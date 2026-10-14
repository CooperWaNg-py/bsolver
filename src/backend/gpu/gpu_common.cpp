#include "backend/gpu/gpu_common.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "core/gtable.h"
#include "core/point_enum.h"
#include "core/random.h"
#include "engine/console.h"
#include "engine/kangaroo_params.h"
#include "engine/target_set.h"

namespace bsolver::gpu {

namespace {
// Upper bound on DP buffer entries the step auto-tuner may request (9 words each).
constexpr uint64_t kMaxDpCapacity = uint64_t{1} << 21;
constexpr uint64_t kMinDpCapacity = 65536;

void packLimbs(const U256& v, uint64_t* out4) {
  for (int l = 0; l < 4; ++l) out4[l] = v.w[l];
}

class GpuScanWorker final : public ScanWorker {
 public:
  GpuScanWorker(std::unique_ptr<ScanKernel> kernel, const ScanSetup& setup)
      : kernel_(std::move(kernel)),
        mode_((hashesCompressed(setup.mode) ? 1u : 0u) | (hashesUncompressed(setup.mode) ? 2u : 0u)) {
    const PointEnumTables& tables = pointEnumTables(kGroup);
    std::vector<uint64_t> gTable((kGroup / 2 + 1) * 8);
    for (uint32_t i = 0; i < kGroup / 2; ++i) packPoint(tables.ig[i], &gTable[8 * i]);
    packPoint(tables.groupG, &gTable[8 * (kGroup / 2)]);

    TargetSet targets(setup.targets);
    targetCount_ = static_cast<uint32_t>(targets.size());
    std::vector<uint32_t> words(5 * std::max<size_t>(targets.size(), 1), 0);
    for (size_t t = 0; t < targets.size(); ++t) {
      const auto& h = targets.items()[t];
      for (int w = 0; w < 5; ++w) {
        words[5 * t + w] = uint32_t{h[4 * w]} << 24 | uint32_t{h[4 * w + 1]} << 16 | uint32_t{h[4 * w + 2]} << 8 |
                           uint32_t{h[4 * w + 3]};
      }
    }
    kernel_->setTables(gTable, words);
    setIters(1);
  }

  uint64_t segmentKeys() const override { return keysPerThread() * kernel_->threads(); }

  void process(const U256& start, uint64_t count, std::vector<U256>& candidates) override {
    if (count == 0) return;
    if (start < U256::fromU64(uint64_t{1} << 20)) throw std::invalid_argument(label() + ": keys below 2^20");
    if (tuned_) {
      launchAt(start, count, candidates);
      return;
    }
    // The first launch (iters = 1, covering all of [start, start+count)) yields the hits; the larger
    // tuning launches over the same keys only measure time.
    tuned_ = true;
    autoTune(
        [&](uint32_t iters) {
          if (iters != iters_) setIters(iters);
          launchAt(start, iters == 1 ? count : 0, candidates);
        },
        [](uint32_t) { return true; });
  }

  std::string label() const override { return kernel_->label(); }

 private:
  uint64_t keysPerThread() const { return uint64_t{iters_} * kGroup; }

  // offsetPts[t] = (t*S + GPU_GROUP/2)*G.
  void setIters(uint32_t iters) {
    iters_ = iters;
    const uint32_t threads = kernel_->threads();
    std::vector<Affine> pts =
        progression(Affine{}, U256::fromU64(kGroup / 2), U256::fromU64(keysPerThread()), threads);
    std::vector<uint64_t> offsets(8 * size_t{threads});
    for (uint32_t t = 0; t < threads; ++t) packPoint(pts[t], &offsets[8 * size_t{t}]);
    kernel_->setOffsets(offsets);
  }

  // One launch over [start, start + segmentKeys()); keeps hits inside [start, start + count).
  void launchAt(const U256& start, uint64_t count, std::vector<U256>& candidates) {
    std::array<uint64_t, kScanParamWords> params{};
    packPoint(mulG(start), params.data());
    params[8] = uint64_t{iters_} | uint64_t{mode_} << 32;
    params[9] = targetCount_;
    uint32_t total = kernel_->launch(params, hits_);
    if (total > kHitCapacity) consoleLog(label() + ": scan hit buffer overflow; some candidates lost");
    const uint64_t s = keysPerThread();
    for (const ScanHit& h : hits_) {
      // key = start + thread*S + iter*GPU_GROUP + GPU_GROUP/2 + offset (offset in [-GPU_GROUP/2, GPU_GROUP/2)).
      uint64_t rel = uint64_t{h.thread} * s + uint64_t{h.iter} * kGroup + kGroup / 2 + static_cast<int64_t>(h.offset);
      if (rel < count) candidates.push_back(start + rel);
    }
  }

  std::unique_ptr<ScanKernel> kernel_;
  uint32_t mode_;
  uint32_t targetCount_ = 0;
  uint32_t iters_ = 1;
  bool tuned_ = false;
  std::vector<ScanHit> hits_;
};

class GpuKangarooWorker final : public KangarooWorker {
 public:
  explicit GpuKangarooWorker(std::unique_ptr<KangarooKernel> kernel)
      : kernel_(std::move(kernel)), n_(size_t{kernel_->threads()} * kKangaroosPerThread) {}

  size_t kangarooCount() const override { return n_; }

  void init(const KangarooParams& params, uint64_t seed) override {
    params_ = params;
    rng_ = Rng(seed ^ 0x9e3779b97f4a7c15ULL);
    steps_ = 0;
    std::vector<uint64_t> jumps(64 * 12);
    for (size_t j = 0; j < 64; ++j) {
      packPoint(params.jumpPoint[j], &jumps[12 * j]);
      packLimbs(params.jumpDist[j], &jumps[12 * j + 8]);
    }
    kernel_->setJumps(jumps);

    auto starts = kangarooStarts(params, n_, seed);
    std::vector<uint64_t> state(kStateComponents * n_);
    uint64_t words[kStateComponents];
    for (size_t i = 0; i < n_; ++i) {
      packKangaroo(starts[i].first, starts[i].second, words);
      for (size_t c = 0; c < kStateComponents; ++c) state[c * n_ + i] = words[c];
    }
    kernel_->writeState(0, state.data(), state.size());
  }

  uint64_t step(std::vector<Dp>& out) override {
    uint64_t jumps = 0;
    auto run = [&](uint32_t steps) { jumps += launchOnce(steps, out); };
    if (steps_ == 0) {
      steps_ = autoTune(run, [&](uint32_t steps) { return dpCapacity(steps) <= kMaxDpCapacity; });
    } else {
      run(steps_);
    }
    return jumps;
  }

  void reset(uint32_t index, uint64_t seed) override {
    auto [p, dist] = kangarooResetStart(params_, index, seed);
    uint64_t words[kStateComponents];
    packKangaroo(p, dist, words);
    for (size_t c = 0; c < kStateComponents; ++c) kernel_->writeState(c * n_ + index, &words[c], 1);
  }

  std::vector<std::pair<Affine, U256>> sample(size_t n) override {
    n = std::min(n, n_);
    std::vector<uint64_t> words(kStateComponents * n);
    for (size_t c = 0; c < kStateComponents; ++c) kernel_->readState(c * n_, &words[c * n], n);
    std::vector<std::pair<Affine, U256>> out(n);
    for (size_t i = 0; i < n; ++i) {
      auto limb = [&](size_t c) { return words[c * n + i]; };
      Affine& p = out[i].first;
      p.inf = false;
      for (int l = 0; l < 4; ++l) {
        p.x.n[l] = limb(l);
        p.y.n[l] = limb(4 + l);
        out[i].second.w[l] = limb(8 + l);
      }
    }
    return out;
  }

  std::string label() const override { return kernel_->label(); }

 private:
  static void packKangaroo(const Affine& p, const U256& dist, uint64_t* words) {
    packPoint(p, words);
    packLimbs(dist, words + 8);
  }

  // max(65536, 4 * expected DPs per launch).
  uint64_t dpCapacity(uint32_t steps) const {
    uint64_t expected = (uint64_t{steps} * n_) >> std::min<uint32_t>(params_.dpBits, 63);
    return std::max(kMinDpCapacity, 4 * (expected + 1));
  }

  uint64_t launchOnce(uint32_t steps, std::vector<Dp>& out) {
    const uint64_t cap = dpCapacity(steps);
    kernel_->reserveDps(static_cast<uint32_t>(cap));
    std::array<uint64_t, kKangarooParamWords> params{n_, steps, params_.dpBits, cap};
    uint32_t total = kernel_->launch(params, raw_);
    if (total > cap) consoleLog(label() + ": DP buffer overflow; some DPs lost");
    stuck_.clear();
    for (size_t e = 0; e + kDpEntryWords <= raw_.size(); e += kDpEntryWords) {
      const uint64_t tag = raw_[e];
      const uint32_t index = static_cast<uint32_t>(tag);
      if (tag >> 63) {
        stuck_.push_back(index);
        continue;
      }
      Dp dp;
      for (int l = 0; l < 4; ++l) {
        dp.x.w[l] = raw_[e + 1 + l];
        dp.dist.w[l] = raw_[e + 5 + l];
      }
      dp.wild = index & 1;
      dp.index = index;
      out.push_back(dp);
    }
    // Kangaroos whose jump point had the same x (dx == 0) did not move; restart them.
    std::sort(stuck_.begin(), stuck_.end());
    stuck_.erase(std::unique(stuck_.begin(), stuck_.end()), stuck_.end());
    for (uint32_t index : stuck_) reset(index, rng_.next());
    return uint64_t{steps} * n_;
  }

  std::unique_ptr<KangarooKernel> kernel_;
  size_t n_;
  KangarooParams params_;
  Rng rng_{0};
  uint32_t steps_ = 0;
  std::vector<uint64_t> raw_;
  std::vector<uint32_t> stuck_;
};
}  // namespace

uint32_t launchThreads(uint32_t computeUnits) {
  uint64_t t = uint64_t{std::max<uint32_t>(computeUnits, 1)} * 512;
  return static_cast<uint32_t>((t + kWorkGroupSize - 1) / kWorkGroupSize * kWorkGroupSize);
}

uint32_t workGroupSize(uint64_t deviceLimit) {
  uint32_t g = 1;
  while (g * 2 <= kWorkGroupSize && g * 2 <= deviceLimit) g *= 2;
  return g;
}

void packPoint(const Affine& p, uint64_t* out8) {
  for (int l = 0; l < 4; ++l) {
    out8[l] = p.x.n[l];
    out8[4 + l] = p.y.n[l];
  }
}

uint32_t autoTune(const std::function<void(uint32_t)>& launch, const std::function<bool(uint32_t)>& allowed) {
  using Clock = std::chrono::steady_clock;
  uint32_t v = 1;
  for (;;) {
    auto t0 = Clock::now();
    launch(v);
    double seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    if (seconds >= kTuneTargetSeconds || v >= kMaxTune || !allowed(v * 2)) return v;
    v *= 2;
  }
}

std::unique_ptr<ScanWorker> makeScanWorker(std::unique_ptr<ScanKernel> kernel, const ScanSetup& setup) {
  return std::make_unique<GpuScanWorker>(std::move(kernel), setup);
}

std::unique_ptr<KangarooWorker> makeKangarooWorker(std::unique_ptr<KangarooKernel> kernel) {
  return std::make_unique<GpuKangarooWorker>(std::move(kernel));
}

}  // namespace bsolver::gpu

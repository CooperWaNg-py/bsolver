#include "backend/gpu/gpu_common.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>

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
// A launch dropped at the smallest size (one work group of scan threads, one kangaroo step) is re-run this many times
// in total before the worker gives up.
constexpr int kLaunchAttempts = 3;

void packLimbs(const U256& v, uint64_t* out4) {
  for (int l = 0; l < 4; ++l) out4[l] = v.w[l];
}

// Empty when all `threads` threads of the launch ran to the end, else why not.
std::string dropReason(const LaunchResult& r, uint32_t threads) {
  if (r.finished == threads && r.error.empty()) return {};
  return r.error.empty() ? std::to_string(r.finished) + " of " + std::to_string(threads) + " threads ran" : r.error;
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
    slice_ = kernel_->threads();
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

  // Covers [start, start + segmentKeys()) in launches of slice_ threads; keeps hits inside [start, start + count).
  // Launches are bounded in time: one slower than kMaxLaunchSeconds, or one the device dropped, halves slice_ (a
  // dropped launch is repeated, so no key is skipped silently); slice_ doubles again (up to all threads) while the
  // doubled launch is projected to stay well below the bound.
  void launchAt(const U256& start, uint64_t count, std::vector<U256>& candidates) {
    using Clock = std::chrono::steady_clock;
    std::array<uint64_t, kScanParamWords> params{};
    packPoint(mulG(start), params.data());
    params[8] = uint64_t{iters_} | uint64_t{mode_} << 32;
    params[9] = targetCount_;
    const uint32_t threads = kernel_->threads();
    const uint32_t group = kernel_->groupSize();
    const uint64_t s = keysPerThread();
    int failures = 0;
    for (uint32_t first = 0; first < threads;) {
      const uint32_t n = std::min(slice_, threads - first);
      params[10] = first;
      auto t0 = Clock::now();
      LaunchResult run = kernel_->launch(params, n, hits_);
      const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();
      if (std::string why = dropReason(run, n); !why.empty()) {
        if (slice_ > group) {
          slice_ = halved(slice_, group);
          consoleLog(label() + ": scan launch dropped (" + why + "); retrying with " + std::to_string(slice_) +
                     " threads per launch");
        } else if (++failures == kLaunchAttempts) {
          throw std::runtime_error(label() + ": scan launch dropped " + std::to_string(failures) + " times: " + why);
        } else {
          consoleLog(label() + ": scan launch dropped (" + why + "); retrying");
        }
        continue;
      }
      if (seconds > kMaxLaunchSeconds && slice_ > group) {
        slice_ = halved(slice_, group);
      } else if (n == slice_ && slice_ < threads && 2 * seconds < 0.75 * kMaxLaunchSeconds) {
        slice_ = std::min(threads, 2 * slice_);
      }
      if (run.count > kHitCapacity) consoleLog(label() + ": scan hit buffer overflow; some candidates lost");
      for (const ScanHit& h : hits_) {
        // key = start + thread*S + iter*GPU_GROUP + GPU_GROUP/2 + offset (offset in [-GPU_GROUP/2, GPU_GROUP/2)).
        uint64_t rel =
            uint64_t{h.thread} * s + uint64_t{h.iter} * kGroup + kGroup / 2 + static_cast<int64_t>(h.offset);
        if (rel < count) candidates.push_back(start + rel);
      }
      first += n;
    }
  }

  // Half of `slice`, rounded down to a multiple of `group` (at least one group).
  static uint32_t halved(uint32_t slice, uint32_t group) { return std::max(group, slice / 2 / group * group); }

  std::unique_ptr<ScanKernel> kernel_;
  uint32_t mode_;
  uint32_t targetCount_ = 0;
  uint32_t iters_ = 1;
  uint32_t slice_ = 0;  // threads per launch
  bool tuned_ = false;
  std::vector<ScanHit> hits_;
};

static_assert(kKangaroosPerThread <= 64, "kangaroo_thread keeps per-kangaroo flags in one 64-bit word");

class GpuKangarooWorker final : public KangarooWorker {
 public:
  explicit GpuKangarooWorker(std::unique_ptr<KangarooKernel> kernel)
      : kernel_(std::move(kernel)), threads_(kernel_->threads()), n_(size_t{threads_} * kKangaroosPerThread) {}

  size_t kangarooCount() const override { return n_; }

  void init(const KangarooParams& params, uint64_t seed, const std::vector<KangarooState>& resume) override {
    if (resume.size() > n_) {
      throw std::invalid_argument(label() + ": " + std::to_string(resume.size()) + " resume states for " +
                                  std::to_string(n_) + " kangaroos");
    }
    params_ = params;
    rng_ = Rng(seed ^ 0x9e3779b97f4a7c15ULL);
    steps_ = 0;
    stepBase_ = 0;
    std::vector<uint64_t> jumps(12 * (kKangarooJumps + kKangarooEscapeJumps));
    packJumps(params.jumps, kKangarooJumps, jumps.data());
    packJumps(params.lookahead, kKangarooEscapeJumps, jumps.data() + 12 * kKangarooJumps);
    kernel_->setJumps(jumps);

    std::vector<KangarooState> starts;
    if (resume.size() == n_) {
      starts = resume;
    } else {
      starts = kangarooStarts(params, n_, seed);
      std::copy(resume.begin(), resume.end(), starts.begin());
    }
    std::vector<uint64_t> state(kangarooStateWords(threads_), 0);
    uint64_t words[kStateComponents];
    for (size_t i = 0; i < n_; ++i) {
      packState(starts[i], words);
      for (size_t c = 0; c < kStateComponents; ++c) state[c * n_ + i] = words[c];
      if (starts[i].lookahead) state[flagWord(i)] |= flagBit(i);
    }
    kernel_->writeState(0, state.data(), state.size());
  }

  uint64_t step(std::vector<Dp>& out) override {
    uint64_t jumps = 0;
    auto run = [&](uint32_t steps) { jumps += launchOnce(steps, out); };
    if (steps_ == 0) {
      steps_ = autoTune(
          run, [&](uint32_t steps) { return steps <= stepCap_ && dpCapacity(steps) <= kMaxDpCapacity; });
    } else {
      run(std::min(steps_, stepCap_));
    }
    return jumps;
  }

  void reset(uint32_t index, uint64_t seed) override { writeKangaroo(index, kangarooResetStart(params_, index, seed)); }

  std::vector<KangarooState> sample(size_t n) override {
    n = std::min(n, n_);
    std::vector<uint64_t> words(kStateComponents * n);
    for (size_t c = 0; c < kStateComponents; ++c) kernel_->readState(c * n_, &words[c * n], n);
    // Kangaroo i's flag is in word i % T, so the first min(n, T) words cover [0, n).
    std::vector<uint64_t> flags(std::min<size_t>(n, threads_));
    kernel_->readState(kStateComponents * n_, flags.data(), flags.size());
    std::vector<KangarooState> out(n);
    uint64_t one[kStateComponents];
    for (size_t i = 0; i < n; ++i) {
      for (size_t c = 0; c < kStateComponents; ++c) one[c] = words[c * n + i];
      out[i] = unpackState(one, flags[i % threads_] & flagBit(i));
    }
    return out;
  }

  std::string label() const override { return kernel_->label(); }

 private:
  void packJumps(const JumpTable& table, size_t count, uint64_t* out) const {
    if (table.dist.size() != count || table.point.size() != count) {
      throw std::invalid_argument(label() + ": jump table needs " + std::to_string(count) + " entries");
    }
    for (size_t j = 0; j < count; ++j) {
      packPoint(table.point[j], out + 12 * j);
      packLimbs(table.dist[j], out + 12 * j + 8);
    }
  }

  // x, y, dist, then the cycle reference: the current x (a walk returning here is on a cycle).
  static void packState(const KangarooState& s, uint64_t* words) {
    packPoint(s.point, words);
    packLimbs(s.dist, words + 8);
    words[12] = s.point.x.n[0];
  }

  static KangarooState unpackState(const uint64_t* words, bool lookahead) {
    KangarooState s;
    s.point.inf = false;
    for (int l = 0; l < 4; ++l) {
      s.point.x.n[l] = words[l];
      s.point.y.n[l] = words[4 + l];
      s.dist.w[l] = words[8 + l];
    }
    s.lookahead = lookahead;
    return s;
  }

  // Kangaroo i = g*T + t: flag bit g of thread t's word.
  size_t flagWord(size_t i) const { return kStateComponents * n_ + i % threads_; }
  uint64_t flagBit(size_t i) const { return uint64_t{1} << (i / threads_); }

  KangarooState readKangaroo(uint32_t index) {
    uint64_t words[kStateComponents], flags = 0;
    for (size_t c = 0; c < kStateComponents; ++c) kernel_->readState(c * n_ + index, &words[c], 1);
    kernel_->readState(flagWord(index), &flags, 1);
    return unpackState(words, flags & flagBit(index));
  }

  void writeKangaroo(uint32_t index, const KangarooState& s) {
    uint64_t words[kStateComponents], flags = 0;
    packState(s, words);
    for (size_t c = 0; c < kStateComponents; ++c) kernel_->writeState(c * n_ + index, &words[c], 1);
    kernel_->readState(flagWord(index), &flags, 1);
    flags = s.lookahead ? flags | flagBit(index) : flags & ~flagBit(index);
    kernel_->writeState(flagWord(index), &flags, 1);
  }

  // max(65536, 4 * expected DPs per launch).
  uint64_t dpCapacity(uint32_t steps) const {
    uint64_t expected = (uint64_t{steps} * n_) >> std::min<uint32_t>(params_.dpBits, 63);
    return std::max(kMinDpCapacity, 4 * (expected + 1));
  }

  // Runs `steps` steps (fewer when the device drops the launch: it is repeated from the unchanged state with half
  // the steps). stepCap_ bounds later launches: halved after a dropped launch or one slower than kMaxLaunchSeconds,
  // doubled again (up to the tuned steps_) while the doubled launch is projected to stay well below that bound.
  // Returns the jumps performed.
  uint64_t launchOnce(uint32_t steps, std::vector<Dp>& out) {
    using Clock = std::chrono::steady_clock;
    uint64_t cap = 0;
    LaunchResult run;
    for (int failures = 0;;) {
      cap = dpCapacity(steps);
      kernel_->reserveDps(static_cast<uint32_t>(cap));
      std::array<uint64_t, kKangarooParamWords> params{n_, steps, params_.dpBits, cap, stepBase_};
      auto t0 = Clock::now();
      run = kernel_->launch(params, raw_);
      const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();
      std::string why = dropReason(run, threads_);
      if (why.empty()) {
        if (seconds > kMaxLaunchSeconds && steps > 1) {
          stepCap_ = std::min(stepCap_, steps / 2);
        } else if (steps == stepCap_ && stepCap_ < steps_ && 2 * seconds < 0.75 * kMaxLaunchSeconds) {
          stepCap_ *= 2;
        }
        break;
      }
      if (steps > 1) {
        steps /= 2;
        stepCap_ = std::min(stepCap_, steps);
        consoleLog(label() + ": kangaroo launch dropped (" + why + "); retrying with " + std::to_string(steps) +
                   " steps per launch");
      } else if (++failures == kLaunchAttempts) {
        throw std::runtime_error(label() + ": kangaroo launch dropped " + std::to_string(failures) + " times: " + why);
      } else {
        consoleLog(label() + ": kangaroo launch dropped (" + why + "); retrying");
      }
    }
    stepBase_ += steps;
    if (run.count > cap) consoleLog(label() + ": DP buffer overflow; some DPs lost");
    stuck_.clear();
    looped_.clear();
    for (size_t e = 0; e + kDpEntryWords <= raw_.size(); e += kDpEntryWords) {
      const uint64_t tag = raw_[e];
      const uint32_t index = static_cast<uint32_t>(tag);
      if (tag >> 63) {
        stuck_.push_back(index);
        continue;
      }
      if ((tag >> 62) & 1) {
        looped_.push_back(index);
        continue;
      }
      Dp dp;
      for (int l = 0; l < 4; ++l) {
        dp.x.w[l] = raw_[e + 1 + l];
        dp.dist.w[l] = raw_[e + 5 + l];
      }
      dp.wild = isWildKangaroo(params_.firstKangaroo + index);
      dp.index = index;
      out.push_back(dp);
    }
    sortUnique(stuck_);
    sortUnique(looped_);
    // Fruitless cycles: the deterministic escape keeps merged walks merged. A false alarm (64-bit reference
    // coincidence) leaves the kangaroo alone.
    for (uint32_t index : looped_) {
      if (std::binary_search(stuck_.begin(), stuck_.end(), index)) continue;
      KangarooState s = readKangaroo(index);
      if (escapeKangarooCycle(params_, s)) writeKangaroo(index, s);
    }
    // Kangaroos whose jump point had the same x (dx == 0) did not move; restart them.
    for (uint32_t index : stuck_) reset(index, rng_.next());
    return uint64_t{steps} * n_;
  }

  static void sortUnique(std::vector<uint32_t>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  }

  std::unique_ptr<KangarooKernel> kernel_;
  uint32_t threads_;
  size_t n_;
  KangarooParams params_;
  Rng rng_{0};
  uint32_t steps_ = 0;
  uint32_t stepCap_ = std::numeric_limits<uint32_t>::max();  // launch-length bound learned from slow/dropped launches
  uint64_t stepBase_ = 0;  // steps walked since init(): the kernel's cycle-reference refresh clock
  std::vector<uint64_t> raw_;
  std::vector<uint32_t> stuck_, looped_;
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

std::string kernelDefines() {
  std::string out;
  auto define = [&](const char* name, size_t value) {
    out += "#define " + std::string(name) + " " + std::to_string(value) + "\n";
  };
  define("GPU_GROUP", kGroup);
  define("SCAN_HIT_CAP", kHitCapacity);
  define("KPT", kKangaroosPerThread);
  define("KANG_COMPONENTS", kStateComponents);
  define("KANG_JUMPS", kKangarooJumps);
  define("KANG_ESC_JUMPS", kKangarooEscapeJumps);
  define("KANG_WINDOW", kKangarooCycleWindow);
  return out;
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

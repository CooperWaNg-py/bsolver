#include "backend/cpu/cpu_kangaroo_worker.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#include "core/random.h"
#include "engine/kangaroo_params.h"

namespace bsolver {

namespace {
constexpr size_t kKangaroos = 1024;
// ~10-50 ms per step() call on current CPUs.
constexpr int kRoundsPerStep = 128;

class CpuKangarooWorker final : public KangarooWorker {
 public:
  explicit CpuKangarooWorker(int index) : index_(index) {}

  size_t kangarooCount() const override { return kKangaroos; }

  void init(const KangarooParams& params, uint64_t seed) override {
    params_ = params;
    rng_ = Rng(seed ^ 0x9e3779b97f4a7c15ULL);
    auto starts = kangarooStarts(params_, kKangaroos, seed);
    for (size_t i = 0; i < kKangaroos; ++i) set(i, starts[i]);
    initialized_ = true;
  }

  uint64_t step(std::vector<Dp>& out) override {
    if (!initialized_) throw std::logic_error("kangaroo worker not initialized");
    uint64_t jumps = 0;
    for (int round = 0; round < kRoundsPerStep; ++round) {
      for (size_t i = 0; i < kKangaroos; ++i) {
        jump_[i] = static_cast<uint8_t>(x_[i].n[0] & 63);
        dx_[i] = feSub(params_.jumpPoint[jump_[i]].x, x_[i]);
      }
      batchInverse(dx_.data(), kKangaroos, scratch_.data());
      for (size_t i = 0; i < kKangaroos; ++i) {
        if (dx_[i].isZero()) {
          // P == +-jumpPoint: the affine add degenerates; restart this kangaroo.
          reset(static_cast<uint32_t>(i), rng_.next());
          continue;
        }
        const Affine& J = params_.jumpPoint[jump_[i]];
        Fe lambda = feMul(feSub(J.y, y_[i]), dx_[i]);
        Fe x3 = feSub(feSub(feSqr(lambda), x_[i]), J.x);
        y_[i] = feSub(feMul(lambda, feSub(x_[i], x3)), y_[i]);
        x_[i] = x3;
        dist_[i] = dist_[i] + params_.jumpDist[jump_[i]];
        ++jumps;
        if (isDistinguished(x3.n[3], params_.dpBits)) {
          out.push_back(Dp{x3.toU256(), dist_[i], i % 2 == 1, static_cast<uint32_t>(i)});
        }
      }
    }
    return jumps;
  }

  void reset(uint32_t index, uint64_t seed) override {
    if (index >= kKangaroos) throw std::out_of_range("kangaroo index out of range");
    set(index, kangarooResetStart(params_, index, seed));
  }

  std::vector<std::pair<Affine, U256>> sample(size_t n) override {
    std::vector<std::pair<Affine, U256>> out;
    for (size_t i = 0; i < std::min(n, kKangaroos); ++i) out.push_back({Affine{x_[i], y_[i], false}, dist_[i]});
    return out;
  }

  std::string label() const override { return "cpu#" + std::to_string(index_); }

 private:
  void set(size_t i, const std::pair<Affine, U256>& s) {
    x_[i] = s.first.x;
    y_[i] = s.first.y;
    dist_[i] = s.second;
  }

  int index_;
  bool initialized_ = false;
  Rng rng_{0};
  KangarooParams params_;
  std::array<Fe, kKangaroos> x_, y_, dx_, scratch_;
  std::array<U256, kKangaroos> dist_;
  std::array<uint8_t, kKangaroos> jump_;
};
}  // namespace

std::unique_ptr<KangarooWorker> makeCpuKangarooWorker(int index) { return std::make_unique<CpuKangarooWorker>(index); }

}  // namespace bsolver

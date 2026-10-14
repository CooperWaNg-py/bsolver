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

// kangarooStep() for 1024 kangaroos per round with one batch inversion, plus Brent-style fruitless-cycle detection:
// each kangaroo's x.limb0 is saved every kKangarooCycleWindow rounds and a later match triggers
// escapeKangarooCycle().
class CpuKangarooWorker final : public KangarooWorker {
 public:
  explicit CpuKangarooWorker(int index) : index_(index) {}

  size_t kangarooCount() const override { return kKangaroos; }

  void init(const KangarooParams& params, uint64_t seed, const std::vector<KangarooState>& resume) override {
    if (resume.size() > kKangaroos) throw std::invalid_argument("too many kangaroo states");
    params_ = params;
    rng_ = Rng(seed ^ 0x9e3779b97f4a7c15ULL);
    auto starts = kangarooStarts(params_, kKangaroos, seed);
    for (size_t i = 0; i < kKangaroos; ++i) {
      wild_[i] = isWildKangaroo(params_.firstKangaroo + i);
      set(i, i < resume.size() ? resume[i] : starts[i]);
    }
    round_ = 0;
    initialized_ = true;
  }

  uint64_t step(std::vector<Dp>& out) override {
    if (!initialized_) throw std::logic_error("kangaroo worker not initialized");
    uint64_t jumps = 0;
    for (int round = 0; round < kRoundsPerStep; ++round) {
      for (size_t i = 0; i < kKangaroos; ++i) {
        const uint64_t mask = flag_[i] ? kKangarooEscapeJumps - 1 : kKangarooJumps - 1;
        jump_[i] = static_cast<uint16_t>(x_[i].n[0] & mask);
        dx_[i] = feSub(table(i).point[jump_[i]].x, x_[i]);
      }
      batchInverse(dx_.data(), kKangaroos, scratch_.data());
      const bool refresh = (round_ + 1) % kKangarooCycleWindow == 0;
      for (size_t i = 0; i < kKangaroos; ++i) {
        if (dx_[i].isZero()) {
          // P == +-jump point: the affine add degenerates; restart this kangaroo.
          reset(static_cast<uint32_t>(i), rng_.next());
          continue;
        }
        const JumpTable& t = table(i);
        const Affine& J = t.point[jump_[i]];
        const bool neg = y_[i].isOdd();
        const Fe jy = neg ? feNeg(J.y) : J.y;
        Fe lambda = feMul(feSub(jy, y_[i]), dx_[i]);
        Fe x3 = feSub(feSub(feSqr(lambda), x_[i]), J.x);
        y_[i] = feSub(feMul(lambda, feSub(x_[i], x3)), y_[i]);
        x_[i] = x3;
        dist_[i] = neg ? dist_[i] - t.dist[jump_[i]] : dist_[i] + t.dist[jump_[i]];
        flag_[i] = !flag_[i] && (x3.n[0] & (kKangarooJumps - 1)) == jump_[i] && y_[i].isOdd() != neg;
        ++jumps;
        if (isDistinguished(x3.n[3], params_.dpBits)) {
          out.push_back(Dp{x3.toU256(), dist_[i], wild_[i] != 0, static_cast<uint32_t>(i)});
        }
        if (x3.n[0] == ref_[i]) {
          KangarooState s = state(i);
          if (escapeKangarooCycle(params_, s)) set(i, s);
        } else if (refresh) {
          ref_[i] = x3.n[0];
        }
      }
      ++round_;
    }
    return jumps;
  }

  void reset(uint32_t index, uint64_t seed) override {
    if (index >= kKangaroos) throw std::out_of_range("kangaroo index out of range");
    set(index, kangarooResetStart(params_, index, seed));
  }

  std::vector<KangarooState> sample(size_t n) override {
    std::vector<KangarooState> out;
    for (size_t i = 0; i < std::min(n, kKangaroos); ++i) out.push_back(state(i));
    return out;
  }

  std::string label() const override { return "cpu#" + std::to_string(index_); }

 private:
  const JumpTable& table(size_t i) const { return flag_[i] ? params_.lookahead : params_.jumps; }

  KangarooState state(size_t i) const { return {Affine{x_[i], y_[i], false}, dist_[i], flag_[i] != 0}; }

  void set(size_t i, const KangarooState& s) {
    x_[i] = s.point.x;
    y_[i] = s.point.y;
    dist_[i] = s.dist;
    flag_[i] = s.lookahead;
    ref_[i] = s.point.x.n[0];
  }

  int index_;
  bool initialized_ = false;
  Rng rng_{0};
  KangarooParams params_;
  uint64_t round_ = 0;
  std::array<Fe, kKangaroos> x_, y_, dx_, scratch_;
  std::array<U256, kKangaroos> dist_;
  std::array<uint64_t, kKangaroos> ref_;
  std::array<uint16_t, kKangaroos> jump_;
  std::array<uint8_t, kKangaroos> flag_, wild_;
};
}  // namespace

std::unique_ptr<KangarooWorker> makeCpuKangarooWorker(int index) { return std::make_unique<CpuKangarooWorker>(index); }

}  // namespace bsolver

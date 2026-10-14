#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "core/field.h"
#include "core/gtable.h"
#include "core/point.h"
#include "core/u256.h"

namespace bsolver {

// Shared tables for PointEnumerator<GROUP>: iG for i = 1..GROUP/2 and GROUP*G.
struct PointEnumTables {
  std::vector<Affine> ig;  // ig[i-1] = i*G
  Affine groupG;           // GROUP*G
};
const PointEnumTables& pointEnumTables(size_t group);

// Enumerates key*G for consecutive keys. Per group of GROUP keys [s, s+GROUP) with center
// c = s + GROUP/2 it does one batch inversion of the GROUP/2 values x(iG) - x(C) plus
// x(GROUP*G) - x(C), yielding C, C +- iG, C - (GROUP/2)G = sG and the next center.
// Groups starting below 2*GROUP (and any group hitting a zero dx) are walked key by key.
template <size_t GROUP>
class PointEnumerator {
  static_assert(GROUP >= 4 && GROUP % 2 == 0, "GROUP must be even");
  static constexpr size_t H = GROUP / 2;

 public:
  PointEnumerator() : tables_(pointEnumTables(GROUP)), dx_(H + 1), scratch_(H + 1) {}

  // Calls fn(offset, P) with P = (start + offset)*G for offset in [0, count). start must be >= 1.
  // Callback order inside a group is arbitrary.
  template <class Fn>
  void run(const U256& start, uint64_t count, Fn&& fn) {
    if (count == 0) return;
    const U256 smallLimit = U256::fromU64(2 * GROUP);
    U256 s = start;
    uint64_t done = 0;
    bool haveCenter = false;
    Affine C;
    while (done < count) {
      const uint64_t n = std::min<uint64_t>(GROUP, count - done);
      bool fast = !(s < smallLimit);
      if (fast) {
        if (!haveCenter) {
          C = mulG(s + H);
          haveCenter = true;
        }
        fast = groupFast(C, done, n, fn);
      }
      if (!fast) {
        walk(s, done, n, fn);
        haveCenter = false;
      }
      s = s + GROUP;
      done += n;
    }
  }

 private:
  // Returns false (without emitting) when C is infinity (s + GROUP/2 == n) or some dx is zero; C advances to the
  // next center on success.
  template <class Fn>
  bool groupFast(Affine& C, uint64_t base, uint64_t n, Fn& fn) {
    if (C.inf) return false;
    const auto& ig = tables_.ig;
    for (size_t i = 0; i < H; ++i) {
      dx_[i] = feSub(ig[i].x, C.x);
      if (dx_[i].isZero()) return false;
    }
    dx_[H] = feSub(tables_.groupG.x, C.x);
    if (dx_[H].isZero()) return false;
    batchInverse(dx_.data(), H + 1, scratch_.data());

    if (H < n) fn(base + H, static_cast<const Affine&>(C));
    Affine P;
    P.inf = false;
    for (size_t i = 1; i <= H; ++i) {
      const Affine& Q = ig[i - 1];
      const Fe& inv = dx_[i - 1];
      if (i < H && H + i < n) {  // C + iG
        Fe lam = feMul(feSub(Q.y, C.y), inv);
        P.x = feSub(feSub(feSqr(lam), C.x), Q.x);
        P.y = feSub(feMul(lam, feSub(C.x, P.x)), C.y);
        fn(base + H + i, static_cast<const Affine&>(P));
      }
      if (H - i < n) {  // C - iG
        Fe lam = feNeg(feMul(feAdd(Q.y, C.y), inv));
        P.x = feSub(feSub(feSqr(lam), C.x), Q.x);
        P.y = feSub(feMul(lam, feSub(C.x, P.x)), C.y);
        fn(base + H - i, static_cast<const Affine&>(P));
      }
    }
    // Next center C + GROUP*G.
    const Affine& GG = tables_.groupG;
    Fe lam = feMul(feSub(GG.y, C.y), dx_[H]);
    Fe nx = feSub(feSub(feSqr(lam), C.x), GG.x);
    Fe ny = feSub(feMul(lam, feSub(C.x, nx)), C.y);
    C.x = nx;
    C.y = ny;
    return true;
  }

  template <class Fn>
  void walk(const U256& s, uint64_t base, uint64_t n, Fn& fn) {
    jac_.resize(n);
    aff_.resize(n);
    jac_[0] = mulGJacobian(s);
    const Affine& G = generator();
    for (uint64_t i = 1; i < n; ++i) jac_[i] = jacAddMixed(jac_[i - 1], G);
    batchToAffine(jac_.data(), aff_.data(), n);
    for (uint64_t i = 0; i < n; ++i) {
      if (!aff_[i].inf) fn(base + i, static_cast<const Affine&>(aff_[i]));
    }
  }

  const PointEnumTables& tables_;
  std::vector<Fe> dx_, scratch_;
  std::vector<Jacobian> jac_;
  std::vector<Affine> aff_;
};

}  // namespace bsolver

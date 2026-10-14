#include "engine/kangaroo_params.h"

#include <algorithm>
#include <cmath>

#include "core/gtable.h"
#include "core/scalar.h"

namespace bsolver {

uint32_t autoDpBits(const U256& width) {
  long double w = width.toLongDouble();
  long double bits = std::ceil(std::log2(2.0L * std::sqrt(w))) - 22.0L;
  return static_cast<uint32_t>(std::clamp(bits, 0.0L, 60.0L));
}

std::array<U256, 64> makeJumpDistances(const U256& width, size_t kangarooCount, Rng& rng) {
  long double w = width.toLongDouble();
  long double mu = static_cast<long double>(kangarooCount) * std::sqrt(w) / 4.0L;
  mu = std::clamp(mu, 1.0L, std::max(1.0L, w / 8.0L));
  U256 hi = U256::fromLongDouble(2.0L * mu);
  if (hi.isZero()) hi = U256::fromU64(1);
  std::array<U256, 64> d;
  for (auto& v : d) v = rng.uniform(U256::fromU64(1), hi);
  return d;
}

KangarooParams makeKangarooParams(const U256& width, const Affine& wildBase, uint32_t dpBits,
                                  const std::array<U256, 64>& jumpDist) {
  KangarooParams p;
  p.width = width;
  p.wildBase = wildBase;
  p.dpBits = dpBits;
  p.jumpDist = jumpDist;
  for (size_t j = 0; j < 64; ++j) p.jumpPoint[j] = mulG(jumpDist[j]);
  return p;
}

Affine wildBaseFor(const Affine& q, const U256& rangeStart) { return add(q, mulG(negMod(rangeStart))); }

bool verifyKangarooPoint(const U256& x, const U256& dist, bool wild, const Affine& wildBase) {
  Affine p = wild ? add(wildBase, mulG(dist)) : mulG(dist);
  return !p.inf && p.x.toU256() == x;
}

double kangarooExpectedWork(const U256& width, size_t kangarooCount, uint32_t dpBits) {
  return 2.0 * std::sqrt(width.toDouble()) +
         static_cast<double>(kangarooCount) * std::ldexp(1.0, static_cast<int>(dpBits));
}

std::vector<std::pair<Affine, U256>> kangarooStarts(const KangarooParams& params, size_t count, uint64_t seed) {
  const uint64_t total = params.totalKangaroos ? params.totalKangaroos : count;
  const uint64_t half = std::max<uint64_t>(1, total / 2);
  const U256 one = U256::fromU64(1);
  U256 tameStep = divU64(params.width, half);
  U256 wildStep = divU64(params.width >> 1, half);
  if (tameStep.isZero()) tameStep = one;
  if (wildStep.isZero()) wildStep = one;
  Rng rng(seed);
  const U256 tameOffset = rng.uniform(one, tameStep);
  const U256 wildOffset = rng.uniform(U256{}, wildStep - 1);
  const uint64_t firstOrdinal = params.firstKangaroo / 2;
  const size_t tameCount = (count + 1) / 2, wildCount = count / 2;
  auto tame = progression(Affine{}, tameOffset + mulU64(tameStep, firstOrdinal), tameStep, tameCount);
  auto wild = progression(params.wildBase, wildOffset + mulU64(wildStep, firstOrdinal), wildStep, wildCount);
  std::vector<std::pair<Affine, U256>> out(count);
  for (size_t i = 0; i < count; ++i) {
    const uint64_t k = firstOrdinal + i / 2;
    if (i % 2 == 0) {
      out[i] = {tame[i / 2], tameOffset + mulU64(tameStep, k)};
    } else {
      out[i] = {wild[i / 2], wildOffset + mulU64(wildStep, k)};
    }
  }
  return out;
}

std::pair<Affine, U256> kangarooResetStart(const KangarooParams& params, uint32_t index, uint64_t seed) {
  Rng rng(seed);
  if (index % 2 == 0) {
    U256 dist = rng.uniform(U256::fromU64(1), params.width - 1);
    return {mulG(dist), dist};
  }
  U256 dist = rng.uniform(U256{}, params.width >> 1);
  return {add(params.wildBase, mulG(dist)), dist};
}

}  // namespace bsolver

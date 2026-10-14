#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "backend/backend.h"
#include "core/random.h"

namespace bsolver {

// Automatic distinguished-point bits: clamp(ceil(log2(2*sqrt(W))) - 22, 0, 60).
uint32_t autoDpBits(const U256& width);

// Mean jump mu = clamp(nK*sqrt(W)/4, 1, W/8); jumpDist[j] uniform in [1, 2*mu]; jumpPoint[j] = jumpDist[j]*G.
std::array<U256, 64> makeJumpDistances(const U256& width, size_t kangarooCount, Rng& rng);

// Fills width/wildBase/dpBits and the jump points for the given distances.
KangarooParams makeKangarooParams(const U256& width, const Affine& wildBase, uint32_t dpBits,
                                  const std::array<U256, 64>& jumpDist);

// Q' = Q - a*G.
Affine wildBaseFor(const Affine& q, const U256& rangeStart);

// Tame: x(dist*G) == x; wild: x(wildBase + dist*G) == x.
bool verifyKangarooPoint(const U256& x, const U256& dist, bool wild, const Affine& wildBase);

// Expected kangaroo work: 2*sqrt(W) + nK*2^dp jumps.
double kangarooExpectedWork(const U256& width, size_t kangarooCount, uint32_t dpBits);

// Starting (point, dist) of local kangaroos [0, count) per the herd layout in KangarooParams
// (tame b_t in [1, W/(nK/2)], wild b_w in [0, (W/2)/(nK/2)), both drawn from `seed`). Two progression() calls.
std::vector<std::pair<Affine, U256>> kangarooStarts(const KangarooParams& params, size_t count, uint64_t seed);

// Fresh random start for local kangaroo `index` (even = tame: dist in [1, W-1]; odd = wild: dist in [0, W/2]).
// One mulG.
std::pair<Affine, U256> kangarooResetStart(const KangarooParams& params, uint32_t index, uint64_t seed);

// Distinguished point: the top dpBits bits of x are zero (every point when dpBits == 0).
inline bool isDistinguished(uint64_t xTopLimb, uint32_t dpBits) {
  return dpBits == 0 || (xTopLimb >> (64 - dpBits)) == 0;
}

}  // namespace bsolver

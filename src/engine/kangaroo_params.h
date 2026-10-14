#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "backend/backend.h"
#include "core/random.h"
#include "engine/bytes.h"
#include "engine/kangaroo_table.h"

// Symmetric (negation-map) kangaroo: geometry, jump tables, the reference step, cycle escape, collision solving and
// the checkpoint payload. Layout (see KangarooParams): K' = (key - a) + shift in [shift, W + shift) with
// shift = max(1, W >> 5); tame points t*G, wild points wildBase + d*G = (d - K')*G. Two kangaroos whose points share x
// are at P1 = +-P2, which gives K' = d -/+ t (tame-wild) or K' = (d1 + d2)/2 (wild-wild mirror).
namespace bsolver {

inline bool isWildKangaroo(uint64_t globalIndex) { return globalIndex % 3 != 0; }

// shift = max(1, W >> 5).
U256 kangarooShift(const U256& width);

// wildBase = -(Q - a*G + shift*G); infinity when Q == (a - shift)*G.
Affine wildBaseFor(const Affine& q, const U256& rangeStart, const U256& shift);

// Signed two's-complement distance -> value mod n.
U256 signedToScalar(const U256& d);

// Automatic distinguished-point bits for nK kangaroos: floor(log2(sqrt(W) / (8*nK))) (DP overhead ratio
// r = nK*2^dp/sqrt(W) ~ 1/8), raised so the table stays below ~2^24 entries (1.3*sqrt(W)/2^dp), clamped to 0..60.
uint32_t autoDpBits(const U256& width, uint64_t kangaroos);

// Expected jumps: 1.15*sqrt(W) + 0.7*nK*2^dp (measured constants of the symmetric walk).
double kangarooExpectedWork(const U256& width, uint64_t kangaroos, uint32_t dpBits);

struct KangarooJumpDistances {
  std::vector<U256> jumps;      // kKangarooJumps, even, uniform in [8*sqrt(W), 16*sqrt(W))
  std::vector<U256> lookahead;  // kKangarooEscapeJumps, even, uniform in [W/1024, W/512)
  std::vector<U256> cycle;      // kKangarooEscapeJumps, even, uniform in [W/4096, W/2048)
};

// Jump sizes do not depend on the kangaroo count, so a checkpoint resumes on any backend with the same tables.
KangarooJumpDistances makeJumpDistances(const U256& width, Rng& rng);

// Fills width/shift/wildBase/dpBits and the jump tables (points = dist*G).
KangarooParams makeKangarooParams(const U256& width, const Affine& wildBase, uint32_t dpBits,
                                  const KangarooJumpDistances& dist);

// Tame: x(dist*G) == x; wild: x(wildBase + dist*G) == x (dist signed).
bool verifyKangarooPoint(const U256& x, const U256& dist, bool wild, const Affine& wildBase);

// Starting states of local kangaroos [0, count) (global g = firstKangaroo + i) across a herd of
// nK = totalKangaroos (or count): tame ordinal k = g/3 at t = t0 + k*(shift/nT), t0 in [1, shift/nT]; wild ordinal
// k = g - g/3 - 1 at d = d0 + k*S_w with S_w = even(34*shift/nW), d0 even in [0, S_w). Offsets come from `seed`.
// Two progression() calls.
std::vector<KangarooState> kangarooStarts(const KangarooParams& params, size_t count, uint64_t seed);

// Fresh random start for local kangaroo `index`: tame t in [1, shift], wild d even in [0, 34*shift). One mulG.
KangarooState kangarooResetStart(const KangarooParams& params, uint32_t index, uint64_t seed);

// The reference step every backend matches: s = -1 if y is odd else +1; with the look-ahead flag set the jump is
// lookahead[x & 63] (flag cleared), else jumps[j = x & 511]; P += s*D, dist += s*d; after a main-table step the flag
// is set iff the new point has x & 511 == j and the opposite y parity (the next main step would undo this one).
// Returns false and leaves s unchanged when P == +-D.
bool kangarooStep(const KangarooParams& params, KangarooState& s);

// Fruitless-cycle escape: replays kangarooStep() until the state returns to s (at most 4*kKangarooCycleWindow steps),
// then from the cycle state with the smallest x steps once with cycle[x & 63] (sign by y parity, flag cleared).
// The result depends only on the cycle's classes, so merged or mirrored walks stay merged. Returns false (s
// unchanged) when s is not on such a cycle.
bool escapeKangarooCycle(const KangarooParams& params, KangarooState& s);

// K' candidates (mod n) for two kangaroos whose points share x, with distances d1, d2 mod n: tame-wild d -/+ t,
// wild-wild (d1 + d2)/2. Empty for tame-tame and for wild-wild at the same distance (merged walks: reset one).
std::vector<U256> collisionCandidates(bool wild1, const U256& d1, bool wild2, const U256& d2);

// Key for K': a - shift + K' (mod n).
U256 kangarooKey(const U256& rangeStart, const U256& shift, const U256& kPrime);

// Points for states holding only dist/lookahead (global index firstKangaroo + i decides the type); batched and
// spread over the hardware threads. A state whose point is infinity gets a fresh kangarooResetStart().
void restoreKangarooPoints(const KangarooParams& params, std::vector<KangarooState>& states, uint64_t seed);

// Kangaroo checkpoint payload: u8 method (1 = symmetric), u32 dpBits, u64 designKangaroos, u32 count + U256 dists for
// each of the three tables, u64 jumps, u64 DP count + entries (u64 x.lo, u64 x.hi, U256 dist mod n, u8 wild),
// u64 state count + states (U256 signed dist, u8 lookahead) in global kangaroo order. Points are not stored.
struct KangarooCheckpoint {
  uint32_t dpBits = 0;
  uint64_t designKangaroos = 0;  // nK the dp was chosen for
  KangarooJumpDistances dist;
  uint64_t jumps = 0;
};
std::vector<uint8_t> writeKangarooCheckpoint(const KangarooCheckpoint& header, const KangarooTable& table,
                                             const std::vector<KangarooState>& states);
// Fills table and states (points left at infinity). Throws std::runtime_error on a malformed payload.
KangarooCheckpoint readKangarooCheckpoint(ByteReader& r, KangarooTable& table, std::vector<KangarooState>& states);

// Distinguished point: the top dpBits bits of x are zero (every point when dpBits == 0).
inline bool isDistinguished(uint64_t xTopLimb, uint32_t dpBits) {
  return dpBits == 0 || (xTopLimb >> (64 - dpBits)) == 0;
}

}  // namespace bsolver

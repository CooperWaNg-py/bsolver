#include "engine/kangaroo_params.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/gtable.h"
#include "core/scalar.h"
#include "engine/job_spec.h"
#include "engine/run_context.h"

namespace bsolver {

namespace {
constexpr uint8_t kCheckpointMethodSymmetric = 1;
constexpr size_t kRestoreChunk = 512;
constexpr uint32_t kMaxCycleWalk = 4 * kKangarooCycleWindow;
// Bound on stored per-entry sizes, used to cap reservations made from untrusted counts.
constexpr size_t kDpEntryBytes = 16 + 32 + 1;
constexpr size_t kStateBytes = 32 + 1;

// Even value uniform in [lo, hi) (lo itself rounded up when the interval holds no even value).
U256 evenUniform(Rng& rng, const U256& lo, const U256& hi) {
  const U256 one = U256::fromU64(1);
  U256 kLo = (lo + one) >> 1;
  U256 kHi = hi.isZero() ? U256{} : (hi - one) >> 1;
  if (kHi < kLo) kHi = kLo;
  return rng.uniform(kLo, kHi) << 1;
}

U256 atLeast(const U256& v, uint64_t floor) { return v < U256::fromU64(floor) ? U256::fromU64(floor) : v; }

std::vector<U256> evenTable(Rng& rng, size_t n, const U256& lo, const U256& hi) {
  std::vector<U256> d(n);
  for (auto& v : d) v = evenUniform(rng, lo, hi);
  return d;
}

JumpTable makeTable(const std::vector<U256>& dist, size_t expected) {
  if (dist.size() != expected) throw std::invalid_argument("kangaroo jump table has the wrong size");
  JumpTable t;
  t.dist = dist;
  t.point.reserve(dist.size());
  for (const U256& d : dist) t.point.push_back(mulG(d));
  return t;
}

// (a + b) mod 2^256 with b signed by neg.
U256 signedAdd(const U256& a, const U256& b, bool neg) { return neg ? a - b : a + b; }

// a/2 mod n (n is odd).
U256 halfMod(const U256& a) {
  if (!(a.w[0] & 1)) return a >> 1;
  U256 s;
  uint64_t carry = add(s, a, kOrderN);
  s = s >> 1;
  s.w[3] |= carry << 63;
  return s;
}

// P + s*D for the jump D (s = -1 when P's y is odd), dist updated alike.
void applyJump(KangarooState& s, const Affine& d, const U256& dist) {
  const bool neg = s.point.y.isOdd();
  Affine jump = d;
  if (neg) jump.y = feNeg(d.y);
  s.point = add(s.point, jump);
  s.dist = signedAdd(s.dist, dist, neg);
}

std::vector<U256> readTable(ByteReader& r, size_t expected) {
  if (r.u32() != expected) throw std::runtime_error("unexpected jump table size");
  std::vector<U256> d(expected);
  for (auto& v : d) v = r.u256();
  return d;
}

void writeTable(ByteWriter& w, const std::vector<U256>& d) {
  w.u32(static_cast<uint32_t>(d.size()));
  for (const U256& v : d) w.u256(v);
}
}  // namespace

U256 kangarooShift(const U256& width) { return atLeast(width >> 5, 1); }

Affine wildBaseFor(const Affine& q, const U256& rangeStart, const U256& shift) {
  Affine s = add(q, mulG(subMod(reduceMod(shift), reduceMod(rangeStart))));
  if (!s.inf) s.y = feNeg(s.y);
  return s;
}

U256 signedToScalar(const U256& d) {
  if (d.w[3] >> 63) return negMod(reduceMod(U256{} - d));
  return reduceMod(d);
}

uint32_t autoDpBits(const U256& width, uint64_t kangaroos) {
  const long double sqrtW = std::sqrt(width.toLongDouble());
  const long double nK = static_cast<long double>(std::max<uint64_t>(kangaroos, 1));
  const long double overhead = std::floor(std::log2(sqrtW / (8.0L * nK)));
  const long double memory = std::ceil(std::log2(1.3L * sqrtW / std::ldexp(1.0L, 24)));
  return static_cast<uint32_t>(std::clamp(std::max(overhead, memory), 0.0L, static_cast<long double>(kMaxDpBits)));
}

double kangarooExpectedWork(const U256& width, uint64_t kangaroos, uint32_t dpBits) {
  return 1.15 * std::sqrt(width.toDouble()) +
         0.7 * static_cast<double>(kangaroos) * std::ldexp(1.0, static_cast<int>(dpBits));
}

KangarooJumpDistances makeJumpDistances(const U256& width, Rng& rng) {
  const long double sqrtW = std::sqrt(width.toLongDouble());
  KangarooJumpDistances d;
  d.jumps = evenTable(rng, kKangarooJumps, atLeast(U256::fromLongDouble(8.0L * sqrtW), 2),
                      atLeast(U256::fromLongDouble(16.0L * sqrtW), 4));
  d.lookahead = evenTable(rng, kKangarooEscapeJumps, atLeast(width >> 10, 2), atLeast(width >> 9, 4));
  d.cycle = evenTable(rng, kKangarooEscapeJumps, atLeast(width >> 12, 2), atLeast(width >> 11, 4));
  return d;
}

KangarooParams makeKangarooParams(const U256& width, const Affine& wildBase, uint32_t dpBits,
                                  const KangarooJumpDistances& dist) {
  KangarooParams p;
  p.width = width;
  p.shift = kangarooShift(width);
  p.wildBase = wildBase;
  p.dpBits = dpBits;
  p.jumps = makeTable(dist.jumps, kKangarooJumps);
  p.lookahead = makeTable(dist.lookahead, kKangarooEscapeJumps);
  p.cycle = makeTable(dist.cycle, kKangarooEscapeJumps);
  return p;
}

bool verifyKangarooPoint(const U256& x, const U256& dist, bool wild, const Affine& wildBase) {
  Affine p = mulG(signedToScalar(dist));
  if (wild) p = add(wildBase, p);
  return !p.inf && p.x.toU256() == x;
}

std::vector<KangarooState> kangarooStarts(const KangarooParams& params, size_t count, uint64_t seed) {
  const uint64_t total = std::max<uint64_t>(params.totalKangaroos ? params.totalKangaroos : count, 1);
  const uint64_t tames = (total + 2) / 3, wilds = std::max<uint64_t>(total - tames, 1);
  const U256 one = U256::fromU64(1);
  const U256 tameStep = atLeast(divU64(params.shift, tames), 1);
  const U256 wildStep = atLeast(divU64(mulU64(params.shift, 34), wilds) >> 1 << 1, 2);
  Rng rng(seed);
  const U256 tameOffset = rng.uniform(one, tameStep);
  const U256 wildOffset = evenUniform(rng, U256{}, wildStep);

  // Local index ranges of each type are consecutive in the type's ordinal.
  const uint64_t first = params.firstKangaroo;
  uint64_t firstTame = 0, firstWild = 0;
  size_t tameCount = 0, wildCount = 0;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t g = first + i;
    if (isWildKangaroo(g)) {
      if (wildCount++ == 0) firstWild = g - g / 3 - 1;
    } else if (tameCount++ == 0) {
      firstTame = g / 3;
    }
  }
  const U256 tameStart = tameOffset + mulU64(tameStep, firstTame);
  const U256 wildStart = wildOffset + mulU64(wildStep, firstWild);
  auto tame = progression(Affine{}, tameStart, tameStep, tameCount);
  auto wild = progression(params.wildBase, wildStart, wildStep, wildCount);

  std::vector<KangarooState> out(count);
  size_t t = 0, w = 0;
  for (size_t i = 0; i < count; ++i) {
    if (isWildKangaroo(first + i)) {
      out[i] = {wild[w], wildStart + mulU64(wildStep, w), false};
      ++w;
    } else {
      out[i] = {tame[t], tameStart + mulU64(tameStep, t), false};
      ++t;
    }
    // A wild start exactly at -K' (point at infinity) cannot walk: draw a random one instead.
    for (uint64_t k = 0; out[i].point.inf; ++k) out[i] = kangarooResetStart(params, static_cast<uint32_t>(i), seed + k);
  }
  return out;
}

KangarooState kangarooResetStart(const KangarooParams& params, uint32_t index, uint64_t seed) {
  Rng rng(seed);
  const bool wild = isWildKangaroo(params.firstKangaroo + index);
  for (;;) {
    KangarooState s;
    if (wild) {
      s.dist = evenUniform(rng, U256{}, mulU64(params.shift, 34));
      s.point = add(params.wildBase, mulG(s.dist));
    } else {
      s.dist = rng.uniform(U256::fromU64(1), params.shift);
      s.point = mulG(s.dist);
    }
    if (!s.point.inf) return s;
  }
}

bool kangarooStep(const KangarooParams& params, KangarooState& s) {
  const bool flag = s.lookahead;
  const size_t j = static_cast<size_t>(s.point.x.n[0] & (flag ? kKangarooEscapeJumps - 1 : kKangarooJumps - 1));
  const JumpTable& table = flag ? params.lookahead : params.jumps;
  if (table.point[j].x == s.point.x) return false;
  const bool neg = s.point.y.isOdd();
  applyJump(s, table.point[j], table.dist[j]);
  s.lookahead = !flag && (s.point.x.n[0] & (kKangarooJumps - 1)) == j && s.point.y.isOdd() != neg;
  return true;
}

bool escapeKangarooCycle(const KangarooParams& params, KangarooState& s) {
  KangarooState cur = s, best = s;
  bool closed = false;
  for (uint32_t k = 0; k < kMaxCycleWalk && !closed; ++k) {
    if (!kangarooStep(params, cur)) return false;
    closed = cur.point.x == s.point.x && cur.lookahead == s.lookahead;
    if (cur.point.x.toU256() < best.point.x.toU256()) best = cur;
  }
  if (!closed) return false;
  const size_t j = static_cast<size_t>(best.point.x.n[0] & (kKangarooEscapeJumps - 1));
  if (params.cycle.point[j].x == best.point.x) return false;
  applyJump(best, params.cycle.point[j], params.cycle.dist[j]);
  best.lookahead = false;
  s = best;
  return true;
}

std::vector<U256> collisionCandidates(bool wild1, const U256& d1, bool wild2, const U256& d2) {
  if (!wild1 && !wild2) return {};
  if (wild1 && wild2) {
    if (d1 == d2) return {};
    return {halfMod(addMod(d1, d2))};
  }
  const U256& t = wild1 ? d2 : d1;
  const U256& d = wild1 ? d1 : d2;
  return {subMod(d, t), addMod(d, t)};
}

U256 kangarooKey(const U256& rangeStart, const U256& shift, const U256& kPrime) {
  return addMod(subMod(reduceMod(rangeStart), reduceMod(shift)), kPrime);
}

void restoreKangarooPoints(const KangarooParams& params, std::vector<KangarooState>& states, uint64_t seed) {
  auto restoreRange = [&](size_t begin, size_t end) {
    std::vector<Jacobian> jac;
    std::vector<Affine> aff;
    for (size_t c = begin; c < end; c += kRestoreChunk) {
      const size_t n = std::min(kRestoreChunk, end - c);
      jac.resize(n);
      aff.resize(n);
      for (size_t i = 0; i < n; ++i) {
        jac[i] = mulGJacobian(signedToScalar(states[c + i].dist));
        if (isWildKangaroo(params.firstKangaroo + c + i)) jac[i] = jacAddMixed(jac[i], params.wildBase);
      }
      batchToAffine(jac.data(), aff.data(), n);
      for (size_t i = 0; i < n; ++i) {
        states[c + i].point = aff[i];
        if (aff[i].inf) states[c + i] = kangarooResetStart(params, static_cast<uint32_t>(c + i), seed + c + i);
      }
    }
  };
  const size_t threads =
      std::min<size_t>(effectiveThreads(0), (states.size() + kRestoreChunk - 1) / kRestoreChunk);
  if (threads <= 1) {
    restoreRange(0, states.size());
    return;
  }
  const size_t per = (states.size() + threads - 1) / threads;
  runThreads(
      threads,
      [&](size_t t) {
        const size_t begin = t * per, end = std::min(states.size(), begin + per);
        if (begin < end) restoreRange(begin, end);
      },
      [] {});
}

std::vector<uint8_t> writeKangarooCheckpoint(const KangarooCheckpoint& header, const KangarooTable& table,
                                             const std::vector<KangarooState>& states) {
  ByteWriter w;
  // Entries inserted while visiting only grow the buffer; the count is patched in afterwards.
  w.reserve(64 + (kKangarooJumps + 2 * kKangarooEscapeJumps) * 32 + table.size() * kDpEntryBytes +
            states.size() * kStateBytes);
  w.u8(kCheckpointMethodSymmetric);
  w.u32(header.dpBits);
  w.u64(header.designKangaroos);
  writeTable(w, header.dist.jumps);
  writeTable(w, header.dist.lookahead);
  writeTable(w, header.dist.cycle);
  w.u64(header.jumps);
  const size_t countPos = w.size();
  w.u64(0);
  uint64_t count = 0;
  table.forEach([&](const X128& x, const KangarooTable::Entry& e) {
    w.u64(x.lo);
    w.u64(x.hi);
    w.u256(e.dist);
    w.u8(e.wild ? 1 : 0);
    ++count;
  });
  w.patchU64(countPos, count);
  w.u64(states.size());
  for (const KangarooState& s : states) {
    w.u256(s.dist);
    w.u8(s.lookahead ? 1 : 0);
  }
  return std::move(w.data());
}

KangarooCheckpoint readKangarooCheckpoint(ByteReader& r, KangarooTable& table, std::vector<KangarooState>& states) {
  if (r.u8() != kCheckpointMethodSymmetric) throw std::runtime_error("unknown kangaroo checkpoint method");
  KangarooCheckpoint h;
  h.dpBits = r.u32();
  if (h.dpBits > kMaxDpBits) throw std::runtime_error("dp bits out of range");
  h.designKangaroos = r.u64();
  h.dist.jumps = readTable(r, kKangarooJumps);
  h.dist.lookahead = readTable(r, kKangarooEscapeJumps);
  h.dist.cycle = readTable(r, kKangarooEscapeJumps);
  h.jumps = r.u64();
  const uint64_t dps = r.u64();
  for (uint64_t i = 0; i < dps; ++i) {
    X128 x;
    x.lo = r.u64();
    x.hi = r.u64();
    KangarooTable::Entry e;
    e.dist = r.u256();
    if (!(e.dist < kOrderN)) throw std::runtime_error("distance out of range");
    const uint8_t wild = r.u8();
    if (wild > 1) throw std::runtime_error("invalid kangaroo type");
    e.wild = wild != 0;
    table.insert(x, e);
  }
  const uint64_t count = r.u64();
  states.clear();
  states.reserve(static_cast<size_t>(std::min<uint64_t>(count, r.remaining() / kStateBytes)));
  for (uint64_t i = 0; i < count; ++i) {
    KangarooState s;
    s.dist = r.u256();
    const uint8_t flag = r.u8();
    if (flag > 1) throw std::runtime_error("invalid kangaroo flags");
    s.lookahead = flag != 0;
    states.push_back(s);
  }
  return h;
}

}  // namespace bsolver

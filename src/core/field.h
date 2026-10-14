#pragma once

#include <cstddef>
#include <cstdint>

#include "core/int128.h"
#include "core/u256.h"

namespace bsolver {

// Element of GF(p), p = 2^256 - 2^32 - 977. Always fully reduced (< p).
struct Fe {
  uint64_t n[4] = {0, 0, 0, 0};

  static Fe fromU64(uint64_t v) { return Fe{{v, 0, 0, 0}}; }
  // Reduces values >= p.
  static Fe fromU256(const U256& v);
  U256 toU256() const { return U256{{n[0], n[1], n[2], n[3]}}; }
  void toBytesBE(uint8_t* out32) const { toU256().toBytesBE(out32); }

  bool isZero() const { return (n[0] | n[1] | n[2] | n[3]) == 0; }
  bool isOdd() const { return n[0] & 1; }
  friend bool operator==(const Fe& a, const Fe& b) = default;
};

namespace fe_detail {
inline constexpr uint64_t kC = 0x1000003D1ULL;  // 2^256 mod p
inline constexpr uint64_t kP0 = 0xFFFFFFFEFFFFFC2FULL;
inline constexpr uint64_t kPHi = 0xFFFFFFFFFFFFFFFFULL;

inline constexpr U256 kFieldP{{kP0, kPHi, kPHi, kPHi}};

// Adds C to r in place when r >= p (i.e. r + C overflows 2^256), giving r - p.
inline void finalReduce(uint64_t r[4]) {
  uint64_t c;
  uint64_t t0 = addc(r[0], kC, 0, &c);
  uint64_t t1 = addc(r[1], 0, c, &c);
  uint64_t t2 = addc(r[2], 0, c, &c);
  uint64_t t3 = addc(r[3], 0, c, &c);
  if (c) {
    r[0] = t0;
    r[1] = t1;
    r[2] = t2;
    r[3] = t3;
  }
}

// Reduces a 512-bit product t[0..7] modulo p.
inline Fe reduce512(const uint64_t t[8]) {
  uint64_t r[4];
  uint64_t carry = 0;
  for (int i = 0; i < 4; ++i) {
    uint64_t hi;
    uint64_t lo = mul64(t[4 + i], kC, &hi);
    uint64_t c1, c2;
    uint64_t s = addc(t[i], lo, 0, &c1);
    s = addc(s, carry, 0, &c2);
    r[i] = s;
    carry = hi + c1 + c2;
  }
  // carry < 2^34; fold carry * C (< 2^67) back in.
  uint64_t hi;
  uint64_t lo = mul64(carry, kC, &hi);
  uint64_t c;
  r[0] = addc(r[0], lo, 0, &c);
  r[1] = addc(r[1], hi, c, &c);
  r[2] = addc(r[2], 0, c, &c);
  r[3] = addc(r[3], 0, c, &c);
  if (c) {
    // Wrapped past 2^256: add C once more (cannot overflow again, r is tiny now).
    r[0] = addc(r[0], kC, 0, &c);
    r[1] = addc(r[1], 0, c, &c);
    r[2] = addc(r[2], 0, c, &c);
    r[3] += c;
  }
  finalReduce(r);
  return Fe{{r[0], r[1], r[2], r[3]}};
}
}  // namespace fe_detail

inline Fe feAdd(const Fe& a, const Fe& b) {
  uint64_t r[4], c;
  r[0] = addc(a.n[0], b.n[0], 0, &c);
  r[1] = addc(a.n[1], b.n[1], c, &c);
  r[2] = addc(a.n[2], b.n[2], c, &c);
  r[3] = addc(a.n[3], b.n[3], c, &c);
  if (c) {
    // a + b - p = (a + b - 2^256) + C
    uint64_t c2;
    r[0] = addc(r[0], fe_detail::kC, 0, &c2);
    r[1] = addc(r[1], 0, c2, &c2);
    r[2] = addc(r[2], 0, c2, &c2);
    r[3] += c2;
  } else {
    fe_detail::finalReduce(r);
  }
  return Fe{{r[0], r[1], r[2], r[3]}};
}

inline Fe feSub(const Fe& a, const Fe& b) {
  uint64_t r[4], c;
  r[0] = subb(a.n[0], b.n[0], 0, &c);
  r[1] = subb(a.n[1], b.n[1], c, &c);
  r[2] = subb(a.n[2], b.n[2], c, &c);
  r[3] = subb(a.n[3], b.n[3], c, &c);
  if (c) {
    // a - b + p = (a - b + 2^256) - C
    uint64_t c2;
    r[0] = subb(r[0], fe_detail::kC, 0, &c2);
    r[1] = subb(r[1], 0, c2, &c2);
    r[2] = subb(r[2], 0, c2, &c2);
    r[3] -= c2;
  }
  return Fe{{r[0], r[1], r[2], r[3]}};
}

inline Fe feNeg(const Fe& a) { return feSub(Fe{}, a); }

inline Fe feMul(const Fe& a, const Fe& b) {
  uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    uint64_t carry = 0;
    for (int j = 0; j < 4; ++j) {
      uint64_t hi;
      uint64_t lo = mul64(a.n[i], b.n[j], &hi);
      uint64_t s = t[i + j] + lo;
      hi += s < lo;
      s += carry;
      hi += s < carry;
      t[i + j] = s;
      carry = hi;
    }
    t[i + 4] = carry;
  }
  return fe_detail::reduce512(t);
}

inline Fe feSqr(const Fe& a) {
  // Off-diagonal products once, doubled, plus the squares on the diagonal.
  uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    uint64_t carry = 0;
    for (int j = i + 1; j < 4; ++j) {
      uint64_t hi;
      uint64_t lo = mul64(a.n[i], a.n[j], &hi);
      uint64_t s = t[i + j] + lo;
      hi += s < lo;
      s += carry;
      hi += s < carry;
      t[i + j] = s;
      carry = hi;
    }
    t[i + 4] = carry;
  }
  // Double.
  uint64_t top = 0;
  for (int i = 0; i < 8; ++i) {
    uint64_t v = t[i];
    t[i] = (v << 1) | top;
    top = v >> 63;
  }
  // Add diagonal squares.
  uint64_t c = 0;
  for (int i = 0; i < 4; ++i) {
    uint64_t hi;
    uint64_t lo = mul64(a.n[i], a.n[i], &hi);
    t[2 * i] = addc(t[2 * i], lo, c, &c);
    t[2 * i + 1] = addc(t[2 * i + 1], hi, c, &c);
  }
  return fe_detail::reduce512(t);
}

// a^(p-2); returns 0 for a == 0.
Fe feInv(const Fe& a);
// Square root if it exists (a^((p+1)/4)); returns false when a is not a square.
bool feSqrt(const Fe& a, Fe* out);
// In-place Montgomery batch inversion. Zero entries stay zero. scratch must hold n elements.
void batchInverse(Fe* v, size_t n, Fe* scratch);

}  // namespace bsolver

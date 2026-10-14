#pragma once

#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace bsolver {

// Full 64x64 -> 128 multiply; returns the low half, stores the high half in *hi.
inline uint64_t mul64(uint64_t a, uint64_t b, uint64_t* hi) {
#if defined(__SIZEOF_INT128__)
  unsigned __int128 r = static_cast<unsigned __int128>(a) * b;
  *hi = static_cast<uint64_t>(r >> 64);
  return static_cast<uint64_t>(r);
#elif defined(_MSC_VER) && defined(_M_X64)
  return _umul128(a, b, hi);
#elif defined(_MSC_VER) && defined(_M_ARM64)
  *hi = __umulh(a, b);
  return a * b;
#else
  uint64_t al = a & 0xffffffffu, ah = a >> 32, bl = b & 0xffffffffu, bh = b >> 32;
  uint64_t ll = al * bl, lh = al * bh, hl = ah * bl, hh = ah * bh;
  uint64_t mid = (ll >> 32) + (lh & 0xffffffffu) + (hl & 0xffffffffu);
  *hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
  return (mid << 32) | (ll & 0xffffffffu);
#endif
}

// s = a + b + carry_in; *carry_out receives 0/1. carry_in must be 0 or 1.
inline uint64_t addc(uint64_t a, uint64_t b, uint64_t carry_in, uint64_t* carry_out) {
  uint64_t s = a + b;
  uint64_t c = s < a;
  uint64_t t = s + carry_in;
  c += t < s;
  *carry_out = c;
  return t;
}

// d = a - b - borrow_in; *borrow_out receives 0/1. borrow_in must be 0 or 1.
inline uint64_t subb(uint64_t a, uint64_t b, uint64_t borrow_in, uint64_t* borrow_out) {
  uint64_t d = a - b;
  uint64_t c = a < b;
  uint64_t t = d - borrow_in;
  c += d < borrow_in;
  *borrow_out = c;
  return t;
}

}  // namespace bsolver

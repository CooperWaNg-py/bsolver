#pragma once

// Lane-generic SHA-256 and RIPEMD-160 compression functions, internal to src/core.
//
// A lane type V holds V::kLanes independent 32-bit words and provides + ^ |, orn(a, b) = a | ~b,
// bitSelect(m, a, b) = (m & a) | (~m & b), shl<S>/shr<S>/rotl<S> (0 < S < 32), bswap, V::splat, V::load and store.
// Lane1 is the scalar instance; the SIMD instances live in hash_batch_impl.h. The rounds are unrolled at compile time
// so rotation counts, message indices and constant (padding) message words fold into the code.
//
// Some including translation units are compiled with extra instruction-set flags (AVX2, SHA, ...). Everything here
// therefore has internal linkage: the linker can never pick a copy compiled with those flags for a baseline caller.

#include <cstdint>
#include <utility>

#if defined(_MSC_VER) && !defined(__clang__)
#include <stdlib.h>
#define BSOLVER_HASH_INLINE __forceinline
#else
#define BSOLVER_HASH_INLINE inline __attribute__((always_inline))
#endif

namespace bsolver::hashimpl {
namespace {

alignas(16) constexpr uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
alignas(16) constexpr uint32_t kSha256Init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                                 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

constexpr uint8_t kRmdRL[80] = {0, 1, 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 7,  4,  13, 1,
                                10, 6, 15, 3,  12, 0,  9,  5,  2,  14, 11, 8,  3,  10, 14, 4,  9,  15, 8,  1,
                                2,  7, 0,  6,  13, 11, 5,  12, 1,  9,  11, 10, 0,  8,  12, 4,  13, 3,  7,  15,
                                14, 5, 6,  2,  4,  0,  5,  9,  7,  12, 2,  10, 14, 1,  3,  8,  11, 6,  15, 13};
constexpr uint8_t kRmdRR[80] = {5,  14, 7,  0, 9, 2,  11, 4,  13, 6,  15, 8,  1,  10, 3,  12, 6,  11, 3,  7,
                                0,  13, 5,  10, 14, 15, 8,  12, 4,  9,  1,  2,  15, 5,  1,  3,  7,  14, 6,  9,
                                11, 8,  12, 2,  10, 0,  4,  13, 8,  6,  4,  1,  3,  11, 15, 0,  5,  12, 2,  13,
                                9,  7,  10, 14, 12, 15, 10, 4,  1,  5,  8,  7,  6,  2,  13, 14, 0,  3,  9,  11};
constexpr uint8_t kRmdSL[80] = {11, 14, 15, 12, 5,  8,  7,  9,  11, 13, 14, 15, 6,  7,  9,  8,  7,  6,  8,  13,
                                11, 9,  7,  15, 7,  12, 15, 9,  11, 7,  13, 12, 11, 13, 6,  7,  14, 9,  13, 15,
                                14, 8,  13, 6,  5,  12, 7,  5,  11, 12, 14, 15, 14, 15, 9,  8,  9,  14, 5,  6,
                                8,  6,  5,  12, 9,  15, 5,  11, 6,  8,  13, 12, 5,  12, 13, 14, 11, 8,  5,  6};
constexpr uint8_t kRmdSR[80] = {8,  9,  9,  11, 13, 15, 15, 5,  7,  7,  8,  11, 14, 14, 12, 6,  9,  13, 15, 7,
                                12, 8,  9,  11, 7,  7,  12, 7,  6,  15, 13, 11, 9,  7,  15, 11, 8,  6,  6,  14,
                                12, 13, 5,  14, 13, 13, 7,  5,  15, 5,  8,  11, 14, 14, 6,  14, 6,  9,  12, 9,
                                12, 5,  15, 8,  8,  5,  12, 9,  12, 5,  14, 6,  8,  13, 6,  5,  15, 13, 11, 11};
constexpr uint32_t kRmdKL[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
constexpr uint32_t kRmdKR[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};
constexpr uint32_t kRmdInit[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};

BSOLVER_HASH_INLINE uint32_t bswap32(uint32_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
  return _byteswap_ulong(v);
#else
  return __builtin_bswap32(v);
#endif
}

// ---- Scalar lane ----------------------------------------------------------------------------------------------------

struct Lane1 {
  uint32_t v;
  static constexpr int kLanes = 1;
  static BSOLVER_HASH_INLINE Lane1 splat(uint32_t x) { return {x}; }
  static BSOLVER_HASH_INLINE Lane1 load(const uint32_t* p) { return {*p}; }
  BSOLVER_HASH_INLINE void store(uint32_t* p) const { *p = v; }
};
BSOLVER_HASH_INLINE Lane1 operator+(Lane1 a, Lane1 b) { return {a.v + b.v}; }
BSOLVER_HASH_INLINE Lane1 operator^(Lane1 a, Lane1 b) { return {a.v ^ b.v}; }
BSOLVER_HASH_INLINE Lane1 operator|(Lane1 a, Lane1 b) { return {a.v | b.v}; }
BSOLVER_HASH_INLINE Lane1 orn(Lane1 a, Lane1 b) { return {a.v | ~b.v}; }
BSOLVER_HASH_INLINE Lane1 bitSelect(Lane1 m, Lane1 a, Lane1 b) { return {((a.v ^ b.v) & m.v) ^ b.v}; }
template <int S>
BSOLVER_HASH_INLINE Lane1 shl(Lane1 a) {
  return {a.v << S};
}
template <int S>
BSOLVER_HASH_INLINE Lane1 shr(Lane1 a) {
  return {a.v >> S};
}
template <int S>
BSOLVER_HASH_INLINE Lane1 rotl(Lane1 a) {
  return {(a.v << S) | (a.v >> (32 - S))};
}
BSOLVER_HASH_INLINE Lane1 bswap(Lane1 a) { return {bswap32(a.v)}; }

// ---- SHA-256 --------------------------------------------------------------------------------------------------------

template <int S, class V>
BSOLVER_HASH_INLINE V rotr(V a) {
  return rotl<32 - S>(a);
}

// Round I; s[] rotates by index instead of moving values (after round I, word k lives in s[(k - I - 1) mod 8]).
template <int I, class V>
BSOLVER_HASH_INLINE void sha256Round(V* s, V* w) {
  if constexpr (I >= 16) {
    const V w15 = w[(I - 15) & 15], w2 = w[(I - 2) & 15];
    const V s0 = rotr<7>(w15) ^ rotr<18>(w15) ^ shr<3>(w15);
    const V s1 = rotr<17>(w2) ^ rotr<19>(w2) ^ shr<10>(w2);
    w[I & 15] = w[I & 15] + s0 + w[(I - 7) & 15] + s1;
  }
  const V a = s[(64 - I) & 7], b = s[(65 - I) & 7], c = s[(66 - I) & 7];
  const V e = s[(68 - I) & 7], f = s[(69 - I) & 7], g = s[(70 - I) & 7], h = s[(71 - I) & 7];
  const V t1 = h + (rotr<6>(e) ^ rotr<11>(e) ^ rotr<25>(e)) + bitSelect(e, f, g) + V::splat(kSha256K[I]) + w[I & 15];
  const V t2 = (rotr<2>(a) ^ rotr<13>(a) ^ rotr<22>(a)) + bitSelect(a ^ b, c, b);  // maj(a, b, c)
  s[(67 - I) & 7] = s[(67 - I) & 7] + t1;
  s[(71 - I) & 7] = t1 + t2;
}

template <class V, int... I>
BSOLVER_HASH_INLINE void sha256Rounds(V* s, V* w, std::integer_sequence<int, I...>) {
  (sha256Round<I>(s, w), ...);
}

// One SHA-256 compression of the 16 big-endian message words w (clobbered) into state st.
template <class V>
BSOLVER_HASH_INLINE void sha256Compress(V (&st)[8], V (&w)[16]) {
  V s[8];
  for (int i = 0; i < 8; ++i) s[i] = st[i];
  sha256Rounds(s, w, std::make_integer_sequence<int, 64>{});
  for (int i = 0; i < 8; ++i) st[i] = st[i] + s[i];
}

template <class V>
BSOLVER_HASH_INLINE void sha256Init(V (&st)[8]) {
  for (int i = 0; i < 8; ++i) st[i] = V::splat(kSha256Init[i]);
}

// ---- RIPEMD-160 -----------------------------------------------------------------------------------------------------

template <int R, class V>
BSOLVER_HASH_INLINE V rmdF(V x, V y, V z) {
  if constexpr (R == 0) return x ^ y ^ z;
  else if constexpr (R == 1) return bitSelect(x, y, z);
  else if constexpr (R == 2) return orn(x, y) ^ z;
  else if constexpr (R == 3) return bitSelect(z, x, y);
  else return x ^ orn(y, z);
}

template <uint32_t K, class V>
BSOLVER_HASH_INLINE V addConst(V a) {
  if constexpr (K == 0) return a;
  else return a + V::splat(K);
}

template <class V>
struct RmdState {
  V al, bl, cl, dl, el, ar, br, cr, dr, er;
};

template <int J, class V>
BSOLVER_HASH_INLINE void rmdStep(RmdState<V>& s, const V* x) {
  constexpr int r = J >> 4;
  V t = rotl<kRmdSL[J]>(addConst<kRmdKL[r]>(s.al + rmdF<r>(s.bl, s.cl, s.dl) + x[kRmdRL[J]])) + s.el;
  s.al = s.el;
  s.el = s.dl;
  s.dl = rotl<10>(s.cl);
  s.cl = s.bl;
  s.bl = t;
  t = rotl<kRmdSR[J]>(addConst<kRmdKR[r]>(s.ar + rmdF<4 - r>(s.br, s.cr, s.dr) + x[kRmdRR[J]])) + s.er;
  s.ar = s.er;
  s.er = s.dr;
  s.dr = rotl<10>(s.cr);
  s.cr = s.br;
  s.br = t;
}

template <class V, int... J>
BSOLVER_HASH_INLINE void rmdSteps(RmdState<V>& s, const V* x, std::integer_sequence<int, J...>) {
  (rmdStep<J>(s, x), ...);
}

// One RIPEMD-160 compression of the 16 little-endian message words x into st.
template <class V>
BSOLVER_HASH_INLINE void rmd160Compress(V (&st)[5], const V (&x)[16]) {
  RmdState<V> s{st[0], st[1], st[2], st[3], st[4], st[0], st[1], st[2], st[3], st[4]};
  rmdSteps(s, x, std::make_integer_sequence<int, 80>{});
  const V t = st[1] + s.cl + s.dr;
  st[1] = st[2] + s.dl + s.er;
  st[2] = st[3] + s.el + s.ar;
  st[3] = st[4] + s.al + s.br;
  st[4] = st[0] + s.bl + s.cr;
  st[0] = t;
}

template <class V>
BSOLVER_HASH_INLINE void rmd160Init(V (&st)[5]) {
  for (int i = 0; i < 5; ++i) st[i] = V::splat(kRmdInit[i]);
}

}  // namespace
}  // namespace bsolver::hashimpl

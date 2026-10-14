#pragma once

// Building blocks of the hash160 batch implementations (hash_batch*.cpp), internal to src/core.
//
// Each implementation file selects what it needs before including this header:
//   BSOLVER_LANES_NEON   Neon8 lane type (arm64 baseline)
//   BSOLVER_LANES_SSE2   Sse8 lane type (x86-64 baseline)
//   BSOLVER_LANES_AVX2   Avx8 lane type (file compiled with AVX2 enabled)
//   BSOLVER_SHA_ARMV8    SHA-256 via the ARMv8 crypto extension (file compiled with it enabled; implies NEON)
//   BSOLVER_SHA_SHANI    SHA-256 via the x86 SHA extensions (file compiled with SHA + SSE4.1 enabled)
// Like hash_lanes.h, everything has internal linkage.

#include <cstdint>
#include <utility>

#include "core/hash_batch.h"
#include "core/hash_lanes.h"

#if defined(BSOLVER_LANES_NEON) || defined(BSOLVER_SHA_ARMV8)
#include <arm_neon.h>
#endif
#if defined(BSOLVER_LANES_SSE2) || defined(BSOLVER_LANES_AVX2) || defined(BSOLVER_SHA_SHANI)
#include <immintrin.h>
#endif

namespace bsolver::hashimpl {

// Implementation tables, defined in the file that implements them (only referenced when compiled in).
extern const Hash160Batch kScalar;
extern const Hash160Batch kSse2;
extern const Hash160Batch kAvx2;
extern const Hash160Batch kShaNi;
extern const Hash160Batch kNeon;
extern const Hash160Batch kArmSha2;

namespace {

// ---- SIMD lane types ------------------------------------------------------------------------------------------------

#if defined(BSOLVER_LANES_NEON)
struct Neon8 {
  uint32x4_t lo, hi;
  static constexpr int kLanes = 8;
  static BSOLVER_HASH_INLINE Neon8 splat(uint32_t x) { return {vdupq_n_u32(x), vdupq_n_u32(x)}; }
  static BSOLVER_HASH_INLINE Neon8 load(const uint32_t* p) { return {vld1q_u32(p), vld1q_u32(p + 4)}; }
  BSOLVER_HASH_INLINE void store(uint32_t* p) const {
    vst1q_u32(p, lo);
    vst1q_u32(p + 4, hi);
  }
};
BSOLVER_HASH_INLINE Neon8 operator+(Neon8 a, Neon8 b) { return {vaddq_u32(a.lo, b.lo), vaddq_u32(a.hi, b.hi)}; }
BSOLVER_HASH_INLINE Neon8 operator^(Neon8 a, Neon8 b) { return {veorq_u32(a.lo, b.lo), veorq_u32(a.hi, b.hi)}; }
BSOLVER_HASH_INLINE Neon8 bitSelect(Neon8 m, Neon8 a, Neon8 b) {
  return {vbslq_u32(m.lo, a.lo, b.lo), vbslq_u32(m.hi, a.hi, b.hi)};
}
BSOLVER_HASH_INLINE Neon8 operator|(Neon8 a, Neon8 b) { return {vorrq_u32(a.lo, b.lo), vorrq_u32(a.hi, b.hi)}; }
BSOLVER_HASH_INLINE Neon8 orn(Neon8 a, Neon8 b) { return {vornq_u32(a.lo, b.lo), vornq_u32(a.hi, b.hi)}; }
template <int S>
BSOLVER_HASH_INLINE Neon8 shl(Neon8 a) {
  return {vshlq_n_u32(a.lo, S), vshlq_n_u32(a.hi, S)};
}
template <int S>
BSOLVER_HASH_INLINE Neon8 shr(Neon8 a) {
  return {vshrq_n_u32(a.lo, S), vshrq_n_u32(a.hi, S)};
}
template <int S>
BSOLVER_HASH_INLINE Neon8 rotl(Neon8 a) {
  return {vsriq_n_u32(vshlq_n_u32(a.lo, S), a.lo, 32 - S), vsriq_n_u32(vshlq_n_u32(a.hi, S), a.hi, 32 - S)};
}
BSOLVER_HASH_INLINE uint32x4_t bswapQ(uint32x4_t v) { return vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(v))); }
BSOLVER_HASH_INLINE Neon8 bswap(Neon8 a) { return {bswapQ(a.lo), bswapQ(a.hi)}; }

// In-register 4x4 transpose of 32-bit elements (row i of the result = column i of the input).
BSOLVER_HASH_INLINE void transpose4(uint32x4_t& r0, uint32x4_t& r1, uint32x4_t& r2, uint32x4_t& r3) {
  const uint64x2_t t0 = vreinterpretq_u64_u32(vtrn1q_u32(r0, r1)), t1 = vreinterpretq_u64_u32(vtrn2q_u32(r0, r1));
  const uint64x2_t t2 = vreinterpretq_u64_u32(vtrn1q_u32(r2, r3)), t3 = vreinterpretq_u64_u32(vtrn2q_u32(r2, r3));
  r0 = vreinterpretq_u32_u64(vtrn1q_u64(t0, t2));
  r1 = vreinterpretq_u32_u64(vtrn1q_u64(t1, t3));
  r2 = vreinterpretq_u32_u64(vtrn2q_u64(t0, t2));
  r3 = vreinterpretq_u32_u64(vtrn2q_u64(t1, t3));
}
#endif

#if defined(BSOLVER_LANES_SSE2) || defined(BSOLVER_LANES_AVX2) || defined(BSOLVER_SHA_SHANI)
// In-register 4x4 transpose of 32-bit elements (row i of the result = column i of the input).
BSOLVER_HASH_INLINE void transpose4(__m128i& r0, __m128i& r1, __m128i& r2, __m128i& r3) {
  const __m128i t0 = _mm_unpacklo_epi32(r0, r1), t1 = _mm_unpacklo_epi32(r2, r3);
  const __m128i t2 = _mm_unpackhi_epi32(r0, r1), t3 = _mm_unpackhi_epi32(r2, r3);
  r0 = _mm_unpacklo_epi64(t0, t1);
  r1 = _mm_unpackhi_epi64(t0, t1);
  r2 = _mm_unpacklo_epi64(t2, t3);
  r3 = _mm_unpackhi_epi64(t2, t3);
}
#endif

#if defined(BSOLVER_LANES_SSE2)
struct Sse8 {
  __m128i lo, hi;
  static constexpr int kLanes = 8;
  static BSOLVER_HASH_INLINE Sse8 splat(uint32_t x) {
    const __m128i v = _mm_set1_epi32(static_cast<int>(x));
    return {v, v};
  }
  static BSOLVER_HASH_INLINE Sse8 load(const uint32_t* p) {
    return {_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)),
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 4))};
  }
  static BSOLVER_HASH_INLINE Sse8 fromHalves(__m128i lo, __m128i hi) { return {lo, hi}; }
  BSOLVER_HASH_INLINE void store(uint32_t* p) const {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p), lo);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(p + 4), hi);
  }
};
BSOLVER_HASH_INLINE Sse8 operator+(Sse8 a, Sse8 b) { return {_mm_add_epi32(a.lo, b.lo), _mm_add_epi32(a.hi, b.hi)}; }
BSOLVER_HASH_INLINE Sse8 operator^(Sse8 a, Sse8 b) { return {_mm_xor_si128(a.lo, b.lo), _mm_xor_si128(a.hi, b.hi)}; }
BSOLVER_HASH_INLINE Sse8 bitSelect(Sse8 m, Sse8 a, Sse8 b) {
  return {_mm_xor_si128(_mm_and_si128(_mm_xor_si128(a.lo, b.lo), m.lo), b.lo),
          _mm_xor_si128(_mm_and_si128(_mm_xor_si128(a.hi, b.hi), m.hi), b.hi)};
}
BSOLVER_HASH_INLINE Sse8 operator|(Sse8 a, Sse8 b) { return {_mm_or_si128(a.lo, b.lo), _mm_or_si128(a.hi, b.hi)}; }
BSOLVER_HASH_INLINE Sse8 orn(Sse8 a, Sse8 b) {
  const __m128i ones = _mm_set1_epi32(-1);
  return {_mm_or_si128(a.lo, _mm_xor_si128(b.lo, ones)), _mm_or_si128(a.hi, _mm_xor_si128(b.hi, ones))};
}
template <int S>
BSOLVER_HASH_INLINE Sse8 shl(Sse8 a) {
  return {_mm_slli_epi32(a.lo, S), _mm_slli_epi32(a.hi, S)};
}
template <int S>
BSOLVER_HASH_INLINE Sse8 shr(Sse8 a) {
  return {_mm_srli_epi32(a.lo, S), _mm_srli_epi32(a.hi, S)};
}
template <int S>
BSOLVER_HASH_INLINE Sse8 rotl(Sse8 a) {
  return shl<S>(a) | shr<32 - S>(a);
}
// SSE2 has no byte shuffle: swap the bytes of each 16-bit half, then the halves.
BSOLVER_HASH_INLINE __m128i bswapQ(__m128i v) {
  v = _mm_or_si128(_mm_slli_epi16(v, 8), _mm_srli_epi16(v, 8));
  return _mm_shufflehi_epi16(_mm_shufflelo_epi16(v, 0xB1), 0xB1);
}
BSOLVER_HASH_INLINE Sse8 bswap(Sse8 a) { return {bswapQ(a.lo), bswapQ(a.hi)}; }
#endif

#if defined(BSOLVER_LANES_AVX2)
struct Avx8 {
  __m256i v;
  static constexpr int kLanes = 8;
  static BSOLVER_HASH_INLINE Avx8 splat(uint32_t x) { return {_mm256_set1_epi32(static_cast<int>(x))}; }
  static BSOLVER_HASH_INLINE Avx8 load(const uint32_t* p) {
    return {_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))};
  }
  static BSOLVER_HASH_INLINE Avx8 fromHalves(__m128i lo, __m128i hi) {
    return {_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)};
  }
  BSOLVER_HASH_INLINE void store(uint32_t* p) const { _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v); }
};
BSOLVER_HASH_INLINE Avx8 operator+(Avx8 a, Avx8 b) { return {_mm256_add_epi32(a.v, b.v)}; }
BSOLVER_HASH_INLINE Avx8 operator^(Avx8 a, Avx8 b) { return {_mm256_xor_si256(a.v, b.v)}; }
BSOLVER_HASH_INLINE Avx8 bitSelect(Avx8 m, Avx8 a, Avx8 b) {
  return {_mm256_xor_si256(_mm256_and_si256(_mm256_xor_si256(a.v, b.v), m.v), b.v)};
}
BSOLVER_HASH_INLINE Avx8 operator|(Avx8 a, Avx8 b) { return {_mm256_or_si256(a.v, b.v)}; }
BSOLVER_HASH_INLINE Avx8 orn(Avx8 a, Avx8 b) {
  return {_mm256_or_si256(a.v, _mm256_xor_si256(b.v, _mm256_set1_epi32(-1)))};
}
template <int S>
BSOLVER_HASH_INLINE Avx8 shl(Avx8 a) {
  return {_mm256_slli_epi32(a.v, S)};
}
template <int S>
BSOLVER_HASH_INLINE Avx8 shr(Avx8 a) {
  return {_mm256_srli_epi32(a.v, S)};
}
template <int S>
BSOLVER_HASH_INLINE Avx8 rotl(Avx8 a) {
  return {_mm256_or_si256(_mm256_slli_epi32(a.v, S), _mm256_srli_epi32(a.v, 32 - S))};
}
BSOLVER_HASH_INLINE Avx8 bswap(Avx8 a) {
  const __m256i mask = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4,
                                        11, 10, 9, 8, 15, 14, 13, 12);
  return {_mm256_shuffle_epi8(a.v, mask)};
}
#endif

// ---- Messages and the lane-generic pipeline -------------------------------------------------------------------------

// SHA-256 block of the compressed encoding (2 + parity) || x, x as 8 big-endian words.
template <class V>
BSOLVER_HASH_INLINE void compressedBlock(const V (&x)[8], V parity, V (&w)[16]) {
  w[0] = shl<24>(parity + V::splat(2)) | shr<8>(x[0]);
  for (int i = 1; i < 8; ++i) w[i] = shl<24>(x[i - 1]) | shr<8>(x[i]);
  w[8] = shl<24>(x[7]) | V::splat(0x800000);
  for (int i = 9; i < 15; ++i) w[i] = V::splat(0);
  w[15] = V::splat(33 * 8);
}

// The two SHA-256 blocks of the uncompressed encoding 04 || x || y.
template <class V>
BSOLVER_HASH_INLINE void uncompressedBlocks(const V (&x)[8], const V (&y)[8], V (&w1)[16], V (&w2)[16]) {
  w1[0] = V::splat(0x04000000) | shr<8>(x[0]);
  for (int i = 1; i < 8; ++i) w1[i] = shl<24>(x[i - 1]) | shr<8>(x[i]);
  w1[8] = shl<24>(x[7]) | shr<8>(y[0]);
  for (int i = 0; i < 7; ++i) w1[9 + i] = shl<24>(y[i]) | shr<8>(y[i + 1]);
  w2[0] = shl<24>(y[7]) | V::splat(0x800000);
  for (int i = 1; i < 15; ++i) w2[i] = V::splat(0);
  w2[15] = V::splat(65 * 8);
}

// RIPEMD-160 of the 32-byte SHA-256 digests in sha (state words, i.e. big-endian), stored to out lanes [g, g+kLanes).
template <class V>
BSOLVER_HASH_INLINE void ripemdOfSha(const V (&sha)[8], HashLaneDigests& out, int g) {
  V x[16];
  for (int i = 0; i < 8; ++i) x[i] = bswap(sha[i]);
  x[8] = V::splat(0x80);
  for (int i = 9; i < 16; ++i) x[i] = V::splat(0);
  x[14] = V::splat(32 * 8);
  V st[5];
  rmd160Init(st);
  rmd160Compress(st, x);
  for (int i = 0; i < 5; ++i) st[i].store(&out[i][g]);
}

template <class V>
BSOLVER_HASH_INLINE void loadWords(const HashLaneWords& in, int g, V (&v)[8]) {
  for (int i = 0; i < 8; ++i) v[i] = V::load(&in[i][g]);
}

// Both hashes computed with lane type V (software SHA-256).
template <class V>
void lanesCompressed(const HashLaneWords& x, const uint32_t (&parity)[kHashLanes], HashLaneDigests& out) {
  for (int g = 0; g < static_cast<int>(kHashLanes); g += V::kLanes) {
    V xv[8], w[16], st[8];
    loadWords(x, g, xv);
    compressedBlock(xv, V::load(&parity[g]), w);
    sha256Init(st);
    sha256Compress(st, w);
    ripemdOfSha(st, out, g);
  }
}

template <class V>
void lanesUncompressed(const HashLaneWords& x, const HashLaneWords& y, HashLaneDigests& out) {
  for (int g = 0; g < static_cast<int>(kHashLanes); g += V::kLanes) {
    V xv[8], yv[8], w1[16], w2[16], st[8];
    loadWords(x, g, xv);
    loadWords(y, g, yv);
    uncompressedBlocks(xv, yv, w1, w2);
    sha256Init(st);
    sha256Compress(st, w1);
    sha256Compress(st, w2);
    ripemdOfSha(st, out, g);
  }
}

// Stream-major message words for the hardware SHA back ends (built with the scalar lane, one lane at a time).
BSOLVER_HASH_INLINE void laneCompressedBlock(const HashLaneWords& x, const uint32_t (&parity)[kHashLanes], int l,
                                             uint32_t (&w)[16]) {
  Lane1 xv[8], wv[16];
  for (int i = 0; i < 8; ++i) xv[i] = {x[i][l]};
  compressedBlock(xv, Lane1{parity[l]}, wv);
  for (int i = 0; i < 16; ++i) w[i] = wv[i].v;
}

BSOLVER_HASH_INLINE void laneUncompressedBlocks(const HashLaneWords& x, const HashLaneWords& y, int l,
                                                uint32_t (&w1)[16], uint32_t (&w2)[16]) {
  Lane1 xv[8], yv[8], v1[16], v2[16];
  for (int i = 0; i < 8; ++i) {
    xv[i] = {x[i][l]};
    yv[i] = {y[i][l]};
  }
  uncompressedBlocks(xv, yv, v1, v2);
  for (int i = 0; i < 16; ++i) {
    w1[i] = v1[i].v;
    w2[i] = v2[i].v;
  }
}

// ---- ARMv8 SHA-256 crypto extension ---------------------------------------------------------------------------------

#if defined(BSOLVER_SHA_ARMV8)
// N interleaved streams. abcd/efgh hold each stream's state; m its 16 message words as 4 quads (clobbered).
template <int Q>
BSOLVER_HASH_INLINE void armShaQuad(uint32x4_t& abcd, uint32x4_t& efgh, uint32x4_t (&m)[4], uint32x4_t k) {
  const uint32x4_t wk = vaddq_u32(m[Q & 3], k);
  if constexpr (Q < 12) {
    m[Q & 3] = vsha256su1q_u32(vsha256su0q_u32(m[Q & 3], m[(Q + 1) & 3]), m[(Q + 2) & 3], m[(Q + 3) & 3]);
  }
  const uint32x4_t t = abcd;
  abcd = vsha256hq_u32(abcd, efgh, wk);
  efgh = vsha256h2q_u32(efgh, t, wk);
}

template <int Q, int N, int... S>
BSOLVER_HASH_INLINE void armShaQuadStreams(uint32x4_t (&abcd)[N], uint32x4_t (&efgh)[N], uint32x4_t (&m)[N][4],
                                           std::integer_sequence<int, S...>) {
  const uint32x4_t k = vld1q_u32(kSha256K + 4 * Q);
  (armShaQuad<Q>(abcd[S], efgh[S], m[S], k), ...);
}

template <int N, int... Q>
BSOLVER_HASH_INLINE void armShaRounds(uint32x4_t (&abcd)[N], uint32x4_t (&efgh)[N], uint32x4_t (&m)[N][4],
                                      std::integer_sequence<int, Q...>) {
  (armShaQuadStreams<Q>(abcd, efgh, m, std::make_integer_sequence<int, N>{}), ...);
}

template <int N>
BSOLVER_HASH_INLINE void armSha256Compress(uint32x4_t (&abcd)[N], uint32x4_t (&efgh)[N], uint32x4_t (&m)[N][4]) {
  uint32x4_t a0[N], e0[N];
  for (int s = 0; s < N; ++s) {
    a0[s] = abcd[s];
    e0[s] = efgh[s];
  }
  armShaRounds(abcd, efgh, m, std::make_integer_sequence<int, 16>{});
  for (int s = 0; s < N; ++s) {
    abcd[s] = vaddq_u32(abcd[s], a0[s]);
    efgh[s] = vaddq_u32(efgh[s], e0[s]);
  }
}

BSOLVER_HASH_INLINE void armLoadBlock(const uint32_t (&w)[16], uint32x4_t (&m)[4]) {
  for (int q = 0; q < 4; ++q) m[q] = vld1q_u32(w + 4 * q);
}

// Streams 0-3 of abcd/efgh become lanes 0-3 of the per-word vectors, streams 4-7 lanes 4-7.
BSOLVER_HASH_INLINE void armStatesToLanes(uint32x4_t (&abcd)[8], uint32x4_t (&efgh)[8], Neon8 (&sha)[8]) {
  transpose4(abcd[0], abcd[1], abcd[2], abcd[3]);
  transpose4(abcd[4], abcd[5], abcd[6], abcd[7]);
  transpose4(efgh[0], efgh[1], efgh[2], efgh[3]);
  transpose4(efgh[4], efgh[5], efgh[6], efgh[7]);
  for (int i = 0; i < 4; ++i) {
    sha[i] = {abcd[i], abcd[4 + i]};
    sha[4 + i] = {efgh[i], efgh[4 + i]};
  }
}

// kArmStreams streams are hashed in lockstep; the rest of the 8 in further passes.
constexpr int kArmStreams = 8;

void armSha2Compressed(const HashLaneWords& x, const uint32_t (&parity)[kHashLanes], HashLaneDigests& out) {
  uint32x4_t abcd[8], efgh[8];
  for (int base = 0; base < 8; base += kArmStreams) {
    uint32x4_t a[kArmStreams], e[kArmStreams], m[kArmStreams][4];
    for (int s = 0; s < kArmStreams; ++s) {
      uint32_t w[16];
      laneCompressedBlock(x, parity, base + s, w);
      armLoadBlock(w, m[s]);
      a[s] = vld1q_u32(kSha256Init);
      e[s] = vld1q_u32(kSha256Init + 4);
    }
    armSha256Compress(a, e, m);
    for (int s = 0; s < kArmStreams; ++s) {
      abcd[base + s] = a[s];
      efgh[base + s] = e[s];
    }
  }
  Neon8 sha[8];
  armStatesToLanes(abcd, efgh, sha);
  ripemdOfSha(sha, out, 0);
}

void armSha2Uncompressed(const HashLaneWords& x, const HashLaneWords& y, HashLaneDigests& out) {
  uint32x4_t abcd[8], efgh[8];
  for (int base = 0; base < 8; base += kArmStreams) {
    uint32x4_t a[kArmStreams], e[kArmStreams], m1[kArmStreams][4], m2[kArmStreams][4];
    for (int s = 0; s < kArmStreams; ++s) {
      uint32_t w1[16], w2[16];
      laneUncompressedBlocks(x, y, base + s, w1, w2);
      armLoadBlock(w1, m1[s]);
      armLoadBlock(w2, m2[s]);
      a[s] = vld1q_u32(kSha256Init);
      e[s] = vld1q_u32(kSha256Init + 4);
    }
    armSha256Compress(a, e, m1);
    armSha256Compress(a, e, m2);
    for (int s = 0; s < kArmStreams; ++s) {
      abcd[base + s] = a[s];
      efgh[base + s] = e[s];
    }
  }
  Neon8 sha[8];
  armStatesToLanes(abcd, efgh, sha);
  ripemdOfSha(sha, out, 0);
}
#endif

// ---- x86 SHA extensions ---------------------------------------------------------------------------------------------

#if defined(BSOLVER_SHA_SHANI)
// States use the SHA-NI register layout: abef = {f, e, b, a} and cdgh = {h, g, d, c} (element 0 first).
template <int Q>
BSOLVER_HASH_INLINE void shaNiQuad(__m128i& abef, __m128i& cdgh, __m128i (&m)[4], __m128i k) {
  __m128i wk = _mm_add_epi32(m[Q & 3], k);
  if constexpr (Q < 12) {
    const __m128i w7 = _mm_alignr_epi8(m[(Q + 3) & 3], m[(Q + 2) & 3], 4);
    m[Q & 3] = _mm_sha256msg2_epu32(_mm_add_epi32(_mm_sha256msg1_epu32(m[Q & 3], m[(Q + 1) & 3]), w7), m[(Q + 3) & 3]);
  }
  cdgh = _mm_sha256rnds2_epu32(cdgh, abef, wk);
  wk = _mm_shuffle_epi32(wk, 0x0E);
  abef = _mm_sha256rnds2_epu32(abef, cdgh, wk);
}

template <int Q, int N, int... S>
BSOLVER_HASH_INLINE void shaNiQuadStreams(__m128i (&abef)[N], __m128i (&cdgh)[N], __m128i (&m)[N][4],
                                          std::integer_sequence<int, S...>) {
  const __m128i k = _mm_load_si128(reinterpret_cast<const __m128i*>(kSha256K + 4 * Q));
  (shaNiQuad<Q>(abef[S], cdgh[S], m[S], k), ...);
}

template <int N, int... Q>
BSOLVER_HASH_INLINE void shaNiRounds(__m128i (&abef)[N], __m128i (&cdgh)[N], __m128i (&m)[N][4],
                                     std::integer_sequence<int, Q...>) {
  (shaNiQuadStreams<Q>(abef, cdgh, m, std::make_integer_sequence<int, N>{}), ...);
}

template <int N>
BSOLVER_HASH_INLINE void shaNiCompress(__m128i (&abef)[N], __m128i (&cdgh)[N], __m128i (&m)[N][4]) {
  __m128i a0[N], c0[N];
  for (int s = 0; s < N; ++s) {
    a0[s] = abef[s];
    c0[s] = cdgh[s];
  }
  shaNiRounds(abef, cdgh, m, std::make_integer_sequence<int, 16>{});
  for (int s = 0; s < N; ++s) {
    abef[s] = _mm_add_epi32(abef[s], a0[s]);
    cdgh[s] = _mm_add_epi32(cdgh[s], c0[s]);
  }
}

BSOLVER_HASH_INLINE void shaNiInit(__m128i& abef, __m128i& cdgh) {
  const uint32_t* h = kSha256Init;
  abef = _mm_set_epi32(static_cast<int>(h[0]), static_cast<int>(h[1]), static_cast<int>(h[4]), static_cast<int>(h[5]));
  cdgh = _mm_set_epi32(static_cast<int>(h[2]), static_cast<int>(h[3]), static_cast<int>(h[6]), static_cast<int>(h[7]));
}

BSOLVER_HASH_INLINE void shaNiLoadBlock(const uint32_t (&w)[16], __m128i (&m)[4]) {
  for (int q = 0; q < 4; ++q) m[q] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 4 * q));
}

// SHA-NI layout -> {a, b, c, d} and {e, f, g, h}.
BSOLVER_HASH_INLINE void shaNiToAbcd(__m128i abef, __m128i cdgh, __m128i& abcd, __m128i& efgh) {
  const __m128i feba = _mm_shuffle_epi32(abef, 0x1B);  // {a, b, e, f}
  const __m128i dchg = _mm_shuffle_epi32(cdgh, 0xB1);  // {g, h, c, d}
  abcd = _mm_blend_epi16(feba, dchg, 0xF0);
  efgh = _mm_alignr_epi8(dchg, feba, 8);
}

// Streams 0-3 become lanes 0-3 of the per-word vectors, streams 4-7 lanes 4-7.
template <class V>
BSOLVER_HASH_INLINE void shaNiStatesToLanes(const __m128i (&abef)[8], const __m128i (&cdgh)[8], V (&sha)[8]) {
  __m128i abcd[8], efgh[8];
  for (int s = 0; s < 8; ++s) shaNiToAbcd(abef[s], cdgh[s], abcd[s], efgh[s]);
  transpose4(abcd[0], abcd[1], abcd[2], abcd[3]);
  transpose4(abcd[4], abcd[5], abcd[6], abcd[7]);
  transpose4(efgh[0], efgh[1], efgh[2], efgh[3]);
  transpose4(efgh[4], efgh[5], efgh[6], efgh[7]);
  for (int i = 0; i < 4; ++i) {
    sha[i] = V::fromHalves(abcd[i], abcd[4 + i]);
    sha[4 + i] = V::fromHalves(efgh[i], efgh[4 + i]);
  }
}

// x86-64 has 16 vector registers; two streams (6 registers each) is the most that runs without spilling.
constexpr int kShaNiStreams = 2;

template <class V>
void shaNiCompressed(const HashLaneWords& x, const uint32_t (&parity)[kHashLanes], HashLaneDigests& out) {
  __m128i abef[8], cdgh[8];
  for (int base = 0; base < 8; base += kShaNiStreams) {
    __m128i a[kShaNiStreams], c[kShaNiStreams], m[kShaNiStreams][4];
    for (int s = 0; s < kShaNiStreams; ++s) {
      uint32_t w[16];
      laneCompressedBlock(x, parity, base + s, w);
      shaNiLoadBlock(w, m[s]);
      shaNiInit(a[s], c[s]);
    }
    shaNiCompress(a, c, m);
    for (int s = 0; s < kShaNiStreams; ++s) {
      abef[base + s] = a[s];
      cdgh[base + s] = c[s];
    }
  }
  V sha[8];
  shaNiStatesToLanes(abef, cdgh, sha);
  ripemdOfSha(sha, out, 0);
}

template <class V>
void shaNiUncompressed(const HashLaneWords& x, const HashLaneWords& y, HashLaneDigests& out) {
  __m128i abef[8], cdgh[8];
  for (int base = 0; base < 8; base += kShaNiStreams) {
    __m128i a[kShaNiStreams], c[kShaNiStreams], m1[kShaNiStreams][4], m2[kShaNiStreams][4];
    for (int s = 0; s < kShaNiStreams; ++s) {
      uint32_t w1[16], w2[16];
      laneUncompressedBlocks(x, y, base + s, w1, w2);
      shaNiLoadBlock(w1, m1[s]);
      shaNiLoadBlock(w2, m2[s]);
      shaNiInit(a[s], c[s]);
    }
    shaNiCompress(a, c, m1);
    shaNiCompress(a, c, m2);
    for (int s = 0; s < kShaNiStreams; ++s) {
      abef[base + s] = a[s];
      cdgh[base + s] = c[s];
    }
  }
  V sha[8];
  shaNiStatesToLanes(abef, cdgh, sha);
  ripemdOfSha(sha, out, 0);
}
#endif

}  // namespace
}  // namespace bsolver::hashimpl

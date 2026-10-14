/* Portable secp256k1 kernel core shared by the OpenCL, Metal and CUDA backends (C99 subset).
 *
 * The per-backend prelude defines:
 *   KFN            device function qualifier   (inline / inline / __device__ inline)
 *   PRIV           private address space       (empty / thread / empty)
 *   GLOB           global address space        (__global / device / empty)
 *   CONSTAS        constant address space      (__constant / constant / __constant__)
 *   MULHI64(a, b)  high 64 bits of a * b       (mul_hi / mulhi / __umul64hi)
 *   U64, U32       unsigned integer types
 *   ATOMIC_INC(p)  atomically increments the U32 at GLOB pointer p, yields the old value
 *
 * Field arithmetic mirrors src/core/field.h/.cpp exactly (reduction by 0x1000003D1, libsecp256k1
 * inversion chain); the scan loop mirrors PointEnumerator (src/core/point_enum.h). Host-side layouts
 * are documented in src/backend/gpu/gpu_common.h.
 */
#ifndef BSOLVER_KERNEL_CORE_H
#define BSOLVER_KERNEL_CORE_H

#define GPU_GROUP 64
#define GPU_HALF (GPU_GROUP / 2)
#define SCAN_HIT_CAP 4096
#define KPT 64

#define FE_C ((U64)0x1000003D1)

typedef struct {
  U64 d[4];
} fe_t;

KFN fe_t fe_load(GLOB const U64* p) {
  fe_t r;
  r.d[0] = p[0];
  r.d[1] = p[1];
  r.d[2] = p[2];
  r.d[3] = p[3];
  return r;
}

KFN void fe_store(GLOB U64* p, fe_t a) {
  p[0] = a.d[0];
  p[1] = a.d[1];
  p[2] = a.d[2];
  p[3] = a.d[3];
}

KFN fe_t fe_one(void) {
  fe_t r;
  r.d[0] = 1;
  r.d[1] = 0;
  r.d[2] = 0;
  r.d[3] = 0;
  return r;
}

KFN int fe_is_zero(fe_t a) { return (a.d[0] | a.d[1] | a.d[2] | a.d[3]) == 0; }

/* r - p when r >= p (r + C overflows 2^256), else r. */
KFN fe_t fe_final_reduce(fe_t r) {
  U64 t0 = r.d[0] + FE_C;
  U64 c = t0 < FE_C;
  U64 t1 = r.d[1] + c;
  c = t1 < c;
  U64 t2 = r.d[2] + c;
  c = t2 < c;
  U64 t3 = r.d[3] + c;
  c = t3 < c;
  U64 m = (U64)0 - c;
  r.d[0] = (t0 & m) | (r.d[0] & ~m);
  r.d[1] = (t1 & m) | (r.d[1] & ~m);
  r.d[2] = (t2 & m) | (r.d[2] & ~m);
  r.d[3] = (t3 & m) | (r.d[3] & ~m);
  return r;
}

KFN fe_t fe_add(fe_t a, fe_t b) {
  fe_t r;
  U64 c, s;
  s = a.d[0] + b.d[0];
  c = s < a.d[0];
  r.d[0] = s;
  s = a.d[1] + b.d[1];
  U64 c1 = s < a.d[1];
  r.d[1] = s + c;
  c = c1 | (r.d[1] < s);
  s = a.d[2] + b.d[2];
  c1 = s < a.d[2];
  r.d[2] = s + c;
  c = c1 | (r.d[2] < s);
  s = a.d[3] + b.d[3];
  c1 = s < a.d[3];
  r.d[3] = s + c;
  c = c1 | (r.d[3] < s);
  if (c) {
    /* a + b - p = (a + b - 2^256) + C; cannot carry again. */
    r.d[0] += FE_C;
    c = r.d[0] < FE_C;
    r.d[1] += c;
    c = r.d[1] < c;
    r.d[2] += c;
    c = r.d[2] < c;
    r.d[3] += c;
    return r;
  }
  return fe_final_reduce(r);
}

KFN fe_t fe_sub(fe_t a, fe_t b) {
  fe_t r;
  U64 bw, d, b1;
  d = a.d[0] - b.d[0];
  bw = a.d[0] < b.d[0];
  r.d[0] = d;
  d = a.d[1] - b.d[1];
  b1 = a.d[1] < b.d[1];
  r.d[1] = d - bw;
  bw = b1 | (d < bw);
  d = a.d[2] - b.d[2];
  b1 = a.d[2] < b.d[2];
  r.d[2] = d - bw;
  bw = b1 | (d < bw);
  d = a.d[3] - b.d[3];
  b1 = a.d[3] < b.d[3];
  r.d[3] = d - bw;
  bw = b1 | (d < bw);
  if (bw) {
    /* a - b + p = (a - b + 2^256) - C */
    U64 c = r.d[0] < FE_C;
    r.d[0] -= FE_C;
    U64 t = r.d[1];
    r.d[1] = t - c;
    c = t < c;
    t = r.d[2];
    r.d[2] = t - c;
    c = t < c;
    r.d[3] -= c;
  }
  return r;
}

KFN fe_t fe_neg(fe_t a) {
  fe_t z;
  z.d[0] = 0;
  z.d[1] = 0;
  z.d[2] = 0;
  z.d[3] = 0;
  return fe_sub(z, a);
}

/* Reduces the 512-bit value t[0..7] modulo p (same steps as fe_detail::reduce512). */
KFN fe_t fe_reduce512(PRIV const U64* t) {
  U64 r[4];
  U64 carry = 0;
  for (int i = 0; i < 4; ++i) {
    U64 lo = t[4 + i] * FE_C;
    U64 hi = MULHI64(t[4 + i], FE_C);
    U64 s = t[i] + lo;
    U64 c1 = s < lo;
    s += carry;
    U64 c2 = s < carry;
    r[i] = s;
    carry = hi + c1 + c2;
  }
  /* carry < 2^34; fold carry * C (< 2^67) back in. */
  U64 lo = carry * FE_C;
  U64 hi = MULHI64(carry, FE_C);
  U64 c, s;
  r[0] += lo;
  c = r[0] < lo;
  s = r[1] + hi;
  U64 c1 = s < hi;
  r[1] = s + c;
  c = c1 | (r[1] < s);
  r[2] += c;
  c = r[2] < c;
  r[3] += c;
  c = r[3] < c;
  if (c) {
    r[0] += FE_C;
    c = r[0] < FE_C;
    r[1] += c;
    c = r[1] < c;
    r[2] += c;
    c = r[2] < c;
    r[3] += c;
  }
  fe_t out;
  out.d[0] = r[0];
  out.d[1] = r[1];
  out.d[2] = r[2];
  out.d[3] = r[3];
  return fe_final_reduce(out);
}

KFN fe_t fe_mul(fe_t a, fe_t b) {
  U64 t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) {
    U64 carry = 0;
    for (int j = 0; j < 4; ++j) {
      U64 lo = a.d[i] * b.d[j];
      U64 hi = MULHI64(a.d[i], b.d[j]);
      U64 s = t[i + j] + lo;
      hi += s < lo;
      s += carry;
      hi += s < carry;
      t[i + j] = s;
      carry = hi;
    }
    t[i + 4] = carry;
  }
  return fe_reduce512(t);
}

KFN fe_t fe_sqr(fe_t a) {
  /* Off-diagonal products once, doubled, plus the squares on the diagonal. */
  U64 t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    U64 carry = 0;
    for (int j = i + 1; j < 4; ++j) {
      U64 lo = a.d[i] * a.d[j];
      U64 hi = MULHI64(a.d[i], a.d[j]);
      U64 s = t[i + j] + lo;
      hi += s < lo;
      s += carry;
      hi += s < carry;
      t[i + j] = s;
      carry = hi;
    }
    t[i + 4] = carry;
  }
  U64 top = 0;
  for (int i = 0; i < 8; ++i) {
    U64 v = t[i];
    t[i] = (v << 1) | top;
    top = v >> 63;
  }
  U64 c = 0;
  for (int i = 0; i < 4; ++i) {
    U64 lo = a.d[i] * a.d[i];
    U64 hi = MULHI64(a.d[i], a.d[i]);
    U64 s = t[2 * i] + lo;
    U64 c1 = s < lo;
    t[2 * i] = s + c;
    c = c1 | (t[2 * i] < s);
    s = t[2 * i + 1] + hi;
    c1 = s < hi;
    t[2 * i + 1] = s + c;
    c = c1 | (t[2 * i + 1] < s);
  }
  return fe_reduce512(t);
}

KFN fe_t fe_sqr_n(fe_t a, int n) {
  for (int i = 0; i < n; ++i) a = fe_sqr(a);
  return a;
}

/* a^(p-2) with libsecp256k1's addition chain (255 squarings, 15 multiplications); 0 -> 0. */
KFN fe_t fe_inv(fe_t a) {
  fe_t x2 = fe_mul(fe_sqr(a), a);
  fe_t x3 = fe_mul(fe_sqr(x2), a);
  fe_t x6 = fe_mul(fe_sqr_n(x3, 3), x3);
  fe_t x9 = fe_mul(fe_sqr_n(x6, 3), x3);
  fe_t x11 = fe_mul(fe_sqr_n(x9, 2), x2);
  fe_t x22 = fe_mul(fe_sqr_n(x11, 11), x11);
  fe_t x44 = fe_mul(fe_sqr_n(x22, 22), x22);
  fe_t x88 = fe_mul(fe_sqr_n(x44, 44), x44);
  fe_t x176 = fe_mul(fe_sqr_n(x88, 88), x88);
  fe_t x220 = fe_mul(fe_sqr_n(x176, 44), x44);
  fe_t x223 = fe_mul(fe_sqr_n(x220, 3), x3);
  fe_t t = fe_mul(fe_sqr_n(x223, 23), x22);
  t = fe_mul(fe_sqr_n(t, 5), a);
  t = fe_mul(fe_sqr_n(t, 3), x2);
  t = fe_mul(fe_sqr_n(t, 2), a);
  return t;
}

/* Affine P + Q given inv = 1 / (Q.x - P.x). */
KFN void pt_add(fe_t px, fe_t py, fe_t qx, fe_t qy, fe_t inv, PRIV fe_t* rx, PRIV fe_t* ry) {
  fe_t lam = fe_mul(fe_sub(qy, py), inv);
  fe_t x = fe_sub(fe_sub(fe_sqr(lam), px), qx);
  *ry = fe_sub(fe_mul(lam, fe_sub(px, x)), py);
  *rx = x;
}

/* ---- hashing ---------------------------------------------------------------------------------- */

#define ROR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define ROL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define BSWAP32(x) (((x) >> 24) | (((x) >> 8) & 0xff00u) | (((x) << 8) & 0xff0000u) | ((x) << 24))

CONSTAS U32 kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

#define SHA_S0(x) (ROR32(x, 2) ^ ROR32(x, 13) ^ ROR32(x, 22))
#define SHA_S1(x) (ROR32(x, 6) ^ ROR32(x, 11) ^ ROR32(x, 25))
#define SHA_s0(x) (ROR32(x, 7) ^ ROR32(x, 18) ^ ((x) >> 3))
#define SHA_s1(x) (ROR32(x, 17) ^ ROR32(x, 19) ^ ((x) >> 10))
#define SHA_CH(e, f, g) ((g) ^ ((e) & ((f) ^ (g))))
#define SHA_MAJ(a, b, c) (((a) & (b)) | ((c) & ((a) | (b))))
#define SHA_RND(a, b, c, d, e, f, g, h, k, w)                 \
  {                                                         \
    U32 t1_ = (h) + SHA_S1(e) + SHA_CH(e, f, g) + (k) + (w); \
    U32 t2_ = SHA_S0(a) + SHA_MAJ(a, b, c);                 \
    (d) += t1_;                                             \
    (h) = t1_ + t2_;                                        \
  }
#define SHA_W(w0, w14, w9, w1) ((w0) += SHA_s1(w14) + (w9) + SHA_s0(w1))

/* One SHA-256 compression of the big-endian message words w0..w15 into st[8]. */
KFN void sha256_block(PRIV U32* st, U32 w0, U32 w1, U32 w2, U32 w3, U32 w4, U32 w5, U32 w6, U32 w7, U32 w8,
                      U32 w9, U32 w10, U32 w11, U32 w12, U32 w13, U32 w14, U32 w15) {
  U32 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
  SHA_RND(a, b, c, d, e, f, g, h, kSha256K[0], w0);
  SHA_RND(h, a, b, c, d, e, f, g, kSha256K[1], w1);
  SHA_RND(g, h, a, b, c, d, e, f, kSha256K[2], w2);
  SHA_RND(f, g, h, a, b, c, d, e, kSha256K[3], w3);
  SHA_RND(e, f, g, h, a, b, c, d, kSha256K[4], w4);
  SHA_RND(d, e, f, g, h, a, b, c, kSha256K[5], w5);
  SHA_RND(c, d, e, f, g, h, a, b, kSha256K[6], w6);
  SHA_RND(b, c, d, e, f, g, h, a, kSha256K[7], w7);
  SHA_RND(a, b, c, d, e, f, g, h, kSha256K[8], w8);
  SHA_RND(h, a, b, c, d, e, f, g, kSha256K[9], w9);
  SHA_RND(g, h, a, b, c, d, e, f, kSha256K[10], w10);
  SHA_RND(f, g, h, a, b, c, d, e, kSha256K[11], w11);
  SHA_RND(e, f, g, h, a, b, c, d, kSha256K[12], w12);
  SHA_RND(d, e, f, g, h, a, b, c, kSha256K[13], w13);
  SHA_RND(c, d, e, f, g, h, a, b, kSha256K[14], w14);
  SHA_RND(b, c, d, e, f, g, h, a, kSha256K[15], w15);
  SHA_W(w0, w14, w9, w1);
  SHA_W(w1, w15, w10, w2);
  SHA_W(w2, w0, w11, w3);
  SHA_W(w3, w1, w12, w4);
  SHA_W(w4, w2, w13, w5);
  SHA_W(w5, w3, w14, w6);
  SHA_W(w6, w4, w15, w7);
  SHA_W(w7, w5, w0, w8);
  SHA_W(w8, w6, w1, w9);
  SHA_W(w9, w7, w2, w10);
  SHA_W(w10, w8, w3, w11);
  SHA_W(w11, w9, w4, w12);
  SHA_W(w12, w10, w5, w13);
  SHA_W(w13, w11, w6, w14);
  SHA_W(w14, w12, w7, w15);
  SHA_W(w15, w13, w8, w0);
  SHA_RND(a, b, c, d, e, f, g, h, kSha256K[16], w0);
  SHA_RND(h, a, b, c, d, e, f, g, kSha256K[17], w1);
  SHA_RND(g, h, a, b, c, d, e, f, kSha256K[18], w2);
  SHA_RND(f, g, h, a, b, c, d, e, kSha256K[19], w3);
  SHA_RND(e, f, g, h, a, b, c, d, kSha256K[20], w4);
  SHA_RND(d, e, f, g, h, a, b, c, kSha256K[21], w5);
  SHA_RND(c, d, e, f, g, h, a, b, kSha256K[22], w6);
  SHA_RND(b, c, d, e, f, g, h, a, kSha256K[23], w7);
  SHA_RND(a, b, c, d, e, f, g, h, kSha256K[24], w8);
  SHA_RND(h, a, b, c, d, e, f, g, kSha256K[25], w9);
  SHA_RND(g, h, a, b, c, d, e, f, kSha256K[26], w10);
  SHA_RND(f, g, h, a, b, c, d, e, kSha256K[27], w11);
  SHA_RND(e, f, g, h, a, b, c, d, kSha256K[28], w12);
  SHA_RND(d, e, f, g, h, a, b, c, kSha256K[29], w13);
  SHA_RND(c, d, e, f, g, h, a, b, kSha256K[30], w14);
  SHA_RND(b, c, d, e, f, g, h, a, kSha256K[31], w15);
  SHA_W(w0, w14, w9, w1);
  SHA_W(w1, w15, w10, w2);
  SHA_W(w2, w0, w11, w3);
  SHA_W(w3, w1, w12, w4);
  SHA_W(w4, w2, w13, w5);
  SHA_W(w5, w3, w14, w6);
  SHA_W(w6, w4, w15, w7);
  SHA_W(w7, w5, w0, w8);
  SHA_W(w8, w6, w1, w9);
  SHA_W(w9, w7, w2, w10);
  SHA_W(w10, w8, w3, w11);
  SHA_W(w11, w9, w4, w12);
  SHA_W(w12, w10, w5, w13);
  SHA_W(w13, w11, w6, w14);
  SHA_W(w14, w12, w7, w15);
  SHA_W(w15, w13, w8, w0);
  SHA_RND(a, b, c, d, e, f, g, h, kSha256K[32], w0);
  SHA_RND(h, a, b, c, d, e, f, g, kSha256K[33], w1);
  SHA_RND(g, h, a, b, c, d, e, f, kSha256K[34], w2);
  SHA_RND(f, g, h, a, b, c, d, e, kSha256K[35], w3);
  SHA_RND(e, f, g, h, a, b, c, d, kSha256K[36], w4);
  SHA_RND(d, e, f, g, h, a, b, c, kSha256K[37], w5);
  SHA_RND(c, d, e, f, g, h, a, b, kSha256K[38], w6);
  SHA_RND(b, c, d, e, f, g, h, a, kSha256K[39], w7);
  SHA_RND(a, b, c, d, e, f, g, h, kSha256K[40], w8);
  SHA_RND(h, a, b, c, d, e, f, g, kSha256K[41], w9);
  SHA_RND(g, h, a, b, c, d, e, f, kSha256K[42], w10);
  SHA_RND(f, g, h, a, b, c, d, e, kSha256K[43], w11);
  SHA_RND(e, f, g, h, a, b, c, d, kSha256K[44], w12);
  SHA_RND(d, e, f, g, h, a, b, c, kSha256K[45], w13);
  SHA_RND(c, d, e, f, g, h, a, b, kSha256K[46], w14);
  SHA_RND(b, c, d, e, f, g, h, a, kSha256K[47], w15);
  SHA_W(w0, w14, w9, w1);
  SHA_W(w1, w15, w10, w2);
  SHA_W(w2, w0, w11, w3);
  SHA_W(w3, w1, w12, w4);
  SHA_W(w4, w2, w13, w5);
  SHA_W(w5, w3, w14, w6);
  SHA_W(w6, w4, w15, w7);
  SHA_W(w7, w5, w0, w8);
  SHA_W(w8, w6, w1, w9);
  SHA_W(w9, w7, w2, w10);
  SHA_W(w10, w8, w3, w11);
  SHA_W(w11, w9, w4, w12);
  SHA_W(w12, w10, w5, w13);
  SHA_W(w13, w11, w6, w14);
  SHA_W(w14, w12, w7, w15);
  SHA_W(w15, w13, w8, w0);
  SHA_RND(a, b, c, d, e, f, g, h, kSha256K[48], w0);
  SHA_RND(h, a, b, c, d, e, f, g, kSha256K[49], w1);
  SHA_RND(g, h, a, b, c, d, e, f, kSha256K[50], w2);
  SHA_RND(f, g, h, a, b, c, d, e, kSha256K[51], w3);
  SHA_RND(e, f, g, h, a, b, c, d, kSha256K[52], w4);
  SHA_RND(d, e, f, g, h, a, b, c, kSha256K[53], w5);
  SHA_RND(c, d, e, f, g, h, a, b, kSha256K[54], w6);
  SHA_RND(b, c, d, e, f, g, h, a, kSha256K[55], w7);
  SHA_RND(a, b, c, d, e, f, g, h, kSha256K[56], w8);
  SHA_RND(h, a, b, c, d, e, f, g, kSha256K[57], w9);
  SHA_RND(g, h, a, b, c, d, e, f, kSha256K[58], w10);
  SHA_RND(f, g, h, a, b, c, d, e, kSha256K[59], w11);
  SHA_RND(e, f, g, h, a, b, c, d, kSha256K[60], w12);
  SHA_RND(d, e, f, g, h, a, b, c, kSha256K[61], w13);
  SHA_RND(c, d, e, f, g, h, a, b, kSha256K[62], w14);
  SHA_RND(b, c, d, e, f, g, h, a, kSha256K[63], w15);
  st[0] += a;
  st[1] += b;
  st[2] += c;
  st[3] += d;
  st[4] += e;
  st[5] += f;
  st[6] += g;
  st[7] += h;
}

KFN void sha256_init(PRIV U32* st) {
  st[0] = 0x6a09e667u;
  st[1] = 0xbb67ae85u;
  st[2] = 0x3c6ef372u;
  st[3] = 0xa54ff53au;
  st[4] = 0x510e527fu;
  st[5] = 0x9b05688cu;
  st[6] = 0x1f83d9abu;
  st[7] = 0x5be0cd19u;
}

#define RMD_F1(x, y, z) ((x) ^ (y) ^ (z))
#define RMD_F2(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define RMD_F3(x, y, z) (((x) | ~(y)) ^ (z))
#define RMD_F4(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define RMD_F5(x, y, z) ((x) ^ ((y) | ~(z)))
#define RMD_R(a, b, c, d, e, F, x, k, s) \
  {                                      \
    (a) += F(b, c, d) + (x) + (k);       \
    (a) = ROL32(a, s) + (e);             \
    (c) = ROL32(c, 10);                  \
  }

/* RIPEMD-160 of the 32-byte SHA-256 digest st[8]; out[5] = digest as big-endian words, so that
   comparing words in order equals comparing the 20 digest bytes lexicographically. */
KFN void ripemd160_of_sha(PRIV const U32* st, PRIV U32* out) {
  U32 x0 = BSWAP32(st[0]), x1 = BSWAP32(st[1]), x2 = BSWAP32(st[2]), x3 = BSWAP32(st[3]);
  U32 x4 = BSWAP32(st[4]), x5 = BSWAP32(st[5]), x6 = BSWAP32(st[6]), x7 = BSWAP32(st[7]);
  U32 x8 = 0x80u, x9 = 0, x10 = 0, x11 = 0, x12 = 0, x13 = 0, x14 = 256u, x15 = 0;
  U32 al = 0x67452301u, bl = 0xEFCDAB89u, cl = 0x98BADCFEu, dl = 0x10325476u, el = 0xC3D2E1F0u;
  U32 ar = al, br = bl, cr = cl, dr = dl, er = el;
  RMD_R(al, bl, cl, dl, el, RMD_F1, x0, 0x00000000u, 11);
  RMD_R(ar, br, cr, dr, er, RMD_F5, x5, 0x50A28BE6u, 8);
  RMD_R(el, al, bl, cl, dl, RMD_F1, x1, 0x00000000u, 14);
  RMD_R(er, ar, br, cr, dr, RMD_F5, x14, 0x50A28BE6u, 9);
  RMD_R(dl, el, al, bl, cl, RMD_F1, x2, 0x00000000u, 15);
  RMD_R(dr, er, ar, br, cr, RMD_F5, x7, 0x50A28BE6u, 9);
  RMD_R(cl, dl, el, al, bl, RMD_F1, x3, 0x00000000u, 12);
  RMD_R(cr, dr, er, ar, br, RMD_F5, x0, 0x50A28BE6u, 11);
  RMD_R(bl, cl, dl, el, al, RMD_F1, x4, 0x00000000u, 5);
  RMD_R(br, cr, dr, er, ar, RMD_F5, x9, 0x50A28BE6u, 13);
  RMD_R(al, bl, cl, dl, el, RMD_F1, x5, 0x00000000u, 8);
  RMD_R(ar, br, cr, dr, er, RMD_F5, x2, 0x50A28BE6u, 15);
  RMD_R(el, al, bl, cl, dl, RMD_F1, x6, 0x00000000u, 7);
  RMD_R(er, ar, br, cr, dr, RMD_F5, x11, 0x50A28BE6u, 15);
  RMD_R(dl, el, al, bl, cl, RMD_F1, x7, 0x00000000u, 9);
  RMD_R(dr, er, ar, br, cr, RMD_F5, x4, 0x50A28BE6u, 5);
  RMD_R(cl, dl, el, al, bl, RMD_F1, x8, 0x00000000u, 11);
  RMD_R(cr, dr, er, ar, br, RMD_F5, x13, 0x50A28BE6u, 7);
  RMD_R(bl, cl, dl, el, al, RMD_F1, x9, 0x00000000u, 13);
  RMD_R(br, cr, dr, er, ar, RMD_F5, x6, 0x50A28BE6u, 7);
  RMD_R(al, bl, cl, dl, el, RMD_F1, x10, 0x00000000u, 14);
  RMD_R(ar, br, cr, dr, er, RMD_F5, x15, 0x50A28BE6u, 8);
  RMD_R(el, al, bl, cl, dl, RMD_F1, x11, 0x00000000u, 15);
  RMD_R(er, ar, br, cr, dr, RMD_F5, x8, 0x50A28BE6u, 11);
  RMD_R(dl, el, al, bl, cl, RMD_F1, x12, 0x00000000u, 6);
  RMD_R(dr, er, ar, br, cr, RMD_F5, x1, 0x50A28BE6u, 14);
  RMD_R(cl, dl, el, al, bl, RMD_F1, x13, 0x00000000u, 7);
  RMD_R(cr, dr, er, ar, br, RMD_F5, x10, 0x50A28BE6u, 14);
  RMD_R(bl, cl, dl, el, al, RMD_F1, x14, 0x00000000u, 9);
  RMD_R(br, cr, dr, er, ar, RMD_F5, x3, 0x50A28BE6u, 12);
  RMD_R(al, bl, cl, dl, el, RMD_F1, x15, 0x00000000u, 8);
  RMD_R(ar, br, cr, dr, er, RMD_F5, x12, 0x50A28BE6u, 6);
  RMD_R(el, al, bl, cl, dl, RMD_F2, x7, 0x5A827999u, 7);
  RMD_R(er, ar, br, cr, dr, RMD_F4, x6, 0x5C4DD124u, 9);
  RMD_R(dl, el, al, bl, cl, RMD_F2, x4, 0x5A827999u, 6);
  RMD_R(dr, er, ar, br, cr, RMD_F4, x11, 0x5C4DD124u, 13);
  RMD_R(cl, dl, el, al, bl, RMD_F2, x13, 0x5A827999u, 8);
  RMD_R(cr, dr, er, ar, br, RMD_F4, x3, 0x5C4DD124u, 15);
  RMD_R(bl, cl, dl, el, al, RMD_F2, x1, 0x5A827999u, 13);
  RMD_R(br, cr, dr, er, ar, RMD_F4, x7, 0x5C4DD124u, 7);
  RMD_R(al, bl, cl, dl, el, RMD_F2, x10, 0x5A827999u, 11);
  RMD_R(ar, br, cr, dr, er, RMD_F4, x0, 0x5C4DD124u, 12);
  RMD_R(el, al, bl, cl, dl, RMD_F2, x6, 0x5A827999u, 9);
  RMD_R(er, ar, br, cr, dr, RMD_F4, x13, 0x5C4DD124u, 8);
  RMD_R(dl, el, al, bl, cl, RMD_F2, x15, 0x5A827999u, 7);
  RMD_R(dr, er, ar, br, cr, RMD_F4, x5, 0x5C4DD124u, 9);
  RMD_R(cl, dl, el, al, bl, RMD_F2, x3, 0x5A827999u, 15);
  RMD_R(cr, dr, er, ar, br, RMD_F4, x10, 0x5C4DD124u, 11);
  RMD_R(bl, cl, dl, el, al, RMD_F2, x12, 0x5A827999u, 7);
  RMD_R(br, cr, dr, er, ar, RMD_F4, x14, 0x5C4DD124u, 7);
  RMD_R(al, bl, cl, dl, el, RMD_F2, x0, 0x5A827999u, 12);
  RMD_R(ar, br, cr, dr, er, RMD_F4, x15, 0x5C4DD124u, 7);
  RMD_R(el, al, bl, cl, dl, RMD_F2, x9, 0x5A827999u, 15);
  RMD_R(er, ar, br, cr, dr, RMD_F4, x8, 0x5C4DD124u, 12);
  RMD_R(dl, el, al, bl, cl, RMD_F2, x5, 0x5A827999u, 9);
  RMD_R(dr, er, ar, br, cr, RMD_F4, x12, 0x5C4DD124u, 7);
  RMD_R(cl, dl, el, al, bl, RMD_F2, x2, 0x5A827999u, 11);
  RMD_R(cr, dr, er, ar, br, RMD_F4, x4, 0x5C4DD124u, 6);
  RMD_R(bl, cl, dl, el, al, RMD_F2, x14, 0x5A827999u, 7);
  RMD_R(br, cr, dr, er, ar, RMD_F4, x9, 0x5C4DD124u, 15);
  RMD_R(al, bl, cl, dl, el, RMD_F2, x11, 0x5A827999u, 13);
  RMD_R(ar, br, cr, dr, er, RMD_F4, x1, 0x5C4DD124u, 13);
  RMD_R(el, al, bl, cl, dl, RMD_F2, x8, 0x5A827999u, 12);
  RMD_R(er, ar, br, cr, dr, RMD_F4, x2, 0x5C4DD124u, 11);
  RMD_R(dl, el, al, bl, cl, RMD_F3, x3, 0x6ED9EBA1u, 11);
  RMD_R(dr, er, ar, br, cr, RMD_F3, x15, 0x6D703EF3u, 9);
  RMD_R(cl, dl, el, al, bl, RMD_F3, x10, 0x6ED9EBA1u, 13);
  RMD_R(cr, dr, er, ar, br, RMD_F3, x5, 0x6D703EF3u, 7);
  RMD_R(bl, cl, dl, el, al, RMD_F3, x14, 0x6ED9EBA1u, 6);
  RMD_R(br, cr, dr, er, ar, RMD_F3, x1, 0x6D703EF3u, 15);
  RMD_R(al, bl, cl, dl, el, RMD_F3, x4, 0x6ED9EBA1u, 7);
  RMD_R(ar, br, cr, dr, er, RMD_F3, x3, 0x6D703EF3u, 11);
  RMD_R(el, al, bl, cl, dl, RMD_F3, x9, 0x6ED9EBA1u, 14);
  RMD_R(er, ar, br, cr, dr, RMD_F3, x7, 0x6D703EF3u, 8);
  RMD_R(dl, el, al, bl, cl, RMD_F3, x15, 0x6ED9EBA1u, 9);
  RMD_R(dr, er, ar, br, cr, RMD_F3, x14, 0x6D703EF3u, 6);
  RMD_R(cl, dl, el, al, bl, RMD_F3, x8, 0x6ED9EBA1u, 13);
  RMD_R(cr, dr, er, ar, br, RMD_F3, x6, 0x6D703EF3u, 6);
  RMD_R(bl, cl, dl, el, al, RMD_F3, x1, 0x6ED9EBA1u, 15);
  RMD_R(br, cr, dr, er, ar, RMD_F3, x9, 0x6D703EF3u, 14);
  RMD_R(al, bl, cl, dl, el, RMD_F3, x2, 0x6ED9EBA1u, 14);
  RMD_R(ar, br, cr, dr, er, RMD_F3, x11, 0x6D703EF3u, 12);
  RMD_R(el, al, bl, cl, dl, RMD_F3, x7, 0x6ED9EBA1u, 8);
  RMD_R(er, ar, br, cr, dr, RMD_F3, x8, 0x6D703EF3u, 13);
  RMD_R(dl, el, al, bl, cl, RMD_F3, x0, 0x6ED9EBA1u, 13);
  RMD_R(dr, er, ar, br, cr, RMD_F3, x12, 0x6D703EF3u, 5);
  RMD_R(cl, dl, el, al, bl, RMD_F3, x6, 0x6ED9EBA1u, 6);
  RMD_R(cr, dr, er, ar, br, RMD_F3, x2, 0x6D703EF3u, 14);
  RMD_R(bl, cl, dl, el, al, RMD_F3, x13, 0x6ED9EBA1u, 5);
  RMD_R(br, cr, dr, er, ar, RMD_F3, x10, 0x6D703EF3u, 13);
  RMD_R(al, bl, cl, dl, el, RMD_F3, x11, 0x6ED9EBA1u, 12);
  RMD_R(ar, br, cr, dr, er, RMD_F3, x0, 0x6D703EF3u, 13);
  RMD_R(el, al, bl, cl, dl, RMD_F3, x5, 0x6ED9EBA1u, 7);
  RMD_R(er, ar, br, cr, dr, RMD_F3, x4, 0x6D703EF3u, 7);
  RMD_R(dl, el, al, bl, cl, RMD_F3, x12, 0x6ED9EBA1u, 5);
  RMD_R(dr, er, ar, br, cr, RMD_F3, x13, 0x6D703EF3u, 5);
  RMD_R(cl, dl, el, al, bl, RMD_F4, x1, 0x8F1BBCDCu, 11);
  RMD_R(cr, dr, er, ar, br, RMD_F2, x8, 0x7A6D76E9u, 15);
  RMD_R(bl, cl, dl, el, al, RMD_F4, x9, 0x8F1BBCDCu, 12);
  RMD_R(br, cr, dr, er, ar, RMD_F2, x6, 0x7A6D76E9u, 5);
  RMD_R(al, bl, cl, dl, el, RMD_F4, x11, 0x8F1BBCDCu, 14);
  RMD_R(ar, br, cr, dr, er, RMD_F2, x4, 0x7A6D76E9u, 8);
  RMD_R(el, al, bl, cl, dl, RMD_F4, x10, 0x8F1BBCDCu, 15);
  RMD_R(er, ar, br, cr, dr, RMD_F2, x1, 0x7A6D76E9u, 11);
  RMD_R(dl, el, al, bl, cl, RMD_F4, x0, 0x8F1BBCDCu, 14);
  RMD_R(dr, er, ar, br, cr, RMD_F2, x3, 0x7A6D76E9u, 14);
  RMD_R(cl, dl, el, al, bl, RMD_F4, x8, 0x8F1BBCDCu, 15);
  RMD_R(cr, dr, er, ar, br, RMD_F2, x11, 0x7A6D76E9u, 14);
  RMD_R(bl, cl, dl, el, al, RMD_F4, x12, 0x8F1BBCDCu, 9);
  RMD_R(br, cr, dr, er, ar, RMD_F2, x15, 0x7A6D76E9u, 6);
  RMD_R(al, bl, cl, dl, el, RMD_F4, x4, 0x8F1BBCDCu, 8);
  RMD_R(ar, br, cr, dr, er, RMD_F2, x0, 0x7A6D76E9u, 14);
  RMD_R(el, al, bl, cl, dl, RMD_F4, x13, 0x8F1BBCDCu, 9);
  RMD_R(er, ar, br, cr, dr, RMD_F2, x5, 0x7A6D76E9u, 6);
  RMD_R(dl, el, al, bl, cl, RMD_F4, x3, 0x8F1BBCDCu, 14);
  RMD_R(dr, er, ar, br, cr, RMD_F2, x12, 0x7A6D76E9u, 9);
  RMD_R(cl, dl, el, al, bl, RMD_F4, x7, 0x8F1BBCDCu, 5);
  RMD_R(cr, dr, er, ar, br, RMD_F2, x2, 0x7A6D76E9u, 12);
  RMD_R(bl, cl, dl, el, al, RMD_F4, x15, 0x8F1BBCDCu, 6);
  RMD_R(br, cr, dr, er, ar, RMD_F2, x13, 0x7A6D76E9u, 9);
  RMD_R(al, bl, cl, dl, el, RMD_F4, x14, 0x8F1BBCDCu, 8);
  RMD_R(ar, br, cr, dr, er, RMD_F2, x9, 0x7A6D76E9u, 12);
  RMD_R(el, al, bl, cl, dl, RMD_F4, x5, 0x8F1BBCDCu, 6);
  RMD_R(er, ar, br, cr, dr, RMD_F2, x7, 0x7A6D76E9u, 5);
  RMD_R(dl, el, al, bl, cl, RMD_F4, x6, 0x8F1BBCDCu, 5);
  RMD_R(dr, er, ar, br, cr, RMD_F2, x10, 0x7A6D76E9u, 15);
  RMD_R(cl, dl, el, al, bl, RMD_F4, x2, 0x8F1BBCDCu, 12);
  RMD_R(cr, dr, er, ar, br, RMD_F2, x14, 0x7A6D76E9u, 8);
  RMD_R(bl, cl, dl, el, al, RMD_F5, x4, 0xA953FD4Eu, 9);
  RMD_R(br, cr, dr, er, ar, RMD_F1, x12, 0x00000000u, 8);
  RMD_R(al, bl, cl, dl, el, RMD_F5, x0, 0xA953FD4Eu, 15);
  RMD_R(ar, br, cr, dr, er, RMD_F1, x15, 0x00000000u, 5);
  RMD_R(el, al, bl, cl, dl, RMD_F5, x5, 0xA953FD4Eu, 5);
  RMD_R(er, ar, br, cr, dr, RMD_F1, x10, 0x00000000u, 12);
  RMD_R(dl, el, al, bl, cl, RMD_F5, x9, 0xA953FD4Eu, 11);
  RMD_R(dr, er, ar, br, cr, RMD_F1, x4, 0x00000000u, 9);
  RMD_R(cl, dl, el, al, bl, RMD_F5, x7, 0xA953FD4Eu, 6);
  RMD_R(cr, dr, er, ar, br, RMD_F1, x1, 0x00000000u, 12);
  RMD_R(bl, cl, dl, el, al, RMD_F5, x12, 0xA953FD4Eu, 8);
  RMD_R(br, cr, dr, er, ar, RMD_F1, x5, 0x00000000u, 5);
  RMD_R(al, bl, cl, dl, el, RMD_F5, x2, 0xA953FD4Eu, 13);
  RMD_R(ar, br, cr, dr, er, RMD_F1, x8, 0x00000000u, 14);
  RMD_R(el, al, bl, cl, dl, RMD_F5, x10, 0xA953FD4Eu, 12);
  RMD_R(er, ar, br, cr, dr, RMD_F1, x7, 0x00000000u, 6);
  RMD_R(dl, el, al, bl, cl, RMD_F5, x14, 0xA953FD4Eu, 5);
  RMD_R(dr, er, ar, br, cr, RMD_F1, x6, 0x00000000u, 8);
  RMD_R(cl, dl, el, al, bl, RMD_F5, x1, 0xA953FD4Eu, 12);
  RMD_R(cr, dr, er, ar, br, RMD_F1, x2, 0x00000000u, 13);
  RMD_R(bl, cl, dl, el, al, RMD_F5, x3, 0xA953FD4Eu, 13);
  RMD_R(br, cr, dr, er, ar, RMD_F1, x13, 0x00000000u, 6);
  RMD_R(al, bl, cl, dl, el, RMD_F5, x8, 0xA953FD4Eu, 14);
  RMD_R(ar, br, cr, dr, er, RMD_F1, x14, 0x00000000u, 5);
  RMD_R(el, al, bl, cl, dl, RMD_F5, x11, 0xA953FD4Eu, 11);
  RMD_R(er, ar, br, cr, dr, RMD_F1, x0, 0x00000000u, 15);
  RMD_R(dl, el, al, bl, cl, RMD_F5, x6, 0xA953FD4Eu, 8);
  RMD_R(dr, er, ar, br, cr, RMD_F1, x3, 0x00000000u, 13);
  RMD_R(cl, dl, el, al, bl, RMD_F5, x15, 0xA953FD4Eu, 5);
  RMD_R(cr, dr, er, ar, br, RMD_F1, x9, 0x00000000u, 11);
  RMD_R(bl, cl, dl, el, al, RMD_F5, x13, 0xA953FD4Eu, 6);
  RMD_R(br, cr, dr, er, ar, RMD_F1, x11, 0x00000000u, 11);
  U32 t = 0xEFCDAB89u + cl + dr;
  U32 h1 = 0x98BADCFEu + dl + er;
  U32 h2 = 0x10325476u + el + ar;
  U32 h3 = 0xC3D2E1F0u + al + br;
  U32 h4 = 0x67452301u + bl + cr;
  out[0] = BSWAP32(t);
  out[1] = BSWAP32(h1);
  out[2] = BSWAP32(h2);
  out[3] = BSWAP32(h3);
  out[4] = BSWAP32(h4);
}

/* Word i of the big-endian 32-byte encoding of a (i = 0 is the most significant). */
#define FE_BE32(a, i) ((U32)((a).d[3 - (i) / 2] >> (((i) & 1) ? 0 : 32)))

/* hash160 of the 33-byte compressed encoding (one SHA-256 block). */
KFN void hash160_compressed(fe_t x, fe_t y, PRIV U32* out) {
  U32 p = (U32)(2 + (y.d[0] & 1));
  U32 X0 = FE_BE32(x, 0), X1 = FE_BE32(x, 1), X2 = FE_BE32(x, 2), X3 = FE_BE32(x, 3);
  U32 X4 = FE_BE32(x, 4), X5 = FE_BE32(x, 5), X6 = FE_BE32(x, 6), X7 = FE_BE32(x, 7);
  U32 st[8];
  sha256_init(st);
  sha256_block(st, (p << 24) | (X0 >> 8), (X0 << 24) | (X1 >> 8), (X1 << 24) | (X2 >> 8), (X2 << 24) | (X3 >> 8),
               (X3 << 24) | (X4 >> 8), (X4 << 24) | (X5 >> 8), (X5 << 24) | (X6 >> 8), (X6 << 24) | (X7 >> 8),
               (X7 << 24) | 0x800000u, 0, 0, 0, 0, 0, 0, 33u * 8u);
  ripemd160_of_sha(st, out);
}

/* hash160 of the 65-byte uncompressed encoding (two SHA-256 blocks). */
KFN void hash160_uncompressed(fe_t x, fe_t y, PRIV U32* out) {
  U32 X0 = FE_BE32(x, 0), X1 = FE_BE32(x, 1), X2 = FE_BE32(x, 2), X3 = FE_BE32(x, 3);
  U32 X4 = FE_BE32(x, 4), X5 = FE_BE32(x, 5), X6 = FE_BE32(x, 6), X7 = FE_BE32(x, 7);
  U32 Y0 = FE_BE32(y, 0), Y1 = FE_BE32(y, 1), Y2 = FE_BE32(y, 2), Y3 = FE_BE32(y, 3);
  U32 Y4 = FE_BE32(y, 4), Y5 = FE_BE32(y, 5), Y6 = FE_BE32(y, 6), Y7 = FE_BE32(y, 7);
  U32 st[8];
  sha256_init(st);
  sha256_block(st, (4u << 24) | (X0 >> 8), (X0 << 24) | (X1 >> 8), (X1 << 24) | (X2 >> 8), (X2 << 24) | (X3 >> 8),
               (X3 << 24) | (X4 >> 8), (X4 << 24) | (X5 >> 8), (X5 << 24) | (X6 >> 8), (X6 << 24) | (X7 >> 8),
               (X7 << 24) | (Y0 >> 8), (Y0 << 24) | (Y1 >> 8), (Y1 << 24) | (Y2 >> 8), (Y2 << 24) | (Y3 >> 8),
               (Y3 << 24) | (Y4 >> 8), (Y4 << 24) | (Y5 >> 8), (Y5 << 24) | (Y6 >> 8), (Y6 << 24) | (Y7 >> 8));
  sha256_block(st, (Y7 << 24) | 0x800000u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65u * 8u);
  ripemd160_of_sha(st, out);
}

/* Binary search of h (5 big-endian words) in the sorted target array (5 words per entry). */
KFN int target_contains(GLOB const U32* targets, U32 count, PRIV const U32* h) {
  U32 lo = 0, hi = count;
  while (lo < hi) {
    U32 mid = (lo + hi) >> 1;
    GLOB const U32* e = targets + 5 * mid;
    int cmp = 0;
    for (int i = 0; i < 5 && cmp == 0; ++i) {
      if (h[i] != e[i]) cmp = h[i] < e[i] ? -1 : 1;
    }
    if (cmp == 0) return 1;
    if (cmp < 0) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return 0;
}

/* ---- scan ------------------------------------------------------------------------------------- */

/* Hashes P per address mode (bit 0 compressed, bit 1 uncompressed) and records target hits as
   {thread, iter, offset, uncompressed}. */
KFN void scan_check(fe_t x, fe_t y, U32 mode, GLOB const U32* targets, U32 targetCount, U32 tid, U32 iter,
                    int offset, GLOB U32* hits, GLOB U32* hitCount) {
  U32 h[5];
  for (U32 unc = 0; unc < 2; ++unc) {
    if (!(mode & (1u << unc))) continue;
    if (unc) {
      hash160_uncompressed(x, y, h);
    } else {
      hash160_compressed(x, y, h);
    }
    if (target_contains(targets, targetCount, h)) {
      U32 slot = ATOMIC_INC(hitCount);
      if (slot < SCAN_HIT_CAP) {
        hits[4 * slot] = tid;
        hits[4 * slot + 1] = iter;
        hits[4 * slot + 2] = (U32)offset;
        hits[4 * slot + 3] = unc;
      }
    }
  }
}

/* Thread t scans `iters` groups of GPU_GROUP keys. params: [0..7] B = segStart*G (x, y), [8] iters | mode << 32,
   [9] target count. offsets[8t..8t+7] = (t*S + GPU_HALF)*G with S = iters*GPU_GROUP. gtable[8(i-1)..] = i*G for
   i = 1..GPU_HALF, then GPU_GROUP*G. Center C covers keys C-GPU_HALF..C+GPU_HALF-1. */
KFN void scan_thread(U32 t, GLOB const U64* params, GLOB const U64* offsets, GLOB const U64* gtable,
                     GLOB const U32* targets, GLOB U32* hits, GLOB U32* hitCount) {
  U32 iters = (U32)params[8];
  U32 mode = (U32)(params[8] >> 32);
  U32 targetCount = (U32)params[9];
  fe_t bx = fe_load(params), by = fe_load(params + 4);
  fe_t ox = fe_load(offsets + 8 * t), oy = fe_load(offsets + 8 * t + 4);

  /* C = B + O (doubling when B == O; B == -O only for keys == 0 mod n, outside every range). */
  fe_t num = fe_sub(oy, by), den = fe_sub(ox, bx);
  if (fe_is_zero(den)) {
    fe_t x2 = fe_sqr(bx);
    num = fe_add(fe_add(x2, x2), x2);
    den = fe_add(by, by);
  }
  fe_t lam = fe_mul(num, fe_inv(den));
  fe_t cx = fe_sub(fe_sub(fe_sqr(lam), bx), ox);
  fe_t cy = fe_sub(fe_mul(lam, fe_sub(bx, cx)), by);

  fe_t pre[GPU_HALF + 1];
  for (U32 it = 0; it < iters; ++it) {
    /* Prefix products of dx_i = x(g_i) - x(C); zero dx entries (keys == +-i mod n) are skipped. */
    fe_t acc = fe_one();
    for (int i = 0; i <= GPU_HALF; ++i) {
      pre[i] = acc;
      fe_t dx = fe_sub(fe_load(gtable + 8 * i), cx);
      if (!fe_is_zero(dx)) acc = fe_mul(acc, dx);
    }
    fe_t inv = fe_inv(acc);
    fe_t nx = cx, ny = cy;
    for (int i = GPU_HALF; i >= 0; --i) {
      fe_t gx = fe_load(gtable + 8 * i), gy = fe_load(gtable + 8 * i + 4);
      fe_t dx = fe_sub(gx, cx);
      if (fe_is_zero(dx)) continue;
      fe_t di = fe_mul(inv, pre[i]);
      inv = fe_mul(inv, dx);
      /* Two points per entry: C + kG and C - kG (k = i + 1), or for i == GPU_HALF the next center
         C + GPU_GROUP*G and C itself. One scan_check call site keeps the inlined hashing code small. */
      for (int s = 0; s < 2; ++s) {
        fe_t px = cx, py = cy;
        int offset = 0;
        if (i < GPU_HALF) {
          if (s == 0 && i + 1 == GPU_HALF) continue; /* C + (GPU_GROUP/2)G is the next group's first key */
          pt_add(cx, cy, gx, s ? fe_neg(gy) : gy, di, &px, &py);
          offset = s ? -(i + 1) : i + 1;
        } else if (s == 0) {
          pt_add(cx, cy, gx, gy, di, &nx, &ny);
          continue;
        }
        scan_check(px, py, mode, targets, targetCount, t, it, offset, hits, hitCount);
      }
    }
    cx = nx;
    cy = ny;
  }
}

/* ---- kangaroo --------------------------------------------------------------------------------- */

/* Thread t of T walks kangaroos i = g*T + t (g < KPT) for `steps` jumps each.
   params: [0] N = T*KPT, [1] steps, [2] dpBits, [3] DP capacity.
   state: 12 words per kangaroo in component-major order, word c*N + i (c: x0..x3, y0..y3, dist0..dist3).
   jumps: 12 words per jump j (x, y, dist). A jump is j = x.d[0] & 63: P += jumpPoint[j], dist += jumpDist[j].
   dps: 9 words per entry {index, x[4], dist[4]}; index bit 63 set = kangaroo stuck on dx == 0 (needs a reset). */
KFN void kangaroo_thread(U32 t, GLOB const U64* params, GLOB U64* state, GLOB const U64* jumps, GLOB U64* dps,
                         GLOB U32* dpCount) {
  U32 n = (U32)params[0];
  U32 steps = (U32)params[1];
  U32 dpBits = (U32)params[2];
  U32 dpCap = (U32)params[3];
  U32 T = n / KPT;
  fe_t px[KPT], py[KPT], dist[KPT], pre[KPT];
  for (int g = 0; g < KPT; ++g) {
    U32 i = (U32)g * T + t;
    for (int l = 0; l < 4; ++l) {
      px[g].d[l] = state[(U64)l * n + i];
      py[g].d[l] = state[(U64)(4 + l) * n + i];
      dist[g].d[l] = state[(U64)(8 + l) * n + i];
    }
  }
  for (U32 s = 0; s < steps; ++s) {
    fe_t acc = fe_one();
    for (int g = 0; g < KPT; ++g) {
      pre[g] = acc;
      fe_t dx = fe_sub(fe_load(jumps + 12 * (px[g].d[0] & 63)), px[g]);
      if (!fe_is_zero(dx)) acc = fe_mul(acc, dx);
    }
    fe_t inv = fe_inv(acc);
    for (int g = KPT - 1; g >= 0; --g) {
      U32 i = (U32)g * T + t;
      GLOB const U64* jp = jumps + 12 * (px[g].d[0] & 63);
      fe_t jx = fe_load(jp);
      fe_t dx = fe_sub(jx, px[g]);
      if (fe_is_zero(dx)) {
        U32 slot = ATOMIC_INC(dpCount);
        if (slot < dpCap) dps[9 * (U64)slot] = ((U64)1 << 63) | i;
        continue;
      }
      fe_t di = fe_mul(inv, pre[g]);
      inv = fe_mul(inv, dx);
      pt_add(px[g], py[g], jx, fe_load(jp + 4), di, &px[g], &py[g]);
      U64 c = 0;
      for (int l = 0; l < 4; ++l) {
        U64 j = jp[8 + l];
        U64 v = dist[g].d[l] + j;
        U64 c1 = v < j;
        dist[g].d[l] = v + c;
        c = c1 | (dist[g].d[l] < v);
      }
      if (dpBits == 0 || (px[g].d[3] >> (64 - dpBits)) == 0) {
        U32 slot = ATOMIC_INC(dpCount);
        if (slot < dpCap) {
          GLOB U64* e = dps + 9 * (U64)slot;
          e[0] = i;
          fe_store(e + 1, px[g]);
          fe_store(e + 5, dist[g]);
        }
      }
    }
  }
  for (int g = 0; g < KPT; ++g) {
    U32 i = (U32)g * T + t;
    for (int l = 0; l < 4; ++l) {
      state[(U64)l * n + i] = px[g].d[l];
      state[(U64)(4 + l) * n + i] = py[g].d[l];
      state[(U64)(8 + l) * n + i] = dist[g].d[l];
    }
  }
}

#endif /* BSOLVER_KERNEL_CORE_H */

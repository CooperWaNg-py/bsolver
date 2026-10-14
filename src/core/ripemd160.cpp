#include "core/ripemd160.h"

#include <cstring>

#include "core/sha256.h"

namespace bsolver {

namespace {
constexpr uint8_t RL[80] = {0, 1, 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 7,  4,  13, 1,
                            10, 6, 15, 3,  12, 0,  9,  5,  2,  14, 11, 8,  3,  10, 14, 4,  9,  15, 8,  1,
                            2,  7, 0,  6,  13, 11, 5,  12, 1,  9,  11, 10, 0,  8,  12, 4,  13, 3,  7,  15,
                            14, 5, 6,  2,  4,  0,  5,  9,  7,  12, 2,  10, 14, 1,  3,  8,  11, 6,  15, 13};
constexpr uint8_t RR[80] = {5,  14, 7,  0, 9, 2,  11, 4,  13, 6,  15, 8,  1,  10, 3,  12, 6,  11, 3,  7,
                            0,  13, 5,  10, 14, 15, 8,  12, 4,  9,  1,  2,  15, 5,  1,  3,  7,  14, 6,  9,
                            11, 8,  12, 2,  10, 0,  4,  13, 8,  6,  4,  1,  3,  11, 15, 0,  5,  12, 2,  13,
                            9,  7,  10, 14, 12, 15, 10, 4,  1,  5,  8,  7,  6,  2,  13, 14, 0,  3,  9,  11};
constexpr uint8_t SL[80] = {11, 14, 15, 12, 5,  8,  7,  9,  11, 13, 14, 15, 6,  7,  9,  8,  7,  6,  8,  13,
                            11, 9,  7,  15, 7,  12, 15, 9,  11, 7,  13, 12, 11, 13, 6,  7,  14, 9,  13, 15,
                            14, 8,  13, 6,  5,  12, 7,  5,  11, 12, 14, 15, 14, 15, 9,  8,  9,  14, 5,  6,
                            8,  6,  5,  12, 9,  15, 5,  11, 6,  8,  13, 12, 5,  12, 13, 14, 11, 8,  5,  6};
constexpr uint8_t SR[80] = {8,  9,  9,  11, 13, 15, 15, 5,  7,  7,  8,  11, 14, 14, 12, 6,  9,  13, 15, 7,
                            12, 8,  9,  11, 7,  7,  12, 7,  6,  15, 13, 11, 9,  7,  15, 11, 8,  6,  6,  14,
                            12, 13, 5,  14, 13, 13, 7,  5,  15, 5,  8,  11, 14, 14, 6,  14, 6,  9,  12, 9,
                            12, 5,  15, 8,  8,  5,  12, 9,  12, 5,  14, 6,  8,  13, 6,  5,  15, 13, 11, 11};
constexpr uint32_t KL[5] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
constexpr uint32_t KR[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};

inline uint32_t rol(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

inline uint32_t f(int j, uint32_t x, uint32_t y, uint32_t z) {
  switch (j >> 4) {
    case 0: return x ^ y ^ z;
    case 1: return (x & y) | (~x & z);
    case 2: return (x | ~y) ^ z;
    case 3: return (x & z) | (y & ~z);
    default: return x ^ (y | ~z);
  }
}

constexpr uint32_t kInit[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
}  // namespace

void Ripemd160::transform(uint32_t st[5], const uint8_t block[64]) {
  uint32_t X[16];
  for (int i = 0; i < 16; ++i) {
    X[i] = static_cast<uint32_t>(block[4 * i]) | static_cast<uint32_t>(block[4 * i + 1]) << 8 |
           static_cast<uint32_t>(block[4 * i + 2]) << 16 | static_cast<uint32_t>(block[4 * i + 3]) << 24;
  }
  uint32_t al = st[0], bl = st[1], cl = st[2], dl = st[3], el = st[4];
  uint32_t ar = al, br = bl, cr = cl, dr = dl, er = el;
  for (int j = 0; j < 80; ++j) {
    uint32_t t = rol(al + f(j, bl, cl, dl) + X[RL[j]] + KL[j >> 4], SL[j]) + el;
    al = el;
    el = dl;
    dl = rol(cl, 10);
    cl = bl;
    bl = t;
    t = rol(ar + f(79 - j, br, cr, dr) + X[RR[j]] + KR[j >> 4], SR[j]) + er;
    ar = er;
    er = dr;
    dr = rol(cr, 10);
    cr = br;
    br = t;
  }
  uint32_t t = st[1] + cl + dr;
  st[1] = st[2] + dl + er;
  st[2] = st[3] + el + ar;
  st[3] = st[4] + al + br;
  st[4] = st[0] + bl + cr;
  st[0] = t;
}

Ripemd160::Ripemd160() { std::memcpy(s_, kInit, sizeof(s_)); }

void Ripemd160::update(const uint8_t* data, size_t len) {
  total_ += len;
  while (len > 0) {
    size_t take = 64 - bufLen_;
    if (take > len) take = len;
    std::memcpy(buf_ + bufLen_, data, take);
    bufLen_ += take;
    data += take;
    len -= take;
    if (bufLen_ == 64) {
      transform(s_, buf_);
      bufLen_ = 0;
    }
  }
}

void Ripemd160::final(uint8_t out[20]) {
  uint64_t bits = total_ * 8;
  uint8_t pad[72] = {0x80};
  size_t padLen = (bufLen_ < 56) ? 56 - bufLen_ : 120 - bufLen_;
  update(pad, padLen);
  uint8_t len[8];
  for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>(bits >> (8 * i));
  update(len, 8);
  for (int i = 0; i < 5; ++i) {
    out[4 * i] = static_cast<uint8_t>(s_[i]);
    out[4 * i + 1] = static_cast<uint8_t>(s_[i] >> 8);
    out[4 * i + 2] = static_cast<uint8_t>(s_[i] >> 16);
    out[4 * i + 3] = static_cast<uint8_t>(s_[i] >> 24);
  }
}

void ripemd160(const uint8_t* data, size_t len, uint8_t out[20]) {
  Ripemd160 h;
  h.update(data, len);
  h.final(out);
}

namespace {
// RIPEMD-160 of a 32-byte SHA-256 state: one padded block.
void ripemdOfSha(const uint32_t sha[8], uint8_t out[20]) {
  uint8_t block[64] = {0};
  for (int i = 0; i < 8; ++i) {
    block[4 * i] = static_cast<uint8_t>(sha[i] >> 24);
    block[4 * i + 1] = static_cast<uint8_t>(sha[i] >> 16);
    block[4 * i + 2] = static_cast<uint8_t>(sha[i] >> 8);
    block[4 * i + 3] = static_cast<uint8_t>(sha[i]);
  }
  block[32] = 0x80;
  block[56] = 0x00;  // 256 bits little-endian = 0x0100
  block[57] = 0x01;
  uint32_t st[5];
  std::memcpy(st, kInit, sizeof(st));
  Ripemd160::transform(st, block);
  for (int i = 0; i < 5; ++i) {
    out[4 * i] = static_cast<uint8_t>(st[i]);
    out[4 * i + 1] = static_cast<uint8_t>(st[i] >> 8);
    out[4 * i + 2] = static_cast<uint8_t>(st[i] >> 16);
    out[4 * i + 3] = static_cast<uint8_t>(st[i] >> 24);
  }
}
}  // namespace

void hash160_33(const uint8_t pubkey[33], uint8_t out[20]) {
  uint8_t block[64] = {0};
  std::memcpy(block, pubkey, 33);
  block[33] = 0x80;
  block[62] = 0x01;  // 264 bits = 0x0108 big-endian
  block[63] = 0x08;
  uint32_t st[8];
  std::memcpy(st, Sha256::kInit, sizeof(st));
  Sha256::transform(st, block);
  ripemdOfSha(st, out);
}

void hash160_65(const uint8_t pubkey[65], uint8_t out[20]) {
  uint32_t st[8];
  std::memcpy(st, Sha256::kInit, sizeof(st));
  Sha256::transform(st, pubkey);
  uint8_t block[64] = {0};
  block[0] = pubkey[64];
  block[1] = 0x80;
  block[62] = 0x02;  // 520 bits = 0x0208 big-endian
  block[63] = 0x08;
  Sha256::transform(st, block);
  ripemdOfSha(st, out);
}

}  // namespace bsolver

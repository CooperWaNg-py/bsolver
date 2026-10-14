#include "core/ripemd160.h"

#include <cstring>

#include "core/hash_lanes.h"

namespace bsolver {

namespace {
using hashimpl::Lane1;

uint32_t loadBE(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) << 24 | static_cast<uint32_t>(p[1]) << 16 | static_cast<uint32_t>(p[2]) << 8 | p[3];
}

uint32_t loadLE(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 | static_cast<uint32_t>(p[2]) << 16 |
         static_cast<uint32_t>(p[3]) << 24;
}

void storeLE(const Lane1 (&st)[5], uint8_t out[20]) {
  for (int i = 0; i < 5; ++i) {
    out[4 * i] = static_cast<uint8_t>(st[i].v);
    out[4 * i + 1] = static_cast<uint8_t>(st[i].v >> 8);
    out[4 * i + 2] = static_cast<uint8_t>(st[i].v >> 16);
    out[4 * i + 3] = static_cast<uint8_t>(st[i].v >> 24);
  }
}

// RIPEMD-160 of a 32-byte SHA-256 state: one padded block.
void ripemdOfSha(const Lane1 (&sha)[8], uint8_t out[20]) {
  Lane1 x[16] = {};
  for (int i = 0; i < 8; ++i) x[i] = hashimpl::bswap(sha[i]);
  x[8] = {0x80};
  x[14] = {256};
  Lane1 st[5];
  hashimpl::rmd160Init(st);
  hashimpl::rmd160Compress(st, x);
  storeLE(st, out);
}
}  // namespace

void Ripemd160::transform(uint32_t st[5], const uint8_t block[64]) {
  Lane1 s[5], x[16];
  for (int i = 0; i < 5; ++i) s[i] = {st[i]};
  for (int i = 0; i < 16; ++i) x[i] = {loadLE(block + 4 * i)};
  hashimpl::rmd160Compress(s, x);
  for (int i = 0; i < 5; ++i) st[i] = s[i].v;
}

Ripemd160::Ripemd160() { std::memcpy(s_, hashimpl::kRmdInit, sizeof(s_)); }

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

void hash160_33(const uint8_t pubkey[33], uint8_t out[20]) {
  Lane1 w[16] = {};
  for (int i = 0; i < 8; ++i) w[i] = {loadBE(pubkey + 4 * i)};
  w[8] = {static_cast<uint32_t>(pubkey[32]) << 24 | 0x800000};
  w[15] = {33 * 8};
  Lane1 st[8];
  hashimpl::sha256Init(st);
  hashimpl::sha256Compress(st, w);
  ripemdOfSha(st, out);
}

void hash160_65(const uint8_t pubkey[65], uint8_t out[20]) {
  Lane1 w[16];
  for (int i = 0; i < 16; ++i) w[i] = {loadBE(pubkey + 4 * i)};
  Lane1 st[8];
  hashimpl::sha256Init(st);
  hashimpl::sha256Compress(st, w);
  for (auto& v : w) v = {0};
  w[0] = {static_cast<uint32_t>(pubkey[64]) << 24 | 0x800000};
  w[15] = {65 * 8};
  hashimpl::sha256Compress(st, w);
  ripemdOfSha(st, out);
}

}  // namespace bsolver

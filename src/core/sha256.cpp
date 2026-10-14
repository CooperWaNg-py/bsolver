#include "core/sha256.h"

#include <cstring>

#include "core/hash_lanes.h"

namespace bsolver {

namespace {
using hashimpl::Lane1;

uint32_t loadBE(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) << 24 | static_cast<uint32_t>(p[1]) << 16 | static_cast<uint32_t>(p[2]) << 8 | p[3];
}
}  // namespace

void Sha256::transform(uint32_t st[8], const uint8_t block[64]) {
  Lane1 s[8], w[16];
  for (int i = 0; i < 8; ++i) s[i] = {st[i]};
  for (int i = 0; i < 16; ++i) w[i] = {loadBE(block + 4 * i)};
  hashimpl::sha256Compress(s, w);
  for (int i = 0; i < 8; ++i) st[i] = s[i].v;
}

Sha256::Sha256() { std::memcpy(s_, hashimpl::kSha256Init, sizeof(s_)); }

void Sha256::update(const uint8_t* data, size_t len) {
  total_ += len;
  while (len > 0) {
    if (bufLen_ == 0 && len >= 64) {
      transform(s_, data);
      data += 64;
      len -= 64;
      continue;
    }
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

void Sha256::final(uint8_t out[32]) {
  uint64_t bits = total_ * 8;
  uint8_t pad[72] = {0x80};
  size_t padLen = (bufLen_ < 56) ? 56 - bufLen_ : 120 - bufLen_;
  update(pad, padLen);
  uint8_t len[8];
  for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
  update(len, 8);
  for (int i = 0; i < 8; ++i) {
    out[4 * i] = static_cast<uint8_t>(s_[i] >> 24);
    out[4 * i + 1] = static_cast<uint8_t>(s_[i] >> 16);
    out[4 * i + 2] = static_cast<uint8_t>(s_[i] >> 8);
    out[4 * i + 3] = static_cast<uint8_t>(s_[i]);
  }
}

void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
  Sha256 h;
  h.update(data, len);
  h.final(out);
}

void sha256d(const uint8_t* data, size_t len, uint8_t out[32]) {
  uint8_t tmp[32];
  sha256(data, len, tmp);
  sha256(tmp, 32, out);
}

}  // namespace bsolver

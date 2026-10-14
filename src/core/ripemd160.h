#pragma once

#include <cstddef>
#include <cstdint>

namespace bsolver {

class Ripemd160 {
 public:
  Ripemd160();
  void update(const uint8_t* data, size_t len);
  void final(uint8_t out[20]);

 private:
  static void transform(uint32_t state[5], const uint8_t block[64]);

  uint32_t s_[5];
  uint8_t buf_[64];
  size_t bufLen_ = 0;
  uint64_t total_ = 0;
};

void ripemd160(const uint8_t* data, size_t len, uint8_t out[20]);

// RIPEMD160(SHA256(pubkey)) with fixed-size single/double block fast paths.
void hash160_33(const uint8_t pubkey[33], uint8_t out[20]);
void hash160_65(const uint8_t pubkey[65], uint8_t out[20]);

}  // namespace bsolver

#pragma once

#include <cstddef>
#include <cstdint>

namespace bsolver {

class Sha256 {
 public:
  Sha256();
  void update(const uint8_t* data, size_t len);
  void final(uint8_t out[32]);

  // Compresses one 64-byte block into state.
  static void transform(uint32_t state[8], const uint8_t block[64]);
  static const uint32_t kInit[8];

 private:
  uint32_t s_[8];
  uint8_t buf_[64];
  size_t bufLen_ = 0;
  uint64_t total_ = 0;
};

void sha256(const uint8_t* data, size_t len, uint8_t out[32]);
void sha256d(const uint8_t* data, size_t len, uint8_t out[32]);

}  // namespace bsolver

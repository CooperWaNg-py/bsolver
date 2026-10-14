#pragma once

#include <cstdint>
#include <random>
#include <string>

#include "core/u256.h"

namespace bsolver {

// Not cryptographic: used for search positions only.
class Rng {
 public:
  Rng();  // seeded from std::random_device (four 32-bit draws)
  explicit Rng(uint64_t seed) : eng_(seed) {}
  uint64_t next() { return eng_(); }
  // Uniform in [lo, hi] (inclusive); requires lo <= hi.
  U256 uniform(const U256& lo, const U256& hi);

 private:
  std::mt19937_64 eng_;
};

// Hex string of `bytes` bytes drawn directly from std::random_device.
std::string randomTokenHex(size_t bytes);

}  // namespace bsolver

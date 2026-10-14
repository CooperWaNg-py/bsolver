#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "core/point.h"
#include "core/u256.h"

namespace bsolver {

using Hash160 = std::array<uint8_t, 20>;

// Throws std::invalid_argument with "unsupported address type (only P2PKH '1...' addresses are
// supported)" or "invalid address checksum".
Hash160 decodeP2PKH(std::string_view address);
std::string encodeP2PKH(const Hash160& h);
std::string encodeP2PKH(const uint8_t h[20]);
std::string wifCompressed(const U256& key);
std::string wifUncompressed(const U256& key);

Hash160 hash160Of(const Affine& p, bool compressed);

}  // namespace bsolver

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace bsolver {

std::string hexEncode(const uint8_t* data, size_t len);
// Even-length hex without prefix; std::nullopt on any invalid character or odd length.
std::optional<std::vector<uint8_t>> hexDecode(std::string_view hex);

}  // namespace bsolver

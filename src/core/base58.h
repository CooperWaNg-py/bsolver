#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace bsolver {

std::string base58Encode(const uint8_t* data, size_t len);
// std::nullopt on characters outside the Bitcoin alphabet.
std::optional<std::vector<uint8_t>> base58Decode(std::string_view s);

}  // namespace bsolver

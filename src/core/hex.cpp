#include "core/hex.h"

namespace bsolver {

std::string hexEncode(const uint8_t* data, size_t len) {
  static const char* digits = "0123456789abcdef";
  std::string s(len * 2, '0');
  for (size_t i = 0; i < len; ++i) {
    s[2 * i] = digits[data[i] >> 4];
    s[2 * i + 1] = digits[data[i] & 0xf];
  }
  return s;
}

std::optional<std::vector<uint8_t>> hexDecode(std::string_view hex) {
  if (hex.size() % 2) return std::nullopt;
  auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  std::vector<uint8_t> out(hex.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) {
    int hi = nibble(hex[2 * i]), lo = nibble(hex[2 * i + 1]);
    if (hi < 0 || lo < 0) return std::nullopt;
    out[i] = static_cast<uint8_t>(hi << 4 | lo);
  }
  return out;
}

}  // namespace bsolver

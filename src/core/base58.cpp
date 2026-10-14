#include "core/base58.h"

namespace bsolver {

namespace {
constexpr const char* kAlphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
}

std::string base58Encode(const uint8_t* data, size_t len) {
  size_t zeros = 0;
  while (zeros < len && data[zeros] == 0) ++zeros;
  // log(256)/log(58) ~ 1.366
  std::vector<uint8_t> digits(len * 138 / 100 + 1, 0);
  size_t used = 0;
  for (size_t i = zeros; i < len; ++i) {
    unsigned carry = data[i];
    size_t j = 0;
    for (auto it = digits.rbegin(); (carry != 0 || j < used) && it != digits.rend(); ++it, ++j) {
      carry += 256u * *it;
      *it = static_cast<uint8_t>(carry % 58);
      carry /= 58;
    }
    used = j;
  }
  auto it = digits.begin();
  while (it != digits.end() && *it == 0) ++it;
  std::string out(zeros, '1');
  for (; it != digits.end(); ++it) out.push_back(kAlphabet[*it]);
  return out;
}

std::optional<std::vector<uint8_t>> base58Decode(std::string_view s) {
  int map[256];
  for (int& m : map) m = -1;
  for (int i = 0; i < 58; ++i) map[static_cast<unsigned char>(kAlphabet[i])] = i;

  size_t zeros = 0;
  while (zeros < s.size() && s[zeros] == '1') ++zeros;
  // log(58)/log(256) ~ 0.733
  std::vector<uint8_t> bytes(s.size() * 733 / 1000 + 1, 0);
  size_t used = 0;
  for (size_t i = zeros; i < s.size(); ++i) {
    int v = map[static_cast<unsigned char>(s[i])];
    if (v < 0) return std::nullopt;
    unsigned carry = static_cast<unsigned>(v);
    size_t j = 0;
    for (auto it = bytes.rbegin(); (carry != 0 || j < used) && it != bytes.rend(); ++it, ++j) {
      carry += 58u * *it;
      *it = static_cast<uint8_t>(carry & 0xff);
      carry >>= 8;
    }
    used = j;
  }
  auto it = bytes.begin();
  while (it != bytes.end() && *it == 0) ++it;
  std::vector<uint8_t> out(zeros, 0);
  out.insert(out.end(), it, bytes.end());
  return out;
}

}  // namespace bsolver

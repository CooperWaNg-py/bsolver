#pragma once

#include <compare>
#include <cstdint>
#include <string>
#include <string_view>

namespace bsolver {

// 256-bit unsigned integer, little-endian 64-bit limbs.
struct U256 {
  uint64_t w[4] = {0, 0, 0, 0};

  static U256 fromU64(uint64_t v) { return U256{{v, 0, 0, 0}}; }
  // Optional 0x prefix; throws std::invalid_argument("invalid hex: ...").
  static U256 fromHex(std::string_view hex);
  // Big-endian 32 bytes.
  static U256 fromBytesBE(const uint8_t* in32);
  void toBytesBE(uint8_t* out32) const;

  std::string toHex(bool pad64 = false) const;
  bool isZero() const { return (w[0] | w[1] | w[2] | w[3]) == 0; }
  unsigned bitLength() const;
  double toDouble() const;
  long double toLongDouble() const;
  static U256 fromLongDouble(long double v);  // truncates; v must be in [0, 2^256)

  friend bool operator==(const U256& a, const U256& b) = default;
  friend std::strong_ordering operator<=>(const U256& a, const U256& b) {
    for (int i = 3; i >= 0; --i) {
      if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? std::strong_ordering::less : std::strong_ordering::greater;
    }
    return std::strong_ordering::equal;
  }
};

// r = a + b, returns carry-out.
uint64_t add(U256& r, const U256& a, const U256& b);
// r = a - b, returns borrow-out.
uint64_t sub(U256& r, const U256& a, const U256& b);
// Wrapping arithmetic modulo 2^256.
inline U256 operator+(const U256& a, const U256& b) { U256 r; add(r, a, b); return r; }
inline U256 operator-(const U256& a, const U256& b) { U256 r; sub(r, a, b); return r; }
inline U256 operator+(const U256& a, uint64_t b) { return a + U256::fromU64(b); }
inline U256 operator-(const U256& a, uint64_t b) { return a - U256::fromU64(b); }
U256 operator<<(const U256& a, unsigned s);
U256 operator>>(const U256& a, unsigned s);
// a * m modulo 2^256; *overflow (optional) receives the bits above 2^256 (top limb).
U256 mulU64(const U256& a, uint64_t m, uint64_t* overflow = nullptr);
// a / d and a % d for a 64-bit divisor.
U256 divU64(const U256& a, uint64_t d, uint64_t* rem = nullptr);

}  // namespace bsolver

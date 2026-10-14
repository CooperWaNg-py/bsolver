#include "core/u256.h"

#include <cmath>
#include <stdexcept>

#include "core/int128.h"

namespace bsolver {

namespace {
int hexDigit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}
}  // namespace

U256 U256::fromHex(std::string_view hex) {
  std::string_view s = hex;
  if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s.remove_prefix(2);
  if (s.empty() || s.size() > 64) throw std::invalid_argument("invalid hex: " + std::string(hex));
  U256 r;
  unsigned bitPos = 0;
  for (size_t i = s.size(); i-- > 0;) {
    int d = hexDigit(s[i]);
    if (d < 0) throw std::invalid_argument("invalid hex: " + std::string(hex));
    r.w[bitPos >> 6] |= static_cast<uint64_t>(d) << (bitPos & 63);
    bitPos += 4;
  }
  return r;
}

U256 U256::fromBytesBE(const uint8_t* in) {
  U256 r;
  for (int limb = 0; limb < 4; ++limb) {
    uint64_t v = 0;
    const uint8_t* p = in + (3 - limb) * 8;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    r.w[limb] = v;
  }
  return r;
}

void U256::toBytesBE(uint8_t* out) const {
  for (int limb = 0; limb < 4; ++limb) {
    uint64_t v = w[limb];
    uint8_t* p = out + (3 - limb) * 8;
    for (int i = 7; i >= 0; --i) {
      p[i] = static_cast<uint8_t>(v);
      v >>= 8;
    }
  }
}

std::string U256::toHex(bool pad64) const {
  static const char* digits = "0123456789abcdef";
  std::string s(64, '0');
  for (int i = 0; i < 64; ++i) {
    unsigned bitPos = static_cast<unsigned>(63 - i) * 4;
    s[i] = digits[(w[bitPos >> 6] >> (bitPos & 63)) & 0xf];
  }
  if (pad64) return s;
  size_t nz = s.find_first_not_of('0');
  return nz == std::string::npos ? std::string("0") : s.substr(nz);
}

unsigned U256::bitLength() const {
  for (int i = 3; i >= 0; --i) {
    if (w[i]) {
      unsigned n = 64;
      uint64_t v = w[i];
      while (!(v >> 63)) {
        v <<= 1;
        --n;
      }
      return static_cast<unsigned>(i) * 64 + n;
    }
  }
  return 0;
}

double U256::toDouble() const {
  return std::ldexp(static_cast<double>(w[3]), 192) + std::ldexp(static_cast<double>(w[2]), 128) +
         std::ldexp(static_cast<double>(w[1]), 64) + static_cast<double>(w[0]);
}

long double U256::toLongDouble() const {
  return std::ldexp(static_cast<long double>(w[3]), 192) + std::ldexp(static_cast<long double>(w[2]), 128) +
         std::ldexp(static_cast<long double>(w[1]), 64) + static_cast<long double>(w[0]);
}

U256 U256::fromLongDouble(long double v) {
  U256 r;
  if (!(v >= 1.0L)) return r;
  for (int i = 3; i >= 0; --i) {
    long double scale = std::ldexp(1.0L, 64 * i);
    long double q = std::floor(v / scale);
    if (q >= 18446744073709551616.0L) q = 18446744073709551615.0L;
    if (q < 0) q = 0;
    r.w[i] = static_cast<uint64_t>(q);
    v -= static_cast<long double>(r.w[i]) * scale;
    if (v < 0) v = 0;
  }
  return r;
}

uint64_t add(U256& r, const U256& a, const U256& b) {
  uint64_t c = 0;
  for (int i = 0; i < 4; ++i) r.w[i] = addc(a.w[i], b.w[i], c, &c);
  return c;
}

uint64_t sub(U256& r, const U256& a, const U256& b) {
  uint64_t c = 0;
  for (int i = 0; i < 4; ++i) r.w[i] = subb(a.w[i], b.w[i], c, &c);
  return c;
}

U256 operator<<(const U256& a, unsigned s) {
  U256 r;
  if (s >= 256) return r;
  unsigned limbs = s / 64, bits = s % 64;
  for (int i = 3; i >= static_cast<int>(limbs); --i) {
    uint64_t v = a.w[i - limbs] << bits;
    if (bits && i - static_cast<int>(limbs) - 1 >= 0) v |= a.w[i - limbs - 1] >> (64 - bits);
    r.w[i] = v;
  }
  return r;
}

U256 operator>>(const U256& a, unsigned s) {
  U256 r;
  if (s >= 256) return r;
  unsigned limbs = s / 64, bits = s % 64;
  for (unsigned i = 0; i + limbs < 4; ++i) {
    uint64_t v = a.w[i + limbs] >> bits;
    if (bits && i + limbs + 1 < 4) v |= a.w[i + limbs + 1] << (64 - bits);
    r.w[i] = v;
  }
  return r;
}

U256 mulU64(const U256& a, uint64_t m, uint64_t* overflow) {
  U256 r;
  uint64_t carry = 0;
  for (int i = 0; i < 4; ++i) {
    uint64_t hi;
    uint64_t lo = mul64(a.w[i], m, &hi);
    uint64_t c;
    r.w[i] = addc(lo, carry, 0, &c);
    carry = hi + c;
  }
  if (overflow) *overflow = carry;
  return r;
}

U256 divU64(const U256& a, uint64_t d, uint64_t* rem) {
  U256 q;
  uint64_t r = 0;
  // Bitwise long division on 32-bit halves keeps this portable without 128-bit division.
  for (int i = 3; i >= 0; --i) {
    for (int half = 1; half >= 0; --half) {
      uint64_t part = (a.w[i] >> (32 * half)) & 0xffffffffu;
      // (r << 32 | part) may overflow 64 bits when d > 2^32; fall back to bit-level steps.
      if (d <= 0xffffffffu) {
        uint64_t cur = (r << 32) | part;
        q.w[i] |= (cur / d) << (32 * half);
        r = cur % d;
      } else {
        uint64_t qpart = 0;
        for (int b = 31; b >= 0; --b) {
          bool top = r >> 63;
          r = (r << 1) | ((part >> b) & 1);
          qpart <<= 1;
          if (top || r >= d) {
            r -= d;
            qpart |= 1;
          }
        }
        q.w[i] |= qpart << (32 * half);
      }
    }
  }
  if (rem) *rem = r;
  return q;
}

}  // namespace bsolver

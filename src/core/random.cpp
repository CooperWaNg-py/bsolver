#include "core/random.h"

#include "core/hex.h"

#include <vector>

namespace bsolver {

Rng::Rng() {
  std::random_device rd;
  std::seed_seq seq{rd(), rd(), rd(), rd()};
  eng_.seed(seq);
}

U256 Rng::uniform(const U256& lo, const U256& hi) {
  U256 span = hi - lo;
  unsigned bits = span.bitLength();
  if (bits == 0) return lo;
  for (;;) {
    U256 r;
    for (unsigned i = 0; i < 4; ++i) {
      if (i * 64 >= bits) break;
      uint64_t v = eng_();
      unsigned remaining = bits - i * 64;
      if (remaining < 64) v &= (uint64_t{1} << remaining) - 1;
      r.w[i] = v;
    }
    if (r <= span) return lo + r;
  }
}

std::string randomTokenHex(size_t bytes) {
  std::random_device rd;
  std::vector<uint8_t> buf(bytes);
  for (size_t i = 0; i < bytes; i += 4) {
    uint32_t v = rd();
    for (size_t j = 0; j < 4 && i + j < bytes; ++j) buf[i + j] = static_cast<uint8_t>(v >> (8 * j));
  }
  return hexEncode(buf.data(), buf.size());
}

}  // namespace bsolver

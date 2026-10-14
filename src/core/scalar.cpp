#include "core/scalar.h"

namespace bsolver {

U256 reduceMod(const U256& a) { return a >= kOrderN ? a - kOrderN : a; }

U256 addMod(const U256& a, const U256& b) {
  U256 r;
  uint64_t carry = add(r, a, b);
  if (carry || r >= kOrderN) r = r - kOrderN;
  return r;
}

U256 subMod(const U256& a, const U256& b) {
  U256 r;
  if (sub(r, a, b)) r = r + kOrderN;
  return r;
}

U256 negMod(const U256& a) { return a.isZero() ? a : kOrderN - a; }

}  // namespace bsolver

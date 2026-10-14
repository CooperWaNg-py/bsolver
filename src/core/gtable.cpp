#include "core/gtable.h"

#include <mutex>
#include <vector>

namespace bsolver {

namespace {
constexpr int kWindows = 32;
constexpr int kEntries = 255;  // digits 1..255

struct GTable {
  // entry[w][d-1] = d * 256^w * G
  std::vector<Affine> entry;
  const Affine& at(int w, int d) const { return entry[static_cast<size_t>(w) * kEntries + (d - 1)]; }
};

const GTable& table() {
  static GTable t;
  static std::once_flag once;
  std::call_once(once, [] {
    t.entry.resize(static_cast<size_t>(kWindows) * kEntries);
    Affine base = generator();
    std::vector<Jacobian> jac(kEntries);
    for (int w = 0; w < kWindows; ++w) {
      jac[0] = toJacobian(base);
      for (int d = 1; d < kEntries; ++d) jac[d] = jacAddMixed(jac[d - 1], base);
      batchToAffine(jac.data(), &t.entry[static_cast<size_t>(w) * kEntries], kEntries);
      // next base = 256 * base = 255*base + base
      base = toAffine(jacAddMixed(jac[kEntries - 1], base));
    }
  });
  return t;
}
}  // namespace

Jacobian mulGJacobian(const U256& k) {
  const GTable& t = table();
  Jacobian acc;
  for (int w = 0; w < kWindows; ++w) {
    int d = static_cast<int>((k.w[w / 8] >> ((w % 8) * 8)) & 0xff);
    if (d) acc = jacAddMixed(acc, t.at(w, d));
  }
  return acc;
}

Affine mulG(const U256& k) { return toAffine(mulGJacobian(k)); }

}  // namespace bsolver

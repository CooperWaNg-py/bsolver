#include "core/field.h"

namespace bsolver {

Fe Fe::fromU256(const U256& v) {
  uint64_t r[4] = {v.w[0], v.w[1], v.w[2], v.w[3]};
  fe_detail::finalReduce(r);
  return Fe{{r[0], r[1], r[2], r[3]}};
}

namespace {
Fe sqrN(Fe a, int n) {
  for (int i = 0; i < n; ++i) a = feSqr(a);
  return a;
}

// Shared prefix of libsecp256k1's inversion / square-root addition chains.
struct ChainBlocks {
  Fe x2, x3, x22, x223;
};

ChainBlocks chainBlocks(const Fe& a) {
  ChainBlocks b;
  b.x2 = feMul(feSqr(a), a);
  b.x3 = feMul(feSqr(b.x2), a);
  Fe x6 = feMul(sqrN(b.x3, 3), b.x3);
  Fe x9 = feMul(sqrN(x6, 3), b.x3);
  Fe x11 = feMul(sqrN(x9, 2), b.x2);
  b.x22 = feMul(sqrN(x11, 11), x11);
  Fe x44 = feMul(sqrN(b.x22, 22), b.x22);
  Fe x88 = feMul(sqrN(x44, 44), x44);
  Fe x176 = feMul(sqrN(x88, 88), x88);
  Fe x220 = feMul(sqrN(x176, 44), x44);
  b.x223 = feMul(sqrN(x220, 3), b.x3);
  return b;
}
}  // namespace

Fe feInv(const Fe& a) {
  ChainBlocks b = chainBlocks(a);
  Fe t = feMul(sqrN(b.x223, 23), b.x22);
  t = feMul(sqrN(t, 5), a);
  t = feMul(sqrN(t, 3), b.x2);
  t = feMul(sqrN(t, 2), a);
  return t;
}

bool feSqrt(const Fe& a, Fe* out) {
  ChainBlocks b = chainBlocks(a);
  Fe t = feMul(sqrN(b.x223, 23), b.x22);
  t = feMul(sqrN(t, 6), b.x2);
  t = sqrN(t, 2);
  if (!(feSqr(t) == a)) return false;
  *out = t;
  return true;
}

void batchInverse(Fe* v, size_t n, Fe* scratch) {
  if (n == 0) return;
  Fe acc = Fe::fromU64(1);
  for (size_t i = 0; i < n; ++i) {
    scratch[i] = acc;
    if (!v[i].isZero()) acc = feMul(acc, v[i]);
  }
  Fe inv = feInv(acc);
  for (size_t i = n; i-- > 0;) {
    if (v[i].isZero()) continue;
    Fe t = feMul(inv, scratch[i]);
    inv = feMul(inv, v[i]);
    v[i] = t;
  }
}

}  // namespace bsolver

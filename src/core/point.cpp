#include "core/point.h"

#include <stdexcept>

#include "core/gtable.h"
#include "core/hex.h"

namespace bsolver {

const Affine& generator() {
  static const Affine g{
      Fe::fromU256(U256::fromHex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")),
      Fe::fromU256(U256::fromHex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8")), false};
  return g;
}

Jacobian toJacobian(const Affine& a) {
  if (a.inf) return Jacobian{};
  return Jacobian{a.x, a.y, Fe::fromU64(1), false};
}

Jacobian jacDouble(const Jacobian& p) {
  if (p.inf || p.y.isZero()) return Jacobian{};
  // dbl-2009-l (a = 0)
  Fe A = feSqr(p.x);
  Fe B = feSqr(p.y);
  Fe C = feSqr(B);
  Fe t = feSub(feSub(feSqr(feAdd(p.x, B)), A), C);
  Fe D = feAdd(t, t);
  Fe E = feAdd(feAdd(A, A), A);
  Fe F = feSqr(E);
  Jacobian r;
  r.inf = false;
  r.x = feSub(F, feAdd(D, D));
  Fe C2 = feAdd(C, C);
  Fe C4 = feAdd(C2, C2);
  Fe C8 = feAdd(C4, C4);
  r.y = feSub(feMul(E, feSub(D, r.x)), C8);
  Fe yz = feMul(p.y, p.z);
  r.z = feAdd(yz, yz);
  return r;
}

Jacobian jacAddMixed(const Jacobian& p, const Affine& q) {
  if (q.inf) return p;
  if (p.inf) return toJacobian(q);
  Fe z1z1 = feSqr(p.z);
  Fe u2 = feMul(q.x, z1z1);
  Fe s2 = feMul(feMul(q.y, p.z), z1z1);
  Fe h = feSub(u2, p.x);
  Fe r = feSub(s2, p.y);
  if (h.isZero()) {
    if (r.isZero()) return jacDouble(p);
    return Jacobian{};
  }
  Fe hh = feSqr(h);
  Fe hhh = feMul(h, hh);
  Fe v = feMul(p.x, hh);
  Jacobian o;
  o.inf = false;
  o.x = feSub(feSub(feSqr(r), hhh), feAdd(v, v));
  o.y = feSub(feMul(r, feSub(v, o.x)), feMul(p.y, hhh));
  o.z = feMul(p.z, h);
  return o;
}

Affine toAffine(const Jacobian& p) {
  if (p.inf) return Affine{};
  Fe zi = feInv(p.z);
  Fe zi2 = feSqr(zi);
  return Affine{feMul(p.x, zi2), feMul(feMul(p.y, zi2), zi), false};
}

void batchToAffine(const Jacobian* in, Affine* out, size_t n) {
  std::vector<Fe> z(n), scratch(n);
  for (size_t i = 0; i < n; ++i) z[i] = in[i].inf ? Fe{} : in[i].z;
  batchInverse(z.data(), n, scratch.data());
  for (size_t i = 0; i < n; ++i) {
    if (in[i].inf) {
      out[i] = Affine{};
      continue;
    }
    Fe zi2 = feSqr(z[i]);
    out[i] = Affine{feMul(in[i].x, zi2), feMul(feMul(in[i].y, zi2), z[i]), false};
  }
}

Affine add(const Affine& a, const Affine& b) { return toAffine(jacAddMixed(toJacobian(a), b)); }

bool isOnCurve(const Affine& a) {
  if (a.inf) return false;
  Fe rhs = feAdd(feMul(feSqr(a.x), a.x), Fe::fromU64(7));
  return feSqr(a.y) == rhs;
}

std::vector<Affine> progression(const Affine& base, const U256& start, const U256& step, size_t count) {
  std::vector<Affine> out(count);
  if (count == 0) return out;
  std::vector<Jacobian> jac(count);
  jac[0] = jacAddMixed(mulGJacobian(start), base);
  if (count > 1) {
    Affine s = mulG(step);
    for (size_t i = 1; i < count; ++i) jac[i] = jacAddMixed(jac[i - 1], s);
  }
  batchToAffine(jac.data(), out.data(), count);
  return out;
}

void serializeCompressed(const Affine& p, uint8_t out[33]) {
  out[0] = p.y.isOdd() ? 0x03 : 0x02;
  p.x.toBytesBE(out + 1);
}

void serializeUncompressed(const Affine& p, uint8_t out[65]) {
  out[0] = 0x04;
  p.x.toBytesBE(out + 1);
  p.y.toBytesBE(out + 33);
}

Affine parsePubkey(std::string_view hex) {
  auto bytes = hexDecode(hex);
  const std::invalid_argument err("invalid public key");
  if (!bytes) throw err;
  const auto& b = *bytes;
  auto fieldFrom = [&](const uint8_t* p) {
    U256 v = U256::fromBytesBE(p);
    if (!(v < fe_detail::kFieldP)) throw err;
    return Fe::fromU256(v);
  };
  Affine a;
  a.inf = false;
  if (b.size() == 33 && (b[0] == 0x02 || b[0] == 0x03)) {
    a.x = fieldFrom(b.data() + 1);
    Fe rhs = feAdd(feMul(feSqr(a.x), a.x), Fe::fromU64(7));
    if (!feSqrt(rhs, &a.y)) throw err;
    if (a.y.isOdd() != (b[0] == 0x03)) a.y = feNeg(a.y);
  } else if (b.size() == 65 && b[0] == 0x04) {
    a.x = fieldFrom(b.data() + 1);
    a.y = fieldFrom(b.data() + 33);
  } else {
    throw err;
  }
  if (!isOnCurve(a)) throw err;
  return a;
}

std::string pubkeyHex(const Affine& p, bool compressed) {
  if (compressed) {
    uint8_t buf[33];
    serializeCompressed(p, buf);
    return hexEncode(buf, 33);
  }
  uint8_t buf[65];
  serializeUncompressed(p, buf);
  return hexEncode(buf, 65);
}

}  // namespace bsolver

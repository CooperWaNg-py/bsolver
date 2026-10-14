#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "core/field.h"
#include "core/u256.h"

namespace bsolver {

struct Affine {
  Fe x, y;
  bool inf = true;
  friend bool operator==(const Affine& a, const Affine& b) {
    if (a.inf || b.inf) return a.inf == b.inf;
    return a.x == b.x && a.y == b.y;
  }
};

struct Jacobian {
  Fe x, y, z;
  bool inf = true;
};

// Generator G.
const Affine& generator();

Jacobian toJacobian(const Affine& a);
Jacobian jacDouble(const Jacobian& p);
// p + q with q affine; handles p == q (double) and p == -q (infinity).
Jacobian jacAddMixed(const Jacobian& p, const Affine& q);
Affine toAffine(const Jacobian& p);
// One batch inversion for all points; infinity stays infinity.
void batchToAffine(const Jacobian* in, Affine* out, size_t n);

Affine add(const Affine& a, const Affine& b);
bool isOnCurve(const Affine& a);

// base + (start + i*step)*G for i in [0, count). Pass base.inf == true for pure multiples of G.
std::vector<Affine> progression(const Affine& base, const U256& start, const U256& step, size_t count);

void serializeCompressed(const Affine& p, uint8_t out[33]);
void serializeUncompressed(const Affine& p, uint8_t out[65]);
// 33- or 65-byte hex encoding; throws std::invalid_argument("invalid public key").
Affine parsePubkey(std::string_view hex);
// Lowercase hex of the compressed encoding.
std::string pubkeyHex(const Affine& p, bool compressed = true);

}  // namespace bsolver

#pragma once

#include "core/point.h"
#include "core/u256.h"

namespace bsolver {

// k*G via a lazily built table of 32 8-bit windows. The only scalar multiplication in bsolver;
// Q*k is never needed (a point plus k*G is always Q + mulG(k)).
Affine mulG(const U256& k);
Jacobian mulGJacobian(const U256& k);

}  // namespace bsolver

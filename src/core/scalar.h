#pragma once

#include "core/u256.h"

namespace bsolver {

// secp256k1 group order n.
inline constexpr U256 kOrderN{{0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL,
                               0xFFFFFFFFFFFFFFFFULL}};

// Inputs must be < n; results are < n.
U256 addMod(const U256& a, const U256& b);
U256 subMod(const U256& a, const U256& b);
U256 negMod(const U256& a);
// Reduces any 256-bit value modulo n.
U256 reduceMod(const U256& a);

}  // namespace bsolver

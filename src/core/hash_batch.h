#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace bsolver {

// Batched hash160 (RIPEMD160(SHA256(pubkey))) of kHashLanes public keys at once, for the CPU scan hot loop.
//
// Inputs are word-major: x[i][l] is the i-th big-endian 32-bit word (i = 0 is the most significant) of lane l's
// x coordinate; y likewise. parity[l] is y's low bit (0 or 1). Output is word-major too: out[i][l] is RIPEMD-160
// state word i of lane l, so lane l's 20-byte digest is the little-endian bytes of out[0][l] .. out[4][l].
// Every lane is always hashed; callers ignore the lanes they did not fill.
inline constexpr size_t kHashLanes = 8;

using HashLaneWords = uint32_t[8][kHashLanes];
using HashLaneDigests = uint32_t[5][kHashLanes];

struct Hash160Batch {
  // 33-byte compressed encoding 02/03 || x.
  void (*compressed)(const HashLaneWords& x, const uint32_t (&parity)[kHashLanes], HashLaneDigests& out);
  // 65-byte uncompressed encoding 04 || x || y.
  void (*uncompressed)(const HashLaneWords& x, const HashLaneWords& y, HashLaneDigests& out);
};

// Implementations; within one architecture, later ones are preferred. Which ones exist depends on the target
// architecture and the compiler; which ones run depends on the CPU (see supportedHashImpls).
enum class HashImpl : uint8_t {
  Scalar,   // portable C++, every platform
  Sse2,     // x86-64: SSE2 8-lane SHA-256 and RIPEMD-160
  Avx2,     // x86-64: AVX2 8-lane SHA-256 and RIPEMD-160
  ShaNi,    // x86-64: SHA extensions + AVX2 RIPEMD-160 (needs SHA, SSE4.1 and AVX2)
  Neon,     // arm64: NEON 8-lane SHA-256 and RIPEMD-160
  ArmSha2,  // arm64: SHA-256 crypto extension + NEON RIPEMD-160
};

std::string_view hashImplName(HashImpl impl);

// Implementations compiled in and supported by this CPU, best last. Never empty (Scalar is always present).
const std::vector<HashImpl>& supportedHashImpls();

// The implementation table for impl; throws std::invalid_argument when impl is not supported here.
const Hash160Batch& hash160Batch(HashImpl impl);

// The fastest supported implementation (selected once).
HashImpl bestHashImpl();

}  // namespace bsolver

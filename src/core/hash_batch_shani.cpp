// Compiled with the SHA extensions and AVX2 enabled (cmake/CpuHash.cmake); selected only when the CPU has SHA,
// SSE4.1 and AVX2. SHA-256 runs on the SHA unit, RIPEMD-160 in AVX2 lanes.
#define BSOLVER_LANES_AVX2 1
#define BSOLVER_SHA_SHANI 1
#include "core/hash_batch_impl.h"

namespace bsolver::hashimpl {
const Hash160Batch kShaNi{shaNiCompressed<Avx8>, shaNiUncompressed<Avx8>};
}  // namespace bsolver::hashimpl

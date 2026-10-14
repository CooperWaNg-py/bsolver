// Compiled with the ARMv8 SHA-256 instructions enabled (cmake/CpuHash.cmake); selected only when the CPU has them.
#define BSOLVER_LANES_NEON 1
#define BSOLVER_SHA_ARMV8 1
#include "core/hash_batch_impl.h"

namespace bsolver::hashimpl {
const Hash160Batch kArmSha2{armSha2Compressed, armSha2Uncompressed};
}  // namespace bsolver::hashimpl

// Compiled with AVX2 enabled (cmake/CpuHash.cmake); selected only when the CPU and OS support it.
#define BSOLVER_LANES_AVX2 1
#include "core/hash_batch_impl.h"

namespace bsolver::hashimpl {
const Hash160Batch kAvx2{lanesCompressed<Avx8>, lanesUncompressed<Avx8>};
}  // namespace bsolver::hashimpl

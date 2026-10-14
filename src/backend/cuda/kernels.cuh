// CUDA prelude + entry points around the portable kernel core (compiled by nvcc, no runtime compile).
#pragma once

#define KFN __device__ inline
#define PRIV
#define GLOB
#define CONSTAS __constant__
#define MULHI64(a, b) __umul64hi((a), (b))
#define U64 unsigned long long
#define U32 unsigned int
#define ATOMIC_INC(p) atomicAdd((p), 1u)

#include "backend/gpu/kernel_core.h"

__global__ void scan_kernel(const U64* params, const U64* offsets, const U64* gtable, const U32* targets, U32* hits,
                            U32* hitCount) {
  scan_thread(blockIdx.x * blockDim.x + threadIdx.x, params, offsets, gtable, targets, hits, hitCount);
}

__global__ void kangaroo_kernel(const U64* params, U64* state, const U64* jumps, U64* dps, U32* dpCount) {
  kangaroo_thread(blockIdx.x * blockDim.x + threadIdx.x, params, state, jumps, dps, dpCount);
}

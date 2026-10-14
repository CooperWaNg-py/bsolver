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
#define GPU_GROUP 1024
#define SCAN_HIT_CAP 4096
#define KPT 64
#define KANG_COMPONENTS 13
#define KANG_JUMPS 512
#define KANG_ESC_JUMPS 64
#define KANG_WINDOW 64

#include "backend/gpu/kernel_core.h"

__global__ void scan_kernel(const U64* params, const U64* offsets, const U64* gtable, const U32* targets, U32* hits,
                            U32* counters) {
  scan_thread(blockIdx.x * blockDim.x + threadIdx.x, params, offsets, gtable, targets, hits, counters);
}

__global__ void kangaroo_kernel(const U64* params, const U64* in, U64* out, const U64* jumps, U64* dps,
                                U32* counters) {
  kangaroo_thread(blockIdx.x * blockDim.x + threadIdx.x, params, in, out, jumps, dps, counters);
}

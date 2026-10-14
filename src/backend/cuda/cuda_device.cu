#include "backend/cuda/cuda_device.h"

#include <cuda_runtime.h>

#include <stdexcept>

#include "backend/cuda/kernels.cuh"

namespace bsolver::cuda {

namespace {
void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(err));
}

void select(int device) { check(cudaSetDevice(device), "cudaSetDevice"); }
}  // namespace

int deviceCount(std::string* error) {
  int n = 0;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (err != cudaSuccess) {
    *error = std::string("cuda: ") + cudaGetErrorString(err);
    return 0;
  }
  if (n == 0) *error = "no CUDA devices";
  return n;
}

DeviceProps deviceProps(int device) {
  cudaDeviceProp p;
  check(cudaGetDeviceProperties(&p, device), "cudaGetDeviceProperties");
  DeviceProps out;
  out.name = p.name;
  out.memoryBytes = p.totalGlobalMem;
  out.multiprocessors = static_cast<uint32_t>(p.multiProcessorCount);
  return out;
}

uint32_t maxThreadsPerBlock(int device, KernelId kernel) {
  select(device);
  cudaFuncAttributes a;
  if (kernel == KernelId::Scan) {
    check(cudaFuncGetAttributes(&a, scan_kernel), "cudaFuncGetAttributes");
  } else {
    check(cudaFuncGetAttributes(&a, kangaroo_kernel), "cudaFuncGetAttributes");
  }
  return static_cast<uint32_t>(a.maxThreadsPerBlock);
}

void* allocate(int device, size_t bytes) {
  select(device);
  void* p = nullptr;
  check(cudaMalloc(&p, bytes), "cudaMalloc");
  return p;
}

void release(int device, void* p) noexcept {
  if (!p) return;
  cudaSetDevice(device);
  cudaFree(p);
}

void copyToDevice(int device, void* dst, const void* src, size_t bytes) {
  select(device);
  check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
}

void copyToHost(int device, void* dst, const void* src, size_t bytes) {
  select(device);
  check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
}

void launchScan(int device, uint32_t threads, uint32_t block, const void* params, const void* offsets,
                const void* gtable, const void* targets, void* hits, void* hitCount) {
  select(device);
  scan_kernel<<<threads / block, block>>>(static_cast<const U64*>(params), static_cast<const U64*>(offsets),
                                          static_cast<const U64*>(gtable), static_cast<const U32*>(targets),
                                          static_cast<U32*>(hits), static_cast<U32*>(hitCount));
  check(cudaGetLastError(), "scan_kernel launch");
  check(cudaDeviceSynchronize(), "scan_kernel");
}

void launchKangaroo(int device, uint32_t threads, uint32_t block, const void* params, void* state, const void* jumps,
                    void* dps, void* dpCount) {
  select(device);
  kangaroo_kernel<<<threads / block, block>>>(static_cast<const U64*>(params), static_cast<U64*>(state),
                                              static_cast<const U64*>(jumps), static_cast<U64*>(dps),
                                              static_cast<U32*>(dpCount));
  check(cudaGetLastError(), "kangaroo_kernel launch");
  check(cudaDeviceSynchronize(), "kangaroo_kernel");
}

}  // namespace bsolver::cuda

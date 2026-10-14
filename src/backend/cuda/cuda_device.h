#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Thin wrappers over the CUDA runtime and the nvcc-compiled kernels (cuda_device.cu), so that the backend
// itself (cuda_backend.cpp) is plain C++20 compiled by the host compiler. All functions select `device`
// first and throw std::runtime_error("cuda: ...") on failure.
namespace bsolver::cuda {

struct DeviceProps {
  std::string name;
  uint64_t memoryBytes = 0;
  uint32_t multiprocessors = 0;
};

enum class KernelId { Scan, Kangaroo };

// Number of CUDA devices; 0 with *error set when the driver/runtime reports none or fails.
int deviceCount(std::string* error);
DeviceProps deviceProps(int device);
// cudaFuncAttributes::maxThreadsPerBlock of the kernel on `device`.
uint32_t maxThreadsPerBlock(int device, KernelId kernel);

void* allocate(int device, size_t bytes);
void release(int device, void* p) noexcept;
void copyToDevice(int device, void* dst, const void* src, size_t bytes);
void copyToHost(int device, void* dst, const void* src, size_t bytes);

// Run `threads` threads in blocks of `block` and wait for completion. Pointers are device buffers.
void launchScan(int device, uint32_t threads, uint32_t block, const void* params, const void* offsets,
                const void* gtable, const void* targets, void* hits, void* hitCount);
void launchKangaroo(int device, uint32_t threads, uint32_t block, const void* params, void* state, const void* jumps,
                    void* dps, void* dpCount);

}  // namespace bsolver::cuda

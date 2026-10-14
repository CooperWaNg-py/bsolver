#include <thread>

#include "backend/backend.h"
#include "backend/cpu/cpu_backend.h"

#if BSOLVER_HAS_CUDA
#include "backend/cuda/cuda_backend.h"
#endif
#if BSOLVER_HAS_OPENCL
#include "backend/opencl/opencl_backend.h"
#endif
#if BSOLVER_HAS_METAL
#include "backend/metal/metal_backend.h"
#endif

namespace bsolver {

std::vector<Backend*> allBackends() {
  std::vector<Backend*> list;
  list.push_back(cpuBackend());
#if BSOLVER_HAS_CUDA
  list.push_back(cudaBackend());
#endif
#if BSOLVER_HAS_OPENCL
  list.push_back(openclBackend());
#endif
#if BSOLVER_HAS_METAL
  list.push_back(metalBackend());
#endif
  return list;
}

Backend* findBackend(std::string_view id) {
  for (Backend* b : allBackends()) {
    if (b->id() == id) return b;
  }
  return nullptr;
}

unsigned effectiveThreads(unsigned threads) {
  if (threads) return threads;
  unsigned hw = std::thread::hardware_concurrency();
  return hw ? hw : 1;
}

}  // namespace bsolver

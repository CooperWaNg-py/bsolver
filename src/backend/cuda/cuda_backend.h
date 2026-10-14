#pragma once

#include "backend/backend.h"

namespace bsolver {

// Every CUDA device (kernels compiled by nvcc).
Backend* cudaBackend();

}  // namespace bsolver

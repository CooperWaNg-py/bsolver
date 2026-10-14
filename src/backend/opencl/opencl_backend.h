#pragma once

#include "backend/backend.h"

namespace bsolver {

// Every CL_DEVICE_TYPE_GPU device across all OpenCL platforms.
Backend* openclBackend();

}  // namespace bsolver

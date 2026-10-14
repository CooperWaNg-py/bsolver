#pragma once

#include <memory>

#include "backend/backend.h"

namespace bsolver {

std::unique_ptr<ScanWorker> makeCpuScanWorker(const ScanSetup& setup, int index);

}  // namespace bsolver

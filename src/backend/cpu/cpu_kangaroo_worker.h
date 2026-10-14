#pragma once

#include <memory>

#include "backend/backend.h"

namespace bsolver {

// 1024 kangaroos per worker (one worker per CPU thread).
std::unique_ptr<KangarooWorker> makeCpuKangarooWorker(int index);

}  // namespace bsolver

#pragma once

#include "backend/backend.h"
#include "engine/run_context.h"

namespace bsolver {

// Each runner loads/saves its own checkpoint through ctx.checkpointer() (SnapshotScope) and
// reports keys via ctx.reportFound(). They return when the job is found, exhausted or stopped.

// Scan keys below this are enumerated by a CPU worker on the job thread before any backend worker starts, so a
// scan whose range ends below it never uses the selected backend.
inline constexpr U256 kScanSmallLimit{{uint64_t{1} << 20, 0, 0, 0}};

// Returns true when the whole range was scanned.
bool runScan(RunContext& ctx, Backend& backend);
// CPU only. Returns true when the whole range was covered.
bool runBsgs(RunContext& ctx);
// Never exhausts; returns on found or stop.
void runKangaroo(RunContext& ctx, Backend& backend);

}  // namespace bsolver

#pragma once

#include "backend/backend.h"
#include "engine/run_context.h"

namespace bsolver {

// Each runner loads/saves its own checkpoint through ctx.checkpointer() (SnapshotScope) and
// reports keys via ctx.reportFound(). They return when the job is found, exhausted or stopped.

// Returns true when the whole range was scanned.
bool runScan(RunContext& ctx, Backend& backend);
// CPU only. Returns true when the whole range was covered.
bool runBsgs(RunContext& ctx);
// Never exhausts; returns on found or stop.
void runKangaroo(RunContext& ctx, Backend& backend);

}  // namespace bsolver

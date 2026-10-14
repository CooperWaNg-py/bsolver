#include <memory>
#include <stdexcept>

#include "backend/cpu/cpu_scan_worker.h"
#include "core/gtable.h"
#include "engine/bytes.h"
#include "engine/runners.h"
#include "engine/segment_source.h"

namespace bsolver {

namespace {
const U256 kSmallLimit = U256::fromU64(uint64_t{1} << 20);

void verifyCandidates(RunContext& ctx, const std::vector<U256>& candidates) {
  for (const U256& c : candidates) ctx.reportFound(c, mulG(c));
}
}  // namespace

// Payload: sequential `U256 cursor, u64 keysDone` (keysDone = cursor - rangeStart); random `u64 keysDone,
// U256 prefixCursor` (keys tested, and how far the sequential small-key prefix got).
bool runScan(RunContext& ctx, Backend& backend) {
  const ResolvedJob& job = ctx.job();
  const bool sequential = job.order == Order::Sequential;
  ScanSetup setup{job.targets, job.addressMode};

  // Every key in [rangeStart, cursor) is done: the whole range in sequential order, the small-key prefix in random.
  U256 cursor = job.rangeStart;
  uint64_t keysDone = 0;
  if (ctx.checkpointer().restore([&](ByteReader& r) {
        if (sequential) cursor = r.u256();
        keysDone = r.u64();
        // Random checkpoints written before the prefix cursor existed end here.
        if (!sequential && r.remaining() != 0) cursor = r.u256();
        if (cursor < job.rangeStart || job.rangeEnd + 1 < cursor) throw std::runtime_error("cursor out of range");
      })) {
    ctx.addResumedWork(sequential ? (cursor - job.rangeStart).toDouble() : static_cast<double>(keysDone));
    ctx.setResumed(true);
    ctx.log(LogLevel::Info, "resuming from checkpoint " + ctx.checkpointer().path().string() +
                                (sequential ? " at " + cursor.toHex() : std::string()));
  }

  // Keys below 2^20 never go to the workers (GPU kernels assume no small-key edge cases).
  bool prefixDone = true;
  if (cursor < kSmallLimit && !(job.rangeEnd < cursor)) {
    U256 prefixEnd = job.rangeEnd < kSmallLimit ? job.rangeEnd : kSmallLimit - 1;
    auto cpu = makeCpuScanWorker(setup, 0);
    std::vector<U256> candidates;
    while (!(prefixEnd < cursor)) {
      ctx.waitIfPaused();
      if (ctx.shouldStop()) break;
      uint64_t n = std::min<uint64_t>(uint64_t{1} << 16, (prefixEnd - cursor).w[0] + 1);
      candidates.clear();
      cpu->process(cursor, n, candidates);
      verifyCandidates(ctx, candidates);
      ctx.addWork(static_cast<double>(n));
      cursor = cursor + n;
    }
    prefixDone = prefixEnd < cursor;
  }

  const U256 mainStart = sequential ? cursor : (job.rangeStart < kSmallLimit ? kSmallLimit : job.rangeStart);
  std::unique_ptr<SequentialSource> seq;
  std::unique_ptr<RandomSource> rnd;
  if (sequential) {
    seq = std::make_unique<SequentialSource>(mainStart, job.rangeEnd, mainStart);
  } else if (!(job.rangeEnd < mainStart)) {
    rnd = std::make_unique<RandomSource>(mainStart, job.rangeEnd);
  }

  SnapshotScope snapshot(ctx.checkpointer(), [&] {
    ByteWriter w;
    if (sequential) {
      // Saved progress is what the resume cursor covers, not the live count (segments done past an in-flight one
      // are rescanned after a resume).
      const U256 resume = seq->resumeCursor();
      const U256 done = resume - job.rangeStart;
      w.u256(resume);
      w.u64(done.bitLength() > 64 ? UINT64_MAX : done.w[0]);
    } else {
      w.u64(static_cast<uint64_t>(ctx.workDone()));
      w.u256(cursor);
    }
    return std::move(w.data());
  });

  if (!ctx.shouldStop() && ((seq && !seq->finished()) || rnd)) {
    auto workers = backend.createScanWorkers(setup, ctx.spec().devices, ctx.spec().threads);
    if (workers.empty()) throw std::runtime_error("backend " + std::string(backend.id()) + " created no workers");
    runThreads(ctx, workers.size(), [&](size_t i) {
      ScanWorker* w = workers[i].get();
      try {
        std::vector<U256> candidates;
        while (!ctx.shouldStop()) {
          ctx.waitIfPaused();
          if (ctx.shouldStop()) break;
          U256 s;
          uint64_t n = 0;
          bool ok = seq ? seq->next(w->segmentKeys(), &s, &n) : rnd->next(w->segmentKeys(), &s, &n);
          if (!ok) break;
          candidates.clear();
          w->process(s, n, candidates);
          verifyCandidates(ctx, candidates);
          if (seq) seq->complete(s);
          ctx.addWork(static_cast<double>(n));
        }
      } catch (const std::exception& ex) {
        throw std::runtime_error(w->label() + ": " + ex.what());
      }
    });
  }

  // Random order is exhausted only when the small-key prefix covered the whole range.
  bool exhausted = seq ? seq->finished() : (!rnd && prefixDone);
  // Without stopOnFound the job's purpose is the whole range, so a stop keeps the checkpoint.
  snapshot.finish(exhausted || (ctx.spec().stopOnFound && ctx.allFound()));
  return exhausted;
}

}  // namespace bsolver

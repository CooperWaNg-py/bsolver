#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

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

bool runScan(RunContext& ctx, Backend& backend) {
  const ResolvedJob& job = ctx.job();
  const bool sequential = job.order == Order::Sequential;
  ScanSetup setup{job.targets, job.addressMode};

  U256 cursor = job.rangeStart;
  if (auto payload = ctx.checkpointer().load()) {
    ByteReader r(*payload);
    if (sequential) cursor = r.u256();
    uint64_t keysDone = r.u64();
    ctx.addResumedWork(static_cast<double>(keysDone));
    ctx.setResumed(true);
    ctx.log(LogLevel::Info, "resuming from checkpoint " + ctx.checkpointer().path().string() +
                                (sequential ? " at " + cursor.toHex() : std::string()));
  }

  // Keys below 2^20 never go to the workers (GPU kernels assume no small-key edge cases).
  U256 prefixStart = sequential ? cursor : job.rangeStart;
  bool prefixDone = true;
  if (prefixStart < kSmallLimit && !(job.rangeEnd < prefixStart)) {
    U256 prefixEnd = job.rangeEnd < kSmallLimit ? job.rangeEnd : kSmallLimit - 1;
    auto cpu = makeCpuScanWorker(setup, 0);
    std::vector<U256> candidates;
    U256 s = prefixStart;
    while (!(prefixEnd < s)) {
      ctx.waitIfPaused();
      if (ctx.shouldStop()) break;
      uint64_t n = std::min<uint64_t>(uint64_t{1} << 16, (prefixEnd - s).w[0] + 1);
      candidates.clear();
      cpu->process(s, n, candidates);
      verifyCandidates(ctx, candidates);
      ctx.addWork(static_cast<double>(n));
      s = s + n;
    }
    prefixDone = prefixEnd < s;
    if (sequential) cursor = s;
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
    if (seq) w.u256(seq->resumeCursor());
    w.u64(static_cast<uint64_t>(ctx.workDone()));
    return std::move(w.data());
  });

  if (!ctx.shouldStop() && ((seq && !seq->finished()) || rnd)) {
    auto workers = backend.createScanWorkers(setup, ctx.spec().devices, ctx.spec().threads);
    if (workers.empty()) throw std::runtime_error("backend " + std::string(backend.id()) + " created no workers");
    std::mutex errMu;
    std::string error;
    std::vector<std::thread> threads;
    for (auto& worker : workers) {
      threads.emplace_back([&, w = worker.get()] {
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
          std::lock_guard<std::mutex> lock(errMu);
          if (error.empty()) error = w->label() + ": " + ex.what();
          ctx.requestStop();
        }
      });
    }
    for (auto& t : threads) t.join();
    if (!error.empty()) throw std::runtime_error(error);
  }

  // Random order is exhausted only when the small-key prefix covered the whole range.
  bool exhausted = seq ? seq->finished() : (!rnd && prefixDone);
  // Without stopOnFound the job's purpose is the whole range, so a stop keeps the checkpoint.
  snapshot.finish(exhausted || (ctx.spec().stopOnFound && ctx.allFound()));
  return exhausted;
}

}  // namespace bsolver

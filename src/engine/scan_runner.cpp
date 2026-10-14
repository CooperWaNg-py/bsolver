#include <memory>
#include <stdexcept>
#include <string>

#include "backend/cpu/cpu_scan_worker.h"
#include "core/gtable.h"
#include "core/scalar.h"
#include "engine/bytes.h"
#include "engine/runners.h"
#include "engine/segment_source.h"

namespace bsolver {

namespace {
// Keys >= n - 2^20 never go to GPU workers either: their kernels have no fallback for the dx == 0 cases within
// GPU_GROUP keys of n. Those keys are scanned by a CPU worker inside the segment that contains them.
const U256 kTopLimit = kOrderN - kScanSmallLimit;

void verifyCandidates(RunContext& ctx, const std::vector<U256>& candidates) {
  for (const U256& c : candidates) ctx.reportFound(c, mulG(c));
}

// Number of keys of [s, s + n) below kTopLimit.
uint64_t keysBelowTop(const U256& s, uint64_t n) {
  if (!(s < kTopLimit)) return 0;
  const U256 room = kTopLimit - s;
  return room < U256::fromU64(n) ? room.w[0] : n;
}

// GPU launches always cover a whole worker segment (not a power of two, and up to 256x its initial size after the
// iters auto-tune), so the last chunk of each random-order block wastes the rest of a launch; blocks of >= 2^32 keys
// keep that under a few percent. CPU segments (2^20 keys) divide every block exactly.
constexpr unsigned kGpuMinBlockBits = 32;

void writePermState(ByteWriter& w, const PermutedSource::State& s) {
  w.u8(static_cast<uint8_t>(s.blockBits));
  w.bytes(s.key.data(), s.key.size());
  w.u256(s.cursor);
  w.u256(s.keysDone);
  w.u32(static_cast<uint32_t>(s.pending.size()));
  for (const PermutedSource::Chunk& c : s.pending) {
    w.u256(c.pos);
    w.u64(c.len);
  }
}

PermutedSource::State readPermState(ByteReader& r) {
  PermutedSource::State s;
  s.blockBits = r.u8();
  r.bytes(s.key.data(), s.key.size());
  s.cursor = r.u256();
  s.keysDone = r.u256();
  for (uint32_t n = r.u32(); n != 0; --n) {
    const U256 pos = r.u256();
    s.pending.push_back({pos, r.u64()});
  }
  return s;
}

std::string blockSummary(const PermutedSource& p) {
  const U256& n = p.blocks();
  const std::string count = n.bitLength() <= 64 ? std::to_string(n.w[0]) : "~2^" + std::to_string(n.bitLength() - 1);
  return "random order: " + count + " blocks of 2^" + std::to_string(p.blockBits()) + " keys";
}
}  // namespace

// Payload: sequential `U256 cursor, u64 keysDone` (keysDone = cursor - rangeStart). Random: `U256 prefixCursor` (how
// far the sequential small-key prefix got) and, when the range reaches 2^20, the block permutation state `u8 blockBits,
// 16-byte key, U256 cursor, U256 keysDone, u32 n, n x (U256 pos, u64 len)` (chunks handed out but not completed).
bool runScan(RunContext& ctx, Backend& backend) {
  const ResolvedJob& job = ctx.job();
  const bool sequential = job.order == Order::Sequential;
  const bool gpu = backend.id() != "cpu";
  ScanSetup setup{job.targets, job.addressMode};
  // Random order visits [blockStart, rangeEnd] as permuted blocks, after the sequential small-key prefix.
  const U256 blockStart = job.rangeStart < kScanSmallLimit ? kScanSmallLimit : job.rangeStart;
  const bool hasBlocks = !sequential && !(job.rangeEnd < blockStart);

  // Every key in [rangeStart, cursor) is done: the whole range in sequential order, the small-key prefix in random.
  U256 cursor = job.rangeStart;
  std::unique_ptr<PermutedSource> perm;
  U256 blockKeysDone;
  if (ctx.checkpointer().restore([&](ByteReader& r) {
        cursor = r.u256();
        if (sequential) r.u64();  // keysDone, implied by the cursor
        if (cursor < job.rangeStart || job.rangeEnd + 1 < cursor) throw std::runtime_error("cursor out of range");
        if (hasBlocks) {
          PermutedSource::State state = readPermState(r);
          blockKeysDone = state.keysDone;
          perm = std::make_unique<PermutedSource>(blockStart, job.rangeEnd, std::move(state));
        }
      })) {
    ctx.addResumedWork((cursor - job.rangeStart).toDouble() + blockKeysDone.toDouble());
    ctx.setResumed(true);
    ctx.log(LogLevel::Info, "resuming from checkpoint " + ctx.checkpointer().path().string() +
                                (sequential ? " at " + cursor.toHex() : std::string()));
  }
  if (hasBlocks && !perm) {
    const unsigned minBits = gpu ? kGpuMinBlockBits : PermutedSource::kMinBlockBits;
    perm = std::make_unique<PermutedSource>(blockStart, job.rangeEnd,
                                            PermutedSource::fresh(job.rangeEnd - blockStart + 1, minBits));
  }
  if (perm) ctx.log(LogLevel::Info, blockSummary(*perm));

  // Keys below 2^20 never go to the workers (GPU kernels assume no small-key edge cases); keys >= kTopLimit go to a
  // CPU worker instead of a GPU one (see kTopLimit).
  bool prefixDone = true;
  if (cursor < kScanSmallLimit && !(job.rangeEnd < cursor)) {
    U256 prefixEnd = job.rangeEnd < kScanSmallLimit ? job.rangeEnd : kScanSmallLimit - 1;
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

  std::unique_ptr<SequentialSource> seq;
  if (sequential) seq = std::make_unique<SequentialSource>(cursor, job.rangeEnd, cursor);
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
      w.u256(cursor);
      if (perm) writePermState(w, perm->snapshot());
    }
    return std::move(w.data());
  });

  if (!ctx.shouldStop() && (seq ? !seq->finished() : perm && !perm->finished())) {
    auto workers = backend.createScanWorkers(setup, ctx.spec().devices, ctx.spec().threads);
    if (workers.empty()) throw std::runtime_error("backend " + std::string(backend.id()) + " created no workers");
    runThreads(ctx, workers.size(), [&](size_t i) {
      ScanWorker* w = workers[i].get();
      std::unique_ptr<ScanWorker> topWorker;  // created on first use
      try {
        std::vector<U256> candidates;
        while (!ctx.shouldStop()) {
          ctx.waitIfPaused();
          if (ctx.shouldStop()) break;
          U256 s;
          uint64_t n = 0;
          bool ok = seq ? seq->next(w->segmentKeys(), &s, &n) : perm->next(w->segmentKeys(), &s, &n);
          if (!ok) break;
          candidates.clear();
          const uint64_t below = gpu ? keysBelowTop(s, n) : n;
          if (below != 0) w->process(s, below, candidates);
          if (below < n) {
            if (!topWorker) topWorker = makeCpuScanWorker(setup, 0);
            for (uint64_t done = below; done < n;) {
              const uint64_t len = std::min(n - done, topWorker->segmentKeys());
              topWorker->process(s + done, len, candidates);
              done += len;
            }
          }
          verifyCandidates(ctx, candidates);
          if (seq) {
            seq->complete(s);
          } else {
            perm->complete(s);
          }
          ctx.addWork(static_cast<double>(n));
        }
      } catch (const std::exception& ex) {
        throw std::runtime_error(w->label() + ": " + ex.what());
      }
    });
  }

  bool exhausted = seq ? seq->finished() : prefixDone && (!perm || perm->finished());
  // Without stopOnFound the job's purpose is the whole range, so a stop keeps the checkpoint.
  snapshot.finish(exhausted || (ctx.spec().stopOnFound && ctx.allFound()));
  return exhausted;
}

}  // namespace bsolver

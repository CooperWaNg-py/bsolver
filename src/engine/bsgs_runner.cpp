#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/gtable.h"
#include "core/point_enum.h"
#include "core/scalar.h"
#include "engine/bytes.h"
#include "engine/format.h"
#include "engine/runners.h"

namespace bsolver {

namespace {

constexpr uint64_t kLanes = 256;
constexpr uint64_t kLaneSteps = 1024;
constexpr uint64_t kBlockGiants = kLanes * kLaneSteps;  // giant indices per work unit
constexpr unsigned kBlockShift = 18;                     // log2(kBlockGiants)
constexpr uint64_t kFillChunk = uint64_t{1} << 18;       // baby steps per fill task
static_assert(kBlockGiants == uint64_t{1} << kBlockShift);

inline void prefetchRead(const void* p) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(p);
#else
  (void)p;
#endif
}

// Open-addressing baby-step table {x low 64 bits -> j} over a power-of-two capacity, indexed by the
// high bits of xlo with linear probing. xlo == 0 marks an empty slot; a genuine zero is stored as 1.
// Equal keys may occupy several slots; lookups report every one of them.
class BabyTable {
 public:
  explicit BabyTable(uint64_t m) {
    const uint64_t slots = bsgsTableSlots(m);
    mask_ = slots - 1;
    shift_ = 64 - static_cast<unsigned>(std::countr_zero(slots));
    slots_.reset(new Slot[slots]());
  }

  uint64_t bytes() const { return (mask_ + 1) * sizeof(Slot); }

  // Thread-safe against concurrent inserts (not against concurrent lookups).
  void insert(uint64_t xlo, uint64_t j) {
    const uint64_t key = xlo ? xlo : 1;
    for (uint64_t idx = key >> shift_;; idx = (idx + 1) & mask_) {
      std::atomic_ref<uint64_t> slot(slots_[idx].xlo);
      uint64_t expected = 0;
      if (slot.load(std::memory_order_relaxed) == 0 &&
          slot.compare_exchange_strong(expected, key, std::memory_order_relaxed)) {
        slots_[idx].j = j;
        return;
      }
    }
  }

  void prefetch(uint64_t xlo) const { prefetchRead(&slots_[(xlo ? xlo : 1) >> shift_]); }

  template <class Fn>
  void lookup(uint64_t xlo, Fn&& fn) const {
    const uint64_t key = xlo ? xlo : 1;
    for (uint64_t idx = key >> shift_; slots_[idx].xlo != 0; idx = (idx + 1) & mask_) {
      if (slots_[idx].xlo == key) fn(slots_[idx].j);
    }
  }

 private:
  struct Slot {
    uint64_t xlo;
    uint64_t j;
  };
  static_assert(sizeof(Slot) == kBsgsSlotBytes);
  std::unique_ptr<Slot[]> slots_;
  uint64_t mask_ = 0;
  unsigned shift_ = 0;
};

// Hands out giant-step blocks in order and tracks in-flight ones so the checkpoint cursor (minimum
// unfinished block) never skips work. Thread-safe.
class BlockSource {
 public:
  BlockSource(uint64_t next, uint64_t total) : next_(next), total_(total) {}

  bool next(uint64_t* block) {
    std::lock_guard<std::mutex> lock(mu_);
    if (next_ >= total_) return false;
    *block = next_++;
    inFlight_.insert(*block);
    return true;
  }
  void complete(uint64_t block) {
    std::lock_guard<std::mutex> lock(mu_);
    inFlight_.erase(block);
  }
  uint64_t resumeBlock() const {
    std::lock_guard<std::mutex> lock(mu_);
    return inFlight_.empty() ? next_ : *inFlight_.begin();
  }
  bool finished() const {
    std::lock_guard<std::mutex> lock(mu_);
    return next_ >= total_ && inFlight_.empty();
  }

 private:
  mutable std::mutex mu_;
  uint64_t next_, total_;
  std::set<uint64_t> inFlight_;
};

// Shared, read-only parameters of one BSGS run. Key offsets k' = key - a lie in [0, W-1]; giant index i
// has center c_i = m + i*s and covers k' in [i*s, i*s + 2m] via R_i = Q' - c_i*G = +-j*G (j <= m).
struct Bsgs {
  RunContext& ctx;
  const BabyTable& table;
  Affine Q, Qp;      // target and Q' = Q - a*G
  U256 a, W;         // range start, width
  uint64_t m, s;     // baby steps, giant stride 2m+1
  U256 giants;       // number of giant indices: floor((W-1)/s) + 1
  Affine negS;       // -s*G
  Affine negLaneS;   // -(kLaneSteps*s)*G

  U256 center(const U256& i) const { return mulU64(i, s) + m; }
  // R_i = Q' - c*G.
  Affine giantPoint(const U256& c) const { return add(Qp, mulG(negMod(reduceMod(c)))); }

  // Verifies key offset k' against Q and reports it.
  void tryOffset(const U256& off) const {
    if (W - 1 < off) return;
    U256 key = a + off;
    Affine P = mulG(key);
    if (P == Q) ctx.reportFound(key, P);
  }

  // Keys covered by the giant indices [i0, i0 + count) (clipped to the range).
  double coveredWork(const U256& i0, uint64_t count) const {
    U256 lo = mulU64(i0, s);
    U256 hi = lo + mulU64(U256::fromU64(count), s);
    return ((W < hi ? W : hi) - lo).toDouble();
  }
};

// Fills the baby-step table with x(j*G) for j in [1, m] using `threads` threads.
void fillTable(RunContext& ctx, BabyTable& table, uint64_t m, unsigned threads) {
  std::atomic<uint64_t> nextChunk{0};
  const uint64_t chunks = (m + kFillChunk - 1) / kFillChunk;
  runThreads(ctx, threads, [&](size_t) {
    auto en = std::make_unique<PointEnumerator<1024>>();
    for (;;) {
      ctx.waitIfPaused();
      if (ctx.shouldStop()) return;
      uint64_t c = nextChunk.fetch_add(1);
      if (c >= chunks) return;
      uint64_t first = 1 + c * kFillChunk;
      uint64_t count = std::min(kFillChunk, m - (first - 1));
      en->run(U256::fromU64(first), count,
              [&](uint64_t off, const Affine& P) { table.insert(P.x.n[0], first + off); });
    }
  });
}

// Processes one block of giant indices: kLanes lanes of kLaneSteps consecutive indices each, all
// lanes stepping together by -s*G with one batch inversion per step. Returns false when interrupted.
class BlockWorker {
 public:
  explicit BlockWorker(const Bsgs& b) : b_(b) {}

  bool run(uint64_t block) {
    const U256 i0 = U256::fromU64(block) << kBlockShift;
    const U256 left = b_.giants - i0;
    const uint64_t n = left < U256::fromU64(kBlockGiants) ? left.w[0] : kBlockGiants;
    const size_t lanes = static_cast<size_t>((n + kLaneSteps - 1) / kLaneSteps);
    const uint64_t steps = std::min(kLaneSteps, n);

    // Lane l starts at giant index i0 + l*kLaneSteps.
    jac_[0] = toJacobian(b_.giantPoint(b_.center(i0)));
    for (size_t l = 1; l < lanes; ++l) jac_[l] = jacAddMixed(jac_[l - 1], b_.negLaneS);
    batchToAffine(jac_, pts_, lanes);
    for (size_t l = 0; l < lanes; ++l) {
      if (!pts_[l].inf) b_.table.prefetch(pts_[l].x.n[0]);
    }

    auto giantIndex = [&](size_t l, uint64_t t) { return i0 + (l * kLaneSteps + t); };
    for (uint64_t t = 0; t < steps; ++t) {
      if (b_.ctx.shouldStop()) return false;
      for (size_t l = 0; l < lanes; ++l) {
        const Affine& P = pts_[l];
        if (P.inf) {  // R_i = O: k' = c_i
          b_.tryOffset(b_.center(giantIndex(l, t)));
          continue;
        }
        b_.table.lookup(P.x.n[0], [&](uint64_t j) {
          U256 c = b_.center(giantIndex(l, t));
          b_.tryOffset(c - j);
          b_.tryOffset(c + j);
        });
      }
      if (t + 1 == steps) break;

      for (size_t l = 0; l < lanes; ++l) {
        const Affine& P = pts_[l];
        reinit_[l] = P.inf;
        dx_[l] = P.inf ? Fe{} : feSub(b_.negS.x, P.x);
        if (!P.inf && dx_[l].isZero()) {  // R_i = +-s*G
          U256 c = b_.center(giantIndex(l, t));
          b_.tryOffset(c + b_.s);
          if (!(c < U256::fromU64(b_.s))) b_.tryOffset(c - b_.s);
          reinit_[l] = true;
        }
      }
      batchInverse(dx_, lanes, scratch_);
      for (size_t l = 0; l < lanes; ++l) {
        Affine& P = pts_[l];
        if (reinit_[l]) {
          P = b_.giantPoint(b_.center(giantIndex(l, t + 1)));
        } else {
          const Affine& S = b_.negS;
          Fe lam = feMul(feSub(S.y, P.y), dx_[l]);
          Fe x = feSub(feSub(feSqr(lam), P.x), S.x);
          P.y = feSub(feMul(lam, feSub(P.x, x)), P.y);
          P.x = x;
        }
        if (!P.inf) b_.table.prefetch(P.x.n[0]);
      }
    }
    return true;
  }

 private:
  const Bsgs& b_;
  Jacobian jac_[kLanes];
  Affine pts_[kLanes];
  Fe dx_[kLanes], scratch_[kLanes];
  bool reinit_[kLanes];
};

}  // namespace

bool runBsgs(RunContext& ctx) {
  const ResolvedJob& job = ctx.job();
  if (!job.pubkey) throw std::runtime_error("bsgs requires a public key");
  const unsigned threads = effectiveThreads(ctx.spec().threads);

  const U256 W = job.width();
  const uint64_t m = bsgsBabySteps(W, ctx.spec().bsgsMemoryMb);
  const uint64_t s = 2 * m + 1;
  const U256 giants = divU64(W - 1, s) + 1;
  const U256 blocks = (giants + (kBlockGiants - 1)) >> kBlockShift;
  // Beyond 2^64 blocks the search cannot finish anyway; saturate.
  const uint64_t totalBlocks = blocks > U256::fromU64(UINT64_MAX) ? UINT64_MAX : blocks.w[0];

  uint64_t startBlock = 0;
  uint64_t savedBlock = 0, savedM = 0;
  if (ctx.checkpointer().restore([&](ByteReader& r) {
        savedBlock = r.u64();
        savedM = r.u64();
      })) {
    if (savedM == m) {
      startBlock = std::min(savedBlock, totalBlocks);
      ctx.setResumed(true);
      ctx.log(LogLevel::Info, "resuming from checkpoint " + ctx.checkpointer().path().string() + " at giant block " +
                                  std::to_string(startBlock) + "/" + std::to_string(totalBlocks));
    } else {
      ctx.log(LogLevel::Warn, "checkpoint " + ctx.checkpointer().path().string() + " used " + std::to_string(savedM) +
                                  " baby steps, this run uses " + std::to_string(m) +
                                  " (--bsgs-mem changed); restarting from block 0");
    }
  }

  BlockSource source(startBlock, totalBlocks);
  SnapshotScope snapshot(ctx.checkpointer(), [&] {
    ByteWriter w;
    w.u64(source.resumeBlock());
    w.u64(m);
    return std::move(w.data());
  });

  const Affine Q = *job.pubkey;
  const Affine Qp = add(Q, mulG(negMod(job.rangeStart)));
  if (Qp.inf) {  // Q = a*G
    ctx.reportFound(job.rangeStart, Q);
    snapshot.finish(true);
    return false;
  }
  if (startBlock > 0) {
    U256 done = mulU64(U256::fromU64(startBlock) << kBlockShift, s);
    ctx.addResumedWork((W < done ? W : done).toDouble());
  }

  std::unique_ptr<BabyTable> table;
  try {
    table = std::make_unique<BabyTable>(m);
  } catch (const std::bad_alloc&) {
    throw std::runtime_error("cannot allocate the baby-step table for " + std::to_string(m) +
                             " baby steps; lower --bsgs-mem");
  }
  ctx.log(LogLevel::Info, "bsgs: " + std::to_string(m) + " baby steps (" + std::to_string(table->bytes() >> 20) +
                              " MiB table), giant stride " + std::to_string(s) + ", " + std::to_string(totalBlocks) +
                              " giant blocks, " + std::to_string(threads) + " threads");
  const auto fillStart = std::chrono::steady_clock::now();
  fillTable(ctx, *table, m, threads);
  if (ctx.shouldStop()) {
    snapshot.finish(ctx.anyFound());
    return false;
  }
  const std::chrono::duration<double> fillTime = std::chrono::steady_clock::now() - fillStart;
  ctx.log(LogLevel::Info, "bsgs: baby-step table built in " + formatDuration(fillTime.count()));

  const Bsgs b{ctx,
               *table,
               Q,
               Qp,
               job.rangeStart,
               W,
               m,
               s,
               giants,
               mulG(negMod(U256::fromU64(s))),
               mulG(negMod(mulU64(U256::fromU64(s), kLaneSteps)))};

  runThreads(ctx, threads, [&](size_t) {
    try {
      auto worker = std::make_unique<BlockWorker>(b);
      while (!ctx.shouldStop()) {
        ctx.waitIfPaused();
        if (ctx.shouldStop()) break;
        uint64_t block;
        if (!source.next(&block)) break;
        if (!worker->run(block)) break;
        source.complete(block);
        ctx.addWork(b.coveredWork(U256::fromU64(block) << kBlockShift, kBlockGiants));
      }
    } catch (const std::exception& ex) {
      throw std::runtime_error(std::string("bsgs: ") + ex.what());
    }
  });

  const bool exhausted = source.finished();
  snapshot.finish(exhausted || ctx.anyFound());
  return exhausted;
}

}  // namespace bsolver

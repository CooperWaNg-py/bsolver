#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "core/gtable.h"
#include "core/random.h"
#include "core/scalar.h"
#include "engine/bytes.h"
#include "engine/kangaroo_params.h"
#include "engine/kangaroo_table.h"
#include "engine/runners.h"

namespace bsolver {

namespace {
using Clock = std::chrono::steady_clock;

constexpr auto kInvalidDpWarnInterval = std::chrono::minutes(1);
constexpr size_t kVerifyChunk = 512;
// DP overhead ratio nK*2^dp/sqrt(W) above which the run is noticeably slowed (+35% of the 1.15*sqrt(W) walk).
constexpr long double kHighDpOverhead = 0.5L;

// Marks valid[i] for each DP whose x matches dist*G (tame) or wildBase + dist*G (wild). One batch inversion per
// chunk; large batches are spread over the hardware threads (GPU workers can return hundreds of thousands).
void verifyDps(RunContext& ctx, const std::vector<Dp>& dps, const Affine& wildBase, std::vector<uint8_t>& valid) {
  valid.assign(dps.size(), 0);
  auto verifyRange = [&](size_t begin, size_t end) {
    std::vector<Jacobian> jac;
    std::vector<Affine> aff;
    for (size_t c = begin; c < end; c += kVerifyChunk) {
      size_t n = std::min(kVerifyChunk, end - c);
      jac.resize(n);
      aff.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const Dp& d = dps[c + i];
        jac[i] = mulGJacobian(signedToScalar(d.dist));
        if (d.wild) jac[i] = jacAddMixed(jac[i], wildBase);
      }
      batchToAffine(jac.data(), aff.data(), n);
      for (size_t i = 0; i < n; ++i) valid[c + i] = !aff[i].inf && aff[i].x.toU256() == dps[c + i].x;
    }
  };
  size_t threads = std::min<size_t>(effectiveThreads(0), (dps.size() + kVerifyChunk - 1) / kVerifyChunk);
  if (threads <= 1) {
    verifyRange(0, dps.size());
    return;
  }
  size_t per = (dps.size() + threads - 1) / threads;
  runThreads(ctx, threads, [&](size_t t) {
    size_t begin = t * per, end = std::min(dps.size(), begin + per);
    if (begin < end) verifyRange(begin, end);
  });
}

// Kangaroo states handed from the worker threads to the checkpoint snapshot, which must not touch the workers
// itself (GPU state can only be read between launches by the thread that launches). A snapshot raises the
// generation and waits until every worker has deposited its states after a step, or is idle (paused, finished
// or never started) with its last deposit.
class StateStage {
 public:
  explicit StateStage(size_t workers) : slots_(workers) {}

  // Worker thread i, between steps: deposits the states if a snapshot is waiting.
  void serve(size_t i, KangarooWorker& w) {
    const uint64_t g = generation_.load();
    if (g == slots_[i].servedLocal) return;
    slots_[i].servedLocal = g;
    deposit(i, w.sample(w.kangarooCount()), g, false);
  }
  // Worker thread i before blocking or exiting: deposits the current states and marks the slot idle.
  void park(size_t i, KangarooWorker& w) { deposit(i, w.sample(w.kangarooCount()), generation_.load(), true); }
  void unpark(size_t i) {
    std::lock_guard<std::mutex> lock(mu_);
    slots_[i].idle = false;
  }
  // All workers are gone: snapshots stop waiting.
  void close() {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    cv_.notify_all();
  }

  // States of all workers in global order, or empty when some worker never deposited.
  std::vector<KangarooState> collect() {
    const uint64_t g = ++generation_;
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] {
      return closed_ || std::all_of(slots_.begin(), slots_.end(), [&](const Slot& s) { return s.idle || s.served >= g; });
    });
    std::vector<KangarooState> out;
    for (const Slot& s : slots_) {
      if (!s.valid) return {};
      out.insert(out.end(), s.states.begin(), s.states.end());
    }
    return out;
  }

 private:
  struct Slot {
    uint64_t servedLocal = 0;  // worker thread only
    uint64_t served = 0;
    bool idle = false, valid = false;
    std::vector<KangarooState> states;
  };

  void deposit(size_t i, std::vector<KangarooState> states, uint64_t g, bool idle) {
    std::lock_guard<std::mutex> lock(mu_);
    Slot& s = slots_[i];
    s.states = std::move(states);
    s.valid = true;
    s.served = std::max(s.served, g);
    s.idle = idle;
    cv_.notify_all();
  }

  std::mutex mu_;
  std::condition_variable cv_;
  std::atomic<uint64_t> generation_{0};
  std::vector<Slot> slots_;
  bool closed_ = false;
};

std::string fmt2(long double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.2Lf", v);
  return buf;
}
}  // namespace

void runKangaroo(RunContext& ctx, Backend& backend) {
  const ResolvedJob& job = ctx.job();
  if (!job.pubkey) throw std::runtime_error("algorithm kangaroo requires a public key");
  const Affine& q = *job.pubkey;
  const U256 width = job.width();
  const U256 shift = kangarooShift(width);
  const Affine wildBase = wildBaseFor(q, job.rangeStart, shift);
  if (wildBase.inf) {
    // Q == (a - shift)*G: no walk needed (and the wild herd would start at infinity).
    ctx.reportFound(kangarooKey(job.rangeStart, shift, U256{}), q);
    return;
  }

  auto workers = backend.createKangarooWorkers(ctx.spec().devices, ctx.spec().threads);
  if (workers.empty()) throw std::runtime_error("backend " + std::string(backend.id()) + " created no workers");
  std::vector<uint64_t> firstIndex;
  uint64_t nK = 0;
  for (const auto& w : workers) {
    firstIndex.push_back(nK);
    nK += w->kangarooCount();
  }

  KangarooTable table;
  KangarooCheckpoint header;
  std::vector<KangarooState> saved;
  Rng rng;
  const bool restored =
      ctx.checkpointer().restore([&](ByteReader& r) { header = readKangarooCheckpoint(r, table, saved); });
  const std::optional<uint32_t> userDp = ctx.spec().dpBits;
  if (restored) {
    ctx.addResumedWork(static_cast<double>(header.jumps));
    ctx.setResumed(true);
    ctx.log(LogLevel::Info, "resuming from checkpoint " + ctx.checkpointer().path().string() + ": " +
                                std::to_string(table.size()) + " distinguished points, " +
                                std::to_string(saved.size()) + " kangaroo states, " + std::to_string(header.jumps) +
                                " jumps");
    // Kangaroo states carry the walks across runs, so the dp may change: the stored DPs remain valid points.
    const uint64_t design = std::max<uint64_t>(header.designKangaroos, 1);
    const bool mismatch = nK > 2 * design || design > 2 * nK;
    if (userDp && *userDp != header.dpBits) {
      ctx.log(LogLevel::Info, "dp " + std::to_string(header.dpBits) + " from the checkpoint replaced by --dp " +
                                  std::to_string(*userDp));
      header.dpBits = *userDp;
      header.designKangaroos = nK;
    } else if (!userDp && mismatch) {
      const uint32_t dp = autoDpBits(width, nK);
      ctx.log(LogLevel::Warn, "checkpoint dp " + std::to_string(header.dpBits) + " was chosen for " +
                                  std::to_string(design) + " kangaroos, resuming with " + std::to_string(nK) +
                                  "; using dp " + std::to_string(dp));
      header.dpBits = dp;
      header.designKangaroos = nK;
    }
    if (saved.size() != nK) {
      ctx.log(LogLevel::Info, "checkpoint holds " + std::to_string(saved.size()) + " kangaroo states for " +
                                  std::to_string(nK) + " kangaroos; " +
                                  (saved.size() > nK ? "dropping the rest" : "starting the rest fresh"));
      if (saved.size() > nK) saved.resize(nK);
    }
  } else {
    header.dpBits = userDp.value_or(autoDpBits(width, nK));
    header.designKangaroos = nK;
    header.dist = makeJumpDistances(width, rng);
  }
  const uint32_t dpBits = header.dpBits;

  const long double ratio = static_cast<long double>(nK) * std::ldexp(1.0L, static_cast<int>(dpBits)) /
                            std::sqrt(width.toLongDouble());
  if (ratio > kHighDpOverhead) ctx.log(LogLevel::Warn, "dp overhead high; consider lower --dp");
  const double expected = kangarooExpectedWork(width, nK, dpBits);
  ctx.setWorkExpected(expected);
  ctx.log(LogLevel::Info, "kangaroo (symmetric): " + std::to_string(nK) + " kangaroos on " +
                              std::to_string(workers.size()) + " workers, dp " + std::to_string(dpBits) +
                              " (overhead ratio " + fmt2(ratio) + "), expected 2^" + fmt2(std::log2(expected)) +
                              " jumps");

  KangarooParams params = makeKangarooParams(width, wildBase, dpBits, header.dist);
  params.totalKangaroos = nK;
  if (!saved.empty()) restoreKangarooPoints(params, saved, rng.next());

  StateStage stage(workers.size());
  SnapshotScope snapshot(ctx.checkpointer(), [&] {
    // Workers add their jumps before depositing, so parked (paused/finished) workers match the count exactly.
    const std::vector<KangarooState> states = stage.collect();
    KangarooCheckpoint h = header;
    h.jumps = static_cast<uint64_t>(ctx.workDone());
    return writeKangarooCheckpoint(h, table, states);
  });

  const bool verify = backend.id() != "cpu";
  const U256& a = job.rangeStart;
  auto done = [&] { return ctx.shouldStop() || ctx.anyFound(); };
  auto solve = [&](const KangarooTable::Entry& e1, const KangarooTable::Entry& e2) {
    for (const U256& k : collisionCandidates(e1.wild, e1.dist, e2.wild, e2.dist)) {
      const U256 key = kangarooKey(a, shift, k);
      Affine p = mulG(key);
      if (p == q) return ctx.reportFound(key, p);
    }
    return false;
  };
  std::vector<uint64_t> seeds(workers.size());
  for (auto& s : seeds) s = rng.next();
  struct CloseStage {
    StateStage& s;
    ~CloseStage() { s.close(); }
  } closeStage{stage};
  runThreads(ctx, workers.size(), [&](size_t i) {
    KangarooWorker& w = *workers[i];
    try {
      struct Park {
        StateStage& stage;
        KangarooWorker& w;
        size_t i;
        bool initialized = false;
        ~Park() {
          if (!initialized) return;
          try {
            stage.park(i, w);
          } catch (...) {
          }
        }
      } park{stage, w, i};
      KangarooParams p = params;
      p.firstKangaroo = firstIndex[i];
      const size_t begin = std::min<size_t>(firstIndex[i], saved.size());
      const size_t end = std::min<size_t>(firstIndex[i] + w.kangarooCount(), saved.size());
      w.init(p, seeds[i], std::vector<KangarooState>(saved.begin() + begin, saved.begin() + end));
      park.initialized = true;
      Rng resetRng(seeds[i] ^ 0x5851f42d4c957f2dULL);
      std::vector<Dp> dps;
      std::vector<uint8_t> valid;
      uint64_t invalid = 0;
      std::optional<Clock::time_point> lastWarn;
      while (!done()) {
        if (ctx.paused()) {
          stage.park(i, w);
          ctx.waitIfPaused();
          stage.unpark(i);
          continue;
        }
        dps.clear();
        ctx.addWork(static_cast<double>(w.step(dps)));
        if (verify) verifyDps(ctx, dps, wildBase, valid);
        for (size_t k = 0; k < dps.size() && !ctx.anyFound(); ++k) {
          const Dp& d = dps[k];
          if (verify && !valid[k]) {
            ++invalid;
            continue;
          }
          const KangarooTable::Entry entry{signedToScalar(d.dist), d.wild};
          auto existing = table.insert(X128::of(d.x), entry);
          if (!existing) continue;
          if (collisionCandidates(existing->wild, existing->dist, entry.wild, entry.dist).empty()) {
            // Tame-tame or a merged wild-wild pair: the newcomer walks a known trail.
            w.reset(d.index, resetRng.next());
          } else if (!solve(*existing, entry)) {
            ctx.log(LogLevel::Warn, "false collision");
          }
        }
        stage.serve(i, w);
        auto now = Clock::now();
        if (invalid > 0 && (!lastWarn || now - *lastWarn >= kInvalidDpWarnInterval)) {
          ctx.log(LogLevel::Warn, "discarded " + std::to_string(invalid) + " invalid DPs from " + w.label());
          invalid = 0;
          lastWarn = now;
        }
      }
    } catch (const std::exception& ex) {
      throw std::runtime_error(w.label() + ": " + ex.what());
    }
  });
  snapshot.finish(ctx.anyFound());
}

}  // namespace bsolver

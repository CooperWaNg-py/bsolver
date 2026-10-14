#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
        jac[i] = d.wild ? jacAddMixed(mulGJacobian(d.dist), wildBase) : mulGJacobian(d.dist);
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

// Tame/wild collision on x: T = W gives k' = tDist - wDist, T = -W gives k' = -(tDist + wDist) (mod n).
bool solveCollision(RunContext& ctx, const U256& tameDist, const U256& wildDist) {
  const U256& a = ctx.job().rangeStart;
  const U256 t = reduceMod(tameDist), w = reduceMod(wildDist);
  for (const U256& key : {addMod(a, subMod(t, w)), addMod(a, negMod(addMod(t, w)))}) {
    Affine p = mulG(key);
    if (p == *ctx.job().pubkey) return ctx.reportFound(key, p);
  }
  return false;
}
}  // namespace

void runKangaroo(RunContext& ctx, Backend& backend) {
  const ResolvedJob& job = ctx.job();
  if (!job.pubkey) throw std::runtime_error("algorithm kangaroo requires a public key");
  const U256 width = job.width();
  const Affine wildBase = wildBaseFor(*job.pubkey, job.rangeStart);
  if (wildBase.inf) {
    // Q == a*G: no walk needed (and the wild herd would start at infinity).
    ctx.reportFound(job.rangeStart, *job.pubkey);
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
  uint32_t dpBits = 0;
  std::array<U256, 64> jumpDist;
  Rng rng;
  uint64_t resumedJumps = 0;
  if (ctx.checkpointer().restore([&](ByteReader& r) {
        dpBits = r.u32();
        if (dpBits > kMaxDpBits) throw std::runtime_error("dp bits out of range");
        for (auto& d : jumpDist) d = r.u256();
        resumedJumps = r.u64();
        uint64_t count = r.u64();
        for (uint64_t i = 0; i < count; ++i) {
          X128 x;
          x.lo = r.u64();
          x.hi = r.u64();
          U256 dist = r.u256();
          bool wild = r.u8() != 0;
          table.insert(x, {dist, wild});
        }
      })) {
    ctx.addResumedWork(static_cast<double>(resumedJumps));
    ctx.setResumed(true);
    ctx.log(LogLevel::Info, "resuming from checkpoint " + ctx.checkpointer().path().string() + ": " +
                                std::to_string(table.size()) + " distinguished points, " +
                                std::to_string(resumedJumps) + " jumps");
    if (ctx.spec().dpBits && *ctx.spec().dpBits != dpBits) {
      ctx.log(LogLevel::Info, "using dp " + std::to_string(dpBits) + " from the checkpoint");
    }
  } else {
    dpBits = ctx.spec().dpBits.value_or(autoDpBits(width));
    jumpDist = makeJumpDistances(width, nK, rng);
  }

  const long double overhead = static_cast<long double>(nK) * std::ldexp(1.0L, static_cast<int>(dpBits));
  if (overhead > std::sqrt(width.toLongDouble()) / 2) {
    ctx.log(LogLevel::Warn, "dp overhead high; consider lower --dp");
  }
  const double expected = kangarooExpectedWork(width, nK, dpBits);
  ctx.setWorkExpected(expected);
  char expectedBits[16];
  std::snprintf(expectedBits, sizeof expectedBits, "%.2f", std::log2(expected));
  ctx.log(LogLevel::Info, "kangaroo: " + std::to_string(nK) + " kangaroos on " + std::to_string(workers.size()) +
                              " workers, dp " + std::to_string(dpBits) + ", expected 2^" + expectedBits + " jumps");

  KangarooParams params = makeKangarooParams(width, wildBase, dpBits, jumpDist);
  params.totalKangaroos = nK;

  SnapshotScope snapshot(ctx.checkpointer(), [&] {
    constexpr size_t kEntryBytes = 16 + 32 + 1;
    ByteWriter w;
    // Entries inserted while visiting only grow the buffer; the count is patched in afterwards.
    w.reserve(4 + 64 * 32 + 8 + 8 + table.size() * kEntryBytes);
    w.u32(dpBits);
    for (const U256& d : jumpDist) w.u256(d);
    w.u64(static_cast<uint64_t>(ctx.workDone()));
    const size_t countPos = w.size();
    w.u64(0);
    uint64_t count = 0;
    table.forEach([&](const X128& x, const KangarooTable::Entry& e) {
      w.u64(x.lo);
      w.u64(x.hi);
      w.u256(e.dist);
      w.u8(e.wild ? 1 : 0);
      ++count;
    });
    w.patchU64(countPos, count);
    return std::move(w.data());
  });

  const bool verify = backend.id() != "cpu";
  auto done = [&] { return ctx.shouldStop() || ctx.anyFound(); };
  std::vector<uint64_t> seeds(workers.size());
  for (auto& s : seeds) s = rng.next();
  runThreads(ctx, workers.size(), [&](size_t i) {
    KangarooWorker& w = *workers[i];
    try {
      KangarooParams p = params;
      p.firstKangaroo = firstIndex[i];
      w.init(p, seeds[i]);
      Rng resetRng(seeds[i] ^ 0x5851f42d4c957f2dULL);
      std::vector<Dp> dps;
      std::vector<uint8_t> valid;
      uint64_t invalid = 0;
      std::optional<Clock::time_point> lastWarn;
      while (!done()) {
        ctx.waitIfPaused();
        if (done()) break;
        dps.clear();
        ctx.addWork(static_cast<double>(w.step(dps)));
        if (verify) verifyDps(ctx, dps, wildBase, valid);
        for (size_t k = 0; k < dps.size() && !ctx.anyFound(); ++k) {
          const Dp& d = dps[k];
          if (verify && !valid[k]) {
            ++invalid;
            continue;
          }
          auto existing = table.insert(X128::of(d.x), {d.dist, d.wild});
          if (!existing) continue;
          if (existing->wild == d.wild) {
            w.reset(d.index, resetRng.next());
          } else if (!solveCollision(ctx, d.wild ? existing->dist : d.dist, d.wild ? d.dist : existing->dist)) {
            ctx.log(LogLevel::Warn, "false collision");
          }
        }
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

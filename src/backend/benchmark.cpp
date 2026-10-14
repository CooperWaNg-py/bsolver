#include "backend/benchmark.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <stdexcept>

#include "backend/backend.h"
#include "catalog/catalog.h"
#include "core/gtable.h"
#include "core/random.h"
#include "engine/kangaroo_params.h"
#include "engine/run_context.h"
#include "engine/segment_source.h"

namespace bsolver {

namespace {
using Clock = std::chrono::steady_clock;

const U256 kPuzzle28Key = U256::fromU64(0xd916ce8);

// Runs fn(i) for every i in [0, n) on its own thread. On failure *stop (if non-null) is set so the throughput loops
// end early; every started thread is joined and the first exception rethrown.
void forEachConcurrently(size_t n, std::atomic<bool>* stop, const std::function<void(size_t)>& fn) {
  runThreads(n, fn, [stop] {
    if (stop) *stop = true;
  });
}

std::array<uint8_t, 20> randomHash(Rng& rng) {
  std::array<uint8_t, 20> h;
  for (auto& b : h) b = static_cast<uint8_t>(rng.next());
  return h;
}

void checkScan(Backend& b, const BenchOptions& opt) {
  const Puzzle* p28 = Catalog::builtin().find(28);
  if (!p28) throw std::runtime_error("puzzle 28 missing from the built-in catalog");
  ScanSetup setup{{p28->hash160}, AddressMode::Compressed};
  auto workers = b.createScanWorkers(setup, opt.devices, opt.threads);
  if (workers.empty()) throw std::runtime_error("no workers");
  forEachConcurrently(workers.size(), nullptr, [&](size_t i) {
    ScanWorker& w = *workers[i];
    std::vector<U256> candidates;
    uint64_t count = std::min<uint64_t>(w.segmentKeys(), uint64_t{1} << 22);
    w.process(kPuzzle28Key - 5000, count, candidates);
    if (std::find(candidates.begin(), candidates.end(), kPuzzle28Key) == candidates.end()) {
      throw std::runtime_error(w.label() + ": correctness check failed (puzzle 28 key not found)");
    }
  });
}

double scanThroughput(Backend& b, const BenchOptions& opt) {
  Rng rng;
  ScanSetup setup{{randomHash(rng)}, AddressMode::Compressed};
  auto workers = b.createScanWorkers(setup, opt.devices, opt.threads);
  const U256 base = U256::fromU64(1) << 70;
  SequentialSource source(base, base + (U256::fromU64(1) << 60), base);
  std::vector<double> rates(workers.size(), 0);
  std::atomic<bool> stop{false};
  forEachConcurrently(workers.size(), &stop, [&](size_t i) {
    ScanWorker& w = *workers[i];
    std::vector<U256> candidates;
    U256 s;
    uint64_t n = 0;
    source.next(w.segmentKeys(), &s, &n);
    w.process(s, n, candidates);  // warm-up (kernel compilation, tuning)
    double work = 0;
    auto t0 = Clock::now();
    double elapsed = 0;
    while (elapsed < opt.seconds && !stop) {
      source.next(w.segmentKeys(), &s, &n);
      w.process(s, n, candidates);
      work += static_cast<double>(n);
      elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
    }
    rates[i] = elapsed > 0 ? work / elapsed : 0;  // no measured interval (seconds <= 0 or stopped): no rate
  });
  double total = 0;
  for (double r : rates) total += r;
  return total;
}

size_t totalKangaroos(const std::vector<std::unique_ptr<KangarooWorker>>& workers) {
  size_t n = 0;
  for (const auto& w : workers) n += w->kangarooCount();
  return n;
}

// Per-worker copies of `params` with the herd layout (totalKangaroos/firstKangaroo) filled in.
std::vector<KangarooParams> layoutHerd(const KangarooParams& params,
                                       const std::vector<std::unique_ptr<KangarooWorker>>& workers) {
  std::vector<KangarooParams> out(workers.size(), params);
  uint64_t first = 0;
  for (size_t i = 0; i < workers.size(); ++i) {
    out[i].firstKangaroo = first;
    first += workers[i]->kangarooCount();
  }
  for (auto& p : out) p.totalKangaroos = first;
  return out;
}

void checkKangaroo(Backend& b, const BenchOptions& opt) {
  auto workers = b.createKangarooWorkers(opt.devices, opt.threads);
  if (workers.empty()) throw std::runtime_error("no workers");
  Rng rng;
  const U256 width = U256::fromU64(1) << 40;
  const U256 a = width;
  const Affine q = mulG(a + rng.uniform(U256{}, width - 1));
  const Affine wildBase = wildBaseFor(q, a);
  auto params =
      layoutHerd(makeKangarooParams(width, wildBase, 8, makeJumpDistances(width, totalKangaroos(workers), rng)), workers);
  forEachConcurrently(workers.size(), nullptr, [&](size_t i) {
    KangarooWorker& w = *workers[i];
    w.init(params[i], 0x5eed0000 + i);
    std::vector<Dp> dps;
    w.step(dps);
    for (size_t k = 0; k < std::min<size_t>(dps.size(), 16); ++k) {
      if (!verifyKangarooPoint(dps[k].x, dps[k].dist, dps[k].wild, wildBase)) {
        throw std::runtime_error(w.label() + ": correctness check failed (invalid distinguished point)");
      }
    }
    auto sample = w.sample(16);
    for (size_t k = 0; k < sample.size(); ++k) {
      bool wild = k % 2 == 1;
      Affine expect = wild ? add(wildBase, mulG(sample[k].second)) : mulG(sample[k].second);
      if (!(expect == sample[k].first)) {
        throw std::runtime_error(w.label() + ": correctness check failed (kangaroo state mismatch)");
      }
    }
  });
}

double kangarooThroughput(Backend& b, const BenchOptions& opt) {
  auto workers = b.createKangarooWorkers(opt.devices, opt.threads);
  Rng rng;
  const U256 width = U256::fromU64(1) << 100;
  const U256 a = width;
  const Affine q = mulG(a + rng.uniform(U256{}, width - 1));
  auto params = layoutHerd(makeKangarooParams(width, wildBaseFor(q, a), autoDpBits(width),
                                              makeJumpDistances(width, totalKangaroos(workers), rng)),
                           workers);
  const uint64_t seed = rng.next();
  std::vector<double> rates(workers.size(), 0);
  std::atomic<bool> stop{false};
  forEachConcurrently(workers.size(), &stop, [&](size_t i) {
    KangarooWorker& w = *workers[i];
    w.init(params[i], seed + i);
    std::vector<Dp> dps;
    w.step(dps);  // warm-up
    double jumps = 0;
    auto t0 = Clock::now();
    double elapsed = 0;
    while (elapsed < opt.seconds && !stop) {
      dps.clear();
      jumps += static_cast<double>(w.step(dps));
      elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
    }
    rates[i] = elapsed > 0 ? jumps / elapsed : 0;
  });
  double total = 0;
  for (double r : rates) total += r;
  return total;
}
}  // namespace

std::vector<BenchResult> runBenchmark(Algo algo, const BenchOptions& opt) {
  if (algo == Algo::Bsgs) throw std::invalid_argument("benchmark supports scan and kangaroo");
  std::vector<BenchResult> results;
  for (Backend* b : allBackends()) {
    if (!b->supports(algo)) continue;
    auto devices = b->devices();
    if (devices.empty()) continue;
    BenchResult r;
    r.backend = std::string(b->id());
    r.algo = algo;
    for (const auto& d : devices) {
      if (b->id() == "cpu" || opt.devices.empty() ||
          std::find(opt.devices.begin(), opt.devices.end(), d.index) != opt.devices.end()) {
        r.devices.push_back(d.name);
      }
    }
    if (b->id() == "cpu") r.devices = {"CPU (" + std::to_string(effectiveThreads(opt.threads)) + " threads)"};
    try {
      if (r.devices.empty()) throw std::runtime_error("no matching devices");
      if (algo == Algo::Scan) {
        checkScan(*b, opt);
        r.verified = true;
        r.rate = scanThroughput(*b, opt);
      } else {
        checkKangaroo(*b, opt);
        r.verified = true;
        r.rate = kangarooThroughput(*b, opt);
      }
    } catch (const std::exception& ex) {
      r.verified = false;
      r.rate = 0;
      r.error = ex.what();
    }
    results.push_back(std::move(r));
  }
  return results;
}

}  // namespace bsolver

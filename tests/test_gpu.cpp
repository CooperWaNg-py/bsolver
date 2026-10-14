#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <algorithm>
#include <filesystem>
#include <string>
#include <unordered_set>

#include "backend/backend.h"
#include "backend/gpu/gpu_common.h"
#include "catalog/catalog.h"
#include "core/address.h"
#include "core/gtable.h"
#include "core/random.h"
#include "core/ripemd160.h"
#include "core/scalar.h"
#include "engine/job_manager.h"
#include "engine/kangaroo_params.h"

using namespace bsolver;
namespace fs = std::filesystem;

namespace {
struct TempDir {
  fs::path path;
  TempDir() : path(fs::temp_directory_path() / ("bsolver-gpu-test-" + randomTokenHex(6))) {
    fs::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

const Catalog& catalog() {
  static Catalog c = Catalog::builtin();
  return c;
}

const U256 kPuzzle28Key = U256::fromU64(0xd916ce8);

// The backend `id`, or SKIP when it is not compiled in or has no devices.
Backend& gpuBackend(std::string_view id) {
  Backend* b = findBackend(id);
  if (!b) SKIP(std::string(id) + " backend not compiled");
  if (b->devices().empty()) SKIP(std::string(id) + " unavailable: " + b->unavailableReason());
  return *b;
}

// Scans [lo, hi] in segmentKeys() chunks on every worker in turn.
std::vector<U256> scanRange(Backend& b, const ScanSetup& setup, const U256& lo, const U256& hi) {
  std::vector<U256> candidates;
  for (auto& w : b.createScanWorkers(setup, {}, 0)) {
    U256 s = lo;
    while (s <= hi) {
      U256 left = hi - s + 1;
      uint64_t n = left < U256::fromU64(w->segmentKeys()) ? left.w[0] : w->segmentKeys();
      w->process(s, n, candidates);
      s = s + n;
    }
  }
  return candidates;
}

std::array<uint8_t, 20> uncompressedHash(const U256& key) {
  uint8_t buf[65];
  serializeUncompressed(mulG(key), buf);
  std::array<uint8_t, 20> h;
  hash160_65(buf, h.data());
  return h;
}

void checkScanFindsPuzzle28(std::string_view id) {
  Backend& b = gpuBackend(id);
  const U256 half = U256::fromU64(uint64_t{1} << 24);
  ScanSetup setup{{catalog().find(28)->hash160}, AddressMode::Compressed};
  auto candidates = scanRange(b, setup, kPuzzle28Key - half, kPuzzle28Key + half);
  REQUIRE(candidates.size() == b.devices().size());
  for (const U256& k : candidates) CHECK(k == kPuzzle28Key);
}

void checkScanRandomTarget(std::string_view id) {
  Backend& b = gpuBackend(id);
  Rng rng;
  std::array<uint8_t, 20> h;
  for (auto& v : h) v = static_cast<uint8_t>(rng.next());
  const U256 half = U256::fromU64(uint64_t{1} << 24);
  ScanSetup setup{{h}, AddressMode::Both};
  CHECK(scanRange(b, setup, kPuzzle28Key - half, kPuzzle28Key + half).empty());
}

void checkScanUncompressed(std::string_view id) {
  Backend& b = gpuBackend(id);
  // Keys at the edges of a segment and of the first group, and inside one.
  const U256 lo = U256::fromU64(uint64_t{1} << 30);
  const U256 hi = lo + (uint64_t{1} << 22);
  std::vector<U256> keys = {lo, lo + (gpu::kGroup - 1), lo + gpu::kGroup, lo + 777777, hi};
  ScanSetup setup;
  setup.mode = AddressMode::Uncompressed;
  for (const U256& k : keys) setup.targets.push_back(uncompressedHash(k));
  std::sort(setup.targets.begin(), setup.targets.end());
  auto candidates = scanRange(b, setup, lo, hi);
  std::sort(candidates.begin(), candidates.end());
  std::vector<U256> expected;
  for (size_t d = 0; d < b.devices().size(); ++d) expected.insert(expected.end(), keys.begin(), keys.end());
  std::sort(expected.begin(), expected.end());
  CHECK(candidates == expected);
}

// Point the state's signed distance stands for (tame dist*G, wild wildBase + dist*G).
Affine kangarooPoint(const KangarooState& s, bool wild, const Affine& wildBase) {
  Affine p = mulG(signedToScalar(s.dist));
  return wild ? add(wildBase, p) : p;
}

// Replays `steps` kangarooStep() calls from each `from` state and compares with `to` bit for bit. Walks that revisit
// an x (fruitless cycle) or hit P == +-D are skipped: the worker escapes or resets those on the host.
void checkReplay(const KangarooParams& params, const std::vector<KangarooState>& from, uint64_t steps,
                 const std::vector<KangarooState>& to) {
  REQUIRE(from.size() == to.size());
  size_t compared = 0;
  for (size_t k = 0; k < from.size(); ++k) {
    KangarooState s = from[k];
    std::unordered_set<uint64_t> seen{s.point.x.n[0]};
    bool diverged = false;
    for (uint64_t i = 0; i < steps && !diverged; ++i) {
      diverged = !kangarooStep(params, s) || !seen.insert(s.point.x.n[0]).second;
    }
    if (diverged) continue;
    ++compared;
    CHECK(s.point == to[k].point);
    CHECK(s.dist == to[k].dist);
    CHECK(s.lookahead == to[k].lookahead);
  }
  CHECK(compared + 2 >= from.size());
}

// Device steps equal the reference kangarooStep() (both jump tables), DPs verify, and resume/reset states
// round-trip through the device buffers.
void checkKangarooWorker(std::string_view id) {
  Backend& b = gpuBackend(id);
  auto workers = b.createKangarooWorkers({}, 0);
  REQUIRE_FALSE(workers.empty());
  KangarooWorker& w = *workers[0];
  Rng rng;
  const U256 width = U256::fromU64(1) << 40;
  const Affine q = mulG(width + rng.uniform(U256{}, width - 1));
  const Affine wildBase = wildBaseFor(q, width, kangarooShift(width));
  KangarooParams params = makeKangarooParams(width, wildBase, 8, makeJumpDistances(width, rng));
  params.totalKangaroos = w.kangarooCount();
  params.firstKangaroo = 0;
  constexpr size_t kSample = 16;
  auto verifyStates = [&](const std::vector<KangarooState>& states) {
    REQUIRE(states.size() == kSample);
    for (size_t k = 0; k < kSample; ++k) {
      CHECK(kangarooPoint(states[k], isWildKangaroo(k), wildBase) == states[k].point);
    }
  };

  w.init(params, 1234, {});
  const std::vector<KangarooState> before = w.sample(kSample);
  verifyStates(before);
  std::vector<Dp> dps;
  uint64_t jumps = w.step(dps);
  REQUIRE(jumps >= w.kangarooCount());
  REQUIRE(jumps % w.kangarooCount() == 0);
  const std::vector<KangarooState> after = w.sample(kSample);
  verifyStates(after);
  checkReplay(params, before, jumps / w.kangarooCount(), after);
  REQUIRE_FALSE(dps.empty());
  for (size_t k = 0; k < std::min<size_t>(dps.size(), 64); ++k) {
    CHECK(isDistinguished(dps[k].x.w[3], 8));
    CHECK(dps[k].wild == isWildKangaroo(dps[k].index));
    CHECK(verifyKangarooPoint(dps[k].x, dps[k].dist, dps[k].wild, wildBase));
  }

  // Resumed states (one with the look-ahead flag set) come back unchanged and keep walking like the reference.
  std::vector<KangarooState> resume = after;
  resume[3].lookahead = true;
  w.init(params, 777, resume);
  const std::vector<KangarooState> resumed = w.sample(kSample);
  for (size_t k = 0; k < kSample; ++k) {
    CHECK(resumed[k].point == resume[k].point);
    CHECK(resumed[k].dist == resume[k].dist);
    CHECK(resumed[k].lookahead == resume[k].lookahead);
  }
  dps.clear();
  jumps = w.step(dps);
  checkReplay(params, resume, jumps / w.kangarooCount(), w.sample(kSample));

  w.reset(0, 99);
  w.reset(1, 100);
  const std::vector<KangarooState> reset = w.sample(kSample);
  verifyStates(reset);
  CHECK_FALSE(reset[0].lookahead);
  CHECK_FALSE(reset[1].lookahead);
}

void checkKangarooJob(std::string_view id) {
  gpuBackend(id);
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  JobSpec s;
  s.puzzle = 48;
  s.algorithm = Algorithm::Kangaroo;
  s.backend = std::string(id);
  s.checkpoint = false;
  m.start(s);
  m.wait();
  JobStatus st = m.status();
  REQUIRE(st.outcome == Outcome::Found);
  REQUIRE(st.found.size() == 1);
  CHECK(st.found[0].privkey == std::string(52, '0') + "ade6d7ce3b9b");
  CHECK(st.backend == std::string(id));
}

// Keys within GPU_GROUP of n hit the kernel's dx == 0 cases; the runner scans keys >= n - 2^20 on the CPU instead.
// Sequential: the range crosses that limit mid-segment and its first key is a GPU key. Random: the range is narrower
// than any GPU segment (>= 512 threads * GPU_GROUP keys), so every draw covers all of it.
void checkScanJobNearOrder(std::string_view id, Order order) {
  gpuBackend(id);
  const bool random = order == Order::Random;
  const U256 first = kOrderN - (random ? uint64_t{1} << 14 : (uint64_t{1} << 21) - 5);
  const std::vector<U256> keys = {first, kOrderN - 64, kOrderN - 2};
  JobSpec s;
  for (const U256& k : keys) {
    uint8_t buf[33], h[20];
    serializeCompressed(mulG(k), buf);
    hash160_33(buf, h);
    s.addresses.push_back(encodeP2PKH(h));
  }
  s.rangeStart = (kOrderN - (random ? uint64_t{1} << 15 : uint64_t{1} << 24)).toHex();
  s.rangeEnd = (kOrderN - 1).toHex();
  s.algorithm = Algorithm::Scan;
  s.order = order;
  s.backend = std::string(id);
  s.checkpoint = false;
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  m.start(s);
  m.wait();
  JobStatus st = m.status();
  REQUIRE(st.outcome == Outcome::Found);
  std::vector<std::string> found;
  for (const auto& f : st.found) found.push_back(f.privkey);
  std::sort(found.begin(), found.end());
  std::vector<std::string> expected;
  for (const U256& k : keys) expected.push_back(k.toHex(true));
  CHECK(found == expected);
}

// Device stand-in for the shared GPU scan worker: reports the hit for one planted key (keys follow the scan_thread
// layout) and drops the launches listed in `drops` (0-based launch numbers) half-way, as a watchdog abort would.
class FakeScanKernel final : public gpu::ScanKernel {
 public:
  FakeScanKernel(U256 start, U256 key, std::vector<int> drops) : start_(start), key_(key), drops_(std::move(drops)) {}
  uint32_t threads() const override { return 256; }
  uint32_t groupSize() const override { return 32; }
  void setTables(const std::vector<uint64_t>&, const std::vector<uint32_t>&) override {}
  void setOffsets(const std::vector<uint64_t>&) override {}
  gpu::LaunchResult launch(const std::array<uint64_t, gpu::kScanParamWords>& params, uint32_t count,
                           std::vector<gpu::ScanHit>& hits) override {
    launched.push_back(count);
    hits.clear();
    gpu::LaunchResult out;
    if (std::find(drops_.begin(), drops_.end(), static_cast<int>(launched.size() - 1)) != drops_.end()) {
      out.finished = count / 2;
      return out;
    }
    const uint64_t perThread = (params[8] & 0xffffffffu) * gpu::kGroup;
    const uint64_t rel = (key_ - start_).w[0];
    const uint64_t thread = rel / perThread, within = rel % perThread;
    if (thread >= params[10] && thread < params[10] + count) {
      hits.push_back(gpu::ScanHit{static_cast<uint32_t>(thread), static_cast<uint32_t>(within / gpu::kGroup),
                                  static_cast<int32_t>(within % gpu::kGroup) - static_cast<int32_t>(gpu::kGroup / 2),
                                  0});
    }
    out.count = static_cast<uint32_t>(hits.size());
    out.finished = count;
    return out;
  }
  std::string label() const override { return "fake"; }

  std::vector<uint32_t> launched;  // threads per launch, in order

 private:
  U256 start_, key_;
  std::vector<int> drops_;
};
}  // namespace

TEST_CASE("gpu scan worker repeats dropped launches with fewer threads", "[gpu]") {
  const U256 start = U256::fromU64(uint64_t{1} << 40);
  const U256 key = start + 77 * gpu::kGroup + 5;  // thread 77 at iters = 1
  auto kernel = std::make_unique<FakeScanKernel>(start, key, std::vector<int>{0, 2});
  FakeScanKernel& fake = *kernel;
  auto worker = gpu::makeScanWorker(std::move(kernel), ScanSetup{});
  std::vector<U256> candidates;
  worker->process(start, worker->segmentKeys(), candidates);
  REQUIRE(candidates.size() == 1);
  CHECK(candidates[0] == key);
  // The dropped 256-thread launch is redone as two 128-thread launches (the second dropped once and redone); fast
  // launches then grow back to all 256 threads.
  REQUIRE(fake.launched.size() >= 5);
  CHECK(std::vector<uint32_t>(fake.launched.begin(), fake.launched.begin() + 5) ==
        std::vector<uint32_t>{256, 128, 128, 128, 256});
}

TEST_CASE("gpu scan worker fails after repeated drops at the smallest launch", "[gpu]") {
  const U256 start = U256::fromU64(uint64_t{1} << 40);
  std::vector<int> drops(16);
  for (int i = 0; i < 16; ++i) drops[i] = i;
  auto worker = gpu::makeScanWorker(std::make_unique<FakeScanKernel>(start, start, drops), ScanSetup{});
  std::vector<U256> candidates;
  CHECK_THROWS_WITH(worker->process(start, worker->segmentKeys(), candidates),
                    "fake: scan launch dropped 3 times: 16 of 32 threads ran");
}

#define BSOLVER_GPU_TESTS(ID)                                                                   \
  TEST_CASE(ID " scan worker finds the puzzle 28 key", "[gpu]") { checkScanFindsPuzzle28(ID); } \
  TEST_CASE(ID " scan worker: no candidate for a random target", "[gpu]") { checkScanRandomTarget(ID); } \
  TEST_CASE(ID " scan worker: uncompressed keys", "[gpu]") { checkScanUncompressed(ID); }   \
  TEST_CASE(ID " kangaroo worker state and DPs verify", "[gpu]") { checkKangarooWorker(ID); } \
  TEST_CASE(ID " kangaroo job finds puzzle 48", "[gpu]") { checkKangarooJob(ID); }            \
  TEST_CASE(ID " sequential scan job finds keys near n", "[gpu]") {                           \
    checkScanJobNearOrder(ID, Order::Sequential);                                             \
  }                                                                                           \
  TEST_CASE(ID " random scan job finds keys near n", "[gpu]") { checkScanJobNearOrder(ID, Order::Random); }

BSOLVER_GPU_TESTS("opencl")
BSOLVER_GPU_TESTS("metal")
BSOLVER_GPU_TESTS("cuda")

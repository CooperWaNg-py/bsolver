#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <string>

#include "backend/backend.h"
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
  // Keys at the edges of a segment and inside one: offsets -32 / +31 around group centers.
  const U256 lo = U256::fromU64(uint64_t{1} << 30);
  const U256 hi = lo + (uint64_t{1} << 22);
  std::vector<U256> keys = {lo, lo + 63, lo + 64, lo + 777777, hi};
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

// Same checks as the benchmark's kangaroo correctness check, plus reset().
void checkKangarooWorker(std::string_view id) {
  Backend& b = gpuBackend(id);
  auto workers = b.createKangarooWorkers({}, 0);
  REQUIRE_FALSE(workers.empty());
  Rng rng;
  const U256 width = U256::fromU64(1) << 40;
  const Affine q = mulG(width + rng.uniform(U256{}, width - 1));
  const Affine wildBase = wildBaseFor(q, width);
  KangarooWorker& w = *workers[0];
  KangarooParams params = makeKangarooParams(width, wildBase, 8, makeJumpDistances(width, w.kangarooCount(), rng));
  params.totalKangaroos = w.kangarooCount();
  params.firstKangaroo = 0;
  w.init(params, 1234);

  auto verifySample = [&] {
    auto sample = w.sample(16);
    REQUIRE(sample.size() == 16);
    for (size_t k = 0; k < sample.size(); ++k) {
      Affine expect = k % 2 ? add(wildBase, mulG(sample[k].second)) : mulG(sample[k].second);
      CHECK(expect == sample[k].first);
    }
  };
  verifySample();

  std::vector<Dp> dps;
  uint64_t jumps = w.step(dps);
  CHECK(jumps >= w.kangarooCount());
  REQUIRE_FALSE(dps.empty());
  for (size_t k = 0; k < std::min<size_t>(dps.size(), 64); ++k) {
    CHECK(isDistinguished(dps[k].x.w[3], 8));
    CHECK(dps[k].wild == (dps[k].index % 2 == 1));
    CHECK(verifyKangarooPoint(dps[k].x, dps[k].dist, dps[k].wild, wildBase));
  }
  verifySample();

  w.reset(0, 99);
  w.reset(1, 100);
  verifySample();
  dps.clear();
  w.step(dps);
  verifySample();
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
// than any GPU segment (>= 512 threads * 64 keys), so every draw covers all of it.
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
}  // namespace

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

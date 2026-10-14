#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

#include "catalog/catalog.h"
#include "core/address.h"
#include "core/base58.h"
#include "core/gtable.h"
#include "core/random.h"
#include "core/ripemd160.h"
#include "engine/bytes.h"
#include "engine/checkpoint.h"
#include "engine/job_manager.h"
#include "engine/run_context.h"

using namespace bsolver;
namespace fs = std::filesystem;

namespace {
struct TempDir {
  fs::path path;
  TempDir() : path(fs::temp_directory_path() / ("bsolver-scan-test-" + randomTokenHex(6))) {
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

JobStatus runToEnd(JobManager& m, const JobSpec& spec) {
  m.start(spec);
  m.wait();
  return m.status();
}

JobSpec cpuScan() {
  JobSpec s;
  s.algorithm = Algorithm::Scan;
  s.backend = "cpu";
  s.checkpoint = false;
  return s;
}
}  // namespace

TEST_CASE("sequential scan finds puzzle 24") {
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  JobSpec s = cpuScan();
  s.puzzle = 24;
  JobStatus st = runToEnd(m, s);
  REQUIRE(st.outcome == Outcome::Found);
  REQUIRE(st.found.size() == 1);
  CHECK(st.found[0].privkey == std::string(58, '0') + "dc2a04");
  auto lines = readResults(dir.path / "found.jsonl");
  REQUIRE(lines.size() == 1);
  CHECK(lines[0].puzzle == 24);
}

TEST_CASE("random scan finds puzzle 20") {
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  JobSpec s = cpuScan();
  s.puzzle = 20;
  s.order = Order::Random;
  auto t0 = std::chrono::steady_clock::now();
  JobStatus st = runToEnd(m, s);
  CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30));
  REQUIRE(st.outcome == Outcome::Found);
  CHECK(st.found[0].privkey.ends_with("d2c55"));
  CHECK_FALSE(st.progress.has_value());
}

TEST_CASE("uncompressed address mode") {
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  JobSpec s = cpuScan();
  s.addresses = {encodeP2PKH(hash160Of(mulG(U256::fromU64(4321)), false))};
  s.rangeStart = "1";
  s.rangeEnd = "1388";  // 5000
  s.addressMode = AddressMode::Uncompressed;
  JobStatus st = runToEnd(m, s);
  REQUIRE(st.outcome == Outcome::Found);
  CHECK(st.found[0].privkey.ends_with("10e1"));  // 4321
  // The WIF must import as the uncompressed key: 0x80 || key, no 0x01 suffix.
  auto wif = base58Decode(st.found[0].wif);
  REQUIRE(wif);
  REQUIRE(wif->size() == 1 + 32 + 4);
  CHECK((*wif)[0] == 0x80);
  uint8_t key[32];
  U256::fromU64(4321).toBytesBE(key);
  CHECK(std::equal(key, key + 32, wif->begin() + 1));
}

TEST_CASE("pubkey-only scan hashes the pubkey per address mode") {
  const Affine pub = mulG(U256::fromU64(4321));
  for (AddressMode mode : {AddressMode::Compressed, AddressMode::Uncompressed, AddressMode::Both}) {
    for (bool compressedInput : {true, false}) {
      TempDir dir;
      JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
      JobSpec s = cpuScan();
      s.pubkey = pubkeyHex(pub, compressedInput);
      s.rangeStart = "1";
      s.rangeEnd = "1388";  // 5000
      s.addressMode = mode;
      INFO("mode " << toString(mode) << ", compressed input " << compressedInput);
      JobStatus st = runToEnd(m, s);
      REQUIRE(st.outcome == Outcome::Found);
      CHECK(st.found[0].privkey.ends_with("10e1"));  // 4321
    }
  }
}

TEST_CASE("both encodings of one key as targets stop the scan once found") {
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  const U256 key = U256::fromU64((uint64_t{1} << 21) + 12345);
  const Affine pub = mulG(key);
  JobSpec s = cpuScan();
  s.addresses = {encodeP2PKH(hash160Of(pub, true)), encodeP2PKH(hash160Of(pub, false))};
  s.rangeStart = U256::fromU64(uint64_t{1} << 21).toHex();
  s.rangeEnd = U256::fromU64(uint64_t{1} << 28).toHex();
  s.addressMode = AddressMode::Both;
  JobStatus st = runToEnd(m, s);
  REQUIRE(st.outcome == Outcome::Found);
  REQUIRE(st.found.size() == 1);
  CHECK(st.found[0].privkey == key.toHex(true));
  // stopOnFound ended the job long before the end of the range.
  CHECK(st.workDone < static_cast<double>(uint64_t{1} << 27));
}

TEST_CASE("corrupt checkpoint payload fails with a hint") {
  TempDir dir;
  JobSpec s = cpuScan();
  s.checkpoint = true;
  s.puzzle = 24;
  const ManagerOptions opts{dir.path / "state", dir.path / "found.jsonl"};
  const ResolvedJob resolved = resolve(s, catalog());
  Checkpointer c(opts.stateDir, Algo::Scan, resolved.identity, true, false);
  c.setSnapshot([] { return std::vector<uint8_t>{1, 2, 3}; });  // header ok, payload truncated
  c.saveNow();

  JobManager m(catalog(), opts);
  JobStatus st = runToEnd(m, s);
  CHECK(st.outcome == Outcome::Error);
  CHECK(st.error == "checkpoint " + c.path().string() + " is corrupt (use --fresh)");
}

TEST_CASE("runThreads stops the others and rethrows the first failure") {
  JobSpec spec;
  ResolvedJob job;
  Checkpointer ckpt(fs::temp_directory_path(), Algo::Scan, "runThreads test", false, false);
  RunContext ctx(job, spec, ckpt, [](LogLevel, const std::string&) {}, [](const FoundKey&) {});
  std::atomic<int> finished{0};
  CHECK_THROWS_WITH(runThreads(ctx, 4,
                               [&](size_t i) {
                                 if (i == 2) throw std::runtime_error("boom");
                                 while (!ctx.stopRequested()) std::this_thread::yield();
                                 ++finished;
                               }),
                    "boom");
  CHECK(ctx.stopRequested());
  CHECK(finished == 3);
}

TEST_CASE("context-free runThreads calls onFailure and rethrows the first failure") {
  std::atomic<bool> stop{false};
  std::atomic<int> finished{0};
  CHECK_THROWS_WITH(runThreads(
                        4,
                        [&](size_t i) {
                          if (i == 1) throw std::runtime_error("bang");
                          while (!stop) std::this_thread::yield();
                          ++finished;
                        },
                        [&] { stop = true; }),
                    "bang");
  CHECK(finished == 3);
}

TEST_CASE("scan exhausts a range without the key") {
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  JobSpec s = cpuScan();
  s.puzzle = 24;
  s.rangeStart = "800000";
  s.rangeEnd = "8fffff";
  JobStatus st = runToEnd(m, s);
  CHECK(st.outcome == Outcome::Exhausted);
}

TEST_CASE("sequential scan checkpoint and resume") {
  TempDir dir;
  Rng rng;
  Hash160 h;
  for (auto& b : h) b = static_cast<uint8_t>(rng.next());
  const U256 start = U256::fromU64(uint64_t{1} << 29);
  const U256 end = start + (U256::fromU64(1) << 26);
  JobSpec s;
  s.algorithm = Algorithm::Scan;
  s.backend = "cpu";
  s.addresses = {encodeP2PKH(h)};
  s.rangeStart = start.toHex();
  s.rangeEnd = end.toHex();
  const ManagerOptions opts{dir.path / "state", dir.path / "found.jsonl"};
  const ResolvedJob resolved = resolve(s, catalog());
  const fs::path ckptPath = Checkpointer(opts.stateDir, Algo::Scan, resolved.identity, true, false).path();

  {
    JobManager m(catalog(), opts);
    m.start(s);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    m.stop();
    m.wait();
    CHECK(m.status().outcome == Outcome::Stopped);
  }
  auto payload = Checkpointer::loadFile(ckptPath, Algo::Scan, resolved.identity);
  REQUIRE(payload);
  ByteReader r(*payload);
  U256 cursor = r.u256();
  uint64_t keysDone = r.u64();
  CHECK(cursor > start);
  CHECK(cursor <= end);
  CHECK(keysDone > 0);

  {
    JobManager m(catalog(), opts);
    m.start(s);
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    JobStatus st = m.status();
    CHECK(st.resumed);
    CHECK(st.workDone >= static_cast<double>(keysDone));
    m.stop();
    m.wait();
  }
  CHECK_THROWS_WITH(Checkpointer::loadFile(ckptPath, Algo::Scan, "other identity"),
                    "checkpoint " + ckptPath.string() + " does not match this job");
}

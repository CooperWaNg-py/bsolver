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

// Polls pred every 10 ms until it holds or the (generous) deadline passes.
template <class Pred>
bool pollUntil(Pred pred, std::chrono::seconds deadline) {
  const auto end = std::chrono::steady_clock::now() + deadline;
  while (!pred()) {
    if (std::chrono::steady_clock::now() > end) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return true;
}

Hash160 randomHash160() {
  Rng rng;
  Hash160 h;
  for (auto& b : h) b = static_cast<uint8_t>(rng.next());
  return h;
}

constexpr uint64_t kSmallLimit = uint64_t{1} << 20;  // scan_runner's small-key prefix bound
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

TEST_CASE("random scan finds puzzle 24") {
  // [2^23, 2^24) lies above the small-key prefix, so only RandomSource segments can find dc2a04; each 2^20 segment
  // covers it with probability ~1/7.
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  JobSpec s = cpuScan();
  s.puzzle = 24;
  s.order = Order::Random;
  s.threads = 2;
  JobStatus st = runToEnd(m, s);
  REQUIRE(st.outcome == Outcome::Found);
  CHECK(st.found[0].privkey == std::string(58, '0') + "dc2a04");
  CHECK(st.workDone >= static_cast<double>(kSmallLimit));  // at least one full random segment
  CHECK_FALSE(st.progress.has_value());
  CHECK_FALSE(st.etaSeconds.has_value());
}

TEST_CASE("random scan checkpoint and resume") {
  TempDir dir;
  const U256 start = U256::fromU64(uint64_t{1} << 19);  // half the range below 2^20 goes through the prefix
  const U256 end = U256::fromU64(uint64_t{1} << 40);
  JobSpec s = cpuScan();
  s.checkpoint = true;
  s.order = Order::Random;
  s.threads = 2;
  s.addresses = {encodeP2PKH(randomHash160())};
  s.rangeStart = start.toHex();
  s.rangeEnd = end.toHex();
  const ManagerOptions opts{dir.path / "state", dir.path / "found.jsonl"};
  const ResolvedJob resolved = resolve(s, catalog());
  const fs::path ckptPath = Checkpointer(opts.stateDir, Algo::Scan, resolved.identity, true, false).path();

  // The prefix (2^19 keys) is done before any worker starts, so this much work includes random segments.
  const double threshold = static_cast<double>((uint64_t{1} << 19) + 2 * kSmallLimit);
  {
    JobManager m(catalog(), opts);
    m.start(s);
    REQUIRE(pollUntil([&] { return m.status().workDone >= threshold; }, std::chrono::seconds(300)));
    m.stop();
    m.wait();
    CHECK(m.status().outcome == Outcome::Stopped);
  }
  auto payload = Checkpointer::loadFile(ckptPath, Algo::Scan, resolved.identity);
  REQUIRE(payload);
  ByteReader r(*payload);
  const uint64_t keysDone = r.u64();
  const U256 prefixCursor = r.u256();
  CHECK(r.remaining() == 0);
  CHECK(static_cast<double>(keysDone) >= threshold);
  CHECK_FALSE(prefixCursor < start);
  CHECK(prefixCursor == U256::fromU64(kSmallLimit));  // the prefix finished before the random segments

  {
    JobManager m(catalog(), opts);
    m.start(s);
    REQUIRE(pollUntil([&] { return m.status().resumed; }, std::chrono::seconds(60)));
    CHECK(m.status().workDone >= static_cast<double>(keysDone));
    m.stop();
    m.wait();
    CHECK(m.status().outcome == Outcome::Stopped);
  }
  payload = Checkpointer::loadFile(ckptPath, Algo::Scan, resolved.identity);
  REQUIRE(payload);
  ByteReader r2(*payload);
  CHECK(r2.u64() >= keysDone);
  CHECK(r2.u256() == U256::fromU64(kSmallLimit));
}

TEST_CASE("random scan resumes a partial small-key prefix at its cursor") {
  // A checkpoint taken mid-prefix: the resumed run must continue the prefix at prefixCursor, so it finds a key just
  // below 2^20 after exactly one more 2^16-key prefix chunk.
  TempDir dir;
  const U256 key = U256::fromU64(kSmallLimit - 100);
  JobSpec s = cpuScan();
  s.checkpoint = true;
  s.order = Order::Random;
  s.threads = 2;
  s.addresses = {encodeP2PKH(hash160Of(mulG(key), true))};
  s.rangeStart = "1";
  s.rangeEnd = U256::fromU64(uint64_t{1} << 40).toHex();
  const ManagerOptions opts{dir.path / "state", dir.path / "found.jsonl"};
  const ResolvedJob resolved = resolve(s, catalog());

  const uint64_t prefixCursor = kSmallLimit - (uint64_t{1} << 16);
  const uint64_t keysDone = prefixCursor - 1;  // [1, prefixCursor) scanned
  {
    Checkpointer c(opts.stateDir, Algo::Scan, resolved.identity, true, false);
    c.setSnapshot([&] {
      ByteWriter w;
      w.u64(keysDone);
      w.u256(U256::fromU64(prefixCursor));
      return std::move(w.data());
    });
    c.saveNow();
    c.clearSnapshot();
  }

  JobManager m(catalog(), opts);
  JobStatus st = runToEnd(m, s);
  REQUIRE(st.outcome == Outcome::Found);
  CHECK(st.resumed);
  CHECK(st.found[0].privkey == key.toHex(true));
  // Resumed work plus the single chunk [prefixCursor, 2^20); a restart from rangeStart would add ~2^20.
  CHECK(st.workDone == static_cast<double>(keysDone + (uint64_t{1} << 16)));
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
  const U256 start = U256::fromU64(uint64_t{1} << 29);
  // 2^34 keys: far more than any machine scans during the sleep below, so the job is stopped, not exhausted.
  const U256 end = start + (U256::fromU64(1) << 34);
  JobSpec s;
  s.algorithm = Algorithm::Scan;
  s.backend = "cpu";
  s.addresses = {encodeP2PKH(randomHash160())};
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
    REQUIRE(pollUntil([&] { return m.status().resumed; }, std::chrono::seconds(60)));
    JobStatus st = m.status();
    CHECK(st.resumed);
    CHECK(st.workDone >= static_cast<double>(keysDone));
    m.stop();
    m.wait();
  }
  CHECK_THROWS_WITH(Checkpointer::loadFile(ckptPath, Algo::Scan, "other identity"),
                    "checkpoint " + ckptPath.string() + " does not match this job");
}

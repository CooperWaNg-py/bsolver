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
#include "engine/segment_source.h"

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

using Interval = std::pair<U256, uint64_t>;  // [first, first + len)

// The intervals tile [first, first + width) exactly: no gap, no overlap.
bool tiles(std::vector<Interval> chunks, const U256& first, const U256& width) {
  std::sort(chunks.begin(), chunks.end());
  U256 next = first;
  for (const auto& [s, n] : chunks) {
    if (s != next || n == 0) return false;
    next = next + n;
  }
  return next == first + width;
}

PermutedSource::State fixedState(unsigned blockBits) {
  PermutedSource::State s;
  for (size_t i = 0; i < s.key.size(); ++i) s.key[i] = static_cast<uint8_t>(7 * i + 1);
  s.blockBits = blockBits;
  return s;
}

// Random-order checkpoint payload (see scan_runner.cpp).
struct RandomPayload {
  U256 prefixCursor;
  PermutedSource::State state;
};

RandomPayload parseRandomPayload(const std::vector<uint8_t>& payload) {
  ByteReader r(payload);
  RandomPayload p;
  p.prefixCursor = r.u256();
  p.state.blockBits = r.u8();
  r.bytes(p.state.key.data(), p.state.key.size());
  p.state.cursor = r.u256();
  p.state.keysDone = r.u256();
  for (uint32_t n = r.u32(); n != 0; --n) {
    const U256 pos = r.u256();
    p.state.pending.push_back({pos, r.u64()});
  }
  REQUIRE(r.remaining() == 0);
  return p;
}

std::vector<uint8_t> randomPayload(const RandomPayload& p) {
  ByteWriter w;
  w.u256(p.prefixCursor);
  w.u8(static_cast<uint8_t>(p.state.blockBits));
  w.bytes(p.state.key.data(), p.state.key.size());
  w.u256(p.state.cursor);
  w.u256(p.state.keysDone);
  w.u32(static_cast<uint32_t>(p.state.pending.size()));
  for (const auto& c : p.state.pending) {
    w.u256(c.pos);
    w.u64(c.len);
  }
  return std::move(w.data());
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

TEST_CASE("block permutation is a keyed bijection") {
  const PermutedSource::State st = fixedState(20);
  std::vector<uint64_t> counts;
  for (uint64_t c = 1; c <= 300; ++c) counts.push_back(c);
  for (uint64_t c : {1000, 4096, 4097, 65537}) counts.push_back(c);
  for (uint64_t count : counts) {
    BlockPermutation p(U256::fromU64(count), st.key);
    std::vector<bool> seen(count);
    bool bijective = true;
    for (uint64_t i = 0; i < count; ++i) {
      const U256 v = p(U256::fromU64(i));
      if (!(v < U256::fromU64(count)) || seen[v.w[0]]) {
        bijective = false;
        break;
      }
      seen[v.w[0]] = true;
    }
    INFO("count " << count);
    CHECK(bijective);
  }

  // Keyed: another key gives another order, and neither is close to the identity.
  BlockPermutation::Key other = st.key;
  other[0] ^= 1;
  BlockPermutation a(U256::fromU64(1000), st.key), b(U256::fromU64(1000), other);
  int fixedA = 0, same = 0;
  for (uint64_t i = 0; i < 1000; ++i) {
    const U256 x = U256::fromU64(i);
    fixedA += a(x) == x;
    same += a(x) == b(x);
  }
  CHECK(fixedA < 20);
  CHECK(same < 20);
}

TEST_CASE("permuted source hands out every key once, block after block in permuted order") {
  const U256 first = U256::fromU64(3 * kSmallLimit + 17);  // blocks are aligned to the range start
  const uint64_t width = 37 * kSmallLimit + 12345;           // 38 blocks of 2^20, the last one clipped to 12345
  PermutedSource src(first, first + (width - 1), fixedState(20));
  CHECK(src.blocks() == U256::fromU64(38));

  // Workers with different segment sizes share the current block.
  const uint64_t sizes[] = {kSmallLimit, 300000, (kSmallLimit >> 1) + 7};
  std::vector<Interval> chunks;
  std::vector<uint64_t> blockOrder;
  U256 s;
  uint64_t n = 0;
  for (size_t i = 0; src.next(sizes[i % 3], &s, &n); ++i) {
    REQUIRE(n <= sizes[i % 3]);
    const uint64_t block = (s - first).w[0] >> 20;
    REQUIRE((s - first + (n - 1)).w[0] >> 20 == block);  // a chunk never crosses a block boundary
    if (blockOrder.empty() || blockOrder.back() != block) blockOrder.push_back(block);
    chunks.emplace_back(s, n);
    src.complete(s);
  }
  CHECK(tiles(chunks, first, U256::fromU64(width)));
  // Each block is visited in one go, once; the order is not the sequential one.
  REQUIRE(blockOrder.size() == 38);
  std::vector<uint64_t> sorted = blockOrder;
  std::sort(sorted.begin(), sorted.end());
  for (uint64_t i = 0; i < 38; ++i) CHECK(sorted[i] == i);
  CHECK(blockOrder != sorted);
  CHECK(src.finished());
  CHECK(src.snapshot().keysDone == U256::fromU64(width));
}

TEST_CASE("permuted source resumes in-flight chunks and continues at the cursor") {
  const U256 first = U256::fromU64(kSmallLimit);
  const uint64_t width = 9 * kSmallLimit + 999;
  const U256 last = first + (width - 1);
  std::vector<Interval> done;
  PermutedSource::State saved;
  {
    PermutedSource src(first, last, fixedState(20));
    std::vector<Interval> inFlight;
    U256 s;
    uint64_t n = 0;
    for (int i = 0; i < 25 && src.next(kSmallLimit >> 1, &s, &n); ++i) {
      if (i == 3 || i == 10 || i == 24) {
        inFlight.emplace_back(s, n);  // never completed: lost when the job stops
      } else {
        done.emplace_back(s, n);
        src.complete(s);
      }
    }
    saved = src.snapshot();
    uint64_t doneKeys = 0;
    for (const auto& c : done) doneKeys += c.second;
    CHECK(saved.keysDone == U256::fromU64(doneKeys));
    CHECK(saved.pending.size() == inFlight.size());
    CHECK_FALSE(src.finished());
  }

  // Smaller segments after the resume: pending chunks are split, not rescanned whole.
  PermutedSource resumed(first, last, saved);
  std::vector<Interval> all = done;
  U256 s;
  uint64_t n = 0;
  while (resumed.next(100000, &s, &n)) {
    all.emplace_back(s, n);
    resumed.complete(s);
  }
  CHECK(tiles(all, first, U256::fromU64(width)));
  CHECK(resumed.finished());
  CHECK(resumed.snapshot().keysDone == U256::fromU64(width));

  // A state that does not fit the range is rejected.
  PermutedSource::State bad = saved;
  bad.cursor = U256::fromU64(11) << 20;  // past the 10 blocks
  CHECK_THROWS(PermutedSource(first, last, bad));
  bad = saved;
  bad.blockBits = 19;
  CHECK_THROWS(PermutedSource(first, last, bad));
}

TEST_CASE("random scan finds puzzle 24") {
  // [2^23, 2^24) lies above the small-key prefix: 8 permuted blocks of 2^20 keys, scanned once each.
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  JobSpec s = cpuScan();
  s.puzzle = 24;
  s.order = Order::Random;
  s.threads = 2;
  JobStatus st = runToEnd(m, s);
  REQUIRE(st.outcome == Outcome::Found);
  CHECK(st.found[0].privkey == std::string(58, '0') + "dc2a04");
  CHECK(st.workDone <= static_cast<double>(uint64_t{1} << 23));
  REQUIRE(st.progress.has_value());
  CHECK(*st.progress == st.workDone / static_cast<double>(uint64_t{1} << 23));
}

TEST_CASE("random scan covers both ends of the range") {
  // 9 blocks, the last one clipped to 5 keys: the first and the last key are found within one pass.
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  const U256 first = U256::fromU64(4 * kSmallLimit);
  const uint64_t width = 8 * kSmallLimit + 5;
  const U256 last = first + (width - 1);
  JobSpec s = cpuScan();
  s.order = Order::Random;
  s.threads = 2;
  s.addresses = {encodeP2PKH(hash160Of(mulG(first), true)), encodeP2PKH(hash160Of(mulG(last), true))};
  s.rangeStart = first.toHex();
  s.rangeEnd = last.toHex();
  JobStatus st = runToEnd(m, s);
  REQUIRE(st.outcome == Outcome::Found);
  std::vector<std::string> found;
  for (const auto& f : st.found) found.push_back(f.privkey);
  std::sort(found.begin(), found.end());
  CHECK(found == std::vector<std::string>{first.toHex(true), last.toHex(true)});
  CHECK(st.workDone <= static_cast<double>(width));
}

TEST_CASE("random scan exhausts a range without the key after exactly one pass") {
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  const uint64_t width = 4 * kSmallLimit + 100;
  JobSpec s = cpuScan();
  s.order = Order::Random;
  s.threads = 2;
  s.addresses = {encodeP2PKH(randomHash160())};
  s.rangeStart = U256::fromU64(kSmallLimit >> 1).toHex();  // half of it through the small-key prefix
  s.rangeEnd = U256::fromU64((kSmallLimit >> 1) + width - 1).toHex();
  JobStatus st = runToEnd(m, s);
  CHECK(st.outcome == Outcome::Exhausted);
  CHECK(st.workDone == static_cast<double>(width));
  REQUIRE(st.progress.has_value());
  CHECK(*st.progress == 1.0);
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
  // Checkpoints of the former with-replacement sampler never match this job.
  CHECK(resolved.identity.ends_with("|perm"));
  const fs::path ckptPath = Checkpointer(opts.stateDir, Algo::Scan, resolved.identity, true, false).path();

  // The prefix (2^19 keys) is done before any worker starts, so this much work includes random blocks.
  const double threshold = static_cast<double>((uint64_t{1} << 19) + 2 * kSmallLimit);
  {
    JobManager m(catalog(), opts);
    m.start(s);
    REQUIRE(pollUntil([&] { return m.status().workDone >= threshold; }, std::chrono::seconds(300)));
    CHECK(m.status().progress.has_value());
    m.stop();
    m.wait();
    CHECK(m.status().outcome == Outcome::Stopped);
  }
  auto payload = Checkpointer::loadFile(ckptPath, Algo::Scan, resolved.identity);
  REQUIRE(payload);
  const RandomPayload first = parseRandomPayload(*payload);
  CHECK(first.prefixCursor == U256::fromU64(kSmallLimit));  // the prefix finished before the random blocks
  CHECK(first.state.blockBits == 31);                        // 2^40 - 2^20 + 1 keys: ~2^9 blocks
  const double saved = (first.prefixCursor - start).toDouble() + first.state.keysDone.toDouble();
  CHECK(saved >= threshold);

  {
    JobManager m(catalog(), opts);
    m.start(s);
    REQUIRE(pollUntil([&] { return m.status().resumed; }, std::chrono::seconds(60)));
    CHECK(m.status().workDone >= saved);
    m.stop();
    m.wait();
    CHECK(m.status().outcome == Outcome::Stopped);
  }
  payload = Checkpointer::loadFile(ckptPath, Algo::Scan, resolved.identity);
  REQUIRE(payload);
  const RandomPayload second = parseRandomPayload(*payload);
  CHECK(second.prefixCursor == U256::fromU64(kSmallLimit));
  CHECK(second.state.key == first.state.key);  // same permutation across the resume
  CHECK(second.state.blockBits == first.state.blockBits);
  CHECK_FALSE(second.state.cursor < first.state.cursor);
  CHECK_FALSE(second.state.keysDone < first.state.keysDone);
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
    RandomPayload p{U256::fromU64(prefixCursor),
                    fixedState(PermutedSource::blockBitsFor(resolved.rangeEnd - U256::fromU64(kSmallLimit) + 1,
                                                            PermutedSource::kMinBlockBits))};
    c.setSnapshot([&] { return randomPayload(p); });
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

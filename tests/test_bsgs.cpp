#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include "catalog/catalog.h"
#include "core/point.h"
#include "core/random.h"
#include "core/u256.h"
#include "engine/job_manager.h"

using namespace bsolver;

namespace {

const Catalog& catalog() {
  static Catalog c = Catalog::builtin();
  return c;
}

struct TempDir {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("bsolver-bsgs-test-" + randomTokenHex(8));
  TempDir() { std::filesystem::create_directories(path); }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

JobSpec bsgsSpec() {
  JobSpec s;
  s.algorithm = Algorithm::Bsgs;
  s.backend = "cpu";
  s.bsgsMemoryMb = 16;
  s.checkpoint = false;
  return s;
}

// Range [start, end] with the pubkey of puzzle `id`.
JobSpec pubkeyRange(int id, const U256& start, const U256& end) {
  JobSpec s = bsgsSpec();
  s.pubkey = pubkeyHex(*catalog().find(id)->pubkey);
  s.rangeStart = start.toHex();
  s.rangeEnd = end.toHex();
  return s;
}

JobStatus runToEnd(const JobSpec& spec, const TempDir& dir) {
  JobManager jm(catalog(), ManagerOptions{dir.path, dir.path / "found.jsonl"});
  jm.start(spec);
  jm.wait();
  return jm.status();
}

void requireFound(const JobStatus& st, const U256& key) {
  INFO(st.error.value_or(""));
  REQUIRE(st.outcome == Outcome::Found);
  REQUIRE(st.found.size() == 1);
  CHECK(st.found[0].privkey == key.toHex(true));
}

const U256 kKey30 = U256::fromHex("3d94cd64");

}  // namespace

TEST_CASE("bsgs finds puzzle 40") {
  TempDir dir;
  JobSpec s = bsgsSpec();
  s.puzzle = 40;
  s.bsgsMemoryMb = 64;
  requireFound(runToEnd(s, dir), U256::fromHex("e9ae4933d6"));
}

TEST_CASE("bsgs finds a key at the range start") {
  TempDir dir;
  requireFound(runToEnd(pubkeyRange(30, kKey30, kKey30 + 1000), dir), kKey30);
}

TEST_CASE("bsgs finds a key at the range end") {
  TempDir dir;
  requireFound(runToEnd(pubkeyRange(30, kKey30 - 5000, kKey30), dir), kKey30);
}

TEST_CASE("bsgs finds a key at the end of a multi-lane range") {
  TempDir dir;
  JobSpec s = pubkeyRange(30, U256::fromU64(2), kKey30);
  s.bsgsMemoryMb = 1;  // m = 32768: the key sits in the last giant step of the last lane
  requireFound(runToEnd(s, dir), kKey30);
}

TEST_CASE("bsgs finds a key in a range spanning many blocks") {
  TempDir dir;
  JobSpec s = bsgsSpec();
  s.puzzle = 40;
  s.bsgsMemoryMb = 1;  // m = 32768: 2^39 / 65537 giant steps -> 32 blocks of 2^18
  requireFound(runToEnd(s, dir), U256::fromHex("e9ae4933d6"));
}

TEST_CASE("bsgs exhausts a range without the key") {
  TempDir dir;
  JobStatus st = runToEnd(pubkeyRange(30, kKey30 + 1, kKey30 + 3000000), dir);
  INFO(st.error.value_or(""));
  CHECK(st.outcome == Outcome::Exhausted);
  CHECK(st.found.empty());
}

TEST_CASE("bsgs resumes from a checkpoint") {
  TempDir dir;
  JobSpec s = bsgsSpec();
  s.puzzle = 60;
  s.bsgsMemoryMb = 1;
  s.checkpoint = true;
  {
    JobManager jm(catalog(), ManagerOptions{dir.path, dir.path / "found.jsonl"});
    jm.start(s);
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    jm.stop();
    jm.wait();
    REQUIRE(jm.status().outcome == Outcome::Stopped);
  }
  bool haveCheckpoint = false;
  for (const auto& e : std::filesystem::directory_iterator(dir.path)) {
    if (e.path().extension() == ".ckpt") haveCheckpoint = true;
  }
  REQUIRE(haveCheckpoint);

  JobManager jm(catalog(), ManagerOptions{dir.path, dir.path / "found.jsonl"});
  jm.start(s);
  JobStatus st;
  for (int i = 0; i < 100; ++i) {
    st = jm.status();
    if (st.workDone > 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  jm.stop();
  jm.wait();
  CHECK(st.resumed);
  CHECK(st.workDone > 0);
}

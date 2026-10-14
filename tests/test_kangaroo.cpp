#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <mutex>
#include <regex>
#include <thread>

#include "backend/cpu/cpu_kangaroo_worker.h"
#include "catalog/catalog.h"
#include "core/gtable.h"
#include "core/random.h"
#include "engine/bytes.h"
#include "engine/checkpoint.h"
#include "engine/errors.h"
#include "engine/job_manager.h"
#include "engine/kangaroo_params.h"
#include "engine/kangaroo_table.h"

using namespace bsolver;
namespace fs = std::filesystem;

namespace {
struct TempDir {
  fs::path path;
  TempDir() : path(fs::temp_directory_path() / ("bsolver-kangaroo-test-" + randomTokenHex(6))) {
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

JobSpec cpuKangaroo(int puzzle) {
  JobSpec s;
  s.puzzle = puzzle;
  s.algorithm = Algorithm::Kangaroo;
  s.backend = "cpu";
  s.checkpoint = false;
  return s;
}

std::string key64(std::string_view hex) { return std::string(64 - hex.size(), '0') + std::string(hex); }
}  // namespace

TEST_CASE("kangaroo table reports the existing entry on collision") {
  KangarooTable t;
  X128 x{1, 2};
  CHECK_FALSE(t.insert(x, {U256::fromU64(5), false}));
  CHECK_FALSE(t.insert(X128{1, 3}, {U256::fromU64(6), true}));
  auto existing = t.insert(x, {U256::fromU64(7), true});
  REQUIRE(existing);
  CHECK(existing->dist == U256::fromU64(5));
  CHECK_FALSE(existing->wild);
  CHECK(t.size() == 2);
}

TEST_CASE("cpu kangaroo worker keeps consistent state and distinguished points") {
  Rng rng(42);
  const U256 width = U256::fromU64(1) << 40;
  const Affine q = mulG(width + rng.uniform(U256{}, width - 1));
  const Affine wildBase = wildBaseFor(q, width);
  KangarooParams params = makeKangarooParams(width, wildBase, 6, makeJumpDistances(width, 2048, rng));
  params.totalKangaroos = 2048;
  params.firstKangaroo = 1024;
  auto w = makeCpuKangarooWorker(1);
  REQUIRE(w->kangarooCount() == 1024);
  w->init(params, 7);

  // Second worker of a 2048 herd: tame ordinals 512.. start at b_t + k*W/1024.
  auto start = w->sample(2);
  const U256 tameStep = width >> 10;
  CHECK(!(start[0].second < mulU64(tameStep, 512)));
  CHECK(!(mulU64(tameStep, 513) < start[0].second));
  CHECK(start[0].first == mulG(start[0].second));
  CHECK(start[1].first == add(wildBase, mulG(start[1].second)));

  std::vector<Dp> dps;
  uint64_t jumps = w->step(dps);
  CHECK(jumps > 0);
  REQUIRE(!dps.empty());
  for (size_t i = 0; i < std::min<size_t>(dps.size(), 32); ++i) {
    CHECK(dps[i].wild == (dps[i].index % 2 == 1));
    CHECK(dps[i].x.w[3] >> 58 == 0);
    CHECK(verifyKangarooPoint(dps[i].x, dps[i].dist, dps[i].wild, wildBase));
  }
  w->reset(3, 99);
  auto s = w->sample(4);
  for (size_t i = 0; i < s.size(); ++i) {
    Affine expect = i % 2 == 1 ? add(wildBase, mulG(s[i].second)) : mulG(s[i].second);
    CHECK(expect == s[i].first);
  }
}

TEST_CASE("kangaroo finds puzzle 40") {
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  auto t0 = std::chrono::steady_clock::now();
  m.start(cpuKangaroo(40));
  m.wait();
  CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(60));
  JobStatus st = m.status();
  REQUIRE(st.outcome == Outcome::Found);
  REQUIRE(st.found.size() == 1);
  CHECK(st.found[0].privkey == key64("e9ae4933d6"));
  CHECK(st.rateUnit == "jumps/s");
}

TEST_CASE("kangaroo finds puzzle 48 with two threads") {
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  JobSpec s = cpuKangaroo(48);
  s.threads = 2;
  m.start(s);
  m.wait();
  JobStatus st = m.status();
  REQUIRE(st.outcome == Outcome::Found);
  CHECK(st.found[0].privkey == key64("ade6d7ce3b9b"));
}

TEST_CASE("kangaroo resumes from a checkpoint") {
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  std::mutex mu;
  std::vector<std::string> logs;
  m.subscribe([&](const Event& e) {
    if (e.type != "log") return;
    std::lock_guard<std::mutex> lock(mu);
    logs.push_back(e.payload["message"].get<std::string>());
  });
  JobSpec s = cpuKangaroo(48);
  s.threads = 1;
  s.checkpoint = true;

  // A single thread needs several seconds on average; retry in the unlikely case the first second already solves it.
  bool interrupted = false;
  for (int attempt = 0; attempt < 5 && !interrupted; ++attempt) {
    s.fresh = true;
    m.start(s);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    try {
      m.stop();
    } catch (const ConflictError&) {
      // Already finished (found).
    }
    m.wait();
    interrupted = m.status().outcome == Outcome::Stopped;
  }
  REQUIRE(interrupted);
  const double firstWork = m.status().workDone;
  CHECK(firstWork > 0);
  REQUIRE_FALSE(fs::is_empty(dir.path / "state"));

  {
    std::lock_guard<std::mutex> lock(mu);
    logs.clear();
  }
  s.fresh = false;
  s.threads = 0;
  m.start(s);
  m.wait();
  JobStatus st = m.status();
  CHECK(st.resumed);
  CHECK(st.workDone > firstWork);
  REQUIRE(st.outcome == Outcome::Found);
  CHECK(st.found[0].privkey == key64("ade6d7ce3b9b"));
  CHECK(fs::is_empty(dir.path / "state"));

  std::lock_guard<std::mutex> lock(mu);
  const std::regex restored(R"(resuming from checkpoint .*: (\d+) distinguished points, (\d+) jumps)");
  bool seen = false;
  for (const auto& line : logs) {
    std::smatch match;
    if (std::regex_search(line, match, restored)) {
      seen = true;
      CHECK(std::stoull(match[1].str()) > 0);
      CHECK(std::stoull(match[2].str()) > 0);
    }
  }
  CHECK(seen);
}

TEST_CASE("kangaroo rejects a checkpoint with dp bits above 60") {
  TempDir dir;
  const ManagerOptions opts{dir.path / "state", dir.path / "found.jsonl"};
  JobSpec s = cpuKangaroo(48);
  s.checkpoint = true;
  Checkpointer c(opts.stateDir, Algo::Kangaroo, resolve(s, catalog()).identity, true, false);
  c.setSnapshot([] {
    ByteWriter w;
    w.u32(61);
    for (int i = 0; i < 64; ++i) w.u256(U256::fromU64(1));
    w.u64(0);  // jumps
    w.u64(0);  // DP count
    return std::move(w.data());
  });
  c.saveNow();

  JobManager m(catalog(), opts);
  m.start(s);
  m.wait();
  JobStatus st = m.status();
  CHECK(st.outcome == Outcome::Error);
  CHECK(st.error == "checkpoint " + c.path().string() + " is corrupt (use --fresh)");
}

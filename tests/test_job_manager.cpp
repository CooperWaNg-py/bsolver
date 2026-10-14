#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <optional>
#include <thread>

#include "catalog/catalog.h"
#include "core/address.h"
#include "core/random.h"
#include "engine/bytes.h"
#include "engine/checkpoint.h"
#include "engine/errors.h"
#include "engine/job_manager.h"

using namespace bsolver;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {
struct TempDir {
  fs::path path;
  TempDir() : path(fs::temp_directory_path() / ("bsolver-jm-test-" + randomTokenHex(6))) {
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

template <class Pred>
bool pollUntil(Pred pred, std::chrono::seconds deadline) {
  const auto end = Clock::now() + deadline;
  while (!pred()) {
    if (Clock::now() > end) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return true;
}

// Waits until workDone stays unchanged for a whole `quiet` window (the in-flight segments have drained and every
// worker blocks in waitIfPaused) and returns that value; nullopt when work still changes at the deadline.
std::optional<double> waitFrozen(const JobManager& m, Clock::duration quiet, std::chrono::seconds deadline) {
  const auto end = Clock::now() + deadline;
  double last = m.status().workDone;
  while (Clock::now() < end) {
    std::this_thread::sleep_for(quiet);
    const double now = m.status().workDone;
    if (now == last) return now;
    last = now;
  }
  return std::nullopt;
}
}  // namespace

TEST_CASE("pause freezes work, resume continues it, stop while paused ends the job") {
  TempDir dir;
  Rng rng;
  Hash160 h;
  for (auto& b : h) b = static_cast<uint8_t>(rng.next());
  const U256 start = U256::fromU64(uint64_t{1} << 40);
  JobSpec s;
  s.algorithm = Algorithm::Scan;
  s.backend = "cpu";  // explicit backend: no benchmark, so the job is Running from the start
  s.threads = 2;
  s.addresses = {encodeP2PKH(h)};  // never found
  s.rangeStart = start.toHex();
  s.rangeEnd = (start + (U256::fromU64(1) << 40)).toHex();
  const ManagerOptions opts{dir.path / "state", dir.path / "found.jsonl"};
  const fs::path ckptPath =
      Checkpointer(opts.stateDir, Algo::Scan, resolve(s, catalog()).identity, true, false).path();

  JobManager m(catalog(), opts);
  CHECK_THROWS_WITH(m.pause(), "no active job");
  const auto t0 = Clock::now();
  CHECK(m.start(s).state == JobState::Running);
  CHECK_THROWS_WITH(m.resume(), "job is not paused");

  // The first completed segment means the runner has registered its checkpoint snapshot. Its duration bounds how long
  // an in-flight segment can take, which sizes the "work is frozen" window on slow and fast machines alike.
  REQUIRE(pollUntil([&] { return m.status().workDone > 0; }, std::chrono::seconds(300)));
  const auto segment = Clock::now() - t0;
  const Clock::duration quiet = std::max<Clock::duration>(std::chrono::milliseconds(1500), 3 * segment);

  JobStatus st = m.pause();
  CHECK(st.state == JobState::Paused);
  CHECK(m.status().state == JobState::Paused);
  // pause() saves synchronously.
  REQUIRE(fs::exists(ckptPath));
  auto payload = Checkpointer::loadFile(ckptPath, Algo::Scan, resolve(s, catalog()).identity);
  REQUIRE(payload);
  ByteReader r(*payload);
  CHECK_FALSE(r.u256() < start);

  CHECK_THROWS_WITH(m.pause(), "job is not running");

  const std::optional<double> frozen = waitFrozen(m, quiet, std::chrono::seconds(300));
  REQUIRE(frozen);
  const double a = m.status().workDone;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  const double b = m.status().workDone;
  CHECK(a == *frozen);
  CHECK(b == a);
  CHECK(m.status().state == JobState::Paused);
  CHECK_THROWS_WITH(m.pause(), "job is not running");

  st = m.resume();
  CHECK(st.state == JobState::Running);
  CHECK_THROWS_WITH(m.resume(), "job is not paused");
  REQUIRE(pollUntil([&] { return m.status().workDone > b; }, std::chrono::seconds(300)));

  // Stop while paused, once every worker blocks in waitIfPaused: the job ends promptly and processes nothing more.
  CHECK(m.pause().state == JobState::Paused);
  const std::optional<double> frozen2 = waitFrozen(m, quiet, std::chrono::seconds(300));
  REQUIRE(frozen2);
  CHECK(*frozen2 > b);
  CHECK(m.stop().state == JobState::Stopping);
  const auto stopAt = Clock::now();
  m.wait();
  CHECK(Clock::now() - stopAt < std::chrono::seconds(30));
  st = m.status();
  CHECK(st.state == JobState::Finished);
  CHECK(st.outcome == Outcome::Stopped);
  CHECK(st.workDone == *frozen2);
  CHECK(fs::exists(ckptPath));  // a stopped job keeps its checkpoint
  CHECK_THROWS_WITH(m.pause(), "no active job");
}

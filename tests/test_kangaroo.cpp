#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <regex>
#include <thread>

#include "backend/cpu/cpu_kangaroo_worker.h"
#include "catalog/catalog.h"
#include "core/gtable.h"
#include "core/random.h"
#include "core/scalar.h"
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

const U256 kWidth = U256::fromU64(1) << 40;

// Q = (a + k)*G in [a, a + W) with a = W; returns the params for it (dp 6) and K' = k + shift.
KangarooParams testParams(uint64_t seed, U256* kPrime = nullptr) {
  Rng rng(seed);
  const U256 k = rng.uniform(U256{}, kWidth - 1);
  const U256 shift = kangarooShift(kWidth);
  if (kPrime) *kPrime = k + shift;
  const Affine wildBase = wildBaseFor(mulG(kWidth + k), kWidth, shift);
  return makeKangarooParams(kWidth, wildBase, 6, makeJumpDistances(kWidth, rng));
}

Affine expectedPoint(const KangarooParams& p, uint64_t globalIndex, const U256& dist) {
  Affine e = mulG(signedToScalar(dist));
  return isWildKangaroo(globalIndex) ? add(p.wildBase, e) : e;
}

KangarooState negated(const KangarooState& s) {
  KangarooState n = s;
  n.point.y = feNeg(s.point.y);
  n.dist = U256{} - s.dist;
  return n;
}
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

TEST_CASE("kangaroo dp bits follow the kangaroo count") {
  const U256 one = U256::fromU64(1);
  CHECK(autoDpBits(one << 47, 1024) == 10);       // floor(23.5 - 13)
  CHECK(autoDpBits(one << 139, 1u << 19) == 47);  // floor(69.5 - 22)
  CHECK(autoDpBits(one << 139, 1024) == 56);
  CHECK(autoDpBits(one << 32, 1u << 19) == 0);
  CHECK(autoDpBits(one << 160, 1) == 60);                  // clamped
  CHECK(autoDpBits(one << 100, uint64_t{1} << 40) == 27);  // DP table bound: ceil(log2(1.3 * 2^50 / 2^24))
}

TEST_CASE("kangaroo starts follow the 1:2 tame/wild herd layout") {
  KangarooParams p = testParams(1);
  p.totalKangaroos = 3072;
  p.firstKangaroo = 1024;
  auto starts = kangarooStarts(p, 1024, 5);
  REQUIRE(starts.size() == 1024);
  const U256 shift = kangarooShift(kWidth);
  const U256 tameStep = divU64(shift, 1024);
  for (size_t i = 0; i < starts.size(); ++i) {
    const uint64_t g = 1024 + i;
    const KangarooState& s = starts[i];
    CHECK_FALSE(s.lookahead);
    CHECK(s.point == expectedPoint(p, g, s.dist));
    if (isWildKangaroo(g)) {
      CHECK((s.dist.w[0] & 1) == 0);
      CHECK(s.dist < mulU64(shift, 34));
    } else {
      // Tame ordinal g/3 sits in [ordinal*step + 1, (ordinal + 1)*step].
      CHECK_FALSE(s.dist < mulU64(tameStep, g / 3) + 1);
      CHECK_FALSE(mulU64(tameStep, g / 3 + 1) < s.dist);
    }
  }
  for (uint32_t i : {0u, 1u, 2u}) {
    KangarooState r = kangarooResetStart(p, i, 77 + i);
    CHECK(r.point == expectedPoint(p, 1024 + i, r.dist));
  }
}

TEST_CASE("cpu kangaroo worker matches the reference step and keeps its invariants") {
  KangarooParams p = testParams(2);
  p.totalKangaroos = 2048;
  p.firstKangaroo = 1024;
  auto w = makeCpuKangarooWorker(1);
  REQUIRE(w->kangarooCount() == 1024);
  w->init(p, 7, {});
  auto before = w->sample(12);

  std::vector<Dp> dps;
  const uint64_t jumps = w->step(dps);
  CHECK(jumps == 128 * 1024);
  REQUIRE(!dps.empty());
  for (size_t i = 0; i < std::min<size_t>(dps.size(), 32); ++i) {
    CHECK(dps[i].wild == isWildKangaroo(1024 + dps[i].index));
    CHECK(dps[i].x.w[3] >> 58 == 0);
    CHECK(verifyKangarooPoint(dps[i].x, dps[i].dist, dps[i].wild, p.wildBase));
  }
  auto after = w->sample(12);
  for (size_t i = 0; i < before.size(); ++i) {
    KangarooState s = before[i];
    for (int k = 0; k < 128; ++k) REQUIRE(kangarooStep(p, s));
    CHECK(s.point == after[i].point);
    CHECK(s.dist == after[i].dist);
    CHECK(s.lookahead == after[i].lookahead);
    CHECK(after[i].point == expectedPoint(p, 1024 + i, after[i].dist));
  }

  w->reset(3, 99);
  auto s = w->sample(4);
  CHECK(s[3].point == expectedPoint(p, 1027, s[3].dist));

  // Resumed states are taken over exactly; the rest start fresh.
  auto other = makeCpuKangarooWorker(2);
  other->init(p, 8, after);
  auto resumed = other->sample(12);
  for (size_t i = 0; i < after.size(); ++i) {
    CHECK(resumed[i].point == after[i].point);
    CHECK(resumed[i].dist == after[i].dist);
    CHECK(resumed[i].lookahead == after[i].lookahead);
  }
}

TEST_CASE("kangaroo look-ahead avoids 2-cycles") {
  KangarooParams p = testParams(3);
  // Find a main-table step whose successor would jump straight back.
  KangarooState hit;
  bool found = false;
  for (uint64_t t = 1; t < 200000 && !found; ++t) {
    KangarooState s{mulG(U256::fromU64(t)), U256::fromU64(t), false};
    KangarooState next = s;
    REQUIRE(kangarooStep(p, next));
    if (next.lookahead) {
      hit = s;
      found = true;
    }
  }
  REQUIRE(found);
  KangarooState s = hit;
  REQUIRE(kangarooStep(p, s));
  REQUIRE(s.lookahead);
  REQUIRE(kangarooStep(p, s));  // look-ahead table step instead of returning to `hit`
  CHECK_FALSE(s.lookahead);
  CHECK_FALSE(s.point.x == hit.point.x);
  CHECK(s.point == mulG(signedToScalar(s.dist)));
}

TEST_CASE("kangaroo collisions solve tame-wild (both signs) and wild-wild mirrors") {
  U256 kPrime;
  const KangarooParams p = testParams(4, &kPrime);
  const U256 shift = kangarooShift(kWidth);
  Rng rng(11);
  const U256 t = rng.uniform(U256::fromU64(1), shift);
  auto pointOf = [&](bool wild, const U256& signedDist) {
    Affine e = mulG(signedToScalar(signedDist));
    return wild ? add(p.wildBase, e) : e;
  };
  const U256 key = kWidth + kPrime - shift;
  auto hasKey = [&](const std::vector<U256>& c) {
    for (const U256& k : c) {
      if (kangarooKey(kWidth, shift, k) == key) return true;
    }
    return false;
  };

  // Wild at the tame point: d - K' = t.
  const U256 dPlus = t + kPrime;
  REQUIRE(pointOf(false, t) == pointOf(true, dPlus));
  CHECK(hasKey(collisionCandidates(false, t, true, dPlus)));
  CHECK(hasKey(collisionCandidates(true, dPlus, false, t)));

  // Wild at the negated tame point: d - K' = -t; the tame distance itself is negative (two's complement).
  const U256 negT = U256{} - t;
  REQUIRE(pointOf(false, negT).x == pointOf(false, t).x);
  const U256 dMinus = kPrime - t;
  REQUIRE(pointOf(true, dMinus).x == pointOf(false, t).x);
  CHECK(hasKey(collisionCandidates(false, signedToScalar(negT), true, dMinus)));
  CHECK(hasKey(collisionCandidates(false, t, true, dMinus)));

  // Wild-wild mirror: d1 - K' = -(d2 - K'), d2 possibly negative (two's complement).
  const U256 d1 = rng.uniform(U256{}, kWidth) << 1;
  const U256 d2 = (kPrime << 1) - d1;
  REQUIRE(pointOf(true, d1).x == pointOf(true, d2).x);
  REQUIRE_FALSE(pointOf(true, d1) == pointOf(true, d2));
  CHECK(hasKey(collisionCandidates(true, d1, true, signedToScalar(d2))));
  CHECK(hasKey(collisionCandidates(true, signedToScalar(d2), true, d1)));
  const U256 d3 = (kPrime << 1) + U256::fromU64(1000);  // mirror of the negative wild distance -1000
  const U256 neg1000 = U256{} - U256::fromU64(1000);
  REQUIRE(pointOf(true, d3).x == pointOf(true, neg1000).x);
  CHECK(hasKey(collisionCandidates(true, d3, true, signedToScalar(neg1000))));

  // Merged walks give nothing to solve.
  CHECK(collisionCandidates(false, t, false, negMod(t)).empty());
  CHECK(collisionCandidates(true, d1, true, d1).empty());
}

TEST_CASE("kangaroo cycle escape is deterministic across merged and mirrored walks") {
  // Degenerate tables (one jump everywhere) trap every walk in a 2-cycle P <-> P + s*J.
  Rng rng(6);
  const U256 shift = kangarooShift(kWidth);
  const Affine wildBase = wildBaseFor(mulG(kWidth + U256::fromU64(12345)), kWidth, shift);
  KangarooJumpDistances d = makeJumpDistances(kWidth, rng);
  std::fill(d.jumps.begin(), d.jumps.end(), d.jumps[0]);
  std::fill(d.lookahead.begin(), d.lookahead.end(), d.jumps[0]);
  KangarooParams p = makeKangarooParams(kWidth, wildBase, 60, d);

  KangarooState a, b;
  bool found = false;
  for (uint64_t t = 1000; !found; ++t) {
    a = {mulG(U256::fromU64(t)), U256::fromU64(t), false};
    b = a;
    REQUIRE(kangarooStep(p, b));
    KangarooState next = b;
    REQUIRE(kangarooStep(p, next));
    found = !b.lookahead && !next.lookahead && next.point == a.point;  // a genuine 2-cycle
  }

  KangarooState ea = a, eb = b, ec = negated(a);
  REQUIRE(escapeKangarooCycle(p, ea));
  REQUIRE(escapeKangarooCycle(p, eb));
  REQUIRE(escapeKangarooCycle(p, ec));
  CHECK(ea.point == eb.point);
  CHECK(ea.dist == eb.dist);
  CHECK_FALSE(ea.point.x == a.point.x);
  CHECK_FALSE(ea.point.x == b.point.x);
  CHECK(ec.point == negated(ea).point);
  CHECK(ec.dist == negated(ea).dist);
  CHECK(ea.point == mulG(signedToScalar(ea.dist)));

  // Off a cycle nothing changes.
  KangarooParams real = testParams(5);
  KangarooState free{mulG(U256::fromU64(424242)), U256::fromU64(424242), false};
  KangarooState copy = free;
  CHECK_FALSE(escapeKangarooCycle(real, copy));
  CHECK(copy.point == free.point);
  CHECK(copy.dist == free.dist);

  // The CPU worker detects the trap itself; the mirrored kangaroo leaves it in lockstep.
  auto w = makeCpuKangarooWorker(0);
  w->init(p, 3, {a, kangarooResetStart(p, 1, 1), kangarooResetStart(p, 2, 2), negated(a)});
  std::vector<Dp> dps;
  w->step(dps);
  auto s = w->sample(4);
  CHECK(s[0].point.x == s[3].point.x);
  CHECK(s[3].point == negated(s[0]).point);
  CHECK_FALSE(s[0].point.x == a.point.x);
  CHECK_FALSE(s[0].point.x == b.point.x);
  CHECK(s[0].point == mulG(signedToScalar(s[0].dist)));
}

TEST_CASE("kangaroo checkpoint payload round-trips") {
  Rng rng(9);
  KangarooCheckpoint h;
  h.dpBits = 12;
  h.designKangaroos = 4096;
  h.dist = makeJumpDistances(kWidth, rng);
  h.jumps = 123456789;
  KangarooTable table;
  table.insert(X128{5, 6}, {U256::fromU64(77), true});
  table.insert(X128{7, 8}, {U256::fromU64(99), false});
  std::vector<KangarooState> states = {{Affine{}, U256{} - U256::fromU64(3), true}, {Affine{}, U256::fromU64(8), false}};
  auto bytes = writeKangarooCheckpoint(h, table, states);

  ByteReader r(bytes);
  KangarooTable table2;
  std::vector<KangarooState> states2;
  KangarooCheckpoint h2 = readKangarooCheckpoint(r, table2, states2);
  CHECK(r.remaining() == 0);
  CHECK(h2.dpBits == 12);
  CHECK(h2.designKangaroos == 4096);
  CHECK(h2.dist.jumps == h.dist.jumps);
  CHECK(h2.dist.lookahead == h.dist.lookahead);
  CHECK(h2.dist.cycle == h.dist.cycle);
  CHECK(h2.jumps == h.jumps);
  CHECK(table2.size() == 2);
  auto e = table2.insert(X128{5, 6}, {});
  REQUIRE(e);
  CHECK(e->dist == U256::fromU64(77));
  CHECK(e->wild);
  REQUIRE(states2.size() == 2);
  CHECK(states2[0].dist == states[0].dist);
  CHECK(states2[0].lookahead);
  CHECK_FALSE(states2[1].lookahead);
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

TEST_CASE("kangaroo resumes its walks from a checkpoint") {
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
  const ResolvedJob job = resolve(s, catalog());
  CHECK(job.identity.ends_with("|sym"));
  const fs::path ckpt = Checkpointer(dir.path / "state", Algo::Kangaroo, job.identity, true, false).path();
  auto load = [&](KangarooTable& table, std::vector<KangarooState>& states) {
    auto payload = Checkpointer::loadFile(ckpt, Algo::Kangaroo, job.identity);
    REQUIRE(payload);
    ByteReader r(*payload);
    KangarooCheckpoint h = readKangarooCheckpoint(r, table, states);
    CHECK(r.remaining() == 0);
    return h;
  };
  auto runFor = [&](std::chrono::milliseconds d) {
    m.start(s);
    std::this_thread::sleep_for(d);
    try {
      m.stop();
    } catch (const ConflictError&) {
      // Already finished (found).
    }
    m.wait();
    return m.status();
  };

  // A single thread needs a few seconds on average; retry in the unlikely case the first second already solves it.
  JobStatus st;
  for (int attempt = 0; attempt < 5; ++attempt) {
    s.fresh = true;
    st = runFor(std::chrono::seconds(1));
    if (st.outcome == Outcome::Stopped) break;
  }
  REQUIRE(st.outcome == Outcome::Stopped);
  const double firstWork = st.workDone;
  KangarooTable table1;
  std::vector<KangarooState> states1;
  const KangarooCheckpoint h1 = load(table1, states1);
  CHECK(static_cast<double>(h1.jumps) == firstWork);
  CHECK(h1.designKangaroos == 1024);
  CHECK(table1.size() > 0);
  REQUIRE(states1.size() == 1024);

  // The second run continues each walk exactly where the first stopped: replaying the CPU worker (step plus
  // cycle detection) from the saved states for the jumps it made reproduces its saved states.
  s.fresh = false;
  st = runFor(std::chrono::milliseconds(300));
  CHECK(st.resumed);
  if (st.outcome == Outcome::Stopped) {
    KangarooTable table2;
    std::vector<KangarooState> states2;
    const KangarooCheckpoint h2 = load(table2, states2);
    REQUIRE(states2.size() == 1024);
    const double live = static_cast<double>(h2.jumps) - firstWork;
    CHECK(live >= 0);
    CHECK(std::fmod(live, 128.0 * 1024) == 0);
    const uint64_t rounds = static_cast<uint64_t>(live / 1024);

    const Affine wildBase = wildBaseFor(*job.pubkey, job.rangeStart, kangarooShift(job.width()));
    KangarooParams p = makeKangarooParams(job.width(), wildBase, h1.dpBits, h1.dist);
    p.totalKangaroos = 1024;
    restoreKangarooPoints(p, states1, 1);
    int same = 0;
    for (size_t i = 0; i < 16; ++i) {
      KangarooState k = states1[i];
      uint64_t ref = k.point.x.n[0];
      for (uint64_t r = 0; r < rounds; ++r) {
        REQUIRE(kangarooStep(p, k));
        if (k.point.x.n[0] == ref) {
          if (escapeKangarooCycle(p, k)) ref = k.point.x.n[0];
        } else if ((r + 1) % kKangarooCycleWindow == 0) {
          ref = k.point.x.n[0];
        }
      }
      same += k.dist == states2[i].dist && k.lookahead == states2[i].lookahead;
    }
    CHECK(same >= 14);  // a kangaroo reset by a same-type collision walks elsewhere
  }

  {
    std::lock_guard<std::mutex> lock(mu);
    logs.clear();
  }
  s.threads = 4;  // 4096 kangaroos: 1024 restored walks plus fresh ones, dp re-tuned
  m.start(s);
  m.wait();
  st = m.status();
  CHECK(st.resumed);
  CHECK(st.workDone > firstWork);
  REQUIRE(st.outcome == Outcome::Found);
  CHECK(st.found[0].privkey == key64("ade6d7ce3b9b"));
  CHECK_FALSE(fs::exists(ckpt));  // a found job deletes its checkpoint (the empty .lock file stays)

  std::lock_guard<std::mutex> lock(mu);
  const std::regex restored(R"(resuming from checkpoint .*: (\d+) distinguished points, (\d+) kangaroo states)");
  bool seen = false, retuned = false;
  for (const auto& line : logs) {
    std::smatch match;
    if (std::regex_search(line, match, restored)) {
      seen = true;
      CHECK(std::stoull(match[1].str()) > 0);
      CHECK(std::stoull(match[2].str()) == 1024);
    }
    retuned = retuned || line.find("was chosen for 1024 kangaroos, resuming with 4096") != std::string::npos;
  }
  CHECK(seen);
  CHECK(retuned);
}

TEST_CASE("kangaroo pause and stop checkpoint every kangaroo state") {
  TempDir dir;
  JobManager m(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  JobSpec s = cpuKangaroo(60);  // ~2^30 jumps: runs far longer than this test
  s.threads = 2;
  s.checkpoint = true;
  const ResolvedJob job = resolve(s, catalog());
  const fs::path ckpt = Checkpointer(dir.path / "state", Algo::Kangaroo, job.identity, true, false).path();
  auto load = [&](std::vector<KangarooState>& states) {
    auto payload = Checkpointer::loadFile(ckpt, Algo::Kangaroo, job.identity);
    REQUIRE(payload);
    ByteReader r(*payload);
    KangarooTable table;
    return readKangarooCheckpoint(r, table, states);
  };

  m.start(s);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  m.pause();  // saves while the workers are parked
  std::vector<KangarooState> paused;
  const uint64_t pausedJumps = load(paused).jumps;
  CHECK(paused.size() == 2048);
  CHECK(pausedJumps > 0);
  m.resume();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  m.stop();
  m.wait();
  CHECK(m.status().outcome == Outcome::Stopped);
  std::vector<KangarooState> stopped;
  CHECK(load(stopped).jumps > pausedJumps);
  CHECK(stopped.size() == 2048);
}

TEST_CASE("kangaroo rejects a checkpoint with dp bits above 60") {
  TempDir dir;
  const ManagerOptions opts{dir.path / "state", dir.path / "found.jsonl"};
  JobSpec s = cpuKangaroo(48);
  s.checkpoint = true;
  Checkpointer c(opts.stateDir, Algo::Kangaroo, resolve(s, catalog()).identity, true, false);
  c.setSnapshot([] {
    Rng rng(1);
    KangarooCheckpoint h;
    h.dpBits = 61;
    h.dist = makeJumpDistances(kWidth, rng);
    return writeKangarooCheckpoint(h, KangarooTable{}, {});
  });
  c.saveNow();

  JobManager m(catalog(), opts);
  m.start(s);
  m.wait();
  JobStatus st = m.status();
  CHECK(st.outcome == Outcome::Error);
  CHECK(st.error == "checkpoint " + c.path().string() + " is corrupt (use --fresh)");
}

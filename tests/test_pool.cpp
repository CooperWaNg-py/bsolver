#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <thread>

#include "catalog/catalog.h"
#include "core/address.h"
#include "core/gtable.h"
#include "core/random.h"
#include "core/scalar.h"
#include "engine/checkpoint.h"
#include "engine/job_manager.h"
#include "engine/job_spec.h"
#include "pool/http_client.h"
#include "pool/pool_api.h"
#include "pool/pool_config.h"
#include "pool/pool_session.h"
#include "pool/pool_state.h"

using namespace bsolver;
namespace fs = std::filesystem;
using nlohmann::json;

TEST_CASE("pool.conf parsing matches the official client format") {
  std::vector<std::string> warnings;
  PoolConfig c = PoolConfig::parse(R"(# Client Pool Configuration
user_token = "tok"
worker_name=rig1
target_puzzle=38
gpu_index=1
untrusted_computer=yes
public_key=-----BEGIN PUBLIC KEY-----|ABC@DEF|-----END PUBLIC KEY-----
telegram_share=false
api_share=1
api_share_url=http://x/hook
save_key=false
custom_range=none
mystery=1
)",
                                   &warnings);
  CHECK(c.userToken == "tok");
  CHECK(c.workerName == "rig1");
  CHECK(c.targetPuzzle == 38);
  CHECK(c.untrustedComputer);
  CHECK(c.publicKey == "-----BEGIN PUBLIC KEY-----\nABC\nDEF\n-----END PUBLIC KEY-----");
  CHECK(c.apiShare);
  CHECK(c.apiShareUrl == "http://x/hook");
  REQUIRE(warnings.size() == 1);
  CHECK(warnings[0] == "pool config: unknown key 'mystery' ignored");
  CHECK_THROWS_WITH(PoolConfig::parse("target_puzzle=abc"), "pool config: target_puzzle must be a number, got 'abc'");
}

TEST_CASE("pool security hash and proof hash match the official client") {
  PoolConfig c;
  c.userToken = std::string(128, 'a');
  c.apiShare = true;
  c.apiShareUrl = "http://x/hook";
  c.targetPuzzle = 38;
  c.buildHash = "abc";
  // Expected values computed independently (Python hashlib) from the official client's formulas.
  CHECK(c.securityHash() == "4101d3111fe2d707ae555cd65c9f2f7fa1a70736d013d92467d3598ec7848a84");
  CHECK(hashedProofKey({"000000000000000000000000000000000000000000000003AB8443AB91BBAFE8",
                        "000000000000000000000000000000000000000000000003AB84435AF1E51FD1"}) ==
        "c6072f7694e85786fbcdb3cd8b94845ef5e191bc706b6e013cb0ba674752224d");
}

TEST_CASE("pool config validation") {
  PoolConfig c;
  CHECK_THROWS_WITH(c.validate(), "user_token is required (get it from your btcpuzzle.info account)");
  c.userToken = "short";
  CHECK_THROWS_WITH(c.validate(), "user_token must be 128 characters long");
  c.userToken = std::string(128, 'a');
  c.validate();
  c.workerName = "J\xf6rg";  // Latin-1 / ANSI bytes would break the pool state JSON and headers
  CHECK_THROWS_WITH(c.validate(), "worker_name must use printable ASCII characters without spaces");
  c.workerName = "rig 1";
  CHECK_THROWS(c.validate());
  c.workerName = "rig-1_a.b";
  c.validate();
  c.untrustedComputer = true;
  CHECK_THROWS(c.validate());
  c.telegramShare = true;
  CHECK_THROWS_WITH(c.validate(), "telegram_share is enabled but telegram_token or telegram_chat_id is missing");
  c.telegramToken = "t";
  c.telegramChatId = "1";
  c.validate();
  c.saveKey = true;
  CHECK_THROWS(c.validate());
}

TEST_CASE("pool range response parsing (API documentation example)") {
  PoolRange r = parseRangeResponse(R"({"hex":"3AB8443","puzzleCode":71,"workloadStart":"0000000000",
    "workloadEnd":"3ffffffffff","targetAddress":"1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ",
    "proofOfWorkAddresses":["1QLLeQ3Hwp9ai7L4KM65e7SxS12skkYUsA","15mLWAX7Hno94fntWj6PEx5wCvZz2zAn8a"]})");
  CHECK(r.start().toHex() == "3ab84430000000000");
  CHECK(r.end().toHex() == "3ab8446ffffffffff");
  CHECK(r.proofAddresses.size() == 2);
  CHECK_THROWS_AS(parseRangeResponse(R"({"hex":"zz","workloadStart":"0","workloadEnd":"1","targetAddress":"x",
    "proofOfWorkAddresses":[]})"),
                  PoolError);
}

TEST_CASE("pool range responses outside the key space are rejected") {
  auto body = [](const std::string& hex, const std::string& start, const std::string& end) {
    return json{{"hex", hex},
                {"workloadStart", start},
                {"workloadEnd", end},
                {"targetAddress", "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ"},
                {"proofOfWorkAddresses", {"1QLLeQ3Hwp9ai7L4KM65e7SxS12skkYUsA"}}}
        .dump();
  };
  auto rejected = [](const std::string& b) {
    try {
      parseRangeResponse(b);
      return false;
    } catch (const PoolError& ex) {
      return !ex.retryable();
    }
  };
  const std::string nHigh = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8C";  // n = nHigh || D0364141
  CHECK(rejected(body("0", "0000", "FFFF")));                                    // starts at key 0
  CHECK(rejected(body("1", std::string(63, '0'), std::string(64, 'F'))));        // end wraps past 2^256
  CHECK(rejected(body(nHigh, "00000000", "D0364141")));                          // end == n
  CHECK(rejected(body(std::string(56, 'F'), "00000000", "1")));                  // start > n
  CHECK(parseRangeResponse(body(nHigh, "00000000", "D0364140")).end() == kOrderN - 1);  // end == n - 1 is fine
  CHECK(parseRangeResponse(body("0", "0000", "FFFF"), false).start().isZero());  // saved state: structure only
}

namespace {
std::string upperHex64(const U256& k) {
  std::string s = k.toHex(true);
  for (auto& ch : s) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  return s;
}

// Minimal btcpuzzle.info API stand-in serving one synthetic range at 2^37 (puzzle 38's range).
struct MockPool {
  httplib::Server server;
  std::thread thread;
  int port = 0;
  std::string hex = "200", workloadStart = "0000000", workloadEnd;
  std::vector<U256> proofKeys;
  U256 targetKey;
  std::mutex mu;
  int gets = 0, pings = 0;
  int unfindableGets = 0;         // the next N ranges carry an extra proof address whose key lies outside the range
  int failPuts = 0;               // the next N submissions answer 503
  int pingsDuringFailedPuts = 0;  // pings seen when the last failing submission arrived
  int failedPuts = 0;             // submissions answered 503 so far
  std::vector<httplib::Headers> puts;
  std::vector<std::string> webhookStatuses;
  httplib::Headers keyFound;

  static std::string address(const U256& k) { return encodeP2PKH(hash160Of(mulG(k), true)); }

  MockPool(std::string end, std::vector<uint64_t> proofOffsets, uint64_t targetOffset) : workloadEnd(std::move(end)) {
    const U256 start = U256::fromHex(hex + workloadStart);
    for (uint64_t o : proofOffsets) proofKeys.push_back(start + o);
    targetKey = start + targetOffset;
    server.Get("/puzzle/38/range", [this](const httplib::Request& req, httplib::Response& res) {
      std::lock_guard<std::mutex> lock(mu);
      ++gets;
      const std::string token = req.get_header_value("UserToken");
      if (token.size() != 128 || token[0] == 'b') {
        res.status = 400;
        res.set_content(R"({"title":"One or more validation errors occurred.","errors":{"UserToken":["bad"]}})",
                        "application/problem+json");
        return;
      }
      json r = rangeJson();
      if (unfindableGets > 0) {
        --unfindableGets;
        r["proofOfWorkAddresses"].push_back(address(U256::fromHex(hex + workloadStart) + (uint64_t{1} << 30)));
      }
      res.set_content(r.dump(), "application/json");
    });
    server.Put("/puzzle/38/range", [this](const httplib::Request& req, httplib::Response& res) {
      std::lock_guard<std::mutex> lock(mu);
      if (failPuts > 0) {
        --failPuts;
        ++failedPuts;
        pingsDuringFailedPuts = pings;
        res.status = 503;
        return;
      }
      puts.push_back(req.headers);
      res.set_content(R"({"isSuccess":true})", "application/json");
    });
    server.Patch("/puzzle/38/range/ping", [this](const httplib::Request&, httplib::Response& res) {
      std::lock_guard<std::mutex> lock(mu);
      ++pings;
      res.set_content("{}", "application/json");
    });
    server.Post("/webhook", [this](const httplib::Request& req, httplib::Response& res) {
      std::lock_guard<std::mutex> lock(mu);
      webhookStatuses.push_back(req.get_header_value("Status"));
      if (req.get_header_value("Status") == "keyFound") keyFound = req.headers;
      res.set_content("true", "text/plain");
    });
    port = server.bind_to_any_port("127.0.0.1");
    thread = std::thread([this] { server.listen_after_bind(); });
    server.wait_until_ready();
  }
  ~MockPool() {
    server.stop();
    thread.join();
  }

  // Caller holds mu (or the server is idle).
  json rangeJson() const {
    json addrs = json::array();
    for (const auto& k : proofKeys) addrs.push_back(address(k));
    return json{{"hex", hex},
                {"puzzleCode", 38},
                {"workloadStart", workloadStart},
                {"workloadEnd", workloadEnd},
                {"targetAddress", address(targetKey)},
                {"proofOfWorkAddresses", addrs}};
  }

  std::string expectedProofHash() const {
    std::vector<std::string> keys;
    for (const auto& k : proofKeys) keys.push_back(upperHex64(k));
    return hashedProofKey(keys);
  }

  PoolSessionOptions options() const {
    PoolSessionOptions o;
    o.config.apiUrl = "http://127.0.0.1:" + std::to_string(port);
    o.config.userToken = std::string(128, 'a');
    o.config.workerName = "testrig";
    o.config.targetPuzzle = 38;
    o.config.apiShare = true;
    o.config.apiShareUrl = o.config.apiUrl + "/webhook";
    o.backend = "cpu";
    o.maxRanges = 1;
    o.pingInterval = std::chrono::seconds(1);
    o.retryDelay = std::chrono::seconds(1);
    return o;
  }
};

struct TempDir {
  fs::path path = fs::temp_directory_path() / ("bsolver-pool-test-" + randomTokenHex(6));
  TempDir() { fs::create_directories(path); }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

const Catalog& catalog() {
  static Catalog c = Catalog::builtin();
  return c;
}
}  // namespace

TEST_CASE("pool session scans, finds the proof keys and target, and submits") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  MockPool mock("FFFFF", {7, 1000, 99999, 500000, 777777, 1048575}, 123456);
  TempDir dir;
  JobManager manager(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  REQUIRE(PoolSession(manager, mock.options()).run() == 0);

  std::lock_guard<std::mutex> lock(mock.mu);
  CHECK(mock.gets == 1);
  REQUIRE(mock.puts.size() == 1);
  const auto& put = mock.puts[0];
  CHECK(put.find("HEX")->second == "200");
  CHECK(put.find("HashedProofKey")->second == mock.expectedProofHash());
  CHECK(put.find("WorkerName")->second == "testrig");
  CHECK(put.find("GPUCount")->second == "1");
  REQUIRE(mock.keyFound.count("PrivateKey"));
  CHECK(mock.keyFound.find("PrivateKey")->second == upperHex64(mock.targetKey));
  CHECK(mock.keyFound.find("Encrypted")->second == "false");
  CHECK(mock.webhookStatuses.front() == "workerStarted");
  CHECK(mock.webhookStatuses.back() == "workerExited");

  // Only the target key is recorded as a result; proof keys are pool bookkeeping.
  auto results = readResults(dir.path / "found.jsonl");
  REQUIRE(results.size() == 1);
  CHECK(results[0].address == MockPool::address(mock.targetKey));
  auto state = PoolState::load(poolStatePath(dir.path / "state", 38));
  REQUIRE(state);
  CHECK_FALSE(state->assignment);
  CHECK(state->rangesSubmitted == 1);
}

TEST_CASE("pool session in untrusted mode never records the target key in plaintext") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  MockPool mock("FFFF", {7, 1000}, 4321);
  TempDir dir;
  JobManager manager(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  PoolSessionOptions o = mock.options();
  o.config.untrustedComputer = true;
  // Example key from the official client's README.
  o.config.setPublicKey(
      "-----BEGIN PUBLIC KEY-----|MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCZkY6nLcQf2OUGKk2qZO34dWhh|"
      "chPCe4htDumUDLo3+ZceZPBrFyKBeL7kuv+tp+sh5x78AhLuLGZjBdoV0Rzospqr|"
      "DXadt62qyNJpa7+ajtE0HRJmepis7QAurX5HcP4zszkzq2Y0u219whtwm/WDdE3p|5fuwuNwz8WkcQlXs9wIDAQAB|"
      "-----END PUBLIC KEY-----");
  REQUIRE(PoolSession(manager, o).run() == 0);

  const std::string plain = upperHex64(mock.targetKey);
  std::lock_guard<std::mutex> lock(mock.mu);
  CHECK(mock.keyFound.find("Encrypted")->second == "true");
  CHECK(mock.keyFound.find("PrivateKey")->second.size() == 172);  // base64 of a 1024-bit RSA block
  auto results = readResults(dir.path / "found.jsonl");
  REQUIRE(results.size() == 1);
  CHECK(results[0].privkey.rfind("rsa-oaep:", 0) == 0);
  CHECK(results[0].wif.empty());
  CHECK(results[0].privkey.find(plain) == std::string::npos);
}

TEST_CASE("pool session in untrusted mode without public_key keeps no local copy of the target key") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  MockPool mock("FFFF", {7, 1000}, 4321);
  TempDir dir;
  JobManager manager(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  PoolSessionOptions o = mock.options();
  o.config.untrustedComputer = true;  // api_share is the delivery channel
  REQUIRE(PoolSession(manager, o).run() == 0);

  std::lock_guard<std::mutex> lock(mock.mu);
  REQUIRE(mock.keyFound.count("PrivateKey"));
  CHECK(mock.keyFound.find("PrivateKey")->second == upperHex64(mock.targetKey));
  CHECK(mock.keyFound.find("Encrypted")->second == "false");
  CHECK(readResults(dir.path / "found.jsonl").empty());
  const std::string lowerKey = mock.targetKey.toHex(true), upperKey = upperHex64(mock.targetKey);
  for (const auto& e : fs::recursive_directory_iterator(dir.path)) {
    if (!e.is_regular_file()) continue;
    std::ifstream in(e.path(), std::ios::binary);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    INFO(e.path().string());
    CHECK(content.find(lowerKey) == std::string::npos);
    CHECK(content.find(upperKey) == std::string::npos);
  }
}

TEST_CASE("pool session drops a range with missing proof keys and fetches a new one") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  MockPool mock("FFFF", {7, 1000}, 4321);
  mock.unfindableGets = 1;
  TempDir dir;
  JobManager manager(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  REQUIRE(PoolSession(manager, mock.options()).run() == 0);

  std::lock_guard<std::mutex> lock(mock.mu);
  CHECK(mock.gets == 2);
  REQUIRE(mock.puts.size() == 1);
  CHECK(mock.puts[0].find("HashedProofKey")->second == mock.expectedProofHash());
  CHECK(fs::exists(dir.path / "state" / "pool-38-incomplete-200.txt"));
  auto state = PoolState::load(poolStatePath(dir.path / "state", 38));
  REQUIRE(state);
  CHECK_FALSE(state->assignment);
  CHECK(state->rangesSubmitted == 1);
}

TEST_CASE("pool session rejects an out-of-range assignment before persisting it") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  MockPool mock("FFFF", {7}, 4321);
  mock.hex = "0";  // range starts at key 0
  TempDir dir;
  JobManager manager(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  CHECK(PoolSession(manager, mock.options()).run() == 1);
  auto state = PoolState::load(poolStatePath(dir.path / "state", 38));
  REQUIRE(state);
  CHECK_FALSE(state->assignment);
  std::lock_guard<std::mutex> lock(mock.mu);
  CHECK(mock.gets == 1);
}

namespace {
PoolState savedState(const PoolRange& range) {
  PoolState s;
  s.puzzle = 38;
  s.tokenId = poolTokenId(std::string(128, 'a'));
  s.workerName = "testrig";
  s.assignment = PoolAssignment{range, {}, false, "2026-01-01T00:00:00Z"};
  return s;
}
}  // namespace

TEST_CASE("pool session drops a saved range outside the key space") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  MockPool mock("FFFF", {7, 1000}, 4321);
  TempDir dir;
  json bad = mock.rangeJson();
  bad["hex"] = "0";
  savedState(parseRangeResponse(bad.dump(), false)).save(poolStatePath(dir.path / "state", 38));
  JobManager manager(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  REQUIRE(PoolSession(manager, mock.options()).run() == 0);
  std::lock_guard<std::mutex> lock(mock.mu);
  CHECK(mock.gets == 1);
  REQUIRE(mock.puts.size() == 1);
  CHECK(mock.puts[0].find("HEX")->second == "200");
}

TEST_CASE("pool session keeps a saved range when a local setting is rejected") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  MockPool mock("FFFF", {7, 1000}, 4321);
  TempDir dir;
  const fs::path statePath = poolStatePath(dir.path / "state", 38);
  savedState(parseRangeResponse(mock.rangeJson().dump())).save(statePath);
  JobManager manager(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  PoolSessionOptions o = mock.options();
  o.threads = kMaxThreads + 1;  // rejected by the engine: fatal, and not the saved range's fault
  CHECK(PoolSession(manager, o).run() == 1);
  auto state = PoolState::load(statePath);
  REQUIRE(state);
  REQUIRE(state->assignment);
  CHECK(state->assignment->range.hex == "200");
  std::lock_guard<std::mutex> lock(mock.mu);
  CHECK(mock.gets == 0);
}

TEST_CASE("pool session rescans a saved range whose checkpoint is corrupt") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  MockPool mock("FFFF", {7, 1000}, 4321);
  TempDir dir;
  const fs::path stateDir = dir.path / "state";
  const PoolRange range = parseRangeResponse(mock.rangeJson().dump());
  savedState(range).save(poolStatePath(stateDir, 38));
  // The checkpoint the session's scan job resumes from (same spec fields that determine the identity).
  JobSpec spec;
  spec.addresses = range.proofAddresses;
  spec.addresses.push_back(range.targetAddress);
  spec.rangeStart = range.start().toHex();
  spec.rangeEnd = range.end().toHex();
  spec.algorithm = Algorithm::Scan;
  const fs::path ckpt = Checkpointer(stateDir, Algo::Scan, resolve(spec, catalog()).identity, true, false).path();
  {
    std::ofstream out(ckpt, std::ios::binary);
    out << "BS";  // torn write
  }
  JobManager manager(catalog(), ManagerOptions{stateDir, dir.path / "found.jsonl"});
  REQUIRE(PoolSession(manager, mock.options()).run() == 0);
  CHECK_FALSE(manager.status().resumed);
  CHECK_FALSE(fs::exists(ckpt));
  std::lock_guard<std::mutex> lock(mock.mu);
  CHECK(mock.gets == 0);
  REQUIRE(mock.puts.size() == 1);
  CHECK(mock.puts[0].find("HashedProofKey")->second == mock.expectedProofHash());
}

TEST_CASE("pool session pings a scanned range while its submission is retried") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  MockPool mock("FFFF", {7, 1000}, 4321);
  mock.failPuts = 4;
  TempDir dir;
  PoolState s = savedState(parseRangeResponse(mock.rangeJson().dump()));
  for (const auto& k : mock.proofKeys) s.assignment->proofKeys[MockPool::address(k)] = upperHex64(k);
  s.assignment->scanComplete = true;
  s.save(poolStatePath(dir.path / "state", 38));
  JobManager manager(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  REQUIRE(PoolSession(manager, mock.options()).run() == 0);

  std::lock_guard<std::mutex> lock(mock.mu);
  CHECK(mock.gets == 0);
  REQUIRE(mock.puts.size() == 1);
  CHECK(mock.puts[0].find("HashedProofKey")->second == mock.expectedProofHash());
  CHECK(mock.pingsDuringFailedPuts >= 1);
}

TEST_CASE("pool session records keys that do not belong to its assignment like a normal job") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  MockPool mock("FFFF", {7, 1000}, 4321);
  mock.failPuts = 1000;  // hold the session in its submission retry loop until the foreign job has finished
  TempDir dir;
  PoolState s = savedState(parseRangeResponse(mock.rangeJson().dump()));
  for (const auto& k : mock.proofKeys) s.assignment->proofKeys[MockPool::address(k)] = upperHex64(k);
  s.assignment->scanComplete = true;
  s.save(poolStatePath(dir.path / "state", 38));
  const fs::path results = dir.path / "found.jsonl";
  JobManager manager(catalog(), ManagerOptions{dir.path / "state", results});
  PoolSession session(manager, mock.options());
  std::atomic<int> rc{-1};
  std::thread t([&] { rc = session.run(); });
  auto failedPuts = [&] {
    std::lock_guard<std::mutex> lock(mock.mu);
    return mock.failedPuts;
  };
  // The first failed submission comes after the session installed its found handler.
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (failedPuts() == 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  const bool submitting = failedPuts() > 0;
  if (submitting) {
    JobSpec spec;
    spec.puzzle = 20;
    spec.backend = "cpu";
    spec.checkpoint = false;
    manager.start(spec);
    manager.wait();
  }
  {
    std::lock_guard<std::mutex> lock(mock.mu);
    mock.failPuts = 0;
  }
  t.join();
  REQUIRE(submitting);
  CHECK(rc == 0);
  auto found = readResults(results);
  REQUIRE(found.size() == 1);
  CHECK(found[0].puzzle == 20);
  CHECK(found[0].privkey.ends_with("d2c55"));
  std::lock_guard<std::mutex> lock(mock.mu);
  CHECK(mock.puts.size() == 1);
}

TEST_CASE("pool session resumes an interrupted range") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  // 2^25 keys: long enough to interrupt; three proof keys near the start, three at the end.
  MockPool mock("1FFFFFF", {100, 2000, 30000, 0x1FFFFFE, 0x1FFFF00, 0x1000000}, 0x1FFFFFF);
  TempDir dir;
  const ManagerOptions mopts{dir.path / "state", dir.path / "found.jsonl"};
  const fs::path statePath = poolStatePath(mopts.stateDir, 38);
  {
    JobManager manager(catalog(), mopts);
    PoolSession session(manager, mock.options());
    std::atomic<int> rc{-1};
    std::thread t([&] { rc = session.run(); });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      auto s = PoolState::load(statePath);
      if ((s && s->assignment && s->assignment->proofKeys.size() >= 3) || std::chrono::steady_clock::now() > deadline) {
        break;
      }
    }
    session.requestStop();
    t.join();
    REQUIRE(rc == 130);
  }
  auto saved = PoolState::load(statePath);
  REQUIRE(saved);
  REQUIRE(saved->assignment);
  CHECK(saved->assignment->proofKeys.size() >= 3);
  CHECK_FALSE(saved->assignment->scanComplete);
  {
    std::lock_guard<std::mutex> lock(mock.mu);
    REQUIRE(mock.puts.empty());
  }

  JobManager manager(catalog(), mopts);
  REQUIRE(PoolSession(manager, mock.options()).run() == 0);
  CHECK(manager.status().resumed);
  std::lock_guard<std::mutex> lock(mock.mu);
  CHECK(mock.gets == 1);  // the saved range was resumed, not replaced
  CHECK(mock.pings >= 1);
  REQUIRE(mock.puts.size() == 1);
  CHECK(mock.puts[0].find("HashedProofKey")->second == mock.expectedProofHash());
}

TEST_CASE("pool session stops on invalid tokens") {
  if (!httpsSupported()) SKIP("built without an HTTP client");
  MockPool mock("FF", {1}, 2);
  TempDir dir;
  JobManager manager(catalog(), ManagerOptions{dir.path / "state", dir.path / "found.jsonl"});
  PoolSessionOptions o = mock.options();
  o.config.userToken = "x";  // wrong length: rejected locally
  CHECK(PoolSession(manager, o).run() == 1);
  o.config.userToken = std::string(128, 'b');  // the mock answers 400 (validation error): not retried
  CHECK(PoolSession(manager, o).run() == 1);
  std::lock_guard<std::mutex> lock(mock.mu);
  CHECK(mock.gets == 1);
}

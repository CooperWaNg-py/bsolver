#include <catch2/catch_test_macros.hpp>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <thread>

#include "catalog/catalog.h"
#include "core/random.h"
#include "engine/job_manager.h"
#include "server/http_server.h"

using namespace bsolver;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {
struct Fixture {
  fs::path dir = fs::temp_directory_path() / ("bsolver-server-test-" + randomTokenHex(6));
  Catalog catalog = Catalog::builtin();
  JobManager manager{catalog, ManagerOptions{dir / "state", dir / "found.jsonl"}};
  HttpServer server{manager, catalog, ServerOptions{"127.0.0.1", 0, "t", true}};
  int port = server.start();
  httplib::Client client{"127.0.0.1", port};
  httplib::Headers auth{{"Authorization", "Bearer t"}};

  ~Fixture() {
    try {
      manager.stop();
    } catch (...) {
    }
    manager.wait();
    server.stop();
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};
}  // namespace

TEST_CASE("API requires the bearer token") {
  Fixture f;
  auto r = f.client.Get("/api/info");
  REQUIRE(r);
  CHECK(r->status == 401);
  CHECK(json::parse(r->body)["error"] == "unauthorized");
  r = f.client.Get("/api/info", httplib::Headers{{"Authorization", "Bearer wrong"}});
  REQUIRE(r);
  CHECK(r->status == 401);
  r = f.client.Get("/api/info", f.auth);
  REQUIRE(r);
  CHECK(r->status == 200);
  auto info = json::parse(r->body);
  CHECK(info["backends"][0]["id"] == "cpu");
}

TEST_CASE("API job lifecycle finds puzzle 20") {
  Fixture f;
  auto r = f.client.Post("/api/job", f.auth, R"({"puzzle":20,"backend":"cpu"})", "application/json");
  REQUIRE(r);
  REQUIRE(r->status == 202);
  json job;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  do {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto g = f.client.Get("/api/job", f.auth);
    REQUIRE(g);
    job = json::parse(g->body);
  } while (job["state"] != "finished" && job["state"] != "failed" && std::chrono::steady_clock::now() < deadline);
  REQUIRE(job["state"] == "finished");
  REQUIRE(job["found"].size() == 1);
  CHECK(job["found"][0]["privkey"].get<std::string>().ends_with("d2c55"));

  auto res = f.client.Get("/api/results", f.auth);
  REQUIRE(res);
  CHECK(json::parse(res->body).size() == 1);
}

TEST_CASE("API rejects a second job and bad specs") {
  Fixture f;
  const char* body = R"({"puzzle":60,"algorithm":"scan","backend":"cpu","checkpoint":false})";
  auto r1 = f.client.Post("/api/job", f.auth, body, "application/json");
  REQUIRE(r1);
  REQUIRE(r1->status == 202);
  auto r2 = f.client.Post("/api/job", f.auth, body, "application/json");
  REQUIRE(r2);
  CHECK(r2->status == 409);
  CHECK(json::parse(r2->body)["error"] == "a job is already active");

  auto resume = f.client.Post("/api/job/resume", f.auth, "", "application/json");
  REQUIRE(resume);
  CHECK(resume->status == 409);
  CHECK(json::parse(resume->body)["error"] == "job is not paused");
  auto stop = f.client.Post("/api/job/stop", f.auth, "", "application/json");
  REQUIRE(stop);
  CHECK(stop->status == 200);
  f.manager.wait();

  auto bad = f.client.Post("/api/job", f.auth, R"({"puzzle":1,"bogus":1})", "application/json");
  REQUIRE(bad);
  CHECK(bad->status == 400);
  CHECK(json::parse(bad->body)["error"] == "unknown field 'bogus'");
  bad = f.client.Post("/api/job", f.auth, "{not json", "application/json");
  REQUIRE(bad);
  CHECK(bad->status == 400);
  auto stopAgain = f.client.Post("/api/job/stop", f.auth, "", "application/json");
  REQUIRE(stopAgain);
  CHECK(stopAgain->status == 409);
  CHECK(json::parse(stopAgain->body)["error"] == "no active job");
}

TEST_CASE("event stream sends status first and accepts the query token") {
  Fixture f;
  std::string received;
  auto r = f.client.Get("/api/events?token=t", [&](const char* data, size_t len) {
    received.append(data, len);
    return received.find("\n\n") == std::string::npos;  // stop after the first event
  });
  CHECK(received.rfind("event: status\ndata: {", 0) == 0);
  auto denied = f.client.Get("/api/events?token=nope");
  REQUIRE(denied);
  CHECK(denied->status == 401);
}

TEST_CASE("a throwing subscriber does not take down the manager") {
  Fixture f;
  f.manager.subscribe([](const Event&) { throw std::runtime_error("subscriber failure"); });
  JobSpec spec;
  spec.puzzle = 20;  // solved: start() also publishes a warning log
  spec.backend = "cpu";
  spec.checkpoint = false;
  REQUIRE_NOTHROW(f.manager.start(spec));
  f.manager.wait();
  JobStatus s = f.manager.status();
  CHECK(s.state == JobState::Finished);
  CHECK(s.found.size() == 1);
}

TEST_CASE("event stream replaces invalid UTF-8 in log text") {
  Fixture f;
  f.client.set_read_timeout(std::chrono::seconds(5));
  const std::string expected = "bad byte \xEF\xBF\xBD here";  // U+FFFD replaces the invalid 0xFF
  std::string received;
  bool logged = false;
  f.client.Get("/api/events?token=t", [&](const char* data, size_t len) {
    received.append(data, len);
    if (!logged && received.find("\n\n") != std::string::npos) {  // subscribed once the first frame arrived
      logged = true;
      f.manager.log(LogLevel::Warn, "bad byte \xff here");
    }
    return received.find(expected) == std::string::npos;
  });
  CHECK(received.find("event: log\ndata: ") != std::string::npos);
  CHECK(received.find(expected) != std::string::npos);
}

TEST_CASE("job start and benchmarks fail fast while a benchmark runs") {
  Fixture f;
  httplib::Client benchClient{"127.0.0.1", f.port};
  benchClient.set_read_timeout(std::chrono::seconds(120));
  int benchStatus = 0;
  std::thread bench([&] {
    auto r = benchClient.Post("/api/bench", f.auth, R"({"algorithm":"scan","seconds":2})", "application/json");
    if (r) benchStatus = r->status;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  auto t0 = std::chrono::steady_clock::now();
  const char* job = R"({"puzzle":20,"backend":"cpu","checkpoint":false})";
  auto r = f.client.Post("/api/job", f.auth, job, "application/json");
  auto r2 = f.client.Post("/api/bench", f.auth, R"({"seconds":1})", "application/json");
  auto elapsed = std::chrono::steady_clock::now() - t0;
  bench.join();

  REQUIRE(r);
  CHECK(r->status == 409);
  CHECK(json::parse(r->body)["error"] == "a benchmark is running");
  REQUIRE(r2);
  CHECK(r2->status == 409);
  CHECK(json::parse(r2->body)["error"] == "a benchmark is running");
  CHECK(elapsed < std::chrono::seconds(1));
  CHECK(benchStatus == 200);

  auto after = f.client.Post("/api/job", f.auth, job, "application/json");
  REQUIRE(after);
  CHECK(after->status == 202);
}

TEST_CASE("SSE queue keeps the newest status and never evicts pinned frames") {
  SseQueue q(4);
  q.push("status", "S1;", false);
  q.push("found", "F;", true);
  q.push("status", "S2;", false);
  CHECK(q.size() == 2);
  for (int i = 0; i < 10; ++i) q.push("log", "L" + std::to_string(i) + ";", false);
  q.push("log", "E;", true);
  q.push("status", "S3;", false);
  // Oldest unpinned frames were evicted; the found and error frames survived and only the newest status remains.
  CHECK(q.drain() == "F;L9;E;S3;");
  CHECK(q.empty());

  SseQueue full(2);
  full.push("found", "F1;", true);
  full.push("log", "E1;", true);
  full.push("log", "I;", false);  // dropped: nothing unpinned to evict
  full.push("found", "F2;", true);
  CHECK(full.drain() == "F1;E1;F2;");
}

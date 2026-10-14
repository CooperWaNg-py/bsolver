#include <catch2/catch_test_macros.hpp>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
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

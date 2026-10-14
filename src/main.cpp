#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "backend/backend.h"
#include "backend/benchmark.h"
#include "catalog/catalog.h"
#include "core/random.h"
#include "engine/console.h"
#include "engine/errors.h"
#include "engine/format.h"
#include "engine/job_manager.h"
#include "server/http_server.h"

using namespace bsolver;

namespace {

volatile std::sig_atomic_t gSignal = 0;
extern "C" void onSignal(int sig) { gSignal = sig; }

void installSignalHandlers() {
  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);
}

Catalog loadCatalog(const std::string& path) { return path.empty() ? Catalog::builtin() : Catalog::fromFile(path); }

int cmdPuzzles(bool unsolvedOnly, std::optional<int> id, const std::string& catalogPath, bool json) {
  Catalog catalog = loadCatalog(catalogPath);
  std::vector<const Puzzle*> rows;
  for (const auto& p : catalog.all()) {
    if (unsolvedOnly && p.solved) continue;
    if (id && p.id != *id) continue;
    rows.push_back(&p);
  }
  if (id && rows.empty()) {
    std::cerr << "error: unknown puzzle " << *id << "\n";
    return 1;
  }
  if (json) {
    nlohmann::json arr = nlohmann::json::array();
    for (const Puzzle* p : rows) arr.push_back(puzzleToJson(*p));
    std::cout << arr.dump(2) << "\n";
    return 0;
  }
  std::printf("%-4s %-4s %-35s %-6s %-9s %s\n", "id", "bits", "address", "pubkey", "status", "key");
  for (const Puzzle* p : rows) {
    std::printf("%-4d %-4d %-35s %-6s %-9s %s\n", p->id, p->id, p->address.c_str(), p->pubkey ? "yes" : "no",
                p->solved ? "solved" : "unsolved", p->privkey ? p->privkey->toHex().c_str() : "-");
  }
  return 0;
}

int cmdDevices() {
  for (Backend* b : allBackends()) {
    auto devices = b->devices();
    std::printf("%s:", std::string(b->id()).c_str());
    if (devices.empty()) {
      std::printf(" unavailable (%s)\n", b->unavailableReason().c_str());
      continue;
    }
    std::printf("\n");
    for (const auto& d : devices) {
      std::printf("  [%d] %s", d.index, d.name.c_str());
      if (d.computeUnits) std::printf(", %u compute units", d.computeUnits);
      if (d.memoryBytes) std::printf(", %.1f GiB", static_cast<double>(d.memoryBytes) / (1024.0 * 1024 * 1024));
      std::printf("\n");
    }
  }
  return 0;
}

std::string joinDevices(const std::vector<std::string>& v) {
  std::string s;
  for (const auto& d : v) s += (s.empty() ? "" : ", ") + d;
  return s;
}

int cmdBench(const std::string& algoName, double seconds, unsigned threads) {
  Algo algo = algoName == "kangaroo" ? Algo::Kangaroo : Algo::Scan;
  const std::string unit = algo == Algo::Kangaroo ? "jumps/s" : "keys/s";
  std::fprintf(stderr, "benchmarking %s for %.1f s per backend...\n", algoName.c_str(), seconds);
  auto results = runBenchmark(algo, BenchOptions{seconds, threads, {}});
  std::printf("%-8s %-40s %-20s %-8s %s\n", "backend", "devices", "rate", "verified", "error");
  for (const auto& r : results) {
    std::printf("%-8s %-40s %-20s %-8s %s\n", r.backend.c_str(), joinDevices(r.devices).c_str(),
                r.verified ? formatRate(r.rate, unit).c_str() : "-", r.verified ? "yes" : "no", r.error.c_str());
  }
  return 0;
}

std::string progressLine(const JobStatus& s) {
  if (s.state == JobState::Benchmarking) return "[benchmarking] elapsed " + formatDuration(s.elapsedSeconds);
  std::string line = "[" + std::string(s.algorithm ? toString(*s.algorithm) : "?") + "/" + s.backend.value_or("?") +
                     "] " + formatRate(s.rate, s.rateUnit);
  char buf[64];
  if (s.progress) {
    std::snprintf(buf, sizeof(buf), " | %.2f%%", *s.progress * 100.0);
    line += buf;
  }
  if (s.workDone >= 1) {
    std::snprintf(buf, sizeof(buf), " | done 2^%.2f", std::log2(s.workDone));
    line += buf;
  } else {
    line += " | done 0";
  }
  line += " | elapsed " + formatDuration(s.elapsedSeconds);
  if (s.etaSeconds) {
    line += " | ETA " + formatDuration(*s.etaSeconds);
  } else if (s.workExpected && s.rate > 0 && isActive(s.state)) {
    line += " | expected " + formatDuration(*s.workExpected / s.rate);
  }
  if (s.state == JobState::Paused) line += " | paused";
  if (s.state == JobState::Stopping) line += " | stopping";
  return line;
}

struct RunOptions {
  std::optional<int> puzzle;
  std::vector<std::string> addresses;
  std::optional<std::string> pubkey;
  std::optional<std::string> range;
  std::optional<int> bits;
  std::string algo = "auto";
  std::string order = "sequential";
  std::string addrMode = "compressed";
  std::string backend = "auto";
  unsigned threads = 0;
  std::vector<int> devices;
  uint32_t bsgsMem = 1024;
  std::optional<uint32_t> dp;
  std::string stateDir = "bsolver-state";
  bool fresh = false;
  bool noCheckpoint = false;
  std::string results = "bsolver-found.jsonl";
  std::string catalog;
  std::optional<std::string> web;
  std::string token;
};

JobSpec buildSpec(const RunOptions& o) {
  JobSpec s;
  s.puzzle = o.puzzle;
  s.addresses = o.addresses;
  s.pubkey = o.pubkey;
  if (o.range) {
    auto colon = o.range->find(':');
    if (colon == std::string::npos) throw JobError("--range must be START:END (hex)");
    s.rangeStart = o.range->substr(0, colon);
    s.rangeEnd = o.range->substr(colon + 1);
  }
  s.bits = o.bits;
  nlohmann::json j = {{"algorithm", o.algo}, {"order", o.order}, {"addressMode", o.addrMode}};
  JobSpec enums = j.get<JobSpec>();
  s.algorithm = enums.algorithm;
  s.order = enums.order;
  s.addressMode = enums.addressMode;
  s.backend = o.backend;
  s.threads = o.threads;
  s.devices = o.devices;
  s.bsgsMemoryMb = o.bsgsMem;
  s.dpBits = o.dp;
  s.checkpoint = !o.noCheckpoint;
  s.fresh = o.fresh;
  return s;
}

struct Listen {
  std::string host;
  int port;
};

Listen parseListen(const std::string& s) {
  auto colon = s.rfind(':');
  if (colon == std::string::npos) throw JobError("listen address must be HOST:PORT");
  Listen l{s.substr(0, colon), 0};
  if (l.host.size() >= 2 && l.host.front() == '[' && l.host.back() == ']') l.host = l.host.substr(1, l.host.size() - 2);
  try {
    l.port = std::stoi(s.substr(colon + 1));
  } catch (const std::exception&) {
    throw JobError("invalid port in '" + s + "'");
  }
  if (l.port < 0 || l.port > 65535) throw JobError("invalid port in '" + s + "'");
  return l;
}

std::string resolveToken(const std::string& flag) {
  if (!flag.empty()) return flag;
  if (const char* env = std::getenv("BSOLVER_TOKEN"); env && *env) return env;
  return randomTokenHex(16);
}

std::unique_ptr<HttpServer> startWeb(JobManager& manager, const Catalog& catalog, const std::string& listen,
                                     const std::string& tokenFlag) {
  Listen l = parseListen(listen);
  std::string token = resolveToken(tokenFlag);
  auto server = std::make_unique<HttpServer>(manager, catalog, ServerOptions{l.host, l.port, token, true});
  int port = server->start();
  std::string host = l.host.find(':') != std::string::npos ? "[" + l.host + "]" : l.host;
  consoleOut("Web UI: http://" + host + ":" + std::to_string(port) + "/#token=" + token);
  return server;
}

int cmdServe(const std::string& listen, const std::string& token, const std::string& stateDir,
             const std::string& results, const std::string& catalogPath) {
  Catalog catalog = loadCatalog(catalogPath);
  JobManager manager(catalog, ManagerOptions{stateDir, results});
  auto server = startWeb(manager, catalog, listen, token);
  installSignalHandlers();
  while (!gSignal) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  consoleLog("[info] shutting down");
  try {
    manager.stop();
  } catch (const ConflictError&) {
  }
  manager.wait();
  server->stop();
  return 0;
}

int cmdRun(const RunOptions& o) {
  Catalog catalog = loadCatalog(o.catalog);
  JobManager manager(catalog, ManagerOptions{o.stateDir, o.results});
  JobSpec spec = buildSpec(o);
  std::unique_ptr<HttpServer> server;
  if (o.web) server = startWeb(manager, catalog, *o.web, o.token);
  installSignalHandlers();
  try {
    manager.start(spec);
  } catch (const JobError& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }

  bool interrupted = false;
  auto lastPrint = std::chrono::steady_clock::now();
  for (;;) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (gSignal && !interrupted) {
      interrupted = true;
      consoleLog("[info] interrupted; stopping and writing checkpoint");
      try {
        manager.stop();
      } catch (const ConflictError&) {
      }
    }
    JobStatus s = manager.status();
    if (!isActive(s.state)) break;
    auto now = std::chrono::steady_clock::now();
    if (now - lastPrint >= std::chrono::seconds(1)) {
      lastPrint = now;
      consoleProgress(progressLine(s));
    }
  }
  manager.wait();
  JobStatus s = manager.status();
  consoleProgress(progressLine(s));
  consoleFinishProgress();
  if (s.state == JobState::Failed) return 1;
  switch (s.outcome.value_or(Outcome::Error)) {
    case Outcome::Found: return 0;
    case Outcome::Exhausted:
      consoleLog("[info] range exhausted without finding the key");
      return 2;
    case Outcome::Stopped: return 130;
    case Outcome::Error: return 1;
  }
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  HANDLE err = GetStdHandle(STD_ERROR_HANDLE);
  DWORD mode = 0;
  if (GetConsoleMode(err, &mode)) SetConsoleMode(err, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif

  CLI::App app{"bsolver - Bitcoin puzzle transaction solver", "bsolver"};
  app.set_version_flag("--version", std::string("bsolver ") + BSOLVER_VERSION);
  app.require_subcommand(0, 1);

  // puzzles
  auto* puzzles = app.add_subcommand("puzzles", "List the built-in puzzle catalog");
  bool unsolvedOnly = false, puzzlesJson = false;
  std::optional<int> puzzleId;
  std::string puzzlesCatalog;
  puzzles->add_flag("--unsolved", unsolvedOnly, "Only unsolved puzzles");
  puzzles->add_option("--id", puzzleId, "Single puzzle id");
  puzzles->add_option("--catalog", puzzlesCatalog, "Catalog JSON file (overrides the built-in snapshot)");
  puzzles->add_flag("--json", puzzlesJson, "JSON output");

  // run
  auto* run = app.add_subcommand("run", "Search for a private key");
  RunOptions ro;
  run->add_option("--puzzle", ro.puzzle, "Puzzle number (1-160)");
  run->add_option("--address", ro.addresses, "Target P2PKH address (repeatable)");
  run->add_option("--pubkey", ro.pubkey, "Target public key (hex, 33 or 65 bytes)");
  run->add_option("--range", ro.range, "Key range START:END (hex)");
  run->add_option("--bits", ro.bits, "Key range [2^(N-1), 2^N - 1]");
  run->add_option("--algo", ro.algo, "Algorithm")->check(CLI::IsMember({"auto", "scan", "bsgs", "kangaroo"}));
  run->add_option("--order", ro.order, "Scan order")->check(CLI::IsMember({"sequential", "random"}));
  run->add_option("--addr-mode", ro.addrMode, "Address types to hash (scan)")
      ->check(CLI::IsMember({"compressed", "uncompressed", "both"}));
  run->add_option("--backend", ro.backend, "Compute backend")
      ->check(CLI::IsMember({"auto", "cpu", "opencl", "metal", "cuda"}));
  run->add_option("--threads", ro.threads, "CPU threads (0 = all)");
  run->add_option("--devices", ro.devices, "GPU device indices (e.g. 0,1)")->delimiter(',');
  run->add_option("--bsgs-mem", ro.bsgsMem, "BSGS baby-step table memory (MB)");
  run->add_option("--dp", ro.dp, "Kangaroo distinguished-point bits");
  run->add_option("--state-dir", ro.stateDir, "Checkpoint directory");
  run->add_flag("--fresh", ro.fresh, "Ignore and overwrite an existing checkpoint");
  run->add_flag("--no-checkpoint", ro.noCheckpoint, "Do not read or write checkpoints");
  run->add_option("--results", ro.results, "Found-keys file (JSON lines)");
  run->add_option("--catalog", ro.catalog, "Catalog JSON file (overrides the built-in snapshot)");
  run->add_option("--web", ro.web, "Also serve the web UI/API on HOST:PORT");
  run->add_option("--token", ro.token, "Web API bearer token (default: $BSOLVER_TOKEN or random)");

  // serve
  auto* serve = app.add_subcommand("serve", "Serve the web UI and API (idle until a job is started)");
  std::string serveListen = "127.0.0.1:8080", serveToken, serveStateDir = "bsolver-state",
              serveResults = "bsolver-found.jsonl", serveCatalog;
  serve->add_option("--listen", serveListen, "HOST:PORT");
  serve->add_option("--token", serveToken, "Bearer token (default: $BSOLVER_TOKEN or random)");
  serve->add_option("--state-dir", serveStateDir, "Checkpoint directory");
  serve->add_option("--results", serveResults, "Found-keys file (JSON lines)");
  serve->add_option("--catalog", serveCatalog, "Catalog JSON file (overrides the built-in snapshot)");

  // devices
  auto* devices = app.add_subcommand("devices", "List compute backends and devices");

  // bench
  auto* bench = app.add_subcommand("bench", "Benchmark every available backend");
  std::string benchAlgo = "scan";
  double benchSeconds = 2.0;
  unsigned benchThreads = 0;
  bench->add_option("--algo", benchAlgo, "Algorithm")->check(CLI::IsMember({"scan", "kangaroo"}));
  bench->add_option("--seconds", benchSeconds, "Seconds per backend");
  bench->add_option("--threads", benchThreads, "CPU threads (0 = all)");

  CLI11_PARSE(app, argc, argv);
  try {
    if (*puzzles) return cmdPuzzles(unsolvedOnly, puzzleId, puzzlesCatalog, puzzlesJson);
    if (*run) return cmdRun(ro);
    if (*devices) return cmdDevices();
    if (*bench) return cmdBench(benchAlgo, benchSeconds, benchThreads);
    if (*serve) return cmdServe(serveListen, serveToken, serveStateDir, serveResults, serveCatalog);
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
  std::cout << app.help();
  return 0;
}

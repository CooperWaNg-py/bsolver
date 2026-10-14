#pragma once

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "backend/benchmark.h"
#include "catalog/catalog.h"
#include "engine/checkpoint.h"
#include "engine/job_spec.h"
#include "engine/job_status.h"
#include "engine/run_context.h"

namespace bsolver {

struct ManagerOptions {
  std::filesystem::path stateDir = "bsolver-state";
  std::filesystem::path resultsPath = "bsolver-found.jsonl";
};

struct Event {
  std::string type;  // "status" | "found" | "log"
  nlohmann::json payload;
};

// Owns at most one active job; shared by the CLI and the HTTP API.
class JobManager {
 public:
  JobManager(const Catalog& catalog, ManagerOptions options);
  ~JobManager();
  JobManager(const JobManager&) = delete;
  JobManager& operator=(const JobManager&) = delete;

  // Throws JobError (invalid spec) or ConflictError("a job is already active" / "a benchmark is running")
  // without waiting for a running benchmark.
  JobStatus start(const JobSpec& spec);
  // Throw ConflictError: "no active job" / "job is not running" / "job is not paused".
  JobStatus stop();
  JobStatus pause();
  JobStatus resume();
  JobStatus status() const;
  // Blocking; throws ConflictError immediately while a job or another benchmark is active.
  std::vector<BenchResult> bench(Algo algo, double seconds, unsigned threads = 0);

  // Callbacks run on manager threads and must not call subscribe/unsubscribe. Exceptions they throw are swallowed.
  int subscribe(std::function<void(const Event&)> fn);
  void unsubscribe(int id);
  // Blocks until no job is active.
  void wait();

  // Replaces the default found-key handling (results file + "FOUND" line) for every later key.
  // Called serialized on a worker thread; keep it short. The "found" event is still published.
  void setFoundHandler(std::function<void(const FoundKey&)> fn);

  void log(LogLevel level, const std::string& msg);
  const ManagerOptions& options() const { return options_; }
  const Catalog& catalog() const { return catalog_; }

 private:
  struct ActiveJob;

  void runJob(std::shared_ptr<ActiveJob> job);
  void tickLoop(std::shared_ptr<ActiveJob> job);
  void onFound(const FoundKey& key);
  void publish(const Event& e);
  JobStatus statusLocked() const;
  void throwIfBusyLocked() const;  // ConflictError when a job is active/starting or a benchmark runs
  void joinThreads();

  const Catalog& catalog_;
  ManagerOptions options_;

  mutable std::mutex mu_;
  std::condition_variable doneCv_;
  JobStatus st_;
  std::shared_ptr<ActiveJob> job_;
  std::thread jobThread_, tickThread_;
  // start() and bench() claim these under mu_ and fail fast on conflict; thread joins happen outside mu_.
  bool starting_ = false;
  bool benchRunning_ = false;
  std::mutex resultsMu_;
  std::function<void(const FoundKey&)> foundHandler_;  // guarded by resultsMu_

  std::mutex subMu_;
  std::map<int, std::function<void(const Event&)>> subs_;
  int nextSub_ = 1;
};

}  // namespace bsolver

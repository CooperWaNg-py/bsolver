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

  // Throws JobError (invalid spec) or ConflictError("a job is already active").
  JobStatus start(const JobSpec& spec);
  // Throw ConflictError: "no active job" / "job is not running" / "job is not paused".
  JobStatus stop();
  JobStatus pause();
  JobStatus resume();
  JobStatus status() const;
  // Blocking; throws ConflictError("a job is already active") while a job is active.
  std::vector<BenchResult> bench(Algo algo, double seconds, unsigned threads = 0);

  // Callbacks run on manager threads and must not call subscribe/unsubscribe.
  int subscribe(std::function<void(const Event&)> fn);
  void unsubscribe(int id);
  // Blocks until no job is active.
  void wait();

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
  void joinThreads();

  const Catalog& catalog_;
  ManagerOptions options_;

  mutable std::mutex mu_;
  std::condition_variable doneCv_;
  JobStatus st_;
  std::shared_ptr<ActiveJob> job_;
  std::thread jobThread_, tickThread_;
  std::mutex controlMu_;  // serializes start()/bench() (thread joins happen outside mu_)
  bool benchRunning_ = false;
  std::mutex resultsMu_;

  std::mutex subMu_;
  std::map<int, std::function<void(const Event&)>> subs_;
  int nextSub_ = 1;
};

}  // namespace bsolver

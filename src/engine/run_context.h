#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "core/point.h"
#include "core/u256.h"
#include "engine/checkpoint.h"
#include "engine/job_spec.h"
#include "engine/results.h"
#include "engine/target_set.h"

namespace bsolver {

enum class LogLevel { Info, Warn, Error };
std::string_view toString(LogLevel l);

// Shared state between the JobManager and a running algorithm.
class RunContext {
 public:
  using LogFn = std::function<void(LogLevel, const std::string&)>;
  using FoundFn = std::function<void(const FoundKey&)>;

  RunContext(const ResolvedJob& job, const JobSpec& spec, Checkpointer& checkpointer, LogFn log, FoundFn found);

  const ResolvedJob& job() const { return job_; }
  const JobSpec& spec() const { return spec_; }
  const TargetSet& targets() const { return targets_; }
  Checkpointer& checkpointer() { return checkpointer_; }
  void log(LogLevel level, const std::string& msg) const { log_(level, msg); }

  void setBackend(std::string id) { backend_ = std::move(id); }
  const std::string& backend() const { return backend_; }

  // Control.
  void requestStop();
  bool stopRequested() const { return stop_.load(std::memory_order_relaxed); }
  void setPaused(bool paused);
  bool paused() const { return paused_.load(std::memory_order_relaxed); }
  // Blocks while paused (returns immediately once a stop is requested).
  void waitIfPaused();
  // stopRequested() or (stopOnFound and every target found).
  bool shouldStop() const { return stopRequested() || (spec_.stopOnFound && allFound()); }

  // Progress.
  void addWork(double w);
  // Work restored from a checkpoint: counted in workDone() but not in the live rate.
  void addResumedWork(double w);
  double workDone() const { return work_.load(std::memory_order_relaxed); }
  double liveWork() const { return work_.load(std::memory_order_relaxed) - resumedWork_.load(); }
  void setResumed(bool r) { resumed_ = r; }
  bool resumed() const { return resumed_.load(); }
  void setWorkExpected(double w) { workExpected_ = w; }
  double workExpected() const { return workExpected_.load(); }

  // The single verification gate: recomputes key*G, requires it to equal pub and that its
  // hash160 (per address mode) is a target or that it equals the target pubkey.
  // Returns false (logging "rejected unverified candidate <hex>") otherwise. Thread-safe.
  bool reportFound(const U256& key, const Affine& pub);
  bool anyFound() const { return foundCount_.load() > 0; }
  bool allFound() const;
  std::vector<FoundKey> found() const;

 private:
  const ResolvedJob& job_;
  const JobSpec& spec_;
  TargetSet targets_;
  Checkpointer& checkpointer_;
  LogFn log_;
  FoundFn found_;
  std::string backend_;

  std::atomic<bool> stop_{false}, paused_{false}, resumed_{false};
  std::mutex pauseMu_;
  std::condition_variable pauseCv_;

  std::atomic<double> work_{0}, resumedWork_{0}, workExpected_{0};

  mutable std::mutex foundMu_;
  std::set<U256> foundKeys_;
  std::set<std::array<uint8_t, 20>> foundHashes_;
  std::vector<FoundKey> foundList_;
  std::atomic<size_t> foundCount_{0};
};

// Runs body(i) for i in [0, n) on n threads and joins them all. If a thread cannot be started or a body throws,
// requests a stop so the other bodies wind down, joins every started thread and rethrows the first exception.
void runThreads(RunContext& ctx, size_t n, const std::function<void(size_t)>& body);

}  // namespace bsolver

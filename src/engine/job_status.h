#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "backend/benchmark.h"
#include "engine/job_spec.h"
#include "engine/results.h"
#include "engine/types.h"

namespace bsolver {

enum class JobState { Idle, Benchmarking, Running, Paused, Stopping, Finished, Failed };
enum class Outcome { Found, Exhausted, Stopped, Error };

std::string_view toString(JobState s);
std::string_view toString(Outcome o);

inline bool isActive(JobState s) {
  return s == JobState::Benchmarking || s == JobState::Running || s == JobState::Paused || s == JobState::Stopping;
}

struct JobStatus {
  uint64_t id = 0;
  JobState state = JobState::Idle;
  std::optional<Outcome> outcome;
  std::optional<std::string> error;
  std::optional<JobSpec> spec;
  std::optional<int> puzzle;
  std::optional<Algo> algorithm;
  std::optional<Order> order;  // scan only
  std::optional<std::string> backend;
  std::vector<std::string> devices;
  std::optional<std::string> rangeStart, rangeEnd;
  std::optional<double> rangeBits;
  double rate = 0;
  std::string rateUnit = "keys/s";
  double workDone = 0;
  std::optional<double> workExpected;
  std::optional<double> progress;
  std::optional<double> etaSeconds;
  double elapsedSeconds = 0;
  std::optional<std::string> startedAt;
  bool resumed = false;
  bool exhausted = false;  // the whole range was covered (scan/bsgs)
  std::vector<FoundKey> found;
  std::vector<BenchResult> benchmark;
};

void to_json(nlohmann::json& j, const JobStatus& s);

}  // namespace bsolver

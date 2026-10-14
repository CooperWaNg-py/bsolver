#include "engine/job_status.h"

#include <nlohmann/json.hpp>

namespace bsolver {

std::string_view toString(JobState s) {
  switch (s) {
    case JobState::Idle: return "idle";
    case JobState::Benchmarking: return "benchmarking";
    case JobState::Running: return "running";
    case JobState::Paused: return "paused";
    case JobState::Stopping: return "stopping";
    case JobState::Finished: return "finished";
    case JobState::Failed: return "failed";
  }
  return "idle";
}

std::string_view toString(Outcome o) {
  switch (o) {
    case Outcome::Found: return "found";
    case Outcome::Exhausted: return "exhausted";
    case Outcome::Stopped: return "stopped";
    case Outcome::Error: return "error";
  }
  return "error";
}

void to_json(nlohmann::json& j, const BenchResult& r) {
  j = nlohmann::json{{"backend", r.backend}, {"devices", r.devices}, {"algorithm", toString(r.algo)},
                     {"rate", r.rate},       {"verified", r.verified}, {"error", r.error}};
}

void to_json(nlohmann::json& j, const JobStatus& s) {
  auto opt = [](const auto& o) { return o ? nlohmann::json(*o) : nlohmann::json(nullptr); };
  auto optStr = [](const auto& o) { return o ? nlohmann::json(std::string(toString(*o))) : nlohmann::json(nullptr); };
  j = nlohmann::json{
      {"id", s.id},
      {"state", toString(s.state)},
      {"outcome", optStr(s.outcome)},
      {"error", opt(s.error)},
      {"spec", opt(s.spec)},
      {"puzzle", opt(s.puzzle)},
      {"algorithm", optStr(s.algorithm)},
      {"order", optStr(s.order)},
      {"backend", opt(s.backend)},
      {"devices", s.devices},
      {"rangeStart", opt(s.rangeStart)},
      {"rangeEnd", opt(s.rangeEnd)},
      {"rangeBits", opt(s.rangeBits)},
      {"rate", s.rate},
      {"rateUnit", s.rateUnit},
      {"workDone", s.workDone},
      {"workExpected", opt(s.workExpected)},
      {"progress", opt(s.progress)},
      {"etaSeconds", opt(s.etaSeconds)},
      {"elapsedSeconds", s.elapsedSeconds},
      {"startedAt", opt(s.startedAt)},
      {"resumed", s.resumed},
      {"exhausted", s.exhausted},
      {"found", s.found},
      {"benchmark", s.benchmark},
  };
}

}  // namespace bsolver

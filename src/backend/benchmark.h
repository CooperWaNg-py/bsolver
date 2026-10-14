#pragma once

#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "engine/types.h"

namespace bsolver {

struct BenchResult {
  std::string backend;
  std::vector<std::string> devices;
  Algo algo = Algo::Scan;
  double rate = 0;  // keys/s (scan) or jumps/s (kangaroo)
  bool verified = false;
  std::string error;
};

void to_json(nlohmann::json& j, const BenchResult& r);

struct BenchOptions {
  double seconds = 2.0;
  unsigned threads = 0;
  std::vector<int> devices;
};

// Correctness check plus throughput run on every compiled-in backend that has devices and supports
// `algo` (scan or kangaroo). Backends failing the check have verified=false and an error.
std::vector<BenchResult> runBenchmark(Algo algo, const BenchOptions& options);

}  // namespace bsolver

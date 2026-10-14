#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/point.h"
#include "core/u256.h"
#include "engine/types.h"

namespace bsolver {

struct DeviceInfo {
  int index = 0;
  std::string name;
  uint64_t memoryBytes = 0;
  uint32_t computeUnits = 0;
};

struct ScanSetup {
  std::vector<std::array<uint8_t, 20>> targets;  // sorted
  AddressMode mode = AddressMode::Compressed;
};

class ScanWorker {
 public:
  virtual ~ScanWorker() = default;
  // Keys per process() call.
  virtual uint64_t segmentKeys() const = 0;
  // Scans keys [start, start+count), count <= segmentKeys(), start >= 2^20 for GPU workers.
  // Appends candidate keys whose hash160 (per address mode) is in the target set.
  virtual void process(const U256& start, uint64_t count, std::vector<U256>& candidates) = 0;
  virtual std::string label() const = 0;
};

struct KangarooParams {
  U256 width;
  Affine wildBase;  // Q' = Q - a*G
  uint32_t dpBits = 0;
  std::array<U256, 64> jumpDist;
  std::array<Affine, 64> jumpPoint;
  // Herd layout across all workers of a job, set per worker before init(). Global kangaroo index
  // g = firstKangaroo + i (firstKangaroo even); even g is tame, odd g is wild; ordinal k = g / 2.
  // Start distances: tame b_t + k*(W/(nK/2)), wild b_w + k*((W/2)/(nK/2)), with nK = totalKangaroos and
  // b_t in [1, spacing], b_w in [0, spacing) drawn from the init seed (see kangarooStarts() in
  // engine/kangaroo_params.h). totalKangaroos == 0 means this worker alone.
  uint64_t totalKangaroos = 0;
  uint64_t firstKangaroo = 0;
};

struct Dp {
  U256 x;
  U256 dist;
  bool wild = false;
  uint32_t index = 0;
};

class KangarooWorker {
 public:
  virtual ~KangarooWorker() = default;
  virtual size_t kangarooCount() const = 0;
  virtual void init(const KangarooParams& params, uint64_t seed) = 0;
  // Advances every kangaroo; appends distinguished points. Returns the number of jumps performed.
  virtual uint64_t step(std::vector<Dp>& out) = 0;
  // Restarts kangaroo `index` from a fresh random distance.
  virtual void reset(uint32_t index, uint64_t seed) = 0;
  // Current (point, dist) of the first n kangaroos; even index = tame, odd = wild.
  virtual std::vector<std::pair<Affine, U256>> sample(size_t n) = 0;
  virtual std::string label() const = 0;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual std::string_view id() const = 0;  // "cpu" | "opencl" | "metal" | "cuda"
  virtual std::vector<DeviceInfo> devices() = 0;  // empty => unavailable
  virtual std::string unavailableReason() const = 0;
  virtual bool supports(Algo algo) const = 0;  // cpu: all; GPU: Scan, Kangaroo
  // CPU: `threads` workers (0 = hardware concurrency). GPU: one worker per device (empty = all).
  virtual std::vector<std::unique_ptr<ScanWorker>> createScanWorkers(const ScanSetup& setup,
                                                                     const std::vector<int>& devices,
                                                                     unsigned threads) = 0;
  virtual std::vector<std::unique_ptr<KangarooWorker>> createKangarooWorkers(const std::vector<int>& devices,
                                                                             unsigned threads) = 0;
};

// Compiled-in backends in the order cpu, cuda, opencl, metal.
std::vector<Backend*> allBackends();
Backend* findBackend(std::string_view id);

// Resolves 0 to std::thread::hardware_concurrency() (at least 1).
unsigned effectiveThreads(unsigned threads);

}  // namespace bsolver

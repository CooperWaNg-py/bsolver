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

inline constexpr size_t kKangarooJumps = 512;        // main jump table, index x.limb0 & 511
inline constexpr size_t kKangarooEscapeJumps = 64;   // look-ahead and cycle tables, index x.limb0 & 63
inline constexpr uint32_t kKangarooCycleWindow = 64;  // fruitless-cycle reference refresh period (steps)

struct JumpTable {
  std::vector<U256> dist;     // even distances
  std::vector<Affine> point;  // point[j] = dist[j]*G
};

// Symmetric (negation-map) kangaroo over K' = (key - rangeStart) + shift in [shift, W + shift), shift = max(1, W >> 5).
// Tame points are t*G (t in [1, shift) at start), wild points wildBase + d*G = (d - K')*G (d even in [0, 34*shift)).
// A step from P = (x, y) adds -D when y is odd and +D otherwise (D from `lookahead` right after a 2-cycle look-ahead
// hit, else from `jumps`), so the walk is a function of the class {P, -P}. See kangarooStep() in
// engine/kangaroo_params.h, the reference every backend matches.
struct KangarooParams {
  U256 width;
  U256 shift;
  Affine wildBase;  // -(Q - rangeStart*G + shift*G)
  uint32_t dpBits = 0;
  JumpTable jumps;      // kKangarooJumps entries
  JumpTable lookahead;  // kKangarooEscapeJumps entries
  JumpTable cycle;      // kKangarooEscapeJumps entries (deterministic fruitless-cycle escape, host side)
  // Herd layout across all workers of a job, set per worker before init(): global kangaroo index
  // g = firstKangaroo + i; g % 3 == 0 is tame, otherwise wild (see kangarooStarts() in engine/kangaroo_params.h).
  // totalKangaroos == 0 means this worker alone.
  uint64_t totalKangaroos = 0;
  uint64_t firstKangaroo = 0;
};

// dist is a signed two's-complement 256-bit distance: the point is dist*G (tame) or wildBase + dist*G (wild).
struct KangarooState {
  Affine point;
  U256 dist;
  bool lookahead = false;  // the next step uses params.lookahead
};

struct Dp {
  U256 x;
  U256 dist;  // signed, as in KangarooState
  bool wild = false;
  uint32_t index = 0;  // local kangaroo index
};

class KangarooWorker {
 public:
  virtual ~KangarooWorker() = default;
  virtual size_t kangarooCount() const = 0;
  // Local kangaroo i < resume.size() (<= kangarooCount()) continues from resume[i]; the others start fresh from
  // kangarooStarts(params, kangarooCount(), seed).
  virtual void init(const KangarooParams& params, uint64_t seed, const std::vector<KangarooState>& resume) = 0;
  // Advances every kangaroo (escaping fruitless cycles); appends distinguished points. Returns the jumps performed.
  virtual uint64_t step(std::vector<Dp>& out) = 0;
  // Restarts kangaroo `index` from a fresh random distance.
  virtual void reset(uint32_t index, uint64_t seed) = 0;
  // Current states of the first n kangaroos. Called between step() calls by the thread that calls step().
  virtual std::vector<KangarooState> sample(size_t n) = 0;
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

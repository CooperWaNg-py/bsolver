#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "backend/backend.h"

// Host code shared by the OpenCL, Metal and CUDA backends. Each backend only implements the device
// half (buffers + launches of the kernels in kernel_core.h) behind ScanKernel / KangarooKernel; the
// ScanWorker / KangarooWorker logic (tables, auto-tuning, hit and DP decoding) lives here.
namespace bsolver::gpu {

inline constexpr uint32_t kGroup = 64;               // GPU_GROUP
inline constexpr uint32_t kHitCapacity = 4096;       // SCAN_HIT_CAP
inline constexpr uint32_t kKangaroosPerThread = 64;  // KPT
inline constexpr uint32_t kWorkGroupSize = 128;
inline constexpr uint32_t kMaxTune = 256;
inline constexpr double kTuneTargetSeconds = 0.05;
inline constexpr size_t kScanParamWords = 10;
inline constexpr size_t kKangarooParamWords = 4;
inline constexpr size_t kDpEntryWords = 9;
inline constexpr size_t kStateComponents = 12;  // x[4], y[4], dist[4]

// Threads per launch: computeUnits * 512 rounded to a multiple of kWorkGroupSize.
uint32_t launchThreads(uint32_t computeUnits);

// Largest power of two <= min(kWorkGroupSize, deviceLimit) (at least 1).
uint32_t workGroupSize(uint64_t deviceLimit);

// Points are 8 words: x limbs then y limbs (little-endian 64-bit limbs, as Fe).
void packPoint(const Affine& p, uint64_t* out8);

// One scan hit as written by scan_check.
struct ScanHit {
  uint32_t thread;
  uint32_t iter;
  int32_t offset;
  uint32_t uncompressed;
};
static_assert(sizeof(ScanHit) == 16);

// Device half of a GPU scan worker (scan_kernel).
class ScanKernel {
 public:
  virtual ~ScanKernel() = default;
  virtual uint32_t threads() const = 0;
  // Uploads the group table (GPU_GROUP/2 + 1 points) and the targets (5 words each, at least one entry).
  virtual void setTables(const std::vector<uint64_t>& gTable, const std::vector<uint32_t>& targets) = 0;
  // Uploads offsetPts (8 words per thread); called again whenever iters changes.
  virtual void setOffsets(const std::vector<uint64_t>& offsets) = 0;
  // Runs one launch and waits for it. Returns the hit counter (may exceed kHitCapacity); hits receives the
  // first min(counter, kHitCapacity) entries.
  virtual uint32_t launch(const std::array<uint64_t, kScanParamWords>& params, std::vector<ScanHit>& hits) = 0;
  virtual std::string label() const = 0;
};

// Device half of a GPU kangaroo worker (kangaroo_kernel). The state buffer holds kStateComponents words per
// kangaroo in component-major order (word c*N + i).
class KangarooKernel {
 public:
  virtual ~KangarooKernel() = default;
  virtual uint32_t threads() const = 0;
  // 64 jumps x 12 words (x, y, dist).
  virtual void setJumps(const std::vector<uint64_t>& table) = 0;
  virtual void writeState(size_t offsetWords, const uint64_t* data, size_t words) = 0;
  virtual void readState(size_t offsetWords, uint64_t* data, size_t words) = 0;
  // Grows the DP buffer to at least `entries` entries.
  virtual void reserveDps(uint32_t entries) = 0;
  // Runs one launch and waits for it. Returns the DP counter (may exceed params[3]); dps receives the first
  // min(counter, params[3]) entries (kDpEntryWords each).
  virtual uint32_t launch(const std::array<uint64_t, kKangarooParamWords>& params, std::vector<uint64_t>& dps) = 0;
  virtual std::string label() const = 0;
};

// Doubles a per-launch work factor from 1 while one launch takes < kTuneTargetSeconds, capped at kMaxTune and
// by allowed(next). launch(v) runs one launch with factor v. Returns the chosen factor.
uint32_t autoTune(const std::function<void(uint32_t)>& launch, const std::function<bool(uint32_t)>& allowed);

std::unique_ptr<ScanWorker> makeScanWorker(std::unique_ptr<ScanKernel> kernel, const ScanSetup& setup);
std::unique_ptr<KangarooWorker> makeKangarooWorker(std::unique_ptr<KangarooKernel> kernel);

}  // namespace bsolver::gpu

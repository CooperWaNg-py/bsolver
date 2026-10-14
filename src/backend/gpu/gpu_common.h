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

// Kernel layout constants; Metal/OpenCL pass them to kernel_core.h via kernelDefines(), CUDA's kernels.cuh
// static_asserts them.
inline constexpr uint32_t kGroup = 1024;             // GPU_GROUP
inline constexpr uint32_t kHitCapacity = 4096;       // SCAN_HIT_CAP
inline constexpr uint32_t kKangaroosPerThread = 64;  // KPT
inline constexpr uint32_t kWorkGroupSize = 128;
inline constexpr uint32_t kMaxTune = 256;
inline constexpr double kTuneTargetSeconds = 0.05;
// A scan launch slower than this halves the threads per launch: macOS kills command buffers that keep the GPU from
// the display for long (kIOGPUCommandBufferCallbackErrorImpactingInteractivity), more readily while it is shared.
inline constexpr double kMaxLaunchSeconds = 0.1;
inline constexpr size_t kScanParamWords = 11;
inline constexpr size_t kKangarooParamWords = 5;
inline constexpr size_t kDpEntryWords = 9;
inline constexpr size_t kStateComponents = 13;  // x[4], y[4], dist[4], cycle reference (KANG_COMPONENTS)

// Kangaroo state buffer words for `threads` threads: kStateComponents words per kangaroo, component-major (word
// c*N + i), then one look-ahead flag word per thread (bit g of word kStateComponents*N + t = kangaroo g*T + t).
inline size_t kangarooStateWords(uint32_t threads) {
  return kStateComponents * size_t{threads} * kKangaroosPerThread + threads;
}

// Threads per launch: computeUnits * 512 rounded to a multiple of kWorkGroupSize.
uint32_t launchThreads(uint32_t computeUnits);

// Largest power of two <= min(kWorkGroupSize, deviceLimit) (at least 1).
uint32_t workGroupSize(uint64_t deviceLimit);

// "#define GPU_GROUP ...\n..." with every kernel layout constant (see kernel_core.h) for runtime-compiled preludes.
std::string kernelDefines();

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

// Scan and kangaroo counter buffers: u32 entry counter (hits / DP records), u32 count of threads that ran to the end.
inline constexpr size_t kCounterWords = 2;

// Outcome of one launch. finished < the launched thread count means the device dropped (part of) the launch: macOS
// aborts command buffers that keep the GPU from the display for long, and Apple's OpenCL reports those as complete.
// `error` is the device's reason when it gave one. Workers repeat such launches with less work.
struct LaunchResult {
  uint32_t count = 0;  // entry counter (may exceed the buffer capacity)
  uint32_t finished = 0;
  std::string error;
};

// Device half of a GPU scan worker (scan_kernel).
class ScanKernel {
 public:
  virtual ~ScanKernel() = default;
  // T: threads of the offsets table, a multiple of groupSize().
  virtual uint32_t threads() const = 0;
  // Work-group size; launches cover a multiple of it.
  virtual uint32_t groupSize() const = 0;
  // Uploads the group table (GPU_GROUP/2 + 1 points) and the targets (5 words each, at least one entry).
  virtual void setTables(const std::vector<uint64_t>& gTable, const std::vector<uint32_t>& targets) = 0;
  // Uploads offsetPts (8 words per thread); called again whenever iters changes.
  virtual void setOffsets(const std::vector<uint64_t>& offsets) = 0;
  // Runs threads [params[10], params[10] + count) (count a multiple of groupSize()) and waits for them; hits
  // receives the first min(result.count, kHitCapacity) entries.
  virtual LaunchResult launch(const std::array<uint64_t, kScanParamWords>& params, uint32_t count,
                              std::vector<ScanHit>& hits) = 0;
  virtual std::string label() const = 0;
};

// Device half of a GPU kangaroo worker (kangaroo_kernel). The current state buffer holds
// kangarooStateWords(threads()) words; a launch reads it and writes the next state to a second buffer, which becomes
// current only when every thread finished (so a dropped launch can simply be repeated).
class KangarooKernel {
 public:
  virtual ~KangarooKernel() = default;
  virtual uint32_t threads() const = 0;
  // (kKangarooJumps + kKangarooEscapeJumps) entries x 12 words (x, y, dist): main table, then look-ahead table.
  virtual void setJumps(const std::vector<uint64_t>& table) = 0;
  virtual void writeState(size_t offsetWords, const uint64_t* data, size_t words) = 0;
  virtual void readState(size_t offsetWords, uint64_t* data, size_t words) = 0;
  // Grows the DP buffer to at least `entries` entries.
  virtual void reserveDps(uint32_t entries) = 0;
  // Runs one launch over all threads and waits for it; dps receives the first min(count, params[3]) entries
  // (kDpEntryWords each).
  virtual LaunchResult launch(const std::array<uint64_t, kKangarooParamWords>& params, std::vector<uint64_t>& dps) = 0;
  virtual std::string label() const = 0;
};

// Doubles a per-launch work factor from 1 while one launch takes < kTuneTargetSeconds, capped at kMaxTune and
// by allowed(next). launch(v) runs one launch with factor v. Returns the chosen factor.
uint32_t autoTune(const std::function<void(uint32_t)>& launch, const std::function<bool(uint32_t)>& allowed);

std::unique_ptr<ScanWorker> makeScanWorker(std::unique_ptr<ScanKernel> kernel, const ScanSetup& setup);
std::unique_ptr<KangarooWorker> makeKangarooWorker(std::unique_ptr<KangarooKernel> kernel);

}  // namespace bsolver::gpu

#include "backend/metal/metal_backend.h"

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "assets/assets.h"
#include "backend/gpu/gpu_common.h"

namespace bsolver {

namespace {
constexpr const char* kPrelude =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#define KFN inline\n"
    "#define PRIV thread\n"
    "#define GLOB device\n"
    "#define CONSTAS constant\n"
    "#define MULHI64(a, b) mulhi((ulong)(a), (ulong)(b))\n"
    "#define U64 ulong\n"
    "#define U32 uint\n"
    "#define ATOMIC_INC(p) atomic_fetch_add_explicit((device atomic_uint*)(p), 1u, memory_order_relaxed)\n";

// Metal has no compute-unit query. Apple GPUs publish their core count as "gpu-core-count" in the
// IORegistry entry of the accelerator; other GPUs (or a missing entry) fall back to 8.
uint32_t gpuCoreCount(id<MTLDevice> device) {
  constexpr uint32_t kFallback = 8;
  io_registry_entry_t entry = IOServiceGetMatchingService(MACH_PORT_NULL, IORegistryEntryIDMatching(device.registryID));
  if (entry == IO_OBJECT_NULL) return kFallback;
  CFTypeRef value = IORegistryEntrySearchCFProperty(entry, kIOServicePlane, CFSTR("gpu-core-count"), kCFAllocatorDefault,
                                                    kIORegistryIterateRecursively | kIORegistryIterateParents);
  IOObjectRelease(entry);
  uint32_t cores = kFallback;
  if (value) {
    int n = 0;
    if (CFGetTypeID(value) == CFNumberGetTypeID() && CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberIntType, &n) &&
        n > 0) {
      cores = static_cast<uint32_t>(n);
    }
    CFRelease(value);
  }
  return cores;
}

// A device with its compiled pipelines (built once, shared by every worker).
struct MtlDevice {
  DeviceInfo info;
  id<MTLDevice> device = nil;
  id<MTLComputePipelineState> scan = nil;
  id<MTLComputePipelineState> kangaroo = nil;
};

std::string nsString(NSString* s) { return s ? std::string(s.UTF8String) : std::string(); }

// Per-worker command queue plus helpers for shared-storage buffers.
class MtlQueue {
 public:
  MtlQueue(const MtlDevice& dev, id<MTLComputePipelineState> pipeline) : dev_(dev), pipeline_(pipeline) {
    queue_ = [dev.device newCommandQueue];
    if (!queue_) throw std::runtime_error("metal: newCommandQueue failed");
    group_ = gpu::workGroupSize(pipeline.maxTotalThreadsPerThreadgroup);
    threads_ = gpu::launchThreads(dev.info.computeUnits);
  }

  uint32_t threads() const { return threads_; }
  std::string label() const { return "metal#" + std::to_string(dev_.info.index); }

  id<MTLBuffer> buffer(size_t bytes, const void* init = nullptr) {
    id<MTLBuffer> b = init ? [dev_.device newBufferWithBytes:init length:bytes options:MTLResourceStorageModeShared]
                           : [dev_.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("metal: buffer allocation of " + std::to_string(bytes) + " bytes failed");
    return b;
  }

  // Binds the buffers at indices 0..n-1, runs threads() threads and waits.
  void run(std::initializer_list<id<MTLBuffer>> args) {
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pipeline_];
      NSUInteger i = 0;
      for (id<MTLBuffer> b : args) [enc setBuffer:b offset:0 atIndex:i++];
      [enc dispatchThreadgroups:MTLSizeMake(threads_ / group_, 1, 1) threadsPerThreadgroup:MTLSizeMake(group_, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError) {
        throw std::runtime_error("metal: command buffer failed: " + nsString(cb.error.localizedDescription));
      }
    }
  }

 private:
  const MtlDevice& dev_;
  id<MTLComputePipelineState> pipeline_;
  id<MTLCommandQueue> queue_;
  uint32_t group_ = 1, threads_ = 0;
};

template <class T>
T* contents(id<MTLBuffer> b) {
  return static_cast<T*>(b.contents);
}

class MtlScanKernel final : public gpu::ScanKernel {
 public:
  explicit MtlScanKernel(const MtlDevice& dev) : q_(dev, dev.scan) {
    params_ = q_.buffer(gpu::kScanParamWords * 8);
    hits_ = q_.buffer(gpu::kHitCapacity * sizeof(gpu::ScanHit));
    hitCount_ = q_.buffer(4);
  }
  uint32_t threads() const override { return q_.threads(); }
  void setTables(const std::vector<uint64_t>& gTable, const std::vector<uint32_t>& targets) override {
    gTable_ = q_.buffer(gTable.size() * 8, gTable.data());
    targets_ = q_.buffer(targets.size() * 4, targets.data());
  }
  void setOffsets(const std::vector<uint64_t>& offsets) override {
    offsets_ = q_.buffer(offsets.size() * 8, offsets.data());
  }
  uint32_t launch(const std::array<uint64_t, gpu::kScanParamWords>& params, std::vector<gpu::ScanHit>& hits) override {
    std::memcpy(params_.contents, params.data(), sizeof(params));
    *contents<uint32_t>(hitCount_) = 0;
    q_.run({params_, offsets_, gTable_, targets_, hits_, hitCount_});
    uint32_t count = *contents<uint32_t>(hitCount_);
    const auto* h = contents<gpu::ScanHit>(hits_);
    hits.assign(h, h + std::min(count, gpu::kHitCapacity));
    return count;
  }
  std::string label() const override { return q_.label(); }

 private:
  MtlQueue q_;
  id<MTLBuffer> params_, offsets_, gTable_, targets_, hits_, hitCount_;
};

class MtlKangarooKernel final : public gpu::KangarooKernel {
 public:
  explicit MtlKangarooKernel(const MtlDevice& dev) : q_(dev, dev.kangaroo) {
    params_ = q_.buffer(gpu::kKangarooParamWords * 8);
    state_ = q_.buffer(gpu::kStateComponents * size_t{threads()} * gpu::kKangaroosPerThread * 8);
    dpCount_ = q_.buffer(4);
  }
  uint32_t threads() const override { return q_.threads(); }
  void setJumps(const std::vector<uint64_t>& table) override { jumps_ = q_.buffer(table.size() * 8, table.data()); }
  void writeState(size_t offsetWords, const uint64_t* data, size_t words) override {
    std::memcpy(contents<uint64_t>(state_) + offsetWords, data, words * 8);
  }
  void readState(size_t offsetWords, uint64_t* data, size_t words) override {
    std::memcpy(data, contents<uint64_t>(state_) + offsetWords, words * 8);
  }
  void reserveDps(uint32_t entries) override {
    if (entries <= dpCapacity_) return;
    dps_ = q_.buffer(size_t{entries} * gpu::kDpEntryWords * 8);
    dpCapacity_ = entries;
  }
  uint32_t launch(const std::array<uint64_t, gpu::kKangarooParamWords>& params, std::vector<uint64_t>& dps) override {
    std::memcpy(params_.contents, params.data(), sizeof(params));
    *contents<uint32_t>(dpCount_) = 0;
    q_.run({params_, state_, jumps_, dps_, dpCount_});
    uint32_t count = *contents<uint32_t>(dpCount_);
    const uint64_t* d = contents<uint64_t>(dps_);
    dps.assign(d, d + size_t{std::min<uint64_t>(count, params[3])} * gpu::kDpEntryWords);
    return count;
  }
  std::string label() const override { return q_.label(); }

 private:
  MtlQueue q_;
  id<MTLBuffer> params_, state_, jumps_, dps_, dpCount_;
  uint32_t dpCapacity_ = 0;
};

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name,
                                         std::string* error) {
  id<MTLFunction> fn = [library newFunctionWithName:name];
  if (!fn) {
    *error = "missing kernel function " + nsString(name);
    return nil;
  }
  NSError* err = nil;
  id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:fn error:&err];
  if (!p) *error = "pipeline creation failed: " + nsString(err.localizedDescription);
  return p;
}

// Enumerates the devices and compiles the kernels for each; *reason explains an empty result.
std::vector<MtlDevice> loadDevices(std::string* reason) {
  std::vector<MtlDevice> out;
  @autoreleasepool {
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    if (all.count == 0) {
      *reason = "no Metal devices";
      return out;
    }
    const std::string source = std::string(kPrelude) + std::string(assets::kMetalKernelSource);
    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    std::vector<std::string> errors;
    int index = 0;
    for (id<MTLDevice> device in all) {
      MtlDevice d;
      d.device = device;
      d.info.index = index++;
      d.info.name = nsString(device.name);
      d.info.memoryBytes = device.recommendedMaxWorkingSetSize;
      d.info.computeUnits = gpuCoreCount(device);
      NSError* err = nil;
      id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&err];
      std::string error;
      if (!library) {
        error = "kernel build failed: " + nsString(err.localizedDescription);
      } else {
        d.scan = makePipeline(device, library, @"scan_kernel", &error);
        if (d.scan) d.kangaroo = makePipeline(device, library, @"kangaroo_kernel", &error);
      }
      if (!d.kangaroo) {
        errors.push_back(d.info.name + ": " + error);
        continue;
      }
      out.push_back(std::move(d));
    }
    if (out.empty()) {
      for (const auto& e : errors) *reason += (reason->empty() ? "" : "; ") + e;
    }
  }
  return out;
}

class MetalBackend final : public Backend {
 public:
  std::string_view id() const override { return "metal"; }

  std::vector<DeviceInfo> devices() override {
    initialize();
    std::vector<DeviceInfo> out;
    for (const MtlDevice& d : devices_) out.push_back(d.info);
    return out;
  }

  std::string unavailableReason() const override {
    initialize();
    return reason_;
  }

  bool supports(Algo algo) const override { return algo != Algo::Bsgs; }

  std::vector<std::unique_ptr<ScanWorker>> createScanWorkers(const ScanSetup& setup, const std::vector<int>& devices,
                                                             unsigned) override {
    std::vector<std::unique_ptr<ScanWorker>> out;
    for (const MtlDevice* d : select(devices)) {
      out.push_back(gpu::makeScanWorker(std::make_unique<MtlScanKernel>(*d), setup));
    }
    return out;
  }

  std::vector<std::unique_ptr<KangarooWorker>> createKangarooWorkers(const std::vector<int>& devices,
                                                                     unsigned) override {
    std::vector<std::unique_ptr<KangarooWorker>> out;
    for (const MtlDevice* d : select(devices)) {
      out.push_back(gpu::makeKangarooWorker(std::make_unique<MtlKangarooKernel>(*d)));
    }
    return out;
  }

 private:
  std::vector<const MtlDevice*> select(const std::vector<int>& indices) {
    initialize();
    std::vector<const MtlDevice*> out;
    for (const MtlDevice& d : devices_) {
      if (indices.empty() || std::find(indices.begin(), indices.end(), d.info.index) != indices.end()) out.push_back(&d);
    }
    for (int i : indices) {
      if (std::none_of(devices_.begin(), devices_.end(), [&](const MtlDevice& d) { return d.info.index == i; })) {
        throw std::runtime_error("metal: no device " + std::to_string(i));
      }
    }
    return out;
  }

  void initialize() const {
    std::call_once(once_, [this] { devices_ = loadDevices(&reason_); });
  }

  mutable std::once_flag once_;
  mutable std::vector<MtlDevice> devices_;
  mutable std::string reason_;
};
}  // namespace

Backend* metalBackend() {
  static MetalBackend backend;
  return &backend;
}

}  // namespace bsolver

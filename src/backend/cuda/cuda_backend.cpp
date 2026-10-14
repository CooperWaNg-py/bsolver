#include "backend/cuda/cuda_backend.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "backend/cuda/cuda_device.h"
#include "backend/gpu/gpu_common.h"

namespace bsolver {

static_assert(cuda::kScanGroup == gpu::kGroup && cuda::kScanHitCapacity == gpu::kHitCapacity &&
                  cuda::kKangaroosPerThread == gpu::kKangaroosPerThread &&
                  cuda::kKangarooComponents == gpu::kStateComponents && cuda::kKangarooJumps == kKangarooJumps &&
                  cuda::kKangarooEscapeJumps == kKangarooEscapeJumps &&
                  cuda::kKangarooCycleWindow == kKangarooCycleWindow,
              "CUDA kernel layout constants out of sync with gpu_common.h / backend.h");

namespace {
// Owning device allocation.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(int device, size_t bytes) : device_(device), ptr_(cuda::allocate(device, bytes)) {}
  DeviceBuffer(DeviceBuffer&& o) noexcept : device_(o.device_), ptr_(std::exchange(o.ptr_, nullptr)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      cuda::release(device_, ptr_);
      device_ = o.device_;
      ptr_ = std::exchange(o.ptr_, nullptr);
    }
    return *this;
  }
  ~DeviceBuffer() { cuda::release(device_, ptr_); }
  void* get() const { return ptr_; }

 private:
  int device_ = 0;
  void* ptr_ = nullptr;
};

// Per-worker device context: launch geometry plus buffer helpers.
class CudaQueue {
 public:
  CudaQueue(const DeviceInfo& info, cuda::KernelId kernel)
      : info_(info),
        block_(gpu::workGroupSize(cuda::maxThreadsPerBlock(info.index, kernel))),
        threads_(gpu::launchThreads(info.computeUnits)) {}

  int device() const { return info_.index; }
  uint32_t threads() const { return threads_; }
  uint32_t block() const { return block_; }
  std::string label() const { return "cuda#" + std::to_string(info_.index); }

  DeviceBuffer buffer(size_t bytes, const void* init = nullptr) {
    DeviceBuffer b(info_.index, bytes);
    if (init) cuda::copyToDevice(info_.index, b.get(), init, bytes);
    return b;
  }
  void write(const DeviceBuffer& b, size_t offset, const void* data, size_t bytes) {
    cuda::copyToDevice(info_.index, static_cast<char*>(b.get()) + offset, data, bytes);
  }
  void read(const DeviceBuffer& b, size_t offset, void* data, size_t bytes) {
    cuda::copyToHost(info_.index, data, static_cast<const char*>(b.get()) + offset, bytes);
  }

 private:
  DeviceInfo info_;
  uint32_t block_, threads_;
};

class CudaScanKernel final : public gpu::ScanKernel {
 public:
  explicit CudaScanKernel(const DeviceInfo& info) : q_(info, cuda::KernelId::Scan) {
    params_ = q_.buffer(gpu::kScanParamWords * 8);
    hits_ = q_.buffer(gpu::kHitCapacity * sizeof(gpu::ScanHit));
    counters_ = q_.buffer(gpu::kCounterWords * 4);
  }
  uint32_t threads() const override { return q_.threads(); }
  uint32_t groupSize() const override { return q_.block(); }
  void setTables(const std::vector<uint64_t>& gTable, const std::vector<uint32_t>& targets) override {
    gTable_ = q_.buffer(gTable.size() * 8, gTable.data());
    targets_ = q_.buffer(targets.size() * 4, targets.data());
  }
  void setOffsets(const std::vector<uint64_t>& offsets) override {
    offsets_ = q_.buffer(offsets.size() * 8, offsets.data());
  }
  gpu::LaunchResult launch(const std::array<uint64_t, gpu::kScanParamWords>& params, uint32_t count,
                           std::vector<gpu::ScanHit>& hits) override {
    const uint32_t zero[gpu::kCounterWords] = {};
    q_.write(params_, 0, params.data(), sizeof(params));
    q_.write(counters_, 0, zero, sizeof(zero));
    cuda::launchScan(q_.device(), count, q_.block(), params_.get(), offsets_.get(), gTable_.get(), targets_.get(),
                     hits_.get(), counters_.get());
    uint32_t counters[gpu::kCounterWords];
    q_.read(counters_, 0, counters, sizeof(counters));
    gpu::LaunchResult out;
    out.count = counters[0];
    out.finished = counters[1];
    hits.resize(std::min(out.count, gpu::kHitCapacity));
    if (!hits.empty()) q_.read(hits_, 0, hits.data(), hits.size() * sizeof(gpu::ScanHit));
    return out;
  }
  std::string label() const override { return q_.label(); }

 private:
  CudaQueue q_;
  DeviceBuffer params_, offsets_, gTable_, targets_, hits_, counters_;
};

class CudaKangarooKernel final : public gpu::KangarooKernel {
 public:
  explicit CudaKangarooKernel(const DeviceInfo& info) : q_(info, cuda::KernelId::Kangaroo) {
    params_ = q_.buffer(gpu::kKangarooParamWords * 8);
    for (auto& s : state_) s = q_.buffer(gpu::kangarooStateWords(threads()) * 8);
    counters_ = q_.buffer(gpu::kCounterWords * 4);
  }
  uint32_t threads() const override { return q_.threads(); }
  void setJumps(const std::vector<uint64_t>& table) override { jumps_ = q_.buffer(table.size() * 8, table.data()); }
  void writeState(size_t offsetWords, const uint64_t* data, size_t words) override {
    q_.write(state_[cur_], offsetWords * 8, data, words * 8);
  }
  void readState(size_t offsetWords, uint64_t* data, size_t words) override {
    q_.read(state_[cur_], offsetWords * 8, data, words * 8);
  }
  void reserveDps(uint32_t entries) override {
    if (entries <= dpCapacity_) return;
    dps_ = q_.buffer(size_t{entries} * gpu::kDpEntryWords * 8);
    dpCapacity_ = entries;
  }
  gpu::LaunchResult launch(const std::array<uint64_t, gpu::kKangarooParamWords>& params,
                           std::vector<uint64_t>& dps) override {
    const uint32_t zero[gpu::kCounterWords] = {};
    q_.write(params_, 0, params.data(), sizeof(params));
    q_.write(counters_, 0, zero, sizeof(zero));
    cuda::launchKangaroo(q_.device(), q_.threads(), q_.block(), params_.get(), state_[cur_].get(),
                         state_[1 - cur_].get(), jumps_.get(), dps_.get(), counters_.get());
    uint32_t counters[gpu::kCounterWords];
    q_.read(counters_, 0, counters, sizeof(counters));
    gpu::LaunchResult out;
    out.count = counters[0];
    out.finished = counters[1];
    if (out.finished != threads()) return out;
    cur_ = 1 - cur_;
    dps.resize(size_t{std::min<uint64_t>(out.count, params[3])} * gpu::kDpEntryWords);
    if (!dps.empty()) q_.read(dps_, 0, dps.data(), dps.size() * 8);
    return out;
  }
  std::string label() const override { return q_.label(); }

 private:
  CudaQueue q_;
  DeviceBuffer params_, jumps_, dps_, counters_;
  DeviceBuffer state_[2];  // current state_[cur_]; a launch writes the other one
  int cur_ = 0;
  uint32_t dpCapacity_ = 0;
};

class CudaBackend final : public Backend {
 public:
  std::string_view id() const override { return "cuda"; }

  std::vector<DeviceInfo> devices() override {
    initialize();
    return devices_;
  }

  std::string unavailableReason() const override {
    initialize();
    return reason_;
  }

  bool supports(Algo algo) const override { return algo != Algo::Bsgs; }

  std::vector<std::unique_ptr<ScanWorker>> createScanWorkers(const ScanSetup& setup, const std::vector<int>& devices,
                                                             unsigned) override {
    std::vector<std::unique_ptr<ScanWorker>> out;
    for (const DeviceInfo* d : select(devices)) {
      out.push_back(gpu::makeScanWorker(std::make_unique<CudaScanKernel>(*d), setup));
    }
    return out;
  }

  std::vector<std::unique_ptr<KangarooWorker>> createKangarooWorkers(const std::vector<int>& devices,
                                                                     unsigned) override {
    std::vector<std::unique_ptr<KangarooWorker>> out;
    for (const DeviceInfo* d : select(devices)) {
      out.push_back(gpu::makeKangarooWorker(std::make_unique<CudaKangarooKernel>(*d)));
    }
    return out;
  }

 private:
  std::vector<const DeviceInfo*> select(const std::vector<int>& indices) {
    initialize();
    std::vector<const DeviceInfo*> out;
    for (const DeviceInfo& d : devices_) {
      if (indices.empty() || std::find(indices.begin(), indices.end(), d.index) != indices.end()) out.push_back(&d);
    }
    for (int i : indices) {
      if (std::none_of(devices_.begin(), devices_.end(), [&](const DeviceInfo& d) { return d.index == i; })) {
        throw std::runtime_error("cuda: no device " + std::to_string(i));
      }
    }
    return out;
  }

  void initialize() const {
    std::call_once(once_, [this] {
      int n = cuda::deviceCount(&reason_);
      for (int i = 0; i < n; ++i) {
        cuda::DeviceProps p = cuda::deviceProps(i);
        devices_.push_back(DeviceInfo{i, p.name, p.memoryBytes, p.multiprocessors});
      }
    });
  }

  mutable std::once_flag once_;
  mutable std::vector<DeviceInfo> devices_;
  mutable std::string reason_;
};
}  // namespace

Backend* cudaBackend() {
  static CudaBackend backend;
  return &backend;
}

}  // namespace bsolver

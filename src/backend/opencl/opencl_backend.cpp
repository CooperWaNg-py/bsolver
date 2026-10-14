#include "backend/opencl/opencl_backend.h"

#define CL_TARGET_OPENCL_VERSION 120
#if defined(__APPLE__)
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>

#include "assets/assets.h"
#include "backend/gpu/gpu_common.h"

namespace bsolver {

namespace {
constexpr const char* kPrelude =
    "#define KFN inline\n"
    "#define PRIV\n"
    "#define GLOB __global\n"
    "#define CONSTAS __constant\n"
    "#define MULHI64(a, b) mul_hi((ulong)(a), (ulong)(b))\n"
    "#define U64 ulong\n"
    "#define U32 uint\n"
    "#define ATOMIC_INC(p) atomic_inc(p)\n";

void check(cl_int err, const char* what) {
  if (err != CL_SUCCESS) throw std::runtime_error(std::string("opencl: ") + what + " failed (error " + std::to_string(err) + ")");
}

struct ReleaseMem {
  void operator()(cl_mem m) const { clReleaseMemObject(m); }
};
struct ReleaseKernel {
  void operator()(cl_kernel k) const { clReleaseKernel(k); }
};
struct ReleaseQueue {
  void operator()(cl_command_queue q) const { clReleaseCommandQueue(q); }
};
using Mem = std::unique_ptr<std::remove_pointer_t<cl_mem>, ReleaseMem>;
using Kernel = std::unique_ptr<std::remove_pointer_t<cl_kernel>, ReleaseKernel>;
using Queue = std::unique_ptr<std::remove_pointer_t<cl_command_queue>, ReleaseQueue>;

std::string deviceString(cl_device_id d, cl_device_info what) {
  size_t n = 0;
  clGetDeviceInfo(d, what, 0, nullptr, &n);
  std::string s(n, '\0');
  clGetDeviceInfo(d, what, n, s.data(), nullptr);
  while (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

// A GPU device with its context and compiled program (built once, shared by every worker).
struct ClDevice {
  DeviceInfo info;
  cl_device_id id = nullptr;
  cl_context context = nullptr;
  cl_program program = nullptr;
};

// Per-worker queue and kernel object (cl_kernel arguments are not thread-safe to share).
class ClQueue {
 public:
  ClQueue(const ClDevice& dev, const char* kernelName) : dev_(dev) {
    cl_int err;
    queue_.reset(clCreateCommandQueue(dev.context, dev.id, 0, &err));
    check(err, "clCreateCommandQueue");
    kernel_.reset(clCreateKernel(dev.program, kernelName, &err));
    check(err, "clCreateKernel");
    size_t limit = 0;
    check(clGetKernelWorkGroupInfo(kernel_.get(), dev.id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(limit), &limit, nullptr),
          "clGetKernelWorkGroupInfo");
    group_ = gpu::workGroupSize(limit);
    threads_ = gpu::launchThreads(dev.info.computeUnits);
  }

  uint32_t threads() const { return threads_; }
  std::string label() const { return "opencl#" + std::to_string(dev_.info.index); }

  Mem buffer(size_t bytes, const void* init = nullptr) {
    cl_int err;
    cl_mem m = clCreateBuffer(dev_.context, CL_MEM_READ_WRITE | (init ? CL_MEM_COPY_HOST_PTR : 0), bytes,
                              const_cast<void*>(init), &err);
    check(err, "clCreateBuffer");
    return Mem(m);
  }
  void write(const Mem& m, size_t offset, const void* data, size_t bytes) {
    check(clEnqueueWriteBuffer(queue_.get(), m.get(), CL_TRUE, offset, bytes, data, 0, nullptr, nullptr),
          "clEnqueueWriteBuffer");
  }
  void read(const Mem& m, size_t offset, void* data, size_t bytes) {
    check(clEnqueueReadBuffer(queue_.get(), m.get(), CL_TRUE, offset, bytes, data, 0, nullptr, nullptr),
          "clEnqueueReadBuffer");
  }
  // Binds the buffers as kernel arguments 0..n-1, runs threads() work items and waits.
  void run(std::initializer_list<const Mem*> args) {
    cl_uint i = 0;
    for (const Mem* m : args) {
      cl_mem h = m->get();
      check(clSetKernelArg(kernel_.get(), i++, sizeof(cl_mem), &h), "clSetKernelArg");
    }
    size_t global = threads_, local = group_;
    check(clEnqueueNDRangeKernel(queue_.get(), kernel_.get(), 1, nullptr, &global, &local, 0, nullptr, nullptr),
          "clEnqueueNDRangeKernel");
    check(clFinish(queue_.get()), "clFinish");
  }

 private:
  const ClDevice& dev_;
  Queue queue_;
  Kernel kernel_;
  uint32_t group_ = 1, threads_ = 0;
};

class ClScanKernel final : public gpu::ScanKernel {
 public:
  explicit ClScanKernel(const ClDevice& dev) : q_(dev, "scan_kernel") {
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
    const uint32_t zero = 0;
    q_.write(params_, 0, params.data(), sizeof(params));
    q_.write(hitCount_, 0, &zero, 4);
    q_.run({&params_, &offsets_, &gTable_, &targets_, &hits_, &hitCount_});
    uint32_t count = 0;
    q_.read(hitCount_, 0, &count, 4);
    hits.resize(std::min(count, gpu::kHitCapacity));
    if (!hits.empty()) q_.read(hits_, 0, hits.data(), hits.size() * sizeof(gpu::ScanHit));
    return count;
  }
  std::string label() const override { return q_.label(); }

 private:
  ClQueue q_;
  Mem params_, offsets_, gTable_, targets_, hits_, hitCount_;
};

class ClKangarooKernel final : public gpu::KangarooKernel {
 public:
  explicit ClKangarooKernel(const ClDevice& dev) : q_(dev, "kangaroo_kernel") {
    params_ = q_.buffer(gpu::kKangarooParamWords * 8);
    state_ = q_.buffer(gpu::kStateComponents * size_t{threads()} * gpu::kKangaroosPerThread * 8);
    dpCount_ = q_.buffer(4);
  }
  uint32_t threads() const override { return q_.threads(); }
  void setJumps(const std::vector<uint64_t>& table) override { jumps_ = q_.buffer(table.size() * 8, table.data()); }
  void writeState(size_t offsetWords, const uint64_t* data, size_t words) override {
    q_.write(state_, offsetWords * 8, data, words * 8);
  }
  void readState(size_t offsetWords, uint64_t* data, size_t words) override {
    q_.read(state_, offsetWords * 8, data, words * 8);
  }
  void reserveDps(uint32_t entries) override {
    if (entries <= dpCapacity_) return;
    dps_ = q_.buffer(size_t{entries} * gpu::kDpEntryWords * 8);
    dpCapacity_ = entries;
  }
  uint32_t launch(const std::array<uint64_t, gpu::kKangarooParamWords>& params, std::vector<uint64_t>& dps) override {
    const uint32_t zero = 0;
    q_.write(params_, 0, params.data(), sizeof(params));
    q_.write(dpCount_, 0, &zero, 4);
    q_.run({&params_, &state_, &jumps_, &dps_, &dpCount_});
    uint32_t count = 0;
    q_.read(dpCount_, 0, &count, 4);
    dps.resize(size_t{std::min<uint64_t>(count, params[3])} * gpu::kDpEntryWords);
    if (!dps.empty()) q_.read(dps_, 0, dps.data(), dps.size() * 8);
    return count;
  }
  std::string label() const override { return q_.label(); }

 private:
  ClQueue q_;
  Mem params_, state_, jumps_, dps_, dpCount_;
  uint32_t dpCapacity_ = 0;
};

class OpenClBackend final : public Backend {
 public:
  std::string_view id() const override { return "opencl"; }

  std::vector<DeviceInfo> devices() override {
    initialize();
    std::vector<DeviceInfo> out;
    for (const ClDevice& d : devices_) out.push_back(d.info);
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
    for (const ClDevice* d : select(devices)) out.push_back(gpu::makeScanWorker(std::make_unique<ClScanKernel>(*d), setup));
    return out;
  }

  std::vector<std::unique_ptr<KangarooWorker>> createKangarooWorkers(const std::vector<int>& devices,
                                                                     unsigned) override {
    std::vector<std::unique_ptr<KangarooWorker>> out;
    for (const ClDevice* d : select(devices)) {
      out.push_back(gpu::makeKangarooWorker(std::make_unique<ClKangarooKernel>(*d)));
    }
    return out;
  }

 private:
  std::vector<const ClDevice*> select(const std::vector<int>& indices) {
    initialize();
    std::vector<const ClDevice*> out;
    for (const ClDevice& d : devices_) {
      if (indices.empty() || std::find(indices.begin(), indices.end(), d.info.index) != indices.end()) out.push_back(&d);
    }
    for (int i : indices) {
      if (std::none_of(devices_.begin(), devices_.end(), [&](const ClDevice& d) { return d.info.index == i; })) {
        throw std::runtime_error("opencl: no device " + std::to_string(i));
      }
    }
    return out;
  }

  // Enumerates the GPU devices and builds the program for each (once per process).
  void initialize() const {
    std::call_once(once_, [this] {
      cl_uint np = 0;
      if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) {
        reason_ = "no OpenCL platforms";
        return;
      }
      std::vector<cl_platform_id> platforms(np);
      clGetPlatformIDs(np, platforms.data(), nullptr);
      const std::string source = std::string(kPrelude) + std::string(assets::kOpenCLKernelSource);
      const char* src = source.c_str();
      std::vector<std::string> errors;
      int index = 0;
      for (cl_platform_id p : platforms) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> ids(nd);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, ids.data(), nullptr);
        for (cl_device_id id : ids) {
          ClDevice d;
          d.id = id;
          d.info.index = index++;
          d.info.name = deviceString(id, CL_DEVICE_NAME);
          cl_ulong mem = 0;
          cl_uint cu = 0;
          clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, nullptr);
          clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
          d.info.memoryBytes = mem;
          d.info.computeUnits = cu;
          cl_int err;
          d.context = clCreateContext(nullptr, 1, &id, nullptr, nullptr, &err);
          if (err != CL_SUCCESS) {
            errors.push_back(d.info.name + ": clCreateContext failed (error " + std::to_string(err) + ")");
            continue;
          }
          d.program = clCreateProgramWithSource(d.context, 1, &src, nullptr, &err);
          if (err == CL_SUCCESS) err = clBuildProgram(d.program, 1, &id, "-cl-std=CL1.2", nullptr, nullptr);
          if (err != CL_SUCCESS) {
            std::string log = d.program ? deviceBuildLog(d.program, id) : "";
            errors.push_back(d.info.name + ": kernel build failed (error " + std::to_string(err) + ")" +
                             (log.empty() ? "" : ":\n" + log));
            if (d.program) clReleaseProgram(d.program);
            clReleaseContext(d.context);
            continue;
          }
          devices_.push_back(std::move(d));
        }
      }
      if (devices_.empty()) {
        reason_ = errors.empty() ? "no OpenCL GPU devices" : "";
        for (const auto& e : errors) reason_ += (reason_.empty() ? "" : "; ") + e;
      }
    });
  }

  static std::string deviceBuildLog(cl_program program, cl_device_id id) {
    size_t n = 0;
    clGetProgramBuildInfo(program, id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
    std::string log(n, '\0');
    clGetProgramBuildInfo(program, id, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    while (!log.empty() && (log.back() == '\0' || log.back() == '\n')) log.pop_back();
    return log;
  }

  mutable std::once_flag once_;
  mutable std::vector<ClDevice> devices_;
  mutable std::string reason_;
};
}  // namespace

Backend* openclBackend() {
  static OpenClBackend backend;
  return &backend;
}

}  // namespace bsolver

#include "backend/cpu/cpu_backend.h"

#include "backend/cpu/cpu_kangaroo_worker.h"
#include "backend/cpu/cpu_scan_worker.h"

namespace bsolver {

namespace {
class CpuBackend final : public Backend {
 public:
  std::string_view id() const override { return "cpu"; }

  std::vector<DeviceInfo> devices() override {
    unsigned n = effectiveThreads(0);
    return {DeviceInfo{0, "CPU (" + std::to_string(n) + " threads)", 0, n}};
  }

  std::string unavailableReason() const override { return ""; }

  bool supports(Algo) const override { return true; }

  std::vector<std::unique_ptr<ScanWorker>> createScanWorkers(const ScanSetup& setup, const std::vector<int>&,
                                                             unsigned threads) override {
    std::vector<std::unique_ptr<ScanWorker>> out;
    unsigned n = effectiveThreads(threads);
    for (unsigned i = 0; i < n; ++i) out.push_back(makeCpuScanWorker(setup, static_cast<int>(i)));
    return out;
  }

  std::vector<std::unique_ptr<KangarooWorker>> createKangarooWorkers(const std::vector<int>&,
                                                                     unsigned threads) override {
    std::vector<std::unique_ptr<KangarooWorker>> out;
    unsigned n = effectiveThreads(threads);
    for (unsigned i = 0; i < n; ++i) out.push_back(makeCpuKangarooWorker(static_cast<int>(i)));
    return out;
  }
};
}  // namespace

Backend* cpuBackend() {
  static CpuBackend backend;
  return &backend;
}

}  // namespace bsolver

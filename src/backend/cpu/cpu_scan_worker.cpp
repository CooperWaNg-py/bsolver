#include "backend/cpu/cpu_scan_worker.h"

#include "core/point_enum.h"
#include "core/ripemd160.h"
#include "engine/target_set.h"

namespace bsolver {

namespace {
class CpuScanWorker final : public ScanWorker {
 public:
  CpuScanWorker(const ScanSetup& setup, int index)
      : targets_(setup.targets),
        compressed_(hashesCompressed(setup.mode)),
        uncompressed_(hashesUncompressed(setup.mode)),
        index_(index) {}

  uint64_t segmentKeys() const override { return uint64_t{1} << 20; }

  void process(const U256& start, uint64_t count, std::vector<U256>& candidates) override {
    uint8_t buf[65];
    uint8_t h[20];
    enumerator_.run(start, count, [&](uint64_t offset, const Affine& P) {
      if (compressed_) {
        serializeCompressed(P, buf);
        hash160_33(buf, h);
        if (targets_.contains(h)) candidates.push_back(start + offset);
      }
      if (uncompressed_) {
        serializeUncompressed(P, buf);
        hash160_65(buf, h);
        if (targets_.contains(h)) candidates.push_back(start + offset);
      }
    });
  }

  std::string label() const override { return "cpu#" + std::to_string(index_); }

 private:
  TargetSet targets_;
  bool compressed_, uncompressed_;
  int index_;
  PointEnumerator<1024> enumerator_;
};
}  // namespace

std::unique_ptr<ScanWorker> makeCpuScanWorker(const ScanSetup& setup, int index) {
  return std::make_unique<CpuScanWorker>(setup, index);
}

}  // namespace bsolver

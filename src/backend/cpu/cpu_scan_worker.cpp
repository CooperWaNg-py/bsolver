#include "backend/cpu/cpu_scan_worker.h"

#include <algorithm>

#include "core/hash_batch.h"
#include "core/point_enum.h"
#include "engine/target_set.h"

namespace bsolver {

namespace {
// Keys per enumerator group: one field inversion per group, so larger groups amortize it better.
constexpr size_t kGroup = 4096;

uint32_t loadLE32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | static_cast<uint32_t>(p[1]) << 8 | static_cast<uint32_t>(p[2]) << 16 |
         static_cast<uint32_t>(p[3]) << 24;
}

// Big-endian 32-bit words of a field element (word 0 most significant), written to lane l of a word-major batch.
void storeWords(const Fe& v, HashLaneWords& out, size_t l) {
  for (int i = 0; i < 4; ++i) {
    out[2 * i][l] = static_cast<uint32_t>(v.n[3 - i] >> 32);
    out[2 * i + 1][l] = static_cast<uint32_t>(v.n[3 - i]);
  }
}

class CpuScanWorker final : public ScanWorker {
 public:
  CpuScanWorker(const ScanSetup& setup, int index)
      : targets_(setup.targets),
        hash_(hash160Batch(bestHashImpl())),
        compressed_(hashesCompressed(setup.mode)),
        uncompressed_(hashesUncompressed(setup.mode)),
        index_(index) {
    for (const auto& t : targets_.items()) prefixes_.push_back(loadLE32(t.data()));
    std::sort(prefixes_.begin(), prefixes_.end());
    prefixes_.erase(std::unique(prefixes_.begin(), prefixes_.end()), prefixes_.end());
  }

  uint64_t segmentKeys() const override { return uint64_t{1} << 20; }

  void process(const U256& start, uint64_t count, std::vector<U256>& candidates) override {
    size_t lanes = 0;
    enumerator_.run(start, count, [&](uint64_t offset, const Affine& P) {
      storeWords(P.x, x_, lanes);
      if (uncompressed_) storeWords(P.y, y_, lanes);
      parity_[lanes] = static_cast<uint32_t>(P.y.n[0] & 1);
      offsets_[lanes] = offset;
      if (++lanes == kHashLanes) {
        hashBatch(lanes, start, candidates);
        lanes = 0;
      }
    });
    if (lanes > 0) hashBatch(lanes, start, candidates);
  }

  std::string label() const override { return "cpu#" + std::to_string(index_); }

 private:
  // Hashes the first `lanes` buffered points and reports the target hits.
  void hashBatch(size_t lanes, const U256& start, std::vector<U256>& candidates) {
    if (compressed_) {
      hash_.compressed(x_, parity_, digests_);
      collectHits(lanes, start, candidates);
    }
    if (uncompressed_) {
      hash_.uncompressed(x_, y_, digests_);
      collectHits(lanes, start, candidates);
    }
  }

  void collectHits(size_t lanes, const U256& start, std::vector<U256>& candidates) const {
    for (size_t l = 0; l < lanes; ++l) {
      if (!std::binary_search(prefixes_.begin(), prefixes_.end(), digests_[0][l])) continue;
      std::array<uint8_t, 20> h;
      for (int i = 0; i < 5; ++i) {
        for (int b = 0; b < 4; ++b) h[4 * i + b] = static_cast<uint8_t>(digests_[i][l] >> (8 * b));
      }
      if (targets_.contains(h)) candidates.push_back(start + offsets_[l]);
    }
  }

  TargetSet targets_;
  std::vector<uint32_t> prefixes_;  // first 4 digest bytes of each target (little-endian word), sorted
  const Hash160Batch& hash_;
  bool compressed_, uncompressed_;
  int index_;
  PointEnumerator<kGroup> enumerator_;
  HashLaneWords x_{}, y_{};
  uint32_t parity_[kHashLanes]{};
  uint64_t offsets_[kHashLanes]{};
  HashLaneDigests digests_{};
};
}  // namespace

std::unique_ptr<ScanWorker> makeCpuScanWorker(const ScanSetup& setup, int index) {
  return std::make_unique<CpuScanWorker>(setup, index);
}

}  // namespace bsolver

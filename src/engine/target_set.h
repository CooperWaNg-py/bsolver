#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace bsolver {

// Sorted hash160 set; lexicographic byte order (the GPU kernels use the same order).
class TargetSet {
 public:
  TargetSet() = default;
  explicit TargetSet(std::vector<std::array<uint8_t, 20>> v) : v_(std::move(v)) {
    std::sort(v_.begin(), v_.end());
    v_.erase(std::unique(v_.begin(), v_.end()), v_.end());
  }
  bool contains(const uint8_t* h) const {
    std::array<uint8_t, 20> k;
    std::copy(h, h + 20, k.begin());
    return std::binary_search(v_.begin(), v_.end(), k);
  }
  bool contains(const std::array<uint8_t, 20>& h) const { return std::binary_search(v_.begin(), v_.end(), h); }
  const std::vector<std::array<uint8_t, 20>>& items() const { return v_; }
  size_t size() const { return v_.size(); }

 private:
  std::vector<std::array<uint8_t, 20>> v_;
};

}  // namespace bsolver

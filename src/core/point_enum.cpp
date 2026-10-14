#include "core/point_enum.h"

#include <map>

namespace bsolver {

const PointEnumTables& pointEnumTables(size_t group) {
  static std::mutex mu;
  static std::map<size_t, std::unique_ptr<PointEnumTables>> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto& slot = cache[group];
  if (!slot) {
    auto t = std::make_unique<PointEnumTables>();
    t->ig = progression(Affine{}, U256::fromU64(1), U256::fromU64(1), group / 2);
    t->groupG = mulG(U256::fromU64(group));
    slot = std::move(t);
  }
  return *slot;
}

}  // namespace bsolver

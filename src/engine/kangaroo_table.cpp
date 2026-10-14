#include "engine/kangaroo_table.h"

namespace bsolver {

std::optional<KangarooTable::Entry> KangarooTable::insert(const X128& x, const Entry& e) {
  Shard& s = shards_[shardOf(x)];
  std::lock_guard<std::mutex> lock(s.mu);
  auto [it, inserted] = s.map.try_emplace(x, e);
  if (inserted) return std::nullopt;
  return it->second;
}

size_t KangarooTable::size() const {
  size_t n = 0;
  for (const Shard& s : shards_) {
    std::lock_guard<std::mutex> lock(s.mu);
    n += s.map.size();
  }
  return n;
}

void KangarooTable::forEach(const std::function<void(const X128&, const Entry&)>& fn) const {
  for (const Shard& s : shards_) {
    std::lock_guard<std::mutex> lock(s.mu);
    for (const auto& [x, e] : s.map) fn(x, e);
  }
}

}  // namespace bsolver

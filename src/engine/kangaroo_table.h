#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "core/u256.h"

namespace bsolver {

// Low 128 bits of a distinguished point's x coordinate.
struct X128 {
  uint64_t lo = 0, hi = 0;
  static X128 of(const U256& x) { return X128{x.w[0], x.w[1]}; }
  friend bool operator==(const X128&, const X128&) = default;
};

// Thread-safe distinguished-point table: 256 mutex-guarded shards keyed by x low 128 bits.
class KangarooTable {
 public:
  struct Entry {
    U256 dist;  // mod n
    bool wild = false;
  };

  // Inserts (x, e) unless x is present; returns the existing entry on collision (the table keeps it).
  std::optional<Entry> insert(const X128& x, const Entry& e);
  size_t size() const;
  // Visits every entry, one locked shard at a time.
  void forEach(const std::function<void(const X128&, const Entry&)>& fn) const;

 private:
  struct Hash {
    size_t operator()(const X128& x) const { return static_cast<size_t>(x.lo ^ (x.hi * 0x9e3779b97f4a7c15ULL)); }
  };
  struct Shard {
    mutable std::mutex mu;
    std::unordered_map<X128, Entry, Hash> map;
  };
  static size_t shardOf(const X128& x) { return static_cast<size_t>(x.hi >> 56); }

  std::array<Shard, 256> shards_;
};

}  // namespace bsolver

#pragma once

#include <cstdint>
#include <mutex>
#include <set>

#include "core/random.h"
#include "core/u256.h"

namespace bsolver {

// Hands out consecutive segments [cursor, cursor+len) clipped to the range end. Tracks in-flight
// segment starts so the resume cursor never skips unfinished work. Thread-safe.
class SequentialSource {
 public:
  SequentialSource(const U256& rangeStart, const U256& rangeEnd, const U256& cursor)
      : end_(rangeEnd), cursor_(cursor < rangeStart ? rangeStart : cursor), exhausted_(cursor_ > rangeEnd) {}

  bool next(uint64_t maxLen, U256* start, uint64_t* len) {
    std::lock_guard<std::mutex> lock(mu_);
    if (exhausted_) return false;
    U256 remaining = end_ - cursor_;  // count - 1
    uint64_t n = maxLen;
    if (remaining < U256::fromU64(maxLen)) n = remaining.w[0] + 1;
    *start = cursor_;
    *len = n;
    inFlight_.insert(cursor_);
    U256 next = cursor_ + n;
    if (next > end_ || next.isZero()) {
      exhausted_ = true;
    }
    cursor_ = next;
    return true;
  }

  void complete(const U256& start) {
    std::lock_guard<std::mutex> lock(mu_);
    inFlight_.erase(start);
  }

  // Minimum in-flight start, or the cursor when nothing is in flight.
  U256 resumeCursor() const {
    std::lock_guard<std::mutex> lock(mu_);
    return inFlight_.empty() ? cursor_ : *inFlight_.begin();
  }

  // Every segment handed out and completed.
  bool finished() const {
    std::lock_guard<std::mutex> lock(mu_);
    return exhausted_ && inFlight_.empty();
  }

 private:
  mutable std::mutex mu_;
  U256 end_, cursor_;
  bool exhausted_;
  std::set<U256> inFlight_;
};

// Independent uniform segments start = uniform(rangeStart, max(rangeStart, rangeEnd - len + 1)).
class RandomSource {
 public:
  RandomSource(const U256& rangeStart, const U256& rangeEnd) : start_(rangeStart), end_(rangeEnd) {}

  bool next(uint64_t maxLen, U256* start, uint64_t* len) {
    U256 width = end_ - start_ + 1;
    U256 hi = start_;
    if (width > U256::fromU64(maxLen)) hi = end_ - maxLen + 1;
    std::lock_guard<std::mutex> lock(mu_);
    *start = rng_.uniform(start_, hi);
    U256 remaining = end_ - *start;
    *len = remaining < U256::fromU64(maxLen) ? remaining.w[0] + 1 : maxLen;
    return true;
  }

 private:
  std::mutex mu_;
  U256 start_, end_;
  Rng rng_;
};

}  // namespace bsolver

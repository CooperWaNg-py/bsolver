#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <vector>

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

// Keyed pseudorandom permutation of [0, count): a 4-round Feistel network over the smallest power-of-two domain with
// an even number of bits >= count (round function: SHA-256 of key, round, half width and half), cycle-walked into
// [0, count) (under 4 iterations on average, since the domain is < 4·count).
class BlockPermutation {
 public:
  using Key = std::array<uint8_t, 16>;
  BlockPermutation(const U256& count, const Key& key);  // count >= 1
  U256 operator()(const U256& i) const;                 // i < count

 private:
  U256 encrypt(const U256& x) const;

  U256 count_;
  Key key_;
  unsigned half_;  // bits per Feistel half; 0 when count == 1
};

// Random order without replacement. The range is cut into aligned blocks of 2^blockBits keys (the last one clipped
// to the range end), visited in the order of a keyed BlockPermutation. All workers share the current block, each
// taking chunks of its own segment size, so mixed segment sizes need no alignment. Chunks are tracked by their stream
// position `counter << blockBits | offset`; a snapshot lists the chunks handed out but not completed, which a resumed
// source hands out again before continuing at the cursor: no key is skipped or scanned twice. Thread-safe.
class PermutedSource {
 public:
  static constexpr unsigned kMinBlockBits = 20;
  static constexpr unsigned kMaxBlockBits = 40;

  struct Chunk {
    U256 pos;  // stream position
    uint64_t len;
  };
  struct State {
    BlockPermutation::Key key{};
    unsigned blockBits = kMinBlockBits;
    U256 cursor;                 // first stream position never handed out
    U256 keysDone;               // keys of completed chunks
    std::vector<Chunk> pending;  // handed out but not completed
  };

  // ~W/256..W/512 keys per block (at least 256 blocks when the range allows), clamped to [minBits, kMaxBlockBits].
  static unsigned blockBitsFor(const U256& width, unsigned minBits);
  // Fresh state with a random key (std::random_device).
  static State fresh(const U256& width, unsigned minBits);

  // Throws std::runtime_error when the state does not fit the range (corrupt checkpoint).
  PermutedSource(const U256& rangeStart, const U256& rangeEnd, State state);

  // Next chunk of at most maxLen keys: *start is its first key. False once every chunk was handed out.
  bool next(uint64_t maxLen, U256* start, uint64_t* len);
  void complete(const U256& start);
  State snapshot() const;
  const U256& blocks() const { return blocks_; }
  unsigned blockBits() const { return blockBits_; }
  // Every chunk handed out and completed.
  bool finished() const;

 private:
  uint64_t blockLen(const U256& block) const;
  U256 keyAt(const U256& pos) const;
  bool exhaustedLocked() const { return !((cursor_ >> blockBits_) < blocks_); }

  mutable std::mutex mu_;
  U256 start_;
  U256 width_;
  unsigned blockBits_;
  U256 blocks_;
  BlockPermutation perm_;
  BlockPermutation::Key key_;
  U256 cursor_;
  U256 keysDone_;
  std::vector<Chunk> pending_;
  std::map<U256, Chunk> inFlight_;  // by first key
};

}  // namespace bsolver

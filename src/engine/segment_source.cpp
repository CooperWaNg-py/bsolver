#include "engine/segment_source.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>

#include "core/sha256.h"

namespace bsolver {

namespace {
// v mod 2^bits.
U256 lowBits(const U256& v, unsigned bits) {
  U256 r = v;
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned lo = 64 * i;
    if (bits <= lo) {
      r.w[i] = 0;
    } else if (bits < lo + 64) {
      r.w[i] &= (uint64_t{1} << (bits - lo)) - 1;
    }
  }
  return r;
}

U256 xorU256(const U256& a, const U256& b) {
  U256 r;
  for (int i = 0; i < 4; ++i) r.w[i] = a.w[i] ^ b.w[i];
  return r;
}

[[noreturn]] void corrupt() { throw std::runtime_error("random-order state does not fit the range"); }

unsigned checkedBlockBits(unsigned bits) {
  if (bits < PermutedSource::kMinBlockBits || bits > PermutedSource::kMaxBlockBits) corrupt();
  return bits;
}
}  // namespace

BlockPermutation::BlockPermutation(const U256& count, const Key& key)
    : count_(count), key_(key), half_(((count - 1).bitLength() + 1) / 2) {
  if (count.isZero()) throw std::invalid_argument("BlockPermutation: empty domain");
}

U256 BlockPermutation::encrypt(const U256& x) const {
  U256 l = x >> half_;
  U256 r = lowBits(x, half_);
  // key (16) | round (1) | half width (1) | right half (32, big-endian)
  uint8_t msg[50];
  std::memcpy(msg, key_.data(), key_.size());
  msg[17] = static_cast<uint8_t>(half_);
  for (uint8_t round = 0; round < 4; ++round) {
    msg[16] = round;
    r.toBytesBE(msg + 18);
    uint8_t h[32];
    sha256(msg, sizeof msg, h);
    const U256 f = lowBits(U256::fromBytesBE(h), half_);
    const U256 nextR = xorU256(l, f);
    l = r;
    r = nextR;
  }
  return (l << half_) + r;
}

U256 BlockPermutation::operator()(const U256& i) const {
  if (half_ == 0) return i;
  // Cycle walking: the Feistel network permutes [0, 2^(2·half)) ⊇ [0, count); following the cycle of i until it
  // re-enters [0, count) restricts it to a permutation of [0, count).
  U256 x = encrypt(i);
  while (!(x < count_)) x = encrypt(x);
  return x;
}

unsigned PermutedSource::blockBitsFor(const U256& width, unsigned minBits) {
  const int bits = static_cast<int>(width.bitLength()) - 9;
  return static_cast<unsigned>(std::clamp(bits, static_cast<int>(minBits), static_cast<int>(kMaxBlockBits)));
}

PermutedSource::State PermutedSource::fresh(const U256& width, unsigned minBits) {
  State s;
  std::random_device rd;
  for (size_t i = 0; i < s.key.size(); i += 4) {
    const uint32_t v = rd();
    std::memcpy(s.key.data() + i, &v, 4);
  }
  s.blockBits = blockBitsFor(width, minBits);
  return s;
}

PermutedSource::PermutedSource(const U256& rangeStart, const U256& rangeEnd, State state)
    : start_(rangeStart),
      width_(rangeEnd - rangeStart + 1),
      blockBits_(checkedBlockBits(state.blockBits)),
      blocks_(((width_ - 1) >> blockBits_) + 1),
      perm_(blocks_, state.key),
      key_(state.key),
      cursor_(state.cursor),
      keysDone_(state.keysDone),
      pending_(std::move(state.pending)) {
  const U256 counter = cursor_ >> blockBits_;
  const uint64_t mask = (uint64_t{1} << blockBits_) - 1;
  if (blocks_ < counter || width_ < keysDone_) corrupt();
  if (counter < blocks_ ? (cursor_.w[0] & mask) >= blockLen(perm_(counter)) : (cursor_.w[0] & mask) != 0) corrupt();
  for (const Chunk& c : pending_) {
    const U256 block = c.pos >> blockBits_;
    if (c.len == 0 || !(c.pos < cursor_) || !(block < blocks_)) corrupt();
    if ((c.pos.w[0] & mask) + c.len > blockLen(perm_(block))) corrupt();
  }
}

uint64_t PermutedSource::blockLen(const U256& block) const {
  const U256 last = blocks_ - 1;
  return block < last ? uint64_t{1} << blockBits_ : (width_ - (last << blockBits_)).w[0];
}

U256 PermutedSource::keyAt(const U256& pos) const {
  const uint64_t offset = pos.w[0] & ((uint64_t{1} << blockBits_) - 1);
  return start_ + (perm_(pos >> blockBits_) << blockBits_) + offset;
}

bool PermutedSource::next(uint64_t maxLen, U256* start, uint64_t* len) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!pending_.empty()) {
    Chunk& c = pending_.back();
    const uint64_t n = std::min(maxLen, c.len);
    *start = keyAt(c.pos);
    *len = n;
    inFlight_.emplace(*start, Chunk{c.pos, n});
    if (n == c.len) {
      pending_.pop_back();
    } else {
      c.pos = c.pos + n;
      c.len -= n;
    }
    return true;
  }
  if (exhaustedLocked()) return false;
  const U256 counter = cursor_ >> blockBits_;
  const uint64_t offset = cursor_.w[0] & ((uint64_t{1} << blockBits_) - 1);
  const U256 block = perm_(counter);
  const uint64_t remaining = blockLen(block) - offset;
  const uint64_t n = std::min(maxLen, remaining);
  *start = start_ + (block << blockBits_) + offset;
  *len = n;
  inFlight_.emplace(*start, Chunk{cursor_, n});
  cursor_ = n == remaining ? (counter + 1) << blockBits_ : cursor_ + n;
  return true;
}

void PermutedSource::complete(const U256& start) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = inFlight_.find(start);
  if (it == inFlight_.end()) return;
  keysDone_ = keysDone_ + it->second.len;
  inFlight_.erase(it);
}

PermutedSource::State PermutedSource::snapshot() const {
  std::lock_guard<std::mutex> lock(mu_);
  State s{key_, blockBits_, cursor_, keysDone_, pending_};
  for (const auto& [start, chunk] : inFlight_) s.pending.push_back(chunk);
  return s;
}

bool PermutedSource::finished() const {
  std::lock_guard<std::mutex> lock(mu_);
  return exhaustedLocked() && pending_.empty() && inFlight_.empty();
}

}  // namespace bsolver

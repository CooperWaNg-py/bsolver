#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/u256.h"

namespace bsolver {

// Little-endian binary writer used for checkpoints.
class ByteWriter {
 public:
  void u8(uint8_t v) { buf_.push_back(v); }
  void u32(uint32_t v) {
    for (int i = 0; i < 4; ++i) buf_.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
  void u64(uint64_t v) {
    for (int i = 0; i < 8; ++i) buf_.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
  void u256(const U256& v) {
    for (uint64_t w : v.w) u64(w);
  }
  void bytes(const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    buf_.insert(buf_.end(), b, b + n);
  }
  std::vector<uint8_t>& data() { return buf_; }

 private:
  std::vector<uint8_t> buf_;
};

// Bounds-checked reader; throws std::runtime_error("truncated data").
class ByteReader {
 public:
  ByteReader(const uint8_t* p, size_t n) : p_(p), n_(n) {}
  explicit ByteReader(const std::vector<uint8_t>& v) : p_(v.data()), n_(v.size()) {}
  uint8_t u8() {
    need(1);
    return p_[pos_++];
  }
  uint32_t u32() {
    need(4);
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p_[pos_++]) << (8 * i);
    return v;
  }
  uint64_t u64() {
    need(8);
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p_[pos_++]) << (8 * i);
    return v;
  }
  U256 u256() {
    U256 v;
    for (auto& w : v.w) w = u64();
    return v;
  }
  void bytes(void* out, size_t n) {
    need(n);
    std::memcpy(out, p_ + pos_, n);
    pos_ += n;
  }
  std::string str(size_t n) {
    need(n);
    std::string s(reinterpret_cast<const char*>(p_ + pos_), n);
    pos_ += n;
    return s;
  }
  size_t remaining() const { return n_ - pos_; }

 private:
  void need(size_t k) const {
    if (n_ - pos_ < k) throw std::runtime_error("truncated data");
  }
  const uint8_t* p_;
  size_t n_;
  size_t pos_ = 0;
};

}  // namespace bsolver

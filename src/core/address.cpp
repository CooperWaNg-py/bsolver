#include "core/address.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "core/base58.h"
#include "core/ripemd160.h"
#include "core/sha256.h"

namespace bsolver {

namespace {
std::string base58Check(std::vector<uint8_t> payload) {
  uint8_t chk[32];
  sha256d(payload.data(), payload.size(), chk);
  payload.insert(payload.end(), chk, chk + 4);
  return base58Encode(payload.data(), payload.size());
}
}  // namespace

Hash160 decodeP2PKH(std::string_view address) {
  const std::invalid_argument unsupported(
      "unsupported address type (only P2PKH '1...' addresses are supported)");
  auto raw = base58Decode(address);
  if (!raw || raw->size() != 25 || (*raw)[0] != 0x00) throw unsupported;
  uint8_t chk[32];
  sha256d(raw->data(), 21, chk);
  if (std::memcmp(chk, raw->data() + 21, 4) != 0) throw std::invalid_argument("invalid address checksum");
  Hash160 h;
  std::memcpy(h.data(), raw->data() + 1, 20);
  return h;
}

std::string encodeP2PKH(const uint8_t h[20]) {
  std::vector<uint8_t> payload(21);
  payload[0] = 0x00;
  std::memcpy(payload.data() + 1, h, 20);
  return base58Check(std::move(payload));
}

std::string encodeP2PKH(const Hash160& h) { return encodeP2PKH(h.data()); }

std::string wifCompressed(const U256& key) {
  std::vector<uint8_t> payload(34);
  payload[0] = 0x80;
  key.toBytesBE(payload.data() + 1);
  payload[33] = 0x01;
  return base58Check(std::move(payload));
}

Hash160 hash160Of(const Affine& p, bool compressed) {
  Hash160 h;
  if (compressed) {
    uint8_t buf[33];
    serializeCompressed(p, buf);
    hash160_33(buf, h.data());
  } else {
    uint8_t buf[65];
    serializeUncompressed(p, buf);
    hash160_65(buf, h.data());
  }
  return h;
}

}  // namespace bsolver

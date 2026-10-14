#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <nlohmann/json.hpp>

#include <cstring>
#include <string>

#include "assets/assets.h"
#include "core/address.h"
#include "core/field.h"
#include "core/gtable.h"
#include "core/hex.h"
#include "core/point.h"
#include "core/point_enum.h"
#include "core/random.h"
#include "core/ripemd160.h"
#include "core/sha256.h"

using namespace bsolver;

namespace {
const uint8_t* bytes(const char* s) { return reinterpret_cast<const uint8_t*>(s); }
}  // namespace

TEST_CASE("hash vectors") {
  uint8_t d[32];
  sha256(bytes("abc"), 3, d);
  CHECK(hexEncode(d, 32) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  uint8_t r[20];
  ripemd160(bytes("abc"), 3, r);
  CHECK(hexEncode(r, 20) == "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc");
  ripemd160(bytes(""), 0, r);
  CHECK(hexEncode(r, 20) == "9c1185a5c5e9fc54612808977ee8f548b2258d31");
}

TEST_CASE("hash160 fast paths match streaming implementations") {
  Rng rng(7);
  for (int i = 0; i < 20; ++i) {
    uint8_t pub[65];
    for (auto& b : pub) b = static_cast<uint8_t>(rng.next());
    uint8_t s[32], expect[20], got[20];
    sha256(pub, 33, s);
    ripemd160(s, 32, expect);
    hash160_33(pub, got);
    CHECK(std::memcmp(expect, got, 20) == 0);
    sha256(pub, 65, s);
    ripemd160(s, 32, expect);
    hash160_65(pub, got);
    CHECK(std::memcmp(expect, got, 20) == 0);
  }
}

TEST_CASE("WIF of key 1") {
  CHECK(wifCompressed(U256::fromU64(1)) == "KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn");
}

TEST_CASE("U256 hex parsing") {
  CHECK(U256::fromHex("0x1f").w[0] == 0x1f);
  CHECK(U256::fromHex("dc2a04").toHex() == "dc2a04");
  CHECK(U256::fromHex("0").toHex(true) == std::string(64, '0'));
  CHECK_THROWS_WITH(U256::fromHex("xyz"), "invalid hex: xyz");
  CHECK_THROWS(U256::fromHex(std::string(65, 'f')));
  CHECK_THROWS(U256::fromHex(""));
}

TEST_CASE("solved catalog entries derive their pubkey and address") {
  auto catalog = nlohmann::json::parse(assets::kPuzzleCatalogJson);
  int checked = 0;
  for (const auto& e : catalog) {
    if (!e["solved"].get<bool>()) continue;
    U256 key = U256::fromHex(e["privkey"].get<std::string>());
    Affine p = mulG(key);
    uint8_t ser[33];
    serializeCompressed(p, ser);
    INFO("puzzle " << e["id"]);
    CHECK(hexEncode(ser, 33) == e["pubkey"].get<std::string>());
    uint8_t h[20];
    hash160_33(ser, h);
    CHECK(encodeP2PKH(h) == e["address"].get<std::string>());
    CHECK(parsePubkey(e["pubkey"].get<std::string>()) == p);
    ++checked;
  }
  CHECK(checked > 80);
}

TEST_CASE("address decoding errors") {
  CHECK(hexEncode(decodeP2PKH("1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU").data(), 20).size() == 40);
  CHECK_THROWS_WITH(decodeP2PKH("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy"),
                    "unsupported address type (only P2PKH '1...' addresses are supported)");
  CHECK_THROWS_WITH(decodeP2PKH("1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXV"), "invalid address checksum");
}

TEST_CASE("uncompressed pubkey round trip and invalid pubkeys") {
  Affine p = mulG(U256::fromHex("e9ae4933d6"));
  uint8_t u[65];
  serializeUncompressed(p, u);
  CHECK(parsePubkey(hexEncode(u, 65)) == p);
  CHECK_THROWS_WITH(parsePubkey("02" + std::string(64, '0')), "invalid public key");
  CHECK_THROWS_WITH(parsePubkey("05" + std::string(64, '1')), "invalid public key");
}

TEST_CASE("PointEnumerator matches mulG") {
  PointEnumerator<1024> e;
  auto check = [&](const U256& start, uint64_t count) {
    std::vector<int> seen(count, 0);
    uint64_t bad = 0;
    e.run(start, count, [&](uint64_t off, const Affine& P) {
      REQUIRE(off < count);
      seen[off]++;
      if (!(P == mulG(start + off))) ++bad;
    });
    CHECK(bad == 0);
    for (uint64_t i = 0; i < count; ++i) {
      if (seen[i] != 1) FAIL("offset " << i << " seen " << seen[i] << " times");
    }
  };
  check(U256::fromU64(1), 5000);
  check(U256::fromU64((uint64_t{1} << 40) - 100), 3101);
}

TEST_CASE("progression matches mulG") {
  Affine base = mulG(U256::fromU64(12345));
  auto pts = progression(base, U256::fromU64(10), U256::fromU64(7), 50);
  for (size_t i = 0; i < pts.size(); ++i) CHECK(pts[i] == mulG(U256::fromU64(12345 + 10 + 7 * i)));
}

TEST_CASE("batchInverse skips zeros") {
  Rng rng(1);
  Fe v[16], orig[16], scratch[16];
  for (int i = 0; i < 16; ++i) {
    v[i] = Fe::fromU256(rng.uniform(U256::fromU64(1), U256::fromHex("ffffffffffffffffffffffffffffffff")));
  }
  v[0] = Fe{};
  v[7] = Fe{};
  std::copy(v, v + 16, orig);
  batchInverse(v, 16, scratch);
  for (int i = 0; i < 16; ++i) {
    if (i == 0 || i == 7) {
      CHECK(v[i].isZero());
    } else {
      CHECK(feMul(orig[i], v[i]) == Fe::fromU64(1));
    }
  }
}

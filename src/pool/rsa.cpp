#include "pool/rsa.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>

namespace bsolver {

namespace {

// ---------------------------------------------------------------- SHA-1 (FIPS 180-4)

class Sha1 {
 public:
  void update(const uint8_t* data, size_t len) {
    total_ += len;
    while (len > 0) {
      size_t take = std::min(len, sizeof(buf_) - bufLen_);
      std::memcpy(buf_ + bufLen_, data, take);
      bufLen_ += take;
      data += take;
      len -= take;
      if (bufLen_ == 64) {
        transform(buf_);
        bufLen_ = 0;
      }
    }
  }

  void final(uint8_t out[20]) {
    uint64_t bits = total_ * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t zero = 0;
    while (bufLen_ != 56) update(&zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
    update(len, 8);
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 4; ++j) out[4 * i + j] = static_cast<uint8_t>(s_[i] >> (24 - 8 * j));
  }

 private:
  static uint32_t rol(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

  void transform(const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(block[4 * i]) << 24) | (uint32_t(block[4 * i + 1]) << 16) |
             (uint32_t(block[4 * i + 2]) << 8) | uint32_t(block[4 * i + 3]);
    for (int i = 16; i < 80; ++i) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = s_[0], b = s_[1], c = s_[2], d = s_[3], e = s_[4];
    for (int i = 0; i < 80; ++i) {
      uint32_t f, k;
      if (i < 20) {
        f = (b & c) | (~b & d);
        k = 0x5A827999;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ED9EBA1;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8F1BBCDC;
      } else {
        f = b ^ c ^ d;
        k = 0xCA62C1D6;
      }
      uint32_t t = rol(a, 5) + f + e + k + w[i];
      e = d;
      d = c;
      c = rol(b, 30);
      b = a;
      a = t;
    }
    s_[0] += a;
    s_[1] += b;
    s_[2] += c;
    s_[3] += d;
    s_[4] += e;
  }

  uint32_t s_[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
  uint8_t buf_[64];
  size_t bufLen_ = 0;
  uint64_t total_ = 0;
};

constexpr size_t kHLen = 20;

// MGF1 with SHA-1 (RFC 8017 B.2.1): XORs the mask derived from `seed` into out[0..len).
void mgf1XorInto(const uint8_t* seed, size_t seedLen, uint8_t* out, size_t len) {
  uint8_t digest[kHLen];
  for (uint32_t counter = 0; len > 0; ++counter) {
    Sha1 h;
    h.update(seed, seedLen);
    uint8_t c[4] = {uint8_t(counter >> 24), uint8_t(counter >> 16), uint8_t(counter >> 8),
                    uint8_t(counter)};
    h.update(c, 4);
    h.final(digest);
    size_t take = std::min(len, kHLen);
    for (size_t i = 0; i < take; ++i) out[i] ^= digest[i];
    out += take;
    len -= take;
  }
}

// ---------------------------------------------------------------- errors / base64

[[noreturn]] void fail(const std::string& detail) {
  throw std::invalid_argument("invalid RSA public key: " + detail);
}

constexpr char kB64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int b64Value(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

bool isSpace(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

// Strict padded base64; whitespace is skipped.
std::vector<uint8_t> base64Decode(std::string_view text) {
  std::string s;
  s.reserve(text.size());
  for (char c : text)
    if (!isSpace(c)) s.push_back(c);
  if (s.empty() || s.size() % 4 != 0) fail("bad base64 length");
  size_t pad = 0;
  if (s.back() == '=') pad = (s[s.size() - 2] == '=') ? 2 : 1;
  std::vector<uint8_t> out;
  out.reserve(s.size() / 4 * 3);
  for (size_t i = 0; i < s.size(); i += 4) {
    bool last = i + 4 == s.size();
    uint32_t v = 0;
    for (size_t j = 0; j < 4; ++j) {
      char c = s[i + j];
      int d;
      if (c == '=' && last && j >= 4 - pad) {
        d = 0;
      } else {
        d = b64Value(c);
        if (d < 0) fail("bad base64 character");
      }
      v = (v << 6) | uint32_t(d);
    }
    out.push_back(uint8_t(v >> 16));
    if (!last || pad < 2) out.push_back(uint8_t(v >> 8));
    if (!last || pad < 1) out.push_back(uint8_t(v));
  }
  return out;
}

// ---------------------------------------------------------------- DER

class Der {
 public:
  Der(const uint8_t* p, size_t n) : p_(p), end_(p + n) {}

  bool empty() const { return p_ == end_; }

  // Reads one TLV with the expected tag; returns its contents.
  Der read(uint8_t tag, const char* what) {
    if (end_ - p_ < 2) fail(std::string("truncated ") + what);
    if (*p_++ != tag) fail(std::string("expected ") + what);
    size_t len = *p_++;
    if (len & 0x80) {
      size_t nbytes = len & 0x7f;
      if (nbytes == 0 || nbytes > 4) fail(std::string("bad length in ") + what);
      if (size_t(end_ - p_) < nbytes) fail(std::string("truncated ") + what);
      len = 0;
      for (size_t i = 0; i < nbytes; ++i) len = (len << 8) | *p_++;
    }
    if (size_t(end_ - p_) < len) fail(std::string("truncated ") + what);
    Der inner(p_, len);
    p_ += len;
    return inner;
  }

  // Positive INTEGER as big-endian bytes without leading zeros.
  std::vector<uint8_t> readUnsigned(const char* what) {
    Der v = read(0x02, what);
    if (v.empty()) fail(std::string("empty ") + what);
    if (*v.p_ & 0x80) fail(std::string("negative ") + what);
    while (v.p_ != v.end_ && *v.p_ == 0) ++v.p_;
    if (v.empty()) fail(std::string("zero ") + what);
    return std::vector<uint8_t>(v.p_, v.end_);
  }

  const uint8_t* data() const { return p_; }
  size_t size() const { return size_t(end_ - p_); }

 private:
  const uint8_t* p_;
  const uint8_t* end_;
};

// RSAPublicKey ::= SEQUENCE { modulus INTEGER, publicExponent INTEGER }
RsaPublicKey parsePkcs1(const uint8_t* p, size_t n) {
  Der outer(p, n);
  Der seq = outer.read(0x30, "RSAPublicKey sequence");
  if (!outer.empty()) fail("trailing data after RSAPublicKey");
  RsaPublicKey key;
  key.n = seq.readUnsigned("modulus");
  key.e = seq.readUnsigned("exponent");
  if (!seq.empty()) fail("trailing data in RSAPublicKey");
  return key;
}

// SubjectPublicKeyInfo ::= SEQUENCE { AlgorithmIdentifier, BIT STRING RSAPublicKey }
RsaPublicKey parseSpki(const uint8_t* p, size_t n) {
  static const uint8_t kRsaOid[] = {0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01};
  Der outer(p, n);
  Der spki = outer.read(0x30, "SubjectPublicKeyInfo sequence");
  if (!outer.empty()) fail("trailing data after SubjectPublicKeyInfo");
  Der alg = spki.read(0x30, "AlgorithmIdentifier");
  Der oid = alg.read(0x06, "algorithm OID");
  if (oid.size() != sizeof(kRsaOid) || std::memcmp(oid.data(), kRsaOid, sizeof(kRsaOid)) != 0)
    fail("not an rsaEncryption key");
  if (!alg.empty()) {
    Der params = alg.read(0x05, "NULL parameters");
    if (!params.empty() || !alg.empty()) fail("bad algorithm parameters");
  }
  Der bits = spki.read(0x03, "public key BIT STRING");
  if (!spki.empty()) fail("trailing data in SubjectPublicKeyInfo");
  if (bits.empty() || bits.data()[0] != 0) fail("bad BIT STRING");
  return parsePkcs1(bits.data() + 1, bits.size() - 1);
}

// ---------------------------------------------------------------- big integer (Montgomery)

using Limbs = std::vector<uint32_t>;  // little-endian 32-bit limbs

Limbs toLimbs(const std::vector<uint8_t>& be, size_t limbs) {
  Limbs out(limbs, 0);
  for (size_t i = 0; i < be.size(); ++i) {
    size_t bytePos = be.size() - 1 - i;  // significance of be[bytePos] is i
    out[i / 4] |= uint32_t(be[bytePos]) << (8 * (i % 4));
  }
  return out;
}

// a >= b (equal length)
bool geq(const Limbs& a, const Limbs& b) {
  for (size_t i = a.size(); i-- > 0;)
    if (a[i] != b[i]) return a[i] > b[i];
  return true;
}

// a -= b (equal length); returns borrow
uint32_t subInPlace(Limbs& a, const Limbs& b) {
  uint64_t borrow = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    uint64_t d = uint64_t(a[i]) - b[i] - borrow;
    a[i] = uint32_t(d);
    borrow = (d >> 63) & 1;
  }
  return uint32_t(borrow);
}

class Montgomery {
 public:
  explicit Montgomery(const Limbs& n) : n_(n), s_(n.size()) {
    // n0inv = -n^{-1} mod 2^32 by Newton iteration (n odd).
    uint32_t inv = 1;
    for (int i = 0; i < 5; ++i) inv *= 2 - n_[0] * inv;
    n0inv_ = 0u - inv;
    // r2 = R^2 mod n, R = 2^(32 s), by repeated doubling of 1.
    r2_.assign(s_, 0);
    r2_[0] = 1;
    for (size_t i = 0; i < 64 * s_; ++i) {
      uint32_t carry = 0;
      for (size_t j = 0; j < s_; ++j) {
        uint32_t next = r2_[j] >> 31;
        r2_[j] = (r2_[j] << 1) | carry;
        carry = next;
      }
      if (carry || geq(r2_, n_)) subInPlace(r2_, n_);
    }
  }

  // a * b * R^-1 mod n (CIOS); inputs < n.
  Limbs mul(const Limbs& a, const Limbs& b) const {
    std::vector<uint32_t> t(s_ + 2, 0);
    for (size_t i = 0; i < s_; ++i) {
      uint64_t c = 0;
      for (size_t j = 0; j < s_; ++j) {
        uint64_t v = uint64_t(t[j]) + uint64_t(a[j]) * b[i] + c;
        t[j] = uint32_t(v);
        c = v >> 32;
      }
      uint64_t v = uint64_t(t[s_]) + c;
      t[s_] = uint32_t(v);
      t[s_ + 1] = uint32_t(v >> 32);

      uint32_t m = t[0] * n0inv_;
      c = (uint64_t(t[0]) + uint64_t(m) * n_[0]) >> 32;
      for (size_t j = 1; j < s_; ++j) {
        v = uint64_t(t[j]) + uint64_t(m) * n_[j] + c;
        t[j - 1] = uint32_t(v);
        c = v >> 32;
      }
      v = uint64_t(t[s_]) + c;
      t[s_ - 1] = uint32_t(v);
      t[s_] = t[s_ + 1] + uint32_t(v >> 32);
    }
    Limbs r(t.begin(), t.begin() + s_);
    if (t[s_] != 0 || geq(r, n_)) subInPlace(r, n_);
    return r;
  }

  // base^e mod n; base < n, e big-endian bytes.
  Limbs pow(const Limbs& base, const std::vector<uint8_t>& e) const {
    Limbs one(s_, 0);
    one[0] = 1;
    Limbs xm = mul(base, r2_);
    Limbs acc = mul(one, r2_);  // R mod n
    for (uint8_t byte : e)
      for (int bit = 7; bit >= 0; --bit) {
        acc = mul(acc, acc);
        if ((byte >> bit) & 1) acc = mul(acc, xm);
      }
    return mul(acc, one);
  }

 private:
  Limbs n_;
  size_t s_;
  uint32_t n0inv_;
  Limbs r2_;
};

}  // namespace

// ---------------------------------------------------------------- public API

std::string base64Encode(const uint8_t* data, size_t len) {
  std::string out;
  out.reserve((len + 2) / 3 * 4);
  for (size_t i = 0; i < len; i += 3) {
    size_t rem = std::min<size_t>(3, len - i);
    uint32_t v = uint32_t(data[i]) << 16;
    if (rem > 1) v |= uint32_t(data[i + 1]) << 8;
    if (rem > 2) v |= data[i + 2];
    out.push_back(kB64[(v >> 18) & 63]);
    out.push_back(kB64[(v >> 12) & 63]);
    out.push_back(rem > 1 ? kB64[(v >> 6) & 63] : '=');
    out.push_back(rem > 2 ? kB64[v & 63] : '=');
  }
  return out;
}

RsaPublicKey parseRsaPublicKeyPem(std::string_view pem) {
  struct Kind {
    std::string_view begin, end;
    bool spki;
  };
  static constexpr Kind kKinds[] = {
      {"-----BEGIN PUBLIC KEY-----", "-----END PUBLIC KEY-----", true},
      {"-----BEGIN RSA PUBLIC KEY-----", "-----END RSA PUBLIC KEY-----", false},
  };
  for (const Kind& kind : kKinds) {
    size_t b = pem.find(kind.begin);
    if (b == std::string_view::npos) continue;
    size_t bodyStart = b + kind.begin.size();
    size_t e = pem.find(kind.end, bodyStart);
    if (e == std::string_view::npos) fail("missing PEM end line");
    std::vector<uint8_t> der = base64Decode(pem.substr(bodyStart, e - bodyStart));
    RsaPublicKey key = kind.spki ? parseSpki(der.data(), der.size())
                                 : parsePkcs1(der.data(), der.size());
    // OAEP-SHA1 of a 64-hex-digit private key needs a modulus of at least 106 bytes; require 1024 bits.
    if (key.n.size() < 128 || key.n.size() > 2048) fail("modulus must be 1024-16384 bits");
    if ((key.n.back() & 1) == 0) fail("even modulus");
    if (key.e.size() > key.n.size() || (key.e.size() == 1 && key.e[0] < 3) ||
        (key.e.back() & 1) == 0)
      fail("bad public exponent");
    return key;
  }
  fail("no PEM public key block found");
}

std::vector<uint8_t> rsaOaepEncrypt(const RsaPublicKey& key, std::string_view msg,
                                    const std::array<uint8_t, 20>& seed) {
  const size_t k = key.modulusBytes();
  if (k < 2 * kHLen + 2 || msg.size() > k - 2 * kHLen - 2)
    throw std::invalid_argument("message too long for RSA key");

  // EM = 0x00 || maskedSeed || maskedDB, DB = lHash || PS || 0x01 || M
  std::vector<uint8_t> em(k, 0);
  uint8_t* emSeed = em.data() + 1;
  uint8_t* db = em.data() + 1 + kHLen;
  const size_t dbLen = k - kHLen - 1;
  Sha1().final(db);  // lHash = SHA1("")
  db[dbLen - msg.size() - 1] = 0x01;
  std::memcpy(db + dbLen - msg.size(), msg.data(), msg.size());
  std::memcpy(emSeed, seed.data(), kHLen);
  mgf1XorInto(emSeed, kHLen, db, dbLen);
  mgf1XorInto(db, dbLen, emSeed, kHLen);

  // c = m^e mod n; m < n since the top byte of EM is zero and n has k bytes.
  const size_t limbs = (k + 3) / 4;
  Montgomery mont(toLimbs(key.n, limbs));
  Limbs c = mont.pow(toLimbs(em, limbs), key.e);

  std::vector<uint8_t> out(k);
  for (size_t i = 0; i < k; ++i) out[k - 1 - i] = uint8_t(c[i / 4] >> (8 * (i % 4)));
  return out;
}

std::string rsaOaepEncryptBase64(const RsaPublicKey& key, std::string_view msg) {
  std::random_device rd;
  std::array<uint8_t, 20> seed;
  for (size_t i = 0; i < seed.size(); i += 4) {
    uint32_t v = rd();
    for (size_t j = 0; j < 4; ++j) seed[i + j] = uint8_t(v >> (8 * j));
  }
  std::vector<uint8_t> ct = rsaOaepEncrypt(key, msg, seed);
  return base64Encode(ct.data(), ct.size());
}

}  // namespace bsolver

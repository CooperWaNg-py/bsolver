#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "pool/rsa.h"

using namespace bsolver;

namespace {

// Example key from the official btcpuzzle.info pool client README.
constexpr const char* kExamplePubPem =
    "-----BEGIN PUBLIC KEY-----\n"
    "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCZkY6nLcQf2OUGKk2qZO34dWhh\n"
    "chPCe4htDumUDLo3+ZceZPBrFyKBeL7kuv+tp+sh5x78AhLuLGZjBdoV0Rzospqr\n"
    "DXadt62qyNJpa7+ajtE0HRJmepis7QAurX5HcP4zszkzq2Y0u219whtwm/WDdE3p\n"
    "5fuwuNwz8WkcQlXs9wIDAQAB\n"
    "-----END PUBLIC KEY-----\n";

// The same key in PKCS#1 form (`openssl rsa -pubin -RSAPublicKey_out`), with CRLF line endings.
constexpr const char* kExamplePkcs1Pem =
    "  -----BEGIN RSA PUBLIC KEY-----\r\n"
    "MIGJAoGBAJmRjqctxB/Y5QYqTapk7fh1aGFyE8J7iG0O6ZQMujf5lx5k8GsXIoF4\r\n"
    "vuS6/62n6yHnHvwCEu4sZmMF2hXRHOiymqsNdp23rarI0mlrv5qO0TQdEmZ6mKzt\r\n"
    "AC6tfkdw/jOzOTOrZjS7bX3CG3Cb9YN0Tenl+7C43DPxaRxCVez3AgMBAAE=\r\n"
    "-----END RSA PUBLIC KEY-----\r\n\r\n";

constexpr const char* kExampleModulusHex =
    "99918ea72dc41fd8e5062a4daa64edf87568617213c27b886d0ee9940cba37f9971e64f06b17228178bee4baff"
    "ada7eb21e71efc0212ee2c666305da15d11ce8b29aab0d769db7adaac8d2696bbf9a8ed1341d12667a98aced00"
    "2ead7e4770fe33b33933ab6634bb6d7dc21b709bf583744de9e5fbb0b8dc33f1691c4255ecf7";

std::string toHex(const std::vector<uint8_t>& v) {
  static const char* digits = "0123456789abcdef";
  std::string s;
  for (uint8_t b : v) {
    s.push_back(digits[b >> 4]);
    s.push_back(digits[b & 15]);
  }
  return s;
}

std::string b64(std::string_view s) {
  return base64Encode(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

}  // namespace

TEST_CASE("RSA: parses SubjectPublicKeyInfo PEM", "[rsa]") {
  RsaPublicKey key = parseRsaPublicKeyPem(kExamplePubPem);
  CHECK(key.modulusBytes() == 128);
  CHECK(toHex(key.n) == kExampleModulusHex);
  CHECK(key.e == std::vector<uint8_t>{0x01, 0x00, 0x01});
}

TEST_CASE("RSA: parses PKCS#1 PEM with CRLF and surrounding whitespace", "[rsa]") {
  RsaPublicKey key = parseRsaPublicKeyPem(kExamplePkcs1Pem);
  CHECK(toHex(key.n) == kExampleModulusHex);
  CHECK(key.e == std::vector<uint8_t>{0x01, 0x00, 0x01});
}

TEST_CASE("RSA: OAEP-SHA1 known answers", "[rsa]") {
  // Expected ciphertexts were computed by an independent RFC 8017 implementation
  // (Python hashlib + pow) and confirmed to decrypt with
  // `openssl pkeyutl -decrypt -pkeyopt rsa_padding_mode:oaep` using the example private key.
  RsaPublicKey key = parseRsaPublicKeyPem(kExamplePubPem);

  std::array<uint8_t, 20> seed{};
  for (size_t i = 0; i < seed.size(); ++i) seed[i] = uint8_t(i + 1);
  CHECK(toHex(rsaOaepEncrypt(key, "e9ae4933d6", seed)) ==
        "4df1c0791bc0cc15c720abaffb98f06663a39bc4e8c5f1c0ffba0b8b87a8f69be26da8b66f8aa5925f8147"
        "948d1dd88f69a18a07fb15d04e5ad38b32c0e6ced01b88070c4c58b4ce0f4f9ae23abc6592bf0ed8e29103"
        "bc6706d6d08cec2d576354d578e06631205a747492d8b7f2f9670f8e65d48e81994eb692fd6a67bc7c5d");

  // Maximum message length for a 1024-bit key: k - 42 = 86 bytes.
  std::array<uint8_t, 20> seed2;
  seed2.fill(0xa5);
  std::string msg2;
  for (int i = 0; i < 86; ++i) msg2.push_back(char('a' + i % 26));
  CHECK(toHex(rsaOaepEncrypt(key, msg2, seed2)) ==
        "428b7f0748aa68ec22270c78e975e5d0e3f98b0c1872380524319e0c0b766e04d934c337f30ddcec4403e6"
        "e7e637885c8798229d77fc9bb20bc57b9ef80bdd5a2c1a05fde85f8af2bfaa5cbb9ef500531cafae940c20"
        "dc23a002f323830c15ddb215b2cbfd8c671f6f4e49a025f9a7229fbd0dc8ae4611e273386b9fc1218bf7");
}

TEST_CASE("RSA: random-seed base64 output has the modulus length", "[rsa]") {
  RsaPublicKey key = parseRsaPublicKeyPem(kExamplePubPem);
  std::string a = rsaOaepEncryptBase64(key, "hello");
  std::string b = rsaOaepEncryptBase64(key, "hello");
  CHECK(a.size() == 172);  // 128 bytes -> 172 base64 chars
  CHECK(a.find('\n') == std::string::npos);
  CHECK(a != b);
}

TEST_CASE("RSA: base64 encoding", "[rsa]") {
  CHECK(b64("") == "");
  CHECK(b64("f") == "Zg==");
  CHECK(b64("fo") == "Zm8=");
  CHECK(b64("foo") == "Zm9v");
  CHECK(b64("foobar") == "Zm9vYmFy");
  const uint8_t high[] = {0xfb, 0xff, 0xbf};
  CHECK(base64Encode(high, 3) == "+/+/");
}

TEST_CASE("RSA: error cases", "[rsa]") {
  using Catch::Matchers::StartsWith;
  CHECK_THROWS_WITH(parseRsaPublicKeyPem("garbage"), StartsWith("invalid RSA public key: "));
  CHECK_THROWS_WITH(
      parseRsaPublicKeyPem("-----BEGIN PUBLIC KEY-----\nZm9vYmFy\n-----END PUBLIC KEY-----\n"),
      StartsWith("invalid RSA public key: "));
  CHECK_THROWS_WITH(
      parseRsaPublicKeyPem("-----BEGIN PUBLIC KEY-----\n!!!!\n-----END PUBLIC KEY-----\n"),
      StartsWith("invalid RSA public key: "));
  // Truncated body.
  std::string truncated(kExamplePubPem);
  truncated.erase(truncated.find("5fuwuNwz"), 25);
  CHECK_THROWS_WITH(parseRsaPublicKeyPem(truncated), StartsWith("invalid RSA public key: "));
  CHECK_THROWS_WITH(parseRsaPublicKeyPem("-----BEGIN PUBLIC KEY-----\nMIGf"),
                    StartsWith("invalid RSA public key: "));
  // 512-bit key: too small for OAEP-SHA1 of a 64-digit private key, rejected when configured.
  CHECK_THROWS_WITH(parseRsaPublicKeyPem("-----BEGIN PUBLIC KEY-----\n"
                                         "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAKkwhN3yc73MUMqVzdk5h8BDaXFpqm/y\n"
                                         "v4uR5XGkczSLOSxzf88xlwNI8vqZx3SvOwWBSI5keEtglezUmA4SOncCAwEAAQ==\n"
                                         "-----END PUBLIC KEY-----\n"),
                    "invalid RSA public key: modulus must be 1024-16384 bits");

  RsaPublicKey key = parseRsaPublicKeyPem(kExamplePubPem);
  std::array<uint8_t, 20> seed{};
  CHECK_NOTHROW(rsaOaepEncrypt(key, std::string(86, 'x'), seed));
  CHECK_THROWS_WITH(rsaOaepEncrypt(key, std::string(87, 'x'), seed),
                    "message too long for RSA key");
}

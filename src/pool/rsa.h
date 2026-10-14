#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace bsolver {

// RSA public key; both values big-endian without leading zero bytes.
struct RsaPublicKey {
  std::vector<uint8_t> n, e;
  size_t modulusBytes() const { return n.size(); }
};

// Accepts PEM SubjectPublicKeyInfo ("-----BEGIN PUBLIC KEY-----", as read by OpenSSL's
// PEM_read_bio_RSA_PUBKEY) and PKCS#1 ("-----BEGIN RSA PUBLIC KEY-----"). Text outside the
// PEM block (e.g. a preceding private key block) is ignored.
// Throws std::invalid_argument("invalid RSA public key: <detail>").
RsaPublicKey parseRsaPublicKeyPem(std::string_view pem);

// RSAES-OAEP (RFC 8017) with SHA-1, MGF1-SHA-1 and an empty label, i.e. OpenSSL's
// RSA_PKCS1_OAEP_PADDING, using an explicit 20-byte seed. Returns modulusBytes() bytes.
// Throws std::invalid_argument("message too long for RSA key") when msg.size() > k - 42.
std::vector<uint8_t> rsaOaepEncrypt(const RsaPublicKey& key, std::string_view msg,
                                    const std::array<uint8_t, 20>& seed);

// rsaOaepEncrypt with a seed from std::random_device, base64-encoded without newlines.
std::string rsaOaepEncryptBase64(const RsaPublicKey& key, std::string_view msg);

// Standard alphabet, '=' padded, no line breaks.
std::string base64Encode(const uint8_t* data, size_t len);

}  // namespace bsolver

#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/u256.h"
#include "pool/pool_config.h"

namespace bsolver {

// A range assigned by the pool: keys [hex||workloadStart, hex||workloadStart + workloadEnd].
struct PoolRange {
  std::string hex;
  std::string targetAddress;
  std::string workloadStart, workloadEnd;
  std::vector<std::string> proofAddresses;  // proofOfWorkAddresses, in submission order

  U256 start() const;
  U256 end() const;  // inclusive
};

class PoolError : public std::runtime_error {
 public:
  PoolError(const std::string& msg, bool retryable) : std::runtime_error(msg), retryable_(retryable) {}
  bool retryable() const { return retryable_; }

 private:
  bool retryable_;
};

// Parses the GET /puzzle/{n}/range response. Throws PoolError (not retryable) when fields are missing/invalid.
PoolRange parseRangeResponse(const std::string& body);

// SHA256 over the concatenated proof keys (64 uppercase hex digits each, in proofAddresses order), lowercase hex:
// the HashedProofKey header.
std::string hashedProofKey(const std::vector<std::string>& proofKeys);

// btcpuzzle.info REST API (https://btcpuzzle.info/api-documentation) plus the official client's notifications.
class PoolApi {
 public:
  PoolApi(PoolConfig config, std::string workerName, std::string gpuName, std::string telegramApiUrl);

  // GET /puzzle/{n}/range. Throws PoolError; validation errors (bad token format etc.) are not retryable.
  PoolRange getRange() const;
  // PUT /puzzle/{n}/range (flag the range as scanned). Throws PoolError; HTTP 4xx (except 429) is not retryable.
  void submitRange(const std::string& hex, const std::string& hashedProof, int gpuCount) const;
  // PATCH /puzzle/{n}/range/ping. Throws PoolError.
  void ping(const std::string& hex) const;
  // POST /puzzle/{n}/save-private-key with an RSA-encrypted key. Throws PoolError.
  void saveKey(const std::string& encryptedKey) const;

  // Notifications; return false on failure (never throw).
  bool telegram(const std::string& text) const;
  bool apiShare(const std::string& status, const std::map<std::string, std::string>& data) const;

  const std::string& securityHash() const { return securityHash_; }

 private:
  std::string puzzleUrl() const;

  PoolConfig config_;
  std::string workerName_, gpuName_, telegramApiUrl_, securityHash_;
};

}  // namespace bsolver

#include "pool/pool_api.h"

#include <algorithm>
#include <cctype>

#include <nlohmann/json.hpp>

#include "core/address.h"
#include "core/hex.h"
#include "core/sha256.h"
#include "pool/http_client.h"

namespace bsolver {

namespace {
using nlohmann::json;

bool isHex(const std::string& s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
}

// Best human-readable error from an API response (plain {"error"}, ASP.NET problem+json, or raw text).
std::string errorText(const HttpResponse& r) {
  std::string prefix = "HTTP " + std::to_string(r.status);
  try {
    json j = json::parse(r.body);
    if (j.is_object()) {
      if (j.contains("error") && j["error"].is_string()) return prefix + ": " + j["error"].get<std::string>();
      if (j.contains("message") && j["message"].is_string()) return prefix + ": " + j["message"].get<std::string>();
      std::string msg = j.value("title", std::string());
      if (j.contains("errors") && j["errors"].is_object()) {
        for (const auto& [field, list] : j["errors"].items()) {
          for (const auto& e : list) {
            if (e.is_string()) msg += (msg.empty() ? "" : " ") + field + ": " + e.get<std::string>();
          }
        }
      }
      if (!msg.empty()) return prefix + ": " + msg;
    }
  } catch (const json::exception&) {
  }
  std::string body = r.body.substr(0, 200);
  return body.empty() ? prefix : prefix + ": " + body;
}

bool retryableStatus(long status) { return status == 429 || status >= 500 || status == 0; }

HttpResponse send(HttpRequest req) {
  try {
    return httpRequest(req);
  } catch (const std::invalid_argument& ex) {
    throw PoolError(ex.what(), false);
  } catch (const std::runtime_error& ex) {
    throw PoolError(ex.what(), true);
  }
}
}  // namespace

U256 PoolRange::start() const { return U256::fromHex(hex + workloadStart); }

U256 PoolRange::end() const { return start() + U256::fromHex(workloadEnd); }

PoolRange parseRangeResponse(const std::string& body) {
  PoolRange r;
  try {
    json j = json::parse(body);
    r.hex = j.at("hex").get<std::string>();
    r.targetAddress = j.at("targetAddress").get<std::string>();
    r.workloadStart = j.at("workloadStart").get<std::string>();
    r.workloadEnd = j.at("workloadEnd").get<std::string>();
    r.proofAddresses = j.at("proofOfWorkAddresses").get<std::vector<std::string>>();
  } catch (const json::exception& ex) {
    throw PoolError(std::string("invalid range response: ") + ex.what(), false);
  }
  if (!isHex(r.hex) || !isHex(r.workloadStart) || !isHex(r.workloadEnd) ||
      r.hex.size() + r.workloadStart.size() > 64 || r.workloadEnd.size() > 64) {
    throw PoolError("invalid range response: bad hex/workload values", false);
  }
  try {
    decodeP2PKH(r.targetAddress);
    for (const auto& a : r.proofAddresses) decodeP2PKH(a);
  } catch (const std::invalid_argument& ex) {
    throw PoolError(std::string("invalid range response: ") + ex.what(), false);
  }
  return r;
}

std::string hashedProofKey(const std::vector<std::string>& proofKeys) {
  std::string concatenated;
  for (const auto& k : proofKeys) concatenated += k;
  uint8_t digest[32];
  sha256(reinterpret_cast<const uint8_t*>(concatenated.data()), concatenated.size(), digest);
  return hexEncode(digest, 32);
}

PoolApi::PoolApi(PoolConfig config, std::string workerName, std::string gpuName, std::string telegramApiUrl)
    : config_(std::move(config)),
      workerName_(std::move(workerName)),
      gpuName_(std::move(gpuName)),
      telegramApiUrl_(std::move(telegramApiUrl)),
      securityHash_(config_.securityHash()) {}

std::string PoolApi::puzzleUrl() const { return config_.apiUrl + "/puzzle/" + std::to_string(config_.targetPuzzle); }

PoolRange PoolApi::getRange() const {
  HttpRequest req;
  req.url = puzzleUrl() + "/range";
  req.headers = {{"UserToken", config_.userToken},   {"SecurityHash", securityHash_},
                 {"WorkerName", workerName_},        {"CustomRange", config_.customRange},
                 {"GPUName", gpuName_}};
  HttpResponse res = send(req);
  if (res.status != 200) {
    // 400 = request validation (token format, worker name, ...): retrying cannot help.
    throw PoolError("range request failed: " + errorText(res), res.status != 400 && retryableStatus(res.status));
  }
  return parseRangeResponse(res.body);
}

void PoolApi::submitRange(const std::string& hex, const std::string& hashedProof, int gpuCount) const {
  HttpRequest req;
  req.method = "PUT";
  req.url = puzzleUrl() + "/range";
  req.headers = {{"HEX", hex},
                 {"WorkerName", workerName_},
                 {"GPUName", gpuName_},
                 {"HashedProofKey", hashedProof},
                 {"UserToken", config_.userToken},
                 {"GPUCount", std::to_string(gpuCount)}};
  HttpResponse res = send(req);
  if (res.status != 200) throw PoolError("range submit failed: " + errorText(res), retryableStatus(res.status));
}

void PoolApi::ping(const std::string& hex) const {
  HttpRequest req;
  req.method = "PATCH";
  req.url = puzzleUrl() + "/range/ping";
  req.timeout = std::chrono::seconds(10);
  req.headers = {{"hex", hex}, {"userToken", config_.userToken}, {"workerName", workerName_}, {"gpuName", gpuName_}};
  HttpResponse res = send(req);
  if (res.status != 200) throw PoolError("ping failed: " + errorText(res), retryableStatus(res.status));
}

void PoolApi::saveKey(const std::string& encryptedKey) const {
  HttpRequest req;
  req.method = "POST";
  req.url = puzzleUrl() + "/save-private-key";
  req.headers = {{"Content-Type", "application/json"}, {"Accept", "application/json"}, {"UserToken", config_.userToken}};
  req.body = json{{"name", workerName_}, {"encryptedKey", encryptedKey}}.dump();
  HttpResponse res = send(req);
  if (res.status != 200) throw PoolError("save-private-key failed: " + errorText(res), retryableStatus(res.status));
}

bool PoolApi::telegram(const std::string& text) const {
  if (!config_.telegramShare) return false;
  HttpRequest req;
  req.method = "POST";
  req.url = telegramApiUrl_ + "/bot" + config_.telegramToken + "/sendMessage";
  req.headers = {{"Content-Type", "application/json"}};
  req.body = json{{"chat_id", config_.telegramChatId}, {"text", text}}.dump();
  try {
    return httpRequest(req).status == 200;
  } catch (const std::exception&) {
    return false;
  }
}

bool PoolApi::apiShare(const std::string& status, const std::map<std::string, std::string>& data) const {
  if (!config_.apiShare) return false;
  HttpRequest req;
  req.method = "POST";
  req.url = config_.apiShareUrl;
  req.timeout = std::chrono::seconds(10);
  req.headers.emplace_back("Status", status);
  for (const auto& [k, v] : data) req.headers.emplace_back(k, v);
  req.headers.emplace_back("Targetpuzzle", std::to_string(config_.targetPuzzle));
  req.headers.emplace_back("Workername", workerName_);
  req.headers.emplace_back("Customrange", config_.customRange);
  req.headers.emplace_back("Securityhash", securityHash_);
  try {
    HttpResponse res = httpRequest(req);
    std::string b = res.body;
    b.erase(0, b.find_first_not_of(" \t\r\n"));
    b.erase(b.find_last_not_of(" \t\r\n") + 1);
    std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return b == "true" || b == "1";
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace bsolver

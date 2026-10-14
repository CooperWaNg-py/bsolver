#include "pool/pool_state.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/hex.h"
#include "core/sha256.h"

namespace bsolver {

namespace {
using nlohmann::json;
constexpr int kVersion = 1;

json rangeToJson(const PoolRange& r) {
  return json{{"hex", r.hex},
              {"targetAddress", r.targetAddress},
              {"workloadStart", r.workloadStart},
              {"workloadEnd", r.workloadEnd},
              {"proofOfWorkAddresses", r.proofAddresses}};
}
}  // namespace

std::filesystem::path poolStatePath(const std::filesystem::path& stateDir, int puzzle) {
  return stateDir / ("pool-" + std::to_string(puzzle) + ".json");
}

std::string poolTokenId(const std::string& userToken) {
  uint8_t digest[32];
  sha256(reinterpret_cast<const uint8_t*>(userToken.data()), userToken.size(), digest);
  return hexEncode(digest, 8);
}

std::optional<PoolState> PoolState::load(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::nullopt;
  std::stringstream ss;
  ss << in.rdbuf();
  try {
    json j = json::parse(ss.str());
    if (j.at("version").get<int>() != kVersion) throw std::runtime_error("unsupported version");
    PoolState s;
    s.puzzle = j.at("puzzle").get<int>();
    s.tokenId = j.at("tokenId").get<std::string>();
    s.workerName = j.at("workerName").get<std::string>();
    s.rangesSubmitted = j.at("rangesSubmitted").get<uint64_t>();
    s.lastSubmittedHex = j.value("lastSubmittedHex", std::string());
    s.lastSubmittedAt = j.value("lastSubmittedAt", std::string());
    const json& a = j.at("assignment");
    if (!a.is_null()) {
      PoolAssignment pa;
      // Re-validate through the same parser as live API responses.
      pa.range = parseRangeResponse(a.at("range").dump());
      pa.proofKeys = a.at("proofKeys").get<std::map<std::string, std::string>>();
      pa.scanComplete = a.at("scanComplete").get<bool>();
      pa.assignedAt = a.value("assignedAt", std::string());
      s.assignment = std::move(pa);
    }
    return s;
  } catch (const std::exception& ex) {
    throw std::runtime_error("pool state " + path.string() + " is unreadable: " + ex.what());
  }
}

void PoolState::save(const std::filesystem::path& path) const {
  json a = nullptr;
  if (assignment) {
    a = json{{"range", rangeToJson(assignment->range)},
             {"proofKeys", assignment->proofKeys},
             {"scanComplete", assignment->scanComplete},
             {"assignedAt", assignment->assignedAt}};
  }
  json j{{"version", kVersion},
         {"puzzle", puzzle},
         {"tokenId", tokenId},
         {"workerName", workerName},
         {"assignment", a},
         {"rangesSubmitted", rangesSubmitted},
         {"lastSubmittedHex", lastSubmittedHex},
         {"lastSubmittedAt", lastSubmittedAt}};
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::filesystem::path tmp = path;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    out << j.dump(2) << "\n";
    if (!out) throw std::runtime_error("cannot write pool state " + tmp.string());
  }
  std::filesystem::rename(tmp, path);
}

}  // namespace bsolver

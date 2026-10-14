#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace bsolver {

struct FoundKey {
  std::string time;  // ISO8601 UTC
  std::optional<int> puzzle;
  std::string address;
  std::string pubkey;   // hex
  std::string privkey;  // hex64
  std::string wif;
  std::string algorithm;
  std::string backend;
};

void to_json(nlohmann::json& j, const FoundKey& k);
void from_json(const nlohmann::json& j, FoundKey& k);

// Appends one JSON line to the results file (creating parent directories).
void appendResult(const std::filesystem::path& path, const FoundKey& key);
// All lines of the results file; empty when missing. Malformed lines are skipped.
std::vector<FoundKey> readResults(const std::filesystem::path& path);

// "FOUND puzzle #N address <a> privkey <hex64> WIF <wif>"
std::string foundLine(const FoundKey& key);

// Current time as ISO8601 with 'Z' suffix.
std::string isoTimeNow();

}  // namespace bsolver

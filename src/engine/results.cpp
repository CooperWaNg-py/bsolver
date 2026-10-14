#include "engine/results.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace bsolver {

void to_json(nlohmann::json& j, const FoundKey& k) {
  j = nlohmann::json{
      {"time", k.time},
      {"puzzle", k.puzzle ? nlohmann::json(*k.puzzle) : nlohmann::json(nullptr)},
      {"address", k.address},
      {"pubkey", k.pubkey},
      {"privkey", k.privkey},
      {"wif", k.wif},
      {"algorithm", k.algorithm},
      {"backend", k.backend},
  };
}

void from_json(const nlohmann::json& j, FoundKey& k) {
  k.time = j.at("time").get<std::string>();
  k.puzzle = j.at("puzzle").is_null() ? std::nullopt : std::optional<int>(j.at("puzzle").get<int>());
  k.address = j.at("address").get<std::string>();
  k.pubkey = j.at("pubkey").get<std::string>();
  k.privkey = j.at("privkey").get<std::string>();
  k.wif = j.at("wif").get<std::string>();
  k.algorithm = j.at("algorithm").get<std::string>();
  k.backend = j.at("backend").get<std::string>();
}

void appendResult(const std::filesystem::path& path, const FoundKey& key) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::app | std::ios::binary);
  if (!out) throw std::runtime_error("cannot open results file " + path.string());
  out << nlohmann::json(key).dump() << "\n";
}

std::vector<FoundKey> readResults(const std::filesystem::path& path) {
  std::vector<FoundKey> out;
  std::ifstream in(path, std::ios::binary);
  if (!in) return out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    try {
      out.push_back(nlohmann::json::parse(line).get<FoundKey>());
    } catch (const nlohmann::json::exception&) {
    }
  }
  return out;
}

std::string foundLine(const FoundKey& key) {
  std::string s = "FOUND ";
  if (key.puzzle) s += "puzzle #" + std::to_string(*key.puzzle) + " ";
  s += "address " + key.address + " privkey " + key.privkey + " WIF " + key.wif;
  return s;
}

std::string isoTimeNow() {
  std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

}  // namespace bsolver

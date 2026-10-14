#include "pool/pool_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "core/hex.h"
#include "core/sha256.h"

namespace bsolver {

namespace {
std::string trim(std::string_view s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string_view::npos) return {};
  size_t b = s.find_last_not_of(" \t\r\n");
  return std::string(s.substr(a, b - a + 1));
}

bool parseBool(const std::string& v) {
  std::string s = v;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s == "true" || s == "1" || s == "yes";
}

int parseInt(const std::string& key, const std::string& v) {
  try {
    size_t used = 0;
    int n = std::stoi(v, &used);
    if (used != v.size()) throw std::invalid_argument(v);
    return n;
  } catch (const std::exception&) {
    throw std::invalid_argument("pool config: " + key + " must be a number, got '" + v + "'");
  }
}
}  // namespace

PoolConfig PoolConfig::parse(std::string_view text, std::vector<std::string>* warnings) {
  PoolConfig c;
  std::istringstream in{std::string(text)};
  std::string raw;
  int lineNo = 0;
  while (std::getline(in, raw)) {
    ++lineNo;
    std::string line = trim(raw);
    if (line.empty() || line[0] == '#' || line[0] == ';') continue;
    size_t eq = line.find('=');
    if (eq == std::string::npos) {
      if (warnings) warnings->push_back("pool config line " + std::to_string(lineNo) + " ignored: no '='");
      continue;
    }
    std::string key = trim(std::string_view(line).substr(0, eq));
    std::string value = trim(std::string_view(line).substr(eq + 1));
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') value = value.substr(1, value.size() - 2);

    if (key == "user_token") {
      c.userToken = value;
    } else if (key == "worker_name") {
      c.workerName = value;
    } else if (key == "target_puzzle") {
      c.targetPuzzle = parseInt(key, value);
    } else if (key == "gpu_index") {
      parseInt(key, value);  // official-client setting; bsolver selects devices with --devices
    } else if (key == "untrusted_computer") {
      c.untrustedComputer = parseBool(value);
    } else if (key == "public_key") {
      c.setPublicKey(value);
    } else if (key == "telegram_share") {
      c.telegramShare = parseBool(value);
    } else if (key == "telegram_token") {
      c.telegramToken = value;
    } else if (key == "telegram_chat_id") {
      c.telegramChatId = value;
    } else if (key == "api_share") {
      c.apiShare = parseBool(value);
    } else if (key == "api_share_url") {
      c.apiShareUrl = value;
    } else if (key == "save_key") {
      c.saveKey = parseBool(value);
    } else if (key == "custom_range") {
      c.customRange = value.empty() ? "none" : value;
    } else if (key == "build_hash") {
      c.buildHash = value;
    } else if (key == "api_url") {
      c.apiUrl = value;
    } else if (warnings) {
      warnings->push_back("pool config: unknown key '" + key + "' ignored");
    }
  }
  return c;
}

PoolConfig PoolConfig::loadFile(const std::filesystem::path& path, std::vector<std::string>* warnings) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::invalid_argument("cannot open pool config " + path.string());
  std::stringstream ss;
  ss << in.rdbuf();
  return parse(ss.str(), warnings);
}

void PoolConfig::setPublicKey(std::string_view raw) {
  publicKey.assign(raw);
  std::replace(publicKey.begin(), publicKey.end(), '|', '\n');
  std::replace(publicKey.begin(), publicKey.end(), '@', '\n');
}

void PoolConfig::validate() const {
  auto fail = [](const std::string& msg) { throw std::invalid_argument(msg); };
  if (userToken.empty()) fail("user_token is required (get it from your btcpuzzle.info account)");
  if (userToken == "YOUR_TOKEN_HERE") fail("please set your actual user token in pool.conf");
  if (userToken.size() != 128) fail("user_token must be 128 characters long");
  if (targetPuzzle < 1 || targetPuzzle > 160) fail("target_puzzle must be 1-160");
  if (workerName.size() > 15) fail("worker_name must be at most 15 characters");
  for (unsigned char ch : workerName) {
    if (ch < 0x21 || ch > 0x7e) fail("worker_name must use printable ASCII characters without spaces");
  }
  if (untrustedComputer && !telegramShare && !apiShare && publicKey.empty()) {
    fail("untrusted_computer requires telegram_share, api_share or public_key; otherwise found keys would be lost");
  }
  if (telegramShare && (telegramToken.empty() || telegramChatId.empty())) {
    fail("telegram_share is enabled but telegram_token or telegram_chat_id is missing");
  }
  if (apiShare && apiShareUrl.empty()) fail("api_share is enabled but api_share_url is missing");
  if (saveKey && (!untrustedComputer || publicKey.empty())) {
    fail("save_key needs untrusted_computer=true and public_key: the key is RSA-encrypted before it is saved");
  }
}

std::string PoolConfig::securityHash() const {
  std::string s = std::to_string(untrustedComputer ? 1 : 0) + std::to_string(apiShare ? 1 : 0) + telegramChatId +
                  telegramToken + userToken + apiShareUrl + std::to_string(targetPuzzle) + buildHash;
  uint8_t digest[32];
  sha256(reinterpret_cast<const uint8_t*>(s.data()), s.size(), digest);
  return hexEncode(digest, 32);
}

}  // namespace bsolver

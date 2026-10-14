#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace bsolver {

// Settings of the btcpuzzle.info pool client. Field names follow the official client's pool.conf
// (https://github.com/ilkerccom/btcpuzzle), so the same file works for both clients.
struct PoolConfig {
  std::string apiUrl = "https://api.btcpuzzle.info";
  std::string userToken;   // user_token (128 characters)
  std::string workerName;  // worker_name (1-15 characters; generated and remembered when empty)
  int targetPuzzle = 71;   // target_puzzle (38 is the test pool)
  bool untrustedComputer = false;  // untrusted_computer: found keys leave this machine RSA-encrypted only
  std::string publicKey;           // public_key: RSA public key PEM ('|' or '@' as line separators)
  bool telegramShare = false;      // telegram_share / telegram_token / telegram_chat_id
  std::string telegramToken, telegramChatId;
  bool apiShare = false;  // api_share / api_share_url (custom webhook)
  std::string apiShareUrl;
  bool saveKey = false;               // save_key: store the encrypted key in the btcpuzzle.info account
  std::string customRange = "none";   // custom_range
  std::string buildHash;              // build_hash: the official client's BUILD_HASH (part of the security hash)

  // Parses pool.conf text (`key = value`, '#'/';' comments, optional double quotes). gpu_index is
  // accepted and ignored (use --devices); other unknown keys are reported in `warnings`.
  // Throws std::invalid_argument for malformed numbers.
  static PoolConfig parse(std::string_view text, std::vector<std::string>* warnings = nullptr);
  static PoolConfig loadFile(const std::filesystem::path& path, std::vector<std::string>* warnings = nullptr);

  // Stores a PEM key, turning the official client's '|' / '@' line separators into newlines.
  void setPublicKey(std::string_view raw);

  // Throws std::invalid_argument with the first configuration error.
  void validate() const;

  // SHA256(untrusted + apiShare + telegramChatId + telegramToken + userToken + apiShareUrl + puzzle + buildHash),
  // the SecurityHash header computed exactly like the official client.
  std::string securityHash() const;
};

}  // namespace bsolver

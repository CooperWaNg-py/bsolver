#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>

#include "pool/pool_api.h"

namespace bsolver {

// The range currently being worked on, persisted so an interrupted range is resumed instead of lost.
// The scan position itself lives in the regular scan checkpoint (same state directory).
struct PoolAssignment {
  PoolRange range;
  std::map<std::string, std::string> proofKeys;  // proof address -> 64 uppercase hex digits
  bool scanComplete = false;                     // scanned; only the submission is outstanding
  std::string assignedAt;
};

// <stateDir>/pool-<puzzle>.json, written atomically after every change.
struct PoolState {
  int puzzle = 0;
  std::string tokenId;     // first 16 hex digits of SHA256(user token): detects a changed account
  std::string workerName;  // remembered so a generated name survives restarts
  std::optional<PoolAssignment> assignment;
  uint64_t rangesSubmitted = 0;
  std::string lastSubmittedHex;
  std::string lastSubmittedAt;

  // std::nullopt when the file does not exist; throws std::runtime_error when it is unreadable.
  static std::optional<PoolState> load(const std::filesystem::path& path);
  // Atomic: writes a uniquely named `<path>.<random>.tmp` and renames it over `path`.
  void save(const std::filesystem::path& path) const;
};

std::filesystem::path poolStatePath(const std::filesystem::path& stateDir, int puzzle);
// Lock file a PoolSession holds for its whole run, so two processes never share one puzzle's pool state.
std::filesystem::path poolLockPath(const std::filesystem::path& stateDir, int puzzle);
std::string poolTokenId(const std::string& userToken);

}  // namespace bsolver

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "catalog/catalog.h"
#include "core/address.h"
#include "core/point.h"
#include "core/u256.h"
#include "engine/types.h"

namespace bsolver {

enum class Algorithm { Auto, Scan, Bsgs, Kangaroo };

// Job request; identical for CLI and HTTP API (JSON field names match the members).
struct JobSpec {
  std::optional<int> puzzle;
  std::vector<std::string> addresses;
  std::optional<std::string> pubkey;
  std::optional<std::string> rangeStart, rangeEnd;
  std::optional<int> bits;
  Algorithm algorithm = Algorithm::Auto;
  Order order = Order::Sequential;
  AddressMode addressMode = AddressMode::Compressed;
  std::string backend = "auto";
  unsigned threads = 0;  // 0 = hardware_concurrency
  std::vector<int> devices;
  uint32_t bsgsMemoryMb = 1024;
  std::optional<uint32_t> dpBits;
  bool checkpoint = true;
  bool fresh = false;
  bool stopOnFound = true;
};

// Throws JobError on unknown fields or invalid values.
void to_json(nlohmann::json& j, const JobSpec& s);
void from_json(const nlohmann::json& j, JobSpec& s);

struct ResolvedJob {
  Algo algorithm = Algo::Scan;
  Order order = Order::Sequential;
  AddressMode addressMode = AddressMode::Compressed;
  U256 rangeStart, rangeEnd;
  std::vector<Hash160> targets;  // sorted, unique
  std::optional<Affine> pubkey;
  std::optional<int> puzzle;
  std::string identity;  // checkpoint naming/matching
  std::vector<std::string> warnings;

  U256 width() const { return rangeEnd - rangeStart + 1; }
};

// Limits enforced by resolve().
constexpr unsigned kMaxThreads = 1024;
constexpr uint32_t kMaxDpBits = 60;
// BSGS table limit: the largest power of two not above half the physical memory, in MiB (fallback when the
// physical memory cannot be determined).
constexpr uint32_t kFallbackMaxBsgsMemoryMb = 65536;
uint32_t maxBsgsMemoryMb();

// Validates the spec against the catalog. Throws JobError.
ResolvedJob resolve(const JobSpec& spec, const Catalog& catalog);

}  // namespace bsolver

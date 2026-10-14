#include "engine/job_spec.h"

#include <algorithm>
#include <set>

#include <nlohmann/json.hpp>

#include "core/hex.h"
#include "core/scalar.h"
#include "engine/errors.h"

namespace bsolver {

namespace {
std::string_view algorithmName(Algorithm a) {
  switch (a) {
    case Algorithm::Auto: return "auto";
    case Algorithm::Scan: return "scan";
    case Algorithm::Bsgs: return "bsgs";
    case Algorithm::Kangaroo: return "kangaroo";
  }
  return "auto";
}

template <class T>
T getAs(const nlohmann::json& v, const char* field) {
  try {
    return v.get<T>();
  } catch (const nlohmann::json::exception&) {
    throw JobError(std::string("invalid value for '") + field + "'");
  }
}

template <class T>
std::optional<T> optAs(const nlohmann::json& v, const char* field) {
  if (v.is_null()) return std::nullopt;
  return getAs<T>(v, field);
}

std::string enumString(const nlohmann::json& v, const char* field) {
  if (!v.is_string()) throw JobError(std::string("invalid value for '") + field + "'");
  return v.get<std::string>();
}
}  // namespace

void to_json(nlohmann::json& j, const JobSpec& s) {
  auto opt = [](const auto& o) { return o ? nlohmann::json(*o) : nlohmann::json(nullptr); };
  j = nlohmann::json{
      {"puzzle", opt(s.puzzle)},
      {"addresses", s.addresses},
      {"pubkey", opt(s.pubkey)},
      {"rangeStart", opt(s.rangeStart)},
      {"rangeEnd", opt(s.rangeEnd)},
      {"bits", opt(s.bits)},
      {"algorithm", algorithmName(s.algorithm)},
      {"order", toString(s.order)},
      {"addressMode", toString(s.addressMode)},
      {"backend", s.backend},
      {"threads", s.threads},
      {"devices", s.devices},
      {"bsgsMemoryMb", s.bsgsMemoryMb},
      {"dpBits", opt(s.dpBits)},
      {"checkpoint", s.checkpoint},
      {"fresh", s.fresh},
      {"stopOnFound", s.stopOnFound},
  };
}

void from_json(const nlohmann::json& j, JobSpec& s) {
  if (!j.is_object()) throw JobError("job spec must be a JSON object");
  s = JobSpec{};
  for (const auto& [k, v] : j.items()) {
    if (k == "puzzle") {
      s.puzzle = optAs<int>(v, "puzzle");
    } else if (k == "addresses") {
      if (!v.is_null()) s.addresses = getAs<std::vector<std::string>>(v, "addresses");
    } else if (k == "pubkey") {
      s.pubkey = optAs<std::string>(v, "pubkey");
    } else if (k == "rangeStart") {
      s.rangeStart = optAs<std::string>(v, "rangeStart");
    } else if (k == "rangeEnd") {
      s.rangeEnd = optAs<std::string>(v, "rangeEnd");
    } else if (k == "bits") {
      s.bits = optAs<int>(v, "bits");
    } else if (k == "algorithm") {
      std::string a = enumString(v, "algorithm");
      if (a == "auto") {
        s.algorithm = Algorithm::Auto;
      } else if (a == "scan") {
        s.algorithm = Algorithm::Scan;
      } else if (a == "bsgs") {
        s.algorithm = Algorithm::Bsgs;
      } else if (a == "kangaroo") {
        s.algorithm = Algorithm::Kangaroo;
      } else {
        throw JobError("invalid algorithm '" + a + "' (auto|scan|bsgs|kangaroo)");
      }
    } else if (k == "order") {
      std::string o = enumString(v, "order");
      auto parsed = parseOrder(o);
      if (!parsed) throw JobError("invalid order '" + o + "' (sequential|random)");
      s.order = *parsed;
    } else if (k == "addressMode") {
      std::string m = enumString(v, "addressMode");
      auto parsed = parseAddressMode(m);
      if (!parsed) throw JobError("invalid addressMode '" + m + "' (compressed|uncompressed|both)");
      s.addressMode = *parsed;
    } else if (k == "backend") {
      s.backend = enumString(v, "backend");
    } else if (k == "threads") {
      s.threads = getAs<unsigned>(v, "threads");
    } else if (k == "devices") {
      if (!v.is_null()) s.devices = getAs<std::vector<int>>(v, "devices");
    } else if (k == "bsgsMemoryMb") {
      s.bsgsMemoryMb = getAs<uint32_t>(v, "bsgsMemoryMb");
    } else if (k == "dpBits") {
      s.dpBits = optAs<uint32_t>(v, "dpBits");
    } else if (k == "checkpoint") {
      s.checkpoint = getAs<bool>(v, "checkpoint");
    } else if (k == "fresh") {
      s.fresh = getAs<bool>(v, "fresh");
    } else if (k == "stopOnFound") {
      s.stopOnFound = getAs<bool>(v, "stopOnFound");
    } else {
      throw JobError("unknown field '" + k + "'");
    }
  }
}

ResolvedJob resolve(const JobSpec& spec, const Catalog& catalog) {
  ResolvedJob r;
  r.addressMode = spec.addressMode;
  r.order = spec.order;

  if (spec.puzzle && (!spec.addresses.empty() || spec.pubkey)) {
    throw JobError("puzzle and explicit addresses/pubkey are mutually exclusive");
  }
  if (spec.bits && (*spec.bits < 1 || *spec.bits > 256)) throw JobError("bits must be 1-256");

  const Puzzle* puzzle = nullptr;
  std::set<Hash160> targets;
  std::optional<std::string> pubkeyText;
  if (spec.puzzle) {
    puzzle = catalog.find(*spec.puzzle);
    if (!puzzle) throw JobError("unknown puzzle " + std::to_string(*spec.puzzle) + " (valid 1-160)");
    r.puzzle = puzzle->id;
    targets.insert(puzzle->hash160);
    r.pubkey = puzzle->pubkey;
    if (puzzle->solved) {
      r.warnings.push_back("puzzle " + std::to_string(puzzle->id) + " is already solved (key " +
                           (puzzle->privkey ? puzzle->privkey->toHex() : std::string("unknown")) + ")");
    }
  } else {
    for (const auto& a : spec.addresses) {
      try {
        targets.insert(decodeP2PKH(a));
      } catch (const std::invalid_argument& ex) {
        throw JobError(std::string(ex.what()) + ": " + a);
      }
    }
    if (spec.pubkey) {
      try {
        r.pubkey = parsePubkey(*spec.pubkey);
      } catch (const std::invalid_argument& ex) {
        throw JobError(ex.what());
      }
      pubkeyText = *spec.pubkey;
    }
    if (targets.empty() && !r.pubkey) throw JobError("no target: give --puzzle, --address or --pubkey");
  }

  // Range: explicit > bits > puzzle.
  try {
    if (spec.rangeStart || spec.rangeEnd) {
      if (!spec.rangeStart || !spec.rangeEnd) throw JobError("range needs both start and end");
      r.rangeStart = U256::fromHex(*spec.rangeStart);
      r.rangeEnd = U256::fromHex(*spec.rangeEnd);
    } else if (spec.bits) {
      unsigned b = static_cast<unsigned>(*spec.bits);
      r.rangeStart = U256::fromU64(1) << (b - 1);
      r.rangeEnd = (b == 256 ? U256{} : (U256::fromU64(1) << b)) - 1;
    } else if (puzzle) {
      r.rangeStart = puzzle->rangeStart;
      r.rangeEnd = puzzle->rangeEnd;
    } else {
      throw JobError("no range: give --range, --bits or --puzzle");
    }
  } catch (const std::invalid_argument& ex) {
    throw JobError(ex.what());
  }
  if (r.rangeStart.isZero()) throw JobError("range start must be >= 1");
  if (r.rangeEnd < r.rangeStart) throw JobError("range end must be >= range start");
  if (!(r.rangeEnd < kOrderN)) throw JobError("range end must be below the secp256k1 order");

  const U256 width = r.width();
  const bool wide = width.bitLength() > 32;  // width >= 2^32
  switch (spec.algorithm) {
    case Algorithm::Auto:
      // Random order only exists for scan, so asking for it selects scan.
      if (spec.order == Order::Random || !r.pubkey) {
        r.algorithm = Algo::Scan;
      } else {
        r.algorithm = wide ? Algo::Kangaroo : Algo::Bsgs;
      }
      break;
    case Algorithm::Scan: r.algorithm = Algo::Scan; break;
    case Algorithm::Bsgs: r.algorithm = Algo::Bsgs; break;
    case Algorithm::Kangaroo: r.algorithm = Algo::Kangaroo; break;
  }
  if (r.algorithm != Algo::Scan && !r.pubkey) {
    throw JobError("algorithm " + std::string(toString(r.algorithm)) + " requires a public key");
  }
  if (r.algorithm == Algo::Kangaroo && !wide) {
    throw JobError("kangaroo requires a range width >= 2^32; use bsgs or scan");
  }
  if (r.algorithm != Algo::Scan && spec.order == Order::Random) throw JobError("--order applies only to scan");

  if (r.algorithm == Algo::Scan && targets.empty()) {
    // Only a pubkey: hash it in its given encoding.
    bool compressed = pubkeyText->size() == 66;
    targets.insert(hash160Of(*r.pubkey, compressed));
  }
  r.targets.assign(targets.begin(), targets.end());

  r.identity = std::string(toString(r.algorithm));
  if (r.algorithm == Algo::Scan) r.identity += "|" + std::string(toString(r.order));
  r.identity += "|" + r.rangeStart.toHex() + "|" + r.rangeEnd.toHex() + "|";
  if (r.algorithm == Algo::Scan) {
    for (size_t i = 0; i < r.targets.size(); ++i) {
      if (i) r.identity += ",";
      r.identity += hexEncode(r.targets[i].data(), 20);
    }
  } else {
    r.identity += pubkeyHex(*r.pubkey);
  }
  r.identity += "|" + std::string(toString(r.addressMode));
  return r;
}

}  // namespace bsolver

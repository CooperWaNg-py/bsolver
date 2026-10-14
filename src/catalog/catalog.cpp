#include "catalog/catalog.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "assets/assets.h"
#include "core/hex.h"

namespace bsolver {

namespace {
[[noreturn]] void fail(const std::string& detail) { throw std::runtime_error("catalog: " + detail); }

Puzzle parseEntry(const nlohmann::json& e, size_t index) {
  const std::string where = "entry " + std::to_string(index);
  if (!e.is_object()) fail(where + " is not an object");
  for (const char* k : {"id", "address", "hash160", "pubkey", "privkey", "solved"}) {
    if (!e.contains(k)) fail(where + " missing '" + k + "'");
  }
  Puzzle p;
  if (!e["id"].is_number_integer()) fail(where + ": id must be an integer");
  p.id = e["id"].get<int>();
  if (p.id < 1 || p.id > 256) fail(where + ": id out of range");
  const std::string tag = "puzzle " + std::to_string(p.id);
  if (!e["address"].is_string() || !e["hash160"].is_string() || !e["solved"].is_boolean()) {
    fail(tag + ": address/hash160 must be strings and solved a boolean");
  }
  p.address = e["address"].get<std::string>();
  p.solved = e["solved"].get<bool>();
  auto h = hexDecode(e["hash160"].get<std::string>());
  if (!h || h->size() != 20) fail(tag + ": hash160 must be 40 hex digits");
  std::copy(h->begin(), h->end(), p.hash160.begin());
  try {
    if (decodeP2PKH(p.address) != p.hash160) fail(tag + ": address does not match hash160");
  } catch (const std::invalid_argument& ex) {
    fail(tag + ": " + ex.what());
  }
  if (!e["pubkey"].is_null()) {
    if (!e["pubkey"].is_string()) fail(tag + ": pubkey must be a string or null");
    try {
      p.pubkey = parsePubkey(e["pubkey"].get<std::string>());
    } catch (const std::invalid_argument& ex) {
      fail(tag + ": " + ex.what());
    }
  }
  if (!e["privkey"].is_null()) {
    if (!e["privkey"].is_string()) fail(tag + ": privkey must be a string or null");
    try {
      p.privkey = U256::fromHex(e["privkey"].get<std::string>());
    } catch (const std::invalid_argument& ex) {
      fail(tag + ": " + ex.what());
    }
  }
  p.rangeStart = U256::fromU64(1) << static_cast<unsigned>(p.id - 1);
  p.rangeEnd = (p.id == 256 ? U256{} : (U256::fromU64(1) << static_cast<unsigned>(p.id))) - 1;
  return p;
}
}  // namespace

Catalog Catalog::fromJson(std::string_view text) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& ex) {
    fail(ex.what());
  }
  if (!j.is_array()) fail("top level must be an array");
  Catalog c;
  for (size_t i = 0; i < j.size(); ++i) c.puzzles_.push_back(parseEntry(j[i], i));
  std::sort(c.puzzles_.begin(), c.puzzles_.end(), [](const Puzzle& a, const Puzzle& b) { return a.id < b.id; });
  for (size_t i = 1; i < c.puzzles_.size(); ++i) {
    if (c.puzzles_[i].id == c.puzzles_[i - 1].id) fail("duplicate id " + std::to_string(c.puzzles_[i].id));
  }
  return c;
}

Catalog Catalog::builtin() { return fromJson(assets::kPuzzleCatalogJson); }

Catalog Catalog::fromFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot open " + path.string());
  std::stringstream ss;
  ss << in.rdbuf();
  return fromJson(ss.str());
}

const Puzzle* Catalog::find(int id) const {
  auto it = std::lower_bound(puzzles_.begin(), puzzles_.end(), id, [](const Puzzle& p, int v) { return p.id < v; });
  return it != puzzles_.end() && it->id == id ? &*it : nullptr;
}

nlohmann::json puzzleToJson(const Puzzle& p) {
  return nlohmann::json{
      {"id", p.id},
      {"address", p.address},
      {"hash160", hexEncode(p.hash160.data(), 20)},
      {"pubkey", p.pubkey ? nlohmann::json(pubkeyHex(*p.pubkey)) : nlohmann::json(nullptr)},
      {"privkey", p.privkey ? nlohmann::json(p.privkey->toHex()) : nlohmann::json(nullptr)},
      {"solved", p.solved},
      {"bits", p.id},
      {"rangeStart", p.rangeStart.toHex()},
      {"rangeEnd", p.rangeEnd.toHex()},
  };
}

}  // namespace bsolver

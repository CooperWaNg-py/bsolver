#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "core/address.h"
#include "core/point.h"
#include "core/u256.h"

namespace bsolver {

struct Puzzle {
  int id = 0;
  std::string address;
  Hash160 hash160{};
  std::optional<Affine> pubkey;
  std::optional<U256> privkey;
  bool solved = false;
  U256 rangeStart, rangeEnd;  // [2^(id-1), 2^id - 1]
};

class Catalog {
 public:
  // The embedded data/puzzles.json snapshot.
  static Catalog builtin();
  // Same schema as data/puzzles.json. Throws std::runtime_error("catalog: <detail>").
  static Catalog fromFile(const std::filesystem::path& path);
  static Catalog fromJson(std::string_view text);

  const Puzzle* find(int id) const;
  const std::vector<Puzzle>& all() const { return puzzles_; }

 private:
  std::vector<Puzzle> puzzles_;
};

// API/CLI representation: catalog fields plus bits, rangeStart, rangeEnd (hex).
nlohmann::json puzzleToJson(const Puzzle& p);

}  // namespace bsolver

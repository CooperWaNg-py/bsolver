#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "assets/assets.h"

TEST_CASE("embedded puzzle catalog parses with 160 entries") {
  using bsolver::assets::kPuzzleCatalogJson;
  REQUIRE(kPuzzleCatalogJson.data()[kPuzzleCatalogJson.size()] == '\0');
  auto j = nlohmann::json::parse(kPuzzleCatalogJson);
  REQUIRE(j.is_array());
  REQUIRE(j.size() == 160);
  CHECK(j[70]["id"] == 71);
}

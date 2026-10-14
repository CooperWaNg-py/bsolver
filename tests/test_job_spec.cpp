#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <nlohmann/json.hpp>

#include "catalog/catalog.h"
#include "engine/errors.h"
#include "engine/job_spec.h"

using namespace bsolver;

namespace {
const Catalog& catalog() {
  static Catalog c = Catalog::builtin();
  return c;
}
JobSpec spec(const char* json) { return nlohmann::json::parse(json).get<JobSpec>(); }
}  // namespace

TEST_CASE("builtin catalog") {
  REQUIRE(catalog().all().size() == 160);
  const Puzzle* p71 = catalog().find(71);
  REQUIRE(p71);
  CHECK(p71->address == "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU");
  CHECK_FALSE(p71->solved);
  CHECK_FALSE(p71->pubkey);
  CHECK(catalog().find(140)->pubkey);
  CHECK(catalog().find(0) == nullptr);
}

TEST_CASE("resolve puzzle 71 to a scan of its range") {
  auto r = resolve(spec(R"({"puzzle":71})"), catalog());
  CHECK(r.algorithm == Algo::Scan);
  CHECK(r.rangeStart.toHex() == "400000000000000000");
  CHECK(r.rangeEnd.toHex() == "7fffffffffffffffff");
  CHECK(r.targets.size() == 1);
}

TEST_CASE("resolve puzzle 140 to kangaroo") {
  CHECK(resolve(spec(R"({"puzzle":140})"), catalog()).algorithm == Algo::Kangaroo);
  CHECK(resolve(spec(R"({"puzzle":140,"bits":20})"), catalog()).algorithm == Algo::Bsgs);
  CHECK(resolve(spec(R"({"puzzle":140,"order":"random"})"), catalog()).algorithm == Algo::Scan);
}

TEST_CASE("resolve errors") {
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":71,"algorithm":"bsgs"})"), catalog()),
                    "algorithm bsgs requires a public key");
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":71,"addresses":["1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"]})"), catalog()),
                    "puzzle and explicit addresses/pubkey are mutually exclusive");
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":71,"bits":0})"), catalog()), "bits must be 1-256");
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":161})"), catalog()), "unknown puzzle 161 (valid 1-160)");
  CHECK_THROWS_WITH(resolve(spec(R"({"bits":20})"), catalog()), "no target: give --puzzle, --address or --pubkey");
  CHECK_THROWS_WITH(resolve(spec(R"({"addresses":["1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"]})"), catalog()),
                    "no range: give --range, --bits or --puzzle");
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":20,"rangeStart":"0","rangeEnd":"10"})"), catalog()),
                    "range start must be >= 1");
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":20,"rangeStart":"10","rangeEnd":"1"})"), catalog()),
                    "range end must be >= range start");
  CHECK_THROWS_WITH(
      resolve(spec(R"({"puzzle":20,"rangeStart":"1","rangeEnd":"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141"})"),
              catalog()),
      "range end must be below the secp256k1 order");
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":140,"bits":20,"algorithm":"kangaroo"})"), catalog()),
                    "kangaroo requires a range width >= 2^32; use bsgs or scan");
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":140,"algorithm":"kangaroo","order":"random"})"), catalog()),
                    "--order applies only to scan");
  CHECK_THROWS_WITH(spec(R"({"puzzle":1,"bogus":2})"), "unknown field 'bogus'");
}

TEST_CASE("resolve rejects out-of-range resource limits") {
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":140,"dpBits":61})"), catalog()), "dp bits must be 0-60");
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":140,"dpBits":-1})"), catalog()), "dp bits must be 0-60");
  CHECK(resolve(spec(R"({"puzzle":140,"dpBits":60})"), catalog()).algorithm == Algo::Kangaroo);

  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":20,"threads":1025})"), catalog()), "threads must be 0-1024");
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":20,"threads":-1})"), catalog()), "threads must be 0-1024");
  CHECK_NOTHROW(resolve(spec(R"({"puzzle":20,"threads":1024})"), catalog()));

  const std::string memError = "bsgsMemoryMb must be 1-" + std::to_string(maxBsgsMemoryMb());
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":40,"algorithm":"bsgs","bsgsMemoryMb":0})"), catalog()), memError);
  CHECK_THROWS_WITH(resolve(spec(R"({"puzzle":40,"algorithm":"bsgs","bsgsMemoryMb":-1})"), catalog()), memError);
  CHECK_NOTHROW(resolve(spec(R"({"puzzle":40,"algorithm":"bsgs","bsgsMemoryMb":64})"), catalog()));
  // Only BSGS allocates the table.
  CHECK_NOTHROW(resolve(spec(R"({"puzzle":20,"algorithm":"scan","bsgsMemoryMb":-1})"), catalog()));
}

TEST_CASE("solved puzzle warns but resolves") {
  auto r = resolve(spec(R"({"puzzle":20})"), catalog());
  REQUIRE(r.warnings.size() == 1);
  CHECK(r.warnings[0] == "puzzle 20 is already solved (key d2c55)");
}

TEST_CASE("job spec JSON round trip") {
  JobSpec s = spec(R"({"puzzle":66,"order":"random","threads":4,"devices":[0,1],"dpBits":12})");
  nlohmann::json j = s;
  JobSpec back = j.get<JobSpec>();
  CHECK(back.puzzle == 66);
  CHECK(back.order == Order::Random);
  CHECK(back.threads == 4);
  CHECK(back.devices == std::vector<int>{0, 1});
  CHECK(back.dpBits == 12u);
}

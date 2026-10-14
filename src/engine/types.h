#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace bsolver {

enum class Algo { Scan, Bsgs, Kangaroo };
enum class Order { Sequential, Random };
enum class AddressMode { Compressed, Uncompressed, Both };

inline std::string_view toString(Algo a) {
  switch (a) {
    case Algo::Scan: return "scan";
    case Algo::Bsgs: return "bsgs";
    case Algo::Kangaroo: return "kangaroo";
  }
  return "?";
}
inline std::string_view toString(Order o) { return o == Order::Sequential ? "sequential" : "random"; }
inline std::string_view toString(AddressMode m) {
  switch (m) {
    case AddressMode::Compressed: return "compressed";
    case AddressMode::Uncompressed: return "uncompressed";
    case AddressMode::Both: return "both";
  }
  return "?";
}

inline std::optional<Algo> parseAlgo(std::string_view s) {
  if (s == "scan") return Algo::Scan;
  if (s == "bsgs") return Algo::Bsgs;
  if (s == "kangaroo") return Algo::Kangaroo;
  return std::nullopt;
}
inline std::optional<Order> parseOrder(std::string_view s) {
  if (s == "sequential") return Order::Sequential;
  if (s == "random") return Order::Random;
  return std::nullopt;
}
inline std::optional<AddressMode> parseAddressMode(std::string_view s) {
  if (s == "compressed") return AddressMode::Compressed;
  if (s == "uncompressed") return AddressMode::Uncompressed;
  if (s == "both") return AddressMode::Both;
  return std::nullopt;
}

inline bool hashesCompressed(AddressMode m) { return m != AddressMode::Uncompressed; }
inline bool hashesUncompressed(AddressMode m) { return m != AddressMode::Compressed; }

}  // namespace bsolver

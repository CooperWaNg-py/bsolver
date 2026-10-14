#include "engine/format.h"

#include <cmath>
#include <cstdio>

namespace bsolver {

std::string formatRate(double rate, const std::string& unit) {
  const char* prefixes[] = {"", "k", "M", "G", "T", "P", "E"};
  int i = 0;
  while (rate >= 1000.0 && i < 6) {
    rate /= 1000.0;
    ++i;
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.2f %s%s", rate, prefixes[i], unit.c_str());
  return buf;
}

std::string formatDuration(double seconds) {
  if (!std::isfinite(seconds) || seconds < 0) return "--:--:--";
  char buf[64];
  constexpr double kYear = 365.25 * 86400;
  if (seconds >= 1000 * kYear) {
    std::snprintf(buf, sizeof(buf), "%.2ey", seconds / kYear);
    return buf;
  }
  auto total = static_cast<unsigned long long>(seconds);
  unsigned long long days = total / 86400, h = total / 3600 % 24, m = total / 60 % 60, s = total % 60;
  if (days > 0) {
    std::snprintf(buf, sizeof(buf), "%llud %02llu:%02llu:%02llu", days, h, m, s);
  } else {
    std::snprintf(buf, sizeof(buf), "%02llu:%02llu:%02llu", h, m, s);
  }
  return buf;
}

}  // namespace bsolver

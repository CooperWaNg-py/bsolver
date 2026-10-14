#pragma once

#include <string>

namespace bsolver {

// "12.34 Mkeys/s" (k/M/G/T/P/E scaling).
std::string formatRate(double rate, const std::string& unit);
// "HH:MM:SS", "Nd HH:MM:SS", or "1.23e+05y" for very long durations.
std::string formatDuration(double seconds);

}  // namespace bsolver

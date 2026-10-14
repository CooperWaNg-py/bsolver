#pragma once

#include <string>

namespace bsolver {

// Serialized terminal output. On a TTY the progress line is redrawn in place (\r) and
// log/stdout lines are printed above it; otherwise progress lines are plain lines.
void consoleLog(const std::string& line);     // stderr
void consoleOut(const std::string& line);     // stdout
void consoleProgress(const std::string& line);  // stderr
// Ends an in-place progress line with a newline (TTY only).
void consoleFinishProgress();

}  // namespace bsolver

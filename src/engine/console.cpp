#include "engine/console.h"

#include <cstdio>
#include <mutex>

#if defined(_WIN32)
#include <io.h>
#define BSOLVER_ISATTY _isatty
#define BSOLVER_FILENO _fileno
#else
#include <unistd.h>
#define BSOLVER_ISATTY isatty
#define BSOLVER_FILENO fileno
#endif

namespace bsolver {

namespace {
std::mutex& mu() {
  static std::mutex m;
  return m;
}
bool stderrTty() {
  static const bool tty = BSOLVER_ISATTY(BSOLVER_FILENO(stderr)) != 0;
  return tty;
}
std::string& progressLine() {
  static std::string s;
  return s;
}

void clearProgress() {
  if (stderrTty() && !progressLine().empty()) std::fputs("\r\033[K", stderr);
}
void redrawProgress() {
  if (stderrTty() && !progressLine().empty()) {
    std::fputs(progressLine().c_str(), stderr);
    std::fflush(stderr);
  }
}
}  // namespace

void consoleLog(const std::string& line) {
  std::lock_guard<std::mutex> lock(mu());
  clearProgress();
  std::fputs(line.c_str(), stderr);
  std::fputc('\n', stderr);
  redrawProgress();
  std::fflush(stderr);
}

void consoleOut(const std::string& line) {
  std::lock_guard<std::mutex> lock(mu());
  clearProgress();
  std::fflush(stderr);
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
  redrawProgress();
}

void consoleProgress(const std::string& line) {
  std::lock_guard<std::mutex> lock(mu());
  if (stderrTty()) {
    progressLine() = line;
    std::fputs("\r\033[K", stderr);
    std::fputs(line.c_str(), stderr);
  } else {
    std::fputs(line.c_str(), stderr);
    std::fputc('\n', stderr);
  }
  std::fflush(stderr);
}

void consoleFinishProgress() {
  std::lock_guard<std::mutex> lock(mu());
  if (stderrTty() && !progressLine().empty()) {
    std::fputc('\n', stderr);
    std::fflush(stderr);
  }
  progressLine().clear();
}

}  // namespace bsolver

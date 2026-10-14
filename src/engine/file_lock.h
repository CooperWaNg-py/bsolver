#pragma once

#include <filesystem>
#include <optional>

namespace bsolver {

// Exclusive advisory lock on a lock file (flock on POSIX, LockFileEx on Windows), held until destruction. Two
// holders conflict whether they are separate processes or separate locks in one process. The lock file itself is
// left in place: it is empty, and deleting it while another process waits for it would split the lock.
class FileLock {
 public:
  // Opens (creating parent directories and the file) and locks `path` without blocking. Returns nullopt when
  // another holder has the lock; throws std::runtime_error when the file cannot be opened or locked.
  static std::optional<FileLock> tryAcquire(const std::filesystem::path& path);

  FileLock(FileLock&& other) noexcept;
  FileLock& operator=(FileLock&& other) noexcept;
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;
  ~FileLock();

 private:
#if defined(_WIN32)
  explicit FileLock(void* handle) : handle_(handle) {}
  void* handle_ = nullptr;
#else
  explicit FileLock(int fd) : fd_(fd) {}
  int fd_ = -1;
#endif
  void release() noexcept;
};

}  // namespace bsolver

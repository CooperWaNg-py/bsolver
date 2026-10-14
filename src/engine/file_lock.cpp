#include "engine/file_lock.h"

#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace bsolver {

namespace {
std::runtime_error lockError(const std::filesystem::path& path, std::error_code ec) {
  return std::runtime_error("cannot lock " + path.string() + ": " + ec.message());
}
}  // namespace

std::optional<FileLock> FileLock::tryAcquire(const std::filesystem::path& path) {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
#if defined(_WIN32)
  HANDLE h = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    throw lockError(path, std::error_code(static_cast<int>(GetLastError()), std::system_category()));
  }
  OVERLAPPED ov{};
  if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, MAXDWORD, MAXDWORD, &ov)) {
    const DWORD err = GetLastError();
    CloseHandle(h);
    if (err == ERROR_LOCK_VIOLATION) return std::nullopt;
    throw lockError(path, std::error_code(static_cast<int>(err), std::system_category()));
  }
  return FileLock(static_cast<void*>(h));
#else
  int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) throw lockError(path, std::error_code(errno, std::generic_category()));
  int rc;
  do {
    rc = ::flock(fd, LOCK_EX | LOCK_NB);
  } while (rc != 0 && errno == EINTR);
  if (rc != 0) {
    const int err = errno;
    ::close(fd);
    if (err == EWOULDBLOCK) return std::nullopt;
    throw lockError(path, std::error_code(err, std::generic_category()));
  }
  return FileLock(fd);
#endif
}

FileLock::FileLock(FileLock&& other) noexcept {
#if defined(_WIN32)
  handle_ = std::exchange(other.handle_, nullptr);
#else
  fd_ = std::exchange(other.fd_, -1);
#endif
}

FileLock& FileLock::operator=(FileLock&& other) noexcept {
  if (this != &other) {
    release();
#if defined(_WIN32)
    handle_ = std::exchange(other.handle_, nullptr);
#else
    fd_ = std::exchange(other.fd_, -1);
#endif
  }
  return *this;
}

FileLock::~FileLock() { release(); }

void FileLock::release() noexcept {
#if defined(_WIN32)
  if (!handle_) return;
  HANDLE h = static_cast<HANDLE>(handle_);
  OVERLAPPED ov{};
  UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &ov);
  CloseHandle(h);
  handle_ = nullptr;
#else
  if (fd_ < 0) return;
  ::close(fd_);  // closing the only descriptor of the open file description releases the flock
  fd_ = -1;
#endif
}

}  // namespace bsolver

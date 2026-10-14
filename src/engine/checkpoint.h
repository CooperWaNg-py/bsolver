#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "engine/bytes.h"
#include "engine/file_lock.h"
#include "engine/types.h"

namespace bsolver {

// Checkpoint file `<stateDir>/<algo>-<first 16 hex of SHA256(identity)>.ckpt`:
// "BSCK", u32 version=1, u32 identityLen, identity, u8 algo, algorithm payload.
// Writes go to a uniquely named `<file>.<random>.tmp` and are renamed into place. While a job runs, it holds an
// exclusive lock on `<file>.lock` so two jobs (in any processes) never resume from and write the same checkpoint;
// the empty lock file stays when the checkpoint is deleted (see FileLock).
class Checkpointer {
 public:
  static constexpr std::chrono::seconds kInterval{60};

  // enabled=false: never reads or writes. fresh=true: ignores (and later overwrites) an existing file.
  Checkpointer(const std::filesystem::path& stateDir, Algo algo, std::string identity, bool enabled, bool fresh);

  bool enabled() const { return enabled_; }
  const std::filesystem::path& path() const { return path_; }

  // Takes the exclusive lock on `<file>.lock` (no-op when disabled). Returns false when another job holds it.
  // Throws std::runtime_error when the lock file cannot be opened. Held until unlock() or destruction.
  bool lock();
  void unlock();

  // Algorithm payload of an existing checkpoint, or nullopt when disabled/fresh/missing.
  // Throws std::runtime_error("checkpoint <path> does not match this job") on identity mismatch.
  std::optional<std::vector<uint8_t>> load();
  // load() and hand the payload to parse; returns false when there is nothing to resume. A payload parse cannot
  // read (truncated, trailing bytes, or a value it rejects by throwing std::runtime_error) throws
  // std::runtime_error("checkpoint <path> is corrupt (use --fresh)").
  bool restore(const std::function<void(ByteReader&)>& parse);

  // The runner registers a thread-safe payload snapshot function.
  void setSnapshot(std::function<std::vector<uint8_t>()> fn);
  void clearSnapshot();
  // Saves when kInterval has elapsed since the last save.
  void maybeSave();
  void saveNow();
  // Deletes the checkpoint file (job finished with found/exhausted).
  void remove();

  // Reads a checkpoint file at an explicit path.
  static std::optional<std::vector<uint8_t>> loadFile(const std::filesystem::path& path, Algo algo,
                                                      const std::string& identity);

 private:
  void write(const std::vector<uint8_t>& payload);

  std::filesystem::path path_;
  Algo algo_;
  std::string identity_;
  bool enabled_, fresh_;
  std::mutex mu_;
  std::function<std::vector<uint8_t>()> snapshot_;
  std::chrono::steady_clock::time_point lastSave_;
  std::optional<FileLock> lock_;  // guarded by mu_
};

// Registers a runner's snapshot function for the lifetime of the run. finish(complete) deletes the
// checkpoint when the job is complete (found/exhausted) and otherwise saves it; if finish() is never
// reached (exception), the destructor saves best-effort. Either way the snapshot is unregistered.
class SnapshotScope {
 public:
  SnapshotScope(Checkpointer& c, std::function<std::vector<uint8_t>()> fn) : c_(c) { c_.setSnapshot(std::move(fn)); }
  SnapshotScope(const SnapshotScope&) = delete;
  SnapshotScope& operator=(const SnapshotScope&) = delete;
  void finish(bool complete) {
    done_ = true;
    if (complete) {
      c_.remove();
      return;
    }
    try {
      c_.saveNow();
    } catch (...) {
      c_.clearSnapshot();  // the snapshot references the runner's locals, which unwinding destroys
      throw;
    }
    c_.clearSnapshot();
  }
  ~SnapshotScope() {
    if (done_) return;
    try {
      c_.saveNow();
    } catch (...) {
    }
    c_.clearSnapshot();
  }

 private:
  Checkpointer& c_;
  bool done_ = false;
};

}  // namespace bsolver

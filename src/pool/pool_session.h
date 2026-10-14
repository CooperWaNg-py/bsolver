#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "engine/job_manager.h"
#include "pool/pool_api.h"
#include "pool/pool_config.h"
#include "pool/pool_state.h"
#include "pool/rsa.h"

namespace bsolver {

struct PoolSessionOptions {
  PoolConfig config;
  std::string backend = "auto";  // benchmarked once per session when "auto"
  unsigned threads = 0;
  std::vector<int> devices;
  std::optional<uint64_t> maxRanges;  // stop after submitting this many ranges
  std::string telegramApiUrl = "https://api.telegram.org";
  std::chrono::seconds pingInterval{120};
  std::chrono::seconds retryDelay{30};
};

// btcpuzzle.info worker loop: fetch a range, scan it (proof-of-work addresses + target) with the
// JobManager, submit SHA256(proof keys), repeat. The assignment, found proof keys and scan checkpoint
// are persisted in the manager's state directory, so an interrupted range resumes where it stopped.
class PoolSession {
 public:
  PoolSession(JobManager& manager, PoolSessionOptions options);
  ~PoolSession();
  PoolSession(const PoolSession&) = delete;
  PoolSession& operator=(const PoolSession&) = delete;

  // Blocks until stopped (130), maxRanges submitted (0) or a fatal error (1).
  int run();
  // Thread-safe; stops the running scan (its checkpoint is written) and returns from run().
  void requestStop();

 private:
  struct Notification {
    std::string address, secret, hex;
    bool encrypted;
  };

  void setup();
  std::string chooseBackend(std::vector<std::string>* deviceNames);
  bool sleepUnlessStopped(std::chrono::seconds d);
  bool fetchAssignment();
  bool scanAssignment();
  void shutdownHelpers();
  bool submitAssignment();
  void onFound(const FoundKey& key);
  void recordTarget(const FoundKey& key, const std::string& hex);
  void notifierLoop();
  void pingLoop();
  void saveState();
  void setPingHex(const std::string& hex);  // empty: no keepalive pings
  // Forgets the current assignment (persisted) so the next loop iteration fetches a new range.
  void dropAssignment();
  std::string keyChannels() const;  // notification channels that carry a found key
  std::string display(const std::string& hex) const;
  void log(LogLevel level, const std::string& msg) { manager_.log(level, "pool: " + msg); }

  JobManager& manager_;
  PoolSessionOptions opts_;
  std::filesystem::path statePath_;
  std::optional<RsaPublicKey> rsaKey_;
  std::unique_ptr<PoolApi> api_;
  std::string backend_;
  int gpuCount_ = 1;
  uint64_t submittedThisRun_ = 0;
  bool resumedAssignment_ = false;  // the current assignment came from the saved state (run() thread only)

  std::mutex stateMu_;  // guards state_
  PoolState state_;

  std::atomic<bool> stop_{false};
  std::mutex waitMu_;
  std::condition_variable waitCv_;
  bool helpersDone_ = false;  // guarded by waitMu_

  std::mutex pingMu_;
  std::string pingHex_;
  std::thread pingThread_;

  std::mutex notifyMu_;
  std::condition_variable notifyCv_;
  std::deque<Notification> notifications_;
  bool notifierDone_ = false;
  std::thread notifierThread_;
};

}  // namespace bsolver

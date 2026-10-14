#pragma once

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include "catalog/catalog.h"
#include "engine/job_manager.h"

namespace httplib {
class Server;
}

namespace bsolver {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;  // 0 = any free port
  std::string token;
  bool serveUi = true;
};

// Pending SSE frames of one event-stream client (not thread-safe). Only the newest pending "status" frame is kept.
// When full, the oldest unpinned frame is evicted; pinned frames ("found", error logs) are never evicted, so the
// queue may exceed its limit only when every pending frame is pinned.
class SseQueue {
 public:
  explicit SseQueue(size_t limit) : limit_(limit) {}
  void push(const std::string& type, std::string frame, bool pinned);
  bool empty() const { return frames_.empty(); }
  size_t size() const { return frames_.size(); }
  // Concatenates every pending frame in order and clears the queue.
  std::string drain();

 private:
  struct Frame {
    std::string type;
    std::string data;
    bool pinned;
  };
  size_t limit_;
  std::deque<Frame> frames_;
};

// REST API + SSE event stream + embedded web UI. Plain HTTP; every /api route needs the bearer token.
class HttpServer {
 public:
  static constexpr size_t kMaxEventStreams = 8;
  static constexpr size_t kEventQueueLimit = 256;

  HttpServer(JobManager& manager, const Catalog& catalog, ServerOptions options);
  ~HttpServer();
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // Binds and starts serving on a background thread; returns the bound port. Throws on bind failure.
  int start();
  void stop();

 private:
  struct EventClient;
  void setupRoutes();
  bool authorized(const std::string& header, const std::string& queryToken) const;

  JobManager& manager_;
  const Catalog& catalog_;
  ServerOptions options_;
  std::unique_ptr<httplib::Server> svr_;
  std::thread thread_;
  std::atomic<size_t> streams_{0};
  std::mutex clientsMu_;
  std::set<std::shared_ptr<EventClient>> clients_;
};

// True for 127.0.0.0/8, ::1 and "localhost".
bool isLoopbackHost(const std::string& host);

}  // namespace bsolver

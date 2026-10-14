#pragma once

#include <atomic>
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

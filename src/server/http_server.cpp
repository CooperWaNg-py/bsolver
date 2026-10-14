#include "server/http_server.h"

#include <httplib.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <stdexcept>

#include "assets/assets.h"
#include "backend/backend.h"
#include "engine/errors.h"
#include "engine/results.h"

namespace bsolver {

namespace {
using nlohmann::json;

void sendJson(httplib::Response& res, int status, const json& body) {
  res.status = status;
  res.set_content(body.dump(), "application/json");
}

void sendError(httplib::Response& res, int status, const std::string& msg) {
  sendJson(res, status, json{{"error", msg}});
}

bool constantTimeEqual(std::string_view a, std::string_view b) {
  size_t n = std::max(a.size(), b.size());
  unsigned diff = static_cast<unsigned>(a.size() ^ b.size());
  for (size_t i = 0; i < n; ++i) {
    unsigned char x = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
    unsigned char y = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
    diff |= static_cast<unsigned>(x ^ y);
  }
  return diff == 0;
}

std::string sseFrame(const std::string& type, const json& payload) {
  return "event: " + type + "\ndata: " + payload.dump() + "\n\n";
}

const assets::Asset* findAsset(std::string_view path) {
  for (size_t i = 0; i < assets::kWebAssetsCount; ++i) {
    if (assets::kWebAssets[i].path == path) return &assets::kWebAssets[i];
  }
  return nullptr;
}

json backendInfo(Backend& b) {
  auto devices = b.devices();
  json devs = json::array();
  for (const auto& d : devices) {
    devs.push_back({{"index", d.index}, {"name", d.name}, {"memoryBytes", d.memoryBytes}, {"computeUnits", d.computeUnits}});
  }
  json algos = json::array();
  for (Algo a : {Algo::Scan, Algo::Bsgs, Algo::Kangaroo}) {
    if (b.supports(a)) algos.push_back(toString(a));
  }
  return json{{"id", b.id()},
              {"available", !devices.empty()},
              {"reason", devices.empty() ? b.unavailableReason() : std::string()},
              {"devices", devs},
              {"algorithms", algos}};
}
}  // namespace

bool isLoopbackHost(const std::string& host) {
  return host == "localhost" || host == "::1" || host == "[::1]" || host.rfind("127.", 0) == 0;
}

struct HttpServer::EventClient {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::string> queue;
  bool closed = false;
  int subscription = 0;

  void push(std::string frame) {
    std::lock_guard<std::mutex> lock(mu);
    if (queue.size() >= kEventQueueLimit) queue.pop_front();
    queue.push_back(std::move(frame));
    cv.notify_one();
  }
  void close() {
    std::lock_guard<std::mutex> lock(mu);
    closed = true;
    cv.notify_all();
  }
};

HttpServer::HttpServer(JobManager& manager, const Catalog& catalog, ServerOptions options)
    : manager_(manager), catalog_(catalog), options_(std::move(options)), svr_(std::make_unique<httplib::Server>()) {
  svr_->new_task_queue = [] { return new httplib::ThreadPool(16); };
  setupRoutes();
}

HttpServer::~HttpServer() { stop(); }

bool HttpServer::authorized(const std::string& header, const std::string& queryToken) const {
  if (!header.empty() && constantTimeEqual(header, "Bearer " + options_.token)) return true;
  return !queryToken.empty() && constantTimeEqual(queryToken, options_.token);
}

void HttpServer::setupRoutes() {
  httplib::Server& s = *svr_;

  s.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
    if (req.path.rfind("/api/", 0) != 0 && req.path != "/api") return httplib::Server::HandlerResponse::Unhandled;
    std::string queryToken = req.path == "/api/events" ? req.get_param_value("token") : std::string();
    if (!authorized(req.get_header_value("Authorization"), queryToken)) {
      sendError(res, 401, "unauthorized");
      return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });

  s.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
    std::string msg = "internal error";
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception& ex) {
      msg = ex.what();
    } catch (...) {
    }
    sendError(res, 500, msg);
  });

  s.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
    if (req.path.rfind("/api", 0) == 0 && res.body.empty()) {
      sendError(res, res.status, res.status == 404 ? "not found" : httplib::status_message(res.status));
    }
  });

  s.Get("/api/info", [](const httplib::Request&, httplib::Response& res) {
    json backends = json::array();
    for (Backend* b : allBackends()) backends.push_back(backendInfo(*b));
    sendJson(res, 200, json{{"version", BSOLVER_VERSION}, {"backends", backends}});
  });

  s.Get("/api/puzzles", [this](const httplib::Request&, httplib::Response& res) {
    json arr = json::array();
    for (const auto& p : catalog_.all()) arr.push_back(puzzleToJson(p));
    sendJson(res, 200, arr);
  });

  s.Get("/api/job", [this](const httplib::Request&, httplib::Response& res) {
    sendJson(res, 200, json(manager_.status()));
  });

  s.Post("/api/job", [this](const httplib::Request& req, httplib::Response& res) {
    JobSpec spec;
    try {
      spec = json::parse(req.body).get<JobSpec>();
    } catch (const json::exception& ex) {
      return sendError(res, 400, std::string("invalid JSON: ") + ex.what());
    } catch (const JobError& ex) {
      return sendError(res, 400, ex.what());
    }
    try {
      sendJson(res, 202, json(manager_.start(spec)));
    } catch (const JobError& ex) {
      sendError(res, 400, ex.what());
    } catch (const ConflictError& ex) {
      sendError(res, 409, ex.what());
    }
  });

  auto control = [this](JobStatus (JobManager::*fn)()) {
    return [this, fn](const httplib::Request&, httplib::Response& res) {
      try {
        sendJson(res, 200, json((manager_.*fn)()));
      } catch (const ConflictError& ex) {
        sendError(res, 409, ex.what());
      }
    };
  };
  s.Post("/api/job/stop", control(&JobManager::stop));
  s.Post("/api/job/pause", control(&JobManager::pause));
  s.Post("/api/job/resume", control(&JobManager::resume));

  s.Post("/api/bench", [this](const httplib::Request& req, httplib::Response& res) {
    Algo algo = Algo::Scan;
    double seconds = 2.0;
    try {
      json body = req.body.empty() ? json::object() : json::parse(req.body);
      if (!body.is_object()) return sendError(res, 400, "bench request must be a JSON object");
      for (const auto& [k, v] : body.items()) {
        if (k == "algorithm") {
          std::string a = v.get<std::string>();
          if (a != "scan" && a != "kangaroo") return sendError(res, 400, "algorithm must be scan or kangaroo");
          algo = *parseAlgo(a);
        } else if (k == "seconds") {
          seconds = v.get<double>();
        } else {
          return sendError(res, 400, "unknown field '" + k + "'");
        }
      }
    } catch (const json::exception& ex) {
      return sendError(res, 400, std::string("invalid JSON: ") + ex.what());
    }
    if (!(seconds > 0 && seconds <= 60)) return sendError(res, 400, "seconds must be in (0, 60]");
    try {
      sendJson(res, 200, json(manager_.bench(algo, seconds)));
    } catch (const ConflictError& ex) {
      sendError(res, 409, ex.what());
    }
  });

  s.Get("/api/results", [this](const httplib::Request&, httplib::Response& res) {
    sendJson(res, 200, json(readResults(manager_.options().resultsPath)));
  });

  s.Get("/api/events", [this](const httplib::Request&, httplib::Response& res) {
    if (streams_.fetch_add(1) >= kMaxEventStreams) {
      streams_--;
      return sendError(res, 503, "too many event streams");
    }
    auto client = std::make_shared<EventClient>();
    client->push(sseFrame("status", json(manager_.status())));
    client->subscription = manager_.subscribe([client](const Event& e) { client->push(sseFrame(e.type, e.payload)); });
    {
      std::lock_guard<std::mutex> lock(clientsMu_);
      clients_.insert(client);
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [client](size_t, httplib::DataSink& sink) {
          std::string out;
          {
            std::unique_lock<std::mutex> lock(client->mu);
            client->cv.wait_for(lock, std::chrono::seconds(15), [&] { return !client->queue.empty() || client->closed; });
            if (client->closed) return false;
            if (client->queue.empty()) {
              out = ": ping\n\n";
            } else {
              while (!client->queue.empty()) {
                out += client->queue.front();
                client->queue.pop_front();
              }
            }
          }
          return sink.write(out.data(), out.size());
        },
        [this, client](bool) {
          manager_.unsubscribe(client->subscription);
          {
            std::lock_guard<std::mutex> lock(clientsMu_);
            clients_.erase(client);
          }
          streams_--;
        });
  });

  s.Get(".*", [this](const httplib::Request& req, httplib::Response& res) {
    if (req.path.rfind("/api", 0) == 0) return sendError(res, 404, "not found");
    if (!options_.serveUi || assets::kWebAssetsCount == 0) {
      res.status = 404;
      res.set_content("web UI not built (BSOLVER_WEBUI=OFF); API available under /api\n", "text/plain");
      return;
    }
    const assets::Asset* a = req.path == "/" ? nullptr : findAsset(req.path);
    if (!a) a = findAsset("/index.html");  // SPA fallback
    if (!a) {
      res.status = 404;
      res.set_content("not found\n", "text/plain");
      return;
    }
    bool immutable = a->path.rfind("/assets/", 0) == 0;
    res.set_header("Cache-Control", immutable ? "public, max-age=31536000, immutable" : "no-cache");
    res.set_content(reinterpret_cast<const char*>(a->data), a->size, std::string(a->mime));
  });
}

int HttpServer::start() {
  int port = options_.port;
  if (port == 0) {
    port = svr_->bind_to_any_port(options_.host);
    if (port < 0) throw std::runtime_error("cannot bind " + options_.host + " (any port)");
  } else if (!svr_->bind_to_port(options_.host, port)) {
    throw std::runtime_error("cannot bind " + options_.host + ":" + std::to_string(port));
  }
  if (!isLoopbackHost(options_.host)) {
    manager_.log(LogLevel::Warn, "web UI exposed on " + options_.host +
                                     " without TLS; use a reverse proxy or SSH tunnel for remote access");
  }
  thread_ = std::thread([this] { svr_->listen_after_bind(); });
  svr_->wait_until_ready();
  return port;
}

void HttpServer::stop() {
  {
    std::lock_guard<std::mutex> lock(clientsMu_);
    for (const auto& c : clients_) c->close();
  }
  if (svr_) svr_->stop();
  if (thread_.joinable()) thread_.join();
}

}  // namespace bsolver

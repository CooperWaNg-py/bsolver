#include "engine/job_manager.h"

#include <algorithm>
#include <cmath>

#include "backend/backend.h"
#include "engine/console.h"
#include "engine/errors.h"
#include "engine/format.h"
#include "engine/runners.h"

namespace bsolver {

namespace {
constexpr double kEmaAlpha = 2.0 / (10.0 + 1.0);  // EMA over ~10 one-second samples

std::vector<std::string> deviceNames(Backend& backend, const JobSpec& spec) {
  std::vector<std::string> names;
  if (backend.id() == "cpu") {
    names.push_back("CPU (" + std::to_string(effectiveThreads(spec.threads)) + " threads)");
    return names;
  }
  for (const auto& d : backend.devices()) {
    if (spec.devices.empty() ||
        std::find(spec.devices.begin(), spec.devices.end(), d.index) != spec.devices.end()) {
      names.push_back(d.name);
    }
  }
  return names;
}

Backend& validateBackend(const JobSpec& spec, Algo algo) {
  static const std::vector<std::string> known = {"cpu", "cuda", "opencl", "metal"};
  const std::string& id = spec.backend;
  if (id == "auto") return *findBackend("cpu");
  if (std::find(known.begin(), known.end(), id) == known.end()) {
    throw JobError("unknown backend '" + id + "' (auto|cpu|opencl|metal|cuda)");
  }
  Backend* b = findBackend(id);
  if (!b) throw JobError("backend " + id + " not available: not compiled in");
  if (b->devices().empty()) throw JobError("backend " + id + " not available: " + b->unavailableReason());
  if (!b->supports(algo)) throw JobError("backend " + id + " does not support " + std::string(toString(algo)));
  return *b;
}
}  // namespace

struct JobManager::ActiveJob {
  JobSpec spec;
  ResolvedJob resolved;
  std::unique_ptr<Checkpointer> ckpt;
  std::unique_ptr<RunContext> ctx;
  std::chrono::steady_clock::time_point started;
  std::optional<std::chrono::steady_clock::time_point> finished;

  // Ticker state (rate fields guarded by JobManager::mu_).
  std::mutex tickMu;
  std::condition_variable tickCv;
  bool done = false;
  double rate = 0;
  bool haveRate = false;
  double lastLive = 0;
  std::chrono::steady_clock::time_point lastTick;
};

JobManager::JobManager(const Catalog& catalog, ManagerOptions options)
    : catalog_(catalog), options_(std::move(options)) {}

JobManager::~JobManager() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (job_ && isActive(st_.state)) {
      st_.state = JobState::Stopping;
      job_->ctx->requestStop();
    }
  }
  joinThreads();
}

void JobManager::joinThreads() {
  if (jobThread_.joinable()) jobThread_.join();
  if (tickThread_.joinable()) tickThread_.join();
}

JobStatus JobManager::start(const JobSpec& spec) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    throwIfBusyLocked();
    starting_ = true;
  }
  struct StartGuard {
    JobManager* m;
    ~StartGuard() {
      std::lock_guard<std::mutex> lock(m->mu_);
      m->starting_ = false;
    }
  } guard{this};
  ResolvedJob resolved = resolve(spec, catalog_);
  validateBackend(spec, resolved.algorithm);
  joinThreads();

  auto job = std::make_shared<ActiveJob>();
  job->spec = spec;
  job->resolved = std::move(resolved);
  const ResolvedJob& r = job->resolved;
  job->ckpt = std::make_unique<Checkpointer>(options_.stateDir, r.algorithm, r.identity, spec.checkpoint, spec.fresh);
  job->ctx = std::make_unique<RunContext>(
      job->resolved, job->spec, *job->ckpt, [this](LogLevel l, const std::string& m) { log(l, m); },
      [this](const FoundKey& k) { onFound(k); });
  const U256 width = r.width();
  if (r.algorithm != Algo::Kangaroo) job->ctx->setWorkExpected(width.toDouble());
  job->started = std::chrono::steady_clock::now();
  job->lastTick = job->started;

  {
    std::lock_guard<std::mutex> lock(mu_);
    JobStatus s;
    s.id = st_.id + 1;
    bool benchmarking = spec.backend == "auto" && r.algorithm != Algo::Bsgs;
    s.state = benchmarking ? JobState::Benchmarking : JobState::Running;
    s.spec = spec;
    s.puzzle = r.puzzle;
    s.algorithm = r.algorithm;
    if (r.algorithm == Algo::Scan) s.order = r.order;
    s.rangeStart = r.rangeStart.toHex();
    s.rangeEnd = r.rangeEnd.toHex();
    s.rangeBits = std::log2(width.toDouble());
    s.rateUnit = r.algorithm == Algo::Kangaroo ? "jumps/s" : "keys/s";
    s.startedAt = isoTimeNow();
    st_ = std::move(s);
    job_ = job;
  }
  for (const auto& w : r.warnings) log(LogLevel::Warn, w);

  jobThread_ = std::thread(&JobManager::runJob, this, job);
  tickThread_ = std::thread(&JobManager::tickLoop, this, job);
  return status();
}

void JobManager::runJob(std::shared_ptr<ActiveJob> job) {
  RunContext& ctx = *job->ctx;
  const Algo algo = job->resolved.algorithm;
  try {
    Backend* backend = nullptr;
    if (algo == Algo::Bsgs) {
      backend = findBackend("cpu");
    } else if (job->spec.backend == "auto") {
      log(LogLevel::Info, "benchmarking backends for " + std::string(toString(algo)));
      auto results = runBenchmark(algo, BenchOptions{2.0, job->spec.threads, job->spec.devices});
      const BenchResult* best = nullptr;
      std::string errors;
      for (const auto& b : results) {
        std::string devs;
        for (const auto& d : b.devices) devs += (devs.empty() ? "" : ", ") + d;
        log(LogLevel::Info, "  " + b.backend + " [" + devs + "] " +
                                (b.verified ? formatRate(b.rate, algo == Algo::Kangaroo ? "jumps/s" : "keys/s")
                                            : "FAILED: " + b.error));
        if (!b.verified) {
          errors += (errors.empty() ? "" : "; ") + b.backend + ": " + b.error;
        } else if (!best || b.rate > best->rate) {
          best = &b;
        }
      }
      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.benchmark = results;
      }
      if (!best) throw std::runtime_error("no working backend: " + errors);
      backend = findBackend(best->backend);
      log(LogLevel::Info, "selected backend " + best->backend + " (" +
                              formatRate(best->rate, algo == Algo::Kangaroo ? "jumps/s" : "keys/s") + ")");
    } else {
      backend = findBackend(job->spec.backend);
    }

    bool exhausted = false;
    if (!ctx.stopRequested()) {
      ctx.setBackend(std::string(backend->id()));
      {
        std::lock_guard<std::mutex> lock(mu_);
        st_.backend = std::string(backend->id());
        st_.devices = deviceNames(*backend, job->spec);
        if (st_.state == JobState::Benchmarking) st_.state = JobState::Running;
      }
      switch (algo) {
        case Algo::Scan: exhausted = runScan(ctx, *backend); break;
        case Algo::Bsgs: exhausted = runBsgs(ctx); break;
        case Algo::Kangaroo: runKangaroo(ctx, *backend); break;
      }
    }
    Outcome outcome = ctx.anyFound() ? Outcome::Found : exhausted ? Outcome::Exhausted : Outcome::Stopped;
    {
      std::lock_guard<std::mutex> lock(mu_);
      st_.state = JobState::Finished;
      st_.outcome = outcome;
      st_.exhausted = exhausted;
      job->finished = std::chrono::steady_clock::now();
    }
    log(LogLevel::Info, "job finished: " + std::string(toString(outcome)));
  } catch (const std::exception& ex) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      st_.state = JobState::Failed;
      st_.outcome = Outcome::Error;
      st_.error = ex.what();
      job->finished = std::chrono::steady_clock::now();
    }
    log(LogLevel::Error, ex.what());
  }
  {
    std::lock_guard<std::mutex> lock(job->tickMu);
    job->done = true;
  }
  job->tickCv.notify_all();
  publishStatus();
  doneCv_.notify_all();
}

void JobManager::tickLoop(std::shared_ptr<ActiveJob> job) {
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(job->tickMu);
      if (job->tickCv.wait_for(lock, std::chrono::seconds(1), [&] { return job->done; })) return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      auto now = std::chrono::steady_clock::now();
      double live = job->ctx->liveWork();
      double dt = std::chrono::duration<double>(now - job->lastTick).count();
      if (st_.state != JobState::Running) {
        // Rate is frozen while benchmarking/paused/stopping; restart the baseline.
        job->lastLive = live;
        job->lastTick = now;
      } else if (dt > 0) {
        double sample = (live - job->lastLive) / dt;
        job->rate = job->haveRate ? kEmaAlpha * sample + (1 - kEmaAlpha) * job->rate : sample;
        job->haveRate = true;
        job->lastLive = live;
        job->lastTick = now;
      }
    }
    try {
      job->ckpt->maybeSave();
    } catch (const std::exception& ex) {
      log(LogLevel::Error, std::string("checkpoint save failed: ") + ex.what());
    }
    publishStatus();
  }
}

JobStatus JobManager::stop() {
  std::lock_guard<std::mutex> lock(mu_);
  if (!job_ || !isActive(st_.state)) throw ConflictError("no active job");
  if (st_.state != JobState::Stopping) {
    st_.state = JobState::Stopping;
    job_->ctx->requestStop();
    job_->ctx->setPaused(false);
  }
  return statusLocked();
}

JobStatus JobManager::pause() {
  std::shared_ptr<ActiveJob> job;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!job_ || !isActive(st_.state)) throw ConflictError("no active job");
    if (st_.state != JobState::Running) throw ConflictError("job is not running");
    st_.state = JobState::Paused;
    job_->ctx->setPaused(true);
    job = job_;
  }
  log(LogLevel::Info, "job paused");
  try {
    job->ckpt->saveNow();
  } catch (const std::exception& ex) {
    log(LogLevel::Error, std::string("checkpoint save failed: ") + ex.what());
  }
  return status();
}

JobStatus JobManager::resume() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!job_ || !isActive(st_.state)) throw ConflictError("no active job");
    if (st_.state != JobState::Paused) throw ConflictError("job is not paused");
    st_.state = JobState::Running;
    job_->ctx->setPaused(false);
  }
  log(LogLevel::Info, "job resumed");
  return status();
}

JobStatus JobManager::status() const {
  std::lock_guard<std::mutex> lock(mu_);
  return statusLocked();
}

JobStatus JobManager::statusLocked() const {
  JobStatus s = st_;
  s.seq = ++seq_;
  if (!job_) return s;
  const RunContext& ctx = *job_->ctx;
  auto end = job_->finished.value_or(std::chrono::steady_clock::now());
  s.elapsedSeconds = std::chrono::duration<double>(end - job_->started).count();
  s.workDone = ctx.workDone();
  s.resumed = ctx.resumed();
  s.found = ctx.found();
  s.rate = job_->rate;
  const double expected = ctx.workExpected();
  const Algo algo = job_->resolved.algorithm;
  if (expected > 0) s.workExpected = expected;
  if (algo == Algo::Scan && job_->resolved.order == Order::Random) {
    // Random scan: no progress/ETA; the UI shows workExpected/rate as the expected time.
  } else if (expected > 0) {
    double remaining = expected - s.workDone;
    if (algo == Algo::Kangaroo) {
      s.progress = s.workDone / expected;
      remaining = std::max(0.0, remaining);
    } else {
      s.progress = std::min(1.0, s.workDone / expected);
      remaining = std::max(0.0, remaining);
    }
    if (s.rate > 0 && isActive(s.state)) s.etaSeconds = remaining / s.rate;
  }
  return s;
}

void JobManager::throwIfBusyLocked() const {
  if (benchRunning_) throw ConflictError("a benchmark is running");
  if (starting_ || (job_ && isActive(st_.state))) throw ConflictError("a job is already active");
}

std::vector<BenchResult> JobManager::bench(Algo algo, double seconds, unsigned threads) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    throwIfBusyLocked();
    benchRunning_ = true;
  }
  struct Reset {
    JobManager* m;
    ~Reset() {
      std::lock_guard<std::mutex> lock(m->mu_);
      m->benchRunning_ = false;
    }
  } reset{this};
  return runBenchmark(algo, BenchOptions{seconds, threads, {}});
}

int JobManager::subscribe(std::function<void(const Event&)> fn, bool initialStatus) {
  std::lock_guard<std::mutex> lock(subMu_);
  int id = nextSub_++;
  // Under subMu_, so the initial snapshot is ordered before every status event published to this subscriber.
  if (initialStatus) fn(Event{"status", nlohmann::json(status())});
  subs_[id] = std::move(fn);
  return id;
}

void JobManager::unsubscribe(int id) {
  std::lock_guard<std::mutex> lock(subMu_);
  subs_.erase(id);
}

void JobManager::publish(const Event& e) {
  std::lock_guard<std::mutex> lock(subMu_);
  deliverLocked(e);
}

void JobManager::publishStatus() {
  // Taking the snapshot under subMu_ makes subscribers see status events in seq order.
  std::lock_guard<std::mutex> lock(subMu_);
  deliverLocked(Event{"status", nlohmann::json(status())});
}

void JobManager::deliverLocked(const Event& e) {
  for (auto& [id, fn] : subs_) {
    // A failing subscriber must not take down the manager thread (or the whole process) that published.
    try {
      fn(e);
    } catch (...) {
    }
  }
}

void JobManager::wait() {
  std::unique_lock<std::mutex> lock(mu_);
  doneCv_.wait(lock, [&] { return !job_ || !isActive(st_.state); });
}

void JobManager::log(LogLevel level, const std::string& msg) {
  consoleLog("[" + std::string(toString(level)) + "] " + msg);
  publish(Event{"log", nlohmann::json{{"time", isoTimeNow()}, {"level", toString(level)}, {"message", msg}}});
}

void JobManager::setFoundHandler(std::function<void(const FoundKey&)> fn) {
  std::lock_guard<std::mutex> lock(resultsMu_);
  foundHandler_ = std::move(fn);
}

void JobManager::onFound(const FoundKey& key) {
  bool handled = false;
  {
    std::lock_guard<std::mutex> lock(resultsMu_);
    if (foundHandler_) {
      foundHandler_(key);
      handled = true;
    } else {
      try {
        appendResult(options_.resultsPath, key);
      } catch (const std::exception& ex) {
        log(LogLevel::Error, ex.what());
      }
    }
  }
  if (!handled) consoleOut(foundLine(key));
  publish(Event{"found", nlohmann::json(key)});
}

}  // namespace bsolver

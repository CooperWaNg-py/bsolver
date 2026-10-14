#include "engine/run_context.h"

#include <exception>
#include <thread>

#include "core/address.h"
#include "core/gtable.h"

namespace bsolver {

std::string_view toString(LogLevel l) {
  switch (l) {
    case LogLevel::Info: return "info";
    case LogLevel::Warn: return "warn";
    case LogLevel::Error: return "error";
  }
  return "info";
}

namespace {
void atomicAdd(std::atomic<double>& a, double v) {
  double cur = a.load(std::memory_order_relaxed);
  while (!a.compare_exchange_weak(cur, cur + v, std::memory_order_relaxed)) {
  }
}
}  // namespace

RunContext::RunContext(const ResolvedJob& job, const JobSpec& spec, Checkpointer& checkpointer, LogFn log,
                       FoundFn found)
    : job_(job),
      spec_(spec),
      targets_(job.targets),
      checkpointer_(checkpointer),
      log_(std::move(log)),
      found_(std::move(found)) {}

void RunContext::requestStop() {
  stop_ = true;
  std::lock_guard<std::mutex> lock(pauseMu_);
  pauseCv_.notify_all();
}

void RunContext::setPaused(bool paused) {
  std::lock_guard<std::mutex> lock(pauseMu_);
  paused_ = paused;
  pauseCv_.notify_all();
}

void RunContext::waitIfPaused() {
  if (!paused_) return;
  std::unique_lock<std::mutex> lock(pauseMu_);
  pauseCv_.wait(lock, [&] { return !paused_ || stop_; });
}

void RunContext::addWork(double w) { atomicAdd(work_, w); }

void RunContext::addResumedWork(double w) {
  atomicAdd(resumedWork_, w);
  atomicAdd(work_, w);
}

bool RunContext::allFound() const {
  if (foundCount_.load() == 0) return false;
  if (job_.algorithm != Algo::Scan) return true;
  std::lock_guard<std::mutex> lock(foundMu_);
  return foundHashes_.size() >= targets_.size();
}

std::vector<FoundKey> RunContext::found() const {
  std::lock_guard<std::mutex> lock(foundMu_);
  return foundList_;
}

bool RunContext::reportFound(const U256& key, const Affine& pub) {
  Affine actual = mulG(key);
  // Every target this key matches; hits[0] (compressed first) is the one reported.
  std::vector<Hash160> hits;
  bool compressedHit = true;
  if (!pub.inf && actual == pub) {
    if (hashesCompressed(job_.addressMode)) {
      Hash160 h = hash160Of(actual, true);
      if (targets_.contains(h)) hits.push_back(h);
    }
    if (hashesUncompressed(job_.addressMode)) {
      Hash160 h = hash160Of(actual, false);
      if (targets_.contains(h)) {
        if (hits.empty()) compressedHit = false;
        hits.push_back(h);
      }
    }
    if (hits.empty() && job_.pubkey && actual == *job_.pubkey) hits.push_back(hash160Of(actual, true));
  }
  if (hits.empty()) {
    log(LogLevel::Warn, "rejected unverified candidate " + key.toHex());
    return false;
  }

  FoundKey fk;
  {
    std::lock_guard<std::mutex> lock(foundMu_);
    // Recorded before the duplicate check: a repeat report may be the only one carrying the other encoding.
    foundHashes_.insert(hits.begin(), hits.end());
    if (!foundKeys_.insert(key).second) return true;
    const Hash160& hit = hits.front();
    fk.time = isoTimeNow();
    fk.puzzle = job_.puzzle;
    fk.address = encodeP2PKH(hit);
    fk.pubkey = pubkeyHex(actual, compressedHit);
    fk.privkey = key.toHex(true);
    fk.wif = compressedHit ? wifCompressed(key) : wifUncompressed(key);
    fk.algorithm = std::string(toString(job_.algorithm));
    fk.backend = backend_;
    foundList_.push_back(fk);
    foundCount_++;
  }
  found_(fk);
  return true;
}

void runThreads(size_t n, const std::function<void(size_t)>& body, const std::function<void()>& onFailure) {
  std::mutex mu;
  std::exception_ptr error;
  auto fail = [&](std::exception_ptr e) {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (!error) error = std::move(e);
    }
    onFailure();
  };
  std::vector<std::thread> threads;
  try {
    threads.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      threads.emplace_back([&, i] {
        try {
          body(i);
        } catch (...) {
          fail(std::current_exception());
        }
      });
    }
  } catch (...) {
    fail(std::current_exception());
  }
  for (auto& t : threads) t.join();
  if (error) std::rethrow_exception(error);
}

void runThreads(RunContext& ctx, size_t n, const std::function<void(size_t)>& body) {
  runThreads(n, body, [&ctx] { ctx.requestStop(); });
}

}  // namespace bsolver

#include "engine/run_context.h"

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
  std::optional<Hash160> hit;
  bool compressedHit = true;
  if (!pub.inf && actual == pub) {
    if (hashesCompressed(job_.addressMode)) {
      Hash160 h = hash160Of(actual, true);
      if (targets_.contains(h)) hit = h;
    }
    if (!hit && hashesUncompressed(job_.addressMode)) {
      Hash160 h = hash160Of(actual, false);
      if (targets_.contains(h)) {
        hit = h;
        compressedHit = false;
      }
    }
    if (!hit && job_.pubkey && actual == *job_.pubkey) hit = hash160Of(actual, true);
  }
  if (!hit) {
    log(LogLevel::Warn, "rejected unverified candidate " + key.toHex());
    return false;
  }

  FoundKey fk;
  {
    std::lock_guard<std::mutex> lock(foundMu_);
    if (!foundKeys_.insert(key).second) return true;
    foundHashes_.insert(*hit);
    fk.time = isoTimeNow();
    fk.puzzle = job_.puzzle;
    fk.address = encodeP2PKH(*hit);
    fk.pubkey = pubkeyHex(actual, compressedHit);
    fk.privkey = key.toHex(true);
    fk.wif = wifCompressed(key);
    fk.algorithm = std::string(toString(job_.algorithm));
    fk.backend = backend_;
    foundList_.push_back(fk);
    foundCount_++;
  }
  found_(fk);
  return true;
}

}  // namespace bsolver

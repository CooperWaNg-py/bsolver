#include "pool/pool_session.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>

#include "backend/backend.h"
#include "core/random.h"
#include "engine/console.h"
#include "engine/errors.h"
#include "engine/format.h"
#include "pool/http_client.h"

namespace bsolver {

namespace {
std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return s;
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string shortHash(const std::string& h) { return h.size() < 8 ? h : h.substr(0, 4) + "..." + h.substr(h.size() - 4); }

std::string joined(const std::vector<std::string>& v, const char* sep) {
  std::string s;
  for (const auto& x : v) s += (s.empty() ? "" : sep) + x;
  return s;
}

// Checkpointer::restore's error for an unreadable (torn or truncated) checkpoint file.
bool isCorruptCheckpointError(const std::string& error) {
  return error.rfind("checkpoint ", 0) == 0 && error.find(" is corrupt") != std::string::npos;
}
}  // namespace

PoolSession::PoolSession(JobManager& manager, PoolSessionOptions options)
    : manager_(manager), opts_(std::move(options)) {}

PoolSession::~PoolSession() { shutdownHelpers(); }

void PoolSession::shutdownHelpers() {
  {
    std::lock_guard<std::mutex> lock(waitMu_);
    helpersDone_ = true;
    waitCv_.notify_all();
  }
  if (pingThread_.joinable()) pingThread_.join();
  {
    std::lock_guard<std::mutex> lock(notifyMu_);
    notifierDone_ = true;
    notifyCv_.notify_all();
  }
  if (notifierThread_.joinable()) notifierThread_.join();
}

void PoolSession::requestStop() {
  {
    std::lock_guard<std::mutex> lock(waitMu_);
    stop_ = true;
    waitCv_.notify_all();
  }
  try {
    manager_.stop();
  } catch (const ConflictError&) {
  }
}

bool PoolSession::sleepUnlessStopped(std::chrono::seconds d) {
  std::unique_lock<std::mutex> lock(waitMu_);
  return !waitCv_.wait_for(lock, d, [&] { return stop_.load(); });
}

std::string PoolSession::display(const std::string& hex) const {
  if (!opts_.config.untrustedComputer || hex.size() < 2) return hex;
  std::string m = hex;
  m[1] = '*';
  m.back() = '*';
  return m;
}

void PoolSession::saveState() {
  std::lock_guard<std::mutex> lock(stateMu_);
  state_.save(statePath_);
}

void PoolSession::setPingHex(const std::string& hex) {
  std::lock_guard<std::mutex> lock(pingMu_);
  pingHex_ = hex;
}

void PoolSession::dropAssignment() {
  {
    std::lock_guard<std::mutex> lock(stateMu_);
    state_.assignment.reset();
    state_.save(statePath_);
  }
  setPingHex({});
}

std::string PoolSession::keyChannels() const {
  std::vector<std::string> c;
  if (opts_.config.telegramShare) c.push_back("telegram_share");
  if (opts_.config.apiShare) c.push_back("api_share");
  return joined(c, " and ");
}

std::string PoolSession::chooseBackend(std::vector<std::string>* names) {
  std::string id = opts_.backend;
  if (id == "auto") {
    log(LogLevel::Info, "benchmarking backends for scan (once per session)");
    auto results = manager_.bench(Algo::Scan, 2.0, opts_.threads);
    const BenchResult* best = nullptr;
    for (const auto& r : results) {
      log(LogLevel::Info, "  " + r.backend + " [" + joined(r.devices, ", ") + "] " +
                              (r.verified ? formatRate(r.rate, "keys/s") : "FAILED: " + r.error));
      if (r.verified && (!best || r.rate > best->rate)) best = &r;
    }
    if (!best) throw std::runtime_error("no working backend");
    id = best->backend;
  }
  Backend* b = findBackend(id);
  if (!b) throw std::runtime_error("backend " + id + " not available: not compiled in");
  auto devices = b->devices();
  if (devices.empty()) throw std::runtime_error("backend " + id + " not available: " + b->unavailableReason());
  names->clear();
  if (id == "cpu") {
    names->push_back("CPU (" + std::to_string(effectiveThreads(opts_.threads)) + " threads)");
  } else {
    for (const auto& d : devices) {
      if (opts_.devices.empty() || std::find(opts_.devices.begin(), opts_.devices.end(), d.index) != opts_.devices.end()) {
        names->push_back(d.name);
      }
    }
    if (names->empty()) throw std::runtime_error("no " + id + " device matches --devices");
  }
  return id;
}

void PoolSession::setup() {
  const PoolConfig& cfg = opts_.config;
  cfg.validate();
  if (!httpsSupported()) {
    throw std::runtime_error("pool mode needs HTTPS, but bsolver was built without libcurl (WinHTTP on Windows)");
  }
  if (!cfg.publicKey.empty()) rsaKey_ = parseRsaPublicKeyPem(cfg.publicKey);

  statePath_ = poolStatePath(manager_.options().stateDir, cfg.targetPuzzle);
  std::optional<PoolState> loaded = PoolState::load(statePath_);
  const std::string tokenId = poolTokenId(cfg.userToken);
  if (loaded && loaded->tokenId != tokenId) {
    log(LogLevel::Warn, "saved pool state belongs to a different user token; discarding its range assignment");
    loaded->assignment.reset();
    loaded->workerName.clear();
  }
  state_ = loaded.value_or(PoolState{});
  if (state_.assignment) {
    const std::string bad = rangeBoundsError(state_.assignment->range);
    if (!bad.empty()) {
      log(LogLevel::Error, "saved range " + display(state_.assignment->range.hex) + " cannot be scanned (" + bad +
                               "); dropping it and fetching a new range");
      state_.assignment.reset();
    }
  }
  state_.puzzle = cfg.targetPuzzle;
  state_.tokenId = tokenId;
  if (!cfg.workerName.empty()) {
    state_.workerName = cfg.workerName;
  } else if (state_.workerName.empty()) {
    state_.workerName = "worker" + std::to_string(Rng().uniform(U256::fromU64(10000), U256::fromU64(99999)).w[0]);
  }
  saveState();

  std::vector<std::string> deviceNames;
  backend_ = chooseBackend(&deviceNames);
  gpuCount_ = static_cast<int>(deviceNames.size());
  api_ = std::make_unique<PoolApi>(cfg, state_.workerName, joined(deviceNames, " + "), opts_.telegramApiUrl);

  log(LogLevel::Info, "btcpuzzle.info worker " + state_.workerName + ", puzzle " + std::to_string(cfg.targetPuzzle) +
                          ", backend " + backend_ + " (" + joined(deviceNames, ", ") + "), custom range " +
                          cfg.customRange);
  log(LogLevel::Info, "security hash " + api_->securityHash());
  log(LogLevel::Info, std::string("untrusted computer ") + (cfg.untrustedComputer ? "on" : "off") +
                          (rsaKey_ ? " (RSA " + std::to_string(rsaKey_->modulusBytes() * 8) + "-bit)" : "") +
                          ", telegram " + (cfg.telegramShare ? "on" : "off") + ", api share " +
                          (cfg.apiShare ? "on" : "off") + ", save key " + (cfg.saveKey ? "on" : "off"));
  if (cfg.untrustedComputer && cfg.apiShare && lower(cfg.apiShareUrl).rfind("http://", 0) == 0) {
    log(LogLevel::Warn, std::string("api_share_url uses plain http:// on an untrusted computer; ") +
                            (rsaKey_ ? "notifications are sent without TLS"
                                     : "a found key would be sent in plaintext over the network") +
                            " (use https://)");
  }
  if (cfg.untrustedComputer && !rsaKey_) {
    log(LogLevel::Info, "untrusted computer without public_key: a found target key is delivered via " + keyChannels() +
                            " only and is not stored or printed locally");
  }
  if (state_.rangesSubmitted) {
    log(LogLevel::Info, std::to_string(state_.rangesSubmitted) + " ranges submitted so far from this state directory");
  }
  if (state_.assignment) {
    const auto& a = *state_.assignment;
    log(LogLevel::Info, "resuming range " + display(a.range.hex) + " assigned " + a.assignedAt + " (" +
                            std::to_string(a.proofKeys.size()) + "/" + std::to_string(a.range.proofAddresses.size()) +
                            " proof keys found" + (a.scanComplete ? ", scan complete" : "") + ")");
    setPingHex(a.range.hex);
  }
}

int PoolSession::run() {
  int rc = 0;
  bool started = false;
  try {
    setup();
    manager_.setFoundHandler([this](const FoundKey& k) { onFound(k); });
    pingThread_ = std::thread(&PoolSession::pingLoop, this);
    notifierThread_ = std::thread(&PoolSession::notifierLoop, this);
    started = true;
    const std::string startMsg = state_.workerName + " started job! (" + shortHash(api_->securityHash()) + ")";
    if (!api_->telegram(startMsg)) api_->apiShare("workerStarted", {});

    while (!stop_) {
      bool haveAssignment, scanned;
      {
        std::lock_guard<std::mutex> lock(stateMu_);
        haveAssignment = state_.assignment.has_value();
        scanned = haveAssignment && state_.assignment->scanComplete;
      }
      // One step per iteration: each step may drop the assignment (rejected/unscannable range).
      if (!haveAssignment) {
        if (!fetchAssignment()) break;
        continue;
      }
      if (!scanned) {
        if (!scanAssignment()) break;
        continue;
      }
      if (!submitAssignment()) break;
      if (opts_.maxRanges && submittedThisRun_ >= *opts_.maxRanges) {
        log(LogLevel::Info, "submitted " + std::to_string(submittedThisRun_) + " range(s); stopping as requested");
        break;
      }
    }
    rc = stop_ ? 130 : 0;
  } catch (const std::exception& ex) {
    log(LogLevel::Error, ex.what());
    rc = 1;
  }
  manager_.setFoundHandler(nullptr);
  if (started) {
    shutdownHelpers();
    const std::string stopMsg = state_.workerName + " went offline! (" + shortHash(api_->securityHash()) + ")";
    if (!api_->telegram(stopMsg)) api_->apiShare("workerExited", {});
  }
  return rc;
}

bool PoolSession::fetchAssignment() {
  for (;;) {
    if (stop_) return false;
    try {
      PoolRange r = api_->getRange();
      PoolAssignment a{r, {}, false, isoTimeNow()};
      {
        std::lock_guard<std::mutex> lock(stateMu_);
        state_.assignment = a;
        state_.save(statePath_);
      }
      setPingHex(r.hex);
      char bits[32];
      std::snprintf(bits, sizeof(bits), "2^%.2f", std::log2((r.end() - r.start() + 1).toDouble()));
      log(LogLevel::Info, "assigned range " + display(r.hex) + " (" + std::to_string(r.proofAddresses.size()) +
                              " proof addresses, " + bits + " keys)");
      return true;
    } catch (const PoolError& ex) {
      if (!ex.retryable()) throw std::runtime_error(ex.what());
      log(LogLevel::Warn, std::string(ex.what()) + "; retrying in " + std::to_string(opts_.retryDelay.count()) + " s");
      if (!sleepUnlessStopped(opts_.retryDelay)) return false;
    }
  }
}

bool PoolSession::scanAssignment() {
  PoolRange r;
  {
    std::lock_guard<std::mutex> lock(stateMu_);
    r = state_.assignment->range;
  }
  JobSpec spec;
  spec.addresses = r.proofAddresses;
  spec.addresses.push_back(r.targetAddress);
  spec.rangeStart = r.start().toHex();
  spec.rangeEnd = r.end().toHex();
  spec.algorithm = Algorithm::Scan;
  spec.order = Order::Sequential;
  spec.addressMode = AddressMode::Compressed;
  spec.backend = backend_;
  spec.threads = opts_.threads;
  spec.devices = opts_.devices;
  spec.checkpoint = true;
  spec.stopOnFound = false;  // the whole range must be covered to find every proof key (and the target)

  // Engine rejections (JobError) are fatal: fetched ranges passed parseRangeResponse and setup() dropped any saved
  // range outside the key space, so what remains is a local configuration problem that would fail every range.
  for (;;) {
    manager_.start(spec);
    if (stop_) {
      try {
        manager_.stop();
      } catch (const ConflictError&) {
      }
    }
    manager_.wait();
    JobStatus st = manager_.status();
    if (st.state == JobState::Failed) {
      const std::string err = st.error.value_or("unknown error");
      if (!spec.fresh && !stop_ && isCorruptCheckpointError(err)) {
        // A crash or power loss can leave a torn checkpoint (the engine logged its path); rescanning the pool range
        // from its start is always correct.
        log(LogLevel::Warn, "range " + display(r.hex) + " has a corrupt scan checkpoint; rescanning it from its start");
        spec.fresh = true;
        continue;
      }
      throw std::runtime_error("scan failed: " + err);
    }
    if (!st.exhausted) {
      // Stopped (Ctrl-C or the web UI): the checkpoint and the assignment stay for the next start.
      stop_ = true;
      return false;
    }
    break;
  }
  std::lock_guard<std::mutex> lock(stateMu_);
  state_.assignment->scanComplete = true;
  state_.save(statePath_);
  return true;
}

bool PoolSession::submitAssignment() {
  PoolRange r;
  std::vector<std::string> keys;
  {
    std::lock_guard<std::mutex> lock(stateMu_);
    const PoolAssignment& a = *state_.assignment;
    r = a.range;
    for (const auto& addr : r.proofAddresses) {
      auto it = a.proofKeys.find(addr);
      if (it != a.proofKeys.end()) keys.push_back(it->second);
    }
  }
  const std::filesystem::path stateDir = manager_.options().stateDir;
  auto writeReport = [&](const std::string& kind) {
    std::filesystem::path p = stateDir / ("pool-" + std::to_string(opts_.config.targetPuzzle) + "-" + kind + "-" + r.hex + ".txt");
    std::ofstream out(p);
    out << "range " << r.hex << " (" << r.start().toHex() << ".." << r.end().toHex() << ")\ntarget " << r.targetAddress
        << "\nproof keys found " << keys.size() << "/" << r.proofAddresses.size() << "\n";
    std::lock_guard<std::mutex> lock(stateMu_);
    for (const auto& addr : r.proofAddresses) {
      auto it = state_.assignment->proofKeys.find(addr);
      out << addr << " " << (it == state_.assignment->proofKeys.end() ? "MISSING" : it->second) << "\n";
    }
    return p;
  };
  if (keys.size() != r.proofAddresses.size()) {
    // A backend miss or a server anomaly: rescanning cannot help, and keeping the assignment would wedge the client.
    auto p = writeReport("incomplete");
    log(LogLevel::Error, "range " + display(r.hex) + " was scanned but only " + std::to_string(keys.size()) + "/" +
                             std::to_string(r.proofAddresses.size()) + " proof keys were found (details: " +
                             p.string() + "); dropping it and fetching a new range");
    dropAssignment();
    return true;
  }

  const std::string proof = hashedProofKey(keys);
  for (;;) {
    try {
      api_->submitRange(r.hex, proof, gpuCount_);
      break;
    } catch (const PoolError& ex) {
      if (!ex.retryable()) {
        auto p = writeReport("rejected");
        log(LogLevel::Error, "pool rejected range " + display(r.hex) + ": " + ex.what() + " (details: " + p.string() +
                                 "); fetching a new range");
        dropAssignment();
        return true;
      }
      log(LogLevel::Warn, std::string(ex.what()) + "; retrying in " + std::to_string(opts_.retryDelay.count()) + " s");
      if (!sleepUnlessStopped(opts_.retryDelay)) return false;
    }
  }
  uint64_t total;
  {
    std::lock_guard<std::mutex> lock(stateMu_);
    state_.rangesSubmitted++;
    state_.lastSubmittedHex = r.hex;
    state_.lastSubmittedAt = isoTimeNow();
    state_.assignment.reset();
    state_.save(statePath_);
    total = state_.rangesSubmitted;
  }
  setPingHex({});
  submittedThisRun_++;
  log(LogLevel::Info, "range " + display(r.hex) + " submitted (" + std::to_string(total) + " total)");
  const std::string msg = "\u2705 " + r.hex + " scanned by " + state_.workerName + " (" + shortHash(api_->securityHash()) + ")";
  if (!api_->telegram(msg)) api_->apiShare("rangeScanned", {{"HEX", r.hex}});
  return true;
}

void PoolSession::onFound(const FoundKey& key) {
  std::string hex;
  bool target = false;
  bool pool = false;
  {
    std::lock_guard<std::mutex> lock(stateMu_);
    if (state_.assignment) {
      PoolAssignment& a = *state_.assignment;
      const auto& proofs = a.range.proofAddresses;
      const bool proof = std::find(proofs.begin(), proofs.end(), key.address) != proofs.end();
      if (proof && !a.proofKeys.count(key.address)) {
        a.proofKeys[key.address] = upper(key.privkey);
        state_.save(statePath_);
        manager_.log(LogLevel::Info, "pool: proof key " + std::to_string(a.proofKeys.size()) + "/" +
                                         std::to_string(proofs.size()) + " found");
      }
      target = key.address == a.range.targetAddress;
      pool = proof || target;
      hex = a.range.hex;
    }
  }
  if (target) {
    recordTarget(key, hex);
  } else if (!pool) {
    // Not part of the current assignment (e.g. a job started outside the pool loop): record it like a normal job.
    try {
      appendResult(manager_.options().resultsPath, key);
    } catch (const std::exception& ex) {
      log(LogLevel::Error, ex.what());
    }
    consoleOut(foundLine(key));
  }
}

void PoolSession::recordTarget(const FoundKey& key, const std::string& hex) {
  const PoolConfig& cfg = opts_.config;
  std::string secret = upper(key.privkey);
  bool encrypted = false;
  if (cfg.untrustedComputer && rsaKey_) {
    secret = rsaOaepEncryptBase64(*rsaKey_, secret);
    encrypted = true;
  }
  const std::string found = "TARGET KEY FOUND puzzle #" + std::to_string(cfg.targetPuzzle) + " address " + key.address;
  if (cfg.untrustedComputer && !encrypted) {
    // No public_key: like the official client, the plaintext key leaves only through the notification channels;
    // nothing is written to disk or the console of a computer declared untrusted.
    consoleOut(found + "; key delivered via " + keyChannels() + " only");
  } else {
    FoundKey rec = key;
    if (encrypted) {
      rec.privkey = "rsa-oaep:" + secret;
      rec.wif.clear();
    }
    try {
      appendResult(manager_.options().resultsPath, rec);
    } catch (const std::exception& ex) {
      log(LogLevel::Error, ex.what());
    }
    consoleOut(found +
               (encrypted ? " privkey (RSA-encrypted) " + secret : " privkey " + key.privkey + " WIF " + key.wif));
  }
  std::lock_guard<std::mutex> lock(notifyMu_);
  notifications_.push_back(Notification{key.address, secret, hex, encrypted});
  notifyCv_.notify_all();
}

void PoolSession::notifierLoop() {
  const PoolConfig& cfg = opts_.config;
  std::unique_lock<std::mutex> lock(notifyMu_);
  for (;;) {
    notifyCv_.wait(lock, [&] { return !notifications_.empty() || notifierDone_; });
    if (notifications_.empty()) return;
    Notification n = notifications_.front();
    lock.unlock();

    std::string text = "\U0001F3AF TARGET KEY FOUND!\n\nPuzzle: " + std::to_string(cfg.targetPuzzle) +
                       "\nWorker: " + state_.workerName + "\n\nAddress:\n" + n.address + "\n\n" +
                       (n.encrypted ? "Encrypted Private Key:\n" : "Private Key:\n") + n.secret;
    bool telegramOk = !cfg.telegramShare, apiOk = !cfg.apiShare, savedOk = !(cfg.saveKey && n.encrypted);
    for (int attempt = 1;; ++attempt) {
      if (!telegramOk) telegramOk = api_->telegram(text);
      if (!apiOk) {
        apiOk = api_->apiShare("keyFound", {{"HEX", n.hex},
                                            {"PrivateKey", n.secret},
                                            {"Address", n.address},
                                            {"Encrypted", n.encrypted ? "true" : "false"}});
      }
      if (!savedOk) {
        try {
          api_->saveKey(n.secret);
          savedOk = true;
        } catch (const PoolError& ex) {
          log(LogLevel::Warn, ex.what());
        }
      }
      if (telegramOk && apiOk && savedOk) {
        if (cfg.telegramShare || cfg.apiShare || cfg.saveKey) log(LogLevel::Info, "found key delivered");
        break;
      }
      lock.lock();
      bool done = notifierDone_;
      // Without a local copy (untrusted computer, no public_key) delivery is retried until it succeeds.
      const bool localCopy = n.encrypted || !cfg.untrustedComputer;
      if (done && attempt >= 3 && localCopy) {
        lock.unlock();
        log(LogLevel::Error, "could not deliver the found key notification; it is in the results file " +
                                 manager_.options().resultsPath.string());
        break;
      }
      notifyCv_.wait_for(lock, std::chrono::seconds(5));
      lock.unlock();
      log(LogLevel::Warn, "found key notification failed; retrying (attempt " + std::to_string(attempt + 1) + ")");
    }
    lock.lock();
    notifications_.pop_front();
  }
}

void PoolSession::pingLoop() {
  std::unique_lock<std::mutex> lock(waitMu_);
  while (!waitCv_.wait_for(lock, opts_.pingInterval, [&] { return stop_.load() || helpersDone_; })) {
    std::string hex;
    {
      std::lock_guard<std::mutex> p(pingMu_);
      hex = pingHex_;
    }
    if (hex.empty()) continue;
    lock.unlock();
    try {
      api_->ping(hex);
    } catch (const PoolError& ex) {
      log(LogLevel::Warn, ex.what());
    }
    lock.lock();
  }
}

}  // namespace bsolver

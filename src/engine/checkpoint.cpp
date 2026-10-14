#include "engine/checkpoint.h"

#include <fstream>
#include <iterator>
#include <stdexcept>

#include "core/hex.h"
#include "core/sha256.h"
#include "engine/bytes.h"

namespace bsolver {

namespace {
constexpr char kMagic[4] = {'B', 'S', 'C', 'K'};
constexpr uint32_t kVersion = 1;

uint8_t algoByte(Algo a) { return static_cast<uint8_t>(a); }

std::runtime_error corrupt(const std::filesystem::path& path) {
  return std::runtime_error("checkpoint " + path.string() + " is corrupt (use --fresh)");
}
}  // namespace

Checkpointer::Checkpointer(const std::filesystem::path& stateDir, Algo algo, std::string identity, bool enabled,
                           bool fresh)
    : algo_(algo),
      identity_(std::move(identity)),
      enabled_(enabled),
      fresh_(fresh),
      lastSave_(std::chrono::steady_clock::now()) {
  uint8_t digest[32];
  sha256(reinterpret_cast<const uint8_t*>(identity_.data()), identity_.size(), digest);
  path_ = stateDir / (std::string(toString(algo)) + "-" + hexEncode(digest, 8) + ".ckpt");
}

std::optional<std::vector<uint8_t>> Checkpointer::loadFile(const std::filesystem::path& path, Algo algo,
                                                           const std::string& identity) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::nullopt;
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::string mismatch = "checkpoint " + path.string() + " does not match this job";
  try {
    ByteReader r(data);
    char magic[4];
    r.bytes(magic, 4);
    if (std::memcmp(magic, kMagic, 4) != 0) throw corrupt(path);
    uint32_t version = r.u32();
    if (version != kVersion) {
      throw std::runtime_error("checkpoint " + path.string() + " has unsupported version " + std::to_string(version));
    }
    uint32_t len = r.u32();
    std::string id = r.str(len);
    uint8_t a = r.u8();
    if (id != identity || a != algoByte(algo)) throw std::runtime_error(mismatch);
    std::vector<uint8_t> payload(data.end() - static_cast<std::ptrdiff_t>(r.remaining()), data.end());
    return payload;
  } catch (const std::runtime_error& ex) {
    if (std::string(ex.what()) == "truncated data") throw corrupt(path);
    throw;
  }
}

std::optional<std::vector<uint8_t>> Checkpointer::load() {
  if (!enabled_ || fresh_) return std::nullopt;
  return loadFile(path_, algo_, identity_);
}

bool Checkpointer::restore(const std::function<void(ByteReader&)>& parse) {
  auto payload = load();
  if (!payload) return false;
  try {
    ByteReader r(*payload);
    parse(r);
    if (r.remaining() != 0) throw std::runtime_error("trailing bytes");
  } catch (const std::runtime_error&) {
    throw corrupt(path_);
  }
  return true;
}

void Checkpointer::setSnapshot(std::function<std::vector<uint8_t>()> fn) {
  std::lock_guard<std::mutex> lock(mu_);
  snapshot_ = std::move(fn);
}

void Checkpointer::clearSnapshot() {
  std::lock_guard<std::mutex> lock(mu_);
  snapshot_ = nullptr;
}

void Checkpointer::maybeSave() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (std::chrono::steady_clock::now() - lastSave_ < kInterval) return;
  }
  saveNow();
}

void Checkpointer::saveNow() {
  std::lock_guard<std::mutex> lock(mu_);
  lastSave_ = std::chrono::steady_clock::now();
  if (!enabled_ || !snapshot_) return;
  write(snapshot_());
}

void Checkpointer::write(const std::vector<uint8_t>& payload) {
  ByteWriter header;
  header.bytes(kMagic, 4);
  header.u32(kVersion);
  header.u32(static_cast<uint32_t>(identity_.size()));
  header.bytes(identity_.data(), identity_.size());
  header.u8(algoByte(algo_));

  std::filesystem::create_directories(path_.parent_path());
  std::filesystem::path tmp = path_;
  tmp += ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write checkpoint " + tmp.string());
    out.write(reinterpret_cast<const char*>(header.data().data()), static_cast<std::streamsize>(header.data().size()));
    out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    // close() flushes; a failed flush (disk full, I/O error) must not replace the previous checkpoint.
    out.close();
    if (!out) {
      std::error_code ec;
      std::filesystem::remove(tmp, ec);
      throw std::runtime_error("cannot write checkpoint " + tmp.string());
    }
  }
  std::filesystem::rename(tmp, path_);
}

void Checkpointer::remove() {
  std::lock_guard<std::mutex> lock(mu_);
  snapshot_ = nullptr;
  if (!enabled_) return;
  std::error_code ec;
  std::filesystem::remove(path_, ec);
}

}  // namespace bsolver

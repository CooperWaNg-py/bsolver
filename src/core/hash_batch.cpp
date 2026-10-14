// Baseline implementations (scalar; NEON on arm64, SSE2 on x86-64) and the run-time selection. The implementations
// that need extra instruction sets live in hash_batch_{armv8,avx2,shani}.cpp, compiled with their own flags when the
// compiler supports them (cmake/CpuHash.cmake, which defines BSOLVER_HAVE_HASH_* for this file).
#if defined(__aarch64__) || defined(_M_ARM64)
#define BSOLVER_LANES_NEON 1
#elif defined(__x86_64__) || defined(_M_X64)
#define BSOLVER_LANES_SSE2 1
#endif
#include "core/hash_batch_impl.h"

#include <stdexcept>
#include <string>

#include "core/cpu_features.h"

namespace bsolver {

namespace hashimpl {
const Hash160Batch kScalar{lanesCompressed<Lane1>, lanesUncompressed<Lane1>};
#if defined(BSOLVER_LANES_NEON)
const Hash160Batch kNeon{lanesCompressed<Neon8>, lanesUncompressed<Neon8>};
#endif
#if defined(BSOLVER_LANES_SSE2)
const Hash160Batch kSse2{lanesCompressed<Sse8>, lanesUncompressed<Sse8>};
#endif
}  // namespace hashimpl

namespace {
// The table for impl when it is compiled in and this CPU runs it, else nullptr.
const Hash160Batch* usableTable(HashImpl impl) {
  [[maybe_unused]] const CpuFeatures& cpu = cpuFeatures();
  switch (impl) {
    case HashImpl::Scalar: return &hashimpl::kScalar;
#if defined(BSOLVER_LANES_SSE2)
    case HashImpl::Sse2: return &hashimpl::kSse2;
#endif
#if defined(BSOLVER_HAVE_HASH_AVX2)
    case HashImpl::Avx2: return cpu.x86Avx2 ? &hashimpl::kAvx2 : nullptr;
#endif
#if defined(BSOLVER_HAVE_HASH_SHANI)
    case HashImpl::ShaNi: return cpu.x86Sha && cpu.x86Sse41 && cpu.x86Avx2 ? &hashimpl::kShaNi : nullptr;
#endif
#if defined(BSOLVER_LANES_NEON)
    case HashImpl::Neon: return &hashimpl::kNeon;
#endif
#if defined(BSOLVER_HAVE_HASH_ARMV8)
    case HashImpl::ArmSha2: return cpu.armSha2 ? &hashimpl::kArmSha2 : nullptr;
#endif
    default: return nullptr;
  }
}
}  // namespace

std::string_view hashImplName(HashImpl impl) {
  switch (impl) {
    case HashImpl::Scalar: return "scalar";
    case HashImpl::Sse2: return "sse2";
    case HashImpl::Avx2: return "avx2";
    case HashImpl::ShaNi: return "sha-ni";
    case HashImpl::Neon: return "neon";
    case HashImpl::ArmSha2: return "armv8-sha2";
  }
  return "unknown";
}

const std::vector<HashImpl>& supportedHashImpls() {
  static const std::vector<HashImpl> impls = [] {
    std::vector<HashImpl> v;
    for (HashImpl impl : {HashImpl::Scalar, HashImpl::Sse2, HashImpl::Avx2, HashImpl::ShaNi, HashImpl::Neon,
                          HashImpl::ArmSha2}) {
      if (usableTable(impl)) v.push_back(impl);
    }
    return v;
  }();
  return impls;
}

const Hash160Batch& hash160Batch(HashImpl impl) {
  const Hash160Batch* t = usableTable(impl);
  if (!t) throw std::invalid_argument("hash implementation " + std::string(hashImplName(impl)) + " not supported here");
  return *t;
}

HashImpl bestHashImpl() { return supportedHashImpls().back(); }

}  // namespace bsolver

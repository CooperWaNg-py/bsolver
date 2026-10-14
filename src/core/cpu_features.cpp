#include "core/cpu_features.h"

#if defined(__x86_64__) || defined(_M_X64)
#define BSOLVER_CPU_X86_64 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define BSOLVER_CPU_ARM64 1
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif

namespace bsolver {

namespace {

#if defined(BSOLVER_CPU_X86_64)
void cpuid(unsigned leaf, unsigned sub, unsigned (&r)[4]) {
#if defined(_MSC_VER) && !defined(__clang__)
  int v[4];
  __cpuidex(v, static_cast<int>(leaf), static_cast<int>(sub));
  for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(v[i]);
#else
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

// XCR0: which register states the OS saves on context switch.
unsigned long long xcr0() {
#if defined(_MSC_VER) && !defined(__clang__)
  return _xgetbv(0);
#else
  unsigned lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
}
#endif

CpuFeatures detect() {
  CpuFeatures f;
#if defined(BSOLVER_CPU_X86_64)
  unsigned r[4];
  cpuid(0, 0, r);
  const unsigned maxLeaf = r[0];
  cpuid(1, 0, r);
  const unsigned ecx1 = r[2];
  f.x86Sse41 = (ecx1 >> 19) & 1;
  const bool osxsave = (ecx1 >> 27) & 1, avx = (ecx1 >> 28) & 1;
  const bool ymmEnabled = osxsave && avx && (xcr0() & 0x6) == 0x6;  // XMM and YMM state
  if (maxLeaf >= 7) {
    cpuid(7, 0, r);
    f.x86Avx2 = ymmEnabled && ((r[1] >> 5) & 1);
    f.x86Sha = (r[1] >> 29) & 1;
  }
#elif defined(BSOLVER_CPU_ARM64)
#if defined(__APPLE__)
  int v = 0;
  size_t len = sizeof(v);
  f.armSha2 = sysctlbyname("hw.optional.arm.FEAT_SHA256", &v, &len, nullptr, 0) == 0 && v != 0;
#elif defined(_WIN32)
  f.armSha2 = IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE) != 0;
#elif defined(__linux__)
  f.armSha2 = (getauxval(AT_HWCAP) & HWCAP_SHA2) != 0;
#endif
#endif
  return f;
}

}  // namespace

const CpuFeatures& cpuFeatures() {
  static const CpuFeatures features = detect();
  return features;
}

}  // namespace bsolver

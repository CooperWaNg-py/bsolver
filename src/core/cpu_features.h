#pragma once

namespace bsolver {

// Instruction-set extensions usable on this machine (CPU and OS support), detected once at first use.
struct CpuFeatures {
  bool armSha2 = false;   // ARMv8 SHA-256 instructions
  bool x86Sse41 = false;  // SSE4.1
  bool x86Avx2 = false;   // AVX2 with OS-enabled YMM state
  bool x86Sha = false;    // SHA extensions (SHA-NI)
};

const CpuFeatures& cpuFeatures();

}  // namespace bsolver

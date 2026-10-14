# Instruction-set-specific hash160 batch implementations (src/core/hash_batch_{armv8,avx2,shani}.cpp).
#
# Each file is compiled with its own instruction-set flags, and only when the compiler accepts them; the program picks
# an implementation at run time from the CPU's features (src/core/cpu_features.cpp), so the flags never reach code that
# runs unconditionally. Produces BSOLVER_HASH_SOURCES (the variant files to compile) and tells src/core/hash_batch.cpp
# which variants exist (BSOLVER_HAVE_HASH_*). Must be included from the directory that creates bsolver_core.
include(CheckCXXSourceCompiles)

set(_bsolver_hash_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
set(BSOLVER_HASH_SOURCES "")
set(_bsolver_hash_have "")

# _bsolver_hash_variant(<name> <file> <HAVE definition> <test source> <flag alternatives...>)
# Each flag alternative is a space-separated flag string ("" = no extra flags); the first that compiles wins.
function(_bsolver_hash_variant name file have source)
  foreach(flags IN LISTS ARGN)
    string(MAKE_C_IDENTIFIER "BSOLVER_HASH_${name}_${flags}" var)
    set(CMAKE_REQUIRED_FLAGS "${flags}")
    set(CMAKE_REQUIRED_QUIET ON)
    check_cxx_source_compiles("${source}" ${var})
    if(${var})
      separate_arguments(opts NATIVE_COMMAND "${flags}")
      set_source_files_properties(${_bsolver_hash_dir}/${file} PROPERTIES COMPILE_OPTIONS "${opts}")
      set(BSOLVER_HASH_SOURCES ${BSOLVER_HASH_SOURCES} ${_bsolver_hash_dir}/${file} PARENT_SCOPE)
      set(_bsolver_hash_have ${_bsolver_hash_have} ${have} PARENT_SCOPE)
      message(STATUS "bsolver: ${name} hash160 enabled (flags: '${flags}')")
      return()
    endif()
  endforeach()
  message(STATUS "bsolver: ${name} hash160 not supported by this compiler/target")
endfunction()

_bsolver_hash_variant(armv8-sha2 hash_batch_armv8.cpp BSOLVER_HAVE_HASH_ARMV8 [[
#include <arm_neon.h>
int main() {
  uint32x4_t a = vdupq_n_u32(1), b = vdupq_n_u32(2), c = vdupq_n_u32(3);
  a = vsha256su1q_u32(vsha256su0q_u32(a, b), b, c);
  b = vsha256h2q_u32(vsha256hq_u32(a, b, c), a, c);
  return static_cast<int>(vgetq_lane_u32(b, 0));
}]] "" "-march=armv8-a+crypto")

if(MSVC)
  set(_bsolver_avx2_flags "/arch:AVX2")
  set(_bsolver_shani_flags "/arch:AVX2")
else()
  set(_bsolver_avx2_flags "-mavx2")
  set(_bsolver_shani_flags "-mavx2 -msha")
endif()

_bsolver_hash_variant(avx2 hash_batch_avx2.cpp BSOLVER_HAVE_HASH_AVX2 [[
#include <immintrin.h>
int main() {
  __m256i a = _mm256_set1_epi32(1);
  a = _mm256_shuffle_epi8(_mm256_add_epi32(a, _mm256_slli_epi32(a, 3)), a);
  return _mm256_extract_epi32(a, 0);
}]] "${_bsolver_avx2_flags}")

_bsolver_hash_variant(sha-ni hash_batch_shani.cpp BSOLVER_HAVE_HASH_SHANI [[
#include <immintrin.h>
int main() {
  __m128i a = _mm_set1_epi32(1), b = _mm_set1_epi32(2);
  a = _mm_sha256msg2_epu32(_mm_sha256msg1_epu32(a, b), _mm_alignr_epi8(a, b, 4));
  a = _mm_sha256rnds2_epu32(a, b, _mm_blend_epi16(a, b, 0xF0));
  __m256i c = _mm256_add_epi32(_mm256_castsi128_si256(a), _mm256_set1_epi32(1));
  return _mm256_extract_epi32(c, 0);
}]] "${_bsolver_shani_flags}")

if(_bsolver_hash_have)
  set_source_files_properties(${_bsolver_hash_dir}/hash_batch.cpp PROPERTIES COMPILE_DEFINITIONS "${_bsolver_hash_have}")
endif()

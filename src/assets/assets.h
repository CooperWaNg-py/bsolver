#pragma once

#include <cstddef>
#include <string_view>

namespace bsolver::assets {

struct Asset {
  std::string_view path, mime;
  const unsigned char* data;
  size_t size;
};

// Web UI files (count 0 when built with BSOLVER_WEBUI=OFF).
extern const Asset kWebAssets[];
extern const size_t kWebAssetsCount;

// data/puzzles.json (NUL-terminated).
extern const std::string_view kPuzzleCatalogJson;

// GPU kernel sources (kernel_core.h + backend kernel file), only defined when the backend is built.
extern const std::string_view kOpenCLKernelSource;
extern const std::string_view kMetalKernelSource;

}  // namespace bsolver::assets

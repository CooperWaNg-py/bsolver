cmake_minimum_required(VERSION 3.24)
project(bsolver VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BSOLVER_WEBUI "Build and embed the React web UI (needs npm)" ON)
option(BSOLVER_OPENCL "Build the OpenCL backend when OpenCL is found" ON)
option(BSOLVER_METAL "Build the Metal backend (Apple only)" ON)
option(BSOLVER_CUDA "Build the CUDA backend when a CUDA toolkit is found" ON)
option(BSOLVER_NATIVE "Optimize for the host CPU" OFF)
option(BSOLVER_POOL "btcpuzzle.info pool client (needs libcurl except on Windows)" ON)
option(BSOLVER_TESTS "Build the test suite" ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(MSVC)
  add_compile_options(/W4 /EHsc /permissive- /bigobj /utf-8)
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " /O2")
  if(BSOLVER_NATIVE)
    message(WARNING "BSOLVER_NATIVE is not supported with MSVC (no -march=native equivalent); ignoring it")
  endif()
else()
  add_compile_options(-Wall -Wextra)
  string(REPLACE "-O2" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
  if(BSOLVER_NATIVE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
      add_compile_options(-mcpu=native)
    else()
      add_compile_options(-march=native)
    endif()
  endif()
endif()

find_package(Threads REQUIRED)
include(Dependencies)
include(Embed)
include(Backends)
include(WebUI)
include(Https)
include(CpuHash)

file(GLOB_RECURSE BSOLVER_CORE_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/catalog/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/server/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cpu/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/gpu/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pool/*.cpp)
# Toolkit-specific backends are added by cmake/Backends.cmake only when their toolkit is found.
list(FILTER BSOLVER_CORE_SOURCES EXCLUDE REGEX "/src/backend/(opencl|metal|cuda)/")
# Instruction-set-specific hash variants are added by cmake/CpuHash.cmake only when the compiler supports them.
list(FILTER BSOLVER_CORE_SOURCES EXCLUDE REGEX "/src/core/hash_batch_[a-z0-9]+\\.cpp$")

bsolver_embed(OUTPUT ${CMAKE_BINARY_DIR}/generated/catalog_data.cpp
  SYMBOL PuzzleCatalogJson MODE text
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/puzzles.json)

add_library(bsolver_core STATIC
  ${BSOLVER_CORE_SOURCES}
  ${CMAKE_BINARY_DIR}/generated/catalog_data.cpp
  ${BSOLVER_WEBUI_ASSETS_CPP}
  ${BSOLVER_BACKEND_SOURCES}
  ${BSOLVER_HASH_SOURCES})
target_include_directories(bsolver_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(bsolver_core PUBLIC
  BSOLVER_VERSION="${PROJECT_VERSION}"
  ${BSOLVER_BACKEND_DEFINITIONS}
  ${BSOLVER_HTTPS_DEFINITIONS})
target_link_libraries(bsolver_core PUBLIC
  Threads::Threads nlohmann_json::nlohmann_json httplib::httplib
  ${BSOLVER_BACKEND_LIBRARIES}
  ${BSOLVER_HTTPS_LIBRARIES})
if(WIN32)
  target_link_libraries(bsolver_core PUBLIC ws2_32)
  target_compile_definitions(bsolver_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00)
endif()
bsolver_configure_backend_target(bsolver_core)

add_executable(bsolver src/main.cpp)
target_link_libraries(bsolver PRIVATE bsolver_core CLI11::CLI11)

if(BSOLVER_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

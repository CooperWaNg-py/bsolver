# GPU backend detection. Produces BSOLVER_BACKEND_SOURCES / _DEFINITIONS / _LIBRARIES
# and bsolver_configure_backend_target(<target>) for per-target settings.
#
# Each backend is compiled only when its toolkit is present; the kernel core
# (src/backend/gpu/kernel_core.h) is text-embedded for the runtime-compiled backends.
set(BSOLVER_BACKEND_SOURCES "")
set(BSOLVER_BACKEND_DEFINITIONS "")
set(BSOLVER_BACKEND_LIBRARIES "")
set(BSOLVER_HAS_OPENCL OFF)
set(BSOLVER_HAS_METAL OFF)
set(BSOLVER_HAS_CUDA OFF)

set(_bsolver_backend_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/backend)
set(_bsolver_kernel_core ${_bsolver_backend_dir}/gpu/kernel_core.h)

if(BSOLVER_OPENCL)
  find_package(OpenCL)
  if(OpenCL_FOUND)
    set(BSOLVER_HAS_OPENCL ON)
    bsolver_embed(OUTPUT ${CMAKE_BINARY_DIR}/generated/opencl_kernels.cpp
      SYMBOL OpenCLKernelSource MODE text
      FILES ${_bsolver_kernel_core} ${_bsolver_backend_dir}/opencl/kernels.cl)
    list(APPEND BSOLVER_BACKEND_SOURCES
      ${_bsolver_backend_dir}/opencl/opencl_backend.cpp
      ${CMAKE_BINARY_DIR}/generated/opencl_kernels.cpp)
    list(APPEND BSOLVER_BACKEND_DEFINITIONS BSOLVER_HAS_OPENCL=1)
    list(APPEND BSOLVER_BACKEND_LIBRARIES OpenCL::OpenCL)
    message(STATUS "bsolver: OpenCL backend enabled")
  else()
    message(STATUS "bsolver: OpenCL not found; OpenCL backend disabled")
  endif()
endif()

if(BSOLVER_METAL AND APPLE)
  enable_language(OBJCXX)
  set(BSOLVER_HAS_METAL ON)
  bsolver_embed(OUTPUT ${CMAKE_BINARY_DIR}/generated/metal_kernels.cpp
    SYMBOL MetalKernelSource MODE text
    FILES ${_bsolver_kernel_core} ${_bsolver_backend_dir}/metal/kernels.metal)
  list(APPEND BSOLVER_BACKEND_SOURCES
    ${_bsolver_backend_dir}/metal/metal_backend.mm
    ${CMAKE_BINARY_DIR}/generated/metal_kernels.cpp)
  set_source_files_properties(${_bsolver_backend_dir}/metal/metal_backend.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)
  list(APPEND BSOLVER_BACKEND_DEFINITIONS BSOLVER_HAS_METAL=1)
  list(APPEND BSOLVER_BACKEND_LIBRARIES "-framework Metal" "-framework Foundation" "-framework IOKit")
  message(STATUS "bsolver: Metal backend enabled")
endif()

if(BSOLVER_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    # `native` fails at generation when the toolkit is installed but no NVIDIA GPU is visible (container/CI builds).
    if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native" AND NOT CMAKE_CUDA_ARCHITECTURES_NATIVE)
      set(CMAKE_CUDA_ARCHITECTURES all-major)
      message(STATUS "bsolver: no NVIDIA GPU detected for CUDA_ARCHITECTURES=native; building for all-major")
    endif()
    find_package(CUDAToolkit REQUIRED)
    set(BSOLVER_HAS_CUDA ON)
    list(APPEND BSOLVER_BACKEND_SOURCES
      ${_bsolver_backend_dir}/cuda/cuda_backend.cpp
      ${_bsolver_backend_dir}/cuda/cuda_device.cu)
    list(APPEND BSOLVER_BACKEND_DEFINITIONS BSOLVER_HAS_CUDA=1)
    list(APPEND BSOLVER_BACKEND_LIBRARIES CUDA::cudart)
    # The directory-wide warning flags are host-compiler flags; keep them away from nvcc.
    get_property(_bsolver_opts DIRECTORY PROPERTY COMPILE_OPTIONS)
    set(_bsolver_wrapped "")
    foreach(_opt IN LISTS _bsolver_opts)
      list(APPEND _bsolver_wrapped "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${_opt}>")
    endforeach()
    set_property(DIRECTORY PROPERTY COMPILE_OPTIONS ${_bsolver_wrapped})
    message(STATUS "bsolver: CUDA backend enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(STATUS "bsolver: no CUDA compiler found; CUDA backend disabled")
  endif()
endif()

function(bsolver_configure_backend_target target)
  if(BSOLVER_HAS_CUDA)
    set_target_properties(${target} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  endif()
endfunction()

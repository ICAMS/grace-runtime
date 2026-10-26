# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
# GPU module grace_runtime._kk_<arch> for ONE GPU architecture (Kokkos builds one architecture per install):
#   cmake -S python/kk -B build -DKokkos_ROOT=<Kokkos 5 CUDA or HIP install, static, PIC> -DGRACE_KK_ARCH=sm86
# The Kokkos code (kk_core.cpp, kk_core_3l.cpp) goes through the Kokkos launch compiler (CUDA: nvcc_wrapper; HIP with
# Kokkos compiled by hipcc / amdclang++ as CXX: that compiler), or, when Kokkos was built with
# Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON for HIP, through the CMake HIP language (amdclang++). The binding
# (kk_module.cpp, nanobind) and the CPU sources (model loaders) go through the host compiler,
# which also links the module: with g++ the only OpenMP runtime is libgomp, as in grace_runtime._core_*.
cmake_minimum_required(VERSION 3.21)
project(grace_runtime_kk CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(GRACE_KK_ARCH "sm86" CACHE STRING "GPU architecture tag of the module (_kk_<arch>)")
set(GRACE_KK_PACKAGE "grace_runtime" CACHE STRING "Python package the module installs into (wheels: grace_runtime_cuda12)")
set(GRACE_KK_CUBLAS "dlopen" CACHE STRING "CUDA: cuBLAS (UQ projection GEMMs): dlopen (loaded at run time when present, else hand-written kernels; no link dependency) or link. HIP: rocBLAS is always loaded at run time")

find_package(Python 3.10 COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
find_package(Kokkos REQUIRED)
if(Kokkos_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
elseif(NOT Kokkos_ENABLE_HIP)
  message(FATAL_ERROR "Kokkos at ${Kokkos_DIR} has neither the CUDA nor the HIP backend")
endif()
# Stream-ordered allocation (Kokkos' default for HIP before ROCm 7) keeps freed buffers in the driver's pool, unseen by
# cudaMemGetInfo / hipMemGetInfo, which chunksize="auto" and free_gpu_memory() rely on
if(Kokkos_ENABLE_IMPL_HIP_MALLOC_ASYNC OR Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC)
  message(FATAL_ERROR "Kokkos at ${Kokkos_DIR} was built with Kokkos_ENABLE_IMPL_HIP_MALLOC_ASYNC or "
                      "Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC: rebuild it with them OFF")
endif()
# Kokkos HIP compiled as the CMake HIP language: the engines' sources are HIP, everything else (and the link) CXX
set(_kk_hip_lang OFF)
if(Kokkos_ENABLE_HIP AND Kokkos_COMPILE_LANGUAGE STREQUAL "HIP")
  set(_kk_hip_lang ON)
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES ${Kokkos_HIP_ARCHITECTURES})
  endif()
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  enable_language(HIP)
  find_package(hip REQUIRED CONFIG)    # hip::amdhip64 for the CXX link
  # g++ links the module: the link flags of the imported targets are for a clang (HIP) link (Kokkos: -fno-gpu-rdc;
  # hip::device through rocThrust / rocPRIM: --hip-link, --offload-arch, as link "libraries") and g++ rejects them.
  # The HIP code is not relocatable (-fno-gpu-rdc: no device link step) and libamdhip64 is linked explicitly.
  get_directory_property(_imported IMPORTED_TARGETS)
  foreach(_t IN LISTS _imported)
    get_target_property(_lo ${_t} INTERFACE_LINK_OPTIONS)
    if(_lo)
      set_target_properties(${_t} PROPERTIES INTERFACE_LINK_OPTIONS "")
    endif()
    get_target_property(_ll ${_t} INTERFACE_LINK_LIBRARIES)
    if(_ll)
      list(FILTER _ll EXCLUDE REGEX "--hip-link|--offload-arch|-fgpu-rdc|-fno-gpu-rdc")
      set_target_properties(${_t} PROPERTIES INTERFACE_LINK_LIBRARIES "${_ll}")
    endif()
  endforeach()
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/GraceRuntime.cmake")

# CPU part: model loaders (libgrace)
set(_lg "${GRACE_RUNTIME_ROOT}/libgrace")
find_package(OpenMP REQUIRED COMPONENTS CXX)
add_library(grace_kk_cpu STATIC ${_lg}/src/model_1l.cpp ${_lg}/src/model_2l.cpp ${_lg}/src/model_3l.cpp
            ${_lg}/src/uq.cpp ${_lg}/src/uq_head.cpp
            ${_lg}/third_party/grace_cnpy/cnpy.cpp)
target_include_directories(grace_kk_cpu PUBLIC ${GRACE_RUNTIME_INCLUDE_DIRS} PRIVATE ${GRACE_RUNTIME_PRIVATE_INCLUDE_DIRS})
target_compile_options(grace_kk_cpu PRIVATE -O3)
target_link_libraries(grace_kk_cpu PRIVATE OpenMP::OpenMP_CXX)    # link-only for its users
set_target_properties(grace_kk_cpu PROPERTIES CXX_VISIBILITY_PRESET hidden)

# GPU part: the fused engines on CSR images (the GRACE-3L engine instantiated in its own translation unit)
add_library(grace_kk_core STATIC kk_core.cpp kk_core_3l.cpp)
target_include_directories(grace_kk_core PRIVATE ${GRACE_RUNTIME_PRIVATE_INCLUDE_DIRS})
target_compile_definitions(grace_kk_core PRIVATE GRACE_KK_ARCH="${GRACE_KK_ARCH}")
target_link_libraries(grace_kk_core PUBLIC grace_kk_cpu PRIVATE Kokkos::kokkos)
if(Kokkos_ENABLE_CUDA AND GRACE_KK_CUBLAS STREQUAL "link")
  target_link_libraries(grace_kk_core PRIVATE CUDA::cublas)
else()    # CUDA: cuBLAS dlopen'ed; HIP: rocBLAS dlopen'ed (grace/kk/rocblas_guard.h)
  if(Kokkos_ENABLE_CUDA)
    target_compile_definitions(grace_kk_core PRIVATE GRACE_KK_CUBLAS_DLOPEN)
  endif()
  target_link_libraries(grace_kk_core PRIVATE ${CMAKE_DL_LIBS})
endif()
if(_kk_hip_lang)
  set_source_files_properties(kk_core.cpp kk_core_3l.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(grace_kk_core PUBLIC hip::amdhip64)
  set_target_properties(grace_kk_core PROPERTIES HIP_VISIBILITY_PRESET hidden)
else()
  kokkos_compilation(TARGET grace_kk_core)
endif()
# AMD GPUs (Kokkos HIP): hardware FP atomics (GRACE_HIP_UNSAFE_FP_ATOMICS); a no-op for CUDA
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/GraceHipAtomics.cmake")
grace_hip_unsafe_fp_atomics(grace_kk_core)
set_target_properties(grace_kk_core PROPERTIES CXX_VISIBILITY_PRESET hidden)

set(t _kk_${GRACE_KK_ARCH})
nanobind_add_module(${t} NB_STATIC NOMINSIZE STABLE_ABI kk_module.cpp)
target_compile_definitions(${t} PRIVATE GRACE_KK_MODULE=${t})
target_link_libraries(${t} PRIVATE grace_kk_core)
target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)    # dispersion_args.h
if(Kokkos_ENABLE_HIP)
  target_compile_definitions(${t} PRIVATE GRACE_KK_PLATFORM_HIP)    # DLPack device type kDLROCM
endif()
if(_kk_hip_lang)
  set_target_properties(${t} PROPERTIES LINKER_LANGUAGE CXX)    # g++ links: libgomp, static libstdc++
endif()
# wheels: static libstdc++, only PyInit_* exported (Kokkos, libstdc++ and nanobind symbols stay local)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(${t} PRIVATE -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL
                      -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../src/exports.map)
endif()
install(TARGETS ${t} LIBRARY DESTINATION ${GRACE_KK_PACKAGE})

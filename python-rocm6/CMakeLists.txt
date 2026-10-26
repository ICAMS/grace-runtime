# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
# grace-runtime-rocm6: one sub-build of python/kk per AMD GPU architecture (ExternalProject: each with its own static
# Kokkos, so the Kokkos installs never meet in one CMake project); the modules go into grace_runtime_rocm6.
#   GRACE_KK_ARCHS          architectures (CMake or environment, "," or ";"), default gfx90a;gfx942 (MI210 / MI250 /
#                           MI250X; MI300A / MI300X / MI325X). AMD code runs only on its own architecture: an RX 7900
#                           needs gfx1100, an RX 9070 gfx1201
#   GRACE_KOKKOS_ROOT_<ARCH> (environment) an installed Kokkos per architecture (HIP compiled as the CMake HIP language,
#                           static, PIC, C++20), e.g. GRACE_KOKKOS_ROOT_GFX90A; else the pinned Kokkos release is built
#                           (GRACE_KOKKOS_SOURCE: a local Kokkos source tree for offline builds)
# ROCm 6 from ROCM_PATH (default /opt/rocm): its clang++ compiles the Kokkos code as HIP (HIPCXX overrides it), the C++
# compiler (CXX, default g++) everything else and links, so the modules have one OpenMP runtime (libgomp) and need
# only libamdhip64.so.6 of ROCm at run time. rocBLAS is loaded at run time when present, never linked.
cmake_minimum_required(VERSION 3.24)
project(grace_runtime_rocm6 NONE)
include(ExternalProject)

set(GRACE_KK_ARCHS "gfx90a;gfx942" CACHE STRING "GPU architectures, one module each")
if(DEFINED ENV{GRACE_KK_ARCHS})    # environment (pip / build): "gfx90a,gfx1100" or "gfx90a;gfx1100"
  string(REPLACE "," ";" GRACE_KK_ARCHS "$ENV{GRACE_KK_ARCHS}")
endif()
set(GRACE_KOKKOS_VERSION "5.2.2")
set(GRACE_KOKKOS_SHA256 "d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f")
set(GRACE_KOKKOS_URL "https://github.com/kokkos/kokkos/releases/download/${GRACE_KOKKOS_VERSION}/kokkos-${GRACE_KOKKOS_VERSION}.tar.gz")
set(GRACE_KOKKOS_SOURCE "" CACHE PATH "Kokkos source tree (offline builds); empty: the pinned release")

# sources: staged next to this file (sdist), else the repository around python-rocm6/
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/kk/CMakeLists.txt")
  set(_root "${CMAKE_CURRENT_SOURCE_DIR}")
else()
  get_filename_component(_root "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
endif()
find_package(Python 3.10 COMPONENTS Interpreter REQUIRED)
set(_cxx "$ENV{CXX}")
if(NOT _cxx)
  find_program(_cxx NAMES g++ c++ REQUIRED)
endif()
set(_rocm "$ENV{ROCM_PATH}")
if(NOT _rocm)
  set(_rocm "/opt/rocm")
endif()
set(_hipcxx "$ENV{HIPCXX}")
if(NOT _hipcxx)
  set(_hipcxx "${_rocm}/llvm/bin/clang++")
endif()
if(NOT EXISTS "${_hipcxx}")
  message(FATAL_ERROR "no ROCm clang++ at ${_hipcxx}: set ROCM_PATH to a ROCm 6 installation (or HIPCXX)")
endif()
# grace-runtime-rocm6: the modules need libamdhip64.so.6 (ROCM_MAJOR = 6 of the plug-in)
if(EXISTS "${_rocm}/.info/version")
  file(STRINGS "${_rocm}/.info/version" _rver LIMIT_COUNT 1)
  if(NOT _rver MATCHES "^6\\.")
    message(FATAL_ERROR "ROCm ${_rver} at ${_rocm}: grace-runtime-rocm6 needs ROCm 6 (set ROCM_PATH)")
  endif()
endif()
# clang++ uses the libstdc++ of the C++ compiler: the HIP and the C++ objects of a module share one standard library
execute_process(COMMAND "${_cxx}" -print-libgcc-file-name OUTPUT_VARIABLE _libgcc OUTPUT_STRIP_TRAILING_WHITESPACE)
get_filename_component(_gccdir "${_libgcc}" DIRECTORY)
set(_hipflags "$ENV{HIPFLAGS} --gcc-install-dir=${_gccdir}")
set(_known gfx906 gfx908 gfx90a gfx940 gfx942 gfx1030 gfx1100 gfx1103 gfx1201)    # the Kokkos_ARCH_AMD_* of Kokkos 5.2
set(_stage "${CMAKE_CURRENT_BINARY_DIR}/stage")
set(_prev "")       # architectures one after another (memory of the build host)

foreach(arch IN LISTS GRACE_KK_ARCHS)
  if(NOT arch IN_LIST _known)
    list(JOIN _known ", " _k)
    message(FATAL_ERROR "GRACE_KK_ARCHS: unknown architecture ${arch} (${_k})")
  endif()
  string(TOUPPER "${arch}" _ARCH)
  string(REPLACE "GFX" "AMD_GFX" _karch "${_ARCH}")
  set(_kroot "$ENV{GRACE_KOKKOS_ROOT_${_ARCH}}")
  set(_deps ${_prev})
  if(NOT _kroot)
    set(_kroot "${CMAKE_CURRENT_BINARY_DIR}/kokkos-${arch}")
    if(GRACE_KOKKOS_SOURCE)
      set(_src SOURCE_DIR "${GRACE_KOKKOS_SOURCE}")
    else()
      set(_src URL "${GRACE_KOKKOS_URL}" URL_HASH "SHA256=${GRACE_KOKKOS_SHA256}")
    endif()
    # MallocAsync off: with it, freed buffers stay in the stream-ordered pool, unseen by hipMemGetInfo
    # (chunksize="auto", free_gpu_memory())
    ExternalProject_Add(kokkos_${arch} ${_src}
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${_kroot} -DCMAKE_CXX_COMPILER=${_cxx}
                 -DCMAKE_HIP_COMPILER=${_hipcxx} "-DCMAKE_HIP_FLAGS=${_hipflags}" -DCMAKE_HIP_ARCHITECTURES=${arch}
                 -DCMAKE_PREFIX_PATH=${_rocm} -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON
                 -DKokkos_ENABLE_HIP=ON -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ARCH_${_karch}=ON
                 -DKokkos_ENABLE_IMPL_HIP_MALLOC_ASYNC=OFF -DKokkos_ENABLE_TESTS=OFF -DCMAKE_CXX_STANDARD=20
                 -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
    list(APPEND _deps kokkos_${arch})
  endif()
  ExternalProject_Add(kk_${arch}
    SOURCE_DIR "${_root}/python/kk"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${_stage} -DCMAKE_CXX_COMPILER=${_cxx}
               -DCMAKE_HIP_COMPILER=${_hipcxx} "-DCMAKE_HIP_FLAGS=${_hipflags}" -DCMAKE_PREFIX_PATH=${_rocm}
               -DKokkos_ROOT=${_kroot} -DGRACE_KK_ARCH=${arch} -DGRACE_KK_PACKAGE=grace_runtime_rocm6
               -DPython_EXECUTABLE=${Python_EXECUTABLE} "-DSKBUILD_SABI_COMPONENT=${SKBUILD_SABI_COMPONENT}"
    BUILD_ALWAYS ON
    DEPENDS ${_deps})
  set(_prev kk_${arch})
endforeach()
install(DIRECTORY "${_stage}/grace_runtime_rocm6/" DESTINATION grace_runtime_rocm6)

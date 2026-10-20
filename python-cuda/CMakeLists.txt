# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
# grace-runtime-cuda12: one sub-build of python/kk per GPU architecture (ExternalProject: each with its own static
# Kokkos, so the Kokkos installs never meet in one CMake project); the modules go into grace_runtime_cuda12.
#   GRACE_KK_ARCHS          architectures (CMake or environment, "," or ";"), default sm70;sm80;sm90;sm100;sm120
#                           (sm75 devices use sm70, 3-6 % slower than an sm75 build on an RTX 2070; sm86 / sm87 /
#                           sm89 devices use sm80, as fast or faster than their own builds on RTX 3090 / 4070 SUPER;
#                           sm103 (B300 / GB300) uses sm100, sm121 uses sm120)
#   GRACE_KOKKOS_ROOT_<ARCH> (environment) an installed Kokkos per architecture (CUDA, static, PIC, C++20), e.g.
#                           GRACE_KOKKOS_ROOT_SM80; else the pinned Kokkos release is built (GRACE_KOKKOS_SOURCE:
#                           a local Kokkos source tree for offline builds)
#   GRACE_KK_CUBLAS         dlopen (default: cuBLAS loaded at run time when installed) or link
# The C++ compiler (CXX) is the host compiler of nvcc; nvcc from PATH or CUDACXX.
cmake_minimum_required(VERSION 3.24)
project(grace_runtime_cuda12 NONE)
include(ExternalProject)

set(GRACE_KK_ARCHS "sm70;sm80;sm90;sm100;sm120" CACHE STRING "GPU architectures, one module each")
if(DEFINED ENV{GRACE_KK_ARCHS})    # environment (pip / build): "sm80,sm89" or "sm80;sm89"
  string(REPLACE "," ";" GRACE_KK_ARCHS "$ENV{GRACE_KK_ARCHS}")
endif()
set(GRACE_KK_CUBLAS "dlopen" CACHE STRING "cuBLAS: dlopen or link")
set(GRACE_KOKKOS_VERSION "5.2.2")
set(GRACE_KOKKOS_SHA256 "d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f")
set(GRACE_KOKKOS_URL "https://github.com/kokkos/kokkos/releases/download/${GRACE_KOKKOS_VERSION}/kokkos-${GRACE_KOKKOS_VERSION}.tar.gz")
set(GRACE_KOKKOS_SOURCE "" CACHE PATH "Kokkos source tree (offline builds); empty: the pinned release")

# sources: staged next to this file (sdist), else the repository around python-cuda/
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
set(_nvcc "$ENV{CUDACXX}")
if(NOT _nvcc)
  find_program(_nvcc nvcc REQUIRED)
endif()
set(_kokkos_arch_sm70 VOLTA70)
set(_kokkos_arch_sm75 TURING75)
set(_kokkos_arch_sm80 AMPERE80)
set(_kokkos_arch_sm86 AMPERE86)
set(_kokkos_arch_sm89 ADA89)
set(_kokkos_arch_sm90 HOPPER90)
set(_kokkos_arch_sm100 BLACKWELL100)
set(_kokkos_arch_sm120 BLACKWELL120)
set(_stage "${CMAKE_CURRENT_BINARY_DIR}/stage")
set(_prev "")       # architectures one after another: four parallel nvcc builds of the engines exhaust a runner's memory

foreach(arch IN LISTS GRACE_KK_ARCHS)
  if(NOT DEFINED _kokkos_arch_${arch})
    message(FATAL_ERROR "GRACE_KK_ARCHS: unknown architecture ${arch} (sm70, sm75, sm80, sm86, sm89, sm90, sm100, sm120)")
  endif()
  string(TOUPPER "${arch}" _ARCH)
  set(_kroot "$ENV{GRACE_KOKKOS_ROOT_${_ARCH}}")
  set(_deps ${_prev})
  if(NOT _kroot)
    set(_kroot "${CMAKE_CURRENT_BINARY_DIR}/kokkos-${arch}")
    if(GRACE_KOKKOS_SOURCE)
      set(_src SOURCE_DIR "${GRACE_KOKKOS_SOURCE}")
    else()
      set(_src URL "${GRACE_KOKKOS_URL}" URL_HASH "SHA256=${GRACE_KOKKOS_SHA256}")
    endif()
    ExternalProject_Add(kokkos_${arch} ${_src}
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${_kroot} -DCMAKE_CXX_COMPILER=${_cxx}
                 -DCMAKE_CUDA_COMPILER=${_nvcc} -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON
                 -DKokkos_ARCH_${_kokkos_arch_${arch}}=ON -DCMAKE_CXX_STANDARD=20 -DBUILD_SHARED_LIBS=OFF
                 -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
    list(APPEND _deps kokkos_${arch})
  endif()
  ExternalProject_Add(kk_${arch}
    SOURCE_DIR "${_root}/python/kk"
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${_stage} -DCMAKE_CXX_COMPILER=${_cxx}
               -DKokkos_ROOT=${_kroot} -DGRACE_KK_ARCH=${arch} -DGRACE_KK_PACKAGE=grace_runtime_cuda12
               -DGRACE_KK_CUBLAS=${GRACE_KK_CUBLAS} -DPython_EXECUTABLE=${Python_EXECUTABLE}
               "-DSKBUILD_SABI_COMPONENT=${SKBUILD_SABI_COMPONENT}"
    BUILD_ALWAYS ON
    DEPENDS ${_deps})
  set(_prev kk_${arch})
endforeach()
install(DIRECTORY "${_stage}/grace_runtime_cuda12/" DESTINATION grace_runtime_cuda12)

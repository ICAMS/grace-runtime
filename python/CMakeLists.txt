# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
cmake_minimum_required(VERSION 3.16)
project(grace_runtime_py CXX)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)    # default; a -DCMAKE_CXX_STANDARD given to the build is kept
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# SKBUILD_SABI_COMPONENT (scikit-build-core, wheel.py-api = cp312): Development.SABIModule, so STABLE_ABI builds abi3
find_package(Python 3.10 COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
# Apple clang has no OpenMP runtime: Homebrew's keg-only libomp, or that of the conda environment of the building Python
# (llvm-openmp, shared with conda's NumPy / PyTorch in that environment), unless OpenMP_ROOT names another one
if(APPLE AND NOT OpenMP_ROOT AND NOT DEFINED ENV{OpenMP_ROOT})
  set(_omp_roots /opt/homebrew/opt/libomp /usr/local/opt/libomp)
  string(FIND "${Python_EXECUTABLE}" "$ENV{CONDA_PREFIX}/" _in_conda)
  if(DEFINED ENV{CONDA_PREFIX} AND _in_conda EQUAL 0)
    list(PREPEND _omp_roots "$ENV{CONDA_PREFIX}")
  endif()
  foreach(p IN LISTS _omp_roots)
    if(EXISTS "${p}/include/omp.h")
      set(OpenMP_ROOT "${p}")
      break()
    endif()
  endforeach()
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GraceRuntime.cmake")
# one module per ISA level, chosen at import by grace_runtime/_dispatch.py; -DGRACE_RUNTIME_ISAS=native builds a single
# -march=native module instead (from source: SKBUILD_CMAKE_DEFINE="GRACE_RUNTIME_ISAS=native" pip install ...). macOS
# (no /proc/cpuinfo for the dispatch) and other CPUs than x86-64 build only the native module from source; the ARM
# wheels the portable aarch64 one (scripts/build_wheels.sh).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$" AND NOT APPLE)
  set(_isas "x86_64_v3;x86_64_v4")
else()
  set(_isas native)
endif()
set(GRACE_RUNTIME_ISAS "${_isas}" CACHE STRING
    "ISA builds of the extension: x86_64_v3, x86_64_v4, aarch64 (the ARM wheels), native")
set(GRACE_RUNTIME_FLAGS_x86_64_v3 -march=x86-64-v3)
set(GRACE_RUNTIME_FLAGS_x86_64_v4 -march=x86-64-v4 -mprefer-vector-width=512)
set(GRACE_RUNTIME_FLAGS_native -march=native)
# macOS: INSTALL_RPATH = the directory of libomp. GRACE_RUNTIME_BUNDLE_OPENMP (the wheels): a copy of an
# @rpath/libomp.dylib (conda-forge's) in grace_runtime/.dylibs and that run path, so that an @rpath/libomp.dylib already
# in the process (conda's NumPy / SciPy) is used instead: one OpenMP runtime, no "OMP: Error #15"
option(GRACE_RUNTIME_BUNDLE_OPENMP "macOS: ship the @rpath libomp in the package (the wheels)" OFF)
if(APPLE)
  set(GRACE_RUNTIME_FLAGS_aarch64 -mcpu=apple-m1)
  set(_omp_lib "")
  set(_omp_rpath "")
  if(OpenMP_CXX_LIBRARIES)    # empty when the compiler brings its OpenMP runtime itself
    list(GET OpenMP_CXX_LIBRARIES 0 _omp_lib)
    get_filename_component(_omp_rpath "${_omp_lib}" DIRECTORY)
  endif()
  if(GRACE_RUNTIME_BUNDLE_OPENMP)
    if(NOT _omp_lib)
      message(FATAL_ERROR "GRACE_RUNTIME_BUNDLE_OPENMP: no OpenMP library file found to bundle")
    endif()
    execute_process(COMMAND otool -D "${_omp_lib}" OUTPUT_VARIABLE _omp_id)
    if(NOT _omp_id MATCHES "@rpath/libomp.dylib")
      message(FATAL_ERROR "GRACE_RUNTIME_BUNDLE_OPENMP: ${_omp_lib} is not @rpath/libomp.dylib "
                          "(conda-forge llvm-openmp)")
    endif()
    set(_omp_rpath "@loader_path/.dylibs")
    get_filename_component(_omp_real "${_omp_lib}" REALPATH)
    install(FILES "${_omp_real}" DESTINATION grace_runtime/.dylibs RENAME libomp.dylib)
    install(FILES LICENSE-libomp DESTINATION grace_runtime/.dylibs)
  endif()
else()
  set(GRACE_RUNTIME_FLAGS_aarch64 -march=armv8.2-a)
endif()
foreach(isa IN LISTS GRACE_RUNTIME_ISAS)
  set(t _core_${isa})
  nanobind_add_module(${t} NB_STATIC NOMINSIZE STABLE_ABI src/core.cpp src/neighbors.cpp ${GRACE_RUNTIME_SOURCES})
  target_compile_definitions(${t} PRIVATE GRACE_RUNTIME_MODULE=${t})
  target_include_directories(${t} PRIVATE ${GRACE_RUNTIME_INCLUDE_DIRS} ${GRACE_RUNTIME_PRIVATE_INCLUDE_DIRS}
                                           ${GRACE_RUNTIME_ROOT}/libgrace/src)    # pace_convert.h
  target_link_libraries(${t} PRIVATE OpenMP::OpenMP_CXX)
  # wheels: no dependency on the build compiler's libstdc++ (newer than the target systems'); libgomp stays shared
  # (auditwheel vendors it under a mangled name). --exclude-libs: the static libstdc++ symbols stay local, else
  # another libstdc++ in the process interposes them; the version script exports only PyInit_*
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(${t} PRIVATE -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL
                        -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map)
  endif()
  target_compile_options(${t} PRIVATE ${GRACE_RUNTIME_FLAGS_${isa}} -O3)
  if(APPLE)
    # a * b + c fused also across the SIMD operator overloads, as GCC (the Linux wheels) does by default
    target_compile_options(${t} PRIVATE -ffp-contract=fast)
    # the GRACE-3L GEMMs through Accelerate's cblas_sgemm (GRACE_GEMM=builtin at run time: the built-in kernel)
    target_compile_definitions(${t} PRIVATE GRACE_BLAS)
    target_link_libraries(${t} PRIVATE "-framework Accelerate")
    # cpu_gemm.h probes for MKL / OpenBLAS through weak symbols, which Apple's linker accepts undefined only so
    target_link_options(${t} PRIVATE -Wl,-U,_MKL_Get_Version_String -Wl,-U,_openblas_get_parallel)
    if(_omp_rpath)
      set_target_properties(${t} PROPERTIES INSTALL_RPATH "${_omp_rpath}")
    endif()
  endif()
  install(TARGETS ${t} LIBRARY DESTINATION grace_runtime)
endforeach()

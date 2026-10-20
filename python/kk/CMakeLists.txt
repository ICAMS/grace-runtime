# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
# GPU module grace_runtime._kk_<arch> for ONE GPU architecture (Kokkos builds one architecture per install):
#   cmake -S python/kk -B build -DKokkos_ROOT=<Kokkos 5 CUDA install, static, PIC> -DGRACE_KK_ARCH=sm86
# The Kokkos code (kk_core.cpp) goes through the Kokkos launch compiler (nvcc_wrapper); the binding (kk_module.cpp,
# nanobind) and the CPU sources (model loaders, neighbour list) through the host compiler.
cmake_minimum_required(VERSION 3.18)
project(grace_runtime_kk CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(GRACE_KK_ARCH "sm86" CACHE STRING "GPU architecture tag of the module (_kk_<arch>)")
set(GRACE_KK_PACKAGE "grace_runtime" CACHE STRING "Python package the module installs into (wheels: grace_runtime_cuda12)")
set(GRACE_KK_CUBLAS "dlopen" CACHE STRING "cuBLAS (UQ projection GEMMs): dlopen (loaded at run time when present, else hand-written kernels; no link dependency) or link")

find_package(Python 3.10 COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
find_package(Kokkos REQUIRED)
find_package(CUDAToolkit REQUIRED)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/GraceRuntime.cmake")

# CPU part: model loaders (libgrace), the cell-list neighbour search of the CPU module and the batch CSR (OpenMP)
set(_lg "${GRACE_RUNTIME_ROOT}/libgrace")
find_package(OpenMP REQUIRED COMPONENTS CXX)
add_library(grace_kk_cpu STATIC ${_lg}/src/model_1l.cpp ${_lg}/src/model_2l.cpp ${_lg}/src/model_3l.cpp
            ${_lg}/src/uq.cpp ${_lg}/src/uq_head.cpp
            ${_lg}/third_party/grace_cnpy/cnpy.cpp ../src/neighbors.cpp batch_csr.cpp)
target_include_directories(grace_kk_cpu PUBLIC ${GRACE_RUNTIME_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../src
                           PRIVATE ${GRACE_RUNTIME_PRIVATE_INCLUDE_DIRS})
target_compile_options(grace_kk_cpu PRIVATE -O3)
target_link_libraries(grace_kk_cpu PRIVATE OpenMP::OpenMP_CXX)    # link-only for its users
set_target_properties(grace_kk_cpu PROPERTIES CXX_VISIBILITY_PRESET hidden)

# GPU part: the fused engines on CSR images (the GRACE-3L engine instantiated in its own translation unit)
add_library(grace_kk_core STATIC kk_core.cpp kk_core_3l.cpp)
target_include_directories(grace_kk_core PRIVATE ${GRACE_RUNTIME_PRIVATE_INCLUDE_DIRS})
target_compile_definitions(grace_kk_core PRIVATE GRACE_KK_ARCH="${GRACE_KK_ARCH}")
target_link_libraries(grace_kk_core PUBLIC grace_kk_cpu PRIVATE Kokkos::kokkos)
if(GRACE_KK_CUBLAS STREQUAL "link")
  target_link_libraries(grace_kk_core PRIVATE CUDA::cublas)
else()
  target_compile_definitions(grace_kk_core PRIVATE GRACE_KK_CUBLAS_DLOPEN)
  target_link_libraries(grace_kk_core PRIVATE ${CMAKE_DL_LIBS})
endif()
kokkos_compilation(TARGET grace_kk_core)
set_target_properties(grace_kk_core PROPERTIES CXX_VISIBILITY_PRESET hidden)

set(t _kk_${GRACE_KK_ARCH})
nanobind_add_module(${t} NB_STATIC NOMINSIZE STABLE_ABI kk_module.cpp)
target_compile_definitions(${t} PRIVATE GRACE_KK_MODULE=${t})
target_link_libraries(${t} PRIVATE grace_kk_core)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(${t} PRIVATE -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL
                      -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../src/exports.map)
endif()
install(TARGETS ${t} LIBRARY DESTINATION ${GRACE_KK_PACKAGE})

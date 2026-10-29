cmake_minimum_required(VERSION 3.16)
project(grace_runtime CXX)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)    # default; LAMMPS / Kokkos builds use 20 (-DCMAKE_CXX_STANDARD=20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# default -march=native (empty when cross-compiling or unsupported): the tests run where they are built; AVX-512 timing
# builds: docs/development.md
if(NOT DEFINED GRACE_ARCH_FLAGS)
  set(_grace_arch "")
  if(NOT CMAKE_CROSSCOMPILING)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native GRACE_HAS_MARCH_NATIVE)
    if(GRACE_HAS_MARCH_NATIVE)
      set(_grace_arch -march=native)
    endif()
  endif()
  set(GRACE_ARCH_FLAGS "${_grace_arch}" CACHE STRING "ISA flags of libgrace and the tests")
endif()
message(STATUS "GRACE_ARCH_FLAGS: '${GRACE_ARCH_FLAGS}'")
set(GRACE_RUNTIME_STATIC_LIBRARY ON)
include(cmake/GraceRuntime.cmake)
target_compile_options(libgrace PRIVATE ${GRACE_ARCH_FLAGS})
option(GRACE_RUNTIME_TESTS "build tests" ON)
if(GRACE_RUNTIME_TESTS)
  enable_testing()
  foreach(t test_error test_model_load test_kernels_compile test_bonds test_driver test_model_limits test_uq test_driver_uq test_gemm test_coupling test_d3 test_model_shapes test_pace_yaml test_model_kind test_host_memory)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE grace::libgrace)
    target_include_directories(${t} PRIVATE tests libgrace/third_party)
    target_compile_options(${t} PRIVATE ${GRACE_ARCH_FLAGS})
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} SKIP_RETURN_CODE 77)
  endforeach()
  target_include_directories(test_model_shapes PRIVATE libgrace/src)    # npz3.h: the loaders' array access
  target_include_directories(test_pace_yaml PRIVATE libgrace/src)    # pace_yaml.h, pace_convert.h
endif()

# Kokkos engines (libgrace/include/grace/kk): standalone tests against an installed Kokkos (Kokkos_ROOT)
option(GRACE_WITH_KOKKOS "build the Kokkos engine tests" OFF)
if(GRACE_WITH_KOKKOS)
  find_package(Kokkos REQUIRED)
  include(cmake/GraceHipAtomics.cmake)
  set(GRACE_KK_TESTS test_kk_1l_compile test_kk_2l_compile test_kk_3l_compile test_kk_csr test_kk_csr_3l
                     test_kk_2l_large test_kk_neighbors test_kk_coupling test_kk_large test_kk_gemm test_kk_device
                     test_kk_launch_check test_kk_d3)
  foreach(t ${GRACE_KK_TESTS})
    add_executable(${t} tests/kk/${t}.cpp)
    target_link_libraries(${t} PRIVATE grace::libgrace Kokkos::kokkos)
    if(Kokkos_ENABLE_CUDA)
      find_package(CUDAToolkit REQUIRED)
      target_link_libraries(${t} PRIVATE CUDA::cublas)
    endif()
    target_include_directories(${t} PRIVATE tests libgrace/third_party)
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} SKIP_RETURN_CODE 77)
  endforeach()
  # a missing test model is a skip (exit 77) also with Kokkos initialised on a GPU, not an abort (test_exit)
  add_test(NAME test_kk_skip_status COMMAND sh -c
           "unset GRACE_TEST_M1; GRACE_CACHE=/nonexistent $<TARGET_FILE:test_kk_1l_compile>; test $? -eq 77")
  set_tests_properties(test_kk_skip_status PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  grace_hip_unsafe_fp_atomics(${GRACE_KK_TESTS})    # AMD GPUs: hardware FP atomics (GRACE_HIP_UNSAFE_FP_ATOMICS)
  # HIP: the GRACE-3L tests (SPBF) and the GRACE-1L / 2L tests (bond kernels) once more with the other kernels than the
  # build's default: gfx90a (matrix-core kernels by default) with GRACE3L_SPBF_MFMA=0 / GRACE_FUSED_MFMA=0, the other
  # targets with =1 (gfx94x: the opt-in matrix-core kernels; RDNA: ignored)
  if(Kokkos_ENABLE_HIP)
    if("AMD_GFX90A" IN_LIST Kokkos_ARCH)
      set(_sfx spbf_mfma_off)
      set(_env GRACE3L_SPBF_MFMA=0)
      set(_fsfx fused_mfma_off)
      set(_fenv GRACE_FUSED_MFMA=0)
    else()
      set(_sfx spbf_mfma)
      set(_env GRACE3L_SPBF_MFMA=1)
      set(_fsfx fused_mfma)
      set(_fenv GRACE_FUSED_MFMA=1)
    endif()
    # and with the engines' own GEMMs instead of rocBLAS (GRACE_RUNTIME_NO_ROCBLAS=1: the BLAS paths, fc_fold among
    # them); CDNA (64-lane wavefronts) also with the 32-lane layer-1 force kernel (GRACE3L_FORCE_L1_WAVE=0)
    set(_wave64 OFF)
    foreach(a AMD_GFX906 AMD_GFX908 AMD_GFX90A AMD_GFX940 AMD_GFX942 AMD_GFX942_APU AMD_GFX950)
      if(a IN_LIST Kokkos_ARCH)
        set(_wave64 ON)
      endif()
    endforeach()
    foreach(t test_kk_3l_compile test_kk_csr_3l)
      add_test(NAME ${t}_${_sfx} COMMAND ${t})
      set_tests_properties(${t}_${_sfx} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} SKIP_RETURN_CODE 77
                           ENVIRONMENT ${_env})
      if(_wave64)
        add_test(NAME ${t}_force_l1_wave_off COMMAND ${t})
        set_tests_properties(${t}_force_l1_wave_off PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                             SKIP_RETURN_CODE 77 ENVIRONMENT GRACE3L_FORCE_L1_WAVE=0)
      endif()
      add_test(NAME ${t}_no_rocblas COMMAND ${t})
      set_tests_properties(${t}_no_rocblas PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                           SKIP_RETURN_CODE 77 ENVIRONMENT GRACE_RUNTIME_NO_ROCBLAS=1)
    endforeach()
    # and the GRACE-1L / 2L tests with the team-of-256 layer-1 bond backward (GRACE_FUSED_WAVE_BWD=0), every HIP target
    foreach(t test_kk_1l_compile test_kk_2l_compile test_kk_2l_large test_kk_coupling test_kk_large)
      add_test(NAME ${t}_${_fsfx} COMMAND ${t})
      set_tests_properties(${t}_${_fsfx} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} SKIP_RETURN_CODE 77
                           ENVIRONMENT ${_fenv})
      add_test(NAME ${t}_wave_bwd_off COMMAND ${t})
      set_tests_properties(${t}_wave_bwd_off PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} SKIP_RETURN_CODE 77
                           ENVIRONMENT GRACE_FUSED_WAVE_BWD=0)
    endforeach()
  endif()
  # CUDA: the team-of-256 layer-1 bond backward of GRACE-1L / -2L (GRACE_FUSED_WAVE_BWD=0: also for 2L, whose CUDA
  # default is the wave-per-atom kernel) with its launch bounds where they fit, and in the fallback without them
  # (GRACE_LB_FALLBACK=1, as on RTX 20xx / T4; the test passes only when the fallback says it ran); the GRACE-3L SPBF
  # backward in the fallback (its bounded kernel runs only on GPUs with >= 160 KB opt-in shared memory per block, e.g.
  # A100 / H100; elsewhere this repeats test_kk_csr_3l)
  if(Kokkos_ENABLE_CUDA)
    add_test(NAME test_kk_csr_wave_bwd_off COMMAND test_kk_csr)
    set_tests_properties(test_kk_csr_wave_bwd_off PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} SKIP_RETURN_CODE 77
                         ENVIRONMENT GRACE_FUSED_WAVE_BWD=0)
    add_test(NAME test_kk_csr_lb_fallback COMMAND test_kk_csr)
    set_tests_properties(test_kk_csr_lb_fallback PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} SKIP_RETURN_CODE 77
                         ENVIRONMENT "GRACE_LB_FALLBACK=1;GRACE_FUSED_WAVE_BWD=0"
                         PASS_REGULAR_EXPRESSION "g1lf:bond_bwd without launch bounds.*g2lf:L1_bwd without launch bounds.*kk csr ok")
    add_test(NAME test_kk_csr_3l_lb_fallback COMMAND test_kk_csr_3l)
    set_tests_properties(test_kk_csr_3l_lb_fallback PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                         SKIP_RETURN_CODE 77 ENVIRONMENT GRACE_LB_FALLBACK=1)
  endif()
  # the CPU neighbour list of the Python module is the oracle of the device list
  target_sources(test_kk_neighbors PRIVATE python/src/neighbors.cpp)
  target_include_directories(test_kk_neighbors PRIVATE python/src)
endif()

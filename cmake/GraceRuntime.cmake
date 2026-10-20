# libgrace: included by the top-level CMakeLists and by LAMMPS' ML-PACE.cmake.
#   GRACE_RUNTIME_SOURCES / GRACE_RUNTIME_INCLUDE_DIRS / GRACE_RUNTIME_PRIVATE_INCLUDE_DIRS: for consumers that
#   compile the sources into their own target (LAMMPS: same flags as the rest of the code, incl. target-specific
#   -march/-mtune from Kokkos), which is the default for consumers.
#   GRACE_RUNTIME_STATIC_LIBRARY=ON (the top level sets it): also define the static library target libgrace.
get_filename_component(GRACE_RUNTIME_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
find_package(OpenMP REQUIRED COMPONENTS CXX)
file(GLOB GRACE_RUNTIME_SOURCES "${GRACE_RUNTIME_ROOT}/libgrace/src/*.cpp")
list(APPEND GRACE_RUNTIME_SOURCES "${GRACE_RUNTIME_ROOT}/libgrace/third_party/grace_cnpy/cnpy.cpp")
set(GRACE_RUNTIME_INCLUDE_DIRS "${GRACE_RUNTIME_ROOT}/libgrace/include")
set(GRACE_RUNTIME_PRIVATE_INCLUDE_DIRS "${GRACE_RUNTIME_ROOT}/libgrace/third_party")
if(GRACE_RUNTIME_STATIC_LIBRARY)
  add_library(libgrace STATIC ${GRACE_RUNTIME_SOURCES})
  add_library(grace::libgrace ALIAS libgrace)
  target_include_directories(libgrace PUBLIC ${GRACE_RUNTIME_INCLUDE_DIRS} PRIVATE ${GRACE_RUNTIME_PRIVATE_INCLUDE_DIRS})
  target_link_libraries(libgrace PUBLIC OpenMP::OpenMP_CXX)
  # GRACE_BLAS: the GRACE-3L CPU GEMM calls cblas_sgemm (LP64) of a BLAS found by CMake; off: the built-in kernel
  # (the Python wheels). LAMMPS sets the definition on its own target (ML-PACE.cmake).
  option(GRACE_BLAS "GRACE-3L CPU GEMM through cblas_sgemm" OFF)
  if(GRACE_BLAS)
    find_package(BLAS REQUIRED)
    target_compile_definitions(libgrace PUBLIC GRACE_BLAS)
    target_link_libraries(libgrace PUBLIC ${BLAS_LIBRARIES})
  endif()
  set_target_properties(libgrace PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden)
endif()

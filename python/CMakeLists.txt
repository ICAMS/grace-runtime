# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
cmake_minimum_required(VERSION 3.16)
project(grace_runtime_py CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# SKBUILD_SABI_COMPONENT (scikit-build-core, wheel.py-api = cp312): Development.SABIModule, so STABLE_ABI builds abi3
find_package(Python 3.10 COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/GraceRuntime.cmake")
# one module per ISA level, chosen at import by grace_runtime/_dispatch.py; -DGRACE_RUNTIME_ISAS=native builds a single
# -march=native module instead (from source: SKBUILD_CMAKE_DEFINE="GRACE_RUNTIME_ISAS=native" pip install ...)
set(GRACE_RUNTIME_ISAS "x86_64_v3;x86_64_v4" CACHE STRING "ISA builds of the extension: x86_64_v3, x86_64_v4, native")
set(GRACE_RUNTIME_FLAGS_x86_64_v3 -march=x86-64-v3)
set(GRACE_RUNTIME_FLAGS_x86_64_v4 -march=x86-64-v4 -mprefer-vector-width=512)
set(GRACE_RUNTIME_FLAGS_native -march=native)
foreach(isa IN LISTS GRACE_RUNTIME_ISAS)
  set(t _core_${isa})
  nanobind_add_module(${t} NB_STATIC NOMINSIZE STABLE_ABI src/core.cpp src/neighbors.cpp ${GRACE_RUNTIME_SOURCES})
  target_compile_definitions(${t} PRIVATE GRACE_RUNTIME_MODULE=${t})
  target_include_directories(${t} PRIVATE ${GRACE_RUNTIME_INCLUDE_DIRS} ${GRACE_RUNTIME_PRIVATE_INCLUDE_DIRS})
  target_link_libraries(${t} PRIVATE OpenMP::OpenMP_CXX)
  # wheels: no dependency on the build compiler's libstdc++ (newer than the target systems'); libgomp stays shared
  # (auditwheel vendors it under a mangled name). --exclude-libs: the static libstdc++ symbols stay local, else
  # another libstdc++ in the process interposes them; the version script exports only PyInit_*
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(${t} PRIVATE -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL
                        -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map)
  endif()
  target_compile_options(${t} PRIVATE ${GRACE_RUNTIME_FLAGS_${isa}} -O3)
  install(TARGETS ${t} LIBRARY DESTINATION grace_runtime)
endforeach()

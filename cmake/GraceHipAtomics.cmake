# Hardware floating-point atomics for the Kokkos engines on AMD GPUs (HIP backend). Include after find_package(Kokkos).
#   grace_hip_unsafe_fp_atomics(<target>...): compiles <target> with -munsafe-fp-atomics when
#   GRACE_HIP_UNSAFE_FP_ATOMICS resolves to ON. Without the flag, clang lowers fp32/fp64 atomicAdd to global memory
#   into compare-and-swap loops on CDNA; with it, into the hardware FP atomic instructions (5-16 % faster engines on
#   MI250X). The hardware atomics are only correct on coarse-grained device memory (hipMalloc, which is what the
#   Kokkos HIP device views are on a discrete GPU); on fine-grained (host-coherent, managed, system) memory they can
#   silently drop updates.
#   AUTO (default): ON for the discrete CDNA GPUs gfx908, gfx90a, gfx940, gfx941, gfx942 (MI100, MI200, MI300X);
#   OFF for the MI300A APU (Kokkos AMD_GFX942_APU: Kokkos may put device views in unified memory there; untested),
#   for RDNA and other targets, and always OFF without the Kokkos HIP backend (CUDA, host-only).
#   ON / OFF: force it on (HIP only) or off.
set(GRACE_HIP_UNSAFE_FP_ATOMICS "AUTO" CACHE STRING
    "-munsafe-fp-atomics for the HIP Kokkos engines: AUTO (discrete CDNA GPUs), ON or OFF")
set_property(CACHE GRACE_HIP_UNSAFE_FP_ATOMICS PROPERTY STRINGS AUTO ON OFF)

function(grace_hip_unsafe_fp_atomics)
  string(TOUPPER "${GRACE_HIP_UNSAFE_FP_ATOMICS}" _mode)
  set(_on OFF)
  if(NOT Kokkos_ENABLE_HIP)
    set(_why "no Kokkos HIP backend")
  elseif(_mode STREQUAL "AUTO")
    set(_why "Kokkos_ARCH ${Kokkos_ARCH}")
    foreach(_a IN LISTS Kokkos_ARCH)
      if(_a MATCHES "^(AMD_GFX(908|90A|940|941|942)|VEGA(908|90A))$")    # not AMD_GFX942_APU
        set(_on ON)
      endif()
    endforeach()
  elseif(_mode STREQUAL "ON" OR _mode STREQUAL "OFF")
    set(_on ${_mode})
    set(_why "forced")
  else()
    message(FATAL_ERROR "GRACE_HIP_UNSAFE_FP_ATOMICS=${GRACE_HIP_UNSAFE_FP_ATOMICS}: expected AUTO, ON or OFF")
  endif()
  message(STATUS "GRACE_HIP_UNSAFE_FP_ATOMICS=${GRACE_HIP_UNSAFE_FP_ATOMICS}: -munsafe-fp-atomics ${_on} (${_why})")
  if(_on)
    foreach(_t IN LISTS ARGN)
      target_compile_options(${_t} PRIVATE $<$<COMPILE_LANGUAGE:CXX,HIP>:-munsafe-fp-atomics>)
    endforeach()
  endif()
endfunction()

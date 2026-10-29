// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// CDNA matrix-core and wavefront primitives shared by the engines' own matrix-core kernels (kk_gemm.h, GRACE-3L SPBF,
// GRACE-1L / 2L bond kernels), the build switches they depend on, and the on / off decision of those kernels.
#pragma once

#include "grace/kk/support.h"    // env_switch

#include <Kokkos_Core.hpp>

#include <string>
#include <type_traits>

// CDNA (gfx90a, gfx94x; wave64) builds: the fp32 matrix cores (v_mfma_f32_*f32, full fp32 products and fp32
// accumulation). Selected from the Kokkos GPU architecture of the build (KOKKOS_ARCH_AMD_GFX90A / GFX940 / GFX942 /
// GFX942_APU), so RDNA (wave32) and the other targets keep their vector-ALU kernels. The MI300A APU compiles for gfx942
// too. gfx950 is not listed: untested (no gfx950 target in the ROCm 6.3 compiler).
#if defined(KOKKOS_ARCH_AMD_GFX90A) || defined(KOKKOS_ARCH_AMD_GFX940) || defined(KOKKOS_ARCH_AMD_GFX942) || \
    defined(KOKKOS_ARCH_AMD_GFX942_APU)
#define GRACE_KK_GEMM_MFMA 1
#else
#define GRACE_KK_GEMM_MFMA 0
#endif
// The default of the engines' own matrix-core kernels (GRACE-3L SPBF, GRACE-1L / 2L bond kernels) where they are
// compiled: on for gfx90a (validated and timed on MI250X), off (opt-in) for gfx940 / gfx941 / gfx942 / gfx942_APU until
// they have run on MI300 hardware
#if GRACE_KK_GEMM_MFMA && defined(KOKKOS_ARCH_AMD_GFX90A)
#define GRACE_KK_MFMA_DEFAULT 1
#else
#define GRACE_KK_MFMA_DEFAULT 0
#endif
// CDNA builds (64-lane wavefronts: gfx906, gfx908, gfx90a, gfx94x, gfx950), from the Kokkos GPU architecture as above; RDNA
// (gfx10 / 11 / 12) runs 32-lane wavefronts. Selects the kernels written for 64-lane wavefronts at compile time.
#if defined(KOKKOS_ARCH_AMD_GFX906) || defined(KOKKOS_ARCH_AMD_GFX908) || defined(KOKKOS_ARCH_AMD_GFX90A) || \
    defined(KOKKOS_ARCH_AMD_GFX940) || defined(KOKKOS_ARCH_AMD_GFX942) || defined(KOKKOS_ARCH_AMD_GFX942_APU) || \
    defined(KOKKOS_ARCH_AMD_GFX950)
#define GRACE_KK_WAVE64 1
#else
#define GRACE_KK_WAVE64 0
#endif
// 1 in the device pass of a CDNA target (the MFMA builtins exist), 0 in the host pass and for the other targets
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
#define GRACE_KK_MFMA_DEVICE 1
#else
#define GRACE_KK_MFMA_DEVICE 0
#endif

namespace grace {
namespace kk {

// Whether an engine's matrix-core kernel runs, and why (one line on stderr per process names the reason): env is the
// value of the kernel's environment variable var (env_switch, support.h: 1, 0, or -1 when unset), fits whether the
// model is inside the kernel's limits. CDNA builds: var=1 or the build's default (GRACE_KK_MFMA_DEFAULT) unless var=0;
// other builds: off.
struct KernelChoice {
  bool on;
  std::string why;
};
// Whether a kernel written for 64-lane wavefronts runs: CDNA builds (GRACE_KK_WAVE64) unless var=0; other builds:
// off (env as for mfma_switch)
inline KernelChoice wave64_switch(const char *var, int env)
{
  const std::string v(var);
#if GRACE_KK_WAVE64
  return {env != 0, env == 0 ? v + "=0" : env == 1 ? v + "=1" : "default with 64-lane wavefronts; " + v + "=0 disables it"};
#else
  return {false, env == 1 ? v + "=1 ignored: no 64-lane wavefronts in this build target"
                          : std::string("no 64-lane wavefronts in this build target")};
#endif
}
inline KernelChoice wave64_switch(const char *var) { return wave64_switch(var, env_switch(var)); }
inline KernelChoice mfma_switch(const char *var, int env, bool fits)
{
  const std::string v(var);
#if GRACE_KK_GEMM_MFMA
  const bool want = env < 0 ? bool(GRACE_KK_MFMA_DEFAULT) : env == 1;
  return {want && fits, env == 0  ? v + "=0"
                        : !want   ? "opt-in on gfx94x until tested on MI300: " + v + "=1 enables them"
                        : !fits   ? std::string("model outside the limits of the MFMA kernels")
                        : env == 1 ? v + "=1"
                                   : "default on gfx90a; " + v + "=0 disables them"};
#else
  (void) fits;
  return {false, env == 1 ? v + "=1 ignored: no fp32 matrix cores in this build target (CDNA only)"
                          : std::string("no fp32 matrix cores in this build target")};
#endif
}

// l of the standard Y index y = l^2 + m, l <= 4
KOKKOS_INLINE_FUNCTION constexpr int l_of_y(int y) { return y < 1 ? 0 : y < 4 ? 1 : y < 9 ? 2 : y < 16 ? 3 : 4; }

// The device-only bodies below abort in the host pass: the kernels' lambdas must capture the same variables in the host
// and the device pass, so the device-only code is these functions, not an #if inside the lambda.

// shared memory written by some lanes of a wavefront, read by others of the same wavefront
KOKKOS_INLINE_FUNCTION void wave_sync()
{
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#else
  Kokkos::abort("grace kk: wavefront barrier outside HIP device code");
#endif
}

// a value equal in all lanes of the wavefront (e.g. team_rank() / 64) marked as such, so that what is indexed by it
// stays in scalar registers and loads (identity outside HIP device code)
KOKKOS_INLINE_FUNCTION int wave_uniform(int v)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_readfirstlane(v);
#else
  return v;
#endif
}

// v of lane k (wavefront-uniform k, e.g. a constant) for all lanes, in scalar registers: one load per lane and
// broadcasts instead of the same address loaded by all lanes (float or double)
template<class T>
KOKKOS_INLINE_FUNCTION T wave_readlane(T v, int k)
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "wave_readlane: float or double");
#if defined(__HIP_DEVICE_COMPILE__)
  if constexpr (sizeof(T) == 4) {
    return __builtin_bit_cast(T, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), k));
  } else {
    const unsigned long long b = __builtin_bit_cast(unsigned long long, v);
    const unsigned lo = (unsigned) __builtin_amdgcn_readlane((int) (unsigned) b, k);
    const unsigned hi = (unsigned) __builtin_amdgcn_readlane((int) (unsigned) (b >> 32), k);
    return __builtin_bit_cast(T, ((unsigned long long) hi << 32) | lo);
  }
#else
  (void) k;
  Kokkos::abort("grace kk: wavefront readlane outside HIP device code");
  return v;
#endif
}

// 64-lane wavefront shuffles
KOKKOS_INLINE_FUNCTION int wave_shfl(int v, int src)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return __shfl(v, src, 64);
#else
  (void) src;
  Kokkos::abort("grace kk: wavefront shuffle outside HIP device code");
  return v;
#endif
}
KOKKOS_INLINE_FUNCTION float wave_shfl(float v, int src)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return __shfl(v, src, 64);
#else
  (void) src;
  Kokkos::abort("grace kk: wavefront shuffle outside HIP device code");
  return v;
#endif
}
KOKKOS_INLINE_FUNCTION float wave_shfl_xor(float v, int m)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return __shfl_xor(v, m, 64);
#else
  (void) m;
  Kokkos::abort("grace kk: wavefront shuffle outside HIP device code");
  return v;
#endif
}
KOKKOS_INLINE_FUNCTION double wave_shfl_xor(double v, int m)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return __shfl_xor(v, m, 64);
#else
  (void) m;
  Kokkos::abort("grace kk: wavefront shuffle outside HIP device code");
  return v;
#endif
}

#if GRACE_KK_GEMM_MFMA
typedef float mfma_f4v __attribute__((ext_vector_type(4)));
// D += A B for one 16x16x4 fp32 block (v_mfma_f32_16x16x4f32: A[i][k] in lane i + 16 k, B[k][j] in lane j + 16 k,
// D[i][j] in register i % 4 of lane j + 16 (i / 4))
KOKKOS_INLINE_FUNCTION void mfma_16x16x4(float a, float b, mfma_f4v &acc)
{
#if GRACE_KK_MFMA_DEVICE
  acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
#else
  (void) a; (void) b; (void) acc;
  Kokkos::abort("grace kk: MFMA kernel outside CDNA device code");
#endif
}
#endif

}    // namespace kk
}    // namespace grace

// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// rocBLAS for the GRACE Kokkos engines on HIP (AMD GPUs), loaded at run time: librocblas.so.5 (ROCm 7) or
// librocblas.so.4 (ROCm 6) is dlopen'ed on first use (never linked: no DT_NEEDED, no hipBLAS; the symbols used are the
// same in both). Without the library (reported once on stderr), or with GRACE_RUNTIME_NO_ROCBLAS set (non-empty,
// not "0"), rocblas_ok() is false. The engines call it through grace/kk/blas.h, which then runs the hand-written
// Kokkos GEMMs of kk_gemm.h instead.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

struct _rocblas_handle;

namespace grace {
namespace kk {
namespace detail {
// rocBLAS ABI (rocblas/internal/rocblas-types.h, LP64 build: rocblas_int = int32_t, rocblas_stride = int64_t)
using rocblas_handle_t = _rocblas_handle *;
using rocblas_status_t = int;    // rocblas_status, rocblas_status_success = 0
using rocblas_op_t = int;        // rocblas_operation: none = 111, transpose = 112
struct RocblasApi {
  bool ok = false;
  rocblas_status_t (*create)(rocblas_handle_t *) = nullptr;
  rocblas_status_t (*destroy)(rocblas_handle_t) = nullptr;
  rocblas_status_t (*set_stream)(rocblas_handle_t, hipStream_t) = nullptr;
  rocblas_status_t (*sgemm)(rocblas_handle_t, rocblas_op_t, rocblas_op_t, int32_t, int32_t, int32_t, const float *,
                            const float *, int32_t, const float *, int32_t, const float *, float *, int32_t) = nullptr;
  rocblas_status_t (*dgemm)(rocblas_handle_t, rocblas_op_t, rocblas_op_t, int32_t, int32_t, int32_t, const double *,
                            const double *, int32_t, const double *, int32_t, const double *, double *,
                            int32_t) = nullptr;
  rocblas_status_t (*sgemm_batched)(rocblas_handle_t, rocblas_op_t, rocblas_op_t, int32_t, int32_t, int32_t,
                                    const float *, const float *const[], int32_t, const float *const[], int32_t,
                                    const float *, float *const[], int32_t, int32_t) = nullptr;
  rocblas_status_t (*sgemm_strided_batched)(rocblas_handle_t, rocblas_op_t, rocblas_op_t, int32_t, int32_t, int32_t,
                                            const float *, const float *, int32_t, int64_t, const float *, int32_t,
                                            int64_t, const float *, float *, int32_t, int64_t, int32_t) = nullptr;
};
inline const RocblasApi &rocblas_api()
{
  static const RocblasApi api = [] {
    RocblasApi a;
    const char *off = std::getenv("GRACE_RUNTIME_NO_ROCBLAS");
    if (off && *off && *off != '0') return a;
    void *h = nullptr;
    // the rocBLAS of the build's ROCm (ROCm 6: librocblas.so.4, ROCm 7: .so.5): one of another ROCm would load a second
    // HIP runtime into the process
#if HIP_VERSION_MAJOR >= 7
    const char *lib = "librocblas.so.5";
#else
    const char *lib = "librocblas.so.4";
#endif
    h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
      const char *err = dlerror();
      std::fprintf(stderr, "grace kk: rocBLAS not loaded (%s); the hand-written Kokkos GEMMs are used instead "
                   "(GRACE_RUNTIME_NO_ROCBLAS=1 silences this)\n", err ? err : lib);
      return a;
    }
    a.create = (decltype(a.create)) dlsym(h, "rocblas_create_handle");
    a.destroy = (decltype(a.destroy)) dlsym(h, "rocblas_destroy_handle");
    a.set_stream = (decltype(a.set_stream)) dlsym(h, "rocblas_set_stream");
    a.sgemm = (decltype(a.sgemm)) dlsym(h, "rocblas_sgemm");
    a.dgemm = (decltype(a.dgemm)) dlsym(h, "rocblas_dgemm");
    a.sgemm_batched = (decltype(a.sgemm_batched)) dlsym(h, "rocblas_sgemm_batched");
    a.sgemm_strided_batched = (decltype(a.sgemm_strided_batched)) dlsym(h, "rocblas_sgemm_strided_batched");
    a.ok = a.create && a.destroy && a.set_stream && a.sgemm && a.dgemm && a.sgemm_batched && a.sgemm_strided_batched;
    return a;
  }();
  return api;
}
inline bool rocblas_ok() { return rocblas_api().ok; }
}    // namespace detail
}    // namespace kk
}    // namespace grace

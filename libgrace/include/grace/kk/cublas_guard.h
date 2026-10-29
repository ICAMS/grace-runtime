// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// cuBLAS of the GRACE Kokkos engines (Sgemm, Dgemm; GRACE-3L also SgemmBatched, SgemmStridedBatched). Default
// (LAMMPS): linked. GRACE_KK_CUBLAS_DLOPEN (the Python GPU wheels): libcublas.so.12 is loaded at run time, in this
// order: the nvidia-cublas-cu12 wheel (the Python side preloads it, so the soname matches that copy), a system
// libcublas.so.12 on the loader path (ldconfig, LD_LIBRARY_PATH, a CUDA toolkit), none. Without one, below the
// version floor (cublas_floor), or with GRACE_RUNTIME_NO_CUBLAS set, the engines create no handle and take their
// hand-written / Kokkos GEMM paths.
#pragma once

#include <cublas_v2.h>

#ifdef GRACE_KK_CUBLAS_DLOPEN
#include <Kokkos_Core.hpp>
#include <cuda_runtime_api.h>
#include <library_types.h>

#include <cstdlib>
#include <dlfcn.h>

namespace grace {
namespace kk {
namespace detail {
struct CublasApi {
  bool ok = false;
  decltype(&cublasCreate_v2) create = nullptr;
  decltype(&cublasDestroy_v2) destroy = nullptr;
  decltype(&cublasSetStream_v2) set_stream = nullptr;
  decltype(&cublasSetMathMode) set_math_mode = nullptr;
  decltype(&cublasSgemm_v2) sgemm = nullptr;
  decltype(&cublasDgemm_v2) dgemm = nullptr;
  decltype(&cublasSgemmBatched) sgemm_batched = nullptr;
  decltype(&cublasSgemmStridedBatched) sgemm_strided_batched = nullptr;
};
// the oldest cuBLAS (major * 100 + minor) the engines take on a device of compute capability major cc: 12.0 in
// general (any libcublas.so.12: the calls above exist since CUDA 11, and nvidia-cublas-cu12 12.0.2 to 12.9 pass the
// GPU tests on an sm86 card), 12.8 from compute capability 10.0 on (Blackwell: the first cuBLAS with Blackwell kernels,
// per NVIDIA's CUDA 12.8 release notes; not measured). Same rule as grace_runtime._cuda.cublas_floor.
inline int cublas_floor(int cc) { return cc >= 10 ? 1208 : 1200; }
// the loaded cuBLAS meets the floor of the current device; false when either cannot be queried
inline bool cublas_version_ok(void *h)
{
  using property_t = cublasStatus_t (*)(libraryPropertyType, int *);
  const auto property = (property_t) dlsym(h, "cublasGetProperty");
  int major = 0, minor = 0, dev = 0, cc = 0;
  if (!property || property(MAJOR_VERSION, &major) != CUBLAS_STATUS_SUCCESS ||
      property(MINOR_VERSION, &minor) != CUBLAS_STATUS_SUCCESS)
    return false;
  // the device Kokkos runs on (the first call may come from any host thread), else the thread's current one
  if (Kokkos::is_initialized())
    dev = Kokkos::Cuda().cuda_device();
  else if (cudaGetDevice(&dev) != cudaSuccess) {
    cudaGetLastError();    // clear it: not a later kernel's error
    return false;
  }
  if (cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return major * 100 + minor >= cublas_floor(cc);
}
inline const CublasApi &cublas_api()
{
  static const CublasApi api = [] {
    CublasApi a;
    const char *off = std::getenv("GRACE_RUNTIME_NO_CUBLAS");
    if (off && *off && *off != '0') return a;
    void *h = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
    if (!h) return a;
    a.create = (decltype(a.create)) dlsym(h, "cublasCreate_v2");
    a.destroy = (decltype(a.destroy)) dlsym(h, "cublasDestroy_v2");
    a.set_stream = (decltype(a.set_stream)) dlsym(h, "cublasSetStream_v2");
    a.set_math_mode = (decltype(a.set_math_mode)) dlsym(h, "cublasSetMathMode");
    a.sgemm = (decltype(a.sgemm)) dlsym(h, "cublasSgemm_v2");
    a.dgemm = (decltype(a.dgemm)) dlsym(h, "cublasDgemm_v2");
    a.sgemm_batched = (decltype(a.sgemm_batched)) dlsym(h, "cublasSgemmBatched");
    a.sgemm_strided_batched = (decltype(a.sgemm_strided_batched)) dlsym(h, "cublasSgemmStridedBatched");
    a.ok = a.create && a.destroy && a.set_stream && a.set_math_mode && a.sgemm && a.dgemm && a.sgemm_batched &&
           a.sgemm_strided_batched && cublas_version_ok(h);
    if (!a.ok) dlclose(h);    // missing symbols or below the floor: not used
    return a;
  }();
  return api;
}
inline bool cublas_ok() { return cublas_api().ok; }
}    // namespace detail
}    // namespace kk
}    // namespace grace
#undef cublasCreate
#undef cublasDestroy
#undef cublasSetStream
#undef cublasSgemm
#undef cublasDgemm
#undef cublasSgemmBatched
#undef cublasSgemmStridedBatched
#define cublasCreate(...) ::grace::kk::detail::cublas_api().create(__VA_ARGS__)
#define cublasDestroy(...) ::grace::kk::detail::cublas_api().destroy(__VA_ARGS__)
#define cublasSetStream(...) ::grace::kk::detail::cublas_api().set_stream(__VA_ARGS__)
#define cublasSetMathMode(...) ::grace::kk::detail::cublas_api().set_math_mode(__VA_ARGS__)
#define cublasSgemm(...) ::grace::kk::detail::cublas_api().sgemm(__VA_ARGS__)
#define cublasDgemm(...) ::grace::kk::detail::cublas_api().dgemm(__VA_ARGS__)
#define cublasSgemmBatched(...) ::grace::kk::detail::cublas_api().sgemm_batched(__VA_ARGS__)
#define cublasSgemmStridedBatched(...) ::grace::kk::detail::cublas_api().sgemm_strided_batched(__VA_ARGS__)
#else
namespace grace {
namespace kk {
namespace detail {
inline bool cublas_ok() { return true; }
}    // namespace detail
}    // namespace kk
}    // namespace grace
#endif

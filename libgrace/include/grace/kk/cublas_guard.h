// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// cuBLAS of the GRACE Kokkos engines (Sgemm, Dgemm; GRACE-3L also SgemmBatched, SgemmStridedBatched). Default
// (LAMMPS): linked. GRACE_KK_CUBLAS_DLOPEN (the Python GPU wheels):
// libcublas.so.12 is loaded at run time (the Python side preloads nvidia-cublas-cu12 when it is installed); without
// it, or with GRACE_RUNTIME_NO_CUBLAS set, the engines create no handle and take their hand-written paths.
#pragma once

#include <cublas_v2.h>

#ifdef GRACE_KK_CUBLAS_DLOPEN
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
           a.sgemm_strided_batched;
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

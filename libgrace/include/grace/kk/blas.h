// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The BLAS of the GEMM paths of the GRACE Kokkos engines, one interface for the GPU backends (grace::kk::blas):
// a handle, create / destroy, set_stream (binds the handle to a Kokkos execution space instance: the GEMMs run in
// issue order with the kernels on that instance), set_pedantic_math (true fp32 / fp64: no tf32), and the GEMMs the
// engines call: sgemm, dgemm, sgemm_batched (device pointer arrays), sgemm_strided_batched. The arguments are those
// of cuBLAS: column-major operands, alpha and beta by host pointer, status SUCCESS on success.
// GRACE_KK_GEMM is defined when the GEMM paths exist in the build (CUDA or HIP Kokkos); without it (host backends)
// this header declares nothing.
//   CUDA: cuBLAS (grace/kk/cublas_guard.h), linked or, with GRACE_KK_CUBLAS_DLOPEN, loaded at run time; the calls are
//         the cuBLAS calls of the engines before this interface, one to one.
//   HIP:  rocBLAS loaded at run time (grace/kk/rocblas_guard.h: the soname of the build's ROCm, .so.4 / .so.5; never linked or
//         bundled; GRACE_RUNTIME_NO_ROCBLAS skips it). Without rocBLAS the handle runs the hand-written Kokkos GEMMs
//         of grace/kk/kk_gemm.h (fp32 only; dgemm fails) on the instance given to set_stream;
//         GRACE_RUNTIME_NO_KKGEMM as well: create fails, the engines run their hand-written fused kernels.
// available(): a handle can be created (CUDA: libcublas loaded, unless linked; HIP: rocBLAS or the Kokkos GEMMs).
// optional: no handle is a supported configuration (dlopen'ed cuBLAS, HIP); else a failed create is an error.
// name(): "cublas", "rocblas", "kkgemm", or "" (none available).
// selftest_error(), gemm_selftest(): the engines' GEMM self-tests at construction.
#pragma once

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cublas_v2.h>
#include "grace/kk/cublas_guard.h"
#define GRACE_KK_GEMM 1
#elif defined(KOKKOS_ENABLE_HIP)
#include "grace/kk/kk_gemm.h"
#include "grace/kk/rocblas_guard.h"
#define GRACE_KK_GEMM 1
#endif

#ifdef GRACE_KK_GEMM
#include <cmath>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace grace {
namespace kk {
namespace blas {

#if defined(KOKKOS_ENABLE_CUDA)
using handle_t = cublasHandle_t;
using status_t = cublasStatus_t;
using op_t = cublasOperation_t;
inline constexpr status_t SUCCESS = CUBLAS_STATUS_SUCCESS;
inline constexpr op_t OP_N = CUBLAS_OP_N;
inline constexpr op_t OP_T = CUBLAS_OP_T;
#ifdef GRACE_KK_CUBLAS_DLOPEN
inline constexpr bool optional = true;
#else
inline constexpr bool optional = false;
#endif

inline bool available() { return detail::cublas_ok(); }
inline const char *name() { return detail::cublas_ok() ? "cublas" : ""; }
inline status_t create(handle_t *h) { return cublasCreate(h); }
inline status_t destroy(handle_t h) { return cublasDestroy(h); }
template<class Exec> status_t set_stream(handle_t h, const Exec &exec)
{
  static_assert(std::is_same_v<Exec, Kokkos::Cuda>, "blas::set_stream: a Kokkos::Cuda instance");
  return cublasSetStream(h, exec.cuda_stream());
}
inline status_t set_pedantic_math(handle_t h) { return cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH); }
inline status_t sgemm(handle_t h, op_t ta, op_t tb, int m, int n, int k, const float *alpha, const float *A, int lda,
                      const float *B, int ldb, const float *beta, float *C, int ldc)
{
  return cublasSgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline status_t dgemm(handle_t h, op_t ta, op_t tb, int m, int n, int k, const double *alpha, const double *A, int lda,
                      const double *B, int ldb, const double *beta, double *C, int ldc)
{
  return cublasDgemm(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline status_t sgemm_batched(handle_t h, op_t ta, op_t tb, int m, int n, int k, const float *alpha,
                              const float *const A[], int lda, const float *const B[], int ldb, const float *beta,
                              float *const C[], int ldc, int batch)
{
  return cublasSgemmBatched(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch);
}
inline status_t sgemm_strided_batched(handle_t h, op_t ta, op_t tb, int m, int n, int k, const float *alpha,
                                      const float *A, int lda, long long sA, const float *B, int ldb, long long sB,
                                      const float *beta, float *C, int ldc, long long sC, int batch)
{
  return cublasSgemmStridedBatched(h, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
}

#elif defined(KOKKOS_ENABLE_HIP)
namespace detail {
inline bool kk_gemm_on()
{
  static const bool on = [] {
    const char *off = std::getenv("GRACE_RUNTIME_NO_KKGEMM");
    return !(off && *off && *off != '0');
  }();
  return on;
}
}    // namespace detail
// rocBLAS handle, or none: the Kokkos GEMMs on exec
struct Handle {
  ::grace::kk::detail::rocblas_handle_t rb = nullptr;
  Kokkos::HIP exec;
};
using handle_t = Handle *;
using status_t = int;    // rocblas_status
using op_t = int;        // rocblas_operation
inline constexpr status_t SUCCESS = 0;
inline constexpr status_t NOT_IMPLEMENTED = 2;    // rocblas_status_not_implemented
inline constexpr op_t OP_N = 111;
inline constexpr op_t OP_T = 112;
inline constexpr bool optional = true;

inline bool available() { return ::grace::kk::detail::rocblas_ok() || detail::kk_gemm_on(); }
inline const char *name()
{
  return ::grace::kk::detail::rocblas_ok() ? "rocblas" : detail::kk_gemm_on() ? "kkgemm" : "";
}
inline status_t create(handle_t *h)
{
  *h = nullptr;
  ::grace::kk::detail::rocblas_handle_t rb = nullptr;
  if (::grace::kk::detail::rocblas_ok()) {
    const status_t st = ::grace::kk::detail::rocblas_api().create(&rb);
    if (st != SUCCESS) return st;
  } else if (!detail::kk_gemm_on()) {
    return NOT_IMPLEMENTED;
  }
  *h = new Handle{rb, Kokkos::HIP()};
  return SUCCESS;
}
inline status_t destroy(handle_t h)
{
  const status_t st = h && h->rb ? ::grace::kk::detail::rocblas_api().destroy(h->rb) : SUCCESS;
  delete h;
  return st;
}
template<class Exec> status_t set_stream(handle_t h, const Exec &exec)
{
  static_assert(std::is_same_v<Exec, Kokkos::HIP>, "blas::set_stream: a Kokkos::HIP instance");
  h->exec = exec;
  return h->rb ? ::grace::kk::detail::rocblas_api().set_stream(h->rb, exec.hip_stream()) : SUCCESS;
}
// rocBLAS fp32 / fp64 GEMMs compute in full precision by default; the Kokkos GEMMs too
inline status_t set_pedantic_math(handle_t) { return SUCCESS; }
inline status_t sgemm(handle_t h, op_t ta, op_t tb, int m, int n, int k, const float *alpha, const float *A, int lda,
                      const float *B, int ldb, const float *beta, float *C, int ldc)
{
  if (h->rb)
    return ::grace::kk::detail::rocblas_api().sgemm(h->rb, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  gemm::sgemm(h->exec, ta == OP_T, tb == OP_T, m, n, k, *alpha, gemm::StridedOperands{A, B, C, 0, 0, 0}, lda, ldb,
              *beta, ldc, 1);
  return SUCCESS;
}
inline status_t dgemm(handle_t h, op_t ta, op_t tb, int m, int n, int k, const double *alpha, const double *A, int lda,
                      const double *B, int ldb, const double *beta, double *C, int ldc)
{
  if (!h->rb) return NOT_IMPLEMENTED;    // no fp64 Kokkos GEMM (no HIP path calls dgemm)
  return ::grace::kk::detail::rocblas_api().dgemm(h->rb, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline status_t sgemm_batched(handle_t h, op_t ta, op_t tb, int m, int n, int k, const float *alpha,
                              const float *const A[], int lda, const float *const B[], int ldb, const float *beta,
                              float *const C[], int ldc, int batch)
{
  if (h->rb)
    return ::grace::kk::detail::rocblas_api().sgemm_batched(h->rb, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C,
                                                            ldc, batch);
  gemm::sgemm(h->exec, ta == OP_T, tb == OP_T, m, n, k, *alpha, gemm::ArrayOperands{A, B, C}, lda, ldb, *beta, ldc,
              batch);
  return SUCCESS;
}
inline status_t sgemm_strided_batched(handle_t h, op_t ta, op_t tb, int m, int n, int k, const float *alpha,
                                      const float *A, int lda, long long sA, const float *B, int ldb, long long sB,
                                      const float *beta, float *C, int ldc, long long sC, int batch)
{
  if (h->rb)
    return ::grace::kk::detail::rocblas_api().sgemm_strided_batched(h->rb, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb,
                                                                    sB, beta, C, ldc, sC, batch);
  gemm::sgemm(h->exec, ta == OP_T, tb == OP_T, m, n, k, *alpha, gemm::StridedOperands{A, B, C, sA, sB, sC}, lda, ldb,
              *beta, ldc, batch);
  return SUCCESS;
}
#endif

// Self-test of a GEMM C = X W (X: Mt x n_in, W: n_in x n_out; x(b, k), w(k, j), c(b, j) read host copies) against the
// fp64 product of the same operands: the largest error per element in units of its fp32 rounding bound
// 2 (n_in + 2) u sum_k |x w| (u = 2^-24; w may be the fp64 weights the device copy was rounded from). At most 1 for a
// correct fp32 GEMM however much the terms of an element cancel (an error relative to |C| is not: it fails correct
// GEMMs for some weights); a transpose or leading-dimension error gives orders of magnitude more. tf32 inputs
// (u = 2^-11) give ~2^12 / n_in^1.5 on average: caught only for narrow inputs (set_pedantic_math excludes tf32).
template<class X, class W, class C>
double selftest_error(int Mt, int n_in, int n_out, const X &x, const W &w, const C &c)
{
  const double u = std::ldexp(1.0, -24);
  double worst = 0.0;
  for (int b = 0; b < Mt; b++)
    for (int j = 0; j < n_out; j++) {
      double ref = 0.0, sabs = 0.0;
      for (int k = 0; k < n_in; k++) {
        const double t = (double) x(b, k) * (double) w(k, j);
        ref += t;
        sabs += std::fabs(t);
      }
      const double r = std::fabs((double) c(b, j) - ref) / (2.0 * (n_in + 2) * u * sabs + 1e-30);
      if (!(r <= worst)) worst = std::isnan(r) ? 1e30 : r;    // NaN in C or in the weights fails
    }
  return worst;
}

// Input of the engines' GEMM self-tests: selftest_rows x n_in values selftest_input(b, k), small integers / 100
constexpr int selftest_rows = 4;
inline double selftest_input(int b, int k) { return 0.01 * (((b * 7 + k * 3) % 13) - 6); }

// The engines' GEMM self-test: gemm(X, C, M) computes C = X W on the device (X: the M x n_in self-test input, C:
// M x n_out, both contiguous row-major) with the engine's own GEMM call; W: the host fp64 weights, row-major n_in x
// n_out; selftest_error of C. tag: the engine's label prefix of the two device views ("g2l": g2l:selftest_X).
template<class T, class Dev, class Gemm>
double gemm_selftest(const std::string &tag, int n_in, int n_out, const double *W, const Gemm &gemm)
{
  constexpr int M = selftest_rows;
  Kokkos::View<T **, Kokkos::LayoutRight, Dev> X(tag + ":selftest_X", M, n_in), C(tag + ":selftest_C", M, n_out);
  auto hX = Kokkos::create_mirror_view(X);
  for (int b = 0; b < M; b++)
    for (int k = 0; k < n_in; k++) hX(b, k) = (T) selftest_input(b, k);
  Kokkos::deep_copy(X, hX);
  gemm(X.data(), C.data(), M);
  Kokkos::fence();
  auto hC = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);
  return selftest_error(M, n_in, n_out, hX, [&](int k, int j) { return W[(size_t) k * n_out + j]; }, hC);
}

}    // namespace blas
}    // namespace kk
}    // namespace grace
#endif    // GRACE_KK_GEMM

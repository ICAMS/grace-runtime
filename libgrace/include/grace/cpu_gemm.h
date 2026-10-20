// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from lammps-ace src/ML-PACE/grace_cpu_gemm.h (feature/grace-runtime a73b8b17ee); contributing author: Yury
// Lysogorskiy (ICAMS). Single-precision row-major GEMM of the CPU GRACE-3L driver: C = alpha A B + beta C.
// With GRACE_BLAS it calls cblas_sgemm (LP64 interface); the built-in kernel is a register-blocked loop (4 rows x 64
// columns of accumulators; 8 rows for 32- and 16-column bands). GRACE_GEMM=builtin (or G3LF_GEMM=builtin) forces it.
// Both are called from inside OpenMP parallel regions, one thread per call: MKL and OpenMP-built OpenBLAS run
// single-threaded there.
#ifndef GRACE_CPU_GEMM_H
#define GRACE_CPU_GEMM_H

#include "grace/cpu_simd.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef GRACE_BLAS
// the reference CBLAS interface with 32-bit integers (LP64), as exported by MKL (lp64), OpenBLAS and BLIS;
// declared here so that no BLAS header is needed at build time
extern "C" void cblas_sgemm(int layout, int transa, int transb, int M, int N, int K, float alpha, const float *A,
                            int lda, const float *B, int ldb, float beta, float *C, int ldc);
#endif

extern "C" int openblas_get_parallel(void) __attribute__((weak));
extern "C" void MKL_Get_Version_String(char *, int) __attribute__((weak));

namespace grace {
namespace gemm {

typedef simd::f16 gv16;
inline gv16 gld(const float *p) { return simd::load(p); }
inline void gst(float *p, const gv16 v) { simd::store(p, v); }

// C[MR x 16 NV] = alpha A B + beta C, register tile (A, B and C point at the tile's first row / column)
template<int MR, int NV>
inline void tile(int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
  gv16 acc[MR][NV];
  for (int r = 0; r < MR; r++)
    for (int v = 0; v < NV; v++) acc[r][v] = gv16{};
  for (int k = 0; k < K; k++) {
    gv16 b[NV];
    for (int v = 0; v < NV; v++) b[v] = gld(B + (size_t) k * ldb + 16 * v);
    for (int r = 0; r < MR; r++) {
      const float a = A[(size_t) r * lda + k];
      for (int v = 0; v < NV; v++) acc[r][v] += a * b[v];
    }
  }
  for (int r = 0; r < MR; r++)
    for (int v = 0; v < NV; v++) {
      float *c = C + (size_t) r * ldc + 16 * v;
      gv16 o = alpha * acc[r][v];
      if (beta != 0.0f) o += beta * gld(c);
      gst(c, o);
    }
}

template<int MR>
inline void scalar_cols(int j0, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                        float *C, int ldc)
{
  for (int j = j0; j < N; j++)
    for (int r = 0; r < MR; r++) {
      float s = 0.0f;
      for (int k = 0; k < K; k++) s += A[(size_t) r * lda + k] * B[(size_t) k * ldb + j];
      float &c = C[(size_t) r * ldc + j];
      c = alpha * s + (beta != 0.0f ? beta * c : 0.0f);
    }
}

inline void gemm_builtin(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                         float *C, int ldc)
{
  // column bands outermost: the band of B (K x 64 floats) stays in L1 while all row tiles stream through it
  auto at = [](const float *X, int i, int ld) { return X + (size_t) i * ld; };
  auto ct = [](float *X, int i, int ld) { return X + (size_t) i * ld; };
  int j = 0;
  for (; j + 64 <= N; j += 64) {
    int i = 0;
    for (; i + 4 <= M; i += 4) tile<4, 4>(K, alpha, at(A, i, lda), lda, B + j, ldb, beta, ct(C, i, ldc) + j, ldc);
    for (; i < M; i++) tile<1, 4>(K, alpha, at(A, i, lda), lda, B + j, ldb, beta, ct(C, i, ldc) + j, ldc);
  }
  for (; j + 32 <= N; j += 32) {
    int i = 0;
    for (; i + 8 <= M; i += 8) tile<8, 2>(K, alpha, at(A, i, lda), lda, B + j, ldb, beta, ct(C, i, ldc) + j, ldc);
    for (; i < M; i++) tile<1, 2>(K, alpha, at(A, i, lda), lda, B + j, ldb, beta, ct(C, i, ldc) + j, ldc);
  }
  for (; j + 16 <= N; j += 16) {
    int i = 0;
    for (; i + 8 <= M; i += 8) tile<8, 1>(K, alpha, at(A, i, lda), lda, B + j, ldb, beta, ct(C, i, ldc) + j, ldc);
    for (; i < M; i++) tile<1, 1>(K, alpha, at(A, i, lda), lda, B + j, ldb, beta, ct(C, i, ldc) + j, ldc);
  }
  if (j < N)
    for (int i = 0; i < M; i++) scalar_cols<1>(j, N, K, alpha, at(A, i, lda), lda, B, ldb, beta, ct(C, i, ldc), ldc);
}

// use the built-in kernel: G3LF_GEMM=builtin, or set when the linked BLAS cannot be called from OpenMP threads
// (pthreads OpenBLAS)
inline bool &force_builtin()
{
  static bool f = [] {
    const char *e = getenv("GRACE_GEMM") ? getenv("GRACE_GEMM") : getenv("G3LF_GEMM");
    return e != nullptr && std::strcmp(e, "builtin") == 0;
  }();
  return f;
}

// G3LF_GEMMSTAT: time per (M, N, K) on OpenMP thread 0 (diagnostics)
struct GemmStat {
  bool on = getenv("G3LF_GEMMSTAT") != nullptr;
  std::map<std::tuple<int, int, int>, std::pair<double, long>> t;
  ~GemmStat()
  {
    if (!on || t.empty()) return;
    double tot = 0;
    for (auto &e : t) tot += e.second.first;
    fprintf(stderr, "GEMM stats (thread 0): %.3f s\n", tot);
    for (auto &e : t)
      if (e.second.first > 0.01 * tot)
        fprintf(stderr, "  M %4d N %4d K %4d  calls %8ld  %7.3f s  %6.1f GFLOP/s\n", std::get<0>(e.first),
                std::get<1>(e.first), std::get<2>(e.first), e.second.second, e.second.first,
                2.0 * std::get<0>(e.first) * std::get<1>(e.first) * std::get<2>(e.first) * e.second.second /
                    e.second.first * 1e-9);
  }
};
inline GemmStat &gemm_stat()
{
  static GemmStat s;
  return s;
}

// C[M x N] = alpha A[M x K] B[K x N] + beta C, all row-major
inline void gemm_impl(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                      float *C, int ldc)
{
  if (M <= 0 || N <= 0) return;
  if (K <= 0) {
    for (int i = 0; i < M; i++)
      for (int j = 0; j < N; j++) C[(size_t) i * ldc + j] = beta != 0.0f ? beta * C[(size_t) i * ldc + j] : 0.0f;
    return;
  }
#ifdef GRACE_BLAS
  if (force_builtin())
    gemm_builtin(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  else
    cblas_sgemm(101, 111, 111, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);    // row major, no transposes
#else
  gemm_builtin(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

inline void gemm(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C,
                 int ldc)
{
  GemmStat &st = gemm_stat();
#ifdef _OPENMP
  if (st.on && omp_get_thread_num() == 0) {
    const double t0 = omp_get_wtime();
    gemm_impl(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    auto &e = st.t[std::make_tuple(M, N, K)];
    e.first += omp_get_wtime() - t0;
    e.second++;
    return;
  }
#endif
  gemm_impl(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// description of the compiled path, and a warning when the BLAS cannot be called from OpenMP threads
inline const char *backend()
{
#ifdef GRACE_BLAS
  if (force_builtin()) return "built-in (linked BLAS not usable from OpenMP threads)";
  if (MKL_Get_Version_String) return "cblas (MKL)";
  if (openblas_get_parallel) return "cblas (OpenBLAS)";
  return "cblas";
#else
  return "built-in";
#endif
}
inline bool blas_thread_unsafe()
{
#ifdef GRACE_BLAS
  return openblas_get_parallel && openblas_get_parallel() == 1;    // pthreads OpenBLAS
#else
  return false;
#endif
}

}    // namespace gemm
}    // namespace grace

#endif

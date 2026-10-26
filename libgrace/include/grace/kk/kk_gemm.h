// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Hand-written fp32 GEMM in Kokkos (no BLAS library): C = alpha op(A) op(B) + beta C, column-major operands as in
// BLAS, batched either by strides or by device pointer arrays, launched on the execution space instance exec. Used on
// HIP by grace/kk/blas.h when rocBLAS is not loaded, so the engines' GEMM paths do not need a BLAS library.
// Generic (sgemm_tiled): team per (batch, 64 x 64 tile of C), 256 threads with a 4 x 4 register tile each (rows
// tx + 16 r, columns ty + 16 c), k in steps of 16 through padded team scratch; fp32 accumulation in k order.
// beta == 0 does not read C (both kernels).
#pragma once

#include "grace/kk/kk_mfma.h"    // GRACE_KK_GEMM_MFMA, GRACE_KK_MFMA_DEVICE

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace grace {
namespace kk {
namespace gemm {

struct StridedOperands {
  const float *A, *B;
  float *C;
  long long sA, sB, sC;
  KOKKOS_INLINE_FUNCTION const float *a(int b) const { return A + sA * b; }
  KOKKOS_INLINE_FUNCTION const float *bm(int b) const { return B + sB * b; }
  KOKKOS_INLINE_FUNCTION float *c(int b) const { return C + sC * b; }
};
struct ArrayOperands {
  const float *const *A, *const *B;
  float *const *C;
  KOKKOS_INLINE_FUNCTION const float *a(int b) const { return A[b]; }
  KOKKOS_INLINE_FUNCTION const float *bm(int b) const { return B[b]; }
  KOKKOS_INLINE_FUNCTION float *c(int b) const { return C[b]; }
};

template<class Exec, bool TA, bool TB, class Ops>
void sgemm_tiled(const Exec &exec, int m, int n, int k, float alpha, const Ops &ops, int lda, int ldb, float beta,
                 int ldc, int batch)
{
  constexpr int BM = 64, BN = 64, BK = 16, LA = BM + 1, LB = BN + 1, NT = 256;
  if (m <= 0 || n <= 0 || batch <= 0) return;
  const int tm = (m + BM - 1) / BM, tn = (n + BN - 1) / BN;
  // launch bounds = the team size: without them the HIP kernel is compiled for up to 1024 threads per block, i.e.
  // at most 128 VGPRs, and the 4 x 4 tile spilled to scratch (160 B / lane; MI250X: 3.2 -> 6.4 TF on the 3L shapes)
  using policy_t = Kokkos::TeamPolicy<Exec, Kokkos::LaunchBounds<NT, 1>>;
  using member_t = typename policy_t::member_type;
  const size_t shm = sizeof(float) * BK * (LA + LB);
  policy_t pol(exec, (long) batch * tm * tn, NT);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(shm));
  Kokkos::parallel_for("kk:sgemm", pol, KOKKOS_LAMBDA(const member_t &t) {
    float *As = (float *) t.team_shmem().get_shmem(sizeof(float) * BK * (LA + LB));
    float *Bs = As + BK * LA;
    const int lr = t.league_rank(), b = lr / (tm * tn), rr = lr - b * (tm * tn);
    const int bj = rr / tm, bi = rr - bj * tm;
    const float *A = ops.a(b), *B = ops.bm(b);
    float *C = ops.c(b);
    const int tid = t.team_rank(), tx = tid & 15, ty = tid >> 4;
    const int i0 = bi * BM, j0 = bj * BN;
    float acc[4][4];
    for (int r = 0; r < 4; r++)
      for (int c = 0; c < 4; c++) acc[r][c] = 0.0f;
    for (int k0 = 0; k0 < k; k0 += BK) {
      for (int e = tid; e < BM * BK; e += NT) {
        // contiguous global reads: along i for op(A) = A, along k for op(A) = A^T
        const int ii = TA ? e / BK : e % BM, kk = TA ? e % BK : e / BM;
        const int gi = i0 + ii, gk = k0 + kk;
        float v = 0.0f;
        if (gi < m && gk < k) v = TA ? A[gk + (long) gi * lda] : A[gi + (long) gk * lda];
        As[kk * LA + ii] = v;
      }
      for (int e = tid; e < BN * BK; e += NT) {
        // contiguous global reads: along k for op(B) = B, along j for op(B) = B^T
        const int jj = TB ? e % BN : e / BK, kk = TB ? e / BN : e % BK;
        const int gj = j0 + jj, gk = k0 + kk;
        float v = 0.0f;
        if (gj < n && gk < k) v = TB ? B[gj + (long) gk * ldb] : B[gk + (long) gj * ldb];
        Bs[kk * LB + jj] = v;
      }
      t.team_barrier();
#pragma unroll
      for (int kk = 0; kk < BK; kk++) {
        float a[4], bb[4];
        for (int r = 0; r < 4; r++) a[r] = As[kk * LA + tx + 16 * r];
        for (int c = 0; c < 4; c++) bb[c] = Bs[kk * LB + ty + 16 * c];
        for (int r = 0; r < 4; r++)
          for (int c = 0; c < 4; c++) acc[r][c] += a[r] * bb[c];
      }
      t.team_barrier();
    }
    for (int c = 0; c < 4; c++) {
      const int gj = j0 + ty + 16 * c;
      if (gj >= n) continue;
      for (int r = 0; r < 4; r++) {
        const int gi = i0 + tx + 16 * r;
        if (gi >= m) continue;
        float &o = C[gi + (long) gj * ldc];
        o = beta == 0.0f ? alpha * acc[r][c] : alpha * acc[r][c] + beta * o;
      }
    }
  });
}

// CDNA (gfx90a, gfx94x; wave64; GRACE_KK_GEMM_MFMA, kk_mfma.h): the same GEMM on the fp32 matrix cores
// (v_mfma_f32_32x32x2f32, full fp32 products and fp32 accumulation, as rocBLAS uses them); RDNA (wave32) and other
// targets keep sgemm_tiled. MI250X GCD on the 3L shapes (m 8192, n 32-128, k 32-128, 25 batches): 3-6.5 TF ->
// 3.5-17.7 TF (rocBLAS 6-22.6 TF).
#if GRACE_KK_GEMM_MFMA
typedef float f16v __attribute__((ext_vector_type(16)));
// one 32x32x2 fp32 MFMA (CDNA2/3). The kernels' lambdas must capture the same variables in the host and the device
// pass, so the device-only part is this function, not an #if inside the lambda.
KOKKOS_INLINE_FUNCTION void mfma_32x32x2(float a, float b, f16v &acc)
{
#if GRACE_KK_MFMA_DEVICE
  acc = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, acc, 0, 0, 0);
#else
  (void) a; (void) b; (void) acc;
  Kokkos::abort("kk_gemm: MFMA kernel outside CDNA device code");
#endif
}
// BM x BN tile per team of 256 threads = 4 wavefronts (WM x WN), wave tile (BM / WM) x (BN / WN) = NI x NJ MFMA
// 32x32 blocks; k steps of BK through team scratch, the next step prefetched into registers.
template<class Exec, bool TA, bool TB, int BM, int BN, int WM, class Ops>
void sgemm_mfma(const Exec &exec, int m, int n, int k, float alpha, const Ops &ops, int lda, int ldb, float beta,
                int ldc, int batch)
{
  constexpr int BK = 16, NT = 256, WN = 4 / WM, NI = BM / WM / 32, NJ = BN / WN / 32;
  constexpr int LA = BM + 4, LB = BN + 4;
  constexpr int NAL = BM * BK / NT, NBL = BN * BK / NT;
  if (m <= 0 || n <= 0 || batch <= 0) return;
  const int tm = (m + BM - 1) / BM, tn = (n + BN - 1) / BN;
  using policy_t = Kokkos::TeamPolicy<Exec, Kokkos::LaunchBounds<NT, 1>>;
  using member_t = typename policy_t::member_type;
  const size_t shm = sizeof(float) * BK * (LA + LB);
  policy_t pol(exec, (long) batch * tm * tn, NT);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(shm));
  Kokkos::parallel_for("kk:sgemm_mfma", pol, KOKKOS_LAMBDA(const member_t &t) {
    float *As = (float *) t.team_shmem().get_shmem(sizeof(float) * BK * (LA + LB));
    float *Bs = As + BK * LA;
    const int lr = t.league_rank(), b = lr / (tm * tn), rr = lr - b * (tm * tn);
    const int bj = rr / tm, bi = rr - bj * tm;
    const float *A = ops.a(b), *B = ops.bm(b);
    float *C = ops.c(b);
    const int tid = t.team_rank();
    const int i0 = bi * BM, j0 = bj * BN;
    const int w = tid >> 6, l = tid & 63, wm = w % WM, wn = w / WM;
    f16v acc[NI][NJ];
#pragma unroll
    for (int p = 0; p < NI; p++)
#pragma unroll
      for (int q = 0; q < NJ; q++)
#pragma unroll
        for (int r = 0; r < 16; r++) acc[p][q][r] = 0.0f;
    float ra[NAL], rb[NBL];
#define KKG_LOAD(K0)                                                                                               \
  _Pragma("unroll") for (int u = 0; u < NAL; u++) {                                                               \
    const int e = tid + u * NT, ii = TA ? e / BK : e % BM, kk = TA ? e % BK : e / BM;                             \
    const int gi = i0 + ii, gk = (K0) + kk;                                                                       \
    ra[u] = (gi < m && gk < k) ? (TA ? A[gk + (long) gi * lda] : A[gi + (long) gk * lda]) : 0.0f;                \
  }                                                                                                               \
  _Pragma("unroll") for (int u = 0; u < NBL; u++) {                                                               \
    const int e = tid + u * NT, jj = TB ? e % BN : e / BK, kk = TB ? e / BN : e % BK;                             \
    const int gj = j0 + jj, gk = (K0) + kk;                                                                       \
    rb[u] = (gj < n && gk < k) ? (TB ? B[gj + (long) gk * ldb] : B[gk + (long) gj * ldb]) : 0.0f;                \
  }
    KKG_LOAD(0)
    for (int k0 = 0; k0 < k; k0 += BK) {
#pragma unroll
      for (int u = 0; u < NAL; u++) {
        const int e = tid + u * NT, ii = TA ? e / BK : e % BM, kk = TA ? e % BK : e / BM;
        As[kk * LA + ii] = ra[u];
      }
#pragma unroll
      for (int u = 0; u < NBL; u++) {
        const int e = tid + u * NT, jj = TB ? e % BN : e / BK, kk = TB ? e / BN : e % BK;
        Bs[kk * LB + jj] = rb[u];
      }
      t.team_barrier();
      if (k0 + BK < k) { KKG_LOAD(k0 + BK) }
#pragma unroll
      for (int kk = 0; kk < BK; kk += 2) {
        const int kr = kk + (l >> 5);
        float av[NI], bv[NJ];
#pragma unroll
        for (int p = 0; p < NI; p++) av[p] = As[kr * LA + (wm * NI + p) * 32 + (l & 31)];
#pragma unroll
        for (int q = 0; q < NJ; q++) bv[q] = Bs[kr * LB + (wn * NJ + q) * 32 + (l & 31)];
#pragma unroll
        for (int p = 0; p < NI; p++)
#pragma unroll
          for (int q = 0; q < NJ; q++) mfma_32x32x2(bv[q], av[p], acc[p][q]);   // D = C^T: lanes along i
      }
      t.team_barrier();
    }
#undef KKG_LOAD
    // MFMA output D[x][y] with y = lane % 32 = row i of C, x = 8 (r / 4) + 4 (lane / 32) + r % 4 = column j: each
    // store instruction writes 32 consecutive rows of two columns (coalesced along the column-major C)
#pragma unroll
    for (int p = 0; p < NI; p++) {
      const int gi = i0 + (wm * NI + p) * 32 + (l & 31);
      if (gi >= m) continue;
#pragma unroll
      for (int q = 0; q < NJ; q++)
#pragma unroll
        for (int r = 0; r < 16; r++) {
          const int gj = j0 + (wn * NJ + q) * 32 + 8 * (r >> 2) + 4 * (l >> 5) + (r & 3);
          if (gj < n) {
            float &o = C[gi + (long) gj * ldc];
            o = beta == 0.0f ? alpha * acc[p][q][r] : alpha * acc[p][q][r] + beta * o;
          }
        }
    }
  });
}
#endif

#if GRACE_KK_GEMM_MFMA
// sgemm_mfma with the op codes as run-time arguments
template<class Exec, int BM, int BN, int WM, class Ops>
void sgemm_mfma_op(const Exec &exec, bool ta, bool tb, int m, int n, int k, float alpha, const Ops &ops, int lda,
                   int ldb, float beta, int ldc, int batch)
{
  if (!ta && !tb) sgemm_mfma<Exec, false, false, BM, BN, WM>(exec, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
  else if (!ta && tb) sgemm_mfma<Exec, false, true, BM, BN, WM>(exec, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
  else if (ta && !tb) sgemm_mfma<Exec, true, false, BM, BN, WM>(exec, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
  else sgemm_mfma<Exec, true, true, BM, BN, WM>(exec, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
}
#endif

// op codes as in BLAS: false = N, true = T
template<class Exec, class Ops>
void sgemm(const Exec &exec, bool ta, bool tb, int m, int n, int k, float alpha, const Ops &ops, int lda, int ldb,
           float beta, int ldc, int batch)
{
#if GRACE_KK_GEMM_MFMA
  if constexpr (std::is_same_v<Exec, Kokkos::HIP>) {
    if (n <= 32) sgemm_mfma_op<Exec, 256, 32, 4>(exec, ta, tb, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
    else if (n <= 64) sgemm_mfma_op<Exec, 256, 64, 4>(exec, ta, tb, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
    else sgemm_mfma_op<Exec, 128, 128, 2>(exec, ta, tb, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
    return;
  }
#endif
  if (!ta && !tb) sgemm_tiled<Exec, false, false>(exec, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
  else if (!ta && tb) sgemm_tiled<Exec, false, true>(exec, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
  else if (ta && !tb) sgemm_tiled<Exec, true, false>(exec, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
  else sgemm_tiled<Exec, true, true>(exec, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
}

}    // namespace gemm
}    // namespace kk
}    // namespace grace

// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The hand-written Kokkos fp32 GEMMs of grace/kk/kk_gemm.h (C = alpha op(A) op(B) + beta C, column-major, batched by
// strides or by device pointer arrays) against a host fp64 reference: the dispatcher sgemm, sgemm_tiled and, in builds
// for CDNA GPUs, sgemm_mfma with each of its three tile shapes; m, n, k off the tile sizes, leading dimensions larger
// than the matrices, beta 0 (C holds NaN: must not be read) and 0.5, op N / T on both sides. Elements of C outside
// op(A) op(B) (padding rows, gaps between batches) must stay untouched. Tolerance: fp32 accumulation, per element
// 2 (k + 2) u sum_k |alpha a b| + |beta c| (u = 2^-24). GPU backends only (the kernels use teams of 256 threads).
// The same problems through grace/kk/blas.h (cuBLAS; on HIP rocBLAS or, without it, the kernels above). Every launch
// goes to a second instance of the device (its own stream: the kernels' exec argument, blas::set_stream), and the
// result is read back on that instance alone, without a device-wide fence.
#include "grace/kk/blas.h"
#include "grace/kk/kk_gemm.h"
#include "../test_util.h"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using Dev = Kokkos::DefaultExecutionSpace;
using Mem = Dev::memory_space;
namespace g = grace::kk::gemm;

namespace {
int n_checked = 0;
const Dev *g_ex = nullptr;    // the instance of every launch (main: its own stream)
#ifdef GRACE_KK_GEMM
namespace bl = grace::kk::blas;
bl::handle_t g_blas = nullptr;
#endif

struct Case {
  int m, n, k, batch;
};

// one kernel variant: (ta, tb, ops) -> launch
template<bool TA, bool TB> struct Problem {
  int m, n, k, batch, lda, ldb, ldc;
  long long sA, sB, sC;                 // strides of the strided layout (also the slot size of the array layout)
  std::vector<float> A, B, C0;          // host, batch slots of sA / sB / sC
  std::vector<double> S, Sabs;          // sum_k a b and sum_k |a b| per (batch, i, j)
  Problem(const Case &c, std::mt19937 &rng) : m(c.m), n(c.n), k(c.k), batch(c.batch)
  {
    lda = (TA ? k : m) + 3;
    ldb = (TB ? n : k) + 5;
    ldc = m + 7;
    sA = (long long) lda * (TA ? m : k) + 11;
    sB = (long long) ldb * (TB ? k : n) + 13;
    sC = (long long) ldc * n + 17;
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    A.resize(sA * batch);
    B.resize(sB * batch);
    C0.resize(sC * batch);
    for (auto &x : A) x = u(rng);
    for (auto &x : B) x = u(rng);
    for (auto &x : C0) x = u(rng);
    S.assign((size_t) batch * m * n, 0.0);
    Sabs.assign((size_t) batch * m * n, 0.0);
    for (int b = 0; b < batch; b++) {
      const float *a = A.data() + sA * b, *bm = B.data() + sB * b;
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
          double s = 0.0, sa = 0.0;
          for (int q = 0; q < k; q++) {
            const double p = (double) (TA ? a[q + (long) i * lda] : a[i + (long) q * lda]) *
                             (double) (TB ? bm[j + (long) q * ldb] : bm[q + (long) j * ldb]);
            s += p;
            sa += std::fabs(p);
          }
          S[((size_t) b * n + j) * m + i] = s;
          Sabs[((size_t) b * n + j) * m + i] = sa;
        }
    }
  }
  // C (host copy after the run) against alpha S + beta C0; untouched elsewhere
  void check(const std::vector<float> &C, float alpha, float beta, const char *what) const
  {
    const double u = std::ldexp(1.0, -24);
    double worst = 0.0;    // max err / tol
    long bad = 0;
    for (int b = 0; b < batch; b++)
      for (long e = 0; e < sC; e++) {
        const long idx = sC * b + e;
        const int j = (int) (e / ldc), i = (int) (e % ldc);
        const float c0 = C0[idx];
        if (j < n && i < m) {
          const size_t r = ((size_t) b * n + j) * m + i;
          const double bc = beta == 0.0f ? 0.0 : beta * (double) c0;    // beta 0: c0 is NaN, not read
          const double ref = alpha * S[r] + bc;
          const double tol = 2.0 * (k + 2) * u * (std::fabs(alpha) * Sabs[r] + std::fabs(bc)) + 1e-30;
          const double err = std::fabs((double) C[idx] - ref);
          if (!(err <= tol)) bad++;
          if (err / tol > worst || std::isnan(err)) worst = std::isnan(err) ? 1e30 : err / tol;
        } else {
          // outside op(A) op(B): bit-identical (NaN stays NaN)
          const bool same = (std::isnan(c0) && std::isnan(C[idx])) || C[idx] == c0;
          if (!same) bad++;
        }
      }
    n_checked++;
    if (bad) {
      std::fprintf(stderr, "FAIL %s TA %d TB %d m %d n %d k %d batch %d beta %g: %ld bad elements, max err/tol %.3g\n",
                   what, (int) TA, (int) TB, m, n, k, batch, beta, bad, worst);
      test_exit(1);
    }
  }
};

template<class T> Kokkos::View<T *, Mem> to_dev(const std::vector<T> &h)
{
  Kokkos::View<T *, Mem> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, "kkg"), h.size());
  Kokkos::deep_copy(d, Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h.data(), h.size()));
  return d;
}
// copied on g_ex after the launches there, no device-wide fence
template<class T> std::vector<T> to_host(const Kokkos::View<T *, Mem> &d)
{
  std::vector<T> h(d.extent(0));
  Kokkos::deep_copy(*g_ex, Kokkos::View<T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h.data(), h.size()), d);
  g_ex->fence();
  return h;
}

// runs launch(ops, lda, ldb, ldc) on fresh device copies, strided and pointer-array batches, beta 0 and 0.5
template<bool TA, bool TB, class Launch> void run_variant(const Problem<TA, TB> &p, const char *what, Launch launch)
{
  const float alpha = 0.75f;
  for (float beta : {0.0f, 0.5f}) {
    std::vector<float> C0 = p.C0;
    if (beta == 0.0f)    // beta 0 must not read C: NaN in op(A) op(B), the rest keeps its values
      for (int b = 0; b < p.batch; b++)
        for (int j = 0; j < p.n; j++)
          for (int i = 0; i < p.m; i++) C0[p.sC * b + (long) j * p.ldc + i] = std::nanf("");
    Problem<TA, TB> q = p;
    q.C0 = C0;
    {    // strided
      auto dA = to_dev(p.A), dB = to_dev(p.B);
      auto dC = to_dev(C0);
      launch(g::StridedOperands{dA.data(), dB.data(), dC.data(), p.sA, p.sB, p.sC}, alpha, beta);
      q.check(to_host(dC), alpha, beta, (std::string(what) + " strided").c_str());
    }
    {    // pointer arrays: one allocation per batch entry, the pointer arrays themselves on the device
      std::vector<Kokkos::View<float *, Mem>> vA, vB, vC;
      std::vector<std::uintptr_t> pA, pB, pC;
      for (int b = 0; b < p.batch; b++) {
        vA.push_back(to_dev(std::vector<float>(p.A.begin() + p.sA * b, p.A.begin() + p.sA * (b + 1))));
        vB.push_back(to_dev(std::vector<float>(p.B.begin() + p.sB * b, p.B.begin() + p.sB * (b + 1))));
        vC.push_back(to_dev(std::vector<float>(C0.begin() + p.sC * b, C0.begin() + p.sC * (b + 1))));
        pA.push_back((std::uintptr_t) vA.back().data());
        pB.push_back((std::uintptr_t) vB.back().data());
        pC.push_back((std::uintptr_t) vC.back().data());
      }
      auto dpA = to_dev(pA), dpB = to_dev(pB), dpC = to_dev(pC);
      launch(g::ArrayOperands{reinterpret_cast<const float *const *>(dpA.data()),
                              reinterpret_cast<const float *const *>(dpB.data()),
                              reinterpret_cast<float *const *>(dpC.data())},
             alpha, beta);
      std::vector<float> C(C0.size());
      for (int b = 0; b < p.batch; b++) {
        const auto h = to_host(vC[b]);
        std::copy(h.begin(), h.end(), C.begin() + p.sC * b);
      }
      q.check(C, alpha, beta, (std::string(what) + " array").c_str());
    }
  }
}

template<bool TA, bool TB> void run_ops(const Case &c, std::mt19937 &rng)
{
  const Problem<TA, TB> p(c, rng);
  const int m = p.m, n = p.n, k = p.k, lda = p.lda, ldb = p.ldb, ldc = p.ldc, batch = p.batch;
  run_variant(p, "sgemm", [&](const auto &ops, float alpha, float beta) {
    g::sgemm(*g_ex, TA, TB, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
  });
  run_variant(p, "sgemm_tiled", [&](const auto &ops, float alpha, float beta) {
    g::sgemm_tiled<Dev, TA, TB>(*g_ex, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
  });
#if GRACE_KK_GEMM_MFMA
  if constexpr (std::is_same_v<Dev, Kokkos::HIP>) {
    run_variant(p, "sgemm_mfma 256x32", [&](const auto &ops, float alpha, float beta) {
      g::sgemm_mfma<Dev, TA, TB, 256, 32, 4>(*g_ex, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
    });
    run_variant(p, "sgemm_mfma 256x64", [&](const auto &ops, float alpha, float beta) {
      g::sgemm_mfma<Dev, TA, TB, 256, 64, 4>(*g_ex, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
    });
    run_variant(p, "sgemm_mfma 128x128", [&](const auto &ops, float alpha, float beta) {
      g::sgemm_mfma<Dev, TA, TB, 128, 128, 2>(*g_ex, m, n, k, alpha, ops, lda, ldb, beta, ldc, batch);
    });
  }
#endif
#ifdef GRACE_KK_GEMM
  if (g_blas)
    run_variant(p, "blas", [&](const auto &ops, float alpha, float beta) {
      const bl::op_t ta = TA ? bl::OP_T : bl::OP_N, tb = TB ? bl::OP_T : bl::OP_N;
      bl::status_t st;
      if constexpr (std::is_same_v<std::decay_t<decltype(ops)>, g::StridedOperands>)
        st = batch == 1 ? bl::sgemm(g_blas, ta, tb, m, n, k, &alpha, ops.A, lda, ops.B, ldb, &beta, ops.C, ldc)
                        : bl::sgemm_strided_batched(g_blas, ta, tb, m, n, k, &alpha, ops.A, lda, ops.sA, ops.B, ldb,
                                                    ops.sB, &beta, ops.C, ldc, ops.sC, batch);
      else
        st = bl::sgemm_batched(g_blas, ta, tb, m, n, k, &alpha, ops.A, lda, ops.B, ldb, &beta, ops.C, ldc, batch);
      CHECK(st == bl::SUCCESS);
    });
#endif
}
#ifdef GRACE_KK_GEMM
// blas::selftest_error on the shape of the engines' self-tests (4 x 138 inputs of their pattern, 138 x 256 weights):
// an output whose terms cancel to ~0 passes; outputs in the wrong place and a NaN fail. blas::gemm_selftest with
// the engines' GEMM call passes, with the weights off by one element it fails.
void check_selftest_error()
{
  const int Mt = bl::selftest_rows, n_in = 138, n_out = 256;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<float> X((size_t) Mt * n_in);    // X(b, k) = X[k + n_in b]: column-major n_in x Mt
  for (int b = 0; b < Mt; b++)
    for (int k = 0; k < n_in; k++) X[k + (size_t) n_in * b] = (float) bl::selftest_input(b, k);
  std::vector<double> W((size_t) n_in * n_out);    // W(k, j) = W[j + n_out k], the model layout
  for (auto &w : W) w = u(rng);
  double s = 0.0;    // output (b 0, j 0): its last weight makes the sum vanish
  for (int k = 0; k < n_in - 1; k++) s += (double) X[k] * W[(size_t) n_out * k];
  W[(size_t) n_out * (n_in - 1)] = -s / (double) X[n_in - 1];

  auto dW = to_dev(std::vector<float>(W.begin(), W.end())), dX = to_dev(X);
  auto dC = to_dev(std::vector<float>((size_t) n_out * Mt));
  const float one = 1.0f, zero = 0.0f;
  CHECK(bl::sgemm(g_blas, bl::OP_N, bl::OP_N, n_out, Mt, n_in, &one, dW.data(), n_out, dX.data(), n_in, &zero,
                  dC.data(), n_out) == bl::SUCCESS);
  std::vector<float> C = to_host(dC);

  auto x = [&](int b, int k) { return X[k + (size_t) n_in * b]; };
  auto w = [&](int k, int j) { return W[j + (size_t) n_out * k]; };
  auto c = [&](int b, int j) { return C[j + (size_t) n_out * b]; };
  const double err = bl::selftest_error(Mt, n_in, n_out, x, w, c);
  std::printf("kk_gemm selftest_error: %.3f x bound (the cancelling output: %.1e, the sum of its other terms %.1e)\n",
              err, (double) c(0, 0), s);
  CHECK(err <= 1.0);

  auto c_swap = [&](int b, int j) { return c(b, j ^ 1); };    // neighbouring outputs exchanged
  CHECK(bl::selftest_error(Mt, n_in, n_out, x, w, c_swap) > 1e3);
  auto c_nan = [&](int b, int j) { return b == 1 && j == 5 ? std::nan("") : (double) c(b, j); };
  CHECK(bl::selftest_error(Mt, n_in, n_out, x, w, c_nan) > 1.0);

  for (bool wrong : {false, true}) {    // the engines' call: C^T (n_out x M) = W^T (n_out x n_in, ld n_out) X^T;
    // wrong: the same weights read as transposed (ld n_in), the transpose / leading-dimension error the gate is for
    const double e = bl::gemm_selftest<float, Dev>("kkg", n_in, n_out, W.data(), [&](const float *X, float *C, int M) {
      CHECK(bl::sgemm(g_blas, wrong ? bl::OP_T : bl::OP_N, bl::OP_N, n_out, M, n_in, &one, dW.data(),
                      wrong ? n_in : n_out, X, n_in, &zero, C, n_out) == bl::SUCCESS);
    });
    std::printf("kk_gemm gemm_selftest%s: %.3g x bound\n", wrong ? ", weights read transposed" : "", e);
    CHECK(wrong ? e > 1e3 : e <= 1.0);
  }
  n_checked += 5;
}
#endif
}    // namespace

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, Mem>::accessible) {
    std::fprintf(stderr, "SKIP: kk_gemm is for GPU backends (default execution space %s)\n", Dev::name());
    return 77;
  }
#if GRACE_KK_GEMM_MFMA
  std::printf("kk_gemm: MFMA kernels compiled in (CDNA build)\n");
#endif
  const Dev ex = Kokkos::Experimental::partition_space(Dev(), 1)[0];
  g_ex = &ex;
#ifdef GRACE_KK_GEMM
  if (bl::available()) {
    CHECK(bl::create(&g_blas) == bl::SUCCESS);
    CHECK(bl::set_stream(g_blas, *g_ex) == bl::SUCCESS);
    CHECK(bl::set_pedantic_math(g_blas) == bl::SUCCESS);
#ifdef KOKKOS_ENABLE_HIP
    CHECK(g_blas->exec.hip_stream() == g_ex->hip_stream());    // the Kokkos GEMMs launch there
#endif
  }
  std::printf("kk_gemm: blas %s\n", g_blas ? bl::name() : "none (not checked)");
#endif
  std::mt19937 rng(12345);
  // off-tile m / n / k (tiles: 64 x 64 x 16 tiled, 256 x 32 / 256 x 64 / 128 x 128 x 16 MFMA); the 3L shape last
  const Case cases[] = {{37, 5, 3, 3}, {64, 64, 16, 2}, {129, 65, 17, 3}, {300, 33, 40, 2}, {200, 130, 50, 2},
                        {1, 1, 1, 1}, {8192, 64, 128, 2}};
  for (const Case &c : cases) {
    run_ops<false, false>(c, rng);
    run_ops<false, true>(c, rng);
    run_ops<true, false>(c, rng);
    run_ops<true, true>(c, rng);
    std::printf("kk_gemm m %5d n %4d k %4d batch %d: ok\n", c.m, c.n, c.k, c.batch);
  }
#ifdef GRACE_KK_GEMM
  if (g_blas) check_selftest_error();
#endif
  std::printf("kk_gemm: %d checks ok\n", n_checked);
#ifdef GRACE_KK_GEMM
  if (g_blas) CHECK(bl::destroy(g_blas) == bl::SUCCESS);
#endif
  return 0;
}

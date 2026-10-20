// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// grace::gemm::gemm (row-major C = alpha A B + beta C, the GEMM of the GRACE-3L CPU driver) against a naive loop in
// double: band and tile edges (N around 16 / 32 / 64), small M, beta 0 and 0.5, padded leading dimensions; the
// built-in kernel always, cblas_sgemm too when built with GRACE_BLAS.
#include "grace/cpu_gemm.h"
#include "test_util.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
static double check(int M, int N, int K, float beta, std::mt19937 &rng)
{
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  const int lda = K + 3, ldb = N + 5, ldc = N + 7;
  std::vector<float> A((size_t) M * lda), B((size_t) K * ldb), C((size_t) M * ldc);
  for (auto &x : A) x = u(rng);
  for (auto &x : B) x = u(rng);
  for (auto &x : C) x = u(rng);
  std::vector<double> ref((size_t) M * ldc);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      double s = 0.0;
      for (int k = 0; k < K; k++) s += (double) A[(size_t) i * lda + k] * B[(size_t) k * ldb + j];
      ref[(size_t) i * ldc + j] = 0.75 * s + beta * (double) C[(size_t) i * ldc + j];
    }
  const std::vector<float> C0 = C;
  grace::gemm::gemm(M, N, K, 0.75f, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
  double err = 0.0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      const double r = ref[(size_t) i * ldc + j];
      err = std::fmax(err, std::fabs(C[(size_t) i * ldc + j] - r) / std::fmax(1.0, std::fabs(r)));
    }
    for (int j = N; j < ldc; j++) CHECK(C[(size_t) i * ldc + j] == C0[(size_t) i * ldc + j]);    // padding untouched
  }
  return err;
}
int main()
{
  std::mt19937 rng(20261011);
  for (int pass = 0; pass < 2; pass++) {
    grace::gemm::force_builtin() = pass == 0;
    std::printf("backend %s\n", grace::gemm::backend());
    double worst = 0.0;
    for (int M : {1, 3, 4, 8, 17})
      for (int N : {1, 15, 16, 32, 64, 80, 133})
        for (int K : {1, 7, 64})
          for (float beta : {0.0f, 0.5f}) worst = std::fmax(worst, check(M, N, K, beta, rng));
    std::printf("max relative error %.3g\n", worst);
    CHECK(worst <= 1e-5);
  }
  return 0;
}

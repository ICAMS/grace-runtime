// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)


/* ----------------------------------------------------------------------
   Contributing author: Yury Lysogorskiy (ICAMS)
   Device and host helpers shared by Grace1LFusedKK and Grace2LFusedKK.
   Internal: include only from their .cpp files.
------------------------------------------------------------------------- */

#pragma once

#include <Kokkos_Core.hpp>


#include <cmath>
#include <vector>

namespace grace {
namespace kk {
namespace GraceFusedKK {

// the last sentence of the warnings for a model without GPU kernels of its own widths
constexpr const char *GRACE_KK_ISSUE =
    "Please open an issue at https://github.com/ICAMS/grace-runtime/issues with these numbers so that fast kernels "
    "can be added";

constexpr int GRACE_FUSED_MAX_LMAX = 4;

struct alignas(16) GF4 { float x, y, z, w; };
// HIP bond backward kernels: row length of the dE/dA rows [channel][lm] (lm contiguous, 16-byte loads): (lmax + 1)^2
// rounded up to whole float4 with an odd number of them (no bank conflicts between consecutive channels)
KOKKOS_INLINE_FUNCTION constexpr int fused_lmp(int nlm)
{
  return 4 * ((nlm + 3) / 4 + ((nlm + 3) / 4 % 2 == 0 ? 1 : 0));
}

// l_of_y (kk_mfma.h): l of the flat index lm = l^2 + l + m, for loops over lm with compile-time bounds
// (lm < (GRACE_FUSED_MAX_LMAX + 1)^2): nvcc does not unroll an inner lm loop whose bounds depend on the outer l, and a
// register array indexed in it (dE/dA row, Y, dY / dr) then goes to local memory
static_assert(GRACE_FUSED_MAX_LMAX == 4, "l_of_y: extend to the new GRACE_FUSED_MAX_LMAX");

// Real spherical harmonics, identical to PairGRACE2LKokkos::compute_Y_bond_chunk.
template<class ViewG>
KOKKOS_INLINE_FUNCTION
void real_sph(const int lmax, const float rx, const float ry, const float rz,
              const ViewG &idx_s, const ViewG &alm, const ViewG &blm, const ViewG &cl,
              const ViewG &dl, float *Y)
{
  float plm[(GRACE_FUSED_MAX_LMAX + 1) * (GRACE_FUSED_MAX_LMAX + 2) / 2];
  plm[0] = 1.0f;
  if (lmax > 0) {
    plm[1] = 1.7320508075688772935f * rz;
    plm[2] = -1.2247448713915890491f;
    for (int ll = 2; ll <= lmax; ll++) {
      for (int mm = 0; mm < ll - 1; mm++) {
        const int idx = ll * (ll + 1) / 2 + mm;
        const int i1 = (ll - 1) * ll / 2 + mm, i2 = (ll - 2) * (ll - 1) / 2 + mm;
        const int ai = idx_s(ll * (ll + 1) + mm);
        plm[idx] = alm(ai) * (rz * plm[i1] + blm(ai) * plm[i2]);
      }
      plm[ll * (ll + 1) / 2 + ll - 1] = dl(ll) * plm[(ll - 1) * ll / 2 + ll - 1] * rz;
      plm[ll * (ll + 1) / 2 + ll] = cl(ll) * plm[(ll - 1) * ll / 2 + ll - 1];
    }
  }
  const float sq2f = 1.4142135623730950488f;
  for (int l = 0; l <= lmax; l++) {
    Y[l * (l + 1)] = plm[l * (l + 1) / 2];
    float pm_re = rx, pm_im = ry;
    for (int m = 1; m <= l; m++) {
      const float fac = (m % 2 == 0) ? 1.0f : -1.0f;
      const float pv = plm[l * (l + 1) / 2 + m];
      Y[l * (l + 1) + m] = sq2f * fac * pm_re * pv;
      Y[l * (l + 1) - m] = sq2f * fac * pm_im * pv;
      const float t_re = pm_re * rx - pm_im * ry;
      const float t_im = pm_re * ry + pm_im * rx;
      pm_re = t_re; pm_im = t_im;
    }
  }
}

struct alignas(16) F4 { float x, y, z, w; };

KOKKOS_INLINE_FUNCTION
void fma4(F4 &a, const F4 &w, const float x)
{
  a.x += w.x * x; a.y += w.y * x; a.z += w.z * x; a.w += w.w * x;
}

// per-atom virial of one bond, as Grace2LKK: half of del (x) f to each atom; centroid form with 9 components
template<class VA, class CVA>
KOKKOS_INLINE_FUNCTION
void tally_vatom(const VA &va, const CVA &cva, bool do_va, bool do_cva, int i, int j, double dx, double dy, double dz,
                 double fx, double fy, double fz)
{
  const double v[9] = {dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz, dy * fx, dz * fx, dz * fy};
  if (do_va)
    for (int k = 0; k < 6; k++) {
      Kokkos::atomic_add(&va(i, k), 0.5 * v[k]);
      Kokkos::atomic_add(&va(j, k), 0.5 * v[k]);
    }
  if (do_cva)
    for (int k = 0; k < 9; k++) {
      Kokkos::atomic_add(&cva(i, k), 0.5 * v[k]);
      Kokkos::atomic_add(&cva(j, k), 0.5 * v[k]);
    }
}

KOKKOS_INLINE_FUNCTION
float dot4(const F4 &a, const F4 &b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

// Radial table: per interval k and output o the cubic a0 + a1 t + a2 t^2 + a3 t^3, t in [0, 1),
// coefficients from the Hermite data in fp64 (no f(k+1) - f(k) cancellation in fp32).
template<class View3>
KOKKOS_INLINE_FUNCTION
float spline_v(const View3 &c, const int N, const float x, const int o)
{
  float s = x * N;
  int k = (int) s;
  if (k > N - 1) k = N - 1;
  const float t = s - k;
  return c(k, o, 0) + t * (c(k, o, 1) + t * (c(k, o, 2) + t * c(k, o, 3)));
}

// Y_lm and G = dY_lm / d(rx, ry, rz) of the polynomial extension of the same real harmonics
// (dual numbers through the recurrences). Surface gradient: dY/dr = (G - rhat (rhat.G)) / r.
// Y_lm at Y[YS lm], G at G[GS lm + 0..2] (default: Y[nlm], G[nlm][3]; 4, 4 with G = Y + 1: [nlm][Y, Gx, Gy, Gz]).
// LM > 0: lmax = LM at compile time (constant trip counts: Y, G and the recursion can stay in registers), the argument
// lmax is ignored.
template<int YS = 1, int GS = 3, int LM = 0, class ViewG>
KOKKOS_INLINE_FUNCTION
void real_sph_grad(const int lmax_rt, const float rx, const float ry, const float rz,
                   const ViewG &idx_s, const ViewG &alm, const ViewG &blm, const ViewG &cl,
                   const ViewG &dl, float *Y, float *G)
{
  const int lmax = LM > 0 ? LM : lmax_rt;
  float plm[(GRACE_FUSED_MAX_LMAX + 1) * (GRACE_FUSED_MAX_LMAX + 2) / 2];
  float dplm[(GRACE_FUSED_MAX_LMAX + 1) * (GRACE_FUSED_MAX_LMAX + 2) / 2];
  plm[0] = 1.0f; dplm[0] = 0.0f;
  if (lmax > 0) {
    plm[1] = 1.7320508075688772935f * rz; dplm[1] = 1.7320508075688772935f;
    plm[2] = -1.2247448713915890491f;     dplm[2] = 0.0f;
    for (int ll = 2; ll <= lmax; ll++) {
      for (int mm = 0; mm < ll - 1; mm++) {
        const int idx = ll * (ll + 1) / 2 + mm;
        const int i1 = (ll - 1) * ll / 2 + mm, i2 = (ll - 2) * (ll - 1) / 2 + mm;
        const int ai = idx_s(ll * (ll + 1) + mm);
        const float a = alm(ai), b = blm(ai);
        plm[idx] = a * (rz * plm[i1] + b * plm[i2]);
        dplm[idx] = a * (plm[i1] + rz * dplm[i1] + b * dplm[i2]);
      }
      const int prev = (ll - 1) * ll / 2 + ll - 1;
      plm[ll * (ll + 1) / 2 + ll - 1] = dl(ll) * plm[prev] * rz;
      dplm[ll * (ll + 1) / 2 + ll - 1] = dl(ll) * (dplm[prev] * rz + plm[prev]);
      plm[ll * (ll + 1) / 2 + ll] = cl(ll) * plm[prev];
      dplm[ll * (ll + 1) / 2 + ll] = cl(ll) * dplm[prev];
    }
  }
  const float sq2f = 1.4142135623730950488f;
  for (int l = 0; l <= lmax; l++) {
    const int c = l * (l + 1);
    Y[YS * c] = plm[l * (l + 1) / 2];
    G[GS * c] = 0.0f; G[GS * c + 1] = 0.0f; G[GS * c + 2] = dplm[l * (l + 1) / 2];
    float pr = rx, pi = ry, prx = 1.0f, pry = 0.0f, pix = 0.0f, piy = 1.0f;
    for (int m = 1; m <= l; m++) {
      const float s = (m % 2 == 0) ? sq2f : -sq2f;
      const float pv = plm[l * (l + 1) / 2 + m], dpv = dplm[l * (l + 1) / 2 + m];
      Y[YS * (c + m)] = s * pr * pv;
      G[GS * (c + m)] = s * prx * pv; G[GS * (c + m) + 1] = s * pry * pv; G[GS * (c + m) + 2] = s * pr * dpv;
      Y[YS * (c - m)] = s * pi * pv;
      G[GS * (c - m)] = s * pix * pv; G[GS * (c - m) + 1] = s * piy * pv; G[GS * (c - m) + 2] = s * pi * dpv;
      const float nr = pr * rx - pi * ry, ni = pr * ry + pi * rx;
      const float nrx = prx * rx + pr - pix * ry, nry = pry * rx - piy * ry - pi;
      const float nix = prx * ry + pix * rx + pi, niy = pry * ry + pr + piy * rx;
      pr = nr; pi = ni; prx = nrx; pry = nry; pix = nix; piy = niy;
    }
  }
}

// Value and d/dx.
template<class View3>
KOKKOS_INLINE_FUNCTION
void spline_vd(const View3 &c, const int N, const float x, const int o, float &v, float &dv)
{
  float s = x * N;
  int k = (int) s;
  if (k > N - 1) k = N - 1;
  const float t = s - k;
  const float a0 = c(k, o, 0), a1 = c(k, o, 1), a2 = c(k, o, 2), a3 = c(k, o, 3);
  v = a0 + t * (a1 + t * (a2 + t * a3));
  dv = (a1 + t * (2.0f * a2 + t * 3.0f * a3)) * (float) N;
}

}    // namespace GraceFusedKK
}    // namespace kk
}    // namespace grace


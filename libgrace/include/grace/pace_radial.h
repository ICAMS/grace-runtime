// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// PACE radial functions in fp64: the radial basis g_k(r), R_nl(r) = sum_k crad[n][l][k] g_k(r), their derivatives and
// the core repulsion (exponential or ZBL). Ported from lammps-user-pace ML-PACE/ace-evaluator/ace_radial.cpp
// (radbase, chebExpCos, chebPow, chebLinear, simplified_bessel, radfunc, radcore, ZBL, cutoff_func_poly), and the
// cubic tables of them the engines evaluate (RadialSpline). The functions of a pair take it as a template (PaceBond on
// the host, the GPU engine's device copy: the same fields, crad a pointer) and compile for the device (GRACE_HD).
#ifndef GRACE_PACE_RADIAL_H
#define GRACE_PACE_RADIAL_H

#include "grace/hd.h"
#include "grace/pace_model.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace grace {
namespace pace {

constexpr double kPi = 3.14159265358979323846264338327950288419;

// 1 below r_in - delta_in, 0 above r_in, a smooth polynomial between
GRACE_HD void cutoff_func_poly(double r, double r_in, double delta_in, double &fc, double &dfc)
{
  if (r <= r_in - delta_in) {
    fc = 1;
    dfc = 0;
  } else if (r >= r_in) {
    fc = 0;
    dfc = 0;
  } else {
    const double x = 1 - 2 * (1 + (r - r_in) / delta_in);
    const double x2 = x * x;
    fc = 0.5 + 7.5 / 2. * (x / 4. - x2 * x / 6. + x2 * x2 * x / 20.);
    dfc = -7.5 / delta_in * (0.25 - x2 / 2.0 + x2 * x2 / 4.);
  }
}

// Chebyshev polynomials T_0 .. T_n(x) and dT/dx (through U_{m-1})
GRACE_HD void cheb(int n, double x, double *t, double *dt)
{
  double u[kPaceMaxNradbase + 2];
  t[0] = 1.0;
  dt[0] = 0.0;
  u[0] = 1.0;
  if (n >= 1) {
    t[1] = x;
    u[1] = 2.0 * x;
  }
  for (int m = 1; m <= n - 1; m++) {
    t[m + 1] = 2.0 * x * t[m] - t[m - 1];
    u[m + 1] = 2.0 * x * u[m] - u[m - 1];
  }
  for (int m = 1; m <= n; m++) dt[m] = m * u[m - 1];
}

GRACE_HD void simplified_bessel(int nradbase, double rc, double x, double *gr, double *dgr)
{
  if (x >= rc) {
    for (int n = 0; n < nradbase; n++) gr[n] = dgr[n] = 0.0;
    return;
  }
  // sin / cos of k theta, theta = pi x / rc, for k = 0 .. nradbase + 1 by the Chebyshev recurrence
  const double th = kPi * x / rc;
  double sk[kPaceMaxNradbase + 2], ck[kPaceMaxNradbase + 2];
  sk[0] = 0.0;
  ck[0] = 1.0;
  sk[1] = std::sin(th);
  ck[1] = std::cos(th);
  for (int k = 2; k <= nradbase + 1; k++) {
    sk[k] = 2.0 * ck[1] * sk[k - 1] - sk[k - 2];
    ck[k] = 2.0 * ck[1] * ck[k - 1] - ck[k - 2];
  }
  // sinc(k theta) and d sinc(k theta) / dx
  auto sinc_k = [&](int k, double &v, double &d) {
    const double z = k * th;
    if (z < 1e-4) {    // series: (cos z z - sin z) / z^2 loses all digits there
      v = 1.0 - z * z / 6.0;
      d = -z / 3.0 * k * kPi / rc;
    } else {
      v = sk[k] / z;
      d = (ck[k] * z - sk[k]) / (z * z) * k * kPi / rc;
    }
  };
  const double base = std::sqrt(2.0) * kPi / (rc * std::sqrt(rc));
  auto f = [&](int n, double &v, double &d) {
    const double pre = (n % 2 ? -1.0 : 1.0) * base * (n + 1) * (n + 2) /
                       std::sqrt((double) (n + 1) * (n + 1) + (double) (n + 2) * (n + 2));
    double v1, d1, v2, d2;
    sinc_k(n + 1, v1, d1);
    sinc_k(n + 2, v2, d2);
    v = pre * (v1 + v2);
    d = pre * (d1 + d2);
  };
  double v, d;
  f(0, gr[0], dgr[0]);
  double d_prev = 1.0;
  for (int n = 1; n < nradbase; n++) {
    const double n1 = n + 1;
    const double en = (double) n * n * (n + 2) * (n + 2) / (4 * n1 * n1 * n1 * n1 + 1);
    const double dn = 1 - en / d_prev;
    f(n, v, d);
    gr[n] = 1 / std::sqrt(dn) * (v + std::sqrt(en / d_prev) * gr[n - 1]);
    dgr[n] = 1 / std::sqrt(dn) * (d + std::sqrt(en / d_prev) * dgr[n - 1]);
    d_prev = dn;
  }
}

// radial basis g_k(r), k < nradbase, with the inner cutoff of the model
template<class Bond> GRACE_HD void radbase(const Bond &b, int inner_cutoff_type, double r, double *gr, double *dgr)
{
  const int nb = b.nradbase;
  const double cut = b.rcut, dcut = b.dcut, lam = b.lambda;
  double cut_in = b.r_in;
  const double dcut_in = b.delta_in;
  if (inner_cutoff_type == kInnerZbl) cut_in = dcut_in == 0 ? 1.0 : 0.0;    // as ML-PACE
  if (r <= cut_in - dcut_in || r >= cut) {
    for (int k = 0; k < nb; k++) gr[k] = dgr[k] = 0.0;
    return;
  }
  double t[kPaceMaxNradbase + 2], dt[kPaceMaxNradbase + 2];
  switch (b.radbase) {
    case kChebExpCos: {
      const double y1 = std::exp(-lam * r / cut), y2 = std::exp(-lam);
      const double x = 1.0 - 2.0 * ((y1 - y2) / (1 - y2));
      const double dx = 2 * (lam / cut) * (y1 / (1 - y2));
      cheb(nb, x, t, dt);
      gr[0] = t[0];
      dgr[0] = dt[0] * dx;
      for (int n = 2; n <= nb; n++) {
        gr[n - 1] = 0.5 - 0.5 * t[n - 1];
        dgr[n - 1] = -0.5 * dt[n - 1] * dx;
      }
      const double env = 0.5 * (1.0 + std::cos(kPi * r / cut));
      const double denv = -0.5 * std::sin(kPi * r / cut) * kPi / cut;
      for (int n = 0; n < nb; n++) {
        dgr[n] = gr[n] * denv + dgr[n] * env;
        gr[n] = gr[n] * env;
      }
      const double r0 = cut - dcut;
      if (r > r0) {
        const double fcut = 0.5 * (1.0 + std::cos(kPi * (r - r0) / dcut));
        const double dfcut = -0.5 * std::sin(kPi * (r - r0) / dcut) * kPi / dcut;
        for (int n = 0; n < nb; n++) {
          dgr[n] = gr[n] * dfcut + dgr[n] * fcut;
          gr[n] = gr[n] * fcut;
        }
      }
      break;
    }
    case kChebPow: {
      double y = 1.0 - r / cut;
      double dy = std::pow(y, lam - 1.0);
      y = dy * y;
      dy = -lam / cut * dy;
      const double x = 2.0 * (1.0 - y) - 1.0, dx = -2.0 * dy;
      cheb(nb, x, t, dt);
      for (int n = 1; n <= nb; n++) {
        gr[n - 1] = 0.5 - 0.5 * t[n];
        dgr[n - 1] = -0.5 * dt[n] * dx;
      }
      break;
    }
    case kChebLinear: {
      const double x = 1.0 - r / cut, dx = -1 / cut;
      cheb(nb, x, t, dt);
      for (int n = 1; n <= nb; n++) {
        gr[n - 1] = 0.5 - 0.5 * t[n];
        dgr[n - 1] = -0.5 * dt[n] * dx;
      }
      break;
    }
    default:
      simplified_bessel(nb, cut, r, gr, dgr);
  }
  if (inner_cutoff_type == kInnerDistance || inner_cutoff_type == kInnerZbl) {
    double fc, dfc;
    cutoff_func_poly(r, cut_in, dcut_in, fc, dfc);
    fc = 1 - fc;
    dfc = -dfc;
    for (int k = 0; k < nb; k++) {
      dgr[k] = dgr[k] * fc + gr[k] * dfc;
      gr[k] = gr[k] * fc;
    }
  }
}

// R_nl = sum_k crad[n][l][k] g_k: fr[n * (lmax + 1) + l]
template<class Bond> GRACE_HD void radfunc(const Bond &b, const double *gr, const double *dgr, double *fr, double *dfr)
{
  const int L = b.lmax + 1, nb = b.nradbase;
  for (int n = 0; n < b.nradmax; n++)
    for (int l = 0; l < L; l++) {
      const double *c = &b.crad[((size_t) n * L + l) * nb];
      double v = 0.0, d = 0.0;
      for (int k = 0; k < nb; k++) {
        v += c[k] * gr[k];
        d += c[k] * dgr[k];
      }
      fr[n * L + l] = v;
      dfr[n * L + l] = d;
    }
}

// exponential core repulsion pre exp(-lambda r^2) / r with a cos envelope (inner cutoff "distance": only inside r_in)
template<class Bond> GRACE_HD void radcore(const Bond &b, int inner_cutoff_type, double r, double &cr, double &dcr)
{
  const double pre = std::abs(b.prehc), lambda = std::abs(b.lambdahc);
  const double r2 = r * r, lr2 = lambda * r2;
  if (lr2 < 50.0) {
    const double y = std::exp(-lr2);
    cr = pre * y / r;
    dcr = -pre * y * (2.0 * lr2 + 1.0) / r2;
    const double x0 = r / b.rcut;
    const double env = 0.5 * (1.0 + std::cos(kPi * x0)), denv = -0.5 * std::sin(kPi * x0) * kPi / b.rcut;
    dcr = cr * denv + dcr * env;
    cr = cr * env;
  } else {
    cr = dcr = 0.0;
  }
  if (inner_cutoff_type == kInnerDistance) {
    double fc, dfc;
    cutoff_func_poly(r, b.r_in, b.delta_in, fc, dfc);
    dcr = dcr * fc + cr * dfc;
    cr = cr * fc;
  }
}

// ZBL screened Coulomb (eV), shifted and switched to zero between rcut - dcut and rcut, times prehc
template<class Bond> GRACE_HD void zbl(const Bond &b, double r, double &cr, double &dcr)
{
  constexpr double K = 14.399645351950543;    // e^2 / (4 pi eps0) in eV A
  constexpr double c[4] = {0.18175, 0.50986, 0.28022, 0.02817}, e[4] = {-3.19980, -0.94229, -0.40290, -0.20162};
  const double out = b.rcut, in = b.rcut - b.dcut;
  if (r >= out) {
    cr = dcr = 0.0;
    return;
  }
  const double a = 0.46850 / (std::pow(b.zi, 0.23) + std::pow(b.zj, 0.23));
  auto Eij = [a, &c, &e](double r, double &E, double &dE, double &d2E) {
    double phi = 0, dphi = 0, d2phi = 0;
    for (int k = 0; k < 4; k++) {
      double t = c[k] * std::exp(e[k] * r / a);
      phi += t;
      t *= e[k];
      dphi += t;
      t *= e[k];
      d2phi += t;
    }
    E = phi / r;
    dE = -phi / (r * r) + dphi / a / r;
    d2E = 2 * phi / (r * r * r) - 2 * dphi / a / (r * r) + d2phi / (a * a) / r;
  };
  double E, dE, d2E, Ec, dEc, d2Ec;
  Eij(r, E, dE, d2E);
  Eij(out, Ec, dEc, d2Ec);
  const double dr = out - in;
  const double A = (-3 * dEc + dr * d2Ec) / (dr * dr);
  const double B = (2 * dEc - dr * d2Ec) / (dr * dr * dr);
  const double C = -Ec + 0.5 * dr * dEc - dr * dr * d2Ec / 12.0;
  double S = C, dS = 0.0;
  if (r > in) {
    const double x = r - in;
    S = A / 3 * x * x * x + B / 4 * x * x * x * x + C;
    dS = A * x * x + B * x * x * x;
  }
  const double pre = b.prehc * K / 2 * b.zi * b.zj;
  cr = pre * (E + S);
  dcr = pre * (dE + dS);
  double fc, dfc;
  cutoff_func_poly(r, out, dr, fc, dfc);
  dcr = dcr * fc + cr * dfc;
  cr = cr * fc;
}

// the core repulsion of the model's inner cutoff type
template<class Bond> GRACE_HD void core(const Bond &b, int inner_cutoff_type, double r, double &cr, double &dcr)
{
  if (inner_cutoff_type == kInnerZbl) zbl(b, r, cr, dcr);
  else radcore(b, inner_cutoff_type, r, cr, dcr);
}

// all radial quantities of one pair at distance r (r < rcut)
template<class Bond> GRACE_HD void radial(const Bond &b, int inner_cutoff_type, double r, double *gr, double *dgr, double *fr,
                   double *dfr, double &cr, double &dcr)
{
  radbase(b, inner_cutoff_type, r, gr, dgr);
  radfunc(b, gr, dgr, fr, dfr);
  core(b, inner_cutoff_type, r, cr, dcr);
}

// the radial quantities the engine needs of one pair: h_p = sum_k hc[p][k] g_k (the rank-1 part of the densities, hc:
// the rank-1 C-tilde coefficients of the pair, [nh][nradbase]), R_nl, the core repulsion; v / d[nh + nradmax (lmax + 1)
// + 1] in that order
template<class Bond> GRACE_HD void radial_h(const Bond &b, int inner_cutoff_type, double r, const double *hc, int nh, double *v, double *d)
{
  double g[kPaceMaxNradbase], dg[kPaceMaxNradbase];
  radbase(b, inner_cutoff_type, r, g, dg);
  for (int p = 0; p < nh; p++) {
    double s = 0.0, ds = 0.0;
    for (int k = 0; k < b.nradbase; k++) {
      s += hc[p * b.nradbase + k] * g[k];
      ds += hc[p * b.nradbase + k] * dg[k];
    }
    v[p] = s;
    d[p] = ds;
  }
  radfunc(b, g, dg, v + nh, d + nh);
  const int qc = nh + b.nradmax * (b.lmax + 1);
  core(b, inner_cutoff_type, r, v[qc], d[qc]);
}

// cubic Hermite tables of the radial_h quantities of one pair on [r0, rcut] in two segments: [r0, r1) with n0
// intervals and [r1, rcut) with n1 (the functions vary fastest at short range: the core repulsion goes as 1 / r, the
// Chebyshev bases of r / rcut too); below r0 they are evaluated directly. Values and derivatives at the nodes are
// exact, so energies and forces stay consistent.
struct RadialSpline {
  int n0 = 0, n1 = 0, nf = 0;    // intervals of the segments, functions (as radial_h)
  double r0 = 0.0, r1 = 0.0;
  double inv_h0 = 0.0, inv_h1 = 0.0;
  std::vector<double> c;    // [n0 + n1][4][nf]: the cubic's coefficients of all functions of an interval together
  void build(const PaceBond &b, int inner_cutoff_type, int intervals0, int intervals1, double r_lo, double r_mid,
             const double *hc, int nh)
  {
    n0 = intervals0;
    n1 = intervals1;
    nf = nh + b.nradmax * (b.lmax + 1) + 1;
    r0 = r_lo;
    r1 = r_mid;
    const double h0 = (r1 - r0) / n0, h1 = (b.rcut - r1) / n1;
    inv_h0 = 1.0 / h0;
    inv_h1 = 1.0 / h1;
    c.resize((size_t) (n0 + n1) * 4 * nf);
    std::vector<double> v((size_t) (std::max(n0, n1) + 1) * nf), d(v.size());
    for (int seg = 0; seg < 2; seg++) {
      const int n = seg ? n1 : n0, k0 = seg ? n0 : 0;
      const double a = seg ? r1 : r0, h = seg ? h1 : h0;
      for (int k = 0; k <= n; k++) {
        // the limit below rcut (ChebLinear: not 0)
        const double r = seg && k == n ? b.rcut * (1.0 - 1e-12) : k == n ? r1 : a + k * h;
        radial_h(b, inner_cutoff_type, r, hc, nh, &v[(size_t) k * nf], &d[(size_t) k * nf]);
      }
      for (int k = 0; k < n; k++)
        for (int f = 0; f < nf; f++) {
          const double v0 = v[(size_t) k * nf + f], v1 = v[(size_t) (k + 1) * nf + f];
          const double d0 = d[(size_t) k * nf + f] * h, d1 = d[(size_t) (k + 1) * nf + f] * h;
          double *q = &c[(size_t) (k0 + k) * 4 * nf + f];
          q[0] = v0;
          q[nf] = d0;
          q[2 * nf] = 3 * (v1 - v0) - 2 * d0 - d1;
          q[3 * nf] = 2 * (v0 - v1) + d0 + d1;
        }
    }
  }
  // the interval of r (r0 <= r < rcut): its coefficients ([4][nf]: value = c0 + x (c1 + x (c2 + x c3)),
  // d/dr = (c1 + x (2 c2 + 3 x c3)) inv_h), the position x in it and inv_h of its segment
  const double *at(double r, double &x, double &inv_h) const
  {
    const bool near = r < r1;
    inv_h = near ? inv_h0 : inv_h1;
    const double t = (r - (near ? r0 : r1)) * inv_h;
    const int k = std::min((int) t, (near ? n0 : n1) - 1);
    x = t - k;
    return &c[(size_t) ((near ? 0 : n0) + k) * 4 * nf];
  }
};

}    // namespace pace
}    // namespace grace

#endif

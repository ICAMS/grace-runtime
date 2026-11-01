// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Host fp64 radial functions in forward mode (value and d/dx), for the spline tables of the CPU and Kokkos engines
#pragma once

#include <cmath>
#include <vector>

namespace grace {

// Chebyshev T_1..T_nrb of (2x-1) times fcut_p(x), and d/dx (x = r / rcut_ij)
inline void radial_basis_x(double x, int p, int nrb, std::vector<double> &g, std::vector<double> &dg)
{
  const double xp = std::pow(x, p), xp1 = xp * x, xp2 = xp1 * x;
  const double fc = 1.0 - 0.5 * (p + 1) * (p + 2) * xp + p * (p + 2) * xp1 - 0.5 * p * (p + 1) * xp2;
  const double dfc = 0.5 * p * (p + 1) * (p + 2) * (-std::pow(x, p - 1) + 2.0 * xp - xp1);
  const double t = 2.0 * x - 1.0;
  double Tp = 1.0, Tc = t, dTp = 0.0, dTc = 2.0;
  g.resize(nrb);
  dg.resize(nrb);
  for (int k = 0; k < nrb; k++) {
    g[k] = Tc * fc;
    dg[k] = dTc * fc + Tc * dfc;
    const double Tn = 2.0 * t * Tc - Tp, dTn = 2.0 * (2.0 * Tc + t * dTc) - dTp;
    Tp = Tc; dTp = dTc; Tc = Tn; dTc = dTn;
  }
}

// the radial MLP (hidden layers silu, or tanh with act 1; linear output), as the device kernels
template<class Layers>
void mlp_x(const Layers &layers, std::vector<double> h, std::vector<double> dh,
           std::vector<double> &out, std::vector<double> &dout, int act)
{
  const int n = layers.size();
  for (int il = 0; il < n; il++) {
    const auto &L = layers[il];
    std::vector<double> s(L.n_out, 0.0), ds(L.n_out, 0.0);
    for (int k = 0; k < L.n_in; k++)
      for (int j = 0; j < L.n_out; j++) {
        const double w = L.W[(size_t) k * L.n_out + j];
        s[j] += w * h[k];
        ds[j] += w * dh[k];
      }
    for (int j = 0; j < L.n_out; j++) {
      s[j] *= L.norm;
      ds[j] *= L.norm;
      if (il < n - 1 && act == 1) {
        s[j] = std::tanh(s[j]);
        ds[j] *= 1.0 - s[j] * s[j];
      } else if (il < n - 1) {
        const double sg = 1.0 / (1.0 + std::exp(-s[j]));
        ds[j] = sg * (1.0 + s[j] * (1.0 - sg)) * ds[j];
        s[j] = s[j] * sg;
      }
    }
    h.swap(s);
    dh.swap(ds);
  }
  out.swap(h);
  dout.swap(dh);
}

}    // namespace grace

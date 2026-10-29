// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// PACE embedding of one atom (ML-PACE ace_abstract_basis.cpp and ACECTildeEvaluator::compute_atom): the Finnis-Sinclair
// functions of its densities, the inner cutoff (ZBL switch on the closest neighbour, or the rho_core cutoff) and the
// core repulsion. For the host and the device (GRACE_HD).
#ifndef GRACE_PACE_EMBEDDING_H
#define GRACE_PACE_EMBEDDING_H

#include "grace/hd.h"
#include "grace/pace_model.h"
#include "grace/pace_radial.h"

#include <cmath>

namespace grace {
namespace pace {

// Finnis-Sinclair F(x) = sign(x) ((1 - g) |x|^m + lambda g |x|), g = exp(-(w |x|)^3)
GRACE_HD void fexp(double x, double m, double &F, double &DF)
{
  const double w = 1.e6, eps = 1e-10;
  const double lambda = std::pow(1.0 / w, m - 1.0);
  if (std::abs(x) > eps) {
    const double a = std::abs(x), am = std::pow(a, m), w3x3 = std::pow(w * a, 3);
    const double sign = x < 0 ? -1 : 1;
    const double g = w3x3 > 30.0 ? 0.0 : std::exp(-w3x3);
    const double omg = 1.0 - g;
    F = sign * (omg * am + lambda * g * a);
    const double dg = -3.0 * w * w * w * a * a * g;
    DF = m * std::pow(a, m - 1.0) * omg - am * dg + lambda * dg * a + lambda * g;
  } else {
    F = lambda * x;
    DF = lambda;
  }
}

// shifted and scaled Finnis-Sinclair (no prefactor)
GRACE_HD void fexp_shifted_scaled(double rho, double mexp, double &F, double &DF)
{
  if (std::abs(mexp - 1.0) < 1e-10) {
    F = rho;
    DF = 1;
    return;
  }
  const double a = std::abs(rho), exprho = std::exp(-a), nx = 1. / mexp;
  const double xoff = std::pow(nx, nx / (1.0 - nx)) * exprho, yoff = std::pow(nx, 1 / (1.0 - nx)) * exprho;
  const double sign = rho < 0 ? -1 : 1;
  F = sign * (std::pow(xoff + a, mexp) - yoff);
  DF = yoff + mexp * (-xoff + 1.0) * std::pow(xoff + a, mexp - 1.);
}

// what the forces need of an atom's embedding
struct Embedded {
  double energy = 0.0;          // the atom's energy
  double dF_drho_core = 0.0;    // dE / d(core repulsion sum)
  double dF_dfcut = 0.0;        // ZBL: dE / d(distance of the closest neighbour)
};

// the atom's energy from its densities rho[nd] and core repulsion sum rho_core; dF[nd]: dE/drho. fs: the element's
// fs_parameters [2 nd]; zbl: inner cutoff type zbl, then jjmin (a neighbour within the switch: dmin its distance minus
// (r_in - delta_in), delta_in of its pair)
// dF_fs (optional): dF/drho_p of the embedding alone, before the inner cutoff (the non-linear extrapolation grade of
// ML-PACE's ACEBEvaluator takes these)
GRACE_HD Embedded embed(int npoti, const double *fs, int nd, const double *rho, double *dF, double rho_core,
                        double rho_core_cut, double drho_core_cut, bool zbl, bool jjmin, double dmin, double delta_in,
                        double *dF_fs = nullptr)
{
  double evdwl = 0.0;
  for (int p = 0; p < nd; p++) {
    double F, DF;
    if (npoti == kFinnisSinclair) fexp(rho[p], fs[2 * p + 1], F, DF);
    else fexp_shifted_scaled(rho[p], fs[2 * p + 1], F, DF);
    evdwl += F * fs[2 * p];
    dF[p] = DF * fs[2 * p];
  }
  Embedded o;
  double fcut, dfcut;
  if (zbl) {
    if (jjmin) {
      cutoff_func_poly(delta_in - dmin, delta_in, delta_in, fcut, dfcut);
      dfcut = -dfcut;
    } else {
      fcut = 1;
      dfcut = 0;
    }
    o.energy = evdwl * fcut + rho_core * (1 - fcut);
    o.dF_drho_core = 1 - fcut;
    o.dF_dfcut = evdwl * dfcut - rho_core * dfcut;
  } else {
    if (rho_core >= rho_core_cut) {
      fcut = 0;
      dfcut = 0;
    } else if (rho_core <= rho_core_cut - drho_core_cut) {
      fcut = 1;
      dfcut = 0;
    } else {
      cutoff_func_poly(rho_core, rho_core_cut, drho_core_cut, fcut, dfcut);
    }
    o.energy = evdwl * fcut + rho_core;
    o.dF_drho_core = evdwl * dfcut + 1;
  }
  if (dF_fs)
    for (int p = 0; p < nd; p++) dF_fs[p] = dF[p];
  for (int p = 0; p < nd; p++) dF[p] *= fcut;
  return o;
}

}    // namespace pace
}    // namespace grace

#endif

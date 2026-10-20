// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Bond gathering of the fused CPU drivers, one specialisation per bond source. The LammpsBonds version is
// the former gather_bonds of the LAMMPS styles with the same statements in the same order (the LAMMPS
// results must stay bit-identical); the CsrBonds version differs only in where j and r_ij come from.
#pragma once

#include "grace/fused_cpu.h"
#include "grace/system.h"

#include <algorithm>
#include <cmath>

namespace grace {

template<class Src> struct BondGather;

namespace detail {
// spherical harmonics of the gathered bonds, 8 at a time
template<class W> void bond_sh(W &w, int nb, const fused::SHCoefs &sh, bool grad)
{
  for (int b0 = 0; b0 < nb; b0 += 8) {
    const int n = std::min(8, nb - b0);
    const double *u = w.rh.data() + 3 * b0;
    if (grad)
      fused::sh_block<true>(sh, n, u, u + 1, u + 2, w.Y.data() + (size_t) b0 * sh.nlm,
                            w.G.data() + (size_t) b0 * sh.nlm * 3);
    else
      fused::sh_block<false>(sh, n, u, u + 1, u + 2, w.Y.data() + (size_t) b0 * sh.nlm, nullptr);
  }
}
template<class W> void bond_reserve(W &w, int ntot, int nlm)
{
  if ((int) w.bj.size() < ntot) {
    w.bj.resize(ntot); w.bt.resize(ntot);
    w.rh.resize(3 * ntot); w.rr.resize(ntot); w.rc.resize(ntot);
    w.Y.resize((size_t) ntot * nlm); w.G.resize((size_t) ntot * nlm * 3);
  }
}
// two-group coupling (grace/coupling.h): r' of bond nb (>= rc: dropped); true r, dr'/dr, dr'/dlambda into w.cp
template<class W> double bond_couple(W &w, const Coupling &c, int i, int j, double r, double rc, int nb)
{
  double rp = r, drp = 1.0, dlp = 0.0;
  if (coupling_cross(c.mask[i], c.mask[j], c.bit1, c.bit2)) coupling_shift(c.eq, r, rc, c.lambda, rp, drp, dlp);
  w.cp[3 * nb] = r; w.cp[3 * nb + 1] = drp; w.cp[3 * nb + 2] = dlp;
  return rp;
}
}    // namespace detail

// two-group coupling: the model's bond gradient fv (at r' = rm) -> the gradient in r_ij, fv = J^T fv with
// J = dr'/dr n n^T + r'/r (1 - n n^T); dE/dlambda into dedl; returns the true r (for the virial)
inline double couple_force(const double *cp, double rm, const double *rh, double *fv, double &dedl)
{
  const double fn = fv[0] * rh[0] + fv[1] * rh[1] + fv[2] * rh[2], s = rm / cp[0], c = (cp[1] - s) * fn;
  dedl += cp[2] * fn;
  for (int d = 0; d < 3; d++) fv[d] = s * fv[d] + c * rh[d];
  return cp[0];
}

// bonds of the batch atoms w.atom[0 .. nbat): w.boff, bj, bt, rh, rr, rc, Y (and G if grad)
template<> struct BondGather<LammpsBonds> {
  template<class W>
  static void run(W &w, const LammpsBonds &s, const BondFilter &f, const fused::SHCoefs &sh, bool grad)
  {
    int ntot = 0;
    for (int a = 0; a < w.nbat; a++) ntot += s.numneigh[w.atom[a]];
    detail::bond_reserve(w, ntot, sh.nlm);
    if (f.cpl && (int) w.cp.size() < 3 * ntot) w.cp.resize(3 * ntot);
    int nb = 0;
    for (int a = 0; a < w.nbat; a++) {
      w.boff[a] = nb;
      const int i = w.atom[a];
      const int itype = s.type[i];
      const int *jl = s.firstneigh[i];
      const int jnum = s.numneigh[i];
      const double xi = s.x[i][0], yi = s.x[i][1], zi = s.x[i][2];
      for (int jj = 0; jj < jnum; jj++) {
        const int j = jl[jj] & s.neighmask;
        const double dx = s.x[j][0] - xi, dy = s.x[j][1] - yi, dz = s.x[j][2] - zi;
        const double rsq = dx * dx + dy * dy + dz * dz;
        const int jt = s.type[j];
        if (rsq >= f.cutsq[itype][jt]) continue;
        const double r = std::sqrt(rsq), ir = 1.0 / r;
        double rm = r;    // the model's bond length (coupling: r')
        if (f.cpl) {
          rm = detail::bond_couple(w, *f.cpl, i, j, r, f.rcut_type[itype * f.ntp + jt], nb);
          if (rm >= f.rcut_type[itype * f.ntp + jt]) continue;
        }
        w.bj[nb] = j;
        w.bt[nb] = jt;
        w.rh[3 * nb] = dx * ir; w.rh[3 * nb + 1] = dy * ir; w.rh[3 * nb + 2] = dz * ir;
        w.rr[nb] = rm;
        w.rc[nb] = f.rcut_type[itype * f.ntp + jt];
        nb++;
      }
    }
    w.boff[w.nbat] = nb;
    detail::bond_sh(w, nb, sh, grad);
  }
};

template<> struct BondGather<CsrBonds> {
  template<class W>
  static void run(W &w, const CsrBonds &s, const BondFilter &f, const fused::SHCoefs &sh, bool grad)
  {
    int ntot = 0;
    for (int a = 0; a < w.nbat; a++) ntot += (int) (s.offset[w.atom[a] + 1] - s.offset[w.atom[a]]);
    detail::bond_reserve(w, ntot, sh.nlm);
    if (f.cpl && (int) w.cp.size() < 3 * ntot) w.cp.resize(3 * ntot);
    int nb = 0;
    for (int a = 0; a < w.nbat; a++) {
      w.boff[a] = nb;
      const int i = w.atom[a];
      const int itype = s.type[i];
      for (int64_t q = s.offset[i]; q < s.offset[i + 1]; q++) {
        const int j = s.j[q];
        const double dx = s.rij[3 * q], dy = s.rij[3 * q + 1], dz = s.rij[3 * q + 2];
        const double rsq = dx * dx + dy * dy + dz * dz;
        const int jt = s.type[j];
        if (rsq >= f.cutsq[itype][jt]) continue;
        const double r = std::sqrt(rsq), ir = 1.0 / r;
        double rm = r;    // the model's bond length (coupling: r')
        if (f.cpl) {
          rm = detail::bond_couple(w, *f.cpl, i, j, r, f.rcut_type[itype * f.ntp + jt], nb);
          if (rm >= f.rcut_type[itype * f.ntp + jt]) continue;
        }
        w.bj[nb] = j;
        w.bt[nb] = jt;
        w.rh[3 * nb] = dx * ir; w.rh[3 * nb + 1] = dy * ir; w.rh[3 * nb + 2] = dz * ir;
        w.rr[nb] = rm;
        w.rc[nb] = f.rcut_type[itype * f.ntp + jt];
        nb++;
      }
    }
    w.boff[w.nbat] = nb;
    detail::bond_sh(w, nb, sh, grad);
  }
};

}    // namespace grace

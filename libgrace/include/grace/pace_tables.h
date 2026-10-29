// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The evaluation tables of a PACE model, built once on the host and used by the CPU and the GPU engines: the
// C-tilde products as flat indices into the per-atom A-basis, the rank-1 terms folded into per-pair radial functions
// h_p, and the cubic radial tables.
#ifndef GRACE_PACE_TABLES_H
#define GRACE_PACE_TABLES_H

#include "grace/pace_model.h"
#include "grace/pace_radial.h"

#include <cstddef>
#include <vector>

namespace grace {

// the radial tables of a pair (fractions of its cutoff): below kPaceExactBelow the functions are evaluated directly
// (the core repulsion grows as 1 / r), on [kPaceExactBelow, kPaceTableSplit) from a table of spline / 2 intervals,
// above from one of spline intervals. One uniform table over [0, rcut] of 1024 intervals missed the fp32 gate by
// 1e-4 .. 1e-1 eV/A at close contacts; the short-range segment is finer than that and more accurate than the
// long-range one (O-H, H-H pairs of water are this close)
constexpr double kPaceExactBelow = 0.1, kPaceTableSplit = 0.25;

struct PaceTables {
  int nel = 0, nr = 1, lmax = 0, nlm = 1;    // largest nradmax, lmax over the pairs
  size_t na = 0;                             // complex A entries per atom (all elements, n, l, m)
  struct Elem {
    int nd = 0;
    std::vector<int> off;             // rank > 1: factors of row k at off[k] .. off[k + 1]
    std::vector<int> fidx;            // A index of the factor
    // dE/dA is needed at m >= 0 only (A_{l,-m} = (-1)^m conj A_{l,m}): the factor's gradient dB goes to ftgt as dB
    // (m > 0), (-1)^m conj(dB) (m < 0) or 2 Re dB (m = 0): Re += fre Re dB, Im += fim Im dB
    std::vector<int> ftgt;
    std::vector<double> fre, fim;
    std::vector<double> c;            // [rows][nd]
    // extrapolation grade (PaceModel::has_uq): the B functions of each row k (CSR), and the rank-1 rows (folded into
    // h_p above) apart: their value is sum_j g_n(r_ij) over the neighbours of element r1_mu, n - 1 = r1_n
    std::vector<int> uq_off{0}, uq_func;
    std::vector<double> uq_cg;
    std::vector<int> r1_mu, r1_n, r1_off{0}, r1_func;
    std::vector<double> r1_cg;
  };
  std::vector<Elem> el;
  std::vector<pace::RadialSpline> spl;    // [mui][muj], empty without tables
  int nf = 1;                             // largest radial block (h_p, R_nl, core repulsion) over the pairs
  std::vector<std::vector<double>> hc;    // [mui][muj]: rank-1 C-tilde coefficients [ndensity(mui)][nradbase]
  // the associated Legendre recurrences of the spherical harmonics (ACECartesianSphericalHarmonics), l <= lmax:
  // P_lm = alm (z P_l-1,m + blm P_l-2,m) (m < l - 1), P_l,l-1 = dl z P_l-1,l-1, P_ll = cl P_l-1,l-1; [l (lmax + 1) + m]
  std::vector<double> alm, blm, cl, dl;
  size_t a(int mu, int n0, int l, int m) const { return ((size_t) mu * nr + n0) * nlm + l * l + l + m; }

  // spline: intervals of the cubic radial tables per pair (0: none, the radial functions evaluated directly)
  PaceTables(const PaceModel &m, int spline);
};

}    // namespace grace

#endif

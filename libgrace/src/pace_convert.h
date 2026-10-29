// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// A pacemaker B-basis .yaml (and its active set .asi) as the arrays of the PACE npz (grace/pace_model.h): the reader
// of ML-PACE's format (ace_yaml_input.cpp, ace_b_basis.cpp), the B -> C-tilde conversion (ace_couplings.cpp,
// to_ACECTildeBasisSet) and the order of the B functions of the extrapolation grade (ace_abstract_basis.cpp compare).
// The values are those of the Python converter it replaced, bit for bit: the Clebsch-Gordan coefficients exactly
// (Racah's sum over integers, rounded once), every sum in the same order.
#ifndef GRACE_PACE_CONVERT_H
#define GRACE_PACE_CONVERT_H

#include "grace_cnpy/cnpy.h"

#include <string>
#include <vector>

namespace grace::pace {

// <j1 m1 j2 m2 | J M> (Condon-Shortley phase); exactly 0.0 when it is zero
double clebsch_gordan(int j1, int m1, int j2, int m2, int J, int M);

// the m-combinations of the rotational invariant (L = 0) of the coupling tree of (ls, lint), with their generalized
// Clebsch-Gordan coefficients, half basis (combinations whose first non-zero m is negative dropped, positive doubled)
struct MsList {
  int rank = 0;
  std::vector<int> ms;       // [size()][rank]
  std::vector<double> cg;    // [size()]
  size_t size() const { return cg.size(); }
};
MsList ms_cg_list(const std::vector<int> &ls, const std::vector<int> &lint);

// the PACE npz arrays of a B-basis yaml; asi: its active set (pacemaker .asi), or empty. grace::Error names the
// key (or the line) for anything the engine cannot evaluate. Flat int32 / float64 / byte-string arrays, scalars of
// shape (1,):
//   pace_format_version, element_names (|S), element_z (atomic numbers, 0: unknown), inner_cutoff_type (0 density,
//   1 distance, 2 zbl), bond_defined [nel, nel] (0: the pair does not interact)
//   e<mu>_  ndensity, npoti (0 FinnisSinclair, 1 FinnisSinclairShiftedScaled), fs_parameters [2 ndensity],
//           rho_core_cut, drho_core_cut, and the C-tilde products: rank [rows], mus / ns / ls / ms [rows, rankmax]
//           (padded with 0), ctildes [rows, ndensity]
//   b<i>_<j>_  radbase (0 ChebExpCos, 1 ChebPow, 2 ChebLinear, 3 SBessel), radparameters, crad [nradmax, lmax+1,
//           nradbase], nradmax, lmax, nradbase, rcut, dcut, r_in, delta_in, prehc, lambdahc   (defined pairs only)
// With an active set: uq_grade (0 linear, 1 non-linear) and per element
//   e<mu>_  uq_nfunc (B functions), uq_off [rows + 1] / uq_func / uq_cg (the B functions of each C-tilde row: index in
//           ML-PACE's order and generalized Clebsch-Gordan coefficient), asi [n, n] (the active set inverse, n =
//           uq_nfunc or uq_nfunc * ndensity)
cnpy::npz_t npz_from_yaml(const std::string &yaml_path, const std::string &asi_path = "");

}    // namespace grace::pace

#endif

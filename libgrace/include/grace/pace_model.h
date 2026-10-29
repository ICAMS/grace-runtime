// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// A PACE (linear / nonlinear ACE) model: a pacemaker B-basis .yaml (converted on load, src/pace_convert.h) or the PACE
// npz of those arrays (their layout: src/pace_convert.h): per element the embedding and the C-tilde products, per
// interacting element pair the radial basis.
#ifndef GRACE_PACE_MODEL_H
#define GRACE_PACE_MODEL_H

#include <string>
#include <vector>

namespace grace {

constexpr int kPaceFormatVersion = 1;
constexpr int kPaceMaxElements = 128;
constexpr int kPaceMaxNradbase = 64;    // radial basis functions per pair (Chebyshev / Bessel recursion buffers)
constexpr int kPaceMaxLmax = 9;
constexpr int kPaceMaxRank = 6;
constexpr int kPaceMaxDensity = 64;

enum PaceRadbase { kChebExpCos = 0, kChebPow = 1, kChebLinear = 2, kSBessel = 3 };
enum PaceInnerCutoff { kInnerDensity = 0, kInnerDistance = 1, kInnerZbl = 2 };
enum PaceEmbedding { kFinnisSinclair = 0, kFinnisSinclairShiftedScaled = 1 };

struct PaceBond {
  bool defined = false;    // false: the pair does not interact
  int radbase = 0, nradmax = 0, lmax = 0, nradbase = 0;
  double lambda = 0.0;     // radparameters[0] (ChebExpCos, ChebPow)
  double rcut = 0.0, dcut = 0.0, r_in = 0.0, delta_in = 0.0, prehc = 0.0, lambdahc = 0.0;
  int zi = 0, zj = 0;      // atomic numbers (ZBL)
  std::vector<double> crad;    // [nradmax][lmax + 1][nradbase]
};

struct PaceElement {
  int ndensity = 0, npoti = 0;
  std::vector<double> fs_parameters;    // [2 ndensity]
  double rho_core_cut = 0.0, drho_core_cut = 0.0;
  int nrows = 0, rankmax = 0;
  std::vector<int> rank;                // [nrows]
  std::vector<int> mus, ns, ls, ms;     // [nrows][rankmax]
  std::vector<double> ctildes;          // [nrows][ndensity]
  // extrapolation grade (PaceModel::has_uq): the B functions (ML-PACE's order) of each row, the active set inverse
  int uq_nfunc = 0;
  std::vector<int> uq_off, uq_func;     // [nrows + 1], [nnz]
  std::vector<double> uq_cg;            // [nnz]
  std::vector<double> asi;              // [n][n], n = uq_nfunc (linear) or uq_nfunc * ndensity (non-linear)
};

struct PaceModel {
  std::vector<std::string> element_names;
  std::vector<int> element_z;
  int n_elements = 0;
  int inner_cutoff_type = kInnerDensity;
  bool has_uq = false;    // an active set (.asi, or in the npz): the extrapolation grade gamma
  int uq_grade = 0;       // 0 linear (projections), 1 non-linear (dF/drho_p times the projections, [p][function])
  std::vector<PaceElement> elements;
  std::vector<PaceBond> bonds;    // [n_elements][n_elements]

  // path: a .yaml / .yml (asi: its active set, pacemaker .asi, or empty) or a PACE npz (asi empty: the npz
  // carries it); grace::Error for a model that is not valid
  void load(const std::string &path, const std::string &asi = "");
  const PaceBond &bond(int mui, int muj) const { return bonds[(size_t) mui * n_elements + muj]; }
  double cutoff(int mui, int muj) const { const PaceBond &b = bond(mui, muj); return b.defined ? b.rcut : 0.0; }
};

// a pacemaker B-basis file by its suffix (.yaml / .yml), which PaceModel::load converts; anything else is read as npz
bool is_pace_yaml(const std::string &path);

}    // namespace grace

#endif

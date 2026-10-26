// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from LAMMPS src/ML-PACE (lammps-ace, feature/grace-fused); contributing author: Yury Lysogorskiy (ICAMS)
#ifndef GRACE_MODEL_1L_H
#define GRACE_MODEL_1L_H

#include "grace/error.h"
#include "grace/uq.h"

#include <map>
#include <string>
#include <vector>

namespace grace {
namespace cnpy {
struct NpyArray;
}


// ============================================================================
// GRACE1LModel: Host-side weight container loaded from .npz file
// ============================================================================
struct GRACE1LModel {
  // Architecture metadata
  int n_elements = 0;
  int embedding_size = 0;
  int lmax = 0;
  int nradbase = 0;
  int nradmax = 0;
  int radial_basis_p = 0;
  double rcut = 0.0;          // max cutoff (or single cutoff if uniform)
  bool has_bond_specific_cutoff = false;
  std::vector<double> bond_cutoff_map;  // [n_elements * n_elements] per-pair cutoffs

  // MLP radial function (of the bond length only: the loader rejects a chem-embedded one)
  int mlp_rad_n_layers = 0;

  // Chemical embedding [n_elements, embedding_size]
  std::vector<double> chem_embedding;

  // A indicator transform [embedding_size, nradmax]
  std::vector<double> A_lin_transform_W;
  double A_lin_transform_norm = 1.0;
  double A_inv_avg_n_neigh = 1.0;

  // MLP radial weights: layers[i] has W[n_in, n_out] and norm
  struct MLPLayer {
    std::vector<double> W;
    double norm = 1.0;
    int n_in = 0, n_out = 0;
  };
  std::vector<MLPLayer> mlp_rad_layers;

  // FC weights (A1, AA1, AA2); left weights in every FC (left_coefs = 1, checked by the loader)
  struct FCWeights {
    std::vector<double> w_left, w_right;
    std::vector<int> w_tile_left, w_tile_right;
    std::vector<int> collect_to, collect_from;
    std::vector<double> norm_out_factor;
    double norm_left = 1.0, norm_right = 1.0;
    int n_out = 0;
    int n_funcs_left = 0, n_funcs_right = 0;
    int w_shape_left = 0, w_shape_right = 0;
  };
  FCWeights fc_A1, fc_AA1, fc_AA2;

  // Product CG metadata (AA, AAA, AAAA)
  struct CGProduct {
    std::vector<int> left_ind, right_ind, m_sum_ind;
    std::vector<double> cg_coeff;
    int n_output_funcs = 0;
    int n_cg_terms = 0;
  };
  CGProduct prod_AA, prod_AAA, prod_AAAA;

  // FunctionReduceN (rho; weights per central element, checked by the loader)
  struct ReduceWeights {
    std::vector<double> W;  // [n_elements, n_out, n_in, w_shape]
    double norm = 1.0;
    int n_in = 0, w_shape = 0;
    std::vector<int> collect_ind;
    std::vector<int> w_l_tile;
    std::vector<int> total_sum_ind;
  };
  int rho_n_out = 0;
  ReduceWeights reduce_A, reduce_AA, reduce_AAA, reduce_AAAA;

  // Energy MLP
  std::vector<MLPLayer> energy_mlp_layers;
  int energy_mlp_activation = 0;  // 0=silu, 1=tanh

  // Per-element shifts
  std::vector<double> shift_values;

  // ConstantScaleShiftTarget output scale; 1.0 if absent from npz
  double output_scale = 1.0;

  // Element names (for type mapping)
  std::vector<std::string> element_names;

  // ---- UQ / extrapolation-grade artifacts (optional; present iff has_uq) ----
  // Schema v6 (basis-RP, normalize+density): the GMM feature phi is built from the
  // invariant (l=0) energy-path B-basis as proj = (B/||B||) * R (L2-normalized) with
  // n_density log-norm density channels appended: phi = concat(proj, dens). R is
  // stored verbatim; gamma is the only UQ signal (no force-error model).
  bool has_uq = false;
  int uq_schema_version = 0;
  std::string uq_error;           // why the UQ arrays are unusable (checked by validate_uq)
  int uq_n_elements = 0;
  int uq_max_clusters = 0;            // Kmax
  int uq_feature_dim = 0;             // D = rp_dim + n_density (full GMM feature dim)
  int uq_rp_dim = 0;                  // random-projection output dim (R cols; e.g. 128, read from artifact, not enforced)
  int uq_n_density = 0;              // density channels appended after proj (D - rp_dim)
  int uq_normalize = 0;             // 1 -> L2-normalize basis before projection (v6)
  double uq_density_scale = 1.0;     // density_scale (=1.0 for v6)
  int uq_d_basis = 0;                 // concatenated invariant basis width (rows of R)
  std::vector<double> uq_centroids;          // [E*Kmax*D]
  std::vector<double> uq_inv_cov;            // [E*Kmax*D*D] (pre-inverted precision)
  std::vector<int>    uq_n_clusters;         // [E]
  std::vector<double> uq_interp_thresholds;  // [E*Kmax]
  std::vector<double> uq_rp_matrix;          // [D_basis*rp_dim] row-major projection R

  void load(const std::string &filepath);
  // load() of an archive already read (key -> array), e.g. a modified copy in a test
  void load_arrays(const std::map<std::string, cnpy::NpyArray> &npz);

  // throws grace::Error unless the model carries a usable schema-6 UQ head

  void validate_uq() const;

  // the basis blocks of the UQ projection (grace/uq.h), from the reduce weights
  std::vector<UqBlock> uq_layout() const;
  std::vector<std::string> rho_instruction_names;   // npz rho_instruction_names (empty if absent)
};

}    // namespace grace

#endif

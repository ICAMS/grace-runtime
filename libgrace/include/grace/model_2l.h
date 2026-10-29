// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Contributing author: Yury Lysogorskiy (ICAMS)
#ifndef GRACE_MODEL_2L_H
#define GRACE_MODEL_2L_H

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
// GRACE2LModel: Host-side weight container loaded from .npz file
// ============================================================================
struct GRACE2LModel {
  // Architecture metadata
  int n_elements = 0;
  int embedding_size = 0;
  int lmax = 0;
  int nradbase = 0;
  int radial_basis_p = 0;
  double rcut = 0.0;
  bool has_bond_specific_cutoff = false;
  std::vector<double> bond_cutoff_map;  // [n_elements * n_elements]

  // Chemical embedding [n_elements, embedding_size]
  std::vector<double> chem_embedding;

  // MLP layer struct (reused for all MLPs)
  struct MLPLayer {
    std::vector<double> W;
    double norm = 1.0;
    int n_in = 0, n_out = 0;
  };

  // ---- Layer 1 ----
  int L1_nradmax = 42;
  int L1_mlp_rad_n_layers = 0;    // radial MLPs of the bond length only (the loader rejects chem-embedded ones)
  std::vector<MLPLayer> L1_mlp_rad_layers;  // R

  // A indicator transform
  std::vector<double> A_lin_transform_W;  // [embedding_size, L1_nradmax]
  double A_lin_transform_norm = 1.0;
  double A_inv_avg_n_neigh = 1.0;

  // FC weights; left weights in every FC (left_coefs = 1, checked by the loader)
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
  int L1_fc_n_out = 42;

  // CG products
  struct CGProduct {
    std::vector<int> left_ind, right_ind, m_sum_ind;
    std::vector<double> cg_coeff;
    int n_output_funcs = 0;
    int n_cg_terms = 0;
  };
  CGProduct prod_AA, prod_AAA, prod_AAAA;

  // Generalized ReduceN weights (supports equivariant output)
  struct ReduceWeights {
    std::vector<double> W;  // element-dep: [n_elem, n_out, n_in, w_shape]; non-elem: [n_out, n_in, w_shape]
    double norm = 1.0;
    int n_in = 0, w_shape = 0;
    int n_connections = 0;         // len(collect_ind) — may differ from w_shape for equivariant
    std::vector<int> collect_ind;  // [n_connections] input function indices
    std::vector<int> w_l_tile;     // [n_connections] tile index into W's last dim
    std::vector<int> total_sum_ind; // [n_connections] output function index
  };

  // I1 reducer (equivariant, element-dependent)
  ReduceWeights I1_reduce_A, I1_reduce_AA, I1_reduce_AAA, I1_reduce_AAAA;
  int I1_n_out = 12;
  int I1_n_funcs = 4;

  // I reducer (equivariant, NOT element-dependent)
  ReduceWeights I_reduce_I1;
  int I_n_out = 32;
  int I_n_funcs = 4;

  // rho reducer (scalar, element-dependent) — same structure as 1L
  ReduceWeights rho_reduce_A, rho_reduce_AA, rho_reduce_AAA, rho_reduce_AAAA;
  int rho_n_out = 17;

  // RMSNorm: I_nl_LN (only_nonlin: scale has n_out-1 elements, rho[0] passed through)
  std::vector<double> I_nl_LN_scale;  // [rho_n_out - 1] = [16]
  int I_nl_LN_type = 1;  // 1 = only_nonlin
  int I_nl_LN_n_out = 17;

  // ---- Layer 2 ----
  int L2_nradmax = 32;
  int L2_mlp_rad_n_layers = 0;
  std::vector<MLPLayer> L2_mlp_rad_layers;  // R1

  // B0 indicator transform
  std::vector<double> B0_lin_transform_W;  // [embedding_size, L2_nradmax]
  double B0_lin_transform_norm = 1.0;
  double B0_inv_avg_n_neigh = 1.0;

  // YI equivariant indicator CG coupling
  struct YICoupling {
    std::vector<int> lr_inds;     // [n_cg_terms * 2] flattened (y_idx, I_idx pairs)
    std::vector<int> m_sum_ind;   // [n_cg_terms] output function index
    std::vector<double> cg_coeff; // [n_cg_terms]
    int n_output_funcs = 0;
    int n_cg_terms = 0;
    int n_rad_max = 0;
    double inv_avg_n_neigh = 1.0;
  } yi;

  // B reducer (equivariant, NOT element-dependent)
  ReduceWeights B_reduce_YI, B_reduce_B0;
  int B_n_out = 64;
  int B_n_funcs = 31;

  // Layer 2 FC/Products
  FCWeights fc_B1, fc_BB1, fc_BB2;
  int L2_fc_n_out = 64;
  CGProduct prod_BB, prod_BBB, prod_BBBB;

  // I2 reducer (scalar, element-dependent)
  ReduceWeights I2_reduce_B, I2_reduce_BB, I2_reduce_BBB, I2_reduce_BBBB;
  int I2_n_out = 17;

  // RMSNorm: I_0_LN (full: scale has n_out elements, all scaled)
  std::vector<double> I_0_LN_scale;  // [I2_n_out] = [17]
  int I_0_LN_type = 0;  // 0 = full
  int I_0_LN_n_out = 17;

  // Energy MLP (input=16, tanh activation)
  std::vector<MLPLayer> energy_mlp_layers;
  int energy_mlp_activation = 1;  // 0=silu, 1=tanh

  // Per-element shifts
  std::vector<double> shift_values;

  // ConstantScaleShiftTarget output scale; 1.0 if absent from npz
  double output_scale = 1.0;

  // Element names
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
  std::vector<std::string> I2_instruction_names;   // npz I2_instruction_names (empty if absent)
  std::vector<std::string> rho_instruction_names;   // npz rho_instruction_names (empty if absent)
};

}    // namespace grace

#endif

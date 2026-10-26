// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from lammps-ace src/ML-PACE/grace_3l_model.h (feature/grace-runtime a73b8b17ee); contributing author:
// Yury Lysogorskiy (ICAMS). GRACE-3L model container (kokkos.npz weights), shared by grace/3l/fused/kk and grace/3l/fused.
#ifndef GRACE_MODEL_3L_H
#define GRACE_MODEL_3L_H

#include "grace/uq_head.h"

#include <map>
#include <string>
#include <vector>

namespace grace {
namespace cnpy {
struct NpyArray;
}


// ============================================================================
// GRACE3LModel: Host-side weight container loaded from .npz file.
//
// Holds EVERY weight tensor grace_3l_weights.npz carries (exporter
// kokkos_export_3l.py), stored RAW (float64 on host) exactly as written.
// copy_weights_to_device() casts these into the NNScalar/GeomScalar device
// views. The forward/backward kernels (Tasks 2.3-3.4) consume the device
// layout; this struct is the CPU staging area + the name/index directory.
//
// NORM CONVENTION (PINNED, applied exactly ONCE downstream): every Linear-like
// weight is stored RAW together with a SEPARATE scalar norm = 1/sqrt(n_in).
// The consumer forms w*norm (and b*norm) once. The ONLY place this file folds
// the norm early is the per-element species-modulation precompute
// (SPBF::z_tr / z_proj built in copy_weights_to_device): there chem_embed is
// contracted with the (lin_transform / chem_linear) weight and the norm folded
// in once — mirroring GRACE-2L's d_z_tr_A. Everywhere else weights go to the
// device raw and the kernel multiplies the stored norm scalar once.
// ============================================================================
struct GRACE3LModel : UqHead {
  // ---- Architecture metadata ----
  int n_elements = 0;
  int embedding_size = 0;
  int lmax = 0;                 // angular lmax (== SPBF _lmax; no bare npz key -> from A1_lmax)
  int nradbase = 0;             // Chebyshev basis count (== SPBF n_rad_basis)
  int radial_basis_p = 0;       // envelope polynomial order used by geometry (== SPBF p)
  double rcut = 0.0;
  bool has_bond_specific_cutoff = false;
  std::vector<double> bond_cutoff_map;   // [n_elements * n_elements]

  // Per-layer radial ceilings (3 layers). Set from spbf[k].n_rad_max.
  int L1_nradmax = 0;
  int L2_nradmax = 0;
  int L3_nradmax = 0;

  // Chemical embedding [n_elements, embedding_size]
  std::vector<double> chem_embedding;

  // Per-element energy shifts and global output scale
  std::vector<double> shift_values;      // [n_elements]
  double output_scale = 1.0;             // ConstantScaleShiftTarget; 1.0 if absent

  // Element names (model index -> chemical symbol)
  std::vector<std::string> element_names;

  // UQ head (has_uq, uq_*): GraceUQHead. The invariant basis is that of the three per-layer scalar reduces
  // rho1 | rho2 | rho3, so uq_n_density = 4 ([full, rho1, rho2, rho3]); forward only.

  // ---- Generic MLP (radial MLP per SPBF; energy readout MLP) ----
  // Each layer stored raw: W flat [n_in*n_out] row-major, optional bias b
  // flat [n_out] (hidden layers only), scalar norm. silu on hidden layers,
  // linear (no bias/activation) on the output layer.
  struct MLP {
    int n_layers = 0;
    bool has_bias = false;
    std::vector<std::vector<double>> W;   // [n_layers][n_in*n_out]
    std::vector<std::vector<double>> b;   // [n_layers][n_out] (empty entry = no bias)
    std::vector<double> norm;             // [n_layers]
    std::vector<int> n_in;                // [n_layers]
    std::vector<int> n_out;               // [n_layers]
  };

  // ---- SPBF (Single Particle Basis Function) per layer: A1 (scalar), A2/A3 (equivariant) ----
  struct SPBF {
    std::string name;
    bool equivariant = false;
    int n_rad_max = 0, n_rad_basis = 0, lmax = 0, Lmax = 0, p = 0;
    double rcut = 0.0, inv_avg_n_neigh = 1.0;
    std::vector<int> l_tile;              // [nlm] radial-channel -> lm tile
    MLP mlp;                              // radial MLP (bias on hidden layers)
    // scalar mode (A1): species modulation z_tr = lin_transform(Z)
    std::vector<double> lin_transform_W;  // [embedding_size, n_rad_max]
    double lin_transform_norm = 1.0;
    // equivariant mode (A2/A3): L=0 chemistry injection + dense CG couple
    std::vector<double> chem_linear_W;    // [embedding_size, n_rad_max]
    double chem_linear_norm = 1.0;
    std::vector<double> chem_l0_mask;     // [n_lm_ind]
    std::vector<double> cg_W;             // [(lmax+1)^2 * n_lm_ind, nfunc]
    int nfunc = 0, n_lm_ind = 0;
  };

  // ---- GeneralProductFunction, cp_l mode (use_S=False => rank == n_out) ----
  struct CPL {
    std::string name;
    int rank = 0, nfunc = 0, n_left = 0, n_right = 0;
    int n_groups_left = 0, n_groups_right = 0, n_cg = 0;
    double norm_u = 1.0, norm_v = 1.0;
    std::vector<double> U;                // [n_groups_left, rank, n_left]
    std::vector<double> V;                // [n_groups_right, rank, n_right]
    std::vector<double> cg;               // [n_cg]
    std::vector<int> group_left, group_right;   // [n_lm_left], [n_lm_right]
    std::vector<int> left_ind, right_ind, m_sum_ind;  // [n_cg]
    std::vector<int> out_l;               // [nfunc]
    std::vector<int> newcol;              // [nfunc] file function -> output column (group_reduce_inputs), or empty
  };

  // ---- FunctionReduceN (metadata driven; per-instruction collectors) ----
  struct ReduceInstr {
    std::string name;
    std::vector<double> W;                // flat: elem_dep [n_types,n_out,n_in,w_shape] else [n_out,n_in,w_shape]
    double norm = 1.0;
    int n_in = 0, w_shape = 0, n_conn = 0;
    std::vector<int> collect_ind;         // [n_conn]
    std::vector<int> w_l_tile;            // [n_conn]
    std::vector<int> total_sum_ind;       // [n_conn]
  };
  struct Reduce {
    std::string name;
    int n_out = 0, n_funcs = 0;
    bool only_invar = false, elem_dep = false;
    std::vector<ReduceInstr> instr;       // 1 (product reduces) or 4 (eq*/rho*)
    std::vector<double> norm_map;         // [n_funcs] (or [1] for only_invar)
    bool has_norm_map = false;
  };

  // ---- FCRight2Left (metadata driven; 3L FCs are right->left, left_coefs=False) ----
  struct FC {
    std::string name;
    int n_out = 0;
    bool left_coefs = false;
    int n_funcs_left = 0, n_funcs_right = 0;
    int n_in_left = 0, n_in_right = 0, w_shape_left = 0, w_shape_right = 0;
    double norm_left = 1.0, norm_right = 1.0;
    std::vector<double> w_left;           // [n_out, n_in_left, w_shape_left] (empty if !left_coefs)
    std::vector<double> w_right;          // [n_out, n_in_right, w_shape_right]
    std::vector<double> norm_out_factor;  // [n_funcs_left]
    std::vector<int> w_tile_left, w_tile_right;    // [n_funcs_left], [n_funcs_right]
    std::vector<int> collect_to, collect_from;     // [n_funcs_left], [n_funcs_right]
  };

  // ---- EquivariantRMSNorm (eq1_norm, eq2_norm) ----
  struct EqNorm {
    std::string name;
    int center_l0 = 0, n_groups = 0, n_out = 0, M = 0;
    double eps = 0.0;
    std::vector<double> affine_weight;    // [n_groups, n_out]
    std::vector<double> degree_weights;   // [M]
    std::vector<double> l0_mask;          // [M]
    std::vector<int> expand_index;        // [M]
  };

  // ---- InvariantLayerRMSNorm (rho1/2/3_norm) ----
  struct RMSNorm {
    std::string name;
    int type = 0, n_out = 0;              // type: 0=full (scale len n_out), 1=only_nonlin (len n_out-1)
    std::vector<double> scale;            // [n_out] or [n_out-1]
  };

  // ---- Ordered inventories (order == the *_names index lists in the npz) ----
  std::vector<SPBF> spbf;                 // [A1, A2, A3]
  std::vector<CPL> prod;                  // [A1_2, A1_3, A1_4, A2_2, A2_3, A2_4, A3_2, A3_3, A3_4]
  std::vector<Reduce> reduce;             // 16 reduces (product reduces + eq*/rho*)
  std::vector<FC> fc;                     // [A1_2a, A1_2b, A2_2a, A2_2b, A3_2a, A3_2b]
  std::vector<EqNorm> eqnorm;             // [eq1_norm, eq2_norm]
  std::vector<RMSNorm> rmsnorm;           // [rho1_norm, rho2_norm, rho3_norm]

  // ---- Readout ----
  MLP energy_mlp;                         // (31->64) silu, (64->1) linear, no bias
  int energy_mlp_activation = 0;          // 0=silu, 1=tanh
  std::vector<std::string> out1_origins;  // [rho1_norm, rho2_norm, rho3_norm]
  int energy_mlp_origin_n_out = 0;        // per-origin density width (32)
  int energy_mlp_n_origins = 0;           // 3

  // The Kokkos engine (grace::kk::Grace3LKK) reorders the producer columns of the single-input element-independent
  // reduces once per model (a permutation that leaves the model unchanged as a function); set when done, cleared by
  // load().
  bool reduce_inputs_grouped = false;

  void load(const std::string &filepath);
  // load() of an archive already read (key -> array), e.g. a modified copy in a test
  void load_arrays(const std::map<std::string, cnpy::NpyArray> &npz);
};

}    // namespace grace
#endif

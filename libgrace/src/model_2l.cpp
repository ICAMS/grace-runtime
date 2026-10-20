// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from LAMMPS src/ML-PACE (lammps-ace, feature/grace-fused); contributing author: Yury Lysogorskiy (ICAMS)
#include "grace/model_2l.h"
#include "grace/error.h"

#include "npz3.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace grace {

using namespace npz;

// ======================================================================
// GRACE2LModel::load — load weights from .npz
// ======================================================================


void GRACE2LModel::load(const std::string &filepath) {
  cnpy::npz_t npz = npz_open(filepath);
  // a second load into the same object (LAMMPS pair_coeff twice) must keep nothing of the previous model: arrays
  // read only when present (UQ head, optional blocks) would otherwise survive
  *this = GRACE2LModel();

  // Metadata
  n_elements = npz_get_int_scalar(npz, "n_elements", 89);
  embedding_size = npz_get_int_scalar(npz, "embedding_size", 128);
  lmax = npz_get_int_scalar(npz, "lmax", 4);
  nradbase = npz_get_int_scalar(npz, "nradbase", 10);
  radial_basis_p = npz_get_int_scalar(npz, "radial_basis_p", 16);
  rcut = npz_get_cutoff_scalar(npz, "rcut", 6.0, "GRACE-2L/KK");
  has_bond_specific_cutoff = npz_get_int_scalar(npz, "has_bond_specific_cutoff", 0) != 0;
  bond_cutoff_map = npz_get_cutoff(npz, "bond_cutoff_map", "GRACE-2L/KK");
  if (bond_cutoff_map.empty())
    bond_cutoff_map.resize(n_elements * n_elements, rcut);

  // Chemical embedding
  chem_embedding = npz_require_double(npz, "chem_embedding");

  // Helper: load MLP radial layers with prefix
  auto load_mlp_rad = [&](const std::string &prefix, int &n_layers, bool &has_chem,
                           std::vector<MLPLayer> &layers, int &nradmax_out) {
    n_layers = npz_get_int_scalar(npz, prefix + "_n_layers", 3);
    has_chem = npz_get_int_scalar(npz, prefix + "_has_chem_emb", 0) != 0;
    nradmax_out = npz_get_int_scalar(npz, prefix + "_n_rad_max", 32);
    layers.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
      auto &layer = layers[i];
      layer.W = npz_require_double(npz, prefix + "_W" + std::to_string(i));
      layer.norm = npz_get_double_scalar(npz, prefix + "_norm" + std::to_string(i), 1.0);
      auto it = npz.find(prefix + "_W" + std::to_string(i));
      if (it != npz.end() && it->second.shape.size() == 2) {
        layer.n_in = it->second.shape[0];
        layer.n_out = it->second.shape[1];
      }
    }
  };

  // Layer 1 radial MLP (R)
  load_mlp_rad("mlp_rad_R", L1_mlp_rad_n_layers, L1_mlp_rad_has_chem_emb,
               L1_mlp_rad_layers, L1_nradmax);

  // Layer 2 radial MLP (R1)
  bool L2_has_chem = false;
  load_mlp_rad("mlp_rad_R1", L2_mlp_rad_n_layers, L2_has_chem,
               L2_mlp_rad_layers, L2_nradmax);

  // A indicator transform
  A_lin_transform_W = npz_require_double(npz, "A_lin_transform_W");
  A_lin_transform_norm = npz_get_double_scalar(npz, "A_lin_transform_norm", 1.0);
  A_inv_avg_n_neigh = npz_get_double_scalar(npz, "A_inv_avg_n_neigh", 1.0);

  // B0 indicator transform
  B0_lin_transform_W = npz_require_double(npz, "B0_lin_transform_W");
  B0_lin_transform_norm = npz_get_double_scalar(npz, "B0_lin_transform_norm", 1.0);
  B0_inv_avg_n_neigh = npz_get_double_scalar(npz, "B0_inv_avg_n_neigh", 1.0);

  // Helper: load FC weights
  auto load_fc = [&](const std::string &prefix, FCWeights &fc) {
    fc.w_left = npz_require_double(npz, prefix + "_w_left");
    fc.w_right = npz_require_double(npz, prefix + "_w_right");
    fc.w_tile_left = npz_require_int(npz, prefix + "_w_tile_left");
    fc.w_tile_right = npz_require_int(npz, prefix + "_w_tile_right");
    fc.collect_to = npz_require_int(npz, prefix + "_collect_to");
    fc.collect_from = npz_require_int(npz, prefix + "_collect_from");
    fc.norm_out_factor = npz_require_double(npz, prefix + "_norm_out_factor");
    fc.norm_left = npz_get_double_scalar(npz, prefix + "_norm_left", 1.0);
    fc.norm_right = npz_get_double_scalar(npz, prefix + "_norm_right", 1.0);
    fc.n_out = npz_get_int_scalar(npz, prefix + "_n_out", 64);
    fc.left_coefs = npz_get_int_scalar(npz, prefix + "_left_coefs", 1) != 0;
    fc.n_funcs_left = (int)fc.w_tile_left.size();
    fc.n_funcs_right = (int)fc.collect_from.size();
    auto it = npz.find(prefix + "_w_left");
    if (it != npz.end() && it->second.shape.size() == 3) {
      fc.w_shape_left = it->second.shape[2];
      fc.w_shape_right = fc.w_shape_left;
    }
    it = npz.find(prefix + "_w_right");
    if (it != npz.end() && it->second.shape.size() == 3)
      fc.w_shape_right = it->second.shape[2];
  };

  // Layer 1 FC
  load_fc("fc_A1", fc_A1);
  load_fc("fc_AA1", fc_AA1);
  load_fc("fc_AA2", fc_AA2);
  L1_fc_n_out = fc_A1.n_out;  // 42

  // Layer 2 FC
  load_fc("fc_B1", fc_B1);
  load_fc("fc_BB1", fc_BB1);
  load_fc("fc_BB2", fc_BB2);
  L2_fc_n_out = fc_B1.n_out;  // 64

  // Helper: load CG product
  auto load_prod = [&](const std::string &prefix, CGProduct &prod) {
    prod.left_ind = npz_require_int(npz, prefix + "_left_ind");
    prod.right_ind = npz_require_int(npz, prefix + "_right_ind");
    prod.m_sum_ind = npz_require_int(npz, prefix + "_m_sum_ind");
    prod.cg_coeff = npz_require_double(npz, prefix + "_cg_coeff");
    prod.n_output_funcs = npz_get_int_scalar(npz, prefix + "_n_output_funcs", 0);
    prod.n_cg_terms = npz_get_int_scalar(npz, prefix + "_n_cg_terms", 0);
  };

  // Layer 1 products
  load_prod("prod_AA", prod_AA);
  load_prod("prod_AAA", prod_AAA);
  load_prod("prod_AAAA", prod_AAAA);

  // Layer 2 products
  load_prod("prod_BB", prod_BB);
  load_prod("prod_BBB", prod_BBB);
  load_prod("prod_BBBB", prod_BBBB);

  // Helper: load generalized ReduceN (supports equivariant with total_sum_ind)
  auto load_reduce = [&](const std::string &reduce_prefix, const std::string &instr_name,
                          ReduceWeights &rw) {
    rw.W = npz_require_double(npz, reduce_prefix + "_reduce_" + instr_name + "_W");
    rw.norm = npz_get_double_scalar(npz, reduce_prefix + "_reduce_" + instr_name + "_norm", 1.0);
    rw.collect_ind = npz_require_int(npz, reduce_prefix + "_collect_ind_" + instr_name);
    rw.w_l_tile = npz_require_int(npz, reduce_prefix + "_w_l_tile_" + instr_name);
    rw.n_in = npz_get_int_scalar(npz, reduce_prefix + "_n_in_" + instr_name, 0);
    rw.w_shape = npz_get_int_scalar(npz, reduce_prefix + "_w_shape_" + instr_name, 0);
    rw.n_connections = (int)rw.collect_ind.size();
    // total_sum_ind: [n_connections, 1] — flatten to [n_connections]
    auto tsi_raw = npz_require_int(npz, reduce_prefix + "_total_sum_ind_" + instr_name);
    rw.total_sum_ind.resize(rw.n_connections);
    for (int i = 0; i < rw.n_connections; i++)
      rw.total_sum_ind[i] = tsi_raw.empty() ? 0 : tsi_raw[i];
  };

  // I1 reducer (equivariant, element-dependent)
  I1_n_out = npz_get_int_scalar(npz, "I1_n_out", 12);
  I1_n_funcs = npz_get_int_scalar(npz, "I1_n_funcs", 4);
  I1_is_elem_dependent = npz_get_int_scalar(npz, "I1_is_central_atom_type_dependent", 1) != 0;
  load_reduce("I1", "A", I1_reduce_A);
  load_reduce("I1", "AA", I1_reduce_AA);
  load_reduce("I1", "AAA", I1_reduce_AAA);
  load_reduce("I1", "AAAA", I1_reduce_AAAA);

  // I reducer (equivariant, NOT element-dependent)
  I_n_out = npz_get_int_scalar(npz, "I_n_out", 32);
  I_n_funcs = npz_get_int_scalar(npz, "I_n_funcs", 4);
  I_is_elem_dependent = npz_get_int_scalar(npz, "I_is_central_atom_type_dependent", 0) != 0;
  load_reduce("I", "I1", I_reduce_I1);

  // rho reducer (scalar, element-dependent)
  rho_n_out = npz_get_int_scalar(npz, "rho_n_out", 17);
  rho_is_elem_dependent = npz_get_int_scalar(npz, "rho_is_central_atom_type_dependent", 1) != 0;
  load_reduce("rho", "A", rho_reduce_A);
  load_reduce("rho", "AA", rho_reduce_AA);
  load_reduce("rho", "AAA", rho_reduce_AAA);
  load_reduce("rho", "AAAA", rho_reduce_AAAA);

  // I_nl_LN RMSNorm
  I_nl_LN_scale = npz_require_double(npz, "I_nl_LN_scale");
  I_nl_LN_type = npz_get_int_scalar(npz, "I_nl_LN_type", 1);
  I_nl_LN_n_out = npz_get_int_scalar(npz, "I_nl_LN_n_out", 17);

  // YI coupling
  {
    auto lr = npz_require_int(npz, "YI_lr_inds");  // [n_cg * 2] flattened
    yi.lr_inds = lr;
    yi.m_sum_ind = npz_require_int(npz, "YI_m_sum_ind");
    yi.cg_coeff = npz_require_double(npz, "YI_cg_coeff");
    yi.n_cg_terms = npz_get_int_scalar(npz, "YI_n_cg_terms", 0);
    yi.n_output_funcs = npz_get_int_scalar(npz, "YI_n_output_funcs", 0);
    yi.Lmax = npz_get_int_scalar(npz, "YI_Lmax", 3);
    yi.n_rad_max = npz_get_int_scalar(npz, "YI_n_rad_max", 32);
    yi.inv_avg_n_neigh = npz_get_double_scalar(npz, "YI_inv_avg_n_neigh", 1.0);
  }

  // B reducer (equivariant, NOT element-dependent)
  B_n_out = npz_get_int_scalar(npz, "B_n_out", 64);
  B_n_funcs = npz_get_int_scalar(npz, "B_n_funcs", 31);
  B_is_elem_dependent = npz_get_int_scalar(npz, "B_is_central_atom_type_dependent", 0) != 0;
  load_reduce("B", "YI", B_reduce_YI);
  load_reduce("B", "B0", B_reduce_B0);

  // I2 reducer (scalar, element-dependent)
  I2_n_out = npz_get_int_scalar(npz, "I2_n_out", 17);
  I2_is_elem_dependent = npz_get_int_scalar(npz, "I2_is_central_atom_type_dependent", 1) != 0;
  load_reduce("I2", "B", I2_reduce_B);
  load_reduce("I2", "BB", I2_reduce_BB);
  load_reduce("I2", "BBB", I2_reduce_BBB);
  load_reduce("I2", "BBBB", I2_reduce_BBBB);

  // I_0_LN RMSNorm
  I_0_LN_scale = npz_require_double(npz, "I_0_LN_scale");
  I_0_LN_type = npz_get_int_scalar(npz, "I_0_LN_type", 0);
  I_0_LN_n_out = npz_get_int_scalar(npz, "I_0_LN_n_out", 17);

  // Energy MLP
  int energy_n_layers_val = npz_get_int_scalar(npz, "energy_mlp_n_layers", 2);
  energy_mlp_layers.resize(energy_n_layers_val);
  for (int i = 0; i < energy_n_layers_val; i++) {
    auto &layer = energy_mlp_layers[i];
    std::string wkey = "energy_mlp_W" + std::to_string(i);
    layer.W = npz_require_double(npz, wkey);
    layer.norm = npz_get_double_scalar(npz, "energy_mlp_norm" + std::to_string(i), 1.0);
    auto wit = npz.find(wkey);
    if (wit != npz.end() && wit->second.shape.size() == 2) {
      layer.n_in = wit->second.shape[0];
      layer.n_out = wit->second.shape[1];
    }
  }
  energy_mlp_activation = npz_get_int_scalar(npz, "energy_mlp_activation", 1);  // 1=tanh

  // Shifts
  shift_values = npz_get_double(npz, "shift_values");
  // a model without a shift instruction exports no shift_values: zero shift per element (the energy readout
  // indexes shift_values by element, so an empty array was read out of bounds)
  if (shift_values.empty()) shift_values.assign(n_elements, 0.0);
  else if ((int) shift_values.size() != n_elements)
    throw Error("GRACE-2L: shift_values has " + std::to_string(shift_values.size()) + " entries, expected n_elements = " +
                             std::to_string(n_elements));
  output_scale = npz_get_double_scalar(npz, "output_scale", 1.0);

  element_names = npz_names(npz, "element_names");

  // ---- UQ / extrapolation-grade artifacts (optional; schema v6 basis-RP) ----
  // Exclusive to uqv6 (schema 6: L2-normalize + log-norm density channels); older
  // v3/v4 artifacts are rejected at the gate so there is no silent miscompute.
  I2_instruction_names = npz_names(npz, "I2_instruction_names");
  rho_instruction_names = npz_names(npz, "rho_instruction_names");
  uq_schema_version = npz_get_int_scalar(npz, "uq_schema_version", 0);
  has_uq = uq_schema_version != 0;
  if (uq_schema_version == 6) try {
    uq_n_elements = npz_get_int_scalar(npz, "uq_n_elements", n_elements);
    uq_max_clusters = npz_get_int_scalar(npz, "uq_max_clusters", 0);
    uq_feature_dim = npz_get_int_scalar(npz, "uq_feature_dim", 0);
    uq_rp_dim = npz_get_int_scalar(npz, "uq_rp_dim", 0);
    // v6 feature-build flags (confirmatory: must match the v6 profile below).
    uq_normalize = npz_get_int_scalar(npz, "uq_rp_normalize", 0);
    const int uq_add_density = npz_get_int_scalar(npz, "uq_rp_add_density_channel", 0);
    uq_density_scale = npz_get_double_scalar(npz, "uq_rp_density_scale", 1.0);
    const int uq_transform = npz_get_int_scalar(npz, "uq_feature_transform", 0);
    uq_centroids = npz_require_double(npz, "uq_centroids");
    uq_inv_cov = npz_require_double(npz, "uq_inv_cov");
    uq_n_clusters = npz_require_int(npz, "uq_n_clusters");
    uq_interp_thresholds = npz_require_double(npz, "uq_interp_thresholds");
    uq_rp_matrix = npz_require_double(npz, "uq_rp_matrix");   // [D_basis, rp_dim] row-major
    // Confirm the feature-build flags agree with schema v6; a mismatch means the
    // artifact is corrupt/mislabeled (this build implements normalize+density only,
    // identity transform — no asinh).
    if (uq_normalize != 1 || uq_add_density != 1 || uq_transform != 0)
      throw Error("GRACE-2L/KK UQ: schema v6 expects normalize=1, "
          "add_density_channel=1, feature_transform=0 (identity); got normalize=" +
          std::to_string(uq_normalize) + ", add_density=" + std::to_string(uq_add_density) +
          ", transform=" + std::to_string(uq_transform) + ". Artifact mislabeled or unsupported variant.");
    // Validate the dense arrays against the declared (E, Kmax, D) and derive
    // D_basis from R; copy_*d index these flat with no bounds check, so a
    // stale/mismatched export must fail here rather than read out of bounds.
    if (uq_rp_dim <= 0 || uq_feature_dim < uq_rp_dim)
      throw Error("GRACE-2L/KK UQ: uq_feature_dim (" + std::to_string(uq_feature_dim) +
          ") must be >= uq_rp_dim (" + std::to_string(uq_rp_dim) + ")");
    uq_n_density = uq_feature_dim - uq_rp_dim;   // = 1 + n_blocks (=3 for 2L)
    // The 2L UQ kernel writes exactly the [full, L2, L1] density channels (RP+0..RP+2);
    // a higher count would leave f[RP+3..) uninitialized yet read by the GMM loops.
    if (uq_n_density > 3)
      throw Error("GRACE-2L/KK UQ: uq_n_density (" + std::to_string(uq_n_density) +
          ") > 3; the 2L UQ kernel only writes [full, L2, L1] density channels. "
          "Artifact has an unexpected invariant-block count.");
    if (uq_rp_matrix.size() % (size_t)uq_rp_dim != 0)
      throw Error("GRACE-2L/KK UQ: uq_rp_matrix size (" +
          std::to_string(uq_rp_matrix.size()) + ") not divisible by rp_dim " + std::to_string(uq_rp_dim));
    uq_d_basis = (int)(uq_rp_matrix.size() / (size_t)uq_rp_dim);
    // Guard against a transposed/mis-strided R: the flat-size divisibility check
    // above is blind to a [rp_dim, D_basis] layout (identical element count), which
    // copy_2d would then read with the wrong stride -> silently wrong gamma. Verify
    // the stored 2D column count is rp_dim.
    {
      auto rp_it = npz.find("uq_rp_matrix");
      if (rp_it != npz.end() && rp_it->second.shape.size() == 2 &&
          (int)rp_it->second.shape[1] != uq_rp_dim)
        throw Error("GRACE-2L/KK UQ: uq_rp_matrix has shape [" +
            std::to_string(rp_it->second.shape[0]) + ", " + std::to_string(rp_it->second.shape[1]) +
            "]; expected column count == uq_rp_dim (" + std::to_string(uq_rp_dim) +
            "). R may be transposed or mislabeled.");
    }
    const size_t E = uq_n_elements, K = uq_max_clusters, D = uq_feature_dim;
    auto uq_chk = [](const char *name, size_t got, size_t want) {
      if (got != want)
        throw Error(std::string("GRACE-2L/KK UQ: ") + name + " has " +
            std::to_string(got) + " elements, expected " + std::to_string(want));
    };
    uq_chk("uq_centroids",         uq_centroids.size(),         E * K * D);
    uq_chk("uq_inv_cov",           uq_inv_cov.size(),           E * K * D * D);
    uq_chk("uq_n_clusters",        uq_n_clusters.size(),        E);
    uq_chk("uq_interp_thresholds", uq_interp_thresholds.size(), E * K);
    // uqv6 artifacts must cover ALL model elements with real GMM clusters; a zero
    // here means an incomplete export, which we reject rather than silently emit a
    // sentinel gamma for that element at runtime.
    for (size_t e = 0; e < uq_n_clusters.size(); e++)
      if (uq_n_clusters[e] <= 0) {
        throw Error("GRACE-2L/KK UQ: element index " + std::to_string(e) +
            " has no GMM clusters; uqv6 artifacts must cover all elements. Re-export the UQ "
            "artifacts with a current grace_utils.");
      } else if (uq_n_clusters[e] > uq_max_clusters) {    // every GMM loop indexes the arrays by cluster (lammps-ace 3cd0775d99)
        throw Error("GRACE-2L/KK UQ: element index " + std::to_string(e) + " has " + std::to_string(uq_n_clusters[e]) +
            " GMM clusters, more than uq_max_clusters = " + std::to_string(uq_max_clusters));
      }
    // Defensive finite-value check: a partially-populated / mis-inverted export can
    // leave NaN/Inf in the GMM arrays (e.g. a singular covariance), which would
    // silently poison the Mahalanobis sigma at runtime. Reject at load instead.
    auto uq_finite = [](const char *name, const std::vector<double> &v) {
      for (double x : v)
        if (!std::isfinite(x))
          throw Error(std::string("GRACE-2L/KK UQ: ") + name +
              " contains a non-finite value (NaN/Inf); re-export the UQ artifacts.");
    };
    uq_finite("uq_centroids",         uq_centroids);
    uq_finite("uq_inv_cov",           uq_inv_cov);
    uq_finite("uq_interp_thresholds", uq_interp_thresholds);
    uq_finite("uq_rp_matrix",         uq_rp_matrix);
  } catch (const Error &e) {
    uq_error = e.what();
  }
}

void GRACE2LModel::validate_uq() const
{
  if (!has_uq) throw Error("GRACE-2L: the model has no UQ head");
  if (uq_schema_version != 6)
    throw Error("GRACE-2L: UQ schema v" + std::to_string(uq_schema_version) +
                " unsupported (expects v6, uqv6 basis-RP normalize+density). Re-export the UQ artifacts at schema v6 "
                "with a current grace_utils.");
  if (!uq_error.empty()) throw Error(uq_error);
  check_uq_layout("GRACE-2L", uq_layout(), uq_d_basis, uq_n_density, uq_n_elements, n_elements,
                  {I2_instruction_names, rho_instruction_names});
}

std::vector<UqBlock> GRACE2LModel::uq_layout() const
{
  std::vector<UqBlock> L;
  int64_t off = 0;
  {
    const ReduceWeights *rs[4] = {&I2_reduce_B, &I2_reduce_BB, &I2_reduce_BBB, &I2_reduce_BBBB};
    L.push_back(make_uq_block("I2", rs, off));
    off += L.back().width;
  }
  {
    const ReduceWeights *rs[4] = {&rho_reduce_A, &rho_reduce_AA, &rho_reduce_AAA, &rho_reduce_AAAA};
    L.push_back(make_uq_block("rho", rs, off));
    off += L.back().width;
  }
  return L;
}


}    // namespace grace

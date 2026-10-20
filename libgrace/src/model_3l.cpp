// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from lammps-ace src/ML-PACE/grace_3l_model.cpp (GRACE3LModel::load).
#include "grace/model_3l.h"
#include "grace/error.h"
#include "npz3.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace grace {

using namespace npz;

// ======================================================================
// GRACE3LModel::load — minimal loader (globals + cutoffs + element names).
// Per-layer weight tensors are loaded by later Phase 2/3 tasks; the empty
// compute() needs no weights, but coeff()/init_one() need this metadata.
// ======================================================================

void GRACE3LModel::load(const std::string &filepath) {
  cnpy::npz_t npz = npz_open(filepath);
  reduce_inputs_grouped = false;

  // ---- Global metadata ----
  n_elements = npz_get_int_scalar(npz, "n_elements", 0);
  embedding_size = npz_get_int_scalar(npz, "embedding_size", 64);
  nradbase = npz_get_int_scalar(npz, "nradbase", 10);
  // No bare `lmax` key in the 3L npz — the angular lmax is SPBF _lmax (A1_lmax).
  lmax = npz_get_int_scalar(npz, "lmax", npz_get_int_scalar(npz, "A1_lmax", 4));
  // Global radial_basis_p (=16) is NOT the envelope order; geometry uses the
  // SPBF p (=A1_p=5). Keep the global one only for the record; radial_basis_p
  // (the envelope order actually used) is overwritten from A1_p below.
  radial_basis_p_global = npz_get_int_scalar(npz, "radial_basis_p", 0);
  radial_basis_p = radial_basis_p_global;   // provisional; set to SPBF p below
  rcut = npz_get_cutoff_scalar(npz, "rcut", 6.0, "GRACE-3L/KK");
  has_bond_specific_cutoff = npz_get_int_scalar(npz, "has_bond_specific_cutoff", 0) != 0;
  bond_cutoff_map = npz_get_cutoff(npz, "bond_cutoff_map", "GRACE-3L/KK");
  if (bond_cutoff_map.empty())
    bond_cutoff_map.resize((size_t) n_elements * n_elements, rcut);

  chem_embedding = npz_require_double(npz, "chem_embedding");
  shift_values = npz_require_double(npz, "shift_values");
  output_scale = npz_get_double_scalar(npz, "output_scale", 1.0);  // absent -> 1.0
  element_names = npz_names(npz, "element_names");

  // ---- Helper: generic MLP (radial MLP has bias on hidden layers; energy MLP
  //      has no bias). W{i} raw + scalar norm{i}; b{i} (hidden only) if present. ----
  auto load_mlp = [&](const std::string &prefix, MLP &mlp) {
    mlp.n_layers = npz_get_int_scalar(npz, prefix + "_n_layers", 0);
    mlp.W.resize(mlp.n_layers);
    mlp.b.resize(mlp.n_layers);
    mlp.norm.resize(mlp.n_layers, 1.0);
    mlp.n_in.resize(mlp.n_layers, 0);
    mlp.n_out.resize(mlp.n_layers, 0);
    mlp.has_bias = false;
    for (int i = 0; i < mlp.n_layers; i++) {
      const std::string wk = prefix + "_W" + std::to_string(i);
      mlp.W[i] = npz_require_double(npz, wk);
      mlp.norm[i] = npz_get_double_scalar(npz, prefix + "_norm" + std::to_string(i), 1.0);
      auto sh = npz_shape(npz, wk);
      if (sh.size() == 2) { mlp.n_in[i] = (int) sh[0]; mlp.n_out[i] = (int) sh[1]; }
      // bias b{i} is stored [1, n_out] on hidden layers only (output has none).
      auto b = npz_get_double(npz, prefix + "_b" + std::to_string(i));
      if (!b.empty()) { mlp.b[i] = std::move(b); mlp.has_bias = true; }
    }
  };

  // ---- SPBF (A1 scalar, A2/A3 equivariant) ----
  auto spbf_names = npz_names(npz, "spbf_names");
  spbf.clear();
  spbf.reserve(spbf_names.size());
  for (const auto &nm : spbf_names) {
    SPBF s;
    s.name = nm;
    s.equivariant = npz_get_int_scalar(npz, nm + "_equivariant", 0) != 0;
    s.n_rad_max = npz_get_int_scalar(npz, nm + "_n_rad_max", 0);
    s.n_rad_basis = npz_get_int_scalar(npz, nm + "_n_rad_basis", 0);
    s.lmax = npz_get_int_scalar(npz, nm + "_lmax", 4);
    s.Lmax = npz_get_int_scalar(npz, nm + "_Lmax", 4);
    s.p = npz_get_int_scalar(npz, nm + "_p", 5);
    s.rcut = npz_get_cutoff_scalar(npz, nm + "_rcut", rcut, "GRACE-3L/KK");
    s.inv_avg_n_neigh = npz_get_double_scalar(npz, nm + "_inv_avg_n_neigh", 1.0);
    s.l_tile = npz_require_int(npz, nm + "_l_tile");
    load_mlp(nm + "_mlp", s.mlp);
    if (!s.equivariant) {
      s.lin_transform_W = npz_require_double(npz, nm + "_lin_transform_W");
      s.lin_transform_norm = npz_get_double_scalar(npz, nm + "_lin_transform_norm", 1.0);
    } else {
      s.chem_linear_W = npz_require_double(npz, nm + "_chem_linear_W");
      s.chem_linear_norm = npz_get_double_scalar(npz, nm + "_chem_linear_norm", 1.0);
      s.chem_l0_mask = npz_require_double(npz, nm + "_chem_l0_mask");
      s.cg_W = npz_require_double(npz, nm + "_cg_W");
      s.nfunc = npz_get_int_scalar(npz, nm + "_nfunc", 0);
      s.n_lm_ind = npz_get_int_scalar(npz, nm + "_n_lm_ind", 0);
    }
    spbf.push_back(std::move(s));
  }
  // Effective radial ceilings + envelope order from the SPBF layers.
  if (spbf.size() >= 1) { L1_nradmax = spbf[0].n_rad_max; radial_basis_p = spbf[0].p; }
  if (spbf.size() >= 2) L2_nradmax = spbf[1].n_rad_max;
  if (spbf.size() >= 3) L3_nradmax = spbf[2].n_rad_max;

  // ---- cp_l products ----
  auto prod_names = npz_names(npz, "prod_names");
  prod.clear();
  prod.reserve(prod_names.size());
  for (const auto &nm : prod_names) {
    CPL c;
    c.name = nm;
    c.rank = npz_get_int_scalar(npz, nm + "_rank", 0);
    c.nfunc = npz_get_int_scalar(npz, nm + "_nfunc", 0);
    c.n_left = npz_get_int_scalar(npz, nm + "_n_left", 0);
    c.n_right = npz_get_int_scalar(npz, nm + "_n_right", 0);
    c.n_groups_left = npz_get_int_scalar(npz, nm + "_n_groups_left", 0);
    c.n_groups_right = npz_get_int_scalar(npz, nm + "_n_groups_right", 0);
    c.norm_u = npz_get_double_scalar(npz, nm + "_norm_u", 1.0);
    c.norm_v = npz_get_double_scalar(npz, nm + "_norm_v", 1.0);
    c.U = npz_require_double(npz, nm + "_U");
    c.V = npz_require_double(npz, nm + "_V");
    c.cg = npz_require_double(npz, nm + "_cg");
    c.n_cg = (int) c.cg.size();
    c.group_left = npz_require_int(npz, nm + "_group_left");
    c.group_right = npz_require_int(npz, nm + "_group_right");
    c.left_ind = npz_require_int(npz, nm + "_left_ind");
    c.right_ind = npz_require_int(npz, nm + "_right_ind");
    c.m_sum_ind = npz_require_int(npz, nm + "_m_sum_ind");
    c.out_l = npz_require_int(npz, nm + "_out_l");
    c.out_parity = npz_require_int(npz, nm + "_out_parity");
    prod.push_back(std::move(c));
  }

  // ---- FunctionReduceN (product reduces + eq*/rho*) ----
  auto reduce_names = npz_names(npz, "reduce_names");
  reduce.clear();
  reduce.reserve(reduce_names.size());
  for (const auto &nm : reduce_names) {
    Reduce r;
    r.name = nm;
    r.n_out = npz_get_int_scalar(npz, nm + "_n_out", 0);
    r.n_funcs = npz_get_int_scalar(npz, nm + "_n_funcs", 0);
    r.only_invar = npz_get_int_scalar(npz, nm + "_only_invar", 0) != 0;
    r.elem_dep = npz_get_int_scalar(npz, nm + "_is_central_atom_type_dependent", 0) != 0;
    auto instr_names = npz_names(npz, nm + "_instruction_names");
    r.instr.reserve(instr_names.size());
    for (const auto &cn : instr_names) {
      ReduceInstr ri;
      ri.name = cn;
      ri.W = npz_require_double(npz, nm + "_reduce_" + cn + "_W");
      ri.norm = npz_get_double_scalar(npz, nm + "_reduce_" + cn + "_norm", 1.0);
      ri.n_in = npz_get_int_scalar(npz, nm + "_n_in_" + cn, 0);
      ri.w_shape = npz_get_int_scalar(npz, nm + "_w_shape_" + cn, 0);
      ri.collect_ind = npz_require_int(npz, nm + "_collect_ind_" + cn);
      ri.w_l_tile = npz_require_int(npz, nm + "_w_l_tile_" + cn);
      ri.total_sum_ind = npz_require_int(npz, nm + "_total_sum_ind_" + cn);  // [n_conn,1] flat
      ri.n_conn = (int) ri.collect_ind.size();
      r.instr.push_back(std::move(ri));
    }
    // EXACT key — `{name}_norm_map` (NOT the RMS-norm `{name}_norm` footgun).
    r.norm_map = npz_get_double(npz, nm + "_norm_map");
    r.has_norm_map = !r.norm_map.empty();
    reduce.push_back(std::move(r));
  }

  // ---- FCRight2Left (keys prefixed fc_<name>; index list stores bare name) ----
  auto fc_names = npz_names(npz, "fc_names");
  fc.clear();
  fc.reserve(fc_names.size());
  for (const auto &nm : fc_names) {
    FC f;
    f.name = nm;
    const std::string p = "fc_" + nm;
    f.n_out = npz_get_int_scalar(npz, p + "_n_out", 0);
    f.left_coefs = npz_get_int_scalar(npz, p + "_left_coefs", 0) != 0;
    f.w_right = npz_require_double(npz, p + "_w_right");
    f.norm_right = npz_get_double_scalar(npz, p + "_norm_right", 1.0);
    f.w_tile_left = npz_require_int(npz, p + "_w_tile_left");
    f.w_tile_right = npz_require_int(npz, p + "_w_tile_right");
    f.collect_to = npz_require_int(npz, p + "_collect_to");
    f.collect_from = npz_require_int(npz, p + "_collect_from");
    f.norm_out_factor = npz_require_double(npz, p + "_norm_out_factor");  // [n_funcs_left,1,1] flat
    f.n_funcs_left = (int) f.w_tile_left.size();
    f.n_funcs_right = (int) f.collect_from.size();
    auto wr = npz_shape(npz, p + "_w_right");
    if (wr.size() == 3) { f.n_in_right = (int) wr[1]; f.w_shape_right = (int) wr[2]; }
    if (f.left_coefs) {   // 3L FCs are right->left (left_coefs=False); guard anyway
      f.w_left = npz_get_double(npz, p + "_w_left");
      f.norm_left = npz_get_double_scalar(npz, p + "_norm_left", 1.0);
      auto wl = npz_shape(npz, p + "_w_left");
      if (wl.size() == 3) { f.n_in_left = (int) wl[1]; f.w_shape_left = (int) wl[2]; }
    }
    fc.push_back(std::move(f));
  }

  // ---- EquivariantRMSNorm (eq1_norm, eq2_norm) ----
  auto eqnorm_names = npz_names(npz, "eqnorm_names");
  eqnorm.clear();
  eqnorm.reserve(eqnorm_names.size());
  for (const auto &nm : eqnorm_names) {
    EqNorm e;
    e.name = nm;
    e.affine_weight = npz_require_double(npz, nm + "_affine_weight");
    e.degree_weights = npz_require_double(npz, nm + "_degree_weights");
    e.l0_mask = npz_require_double(npz, nm + "_l0_mask");
    e.expand_index = npz_require_int(npz, nm + "_expand_index");
    e.eps = npz_get_double_scalar(npz, nm + "_eps", 1e-8);
    e.center_l0 = npz_get_int_scalar(npz, nm + "_center_l0", 1);
    e.M = (int) e.degree_weights.size();
    auto aw = npz_shape(npz, nm + "_affine_weight");
    if (aw.size() == 2) { e.n_groups = (int) aw[0]; e.n_out = (int) aw[1]; }
    eqnorm.push_back(std::move(e));
  }

  // ---- InvariantLayerRMSNorm (rho1/2/3_norm). EXACT keys — the reduce
  //      `rho1`'s `rho1_norm_map` is a different array from `rho1_norm_scale`. ----
  auto rmsnorm_names = npz_names(npz, "rmsnorm_names");
  rmsnorm.clear();
  rmsnorm.reserve(rmsnorm_names.size());
  for (const auto &nm : rmsnorm_names) {
    RMSNorm r;
    r.name = nm;
    r.scale = npz_require_double(npz, nm + "_scale");
    r.type = npz_get_int_scalar(npz, nm + "_type", 0);
    r.n_out = npz_get_int_scalar(npz, nm + "_n_out", (int) r.scale.size());
    rmsnorm.push_back(std::move(r));
  }

  // ---- Readout energy MLP (no bias) + 3-origin metadata ----
  load_mlp("energy_mlp", energy_mlp);
  energy_mlp_activation = npz_get_int_scalar(npz, "energy_mlp_activation", 0);  // 0=silu
  out1_origins = npz_names(npz, "out1_origins");
  energy_mlp_origin_n_out = npz_get_int_scalar(npz, "energy_mlp_origin_n_out", 0);
  energy_mlp_n_origins = npz_get_int_scalar(npz, "energy_mlp_n_origins", (int) out1_origins.size());

  // ---- UQ / extrapolation-grade artifacts (optional; schema v6 basis-RP): GraceUQHead ----
  load_uq_head(*this, npz, n_elements, 4, "GRACE-3L");
}

}    // namespace grace

// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Contributing author: Yury Lysogorskiy (ICAMS)
#include "grace/model_2l.h"
#include "grace/error.h"

#include "npz3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

namespace grace {

using namespace npz;

// ======================================================================
// GRACE2LModel::load — load weights from .npz
// ======================================================================


void GRACE2LModel::load(const std::string &filepath) { load_arrays(npz_open(filepath)); }

void GRACE2LModel::load_arrays(const cnpy::npz_t &npz) {
  // a second load into the same object (LAMMPS pair_coeff twice) must keep nothing of the previous model: arrays
  // read only when present (UQ head, optional blocks) would otherwise survive
  *this = GRACE2LModel();
  check_grace_format(npz, "GRACE-2L");

  // Every array is checked against the declared sizes before anything is sized from it: a count is tied to the
  // stored shape of an array it describes (so a corrupt count cannot ask for a huge allocation), every index lies in
  // the extent of what it addresses, and the channel / function counts agree across the stages that share them.

  // Metadata
  n_elements = npz_get_int_in(npz, "n_elements", 89, 1, kMaxDim);
  embedding_size = npz_get_int_in(npz, "embedding_size", 128, 1, kMaxDim);
  // Chemical embedding [n_elements, embedding_size]
  chem_embedding = npz_require_double(npz, "chem_embedding");
  require_shape(npz, "chem_embedding", {n_elements, embedding_size});
  lmax = npz_get_int_in(npz, "lmax", 4, 0, kMaxL);
  nradbase = npz_get_int_in(npz, "nradbase", 10, 1, kMaxDim);
  // cutoff polynomial of degree p + 2: x^(p-1) at x = 0 needs p >= 1
  radial_basis_p = npz_get_int_in(npz, "radial_basis_p", 16, 1, kMaxP);
  rcut = npz_get_cutoff_scalar(npz, "rcut", 6.0, "GRACE-2L/KK");
  has_bond_specific_cutoff = npz_get_flag(npz, "has_bond_specific_cutoff", 0);
  bond_cutoff_map = npz_get_cutoff(npz, "bond_cutoff_map", "GRACE-2L/KK");
  require_shape(npz, "bond_cutoff_map", {n_elements, n_elements});
  if (bond_cutoff_map.empty())
    bond_cutoff_map.assign((size_t) n_elements * n_elements, rcut);
  const int64_t nl = (int64_t) lmax + 1, nlm = nl * nl;    // spherical harmonics (l, m), l <= lmax

  // an MLP (npz_mlp) of prefix_n_layers layers (default_layers when absent): n_in inputs, n_out outputs
  auto load_mlp = [&](const std::string &prefix, int default_layers, int64_t n_in, int64_t n_out,
                      const std::string &in_what, const std::string &out_what, std::vector<MLPLayer> &layers) {
    const int n = npz_get_int_in(npz, prefix + "_n_layers", default_layers, 1, kMaxDim);
    for (auto &l : npz_mlp(npz, prefix, n, n_in, n_out, in_what, out_what))
      layers.push_back({std::move(l.W), l.norm, l.n_in, l.n_out});
    return n;
  };

  // radial MLP: nradbase Chebyshev functions -> n_rad_max x (lmax + 1) radial functions
  auto load_mlp_rad = [&](const std::string &prefix, int &n_layers, std::vector<MLPLayer> &layers, int &nradmax_out) {
    // a chem-embedded radial MLP (pair-dependent input) is implemented by none of the GRACE-2L engines
    require_eq(prefix + "_has_chem_emb", npz_get_int_scalar(npz, prefix + "_has_chem_emb", 0), 0,
               "chem-embedded radial MLPs are not supported");
    nradmax_out = npz_get_int_in(npz, prefix + "_n_rad_max", 32, 1, kMaxDim);
    n_layers = load_mlp(prefix, 3, nradbase, nradmax_out * nl, "nradbase", "n_rad_max x (lmax + 1)", layers);
  };

  // Layer 1 radial MLP (R)
  load_mlp_rad("mlp_rad_R", L1_mlp_rad_n_layers, L1_mlp_rad_layers, L1_nradmax);

  // Layer 2 radial MLP (R1)
  load_mlp_rad("mlp_rad_R1", L2_mlp_rad_n_layers, L2_mlp_rad_layers, L2_nradmax);

  // A indicator transform [embedding_size, L1_nradmax]
  A_lin_transform_W = npz_require_double(npz, "A_lin_transform_W");
  require_shape(npz, "A_lin_transform_W", {embedding_size, L1_nradmax});
  A_lin_transform_norm = npz_get_double_scalar(npz, "A_lin_transform_norm", 1.0);
  A_inv_avg_n_neigh = npz_get_double_scalar(npz, "A_inv_avg_n_neigh", 1.0);

  // B0 indicator transform [embedding_size, L2_nradmax]
  B0_lin_transform_W = npz_require_double(npz, "B0_lin_transform_W");
  require_shape(npz, "B0_lin_transform_W", {embedding_size, L2_nradmax});
  B0_lin_transform_norm = npz_get_double_scalar(npz, "B0_lin_transform_norm", 1.0);
  B0_inv_avg_n_neigh = npz_get_double_scalar(npz, "B0_inv_avg_n_neigh", 1.0);

  // Helper: load FC weights (checked by check_fc below, once the function counts are known)
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
    fc.n_out = npz_get_int_in(npz, prefix + "_n_out", 64, 1, kMaxDim);
    // left_coefs = 0 (identity left path, as the 3L FCRight2Left) is not what the 2L engines compute
    require_eq(prefix + "_left_coefs", npz_get_int_scalar(npz, prefix + "_left_coefs", 1), 1,
               "the GRACE-2L engines apply left weights in every FC");
    fc.n_funcs_left = (int)fc.w_tile_left.size();
    fc.n_funcs_right = (int)fc.collect_from.size();
    // w_left [n_out, n_in_left, w_shape_left], w_right [n_out, n_in_right, w_shape_right]
    fc.w_shape_left = (int) npz_shape_rank(npz, prefix + "_w_left", 3)[2];
    fc.w_shape_right = (int) npz_shape_rank(npz, prefix + "_w_right", 3)[2];
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
    prod.n_output_funcs = npz_get_int_in(npz, prefix + "_n_output_funcs", 0, 1, kMaxDim);
    prod.n_cg_terms = npz_get_int_in(npz, prefix + "_n_cg_terms", 0, 1, kMaxDim);
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
    rw.w_shape = npz_get_int_in(npz, reduce_prefix + "_w_shape_" + instr_name, 0, 1, kMaxDim);
    rw.n_connections = (int)rw.collect_ind.size();
    // total_sum_ind: [n_connections, 1] — flatten to [n_connections]
    const std::string tkey = reduce_prefix + "_total_sum_ind_" + instr_name;
    rw.total_sum_ind = npz_require_int(npz, tkey);
    require_size(tkey, rw.total_sum_ind, rw.n_connections);
  };
  // the central-atom-type dependence the engines implement for each reduce (it fixes the layout of W)
  auto elem_dependent = [&](const std::string &prefix, int fixed) {
    const std::string key = prefix + "_is_central_atom_type_dependent";
    require_eq(key, npz_get_int_scalar(npz, key, fixed), fixed, "the GRACE-2L engines implement this layout only");
  };

  // I1 reducer (equivariant, element-dependent)
  I1_n_out = npz_get_int_in(npz, "I1_n_out", 12, 1, kMaxDim);
  I1_n_funcs = npz_get_int_in(npz, "I1_n_funcs", 4, 1, kMaxDim);
  elem_dependent("I1", 1);
  load_reduce("I1", "A", I1_reduce_A);
  load_reduce("I1", "AA", I1_reduce_AA);
  load_reduce("I1", "AAA", I1_reduce_AAA);
  load_reduce("I1", "AAAA", I1_reduce_AAAA);

  // I reducer (equivariant, NOT element-dependent)
  I_n_out = npz_get_int_in(npz, "I_n_out", 32, 1, kMaxDim);
  I_n_funcs = npz_get_int_in(npz, "I_n_funcs", 4, 1, kMaxDim);
  elem_dependent("I", 0);
  load_reduce("I", "I1", I_reduce_I1);

  // rho reducer (scalar, element-dependent)
  rho_n_out = npz_get_int_in(npz, "rho_n_out", 17, 2, kMaxDim);    // rho[0] and the energy MLP input rho[1:]
  elem_dependent("rho", 1);
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
    yi.n_cg_terms = npz_get_int_in(npz, "YI_n_cg_terms", 0, 1, kMaxDim);
    yi.n_output_funcs = npz_get_int_in(npz, "YI_n_output_funcs", 0, 1, kMaxDim);
    yi.n_rad_max = npz_get_int_scalar(npz, "YI_n_rad_max", 32);
    yi.inv_avg_n_neigh = npz_get_double_scalar(npz, "YI_inv_avg_n_neigh", 1.0);
  }

  // B reducer (equivariant, NOT element-dependent)
  B_n_out = npz_get_int_in(npz, "B_n_out", 64, 1, kMaxDim);
  B_n_funcs = npz_get_int_in(npz, "B_n_funcs", 31, 1, kMaxDim);
  elem_dependent("B", 0);
  load_reduce("B", "YI", B_reduce_YI);
  load_reduce("B", "B0", B_reduce_B0);

  // I2 reducer (scalar, element-dependent)
  I2_n_out = npz_get_int_in(npz, "I2_n_out", 17, 1, kMaxDim);
  elem_dependent("I2", 1);
  load_reduce("I2", "B", I2_reduce_B);
  load_reduce("I2", "BB", I2_reduce_BB);
  load_reduce("I2", "BBB", I2_reduce_BBB);
  load_reduce("I2", "BBBB", I2_reduce_BBBB);

  // I_0_LN RMSNorm
  I_0_LN_scale = npz_require_double(npz, "I_0_LN_scale");
  I_0_LN_type = npz_get_int_scalar(npz, "I_0_LN_type", 0);
  I_0_LN_n_out = npz_get_int_scalar(npz, "I_0_LN_n_out", 17);

  // Source names of the reduces (I2 / rho: kept for the UQ basis order)
  I2_instruction_names = npz_names(npz, "I2_instruction_names");
  rho_instruction_names = npz_names(npz, "rho_instruction_names");
  // when exported, the sources the loader reads: in any order (the energy sums them; validate_uq checks the order the
  // UQ basis assumes), none missing, none ignored
  auto sources = [&](const std::string &r, std::vector<std::string> want) {
    const std::string key = r + "_instruction_names";
    if (npz.count(key)) require_name_set(key, npz_names(npz, key), want);
  };
  sources("I1", {"A", "AA", "AAA", "AAAA"});
  sources("rho", {"A", "AA", "AAA", "AAAA"});
  sources("I", {"I1"});
  sources("B", {"YI", "B0"});
  sources("I2", {"B", "BB", "BBB", "BBBB"});

  // ---- consistency of the arrays with the declared sizes ----

  // Channel widths. Layer 1: the radial channels of A and the FC channels of A1 .. AAAA agree (AAA = AA1 x A channel
  // by channel); layer 2: the message I and the radial R1 (YI = R1 I_j channel by channel), the B and FC channels of
  // B1 .. BBBB (BBB = BB1 x B).
  const int64_t C1 = L1_nradmax, CR = L2_nradmax, C2 = B_n_out;
  require_eq("I_n_out", I_n_out, CR, "YI couples I with the layer-2 radial channels mlp_rad_R1_n_rad_max");
  require_eq("YI_n_rad_max", yi.n_rad_max, CR, "the layer-2 radial channels mlp_rad_R1_n_rad_max");

  // CG product: n_cg_terms terms (left_ind, right_ind) -> m_sum_ind; left / right address nl_in / nr_in functions
  auto check_prod = [&](const std::string &p, const CGProduct &c, int64_t nl_in, int64_t nr_in) {
    const size_t nt = c.n_cg_terms;
    require_size(p + "_left_ind", c.left_ind, nt);
    require_size(p + "_right_ind", c.right_ind, nt);
    require_size(p + "_m_sum_ind", c.m_sum_ind, nt);
    require_size(p + "_cg_coeff", c.cg_coeff, nt);
    require_index(p + "_m_sum_ind", c.m_sum_ind, c.n_output_funcs);
    // the CPU engine sizes its tensors from the indices, the Kokkos engine from the declared count
    require_eq(p + "_n_output_funcs", c.n_output_funcs, index_extent(c.m_sum_ind), "1 + the largest " + p + "_m_sum_ind");
    require_index(p + "_left_ind", c.left_ind, nl_in);
    require_index(p + "_right_ind", c.right_ind, nr_in);
    return (int64_t) c.n_output_funcs;
  };

  // FC: out[f] = W_left[tile_left[f]] left[f] + sum over c with collect_to[c] = f of W_right[tile_right[c]]
  // right[collect_from[c]], f < nfl; ch_l / ch_r input channels, nf_r functions of the right input, ch channels out
  auto check_fc = [&](const std::string &f, const FCWeights &w, int64_t nfl, int64_t ch_l, int64_t ch_r, int64_t nf_r,
                      int64_t ch, const std::string &ch_what) {
    require_eq(f + "_n_out", w.n_out, ch, ch_what);
    require_shape(npz, f + "_w_left", {w.n_out, ch_l, w.w_shape_left});
    require_shape(npz, f + "_w_right", {w.n_out, ch_r, w.w_shape_right});
    require_size(f + "_w_tile_left", w.w_tile_left, nfl);
    require_index(f + "_w_tile_left", w.w_tile_left, w.w_shape_left);
    require_size(f + "_norm_out_factor", w.norm_out_factor, nfl);
    const size_t nc = w.collect_from.size();
    require_size(f + "_collect_to", w.collect_to, nc);
    require_size(f + "_w_tile_right", w.w_tile_right, nc);
    require_index(f + "_collect_to", w.collect_to, nfl);
    require_index(f + "_collect_from", w.collect_from, nf_r);
    require_index(f + "_w_tile_right", w.w_tile_right, w.w_shape_right);
  };

  // ReduceN r: n_out channels, nfo output functions (count_key: their declared count, empty for a scalar reduce);
  // per source: W [n_elements,] n_out x ch x w_shape, collect_ind < nf (the source's functions), w_l_tile < w_shape,
  // total_sum_ind < nfo
  struct Src {
    const ReduceWeights *rw;
    const char *name;
    int64_t ch, nf;
  };
  auto check_reduce = [&](const std::string &r, bool elem, int64_t n_out, int64_t nfo, const std::string &count_key,
                          std::initializer_list<Src> srcs) {
    int64_t extent = 0;
    for (const Src &s : srcs) {
      const ReduceWeights &w = *s.rw;
      const std::string n(s.name);
      require_eq(r + "_n_in_" + n, w.n_in, s.ch, "the channels of " + n);
      if (elem) require_shape(npz, r + "_reduce_" + n + "_W", {n_elements, n_out, s.ch, w.w_shape});
      else require_shape(npz, r + "_reduce_" + n + "_W", {n_out, s.ch, w.w_shape});
      require_index(r + "_collect_ind_" + n, w.collect_ind, s.nf);
      require_size(r + "_w_l_tile_" + n, w.w_l_tile, w.collect_ind.size());
      require_index(r + "_w_l_tile_" + n, w.w_l_tile, w.w_shape);
      require_index(r + "_total_sum_ind_" + n, w.total_sum_ind, nfo);
      extent = std::max(extent, index_extent(w.total_sum_ind));
    }
    // the n functions declared are the ones the connections address (the CPU engine sizes its tensors from the
    // indices, the Kokkos engine from the count)
    if (!count_key.empty()) require_eq(count_key, nfo, extent, "1 + the largest " + r + "_total_sum_ind_*");
  };

  // layer 1: A [nlm] -> A1 [nlm] (FC) -> AA (A1 x A1) -> AA1 (FC with A) -> AAA (AA1 x A) -> AA2 (FC with A)
  // -> AAAA (AA2 x AA2); I1, rho from A .. AAAA; I from I1
  const int64_t nAA = check_prod("prod_AA", prod_AA, nlm, nlm);
  const int64_t nAAA = check_prod("prod_AAA", prod_AAA, nAA, nlm);
  const int64_t nAAAA = check_prod("prod_AAAA", prod_AAAA, nAA, nAA);
  const std::string c1_what = "the layer-1 radial channels mlp_rad_R_n_rad_max";
  check_fc("fc_A1", fc_A1, nlm, C1, C1, nlm, C1, c1_what);
  check_fc("fc_AA1", fc_AA1, nAA, C1, C1, nlm, C1, c1_what);
  check_fc("fc_AA2", fc_AA2, nAA, C1, C1, nlm, C1, c1_what);
  check_reduce("I1", true, I1_n_out, I1_n_funcs, "I1_n_funcs",
               {{&I1_reduce_A, "A", C1, nlm}, {&I1_reduce_AA, "AA", C1, nAA}, {&I1_reduce_AAA, "AAA", C1, nAAA},
                {&I1_reduce_AAAA, "AAAA", C1, nAAAA}});
  check_reduce("rho", true, rho_n_out, 1, "",
               {{&rho_reduce_A, "A", C1, nlm}, {&rho_reduce_AA, "AA", C1, nAA}, {&rho_reduce_AAA, "AAA", C1, nAAA},
                {&rho_reduce_AAAA, "AAAA", C1, nAAAA}});
  check_reduce("I", false, I_n_out, I_n_funcs, "I_n_funcs", {{&I_reduce_I1, "I1", I1_n_out, I1_n_funcs}});
  // only_nonlin RMSNorm of rho: rho[0] passed through, rho[1:] scaled
  require_eq("I_nl_LN_type", I_nl_LN_type, 1, "the GRACE-2L engines implement the only_nonlin RMSNorm here");
  require_eq("I_nl_LN_n_out", I_nl_LN_n_out, rho_n_out, "rho_n_out");
  require_size("I_nl_LN_scale", I_nl_LN_scale, rho_n_out - 1);

  // layer 2: YI (Y_lm(r_ij) x I_j) and B0 [nlm] -> B -> B1 (FC) -> BB (B1 x B1) -> BB1 (FC with B) -> BBB (BB1 x B)
  // -> BB2 (FC with B) -> BBBB (BB2 x BB2); I2 from B .. BBBB
  {
    const int64_t nt = yi.n_cg_terms;
    require_shape(npz, "YI_lr_inds", {nt, 2});
    require_size("YI_m_sum_ind", yi.m_sum_ind, nt);
    require_size("YI_cg_coeff", yi.cg_coeff, nt);
    for (int64_t t = 0; t < nt; t++) {
      const int y = yi.lr_inds[2 * t], i = yi.lr_inds[2 * t + 1];
      if (y < 0 || y >= nlm || i < 0 || i >= I_n_funcs)
        throw bad_array("YI_lr_inds", "has (" + std::to_string(y) + ", " + std::to_string(i) + ") at " +
                                          std::to_string(t) + ", outside [0, " + std::to_string(nlm) +
                                          ") x [0, I_n_funcs = " + std::to_string(I_n_funcs) + ")");
    }
    require_index("YI_m_sum_ind", yi.m_sum_ind, yi.n_output_funcs);
    require_eq("YI_n_output_funcs", yi.n_output_funcs, index_extent(yi.m_sum_ind), "1 + the largest YI_m_sum_ind");
  }
  check_reduce("B", false, B_n_out, B_n_funcs, "B_n_funcs",
               {{&B_reduce_YI, "YI", CR, yi.n_output_funcs}, {&B_reduce_B0, "B0", CR, nlm}});
  const int64_t nB = B_n_funcs;
  const int64_t nBB = check_prod("prod_BB", prod_BB, nB, nB);
  const int64_t nBBB = check_prod("prod_BBB", prod_BBB, nBB, nB);
  const int64_t nBBBB = check_prod("prod_BBBB", prod_BBBB, nBB, nBB);
  check_fc("fc_B1", fc_B1, nB, C2, C2, nB, C2, "B_n_out");
  check_fc("fc_BB1", fc_BB1, nBB, C2, C2, nB, C2, "B_n_out");
  check_fc("fc_BB2", fc_BB2, nBB, C2, C2, nB, C2, "B_n_out");
  check_reduce("I2", true, I2_n_out, 1, "",
               {{&I2_reduce_B, "B", C2, nB}, {&I2_reduce_BB, "BB", C2, nBB}, {&I2_reduce_BBB, "BBB", C2, nBBB},
                {&I2_reduce_BBBB, "BBBB", C2, nBBBB}});
  // full RMSNorm of I2; the energy adds I_nl_LN(rho) and I_0_LN(I2) function by function
  require_eq("I_0_LN_type", I_0_LN_type, 0, "the GRACE-2L engines implement the full RMSNorm here");
  require_eq("I2_n_out", I2_n_out, rho_n_out, "the energy adds rho and I2 function by function: rho_n_out");
  require_eq("I_0_LN_n_out", I_0_LN_n_out, I2_n_out, "I2_n_out");
  require_size("I_0_LN_scale", I_0_LN_scale, I2_n_out);

  // Energy MLP: rho[1:] + I2[1:] -> one energy
  load_mlp("energy_mlp", 2, rho_n_out - 1, 1, "rho_n_out - 1", "one energy", energy_mlp_layers);
  energy_mlp_activation = npz_get_int_in(npz, "energy_mlp_activation", 1, 0, 1);    // 0 = silu, 1 = tanh

  // Shifts
  shift_values = npz_get_double(npz, "shift_values");
  // a model without a shift instruction exports no shift_values: zero shift per element (the energy readout
  // indexes shift_values by element, so an empty array was read out of bounds)
  if (npz.count("shift_values")) require_size("shift_values", shift_values, n_elements);
  else shift_values.assign(n_elements, 0.0);
  output_scale = npz_get_double_scalar(npz, "output_scale", 1.0);

  element_names = npz_names(npz, "element_names");
  if (npz.count("element_names")) require_size("element_names", element_names, n_elements);

  // ---- UQ / extrapolation-grade artifacts (optional; schema v6 basis-RP) ----
  // Exclusive to uqv6 (schema 6: L2-normalize + log-norm density channels); older
  // v3/v4 artifacts are rejected at the gate so there is no silent miscompute.
  uq_schema_version = npz_get_int_scalar(npz, "uq_schema_version", 0);
  has_uq = uq_schema_version != 0;
  if (uq_schema_version == 6) try {
    // the GMM arrays cover every model element; clusters and feature widths bounded before any size is formed
    uq_n_elements = npz_get_int_scalar(npz, "uq_n_elements", n_elements);
    require_eq("uq_n_elements", uq_n_elements, n_elements, "n_elements");
    uq_max_clusters = npz_get_int_in(npz, "uq_max_clusters", 0, 1, kMaxDim);
    uq_rp_dim = npz_get_int_in(npz, "uq_rp_dim", 0, 1, kMaxDim);
    uq_feature_dim = npz_get_int_in(npz, "uq_feature_dim", 0, uq_rp_dim, kMaxDim);
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
    uq_n_density = uq_feature_dim - uq_rp_dim;   // = 1 + n_blocks (=3 for 2L)
    // The 2L UQ kernel writes exactly the [full, L2, L1] density channels (RP+0..RP+2);
    // a higher count would leave f[RP+3..) uninitialized yet read by the GMM loops.
    if (uq_n_density > 3)
      throw Error("GRACE-2L/KK UQ: uq_n_density (" + std::to_string(uq_n_density) +
          ") > 3; the 2L UQ kernel only writes [full, L2, L1] density channels. "
          "Artifact has an unexpected invariant-block count.");
    // R is [D_basis, rp_dim]: the column count guards against a transposed / mis-strided R (the same element count
    // as [rp_dim, D_basis]), which copy_2d would read with the wrong stride -> silently wrong gamma
    const auto rs = npz_shape_rank(npz, "uq_rp_matrix", 2);
    require_eq("uq_rp_matrix", rs[1], uq_rp_dim, "uq_rp_dim", "columns");
    uq_d_basis = (int) rs[0];
    // copy_*d index the GMM arrays flat with no bounds check: a stale / mismatched export must fail here
    const size_t E = uq_n_elements, K = uq_max_clusters, D = uq_feature_dim;
    require_size("uq_centroids", uq_centroids, mul_sat(mul_sat(E, K), D));
    require_size("uq_inv_cov", uq_inv_cov, mul_sat(mul_sat(mul_sat(E, K), D), D));
    require_size("uq_n_clusters", uq_n_clusters, E);
    require_size("uq_interp_thresholds", uq_interp_thresholds, mul_sat(E, K));
    // uqv6 artifacts must cover ALL model elements with real GMM clusters; a zero
    // here means an incomplete export, which we reject rather than silently emit a
    // sentinel gamma for that element at runtime.
    for (size_t e = 0; e < uq_n_clusters.size(); e++)
      if (uq_n_clusters[e] <= 0) {
        throw Error("GRACE-2L/KK UQ: element index " + std::to_string(e) +
            " has no GMM clusters; uqv6 artifacts must cover all elements. Re-export the UQ "
            "artifacts with a current grace_utils.");
      } else if (uq_n_clusters[e] > uq_max_clusters) {    // every GMM loop indexes the arrays by cluster
        throw Error("GRACE-2L/KK UQ: element index " + std::to_string(e) + " has " + std::to_string(uq_n_clusters[e]) +
            " GMM clusters, more than uq_max_clusters = " + std::to_string(uq_max_clusters));
      }
    // (non-finite values in the GMM arrays and R are rejected by npz_get_double)
  } catch (const Error &e) {    // reported by validate_uq(), as a UQ-artifact error
    uq_error = e.what();
    if (uq_error.rfind("GRACE-2L/KK UQ: ", 0) != 0) uq_error = "GRACE-2L/KK UQ: " + uq_error;
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

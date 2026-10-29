// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Contributing author: Yury Lysogorskiy (ICAMS)
#include "grace/model_1l.h"
#include "grace/error.h"

#include "npz3.h"

#include <cstdint>

namespace grace {

using namespace npz;

void GRACE1LModel::load(const std::string &filepath) { load_arrays(npz_open(filepath)); }

void GRACE1LModel::load_arrays(const cnpy::npz_t &npz) {
  // a second load into the same object (LAMMPS pair_coeff twice) must keep nothing of the previous model: arrays
  // read only when present (UQ head, optional blocks) would otherwise survive
  *this = GRACE1LModel();
  check_grace_format(npz, "GRACE-1L");

  // Every array is checked against the model's declared sizes before anything is sized from it: the CPU and Kokkos
  // engines index these flat without bounds checks, so a truncated or corrupted file must fail here.

  // Metadata
  n_elements = npz_get_int_in(npz, "n_elements", 89, 1, kMaxDim);
  embedding_size = npz_get_int_in(npz, "embedding_size", 128, 1, kMaxDim);
  lmax = npz_get_int_in(npz, "lmax", 4, 0, kMaxL);
  nradbase = npz_get_int_in(npz, "nradbase", 10, 1, kMaxDim);
  nradmax = npz_get_int_in(npz, "nradmax", 32, 1, kMaxDim);
  // envelope polynomial 1 - (p+1)(p+2)/2 x^p + p(p+2) x^(p+1) - p(p+1)/2 x^(p+2): p >= 1 (x^(p-1) in its derivative)
  radial_basis_p = npz_get_int_in(npz, "radial_basis_p", 16, 1, kMaxP);
  rcut = npz_get_cutoff_scalar(npz, "rcut", 6.0, "GRACE-1L/KK");
  has_bond_specific_cutoff = npz_get_flag(npz, "has_bond_specific_cutoff", 0);
  // a chem-embedded radial MLP (pair-dependent input) is implemented by none of the GRACE-1L engines
  require_eq("mlp_rad_has_chem_emb", npz_get_int_scalar(npz, "mlp_rad_has_chem_emb", 0), 0,
             "chem-embedded radial MLPs are not supported");
  const int64_t nlm = (int64_t) (lmax + 1) * (lmax + 1);    // A and A1 functions

  // Elements: n_elements against the element names and the embedding rows before it sizes anything
  element_names = npz_names(npz, "element_names");
  chem_embedding = npz_require_double(npz, "chem_embedding");
  {
    const auto ce = npz_shape_rank(npz, "chem_embedding", 2);
    if ((long) ce[0] != n_elements && (element_names.empty() || element_names.size() == ce[0]))
      throw bad_array("n_elements", "is " + std::to_string(n_elements) + ", but chem_embedding has " +
                                        std::to_string(ce[0]) + " rows" +
                                        (element_names.empty() ? "" : " and element_names as many names"));
    if (!element_names.empty()) require_size("element_names", element_names, n_elements);
    require_shape(npz, "chem_embedding", {n_elements, embedding_size});
  }
  bond_cutoff_map = npz_get_cutoff(npz, "bond_cutoff_map", "GRACE-1L/KK");
  require_shape(npz, "bond_cutoff_map", {n_elements, n_elements});
  // If no bond_cutoff_map provided, fill uniformly with rcut
  if (bond_cutoff_map.empty()) bond_cutoff_map.assign((size_t) n_elements * n_elements, rcut);

  // A indicator transform [embedding_size, nradmax]
  A_lin_transform_W = npz_require_double(npz, "A_lin_transform_W");
  require_shape(npz, "A_lin_transform_W", {embedding_size, nradmax});
  A_lin_transform_norm = npz_get_double_scalar(npz, "A_lin_transform_norm", 1.0);
  A_inv_avg_n_neigh = npz_get_double_scalar(npz, "A_inv_avg_n_neigh", 1.0);

  // an MLP of n layers (npz_mlp): n_in inputs, n_out outputs
  auto load_mlp = [&](const std::string &pfx, int n, std::vector<MLPLayer> &layers, int64_t n_in, int64_t n_out,
                      const std::string &in_why, const std::string &out_why) {
    for (auto &l : npz_mlp(npz, pfx, n, n_in, n_out, in_why, out_why))
      layers.push_back({std::move(l.W), l.norm, l.n_in, l.n_out});
  };

  // MLP radial layers: nradbase radial basis functions in, nradmax x (lmax + 1) radial functions out
  mlp_rad_n_layers = npz_get_int_in(npz, "mlp_rad_n_layers", 3, 1, kMaxDim);
  load_mlp("mlp_rad", mlp_rad_n_layers, mlp_rad_layers, nradbase, (int64_t) nradmax * (lmax + 1), "nradbase",
           "nradmax x (lmax + 1)");

  // FC weights: left [n_out, n_in_left, tiles] per left function, right [n_out, nradmax, tiles] per connection
  // collect_from (an A function) -> collect_to (an output function)
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
    // left_coefs = 0 (identity left path, as the 3L FCRight2Left) is not what the 1L engines compute
    require_eq(prefix + "_left_coefs", npz_get_int_scalar(npz, prefix + "_left_coefs", 1), 1,
               "the GRACE-1L engines apply left weights in every FC");
    fc.n_funcs_left = (int) fc.w_tile_left.size();
    fc.n_funcs_right = (int) fc.collect_from.size();
    fc.w_shape_left = (int) npz_shape_rank(npz, prefix + "_w_left", 3)[2];
    fc.w_shape_right = (int) npz_shape_rank(npz, prefix + "_w_right", 3)[2];
  };
  // n_funcs: output (= left input) functions, n_in_left: channels of the left input
  auto check_fc = [&](const std::string &prefix, const FCWeights &fc, int64_t n_funcs, int64_t n_in_left) {
    // the CUDA cuBLAS FC path tiles both sides alike
    require_eq(prefix + "_w_right", fc.w_shape_right, fc.w_shape_left, "the tiles of " + prefix + "_w_left", "tiles");
    require_shape(npz, prefix + "_w_left", {fc.n_out, n_in_left, fc.w_shape_left});
    require_shape(npz, prefix + "_w_right", {fc.n_out, nradmax, fc.w_shape_right});
    require_size(prefix + "_w_tile_left", fc.w_tile_left, n_funcs);
    require_size(prefix + "_norm_out_factor", fc.norm_out_factor, n_funcs);
    require_index(prefix + "_w_tile_left", fc.w_tile_left, fc.w_shape_left);
    const size_t nc = fc.collect_from.size();
    require_size(prefix + "_collect_to", fc.collect_to, nc);
    require_size(prefix + "_w_tile_right", fc.w_tile_right, nc);
    require_index(prefix + "_collect_from", fc.collect_from, nlm);
    require_index(prefix + "_collect_to", fc.collect_to, n_funcs);
    require_index(prefix + "_w_tile_right", fc.w_tile_right, fc.w_shape_right);
  };

  load_fc("fc_A1", fc_A1);
  load_fc("fc_AA1", fc_AA1);
  load_fc("fc_AA2", fc_AA2);

  // CG products: out[m_sum_ind[t]] += cg_coeff[t] * left[left_ind[t]] * right[right_ind[t]]
  auto load_prod = [&](const std::string &prefix, CGProduct &prod) {
    prod.left_ind = npz_require_int(npz, prefix + "_left_ind");
    prod.right_ind = npz_require_int(npz, prefix + "_right_ind");
    prod.m_sum_ind = npz_require_int(npz, prefix + "_m_sum_ind");
    prod.cg_coeff = npz_require_double(npz, prefix + "_cg_coeff");
    prod.n_output_funcs = npz_get_int_in(npz, prefix + "_n_output_funcs", 0, 1, kMaxDim);
    prod.n_cg_terms = npz_get_int_in(npz, prefix + "_n_cg_terms", 0, 1, kMaxDim);
  };
  auto check_prod = [&](const std::string &prefix, const CGProduct &prod, int64_t n_left, int64_t n_right) {
    const size_t nt = prod.n_cg_terms;
    require_size(prefix + "_left_ind", prod.left_ind, nt);
    require_size(prefix + "_right_ind", prod.right_ind, nt);
    require_size(prefix + "_m_sum_ind", prod.m_sum_ind, nt);
    require_size(prefix + "_cg_coeff", prod.cg_coeff, nt);
    require_index(prefix + "_left_ind", prod.left_ind, n_left);
    require_index(prefix + "_right_ind", prod.right_ind, n_right);
    require_index(prefix + "_m_sum_ind", prod.m_sum_ind, prod.n_output_funcs);
    // the CPU engine counts the outputs from the CG terms, the Kokkos ones take n_output_funcs: they must agree
    require_eq(prefix + "_n_output_funcs", prod.n_output_funcs, index_extent(prod.m_sum_ind),
               "1 + the largest " + prefix + "_m_sum_ind");
  };

  load_prod("prod_AA", prod_AA);
  load_prod("prod_AAA", prod_AAA);
  load_prod("prod_AAAA", prod_AAAA);

  // A1 = FC(A, A) [nlm functions], AA = A1 x A1, AA1/AA2 = FC(AA, A) [nAA functions], AAA = AA1 x A1,
  // AAAA = AA2 x AA2; every FC block has the same width (the products are channel-wise)
  const int64_t nAA = prod_AA.n_output_funcs, CF = fc_A1.n_out;
  require_eq("fc_AA1_n_out", fc_AA1.n_out, CF, "fc_A1_n_out");
  require_eq("fc_AA2_n_out", fc_AA2.n_out, CF, "fc_A1_n_out");
  check_fc("fc_A1", fc_A1, nlm, nradmax);
  check_prod("prod_AA", prod_AA, nlm, nlm);
  check_fc("fc_AA1", fc_AA1, nAA, CF);
  check_fc("fc_AA2", fc_AA2, nAA, CF);
  check_prod("prod_AAA", prod_AAA, nAA, nlm);
  check_prod("prod_AAAA", prod_AAAA, nAA, nAA);

  // FunctionReduceN: rho[k] = sum over sources s and collected functions c of
  // W_s[mu, k, n, w_l_tile[c]] * X_s[collect_ind[c]][n]
  rho_n_out = npz_get_int_in(npz, "rho_n_out", 17, 2, kMaxDim);    // rho[0] and the energy MLP input rho[1:]
  // the engines read the weights per central element (the layout of W)
  require_eq("rho_is_central_atom_type_dependent", npz_get_int_scalar(npz, "rho_is_central_atom_type_dependent", 1), 1,
             "the GRACE-1L engines implement this layout only");

  // n_in: channels of the source, n_funcs: its function count
  auto load_reduce = [&](const std::string &instr_name, ReduceWeights &rw, int64_t n_in, int64_t n_funcs) {
    const std::string wkey = "rho_reduce_" + instr_name + "_W", ckey = "rho_collect_ind_" + instr_name,
                      tkey = "rho_w_l_tile_" + instr_name;
    rw.W = npz_require_double(npz, wkey);
    rw.norm = npz_get_double_scalar(npz, "rho_reduce_" + instr_name + "_norm", 1.0);
    rw.collect_ind = npz_require_int(npz, ckey);
    rw.w_l_tile = npz_require_int(npz, tkey);
    rw.n_in = npz_get_int_scalar(npz, "rho_n_in_" + instr_name, 0);
    rw.w_shape = npz_get_int_in(npz, "rho_w_shape_" + instr_name, 0, 1, kMaxDim);
    require_eq("rho_n_in_" + instr_name, rw.n_in, n_in, "the channels of " + instr_name);
    require_shape(npz, wkey, {n_elements, rho_n_out, rw.n_in, rw.w_shape});
    require_size(tkey, rw.w_l_tile, rw.collect_ind.size());
    require_index(ckey, rw.collect_ind, n_funcs);
    require_index(tkey, rw.w_l_tile, rw.w_shape);
  };

  load_reduce("A", reduce_A, nradmax, nlm);
  load_reduce("AA", reduce_AA, CF, nAA);
  load_reduce("AAA", reduce_AAA, CF, prod_AAA.n_output_funcs);
  load_reduce("AAAA", reduce_AAAA, CF, prod_AAAA.n_output_funcs);

  // Energy MLP: rho[1:] in, one value out
  load_mlp("energy_mlp", npz_get_int_in(npz, "energy_mlp_n_layers", 2, 1, kMaxDim), energy_mlp_layers, rho_n_out - 1,
           1, "rho_n_out - 1", "one energy");
  energy_mlp_activation = npz_get_int_in(npz, "energy_mlp_activation", 0, 0, 1);    // 0=silu, 1=tanh

  // Shifts
  shift_values = npz_get_double(npz, "shift_values");
  // a model without a shift instruction exports no shift_values: zero shift per element (the energy readout
  // indexes shift_values by element, so an empty array was read out of bounds)
  if (npz.count("shift_values")) require_size("shift_values", shift_values, n_elements);
  else shift_values.assign(n_elements, 0.0);
  output_scale = npz_get_double_scalar(npz, "output_scale", 1.0);

  // the sources of the rho reduce (their order matters to the UQ basis only: checked by validate_uq)
  rho_instruction_names = npz_names(npz, "rho_instruction_names");
  if (npz.count("rho_instruction_names"))
    require_name_set("rho_instruction_names", rho_instruction_names, {"A", "AA", "AAA", "AAAA"});

  // ---- UQ / extrapolation-grade artifacts (optional; schema v6 basis-RP) ----
  // Present when the model was exported with --uq-artifacts. This build is
  // exclusive to uqv6 (schema 6: L2-normalize + log-norm density channels); older
  // v3/v4 artifacts are rejected at the gate so there is no silent miscompute.
  // A broken UQ head leaves the model usable without UQ: its errors are kept in uq_error (thrown by validate_uq).
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
      throw Error("GRACE-1L/KK UQ: schema v6 expects normalize=1, "
          "add_density_channel=1, feature_transform=0 (identity); got normalize=" +
          std::to_string(uq_normalize) + ", add_density=" + std::to_string(uq_add_density) +
          ", transform=" + std::to_string(uq_transform) + ". Artifact mislabeled or unsupported variant.");
    uq_n_density = uq_feature_dim - uq_rp_dim;   // = 1 + n_blocks (=2 for 1L; checked by validate_uq)
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
        throw Error("GRACE-1L/KK UQ: element index " + std::to_string(e) +
            " has no GMM clusters; uqv6 artifacts must cover all elements. Re-export the UQ "
            "artifacts with a current grace_utils.");
      } else if (uq_n_clusters[e] > uq_max_clusters) {    // every GMM loop indexes the arrays by cluster
        throw Error("GRACE-1L/KK UQ: element index " + std::to_string(e) + " has " + std::to_string(uq_n_clusters[e]) +
            " GMM clusters, more than uq_max_clusters = " + std::to_string(uq_max_clusters));
      }
    // (non-finite values in the GMM arrays and R are rejected by npz_get_double)
  } catch (const Error &e) {    // reported by validate_uq(), as a UQ-artifact error
    uq_error = e.what();
    if (uq_error.rfind("GRACE-1L/KK UQ: ", 0) != 0) uq_error = "GRACE-1L/KK UQ: " + uq_error;
  }
}

void GRACE1LModel::validate_uq() const
{
  if (!has_uq) throw Error("GRACE-1L: the model has no UQ head");
  if (uq_schema_version != 6)
    throw Error("GRACE-1L: UQ schema v" + std::to_string(uq_schema_version) +
                " unsupported (expects v6, uqv6 basis-RP normalize+density). Re-export the UQ artifacts at schema v6 "
                "with a current grace_utils.");
  if (!uq_error.empty()) throw Error(uq_error);
  check_uq_layout("GRACE-1L", uq_layout(), uq_d_basis, uq_n_density, uq_n_elements, n_elements, {rho_instruction_names});
}

std::vector<UqBlock> GRACE1LModel::uq_layout() const
{
  std::vector<UqBlock> L;
  int64_t off = 0;
  {
    const ReduceWeights *rs[4] = {&reduce_A, &reduce_AA, &reduce_AAA, &reduce_AAAA};
    L.push_back(make_uq_block("rho", rs, off));
    off += L.back().width;
  }
  return L;
}


}    // namespace grace

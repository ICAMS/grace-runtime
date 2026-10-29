// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// GRACE3LModel::load: the GRACE-3L model from kokkos.npz.
#include "grace/model_3l.h"
#include "grace/error.h"
#include "npz3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace grace {

using namespace npz;

namespace {

constexpr int kLayers = 3;    // SPBF layers of GRACE-3L

bool has(const cnpy::npz_t &npz, const std::string &key) { return npz.find(key) != npz.end(); }

// a size: in [1, kMaxDim], required
int get_size(const cnpy::npz_t &npz, const std::string &key) { return npz_require_int_in(npz, key, 1, kMaxDim); }

// The GRACE-3L dataflow, as both engines (Grace3LFused, Grace3LKK) resolve it by name. Per layer k = 1, 2, 3
// (K = A1, A2, A3), X are the layer's lm features: A1 the scalar SPBF itself; A2, A3 the reduce K_red of the
// equivariant SPBF K, which couples the message eq{k-1}_norm of the layer before:
//   K_2 = cp_l(X, X), K_2_red = reduce(K_2), K_2a = fc(K_2_red, X), K_3 = cp_l(K_2a, X), K_3_red = reduce(K_3),
//   K_2b = fc(K_2_red, X), K_4 = cp_l(K_2b, K_2b), K_4_red = reduce(K_4),
//   eq{k} = reduce(X, K_2_red, K_3_red, K_4_red) -> eq{k}_norm (k < 3), rho{k} = reduce(the same) -> rho{k}_norm,
//   energy = MLP(sum of out1_origins = rho1_norm, rho2_norm, rho3_norm; channel 0 a linear skip).
// Each component is checked against the functions and channels of its inputs; a component outside this graph, or
// one missing, is an error. Returns the channels of the densities summed by the readout.
int validate_layers(GRACE3LModel &m, bool has_origin_n_out)
{
  struct Shape {
    int nf, nc;    // functions (lm), channels
  };
  std::map<std::string, Shape> T;
  std::vector<char> us(m.spbf.size()), up(m.prod.size()), ur(m.reduce.size()), uf(m.fc.size()),
      ue(m.eqnorm.size()), un(m.rmsnorm.size());
  // component `name` of the list key `list` (once), marked used
  auto find = [](auto &vec, std::vector<char> &used, const char *list, const std::string &name) -> auto & {
    int at = -1;
    for (size_t i = 0; i < vec.size(); i++)
      if (vec[i].name == name) {
        if (at >= 0) throw bad_array(list, "lists '" + name + "' twice");
        at = (int) i;
      }
    if (at < 0) throw bad_array(list, "has no '" + name + "', a part of the GRACE-3L architecture");
    used[at] = 1;
    return vec[at];
  };
  auto unused = [](auto &vec, const std::vector<char> &used, const char *list) {
    for (size_t i = 0; i < vec.size(); i++)
      if (!used[i]) throw bad_array(list, "lists '" + vec[i].name + "', not a part of the GRACE-3L architecture");
  };
  auto red = [&](const std::string &name, const std::vector<std::string> &inputs) {
    auto &r = find(m.reduce, ur, "reduce_names", name);
    std::vector<std::string> got;
    for (const auto &ins : r.instr) got.push_back(ins.name);
    require_name_set(name + "_instruction_names", got, inputs);
    for (const auto &ins : r.instr) {
      const Shape in = T.at(ins.name);
      require_eq(name + "_n_in_" + ins.name, ins.n_in, in.nc, "the channels of " + ins.name);
      require_index(name + "_collect_ind_" + ins.name, ins.collect_ind, in.nf);
    }
    T[name] = {r.only_invar ? 1 : r.n_funcs, r.n_out};
  };
  auto cpl = [&](const std::string &name, const std::string &left, const std::string &right) {
    auto &c = find(m.prod, up, "prod_names", name);
    const Shape l = T.at(left), r = T.at(right);
    require_eq(name + "_group_left", c.group_left.size(), l.nf, "the functions of " + left, "values");
    require_eq(name + "_n_left", c.n_left, l.nc, "the channels of " + left);
    require_eq(name + "_group_right", c.group_right.size(), r.nf, "the functions of " + right, "values");
    require_eq(name + "_n_right", c.n_right, r.nc, "the channels of " + right);
    T[name] = {c.nfunc, c.rank};
  };
  auto fcl = [&](const std::string &name, const std::string &left, const std::string &right) {
    auto &f = find(m.fc, uf, "fc_names", name);
    const std::string p = "fc_" + name;
    const Shape l = T.at(left), r = T.at(right);
    require_eq(p + "_w_tile_left", f.n_funcs_left, l.nf, "the functions of " + left, "values");
    if (f.left_coefs)
      require_eq(p + "_w_left", f.n_in_left, l.nc, "the channels of " + left, "columns");
    else    // out = norm_out_factor (left + W right): the left input passes with its channels
      require_eq(p + "_n_out", f.n_out, l.nc, "the channels of " + left);
    require_eq(p + "_w_right", f.n_in_right, r.nc, "the channels of " + right, "columns");
    require_index(p + "_collect_from", f.collect_from, r.nf);
    T[name] = {f.n_funcs_left, f.n_out};
  };

  if ((int) m.spbf.size() != kLayers)
    throw bad_array("spbf_names", "has " + std::to_string(m.spbf.size()) + " names, expected 3 (A1, A2, A3)");
  const int nlm = (m.lmax + 1) * (m.lmax + 1);
  for (int k = 1; k <= kLayers; k++) {
    const std::string ks = std::to_string(k), K = "A" + ks;
    auto &s = find(m.spbf, us, "spbf_names", K);
    if (s.name != m.spbf[k - 1].name) throw bad_array("spbf_names", "lists '" + K + "' out of order");
    std::string X = K;
    if (k == 1) {
      if (s.equivariant) throw bad_array(K + "_equivariant", "is 1, expected 0 (the first layer has no message to couple)");
      T[K] = {nlm, s.n_rad_max};
    } else {
      if (!s.equivariant)
        throw bad_array(K + "_equivariant", "is 0, expected 1 (the layer couples the message of the one before)");
      const std::string msg = "eq" + std::to_string(k - 1) + "_norm";
      require_eq(K + "_n_lm_ind", s.n_lm_ind, T.at(msg).nf, "the functions of the message " + msg);
      require_eq(K + "_n_rad_max", s.n_rad_max, T.at(msg).nc, "the channels of the message " + msg);
      T[K] = {s.nfunc, s.n_rad_max};
      red(K + "_red", {K});
      X = K + "_red";
    }
    cpl(K + "_2", X, X);
    red(K + "_2_red", {K + "_2"});
    fcl(K + "_2a", K + "_2_red", X);
    cpl(K + "_3", K + "_2a", X);
    red(K + "_3_red", {K + "_3"});
    fcl(K + "_2b", K + "_2_red", X);
    cpl(K + "_4", K + "_2b", K + "_2b");
    red(K + "_4_red", {K + "_4"});
    const std::vector<std::string> collect = {X, K + "_2_red", K + "_3_red", K + "_4_red"};
    if (k < kLayers) {
      red("eq" + ks, collect);
      auto &e = find(m.eqnorm, ue, "eqnorm_names", "eq" + ks + "_norm");
      const Shape in = T.at("eq" + ks);
      require_eq(e.name + "_degree_weights", e.M, in.nf, "the functions of eq" + ks, "values");
      require_eq(e.name + "_affine_weight", e.n_out, in.nc, "the channels of eq" + ks, "columns");
      T[e.name] = {e.M, e.n_out};
    }
    red("rho" + ks, collect);
    auto &n = find(m.rmsnorm, un, "rmsnorm_names", "rho" + ks + "_norm");
    const Shape in = T.at("rho" + ks);
    require_eq(n.name + "_n_out", n.n_out, in.nc, "the channels of rho" + ks);
    T[n.name] = {in.nf, n.n_out};
  }
  unused(m.spbf, us, "spbf_names");
  unused(m.prod, up, "prod_names");
  unused(m.reduce, ur, "reduce_names");
  unused(m.fc, uf, "fc_names");
  unused(m.eqnorm, ue, "eqnorm_names");
  unused(m.rmsnorm, un, "rmsnorm_names");

  // readout: the densities of all layers, summed channel by channel
  std::vector<std::string> want;
  for (int k = 1; k <= kLayers; k++) want.push_back("rho" + std::to_string(k) + "_norm");
  require_name_set("out1_origins", m.out1_origins, want);
  const int w = T.at(want[0]).nc;
  for (const auto &o : want) require_eq(o + "_n_out", T.at(o).nc, w, "the channels of " + want[0]);
  if (has_origin_n_out) require_eq("energy_mlp_origin_n_out", m.energy_mlp_origin_n_out, w, "the channels of out1_origins");
  return w;
}

}    // namespace

// ======================================================================
// GRACE3LModel::load — every array checked against the sizes the model declares (shapes, index ranges, the layer
// dataflow) before anything is built from it: a malformed file is a grace::Error naming the array.
// ======================================================================

void GRACE3LModel::load(const std::string &filepath) { load_arrays(npz_open(filepath)); }

void GRACE3LModel::load_arrays(const cnpy::npz_t &npz) {
  // a second load into the same object must keep nothing of the previous model
  *this = GRACE3LModel();
  check_grace_format(npz, "GRACE-3L");

  // ---- Global metadata ----
  n_elements = get_size(npz, "n_elements");
  embedding_size = npz_get_int_in(npz, "embedding_size", 64, 1, kMaxDim);
  nradbase = npz_get_int_in(npz, "nradbase", 10, 1, kMaxDim);
  // No bare `lmax` key in the 3L npz — the angular lmax is SPBF _lmax (A1_lmax).
  lmax = npz_get_int_in(npz, "lmax", npz_get_int_in(npz, "A1_lmax", 4, 0, kMaxL), 0, kMaxL);
  const int nl = lmax + 1, nlm = nl * nl;
  // The global radial_basis_p key (=16) is NOT the envelope order and is not read; geometry uses the SPBF p (=A1_p=5),
  // set from the SPBF layers below.
  rcut = npz_get_cutoff_scalar(npz, "rcut", 6.0, "GRACE-3L/KK");
  has_bond_specific_cutoff = npz_get_flag(npz, "has_bond_specific_cutoff", 0);

  chem_embedding = npz_require_double(npz, "chem_embedding");
  require_shape(npz, "chem_embedding", {n_elements, embedding_size});
  shift_values = npz_require_double(npz, "shift_values");
  require_size("shift_values", shift_values, n_elements);
  output_scale = npz_get_double_scalar(npz, "output_scale", 1.0);  // absent -> 1.0
  element_names = npz_names(npz, "element_names");
  require_size("element_names", element_names, n_elements);
  bond_cutoff_map = npz_get_cutoff(npz, "bond_cutoff_map", "GRACE-3L/KK");
  require_shape(npz, "bond_cutoff_map", {n_elements, n_elements});
  if (bond_cutoff_map.empty()) bond_cutoff_map.assign((size_t) n_elements * n_elements, rcut);

  // ---- Helper: generic MLP (npz_mlp; n_in inputs, n_out outputs). The radial MLP has a bias b{i} [n_out] on hidden
  //      layers, the energy MLP none. ----
  auto load_mlp = [&](const std::string &prefix, MLP &mlp, int64_t n_in, int64_t n_out, const std::string &in_why,
                      const std::string &out_why) {
    mlp.n_layers = npz_require_int_in(npz, prefix + "_n_layers", 1, kMaxDim);
    for (auto &l : npz_mlp(npz, prefix, mlp.n_layers, n_in, n_out, in_why, out_why)) {
      const std::string bk = prefix + "_b" + std::to_string(mlp.W.size());
      auto b = npz_get_double(npz, bk);
      if (has(npz, bk)) {
        require_size(bk, b, l.n_out);
        mlp.has_bias = true;
      }
      mlp.W.push_back(std::move(l.W));
      mlp.b.push_back(std::move(b));
      mlp.norm.push_back(l.norm);
      mlp.n_in.push_back(l.n_in);
      mlp.n_out.push_back(l.n_out);
    }
  };

  // ---- SPBF (A1 scalar, A2/A3 equivariant) ----
  auto spbf_names = npz_names(npz, "spbf_names");
  spbf.clear();
  spbf.reserve(spbf_names.size());
  for (const auto &nm : spbf_names) {
    SPBF s;
    s.name = nm;
    s.equivariant = npz_get_flag(npz, nm + "_equivariant", 0);
    s.n_rad_max = get_size(npz, nm + "_n_rad_max");
    s.n_rad_basis = npz_get_int_scalar(npz, nm + "_n_rad_basis", nradbase);
    require_eq(nm + "_n_rad_basis", s.n_rad_basis, nradbase, "nradbase, the radial MLP's Chebyshev inputs");
    s.lmax = npz_get_int_scalar(npz, nm + "_lmax", 4);
    require_eq(nm + "_lmax", s.lmax, lmax, "the model's lmax");
    s.Lmax = npz_get_int_in(npz, nm + "_Lmax", 4, 0, kMaxL);
    // envelope polynomial order: p >= 1 (x^(p-1) in its derivative)
    s.p = npz_get_int_in(npz, nm + "_p", 5, 1, kMaxP);
    s.rcut = npz_get_cutoff_scalar(npz, nm + "_rcut", rcut, "GRACE-3L/KK");
    s.inv_avg_n_neigh = npz_get_double_scalar(npz, nm + "_inv_avg_n_neigh", 1.0);
    // l of each spherical harmonic lm: selects the radial function n (lmax+1) + l
    s.l_tile = npz_require_int(npz, nm + "_l_tile");
    require_size(nm + "_l_tile", s.l_tile, nlm);
    require_index(nm + "_l_tile", s.l_tile, nl);
    // radial MLP: [Chebyshev basis, embedding(mu_i), embedding(mu_j)] -> n_rad_max (lmax+1) radial functions
    load_mlp(nm + "_mlp", s.mlp, nradbase + 2 * embedding_size, (int64_t) s.n_rad_max * nl,
             "nradbase + 2 embedding_size", "n_rad_max (lmax+1)");
    if (!s.equivariant) {
      s.lin_transform_W = npz_require_double(npz, nm + "_lin_transform_W");
      require_shape(npz, nm + "_lin_transform_W", {embedding_size, s.n_rad_max});
      s.lin_transform_norm = npz_get_double_scalar(npz, nm + "_lin_transform_norm", 1.0);
    } else {
      s.nfunc = get_size(npz, nm + "_nfunc");
      s.n_lm_ind = get_size(npz, nm + "_n_lm_ind");
      s.chem_linear_W = npz_require_double(npz, nm + "_chem_linear_W");
      require_shape(npz, nm + "_chem_linear_W", {embedding_size, s.n_rad_max});
      s.chem_linear_norm = npz_get_double_scalar(npz, nm + "_chem_linear_norm", 1.0);
      s.chem_l0_mask = npz_require_double(npz, nm + "_chem_l0_mask");
      require_size(nm + "_chem_l0_mask", s.chem_l0_mask, s.n_lm_ind);
      // dense CG couple of (Y lm) x (message lm_ind) -> nfunc
      s.cg_W = npz_require_double(npz, nm + "_cg_W");
      require_shape(npz, nm + "_cg_W", {(int64_t) nlm * s.n_lm_ind, s.nfunc});
    }
    spbf.push_back(std::move(s));
  }
  // Effective radial ceilings + envelope order from the SPBF layers.
  if (spbf.size() >= 1) { L1_nradmax = spbf[0].n_rad_max; radial_basis_p = spbf[0].p; }
  for (const auto &s : spbf)    // the Kokkos engine evaluates every layer's envelope with radial_basis_p
    if (s.p != radial_basis_p)
      throw bad_array(s.name + "_p", "is " + std::to_string(s.p) + ", the envelope order of " + spbf[0].name + " is " +
                                         std::to_string(radial_basis_p) + "; layers with different orders are not "
                                         "supported");
  if (spbf.size() >= 2) L2_nradmax = spbf[1].n_rad_max;
  if (spbf.size() >= 3) L3_nradmax = spbf[2].n_rad_max;

  // ---- cp_l products ----
  auto prod_names = npz_names(npz, "prod_names");
  prod.clear();
  prod.reserve(prod_names.size());
  for (const auto &nm : prod_names) {
    CPL c;
    c.name = nm;
    c.rank = get_size(npz, nm + "_rank");
    c.nfunc = get_size(npz, nm + "_nfunc");
    c.n_left = get_size(npz, nm + "_n_left");
    c.n_right = get_size(npz, nm + "_n_right");
    c.n_groups_left = get_size(npz, nm + "_n_groups_left");
    c.n_groups_right = get_size(npz, nm + "_n_groups_right");
    c.norm_u = npz_get_double_scalar(npz, nm + "_norm_u", 1.0);
    c.norm_v = npz_get_double_scalar(npz, nm + "_norm_v", 1.0);
    c.U = npz_require_double(npz, nm + "_U");
    require_shape(npz, nm + "_U", {c.n_groups_left, c.rank, c.n_left});
    c.V = npz_require_double(npz, nm + "_V");
    require_shape(npz, nm + "_V", {c.n_groups_right, c.rank, c.n_right});
    c.cg = npz_require_double(npz, nm + "_cg");
    c.n_cg = (int) c.cg.size();
    // lm functions of the left / right input -> projection group (their count is checked against the inputs below)
    c.group_left = npz_require_int(npz, nm + "_group_left");
    require_index(nm + "_group_left", c.group_left, c.n_groups_left);
    c.group_right = npz_require_int(npz, nm + "_group_right");
    require_index(nm + "_group_right", c.group_right, c.n_groups_right);
    // CG terms: left function x right function -> output function
    c.left_ind = npz_require_int(npz, nm + "_left_ind");
    require_size(nm + "_left_ind", c.left_ind, c.n_cg);
    require_index(nm + "_left_ind", c.left_ind, c.group_left.size());
    c.right_ind = npz_require_int(npz, nm + "_right_ind");
    require_size(nm + "_right_ind", c.right_ind, c.n_cg);
    require_index(nm + "_right_ind", c.right_ind, c.group_right.size());
    c.m_sum_ind = npz_require_int(npz, nm + "_m_sum_ind");
    require_size(nm + "_m_sum_ind", c.m_sum_ind, c.n_cg);
    require_index(nm + "_m_sum_ind", c.m_sum_ind, c.nfunc);
    // l of each output function
    c.out_l = npz_require_int(npz, nm + "_out_l");
    require_size(nm + "_out_l", c.out_l, c.nfunc);
    require_index(nm + "_out_l", c.out_l, kMaxL + 1);
    prod.push_back(std::move(c));
  }

  // ---- FunctionReduceN (product reduces + eq*/rho*) ----
  auto reduce_names = npz_names(npz, "reduce_names");
  reduce.clear();
  reduce.reserve(reduce_names.size());
  for (const auto &nm : reduce_names) {
    Reduce r;
    r.name = nm;
    r.n_out = get_size(npz, nm + "_n_out");
    r.n_funcs = get_size(npz, nm + "_n_funcs");
    r.only_invar = npz_get_flag(npz, nm + "_only_invar", 0);
    r.elem_dep = npz_get_flag(npz, nm + "_is_central_atom_type_dependent", 0);
    const int nfo = r.only_invar ? 1 : r.n_funcs;    // output functions
    auto instr_names = npz_names(npz, nm + "_instruction_names");
    r.instr.reserve(instr_names.size());
    for (const auto &cn : instr_names) {
      ReduceInstr ri;
      ri.name = cn;
      ri.n_in = get_size(npz, nm + "_n_in_" + cn);
      ri.w_shape = get_size(npz, nm + "_w_shape_" + cn);
      const std::string wk = nm + "_reduce_" + cn + "_W";
      ri.W = npz_require_double(npz, wk);
      if (r.elem_dep) require_shape(npz, wk, {n_elements, r.n_out, ri.n_in, ri.w_shape});
      else require_shape(npz, wk, {r.n_out, ri.n_in, ri.w_shape});
      ri.norm = npz_get_double_scalar(npz, nm + "_reduce_" + cn + "_norm", 1.0);
      // connections: input function collect_ind (range checked against the input below), weight tile, output function
      ri.collect_ind = npz_require_int(npz, nm + "_collect_ind_" + cn);
      ri.n_conn = (int) ri.collect_ind.size();
      ri.w_l_tile = npz_require_int(npz, nm + "_w_l_tile_" + cn);
      require_size(nm + "_w_l_tile_" + cn, ri.w_l_tile, ri.n_conn);
      require_index(nm + "_w_l_tile_" + cn, ri.w_l_tile, ri.w_shape);
      ri.total_sum_ind = npz_require_int(npz, nm + "_total_sum_ind_" + cn);  // [n_conn,1] flat
      require_size(nm + "_total_sum_ind_" + cn, ri.total_sum_ind, ri.n_conn);
      require_index(nm + "_total_sum_ind_" + cn, ri.total_sum_ind, nfo);
      r.instr.push_back(std::move(ri));
    }
    // EXACT key — `{name}_norm_map` (NOT the RMS-norm `{name}_norm` footgun).
    r.norm_map = npz_get_double(npz, nm + "_norm_map");
    r.has_norm_map = has(npz, nm + "_norm_map");
    if (r.has_norm_map) require_size(nm + "_norm_map", r.norm_map, nfo);
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
    f.n_out = get_size(npz, p + "_n_out");
    f.left_coefs = npz_get_flag(npz, p + "_left_coefs", 0);
    f.w_right = npz_require_double(npz, p + "_w_right");
    {
      const auto wr = npz_shape_rank(npz, p + "_w_right", 3);    // [n_out, n_in_right, w_shape_right]
      require_eq(p + "_w_right", (long) wr[0], f.n_out, p + "_n_out", "rows");
      f.n_in_right = (int) wr[1];
      f.w_shape_right = (int) wr[2];
    }
    f.norm_right = npz_get_double_scalar(npz, p + "_norm_right", 1.0);
    // left functions (their count n_funcs_left is checked against the left input below); their weight tiles are
    // applied with left_coefs = 1 only (read below), every GRACE-3L export has 0: then only their count is used
    f.n_funcs_left = (int) npz_shape_rank(npz, p + "_w_tile_left", 1)[0];
    f.norm_out_factor = npz_require_double(npz, p + "_norm_out_factor");  // [n_funcs_left,1,1] flat
    require_size(p + "_norm_out_factor", f.norm_out_factor, f.n_funcs_left);
    // right connections: right function collect_from (range checked against the right input below) -> left function
    // collect_to with weight tile w_tile_right
    f.collect_from = npz_require_int(npz, p + "_collect_from");
    f.n_funcs_right = (int) f.collect_from.size();
    f.collect_to = npz_require_int(npz, p + "_collect_to");
    require_size(p + "_collect_to", f.collect_to, f.n_funcs_right);
    require_index(p + "_collect_to", f.collect_to, f.n_funcs_left);
    f.w_tile_right = npz_require_int(npz, p + "_w_tile_right");
    require_size(p + "_w_tile_right", f.w_tile_right, f.n_funcs_right);
    require_index(p + "_w_tile_right", f.w_tile_right, f.w_shape_right);
    if (f.left_coefs) {   // 3L FCs are right->left (left_coefs=False); guard anyway
      f.w_tile_left = npz_require_int(npz, p + "_w_tile_left");
      require_size(p + "_w_tile_left", f.w_tile_left, f.n_funcs_left);
      f.w_left = npz_require_double(npz, p + "_w_left");
      const auto wl = npz_shape_rank(npz, p + "_w_left", 3);
      require_eq(p + "_w_left", (long) wl[0], f.n_out, p + "_n_out", "rows");
      f.n_in_left = (int) wl[1];
      f.w_shape_left = (int) wl[2];
      f.norm_left = npz_get_double_scalar(npz, p + "_norm_left", 1.0);
      require_index(p + "_w_tile_left", f.w_tile_left, f.w_shape_left);
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
    {
      const auto aw = npz_shape_rank(npz, nm + "_affine_weight", 2);    // [n_groups, n_out]
      e.n_groups = (int) aw[0];
      e.n_out = (int) aw[1];
    }
    e.degree_weights = npz_require_double(npz, nm + "_degree_weights");
    e.M = (int) e.degree_weights.size();
    e.l0_mask = npz_require_double(npz, nm + "_l0_mask");
    require_size(nm + "_l0_mask", e.l0_mask, e.M);
    // lm function -> its affine-weight group
    e.expand_index = npz_require_int(npz, nm + "_expand_index");
    require_size(nm + "_expand_index", e.expand_index, e.M);
    require_index(nm + "_expand_index", e.expand_index, e.n_groups);
    e.eps = npz_get_double_scalar(npz, nm + "_eps", 1e-8);
    if (e.eps < 0.0) throw bad_array(nm + "_eps", "is negative");
    e.center_l0 = npz_get_flag(npz, nm + "_center_l0", 1);
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
    r.type = npz_get_flag(npz, nm + "_type", 0);    // 0: full (scale [n_out]), 1: only_nonlin (channel 0 passes, [n_out-1])
    r.n_out = npz_get_int_in(npz, nm + "_n_out", (int) std::min<size_t>(r.scale.size() + r.type, kMaxDim), 1, kMaxDim);
    require_size(nm + "_scale", r.scale, r.n_out - r.type);
    rmsnorm.push_back(std::move(r));
  }

  // ---- 3-origin metadata + readout energy MLP (no bias) ----
  out1_origins = npz_names(npz, "out1_origins");
  energy_mlp_origin_n_out = npz_get_int_scalar(npz, "energy_mlp_origin_n_out", 0);
  energy_mlp_n_origins = npz_get_int_scalar(npz, "energy_mlp_n_origins", (int) out1_origins.size());
  require_eq("energy_mlp_n_origins", energy_mlp_n_origins, out1_origins.size(), "the names in out1_origins");

  const int w = validate_layers(*this, has(npz, "energy_mlp_origin_n_out"));
  load_mlp("energy_mlp", energy_mlp, w - 1, 1, "the channels of out1_origins but the linear channel 0",
           "one energy per atom");
  energy_mlp_activation = npz_get_int_in(npz, "energy_mlp_activation", 0, 0, 1);    // 0=silu, 1=tanh

  // ---- UQ / extrapolation-grade artifacts (optional; schema v6 basis-RP): GraceUQHead ----
  load_uq_head(*this, npz, n_elements, 4, "GRACE-3L");
  if (has_uq) {
    // the GMM feature projects the inputs of the rho1 | rho2 | rho3 reduces: R has one row per basis value
    int64_t width = 0;
    for (int k = 1; k <= kLayers; k++)
      for (const auto &r : reduce)
        if (r.name == "rho" + std::to_string(k))
          for (const auto &ins : r.instr) width += (int64_t) ins.n_in * ins.n_conn;
    require_eq("uq_rp_matrix", uq_d_basis, width, "the inputs of the rho1, rho2, rho3 reduces", "rows");
  }
}

}    // namespace grace

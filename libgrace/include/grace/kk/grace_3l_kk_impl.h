// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Implementation of grace::kk::Grace3LKK (see grace_3l_kk.h): model setup, device weights, buffers, the layered
// message exchange and run(); the kernels follow in grace_3l_kk_{radial,spbf,cpl,dense}.h (included at the end).
// Included by one translation unit per consumer, which instantiates the engine for its Traits (LAMMPS:
// the grace/runtime/kk style; python: _kk module).
#pragma once

#include "grace/kk/grace_3l_kk.h"
#include "grace/kk/grace_3l_kk_cpl_path.h"

#include <Kokkos_Sort.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace grace {
namespace kk {
using namespace sph_const;

namespace g3l_detail {
// ======================================================================
// Reorder the output functions of every SPBF / cp_l product whose only consumer is a single-instruction
// element-independent reduce with a bijective collect_ind, so that the connections of each reduce output f
// read contiguous producer columns (in connection order). Producer and reduce are permuted together (cp_l
// m_sum_ind / out_l, SPBF cg_W columns, reduce collect_ind): the model is unchanged as a function,
// every kernel path stays consistent, and the reduce becomes one GEMM per output (compute_reduceN_grouped).
// ======================================================================
inline void group_reduce_inputs(GRACE3LModel &m, const grace::Logger &log)
{
  if (m.reduce_inputs_grouped) return;    // once per model: engines may share it
  m.reduce_inputs_grouped = true;
  int ngrp = 0;
  for (auto &r : m.reduce) {
    if (r.elem_dep || r.only_invar || r.instr.size() != 1) continue;
    auto &ins = r.instr[0];
    const int nc = ins.n_conn;
    GRACE3LModel::CPL *prod = nullptr;
    GRACE3LModel::SPBF *spbf = nullptr;
    for (auto &p : m.prod) if (p.name == ins.name) prod = &p;
    for (auto &sp : m.spbf) if (sp.name == ins.name && sp.equivariant) spbf = &sp;
    const int nf = prod ? prod->nfunc : (spbf ? spbf->nfunc : -1);
    if (nf != nc || (int) ins.collect_ind.size() != nc || (int) ins.total_sum_ind.size() != nc) continue;
    std::vector<int> seen(nf, 0);
    bool ok = true;
    for (int c = 0; c < nc && ok; c++) {
      const int ci = ins.collect_ind[c];
      ok = ci >= 0 && ci < nf && !seen[ci];
      if (ok) seen[ci] = 1;
    }
    if (!ok) continue;
    // new column of producer function ci[c]: outputs in order, connections of an output in file order
    std::vector<int> newcol(nf, -1);
    int pos = 0;
    for (int f = 0; f < r.n_funcs; f++)
      for (int c = 0; c < nc; c++)
        if (ins.total_sum_ind[c] == f) newcol[ins.collect_ind[c]] = pos++;
    if (pos != nf) continue;
    for (int c = 0; c < nc; c++) ins.collect_ind[c] = newcol[ins.collect_ind[c]];
    if (prod) {
      for (auto &x : prod->m_sum_ind) x = newcol[x];
      prod->newcol = newcol;
      auto perm = [&](std::vector<int> &v) {
        if ((int) v.size() != nf) return;
        std::vector<int> w(nf);
        for (int f = 0; f < nf; f++) w[newcol[f]] = v[f];
        v.swap(w);
      };
      perm(prod->out_l);
    } else {
      const size_t P = spbf->cg_W.size() / nf;
      std::vector<double> w(spbf->cg_W.size());
      for (size_t p = 0; p < P; p++)
        for (int f = 0; f < nf; f++) w[p * nf + newcol[f]] = spbf->cg_W[p * nf + f];
      spbf->cg_W.swap(w);
    }
    ngrp++;
  }
  log.info(fmt("[GRACE-3L/KK] {} reduces grouped by output (producer columns reordered)\n", ngrp));
}
}    // namespace g3l_detail

// ======================================================================
// Constructor / Destructor
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::Grace3LKK()
{
  host_flag = Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename DeviceType::memory_space>::accessible ? 1 : 0;
  couple_supported = true;    // geometry stage + scatter_forces_chunk
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::~Grace3LKK()
{
  // the BLAS handle is released by its shared owner (blas_owner) with the last engine copy
}

// ======================================================================
// set_model (LAMMPS: coeff): the model, its reduce-input grouping (once per model), the UQ flag and the type map
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::set_model(std::shared_ptr<GRACE3LModel> model,
                                                                       const std::vector<int> &type2mu)
{
  if (!model) GRACE_KK_ERROR("GRACE-3L/KK: set_model() without a model");
  grace_model = std::move(model);
  realloc_all = true;    // the per-atom buffers have the previous model's dims
  setup_done = false;
  chunk_auto = 0;        // chunksize auto: decided again for the new model's buffers
  g3l_detail::group_reduce_inputs(*grace_model, log);
  // the message indicators and their dims here, not in setup(): callers size their comm buffers with comm_width()
  // before setup() (setup() refuses a model without eq1_norm / eq2_norm)
  const auto eqnorm = [&](const char *name, int &idx, int &n_funcs, int &n_out) {
    idx = -1;
    n_funcs = n_out = 0;
    for (int i = 0; i < (int) grace_model->eqnorm.size() && idx < 0; i++)
      if (grace_model->eqnorm[i].name == name) {
        idx = i;
        n_funcs = grace_model->eqnorm[i].n_out;
        n_out = grace_model->eqnorm[i].M;
      }
  };
  eqnorm("eq1_norm", idx_eqnorm_eq1_norm, I_n_funcs, I_n_out);
  eqnorm("eq2_norm", idx_eqnorm_eq2_norm, I2_n_funcs, I2_n_out);

  // Publish the UQ availability flag now (LAMMPS: at pair_coeff time) so `fix pair` can extract the
  // gamma/atomic_sigma/gmm_cluster triggers, which it queries at fix-creation time — BEFORE setup() stages the GMM
  // arrays onto the device. The device arrays + dims are set in setup().
  has_uq = grace_model->has_uq;

  {
    auto &m = *grace_model;
    // Loaded-architecture summary.
    int emlp_in  = m.energy_mlp.n_layers > 0 ? m.energy_mlp.n_in.front()  : 0;
    int emlp_hid = m.energy_mlp.n_layers > 1 ? m.energy_mlp.n_out.front() : 0;
    int emlp_out = m.energy_mlp.n_layers > 0 ? m.energy_mlp.n_out.back()  : 0;
    double a1_norm0 = (!m.spbf.empty() && !m.spbf[0].mlp.norm.empty()) ? m.spbf[0].mlp.norm[0] : 0.0;
    double a1_inv_avg = m.spbf.empty() ? 0.0 : m.spbf[0].inv_avg_n_neigh;
    log.info(fmt(
      "[GRACE-3L/KK] GRACE-3L loaded: n_elements={} lmax={} nradbase={} radial_p={} rcut={} | "
      "SPBF={} products={} reduces={} FCs={} eqnorms={} rmsnorms={} | "
      "energy_mlp[{}->{}->{}] act={} origins={} | "
      "A1_inv_avg_n_neigh={} A1_mlp_norm0={} output_scale={}\n",
      m.n_elements, m.lmax, m.nradbase, m.radial_basis_p, m.rcut,
      m.spbf.size(), m.prod.size(), m.reduce.size(), m.fc.size(), m.eqnorm.size(), m.rmsnorm.size(),
      emlp_in, emlp_hid, emlp_out, m.energy_mlp_activation, m.energy_mlp_n_origins,
      a1_inv_avg, a1_norm0, m.output_scale));
  }

  set_type_map(type2mu);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::set_type_map(const std::vector<int> &type2mu)
{
  // atom type -> model element, and the per-type-pair cutoffs (as LAMMPS init_one)
  const int n = (int) type2mu.size();
  check_type2mu(type2mu, n - 1, grace_model->n_elements, "GRACE-3L/KK");
  h_type2model = type2mu;
  MemKK::realloc_kokkos(d_map, "g3l:map", std::max(n, 1));
  auto h_map = Kokkos::create_mirror_view(d_map);
  for (int t = 0; t < n; t++) h_map(t) = type2mu[t];
  Kokkos::deep_copy(d_map, h_map);
  MemKK::realloc_kokkos(k_cutsq, "g3l:cutsq", std::max(n, 1), std::max(n, 1));
  for (int i = 1; i < n; i++)
    for (int j = 1; j < n; j++) {
      const double rc = cutoff(type2mu[i], type2mu[j]);
      k_cutsq.view_host()(i, j) = rc * rc;
    }
  k_cutsq.modify_host();
  k_cutsq.template sync<DeviceType>();
  d_cutsq = k_cutsq.template view<DeviceType>();
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::set_types(const std::vector<int> &type2mu)
{
  if (!setup_done) GRACE_KK_ERROR("GRACE-3L/KK: set_types() before setup()");
  set_type_map(type2mu);
  build_radial_splines();    // the tables hold one block per ordered species pair of the type map
  realloc_all = true;        // the radial-MLP scratch follows the table / GEMM choice
  chunk_auto = 0;            // and so do the work bytes per atom of chunksize auto
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::release_buffers()
{
  if (!setup_done) return;
  restore_fwd_store();
  for (int i = 0; i < 12; i++) {
    st_fwd[i].clear();
    st_base[i] = t_nn_3d();    // the base views of the store (else the chunk-sized buffers outlive the shrink)
  }
  fwd_stored = 0;
  v_buf = decltype(v_buf)();    // the caller's exchange buffer, kept by the last pack / unpack
  // the per-call launch state holds the work buffers of the last kernels it launched (else they outlive the shrink)
  MemKK::drop(force_A_adj, rmsnorm_adj_in, rmsnorm_adj_out, eqnorm_adj_out, eqnorm_adj_in, reduceN_adj_out,
              reduceN_adj_in, fc_adj_out, fc_adj_left, fc_adj_right, cpl_adj_out, cpl_adj_left, cpl_adj_right, cpl_in_left,
              cpl_in_right, cpl_out, spbf_fused_indicator, spbf_fused_out, spbf_bwd_target, reduceN_out, fc_in_left,
              fc_in_right, fc_out, eqnorm_in, eqnorm_out, rmsnorm_in, rmsnorm_out, d_spbf_ydy);
  for (t_nn_3d &v : reduceN_inputs) MemKK::drop(v);
  // the caller's views of the last run() (positions, forces, neighbour list, outputs) and the per-atom type order
  MemKK::drop(x, f, type, d_numneigh, d_neighbors, d_ilist, d_mask, d_eatom, d_vatom, d_cvatom, d_ilist_sorted,
              d_type_key);
  realloc_all = true;
  grow(1, 1);
  grow_global(1);
  h_I_global = decltype(h_I_global)();
  h_grad_I_global = decltype(h_grad_I_global)();
  h_eq2_norm_global = decltype(h_eq2_norm_global)();
  h_eq2_norm_adj = decltype(h_eq2_norm_adj)();
  realloc_all = true;
  chunk_auto = 0;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
typename Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::TypeTables
Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::type_tables() const
{
  TypeTables t;
  t.type2model = h_type2model;
  t.map = d_map;
  t.cutsq = k_cutsq;
  t.spline_on = spline_on;
  t.n_sp = n_sp;
  t.sp_of_mu = d_sp_of_mu;
  t.splR = d_splR;
  std::copy(spl_N, spl_N + 3, t.spl_N);
  std::copy(spl_off, spl_off + 3, t.spl_off);
  t.spl_Ntot = spl_Ntot;
  t.spl_ld = spl_ld;
  t.spl_mus = spl_mus;
  return t;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::use_type_tables(const TypeTables &t)
{
  if (!setup_done) GRACE_KK_ERROR("GRACE-3L/KK: use_type_tables() before setup()");
  if (t.spline_on != spline_on) {    // tables <-> GEMM MLP: other scratch, other work bytes per atom
    realloc_all = true;
    chunk_auto = 0;
  }
  h_type2model = t.type2model;
  d_map = t.map;
  k_cutsq = t.cutsq;
  d_cutsq = k_cutsq.template view<DeviceType>();
  spline_on = t.spline_on;
  n_sp = t.n_sp;
  d_sp_of_mu = t.sp_of_mu;
  d_splR = t.splR;    // with its shape (a later setup may have built tables of another shape)
  std::copy(t.spl_N, t.spl_N + 3, spl_N);
  std::copy(t.spl_off, t.spl_off + 3, spl_off);
  spl_Ntot = t.spl_Ntot;
  spl_ld = t.spl_ld;
  spl_mus = t.spl_mus;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
double Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::cutoff(int mu_i, int mu_j) const
{
  return pair_cutoff(*grace_model, mu_i, mu_j);
}

// ======================================================================
// setup (LAMMPS: setup())
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::setup()
{
  if (!grace_model) GRACE_KK_ERROR("GRACE-3L/KK: setup() before set_model()");
  setup_done = false;
  for (int si = 0; si < 3; si++)
    if (grace::spline_n_3l(spline_n, spline_n_layer, si) < 16)
      GRACE_KK_ERROR("GRACE-3L/KK: spline_n of layer {} must be >= 16 (got {})", si + 1,
                     grace::spline_n_3l(spline_n, spline_n_layer, si));
  realloc_all = true;    // settings may have changed (spline_mb: tables <-> GEMM MLP, whose scratch sizes differ)
  chunk_auto = 0;
  auto &m = *grace_model;

  // Copy scalar architecture params
  nelements = m.n_elements;
  lmax = m.lmax;
  nradbase = m.nradbase;
  embedding_size = m.embedding_size;
  radial_basis_p = m.radial_basis_p;
  rcut = m.rcut;

  L1_nradmax = m.L1_nradmax;
  L2_nradmax = m.L2_nradmax;
  L3_nradmax = m.L3_nradmax;

  // ---- Validate loaded dims against compile-time caps / stack bounds ----
  {
    const int nlm = (lmax + 1) * (lmax + 1);
    const int plm_size = (lmax + 1) * (lmax + 2) / 2;
    if (nlm > 25)
      GRACE_KK_ERROR("GRACE-3L/KK: (lmax+1)^2={} exceeds stack bound 25 (lmax={})", nlm, lmax);
    if (plm_size > 15)
      GRACE_KK_ERROR("GRACE-3L/KK: plm size {} exceeds stack bound 15 (lmax={})", plm_size, lmax);
    if (lmax > GRACE3L_LMAX)
      GRACE_KK_ERROR("GRACE-3L/KK: lmax={} exceeds GRACE3L_LMAX={}", lmax, GRACE3L_LMAX);
    if (nelements > GRACE3L_MAX_ATOM_TYPES)
      GRACE_KK_ERROR("GRACE-3L/KK: n_elements={} exceeds GRACE3L_MAX_ATOM_TYPES={}", nelements, GRACE3L_MAX_ATOM_TYPES);

    if ((int) m.spbf.size() > GRACE3L_N_SPBF)
      GRACE_KK_ERROR("GRACE-3L/KK: #SPBF={} exceeds GRACE3L_N_SPBF={}", (int) m.spbf.size(), GRACE3L_N_SPBF);
    if ((int) m.prod.size() > GRACE3L_MAX_PRODS)
      GRACE_KK_ERROR("GRACE-3L/KK: #products={} exceeds GRACE3L_MAX_PRODS={}", (int) m.prod.size(), GRACE3L_MAX_PRODS);
    if ((int) m.reduce.size() > GRACE3L_MAX_REDUCES)
      GRACE_KK_ERROR("GRACE-3L/KK: #reduces={} exceeds GRACE3L_MAX_REDUCES={}", (int) m.reduce.size(), GRACE3L_MAX_REDUCES);
    if ((int) m.fc.size() > GRACE3L_MAX_FCS)
      GRACE_KK_ERROR("GRACE-3L/KK: #FCs={} exceeds GRACE3L_MAX_FCS={}", (int) m.fc.size(), GRACE3L_MAX_FCS);
    if ((int) m.eqnorm.size() > GRACE3L_MAX_EQNORMS)
      GRACE_KK_ERROR("GRACE-3L/KK: #eqnorms={} exceeds GRACE3L_MAX_EQNORMS={}", (int) m.eqnorm.size(), GRACE3L_MAX_EQNORMS);
    // EquivariantRMSNorm lm-axis M (25 for eq1_norm, 50 for eq2_norm) shares the
    // downstream SPBF indicator lm-axis; both are bounded by the same cap. This
    // guards the per-thread l0mean/colsum stack arrays in TagEqNorm/TagEqNormBwd.
    for (const auto &e : m.eqnorm)
      if (e.M > GRACE3L_MAX_SPBF_NLMIND)
        GRACE_KK_ERROR("GRACE-3L/KK: {}.M={} exceeds GRACE3L_MAX_SPBF_NLMIND={}", e.name, e.M, GRACE3L_MAX_SPBF_NLMIND);
    if ((int) m.rmsnorm.size() > GRACE3L_MAX_RMSNORMS)
      GRACE_KK_ERROR("GRACE-3L/KK: #rmsnorms={} exceeds GRACE3L_MAX_RMSNORMS={}", (int) m.rmsnorm.size(), GRACE3L_MAX_RMSNORMS);

    for (const auto &s : m.spbf) {
      if (s.n_rad_max > GRACE3L_MAX_NRADMAX)
        GRACE_KK_ERROR("GRACE-3L/KK: {}.n_rad_max={} exceeds GRACE3L_MAX_NRADMAX={}", s.name, s.n_rad_max, GRACE3L_MAX_NRADMAX);
      if (s.n_rad_basis > GRACE3L_MAX_NRADBASIS)
        GRACE_KK_ERROR("GRACE-3L/KK: {}.n_rad_basis={} exceeds GRACE3L_MAX_NRADBASIS={}", s.name, s.n_rad_basis, GRACE3L_MAX_NRADBASIS);
      if (s.mlp.n_layers > GRACE3L_MAX_MLP_LAYERS)
        GRACE_KK_ERROR("GRACE-3L/KK: {}.mlp_n_layers={} exceeds GRACE3L_MAX_MLP_LAYERS={}", s.name, s.mlp.n_layers, GRACE3L_MAX_MLP_LAYERS);
      for (int i = 0; i < s.mlp.n_layers; i++)
        if (s.mlp.n_out[i] > GRACE3L_MAX_MLP_DIM || s.mlp.n_in[i] > GRACE3L_MAX_MLP_DIM)
          GRACE_KK_ERROR("GRACE-3L/KK: {}.mlp layer {} dim exceeds GRACE3L_MAX_MLP_DIM={}", s.name, i, GRACE3L_MAX_MLP_DIM);
      if (s.equivariant) {
        if (s.nfunc > GRACE3L_MAX_SPBF_NFUNC)
          GRACE_KK_ERROR("GRACE-3L/KK: {}.nfunc={} exceeds GRACE3L_MAX_SPBF_NFUNC={}", s.name, s.nfunc, GRACE3L_MAX_SPBF_NFUNC);
        if (s.n_lm_ind > GRACE3L_MAX_SPBF_NLMIND)
          GRACE_KK_ERROR("GRACE-3L/KK: {}.n_lm_ind={} exceeds GRACE3L_MAX_SPBF_NLMIND={}", s.name, s.n_lm_ind, GRACE3L_MAX_SPBF_NLMIND);
      }
    }
    for (const auto &c : m.prod) {
      if (c.rank > GRACE3L_MAX_PROD_RANK)
        GRACE_KK_ERROR("GRACE-3L/KK: {}.rank={} exceeds GRACE3L_MAX_PROD_RANK={}", c.name, c.rank, GRACE3L_MAX_PROD_RANK);
      if (c.nfunc > GRACE3L_MAX_PROD_NFUNC)
        GRACE_KK_ERROR("GRACE-3L/KK: {}.nfunc={} exceeds GRACE3L_MAX_PROD_NFUNC={}", c.name, c.nfunc, GRACE3L_MAX_PROD_NFUNC);
      if (c.n_left > GRACE3L_MAX_PROD_NIN || c.n_right > GRACE3L_MAX_PROD_NIN)
        GRACE_KK_ERROR("GRACE-3L/KK: {} n_left/n_right exceeds GRACE3L_MAX_PROD_NIN={}", c.name, GRACE3L_MAX_PROD_NIN);
      if (c.n_groups_left > GRACE3L_MAX_PROD_NGROUPS || c.n_groups_right > GRACE3L_MAX_PROD_NGROUPS)
        GRACE_KK_ERROR("GRACE-3L/KK: {} n_groups exceeds GRACE3L_MAX_PROD_NGROUPS={}", c.name, GRACE3L_MAX_PROD_NGROUPS);
      if (c.n_cg > GRACE3L_MAX_PROD_NCG)
        GRACE_KK_ERROR("GRACE-3L/KK: {}.n_cg={} exceeds GRACE3L_MAX_PROD_NCG={}", c.name, c.n_cg, GRACE3L_MAX_PROD_NCG);
    }
    for (const auto &r : m.reduce) {
      if ((int) r.instr.size() > GRACE3L_MAX_REDUCE_INSTR)
        GRACE_KK_ERROR("GRACE-3L/KK: {} has {} collectors > GRACE3L_MAX_REDUCE_INSTR={}", r.name, (int) r.instr.size(), GRACE3L_MAX_REDUCE_INSTR);
      if (r.n_out > GRACE3L_MAX_REDUCE_NOUT)
        GRACE_KK_ERROR("GRACE-3L/KK: {}.n_out={} exceeds GRACE3L_MAX_REDUCE_NOUT={}", r.name, r.n_out, GRACE3L_MAX_REDUCE_NOUT);
      if (r.n_funcs > GRACE3L_MAX_REDUCE_NFUNCS)
        GRACE_KK_ERROR("GRACE-3L/KK: {}.n_funcs={} exceeds GRACE3L_MAX_REDUCE_NFUNCS={}", r.name, r.n_funcs, GRACE3L_MAX_REDUCE_NFUNCS);
      for (const auto &ri : r.instr) {
        if (ri.n_in > GRACE3L_MAX_REDUCE_NIN)
          GRACE_KK_ERROR("GRACE-3L/KK: {}[{}].n_in={} exceeds GRACE3L_MAX_REDUCE_NIN={}", r.name, ri.name, ri.n_in, GRACE3L_MAX_REDUCE_NIN);
        if (ri.w_shape > GRACE3L_MAX_REDUCE_WSHAPE)
          GRACE_KK_ERROR("GRACE-3L/KK: {}[{}].w_shape={} exceeds GRACE3L_MAX_REDUCE_WSHAPE={}", r.name, ri.name, ri.w_shape, GRACE3L_MAX_REDUCE_WSHAPE);
      }
    }
    for (const auto &f : m.fc) {
      if (f.n_out > GRACE3L_MAX_FC_NOUT)
        GRACE_KK_ERROR("GRACE-3L/KK: fc_{}.n_out={} exceeds GRACE3L_MAX_FC_NOUT={}", f.name, f.n_out, GRACE3L_MAX_FC_NOUT);
      if (f.w_shape_right > GRACE3L_MAX_FC_TILE)
        GRACE_KK_ERROR("GRACE-3L/KK: fc_{} tile={} exceeds GRACE3L_MAX_FC_TILE={}", f.name, f.w_shape_right, GRACE3L_MAX_FC_TILE);
      // TagFCRight2Left accumulates into a GRACE3L_MAX_FC_NFUNCS-sized
      // stack array indexed by n_funcs_left/right; catch an overflowing model here
      // rather than silently corrupting the stack on device.
      if (f.n_funcs_left > GRACE3L_MAX_FC_NFUNCS)
        GRACE_KK_ERROR("GRACE-3L/KK: fc_{}.n_funcs_left={} exceeds GRACE3L_MAX_FC_NFUNCS={}", f.name, f.n_funcs_left, GRACE3L_MAX_FC_NFUNCS);
      if (f.n_funcs_right > GRACE3L_MAX_FC_NFUNCS)
        GRACE_KK_ERROR("GRACE-3L/KK: fc_{}.n_funcs_right={} exceeds GRACE3L_MAX_FC_NFUNCS={}", f.name, f.n_funcs_right, GRACE3L_MAX_FC_NFUNCS);
    }
    for (const auto &r : m.rmsnorm)
      if (r.n_out > GRACE3L_MAX_RHO_NOUT)
        GRACE_KK_ERROR("GRACE-3L/KK: {}.n_out={} exceeds GRACE3L_MAX_RHO_NOUT={}", r.name, r.n_out, GRACE3L_MAX_RHO_NOUT);
    if (m.energy_mlp.n_layers > GRACE3L_MAX_MLP_LAYERS)
      GRACE_KK_ERROR("GRACE-3L/KK: energy_mlp n_layers={} exceeds GRACE3L_MAX_MLP_LAYERS={}", m.energy_mlp.n_layers, GRACE3L_MAX_MLP_LAYERS);
    // TagComputeMLPEnergy/TagReadoutBwd stack arrays (ebuf_a/ebuf_b, adj_a/adj_b,
    // pre_act_buf) are sized GRACE3L_MAX_ENERGY_MLP_DIM; energy_max_dim itself is
    // only computed later in copy_weights_to_device, so validate every energy-MLP
    // layer width here (before any weights are staged to device).
    for (int i = 0; i < m.energy_mlp.n_layers; i++)
      if (m.energy_mlp.n_out[i] > GRACE3L_MAX_ENERGY_MLP_DIM || m.energy_mlp.n_in[i] > GRACE3L_MAX_ENERGY_MLP_DIM)
        GRACE_KK_ERROR("GRACE-3L/KK: energy_mlp layer {} dim exceeds GRACE3L_MAX_ENERGY_MLP_DIM={}", i, GRACE3L_MAX_ENERGY_MLP_DIM);
  }

  // ---- Spherical harmonics precompute (GeomScalar; verbatim from GRACE-2L) ----
  MemKK::realloc_kokkos(d_idx_sph, "g3l:idx_sph", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(alm, "g3l:alm", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(blm, "g3l:blm", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(cl, "g3l:cl", lmax + 1);
  MemKK::realloc_kokkos(dl, "g3l:dl", lmax + 1);
  precompute_harmonics();

  // l_from_lm lookup: for each lm index, store the corresponding l
  {
    int nlm = (lmax + 1) * (lmax + 1);
    MemKK::realloc_kokkos(d_l_from_lm, "g3l:l_from_lm", nlm);
    auto h_lfm = Kokkos::create_mirror_view(d_l_from_lm);
    for (int l = 0; l <= lmax; l++)
      for (int mm = -l; mm <= l; mm++)
        h_lfm(l * (l + 1) + mm) = l;
    Kokkos::deep_copy(d_l_from_lm, h_lfm);
  }

  // ---- Copy all weights to device ----
  copy_weights_to_device();
  build_radial_splines();

  // ---- locate A1_2_red / fc_A1_2a by name (index-parallel with
  // d_reduce/d_fc, filled in the same loop order by copy_weights_to_device
  // above). Match by NAME (not position) so a re-ordered npz still resolves. ----
  idx_reduce_A1_2_red = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "A1_2_red") { idx_reduce_A1_2_red = i; break; }
  if (idx_reduce_A1_2_red < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A1_2_red' not found in loaded model");
  if (m.reduce[idx_reduce_A1_2_red].instr.size() != 1 ||
      m.reduce[idx_reduce_A1_2_red].instr[0].name != "A1_2")
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A1_2_red' instruction layout unexpected "
               "(expected single instruction named 'A1_2'; compute() wires d_A1_2 "
               "into inputs[0] positionally)");

  idx_fc_A1_2a = -1;
  for (int i = 0; i < (int) m.fc.size(); i++)
    if (m.fc[i].name == "A1_2a") { idx_fc_A1_2a = i; break; }
  if (idx_fc_A1_2a < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: FC 'A1_2a' not found in loaded model");

  // ---- locate the remaining L1 chain by NAME (index-parallel with
  // d_prod/d_reduce/d_fc/d_eqnorm/d_rmsnorm, filled in the same order as
  // grace_model->{prod,reduce,fc,eqnorm,rmsnorm} by copy_weights_to_device). ----
  idx_prod_A1_3 = -1;
  for (int i = 0; i < (int) m.prod.size(); i++)
    if (m.prod[i].name == "A1_3") { idx_prod_A1_3 = i; break; }
  if (idx_prod_A1_3 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: product 'A1_3' not found in loaded model");

  idx_prod_A1_4 = -1;
  for (int i = 0; i < (int) m.prod.size(); i++)
    if (m.prod[i].name == "A1_4") { idx_prod_A1_4 = i; break; }
  if (idx_prod_A1_4 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: product 'A1_4' not found in loaded model");

  idx_reduce_A1_3_red = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "A1_3_red") { idx_reduce_A1_3_red = i; break; }
  if (idx_reduce_A1_3_red < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A1_3_red' not found in loaded model");
  if (m.reduce[idx_reduce_A1_3_red].instr.size() != 1 ||
      m.reduce[idx_reduce_A1_3_red].instr[0].name != "A1_3")
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A1_3_red' instruction layout unexpected "
               "(expected single instruction named 'A1_3')");

  idx_fc_A1_2b = -1;
  for (int i = 0; i < (int) m.fc.size(); i++)
    if (m.fc[i].name == "A1_2b") { idx_fc_A1_2b = i; break; }
  if (idx_fc_A1_2b < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: FC 'A1_2b' not found in loaded model");

  idx_reduce_A1_4_red = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "A1_4_red") { idx_reduce_A1_4_red = i; break; }
  if (idx_reduce_A1_4_red < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A1_4_red' not found in loaded model");
  if (m.reduce[idx_reduce_A1_4_red].instr.size() != 1 ||
      m.reduce[idx_reduce_A1_4_red].instr[0].name != "A1_4")
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A1_4_red' instruction layout unexpected "
               "(expected single instruction named 'A1_4')");

  // eq1 / rho1: multi-instruction (4 collectors), elem-dependent reduces over
  // the SAME 4 named L1 tensors. Resolve role[j] by matching instr[j].name —
  // do NOT assume the npz's instruction_names order matches this enumeration.
  auto resolve_l1_roles = [&](const GRACE3LModel::Reduce &r, int roles[GRACE3L_MAX_REDUCE_INSTR]) {
    static const char *l1_names[4] = {"A1", "A1_2_red", "A1_3_red", "A1_4_red"};
    if ((int) r.instr.size() != 4)
      GRACE_KK_ERROR("GRACE-3L/KK: reduce '{}' expected 4 L1 collectors, found {}",
                 r.name, (int) r.instr.size());
    for (int j = 0; j < 4; j++) {
      int role = -1;
      for (int k = 0; k < 4; k++)
        if (r.instr[j].name == l1_names[k]) { role = k; break; }
      if (role < 0)
        GRACE_KK_ERROR("GRACE-3L/KK: reduce '{}' instruction[{}] name '{}' is not one "
                   "of the expected L1 inputs {{A1,A1_2_red,A1_3_red,A1_4_red}}",
                   r.name, j, r.instr[j].name);
      roles[j] = role;
    }
  };

  idx_reduce_eq1 = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "eq1") { idx_reduce_eq1 = i; break; }
  if (idx_reduce_eq1 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'eq1' not found in loaded model");
  resolve_l1_roles(m.reduce[idx_reduce_eq1], eq1_input_role);

  idx_reduce_rho1 = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "rho1") { idx_reduce_rho1 = i; break; }
  if (idx_reduce_rho1 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'rho1' not found in loaded model");
  resolve_l1_roles(m.reduce[idx_reduce_rho1], rho1_input_role);

  if (idx_eqnorm_eq1_norm < 0)    // resolved in set_model()
    GRACE_KK_ERROR("GRACE-3L/KK: eqnorm 'eq1_norm' not found in loaded model");

  idx_rmsnorm_rho1_norm = -1;
  for (int i = 0; i < (int) m.rmsnorm.size(); i++)
    if (m.rmsnorm[i].name == "rho1_norm") { idx_rmsnorm_rho1_norm = i; break; }
  if (idx_rmsnorm_rho1_norm < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: rmsnorm 'rho1_norm' not found in loaded model");

  // ---- locate the L2 SPBF "A2" (equivariant) by NAME (defensive;
  // spbf order is [A1,A2,A3]). Used by the radial-MLP + A2-assembly kernels. ----
  idx_spbf_A2 = -1;
  for (int i = 0; i < (int) m.spbf.size(); i++)
    if (m.spbf[i].name == "A2") { idx_spbf_A2 = i; break; }
  if (idx_spbf_A2 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: SPBF 'A2' not found in loaded model");
  if (!m.spbf[idx_spbf_A2].equivariant)
    GRACE_KK_ERROR("GRACE-3L/KK: SPBF 'A2' is not equivariant (unexpected schema)");

  // ---- locate the remaining L2 chain by NAME (index-parallel with
  // d_prod/d_reduce/d_fc/d_eqnorm/d_rmsnorm, same convention as the
  // L1 block above). L2 differs from L1 in ONE structural way: every
  // downstream product/FC here is built on A2_red (a reduce of the raw SPBF
  // output A2), not on A2 itself — so there is one extra reduce (A2_red)
  // that L1 does not have. ----
  idx_reduce_A2_red = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "A2_red") { idx_reduce_A2_red = i; break; }
  if (idx_reduce_A2_red < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A2_red' not found in loaded model");
  if (m.reduce[idx_reduce_A2_red].instr.size() != 1 ||
      m.reduce[idx_reduce_A2_red].instr[0].name != "A2")
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A2_red' instruction layout unexpected "
               "(expected single instruction named 'A2'; compute() wires d_A2 "
               "into inputs[0] positionally)");

  idx_prod_A2_2 = -1;
  for (int i = 0; i < (int) m.prod.size(); i++)
    if (m.prod[i].name == "A2_2") { idx_prod_A2_2 = i; break; }
  if (idx_prod_A2_2 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: product 'A2_2' not found in loaded model");

  idx_reduce_A2_2_red = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "A2_2_red") { idx_reduce_A2_2_red = i; break; }
  if (idx_reduce_A2_2_red < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A2_2_red' not found in loaded model");
  if (m.reduce[idx_reduce_A2_2_red].instr.size() != 1 ||
      m.reduce[idx_reduce_A2_2_red].instr[0].name != "A2_2")
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A2_2_red' instruction layout unexpected "
               "(expected single instruction named 'A2_2')");

  idx_fc_A2_2a = -1;
  for (int i = 0; i < (int) m.fc.size(); i++)
    if (m.fc[i].name == "A2_2a") { idx_fc_A2_2a = i; break; }
  if (idx_fc_A2_2a < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: FC 'A2_2a' not found in loaded model");

  idx_prod_A2_3 = -1;
  for (int i = 0; i < (int) m.prod.size(); i++)
    if (m.prod[i].name == "A2_3") { idx_prod_A2_3 = i; break; }
  if (idx_prod_A2_3 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: product 'A2_3' not found in loaded model");

  idx_reduce_A2_3_red = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "A2_3_red") { idx_reduce_A2_3_red = i; break; }
  if (idx_reduce_A2_3_red < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A2_3_red' not found in loaded model");
  if (m.reduce[idx_reduce_A2_3_red].instr.size() != 1 ||
      m.reduce[idx_reduce_A2_3_red].instr[0].name != "A2_3")
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A2_3_red' instruction layout unexpected "
               "(expected single instruction named 'A2_3')");

  idx_fc_A2_2b = -1;
  for (int i = 0; i < (int) m.fc.size(); i++)
    if (m.fc[i].name == "A2_2b") { idx_fc_A2_2b = i; break; }
  if (idx_fc_A2_2b < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: FC 'A2_2b' not found in loaded model");

  idx_prod_A2_4 = -1;
  for (int i = 0; i < (int) m.prod.size(); i++)
    if (m.prod[i].name == "A2_4") { idx_prod_A2_4 = i; break; }
  if (idx_prod_A2_4 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: product 'A2_4' not found in loaded model");

  idx_reduce_A2_4_red = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "A2_4_red") { idx_reduce_A2_4_red = i; break; }
  if (idx_reduce_A2_4_red < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A2_4_red' not found in loaded model");
  if (m.reduce[idx_reduce_A2_4_red].instr.size() != 1 ||
      m.reduce[idx_reduce_A2_4_red].instr[0].name != "A2_4")
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A2_4_red' instruction layout unexpected "
               "(expected single instruction named 'A2_4')");

  // eq2 / rho2: multi-instruction (4 collectors), elem-dependent reduces over
  // the SAME 4 named L2 tensors. Resolve role[j] by matching instr[j].name —
  // do NOT assume the npz's instruction_names order matches this enumeration.
  auto resolve_l2_roles = [&](const GRACE3LModel::Reduce &r, int roles[GRACE3L_MAX_REDUCE_INSTR]) {
    static const char *l2_names[4] = {"A2_red", "A2_2_red", "A2_3_red", "A2_4_red"};
    if ((int) r.instr.size() != 4)
      GRACE_KK_ERROR("GRACE-3L/KK: reduce '{}' expected 4 L2 collectors, found {}",
                 r.name, (int) r.instr.size());
    for (int j = 0; j < 4; j++) {
      int role = -1;
      for (int k = 0; k < 4; k++)
        if (r.instr[j].name == l2_names[k]) { role = k; break; }
      if (role < 0)
        GRACE_KK_ERROR("GRACE-3L/KK: reduce '{}' instruction[{}] name '{}' is not one "
                   "of the expected L2 inputs {{A2_red,A2_2_red,A2_3_red,A2_4_red}}",
                   r.name, j, r.instr[j].name);
      roles[j] = role;
    }
  };

  idx_reduce_eq2 = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "eq2") { idx_reduce_eq2 = i; break; }
  if (idx_reduce_eq2 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'eq2' not found in loaded model");
  resolve_l2_roles(m.reduce[idx_reduce_eq2], eq2_input_role);

  idx_reduce_rho2 = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "rho2") { idx_reduce_rho2 = i; break; }
  if (idx_reduce_rho2 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'rho2' not found in loaded model");
  resolve_l2_roles(m.reduce[idx_reduce_rho2], rho2_input_role);

  if (idx_eqnorm_eq2_norm < 0)    // resolved in set_model()
    GRACE_KK_ERROR("GRACE-3L/KK: eqnorm 'eq2_norm' not found in loaded model");

  idx_rmsnorm_rho2_norm = -1;
  for (int i = 0; i < (int) m.rmsnorm.size(); i++)
    if (m.rmsnorm[i].name == "rho2_norm") { idx_rmsnorm_rho2_norm = i; break; }
  if (idx_rmsnorm_rho2_norm < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: rmsnorm 'rho2_norm' not found in loaded model");

  // ---- locate the L3 SPBF "A3" (equivariant) by NAME (defensive;
  // spbf order is [A1,A2,A3]). Its indicator is eq2_norm (parity-doubled). ----
  idx_spbf_A3 = -1;
  for (int i = 0; i < (int) m.spbf.size(); i++)
    if (m.spbf[i].name == "A3") { idx_spbf_A3 = i; break; }
  if (idx_spbf_A3 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: SPBF 'A3' not found in loaded model");
  if (!m.spbf[idx_spbf_A3].equivariant)
    GRACE_KK_ERROR("GRACE-3L/KK: SPBF 'A3' is not equivariant (unexpected schema)");

  // ---- Guard: d_R1_nl/d_DR1_nl are allocated to L1_nradmax (A1's n_rad_max)
  // but reused as shared radial scratch by A2/A3 in TagComputeMLPRadial /
  // TagComputeMLPRadialDeriv. Safe only because every layer currently shares
  // n_rad_max; nothing else enforces that, so a model with a larger L2/L3
  // nradmax would silently overflow the shared scratch on device. ----
  if (m.spbf[idx_spbf_A2].n_rad_max > L1_nradmax)
    GRACE_KK_ERROR("GRACE-3L/KK: A2.n_rad_max={} exceeds shared radial scratch "
               "size L1_nradmax={} (d_R1_nl/d_DR1_nl sized to L1_nradmax)",
               m.spbf[idx_spbf_A2].n_rad_max, L1_nradmax);
  if (m.spbf[idx_spbf_A3].n_rad_max > L1_nradmax)
    GRACE_KK_ERROR("GRACE-3L/KK: A3.n_rad_max={} exceeds shared radial scratch "
               "size L1_nradmax={} (d_R1_nl/d_DR1_nl sized to L1_nradmax)",
               m.spbf[idx_spbf_A3].n_rad_max, L1_nradmax);

  // ---- remaining L3 chain by NAME. Structurally identical to L2
  // (built on A3_red, a reduce of the raw SPBF output A3), EXCEPT L3 is
  // TERMINAL: it produces ONLY rho3/rho3_norm — there is NO eq3. ----
  idx_reduce_A3_red = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "A3_red") { idx_reduce_A3_red = i; break; }
  if (idx_reduce_A3_red < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A3_red' not found in loaded model");
  if (m.reduce[idx_reduce_A3_red].instr.size() != 1 ||
      m.reduce[idx_reduce_A3_red].instr[0].name != "A3")
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A3_red' instruction layout unexpected "
               "(expected single instruction named 'A3'; compute() wires d_A3 "
               "into inputs[0] positionally)");

  idx_prod_A3_2 = -1;
  for (int i = 0; i < (int) m.prod.size(); i++)
    if (m.prod[i].name == "A3_2") { idx_prod_A3_2 = i; break; }
  if (idx_prod_A3_2 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: product 'A3_2' not found in loaded model");

  idx_reduce_A3_2_red = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "A3_2_red") { idx_reduce_A3_2_red = i; break; }
  if (idx_reduce_A3_2_red < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A3_2_red' not found in loaded model");
  if (m.reduce[idx_reduce_A3_2_red].instr.size() != 1 ||
      m.reduce[idx_reduce_A3_2_red].instr[0].name != "A3_2")
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A3_2_red' instruction layout unexpected "
               "(expected single instruction named 'A3_2')");

  idx_fc_A3_2a = -1;
  for (int i = 0; i < (int) m.fc.size(); i++)
    if (m.fc[i].name == "A3_2a") { idx_fc_A3_2a = i; break; }
  if (idx_fc_A3_2a < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: FC 'A3_2a' not found in loaded model");

  idx_prod_A3_3 = -1;
  for (int i = 0; i < (int) m.prod.size(); i++)
    if (m.prod[i].name == "A3_3") { idx_prod_A3_3 = i; break; }
  if (idx_prod_A3_3 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: product 'A3_3' not found in loaded model");

  idx_reduce_A3_3_red = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "A3_3_red") { idx_reduce_A3_3_red = i; break; }
  if (idx_reduce_A3_3_red < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A3_3_red' not found in loaded model");
  if (m.reduce[idx_reduce_A3_3_red].instr.size() != 1 ||
      m.reduce[idx_reduce_A3_3_red].instr[0].name != "A3_3")
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A3_3_red' instruction layout unexpected "
               "(expected single instruction named 'A3_3')");

  idx_fc_A3_2b = -1;
  for (int i = 0; i < (int) m.fc.size(); i++)
    if (m.fc[i].name == "A3_2b") { idx_fc_A3_2b = i; break; }
  if (idx_fc_A3_2b < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: FC 'A3_2b' not found in loaded model");

  idx_prod_A3_4 = -1;
  for (int i = 0; i < (int) m.prod.size(); i++)
    if (m.prod[i].name == "A3_4") { idx_prod_A3_4 = i; break; }
  if (idx_prod_A3_4 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: product 'A3_4' not found in loaded model");

  idx_reduce_A3_4_red = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "A3_4_red") { idx_reduce_A3_4_red = i; break; }
  if (idx_reduce_A3_4_red < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A3_4_red' not found in loaded model");
  if (m.reduce[idx_reduce_A3_4_red].instr.size() != 1 ||
      m.reduce[idx_reduce_A3_4_red].instr[0].name != "A3_4")
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'A3_4_red' instruction layout unexpected "
               "(expected single instruction named 'A3_4')");

  // rho3: multi-instruction (4 collectors), elem-dependent, only_invar reduce
  // over the SAME 4 named L3 tensors. Resolve role[j] by matching instr[j].name
  // — do NOT assume the npz's instruction_names order matches this enumeration.
  // (L3 is terminal: no eq3, so only rho3 needs a role array.)
  auto resolve_l3_roles = [&](const GRACE3LModel::Reduce &r, int roles[GRACE3L_MAX_REDUCE_INSTR]) {
    static const char *l3_names[4] = {"A3_red", "A3_2_red", "A3_3_red", "A3_4_red"};
    if ((int) r.instr.size() != 4)
      GRACE_KK_ERROR("GRACE-3L/KK: reduce '{}' expected 4 L3 collectors, found {}",
                 r.name, (int) r.instr.size());
    for (int j = 0; j < 4; j++) {
      int role = -1;
      for (int k = 0; k < 4; k++)
        if (r.instr[j].name == l3_names[k]) { role = k; break; }
      if (role < 0)
        GRACE_KK_ERROR("GRACE-3L/KK: reduce '{}' instruction[{}] name '{}' is not one "
                   "of the expected L3 inputs {{A3_red,A3_2_red,A3_3_red,A3_4_red}}",
                   r.name, j, r.instr[j].name);
      roles[j] = role;
    }
  };

  idx_reduce_rho3 = -1;
  for (int i = 0; i < (int) m.reduce.size(); i++)
    if (m.reduce[i].name == "rho3") { idx_reduce_rho3 = i; break; }
  if (idx_reduce_rho3 < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: reduce 'rho3' not found in loaded model");
  resolve_l3_roles(m.reduce[idx_reduce_rho3], rho3_input_role);

  idx_rmsnorm_rho3_norm = -1;
  for (int i = 0; i < (int) m.rmsnorm.size(); i++)
    if (m.rmsnorm[i].name == "rho3_norm") { idx_rmsnorm_rho3_norm = i; break; }
  if (idx_rmsnorm_rho3_norm < 0)
    GRACE_KK_ERROR("GRACE-3L/KK: rmsnorm 'rho3_norm' not found in loaded model");

  // ---- UQ basis-RP: pin the per-layer scalar-basis block widths. The GMM feature
  // is the projection of the invariant (l=0) products entering rho1|rho2|rho3, read
  // in the exact (source, n, collect) order TagReduceN iterates them, so R's row
  // count D_basis must equal the summed per-block widths [Σ n_in·n_conn per source].
  // R-row layout is [rho1, rho2, rho3] (sorted by reduce name). This guards against
  // an artifact/model mismatch. ----
  if (m.has_uq) {
    auto block_width = [&](int idx_reduce) {
      const auto &r = m.reduce[idx_reduce];
      int w = 0;
      for (const auto &ins : r.instr) w += ins.n_in * ins.n_conn;
      return w;
    };
    uq_d_basis_rho1 = block_width(idx_reduce_rho1);
    uq_d_basis_rho2 = block_width(idx_reduce_rho2);
    uq_d_basis_rho3 = block_width(idx_reduce_rho3);
    if (uq_d_basis_rho1 + uq_d_basis_rho2 + uq_d_basis_rho3 != m.uq_d_basis)
      GRACE_KK_ERROR("GRACE-3L/KK UQ: scalar-basis width rho1({})+rho2({})+rho3({}) "
                 "!= uq_rp_matrix rows ({})",
                 uq_d_basis_rho1, uq_d_basis_rho2, uq_d_basis_rho3, m.uq_d_basis);
  }

  // ---- inter-layer message comm sizing. Round 1 forward-comms
  // the Layer-2 indicator eq1_norm [nall, n_out=64, M=25] (I_n_funcs*I_n_out =
  // 1600/atom); round 2 forward-comms the Layer-3 indicator eq2_norm
  // [nall, n_out=64, M=50] (I2_n_funcs*I2_n_out = 3200/atom, PARITY-DOUBLED).
  // comm_forward is the MAX of the two payloads so the shared comm buffer fits
  // both stages; comm_stage selects which at each forward_comm. comm_reverse is
  // ALSO the MAX: round-1 reverse-comms grad_I [I_n_funcs*I_n_out=1600],
  // round-2 reverse-comms the eq2_norm message adj [I2_n_funcs*
  // I2_n_out=3200]; the shared reverse buffer must fit the larger stage. The dims are set in set_model(). ----
  log.info(fmt("[GRACE-3L/KK] Initialization complete: weights on device "
               "(SPBF={} products={} reduces={} FCs={} eqnorms={} rmsnorms={}) | "
               "comm_forward={} (round1 I_n_funcs={} I_n_out={}; "
               "round2 I2_n_funcs={} I2_n_out={}).\n",
               n_spbf, n_prods, n_reduces, n_fcs, n_eqnorms, n_rmsnorms,
               std::max(comm_width(0), comm_width(1)), I_n_funcs, I_n_out, I2_n_funcs, I2_n_out));

#ifdef GRACE_KK_GEMM
  // ---- create the BLAS handle (grace/kk/blas.h) once, bind it to the SAME stream the
  // Kokkos parallel_fors launch on (the default DeviceType instance), and force
  // TRUE fp32 (CUBLAS_PEDANTIC_MATH disables the Ampere+ tf32 default). Same-stream
  // binding makes the GEMMs and the surrounding Kokkos assembly/activation kernels
  // execute in issue order with NO race and NO explicit fences. Without cuBLAS
  // (GRACE_KK_CUBLAS_DLOPEN and no libcublas, or GRACE_RUNTIME_NO_CUBLAS) the handle stays
  // null and every BLAS path takes its hand-written kernel. ----
  // HIP: fp32 engines only. rocBLAS (loaded at run time) for the fp32 GEMM paths (cp_l projections,
  // ReduceN, FC GEMM; the radial MLP stays on the tables / hand kernels). Created here, not in the first
  // compute, so the library is loaded during setup. Without librocblas or with GRACE_RUNTIME_NO_ROCBLAS: the same GEMM
  // paths on the hand-written Kokkos GEMMs (kk_gemm.h); GRACE_RUNTIME_NO_KKGEMM as well: no handle, the fused
  // hand-written kernels.
  if constexpr (is_cuda_v<DeviceType> || (is_hip_v<DeviceType> && sizeof(NNScalar) == 4)) {
    if (!blas_handle && blas::available()) {
      if (blas::create(&blas_handle) != blas::SUCCESS) {
        blas_handle = nullptr;    // optional BLAS (e.g. no device memory left): the hand-written kernels
        if constexpr (!blas::optional) GRACE_KK_ERROR("GRACE-3L/KK: BLAS handle creation failed");
      }
      if (blas_handle) {
        blas_owner.reset(blas_handle, [](blas::handle_t h) { blas::destroy(h); });
        blas::set_stream(blas_handle, DeviceType());
        blas::set_pedantic_math(blas_handle);
        mlp_blas_selftest();   // tiny-case transpose/leading-dim gate (aborts on mismatch)
      }
    }
  }
#ifdef KOKKOS_ENABLE_CUDA
  if (blas_handle)
    log.info(fmt("[GRACE-3L/KK] radial MLP: true-fp32 cuBLAS SGEMM enabled (CUDA)\n"));
  else
    log.info(fmt("[GRACE-3L/KK] radial MLP: hand-written Kokkos kernels (no cuBLAS)\n"));
#else
  if (blas_handle)
    log.info(fmt("[GRACE-3L/KK] fp32 GEMMs (cp_l projections, ReduceN, FC): {} (HIP)\n",
                 ::grace::kk::detail::rocblas_ok() ? "rocBLAS" : "hand-written Kokkos GEMMs"));
  else
    log.info(fmt("[GRACE-3L/KK] fp32 GEMMs: hand-written Kokkos kernels (no rocBLAS)\n"));
#endif
#else
  log.info(fmt("[GRACE-3L/KK] radial MLP: hand-written Kokkos kernels "
               "(host build, no GEMM library)\n"));
#endif
  build_fc_fold();
  setup_done = true;
  check_kernel_launches<DeviceType>("GRACE-3L/KK");
}

// ======================================================================
// precompute_harmonics — the real spherical harmonics' recurrence tables (support.h sph_tables) on the device
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::precompute_harmonics()
{
  const SphTables t = sph_tables(lmax);
  upload_prefix(d_idx_sph, t.slot);
  upload_prefix(alm, t.a);
  upload_prefix(blm, t.b);
  upload_prefix(cl, t.c);
  upload_prefix(dl, t.d);
  idx_sph_max = t.npacked;
}

// ======================================================================
// copy_weights_to_device — stage every host GRACE3LModel tensor into the
// constant device views. RAW weight + separate norm scalar (kernel applies
// once), EXCEPT the SPBF per-element species tables (z_tr / z_proj) which
// fold the layer norm in once at precompute time.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::copy_weights_to_device()
{
  auto &m = *grace_model;

  // ---- generic copy helpers (dst value type drives the double->NN/Geom cast) ----
  auto copy_1d_int = [](const std::vector<int> &src, auto &dst, const char *label) {
    MemKK::realloc_kokkos(dst, label, std::max<size_t>(src.size(), 1));
    auto h = Kokkos::create_mirror_view(dst);
    for (size_t i = 0; i < src.size(); i++) h(i) = src[i];
    Kokkos::deep_copy(dst, h);
  };
  auto copy_1d = [](const std::vector<double> &src, auto &dst, const char *label) {
    MemKK::realloc_kokkos(dst, label, std::max<size_t>(src.size(), 1));
    auto h = Kokkos::create_mirror_view(dst);
    for (size_t i = 0; i < src.size(); i++) h(i) = src[i];
    Kokkos::deep_copy(dst, h);
  };
  auto copy_2d = [this](const std::vector<double> &src, auto &dst, int d0, int d1, const char *label) {
    if (src.size() != (size_t) d0 * d1)
      GRACE_KK_ERROR("GRACE-3L/KK: {} size {} does not match expected {}x{}={} "
                 "(desynced weight export?)", label, src.size(), d0, d1, (size_t) d0 * d1);
    MemKK::realloc_kokkos(dst, label, d0, d1);
    auto h = Kokkos::create_mirror_view(dst);
    for (int i = 0; i < d0; i++)
      for (int j = 0; j < d1; j++)
        h(i, j) = src[(size_t) i * d1 + j];
    Kokkos::deep_copy(dst, h);
  };
  auto copy_3d = [this](const std::vector<double> &src, auto &dst, int d0, int d1, int d2, const char *label) {
    if (src.size() != (size_t) d0 * d1 * d2)
      GRACE_KK_ERROR("GRACE-3L/KK: {} size {} does not match expected {}x{}x{}={} "
                 "(desynced weight export?)", label, src.size(), d0, d1, d2, (size_t) d0 * d1 * d2);
    MemKK::realloc_kokkos(dst, label, d0, d1, d2);
    auto h = Kokkos::create_mirror_view(dst);
    for (int i = 0; i < d0; i++)
      for (int j = 0; j < d1; j++)
        for (int k = 0; k < d2; k++)
          h(i, j, k) = src[((size_t) i * d1 + j) * d2 + k];
    Kokkos::deep_copy(dst, h);
  };
  auto copy_4d = [this](const std::vector<double> &src, auto &dst, int d0, int d1, int d2, int d3, const char *label) {
    if (src.size() != (size_t) d0 * d1 * d2 * d3)
      GRACE_KK_ERROR("GRACE-3L/KK: {} size {} does not match expected {}x{}x{}x{}={} "
                 "(desynced weight export?)", label, src.size(), d0, d1, d2, d3, (size_t) d0 * d1 * d2 * d3);
    MemKK::realloc_kokkos(dst, label, d0, d1, d2, d3);
    auto h = Kokkos::create_mirror_view(dst);
    for (int i = 0; i < d0; i++)
      for (int j = 0; j < d1; j++)
        for (int k = 0; k < d2; k++)
          for (int l = 0; l < d3; l++)
            h(i, j, k, l) = src[(((size_t) i * d1 + j) * d2 + k) * d3 + l];
    Kokkos::deep_copy(dst, h);
  };

  // ---- Chemical embedding / shifts / bond cutoffs / output scale ----
  copy_2d(m.chem_embedding, d_chem_embed, m.n_elements, m.embedding_size, "g3l:chem_embed");
  copy_1d(m.shift_values, d_shifts, "g3l:shifts");
  copy_2d(m.bond_cutoff_map, d_bond_cutoff, m.n_elements, m.n_elements, "g3l:bond_cutoff");
  output_scale = (NNScalar) m.output_scale;

  // ---- SPBF layers (radial MLP w/ hidden bias; species modulation; CG couple) ----
  n_spbf = (int) m.spbf.size();
  for (int si = 0; si < n_spbf; si++) {
    const auto &s = m.spbf[si];
    DeviceSPBF &ds = d_spbf[si];
    ds.equivariant = s.equivariant ? 1 : 0;
    ds.n_rad_max = s.n_rad_max; ds.n_rad_basis = s.n_rad_basis;
    ds.spbf_lmax = s.lmax; ds.Lmax = s.Lmax; ds.p = s.p;
    ds.nfunc = s.nfunc; ds.n_lm_ind = s.n_lm_ind;
    ds.rcut = (GeomScalar) s.rcut;
    ds.inv_avg_n_neigh = (GeomScalar) s.inv_avg_n_neigh;
    copy_1d_int(s.l_tile, ds.l_tile, "g3l:spbf_l_tile");
    // the sliced-ELL CG tables of the matrix-core SPBF are built again on first use (their key, the address of the
    // previous CG table, may come back for the new one)
    d_spbf_rsell[si] = SpbfSell();
    d_spbf_csell[si] = SpbfSell();
    spbf_ltile_std[si] = 1;
    for (int y = 0; y < (int) s.l_tile.size(); y++) {
      int l = 0;
      while ((l + 1) * (l + 1) <= y) l++;
      if (s.l_tile[y] != l) spbf_ltile_std[si] = 0;
    }

    // radial MLP (padded), with hidden-layer biases (output row zero)
    const auto &mlp = s.mlp;
    int nl = mlp.n_layers, max_in = 0, max_out = 0;
    for (int i = 0; i < nl; i++) { max_in = std::max(max_in, mlp.n_in[i]); max_out = std::max(max_out, mlp.n_out[i]); }
    ds.mlp_n_layers = nl;
    ds.mlp_max_dim = std::max(max_in, max_out);
    std::string sp = "g3l:" + s.name + "_mlp";
    MemKK::realloc_kokkos(ds.mlp_W, (sp + "_W").c_str(), std::max(nl, 1), std::max(max_in, 1), std::max(max_out, 1));
    MemKK::realloc_kokkos(ds.mlp_b, (sp + "_b").c_str(), std::max(nl, 1), std::max(max_out, 1));
    auto hW = Kokkos::create_mirror_view(ds.mlp_W);
    auto hb = Kokkos::create_mirror_view(ds.mlp_b);
    Kokkos::deep_copy(hW, (NNScalar) 0); Kokkos::deep_copy(hb, (NNScalar) 0);
    for (int l = 0; l < nl; l++) {
      for (int i = 0; i < mlp.n_in[l]; i++)
        for (int j = 0; j < mlp.n_out[l]; j++)
          hW(l, i, j) = (NNScalar) mlp.W[l][(size_t) i * mlp.n_out[l] + j];
      if (!mlp.b[l].empty())
        for (int j = 0; j < mlp.n_out[l] && j < (int) mlp.b[l].size(); j++)
          hb(l, j) = (NNScalar) mlp.b[l][j];
    }
    Kokkos::deep_copy(ds.mlp_W, hW); Kokkos::deep_copy(ds.mlp_b, hb);
#ifdef GRACE_KK_GEMM
    // pack each layer's weights TIGHTLY row-major [n_in x n_out] for cuBLAS
    // (the padded LayoutLeft ds.mlp_W is not a contiguous per-layer block). lda=n_out.
    for (int l = 0; l < nl; l++) {
      const int ni = mlp.n_in[l], no = mlp.n_out[l];
      MemKK::realloc_kokkos(d_mlp_Wg[si][l], (sp + "_Wg").c_str(),
                            std::max(ni, 1), std::max(no, 1));
      auto hWg = Kokkos::create_mirror_view(d_mlp_Wg[si][l]);
      for (int i = 0; i < ni; i++)
        for (int j = 0; j < no; j++)
          hWg(i, j) = (NNScalar) mlp.W[l][(size_t) i * no + j];
      Kokkos::deep_copy(d_mlp_Wg[si][l], hWg);
    }
    // Layer 0 split by input block [cheb, emb(mu_i), emb(mu_j)]: Chebyshev weights and the
    // per-element embedding terms (fp64 on the host), norm0 and the layer-0 bias folded in.
    mlp_split[si] = 0;
    if (nl == 3 && mlp.n_in[0] == m.nradbase + 2 * m.embedding_size) {
      const int nrb = m.nradbase, ne = m.embedding_size, d1 = mlp.n_out[0], nel = m.n_elements;
      const double nrm = mlp.norm[0];
      const auto &W0 = mlp.W[0];
      MemKK::realloc_kokkos(d_mlp_W0c[si], (sp + "_W0c").c_str(), nrb, d1);
      MemKK::realloc_kokkos(d_mlp_E[si], (sp + "_E").c_str(), nel, 2 * d1);
      auto hW0c = Kokkos::create_mirror_view(d_mlp_W0c[si]);
      auto hE = Kokkos::create_mirror_view(d_mlp_E[si]);
      for (int k = 0; k < nrb; k++)
        for (int j = 0; j < d1; j++) hW0c(k, j) = (NNScalar) (nrm * W0[(size_t) k * d1 + j]);
      for (int mu = 0; mu < nel; mu++)
        for (int j = 0; j < d1; j++) {
          double ei = mlp.b[0].empty() ? 0.0 : mlp.b[0][j], ej = 0.0;
          for (int e = 0; e < ne; e++) {
            const double z = m.chem_embedding[(size_t) mu * ne + e];
            ei += z * W0[(size_t) (nrb + e) * d1 + j];
            ej += z * W0[(size_t) (nrb + ne + e) * d1 + j];
          }
          hE(mu, j) = (NNScalar) (nrm * ei);
          hE(mu, d1 + j) = (NNScalar) (nrm * ej);
        }
      Kokkos::deep_copy(d_mlp_W0c[si], hW0c);
      Kokkos::deep_copy(d_mlp_E[si], hE);
      mlp_split[si] = 1;
    }
#endif
    MemKK::realloc_kokkos(ds.mlp_norms, (sp + "_norms").c_str(), std::max(nl, 1));
    auto hn = Kokkos::create_mirror_view(ds.mlp_norms);
    for (int i = 0; i < nl; i++) hn(i) = (NNScalar) mlp.norm[i];
    Kokkos::deep_copy(ds.mlp_norms, hn);
    MemKK::realloc_kokkos(ds.mlp_dims, (sp + "_dims").c_str(), nl + 1);
    auto hd = Kokkos::create_mirror_view(ds.mlp_dims);
    hd(0) = mlp.n_in.empty() ? 0 : mlp.n_in[0];
    for (int i = 0; i < nl; i++) hd(i + 1) = mlp.n_out[i];
    Kokkos::deep_copy(ds.mlp_dims, hd);

    // Per-element species modulation — norm FOLDED here (once).
    if (!s.equivariant) {
      // A1: z_tr[mu][n] = norm * sum_e chem_embed[mu,e] * lin_transform_W[e,n]
      MemKK::realloc_kokkos(ds.z_tr, ("g3l:" + s.name + "_z_tr").c_str(), m.n_elements, s.n_rad_max);
      auto h = Kokkos::create_mirror_view(ds.z_tr);
      for (int mu = 0; mu < m.n_elements; mu++)
        for (int n = 0; n < s.n_rad_max; n++) {
          double v = 0.0;
          for (int e = 0; e < m.embedding_size; e++)
            v += m.chem_embedding[(size_t) mu * m.embedding_size + e] *
                 s.lin_transform_W[(size_t) e * s.n_rad_max + n];
          h(mu, n) = (NNScalar) (v * s.lin_transform_norm);
        }
      Kokkos::deep_copy(ds.z_tr, h);
    } else {
      // A2/A3: z_proj[mu][n] = norm * sum_e chem_embed[mu,e] * chem_linear_W[e,n]
      const int cn_out = (int) (s.chem_linear_W.size() / std::max(m.embedding_size, 1));
      MemKK::realloc_kokkos(ds.z_proj, ("g3l:" + s.name + "_z_proj").c_str(), m.n_elements, cn_out);
      auto h = Kokkos::create_mirror_view(ds.z_proj);
      for (int mu = 0; mu < m.n_elements; mu++)
        for (int n = 0; n < cn_out; n++) {
          double v = 0.0;
          for (int e = 0; e < m.embedding_size; e++)
            v += m.chem_embedding[(size_t) mu * m.embedding_size + e] *
                 s.chem_linear_W[(size_t) e * cn_out + n];
          h(mu, n) = (NNScalar) (v * s.chem_linear_norm);
        }
      Kokkos::deep_copy(ds.z_proj, h);
      copy_1d(s.chem_l0_mask, ds.chem_l0_mask, "g3l:spbf_chem_l0_mask");
      // dense CG couple: [(lmax+1)^2 * n_lm_ind, nfunc]
      const int cg_rows = (int) (s.cg_W.size() / std::max(s.nfunc, 1));
      copy_2d(s.cg_W, ds.cg_W, cg_rows, s.nfunc, "g3l:spbf_cg_W");
      // build sparse CSC/CSR forms of cg_W. CG selection rules make the
      // matrix ~99% zeros (mean ~3-6 nnz per row/column), so the dense couple
      // loops in TagComputeSPBFFused (per-f over P) and TagComputeAdjProd
      // (per-p over f) wasted ~100-200x compute + L2 bandwidth streaming zeros.
      // Ascending index order within each column/row preserves the dense loops'
      // accumulation order exactly (skipped terms are exact +0 contributions),
      // so results are numerically identical. Worst case (a dense cg_W) the
      // sparse loop only adds index-load overhead — still correct.
      {
        const int P = cg_rows, NF = s.nfunc;
        std::vector<int> cptr(NF + 1, 0), ridx;
        std::vector<double> cval;
        std::vector<int> rptr(P + 1, 0), cidx;
        std::vector<double> rval;
        for (int f = 0; f < NF; f++) {
          cptr[f] = (int) ridx.size();
          for (int p = 0; p < P; p++) {
            const double w = s.cg_W[(size_t) p * NF + f];
            if (w != 0.0) { ridx.push_back(p); cval.push_back(w); }
          }
        }
        cptr[NF] = (int) ridx.size();
        for (int p = 0; p < P; p++) {
          rptr[p] = (int) cidx.size();
          for (int f = 0; f < NF; f++) {
            const double w = s.cg_W[(size_t) p * NF + f];
            if (w != 0.0) { cidx.push_back(f); rval.push_back(w); }
          }
        }
        rptr[P] = (int) cidx.size();
        copy_1d_int(cptr, ds.cg_col_ptr, "g3l:spbf_cg_col_ptr");
        copy_1d_int(ridx, ds.cg_row_idx, "g3l:spbf_cg_row_idx");
        copy_1d(cval, ds.cg_col_val, "g3l:spbf_cg_col_val");
        copy_1d_int(rptr, ds.cg_row_ptr, "g3l:spbf_cg_row_ptr");
        copy_1d_int(cidx, ds.cg_col_idx, "g3l:spbf_cg_col_idx");
        copy_1d(rval, ds.cg_row_val, "g3l:spbf_cg_row_val");
        int ew = 0;
        for (int q = 0; q < P; q++) ew = std::max(ew, rptr[q + 1] - rptr[q]);
        std::vector<int> ecol((size_t) ew * P, 0);
        std::vector<double> eval((size_t) ew * P, 0.0);
        for (int q = 0; q < P; q++)
          for (int e = rptr[q]; e < rptr[q + 1]; e++) {
            ecol[(size_t) (e - rptr[q]) * P + q] = cidx[e];
            eval[(size_t) (e - rptr[q]) * P + q] = rval[e];
          }
        ds.ell_w = ew;
        copy_1d_int(ecol, ds.cg_ell_col, "g3l:spbf_cg_ell_col");
        copy_1d(eval, ds.cg_ell_val, "g3l:spbf_cg_ell_val");
        auto pack = [](const std::vector<int> &ix, const std::vector<double> &v) {
          std::vector<int> o(2 * ix.size());
          for (size_t q = 0; q < ix.size(); q++) {
            const float fv = (float) v[q];
            o[2 * q] = ix[q];
            std::memcpy(&o[2 * q + 1], &fv, sizeof(float));
          }
          return o;
        };
        copy_1d_int(pack(ecol, eval), ds.cg_ell2, "g3l:spbf_cg_ell2");
        copy_1d_int(pack(ridx, cval), ds.cg_csc2, "g3l:spbf_cg_csc2");
      }
    }
  }

  // ---- cp_l products ----
  n_prods = (int) m.prod.size();
  for (int pi = 0; pi < n_prods; pi++) {
    const auto &c = m.prod[pi];
    DeviceCPL &dc = d_prod[pi];
    dc.rank = c.rank; dc.nfunc = c.nfunc; dc.n_left = c.n_left; dc.n_right = c.n_right;
    dc.n_groups_left = c.n_groups_left; dc.n_groups_right = c.n_groups_right; dc.n_cg = c.n_cg;
    dc.norm_u = (NNScalar) c.norm_u; dc.norm_v = (NNScalar) c.norm_v;
    copy_3d(c.U, dc.U, c.n_groups_left, c.rank, c.n_left, "g3l:prod_U");
    copy_3d(c.V, dc.V, c.n_groups_right, c.rank, c.n_right, "g3l:prod_V");
    copy_1d(c.cg, dc.cg, "g3l:prod_cg");
    copy_1d_int(c.group_left, dc.group_left, "g3l:prod_gl");
    copy_1d_int(c.group_right, dc.group_right, "g3l:prod_gr");
    copy_1d_int(c.left_ind, dc.left_ind, "g3l:prod_li");
    copy_1d_int(c.right_ind, dc.right_ind, "g3l:prod_ri");
    copy_1d_int(c.m_sum_ind, dc.m_sum_ind, "g3l:prod_si");
    // pack (m_sum_ind, right_ind, left_ind) into one int32 per entry so
    // the fused couple loops stream half the index bytes. Hard-check the packing
    // bounds (li/ri < 256, m < 65536) — far beyond any plausible cp_l dims
    // (nlm <= (Lmax+1)^2, nfunc <= a few thousand).
    {
      std::vector<int> packed(std::max(c.n_cg, 1), 0);
      for (int i = 0; i < c.n_cg; i++) {
        const int li = c.left_ind[i], ri = c.right_ind[i], mi = c.m_sum_ind[i];
        if (li < 0 || li > 255 || ri < 0 || ri > 255 || mi < 0 || mi > 65535)
          GRACE_KK_ERROR("GRACE-3L/KK: cp_l couple index out of packing range "
                     "(li={}, ri={}, m={})", li, ri, mi);
        packed[i] = (mi << 16) | (ri << 8) | li;
      }
      copy_1d_int(packed, dc.cg_packed, "g3l:prod_cg_packed");
    }
    copy_1d_int(c.out_l, dc.out_l, "g3l:prod_out_l");
    // gather-form couple tables (term order within a key = file order)
    {
      auto csr = [&](int nkey, auto key, auto ia, auto ib, t_int_1d &off, t_int_1d &a, t_int_1d &b, t_nn_1d &cv,
                     const char *nm) {
        std::vector<std::vector<int>> by(nkey);
        for (int i = 0; i < c.n_cg; i++) by[key(i)].push_back(i);
        std::vector<int> ho(nkey + 1, 0), ha, hb;
        std::vector<double> hc;
        for (int k = 0; k < nkey; k++) {
          ho[k] = (int) ha.size();
          for (int i : by[k]) { ha.push_back(ia(i)); hb.push_back(ib(i)); hc.push_back(c.cg[i]); }
        }
        ho[nkey] = (int) ha.size();
        copy_1d_int(ho, off, nm); copy_1d_int(ha, a, nm); copy_1d_int(hb, b, nm); copy_1d(hc, cv, nm);
      };
      const int nlm_l = (int) c.group_left.size(), nlm_r = (int) c.group_right.size();
      // forward / output-order VJP rows in file function order (neighbouring functions share operands),
      // each row writing its (possibly reordered) output column fcol
      std::vector<int> col(c.nfunc), inv(c.nfunc);
      for (int f = 0; f < c.nfunc; f++) col[f] = c.newcol.empty() ? f : c.newcol[f];
      for (int f = 0; f < c.nfunc; f++) inv[col[f]] = f;
      copy_1d_int(col, dc.fcol, "g3l:prod_fcol");
      csr(c.nfunc, [&](int i) { return inv[c.m_sum_ind[i]]; }, [&](int i) { return c.left_ind[i]; },
          [&](int i) { return c.right_ind[i]; }, dc.foff, dc.fa, dc.fb, dc.fcg, "g3l:prod_fcsr");
      // coupling paths (grace_3l_cpl_path.h); any term outside the compile-time pattern -> table kernels
      {
        dc.path_n = 0;
        auto blocks = [](const std::vector<int> &g, std::vector<int> &st, std::vector<int> &lv) {
          const int ng = g.empty() ? 0 : *std::max_element(g.begin(), g.end()) + 1;
          st.assign(ng, -1); lv.assign(ng, -1);
          std::vector<int> sz(ng, 0);
          for (int i = 0; i < (int) g.size(); i++) { if (st[g[i]] < 0) st[g[i]] = i; sz[g[i]]++; }
          for (int k = 0; k < ng; k++) {
            if (st[k] < 0 || sz[k] % 2 == 0 || (sz[k] - 1) / 2 > G3LPath::LMAX) return false;
            for (int i = 0; i < sz[k]; i++) if (g[st[k] + i] != k) return false;
            lv[k] = (sz[k] - 1) / 2;
          }
          return true;
        };
        std::vector<int> lst, llv, rst, rlv;
        bool ok = blocks(c.group_left, lst, llv) && blocks(c.group_right, rst, rlv);
        // rows f (file order) -> path key; every row's terms must share one key
        std::map<std::array<int, 3>, std::vector<int>> rows_of;   // key (gl, gr, L) -> rows
        std::vector<std::array<int, 3>> key_of(c.nfunc, {-1, -1, -1});
        for (int i = 0; ok && i < c.n_cg; i++) {
          const int f = inv[c.m_sum_ind[i]];
          const std::array<int, 3> k = {c.group_left[c.left_ind[i]], c.group_right[c.right_ind[i]], c.out_l[c.m_sum_ind[i]]};
          if (key_of[f][0] < 0) { key_of[f] = k; rows_of[k].push_back(f); }
          else if (key_of[f] != k) ok = false;
        }
        struct PathH { int l1, l2, L, loff, roff, f0, cgoff, gl, gr; };
        std::vector<PathH> paths;
        std::vector<double> pcg;
        for (auto &kv : rows_of) {
          if (!ok) break;
          auto fs = kv.second; std::sort(fs.begin(), fs.end());
          const int gl = kv.first[0], gr = kv.first[1], L = kv.first[2];
          if ((int) fs.size() != 2 * L + 1 || fs.back() - fs.front() != 2 * L) { ok = false; break; }
          while (pcg.size() % 4) pcg.push_back(0.0);   // 16-byte aligned cg block per path
          PathH ph{llv[gl], rlv[gr], L, lst[gl], rst[gr], fs.front(), (int) pcg.size(), gl, gr};
          if (std::abs(ph.l1 - ph.l2) > L || L > ph.l1 + ph.l2 || L > G3LPath::LMAX) { ok = false; break; }
          // one cg slot per compile-time candidate, in kernel order (M, m1, m2)
          for (int M = -L; M <= L; M++)
            for (int m1 = -ph.l1; m1 <= ph.l1; m1++)
              for (int m2 = -ph.l2; m2 <= ph.l2; m2++)
                if (G3LPath::cand(ph.l1, ph.l2, L, m1, m2, M)) pcg.push_back(0.0);
          paths.push_back(ph);
        }
        for (int i = 0; ok && i < c.n_cg; i++) {
          const int f = inv[c.m_sum_ind[i]];
          const auto &k = key_of[f];
          // find the path of this row (paths sorted by key in rows_of order; small, linear search is fine)
          const PathH *ph = nullptr;
          for (auto &q : paths) if (q.gl == k[0] && q.gr == k[1] && q.L == k[2]) { ph = &q; break; }
          if (!ph) { ok = false; break; }
          const int m1 = c.left_ind[i] - ph->loff - ph->l1, m2 = c.right_ind[i] - ph->roff - ph->l2, M = f - ph->f0 - ph->L;
          if (!G3LPath::cand(ph->l1, ph->l2, ph->L, m1, m2, M)) { ok = false; break; }
          int kk = ph->cgoff;
          for (int MM = -ph->L; MM <= ph->L; MM++)
            for (int a1 = -ph->l1; a1 <= ph->l1; a1++)
              for (int a2 = -ph->l2; a2 <= ph->l2; a2++)
                if (G3LPath::cand(ph->l1, ph->l2, ph->L, a1, a2, MM)) {
                  if (a1 == m1 && a2 == m2 && MM == M) pcg[kk] += c.cg[i];
                  kk++;
                }
        }
        if (ok && !paths.empty()) {
          // per side: every outer block (in lm order), its pairs (inner blocks in lm order), their paths by L
          for (int side = 0; side < 2; side++) {
            const auto &ost = side == 0 ? lst : rst; const auto &olv = side == 0 ? llv : rlv;
            std::vector<int> ob, pr, pt;
            for (int g = 0; g < (int) ost.size(); g++) {
              const int pr0 = (int) pr.size() / 4;
              std::map<int, std::vector<const PathH *>> by_inner;
              for (auto &q : paths) if ((side == 0 ? q.gl : q.gr) == g) by_inner[side == 0 ? q.gr : q.gl].push_back(&q);
              for (auto &kv : by_inner) {
                auto v = kv.second;
                std::sort(v.begin(), v.end(), [](const PathH *x, const PathH *y) { return x->L < y->L; });
                const int pt0 = (int) pt.size() / 3;
                for (auto *q : v) pt.insert(pt.end(), {q->L, q->f0, q->cgoff});
                const PathH *q = v.front();
                pr.insert(pr.end(), {side == 0 ? q->l2 : q->l1, side == 0 ? q->roff : q->loff, pt0, (int) pt.size() / 3});
              }
              ob.insert(ob.end(), {olv[g], ost[g], pr0, (int) pr.size() / 4});
            }
            dc.path_nob[side] = (int) ob.size() / 4;
            copy_1d_int(ob, dc.path_ob[side], "g3l:prod_path_ob");
            copy_1d_int(pr, dc.path_pr[side], "g3l:prod_path_pr");
            copy_1d_int(pt, dc.path_pt[side], "g3l:prod_path_pt");
          }
          dc.path_n = (int) paths.size();
          while (pcg.size() % 4) pcg.push_back(0.0);
          copy_1d(pcg, dc.path_cg, "g3l:prod_path_cg");
        }
      }
      csr(nlm_l, [&](int i) { return c.left_ind[i]; }, [&](int i) { return c.m_sum_ind[i]; },
          [&](int i) { return c.right_ind[i]; }, dc.loff, dc.la, dc.lb, dc.lcg, "g3l:prod_lcsr");
      csr(nlm_r, [&](int i) { return c.right_ind[i]; }, [&](int i) { return c.m_sum_ind[i]; },
          [&](int i) { return c.left_ind[i]; }, dc.roff, dc.ra, dc.rb, dc.rcg, "g3l:prod_rcsr");
    }
#ifdef GRACE_KK_GEMM
    // pre-gather + transpose U/V into BLAS-friendly per-lm-slab weights
    // for the batched CPLBwdProject GEMM. Uw(w,n,r) = U(group_left(w), r, n) so each
    // w-slab is a tight col-major [rank x n_left] matrix (ldb=rank, w-stride=n_left*rank).
    // Same for Vw. One-time; ~a few MB total across all products.
    {
      const int nlm_l = (int) c.group_left.size();
      const int nlm_r = (int) c.group_right.size();
      MemKK::realloc_kokkos(dc.Uw, "g3l:prod_Uw", std::max(nlm_l, 1),
                            std::max(c.n_left, 1), std::max(c.rank, 1));
      MemKK::realloc_kokkos(dc.Vw, "g3l:prod_Vw", std::max(nlm_r, 1),
                            std::max(c.n_right, 1), std::max(c.rank, 1));
      auto hUw = Kokkos::create_mirror_view(dc.Uw);
      auto hVw = Kokkos::create_mirror_view(dc.Vw);
      for (int w = 0; w < nlm_l; w++) {
        const int g = c.group_left[w];
        for (int n = 0; n < c.n_left; n++)
          for (int r = 0; r < c.rank; r++)
            hUw(w, n, r) = (NNScalar) c.U[((size_t) g * c.rank + r) * c.n_left + n];
      }
      for (int w = 0; w < nlm_r; w++) {
        const int g = c.group_right[w];
        for (int n = 0; n < c.n_right; n++)
          for (int r = 0; r < c.rank; r++)
            hVw(w, n, r) = (NNScalar) c.V[((size_t) g * c.rank + r) * c.n_right + n];
      }
      Kokkos::deep_copy(dc.Uw, hUw);
      Kokkos::deep_copy(dc.Vw, hVw);
      dc.lr_ok = nlm_l == nlm_r && c.n_left == c.n_right && nlm_l > 0 && c.rank > 0;
      if (dc.lr_ok) {
        const bool fold = c.norm_u != c.norm_v;
        dc.lr_alpha = fold ? NNScalar(1) : (NNScalar) c.norm_u;
        MemKK::realloc_kokkos(dc.Wlr, "g3l:prod_Wlr", nlm_l, c.n_left, 2 * c.rank);
        auto hW = Kokkos::create_mirror_view(dc.Wlr);
        for (int w = 0; w < nlm_l; w++)
          for (int n = 0; n < c.n_left; n++)
            for (int r = 0; r < c.rank; r++) {
              hW(w, n, r) = fold ? (NNScalar) (c.norm_u * (double) hUw(w, n, r)) : hUw(w, n, r);
              hW(w, n, c.rank + r) = fold ? (NNScalar) (c.norm_v * (double) hVw(w, n, r)) : hVw(w, n, r);
            }
        Kokkos::deep_copy(dc.Wlr, hW);
      }
    }
#endif
  }
  {
    cpl_path = getenv("GRACE3L_CPL_TABLES") == nullptr;   // GRACE3L_CPL_TABLES: table kernels for every product
    int npath = 0;
    for (int pi = 0; pi < n_prods; pi++) npath += d_prod[pi].path_n > 0;
    if (sizeof(NNScalar) == 4)
      log.info(fmt("[GRACE-3L/KK] cp_l couple: {} of {} products by coupling path\n", cpl_path ? npath : 0, n_prods));
  }

  // cp_l scratch caps: max rank / max n_lm over ALL products so the two shared
  // projection scratch views (d_cpl_lproj/rproj) serve every compute_cp_l call.
  cpl_scratch_rank = 0;
  cpl_scratch_nlm = 0;
  for (int pi = 0; pi < n_prods; pi++) {
    const auto &c = m.prod[pi];
    cpl_scratch_rank = std::max(cpl_scratch_rank, c.rank);
    cpl_scratch_nlm = std::max(cpl_scratch_nlm,
                               std::max((int) c.group_left.size(), (int) c.group_right.size()));
  }

  // ---- FunctionReduceN ----
  n_reduces = (int) m.reduce.size();
  for (int ri = 0; ri < n_reduces; ri++) {
    const auto &r = m.reduce[ri];
    DeviceReduce &dr = d_reduce[ri];
    dr.n_out = r.n_out; dr.n_funcs = r.n_funcs;
    dr.only_invar = r.only_invar ? 1 : 0;
    dr.elem_dep = r.elem_dep ? 1 : 0;
    dr.n_instr = (int) r.instr.size();
    const int n_types = r.elem_dep ? m.n_elements : 1;
    for (int ci = 0; ci < dr.n_instr; ci++) {
      const auto &ins = r.instr[ci];
      dr.n_in[ci] = ins.n_in; dr.w_shape[ci] = ins.w_shape; dr.n_conn[ci] = ins.n_conn;
      dr.norm[ci] = (NNScalar) ins.norm;
      // W stored uniformly 4D [n_types, n_out, n_in, w_shape] (n_types=1 if !elem_dep).
      copy_4d(ins.W, dr.W[ci], n_types, r.n_out, ins.n_in, ins.w_shape, "g3l:reduce_W");
      copy_1d_int(ins.collect_ind, dr.collect_ind[ci], "g3l:reduce_ci");
      copy_1d_int(ins.w_l_tile, dr.w_l_tile[ci], "g3l:reduce_wlt");
      copy_1d_int(ins.total_sum_ind, dr.total_sum_ind[ci], "g3l:reduce_tsi");
    }
    dr.has_norm_map = r.has_norm_map ? 1 : 0;
    if (r.has_norm_map) copy_1d(r.norm_map, dr.norm_map, "g3l:reduce_norm_map");
    // blocked hand path (blocks of 8 outputs / inputs)
    dr.blk_ok = 0;
    if (!(!r.elem_dep && !r.only_invar && r.instr.size() == 1)) {
      bool ok = r.n_out % 8 == 0;
      long tot = 0;
      for (int ci = 0; ci < dr.n_instr; ci++) {
        ok = ok && r.instr[ci].n_in % 8 == 0;
        dr.wb_off[ci] = tot;
        tot += (long) n_types * r.instr[ci].w_shape * r.instr[ci].n_in * r.n_out;
      }
      if (ok) {
        std::vector<double> hWb(tot);
        const int K = r.n_out;
        for (int ci = 0; ci < dr.n_instr; ci++) {
          const auto &ins = r.instr[ci];
          const int T = ins.w_shape, NI = ins.n_in;
          for (int e = 0; e < n_types; e++)
            for (int k = 0; k < K; k++)
              for (int n = 0; n < NI; n++)
                for (int t = 0; t < T; t++)
                  hWb[dr.wb_off[ci] + (((long) e * T + t) * NI + n) * K + k] =
                      ins.norm * ins.W[(((size_t) e * K + k) * NI + n) * T + t];
        }
        copy_1d(hWb, dr.Wb, "g3l:reduce_Wb");
        const int shape0 = r.only_invar ? 1 : r.n_funcs;
        std::vector<std::vector<std::array<int, 3>>> byf(shape0);
        for (int ci = 0; ci < dr.n_instr; ci++) {
          const auto &ins = r.instr[ci];
          for (int c = 0; c < ins.n_conn; c++) {
            const int f = r.only_invar ? 0 : ins.total_sum_ind[c];
            if (f < 0 || f >= shape0) GRACE_KK_ERROR("GRACE-3L/KK: reduce {} output index out of range", r.name);
            byf[f].push_back({ci, ins.collect_ind[c], ins.w_l_tile[c]});
          }
        }
        std::vector<int> ho(shape0 + 1, 0), hi, hc, ht;
        for (int f = 0; f < shape0; f++) {
          ho[f] = (int) hi.size();
          for (auto &x : byf[f]) { hi.push_back(x[0]); hc.push_back(x[1]); ht.push_back(x[2]); }
        }
        ho[shape0] = (int) hi.size();
        copy_1d_int(ho, dr.roff, "g3l:reduce_roff"); copy_1d_int(hi, dr.rins, "g3l:reduce_rins");
        copy_1d_int(hc, dr.rci, "g3l:reduce_rci"); copy_1d_int(ht, dr.rtile, "g3l:reduce_rtile");
        for (int ci = 0; ci < dr.n_instr; ci++) {
          const auto &cis = r.instr[ci].collect_ind;
          dr.ci_unique[ci] = (int) (std::unordered_set<int>(cis.begin(), cis.end()).size() == cis.size());
        }
        dr.blk_ok = 1;
      }
    }
  }

#ifdef GRACE_KK_GEMM
  // ---- BLAS batched-GEMM scratch for the element-independent, non-only_invar,
  //      single-instruction product reduces (A*_*_red). Fold norm*norm_map(f) into a
  //      per-connection col-major [n_out x n_in] weight block Wsc[c] (serves BOTH the
  //      forward rounds and the backward single batch), and precompute the forward round
  //      schedule (connections grouped so every round writes DISTINCT output f -> the
  //      beta=1 batched accumulate into `out` is race-free within a round). collect_ind must
  //      be a bijection over the input lm-columns so the backward writes DISTINCT adj_in
  //      slabs -> a single beta=1 batch with no scatter; this is verified per reduce below
  //      and BLAS is disabled (hand-kernel fallback) for any reduce that violates it. ----
  reduceN_max_conn = 0;
  for (int ri = 0; ri < n_reduces; ri++) {
    const auto &r = m.reduce[ri];
    ReduceHost &hr = h_reduce[ri];
    hr.gemm_form_ok = (!r.elem_dep && !r.only_invar && r.instr.size() == 1);
    if (!hr.gemm_form_ok) continue;
    const auto &ins = r.instr[0];
    // The BLAS backward accumulates into adj_in slabs indexed by collect_ind in a single
    // beta=1 batch, which is only race-free if collect_ind is a bijection (distinct value
    // per connection). Verify it here; if any index repeats, disable BLAS for this reduce
    // and fall back to the hand kernel (which scatters safely).
    {
      std::unordered_set<int> seen_ci;
      bool ci_unique = true;
      for (int c = 0; c < ins.n_conn; c++)
        if (!seen_ci.insert(ins.collect_ind[c]).second) { ci_unique = false; break; }
      if (!ci_unique) { hr.gemm_form_ok = false; continue; }
    }
    const int n_out = r.n_out, n_in = ins.n_in, n_conn = ins.n_conn, w_shape = ins.w_shape;
    const int n_f = r.n_funcs;
    hr.n_conn = n_conn; hr.n_out = n_out; hr.n_in = n_in;
    hr.ci = ins.collect_ind;                 // [n_conn]
    hr.tsi = ins.total_sum_ind;              // [n_conn]
    // per-connection scaled weight blocks: Wsc[c](k,n) = norm*norm_map(tsi[c])*W(0,k,n,tile)
    // stored col-major [n_out x n_in] (k fastest), c-stride = n_out*n_in.
    MemKK::realloc_kokkos(d_reduce[ri].Wsc, "g3l:reduce_Wsc", (size_t) n_conn * n_out * n_in);
    auto hWsc = Kokkos::create_mirror_view(d_reduce[ri].Wsc);
    for (int c = 0; c < n_conn; c++) {
      const int tile = ins.w_l_tile[c];
      const int f = hr.tsi[c];
      const double sc = ins.norm * (r.has_norm_map ? r.norm_map[f] : 1.0);
      const size_t base = (size_t) c * n_out * n_in;
      for (int n = 0; n < n_in; n++)
        for (int k = 0; k < n_out; k++)
          hWsc(base + k + (size_t) n_out * n) =
              (NNScalar) (sc * ins.W[((size_t) k * n_in + n) * w_shape + tile]);
    }
    Kokkos::deep_copy(d_reduce[ri].Wsc, hWsc);
    // forward round schedule: round rr collects the rr-th connection of every output f.
    std::vector<std::vector<int>> byf(n_f);
    for (int c = 0; c < n_conn; c++) byf[hr.tsi[c]].push_back(c);
    int maxr = 0;
    for (int f = 0; f < n_f; f++) maxr = std::max(maxr, (int) byf[f].size());
    if (maxr > GRACE3L_MAX_REDUCE_ROUNDS)
      GRACE_KK_ERROR("GRACE-3L/KK: reduce {} needs {} BLAS fwd rounds > cap {}",
                 r.name, maxr, GRACE3L_MAX_REDUCE_ROUNDS);
    hr.fwd_order.clear();
    hr.round_ptr[0] = 0;
    for (int rr = 0; rr < maxr; rr++) {
      for (int f = 0; f < n_f; f++)
        if ((int) byf[f].size() > rr) hr.fwd_order.push_back(byf[f][rr]);
      hr.round_ptr[rr + 1] = (int) hr.fwd_order.size();
    }
    hr.nrounds = maxr;
    reduceN_max_conn = std::max(reduceN_max_conn, n_conn);
    // grouped form: connections of each output on contiguous columns, in connection order
    hr.grouped = true; hr.has_empty = false;
    hr.gstart.assign(n_f, 0); hr.gn.assign(n_f, 0); hr.boff.assign(n_f, 0); hr.gsets.clear();
    long btot = 0;
    for (int f = 0; f < n_f && hr.grouped; f++) {
      hr.gn[f] = (int) byf[f].size();
      if (hr.gn[f] == 0) { hr.has_empty = true; continue; }
      hr.gstart[f] = hr.ci[byf[f][0]];
      for (int q = 0; q < hr.gn[f]; q++) if (hr.ci[byf[f][q]] != hr.gstart[f] + q) hr.grouped = false;
      hr.boff[f] = btot;
      btot += (long) hr.gn[f] * n_in * n_out;
    }
    if (hr.grouped) {
      std::vector<double> hB(btot);
      for (int f = 0; f < n_f; f++) {
        const int K = hr.gn[f] * n_in;
        for (int q = 0; q < hr.gn[f]; q++) {
          const int c = byf[f][q], tile = ins.w_l_tile[c];
          const double sc = ins.norm * (r.has_norm_map ? r.norm_map[f] : 1.0);
          for (int k = 0; k < n_out; k++)
            for (int n = 0; n < n_in; n++)
              hB[hr.boff[f] + (q * n_in + n) + (long) K * k] = sc * ins.W[((size_t) k * n_in + n) * w_shape + tile];
        }
      }
      copy_1d(hB, d_Bg[ri], "g3l:reduce_Bg");
      std::map<int, std::vector<int>> bysz;
      for (int f = 0; f < n_f; f++) if (hr.gn[f] > 0) bysz[hr.gn[f]].push_back(f);
      for (auto &kv : bysz) hr.gsets.push_back(kv.second);
    }
  }
  if (reduceN_max_conn > 0) {
    reduceN_max_conn = std::max(reduceN_max_conn, GRACE3L_MAX_REDUCE_NFUNCS);
    MemKK::realloc_kokkos(d_redptr, "g3l:reduceN_ptrbuf", (size_t) 3 * reduceN_max_conn);
    h_redptr = Kokkos::View<uintptr_t*, Kokkos::HostSpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "g3l:reduceN_hptr"),
        (size_t) 3 * reduceN_max_conn);
  }
#endif

  // ---- FCRight2Left ----
  n_fcs = (int) m.fc.size();
  for (int fi = 0; fi < n_fcs; fi++) {
    const auto &f = m.fc[fi];
    DeviceFC &df = d_fc[fi];
    df.n_out = f.n_out; df.left_coefs = f.left_coefs ? 1 : 0;
    df.n_funcs_left = f.n_funcs_left; df.n_funcs_right = f.n_funcs_right;
    df.n_in_left = f.n_in_left; df.n_in_right = f.n_in_right;
    df.w_shape_left = f.w_shape_left; df.w_shape_right = f.w_shape_right;
    df.norm_left = (NNScalar) f.norm_left; df.norm_right = (NNScalar) f.norm_right;
    copy_3d(f.w_right, df.w_right, f.n_out, f.n_in_right, f.w_shape_right, "g3l:fc_wr");
    if (f.left_coefs && !f.w_left.empty())
      copy_3d(f.w_left, df.w_left, f.n_out, f.n_in_left, f.w_shape_left, "g3l:fc_wl");
    copy_1d_int(f.w_tile_left, df.w_tile_left, "g3l:fc_wtl");
    copy_1d_int(f.w_tile_right, df.w_tile_right, "g3l:fc_wtr");
    copy_1d_int(f.collect_to, df.collect_to, "g3l:fc_ct");
    copy_1d_int(f.collect_from, df.collect_from, "g3l:fc_cf");
    copy_1d(f.norm_out_factor, df.norm_out_factor, "g3l:fc_nof");
    df.gemm_ok = 0;
    {
      bool ok = !f.left_coefs && f.n_funcs_right == f.n_funcs_left;
      for (int c = 0; ok && c < f.n_funcs_right; c++) ok = f.collect_from[c] == c && f.collect_to[c] == c;
      if (ok) {
        const int nf = f.n_funcs_left, no = f.n_out, ni = f.n_in_right, nt = f.w_shape_right;
        MemKK::realloc_kokkos(df.Bw, "g3l:fc_Bw", nf, no, ni);
        auto hB = Kokkos::create_mirror_view(df.Bw);
        for (int w = 0; w < nf; w++) {
          const int t = f.w_tile_right[w];
          const double sc = f.norm_right * f.norm_out_factor[w];
          for (int k = 0; k < no; k++)
            for (int n = 0; n < ni; n++) hB(w, k, n) = (NNScalar) (sc * f.w_right[((size_t) k * ni + n) * nt + t]);
        }
        Kokkos::deep_copy(df.Bw, hB);
        df.gemm_ok = 1;
      }
    }
  }

  // ---- EquivariantRMSNorm ----
  n_eqnorms = (int) m.eqnorm.size();
  for (int ei = 0; ei < n_eqnorms; ei++) {
    const auto &e = m.eqnorm[ei];
    DeviceEqNorm &de = d_eqnorm[ei];
    de.center_l0 = e.center_l0; de.n_groups = e.n_groups; de.n_out = e.n_out; de.M = e.M;
    de.eps = (NNScalar) e.eps;
    copy_2d(e.affine_weight, de.affine_weight, e.n_groups, e.n_out, "g3l:eqnorm_aff");
    copy_1d(e.degree_weights, de.degree_weights, "g3l:eqnorm_dw");
    copy_1d(e.l0_mask, de.l0_mask, "g3l:eqnorm_l0");
    copy_1d_int(e.expand_index, de.expand_index, "g3l:eqnorm_exp");
  }

  // ---- InvariantLayerRMSNorm ----
  n_rmsnorms = (int) m.rmsnorm.size();
  for (int ni = 0; ni < n_rmsnorms; ni++) {
    const auto &r = m.rmsnorm[ni];
    DeviceRMSNorm &dn = d_rmsnorm[ni];
    dn.type = r.type; dn.n_out = r.n_out; dn.scale_len = (int) r.scale.size();
    copy_1d(r.scale, dn.scale, "g3l:rmsnorm_scale");
  }

  // ---- Readout energy MLP (no bias) ----
  {
    const auto &mlp = m.energy_mlp;
    energy_n_layers = mlp.n_layers;
    energy_activation = m.energy_mlp_activation;
    int max_in = 0, max_out = 0;
    for (int i = 0; i < energy_n_layers; i++) { max_in = std::max(max_in, mlp.n_in[i]); max_out = std::max(max_out, mlp.n_out[i]); }
    energy_max_dim = std::max(max_in, max_out);
    MemKK::realloc_kokkos(d_energy_W, "g3l:energy_W", std::max(energy_n_layers, 1), std::max(max_in, 1), std::max(max_out, 1));
    auto hW = Kokkos::create_mirror_view(d_energy_W);
    Kokkos::deep_copy(hW, (NNScalar) 0);
    for (int l = 0; l < energy_n_layers; l++)
      for (int i = 0; i < mlp.n_in[l]; i++)
        for (int j = 0; j < mlp.n_out[l]; j++)
          hW(l, i, j) = (NNScalar) mlp.W[l][(size_t) i * mlp.n_out[l] + j];
    Kokkos::deep_copy(d_energy_W, hW);
    MemKK::realloc_kokkos(d_energy_norms, "g3l:energy_norms", std::max(energy_n_layers, 1));
    auto hn = Kokkos::create_mirror_view(d_energy_norms);
    for (int i = 0; i < energy_n_layers; i++) hn(i) = (NNScalar) mlp.norm[i];
    Kokkos::deep_copy(d_energy_norms, hn);
    MemKK::realloc_kokkos(d_energy_dims, "g3l:energy_dims", energy_n_layers + 1);
    auto hd = Kokkos::create_mirror_view(d_energy_dims);
    hd(0) = mlp.n_in.empty() ? 0 : mlp.n_in[0];
    for (int i = 0; i < energy_n_layers; i++) hd(i + 1) = mlp.n_out[i];
    Kokkos::deep_copy(d_energy_dims, hd);
  }

  // ---- UQ / extrapolation-grade artifacts (optional; schema v6 basis-RP) ----
  // Stage the GMM arrays + R onto the device; set the scalar members. The
  // per-block basis-width validation (needs idx_reduce_rho{1,2,3}) happens in
  // setup() after those reduce indices are resolved.
  if (m.has_uq) {
    const int E = m.uq_n_elements;
    const int K = m.uq_max_clusters;
    const int D = m.uq_feature_dim;        // = rp_dim + n_density
    const int RP = m.uq_rp_dim;            // projection width (R cols)
    if (E != m.n_elements)
      GRACE_KK_ERROR("GRACE-3L/KK UQ: uq_n_elements ({}) != model n_elements ({})",
                 E, m.n_elements);
    if (D > GRACE3L_UQ_MAX_RP_DIM)
      GRACE_KK_ERROR("GRACE-3L/KK UQ: feature_dim ({}) exceeds GRACE3L_UQ_MAX_RP_DIM ({})",
                 D, GRACE3L_UQ_MAX_RP_DIM);
    has_uq = true;
    uq_Kmax = K;
    uq_D = D;
    uq_rp_dim = RP;
    uq_n_density = m.uq_n_density;
    uq_normalize = m.uq_normalize;
    uq_density_scale = m.uq_density_scale;
    uq_d_basis = m.uq_d_basis;
    copy_3d(m.uq_centroids, d_uq_centroids, E, K, D, "g3l:uq_centroids");
    copy_4d(m.uq_inv_cov, d_uq_inv_cov, E, K, D, D, "g3l:uq_inv_cov");
    copy_1d_int(m.uq_n_clusters, d_uq_n_clusters, "g3l:uq_n_clusters");
    copy_2d(m.uq_interp_thresholds, d_uq_interp_thresholds, E, K, "g3l:uq_interp_thr");
    copy_2d(m.uq_rp_matrix, d_uq_rp_matrix, m.uq_d_basis, RP, "g3l:uq_rp_matrix");
  }
}

// ======================================================================
// grow: allocate the per-chunk geometry arrays and the per-atom / per-bond NN
// intermediate arrays.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::grow(int natom, int maxneigh_in)
{
  // HIP: row count (leading dimension) of the per-chunk arrays the chunk rounded up to a multiple of 128 (aligned GEMM
  // columns; the GEMMs still run over the chunk), but not a power of two (16384 rows put the rank / function strides of the cp_l, eqnorm and A1 kernels on the same
  // memory channels): MI250X GRACE-3L-OMAT-large 5k atoms -0.2 .. -0.6 %, 125k - 128k atoms -2.8 .. -5.3 %. Not on
  // CUDA (A100 0.4 - 0.6 % slower), not up to 1024 atoms (chunksize auto counts the bytes of grow(1024) per atom).
  if constexpr (is_hip_v<DeviceType>) {
    if (natom > 1024) {
      natom = (natom + 127) / 128 * 128;
      if ((natom & (natom - 1)) == 0) natom += 128;
    }
  }
  // Transient tensors (an SPBF / cp_l product output lives only until its reduce, its adjoint only from the
  // reduce backward to the product backward) share two buffers: views into d_pool_fwd / d_pool_adj.
  struct PoolEntry { t_nn_3d *v; int d1, d2; };
  std::vector<PoolEntry> fwd, adj;
  auto pool_add = [](std::vector<PoolEntry> &p, t_nn_3d &v, int d1, int d2) { p.push_back({&v, d1, d2}); };
  bool pool_grown = false;

  if (realloc_all || (int) d_ncount.extent(0) < natom) {
    MemKK::realloc_kokkos(d_ncount, "g3l:ncount", natom);
    MemKK::realloc_kokkos(d_mu_i, "g3l:mu_i", natom);
    MemKK::realloc_kokkos(d_A1, "g3l:A1", natom, L1_nradmax, (lmax + 1) * (lmax + 1));
    // cp_l product output A1_2 [atoms, rank, nfunc] + shared projection scratch.
    pool_grown = true;
    pool_add(fwd, d_A1_2, std::max(d_prod[0].rank, 1), std::max(d_prod[0].nfunc, 1));
    {
      long tot = 0;
      for (int pi = 0; pi < n_prods; pi++) {
        const DeviceCPL &p = d_prod[pi];
        const int nlm = std::max((int) p.group_left.extent(0), (int) p.group_right.extent(0));
        cpl_slot_off[pi] = tot;
        cpl_stamp[pi] = 0;
        tot += (long) natom * 2 * std::max(p.rank, 1) * std::max(nlm, 1);
      }
      cpl_ld = natom;
      MemKK::realloc_kokkos(d_cpl_store, "g3l:cpl_store", std::max(tot, 1L));
    }
    // FunctionReduceN / FCRight2Left outputs.
    MemKK::realloc_kokkos(d_A1_2_red, "g3l:A1_2_red", natom,
                          std::max(d_reduce[idx_reduce_A1_2_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A1_2_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A1_2a, "g3l:A1_2a", fold_on[0] ? 1 : natom,
                          std::max(d_fc[idx_fc_A1_2a].n_out, 1),
                          std::max(d_fc[idx_fc_A1_2a].n_funcs_left, 1));
    // Remaining L1 chain.
    pool_add(fwd, d_A1_3, std::max(d_prod[idx_prod_A1_3].rank, 1), std::max(d_prod[idx_prod_A1_3].nfunc, 1));
    MemKK::realloc_kokkos(d_A1_3_red, "g3l:A1_3_red", natom,
                          std::max(d_reduce[idx_reduce_A1_3_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A1_3_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A1_2b, "g3l:A1_2b", fold_on[0] ? 1 : natom,
                          std::max(d_fc[idx_fc_A1_2b].n_out, 1),
                          std::max(d_fc[idx_fc_A1_2b].n_funcs_left, 1));
    pool_add(fwd, d_A1_4, std::max(d_prod[idx_prod_A1_4].rank, 1), std::max(d_prod[idx_prod_A1_4].nfunc, 1));
    MemKK::realloc_kokkos(d_A1_4_red, "g3l:A1_4_red", natom,
                          std::max(d_reduce[idx_reduce_A1_4_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A1_4_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_eq1, "g3l:eq1", natom,
                          std::max(d_reduce[idx_reduce_eq1].n_out, 1),
                          std::max(d_reduce[idx_reduce_eq1].n_funcs, 1));
    MemKK::realloc_kokkos(d_eq1_norm, "g3l:eq1_norm", natom,
                          std::max(d_eqnorm[idx_eqnorm_eq1_norm].n_out, 1),
                          std::max(d_eqnorm[idx_eqnorm_eq1_norm].M, 1));
    MemKK::realloc_kokkos(d_rho1, "g3l:rho1", natom,
                          std::max(d_reduce[idx_reduce_rho1].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho1].n_funcs, 1));
    MemKK::realloc_kokkos(d_rho1_norm, "g3l:rho1_norm", natom,
                          std::max(d_rmsnorm[idx_rmsnorm_rho1_norm].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho1].n_funcs, 1));
    // L2 equiv-ind SPBF basis A2 [atoms, A2_n_rad_max, A2_nfunc].
    pool_add(fwd, d_A2, std::max(d_spbf[idx_spbf_A2].n_rad_max, 1), std::max(d_spbf[idx_spbf_A2].nfunc, 1));
    // remaining L2 chain (identical structure to L1's
    // allocations, EXCEPT everything downstream of A2 here is built on
    // A2_red — the one extra reduce L2 has that L1 does not).
    MemKK::realloc_kokkos(d_A2_red, "g3l:A2_red", natom,
                          std::max(d_reduce[idx_reduce_A2_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A2_red].n_funcs, 1));
    pool_add(fwd, d_A2_2, std::max(d_prod[idx_prod_A2_2].rank, 1), std::max(d_prod[idx_prod_A2_2].nfunc, 1));
    MemKK::realloc_kokkos(d_A2_2_red, "g3l:A2_2_red", natom,
                          std::max(d_reduce[idx_reduce_A2_2_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A2_2_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A2_2a, "g3l:A2_2a", fold_on[1] ? 1 : natom,
                          std::max(d_fc[idx_fc_A2_2a].n_out, 1),
                          std::max(d_fc[idx_fc_A2_2a].n_funcs_left, 1));
    pool_add(fwd, d_A2_3, std::max(d_prod[idx_prod_A2_3].rank, 1), std::max(d_prod[idx_prod_A2_3].nfunc, 1));
    MemKK::realloc_kokkos(d_A2_3_red, "g3l:A2_3_red", natom,
                          std::max(d_reduce[idx_reduce_A2_3_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A2_3_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A2_2b, "g3l:A2_2b", fold_on[1] ? 1 : natom,
                          std::max(d_fc[idx_fc_A2_2b].n_out, 1),
                          std::max(d_fc[idx_fc_A2_2b].n_funcs_left, 1));
    pool_add(fwd, d_A2_4, std::max(d_prod[idx_prod_A2_4].rank, 1), std::max(d_prod[idx_prod_A2_4].nfunc, 1));
    MemKK::realloc_kokkos(d_A2_4_red, "g3l:A2_4_red", natom,
                          std::max(d_reduce[idx_reduce_A2_4_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A2_4_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_eq2, "g3l:eq2", natom,
                          std::max(d_reduce[idx_reduce_eq2].n_out, 1),
                          std::max(d_reduce[idx_reduce_eq2].n_funcs, 1));
    MemKK::realloc_kokkos(d_eq2_norm, "g3l:eq2_norm", natom,
                          std::max(d_eqnorm[idx_eqnorm_eq2_norm].n_out, 1),
                          std::max(d_eqnorm[idx_eqnorm_eq2_norm].M, 1));
    MemKK::realloc_kokkos(d_rho2, "g3l:rho2", natom,
                          std::max(d_reduce[idx_reduce_rho2].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho2].n_funcs, 1));
    MemKK::realloc_kokkos(d_rho2_norm, "g3l:rho2_norm", natom,
                          std::max(d_rmsnorm[idx_rmsnorm_rho2_norm].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho2].n_funcs, 1));
    // L3 equiv-ind SPBF basis A3 [atoms, A3_n_rad_max, A3_nfunc] +
    // the terminal L3 chain. Identical structure to L2's allocations
    // (everything downstream of A3 is built on A3_red) EXCEPT there is NO eq3;
    // A3_3/A3_4 are Lmax=0 so their dims come from the resolved metadata.
    pool_add(fwd, d_A3, std::max(d_spbf[idx_spbf_A3].n_rad_max, 1), std::max(d_spbf[idx_spbf_A3].nfunc, 1));
    MemKK::realloc_kokkos(d_A3_red, "g3l:A3_red", natom,
                          std::max(d_reduce[idx_reduce_A3_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A3_red].n_funcs, 1));
    pool_add(fwd, d_A3_2, std::max(d_prod[idx_prod_A3_2].rank, 1), std::max(d_prod[idx_prod_A3_2].nfunc, 1));
    MemKK::realloc_kokkos(d_A3_2_red, "g3l:A3_2_red", natom,
                          std::max(d_reduce[idx_reduce_A3_2_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A3_2_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A3_2a, "g3l:A3_2a", fold_on[2] ? 1 : natom,
                          std::max(d_fc[idx_fc_A3_2a].n_out, 1),
                          std::max(d_fc[idx_fc_A3_2a].n_funcs_left, 1));
    pool_add(fwd, d_A3_3, std::max(d_prod[idx_prod_A3_3].rank, 1), std::max(d_prod[idx_prod_A3_3].nfunc, 1));
    MemKK::realloc_kokkos(d_A3_3_red, "g3l:A3_3_red", natom,
                          std::max(d_reduce[idx_reduce_A3_3_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A3_3_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A3_2b, "g3l:A3_2b", fold_on[2] ? 1 : natom,
                          std::max(d_fc[idx_fc_A3_2b].n_out, 1),
                          std::max(d_fc[idx_fc_A3_2b].n_funcs_left, 1));
    pool_add(fwd, d_A3_4, std::max(d_prod[idx_prod_A3_4].rank, 1), std::max(d_prod[idx_prod_A3_4].nfunc, 1));
    MemKK::realloc_kokkos(d_A3_4_red, "g3l:A3_4_red", natom,
                          std::max(d_reduce[idx_reduce_A3_4_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A3_4_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_rho3, "g3l:rho3", natom,
                          std::max(d_reduce[idx_reduce_rho3].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho3].n_funcs, 1));
    MemKK::realloc_kokkos(d_rho3_norm, "g3l:rho3_norm", natom,
                          std::max(d_rmsnorm[idx_rmsnorm_rho3_norm].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho3].n_funcs, 1));
    // per-atom readout energy (fp64), chunk-local like rho3_norm.
    MemKK::realloc_kokkos(d_e_atom, "g3l:e_atom", natom);

    // ---- backward adjoint buffers (each mirrors its forward tensor's
    // shape). d_rho{1,2}_norm_adj are seeded by the readout backward (P3) and
    // consumed in P4 / P5. ----
    // d_rho1_norm_adj/d_rho2_norm_adj are natom ([nall]) and
    // allocated in grow_global(); here we allocate only their chunk-scratch
    // gather targets (fed to the shared TagRMSNormBwd per chunk in P5/P4).
    MemKK::realloc_kokkos(d_rho1_norm_adj_local, "g3l:rho1_norm_adj_local", natom,
                          std::max(d_rmsnorm[idx_rmsnorm_rho1_norm].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho1].n_funcs, 1));
    MemKK::realloc_kokkos(d_rho2_norm_adj_local, "g3l:rho2_norm_adj_local", natom,
                          std::max(d_rmsnorm[idx_rmsnorm_rho2_norm].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho2].n_funcs, 1));
    MemKK::realloc_kokkos(d_rho3_norm_adj, "g3l:rho3_norm_adj", natom,
                          std::max(d_rmsnorm[idx_rmsnorm_rho3_norm].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho3].n_funcs, 1));
    MemKK::realloc_kokkos(d_rho1_adj, "g3l:rho1_adj", natom,
                          std::max(d_reduce[idx_reduce_rho1].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho1].n_funcs, 1));
    MemKK::realloc_kokkos(d_rho2_adj, "g3l:rho2_adj", natom,
                          std::max(d_reduce[idx_reduce_rho2].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho2].n_funcs, 1));
    MemKK::realloc_kokkos(d_rho3_adj, "g3l:rho3_adj", natom,
                          std::max(d_reduce[idx_reduce_rho3].n_out, 1),
                          std::max(d_reduce[idx_reduce_rho3].n_funcs, 1));
    MemKK::realloc_kokkos(d_A3_red_adj, "g3l:A3_red_adj", natom,
                          std::max(d_reduce[idx_reduce_A3_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A3_red].n_funcs, 1));
    pool_add(adj, d_A3_2_adj, std::max(d_prod[idx_prod_A3_2].rank, 1), std::max(d_prod[idx_prod_A3_2].nfunc, 1));
    MemKK::realloc_kokkos(d_A3_2_red_adj, "g3l:A3_2_red_adj", natom,
                          std::max(d_reduce[idx_reduce_A3_2_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A3_2_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A3_2a_adj, "g3l:A3_2a_adj", fold_on[2] ? 1 : natom,
                          std::max(d_fc[idx_fc_A3_2a].n_out, 1),
                          std::max(d_fc[idx_fc_A3_2a].n_funcs_left, 1));
    pool_add(adj, d_A3_3_adj, std::max(d_prod[idx_prod_A3_3].rank, 1), std::max(d_prod[idx_prod_A3_3].nfunc, 1));
    MemKK::realloc_kokkos(d_A3_3_red_adj, "g3l:A3_3_red_adj", natom,
                          std::max(d_reduce[idx_reduce_A3_3_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A3_3_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A3_2b_adj, "g3l:A3_2b_adj", fold_on[2] ? 1 : natom,
                          std::max(d_fc[idx_fc_A3_2b].n_out, 1),
                          std::max(d_fc[idx_fc_A3_2b].n_funcs_left, 1));
    pool_add(adj, d_A3_4_adj, std::max(d_prod[idx_prod_A3_4].rank, 1), std::max(d_prod[idx_prod_A3_4].nfunc, 1));
    MemKK::realloc_kokkos(d_A3_4_red_adj, "g3l:A3_4_red_adj", natom,
                          std::max(d_reduce[idx_reduce_A3_4_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A3_4_red].n_funcs, 1));
    pool_add(adj, d_A3_adj, std::max(d_spbf[idx_spbf_A3].n_rad_max, 1), std::max(d_spbf[idx_spbf_A3].nfunc, 1));
    // shared adj_prod buffer. Sized to the MAX over the equiv layers
    // (A2/A3) of n_rad_max and P = nlm_y*n_lm_ind, so one buffer serves both.
    {
      const int nlm_y = (lmax + 1) * (lmax + 1);
      const int adjp_nrad = std::max({d_spbf[idx_spbf_A2].n_rad_max,
                                      d_spbf[idx_spbf_A3].n_rad_max, 1});
      const int adjp_P = std::max({nlm_y * d_spbf[idx_spbf_A2].n_lm_ind,
                                   nlm_y * d_spbf[idx_spbf_A3].n_lm_ind, 1});
      const bool need_adjp = !(spbf_fast_ok(idx_spbf_A2) && spbf_fast_ok(idx_spbf_A3));
      MemKK::realloc_kokkos(d_adj_prod, "g3l:adj_prod", need_adjp ? natom : 1, adjp_nrad, adjp_P);
      int qmax = 1;
      for (int si : {idx_spbf_A2, idx_spbf_A3})
        if (spbf_fast_ok(si)) qmax = std::max(qmax, d_spbf[si].n_rad_max * d_spbf[si].nfunc);
      MemKK::realloc_kokkos(d_spbf_T, "g3l:spbf_T", qmax > 1 ? natom : 1, qmax);
    }
    // cp_l backward rank-projection scratch (same size as the forward scratch).
    MemKK::realloc_kokkos(d_cpl_adjbuf, "g3l:cpl_adjbuf",
                          (size_t) natom * 2 * std::max(cpl_scratch_rank, 1) * std::max(cpl_scratch_nlm, 1));

    // ---- Layer-2 backward adjoint buffers (mirror the L3 d_A3_* set;
    // each matches its forward L2 tensor's shape). d_rho2_(norm_)adj already
    // allocated above. ----
    MemKK::realloc_kokkos(d_A2_red_adj, "g3l:A2_red_adj", natom,
                          std::max(d_reduce[idx_reduce_A2_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A2_red].n_funcs, 1));
    pool_add(adj, d_A2_2_adj, std::max(d_prod[idx_prod_A2_2].rank, 1), std::max(d_prod[idx_prod_A2_2].nfunc, 1));
    MemKK::realloc_kokkos(d_A2_2_red_adj, "g3l:A2_2_red_adj", natom,
                          std::max(d_reduce[idx_reduce_A2_2_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A2_2_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A2_2a_adj, "g3l:A2_2a_adj", fold_on[1] ? 1 : natom,
                          std::max(d_fc[idx_fc_A2_2a].n_out, 1),
                          std::max(d_fc[idx_fc_A2_2a].n_funcs_left, 1));
    pool_add(adj, d_A2_3_adj, std::max(d_prod[idx_prod_A2_3].rank, 1), std::max(d_prod[idx_prod_A2_3].nfunc, 1));
    MemKK::realloc_kokkos(d_A2_3_red_adj, "g3l:A2_3_red_adj", natom,
                          std::max(d_reduce[idx_reduce_A2_3_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A2_3_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A2_2b_adj, "g3l:A2_2b_adj", fold_on[1] ? 1 : natom,
                          std::max(d_fc[idx_fc_A2_2b].n_out, 1),
                          std::max(d_fc[idx_fc_A2_2b].n_funcs_left, 1));
    pool_add(adj, d_A2_4_adj, std::max(d_prod[idx_prod_A2_4].rank, 1), std::max(d_prod[idx_prod_A2_4].nfunc, 1));
    MemKK::realloc_kokkos(d_A2_4_red_adj, "g3l:A2_4_red_adj", natom,
                          std::max(d_reduce[idx_reduce_A2_4_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A2_4_red].n_funcs, 1));
    pool_add(adj, d_A2_adj, std::max(d_spbf[idx_spbf_A2].n_rad_max, 1), std::max(d_spbf[idx_spbf_A2].nfunc, 1));
    MemKK::realloc_kokkos(d_eq2_adj, "g3l:eq2_adj", natom,
                          std::max(d_reduce[idx_reduce_eq2].n_out, 1),
                          std::max(d_reduce[idx_reduce_eq2].n_funcs, 1));
    // Chunk-local gather of the reverse-comm-completed message adjoint.
    MemKK::realloc_kokkos(d_eq2_norm_adj_local, "g3l:eq2_norm_adj_local", natom,
                          std::max(I2_n_funcs, 1), std::max(I2_n_out, 1));

    // ---- Layer-1 backward adjoint buffers (mirror the L1 forward
    // tensor shapes). d_A1_adj is the direct fan-out target (raw A1). ----
    pool_add(adj, d_A1_2_adj, std::max(d_prod[0].rank, 1), std::max(d_prod[0].nfunc, 1));
    MemKK::realloc_kokkos(d_A1_2_red_adj, "g3l:A1_2_red_adj", natom,
                          std::max(d_reduce[idx_reduce_A1_2_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A1_2_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A1_2a_adj, "g3l:A1_2a_adj", fold_on[0] ? 1 : natom,
                          std::max(d_fc[idx_fc_A1_2a].n_out, 1),
                          std::max(d_fc[idx_fc_A1_2a].n_funcs_left, 1));
    pool_add(adj, d_A1_3_adj, std::max(d_prod[idx_prod_A1_3].rank, 1), std::max(d_prod[idx_prod_A1_3].nfunc, 1));
    MemKK::realloc_kokkos(d_A1_3_red_adj, "g3l:A1_3_red_adj", natom,
                          std::max(d_reduce[idx_reduce_A1_3_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A1_3_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A1_2b_adj, "g3l:A1_2b_adj", fold_on[0] ? 1 : natom,
                          std::max(d_fc[idx_fc_A1_2b].n_out, 1),
                          std::max(d_fc[idx_fc_A1_2b].n_funcs_left, 1));
    pool_add(adj, d_A1_4_adj, std::max(d_prod[idx_prod_A1_4].rank, 1), std::max(d_prod[idx_prod_A1_4].nfunc, 1));
    MemKK::realloc_kokkos(d_A1_4_red_adj, "g3l:A1_4_red_adj", natom,
                          std::max(d_reduce[idx_reduce_A1_4_red].n_out, 1),
                          std::max(d_reduce[idx_reduce_A1_4_red].n_funcs, 1));
    MemKK::realloc_kokkos(d_A1_adj, "g3l:A1_adj", natom,
                          L1_nradmax, (lmax + 1) * (lmax + 1));
    MemKK::realloc_kokkos(d_eq1_adj, "g3l:eq1_adj", natom,
                          std::max(d_reduce[idx_reduce_eq1].n_out, 1),
                          std::max(d_reduce[idx_reduce_eq1].n_funcs, 1));
    // d_rho1_adj already allocated with the rho{1,2,3}_adj set.
    MemKK::realloc_kokkos(d_eq1_norm_adj_local, "g3l:eq1_norm_adj_local", natom,
                          std::max(I_n_funcs, 1), std::max(I_n_out, 1));
  }

  if (pool_grown) {
    for (auto *p : {&fwd, &adj}) {
      size_t mx = 1;
      for (auto &e : *p) mx = std::max(mx, (size_t) e.d1 * e.d2);
      t_nn_1d &buf = (p == &fwd) ? d_pool_fwd : d_pool_adj;
      buf = t_nn_1d();
      for (auto &e : *p) *e.v = t_nn_3d();
      MemKK::realloc_kokkos(buf, (p == &fwd) ? "g3l:pool_fwd" : "g3l:pool_adj", (size_t) natom * mx);
      for (auto &e : *p) *e.v = t_nn_3d(buf.data(), natom, e.d1, e.d2);
    }
  }

  if (realloc_all || (int) d_radial_basis.extent(0) < natom || (int) d_radial_basis.extent(1) < maxneigh_in) {
    MemKK::realloc_kokkos(d_nearest, "g3l:nearest", natom, maxneigh_in);
    MemKK::realloc_kokkos(d_rnorms, "g3l:rnorms", natom, maxneigh_in);
    MemKK::realloc_kokkos(d_rhats, "g3l:rhats", natom, maxneigh_in);
    MemKK::realloc_kokkos(d_mu_j, "g3l:mu_j", natom, maxneigh_in);
    MemKK::realloc_kokkos(d_radial_basis, "g3l:radial_basis", natom, maxneigh_in, nradbase);
    MemKK::realloc_kokkos(d_env, "g3l:env", natom, maxneigh_in);
    MemKK::realloc_kokkos(d_Y_bond, "g3l:Y_bond", natom, maxneigh_in, (lmax + 1) * (lmax + 1));
    MemKK::realloc_kokkos(d_dY_bond, "g3l:dY_bond", natom, maxneigh_in, (lmax + 1) * (lmax + 1), 3);
    // the (Y, dY) records of the matrix-core SPBF backward (spbf_bwd_mfma_t), as large as Y_bond + dY_bond: here, so
    // that chunksize auto counts them and release_buffers() shrinks them
    bool ydy = false;
    for (int si : {idx_spbf_A2, idx_spbf_A3}) ydy = ydy || (spbf_fast_ok(si) && spbf_use_mfma(si));
    if (ydy) MemKK::realloc_kokkos(d_spbf_ydy, "g3l:spbf_ydy", natom, (lmax + 1) * (lmax + 1), 4 * maxneigh_in);
    else MemKK::drop(d_spbf_ydy);
    // the per-bond envelope derivative and the per-bond force buffer.
    MemKK::realloc_kokkos(d_denv, "g3l:denv", natom, maxneigh_in);
    MemKK::realloc_kokkos(d_f_ij, "g3l:f_ij", natom, maxneigh_in);
#ifdef KOKKOS_ENABLE_CUDA
    // batched cuBLAS radial-MLP scratch (value path). M = natom*maxneigh
    // bonds; hidden widths = max n_out of layers 0/1 across all radial SPBFs.
    {
      int64_t Mrows = (int64_t) natom * maxneigh_in;
      int nin0 = 1, mh0 = 1, mh1 = 1, mout = 1;
      for (int si = 0; si < n_spbf; si++) {
        const auto &mlp = grace_model->spbf[si].mlp;
        const int nl = mlp.n_layers;
        if (nl >= 1) { nin0 = std::max(nin0, mlp.n_in[0]); mh0 = std::max(mh0, mlp.n_out[0]); }
        if (nl >= 2) { mh1 = std::max(mh1, mlp.n_out[1]); }
        if (nl >= 1) { mout = std::max(mout, mlp.n_out[nl - 1]); }
      }
      // the assembled input X/dX is needed only by layers without the layer-0 split
      bool all_split = true;
      for (int si = 0; si < n_spbf; si++) all_split = all_split && mlp_split[si];
      const int64_t Mx = (all_split || spline_on) ? 1 : std::max<int64_t>(Mrows, 1);
      if (spline_on) Mrows = 1;
      MemKK::realloc_kokkos(d_mlp_X,  "g3l:mlp_X",  Mx, nin0);
      MemKK::realloc_kokkos(d_mlp_h0, "g3l:mlp_h0", std::max<int64_t>(Mrows, 1), mh0);
      MemKK::realloc_kokkos(d_mlp_h1, "g3l:mlp_h1", std::max<int64_t>(Mrows, 1), mh1);
      // deriv-path scratch: dX is Chebyshev-only (nradbase cols).
      MemKK::realloc_kokkos(d_mlp_dX,  "g3l:mlp_dX",  Mx, std::max(nradbase, 1));
      MemKK::realloc_kokkos(d_mlp_dh0, "g3l:mlp_dh0", std::max<int64_t>(Mrows, 1), mh0);
      MemKK::realloc_kokkos(d_mlp_dh1, "g3l:mlp_dh1", std::max<int64_t>(Mrows, 1), mh1);
      (void) mout;   // the output GEMMs write d_R1_nl / d_DR1_nl directly
    }
#endif
  }

  // radial scratch d_R1_nl / d_DR1_nl [atom, bond, n, l] (value, dR/dr; reused per layer, bond extent ==
  // d_radial_basis's): one element when layer 1 and the fast SPBF forward / backward of layers 2-3 all read the radial
  // from the tables (fuse_radial; the conditions of compute()). Checked on every grow: a new setup can switch
  // tables <-> MLP.
  {
    const bool need = !(l1_radial_fused() && spbf_rm_ok(idx_spbf_A2, idx_reduce_A2_red) &&
                        spbf_rm_ok(idx_spbf_A3, idx_reduce_A3_red));
    const int ra = need ? (int) d_radial_basis.extent(0) : 1, rb = need ? (int) d_radial_basis.extent(1) : 1;
    if (realloc_all || (int) d_R1_nl.extent(0) != ra || (int) d_R1_nl.extent(1) != rb) {
      MemKK::realloc_kokkos(d_R1_nl, "g3l:R1_nl", ra, rb, L1_nradmax, lmax + 1);
      MemKK::realloc_kokkos(d_DR1_nl, "g3l:DR1_nl", ra, rb, L1_nradmax, lmax + 1);
    }
  }
}

// ======================================================================
// grow_global: allocate the global (nall-sized) inter-layer comm
// arrays for the Layer-2 indicator (eq1_norm) + its host mirror. Mirrors
// GRACE-2L's grow_global; d_grad_I_global (the round-1 reverse-comm buffer) is
// sized here too. Sized to I_n_funcs/I_n_out set in set_model().
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::grow_global(int nall)
{
  if (realloc_all || (int) d_I_global.extent(0) < nall) {
    MemKK::realloc_kokkos(d_I_global, "g3l:I_global", nall, I_n_funcs, I_n_out);
    MemKK::realloc_kokkos(d_grad_I_global, "g3l:grad_I_global", nall, I_n_funcs, I_n_out);
    h_I_global = decltype(h_I_global)();    // host copies: on first use (copy_rows_to_host, host comm only)
    h_grad_I_global = decltype(h_grad_I_global)();
  }
  // Round-2 (L3) indicator buffer: eq2_norm is parity-doubled
  // (I2_n_out=50 vs I_n_out=25), so it needs its own [nall, I2_n_funcs, I2_n_out]
  // array — d_I_global cannot hold it. Separate extent check (distinct dims).
  if (realloc_all || (int) d_eq2_norm_global.extent(0) < nall) {
    MemKK::realloc_kokkos(d_eq2_norm_global, "g3l:eq2_norm_global", nall, I2_n_funcs, I2_n_out);
    h_eq2_norm_global = decltype(h_eq2_norm_global)();
  }
  // L3 message adjoint d_eq2_norm_adj — same [nall, I2_n_funcs, I2_n_out]
  // sizing as d_eq2_norm_global (scattered by atomic_add to owned+ghost neighbors).
  if (realloc_all || (int) d_eq2_norm_adj.extent(0) < nall) {
    MemKK::realloc_kokkos(d_eq2_norm_adj, "g3l:eq2_norm_adj", nall, I2_n_funcs, I2_n_out);
    h_eq2_norm_adj = decltype(h_eq2_norm_adj)();    // host copy for the reverse-comm host path, on first use
  }
  // natom promotions of d_rho1_norm/d_rho2_norm (see .h comment).
  // grow() above always runs before grow_global() in compute(), so
  // d_rho1_norm/d_rho2_norm already carry their correct (natom-independent)
  // trailing dims by this point — reuse them rather than re-deriving from the
  // model metadata indices.
  if (realloc_all || (int) d_rho1_norm_full.extent(0) < nall) {
    MemKK::realloc_kokkos(d_rho1_norm_full, "g3l:rho1_norm_full", nall,
                          (int) d_rho1_norm.extent(1), (int) d_rho1_norm.extent(2));
  }
  if (realloc_all || (int) d_rho2_norm_full.extent(0) < nall) {
    MemKK::realloc_kokkos(d_rho2_norm_full, "g3l:rho2_norm_full", nall,
                          (int) d_rho2_norm.extent(1), (int) d_rho2_norm.extent(2));
  }
  // natom promotions of the rho1/rho2 readout-adjoint SEEDS. Written
  // by TagReadoutBwd at (ii+chunk_offset) across ALL P3 chunks (every owned row
  // fully ASSIGNED -> no pre-zero), then gathered natom->_local per chunk in
  // P5/P4. Trailing dims mirror d_rho{1,2}_norm (== d_rho3_norm_adj's dims, which
  // TagReadoutBwd uses to bound the write loop). Sized to nall (safe superset).
  if (realloc_all || (int) d_rho1_norm_adj.extent(0) < nall) {
    MemKK::realloc_kokkos(d_rho1_norm_adj, "g3l:rho1_norm_adj", nall,
                          (int) d_rho1_norm.extent(1), (int) d_rho1_norm.extent(2));
  }
  if (realloc_all || (int) d_rho2_norm_adj.extent(0) < nall) {
    MemKK::realloc_kokkos(d_rho2_norm_adj, "g3l:rho2_norm_adj", nall,
                          (int) d_rho2_norm.extent(1), (int) d_rho2_norm.extent(2));
  }
}

// ======================================================================
// compute_Y_bond_chunk: fill d_Y_bond from d_rhats for the current chunk.
// The same code as in the GRACE-2L engine: the alm/blm/cl/dl/idx_sph recurrence
// tables, the same real-SH sign and ordering convention. Must be (re)called whenever d_rhats for the current chunk is
// refreshed, before any kernel reads d_Y_bond.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_Y_bond_chunk()
{
  auto Y_bond = d_Y_bond;
  auto rhats = d_rhats;
  auto nc_v = d_ncount;
  auto idx_s = d_idx_sph;
  auto alm_v = alm; auto blm_v = blm; auto cl_v = cl; auto dl_v = dl;
  int lmax_v = lmax;
  GeomScalar sq2_v = sq2;
  int cs = chunk_size;
  // row length of the allocation (>= maxneigh: the views only grow)
  const int mn = (int) d_Y_bond.extent(1);

  // Rows of d_Y_bond / d_dY_bond are contiguous in q = ii mn + jj: a team computes TY consecutive rows into
  // shared memory and writes the block out coalesced (rows past the bond count as zeros).
  constexpr int NYB = 25, NDB = 75;
  const int nq = cs * mn, nlm = (int) d_Y_bond.extent(2);
  // team sizes: 128 / 64 rows on GPUs; host execution spaces cap a team at their thread count
  int TY = 128, TD = 64;
#if defined(KOKKOS_ENABLE_CUDA)
  if (!std::is_same<typename DeviceType::execution_space, Kokkos::Cuda>::value)
#elif defined(KOKKOS_ENABLE_HIP)
  if (!is_hip_v<DeviceType>)
#endif
  {
    const int cap = std::max(1, (int) typename DeviceType::execution_space().concurrency());
    TY = std::min(TY, cap); TD = std::min(TD, cap);
  }
  using team_policy = Kokkos::TeamPolicy<DeviceType>;
  using member = typename team_policy::member_type;
  team_policy polY((nq + TY - 1) / TY, TY);
  polY = polY.set_scratch_size(0, Kokkos::PerTeam((size_t) TY * NYB * sizeof(GeomScalar)));
  GeomScalar *Yd = Y_bond.data();
  Kokkos::parallel_for("ComputeYBond", polY, KOKKOS_LAMBDA(const member &t) {
    GeomScalar *sh = (GeomScalar *) t.team_shmem().get_shmem((size_t) TY * NYB * sizeof(GeomScalar));
    const int tid = t.team_rank(), q0 = t.league_rank() * TY, q = q0 + tid;
    GeomScalar *Yrow = sh + tid * NYB;
    for (int lm = 0; lm < NYB; lm++) Yrow[lm] = GeomScalar(0.0);
    const int ii = q / mn, jj = q - ii * mn;
    if (q < nq && jj < nc_v(ii)) {
      const GeomScalar rx = rhats(ii, jj, 0);
      const GeomScalar ry = rhats(ii, jj, 1);
      const GeomScalar rz = rhats(ii, jj, 2);

      GeomScalar plm_arr[15];
      plm_arr[0] = GeomScalar(1.0);
      if (lmax_v > 0) {
        plm_arr[1] = GeomScalar(1.7320508075688772935) * rz;
        plm_arr[2] = GeomScalar(-1.2247448713915890491);
#pragma unroll
        for (int ll = 2; ll <= 4; ll++) {   // bound 4 (plm_arr[15]) so that the arrays stay in registers
          if (ll > lmax_v) break;
          for (int mm = 0; mm < ll - 1; mm++) {
            const int idx = ll*(ll+1)/2 + mm;
            const int i1 = (ll-1)*ll/2 + mm, i2 = (ll-2)*(ll-1)/2 + mm;
            const int ai = idx_s(ll*(ll+1) + mm);
            plm_arr[idx] = alm_v(ai) * (rz * plm_arr[i1] + blm_v(ai) * plm_arr[i2]);
          }
          { int idx = ll*(ll+1)/2+ll-1;
            plm_arr[idx] = dl_v(ll) * plm_arr[(ll-1)*ll/2+ll-1] * rz; }
          { int idx = ll*(ll+1)/2+ll;
            plm_arr[idx] = cl_v(ll) * plm_arr[(ll-1)*ll/2+ll-1]; }
        }
      }

      const GeomScalar phase_re = rx, phase_im = ry;
#pragma unroll
      for (int l = 0; l <= 4; l++) {
        if (l > lmax_v) break;
        Yrow[l*(l+1)] = plm_arr[l*(l+1)/2];
        GeomScalar pm_re = phase_re, pm_im = phase_im;
        for (int m = 1; m <= l; m++) {
          const int fac = (m % 2 == 0) ? 1 : -1;
          const GeomScalar pv = plm_arr[l*(l+1)/2 + m];
          Yrow[l*(l+1)+m] = sq2_v * fac * pm_re * pv;
          Yrow[l*(l+1)-m] = sq2_v * fac * pm_im * pv;
          const GeomScalar t_re = pm_re * phase_re - pm_im * phase_im;
          const GeomScalar t_im = pm_re * phase_im + pm_im * phase_re;
          pm_re = t_re; pm_im = t_im;
        }
      }
    }
    t.team_barrier();
    const long base = (long) q0 * nlm;
    const int nv = ((nq - q0 < TY) ? nq - q0 : TY) * nlm;
    for (int v = tid; v < nv; v += TY) {
      const int r = v / nlm;
      Yd[base + v] = sh[r * NYB + (v - r * nlm)];
    }
  });

  // DY_lm (as in bond_geom_force) for the backward kernels
  auto dYb = d_dY_bond;
  const GeomScalar sq3_v = sq3, sq3o2_v = sq3o2;
  team_policy polD((nq + TD - 1) / TD, TD);
  polD = polD.set_scratch_size(0, Kokkos::PerTeam((size_t) TD * NDB * sizeof(GeomScalar)));
  GeomScalar *dYd = dYb.data();
  Kokkos::parallel_for("ComputeDYBond", polD, KOKKOS_LAMBDA(const member &t) {
    GeomScalar *sh = (GeomScalar *) t.team_shmem().get_shmem((size_t) TD * NDB * sizeof(GeomScalar));
    const int tid = t.team_rank(), q0 = t.league_rank() * TD, q = q0 + tid;
    GeomScalar *D = sh + tid * NDB;   // D[lm * 3 + k] = dYb(ii, jj, lm, k)
    for (int v = 0; v < NDB; v++) D[v] = GeomScalar(0.0);
    const int ii = q / mn, jj = q - ii * mn;
    if (q < nq && jj < nc_v(ii)) {
      const GeomScalar rx = rhats(ii, jj, 0), ry = rhats(ii, jj, 1), rz = rhats(ii, jj, 2);
      GeomScalar plm[15], dplm[15];
      plm[0] = GeomScalar(1.0); dplm[0] = GeomScalar(0.0);
      if (lmax_v > 0) {
        plm[1] = sq3_v * rz; dplm[1] = sq3_v;
        plm[2] = -sq3o2_v;   dplm[2] = GeomScalar(0.0);
#pragma unroll
        for (int l = 2; l <= 4; l++) {
          if (l > lmax_v) break;
          for (int m = 0; m < l - 1; m++) {
            const int idx = l*(l+1)/2 + m;
            const int i1 = (l-1)*l/2 + m, i2 = (l-2)*(l-1)/2 + m;
            const int ai = idx_s(l*(l+1) + m);
            const GeomScalar al = alm_v(ai), bl = blm_v(ai);
            plm[idx]  = al * (rz * plm[i1] + bl * plm[i2]);
            dplm[idx] = al * (plm[i1] + rz * dplm[i1] + bl * dplm[i2]);
          }
          { const int idx = l*(l+1)/2+l-1, prev = (l-1)*l/2+l-1;
            const GeomScalar t = dl_v(l) * plm[prev];
            plm[idx] = t * rz; dplm[idx] = t; }
          { const int idx = l*(l+1)/2+l, prev = (l-1)*l/2+l-1;
            plm[idx] = cl_v(l) * plm[prev]; dplm[idx] = GeomScalar(0.0); }
        }
      }
#pragma unroll
      for (int l = 0; l <= 4; l++) {
        if (l > lmax_v) break;
        {
          const GeomScalar dp = dplm[l*(l+1)/2];
          const GeomScalar rdy = dp * rz;
          const int A0 = l*(l+1);
          D[3 * A0] = -rdy * rx; D[3 * A0 + 1] = -rdy * ry; D[3 * A0 + 2] = dp - rdy * rz;
        }
        GeomScalar pm_re = rx, pm_im = ry;
        for (int m = 1; m <= l; m++) {
          const int fac = (m % 2 == 0) ? 1 : -1;
          const GeomScalar pv = plm[l*(l+1)/2+m], dpv = dplm[l*(l+1)/2+m];
          GeomScalar dyx_re, dyx_im, dyy_re, dyy_im;
          const GeomScalar dyz_re = dpv * pm_re, dyz_im = dpv * pm_im;
          if (m == 1) {
            dyx_re = pv; dyx_im = GeomScalar(0.0); dyy_re = GeomScalar(0.0); dyy_im = pv;
          } else {
            const GeomScalar s2 = rx*rx + ry*ry;
            if (s2 > GeomScalar(1e-14)) {
              const GeomScalar is2 = GeomScalar(1.0)/s2;
              const GeomScalar p1r = (pm_re*rx + pm_im*ry)*is2;
              const GeomScalar p1i = (pm_im*rx - pm_re*ry)*is2;
              const GeomScalar mp = GeomScalar(m) * pv;
              dyx_re = mp*p1r; dyx_im = mp*p1i;
              dyy_re = -dyx_im; dyy_im = dyx_re;
            } else { dyx_re = dyx_im = dyy_re = dyy_im = GeomScalar(0.0); }
          }
          const GeomScalar rdy_re = rx*dyx_re + ry*dyy_re + rz*dyz_re;
          const GeomScalar rdy_im = rx*dyx_im + ry*dyy_im + rz*dyz_im;
          const GeomScalar f2 = sq2_v * fac;
          const int Ap = l*(l+1)+m, An = l*(l+1)-m;
          D[3 * Ap] = f2 * (dyx_re - rdy_re*rx); D[3 * Ap + 1] = f2 * (dyy_re - rdy_re*ry);
          D[3 * Ap + 2] = f2 * (dyz_re - rdy_re*rz);
          D[3 * An] = f2 * (dyx_im - rdy_im*rx); D[3 * An + 1] = f2 * (dyy_im - rdy_im*ry);
          D[3 * An + 2] = f2 * (dyz_im - rdy_im*rz);
          const GeomScalar t_re = pm_re*rx - pm_im*ry;
          const GeomScalar t_im = pm_re*ry + pm_im*rx;
          pm_re = t_re; pm_im = t_im;
        }
      }
    }
    t.team_barrier();
    const int w = 3 * nlm;
    const long base = (long) q0 * w;
    const int nv = ((nq - q0 < TD) ? nq - q0 : TD) * w;
    for (int v = tid; v < nv; v += TD) {
      const int r = v / w;
      dYd[base + v] = sh[r * NDB + (v - r * w)];
    }
  });
}

// ======================================================================
// Kernel: ComputeNeigh — the neighbour list of each atom of the chunk filtered by the pair cutoffs into
// per-atom bond arrays. d_rhats stores the unit vector (x_j - x_i)/r, the model's bond_vector = r_j - r_i.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeNeigh,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>::member_type& team) const
{
  const int ii = team.league_rank();
  const int i = d_ilist[ii + chunk_offset];
  const int itype = type(i);
  using D = std::common_type_t<typename Traits::kk_float, GeomScalar>;    // positions or geometry, the wider
  const D xi = x(i,0), yi = x(i,1), zi = x(i,2);
  // the neighbours inside the pair cutoff, packed in list order by one team scan: both passes of the scan test the
  // cutoff, the first one counting, the second one writing each bond at its prefix slot. With two-group coupling a
  // cross-group bond whose shifted r' reaches rc is dropped as well (the same test in both passes).
  int nin = 0;
  Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, d_numneigh[i]),
      [&] (const int k, int& slot, const bool write) {
    const int j = d_neighbors(i, k) & Traits::neighmask;
    const BondScalar dx = BondScalar(xi - D(x(j,0))), dy = BondScalar(yi - D(x(j,1))), dz = BondScalar(zi - D(x(j,2)));
    const BondScalar rsq = dx*dx + dy*dy + dz*dz;
    if (!(rsq < d_cutsq(itype, type(j)))) return;
    // |r| regularised as the TF model's BondLength: sqrt(|r|^2 + 1e-10)
    const BondScalar r = Kokkos::sqrt(rsq + BondScalar(1e-10));
    double rp = r, drp = 1.0, dlp = 0.0;
    if (couple_eq && coupling_cross(d_mask(i), d_mask(j), couple_bit1, couple_bit2)) {
      const double rc = d_bond_cutoff(d_map(itype), d_map(type(j)));
      coupling_shift(couple_eq, r, rc, couple_lambda, rp, drp, dlp);
      if (!(rp < rc)) return;    // r' >= rc: dropped
    }
    if (write) {
      const BondScalar rinv = BondScalar(1.0) / r;
      d_mu_j(ii, slot) = d_map(type(j));
      if (couple_eq) {    // the model sees r' (d_rnorms); true r, dr'/dr, dr'/dlambda for scatter_forces_chunk
        d_cpl(ii, slot, 0) = r; d_cpl(ii, slot, 1) = drp; d_cpl(ii, slot, 2) = dlp;
        d_rnorms(ii, slot) = rp;
      } else {
        d_rnorms(ii, slot) = r;
      }
      d_rhats(ii, slot, 0) = -dx * rinv;    // unit vector from i to j (the model's bond_vector = r_j - r_i)
      d_rhats(ii, slot, 1) = -dy * rinv;
      d_rhats(ii, slot, 2) = -dz * rinv;
      d_nearest(ii, slot) = j;
    }
    slot++;
  }, nin);
  Kokkos::single(Kokkos::PerTeam(team), [&] () {
    d_ncount(ii) = nin;
    d_mu_i(ii) = d_map(itype);
  });
}

// ======================================================================
// Kernel: ComputeRadialBasis — Chebyshev (kind=1) basis T_1..T_nradbase,
// RAW (no envelope folded in) + a separate polynomial cutoff envelope d_env.
// Adapted from Grace2LKK: 2L multiplies the
// envelope into d_radial_basis and also tracks its r-derivative; the 3L
// reference (chebyshev_basis / envelope) keeps them SEPARATE: the
// envelope and its r-derivative go to d_env / d_denv.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeRadialBasis,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;
  const int jj = slot.neigh;
  if (jj >= d_ncount(ii)) return;

  const BondScalar r = d_rnorms(ii, jj);
  const BondScalar rcut_ij = d_bond_cutoff(d_mu_i(ii), d_mu_j(ii, jj));
  const BondScalar x_norm = r / rcut_ij;

  const int p = radial_basis_p;
  BondScalar xp = BondScalar(1.0);
  for (int ip = 0; ip < p; ip++) xp *= x_norm;
  const BondScalar xp1 = xp * x_norm, xp2 = xp1 * x_norm;
  const BondScalar pp1 = BondScalar(p) * (p + 1);
  const BondScalar pp2 = BondScalar(p) * (p + 2);
  const BondScalar p1p2 = BondScalar(p + 1) * (p + 2);
  const BondScalar fcut = BondScalar(1.0) - BondScalar(0.5) * p1p2 * xp + pp2 * xp1 - BondScalar(0.5) * pp1 * xp2;
  d_env(ii, jj) = fcut;
  // denv/dr (layer-independent), here once per bond in fp64 for the radial kernels of every layer
  {
    const BondScalar xpm1 = xp / x_norm;
    const BondScalar denv_dx = BondScalar(-0.5) * p1p2 * BondScalar(p) * xpm1 + pp2 * BondScalar(p + 1) * xp
                             - BondScalar(0.5) * pp1 * BondScalar(p + 2) * xp1;
    d_denv(ii, jj) = denv_dx / rcut_ij;
  }

  const BondScalar x_cheb = BondScalar(2.0) * x_norm - BondScalar(1.0);

  BondScalar T_prev = BondScalar(1.0), T_curr = x_cheb;
  d_radial_basis(ii, jj, 0) = T_curr;

  for (int k = 1; k < nradbase; k++) {
    const BondScalar T_next = BondScalar(2.0) * x_cheb * T_curr - T_prev;
    T_prev = T_curr;
    T_curr = T_next;
    d_radial_basis(ii, jj, k) = T_curr;
  }
}

// ======================================================================
// Kernel: ComputeA1 — Layer-1 scalar-indicator SPBF assembly,
// per (central-atom ii, radial-channel n). Reproduces
// the reference spbf_scalar EXACTLY:
//   a_nl[lm] = R1_nl[n, l_tile[lm]] * Y_bond[lm] * z_tr[mu_j][n] * env
//   A1[ii,n,lm] = inv_avg_n_neigh * Sum_{j in neigh(ii)} a_nl[lm]
// z_tr is the PRECOMPUTED per-element species table (norm folded once at
// copy_weights_to_device); env/Y_bond (GeomScalar) are cast down to
// NNScalar on read. One thread owns the whole (ii,n,:) row -> no atomics,
// zero-init folded into the same kernel. Team-policy (ii,n) decode mirrors
// TagComputeRadialBasis's (ii,jj) decode, dividing by L1_nradmax instead of
// maxneigh.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeA1,
    const int idx) const
{
  // channel n fastest: a warp reads one bond's contiguous row-major R(ii, jj, :, :)
  const int ii = idx / L1_nradmax;
  const int n = idx - ii * L1_nradmax;

  const DeviceSPBF &sp = d_spbf[0];   // A1
  const int nlm = (lmax + 1) * (lmax + 1);
  NNScalar acc[25];
  for (int lm = 0; lm < nlm; lm++) acc[lm] = NNScalar(0.0);

  const int nct = d_ncount(ii);
  for (int jj = 0; jj < nct; jj++) {
    const int mu_j = d_mu_j(ii, jj);
    const NNScalar z_n = sp.z_tr(mu_j, n);
    const NNScalar env = (NNScalar) d_env(ii, jj);
    for (int lm = 0; lm < nlm; lm++) {
      const int l = sp.l_tile(lm);
      NNScalar a = d_R1_nl(ii, jj, n, l) * (NNScalar) d_Y_bond(ii, jj, lm);
      a *= z_n;
      a *= env;
      acc[lm] += a;
    }
  }
  const NNScalar inv_avg = (NNScalar) sp.inv_avg_n_neigh;
  for (int lm = 0; lm < nlm; lm++)
    d_A1(ii, n, lm) = acc[lm] * inv_avg;
}

// ======================================================================
// Kernel: ComputeMLPEnergy — 3-density readout -> per-atom energy.
// Reference: readout() (out1_origins = [rho1_norm, rho2_norm,
// rho3_norm]). For each origin, s = origin[:, :, 0] (the l=0 slice, LAST axis
// index 0); channel 0 is a weightless linear skip, channels 1..31 feed the
// MLP(31->64 silu ->1). ae_nn = e_mlp + lin; E_atom = ae_nn*output_scale +
// shift[species]. rho3_norm is a chunk-scratch buffer indexed by the SAME
// chunk-local ii used by the L3 rho kernels (written+read within this SAME P3
// chunk, so that is safe). rho1_norm/rho2_norm are instead read from the natom
// buffers d_rho{1,2}_norm_full at ii+chunk_offset — the chunk-local
// d_rho1_norm/d_rho2_norm were written back in P1/P2 and, under multi-chunk,
// hold only the LAST chunk by the time P3 runs, so reading them chunk-local
// here would silently pick up the wrong atoms' densities for every chunk but
// the last. The backward (TagReadoutBwd) reads the same natom buffers and writes
// the rho1/rho2 adjoints to d_rho{1,2}_norm_adj at ii+chunk_offset.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeMLPEnergy,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeMLPEnergy>::member_type& team) const
{
  const int ii = team.league_rank();
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));

  // Linear (weightless) skip: channel 0, l=0 slice, summed over the 3 origins.
  const NNScalar linear_term = d_rho1_norm_full(ii + chunk_offset, 0, 0) +
                                d_rho2_norm_full(ii + chunk_offset, 0, 0) +
                                d_rho3_norm(ii, 0, 0);

  NNScalar ebuf_a[GRACE3L_MAX_ENERGY_MLP_DIM], ebuf_b[GRACE3L_MAX_ENERGY_MLP_DIM];
  NNScalar* h_cur = ebuf_a;
  NNScalar* h_nxt = ebuf_b;

  // MLP input: channels 1..31, l=0 slice, summed over the 3 origins.
  const int nin0 = d_energy_dims(0);
  for (int c = 0; c < nin0; c++)
    h_cur[c] = d_rho1_norm_full(ii + chunk_offset, c + 1, 0) +
               d_rho2_norm_full(ii + chunk_offset, c + 1, 0) +
               d_rho3_norm(ii, c + 1, 0);

  // Hidden layers with activation (3L uses silu: energy_activation == 0).
  for (int layer = 0; layer < energy_n_layers - 1; layer++) {
    const int nin = d_energy_dims(layer);
    const int nout = d_energy_dims(layer + 1);
    const NNScalar norm = d_energy_norms(layer);
    for (int j = 0; j < nout; j++) {
      NNScalar sum = 0.0;
      for (int k = 0; k < nin; k++)
        sum += d_energy_W(layer, k, j) * h_cur[k];
      sum *= norm;
      if (energy_activation == 1)
        h_nxt[j] = tanh_act(sum);
      else
        h_nxt[j] = silu(sum);
    }
    NNScalar* tmp = h_cur; h_cur = h_nxt; h_nxt = tmp;
  }

  // Output layer (no activation).
  const int last = energy_n_layers - 1;
  const int nin_last = d_energy_dims(last);
  const int nout_last = d_energy_dims(last + 1);
  NNScalar e = 0.0;
  for (int o = 0; o < nout_last; o++) {
    NNScalar sum = 0.0;
    for (int k = 0; k < nin_last; k++)
      sum += d_energy_W(last, k, o) * h_cur[k];
    e += sum * d_energy_norms(last);
  }

  // E_atom = (MLP_out + linear_term) * output_scale + shift[species]; the MLP
  // math runs in NNScalar, the readout in fp64 (KKA) as on the CPU: an fp32 sum
  // rounds E_atom to ~1e-6 eV.
  d_e_atom(ii) = ((KKA) e + (KKA) linear_term) * (KKA) output_scale + (KKA) d_shifts(mu_i);
}

// ======================================================================
// UQ basis-RP kernels (forward only). The GMM feature is the projection of the
// invariant (l=0) products entering the three per-layer scalar reduces
// rho1|rho2|rho3, read in the EXACT (source, n, collect) order TagReduceN
// iterates them (n outer, connection inner), so R's row cursor advances in the
// concatenation order [rho1, rho2, rho3]. Because feature_transform is identity
// and normalize is a scalar divide, the projection proj = B·R is linear and is
// stream-accumulated across the three layer phases into d_uq_z; the final kernel
// normalizes by ||B|| = sqrt(Σ n2) and appends the [full, rho1, rho2, rho3]
// density channels before the GMM. FP64 throughout for numerical safety.
// ======================================================================

// Stream one rho block's sources into the RAW projection ACC and its block norm² N2.
// For each gathered product b_inv = A(ii, n, ci(c)) (n outer, connection inner —
// TagReduceN order), accumulate ACC[d] += b_inv*R[B,d] over the RP columns, add
// b_inv² into N2, and advance the R-row cursor B. Expands against each kernel's
// local RP/ii, the d_uq_rp_matrix member, and the reduce metadata.
// Single projection column: for each gathered product _bv = A(ii,_n,_ci(_c)) (n
// outer, connection inner — TagReduceN order), accumulate ACC += _bv*R[B,d] for this
// lane's column d, add _bv² into the block norm² N2, and advance the R-row cursor B.
#define GRACE3L_UQ_WALK_BLOCK(REDUCE, VIEWS, ROLE, ACC, N2, B)                      \
    { const DeviceReduce &_r = (REDUCE);                                           \
      for (int _j = 0; _j < _r.n_instr; _j++) {                                    \
        const t_nn_3d &_A = (VIEWS)[(ROLE)[_j]];                                   \
        const int _nin = _r.n_in[_j];                                             \
        const int _nc = _r.n_conn[_j];                                            \
        const auto &_ci = _r.collect_ind[_j];                                     \
        for (int _n = 0; _n < _nin; _n++)                                         \
          for (int _c = 0; _c < _nc; _c++) {                                      \
            const double _bv = (double)(_A)(ii, _n, _ci(_c));                      \
            (N2)  += _bv * _bv;                                                    \
            (ACC) += _bv * d_uq_rp_matrix((B), d);                                 \
            (B)++;                                                                 \
          } } }

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeUQ_rho1,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeUQ_rho1>::member_type& team) const
{
  const int ii = team.league_rank();
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];
  const int RP = uq_rp_dim;
  const int TS = team.team_size();
  const int lane = team.team_rank();
  // Each lane owns column(s) d, walks the rho1 block once per column with a register
  // accumulator (no per-thread array => no local-memory spill). rho1 occupies R-rows
  // [0, d_basis_rho1). ||B^(rho1)||² recomputed per column (same value); lane 0 stores.
  t_nn_3d views[4] = {d_A1, d_A1_2_red, d_A1_3_red, d_A1_4_red};
  double n2_rho1 = 0.0;
  for (int d = lane; d < RP; d += TS) {
    double z = 0.0; n2_rho1 = 0.0; int b = 0;
    GRACE3L_UQ_WALK_BLOCK(d_reduce[idx_reduce_rho1], views, rho1_input_role, z, n2_rho1, b);
    d_uq_z(i, d) = z;
  }
  if (lane == 0) d_uq_n2_rho1(i) = n2_rho1;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeUQ_rho2,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeUQ_rho2>::member_type& team) const
{
  const int ii = team.league_rank();
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];
  const int RP = uq_rp_dim;
  const int TS = team.team_size();
  const int lane = team.team_rank();
  // rho2 block occupies R-rows [d_basis_rho1, d_basis_rho1 + d_basis_rho2); each lane
  // adds its column onto the carried rho1 projection in d_uq_z.
  t_nn_3d views[4] = {d_A2_red, d_A2_2_red, d_A2_3_red, d_A2_4_red};
  double n2_rho2 = 0.0;
  for (int d = lane; d < RP; d += TS) {
    double z = d_uq_z(i, d); n2_rho2 = 0.0; int b = uq_d_basis_rho1;
    GRACE3L_UQ_WALK_BLOCK(d_reduce[idx_reduce_rho2], views, rho2_input_role, z, n2_rho2, b);
    d_uq_z(i, d) = z;
  }
  if (lane == 0) d_uq_n2_rho2(i) = n2_rho2;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeUQ,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeUQ>::member_type& team) const
{
  const int ii = team.league_rank();
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  const int D  = uq_D;         // full feature dim (rp_dim + n_density)
  const int RP = uq_rp_dim;    // projection width (R cols)
  const int TS = team.team_size();
  const int lane = team.team_rank();

  // Team-shared scratch: normalized feature f[D], Mahalanobis delta[D], and a pair
  // of scalars [n2_rho3, nrm_full] carried from the projection loop to the density code.
  double* f     = (double*) team.team_shmem().get_shmem(D * sizeof(double));
  double* delta = (double*) team.team_shmem().get_shmem(D * sizeof(double));
  double* sc    = (double*) team.team_shmem().get_shmem(2 * sizeof(double));

  const double n2_rho1 = d_uq_n2_rho1(i);
  const double n2_rho2 = d_uq_n2_rho2(i);

  // Combined projection proj = B·R: rho1 (P1) + rho2 (P2) from d_uq_z + rho3 added
  // here. rho3 block occupies R-rows [d_basis_rho1 + d_basis_rho2, d_basis).
  t_nn_3d views[4] = {d_A3_red, d_A3_2_red, d_A3_3_red, d_A3_4_red};
  for (int d = lane; d < RP; d += TS) {
    double acc = d_uq_z(i, d);
    double n2_rho3 = 0.0;
    int b = uq_d_basis_rho1 + uq_d_basis_rho2;
    GRACE3L_UQ_WALK_BLOCK(d_reduce[idx_reduce_rho3], views, rho3_input_role, acc, n2_rho3, b);
    const double nrm_full = Kokkos::sqrt(n2_rho1 + n2_rho2 + n2_rho3);   // ||B||
    if (uq_normalize) acc /= (nrm_full + 1.0e-12);
    f[d] = acc;
    if (d == 0) { sc[0] = n2_rho3; sc[1] = nrm_full; }
  }
  #undef GRACE3L_UQ_WALK_BLOCK
  team.team_barrier();

  // Density channels [full, rho1, rho2, rho3] — sorted-by-reduce-name order matching
  // R's row layout. Cheap; lane 0 writes them.
  if (lane == 0) {
    if (uq_n_density > 0) f[RP + 0] = Kokkos::log(sc[1]                + 1.0e-12) * uq_density_scale;
    if (uq_n_density > 1) f[RP + 1] = Kokkos::log(Kokkos::sqrt(n2_rho1) + 1.0e-12) * uq_density_scale;
    if (uq_n_density > 2) f[RP + 2] = Kokkos::log(Kokkos::sqrt(n2_rho2) + 1.0e-12) * uq_density_scale;
    if (uq_n_density > 3) f[RP + 3] = Kokkos::log(Kokkos::sqrt(sc[0])   + 1.0e-12) * uq_density_scale;
  }
  team.team_barrier();

  // Cluster assignment: nearest centroid (squared Euclidean over D). Each reduce
  // broadcasts d2 to all lanes, so kstar is identical across the team.
  const int ncl = d_uq_n_clusters(mu_i);
  int kstar = 0;
  double best = 1.0e300;
  for (int k = 0; k < ncl; k++) {
    double d2 = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, D),
      [&](const int p, double& s) {
        const double diff = f[p] - d_uq_centroids(mu_i, k, p);
        s += diff * diff;
      }, d2);
    if (d2 < best) { best = d2; kstar = k; }
  }

  // Mahalanobis distance to assigned cluster: sig2 = deltaᵀ · Σ⁻¹ · delta.
  for (int p = lane; p < D; p += TS) delta[p] = f[p] - d_uq_centroids(mu_i, kstar, p);
  team.team_barrier();
  double sig2 = 0.0;
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, D * D),
    [&](const int idx, double& s) {
      const int p = idx / D, q = idx % D;
      s += delta[p] * d_uq_inv_cov(mu_i, kstar, p, q) * delta[q];
    }, sig2);

  Kokkos::single(Kokkos::PerTeam(team), [&]() {
    const double sigma = Kokkos::sqrt(sig2 + 1.0e-8);
    double t = d_uq_interp_thresholds(mu_i, kstar);
    if (t < 1.0e-10) t = 1.0e-10;
    d_sigma(i) = sigma;
    d_gamma(i) = sigma / t;
    d_gmm_cluster(i) = (double) kstar;
  });
}

// ---- multi-chunk forward store ----
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
typename Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::t_nn_3d *
Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::st_ptr(int i)
{
  // with fc_fold the L1 / L2 backward re-projects from A*_2_red (A*_2a / A*_2b are never formed)
  t_nn_3d *v[12] = {&d_A1, fold_on[0] ? &d_A1_2_red : &d_A1_2a, fold_on[0] ? &st_none : &d_A1_2b,
                    &d_eq1, &d_rho1, &d_rho1_norm,
                    &d_A2_red, fold_on[1] ? &d_A2_2_red : &d_A2_2a, fold_on[1] ? &st_none : &d_A2_2b,
                    &d_eq2, &d_rho2, &d_rho2_norm};
  return v[i];
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::setup_fwd_store()
{
  fwd_stored = 0;
  for (int i = 0; i < 12; i++) st_base[i] = *st_ptr(i);
  const int cl = std::min(chunk_lim, inum);
  const int nch = (cl > 0) ? (inum + cl - 1) / cl : 0;
  if (store_fwd == 0 || nch <= 1) return;
  if (store_fwd < 0 && chunksize < 0 && !store_auto_ok) {    // chunksize auto decided without the store
    for (int i = 0; i < 12; i++) st_fwd[i].clear();
    return;
  }
  size_t per = 0;
  for (int i = 0; i < 12; i++) per += st_base[i].size();
  const double bytes = (double) per * nch * sizeof(NNScalar);
  bool fresh = (int) st_fwd[0].size() != nch;
  for (int i = 0; i < 12 && !fresh; i++)
    for (int c = 0; c < nch && !fresh; c++)
      fresh = st_fwd[i][c].extent(0) != st_base[i].extent(0) || st_fwd[i][c].extent(1) != st_base[i].extent(1) ||
              st_fwd[i][c].extent(2) != st_base[i].extent(2);
  if (fresh) {
    for (int i = 0; i < 12; i++) st_fwd[i].clear();
    if (store_fwd < 0) {
      double free_b = 0.0;
      size_t fr = 0, tot = 0;
      if (device_mem_info(fr, tot)) free_b = (double) fr / std::max(1, ranks_per_device);
      if (bytes > 0.7 * free_b) {
          log.info(fmt("[GRACE-3L/KK] {} chunks: L1/L2 forward recomputed in the backward (store would need "
                         "{:.0f} MB, free {:.0f} MB)\n", nch, bytes / 1048576.0, free_b / 1048576.0));
        return;
      }
    }
    for (int i = 0; i < 12; i++)
      for (int c = 0; c < nch; c++)
        st_fwd[i].push_back(t_nn_3d("g3l:st_fwd", st_base[i].extent(0), st_base[i].extent(1), st_base[i].extent(2)));
      log.info(fmt("[GRACE-3L/KK] {} chunks: L1/L2 forward state stored ({:.0f} MB)\n", nch, bytes / 1048576.0));
  }
  fwd_stored = 1;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::select_fwd_store(int layer, int c)
{
  if (!fwd_stored) return;
  for (int i = (layer == 1 ? 0 : 6); i < (layer == 1 ? 6 : 12); i++) *st_ptr(i) = st_fwd[i][c];
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::restore_fwd_store()
{
  for (int i = 0; i < 12; i++) if (st_base[i].data()) *st_ptr(i) = st_base[i];
}

// Geometry of the current chunk only (neighbours, Y / dY, Chebyshev basis + envelope): the first stages of
// l1_forward_chunk / l2_forward_chunk, used by the backward when the forward state is stored.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::geometry_chunk()
{
  int team_size = 1, vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1) team_size = 32;
  {
    team_size = fit_team_size<DeviceType, TagComputeNeigh>(*this, chunk_size, team_size, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>(chunk_size, team_size, vector_length);
    Kokkos::parallel_for("ComputeNeigh", policy, *this);
  }
  compute_Y_bond_chunk();
  {
    int ts = team_size;
    ts = fit_team_size<DeviceType, TagComputeRadialBasis>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>(
        ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    Kokkos::parallel_for("ComputeRadialBasis", policy, *this);
  }
}

// ======================================================================
// Message rows of the layered exchange (LAMMPS pack/unpack callbacks of the pair style). Layer 0: eq1_norm
// (d_I_global, I_n_funcs x I_n_out per atom) and its adjoint d_grad_I_global; layer 1: eq2_norm
// (d_eq2_norm_global, I2_n_funcs x I2_n_out, parity-doubled) and its adjoint d_eq2_norm_adj. comm_stage = layer + 1
// selects the buffers in the pack / unpack kernels (captured by value into the tag functors).
// ======================================================================

// host buffers (legacy pair comm, MPI without GPU support)
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::copy_rows_to_host(int layer, bool reverse)
{
  // the host copies exist only for the host comm path (LAMMPS legacy pair comm): made here on first use
  using M = grace::kk::MemKK;
  M::ensure_mirror(h_I_global, d_I_global);
  M::ensure_mirror(h_grad_I_global, d_grad_I_global);
  M::ensure_mirror(h_eq2_norm_global, d_eq2_norm_global);
  M::ensure_mirror(h_eq2_norm_adj, d_eq2_norm_adj);
  if (layer == 1) {
    if (reverse) Kokkos::deep_copy(h_eq2_norm_adj.base(), d_eq2_norm_adj.base());
    else Kokkos::deep_copy(h_eq2_norm_global.base(), d_eq2_norm_global.base());
  } else {
    if (reverse) Kokkos::deep_copy(h_grad_I_global.base(), d_grad_I_global.base());
    else Kokkos::deep_copy(h_I_global.base(), d_I_global.base());
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::copy_rows_from_host(int layer, bool reverse)
{
  if (layer == 1) {
    if (reverse) Kokkos::deep_copy(d_eq2_norm_adj.base(), h_eq2_norm_adj.base());
    else Kokkos::deep_copy(d_eq2_norm_global.base(), h_eq2_norm_global.base());
  } else {
    if (reverse) Kokkos::deep_copy(d_grad_I_global.base(), h_grad_I_global.base());
    else Kokkos::deep_copy(d_I_global.base(), h_I_global.base());
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
int Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::pack_forward_host(int layer, int n, const int *list, double *buf)
{
  comm_values(n, comm_width(layer));    // the int count LAMMPS gets
  int m = 0;
  // Layer 1 packs the parity-doubled round-2 indicator eq2_norm from
  // h_eq2_norm_global; layer 0 packs eq1_norm from h_I_global. Same buffer layout.
  const bool s2 = (layer == 1);
  const int nf = s2 ? I2_n_funcs : I_n_funcs;
  const int no = s2 ? I2_n_out   : I_n_out;
  auto &G = s2 ? h_eq2_norm_global : h_I_global;
  const int nmax = (int)G.extent(0);
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    if (j >= 0 && j < nmax) {
      for (int f = 0; f < nf; f++)
        for (int k = 0; k < no; k++)
          buf[m++] = G(j, f, k);
    } else {
      for (int f = 0; f < nf; f++)
        for (int k = 0; k < no; k++)
          buf[m++] = 0.0;
    }
  }
  return m;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::unpack_forward_host(int layer, int n, int first, const double *buf)
{
  int m = 0;
  const bool s2 = (layer == 1);
  const int nf = s2 ? I2_n_funcs : I_n_funcs;
  const int no = s2 ? I2_n_out   : I_n_out;
  auto &G = s2 ? h_eq2_norm_global : h_I_global;
  const int nmax = (int)G.extent(0);
  for (int i = 0; i < n; i++) {
    const int j = first + i;
    if (j >= 0 && j < nmax) {
      for (int f = 0; f < nf; f++)
        for (int k = 0; k < no; k++)
          G(j, f, k) = buf[m++];
    } else {
      m += nf * no;  // skip
    }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
int Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::pack_reverse_host(int layer, int n, int first, double *buf)
{
  comm_values(n, comm_width(layer));    // the int count LAMMPS gets
  int m = 0;
  // Layer 0 reverse-comms the round-1 grad_I from h_grad_I_global; layer 1 reverse-comms the
  // parity-doubled round-2 eq2_norm message adjoint from h_eq2_norm_adj.
  const bool s2 = (layer == 1);
  const int nf = s2 ? I2_n_funcs : I_n_funcs;
  const int no = s2 ? I2_n_out   : I_n_out;
  auto &G = s2 ? h_eq2_norm_adj : h_grad_I_global;
  for (int i = 0; i < n; i++) {
    const int j = first + i;
    for (int f = 0; f < nf; f++)
      for (int k = 0; k < no; k++)
        buf[m++] = G(j, f, k);
  }
  return m;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::unpack_reverse_host(int layer, int n, const int *list, const double *buf)
{
  int m = 0;
  const bool s2 = (layer == 1);
  const int nf = s2 ? I2_n_funcs : I_n_funcs;
  const int no = s2 ? I2_n_out   : I_n_out;
  auto &G = s2 ? h_eq2_norm_adj : h_grad_I_global;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    for (int f = 0; f < nf; f++)
      for (int k = 0; k < no; k++)
        G(j, f, k) += buf[m++];
  }
}

// Device pack / unpack: one thread per (atom, value), the row-major message rows are contiguous per atom.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
int Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::pack_forward(int layer, int n,
                                                                        const typename Traits::t_int_1d &sendlist,
                                                                        const typename Traits::t_double_1d &buf)
{
  d_sendlist = sendlist;
  v_buf = buf;
  comm_stage = layer + 1;
  comm_nfo = comm_width(layer);
  const int nv = comm_values(n, comm_nfo);    // the int count LAMMPS gets
  Kokkos::parallel_for("PackForwardComm",
    Kokkos::RangePolicy<DeviceType, TagPackForwardComm>(0, nv), *this);
  return nv;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagPackForwardComm, const int& idx) const
{
  // comm_stage (captured by value in *this) selects round-1 (eq1_norm ->
  // d_I_global) vs round-2 (eq2_norm -> d_eq2_norm_global, parity-doubled lm).
  const int i = idx / comm_nfo, q = idx - i * comm_nfo;
  const int j = d_sendlist(i);
  if (comm_stage == 2) v_buf(idx) = d_eq2_norm_global(j, q / I2_n_out, q % I2_n_out);
  else                 v_buf(idx) = d_I_global(j, q / I_n_out, q % I_n_out);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::unpack_forward(int layer, int n, int first,
                                                                           const typename Traits::t_double_1d &buf)
{
  comm_first = first;
  v_buf = buf;
  comm_stage = layer + 1;
  comm_nfo = comm_width(layer);
  Kokkos::parallel_for("UnpackForwardComm",
    Kokkos::RangePolicy<DeviceType, TagUnpackForwardComm>(0, comm_values(n, comm_nfo)), *this);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagUnpackForwardComm, const int& idx) const
{
  const int i = idx / comm_nfo, q = idx - i * comm_nfo;
  const int j = comm_first + i;
  if (comm_stage == 2) d_eq2_norm_global(j, q / I2_n_out, q % I2_n_out) = v_buf(idx);
  else                 d_I_global(j, q / I_n_out, q % I_n_out) = v_buf(idx);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
int Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::pack_reverse(int layer, int n, int first,
                                                                        const typename Traits::t_double_1d &buf)
{
  comm_first = first;
  v_buf = buf;
  comm_stage = layer + 1;
  comm_nfo = comm_width(layer);
  const int nv = comm_values(n, comm_nfo);    // the int count LAMMPS gets
  Kokkos::parallel_for("PackReverseComm",
    Kokkos::RangePolicy<DeviceType, TagPackReverseComm>(0, nv), *this);
  return nv;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagPackReverseComm, const int& idx) const
{
  // Stage 2: pack the round-2 eq2_norm message adjoint (d_eq2_norm_adj,
  // parity-doubled lm); stage 1: pack round-1 grad_I (d_grad_I_global).
  const int i = idx / comm_nfo, q = idx - i * comm_nfo;
  const int j = comm_first + i;
  if (comm_stage == 2) v_buf(idx) = d_eq2_norm_adj(j, q / I2_n_out, q % I2_n_out);
  else                 v_buf(idx) = d_grad_I_global(j, q / I_n_out, q % I_n_out);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::unpack_reverse(int layer, int n,
                                                                           const typename Traits::t_int_1d &sendlist,
                                                                           const typename Traits::t_double_1d &buf)
{
  d_sendlist = sendlist;
  v_buf = buf;
  comm_stage = layer + 1;
  comm_nfo = comm_width(layer);
  Kokkos::parallel_for("UnpackReverseComm",
    Kokkos::RangePolicy<DeviceType, TagUnpackReverseComm>(0, comm_values(n, comm_nfo)), *this);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagUnpackReverseComm, const int& idx) const
{
  const int i = idx / comm_nfo, q = idx - i * comm_nfo;
  const int j = d_sendlist(i);
  if (comm_stage == 2) Kokkos::atomic_add(&d_eq2_norm_adj(j, q / I2_n_out, q % I2_n_out), (NNScalar) v_buf(idx));
  else                 Kokkos::atomic_add(&d_grad_I_global(j, q / I_n_out, q % I_n_out), (NNScalar) v_buf(idx));
}

// ======================================================================
// forward-chunk recompute helpers. l1_forward_chunk() runs the full
// Layer-1 forward body (geometry -> A1 chain -> eq1_norm/rho1_norm + the natom
// ScatterRho1Norm) for the CURRENT chunk (member chunk_offset/chunk_size), but
// STOPS BEFORE the global eq1_norm scatter (P1 does that; the backward recompute
// in P5 must NOT touch d_I_global). l2_forward_chunk() is the Layer-2 analogue
// (through rho2_norm + ScatterRho2Norm, excluding the eq2_norm global scatter).
// Because the inter-layer indicators d_I_global/d_eq2_norm_global are already
// ghost-complete (forward_comm ran in P1.5/P2.5), re-running these bodies in
// P4/P5 reproduces bit-identical per-chunk forward intermediates.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::l1_forward_chunk()
{
  int team_size = 1;
  int vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
    team_size = 32;

  // ---- Stage 1: ComputeNeigh ----
  {
    team_size = fit_team_size<DeviceType, TagComputeNeigh>(*this, chunk_size, team_size, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>(chunk_size, team_size, vector_length);
    Kokkos::parallel_for("ComputeNeigh", policy, *this);
  }

  // ---- Stage 1b: ComputeYBond — precompute real spherical harmonics per bond ----
  compute_Y_bond_chunk();

  // ---- Stage 2: ComputeRadialBasis — Chebyshev basis (raw) + envelope ----
  {
    int ts = team_size;
    ts = fit_team_size<DeviceType, TagComputeRadialBasis>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>(
        ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    Kokkos::parallel_for("ComputeRadialBasis", policy, *this);
  }

  // ---- Stage 3: ComputeMLPRadial — Layer-1 radial MLP ----
  // ---- Stage 4: ComputeA1 — Layer-1 scalar-indicator SPBF basis ----
  radial_mlp_spbf = 0;   // A1
  if (l1_radial_fused()) {
    compute_A1_tables();    // fuse_radial: the radial from the tables inside ComputeA1
  } else {
    compute_mlp_radial();  // cuBLAS SGEMM path (or hand-kernel fallback)
    Kokkos::parallel_for("ComputeA1",
        Kokkos::RangePolicy<DeviceType, TagComputeA1>(0, chunk_size * L1_nradmax), *this);
  }

  // ---- Stage 5: cp_l product A1_2 = cp_l(A1, A1) ----
  compute_cp_l(d_prod[0], d_A1, d_A1, d_A1_2);

  // ---- Stage 6: FunctionReduceN A1_2_red = reduce_n({"A1_2": A1_2}) ----
  {
    t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
    inputs[0] = d_A1_2;   // matches reduce["A1_2_red"].instr[0].name == "A1_2" (checked in setup())
    compute_reduceN(d_reduce[idx_reduce_A1_2_red], inputs, d_A1_2_red);
  }

  // ---- Stage 7-8: A1_2a = fc_right2left(A1_2_red, A1); A1_3 = cp_l(A1_2a, A1) (fc folded into the projection
  // with fc_fold) ----
  if (fold_on[0]) {
    compute_cp_l_fold(d_prod[idx_prod_A1_3], d_A1_2_red, d_A1, d_A1, d_A1_3);
  } else {
    compute_fc(d_fc[idx_fc_A1_2a], d_A1_2_red, d_A1, d_A1_2a);
    compute_cp_l(d_prod[idx_prod_A1_3], d_A1_2a, d_A1, d_A1_3);
  }

  // ---- Stage 9: FunctionReduceN A1_3_red = reduce_n({"A1_3": A1_3}) ----
  {
    t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
    inputs[0] = d_A1_3;   // matches reduce["A1_3_red"].instr[0].name == "A1_3" (checked in setup())
    compute_reduceN(d_reduce[idx_reduce_A1_3_red], inputs, d_A1_3_red);
  }

  // ---- Stage 10-11: A1_2b = fc_right2left(A1_2_red, A1); A1_4 = cp_l(A1_2b, A1_2b) (fc folded with fc_fold) ----
  if (fold_on[0]) {
    compute_cp_l_fold(d_prod[idx_prod_A1_4], d_A1_2_red, d_A1, t_nn_3d(), d_A1_4);
  } else {
    compute_fc(d_fc[idx_fc_A1_2b], d_A1_2_red, d_A1, d_A1_2b);
    compute_cp_l(d_prod[idx_prod_A1_4], d_A1_2b, d_A1_2b, d_A1_4);
  }

  // ---- Stage 12: FunctionReduceN A1_4_red = reduce_n({"A1_4": A1_4}) ----
  {
    t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
    inputs[0] = d_A1_4;   // matches reduce["A1_4_red"].instr[0].name == "A1_4" (checked in setup())
    compute_reduceN(d_reduce[idx_reduce_A1_4_red], inputs, d_A1_4_red);
  }

  // ---- Stage 13: FunctionReduceN eq1 = reduce_n({A1,A1_2_red,A1_3_red,A1_4_red}) ----
  {
    t_nn_3d l1_views[4] = {d_A1, d_A1_2_red, d_A1_3_red, d_A1_4_red};
    t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
    const int n_instr = d_reduce[idx_reduce_eq1].n_instr;
    for (int j = 0; j < n_instr; j++) inputs[j] = l1_views[eq1_input_role[j]];
    compute_reduceN(d_reduce[idx_reduce_eq1], inputs, d_eq1);
  }

  // ---- Stage 14: EquivariantRMSNorm eq1_norm = equivariant_rms_norm(eq1) ----
  compute_equiv_rms_norm(d_eqnorm[idx_eqnorm_eq1_norm], d_eq1, d_eq1_norm);
  // NOTE: Stage 14b (ScatterEq1Norm -> d_I_global) is done by the P1 caller ONLY.

  // ---- Stage 15: FunctionReduceN rho1 = reduce_n({A1,A1_2_red,A1_3_red,A1_4_red}) ----
  {
    t_nn_3d l1_views[4] = {d_A1, d_A1_2_red, d_A1_3_red, d_A1_4_red};
    t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
    const int n_instr = d_reduce[idx_reduce_rho1].n_instr;
    for (int j = 0; j < n_instr; j++) inputs[j] = l1_views[rho1_input_role[j]];
    compute_reduceN(d_reduce[idx_reduce_rho1], inputs, d_rho1);
  }

  // ---- Stage 16: InvariantLayerRMSNorm rho1_norm = invariant_rms_norm(rho1) ----
  compute_invariant_rms_norm(d_rmsnorm[idx_rmsnorm_rho1_norm], d_rho1, d_rho1_norm);

  // ---- Stage 16b: scatter rho1_norm into the natom buffer
  // d_rho1_norm_full (dense list-order index ii+chunk_offset; local-only, no comm). ----
  {
    auto rho1n = d_rho1_norm;
    auto full = d_rho1_norm_full;
    const int co = chunk_offset;
    const int cs = chunk_size;
    const int nc = (int) rho1n.extent(1), nl = (int) rho1n.extent(2);
    Kokkos::parallel_for("ScatterRho1Norm",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nc, nl}),
      KOKKOS_LAMBDA(const int ii, const int c, const int lm) {
        full(ii + co, c, lm) = rho1n(ii, c, lm);
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::l2_forward_chunk()
{
  int team_size = 1;
  int vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
    team_size = 32;

  // ---- L2 Stage 0: recompute neighbor data for this chunk ----
  {
    team_size = fit_team_size<DeviceType, TagComputeNeigh>(*this, chunk_size, team_size, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>(chunk_size, team_size, vector_length);
    Kokkos::parallel_for("L2_ComputeNeigh", policy, *this);
  }

  // ---- L2 Stage 0b: ComputeYBond ----
  compute_Y_bond_chunk();

  // ---- L2 Stage 0c: ComputeRadialBasis ----
  {
    int ts = team_size;
    ts = fit_team_size<DeviceType, TagComputeRadialBasis>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>(
        ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    Kokkos::parallel_for("L2_ComputeRadialBasis", policy, *this);
  }

  // ---- L2 Stage 1: ComputeMLPRadial — A2 radial MLP (spbf idx_spbf_A2) ----
  // (fuse_radial: the fast SPBF forward reads the tables itself)
  radial_mlp_spbf = idx_spbf_A2;
  if (!(fuse_radial && spline_on && spbf_rm_ok(idx_spbf_A2, idx_reduce_A2_red)))
    compute_mlp_radial();  // cuBLAS SGEMM path (or hand-kernel fallback)

  // ---- L2 Stage 2: ComputeA2 — equiv-indicator SPBF basis ----
  // fused team+shared kernel (indicator = d_I_global, out = d_A2).
  // ---- L2 Stage 3: FunctionReduceN A2_red = reduce_n({"A2": A2}) ----
#ifdef GRACE_KK_GEMM
  if (spbf_rm_ok(idx_spbf_A2, idx_reduce_A2_red)) {
    spbf_fwd_fast(idx_spbf_A2, d_I_global, d_A2, true);
    compute_reduceN_grouped_rm(idx_reduce_A2_red, d_A2_red, false);
  } else
#endif
  {
    compute_spbf_equiv(idx_spbf_A2, d_I_global, d_A2);
    t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
    inputs[0] = d_A2;   // matches reduce["A2_red"].instr[0].name == "A2" (checked in setup())
    compute_reduceN(d_reduce[idx_reduce_A2_red], inputs, d_A2_red);
  }

  // ---- L2 Stage 4: cp_l product A2_2 = cp_l(A2_red, A2_red) ----
  compute_cp_l(d_prod[idx_prod_A2_2], d_A2_red, d_A2_red, d_A2_2);

  // ---- L2 Stage 5: FunctionReduceN A2_2_red = reduce_n({"A2_2": A2_2}) ----
  {
    t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
    inputs[0] = d_A2_2;   // matches reduce["A2_2_red"].instr[0].name == "A2_2" (checked in setup())
    compute_reduceN(d_reduce[idx_reduce_A2_2_red], inputs, d_A2_2_red);
  }

  // ---- L2 Stage 6-7: A2_2a = fc_right2left(A2_2_red, A2_red); A2_3 = cp_l(A2_2a, A2_red) (fc folded into the
  // projection with fc_fold) ----
  if (fold_on[1]) {
    compute_cp_l_fold(d_prod[idx_prod_A2_3], d_A2_2_red, d_A2_red, d_A2_red, d_A2_3);
  } else {
    compute_fc(d_fc[idx_fc_A2_2a], d_A2_2_red, d_A2_red, d_A2_2a);
    compute_cp_l(d_prod[idx_prod_A2_3], d_A2_2a, d_A2_red, d_A2_3);
  }

  // ---- L2 Stage 8: FunctionReduceN A2_3_red = reduce_n({"A2_3": A2_3}) ----
  {
    t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
    inputs[0] = d_A2_3;   // matches reduce["A2_3_red"].instr[0].name == "A2_3" (checked in setup())
    compute_reduceN(d_reduce[idx_reduce_A2_3_red], inputs, d_A2_3_red);
  }

  // ---- L2 Stage 9-10: A2_2b = fc_right2left(A2_2_red, A2_red); A2_4 = cp_l(A2_2b, A2_2b) (fc folded with
  // fc_fold) ----
  if (fold_on[1]) {
    compute_cp_l_fold(d_prod[idx_prod_A2_4], d_A2_2_red, d_A2_red, t_nn_3d(), d_A2_4);
  } else {
    compute_fc(d_fc[idx_fc_A2_2b], d_A2_2_red, d_A2_red, d_A2_2b);
    compute_cp_l(d_prod[idx_prod_A2_4], d_A2_2b, d_A2_2b, d_A2_4);
  }

  // ---- L2 Stage 11: FunctionReduceN A2_4_red = reduce_n({"A2_4": A2_4}) ----
  {
    t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
    inputs[0] = d_A2_4;   // matches reduce["A2_4_red"].instr[0].name == "A2_4" (checked in setup())
    compute_reduceN(d_reduce[idx_reduce_A2_4_red], inputs, d_A2_4_red);
  }

  // ---- L2 Stage 12: FunctionReduceN eq2 = reduce_n({A2_red,A2_2_red,A2_3_red,A2_4_red}) ----
  {
    t_nn_3d l2_views[4] = {d_A2_red, d_A2_2_red, d_A2_3_red, d_A2_4_red};
    t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
    const int n_instr = d_reduce[idx_reduce_eq2].n_instr;
    for (int j = 0; j < n_instr; j++) inputs[j] = l2_views[eq2_input_role[j]];
    compute_reduceN(d_reduce[idx_reduce_eq2], inputs, d_eq2);
  }

  // ---- L2 Stage 13: EquivariantRMSNorm eq2_norm = equivariant_rms_norm(eq2) ----
  compute_equiv_rms_norm(d_eqnorm[idx_eqnorm_eq2_norm], d_eq2, d_eq2_norm);
  // NOTE: Stage 13b (ScatterEq2Norm -> d_eq2_norm_global) is done by the P2 caller ONLY.

  // ---- L2 Stage 14: FunctionReduceN rho2 = reduce_n({A2_red,A2_2_red,A2_3_red,A2_4_red}) ----
  {
    t_nn_3d l2_views[4] = {d_A2_red, d_A2_2_red, d_A2_3_red, d_A2_4_red};
    t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
    const int n_instr = d_reduce[idx_reduce_rho2].n_instr;
    for (int j = 0; j < n_instr; j++) inputs[j] = l2_views[rho2_input_role[j]];
    compute_reduceN(d_reduce[idx_reduce_rho2], inputs, d_rho2);
  }

  // ---- L2 Stage 15: InvariantLayerRMSNorm rho2_norm = invariant_rms_norm(rho2) ----
  compute_invariant_rms_norm(d_rmsnorm[idx_rmsnorm_rho2_norm], d_rho2, d_rho2_norm);

  // ---- L2 Stage 15b: scatter rho2_norm into the natom buffer
  // d_rho2_norm_full (dense list-order index ii+chunk_offset; local-only, no comm). ----
  {
    auto rho2n = d_rho2_norm;
    auto full = d_rho2_norm_full;
    const int co = chunk_offset;
    const int cs = chunk_size;
    const int nc = (int) rho2n.extent(1), nl = (int) rho2n.extent(2);
    Kokkos::parallel_for("ScatterRho2Norm",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nc, nl}),
      KOKKOS_LAMBDA(const int ii, const int c, const int lm) {
        full(ii + co, c, lm) = rho2n(ii, c, lm);
      });
  }
}

// ======================================================================
// scatter_forces_chunk — scatter the CURRENT chunk's per-bond force
// buffer d_f_ij into the global force array f[] (atomic; F_i += f_ij, F_j -=
// f_ij) and accumulate the per-bond virial into virial[] (+=) / d_vatom /
// d_cvatom. Called once per layer per chunk (after that layer's force kernel),
// so each layer's partial force accumulates into f[]/virial[] independently ->
// identical total.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::scatter_forces_chunk()
{
  auto fij = d_f_ij; auto nc = d_ncount; auto near = d_nearest;
  auto fout = f; auto il = d_ilist;
  auto rh = d_rhats; auto rn = d_rnorms;
  int co = chunk_offset, cs = chunk_size;
  bool do_v = (vflag_global != 0);
  bool do_va = (vflag_atom != 0);
  bool do_cva = (cvflag_atom != 0);
  auto va = d_vatom; auto cva = d_cvatom;
  const bool cpl = couple_eq != 0; auto cp = d_cpl; auto dedl = d_dedl;

  EV ev_force;
  auto body = KOKKOS_LAMBDA(const int ii, EV& ev) {
      const int i = il[ii + co];
      const int nct = nc(ii);
      double dl_i = 0.0;
      for (int jj = 0; jj < nct; jj++) {
        const int j = near(ii, jj);
        GeomScalar fx = fij(ii, jj, 0);
        GeomScalar fy = fij(ii, jj, 1);
        GeomScalar fz = fij(ii, jj, 2);
        BondScalar rv = rn(ii, jj);
        if (cpl) {    // two-group coupling: f = J^T dE/dr', J = dr'/dr n n^T + r'/r (1 - n n^T); virial at the true r
          const BondScalar nx = rh(ii, jj, 0), ny = rh(ii, jj, 1), nz = rh(ii, jj, 2);
          const BondScalar rt = cp(ii, jj, 0), fn = fx * nx + fy * ny + fz * nz, sc = rv / rt;
          dl_i += cp(ii, jj, 2) * fn;
          const BondScalar c = (cp(ii, jj, 1) - sc) * fn;
          fx = GeomScalar(sc * fx + c * nx); fy = GeomScalar(sc * fy + c * ny); fz = GeomScalar(sc * fz + c * nz);
          rv = rt;
        }
        Kokkos::atomic_add(&fout(i, 0), fx);
        Kokkos::atomic_add(&fout(i, 1), fy);
        Kokkos::atomic_add(&fout(i, 2), fz);
        Kokkos::atomic_add(&fout(j, 0), -fx);
        Kokkos::atomic_add(&fout(j, 1), -fy);
        Kokkos::atomic_add(&fout(j, 2), -fz);
        if (do_v || do_va || do_cva) {
          const BondScalar delx = -rh(ii, jj, 0) * rv;
          const BondScalar dely = -rh(ii, jj, 1) * rv;
          const BondScalar delz = -rh(ii, jj, 2) * rv;
          const BondScalar v0 = delx*fx, v1 = dely*fy, v2 = delz*fz;
          const BondScalar v3 = delx*fy, v4 = delx*fz, v5 = dely*fz;
          if (do_v) { ev.v[0]+=v0; ev.v[1]+=v1; ev.v[2]+=v2; ev.v[3]+=v3; ev.v[4]+=v4; ev.v[5]+=v5; }
          if (do_va) {
            Kokkos::atomic_add(&va(i,0),(KKF)(BondScalar(0.5)*v0)); Kokkos::atomic_add(&va(i,1),(KKF)(BondScalar(0.5)*v1));
            Kokkos::atomic_add(&va(i,2),(KKF)(BondScalar(0.5)*v2)); Kokkos::atomic_add(&va(i,3),(KKF)(BondScalar(0.5)*v3));
            Kokkos::atomic_add(&va(i,4),(KKF)(BondScalar(0.5)*v4)); Kokkos::atomic_add(&va(i,5),(KKF)(BondScalar(0.5)*v5));
            Kokkos::atomic_add(&va(j,0),(KKF)(BondScalar(0.5)*v0)); Kokkos::atomic_add(&va(j,1),(KKF)(BondScalar(0.5)*v1));
            Kokkos::atomic_add(&va(j,2),(KKF)(BondScalar(0.5)*v2)); Kokkos::atomic_add(&va(j,3),(KKF)(BondScalar(0.5)*v3));
            Kokkos::atomic_add(&va(j,4),(KKF)(BondScalar(0.5)*v4)); Kokkos::atomic_add(&va(j,5),(KKF)(BondScalar(0.5)*v5));
          }
          if (do_cva) {
            const BondScalar v6=dely*fx, v7=delz*fx, v8=delz*fy;
            Kokkos::atomic_add(&cva(i,0),(KKF)(BondScalar(0.5)*v0)); Kokkos::atomic_add(&cva(i,1),(KKF)(BondScalar(0.5)*v1));
            Kokkos::atomic_add(&cva(i,2),(KKF)(BondScalar(0.5)*v2)); Kokkos::atomic_add(&cva(i,3),(KKF)(BondScalar(0.5)*v3));
            Kokkos::atomic_add(&cva(i,4),(KKF)(BondScalar(0.5)*v4)); Kokkos::atomic_add(&cva(i,5),(KKF)(BondScalar(0.5)*v5));
            Kokkos::atomic_add(&cva(i,6),(KKF)(BondScalar(0.5)*v6)); Kokkos::atomic_add(&cva(i,7),(KKF)(BondScalar(0.5)*v7));
            Kokkos::atomic_add(&cva(i,8),(KKF)(BondScalar(0.5)*v8));
            Kokkos::atomic_add(&cva(j,0),(KKF)(BondScalar(0.5)*v0)); Kokkos::atomic_add(&cva(j,1),(KKF)(BondScalar(0.5)*v1));
            Kokkos::atomic_add(&cva(j,2),(KKF)(BondScalar(0.5)*v2)); Kokkos::atomic_add(&cva(j,3),(KKF)(BondScalar(0.5)*v3));
            Kokkos::atomic_add(&cva(j,4),(KKF)(BondScalar(0.5)*v4)); Kokkos::atomic_add(&cva(j,5),(KKF)(BondScalar(0.5)*v5));
            Kokkos::atomic_add(&cva(j,6),(KKF)(BondScalar(0.5)*v6)); Kokkos::atomic_add(&cva(j,7),(KKF)(BondScalar(0.5)*v7));
            Kokkos::atomic_add(&cva(j,8),(KKF)(BondScalar(0.5)*v8));
          }
        }
      }
      if (cpl) Kokkos::atomic_add(&dedl(), dl_i);
    };
  // the global virial sum only when requested (a reduction to a host value is a fence)
  if (do_v) Kokkos::parallel_reduce("ComputeForce3L", Kokkos::RangePolicy<DeviceType>(0, cs), body, ev_force);
  else Kokkos::parallel_for("ComputeForce3L", Kokkos::RangePolicy<DeviceType>(0, cs), KOKKOS_LAMBDA(const int ii) {
    EV e;
    body(ii, e);
  });
  if (vflag_global) {
    virial[0]+=ev_force.v[0]; virial[1]+=ev_force.v[1]; virial[2]+=ev_force.v[2];
    virial[3]+=ev_force.v[3]; virial[4]+=ev_force.v[4]; virial[5]+=ev_force.v[5];
  }
}

// ======================================================================
// run (LAMMPS: compute): full forward BASIS chain (L1/L2/L3) + 3-density readout -> per-atom/total energy, then the
// chunked backward with the per-layer geometry forces. The layer-0 / layer-1 messages go to the ghost atoms through
// in.ex: forward(0) after Phase 1, forward(1) after Phase 2, reverse(1) after Phase 3, reverse(0) after Phase 4.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
typename Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::RunOutput
Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::run(const RunInput3 &in)
{
  if (!setup_done) GRACE_KK_ERROR("GRACE-3L/KK: run() before setup() (after set_model())");
  if (in.nlocal < in.inum || in.nall < in.nlocal)
    GRACE_KK_ERROR("GRACE-3L/KK: need inum <= nlocal <= nall (inum {}, nlocal {}, nall {})", in.inum, in.nlocal, in.nall);
  eflag_global = in.eflag_global;
  eflag_atom = in.eflag_atom;
  vflag_global = in.vflag_global;
  vflag_atom = in.vflag_atom;
  cvflag_atom = in.cvflag_atom;
  eflag = eflag_global || eflag_atom;
  vflag = vflag_global || vflag_atom || cvflag_atom;
  nlocal = in.nlocal;
  eng_vdwl = 0.0;
  for (int k = 0; k < 6; k++) virial[k] = 0.0;
  cpl_gen++;   // invalidates the stored cp_l projections

  // The energy-only fast path skips the entire backward (L3 Stage 15 + PHASES 3.5/4/4.5/5 + per-chunk force
  // scatter) when the caller requests energy without forces (LAMMPS: ENERGY_ONLY in eflag, e.g. MC fixes,
  // fix_numdiff, compute_fep). UQ is forward-only and still runs. `debug_no_energy_only_calc` forces the full
  // backward for validation/benchmarking.
  const bool do_energy_only = in.energy_only && !debug_no_energy_only_calc;

  d_eatom = in.eatom;
  d_vatom = in.vatom;
  d_cvatom = in.cvatom;
  x = in.x;
  couple_eq = in.couple_eq; couple_bit1 = in.couple_bit1; couple_bit2 = in.couple_bit2;
  couple_lambda = in.couple_lambda; d_mask = in.mask;
  if (couple_eq) {
    if (!couple_supported) GRACE_KK_ERROR("GRACE-3L/KK: two-group coupling needs a fused engine");
    if (couple_eq < 1 || couple_eq > 4) GRACE_KK_ERROR("GRACE-3L/KK: coupling equation {} (must be 1..4)", couple_eq);
    if (!d_dedl.data()) d_dedl = Kokkos::View<double, DeviceType>("grace:dedl");
    Kokkos::deep_copy(d_dedl, 0.0);
  }
  f = in.f;
  type = in.type;
  d_numneigh = in.numneigh;
  d_neighbors = in.neighbors;
  d_ilist = in.ilist;
  inum = in.inum;
  // Element-dependent weights are read per atom (one thread or team per atom): order the atoms by type so a
  // warp reads one weight row instead of up to 32 different ones. Stable (key = type * inum + ii).
  static const bool sort_by_type = !getenv("GRACE_NO_TYPE_SORT");
  if (sort_by_type && in.ntypes > 1 && inum > 0) {
    if ((int) d_ilist_sorted.extent(0) < inum) {
      d_ilist_sorted = typename Traits::t_int_1d("g3l:ilist_sorted", inum);
      d_type_key = Kokkos::View<long long *, DeviceType>("g3l:type_key", inum);
    }
    auto src = in.ilist; auto ty = type; auto key = d_type_key; auto dst = d_ilist_sorted;
    const long long n = inum;
    Kokkos::parallel_for("grace_3l::type_key", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii) { key(ii) = (long long) ty(src(ii)) * n + ii; });
    Kokkos::sort(Kokkos::subview(key, std::make_pair(0, inum)));
    Kokkos::parallel_for("grace_3l::type_sort", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii) { dst(ii) = src((int) (key(ii) % n)); });
    d_ilist = d_ilist_sorted;
  }

  maxneigh = 0;
  {
    auto il = in.ilist; auto nn = in.numneigh;
    Kokkos::parallel_reduce("grace_3l::find_maxneigh", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii, int &mx) { const int c = nn[il[ii]]; if (mx < c) mx = c; },
        Kokkos::Max<int>(maxneigh));
  }
  maxneigh = std::max(maxneigh, 0);    // Kokkos::Max over no atoms (an empty rank) gives INT_MIN
  choose_chunk();
  chunk_lim = chunk_cap();

  chunk_size = std::min(chunk_lim, inum);
  chunk_offset = 0;
  grow(chunk_size, maxneigh);
  if (couple_eq && (d_cpl.extent(0) < d_rnorms.extent(0) || d_cpl.extent(1) < d_rnorms.extent(1)))
    MemKK::realloc_kokkos(d_cpl, "g3l:cpl", d_rnorms.extent(0), d_rnorms.extent(1));

  // Global (nall-sized) arrays for the inter-layer message comm.
  nall = in.nall;
  grow_global(nall);
  realloc_all = false;
  setup_fwd_store();
  // Zero the whole indicator array: owned rows are overwritten by the L1 loop,
  // ghost rows by comm->forward_comm; zeroing guards any atom left untouched.
  Kokkos::deep_copy(d_I_global.base(), NNScalar(0.0));
  // Round-2 (L3) indicator buffer: owned rows written by the L2 loop's eq2_norm
  // scatter, ghost rows by the round-2 forward_comm. Zero first.
  Kokkos::deep_copy(d_eq2_norm_global.base(), NNScalar(0.0));

  // ---- UQ / extrapolation grade: allocate per-atom outputs up front. The
  // basis-RP feature is assembled in three parts across the forward: rho1 in
  // Phase 1, rho2 in Phase 2 (both accumulate the RAW proj into d_uq_z), and
  // rho3 + the GMM in Phase 3. All UQ kernels are any_uq-gated, so the energy
  // path is byte-for-byte unchanged when UQ is not requested. ----
  const bool any_uq = in.uq && has_uq;
  if (any_uq) {
    // Also reallocate if d_uq_z's projection width no longer matches uq_rp_dim: a
    // mid-session pair_coeff that loads a model with a different rp_dim (while nlocal
    // has not grown) would otherwise leave d_uq_z sized to the OLD width, and the
    // rho1 kernel would write d_uq_z(i,d) out of bounds.
    if ((int)d_gamma.extent(0) < nlocal || (int)d_uq_z.extent(1) != uq_rp_dim) {
      MemKK::realloc_kokkos(d_gamma, "g3l:d_gamma", nlocal);
      MemKK::realloc_kokkos(d_sigma, "g3l:d_sigma", nlocal);
      MemKK::realloc_kokkos(d_gmm_cluster, "g3l:d_gmm_cluster", nlocal);
      MemKK::realloc_kokkos(d_uq_z, "g3l:d_uq_z", nlocal, uq_rp_dim);  // per-atom RAW proj (rho1+rho2 carry)
      MemKK::realloc_kokkos(d_uq_n2_rho1, "g3l:d_uq_n2_rho1", nlocal); // ||B^(rho1)||^2 carry
      MemKK::realloc_kokkos(d_uq_n2_rho2, "g3l:d_uq_n2_rho2", nlocal); // ||B^(rho2)||^2 carry
    }
  }

  // ============ PHASE 1: Layer-1 forward (chunked, ALL owned atoms) ============
  // Runs the full L1 pipeline (geometry -> A1 chain -> eq1_norm/rho1_norm) for
  // every owned atom, scattering each atom's eq1_norm into the global indicator
  // array d_I_global at its LAMMPS local index. After all chunks, forward_comm
  // fills the ghost rows; then Phase 2 computes A2 reading indicator[ind_j].
  chunk_offset = 0;
  while (chunk_offset < inum) {
    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

    // ---- L1 forward chunk body (Stages 1-16, incl. ScatterRho1Norm), EXCEPT
    // the Stage-14b eq1_norm global scatter which P1 does below (and P5's
    // backward recompute must NOT do). See l1_forward_chunk(). ----
    select_fwd_store(1, chunk_offset / std::min(chunk_lim, inum));
    l1_forward_chunk();

    // ---- Stage 14b: scatter eq1_norm into the global indicator array.
    // d_eq1_norm is chunk-indexed (ii); write it to d_I_global at the atom's
    // LAMMPS local index i=d_ilist(ii+chunk_offset) so forward_comm can then
    // fill the ghost rows. Layout matches: (n_out, M) == (I_n_funcs, I_n_out). ----
    {
      auto eq1n = d_eq1_norm;
      auto Ig = d_I_global;
      auto il = d_ilist;
      const int co = chunk_offset;
      const int nf = I_n_funcs, no = I_n_out, cs = chunk_size;
      Kokkos::parallel_for("ScatterEq1Norm",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nf, no}),
        KOKKOS_LAMBDA(const int ii, const int fdim, const int k) {
          Ig(il(ii + co), fdim, k) = eq1n(ii, fdim, k);
        });
    }

    // ---- UQ basis-RP (Phase 1): project the rho1 (L1) scalar basis into d_uq_z
    // for this chunk (d_A1/d_A1_2_red/d_A1_3_red/d_A1_4_red are current here). ----
    if (any_uq) {
      // Team-per-atom: lanes split the rho1 B·R projection over rp_dim (register acc).
      auto probe = Kokkos::TeamPolicy<DeviceType, TagComputeUQ_rho1>(chunk_size, 1, 1);
      int ts = probe.team_size_max(*this, Kokkos::ParallelForTag());
      if (ts > 128) ts = 128; if (ts < 1) ts = 1;
      Kokkos::parallel_for("ComputeUQ_rho1",
          Kokkos::TeamPolicy<DeviceType, TagComputeUQ_rho1>(chunk_size, ts, 1), *this);
    }

    Kokkos::fence();
    chunk_offset += chunk_size;
  }

  // ============ PHASE 1.5: forward-comm eq1_norm to ghost atoms ====
  // Every owned atom's eq1_norm now lives in d_I_global; forward_comm copies
  // owned rows into their ghost images so the A2 kernel can read the neighbor
  // indicator at ghost ind_j. When legacy pair comm is active (GPU-aware MPI
  // off / multi-proc host path), LAMMPS pack/unpack reads h_I_global, so sync
  // device->host before and host->device after; otherwise the kokkos-native
  // pack/unpack operates on d_I_global directly. (Mirrors GRACE-2L.)
  if (in.ex) in.ex->forward(0);   // round 1: eq1_norm rows of d_I_global

  // ============ PHASE 2: Layer-2 forward (chunked) — A2 ============
  // The per-chunk geometry arrays (d_ncount/d_nearest/d_rhats/d_radial_basis/
  // d_Y_bond/d_R1_nl) hold the LAST L1 chunk's data, so each L2 chunk must
  // recompute geometry + the A2 radial MLP before assembling A2.
  chunk_size = std::min(chunk_lim, inum);
  chunk_offset = 0;
  while (chunk_offset < inum) {
    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

    // ---- L2 forward chunk body (Stages 0-15, incl. ScatterRho2Norm), EXCEPT
    // the Stage-13b eq2_norm global scatter which P2 does below (and P4's
    // backward recompute must NOT do). See l2_forward_chunk(). ----
    select_fwd_store(2, chunk_offset / std::min(chunk_lim, inum));
    l2_forward_chunk();

    // ---- L2 Stage 13b: scatter eq2_norm into the round-2 global indicator
    // array. d_eq2_norm is chunk-indexed (ii); write it to
    // d_eq2_norm_global at the atom's LAMMPS local index i=d_ilist(ii+chunk_offset)
    // so the round-2 forward_comm (after all L2 chunks) can fill the ghost rows.
    // Layout matches: (n_out, M) == (I2_n_funcs, I2_n_out). Mirrors Stage 14b. ----
    {
      auto eq2n = d_eq2_norm;
      auto Ig = d_eq2_norm_global;
      auto il = d_ilist;
      const int co = chunk_offset;
      const int nf = I2_n_funcs, no = I2_n_out, cs = chunk_size;
      Kokkos::parallel_for("ScatterEq2Norm",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nf, no}),
        KOKKOS_LAMBDA(const int ii, const int fdim, const int k) {
          Ig(il(ii + co), fdim, k) = eq2n(ii, fdim, k);
        });
    }

    // ---- UQ basis-RP (Phase 2): accumulate the rho2 (L2) scalar basis into d_uq_z
    // (d_A2_red/d_A2_2_red/d_A2_3_red/d_A2_4_red are current here). ----
    if (any_uq) {
      // Team-per-atom: lanes split the rho2 B·R projection over rp_dim (register acc).
      auto probe = Kokkos::TeamPolicy<DeviceType, TagComputeUQ_rho2>(chunk_size, 1, 1);
      int ts = probe.team_size_max(*this, Kokkos::ParallelForTag());
      if (ts > 128) ts = 128; if (ts < 1) ts = 1;
      Kokkos::parallel_for("ComputeUQ_rho2",
          Kokkos::TeamPolicy<DeviceType, TagComputeUQ_rho2>(chunk_size, ts, 1), *this);
    }

    Kokkos::fence();
    chunk_offset += chunk_size;
  }

  // ============ PHASE 2.5: forward-comm eq2_norm to ghost atoms ====
  // Every owned atom's eq2_norm now lives in d_eq2_norm_global; forward_comm
  // copies owned rows into their ghost images so the A3 kernel can read the
  // neighbor indicator at ghost ind_j. comm_stage=2 selects the round-2 buffer +
  // parity-doubled dims (I2_n_funcs*I2_n_out=3200/atom) in the pack/unpack (host
  // AND kokkos-native) paths. Mirrors the round-1 forward_comm block above.
  if (in.ex) in.ex->forward(1);   // round 2: eq2_norm rows of d_eq2_norm_global

  // ============ PHASE 3: Layer-3 forward (chunked) — A3 chain (TERMINAL) ======
  // The per-chunk geometry arrays hold the LAST L2 chunk's data, so each L3
  // chunk must recompute geometry + the A3 radial MLP before assembling A3.
  // A3's indicator is eq2_norm (in d_eq2_norm_global). L3 is TERMINAL: the
  // A3_red..rho3 chain mirrors L2's A2_red..rho2, but ends at rho3/rho3_norm —
  // there is NO eq3.
  // zero the L3 message adjoint d_eq2_norm_adj ONCE, BEFORE the P3
  // loop. It is an [nall] buffer that compute_A3_spbf_bwd atomic_add-scatters
  // across ALL chunks; zeroing it per-chunk (its old spot, inside the loop)
  // would wipe earlier chunks' contributions under multi-chunk.
  if (!do_energy_only) Kokkos::deep_copy(d_eq2_norm_adj.base(), NNScalar(0.0));
  chunk_size = std::min(chunk_lim, inum);
  chunk_offset = 0;
  while (chunk_offset < inum) {
    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

    int team_size = 1;
    int vector_length = 1;
    if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
      team_size = 32;

    // ---- L3 Stage 0: recompute neighbor data for this chunk ----
    {
      team_size = fit_team_size<DeviceType, TagComputeNeigh>(*this, chunk_size, team_size, vector_length);
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>(chunk_size, team_size, vector_length);
      Kokkos::parallel_for("L3_ComputeNeigh", policy, *this);
    }

    // ---- L3 Stage 0b: ComputeYBond ----
    compute_Y_bond_chunk();

    // ---- L3 Stage 0c: ComputeRadialBasis ----
    {
      int ts = team_size;
      ts = fit_team_size<DeviceType, TagComputeRadialBasis>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>(
          ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
      Kokkos::parallel_for("L3_ComputeRadialBasis", policy, *this);
    }

    // ---- L3 Stage 1: A3 radial MLP (spbf idx_spbf_A3). Run the DERIV
    // variant here instead of the plain forward — it reproduces the forward's
    // d_R1_nl bit-identically AND fills d_DR1_nl/d_denv, which Stage 15's force
    // path needs; nothing between Stage 1 and Stage 15 writes those buffers
    // (SPBFFused/reduce/cp_l/fc only READ d_R1_nl), so the Stage-15 radial
    // recompute is dropped. Saves one full radial-MLP forward per eval. ----
    // (fuse_radial: the fast SPBF forward and backward read the tables themselves)
    radial_mlp_spbf = idx_spbf_A3;
    if (!(fuse_radial && spline_on && spbf_rm_ok(idx_spbf_A3, idx_reduce_A3_red))) compute_mlp_radial_deriv();

    // ---- L3 Stage 2: ComputeA3 — equiv-indicator SPBF basis (indicator=eq2_norm) ----
    // fused team+shared kernel (indicator = d_eq2_norm_global, out = d_A3).
    // ---- L3 Stage 3: FunctionReduceN A3_red = reduce_n({"A3": A3}). L3's
    // downstream products/FCs are ALL built on this reduced A3_red (like L2). ----
#ifdef GRACE_KK_GEMM
    if (spbf_rm_ok(idx_spbf_A3, idx_reduce_A3_red)) {
      spbf_fwd_fast(idx_spbf_A3, d_eq2_norm_global, d_A3, true);
      compute_reduceN_grouped_rm(idx_reduce_A3_red, d_A3_red, false);
    } else
#endif
    {
      compute_spbf_equiv(idx_spbf_A3, d_eq2_norm_global, d_A3);
      t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
      inputs[0] = d_A3;   // matches reduce["A3_red"].instr[0].name == "A3" (checked in setup())
      compute_reduceN(d_reduce[idx_reduce_A3_red], inputs, d_A3_red);
    }

    // ---- L3 Stage 4: cp_l product A3_2 = cp_l(A3_red, A3_red) ----
    compute_cp_l(d_prod[idx_prod_A3_2], d_A3_red, d_A3_red, d_A3_2);

    // ---- L3 Stage 5: FunctionReduceN A3_2_red = reduce_n({"A3_2": A3_2}) ----
    {
      t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
      inputs[0] = d_A3_2;   // matches reduce["A3_2_red"].instr[0].name == "A3_2"
      compute_reduceN(d_reduce[idx_reduce_A3_2_red], inputs, d_A3_2_red);
    }

    // ---- L3 Stage 6-7: A3_2a = fc_right2left(A3_2_red, A3_red); A3_3 = cp_l(A3_2a, A3_red) (Lmax=0; fc folded
    // into the projection with fc_fold) ----
    if (fold_on[2]) {
      compute_cp_l_fold(d_prod[idx_prod_A3_3], d_A3_2_red, d_A3_red, d_A3_red, d_A3_3);
    } else {
      compute_fc(d_fc[idx_fc_A3_2a], d_A3_2_red, d_A3_red, d_A3_2a);
      compute_cp_l(d_prod[idx_prod_A3_3], d_A3_2a, d_A3_red, d_A3_3);
    }

    // ---- L3 Stage 8: FunctionReduceN A3_3_red = reduce_n({"A3_3": A3_3}) ----
    {
      t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
      inputs[0] = d_A3_3;   // matches reduce["A3_3_red"].instr[0].name == "A3_3"
      compute_reduceN(d_reduce[idx_reduce_A3_3_red], inputs, d_A3_3_red);
    }

    // ---- L3 Stage 9-10: A3_2b = fc_right2left(A3_2_red, A3_red); A3_4 = cp_l(A3_2b, A3_2b) (Lmax=0; fc folded
    // with fc_fold) ----
    if (fold_on[2]) {
      compute_cp_l_fold(d_prod[idx_prod_A3_4], d_A3_2_red, d_A3_red, t_nn_3d(), d_A3_4);
    } else {
      compute_fc(d_fc[idx_fc_A3_2b], d_A3_2_red, d_A3_red, d_A3_2b);
      compute_cp_l(d_prod[idx_prod_A3_4], d_A3_2b, d_A3_2b, d_A3_4);
    }

    // ---- L3 Stage 11: FunctionReduceN A3_4_red = reduce_n({"A3_4": A3_4}) ----
    {
      t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
      inputs[0] = d_A3_4;   // matches reduce["A3_4_red"].instr[0].name == "A3_4"
      compute_reduceN(d_reduce[idx_reduce_A3_4_red], inputs, d_A3_4_red);
    }

    // ---- L3 Stage 12: FunctionReduceN rho3 = reduce_n({A3_red,A3_2_red,A3_3_red,A3_4_red})
    // (multi-instruction + elem-dependent + only_invar). inputs[j] matched to
    // instruction j's NAME via rho3_input_role (resolved in setup()). There is
    // NO eq3 — rho3 is the ONLY terminal reduce of the L3 collectors. ----
    {
      t_nn_3d l3_views[4] = {d_A3_red, d_A3_2_red, d_A3_3_red, d_A3_4_red};
      t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR];
      const int n_instr = d_reduce[idx_reduce_rho3].n_instr;
      for (int j = 0; j < n_instr; j++) inputs[j] = l3_views[rho3_input_role[j]];
      compute_reduceN(d_reduce[idx_reduce_rho3], inputs, d_rho3);
    }

    // ---- UQ basis-RP (Phase 3, final): accumulate the rho3 (L3) scalar basis into
    // d_uq_z (d_A3_red/d_A3_2_red/d_A3_3_red/d_A3_4_red are current here), then
    // L2-normalize, append density channels [full, rho1, rho2, rho3], and run the
    // GMM (nearest centroid, Mahalanobis sigma, gamma). Forward only. Placed here so
    // it reads the L3 forward views before the L3 backward (Stage 15) begins. ----
    if (any_uq) {
      // Team-per-atom: lanes split the rho3 projection add over rp_dim, the cluster
      // search over the feature dim, and the D×D Mahalanobis over its quadratic form.
      const int scratch_bytes = (2 * uq_D + 8) * (int)sizeof(double);
      auto probe = Kokkos::TeamPolicy<DeviceType, TagComputeUQ>(chunk_size, 1, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
      int ts = probe.team_size_max(*this, Kokkos::ParallelForTag());
      if (ts > 128) ts = 128; if (ts < 1) ts = 1;
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeUQ>(chunk_size, ts, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
      Kokkos::parallel_for("ComputeUQ", policy, *this);
    }

    // ---- L3 Stage 13: InvariantLayerRMSNorm rho3_norm = invariant_rms_norm(rho3)
    // (FULL branch, same as rho2_norm — data-driven off scale_len). END of the
    // forward feature chain before readout. ----
    compute_invariant_rms_norm(d_rmsnorm[idx_rmsnorm_rho3_norm], d_rho3, d_rho3_norm);

    // ---- L3 Stage 14: Readout — 3-density sum -> energy MLP
    // (31->64 silu ->1) + lin skip + shift -> per-atom energy. rho3_norm is
    // read chunk-local (ii) since it was just written above for THIS chunk;
    // rho1_norm/rho2_norm are read from the natom buffers d_rho{1,2}_norm_full
    // (ii+chunk_offset) because their chunk-scratch originals
    // (d_rho1_norm/d_rho2_norm, written back in P1/P2) hold only the LAST
    // chunk under multi-chunk by the time P3 runs. ----
    {
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeMLPEnergy>(chunk_size, 1, 1);
      Kokkos::parallel_for("ComputeMLPEnergy", policy, *this);
    }

    if (eflag_global) {
      auto atom_energies = d_e_atom;
      KKA energy_partial = 0.0;
      Kokkos::parallel_reduce("ComputeEnergySum", chunk_size,
          KOKKOS_LAMBDA(const int ii, KKA& update) {
          update += atom_energies(ii);
      }, energy_partial);
      eng_vdwl += energy_partial;
    }

    if (eflag_atom) {
      auto e_chunk = d_e_atom;
      auto e_lammps = d_eatom;
      auto il = d_ilist;
      int co = chunk_offset;
      Kokkos::parallel_for("WriteEatom", Kokkos::RangePolicy<DeviceType>(0, chunk_size),
        KOKKOS_LAMBDA(const int ii) {
          e_lammps(il[ii + co]) += e_chunk(ii);
        });
    }

    // ---- L3 Stage 15: readout + Layer-3 backward (adjoints -> A3
    // basis + the L3 message adjoint d_eq2_norm_adj). Skipped in the energy-only
    // fast path (forces not requested); otherwise runs per chunk, reusing this
    // chunk's forward buffers. Zero every ACCUMULATING adjoint buffer, then walk
    // the L3 DAG in reverse topological order, ACCUMULATING (+=) where a tensor
    // fans out to multiple consumers. ----
    if (!do_energy_only) {
      Kokkos::deep_copy(d_A3_red_adj, NNScalar(0.0));
      Kokkos::deep_copy(d_A3_2_red_adj, NNScalar(0.0));
      Kokkos::deep_copy(d_A3_3_red_adj, NNScalar(0.0));
      Kokkos::deep_copy(d_A3_4_red_adj, NNScalar(0.0));
      if (!fold_on[2]) Kokkos::deep_copy(d_A3_2a_adj, NNScalar(0.0));
      if (!fold_on[2]) Kokkos::deep_copy(d_A3_2b_adj, NNScalar(0.0));
      // NOTE: d_eq2_norm_adj is zeroed ONCE before the P3 loop (it
      // is [nall] and accumulated across chunks) — NOT here per chunk.

      // (a) readout bwd -> d_rho{1,2,3}_norm_adj (all EQUAL).
      readout_bwd();

      // (b) rho3_norm bwd (InvariantLayerRMSNorm, FULL branch) -> d_rho3_adj.
      compute_invariant_rms_norm_bwd(d_rmsnorm[idx_rmsnorm_rho3_norm],
          d_rho3, d_rho3_norm, d_rho3_norm_adj, d_rho3_adj);

      // (c) rho3 reduce bwd (only_invar, elem-dep, 4 instr) -> SEEDS the 4 *_red adj.
      {
        t_nn_3d l3_adj[4] = {d_A3_red_adj, d_A3_2_red_adj, d_A3_3_red_adj, d_A3_4_red_adj};
        t_nn_3d adj_in[GRACE3L_MAX_REDUCE_INSTR];
        const int n_instr = d_reduce[idx_reduce_rho3].n_instr;
        for (int j = 0; j < n_instr; j++) adj_in[j] = l3_adj[rho3_input_role[j]];
        compute_reduceN_bwd(d_reduce[idx_reduce_rho3], d_rho3_adj, adj_in);
      }

      // (d) A3_4_red -> A3_4.
      compute_reduceN_bwd_set(d_reduce[idx_reduce_A3_4_red], d_A3_4_red_adj, d_A3_4_adj);   // pooled, single writer
      // (e) A3_4 = cp_l(A3_2b, A3_2b) self-product -> d_A3_2b_adj (both operands).
      if (fold_on[2]) {    // fc folded: straight into A3_2_red_adj / A3_red_adj
        compute_cp_l_fold_bwd(d_prod[idx_prod_A3_4], d_A3_2_red, d_A3_red, t_nn_3d(), d_A3_4_adj,
                              d_A3_2_red_adj, d_A3_red_adj, t_nn_3d());
      } else {
        compute_cp_l_bwd(d_prod[idx_prod_A3_4], d_A3_2b, d_A3_2b, d_A3_4_adj,
                         d_A3_2b_adj, d_A3_2b_adj);
        // (f) A3_2b = fc(A3_2_red, A3_red) -> d_A3_2_red_adj +=, d_A3_red_adj +=.
        compute_fc_bwd(d_fc[idx_fc_A3_2b], d_A3_2b_adj, d_A3_2_red_adj, d_A3_red_adj);
      }
      // (g) A3_3_red -> A3_3.
      compute_reduceN_bwd_set(d_reduce[idx_reduce_A3_3_red], d_A3_3_red_adj, d_A3_3_adj);   // pooled, single writer
      // (h) A3_3 = cp_l(A3_2a, A3_red) -> d_A3_2a_adj +=, d_A3_red_adj +=.
      if (fold_on[2]) {
        compute_cp_l_fold_bwd(d_prod[idx_prod_A3_3], d_A3_2_red, d_A3_red, d_A3_red, d_A3_3_adj,
                              d_A3_2_red_adj, d_A3_red_adj, d_A3_red_adj);
      } else {
        compute_cp_l_bwd(d_prod[idx_prod_A3_3], d_A3_2a, d_A3_red, d_A3_3_adj,
                         d_A3_2a_adj, d_A3_red_adj);
        // (i) A3_2a = fc(A3_2_red, A3_red) -> d_A3_2_red_adj +=, d_A3_red_adj +=.
        compute_fc_bwd(d_fc[idx_fc_A3_2a], d_A3_2a_adj, d_A3_2_red_adj, d_A3_red_adj);
      }
      // (j) A3_2_red -> A3_2  (d_A3_2_red_adj is now COMPLETE).
      compute_reduceN_bwd_set(d_reduce[idx_reduce_A3_2_red], d_A3_2_red_adj, d_A3_2_adj);   // pooled, single writer
      // (k) A3_2 = cp_l(A3_red, A3_red) self-product -> d_A3_red_adj += (both operands).
      compute_cp_l_bwd(d_prod[idx_prod_A3_2], d_A3_red, d_A3_red, d_A3_2_adj,
                       d_A3_red_adj, d_A3_red_adj);
      // (l) A3_red -> A3  (d_A3_red_adj is now COMPLETE) => d_A3_adj.
      const bool rm3 = spbf_rm_ok(idx_spbf_A3, idx_reduce_A3_red);
#ifdef GRACE_KK_GEMM
      if (rm3) compute_reduceN_grouped_rm(idx_reduce_A3_red, d_A3_red_adj, true);   // -> d_spbf_T
      else
#endif
      {
        Kokkos::deep_copy(d_A3_adj, NNScalar(0.0));   // pooled
        t_nn_3d adj_in[GRACE3L_MAX_REDUCE_INSTR]; adj_in[0] = d_A3_adj;
        compute_reduceN_bwd(d_reduce[idx_reduce_A3_red], d_A3_red_adj, adj_in);
      }
      // (m) A3 equiv-SPBF bwd -> d_eq2_norm_adj (L3 message adj;
      // reverse-comm'd in PHASE 3.5, which completes the ghost contributions).
      // compute adj_prod ONCE here; A3SPBFBwd AND the later ForceEquiv(A3)
      // both read d_adj_prod (not overwritten between them in P3).
      const bool fast3 = spbf_fast_ok(idx_spbf_A3);
      if (!fast3) {
        compute_adj_prod(idx_spbf_A3, d_A3_adj);
        compute_A3_spbf_bwd();
      }

      // ---- fold the L3 geometry force into P3 (per chunk). d_A3_adj
      // is complete for THIS chunk; A3's radial + dR/dr (d_R1_nl/d_DR1_nl/d_denv)
      // are still resident from Stage 1's deriv run (nothing in Stages
      // 2-14 writes them), so no radial recompute here. Zero the per-bond force
      // buffer, run the equiv force kernel, and scatter into f[]/virial. f[]
      // (atomic_add) and virial[] (+=) accumulate across the 3 layers/phases and
      // all chunks -> identical total. ----
      Kokkos::deep_copy(d_f_ij, GeomScalar(0.0));
      if (fast3) spbf_bwd_fast(idx_spbf_A3, d_A3_adj, d_eq2_norm_global, d_eq2_norm_adj, rm3);
      else compute_force_equiv(idx_spbf_A3, d_A3_adj, /*ind_stage=*/2, d_spbf[idx_spbf_A3].n_lm_ind);
      scatter_forces_chunk();
    }

    Kokkos::fence();
    chunk_offset += chunk_size;
  }

  // ---- Energy-only fast path: everything below (PHASES 3.5/4/4.5/5) is pure
  // force computation; skip it all when forces are not requested. ----
  if (!do_energy_only) {

  // ============ PHASE 3.5: reverse-comm the L2 message adjoint ======
  // compute_A3_spbf_bwd scattered d_eq2_norm_adj into owned+ghost rows (the owned
  // PARTIAL); reverse_comm ADDS every ghost image back into its owning rank's row
  // so the OWNED rows are now COMPLETE.
  // comm_stage=2 selects the parity-doubled eq2_norm buffer + dims (I2_n_funcs*
  // I2_n_out=3200/atom) in all 4 reverse pack/unpack methods. Mirrors the round-1
  // forward_comm branch structure (device vs legacy-host fallback). Do NOT re-zero
  // d_eq2_norm_adj — the scatter partial must survive into the reverse-comm. ----
  if (in.ex) in.ex->reverse(1);

  // ============ PHASE 4: Layer-2 backward (CHUNKED) ============
  // Each chunk RE-RUNS the L2 forward body (l2_forward_chunk) to repopulate the
  // chunk-scratch L2 forward intermediates (d_A2*, d_eq2, d_rho2, ...) + geometry
  // + A2's radial (d_R1_nl) for THIS chunk (correct because d_I_global is already
  // ghost-complete), gathers the reverse-comm-completed seeds, walks the L2 DAG
  // -> d_A2_adj -> A2-SPBF bwd -> d_grad_I_global (= d_eq1_norm_adj,
  // owned partial), then FOLDS the L2 geometry force into f[]/virial.
  chunk_size = std::min(chunk_lim, inum);
  chunk_offset = 0;
  // Perf (single-chunk fast path): when chunksize >= inum there is exactly ONE
  // chunk, so the L2 forward chunk-scratch (d_A2*, d_eq2, d_rho2, d_rho2_norm) is
  // still RESIDENT from PHASE 2 — PHASE 3 (L3) writes only L3-specific buffers and
  // the shared radial scratch d_R1_nl. So the L2 forward RECOMPUTE is pure overhead
  // here; skip it and restore only d_R1_nl/d_DR1_nl/d_denv (the L3-clobbered shared
  // scratch) via the reordered compute_mlp_radial_deriv below. Multi-chunk (a later
  // chunk overwrites the earlier chunk's scratch) still recomputes.
  const bool single_chunk = (chunk_size >= inum);
  // zero the round-1 message adjoint ONCE before the loop ([nall]
  // buffer, atomic_add-scattered by compute_A2_spbf_bwd across ALL chunks).
  Kokkos::deep_copy(d_grad_I_global.base(), NNScalar(0.0));
  while (chunk_offset < inum) {
    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

    // (i) recompute the L2 forward intermediates + geometry + A2 radial for THIS
    // chunk (leaves d_R1_nl = A2's radial; and rho2_norm bwd reads this chunk's
    // d_rho2/d_rho2_norm). SKIPPED for single_chunk — buffers already resident.
    if (!single_chunk) {
      if (fwd_stored) { select_fwd_store(2, chunk_offset / std::min(chunk_lim, inum)); geometry_chunk(); }
      else l2_forward_chunk();
    }

    // (ii) Gather the reverse-comm-completed L2 message adjoint from LAMMPS-local
    // rows (d_eq2_norm_adj[d_ilist(ii+co)]) into chunk-local rows (comm'd buffer
    // -> keyed by the LAMMPS local index).
    {
      auto src = d_eq2_norm_adj;
      auto dst = d_eq2_norm_adj_local;
      auto il = d_ilist;
      const int co = chunk_offset;
      const int nf = I2_n_funcs, no = I2_n_out, cs = chunk_size;
      Kokkos::parallel_for("GatherEq2NormAdj",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nf, no}),
        KOKKOS_LAMBDA(const int ii, const int f, const int k) {
          dst(ii, f, k) = src(il(ii + co), f, k);
        });
    }
    // Gather the natom rho2 readout-adjoint seed (dense list-order ii+co; NOT
    // d_ilist — this buffer is local-only) into the chunk-local _local buffer
    // that the shared TagRMSNormBwd reads.
    {
      auto src = d_rho2_norm_adj;
      auto dst = d_rho2_norm_adj_local;
      const int co = chunk_offset;
      const int cs = chunk_size;
      const int nc = (int) dst.extent(1), nl = (int) dst.extent(2);
      Kokkos::parallel_for("GatherRho2NormAdj",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nc, nl}),
        KOKKOS_LAMBDA(const int ii, const int c, const int lm) {
          dst(ii, c, lm) = src(ii + co, c, lm);
        });
    }

    // (iii) Zero every ACCUMULATING adjoint buffer (the 4 *_red seeds, the
    // product/FC adjoints, d_A2_adj written by reduceN_bwd +=). d_eq2_adj /
    // d_rho2_adj are ASSIGNED (single consumer). d_grad_I_global is zeroed ONCE
    // before the loop (above), NOT here.
    Kokkos::deep_copy(d_A2_red_adj, NNScalar(0.0));
    Kokkos::deep_copy(d_A2_2_red_adj, NNScalar(0.0));
    Kokkos::deep_copy(d_A2_3_red_adj, NNScalar(0.0));
    Kokkos::deep_copy(d_A2_4_red_adj, NNScalar(0.0));
    if (!fold_on[1]) Kokkos::deep_copy(d_A2_2a_adj, NNScalar(0.0));
    if (!fold_on[1]) Kokkos::deep_copy(d_A2_2b_adj, NNScalar(0.0));

    // (iv) L2 backward body.
    // (a) eq2_norm bwd (EquivariantRMSNorm VJP) -> d_eq2_adj (ASSIGN).
    compute_equiv_rms_norm_bwd(d_eqnorm[idx_eqnorm_eq2_norm], d_eq2,
        d_eq2_norm_adj_local, d_eq2_adj);
    // (b) eq2 reduce bwd -> SEED the 4 *_red adj (by eq2_input_role, ACCUMULATE).
    {
      t_nn_3d l2_adj[4] = {d_A2_red_adj, d_A2_2_red_adj, d_A2_3_red_adj, d_A2_4_red_adj};
      t_nn_3d adj_in[GRACE3L_MAX_REDUCE_INSTR];
      const int n_instr = d_reduce[idx_reduce_eq2].n_instr;
      for (int j = 0; j < n_instr; j++) adj_in[j] = l2_adj[eq2_input_role[j]];
      compute_reduceN_bwd(d_reduce[idx_reduce_eq2], d_eq2_adj, adj_in);
    }
    // (c) rho2_norm bwd (InvariantLayerRMSNorm, FULL) -> d_rho2_adj (ASSIGN).
    //     seed = d_rho2_norm_adj_local (gathered from the natom d_rho2_norm_adj).
    compute_invariant_rms_norm_bwd(d_rmsnorm[idx_rmsnorm_rho2_norm],
        d_rho2, d_rho2_norm, d_rho2_norm_adj_local, d_rho2_adj);
    // (d) rho2 reduce bwd -> ACCUMULATE into the SAME 4 *_red adj (by rho2_input_role).
    {
      t_nn_3d l2_adj[4] = {d_A2_red_adj, d_A2_2_red_adj, d_A2_3_red_adj, d_A2_4_red_adj};
      t_nn_3d adj_in[GRACE3L_MAX_REDUCE_INSTR];
      const int n_instr = d_reduce[idx_reduce_rho2].n_instr;
      for (int j = 0; j < n_instr; j++) adj_in[j] = l2_adj[rho2_input_role[j]];
      compute_reduceN_bwd(d_reduce[idx_reduce_rho2], d_rho2_adj, adj_in);
    }

    // ---- L2 DAG walk (reverse topological; mirrors the L3 walk exactly) ----
    // (e) A2_4_red -> A2_4.
    compute_reduceN_bwd_set(d_reduce[idx_reduce_A2_4_red], d_A2_4_red_adj, d_A2_4_adj);   // pooled, single writer
    // (f) A2_4 = cp_l(A2_2b, A2_2b) self-product -> d_A2_2b_adj (both operands).
    if (fold_on[1]) {    // fc folded: straight into A2_2_red_adj / A2_red_adj
      compute_cp_l_fold_bwd(d_prod[idx_prod_A2_4], d_A2_2_red, d_A2_red, t_nn_3d(), d_A2_4_adj,
                            d_A2_2_red_adj, d_A2_red_adj, t_nn_3d());
    } else {
      compute_cp_l_bwd(d_prod[idx_prod_A2_4], d_A2_2b, d_A2_2b, d_A2_4_adj,
                       d_A2_2b_adj, d_A2_2b_adj);
      // (g) A2_2b = fc(A2_2_red, A2_red) -> d_A2_2_red_adj +=, d_A2_red_adj +=.
      compute_fc_bwd(d_fc[idx_fc_A2_2b], d_A2_2b_adj, d_A2_2_red_adj, d_A2_red_adj);
    }
    // (h) A2_3_red -> A2_3.
    compute_reduceN_bwd_set(d_reduce[idx_reduce_A2_3_red], d_A2_3_red_adj, d_A2_3_adj);   // pooled, single writer
    // (i) A2_3 = cp_l(A2_2a, A2_red) -> d_A2_2a_adj +=, d_A2_red_adj +=.
    if (fold_on[1]) {
      compute_cp_l_fold_bwd(d_prod[idx_prod_A2_3], d_A2_2_red, d_A2_red, d_A2_red, d_A2_3_adj,
                            d_A2_2_red_adj, d_A2_red_adj, d_A2_red_adj);
    } else {
      compute_cp_l_bwd(d_prod[idx_prod_A2_3], d_A2_2a, d_A2_red, d_A2_3_adj,
                       d_A2_2a_adj, d_A2_red_adj);
      // (j) A2_2a = fc(A2_2_red, A2_red) -> d_A2_2_red_adj +=, d_A2_red_adj +=.
      compute_fc_bwd(d_fc[idx_fc_A2_2a], d_A2_2a_adj, d_A2_2_red_adj, d_A2_red_adj);
    }
    // (k) A2_2_red -> A2_2  (d_A2_2_red_adj now COMPLETE: eq2+rho2 seeds + fc_2b/2a left).
    compute_reduceN_bwd_set(d_reduce[idx_reduce_A2_2_red], d_A2_2_red_adj, d_A2_2_adj);   // pooled, single writer
    // (l) A2_2 = cp_l(A2_red, A2_red) self-product -> d_A2_red_adj += (both operands).
    compute_cp_l_bwd(d_prod[idx_prod_A2_2], d_A2_red, d_A2_red, d_A2_2_adj,
                     d_A2_red_adj, d_A2_red_adj);
    // (m) A2_red -> A2  (d_A2_red_adj now COMPLETE) => d_A2_adj.
    const bool rm2 = spbf_rm_ok(idx_spbf_A2, idx_reduce_A2_red);
#ifdef GRACE_KK_GEMM
    if (rm2) compute_reduceN_grouped_rm(idx_reduce_A2_red, d_A2_red_adj, true);   // -> d_spbf_T
    else
#endif
    {
      Kokkos::deep_copy(d_A2_adj, NNScalar(0.0));   // pooled
      t_nn_3d adj_in[GRACE3L_MAX_REDUCE_INSTR]; adj_in[0] = d_A2_adj;
      compute_reduceN_bwd(d_reduce[idx_reduce_A2_red], d_A2_red_adj, adj_in);
    }

    // (n) Restore A2's radial d_R1_nl/d_DR1_nl (+ d_denv) for THIS chunk.
    // compute_mlp_radial_deriv recomputes the radial straight from geometry
    // (d_rnorms), so it repairs the shared d_R1_nl that L3 clobbered — the ONLY
    // thing the single_chunk path needs restored. Reordered BEFORE
    // compute_A2_spbf_bwd (which reads d_R1_nl = A2's radial) and
    // compute_force_equiv. (Multi-chunk: l2_forward_chunk already set d_R1_nl = A2,
    // so this recompute is identical and the reorder is a no-op there.)
    // (fuse_radial: the fast SPBF backward reads the tables itself)
    radial_mlp_spbf = idx_spbf_A2;
    if (!(fuse_radial && spline_on && spbf_fast_ok(idx_spbf_A2))) compute_mlp_radial_deriv();

    // (o) A2 equiv-SPBF bwd -> d_grad_I_global (= d_eq1_norm_adj, owned partial).
    // compute adj_prod ONCE here; A2SPBFBwd AND the later ForceEquiv(A2)
    // both read d_adj_prod (not overwritten between them in P4).
    // (v) fold the L2 geometry force into f[]/virial for THIS chunk.
    Kokkos::deep_copy(d_f_ij, GeomScalar(0.0));
    if (spbf_fast_ok(idx_spbf_A2)) {
      spbf_bwd_fast(idx_spbf_A2, d_A2_adj, d_I_global, d_grad_I_global, rm2);
    } else {
      compute_adj_prod(idx_spbf_A2, d_A2_adj);
      compute_A2_spbf_bwd();
      compute_force_equiv(idx_spbf_A2, d_A2_adj, /*ind_stage=*/1, d_spbf[idx_spbf_A2].n_lm_ind);
    }
    scatter_forces_chunk();

    Kokkos::fence();
    chunk_offset += chunk_size;
  }

  // ============ PHASE 4.5: reverse-comm the L1 message adjoint ======
  // compute_A2_spbf_bwd scattered d_grad_I_global (= d_eq1_norm_adj) into owned+
  // ghost rows (owned PARTIAL); reverse_comm (comm_stage=1 -> the round-1 grad_I
  // branch in all 4 reverse pack/unpack methods) ADDS every ghost image back into
  // its owning rank's row, so the OWNED rows are COMPLETE.
  // Do NOT re-zero d_grad_I_global — the scatter partial must survive. ----
  if (in.ex) in.ex->reverse(0);

  // ============ PHASE 5: Layer-1 backward (CHUNKED) ============
  // Each chunk RE-RUNS the L1 forward body (l1_forward_chunk) to repopulate the
  // chunk-scratch L1 forward buffers (d_A1, d_A1_2, ..., d_eq1, d_rho1,
  // d_rho1_norm) + geometry + A1's radial (d_R1_nl) for THIS chunk. UNLIKE L2/L3,
  // the L1 chain consumes RAW A1 (there is no A1_red), so d_A1_adj is the DIRECT
  // fan-out target. Dual-seed the 4 l1_input adjoints from BOTH eq1 (equiv) and
  // rho1 (invariant), walk the L1 DAG -> d_A1_adj, then FOLD the L1
  // geometry force into f[]/virial.
  chunk_size = std::min(chunk_lim, inum);
  chunk_offset = 0;
  // Perf (single-chunk fast path): the L1 forward chunk-scratch (d_A1*, d_eq1,
  // d_rho1, d_rho1_norm) is still RESIDENT from PHASE 1 — nothing between P1 and
  // here writes the L1-specific buffers (P2/P3/P4 use L2/L3 buffers + the shared
  // radial scratch). Skip the L1 forward recompute; compute_mlp_radial_deriv below
  // (already before compute_force_L1) restores the shared d_R1_nl/d_DR1_nl/d_denv.
  // Reuses `single_chunk` from PHASE 4 (same condition: chunksize >= inum).
  while (chunk_offset < inum) {
    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

    // (i) recompute the L1 forward intermediates + geometry + A1 radial for THIS
    // chunk. SKIPPED for single_chunk — buffers already resident from PHASE 1.
    if (!single_chunk) {
      if (fwd_stored) { select_fwd_store(1, chunk_offset / std::min(chunk_lim, inum)); geometry_chunk(); }
      else l1_forward_chunk();
    }

    // (ii) Gather the reverse-comm-completed round-1 message adjoint from
    // LAMMPS-local rows (d_grad_I_global[d_ilist(ii+co)]) into chunk-local rows.
    {
      auto src = d_grad_I_global;
      auto dst = d_eq1_norm_adj_local;
      auto il = d_ilist;
      const int co = chunk_offset;
      const int nf = I_n_funcs, no = I_n_out, cs = chunk_size;
      Kokkos::parallel_for("GatherEq1NormAdj",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nf, no}),
        KOKKOS_LAMBDA(const int ii, const int f, const int kk) {
          dst(ii, f, kk) = src(il(ii + co), f, kk);
        });
    }
    // Gather the natom rho1 readout-adjoint seed (dense list-order ii+co; NOT
    // d_ilist) into the chunk-local _local buffer the shared TagRMSNormBwd reads.
    {
      auto src = d_rho1_norm_adj;
      auto dst = d_rho1_norm_adj_local;
      const int co = chunk_offset;
      const int cs = chunk_size;
      const int nc = (int) dst.extent(1), nl = (int) dst.extent(2);
      Kokkos::parallel_for("GatherRho1NormAdj",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nc, nl}),
        KOKKOS_LAMBDA(const int ii, const int c, const int lm) {
          dst(ii, c, lm) = src(ii + co, c, lm);
        });
    }

    // Zero every ACCUMULATING L1 adjoint buffer (bwd routines +=). d_eq1_adj /
    // d_rho1_adj are ASSIGNED by the rms-norm bwds (single consumer).
    Kokkos::deep_copy(d_A1_adj, NNScalar(0.0));
    Kokkos::deep_copy(d_A1_2_red_adj, NNScalar(0.0));
    if (!fold_on[0]) Kokkos::deep_copy(d_A1_2a_adj, NNScalar(0.0));
    Kokkos::deep_copy(d_A1_3_red_adj, NNScalar(0.0));
    if (!fold_on[0]) Kokkos::deep_copy(d_A1_2b_adj, NNScalar(0.0));
    Kokkos::deep_copy(d_A1_4_red_adj, NNScalar(0.0));

    // (1) eq1_norm bwd (EquivariantRMSNorm VJP) -> d_eq1_adj (ASSIGN).
    compute_equiv_rms_norm_bwd(d_eqnorm[idx_eqnorm_eq1_norm], d_eq1,
        d_eq1_norm_adj_local, d_eq1_adj);
    // (2) eq1 reduce bwd -> SEED the 4 l1_input adj (by eq1_input_role; role 0 = raw A1).
    {
      t_nn_3d l1_adj[4] = {d_A1_adj, d_A1_2_red_adj, d_A1_3_red_adj, d_A1_4_red_adj};
      t_nn_3d adj_in[GRACE3L_MAX_REDUCE_INSTR];
      const int n_instr = d_reduce[idx_reduce_eq1].n_instr;
      for (int j = 0; j < n_instr; j++) adj_in[j] = l1_adj[eq1_input_role[j]];
      compute_reduceN_bwd(d_reduce[idx_reduce_eq1], d_eq1_adj, adj_in);
    }
    // (3) rho1_norm bwd (InvariantLayerRMSNorm, only_nonlin) -> d_rho1_adj (ASSIGN).
    //     seed = d_rho1_norm_adj_local (gathered from the natom d_rho1_norm_adj).
    compute_invariant_rms_norm_bwd(d_rmsnorm[idx_rmsnorm_rho1_norm],
        d_rho1, d_rho1_norm, d_rho1_norm_adj_local, d_rho1_adj);
    // (4) rho1 reduce bwd -> ACCUMULATE into the SAME 4 (by rho1_input_role).
    {
      t_nn_3d l1_adj[4] = {d_A1_adj, d_A1_2_red_adj, d_A1_3_red_adj, d_A1_4_red_adj};
      t_nn_3d adj_in[GRACE3L_MAX_REDUCE_INSTR];
      const int n_instr = d_reduce[idx_reduce_rho1].n_instr;
      for (int j = 0; j < n_instr; j++) adj_in[j] = l1_adj[rho1_input_role[j]];
      compute_reduceN_bwd(d_reduce[idx_reduce_rho1], d_rho1_adj, adj_in);
    }

    // ---- L1 DAG walk (reverse topological) ----
    // (d) A1_4_red -> A1_4.
    compute_reduceN_bwd_set(d_reduce[idx_reduce_A1_4_red], d_A1_4_red_adj, d_A1_4_adj);   // pooled, single writer
    // (e) A1_4 = cp_l(A1_2b, A1_2b) self-product -> d_A1_2b_adj (both operands).
    if (fold_on[0]) {    // fc folded: straight into A1_2_red_adj / A1_adj
      compute_cp_l_fold_bwd(d_prod[idx_prod_A1_4], d_A1_2_red, d_A1, t_nn_3d(), d_A1_4_adj,
                            d_A1_2_red_adj, d_A1_adj, t_nn_3d());
    } else {
      compute_cp_l_bwd(d_prod[idx_prod_A1_4], d_A1_2b, d_A1_2b, d_A1_4_adj,
                       d_A1_2b_adj, d_A1_2b_adj);
      // (f) A1_2b = fc(A1_2_red, A1) -> d_A1_2_red_adj +=, d_A1_adj += (right = raw A1).
      compute_fc_bwd(d_fc[idx_fc_A1_2b], d_A1_2b_adj, d_A1_2_red_adj, d_A1_adj);
    }
    // (g) A1_3_red -> A1_3.
    compute_reduceN_bwd_set(d_reduce[idx_reduce_A1_3_red], d_A1_3_red_adj, d_A1_3_adj);   // pooled, single writer
    // (h) A1_3 = cp_l(A1_2a, A1) -> d_A1_2a_adj +=, d_A1_adj += (right = raw A1).
    if (fold_on[0]) {
      compute_cp_l_fold_bwd(d_prod[idx_prod_A1_3], d_A1_2_red, d_A1, d_A1, d_A1_3_adj,
                            d_A1_2_red_adj, d_A1_adj, d_A1_adj);
    } else {
      compute_cp_l_bwd(d_prod[idx_prod_A1_3], d_A1_2a, d_A1, d_A1_3_adj,
                       d_A1_2a_adj, d_A1_adj);
      // (i) A1_2a = fc(A1_2_red, A1) -> d_A1_2_red_adj +=, d_A1_adj += (right = raw A1).
      compute_fc_bwd(d_fc[idx_fc_A1_2a], d_A1_2a_adj, d_A1_2_red_adj, d_A1_adj);
    }
    // (j) A1_2_red -> A1_2  (d_A1_2_red_adj now COMPLETE: eq1+rho1 seeds + fc_2b/2a left).
    compute_reduceN_bwd_set(d_reduce[idx_reduce_A1_2_red], d_A1_2_red_adj, d_A1_2_adj);   // pooled, single writer
    // (k) A1_2 = cp_l(A1, A1) self-product -> d_A1_adj += (both operands).
    //     d_A1_adj now COMPLETE: eq1[0]+rho1[0] + fc_2b(R) + cp_l_3(R) + fc_2a(R) + cp_l_2(both).
    compute_cp_l_bwd(d_prod[0], d_A1, d_A1, d_A1_2_adj, d_A1_adj, d_A1_adj);

    // (v) fold the L1 geometry force into f[]/virial for THIS chunk. compute_force_L1
    // needs d_R1_nl/d_DR1_nl = A1's radial + dR/dr, so recompute the radial deriv
    // (l1_forward_chunk left d_R1_nl = A1's radial value, but not d_DR1_nl); with fuse_radial ForceL1F reads the
    // tables itself.
    radial_mlp_spbf = 0;
    if (!l1_radial_fused()) compute_mlp_radial_deriv();
    Kokkos::deep_copy(d_f_ij, GeomScalar(0.0));
    compute_force_L1();
    scatter_forces_chunk();

    Kokkos::fence();
    chunk_offset += chunk_size;
  }

  } // end !do_energy_only (backward PHASES 3.5/4/4.5/5)
  restore_fwd_store();
  RunOutput out;
  out.eng_vdwl = eng_vdwl;
  for (int k = 0; k < 6; k++) out.virial[k] = virial[k];
  if (couple_eq) Kokkos::deep_copy(out.dedl, d_dedl);
  check_kernel_launches<DeviceType>("GRACE-3L/KK");
  return out;
}

// ======================================================================
// chunksize auto (as grace::kk::Grace2LKK): the chunk for up to auto_inum atoms; kept bytes = the store_fwd tensors
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::choose_chunk()
{
  if (chunksize >= 0 || (chunk_auto > 0 && inum <= auto_inum && maxneigh <= auto_maxneigh)) return;
  auto_inum = inum + inum / 10;
  auto_maxneigh = maxneigh + maxneigh / 10 + 1;
  realloc_all = true;    // the buffers of the calibration size go; run() grows them for the chunk
  const auto grow_fn = [this](int a, int m) { grow(a, m); };
  for (int i = 0; i < 12; i++) st_fwd[i].clear();    // the old store out of the free memory the decision sees
  fwd_stored = 0;
  store_auto_ok = true;
  if (store_fwd < 0) {    // store_fwd auto: the forward store only when it fits and does not shrink the chunk much
    const auto ck = grace::kk::auto_chunk_keep<DeviceType>(auto_inum, auto_maxneigh, ranks_per_device, grow_fn,
                                                           [this] { return kept_bytes_per_atom(); }, log, "GRACE-3L/KK",
                                                           2048);
    chunk_auto = ck.first;
    store_auto_ok = ck.second;
  } else {
    chunk_auto = grace::kk::auto_chunk<DeviceType>(auto_inum, auto_maxneigh, ranks_per_device, grow_fn,
                                                   [this] { return kept_bytes_per_atom(); }, log, "GRACE-3L/KK");
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
double Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::kept_bytes_per_atom() const
{
  // the L1 / L2 forward tensors kept per chunk with store_fwd (st_ptr order), sized per atom by grow()
  if (store_fwd == 0) return 0.0;
  double b = 0.0;
  for (int i = 0; i < 12; i++) {
    const t_nn_3d *p = const_cast<Grace3LKK *>(this)->st_ptr(i);
    if (p->extent(0) > 0) b += (double) p->extent(1) * p->extent(2) * sizeof(NNScalar);
  }
  return b;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
double Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::memory_usage()
{
  double bytes = 0.0;
  bytes += MemKK::memory_usage(d_pool_fwd);
  bytes += MemKK::memory_usage(d_pool_adj);
  bytes += MemKK::memory_usage(d_cpl_store);
  bytes += MemKK::memory_usage(d_cpl_adjbuf);
  bytes += MemKK::memory_usage(d_R1_nl);
  bytes += MemKK::memory_usage(d_DR1_nl);
  bytes += MemKK::memory_usage(d_Y_bond);
  bytes += MemKK::memory_usage(d_dY_bond);
  bytes += MemKK::memory_usage(d_spbf_ydy);
  bytes += MemKK::memory_usage(d_radial_basis);
  bytes += MemKK::memory_usage(d_I_global);
  bytes += MemKK::memory_usage(d_grad_I_global);
  bytes += MemKK::memory_usage(d_eq2_norm_global);
  bytes += MemKK::memory_usage(d_eq2_norm_adj);
  bytes += MemKK::memory_usage(d_splR);
  for (int i = 0; i < 12; i++)
    for (const auto &v : st_fwd[i]) bytes += MemKK::memory_usage(v);
  return bytes;
}

}    // namespace kk
}    // namespace grace

// the explicit instantiation of GRACE_KK_EXTERN_3L (grace/kk/grace_3l_kk.h), in one translation unit
#define GRACE_KK_INSTANTIATE_3L(Device, Traits) template class grace::kk::Grace3LKK<Device, float, float, Traits>

// the kernels and their launchers
#include "grace/kk/grace_3l_kk_radial.h"
#include "grace/kk/grace_3l_kk_spbf.h"
#include "grace/kk/grace_3l_kk_cpl.h"
#include "grace/kk/grace_3l_kk_dense.h"

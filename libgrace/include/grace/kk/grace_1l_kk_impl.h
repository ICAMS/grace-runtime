// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Implementation of grace::kk::Grace1LKK (see grace_1l_kk.h). Included by one translation unit per consumer, which
// instantiates the engine for its Traits (LAMMPS: src/KOKKOS/pair_grace_1l_kokkos.cpp; python: _kk module).
#pragma once

#include "grace/kk/grace_1l_kk.h"

#include <Kokkos_Sort.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace grace {
namespace kk {
using namespace sph_const;

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::precompute_harmonics()
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
// copy_weights_to_device: transfer all model weights to Kokkos views
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::copy_weights_to_device()
{
  auto &m = *grace_model;

  // Helper: copy flat vector to 2D Kokkos view
  auto copy_2d = [](const std::vector<double> &src, auto &dst, int d0, int d1, const char *label) {
    grace::kk::MemKK::realloc_kokkos(dst, label, d0, d1);
    auto h = Kokkos::create_mirror_view(dst);
    for (int i = 0; i < d0; i++)
      for (int j = 0; j < d1; j++)
        h(i, j) = src[i * d1 + j];
    Kokkos::deep_copy(dst, h);
  };

  auto copy_3d = [](const std::vector<double> &src, auto &dst, int d0, int d1, int d2, const char *label) {
    grace::kk::MemKK::realloc_kokkos(dst, label, d0, d1, d2);
    auto h = Kokkos::create_mirror_view(dst);
    for (int i = 0; i < d0; i++)
      for (int j = 0; j < d1; j++)
        for (int k = 0; k < d2; k++)
          h(i, j, k) = src[(i * d1 + j) * d2 + k];
    Kokkos::deep_copy(dst, h);
  };

  auto copy_4d = [](const std::vector<double> &src, auto &dst, int d0, int d1, int d2, int d3, const char *label) {
    grace::kk::MemKK::realloc_kokkos(dst, label, d0, d1, d2, d3);
    auto h = Kokkos::create_mirror_view(dst);
    for (int i = 0; i < d0; i++)
      for (int j = 0; j < d1; j++)
        for (int k = 0; k < d2; k++)
          for (int l = 0; l < d3; l++)
            h(i, j, k, l) = src[((i * d1 + j) * d2 + k) * d3 + l];
    Kokkos::deep_copy(dst, h);
  };

  auto copy_1d_int = [](const std::vector<int> &src, auto &dst, const char *label) {
    grace::kk::MemKK::realloc_kokkos(dst, label, src.size());
    auto h = Kokkos::create_mirror_view(dst);
    for (size_t i = 0; i < src.size(); i++) h(i) = src[i];
    Kokkos::deep_copy(dst, h);
  };

  auto copy_1d_double = [](const std::vector<double> &src, auto &dst, const char *label) {
    grace::kk::MemKK::realloc_kokkos(dst, label, src.size());
    auto h = Kokkos::create_mirror_view(dst);
    for (size_t i = 0; i < src.size(); i++) h(i) = src[i];
    Kokkos::deep_copy(dst, h);
  };

  // Chemical embedding [n_elements, embedding_size]
  copy_2d(m.chem_embedding, d_chem_embed, m.n_elements, m.embedding_size, "g1l:chem_embed");

  // A indicator transform [embedding_size, nradmax]
  copy_2d(m.A_lin_transform_W, d_A_lin_W, m.embedding_size, m.nradmax, "g1l:A_lin_W");

  // Precompute z_tr[mu, n] = (A_lin_norm * inv_avg_n_neigh) * Σ_e A_lin_W[e, n] * chem_embed[mu, e]
  {
    grace::kk::MemKK::realloc_kokkos(d_z_tr, "g1l:z_tr", m.n_elements, m.nradmax);
    auto h_z_tr = Kokkos::create_mirror_view(d_z_tr);
    // in fp64, rounded once (as the CPU driver): an fp32 sum over the embedding is ~1e-6 off, the same for every bond of
    // the element, a bias of the energy where the A basis is large (compressed cells)
    const double z_norm = m.A_lin_transform_norm * m.A_inv_avg_n_neigh;
    for (int mu = 0; mu < m.n_elements; mu++)
      for (int n = 0; n < m.nradmax; n++) {
        double val = 0.0;
        for (int e = 0; e < m.embedding_size; e++)
          val += m.A_lin_transform_W[e * m.nradmax + n] * m.chem_embedding[mu * m.embedding_size + e];
        h_z_tr(mu, n) = (NNScalar) (val * z_norm);
      }
    Kokkos::deep_copy(d_z_tr, h_z_tr);
  }

  // MLP radial weights (generalized N-layer)
  {
    mlp_rad_n_layers = m.mlp_rad_n_layers;
    // Compute dims array: dims[0]=input of layer 0, dims[i+1]=output of layer i
    std::vector<int> dims(mlp_rad_n_layers + 1);
    dims[0] = m.mlp_rad_layers[0].n_in;
    for (int i = 0; i < mlp_rad_n_layers; i++)
      dims[i + 1] = m.mlp_rad_layers[i].n_out;
    mlp_rad_last_hidden_dim = dims[mlp_rad_n_layers - 1];
    mlp_rad_output_norm = (NNScalar) m.mlp_rad_layers[mlp_rad_n_layers - 1].norm;
    // Compute max hidden dim for stack array sizing (exclude output layer dim)
    mlp_rad_max_dim = 0;
    for (int i = 0; i < mlp_rad_n_layers; i++)
      if (dims[i] > mlp_rad_max_dim) mlp_rad_max_dim = dims[i];
    // Validate compile-time limits
    if (mlp_rad_n_layers > GRACE1L_MAX_MLP_LAYERS)
      GRACE_KK_ERROR("GRACE-1L: mlp_rad_n_layers ({}) exceeds GRACE1L_MAX_MLP_LAYERS ({})",
                 mlp_rad_n_layers, GRACE1L_MAX_MLP_LAYERS);
    if (mlp_rad_max_dim > GRACE1L_MAX_MLP_DIM)
      GRACE_KK_ERROR("GRACE-1L: mlp hidden dim ({}) exceeds GRACE1L_MAX_MLP_DIM ({})",
                 mlp_rad_max_dim, GRACE1L_MAX_MLP_DIM);
    if (nradmax > GRACE1L_MAX_NRADMAX)
      GRACE_KK_ERROR("GRACE-1L: nradmax ({}) exceeds GRACE1L_MAX_NRADMAX ({})",
                 nradmax, GRACE1L_MAX_NRADMAX);
    // Find max_in and max_out for weight tensor padding
    int max_in = 0, max_out = 0;
    for (int i = 0; i < mlp_rad_n_layers; i++) {
      if (m.mlp_rad_layers[i].n_in > max_in) max_in = m.mlp_rad_layers[i].n_in;
      if (m.mlp_rad_layers[i].n_out > max_out) max_out = m.mlp_rad_layers[i].n_out;
    }
    // Allocate and fill padded 3D weight tensor [n_layers, max_in, max_out]
    grace::kk::MemKK::realloc_kokkos(d_mlp_rad_W, "g1l:mlp_rad_W", mlp_rad_n_layers, max_in, max_out);
    auto h_W = Kokkos::create_mirror_view(d_mlp_rad_W);
    Kokkos::deep_copy(h_W, 0.0);
    for (int layer = 0; layer < mlp_rad_n_layers; layer++) {
      auto &L = m.mlp_rad_layers[layer];
      for (int i = 0; i < L.n_in; i++)
        for (int j = 0; j < L.n_out; j++)
          h_W(layer, i, j) = L.W[i * L.n_out + j];
    }
    Kokkos::deep_copy(d_mlp_rad_W, h_W);
    // Norms
    grace::kk::MemKK::realloc_kokkos(d_mlp_rad_norms, "g1l:mlp_rad_norms", mlp_rad_n_layers);
    auto h_norms = Kokkos::create_mirror_view(d_mlp_rad_norms);
    for (int i = 0; i < mlp_rad_n_layers; i++)
      h_norms(i) = (NNScalar) m.mlp_rad_layers[i].norm;
    Kokkos::deep_copy(d_mlp_rad_norms, h_norms);
    // Dims
    grace::kk::MemKK::realloc_kokkos(d_mlp_rad_dims, "g1l:mlp_rad_dims", mlp_rad_n_layers + 1);
    auto h_dims = Kokkos::create_mirror_view(d_mlp_rad_dims);
    for (int i = 0; i <= mlp_rad_n_layers; i++)
      h_dims(i) = dims[i];
    Kokkos::deep_copy(d_mlp_rad_dims, h_dims);

#ifdef KOKKOS_ENABLE_CUDA
    // cuBLAS radial-MLP weights: tight row-major [n_in x n_out] per layer.
    // The native padded d_mlp_rad_W layer slice is not contiguous enough for
    // SGEMM. Active only for CUDA device float instantiations; fp64/host ignore.
    if constexpr (is_cuda_v<DeviceType> && sizeof(NNScalar) == 4) {
      for (int layer = 0; layer < mlp_rad_n_layers; layer++) {
        auto &L = m.mlp_rad_layers[layer];
        d_mlp_Wg[layer] = t_nn_2d_r("g1l:mlp_Wg", L.n_in, L.n_out);
        auto h = Kokkos::create_mirror_view(d_mlp_Wg[layer]);
        for (int i = 0; i < L.n_in; i++)
          for (int j = 0; j < L.n_out; j++)
            h(i, j) = (NNScalar) L.W[(size_t) i * L.n_out + j];
        Kokkos::deep_copy(d_mlp_Wg[layer], h);
      }
    }
#endif
  }

  // FC weights
  auto copy_fc = [&](const GRACE1LModel::FCWeights &fc, const char *pfx,
                      auto &dwl, auto &dwr, auto &dwtl, auto &dwtr,
                      auto &dct, auto &dcf, auto &dnof,
                      NNScalar &nl, NNScalar &nr) {
    std::string s(pfx);
    copy_3d(fc.w_left, dwl, fc.n_out, fc.w_left.size() / (fc.n_out * fc.w_shape_left), fc.w_shape_left, (s+"_wl").c_str());
    copy_3d(fc.w_right, dwr, fc.n_out, fc.w_right.size() / (fc.n_out * fc.w_shape_right), fc.w_shape_right, (s+"_wr").c_str());
    copy_1d_int(fc.w_tile_left, dwtl, (s+"_wtl").c_str());
    copy_1d_int(fc.w_tile_right, dwtr, (s+"_wtr").c_str());
    copy_1d_int(fc.collect_to, dct, (s+"_ct").c_str());
    copy_1d_int(fc.collect_from, dcf, (s+"_cf").c_str());
    // norm_out_factor is [n_funcs, 1, 1] - flatten to 1d
    std::vector<double> nof_flat(fc.n_funcs_left);
    for (int i = 0; i < fc.n_funcs_left && i * 1 < (int)fc.norm_out_factor.size(); i++)
      nof_flat[i] = fc.norm_out_factor[i];  // stride is 1*1=1 in flattened [n,1,1]
    copy_1d_double(nof_flat, dnof, (s+"_nof").c_str());
    nl = (NNScalar) fc.norm_left;
    nr = (NNScalar) fc.norm_right;
  };

  copy_fc(m.fc_A1, "g1l:fc_A1", d_fc_A1_wl, d_fc_A1_wr, d_fc_A1_wtl, d_fc_A1_wtr,
          d_fc_A1_ct, d_fc_A1_cf, d_fc_A1_nof, fc_A1_nl, fc_A1_nr);
  copy_fc(m.fc_AA1, "g1l:fc_AA1", d_fc_AA1_wl, d_fc_AA1_wr, d_fc_AA1_wtl, d_fc_AA1_wtr,
          d_fc_AA1_ct, d_fc_AA1_cf, d_fc_AA1_nof, fc_AA1_nl, fc_AA1_nr);
  copy_fc(m.fc_AA2, "g1l:fc_AA2", d_fc_AA2_wl, d_fc_AA2_wr, d_fc_AA2_wtl, d_fc_AA2_wtr,
          d_fc_AA2_ct, d_fc_AA2_cf, d_fc_AA2_nof, fc_AA2_nl, fc_AA2_nr);

  // Build gather indices for fused FC kernels (forward: group by target, reverse: group by source)
  auto build_gather = [&](const GRACE1LModel::FCWeights &fc, int n_tgt,
                          auto &rg_start, auto &rg_src, auto &rg_tile,
                          auto &rev_start, auto &rev_src, auto &rev_tile,
                          const char *pfx) {
    int ncol = (int)fc.collect_to.size();
    int n_src_max = 0;
    for (int i = 0; i < ncol; i++)
      if (fc.collect_from[i] + 1 > n_src_max) n_src_max = fc.collect_from[i] + 1;
    std::string s(pfx);

    // Forward gather: group scatter entries by target (ct)
    {
      std::vector<int> count(n_tgt, 0);
      for (int i = 0; i < ncol; i++) count[fc.collect_to[i]]++;
      std::vector<int> start(n_tgt + 1, 0);
      for (int i = 0; i < n_tgt; i++) start[i + 1] = start[i] + count[i];
      std::vector<int> src(ncol), tile(ncol);
      std::vector<int> pos(n_tgt, 0);
      for (int i = 0; i < ncol; i++) {
        int t = fc.collect_to[i];
        int idx = start[t] + pos[t]++;
        src[idx] = fc.collect_from[i];
        tile[idx] = fc.w_tile_right[i];
      }
      copy_1d_int(start, rg_start, (s + "_rg_start").c_str());
      copy_1d_int(src, rg_src, (s + "_rg_src").c_str());
      copy_1d_int(tile, rg_tile, (s + "_rg_tile").c_str());
    }
    // Reverse gather: group scatter entries by source (cf)
    {
      std::vector<int> count(n_src_max, 0);
      for (int i = 0; i < ncol; i++) count[fc.collect_from[i]]++;
      std::vector<int> start(n_src_max + 1, 0);
      for (int i = 0; i < n_src_max; i++) start[i + 1] = start[i] + count[i];
      std::vector<int> tgt(ncol), tile(ncol);
      std::vector<int> pos(n_src_max, 0);
      for (int i = 0; i < ncol; i++) {
        int s2 = fc.collect_from[i];
        int idx = start[s2] + pos[s2]++;
        tgt[idx] = fc.collect_to[i];
        tile[idx] = fc.w_tile_right[i];
      }
      copy_1d_int(start, rev_start, (s + "_rev_start").c_str());
      copy_1d_int(tgt, rev_src, (s + "_rev_tgt").c_str());
      copy_1d_int(tile, rev_tile, (s + "_rev_tile").c_str());
    }
  };
  build_gather(m.fc_A1, n_funcs_A1,
    d_fc_A1_rg_start, d_fc_A1_rg_src, d_fc_A1_rg_tile,
    d_fc_A1_rev_start, d_fc_A1_rev_src, d_fc_A1_rev_tile, "g1l:fc_A1");
  build_gather(m.fc_AA1, n_funcs_AA,
    d_fc_AA1_rg_start, d_fc_AA1_rg_src, d_fc_AA1_rg_tile,
    d_fc_AA1_rev_start, d_fc_AA1_rev_src, d_fc_AA1_rev_tile, "g1l:fc_AA1");
  build_gather(m.fc_AA2, n_funcs_AA,
    d_fc_AA2_rg_start, d_fc_AA2_rg_src, d_fc_AA2_rg_tile,
    d_fc_AA2_rev_start, d_fc_AA2_rev_src, d_fc_AA2_rev_tile, "g1l:fc_AA2");

  // CG product metadata
  auto copy_prod = [&](const GRACE1LModel::CGProduct &prod, const char *pfx,
                        auto &dli, auto &dri, auto &dsi, auto &dcg) {
    std::string s(pfx);
    copy_1d_int(prod.left_ind, dli, (s+"_li").c_str());
    copy_1d_int(prod.right_ind, dri, (s+"_ri").c_str());
    copy_1d_int(prod.m_sum_ind, dsi, (s+"_si").c_str());
    copy_1d_double(prod.cg_coeff, dcg, (s+"_cg").c_str());
  };

  copy_prod(m.prod_AA, "g1l:prod_AA", d_prod_AA_li, d_prod_AA_ri, d_prod_AA_si, d_prod_AA_cg);
  copy_prod(m.prod_AAA, "g1l:prod_AAA", d_prod_AAA_li, d_prod_AAA_ri, d_prod_AAA_si, d_prod_AAA_cg);
  copy_prod(m.prod_AAAA, "g1l:prod_AAAA", d_prod_AAAA_li, d_prod_AAAA_ri, d_prod_AAAA_si, d_prod_AAAA_cg);

  // Build forward product gather indices (group CG terms by output function)
  auto build_prod_gather = [&](const GRACE1LModel::CGProduct &prod,
                               auto &gs, auto &gli, auto &gri, auto &gcg, const char *pfx) {
    int nt = prod.n_cg_terms, no = prod.n_output_funcs;
    std::string s(pfx);
    std::vector<int> count(no, 0);
    for (int t = 0; t < nt; t++) count[prod.m_sum_ind[t]]++;
    std::vector<int> start(no + 1, 0);
    for (int i = 0; i < no; i++) start[i + 1] = start[i] + count[i];
    std::vector<int> li(nt), ri(nt); std::vector<double> cg(nt);
    std::vector<int> pos(no, 0);
    for (int t = 0; t < nt; t++) {
      int o = prod.m_sum_ind[t];
      int idx = start[o] + pos[o]++;
      li[idx] = prod.left_ind[t];
      ri[idx] = prod.right_ind[t];
      cg[idx] = prod.cg_coeff[t];
    }
    copy_1d_int(start, gs, (s + "_gs").c_str());
    copy_1d_int(li, gli, (s + "_gli").c_str());
    copy_1d_int(ri, gri, (s + "_gri").c_str());
    copy_1d_double(cg, gcg, (s + "_gcg").c_str());
  };
  build_prod_gather(m.prod_AA, d_prod_AA_gs, d_prod_AA_gli, d_prod_AA_gri, d_prod_AA_gcg, "g1l:prod_AA");
  build_prod_gather(m.prod_AAA, d_prod_AAA_gs, d_prod_AAA_gli, d_prod_AAA_gri, d_prod_AAA_gcg, "g1l:prod_AAA");
  build_prod_gather(m.prod_AAAA, d_prod_AAAA_gs, d_prod_AAAA_gli, d_prod_AAAA_gri, d_prod_AAAA_gcg, "g1l:prod_AAAA");

  // Build reverse self-product gather: for each adjoint func f, entries where li==f or ri==f
  auto build_rprod_gather = [&](const GRACE1LModel::CGProduct &prod, int n_adj,
                                auto &gs, auto &goa, auto &gfwd, auto &gcg, const char *pfx) {
    int nt = prod.n_cg_terms;
    std::string s(pfx);
    // Count entries per adjoint function
    std::vector<int> count(n_adj, 0);
    for (int t = 0; t < nt; t++) { count[prod.left_ind[t]]++; count[prod.right_ind[t]]++; }
    std::vector<int> start(n_adj + 1, 0);
    for (int i = 0; i < n_adj; i++) start[i + 1] = start[i] + count[i];
    int total = start[n_adj];
    std::vector<int> oa(total), fwd(total); std::vector<double> cg(total);
    std::vector<int> pos(n_adj, 0);
    for (int t = 0; t < nt; t++) {
      // as left: adj[li] += cg * oa[si] * fwd[ri]
      int f = prod.left_ind[t];
      int idx = start[f] + pos[f]++;
      oa[idx] = prod.m_sum_ind[t]; fwd[idx] = prod.right_ind[t]; cg[idx] = prod.cg_coeff[t];
      // as right: adj[ri] += cg * oa[si] * fwd[li]
      f = prod.right_ind[t];
      idx = start[f] + pos[f]++;
      oa[idx] = prod.m_sum_ind[t]; fwd[idx] = prod.left_ind[t]; cg[idx] = prod.cg_coeff[t];
    }
    copy_1d_int(start, gs, (s + "_gs").c_str());
    copy_1d_int(oa, goa, (s + "_goa").c_str());
    copy_1d_int(fwd, gfwd, (s + "_gfwd").c_str());
    copy_1d_double(cg, gcg, (s + "_gcg").c_str());
  };
  build_rprod_gather(m.prod_AA, n_funcs_A1,
    d_rprod_AA_gs, d_rprod_AA_goa, d_rprod_AA_gfwd, d_rprod_AA_gcg, "g1l:rprod_AA");
  build_rprod_gather(m.prod_AAAA, n_funcs_AA,
    d_rprod_AAAA_gs, d_rprod_AAAA_goa, d_rprod_AAAA_gfwd, d_rprod_AAAA_gcg, "g1l:rprod_AAAA");

  // Build reverse gather for asymmetric AAA product (AA1⊗A1 → AAA): separate left/right CSRs
  {
    const auto &prod = m.prod_AAA;
    int nt = prod.n_cg_terms;
    // --- Left gather (adj indexed by left_ind space: [0, n_funcs_AA)) ---
    {
      std::vector<int> count(n_funcs_AA, 0);
      for (int t = 0; t < nt; t++) count[prod.left_ind[t]]++;
      std::vector<int> start(n_funcs_AA + 1, 0);
      for (int i = 0; i < n_funcs_AA; i++) start[i + 1] = start[i] + count[i];
      int total = start[n_funcs_AA];
      std::vector<int> oa(total), fwd(total); std::vector<double> cg(total);
      std::vector<int> pos(n_funcs_AA, 0);
      for (int t = 0; t < nt; t++) {
        int f = prod.left_ind[t], idx = start[f] + pos[f]++;
        oa[idx] = prod.m_sum_ind[t]; fwd[idx] = prod.right_ind[t]; cg[idx] = prod.cg_coeff[t];
      }
      copy_1d_int(start, d_rprod_AAA_left_gs,  "g1l:rprod_AAA_left_gs");
      copy_1d_int(oa,    d_rprod_AAA_left_goa, "g1l:rprod_AAA_left_goa");
      copy_1d_int(fwd,   d_rprod_AAA_left_gfwd,"g1l:rprod_AAA_left_gfwd");
      copy_1d_double(cg, d_rprod_AAA_left_gcg, "g1l:rprod_AAA_left_gcg");
    }
    // --- Right gather (adj indexed by right_ind space: [0, n_funcs_A1)) ---
    {
      std::vector<int> count(n_funcs_A1, 0);
      for (int t = 0; t < nt; t++) count[prod.right_ind[t]]++;
      std::vector<int> start(n_funcs_A1 + 1, 0);
      for (int i = 0; i < n_funcs_A1; i++) start[i + 1] = start[i] + count[i];
      int total = start[n_funcs_A1];
      std::vector<int> oa(total), fwd(total); std::vector<double> cg(total);
      std::vector<int> pos(n_funcs_A1, 0);
      for (int t = 0; t < nt; t++) {
        int f = prod.right_ind[t], idx = start[f] + pos[f]++;
        oa[idx] = prod.m_sum_ind[t]; fwd[idx] = prod.left_ind[t]; cg[idx] = prod.cg_coeff[t];
      }
      copy_1d_int(start, d_rprod_AAA_right_gs,  "g1l:rprod_AAA_right_gs");
      copy_1d_int(oa,    d_rprod_AAA_right_goa, "g1l:rprod_AAA_right_goa");
      copy_1d_int(fwd,   d_rprod_AAA_right_gfwd,"g1l:rprod_AAA_right_gfwd");
      copy_1d_double(cg, d_rprod_AAA_right_gcg, "g1l:rprod_AAA_right_gcg");
    }
  }

  // FunctionReduceN weights [n_elements, rho_n_out, n_in, w_shape]
  auto copy_reduce = [&](const GRACE1LModel::ReduceWeights &rw, const char *pfx,
                          auto &dw, NNScalar &norm, auto &dci) {
    std::string s(pfx);
    copy_4d(rw.W, dw, m.n_elements, m.rho_n_out, rw.n_in, rw.w_shape, (s+"_W").c_str());
    norm = (NNScalar) rw.norm;
    copy_1d_int(rw.collect_ind, dci, (s+"_ci").c_str());
  };

  copy_reduce(m.reduce_A, "g1l:red_A", d_reduce_A_W, reduce_A_norm, d_reduce_A_ci);
  copy_reduce(m.reduce_AA, "g1l:red_AA", d_reduce_AA_W, reduce_AA_norm, d_reduce_AA_ci);
  copy_reduce(m.reduce_AAA, "g1l:red_AAA", d_reduce_AAA_W, reduce_AAA_norm, d_reduce_AAA_ci);
  copy_reduce(m.reduce_AAAA, "g1l:red_AAAA", d_reduce_AAAA_W, reduce_AAAA_norm, d_reduce_AAAA_ci);

  // Energy MLP (generalized N-layer)
  {
    energy_n_layers = (int) m.energy_mlp_layers.size();
    std::vector<int> edims(energy_n_layers + 1);
    edims[0] = m.energy_mlp_layers[0].n_in;
    for (int i = 0; i < energy_n_layers; i++)
      edims[i + 1] = m.energy_mlp_layers[i].n_out;
    energy_max_dim = 0;
    for (int i = 0; i <= energy_n_layers; i++)
      if (edims[i] > energy_max_dim) energy_max_dim = edims[i];
    if (energy_n_layers > GRACE1L_MAX_MLP_LAYERS)
      GRACE_KK_ERROR("GRACE-1L: energy_n_layers ({}) exceeds GRACE1L_MAX_MLP_LAYERS ({})",
                 energy_n_layers, GRACE1L_MAX_MLP_LAYERS);
    if (energy_max_dim > GRACE1L_MAX_MLP_DIM)
      GRACE_KK_ERROR("GRACE-1L: energy hidden dim ({}) exceeds GRACE1L_MAX_MLP_DIM ({})",
                 energy_max_dim, GRACE1L_MAX_MLP_DIM);
    if (m.energy_mlp_activation != 0 && m.energy_mlp_activation != 1)
      GRACE_KK_ERROR("GRACE-1L/KK: energy_mlp_activation={} not supported "
                        "(only 0=silu, 1=tanh)", m.energy_mlp_activation);
    energy_activation = m.energy_mlp_activation;
    int emax_in = 0, emax_out = 0;
    for (int i = 0; i < energy_n_layers; i++) {
      if (m.energy_mlp_layers[i].n_in > emax_in) emax_in = m.energy_mlp_layers[i].n_in;
      if (m.energy_mlp_layers[i].n_out > emax_out) emax_out = m.energy_mlp_layers[i].n_out;
    }
    grace::kk::MemKK::realloc_kokkos(d_energy_W, "g1l:energy_W", energy_n_layers, emax_in, emax_out);
    auto h_eW = Kokkos::create_mirror_view(d_energy_W);
    Kokkos::deep_copy(h_eW, 0.0);
    for (int layer = 0; layer < energy_n_layers; layer++) {
      auto &L = m.energy_mlp_layers[layer];
      for (int i = 0; i < L.n_in; i++)
        for (int j = 0; j < L.n_out; j++)
          h_eW(layer, i, j) = L.W[i * L.n_out + j];
    }
    Kokkos::deep_copy(d_energy_W, h_eW);
    grace::kk::MemKK::realloc_kokkos(d_energy_norms, "g1l:energy_norms", energy_n_layers);
    auto h_enorms = Kokkos::create_mirror_view(d_energy_norms);
    for (int i = 0; i < energy_n_layers; i++)
      h_enorms(i) = (NNScalar) m.energy_mlp_layers[i].norm;
    Kokkos::deep_copy(d_energy_norms, h_enorms);
    grace::kk::MemKK::realloc_kokkos(d_energy_dims, "g1l:energy_dims", energy_n_layers + 1);
    auto h_edims = Kokkos::create_mirror_view(d_energy_dims);
    for (int i = 0; i <= energy_n_layers; i++)
      h_edims(i) = edims[i];
    Kokkos::deep_copy(d_energy_dims, h_edims);
  }

  // Shifts
  copy_1d_double(m.shift_values, d_shifts, "g1l:shifts");
  output_scale = static_cast<NNScalar>(m.output_scale);

  // Bond-specific cutoff map [nelements, nelements]
  copy_2d(m.bond_cutoff_map, d_bond_cutoff, nelements, nelements, "g1l:bond_cutoff");

  // ---- UQ / extrapolation-grade artifacts (optional; schema v6 basis-RP) ----
  if (has_uq) {
    const int E = m.uq_n_elements;
    const int K = m.uq_max_clusters;
    const int D = m.uq_feature_dim;        // = rp_dim + n_density
    const int RP = m.uq_rp_dim;            // projection width (R cols)
    if (E != m.n_elements)
      GRACE_KK_ERROR("GRACE-1L/KK UQ: uq_n_elements ({}) != model n_elements ({})",
                 E, m.n_elements);
    if (D > GRACE1L_UQ_MAX_RP_DIM)
      GRACE_KK_ERROR("GRACE-1L/KK UQ: feature_dim ({}) exceeds GRACE1L_UQ_MAX_RP_DIM ({})",
                 D, GRACE1L_UQ_MAX_RP_DIM);
    // basis-RP reads the gathered scalar-reduce products (b_inv) in the exact
    // (source, n_out, collect) order the energy reduces iterate them, so the
    // R-row count D_basis must equal the rho scalar-basis width
    // [Σ nin·n_collected per source]. 1L has only the L1 (rho) reduces.
    const int d_basis =
        (int)d_reduce_A_W.extent(2)    * (int)d_reduce_A_ci.extent(0) +
        (int)d_reduce_AA_W.extent(2)   * (int)d_reduce_AA_ci.extent(0) +
        (int)d_reduce_AAA_W.extent(2)  * (int)d_reduce_AAA_ci.extent(0) +
        (int)d_reduce_AAAA_W.extent(2) * (int)d_reduce_AAAA_ci.extent(0);
    if (d_basis != m.uq_d_basis)
      GRACE_KK_ERROR("GRACE-1L/KK UQ: scalar-basis width ({}) != uq_rp_matrix rows ({})",
                 d_basis, m.uq_d_basis);
    uq_d_basis = m.uq_d_basis;
    copy_3d(m.uq_centroids, d_uq_centroids, E, K, D, "g1l:uq_centroids");
    copy_4d(m.uq_inv_cov, d_uq_inv_cov, E, K, D, D, "g1l:uq_inv_cov");
    copy_1d_int(m.uq_n_clusters, d_uq_n_clusters, "g1l:uq_n_clusters");
    copy_2d(m.uq_interp_thresholds, d_uq_interp_thresholds, E, K, "g1l:uq_interp_thr");
    copy_2d(m.uq_rp_matrix, d_uq_rp_matrix, m.uq_d_basis, RP, "g1l:uq_rp_matrix");
#ifdef KOKKOS_ENABLE_CUDA
    uq_sgemm = std::is_same<NNScalarT, float>::value;
    if (uq_sgemm) {
      d_uq_rp_matrix_f = Kokkos::View<float*, DeviceType>("g1l:uq_rp_matrix_f", m.uq_rp_matrix.size());
      auto h = Kokkos::create_mirror_view(d_uq_rp_matrix_f);
      for (size_t q = 0; q < m.uq_rp_matrix.size(); q++) h(q) = (float) m.uq_rp_matrix[q];
      Kokkos::deep_copy(d_uq_rp_matrix_f, h);
    }
#endif
  }

#ifdef KOKKOS_ENABLE_CUDA
  // cuBLAS true-fp32 block-sparse FC forward (CUDA device + NNScalar=float only).
  // Create the handle once, bind to the SAME stream the Kokkos parallel_fors launch
  // on (=> GEMMs and the surrounding gather/scatter kernels run in issue order, no
  // race, no fences), and force true fp32 (CUBLAS_PEDANTIC_MATH disables the Ampere+
  // tf32 default — tf32 is FORBIDDEN here). The if constexpr guards the CUDA-only
  // cuda_stream() from the host instantiation; the sizeof gate skips the fp64 style.
  // The handle is needed for the fp32 FC/radial SGEMM path (NNScalar=float) AND
  // for the fp64 UQ DGEMM projection (UQ arrays are double, so the fp64 style
  // uses cuBLAS for UQ even though its FC path stays hand-written).
  if constexpr (is_cuda_v<DeviceType>) {
    if ((sizeof(NNScalar) == 4 || has_uq) && !blas_handle && blas::available()) {
      if (blas::create(&blas_handle) != blas::SUCCESS) {
#ifdef GRACE_KK_CUBLAS_DLOPEN
        blas_handle = nullptr;    // optional cuBLAS (e.g. no device memory left): the hand-written kernels
#else
        GRACE_KK_ERROR("GRACE-1L/KK: cublasCreate failed");
#endif
      }
      if (blas_handle) {
        blas_owner.reset(blas_handle, [](blas::handle_t h) { blas::destroy(h); });
        blas::set_stream(blas_handle, DeviceType());
        blas::set_pedantic_math(blas_handle);  // no tf32 (fp32 & fp64)
        if constexpr (sizeof(NNScalar) == 4) {
          mlp_cublas_selftest();  // tiny-case radial MLP SGEMM gate
          fc_cublas_selftest();   // tiny-case transpose/leading-dim gate (aborts on mismatch)
        }
      }
    }
    if constexpr (sizeof(NNScalar) == 4) {
      build_fc_cublas_op(fc_op_AA1, m.fc_AA1);
      build_fc_cublas_op(fc_op_AA2, m.fc_AA2);
    }
  }
  {
    if (blas_handle && sizeof(NNScalar) == 4) {
      log.info("[GRACE-1L/KK] radial MLP: true-fp32 cuBLAS SGEMM enabled (CUDA)\n");
      log.info("[GRACE-1L/KK] FC forward: true-fp32 cuBLAS SGEMM enabled (CUDA)\n");
    } else {
      log.info("[GRACE-1L/KK] FC forward: hand-written Kokkos kernels "
               "(cuBLAS is CUDA+fp32/Mixed only; fp64 & non-CUDA use the fallback)\n");
    }
    if (blas_handle && has_uq)
      log.info("[GRACE-1L/KK] UQ projection: fp64 cuBLAS DGEMM enabled (CUDA)\n");
  }
#endif
}

// ======================================================================
// grow: allocate per-chunk intermediate arrays
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::grow(int natom, int maxneigh_in)
{
  if (realloc_all || (int)d_A.extent(0) < natom) {
    const int nlm = (lmax + 1) * (lmax + 1);  // 25
    // Layout matches Python: [batch, n_channels, n_funcs]

    grace::kk::MemKK::realloc_kokkos(d_ncount, "g1l:ncount", natom);
    grace::kk::MemKK::realloc_kokkos(d_mu_i, "g1l:mu_i", natom);
    grace::kk::MemKK::realloc_kokkos(d_e_atom, "g1l:e_atom", natom);
    grace::kk::MemKK::realloc_kokkos(d_A, "g1l:A", natom, nradmax, nlm);
    grace::kk::MemKK::realloc_kokkos(d_A1, "g1l:A1", natom, fc_n_out, n_funcs_A1);
    grace::kk::MemKK::realloc_kokkos(d_AA, "g1l:AA", natom, fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_AA1, "g1l:AA1", natom, fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_AAA, "g1l:AAA", natom, fc_n_out, n_funcs_AAA);
    grace::kk::MemKK::realloc_kokkos(d_AA2, "g1l:AA2", natom, fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_AAAA, "g1l:AAAA", natom, fc_n_out, n_funcs_AAAA);
    grace::kk::MemKK::realloc_kokkos(d_rho, "g1l:rho", natom, rho_n_out);

    // Adjoint views for backward pass
    grace::kk::MemKK::realloc_kokkos(d_rho_adj, "g1l:rho_adj", natom, rho_n_out);
    grace::kk::MemKK::realloc_kokkos(d_AAAA_adj, "g1l:AAAA_adj", natom, fc_n_out, n_funcs_AAAA);
    grace::kk::MemKK::realloc_kokkos(d_AA2_adj, "g1l:AA2_adj", natom, fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_AAA_adj, "g1l:AAA_adj", natom, fc_n_out, n_funcs_AAA);
    grace::kk::MemKK::realloc_kokkos(d_AA1_adj, "g1l:AA1_adj", natom, fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_AA_adj, "g1l:AA_adj", natom, fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_A1_adj, "g1l:A1_adj", natom, fc_n_out, n_funcs_A1);
    grace::kk::MemKK::realloc_kokkos(d_A_adj, "g1l:A_adj", natom, nradmax, nlm);

#ifdef KOKKOS_ENABLE_CUDA
    // cuBLAS block-sparse FC scratch. G = gathered in-columns [n_in x cs*N],
    // C = per-tile SGEMM output [n_out x cs*N]; sized to the max over the FC ops
    // (cs = natom = chunk_size). Only allocated when cuBLAS FC is active.
    if (fc_cublas_max_N > 0) {
      const size_t cols = (size_t) fc_cublas_max_N * std::max(natom, 1);
      grace::kk::MemKK::realloc_kokkos(d_fc_G, "g1l:fc_G", (size_t) std::max(fc_cublas_max_nin, 1) * cols);
      grace::kk::MemKK::realloc_kokkos(d_fc_C, "g1l:fc_C", (size_t) std::max(fc_cublas_max_nout, 1) * cols);
    }
#endif
  }

  if (need_bond_buffers() &&
      (realloc_all || (int)d_radial_basis.extent(0) < natom || (int)d_radial_basis.extent(1) < maxneigh_in)) {
    grace::kk::MemKK::realloc_kokkos(d_nearest, "g1l:nearest", natom, maxneigh_in);
    grace::kk::MemKK::realloc_kokkos(d_rnorms, "g1l:rnorms", natom, maxneigh_in);
    grace::kk::MemKK::realloc_kokkos(d_rhats, "g1l:rhats", natom, maxneigh_in);
    grace::kk::MemKK::realloc_kokkos(d_mu_j, "g1l:mu_j", natom, maxneigh_in);
    grace::kk::MemKK::realloc_kokkos(d_radial_basis, "g1l:radial_basis", natom, maxneigh_in, nradbase);
    grace::kk::MemKK::realloc_kokkos(d_dradial_basis, "g1l:dradial_basis", natom, maxneigh_in, nradbase);
    grace::kk::MemKK::realloc_kokkos(d_R_nl, "g1l:R_nl", natom, maxneigh_in, nradmax, lmax + 1);
    if constexpr (!std::is_same<NNScalar, double>::value)
      grace::kk::MemKK::realloc_kokkos(d_dR_nl, "g1l:dR_nl", natom, maxneigh_in, nradmax, lmax + 1);
    if constexpr (std::is_same<NNScalar, double>::value) {
      grace::kk::MemKK::realloc_kokkos(d_h2, "g1l:h2", natom, maxneigh_in, mlp_rad_max_dim);
      grace::kk::MemKK::realloc_kokkos(d_dh2, "g1l:dh2", natom, maxneigh_in, mlp_rad_max_dim);
    }
    grace::kk::MemKK::realloc_kokkos(d_f_ij, "g1l:f_ij", natom, maxneigh_in);

#ifdef KOKKOS_ENABLE_CUDA
    // Batched cuBLAS radial-MLP scratch. M = chunk atoms * padded maxneigh bonds.
    // Widths are physical row strides for LayoutRight row-major operands.
    if constexpr (is_cuda_v<DeviceType> && sizeof(NNScalar) == 4) {
      const int Mrows = natom * maxneigh_in;
      const int nin0 = nradbase;
      const int mh = std::max(mlp_rad_max_dim, 1);
      const int mout = nradmax * (lmax + 1);
      grace::kk::MemKK::realloc_kokkos(d_mlp_X,   "g1l:mlp_X",   std::max(Mrows, 1), std::max(nin0, 1));
      grace::kk::MemKK::realloc_kokkos(d_mlp_dX,  "g1l:mlp_dX",  std::max(Mrows, 1), std::max(nin0, 1));
      grace::kk::MemKK::realloc_kokkos(d_mlp_h0,  "g1l:mlp_h0",  std::max(Mrows, 1), mh);
      grace::kk::MemKK::realloc_kokkos(d_mlp_h1,  "g1l:mlp_h1",  std::max(Mrows, 1), mh);
      grace::kk::MemKK::realloc_kokkos(d_mlp_dh0, "g1l:mlp_dh0", std::max(Mrows, 1), mh);
      grace::kk::MemKK::realloc_kokkos(d_mlp_dh1, "g1l:mlp_dh1", std::max(Mrows, 1), mh);
      grace::kk::MemKK::realloc_kokkos(d_mlp_R,   "g1l:mlp_R",   std::max(Mrows, 1), std::max(mout, 1));
    }
#endif
  }
}

// ======================================================================
// compute stages (virtual: grace/1l/fused/kk replaces them)
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_bond_fwd(int team_size, int vector_length)
{
    // ---- Stage 1: ComputeNeigh ----
    {
      team_size = fit_team_size<DeviceType, TagComputeNeigh>(*this, chunk_size, team_size, vector_length);
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>(chunk_size, team_size, vector_length);
      Kokkos::parallel_for("ComputeNeigh", policy, *this);
    }

    // ---- Stage 2: ComputeRadialBasis ----
    {
      int ts = team_size;
      ts = fit_team_size<DeviceType, TagComputeRadialBasis>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>(
          ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
      Kokkos::parallel_for("ComputeRadialBasis", policy, *this);
    }

    // ---- Stage 3: ComputeMLPRadial ----
    compute_mlp_radial();

    if constexpr (std::is_same<NNScalar, double>::value) {
      auto h2 = d_h2;
      auto R = d_R_nl;
      auto ncount_v = d_ncount;
      auto W = d_mlp_rad_W;
      const int n_hidden = mlp_rad_last_hidden_dim;
      const int last_layer = mlp_rad_n_layers - 1;
      const NNScalar out_norm = mlp_rad_output_norm;
      const int nr = nradmax;
      const int nl = lmax + 1;
      const int mn = maxneigh;
      const int cs = chunk_size;

      Kokkos::parallel_for("ComputeMLPRadialOutput",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, mn, nr * nl}),
        KOKKOS_LAMBDA(const int ii, const int jj, const int jl) {
          if (jj >= ncount_v(ii)) return;
          const int n = jl / nl;
          const int l = jl - n * nl;
          const int j = n * nl + l;
          NNScalar sum = 0.0;
          for (int k = 0; k < n_hidden; k++)
            sum += W(last_layer, k, j) * h2(ii, jj, k);
          R(ii, jj, n, l) = sum * out_norm;
        });
    }

    // ---- Stage 4: ComputeAi ----
    {
      int ts = team_size;
      ts = fit_team_size<DeviceType, TagComputeAi>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
      int plm_size = (lmax + 1) * (lmax + 2) / 2;
      int scratch_size = scratch_bytes<DeviceType, GeomScalar>(plm_size);
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeAi>(
          ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
      policy = policy.set_scratch_size(0, Kokkos::PerThread(scratch_size));
      Kokkos::parallel_for("ComputeAi", policy, *this);
    }

}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_dense_fwd()
{
    // ---- Stage 5: FC A1 = FC(A, A) — fused Left+Right (gather, no atomics) ----
    {
      auto left_in = d_A;  auto right_in = d_A;
      auto out = d_A1;
      auto wl = d_fc_A1_wl;  auto wr = d_fc_A1_wr;
      auto wtl = d_fc_A1_wtl;
      auto rgs = d_fc_A1_rg_start; auto rg_src = d_fc_A1_rg_src; auto rg_tile = d_fc_A1_rg_tile;
      auto nof = d_fc_A1_nof;
      NNScalar nl = fc_A1_nl, nr = fc_A1_nr;
      int n_lm = n_funcs_A1, cs = chunk_size, no = fc_n_out;

      Kokkos::parallel_for("FC_A1",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, no, n_lm}),
        KOKKOS_LAMBDA(const int ii, const int k, const int lm) {
          int nin_l = left_in.extent(1);
          NNScalar sum_l = 0.0;
          int tile_l = wtl(lm);
          for (int n = 0; n < nin_l; n++)
            sum_l += wl(k, n, tile_l) * left_in(ii, n, lm);
          NNScalar result = sum_l * nl;
          int nin_r = right_in.extent(1);
          for (int g = rgs(lm); g < rgs(lm + 1); g++) {
            int src = rg_src(g), tile_r = rg_tile(g);
            NNScalar sum_r = 0.0;
            for (int n = 0; n < nin_r; n++)
              sum_r += wr(k, n, tile_r) * right_in(ii, n, src);
            result += sum_r * nr;
          }
          out(ii, k, lm) = result * nof(lm);
        });
    }

    // ---- Stage 6: Product AA = A1 ⊗ A1 (CG-coupled, gather, no atomics) ----
    {
      auto left = d_A1; auto right = d_A1; auto out = d_AA;
      auto gs = d_prod_AA_gs; auto gli = d_prod_AA_gli;
      auto gri = d_prod_AA_gri; auto gcg = d_prod_AA_gcg;
      int n_out = n_funcs_AA, n_ch = fc_n_out, cs = chunk_size;

      Kokkos::parallel_for("Product_AA",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, n_ch, n_out}),
        KOKKOS_LAMBDA(const int ii, const int k, const int o) {
          NNScalar sum = 0.0;
          for (int g = gs(o); g < gs(o + 1); g++)
            sum += gcg(g) * left(ii, k, gli(g)) * right(ii, k, gri(g));
          out(ii, k, o) = sum;
        });
    }

    // ---- Stage 7: FC AA1(/AA2 for FP64) = FC(AA, A) — fused Left+Right (gather, no atomics) ----
    if constexpr (std::is_same<NNScalar, double>::value) {
      auto left_in = d_AA;  auto right_in = d_A;
      auto out1 = d_AA1; auto out2 = d_AA2;
      auto wl1 = d_fc_AA1_wl; auto wr1 = d_fc_AA1_wr;
      auto wtl1 = d_fc_AA1_wtl;
      auto rgs1 = d_fc_AA1_rg_start; auto rg_src1 = d_fc_AA1_rg_src; auto rg_tile1 = d_fc_AA1_rg_tile;
      auto nof1 = d_fc_AA1_nof;
      auto wl2 = d_fc_AA2_wl; auto wr2 = d_fc_AA2_wr;
      auto wtl2 = d_fc_AA2_wtl;
      auto rgs2 = d_fc_AA2_rg_start; auto rg_src2 = d_fc_AA2_rg_src; auto rg_tile2 = d_fc_AA2_rg_tile;
      auto nof2 = d_fc_AA2_nof;
      NNScalar nl1 = fc_AA1_nl, nr1 = fc_AA1_nr;
      NNScalar nl2 = fc_AA2_nl, nr2 = fc_AA2_nr;
      int nlm = n_funcs_AA, cs = chunk_size, no = fc_n_out;

      Kokkos::parallel_for("FC_AA12",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, no, nlm}),
        KOKKOS_LAMBDA(const int ii, const int k, const int lm) {
          int nin_l = left_in.extent(1);
          NNScalar sum_l1 = 0.0, sum_l2 = 0.0;
          int tile_l1 = wtl1(lm), tile_l2 = wtl2(lm);
          for (int n = 0; n < nin_l; n++) {
            const NNScalar x = left_in(ii, n, lm);
            sum_l1 += wl1(k, n, tile_l1) * x;
            sum_l2 += wl2(k, n, tile_l2) * x;
          }
          NNScalar result1 = sum_l1 * nl1;
          NNScalar result2 = sum_l2 * nl2;
          int nin_r = right_in.extent(1);
          for (int g = rgs1(lm); g < rgs1(lm + 1); g++) {
            int src = rg_src1(g), tile_r = rg_tile1(g);
            NNScalar sum_r = 0.0;
            for (int n = 0; n < nin_r; n++)
              sum_r += wr1(k, n, tile_r) * right_in(ii, n, src);
            result1 += sum_r * nr1;
          }
          for (int g = rgs2(lm); g < rgs2(lm + 1); g++) {
            int src = rg_src2(g), tile_r = rg_tile2(g);
            NNScalar sum_r = 0.0;
            for (int n = 0; n < nin_r; n++)
              sum_r += wr2(k, n, tile_r) * right_in(ii, n, src);
            result2 += sum_r * nr2;
          }
          out1(ii, k, lm) = result1 * nof1(lm);
          out2(ii, k, lm) = result2 * nof2(lm);
        });
    } else {
      bool fc_done = false;
#ifdef KOKKOS_ENABLE_CUDA
      // cuBLAS true-fp32 block-sparse FC (CUDA+fp32/Mixed): gather -> per-tile
      // SGEMM -> scatter; falls back to the hand kernel below when disabled.
      if (blas_handle && fc_op_AA1.active) {
        fc_forward_cublas(fc_op_AA1, d_AA, d_A, d_AA1);
        fc_done = true;
      }
#endif
      if (!fc_done) {
      auto left_in = d_AA;  auto right_in = d_A;
      auto out = d_AA1;
      auto wl = d_fc_AA1_wl; auto wr = d_fc_AA1_wr;
      auto wtl = d_fc_AA1_wtl;
      auto rgs = d_fc_AA1_rg_start; auto rg_src = d_fc_AA1_rg_src; auto rg_tile = d_fc_AA1_rg_tile;
      auto nof = d_fc_AA1_nof;
      NNScalar nl = fc_AA1_nl, nr = fc_AA1_nr;
      int nlm = n_funcs_AA, cs = chunk_size, no = fc_n_out;

      Kokkos::parallel_for("FC_AA1",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, no, nlm}),
        KOKKOS_LAMBDA(const int ii, const int k, const int lm) {
          int nin_l = left_in.extent(1);
          NNScalar sum_l = 0.0;
          int tile_l = wtl(lm);
          for (int n = 0; n < nin_l; n++)
            sum_l += wl(k, n, tile_l) * left_in(ii, n, lm);
          NNScalar result = sum_l * nl;
          int nin_r = right_in.extent(1);
          for (int g = rgs(lm); g < rgs(lm + 1); g++) {
            int src = rg_src(g), tile_r = rg_tile(g);
            NNScalar sum_r = 0.0;
            for (int n = 0; n < nin_r; n++)
              sum_r += wr(k, n, tile_r) * right_in(ii, n, src);
            result += sum_r * nr;
          }
          out(ii, k, lm) = result * nof(lm);
        });
      }
    }

    // ---- Stage 8: Product AAA = AA1 ⊗ A1 (gather, no atomics) ----
    {
      auto left = d_AA1; auto right = d_A1; auto out = d_AAA;
      auto gs = d_prod_AAA_gs; auto gli = d_prod_AAA_gli;
      auto gri = d_prod_AAA_gri; auto gcg = d_prod_AAA_gcg;
      int n_out = n_funcs_AAA, n_ch = fc_n_out, cs = chunk_size;

      Kokkos::parallel_for("Product_AAA",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, n_ch, n_out}),
        KOKKOS_LAMBDA(const int ii, const int k, const int o) {
          NNScalar sum = 0.0;
          for (int g = gs(o); g < gs(o + 1); g++)
            sum += gcg(g) * left(ii, k, gli(g)) * right(ii, k, gri(g));
          out(ii, k, o) = sum;
        });
    }

    // ---- Stage 9: FC AA2 = FC(AA, A), only needed separately for non-FP64 ----
    if constexpr (!std::is_same<NNScalar, double>::value) {
      bool fc_done = false;
#ifdef KOKKOS_ENABLE_CUDA
      if (blas_handle && fc_op_AA2.active) {
        fc_forward_cublas(fc_op_AA2, d_AA, d_A, d_AA2);
        fc_done = true;
      }
#endif
      if (!fc_done) {
      auto left_in = d_AA;  auto right_in = d_A;
      auto out = d_AA2;
      auto wl = d_fc_AA2_wl; auto wr = d_fc_AA2_wr;
      auto wtl = d_fc_AA2_wtl;
      auto rgs = d_fc_AA2_rg_start; auto rg_src = d_fc_AA2_rg_src; auto rg_tile = d_fc_AA2_rg_tile;
      auto nof = d_fc_AA2_nof;
      NNScalar nl = fc_AA2_nl, nr = fc_AA2_nr;
      int n_lm = n_funcs_AA, cs = chunk_size, no = fc_n_out;

      Kokkos::parallel_for("FC_AA2",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, no, n_lm}),
        KOKKOS_LAMBDA(const int ii, const int k, const int lm) {
          int nin_l = left_in.extent(1);
          NNScalar sum_l = 0.0;
          int tile_l = wtl(lm);
          for (int n = 0; n < nin_l; n++)
            sum_l += wl(k, n, tile_l) * left_in(ii, n, lm);
          NNScalar result = sum_l * nl;
          int nin_r = right_in.extent(1);
          for (int g = rgs(lm); g < rgs(lm + 1); g++) {
            int src = rg_src(g), tile_r = rg_tile(g);
            NNScalar sum_r = 0.0;
            for (int n = 0; n < nin_r; n++)
              sum_r += wr(k, n, tile_r) * right_in(ii, n, src);
            result += sum_r * nr;
          }
          out(ii, k, lm) = result * nof(lm);
        });
      }
    }

    // ---- Stage 10: Product AAAA = AA2 ⊗ AA2 (gather, no atomics) ----
    {
      auto left = d_AA2; auto right = d_AA2; auto out = d_AAAA;
      auto gs = d_prod_AAAA_gs; auto gli = d_prod_AAAA_gli;
      auto gri = d_prod_AAAA_gri; auto gcg = d_prod_AAAA_gcg;
      int n_out = n_funcs_AAAA, n_ch = fc_n_out, cs = chunk_size;

      Kokkos::parallel_for("Product_AAAA",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, n_ch, n_out}),
        KOKKOS_LAMBDA(const int ii, const int k, const int o) {
          NNScalar sum = 0.0;
          for (int g = gs(o); g < gs(o + 1); g++)
            sum += gcg(g) * left(ii, k, gli(g)) * right(ii, k, gri(g));
          out(ii, k, o) = sum;
        });
    }

}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_reduce()
{
    // ---- Stage 11: ComputeReduceN ----
    // Parallelize over (atom, rho_channel) for better GPU utilization
    {
      auto rho_out = d_rho;
      auto il = d_ilist; auto mp = d_map; auto tp = type;
      int co = chunk_offset, cs = chunk_size, nrho = rho_n_out;

      // Reduce from A: W[89,17,32,1] * A[ii,32,25]
      {
        auto W = d_reduce_A_W; auto ci = d_reduce_A_ci; auto X = d_A;
        NNScalar norm = reduce_A_norm;
        Kokkos::parallel_for("ReduceN_A",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nrho}),
          KOKKOS_LAMBDA(const int ii, const int k) {
            const int mu_i = mp(tp(il[ii + co]));
            const int nin = W.extent(2), ws = W.extent(3);
            NNScalar val = 0.0;
            for (int w = 0; w < ws; w++) {
              const int fi = ci(w);
              for (int n = 0; n < nin; n++)
                val += W(mu_i, k, n, w) * X(ii, n, fi);
            }
            rho_out(ii, k) += val * norm;
          });
      }
      // Reduce from AA: W[89,17,64,5] * AA[ii,64,141]
      {
        auto W = d_reduce_AA_W; auto ci = d_reduce_AA_ci; auto X = d_AA;
        NNScalar norm = reduce_AA_norm;
        int ws = W.extent(3);
        Kokkos::parallel_for("ReduceN_AA",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nrho, ws}),
          KOKKOS_LAMBDA(const int ii, const int k, const int w) {
            const int mu_i = mp(tp(il[ii + co]));
            const int nin = W.extent(2);
            NNScalar val = 0.0;
            const int fi = ci(w);
            for (int n = 0; n < nin; n++)
              val += W(mu_i, k, n, w) * X(ii, n, fi);
            Kokkos::atomic_add(&rho_out(ii, k), val * norm);
          });
      }
      // Reduce from AAA: W[89,17,64,14] * AAA[ii,64,14]
      {
        auto W = d_reduce_AAA_W; auto ci = d_reduce_AAA_ci; auto X = d_AAA;
        NNScalar norm = reduce_AAA_norm;
        int ws = W.extent(3);
        Kokkos::parallel_for("ReduceN_AAA",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nrho, ws}),
          KOKKOS_LAMBDA(const int ii, const int k, const int w) {
            const int mu_i = mp(tp(il[ii + co]));
            const int nin = W.extent(2);
            NNScalar val = 0.0;
            const int fi = ci(w);
            for (int n = 0; n < nin; n++)
              val += W(mu_i, k, n, w) * X(ii, n, fi);
            Kokkos::atomic_add(&rho_out(ii, k), val * norm);
          });
      }
      // Reduce from AAAA: W[89,17,64,69] * AAAA[ii,64,69]
      {
        auto W = d_reduce_AAAA_W; auto ci = d_reduce_AAAA_ci; auto X = d_AAAA;
        NNScalar norm = reduce_AAAA_norm;
        int ws = W.extent(3);
        Kokkos::parallel_for("ReduceN_AAAA",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nrho, ws}),
          KOKKOS_LAMBDA(const int ii, const int k, const int w) {
            const int mu_i = mp(tp(il[ii + co]));
            const int nin = W.extent(2);
            NNScalar val = 0.0;
            const int fi = ci(w);
            for (int n = 0; n < nin; n++)
              val += W(mu_i, k, n, w) * X(ii, n, fi);
            Kokkos::atomic_add(&rho_out(ii, k), val * norm);
          });
      }
    }

}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_dense_bwd()
{
    // ---- B2: ReverseReduceN → d_A_adj, d_AA_adj, d_AAA_adj, d_AAAA_adj ----
    // Parallelize over (atom, w) for each reduction block
    {
      auto rho_adj_v = d_rho_adj;
      auto il = d_ilist; auto tp = type; auto mp = d_map;
      int co = chunk_offset, cs = chunk_size, n_rho = rho_n_out;

      // All collect_ind arrays are verified injective (unique fi per w), so
      // each (ii, n, fi) is written by exactly one thread — no atomic needed.
      // RevReduceN_AAA/AAAA: ci is a full bijection over all funcs → deep_copy not needed.
      // RevReduceN_A/AA: ci is injective but partial → deep_copy zeros the rest (done above).

      // Reverse A
      {
        auto W = d_reduce_A_W; auto ci = d_reduce_A_ci; auto X = d_A_adj;
        NNScalar norm = reduce_A_norm;
        int ws = W.extent(3);
        int nin = W.extent(2);
        Kokkos::parallel_for("RevReduceN_A",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ws, nin}),
          KOKKOS_LAMBDA(const int ii, const int w, const int n) {
            const int mu_i = mp(tp(il[ii + co]));
            const int fi = ci(w);
            NNScalar val = 0.0;
            for (int k = 0; k < n_rho; k++)
              val += rho_adj_v(ii, k) * W(mu_i, k, n, w);
            X(ii, n, fi) = val * norm;
          });
      }
      // Reverse AA
      {
        auto W = d_reduce_AA_W; auto ci = d_reduce_AA_ci; auto X = d_AA_adj;
        NNScalar norm = reduce_AA_norm;
        int ws = W.extent(3);
        int nin = W.extent(2);
        Kokkos::parallel_for("RevReduceN_AA",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ws, nin}),
          KOKKOS_LAMBDA(const int ii, const int w, const int n) {
            const int mu_i = mp(tp(il[ii + co]));
            const int fi = ci(w);
            NNScalar val = 0.0;
            for (int k = 0; k < n_rho; k++)
              val += rho_adj_v(ii, k) * W(mu_i, k, n, w);
            X(ii, n, fi) = val * norm;
          });
      }
      // Reverse AAA (ci is full bijection → no prior deep_copy needed)
      {
        auto W = d_reduce_AAA_W; auto ci = d_reduce_AAA_ci; auto X = d_AAA_adj;
        NNScalar norm = reduce_AAA_norm;
        int ws = W.extent(3);
        int nin = W.extent(2);
        Kokkos::parallel_for("RevReduceN_AAA",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ws, nin}),
          KOKKOS_LAMBDA(const int ii, const int w, const int n) {
            const int mu_i = mp(tp(il[ii + co]));
            const int fi = ci(w);
            NNScalar val = 0.0;
            for (int k = 0; k < n_rho; k++)
              val += rho_adj_v(ii, k) * W(mu_i, k, n, w);
            X(ii, n, fi) = val * norm;
          });
      }
      // Reverse AAAA (ci is full bijection → no prior deep_copy needed)
      {
        auto W = d_reduce_AAAA_W; auto ci = d_reduce_AAAA_ci; auto X = d_AAAA_adj;
        NNScalar norm = reduce_AAAA_norm;
        int ws = W.extent(3);
        int nin = W.extent(2);
        Kokkos::parallel_for("RevReduceN_AAAA",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ws, nin}),
          KOKKOS_LAMBDA(const int ii, const int w, const int n) {
            const int mu_i = mp(tp(il[ii + co]));
            const int fi = ci(w);
            NNScalar val = 0.0;
            for (int k = 0; k < n_rho; k++)
              val += rho_adj_v(ii, k) * W(mu_i, k, n, w);
            X(ii, n, fi) = val * norm;
          });
      }
    }

    // ---- B3: ReverseProduct_AAAA (AA2 ⊗ AA2, gather, no atomics) → d_AA2_adj ----
    {
      auto fwd = d_AA2; auto oa = d_AAAA_adj; auto adj = d_AA2_adj;
      auto gs = d_rprod_AAAA_gs; auto goa = d_rprod_AAAA_goa;
      auto gfwd = d_rprod_AAAA_gfwd; auto gcg = d_rprod_AAAA_gcg;
      int n_adj = n_funcs_AA, nc = fc_n_out, cs = chunk_size;

      Kokkos::parallel_for("RevProd_AAAA",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nc, n_adj}),
        KOKKOS_LAMBDA(const int ii, const int k, const int f) {
          NNScalar sum = 0.0;
          for (int g = gs(f); g < gs(f + 1); g++)
            sum += gcg(g) * oa(ii, k, goa(g)) * fwd(ii, k, gfwd(g));
          adj(ii, k, f) = sum;
        });
    }

    // ---- B4: ReverseFC_AA2 → d_AA_adj +=, d_A_adj += ----
    {
      auto oa = d_AA2_adj; auto la = d_AA_adj; auto ra = d_A_adj;
      auto wl = d_fc_AA2_wl; auto wr = d_fc_AA2_wr;
      auto wtl = d_fc_AA2_wtl; auto wtr = d_fc_AA2_wtr;
      auto ct = d_fc_AA2_ct; auto cf = d_fc_AA2_cf; auto nof = d_fc_AA2_nof;
      NNScalar nl = fc_AA2_nl, nr = fc_AA2_nr;
      int no = fc_n_out, nlm = n_funcs_AA, ncol = (int)d_fc_AA2_cf.extent(0), cs = chunk_size;

      // Left part: la[ii,n,lm] += nof[lm]*nl * Σ_k wl[k,n,tile] * oa[ii,k,lm]
      { int nil = (int)la.extent(1);
      Kokkos::parallel_for("RevFC_AA2_L",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nil, nlm}),
        KOKKOS_LAMBDA(const int ii, const int n, const int lm) {
          NNScalar nf = nof(lm) * nl;
          int tile = wtl(lm);
          NNScalar val = 0.0;
          for (int k = 0; k < no; k++) val += wl(k, n, tile) * oa(ii, k, lm);
          la(ii, n, lm) += val * nf;
        }); }
      // Right part: gather by source — ra[ii,n,src] += Σ_{g} nof[tgt]*nr * Σ_k wr[k,n,tile] * oa[ii,k,tgt]
      { int nir = (int)ra.extent(1);
        auto revs = d_fc_AA2_rev_start; auto rev_tgt = d_fc_AA2_rev_src; auto rev_tile = d_fc_AA2_rev_tile;
        int n_src = (int)revs.extent(0) - 1;
      Kokkos::parallel_for("RevFC_AA2_R",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nir, n_src}),
        KOKKOS_LAMBDA(const int ii, const int n, const int src) {
          NNScalar val = 0.0;
          for (int g = revs(src); g < revs(src + 1); g++) {
            int tl = rev_tgt(g), tile = rev_tile(g);
            NNScalar gval = 0.0;
            for (int k = 0; k < no; k++) gval += wr(k, n, tile) * oa(ii, k, tl);
            val += gval * nof(tl);
          }
          ra(ii, n, src) += val * nr;
        }); }
    }

    // ---- B5: ReverseProduct_AAA (AA1 ⊗ A1, gather, no atomics) → d_AA1_adj, d_A1_adj ----
    {
      auto fwd_r = d_A1; auto oa = d_AAA_adj; auto adj = d_AA1_adj;
      auto gs = d_rprod_AAA_left_gs; auto goa = d_rprod_AAA_left_goa;
      auto gfwd = d_rprod_AAA_left_gfwd; auto gcg = d_rprod_AAA_left_gcg;
      int n_adj = n_funcs_AA, nc = fc_n_out, cs = chunk_size;

      Kokkos::parallel_for("RevProd_AAA_L",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nc, n_adj}),
        KOKKOS_LAMBDA(const int ii, const int k, const int f) {
          NNScalar sum = 0.0;
          for (int g = gs(f); g < gs(f + 1); g++)
            sum += gcg(g) * oa(ii, k, goa(g)) * fwd_r(ii, k, gfwd(g));
          adj(ii, k, f) = sum;
        });
    }
    {
      auto fwd_l = d_AA1; auto oa = d_AAA_adj; auto adj = d_A1_adj;
      auto gs = d_rprod_AAA_right_gs; auto goa = d_rprod_AAA_right_goa;
      auto gfwd = d_rprod_AAA_right_gfwd; auto gcg = d_rprod_AAA_right_gcg;
      int n_adj = n_funcs_A1, nc = fc_n_out, cs = chunk_size;

      Kokkos::parallel_for("RevProd_AAA_R",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nc, n_adj}),
        KOKKOS_LAMBDA(const int ii, const int k, const int f) {
          NNScalar sum = 0.0;
          for (int g = gs(f); g < gs(f + 1); g++)
            sum += gcg(g) * oa(ii, k, goa(g)) * fwd_l(ii, k, gfwd(g));
          adj(ii, k, f) = sum;
        });
    }

    // ---- B6: ReverseFC_AA1 → d_AA_adj +=, d_A_adj += ----
    {
      auto oa = d_AA1_adj; auto la = d_AA_adj; auto ra = d_A_adj;
      auto wl = d_fc_AA1_wl; auto wr = d_fc_AA1_wr;
      auto wtl = d_fc_AA1_wtl; auto wtr = d_fc_AA1_wtr;
      auto ct = d_fc_AA1_ct; auto cf = d_fc_AA1_cf; auto nof = d_fc_AA1_nof;
      NNScalar nl = fc_AA1_nl, nr = fc_AA1_nr;
      int no = fc_n_out, nlm = n_funcs_AA, ncol = (int)d_fc_AA1_cf.extent(0), cs = chunk_size;

      // Left part
      { int nil = (int)la.extent(1);
      Kokkos::parallel_for("RevFC_AA1_L",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nil, nlm}),
        KOKKOS_LAMBDA(const int ii, const int n, const int lm) {
          NNScalar nf = nof(lm) * nl;
          int tile = wtl(lm);
          NNScalar val = 0.0;
          for (int k = 0; k < no; k++) val += wl(k, n, tile) * oa(ii, k, lm);
          la(ii, n, lm) += val * nf;
        }); }
      // Right part: gather by source
      { int nir = (int)ra.extent(1);
        auto revs = d_fc_AA1_rev_start; auto rev_tgt = d_fc_AA1_rev_src; auto rev_tile = d_fc_AA1_rev_tile;
        int n_src = (int)revs.extent(0) - 1;
      Kokkos::parallel_for("RevFC_AA1_R",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nir, n_src}),
        KOKKOS_LAMBDA(const int ii, const int n, const int src) {
          NNScalar val = 0.0;
          for (int g = revs(src); g < revs(src + 1); g++) {
            int tl = rev_tgt(g), tile = rev_tile(g);
            NNScalar gval = 0.0;
            for (int k = 0; k < no; k++) gval += wr(k, n, tile) * oa(ii, k, tl);
            val += gval * nof(tl);
          }
          ra(ii, n, src) += val * nr;
        }); }
    }

    // ---- B7: ReverseProduct_AA (A1 ⊗ A1, gather, no atomics) → d_A1_adj += ----
    {
      auto fwd = d_A1; auto oa = d_AA_adj; auto adj = d_A1_adj;
      auto gs = d_rprod_AA_gs; auto goa_v = d_rprod_AA_goa;
      auto gfwd_v = d_rprod_AA_gfwd; auto gcg = d_rprod_AA_gcg;
      int n_adj = n_funcs_A1, nc = fc_n_out, cs = chunk_size;

      Kokkos::parallel_for("RevProd_AA",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nc, n_adj}),
        KOKKOS_LAMBDA(const int ii, const int k, const int f) {
          NNScalar sum = 0.0;
          for (int g = gs(f); g < gs(f + 1); g++)
            sum += gcg(g) * oa(ii, k, goa_v(g)) * fwd(ii, k, gfwd_v(g));
          adj(ii, k, f) += sum;
        });
    }

    // ---- B8: ReverseFC_A1 → d_A_adj += (both left and right inputs are A) ----
    {
      auto oa = d_A1_adj; auto aa = d_A_adj;
      auto wl = d_fc_A1_wl; auto wr = d_fc_A1_wr;
      auto wtl = d_fc_A1_wtl; auto wtr = d_fc_A1_wtr;
      auto ct = d_fc_A1_ct; auto cf = d_fc_A1_cf; auto nof = d_fc_A1_nof;
      NNScalar nl = fc_A1_nl, nr = fc_A1_nr;
      int no = fc_n_out, nlm = n_funcs_A1, ncol = (int)d_fc_A1_cf.extent(0), cs = chunk_size;

      // Left part
      { int ni = (int)aa.extent(1);
      Kokkos::parallel_for("RevFC_A1_L",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ni, nlm}),
        KOKKOS_LAMBDA(const int ii, const int n, const int lm) {
          NNScalar nf = nof(lm) * nl;
          int tile = wtl(lm);
          NNScalar val = 0.0;
          for (int k = 0; k < no; k++) val += wl(k, n, tile) * oa(ii, k, lm);
          aa(ii, n, lm) += val * nf;
        }); }
      // Right part: gather by source
      { int ni = (int)aa.extent(1);
        auto revs = d_fc_A1_rev_start; auto rev_tgt = d_fc_A1_rev_src; auto rev_tile = d_fc_A1_rev_tile;
        int n_src = (int)revs.extent(0) - 1;
      Kokkos::parallel_for("RevFC_A1_R",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ni, n_src}),
        KOKKOS_LAMBDA(const int ii, const int n, const int src) {
          NNScalar val = 0.0;
          for (int g = revs(src); g < revs(src + 1); g++) {
            int tl = rev_tgt(g), tile = rev_tile(g);
            NNScalar gval = 0.0;
            for (int k = 0; k < no; k++) gval += wr(k, n, tile) * oa(ii, k, tl);
            val += gval * nof(tl);
          }
          aa(ii, n, src) += val * nr;
        }); }
    }

}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_bond_bwd(int team_size, int vector_length)
{
    // ---- B9: ComputeDerivative (per-bond forces from d_A_adj) ----
    {
      int ts = team_size;
      ts = fit_team_size<DeviceType, TagComputeDerivative>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
      int plm_size = (lmax + 1) * (lmax + 2) / 2;
      // Scratch: plm + dplm (z_tr removed — precomputed as d_z_tr)
      int scratch_size = scratch_bytes<DeviceType, GeomScalar>(2 * plm_size);
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeDerivative>(
          ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
      policy = policy.set_scratch_size(0, Kokkos::PerThread(scratch_size));
      Kokkos::parallel_for("ComputeDerivative", policy, *this);
    }

    // ---- B10: ComputeForce (accumulate f_ij to atom forces + virial) ----
    {
      auto fij = d_f_ij; auto nc = d_ncount; auto near = d_nearest;
      auto fout = f; auto il = d_ilist;
      auto rh = d_rhats; auto rn = d_rnorms;
      int co = chunk_offset, cs = chunk_size;
      bool do_v = (vflag_global != 0);
      bool do_va = (vflag_atom != 0);
      bool do_cva = (cvflag_atom != 0);
      auto va = d_vatom;
      auto cva = d_cvatom;

      EV ev_force;
      Kokkos::parallel_reduce("ComputeForce", Kokkos::RangePolicy<DeviceType>(0, cs),
        KOKKOS_LAMBDA(const int ii, EV& ev) {
          const int i = il[ii + co];
          const int nct = nc(ii);
          for (int jj = 0; jj < nct; jj++) {
            const int j = near(ii, jj);
            const GeomScalar fx = fij(ii, jj, 0);
            const GeomScalar fy = fij(ii, jj, 1);
            const GeomScalar fz = fij(ii, jj, 2);
            Kokkos::atomic_add(&fout(i, 0), fx);
            Kokkos::atomic_add(&fout(i, 1), fy);
            Kokkos::atomic_add(&fout(i, 2), fz);
            Kokkos::atomic_add(&fout(j, 0), -fx);
            Kokkos::atomic_add(&fout(j, 1), -fy);
            Kokkos::atomic_add(&fout(j, 2), -fz);
            if (do_v || do_va || do_cva) {
              const GeomScalar delx = -rh(ii, jj, 0) * rn(ii, jj);
              const GeomScalar dely = -rh(ii, jj, 1) * rn(ii, jj);
              const GeomScalar delz = -rh(ii, jj, 2) * rn(ii, jj);
              const GeomScalar v0 = delx * fx, v1 = dely * fy, v2 = delz * fz;
              const GeomScalar v3 = delx * fy, v4 = delx * fz, v5 = dely * fz;
              if (do_v) {
                ev.v[0] += v0; ev.v[1] += v1; ev.v[2] += v2;
                ev.v[3] += v3; ev.v[4] += v4; ev.v[5] += v5;
              }
              if (do_va) {
                // Half to each atom (half neighbor list)
                const GeomScalar half = GeomScalar(0.5);
                Kokkos::atomic_add(&va(i, 0), (KKF)(half * v0));
                Kokkos::atomic_add(&va(i, 1), (KKF)(half * v1));
                Kokkos::atomic_add(&va(i, 2), (KKF)(half * v2));
                Kokkos::atomic_add(&va(i, 3), (KKF)(half * v3));
                Kokkos::atomic_add(&va(i, 4), (KKF)(half * v4));
                Kokkos::atomic_add(&va(i, 5), (KKF)(half * v5));
                Kokkos::atomic_add(&va(j, 0), (KKF)(half * v0));
                Kokkos::atomic_add(&va(j, 1), (KKF)(half * v1));
                Kokkos::atomic_add(&va(j, 2), (KKF)(half * v2));
                Kokkos::atomic_add(&va(j, 3), (KKF)(half * v3));
                Kokkos::atomic_add(&va(j, 4), (KKF)(half * v4));
                Kokkos::atomic_add(&va(j, 5), (KKF)(half * v5));
              }
              if (do_cva) {
                const GeomScalar half = GeomScalar(0.5);
                const GeomScalar v6 = dely * fx, v7 = delz * fx, v8 = delz * fy;
                Kokkos::atomic_add(&cva(i, 0), (KKF)(half * v0));
                Kokkos::atomic_add(&cva(i, 1), (KKF)(half * v1));
                Kokkos::atomic_add(&cva(i, 2), (KKF)(half * v2));
                Kokkos::atomic_add(&cva(i, 3), (KKF)(half * v3));
                Kokkos::atomic_add(&cva(i, 4), (KKF)(half * v4));
                Kokkos::atomic_add(&cva(i, 5), (KKF)(half * v5));
                Kokkos::atomic_add(&cva(i, 6), (KKF)(half * v6));
                Kokkos::atomic_add(&cva(i, 7), (KKF)(half * v7));
                Kokkos::atomic_add(&cva(i, 8), (KKF)(half * v8));
                Kokkos::atomic_add(&cva(j, 0), (KKF)(half * v0));
                Kokkos::atomic_add(&cva(j, 1), (KKF)(half * v1));
                Kokkos::atomic_add(&cva(j, 2), (KKF)(half * v2));
                Kokkos::atomic_add(&cva(j, 3), (KKF)(half * v3));
                Kokkos::atomic_add(&cva(j, 4), (KKF)(half * v4));
                Kokkos::atomic_add(&cva(j, 5), (KKF)(half * v5));
                Kokkos::atomic_add(&cva(j, 6), (KKF)(half * v6));
                Kokkos::atomic_add(&cva(j, 7), (KKF)(half * v7));
                Kokkos::atomic_add(&cva(j, 8), (KKF)(half * v8));
              }
            }
          }
        }, ev_force);

      if (vflag_global) {
        virial[0] += ev_force.v[0]; virial[1] += ev_force.v[1]; virial[2] += ev_force.v[2];
        virial[3] += ev_force.v[3]; virial[4] += ev_force.v[4]; virial[5] += ev_force.v[5];
      }
    }

}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_zero_adj()
{
    Kokkos::deep_copy(d_A_adj, 0.0);
    Kokkos::deep_copy(d_AA_adj, 0.0);
}

// ======================================================================
// compute: main chunked pipeline
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeNeigh,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>::member_type& team) const
{
  const int ii = team.league_rank();
  const int i = d_ilist[ii + chunk_offset];
  const int itype = type(i);
  const auto xi = x(i,0), yi = x(i,1), zi = x(i,2);    // positions' precision: GeomScalar of the difference
  // the neighbours inside the pair cutoff, packed in list order by one team scan: both passes of the scan test the
  // cutoff, the first one counting, the second one writing each bond at its prefix slot
  int nin = 0;
  Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, d_numneigh[i]),
      [&] (const int k, int& slot, const bool write) {
    const int j = d_neighbors(i, k) & Traits::neighmask;
    const GeomScalar dx = GeomScalar(xi - x(j,0)), dy = GeomScalar(yi - x(j,1)), dz = GeomScalar(zi - x(j,2));
    const GeomScalar rsq = dx*dx + dy*dy + dz*dz;
    if (!(rsq < d_cutsq(itype, type(j)))) return;
    if (write) {
      const GeomScalar r = Kokkos::sqrt(rsq);
      const GeomScalar rinv = GeomScalar(1.0) / r;
      d_mu_j(ii, slot) = d_map(type(j));
      d_rnorms(ii, slot) = r;
      d_rhats(ii, slot, 0) = -dx * rinv;    // unit vector from i to j
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
// Kernel: ComputeRadialBasis - Chebyshev polynomials + cutoff
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeRadialBasis,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;

  const int jj = slot.neigh;
  const int ncount = d_ncount(ii);
  if (jj >= ncount) return;

  const GeomScalar r = d_rnorms(ii, jj);
  // Per-bond cutoff: look up from bond_cutoff_map[mu_i, mu_j]
  const GeomScalar rcut_ij = d_bond_cutoff(d_mu_i(ii), d_mu_j(ii, jj));
  const GeomScalar x_norm = r / rcut_ij;  // x in [0, 1]

  // Smooth polynomial envelope cutoff (p-order):
  // fcut(x) = 1 - (p+1)(p+2)/2 * x^p + p*(p+2) * x^(p+1) - p*(p+1)/2 * x^(p+2)
  const int p = radial_basis_p;
  GeomScalar xp = GeomScalar(1.0);
  for (int ip = 0; ip < p; ip++) xp *= x_norm;
  const GeomScalar xp1 = xp * x_norm;     // x^(p+1)
  const GeomScalar xp2 = xp1 * x_norm;    // x^(p+2)
  const GeomScalar pp1 = GeomScalar(p) * (p + 1);
  const GeomScalar pp2 = GeomScalar(p) * (p + 2);
  const GeomScalar p1p2 = GeomScalar(p + 1) * (p + 2);
  const GeomScalar fcut = GeomScalar(1.0) - GeomScalar(0.5) * p1p2 * xp + pp2 * xp1 - GeomScalar(0.5) * pp1 * xp2;

  // Derivative of fcut w.r.t. r:
  // dfcut/dr = (1/rcut_ij) * [-p*(p+1)*(p+2)/2 * x^(p-1)
  //            + p*(p+1)*(p+2) * x^p - p*(p+1)*(p+2)/2 * x^(p+1)]
  GeomScalar dfcut = GeomScalar(0.0);
  if (x_norm > GeomScalar(1e-14)) {
    const GeomScalar xp_m1 = xp / x_norm;  // x^(p-1)
    const GeomScalar coeff = GeomScalar(p) * (p + 1) * (p + 2) * GeomScalar(0.5);
    dfcut = coeff * (-xp_m1 + GeomScalar(2.0) * xp - xp1) / rcut_ij;
  }

  // Chebyshev variable: x_cheb = 2*(r/rcut_ij) - 1 (maps [0, rcut_ij] to [-1, 1])
  const GeomScalar x_cheb = GeomScalar(2.0) * x_norm - GeomScalar(1.0);
  const GeomScalar dx_cheb_dr = GeomScalar(2.0) / rcut_ij;

  // Chebyshev polynomials T_1..T_{nradbase} (skip T_0, use kind=1)
  // T_0 = 1, T_1 = x, T_k = 2*x*T_{k-1} - T_{k-2}
  GeomScalar T_prev = GeomScalar(1.0);    // T_0
  GeomScalar T_curr = x_cheb;             // T_1
  GeomScalar dT_prev = GeomScalar(0.0);   // dT_0/dx = 0
  GeomScalar dT_curr = GeomScalar(1.0);   // dT_1/dx = 1

  // Store T_1 as first basis function (k=0 in output)
  d_radial_basis(ii, jj, 0) = T_curr * fcut;
  d_dradial_basis(ii, jj, 0) = dT_curr * dx_cheb_dr * fcut + T_curr * dfcut;

  for (int k = 1; k < nradbase; k++) {
    // T_{k+1} = 2*x*T_k - T_{k-1}
    const GeomScalar T_next = GeomScalar(2.0) * x_cheb * T_curr - T_prev;
    const GeomScalar dT_next = GeomScalar(2.0) * (T_curr + x_cheb * dT_curr) - dT_prev;
    T_prev = T_curr;
    dT_prev = dT_curr;
    T_curr = T_next;
    dT_curr = dT_next;

    // g_k = T_{k+1} * fcut  (output index k stores T_{k+1})
    d_radial_basis(ii, jj, k) = T_curr * fcut;
    d_dradial_basis(ii, jj, k) = dT_curr * dx_cheb_dr * fcut + T_curr * dfcut;
  }
}

// ======================================================================
// Kernel: ComputeMLPRadial - MLP(g_k) -> R_nl
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeMLPRadial,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadial>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;

  const int jj = slot.neigh;
  const int ncount = d_ncount(ii);
  if (jj >= ncount) return;

  // Cache layer dims/norms in registers (avoid repeated device view reads)
  const int n_mlp_layers = mlp_rad_n_layers;
  int mlp_dims[GRACE1L_MAX_MLP_LAYERS + 1];
  NNScalar mlp_norms[GRACE1L_MAX_MLP_LAYERS];
  for (int i = 0; i <= n_mlp_layers; i++) mlp_dims[i] = d_mlp_rad_dims(i);
  for (int i = 0; i < n_mlp_layers; i++) mlp_norms[i] = d_mlp_rad_norms(i);

  NNScalar buf_a[GRACE1L_MAX_MLP_DIM], buf_b[GRACE1L_MAX_MLP_DIM];
  NNScalar dbuf_a[GRACE1L_MAX_MLP_DIM], dbuf_b[GRACE1L_MAX_MLP_DIM];
  NNScalar* h_cur = buf_a;  NNScalar* h_nxt = buf_b;
  NNScalar* dh_cur = dbuf_a; NNScalar* dh_nxt = dbuf_b;

  const int n_input = mlp_dims[0];
  for (int k = 0; k < n_input; k++) {
    h_cur[k] = d_radial_basis(ii, jj, k);
    dh_cur[k] = d_dradial_basis(ii, jj, k);
  }

  // Hidden layers: 0 .. n_layers-2 (all with silu activation)
  for (int layer = 0; layer < n_mlp_layers - 1; layer++) {
    const int nin  = mlp_dims[layer];
    const int nout = mlp_dims[layer + 1];
    const NNScalar norm = mlp_norms[layer];
    for (int j = 0; j < nout; j++) {
      NNScalar s = 0.0, ds = 0.0;
      for (int k = 0; k < nin; k++) {
        NNScalar w = d_mlp_rad_W(layer, k, j);
        s  += w * h_cur[k];
        ds += w * dh_cur[k];
      }
      s *= norm; ds *= norm;
      NNScalar sig = 1.0 / (1.0 + Kokkos::exp(-s));
      h_nxt[j]  = s * sig;
      dh_nxt[j] = sig * (1.0 + s * (1.0 - sig)) * ds;
    }
    // Swap cur/nxt
    NNScalar* tmp;
    tmp = h_cur;  h_cur  = h_nxt;  h_nxt  = tmp;
    tmp = dh_cur; dh_cur = dh_nxt; dh_nxt = tmp;
  }

  const int n_last_hidden = mlp_dims[n_mlp_layers - 1];
  if constexpr (std::is_same<NNScalar, double>::value) {
    for (int j = 0; j < n_last_hidden; j++) {
      d_h2(ii, jj, j) = h_cur[j];
      d_dh2(ii, jj, j) = dh_cur[j];
    }
  } else {
    // Output layer (last layer, no activation). For float NN paths, compute
    // dR/dr together with R while h_cur/dh_cur are still in registers; FP64
    // computes R in a separate output-parallel kernel.
    const int last_layer = n_mlp_layers - 1;
    const NNScalar out_norm = mlp_norms[last_layer];
    for (int n = 0; n < nradmax; n++) {
      for (int l = 0; l <= lmax; l++) {
        const int j = n * (lmax + 1) + l;
        NNScalar sum = 0.0, dsum = 0.0;
        for (int k = 0; k < n_last_hidden; k++) {
          const NNScalar w = d_mlp_rad_W(last_layer, k, j);
          sum += w * h_cur[k];
          dsum += w * dh_cur[k];
        }
        d_R_nl(ii, jj, n, l) = sum * out_norm;
        d_dR_nl(ii, jj, n, l) = dsum * out_norm;
      }
    }
  }
}

#ifdef KOKKOS_ENABLE_CUDA
// ======================================================================
// Batched cuBLAS radial-MLP support kernels (CUDA + NNScalar=float).
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPAssemble,
    const int b) const
{
  const int bond_stride = (int) d_radial_basis.extent(1);
  const int ii = b / bond_stride;
  const int jj = b - ii * bond_stride;
  const int n_in = nradbase;
  if (jj >= d_ncount(ii)) {
    for (int c = 0; c < n_in; c++) {
      d_mlp_X(b, c) = NNScalar(0.0);
      d_mlp_dX(b, c) = NNScalar(0.0);
    }
    return;
  }
  for (int c = 0; c < n_in; c++) {
    d_mlp_X(b, c) = (NNScalar) d_radial_basis(ii, jj, c);
    d_mlp_dX(b, c) = (NNScalar) d_dradial_basis(ii, jj, c);
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPActDeriv,
    const int idx) const
{
  const int nout = mlp_act_n;
  const int b = idx / nout;
  const int j = idx - b * nout;
  const NNScalar norm = mlp_act_norm;
  if (mlp_act_layer == 0) {
    const NNScalar s = d_mlp_h0(b, j) * norm;
    const NNScalar ds = d_mlp_dh0(b, j) * norm;
    const NNScalar sig = NNScalar(1.0) / (NNScalar(1.0) + Kokkos::exp(-s));
    d_mlp_h0(b, j) = s * sig;
    d_mlp_dh0(b, j) = sig * (NNScalar(1.0) + s * (NNScalar(1.0) - sig)) * ds;
  } else {
    const NNScalar s = d_mlp_h1(b, j) * norm;
    const NNScalar ds = d_mlp_dh1(b, j) * norm;
    const NNScalar sig = NNScalar(1.0) / (NNScalar(1.0) + Kokkos::exp(-s));
    d_mlp_h1(b, j) = s * sig;
    d_mlp_dh1(b, j) = sig * (NNScalar(1.0) + s * (NNScalar(1.0) - sig)) * ds;
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPScatterR,
    const int idx) const
{
  const int lm1 = lmax + 1;
  const int D3 = nradmax * lm1;
  const int bs = (int) d_radial_basis.extent(1);
  const int b = idx / D3;
  const int j = idx - b * D3;
  const int ii = b / bs;
  const int jj = b - ii * bs;
  if (jj >= d_ncount(ii)) return;
  const int n = j / lm1;
  const int l = j - n * lm1;
  const NNScalar val = d_mlp_R(b, j);
  if (mlp_scatter_deriv == 0)
    d_R_nl(ii, jj, n, l) = val;
  else
    d_dR_nl(ii, jj, n, l) = val;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::mlp_sgemm(
    const NNScalar* Wg, const NNScalar* In, NNScalar* C,
    int n_out, int M, int n_in, int ldW, int ldIn, int ldC, NNScalar alpha)
{
  const float a = (float) alpha, beta = 0.0f;
  blas::status_t st = blas::sgemm(blas_handle, blas::OP_N, blas::OP_N,
      n_out, M, n_in, &a,
      (const float*) Wg, ldW,
      (const float*) In, ldIn,
      &beta, (float*) C, ldC);
  if (st != blas::SUCCESS)
    GRACE_KK_ERROR("GRACE-1L/KK: radial MLP cublasSgemm failed (status {})", (int) st);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::mlp_cublas_selftest()
{
  const auto &layers = grace_model->mlp_rad_layers;
  if ((int) layers.size() < 1) return;
  const int n_in = layers[0].n_in, n_out = layers[0].n_out;
  const auto &Wg = d_mlp_Wg[0];
  const double err = blas::gemm_selftest<NNScalar, DeviceType>("g1l", n_in, n_out, layers[0].W.data(),
      [&](const NNScalar *X, NNScalar *C, int M) {
        mlp_sgemm(Wg.data(), X, C, n_out, M, n_in, (int) Wg.extent(1), n_in, n_out, NNScalar(1.0));
      });
  if (err > 1.0)
    GRACE_KK_ERROR("GRACE-1L/KK: radial MLP {} SGEMM selftest failed (max error {:.3g} x the fp32 rounding bound)",
                   blas::name(), err);
}
#endif  // KOKKOS_ENABLE_CUDA

// ======================================================================
// Batched radial MLP dispatch: true-fp32 cuBLAS for CUDA fp32/Mixed 3-layer
// models; original hand kernel otherwise.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_mlp_radial()
{
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (is_cuda_v<DeviceType> && sizeof(NNScalar) == 4) {
    if (blas_handle && mlp_rad_n_layers == 3) {
      const int bond_stride = (int) d_radial_basis.extent(1);
      const int M = chunk_size * bond_stride;
      mlp_M = M;
      const int nin0 = nradbase;
      const int d1 = (int) d_mlp_Wg[0].extent(1);
      const int d2 = (int) d_mlp_Wg[1].extent(1);
      const int d3 = (int) d_mlp_Wg[2].extent(1);
      const NNScalar norm0 = (NNScalar) grace_model->mlp_rad_layers[0].norm;
      const NNScalar norm1 = (NNScalar) grace_model->mlp_rad_layers[1].norm;
      const NNScalar norm2 = (NNScalar) grace_model->mlp_rad_layers[2].norm;

      Kokkos::parallel_for("ComputeMLPRadial_assemble",
          Kokkos::RangePolicy<DeviceType, TagMLPAssemble>(0, M), *this);

      mlp_sgemm(d_mlp_Wg[0].data(), d_mlp_X.data(), d_mlp_h0.data(),
          d1, M, nin0, (int) d_mlp_Wg[0].extent(1), (int) d_mlp_X.extent(1),
          (int) d_mlp_h0.extent(1), NNScalar(1.0));
      mlp_sgemm(d_mlp_Wg[0].data(), d_mlp_dX.data(), d_mlp_dh0.data(),
          d1, M, nin0, (int) d_mlp_Wg[0].extent(1), (int) d_mlp_dX.extent(1),
          (int) d_mlp_dh0.extent(1), NNScalar(1.0));

      mlp_act_layer = 0; mlp_act_n = d1; mlp_act_norm = norm0;
      Kokkos::parallel_for("ComputeMLPRadial_act",
          Kokkos::RangePolicy<DeviceType, TagMLPActDeriv>(0, M * d1), *this);

      mlp_sgemm(d_mlp_Wg[1].data(), d_mlp_h0.data(), d_mlp_h1.data(),
          d2, M, d1, (int) d_mlp_Wg[1].extent(1), (int) d_mlp_h0.extent(1),
          (int) d_mlp_h1.extent(1), NNScalar(1.0));
      mlp_sgemm(d_mlp_Wg[1].data(), d_mlp_dh0.data(), d_mlp_dh1.data(),
          d2, M, d1, (int) d_mlp_Wg[1].extent(1), (int) d_mlp_dh0.extent(1),
          (int) d_mlp_dh1.extent(1), NNScalar(1.0));

      mlp_act_layer = 1; mlp_act_n = d2; mlp_act_norm = norm1;
      Kokkos::parallel_for("ComputeMLPRadial_act",
          Kokkos::RangePolicy<DeviceType, TagMLPActDeriv>(0, M * d2), *this);

      mlp_sgemm(d_mlp_Wg[2].data(), d_mlp_h1.data(), d_mlp_R.data(),
          d3, M, d2, (int) d_mlp_Wg[2].extent(1), (int) d_mlp_h1.extent(1),
          (int) d_mlp_R.extent(1), norm2);
      mlp_scatter_deriv = 0;
      Kokkos::parallel_for("ComputeMLPRadial_scatterR",
          Kokkos::RangePolicy<DeviceType, TagMLPScatterR>(0, M * d3), *this);

      mlp_sgemm(d_mlp_Wg[2].data(), d_mlp_dh1.data(), d_mlp_R.data(),
          d3, M, d2, (int) d_mlp_Wg[2].extent(1), (int) d_mlp_dh1.extent(1),
          (int) d_mlp_R.extent(1), norm2);
      mlp_scatter_deriv = 1;
      Kokkos::parallel_for("ComputeMLPRadial_scatterDR",
          Kokkos::RangePolicy<DeviceType, TagMLPScatterR>(0, M * d3), *this);
      return;
    }
  }
#endif

  int team_size = 1;
  int vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
    team_size = 32;
  int ts = team_size;
  ts = fit_team_size<DeviceType, TagComputeMLPRadial>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
  auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadial>(
      ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
  Kokkos::parallel_for("ComputeMLPRadial", policy, *this);
}

// ======================================================================
// Kernel: ComputeAi - A[i,lm,n] = sum_j R_nl * Y_lm * Z_tr
// Adapted from GRACE-FS ComputeAi with R_nl from MLP and Z_tr from embedding
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeAi,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeAi>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;

  const int jj = slot.neigh;
  const int ncount = d_ncount(ii);
  if (jj >= ncount) return;

  const int mu_j = d_mu_j(ii, jj);

  // Compute Z_tr[n] = sum_e A_lin_W[e, n] * chem_embed[mu_j, e] * A_lin_norm
  // Z_tr has shape [nradmax]
  // This is a per-bond, per-n operation

  const GeomScalar rx = d_rhats(ii, jj, 0);
  const GeomScalar ry = d_rhats(ii, jj, 1);
  const GeomScalar rz = d_rhats(ii, jj, 2);

  // Local copies of file-scope constants to avoid double promotion
  const GeomScalar Y00_v = Y00;
  const GeomScalar sq3_v = sq3;
  const GeomScalar sq3o2_v = sq3o2;
  const GeomScalar sq2_v = sq2;

  int plm_size = (lmax + 1) * (lmax + 2) / 2;
  GeomScalar* plm = (GeomScalar*) team.thread_scratch(0).get_shmem(
      scratch_bytes<DeviceType, GeomScalar>(plm_size));

  // ---- Compute associated Legendre polynomials ----
  plm[0] = Y00_v;
  if (lmax > 0) {
    plm[1] = Y00_v * sq3_v * rz;
    plm[2] = -sq3o2_v * Y00_v;
    for (int l = 2; l <= lmax; l++) {
      for (int m = 0; m < l - 1; m++) {
        const int idx = l * (l + 1) / 2 + m;
        const int idx_l1_m = (l - 1) * l / 2 + m;
        const int idx_l2_m = (l - 2) * (l - 1) / 2 + m;
        const int alm_idx = d_idx_sph(l * (l + 1) + m);
        plm[idx] = alm(alm_idx) * (rz * plm[idx_l1_m] + blm(alm_idx) * plm[idx_l2_m]);
      }
      {
        const int idx = l * (l + 1) / 2 + l - 1;
        plm[idx] = dl(l) * plm[(l - 1) * l / 2 + l - 1] * rz;
      }
      {
        const int idx = l * (l + 1) / 2 + l;
        plm[idx] = cl(l) * plm[(l - 1) * l / 2 + l - 1];
      }
    }
  }

  // ---- Convert to real Y_lm and accumulate A ----
  // For each (l, m), compute Y_lm and add R_nl[n,l] * Y_lm * Z_tr[n] to A[ii, lm, n]

  const GeomScalar phase_re = rx;
  const GeomScalar phase_im = ry;

  // m = 0
  for (int l = 0; l <= lmax; l++) {
    const GeomScalar Y = plm[l * (l + 1) / 2];
    const int A_idx = l * (l + 1);  // m=0
    for (int n = 0; n < nradmax; n++) {
      Kokkos::atomic_add(&d_A(ii, n, A_idx),
                          (NNScalar)(d_R_nl(ii, jj, n, l) * Y) * d_z_tr(mu_j, n));
    }
  }

  // m >= 1
  GeomScalar phasem_re = phase_re;
  GeomScalar phasem_im = phase_im;

  for (int m = 1; m <= lmax; m++) {
    if (m >= 2) {
      const GeomScalar tmp_re = phasem_re * phase_re - phasem_im * phase_im;
      const GeomScalar tmp_im = phasem_re * phase_im + phasem_im * phase_re;
      phasem_re = tmp_re;
      phasem_im = tmp_im;
    }

    const int factor = (m % 2 == 0) ? 1 : -1;

    for (int l = m; l <= lmax; l++) {
      const GeomScalar plm_val = plm[l * (l + 1) / 2 + m];
      const GeomScalar ylm_re = phasem_re * plm_val;
      const GeomScalar ylm_im = phasem_im * plm_val;

      const GeomScalar real_Y_pos = sq2_v * factor * ylm_re;
      const GeomScalar real_Y_neg = sq2_v * factor * ylm_im;

      const int A_idx_pos = l * (l + 1) + m;
      const int A_idx_neg = l * (l + 1) - m;

      for (int n = 0; n < nradmax; n++) {
        const NNScalar R = d_R_nl(ii, jj, n, l);
        Kokkos::atomic_add(&d_A(ii, n, A_idx_pos), R * (NNScalar)real_Y_pos * d_z_tr(mu_j, n));
        Kokkos::atomic_add(&d_A(ii, n, A_idx_neg), R * (NNScalar)real_Y_neg * d_z_tr(mu_j, n));
      }
    }
  }
}

// ======================================================================
// Kernel: ComputeMLPEnergy - E = MLP(rho[1:]) + linear(rho[0]) + shift
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeMLPEnergyR, const int ii) const
{
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));

  // LinMLPOut2ScalarTarget: input is rho (17 channels)
  // rho[0] is the "linear" channel (added directly)
  // rho[1:] go through MLP
  const NNScalar linear_term = d_rho(ii, 0);

  NNScalar ebuf_a[GRACE1L_MAX_MLP_DIM], ebuf_b[GRACE1L_MAX_MLP_DIM];
  NNScalar* h_cur = ebuf_a;
  NNScalar* h_nxt = ebuf_b;

  // Initialize h_cur = rho[1:] (skip rho[0])
  const int nin0 = d_energy_dims(0);
  for (int k = 0; k < nin0; k++)
    h_cur[k] = d_rho(ii, k + 1);

  // Hidden layers: 0 .. energy_n_layers-2 (silu or tanh activation)
  for (int layer = 0; layer < energy_n_layers - 1; layer++) {
    const int nin  = d_energy_dims(layer);
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

  // Output layer (last layer, no activation)
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

  // Total: E_atom = (MLP_out + linear_term) * output_scale + shift, in fp64 (KKA) as on the CPU: an fp32 sum rounds
  // E_atom to ~1e-6 eV
  d_e_atom(ii) = ((KKA) e + (KKA) linear_term) * (KKA) output_scale + (KKA) d_shifts(mu_i);
}

// ======================================================================
// Kernel: ComputeUQ (schema v6 basis-RP) - per-atom extrapolation grade (gamma)
// and sigma. The GMM feature phi = concat(proj, dens) where proj = (B/||B||)·R is
// the L2-normalized invariant B-basis projected through R, and dens are log-norm
// density channels [log||B||, log||B^(0)||, ...]·density_scale. b_inv is the
// gathered L1 (rho) scalar B-basis products d_A..d_AAAA in the exact (source,
// n_out, collect) order the rho reduce iterates them — Python's reshape order — so
// the running R-row index b matches uq_rp_matrix's layout. 1L is single-phase
// (d_A.. are current here) and has a single invariant block, so ||B|| == ||B^(0)||.
// Normalization exploits linearity: project the raw B once, accumulate ||B||^2 in
// the same pass, then divide proj by ||B|| — no need to materialize B_hat. Then run
// the GMM: nearest centroid, Mahalanobis sigma, gamma = sigma/threshold. Forward
// only; gamma is the sole UQ signal. The energy hot path is untouched and the no-UQ
// run pays nothing (any_uq-gated launch).
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeUQ,
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

  // Team-shared scratch: normalized feature f[D] and Mahalanobis delta[D], plus
  // a scalar for ||B|| shared to the density channels.
  double* f      = (double*) team.team_shmem().get_shmem(D * sizeof(double));
  double* delta  = (double*) team.team_shmem().get_shmem(D * sizeof(double));
  double* nrm_sh = (double*) team.team_shmem().get_shmem(sizeof(double));

  // ---- Projection proj[d] = (B·R)[d] + ||B||, streamed over the L1 scalar basis ----
  // Each lane owns projection column(s) d = lane, lane+TS, ... and walks the full
  // scalar-basis product stream once per column, accumulating that column in a
  // register (no dynamically-indexed per-thread array => no local-memory spill,
  // which was the old 1-thread/atom kernel's dominant cost). ||B||^2 is recomputed
  // per column (identical value, cheap) so no cross-lane reduction is needed.
  // TODO(dedup): the (source, n_out, collect) walk order + schema-v6 load() checks
  // are duplicated across all KK styles (1l/1l_cpu/2l/2l_cpu/3l). The order is
  // coupled to the exporter reshape; the load-time d_basis check guards width, not
  // ordering. A shared helper would remove the drift risk. Follow-up refactor.
  for (int d = lane; d < RP; d += TS) {
    double acc = 0.0;
    double n2  = 0.0;
    int b = 0;
    if (uq_sgemm) {    // fp32 engines: fp32 products and sums against the fp32 R, the precision of the SGEMM path
      float a = 0.0f, q = 0.0f;
      #define GRACE1L_UQ_WALK_F(XVIEW, CIVIEW)                                        \
        { const int nin = (int)(XVIEW).extent(1); const int nc = (int)(CIVIEW).extent(0); \
          for (int n = 0; n < nin; n++)                                            \
            for (int c = 0; c < nc; c++) {                                         \
              const float bval = (float)(XVIEW)(ii, n, (CIVIEW)(c));               \
              q += bval * bval;                                                    \
              a += bval * d_uq_rp_matrix_f((size_t) b * RP + d);                   \
              b++;                                                                 \
            } }
      GRACE1L_UQ_WALK_F(d_A,    d_reduce_A_ci);
      GRACE1L_UQ_WALK_F(d_AA,   d_reduce_AA_ci);
      GRACE1L_UQ_WALK_F(d_AAA,  d_reduce_AAA_ci);
      GRACE1L_UQ_WALK_F(d_AAAA, d_reduce_AAAA_ci);
      #undef GRACE1L_UQ_WALK_F
      acc = a;
      n2 = q;
    } else {
    #define GRACE1L_UQ_WALK(XVIEW, CIVIEW)                                          \
      { const int nin = (int)(XVIEW).extent(1); const int nc = (int)(CIVIEW).extent(0); \
        for (int n = 0; n < nin; n++)                                              \
          for (int c = 0; c < nc; c++) {                                           \
            const double bval = (double)(XVIEW)(ii, n, (CIVIEW)(c));               \
            n2  += bval * bval;                                                    \
            acc += bval * d_uq_rp_matrix(b, d);                                    \
            b++;                                                                   \
          } }
    GRACE1L_UQ_WALK(d_A,    d_reduce_A_ci);
    GRACE1L_UQ_WALK(d_AA,   d_reduce_AA_ci);
    GRACE1L_UQ_WALK(d_AAA,  d_reduce_AAA_ci);
    GRACE1L_UQ_WALK(d_AAAA, d_reduce_AAAA_ci);
    #undef GRACE1L_UQ_WALK
    }
    // L2-normalize: proj = (B/||B||)·R = (B·R)/||B||
    if (uq_normalize) acc /= (Kokkos::sqrt(n2) + 1.0e-12);
    f[d] = acc;
    if (d == 0) *nrm_sh = Kokkos::sqrt(n2);   // ||B|| for the density channels
  }
  team.team_barrier();

  // --- Density channels appended after the projection: [full, block0, ...] ---
  // 1L has one invariant block, so every block norm equals ||B||.
  const double density = Kokkos::log(*nrm_sh + 1.0e-12) * uq_density_scale;
  for (int j = lane; j < uq_n_density; j += TS) f[RP + j] = density;
  team.team_barrier();

  // --- Cluster assignment: nearest centroid (squared Euclidean over D) ---
  // ncl > 0 for every element is guaranteed at load time (uqv6 covers all elements).
  // Each TeamThreadRange reduce broadcasts d2 to every lane, so kstar is identical
  // across the team without an explicit barrier.
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

  // --- Mahalanobis distance to assigned cluster: sig2 = deltaᵀ · Σ⁻¹ · delta ---
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

#ifdef KOKKOS_ENABLE_CUDA
// ======================================================================
// Kernel: UQBmat - materialize the dense invariant-basis matrix for one
// sub-block of atoms so the projection can be a single cuBLAS DGEMM.
// league = sub-block atom, team lanes split the D_basis basis rows: each
// gathers b_inv[b] = view(ii, n, collect) via the precomputed walk map and
// writes it col-major into d_uq_Bmat[b + D_basis*ii_local] (coalesced across
// lanes), team-reducing ||B||^2 for the later L2 normalization. This replaces
// ComputeUQ's per-lane re-walk of the whole basis (128x redundant indirect
// gather) with one coalesced gather + a BLAS GEMM.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagUQBmat,
    const typename Kokkos::TeamPolicy<DeviceType, TagUQBmat>::member_type& team) const
{
  const int ii_local = team.league_rank();
  const int ii = uq_bmat_s0 + ii_local;
  if (ii >= chunk_size) return;
  const int Db = uq_d_basis;
  const size_t base = (size_t) Db * ii_local;

  double n2 = 0.0;
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, Db),
    [&](const int b, double& s) {
      const int vid = d_uq_map_view(b);
      const int n   = d_uq_map_n(b);
      const int col = d_uq_map_col(b);
      double v;
      if      (vid == 0) v = (double) d_A(ii, n, col);
      else if (vid == 1) v = (double) d_AA(ii, n, col);
      else if (vid == 2) v = (double) d_AAA(ii, n, col);
      else               v = (double) d_AAAA(ii, n, col);
      if (uq_sgemm) d_uq_Bmat_f(base + b) = (float) v;
      else d_uq_Bmat(base + b) = v;
      s += v * v;
    }, n2);
  Kokkos::single(Kokkos::PerTeam(team), [&]() { d_uq_n2(ii_local) = n2; });
}

// ======================================================================
// Kernel: UQFinish - from the cuBLAS projection proj[rp_dim x subN] (col-major),
// L2-normalize by ||B||, append the density channels, and run the GMM (nearest
// centroid + Mahalanobis -> gamma). Identical GMM math to ComputeUQ; only the
// projection source changes (BLAS output vs the in-kernel walk).
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagUQFinish,
    const typename Kokkos::TeamPolicy<DeviceType, TagUQFinish>::member_type& team) const
{
  const int ii_local = team.league_rank();
  const int ii = uq_bmat_s0 + ii_local;
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));
  const int D  = uq_D;
  const int RP = uq_rp_dim;
  const int TS = team.team_size();
  const int lane = team.team_rank();

  double* f     = (double*) team.team_shmem().get_shmem(D * sizeof(double));
  double* delta = (double*) team.team_shmem().get_shmem(D * sizeof(double));

  const double nrm = Kokkos::sqrt(d_uq_n2(ii_local));
  const double inv = uq_normalize ? (1.0 / (nrm + 1.0e-12)) : 1.0;
  const size_t pbase = (size_t) RP * ii_local;
  for (int d = lane; d < RP; d += TS)
    f[d] = (uq_sgemm ? (double) d_uq_proj_f(pbase + d) : d_uq_proj(pbase + d)) * inv;

  const double density = Kokkos::log(nrm + 1.0e-12) * uq_density_scale;
  for (int j = lane; j < uq_n_density; j += TS) f[RP + j] = density;
  team.team_barrier();

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
#endif  // KOKKOS_ENABLE_CUDA

// ======================================================================
// Kernel: ComputeDerivative - per-bond forces from d_A_adj
// Uses forward-mode AD through MLP for dR_nl/dr + SH derivatives
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeDerivative,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeDerivative>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;

  const int jj = slot.neigh;
  const int ncount = d_ncount(ii);
  if (jj >= ncount) return;

  const int mu_j = d_mu_j(ii, jj);
  const GeomScalar rinv = GeomScalar(1.0) / d_rnorms(ii, jj);
  const GeomScalar rx = d_rhats(ii, jj, 0);
  const GeomScalar ry = d_rhats(ii, jj, 1);
  const GeomScalar rz = d_rhats(ii, jj, 2);

  // Local copies of file-scope constants to avoid double promotion
  const GeomScalar Y00_v = Y00;
  const GeomScalar sq3_v = sq3;
  const GeomScalar sq3o2_v = sq3o2;
  const GeomScalar sq2_v = sq2;

  // ---- Scratch: plm + dplm (z_tr precomputed in d_z_tr) ----
  int plm_size = (lmax + 1) * (lmax + 2) / 2;
  GeomScalar* scratch_geom = (GeomScalar*) team.thread_scratch(0).get_shmem(
      2 * plm_size * sizeof(GeomScalar));
  GeomScalar* plm = scratch_geom;
  GeomScalar* dplm = scratch_geom + plm_size;

  int n_last_hidden = 0;
  int last_layer = 0;
  NNScalar norm_last = NNScalar(0.0);
  NNScalar dh2_cache[GRACE1L_MAX_MLP_DIM];
  if constexpr (std::is_same<NNScalar, double>::value) {
    n_last_hidden = d_mlp_rad_dims(mlp_rad_n_layers - 1);
    last_layer = mlp_rad_n_layers - 1;
    norm_last = d_mlp_rad_norms(last_layer);
    for (int k = 0; k < n_last_hidden; k++) dh2_cache[k] = d_dh2(ii, jj, k);
  }

  plm[0] = Y00_v; dplm[0] = GeomScalar(0.0);
  if (lmax > 0) {
    plm[1] = Y00_v * sq3_v * rz;     dplm[1] = Y00_v * sq3_v;
    plm[2] = -sq3o2_v * Y00_v;       dplm[2] = GeomScalar(0.0);
    for (int l = 2; l <= lmax; l++) {
      for (int m = 0; m < l - 1; m++) {
        const int idx = l*(l+1)/2 + m;
        const int i1 = (l-1)*l/2 + m, i2 = (l-2)*(l-1)/2 + m;
        const int ai = d_idx_sph(l*(l+1) + m);
        const GeomScalar a = alm(ai), b = blm(ai);
        plm[idx] = a * (rz * plm[i1] + b * plm[i2]);
        dplm[idx] = a * (plm[i1] + rz * dplm[i1] + b * dplm[i2]);
      }
      { const int idx = l*(l+1)/2+l-1, prev = (l-1)*l/2+l-1;
        const GeomScalar t = dl(l) * plm[prev];
        plm[idx] = t * rz;  dplm[idx] = t; }
      { const int idx = l*(l+1)/2+l, prev = (l-1)*l/2+l-1;
        plm[idx] = cl(l) * plm[prev];  dplm[idx] = 0.0; }
    }
  }

  // ---- Cache z_tr[mu_j, n] (read ~25 times per (ii,jj,n) across loops). ----
  NNScalar zn_cache[GRACE1L_MAX_NRADMAX];
  for (int n = 0; n < nradmax; n++) zn_cache[n] = d_z_tr(mu_j, n);

  // ---- Accumulate per-bond force ----
  // Restructured loop: (l, m) outermost, n innermost
  // Angular derivatives depend only on (l, m, rhat) — computed once per (l,m)
  // DR_n[n] and R_n[n] precomputed per l value
  GeomScalar f_ji[3] = {GeomScalar(0.0), GeomScalar(0.0), GeomScalar(0.0)};
  const GeomScalar phase_re = rx, phase_im = ry;

  NNScalar DR_n[GRACE1L_MAX_NRADMAX];
  GeomScalar R_n[GRACE1L_MAX_NRADMAX];

  for (int l = 0; l <= lmax; l++) {
    // Precompute DR_n[n] and R_n[n] for this l
    for (int n = 0; n < nradmax; n++) {
      R_n[n] = d_R_nl(ii, jj, n, l);
      if constexpr (std::is_same<NNScalar, double>::value) {
        const int j = n * (lmax + 1) + l;
        NNScalar dr = 0.0;
        for (int k = 0; k < n_last_hidden; k++)
          dr += d_mlp_rad_W(last_layer, k, j) * dh2_cache[k];
        DR_n[n] = dr * norm_last;
      } else {
        DR_n[n] = d_dR_nl(ii, jj, n, l);
      }
    }

    // m = 0: angular derivatives computed once
    {
      const GeomScalar Y = plm[l*(l+1)/2];
      const GeomScalar dp = dplm[l*(l+1)/2];
      const GeomScalar rdy = dp * rz;
      const GeomScalar DY_x = -rdy * rx, DY_y = -rdy * ry, DY_z = dp - rdy * rz;
      const int A_idx = l * (l + 1);

      for (int n = 0; n < nradmax; n++) {
        const NNScalar w = d_A_adj(ii, n, A_idx) * zn_cache[n];
        const GeomScalar R_over_r = R_n[n] * rinv;
        const GeomScalar YDR = Y * DR_n[n];
        f_ji[0] += w * (YDR * rx + DY_x * R_over_r);
        f_ji[1] += w * (YDR * ry + DY_y * R_over_r);
        f_ji[2] += w * (YDR * rz + DY_z * R_over_r);
      }
    }

    // m >= 1: angular derivatives computed once per m, then n-inner loop
    GeomScalar pm_re = phase_re, pm_im = phase_im;
    for (int m = 1; m <= l; m++) {
      const int fac = (m % 2 == 0) ? 1 : -1;
      const GeomScalar pv = plm[l*(l+1)/2+m], dpv = dplm[l*(l+1)/2+m];
      const GeomScalar ylm_re = pm_re * pv, ylm_im = pm_im * pv;
      const GeomScalar rYp = sq2_v * fac * ylm_re, rYn = sq2_v * fac * ylm_im;

      // Angular derivatives (independent of n — computed once)
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
          const GeomScalar mp = (GeomScalar)m * pv;
          dyx_re = mp*p1r; dyx_im = mp*p1i;
          dyy_re = -dyx_im; dyy_im = dyx_re;
        } else {
          dyx_re = dyx_im = dyy_re = dyy_im = GeomScalar(0.0);
        }
      }

      const GeomScalar rdy_re = rx*dyx_re + ry*dyy_re + rz*dyz_re;
      const GeomScalar rdy_im = rx*dyx_im + ry*dyy_im + rz*dyz_im;
      const GeomScalar Dpx_re = dyx_re - rdy_re*rx, Dpx_im = dyx_im - rdy_im*rx;
      const GeomScalar Dpy_re = dyy_re - rdy_re*ry, Dpy_im = dyy_im - rdy_im*ry;
      const GeomScalar Dpz_re = dyz_re - rdy_re*rz, Dpz_im = dyz_im - rdy_im*rz;
      const GeomScalar DYpx = sq2_v*fac*Dpx_re, DYpy = sq2_v*fac*Dpy_re, DYpz = sq2_v*fac*Dpz_re;
      const GeomScalar DYnx = sq2_v*fac*Dpx_im, DYny = sq2_v*fac*Dpy_im, DYnz = sq2_v*fac*Dpz_im;

      const int Ap = l*(l+1)+m, An = l*(l+1)-m;

      for (int n = 0; n < nradmax; n++) {
        const NNScalar zn = zn_cache[n];
        const NNScalar wp = d_A_adj(ii, n, Ap) * zn;
        const NNScalar wn = d_A_adj(ii, n, An) * zn;
        const GeomScalar R_over_r = R_n[n] * rinv;
        {
          const GeomScalar YDR = rYp * DR_n[n];
          f_ji[0] += wp * (YDR*rx + DYpx*R_over_r);
          f_ji[1] += wp * (YDR*ry + DYpy*R_over_r);
          f_ji[2] += wp * (YDR*rz + DYpz*R_over_r);
        }
        {
          const GeomScalar YDR = rYn * DR_n[n];
          f_ji[0] += wn * (YDR*rx + DYnx*R_over_r);
          f_ji[1] += wn * (YDR*ry + DYny*R_over_r);
          f_ji[2] += wn * (YDR*rz + DYnz*R_over_r);
        }
      }

      const GeomScalar t_re = pm_re*phase_re - pm_im*phase_im;
      const GeomScalar t_im = pm_re*phase_im + pm_im*phase_re;
      pm_re = t_re; pm_im = t_im;
    }
  }

  d_f_ij(ii, jj, 0) = f_ji[0];
  d_f_ij(ii, jj, 1) = f_ji[1];
  d_f_ij(ii, jj, 2) = f_ji[2];
}

// ======================================================================
// Kernel: ComputeForce (placeholder - force done via lambda in compute())
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeForce<NEIGHFLAG,EVFLAG>, const int& /*ii*/) const
{
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeForce<NEIGHFLAG,EVFLAG>, const int& /*ii*/, EV& /*ev*/) const
{
}

#ifdef KOKKOS_ENABLE_CUDA
// ======================================================================
// cuBLAS true-fp32 block-sparse FC forward (mirror of 2l Opt-2L-17).
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::fc_cublas_selftest()
{
  const int n_out = 3, n_in = 2, M = 4;
  t_nn_1d W("g1l:fc_st_W", (size_t) n_out * n_in);   // col-major [n_out x n_in]
  t_nn_1d G("g1l:fc_st_G", (size_t) n_in * M);       // col-major [n_in  x M]
  t_nn_1d C("g1l:fc_st_C", (size_t) n_out * M);      // col-major [n_out x M]
  auto hW = Kokkos::create_mirror_view(W);
  auto hG = Kokkos::create_mirror_view(G);
  for (int n = 0; n < n_in; n++)
    for (int k = 0; k < n_out; k++)
      hW(k + (size_t) n_out * n) = (NNScalar)(0.1 * (double)(((k * 5 + n * 3) % 7) - 3));
  for (int j = 0; j < M; j++)
    for (int n = 0; n < n_in; n++)
      hG(n + (size_t) n_in * j) = (NNScalar)(0.01 * (double)(((j * 7 + n * 2) % 11) - 5));
  Kokkos::deep_copy(W, hW); Kokkos::deep_copy(G, hG);

  const float one = 1.0f, zero = 0.0f;
  blas::status_t st = blas::sgemm(blas_handle, blas::OP_N, blas::OP_N,
      n_out, M, n_in, &one,
      (const float*) W.data(), n_out,
      (const float*) G.data(), n_in,
      &zero, (float*) C.data(), n_out);
  if (st != blas::SUCCESS)
    GRACE_KK_ERROR("GRACE-1L/KK: FC selftest {} SGEMM failed (status {})", blas::name(), (int) st);
  Kokkos::fence();

  auto hC = Kokkos::create_mirror_view(C);
  Kokkos::deep_copy(hC, C);
  const double err = blas::selftest_error(M, n_in, n_out,
      [&](int j, int n) { return hG(n + (size_t) n_in * j); }, [&](int n, int k) { return hW(k + (size_t) n_out * n); },
      [&](int j, int k) { return hC(k + (size_t) n_out * j); });
  if (err > 1.0)
    GRACE_KK_ERROR("GRACE-1L/KK: FC {} SGEMM selftest failed (max error {:.3g} x the fp32 rounding bound)",
                   blas::name(), err);
}

// Build tile-sorted metadata + repacked per-tile weights for one block-sparse FC
// (host-side, from the model FCWeights). Identical to the 2l build. See FCCublasOp.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::build_fc_cublas_op(
    FCCublasOp &op, const GRACE1LModel::FCWeights &fc)
{
  const int n_out = fc.n_out;
  const int nt    = fc.w_shape_left;               // # tiles (== w_shape_right)
  if (n_out <= 0 || nt <= 0) return;
  const int n_in_l = (int)(fc.w_left.size()  / ((size_t) n_out * nt));
  const int n_in_r = (int)(fc.w_right.size() / ((size_t) n_out * nt));
  const int N_l = fc.n_funcs_left;                 // left entries: lm = 0..N_l-1
  const int N_r = (int) fc.collect_from.size();    // right entries: idx = 0..N_r-1
  op.n_out = n_out; op.ntiles = nt;
  op.n_in_l = n_in_l; op.n_in_r = n_in_r;
  op.N_l = N_l; op.N_r = N_r;
  op.nl = (NNScalar) fc.norm_left; op.nr = (NNScalar) fc.norm_right;

  // per-output-function norm factor
  op.nof = t_nn_1d("g1l:fc_nof", std::max(N_l, 1));
  { auto h = Kokkos::create_mirror_view(op.nof);
    for (int i = 0; i < N_l; i++)
      h(i) = (NNScalar)(i < (int) fc.norm_out_factor.size() ? fc.norm_out_factor[i] : 0.0);
    Kokkos::deep_copy(op.nof, h); }

  // LEFT: stable-sort lm by tile wtl(lm) -> tile-contiguous columns
  op.L_src = t_int_1d("g1l:fc_Lsrc", std::max(N_l, 1));
  op.L_pos = t_int_1d("g1l:fc_Lpos", std::max(N_l, 1));
  op.L_pstart.assign(nt, 0); op.L_count.assign(nt, 0);
  { std::vector<int> order(N_l);
    for (int i = 0; i < N_l; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
        [&](int a, int b){ return fc.w_tile_left[a] < fc.w_tile_left[b]; });
    auto hsrc = Kokkos::create_mirror_view(op.L_src);
    auto hpos = Kokkos::create_mirror_view(op.L_pos);
    for (int p = 0; p < N_l; p++) {
      const int lm = order[p];
      hsrc(p) = lm;                  // left: src == tgt == lm
      hpos(lm) = p;
      op.L_count[fc.w_tile_left[lm]]++;
    }
    int acc = 0; for (int t = 0; t < nt; t++) { op.L_pstart[t] = acc; acc += op.L_count[t]; }
    Kokkos::deep_copy(op.L_src, hsrc); Kokkos::deep_copy(op.L_pos, hpos); }

  // RIGHT: stable-sort idx by tile wtr(idx). R_tgt kept in ORIGINAL idx order
  // (the scatter loops idx = 0..N_r-1 exactly like the hand kernel).
  op.R_src = t_int_1d("g1l:fc_Rsrc", std::max(N_r, 1));
  op.R_pos = t_int_1d("g1l:fc_Rpos", std::max(N_r, 1));
  op.R_tgt = t_int_1d("g1l:fc_Rtgt", std::max(N_r, 1));
  op.R_pstart.assign(nt, 0); op.R_count.assign(nt, 0);
  { std::vector<int> order(N_r);
    for (int i = 0; i < N_r; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
        [&](int a, int b){ return fc.w_tile_right[a] < fc.w_tile_right[b]; });
    auto hsrc = Kokkos::create_mirror_view(op.R_src);
    auto hpos = Kokkos::create_mirror_view(op.R_pos);
    auto htgt = Kokkos::create_mirror_view(op.R_tgt);
    for (int p = 0; p < N_r; p++) {
      const int idx = order[p];
      hsrc(p) = fc.collect_from[idx];
      hpos(idx) = p;
      op.R_count[fc.w_tile_right[idx]]++;
    }
    for (int idx = 0; idx < N_r; idx++) htgt(idx) = fc.collect_to[idx];
    int acc = 0; for (int t = 0; t < nt; t++) { op.R_pstart[t] = acc; acc += op.R_count[t]; }
    Kokkos::deep_copy(op.R_src, hsrc); Kokkos::deep_copy(op.R_pos, hpos);
    Kokkos::deep_copy(op.R_tgt, htgt); }

  // Repack weights to column-major per-tile slabs: packed[t*n_out*n_in + n*n_out + k]
  // = fc.w_{left,right}[k*(n_in*nt) + n*nt + t] (native flat [n_out][n_in][nt]).
  { const size_t szl = (size_t) nt * n_out * n_in_l;
    op.Wl_packed = t_nn_1d("g1l:fc_Wl", szl);
    auto h = Kokkos::create_mirror_view(op.Wl_packed);
    for (int k = 0; k < n_out; k++)
      for (int n = 0; n < n_in_l; n++)
        for (int t = 0; t < nt; t++)
          h((size_t) t * n_out * n_in_l + (size_t) n * n_out + k) =
              (NNScalar) fc.w_left[(size_t) k * n_in_l * nt + (size_t) n * nt + t];
    Kokkos::deep_copy(op.Wl_packed, h); }
  { const size_t szr = (size_t) nt * n_out * n_in_r;
    op.Wr_packed = t_nn_1d("g1l:fc_Wr", szr);
    auto h = Kokkos::create_mirror_view(op.Wr_packed);
    for (int k = 0; k < n_out; k++)
      for (int n = 0; n < n_in_r; n++)
        for (int t = 0; t < nt; t++)
          h((size_t) t * n_out * n_in_r + (size_t) n * n_out + k) =
              (NNScalar) fc.w_right[(size_t) k * n_in_r * nt + (size_t) n * nt + t];
    Kokkos::deep_copy(op.Wr_packed, h); }

  fc_cublas_max_nin  = std::max(fc_cublas_max_nin,  std::max(n_in_l, n_in_r));
  fc_cublas_max_nout = std::max(fc_cublas_max_nout, n_out);
  fc_cublas_max_N    = std::max(fc_cublas_max_N,    std::max(N_l, N_r));
  op.active = true;
}

// Run one forward FC (left assign + right accumulate) via gather -> per-tile
// true-fp32 SGEMM -> scatter, into out (float only). The scatter is (ii,k)-
// parallel and loops functions in the same order as the hand kernel (no atomics)
// -> only the Σ_n contraction order changes vs the hand kernel.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::fc_forward_cublas(
    FCCublasOp &op, const t_nn_3d &in_left, const t_nn_3d &in_right, const t_nn_3d &out)
{
  const int cs = chunk_size;
  const int n_out = op.n_out;
  const float one = 1.0f, zero = 0.0f;

  // ---------- LEFT: gather -> per-tile GEMM -> scatter (assign) ----------
  {
    const int nin = op.n_in_l, Nl = op.N_l;
    { auto G = d_fc_G; auto L_src = op.L_src; auto inL = in_left;
      Kokkos::parallel_for("FC_cublas_gatherL",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {Nl*cs, nin}),
        KOKKOS_LAMBDA(const int col, const int n) {
          const int p = col / cs, ii = col - p*cs;
          G((size_t) col * nin + n) = inL(ii, n, L_src(p));
        }); }
    for (int t = 0; t < op.ntiles; t++) {
      const int m = op.L_count[t]; if (!m) continue;
      const int colbase = op.L_pstart[t] * cs, Mt = m * cs;
      blas::status_t st = blas::sgemm(blas_handle, blas::OP_N, blas::OP_N,
          n_out, Mt, nin, &one,
          (const float*) op.Wl_packed.data() + (size_t) t * n_out * nin, n_out,
          (const float*) d_fc_G.data() + (size_t) colbase * nin, nin,
          &zero, (float*) d_fc_C.data() + (size_t) colbase * n_out, n_out);
      if (st != blas::SUCCESS)
        GRACE_KK_ERROR("GRACE-1L/KK: FC cublasSgemm(L) failed (status {})", (int) st);
    }
    { auto C = d_fc_C; auto L_pos = op.L_pos; auto nof = op.nof; auto outv = out;
      const NNScalar nl = op.nl; const int Nl2 = Nl, no = n_out, cs2 = cs;
      Kokkos::parallel_for("FC_cublas_scatterL",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, n_out}),
        KOKKOS_LAMBDA(const int ii, const int k) {
          for (int lm = 0; lm < Nl2; lm++) {
            const size_t col = (size_t)(L_pos(lm) * cs2 + ii);
            outv(ii, k, lm) = C(col * no + k) * nof(lm) * nl;
          }
        }); }
  }

  // ---------- RIGHT: gather -> per-tile GEMM -> scatter (accumulate) ----------
  {
    const int nin = op.n_in_r, Nr = op.N_r;
    { auto G = d_fc_G; auto R_src = op.R_src; auto inR = in_right;
      Kokkos::parallel_for("FC_cublas_gatherR",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {Nr*cs, nin}),
        KOKKOS_LAMBDA(const int col, const int n) {
          const int p = col / cs, ii = col - p*cs;
          G((size_t) col * nin + n) = inR(ii, n, R_src(p));
        }); }
    for (int t = 0; t < op.ntiles; t++) {
      const int m = op.R_count[t]; if (!m) continue;
      const int colbase = op.R_pstart[t] * cs, Mt = m * cs;
      blas::status_t st = blas::sgemm(blas_handle, blas::OP_N, blas::OP_N,
          n_out, Mt, nin, &one,
          (const float*) op.Wr_packed.data() + (size_t) t * n_out * nin, n_out,
          (const float*) d_fc_G.data() + (size_t) colbase * nin, nin,
          &zero, (float*) d_fc_C.data() + (size_t) colbase * n_out, n_out);
      if (st != blas::SUCCESS)
        GRACE_KK_ERROR("GRACE-1L/KK: FC cublasSgemm(R) failed (status {})", (int) st);
    }
    { auto C = d_fc_C; auto R_pos = op.R_pos; auto R_tgt = op.R_tgt; auto nof = op.nof; auto outv = out;
      const NNScalar nr = op.nr; const int Nr2 = Nr, no = n_out, cs2 = cs;
      Kokkos::parallel_for("FC_cublas_scatterR",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, n_out}),
        KOKKOS_LAMBDA(const int ii, const int k) {
          for (int idx = 0; idx < Nr2; idx++) {
            const int tgt = R_tgt(idx);
            const size_t col = (size_t)(R_pos(idx) * cs2 + ii);
            outv(ii, k, tgt) += C(col * no + k) * nof(tgt) * nr;
          }
        }); }
  }
}

// ======================================================================
// cuBLAS DGEMM UQ projection driver. The B->R projection that dominated the
// small-N UQ cost (ComputeUQ re-walked all D_basis basis rows per lane, a 128x
// redundant indirect gather at ~12% occupancy) becomes: gather Bmat once
// (UQBmat) -> one fp64 GEMM proj = R·Bmat -> GMM finish (UQFinish). Processes
// the current chunk in sub-blocks of uq_subN atoms to bound the Bmat buffer
// (D_basis*subN doubles). fp64 throughout (UQ arrays are double); CUBLAS_PEDANTIC
// avoids any tf32 fallthrough so gamma stays at the fp32-basis validation floor.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_uq_cublas()
{
  if (chunk_size <= 0) return;
  const int Db = uq_d_basis, RP = uq_rp_dim;

  // ---- one-time: build the (view,n,collect) gather map replaying ComputeUQ's
  // walk exactly, so Bmat row b lines up with uq_rp_matrix row b ----
  if (!uq_map_built) {
    d_uq_map_view = t_int_1d("g1l:uq_map_view", Db);
    d_uq_map_n    = t_int_1d("g1l:uq_map_n",    Db);
    d_uq_map_col  = t_int_1d("g1l:uq_map_col",  Db);
    auto mv = d_uq_map_view; auto mn = d_uq_map_n; auto mc = d_uq_map_col;
    auto ciA = d_reduce_A_ci; auto ciAA = d_reduce_AA_ci;
    auto ciAAA = d_reduce_AAA_ci; auto ciAAAA = d_reduce_AAAA_ci;
    const int ninA = (int) d_A.extent(1),   ninAA = (int) d_AA.extent(1);
    const int ninAAA = (int) d_AAA.extent(1), ninAAAA = (int) d_AAAA.extent(1);
    const int Dbc = Db;
    Kokkos::parallel_for("UQBuildMap", Kokkos::RangePolicy<DeviceType>(0, 1),
      KOKKOS_LAMBDA(const int) {
        int b = 0;
        #define GRACE1L_UQ_MAP(VID, NIN, CIV)                                   \
          { const int nc = (int)(CIV).extent(0);                               \
            for (int n = 0; n < (NIN); n++)                                    \
              for (int c = 0; c < nc; c++) {                                   \
                if (b < Dbc) { mv(b) = (VID); mn(b) = n; mc(b) = (CIV)(c); }   \
                b++;                                                           \
              } }
        GRACE1L_UQ_MAP(0, ninA,    ciA);
        GRACE1L_UQ_MAP(1, ninAA,   ciAA);
        GRACE1L_UQ_MAP(2, ninAAA,  ciAAA);
        GRACE1L_UQ_MAP(3, ninAAAA, ciAAAA);
        #undef GRACE1L_UQ_MAP
      });
    uq_map_built = true;
  }

  // ---- (re)size sub-block scratch; grow up to the 8192-atom cap on demand ----
  const int want = std::min(8192, chunk_size);
  if (want > uq_subN) {
    uq_subN = want;
    grace::kk::MemKK::realloc_kokkos(d_uq_Bmat, "g1l:uq_Bmat", (size_t) Db * uq_subN);
    grace::kk::MemKK::realloc_kokkos(d_uq_proj, "g1l:uq_proj", (size_t) RP * uq_subN);
    if (uq_sgemm) {
      grace::kk::MemKK::realloc_kokkos(d_uq_Bmat_f, "g1l:uq_Bmat_f", (size_t) Db * uq_subN);
      grace::kk::MemKK::realloc_kokkos(d_uq_proj_f, "g1l:uq_proj_f", (size_t) RP * uq_subN);
    }
    grace::kk::MemKK::realloc_kokkos(d_uq_n2,   "g1l:uq_n2",   (size_t) uq_subN);
  }

  const double one = 1.0, zero = 0.0;
  const int scratch_bytes = (2 * uq_D + 8) * (int) sizeof(double);
  auto fprobe = Kokkos::TeamPolicy<DeviceType, TagUQFinish>(1, 1, 1)
      .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
  int fts = fprobe.team_size_max(*this, Kokkos::ParallelForTag());
  if (fts > 128) fts = 128;
  if (fts < 1)   fts = 1;

  for (int s0 = 0; s0 < chunk_size; s0 += uq_subN) {
    const int sN = std::min(uq_subN, chunk_size - s0);
    uq_bmat_s0 = s0;

    Kokkos::parallel_for("UQBmat",
        Kokkos::TeamPolicy<DeviceType, TagUQBmat>(sN, Kokkos::AUTO), *this);

    // proj[RP x sN] = R'[RP x Db] · Bmat[Db x sN]; R' is uq_rp_matrix's row-major
    // [Db,RP] buffer read as col-major [RP x Db] (lda=RP), no transpose needed.
    const float onef = 1.0f, zerof = 0.0f;
    blas::status_t st = uq_sgemm
        ? blas::sgemm(blas_handle, blas::OP_N, blas::OP_N, RP, sN, Db, &onef,
                      d_uq_rp_matrix_f.data(), RP, d_uq_Bmat_f.data(), Db, &zerof, d_uq_proj_f.data(), RP)
        : blas::dgemm(blas_handle, blas::OP_N, blas::OP_N, RP, sN, Db, &one,
                      d_uq_rp_matrix.data(), RP, d_uq_Bmat.data(), Db, &zero, d_uq_proj.data(), RP);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-1L/KK: UQ projection GEMM failed (status {})", (int) st);

    Kokkos::parallel_for("UQFinish",
        Kokkos::TeamPolicy<DeviceType, TagUQFinish>(sN, fts, 1)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)), *this);
  }
}
#endif


// ======================================================================
// Engine API (replaces the LAMMPS constructor, coeff, init_style, init_one and compute of PairGRACE1LKokkos)
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::Grace1LKK()
{
  host_flag = Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename DeviceType::memory_space>::accessible ? 1 : 0;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::~Grace1LKK()
{
  // the BLAS handle is released by its shared owner (blas_owner) with the last engine copy
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::set_model(std::shared_ptr<GRACE1LModel> model,
                                                                       const std::vector<int> &type2mu)
{
  grace_model = std::move(model);
  realloc_all = true;    // the per-atom buffers have the previous model's dims
  setup_done = false;
#ifdef KOKKOS_ENABLE_CUDA
  // the cuBLAS UQ gather map and Bmat scratch belong to the previous model's basis
  uq_map_built = false;
  uq_subN = 0;
#endif
  // A UQ head the build cannot use (schema other than 6, inconsistent arrays) leaves the engine usable without UQ
  has_uq = false;
  if (grace_model->has_uq) {
    try {
      grace_model->validate_uq();
      has_uq = true;
    } catch (const grace::Error &e) {
      log.info(fmt("[GRACE-1L/KK] UQ head not usable, running without UQ: {}\n", e.what()));
    }
  }
  uq_Kmax = grace_model->uq_max_clusters;
  uq_D = grace_model->uq_feature_dim;
  uq_rp_dim = grace_model->uq_rp_dim;
  uq_n_density = grace_model->uq_n_density;
  uq_normalize = grace_model->uq_normalize;
  uq_density_scale = grace_model->uq_density_scale;
  uq_d_basis = grace_model->uq_d_basis;
  if (has_uq)
    log.info(fmt("[GRACE-1L/KK] UQ artifacts loaded (uqv6 basis-RP): Kmax={}, D={}, rp_dim={}, n_density={}, "
                 "D_basis={}\n", uq_Kmax, uq_D, uq_rp_dim, uq_n_density, uq_d_basis));
  log.info(fmt("[GRACE-1L/KK] n_elements={}, lmax={}, nradbase={}, nradmax={}, rcut={}, bond_specific_cutoff={}\n",
               grace_model->n_elements, grace_model->lmax, grace_model->nradbase, grace_model->nradmax,
               grace_model->rcut, grace_model->has_bond_specific_cutoff ? "yes" : "no"));

  set_type_map(type2mu);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::set_type_map(const std::vector<int> &type2mu)
{
  // atom type -> model element, and the per-type-pair cutoffs (as LAMMPS init_one)
  const int n = (int) type2mu.size();
  check_type2mu(type2mu, n - 1, grace_model->n_elements, "GRACE-1L/KK");
  h_type2model = type2mu;
  MemKK::realloc_kokkos(d_map, "grace_1l:map", n);
  auto h_map = Kokkos::create_mirror_view(d_map);
  for (int t = 0; t < n; t++) h_map(t) = type2mu[t];
  Kokkos::deep_copy(d_map, h_map);
  MemKK::realloc_kokkos(k_cutsq, "grace_1l:cutsq", n, n);
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
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::set_types(const std::vector<int> &type2mu)
{
  if (!setup_done) GRACE_KK_ERROR("GRACE-1L/KK: set_types() before setup()");
  set_type_map(type2mu);
  retype();
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
double Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::cutoff(int mu_i, int mu_j) const
{
  return pair_cutoff(*grace_model, mu_i, mu_j);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::setup()
{
  // Copy scalar architecture params
  nelements = grace_model->n_elements;
  lmax = grace_model->lmax;
  nradmax = grace_model->nradmax;
  nradbase = grace_model->nradbase;
  embedding_size = grace_model->embedding_size;
  radial_basis_p = grace_model->radial_basis_p;
  rcut = grace_model->rcut;
  rho_n_out = grace_model->rho_n_out;
  fc_n_out = grace_model->fc_A1.n_out;  // 64

  n_funcs_A1 = grace_model->fc_A1.n_funcs_left;   // 25
  n_funcs_AA = grace_model->prod_AA.n_output_funcs; // 141
  n_funcs_AA1 = n_funcs_AA;  // FC AA1 has same # output funcs
  n_funcs_AAA = grace_model->prod_AAA.n_output_funcs; // 14
  n_funcs_AA2 = n_funcs_AA;
  n_funcs_AAAA = grace_model->prod_AAAA.n_output_funcs; // 69
  n_cg_AA = grace_model->prod_AA.n_cg_terms;
  n_cg_AAA = grace_model->prod_AAA.n_cg_terms;
  n_cg_AAAA = grace_model->prod_AAAA.n_cg_terms;

  // Precompute spherical harmonics coefficients
  MemKK::realloc_kokkos(d_idx_sph, "grace_1l:idx_sph", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(alm, "grace_1l:alm", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(blm, "grace_1l:blm", (lmax + 1) * (lmax + 1));
  MemKK::realloc_kokkos(cl, "grace_1l:cl", lmax + 1);
  MemKK::realloc_kokkos(dl, "grace_1l:dl", lmax + 1);
  precompute_harmonics();

  // Copy all weights to device
  copy_weights_to_device();
  setup_done = true;
  check_kernel_launches<DeviceType>("GRACE-1L/KK");
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
typename Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::RunOutput
Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::run(const RunInput &in)
{
  if (!setup_done) GRACE_KK_ERROR("GRACE-1L/KK: run() before setup() (after set_model())");
  if (in.nlocal < in.inum) GRACE_KK_ERROR("GRACE-1L/KK: nlocal {} < inum {}", in.nlocal, in.inum);
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
  const bool do_energy_only = in.energy_only && !debug_no_energy_only_calc;
  d_eatom = in.eatom;
  d_vatom = in.vatom;
  d_cvatom = in.cvatom;
  x = in.x;
  couple_eq = in.couple_eq; couple_bit1 = in.couple_bit1; couple_bit2 = in.couple_bit2;
  couple_lambda = in.couple_lambda; d_mask = in.mask;
  if (couple_eq) {
    if (!couple_supported) GRACE_KK_ERROR("GRACE-1L/KK: two-group coupling needs a fused engine");
    if (couple_eq < 1 || couple_eq > 4) GRACE_KK_ERROR("GRACE-1L/KK: coupling equation {} (must be 1..4)", couple_eq);
    if (!d_dedl.data()) d_dedl = Kokkos::View<double, DeviceType>("grace:dedl");
    Kokkos::deep_copy(d_dedl, 0.0);
  }
  f = in.f;
  type = in.type;
  d_numneigh = in.numneigh;
  d_neighbors = in.neighbors;
  d_ilist = in.ilist;
  inum = in.inum;
  // Element-dependent weights are read per atom: order the atoms by type so a warp reads one weight row
  // instead of up to 32 different ones. Stable (key = type * inum + ii).
  static const bool sort_by_type = !getenv("GRACE_NO_TYPE_SORT");
  if (sort_by_type && in.ntypes > 1 && inum > 0) {
    if ((int) d_ilist_sorted.extent(0) < inum) {
      d_ilist_sorted = typename Traits::t_int_1d("g1l:ilist_sorted", inum);
      d_type_key = Kokkos::View<long long *, DeviceType>("g1l:type_key", inum);
    }
    auto src = in.ilist; auto ty = type; auto key = d_type_key; auto dst = d_ilist_sorted;
    const long long n = inum;
    Kokkos::parallel_for("grace_1l::type_key", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii) { key(ii) = (long long) ty(src(ii)) * n + ii; });
    Kokkos::sort(Kokkos::subview(key, std::make_pair(0, inum)));
    Kokkos::parallel_for("grace_1l::type_sort", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii) { dst(ii) = src((int) (key(ii) % n)); });
    d_ilist = d_ilist_sorted;
  }

  // Find max neighbors
  maxneigh = 0;
  {
    auto il = in.ilist; auto nn = in.numneigh;
    Kokkos::parallel_reduce("grace_1l::find_maxneigh", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii, int &mx) { const int c = nn[il[ii]]; if (mx < c) mx = c; },
        Kokkos::Max<int>(maxneigh));
  }
  maxneigh = std::max(maxneigh, 0);    // Kokkos::Max over no atoms (an empty rank) gives INT_MIN

  begin_run();
  choose_chunk();
  chunk_size = std::min(chunk_cap(), inum);
  chunk_offset = 0;
  grow(chunk_size, maxneigh);
  realloc_all = false;

  // ---- UQ / extrapolation grade: allocate per-atom outputs if requested ----
  const bool any_uq = in.uq && has_uq;
  uq_on = any_uq;
  if (any_uq) {
    if ((int)d_gamma.extent(0) < nlocal) {
      MemKK::realloc_kokkos(d_gamma, "g1l:d_gamma", nlocal);
      MemKK::realloc_kokkos(d_sigma, "g1l:d_sigma", nlocal);
      MemKK::realloc_kokkos(d_gmm_cluster, "g1l:d_gmm_cluster", nlocal);
    }
  }
  EV ev;

  while (chunk_offset < inum) {
    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

    // Zero per-chunk arrays
    Kokkos::deep_copy(d_A, 0.0);
    Kokkos::deep_copy(d_rho, 0.0);

    EV ev_tmp;

    int team_size = 1;
    int vector_length = 1;
    if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
      team_size = 32;

    stage_bond_fwd(team_size, vector_length);

    stage_dense_fwd();

    stage_reduce();

    // ---- Stage 12: ComputeMLPEnergy ----
    {
      // a thread per atom (one-thread teams ran one active lane per wavefront / warp: 2L-medium MI250X 31k atoms
      // 4.2 ms per call)
      Kokkos::parallel_for("ComputeMLPEnergy", Kokkos::RangePolicy<DeviceType, TagComputeMLPEnergyR>(0, chunk_size), *this);
    }

    // Accumulate energy
    if (eflag_global) {
      auto atom_energies = d_e_atom;
      KKA energy_partial = 0.0;    // fp64: an fp32 sum loses ~1e-3 eV at 3e4 atoms
      Kokkos::parallel_reduce("ComputeEnergySum", chunk_size,
          KOKKOS_LAMBDA(const int ii, KKA& update) {
          update += atom_energies(ii);
      }, energy_partial);
      eng_vdwl += energy_partial;
    }

    // Write per-atom energy to LAMMPS array
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

    // ---- UQ / extrapolation grade (forward only; runs on energy-only path too) ----
    // Team-per-atom: lanes split the B·R projection over rp_dim, the cluster
    // search over the feature dim, and the Mahalanobis over the D×D quadratic
    // form. Register/scratch accumulators (no dynamically-indexed per-thread
    // arrays) avoid the local-memory spill that made the old 1-thread/atom
    // RangePolicy kernel the dominant hotspot at small/mid N.
    if (any_uq) {
#ifdef KOKKOS_ENABLE_CUDA
      // cuBLAS DGEMM projection path (CUDA): gather Bmat -> DGEMM -> GMM finish.
      // Replaces the per-lane basis re-walk; big small-N win, better large-N L2
      // traffic. Non-CUDA / no-handle builds use the team ComputeUQ fallback.
      if (blas_handle) {
        compute_uq_cublas();
      } else
#endif
      {
      // f_sh[D] + delta_sh[D] + nrm scalar (team-shared scratch, level 0);
      // +8 doubles of slack absorbs per-allocation shmem alignment rounding.
      const int scratch_bytes = (2 * uq_D + 8) * (int)sizeof(double);
      auto probe = Kokkos::TeamPolicy<DeviceType, TagComputeUQ>(chunk_size, 1, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
      int uq_team_size = probe.team_size_max(*this, Kokkos::ParallelForTag());
      if (uq_team_size > 128) uq_team_size = 128;
      if (uq_team_size < 1) uq_team_size = 1;
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeUQ>(chunk_size, uq_team_size, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
      Kokkos::parallel_for("ComputeUQ", policy, *this);
      }
    }

    // ============ BACKWARD PASS (Forces) ============
    if (!do_energy_only) {

    // Zero adjoint arrays that are only partially written by RevReduceN.
    // d_AAA_adj and d_AAAA_adj are fully written (ws == n_funcs, ci injective),
    // so their deep_copy is not needed.
    stage_zero_adj();

    // ---- B1: ReverseMLP_Energy → d_rho_adj ----
    {
      auto rho_in = d_rho;
      auto rho_adj = d_rho_adj;
      auto eW = d_energy_W;
      auto enorms = d_energy_norms;
      auto edims = d_energy_dims;
      int en_layers = energy_n_layers;
      int emax = energy_max_dim;
      int cs = chunk_size;
      int eact = energy_activation;
      const NNScalar os = output_scale;
      Kokkos::parallel_for("ReverseMLPEnergy", Kokkos::RangePolicy<DeviceType>(0, cs),
        KOKKOS_LAMBDA(const int ii) {
          // Stack arrays: pre_act for hidden layers, adj ping-pong
          NNScalar pre_act_buf[GRACE1L_MAX_MLP_LAYERS * GRACE1L_MAX_MLP_DIM];
          NNScalar adj_a[GRACE1L_MAX_MLP_DIM], adj_b[GRACE1L_MAX_MLP_DIM];
          NNScalar* pre_act = pre_act_buf;
          NNScalar* adj_cur = adj_a;
          NNScalar* adj_nxt = adj_b;

          rho_adj(ii, 0) = os;  // linear term, scaled by output_scale

          // --- Forward pass: recompute all hidden pre-activations ---
          // Input to energy MLP is rho[1:] (skip rho[0])
          // Layer 0: pre_act[0] = norm0 * W0 @ rho[1:]
          {
            const int nin = edims(0);
            const int nout = edims(1);
            for (int j = 0; j < nout; j++) {
              NNScalar sum = 0.0;
              for (int k = 0; k < nin; k++)
                sum += eW(0, k, j) * rho_in(ii, k + 1);
              pre_act[0 * emax + j] = sum * enorms(0);
            }
          }
          // Hidden layers 1..en_layers-2: pre_act[layer] = norm * W @ act(pre_act[layer-1])
          for (int layer = 1; layer < en_layers - 1; layer++) {
            const int nin = edims(layer);
            const int nout = edims(layer + 1);
            for (int j = 0; j < nout; j++) {
              NNScalar sum = 0.0;
              for (int k = 0; k < nin; k++) {
                NNScalar pa = pre_act[(layer - 1) * emax + k];
                NNScalar act;
                if (eact == 1) {
                  act = Kokkos::tanh(pa);
                } else {
                  NNScalar sig = 1.0 / (1.0 + Kokkos::exp(-pa));
                  act = pa * sig;
                }
                sum += eW(layer, k, j) * act;
              }
              pre_act[layer * emax + j] = sum * enorms(layer);
            }
          }

          // --- Backward pass ---
          // Last layer (en_layers-1): output layer, no activation
          // adj for last hidden = W_last[:,0] * norm_last (output is scalar, index 0)
          {
            const int last = en_layers - 1;
            const int nin = edims(last);
            const int nout = edims(last + 1);
            // adj_cur = adjoint at input of last layer = silu(pre_act[last-1])
            // d_output/d_silu_input[k] = norm_last * W_last[k,o] for each output o
            // Since output is scalar (nout=1 typically), sum over outputs.
            // Multiply seed by output_scale so the chain rule scales all forces.
            for (int k = 0; k < nin; k++) {
              NNScalar val = 0.0;
              for (int o = 0; o < nout; o++)
                val += eW(last, k, o) * enorms(last);
              adj_cur[k] = val * os;
            }
          }

          // Backward through hidden layers (en_layers-2 down to 0)
          for (int layer = en_layers - 2; layer >= 0; layer--) {
            const int nin = edims(layer);
            const int nout = edims(layer + 1);
            // adj_cur[j] is adjoint at output of layer = act(pre_act[layer])
            // Apply act': d_pre_act = adj_cur * act'(pre_act)
            for (int j = 0; j < nout; j++) {
              NNScalar pa = pre_act[layer * emax + j];
              if (eact == 1) {
                NNScalar t = Kokkos::tanh(pa);
                adj_cur[j] *= (1.0 - t * t);
              } else {
                NNScalar sig = 1.0 / (1.0 + Kokkos::exp(-pa));
                adj_cur[j] *= sig * (1.0 + pa * (1.0 - sig));
              }
            }
            // Propagate through weight matrix: adj_nxt[k] = norm * Σ_j adj_cur[j] * W[layer,k,j]
            for (int k = 0; k < nin; k++) {
              NNScalar val = 0.0;
              for (int j = 0; j < nout; j++)
                val += adj_cur[j] * eW(layer, k, j);
              adj_nxt[k] = val * enorms(layer);
            }
            // Swap for next iteration
            NNScalar* tmp = adj_cur; adj_cur = adj_nxt; adj_nxt = tmp;
          }
          // adj_cur now holds d_output/d_rho[k+1] for k=0..edims(0)-1
          const int nin0 = edims(0);
          for (int k = 0; k < nin0; k++)
            rho_adj(ii, k + 1) = adj_cur[k];
        });
    }

    stage_dense_bwd();

    stage_bond_bwd(team_size, vector_length);

    } // end if (!do_energy_only)

    Kokkos::fence();
    chunk_offset += chunk_size;
  }


  RunOutput out;
  out.eng_vdwl = eng_vdwl;
  for (int k = 0; k < 6; k++) out.virial[k] = virial[k];
  check_kernel_launches<DeviceType>("GRACE-1L/KK");
  return out;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
double Grace1LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::memory_usage()
{
  double bytes = 0;
  bytes += grace::kk::MemKK::memory_usage(d_A);
  bytes += grace::kk::MemKK::memory_usage(d_R_nl);
  if constexpr (!std::is_same<NNScalar, double>::value)
    bytes += grace::kk::MemKK::memory_usage(d_dR_nl);
  if constexpr (std::is_same<NNScalar, double>::value)
    bytes += grace::kk::MemKK::memory_usage(d_h2);
  bytes += grace::kk::MemKK::memory_usage(d_radial_basis);
  bytes += grace::kk::MemKK::memory_usage(d_e_atom);
  bytes += grace::kk::MemKK::memory_usage(d_rho);
  if (has_uq) {
    bytes += grace::kk::MemKK::memory_usage(d_uq_centroids);
    bytes += grace::kk::MemKK::memory_usage(d_uq_inv_cov);
    bytes += grace::kk::MemKK::memory_usage(d_uq_interp_thresholds);
    bytes += grace::kk::MemKK::memory_usage(d_uq_rp_matrix);
  }
  return bytes;
}

}    // namespace kk
}    // namespace grace

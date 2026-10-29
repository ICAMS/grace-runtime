// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Implementation of grace::kk::Grace2LKK (see grace_2l_kk.h). Included by one translation unit per consumer, which
// instantiates the engine for its Traits (LAMMPS: the grace/runtime/kk style; python: _kk module).
#pragma once

#include "grace/kk/grace_2l_kk.h"

#include <Kokkos_Sort.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace grace {
namespace kk {
using namespace sph_const;

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::setup()
{
  auto &m = *grace_model;

  // Copy scalar architecture params
  nelements = m.n_elements;
  lmax = m.lmax;
  nradbase = m.nradbase;
  embedding_size = m.embedding_size;
  radial_basis_p = m.radial_basis_p;
  rcut = m.rcut;

  // Layer-specific dims
  L1_nradmax = m.L1_nradmax;
  L2_nradmax = m.L2_nradmax;
  L1_fc_n_out = m.L1_fc_n_out;
  L2_fc_n_out = m.L2_fc_n_out;
  I1_n_out = m.I1_n_out;
  I1_n_funcs = m.I1_n_funcs;
  rho_n_out = m.rho_n_out;
  B_n_out = m.B_n_out;
  B_n_funcs = m.B_n_funcs;
  I2_n_out = m.I2_n_out;
  n_yi_cg = m.yi.n_cg_terms;
  n_yi_out_funcs = m.yi.n_output_funcs;

  // Layer 1 func counts
  n_funcs_A1 = m.fc_A1.n_funcs_left;
  n_funcs_AA = m.prod_AA.n_output_funcs;
  n_funcs_AA1 = n_funcs_AA;
  n_funcs_AAA = m.prod_AAA.n_output_funcs;
  n_funcs_AA2 = n_funcs_AA;
  n_funcs_AAAA = m.prod_AAAA.n_output_funcs;
  n_cg_AA = m.prod_AA.n_cg_terms;
  n_cg_AAA = m.prod_AAA.n_cg_terms;
  n_cg_AAAA = m.prod_AAAA.n_cg_terms;

  // Layer 2 func counts
  n_funcs_B1 = m.fc_B1.n_funcs_left;
  n_funcs_BB = m.prod_BB.n_output_funcs;
  n_funcs_BB1 = n_funcs_BB;
  n_funcs_BBB = m.prod_BBB.n_output_funcs;
  n_funcs_BB2 = n_funcs_BB;
  n_funcs_BBBB = m.prod_BBBB.n_output_funcs;
  n_cg_BB = m.prod_BB.n_cg_terms;
  n_cg_BBB = m.prod_BBB.n_cg_terms;
  n_cg_BBBB = m.prod_BBBB.n_cg_terms;

  // Validate dimensions against compile-time stack array bounds
  {
    const int nlm = (lmax + 1) * (lmax + 1);
    const int plm_size = (lmax + 1) * (lmax + 2) / 2;
    if (nlm > 25)
      GRACE_KK_ERROR("GRACE-2L/KK: (lmax+1)^2={} exceeds stack array bound 25 (lmax={})", nlm, lmax);
    if (plm_size > 15)
      GRACE_KK_ERROR("GRACE-2L/KK: plm size {} exceeds stack array bound 15 (lmax={})", plm_size, lmax);
    if (L1_nradmax > GRACE2L_MAX_NRADMAX)
      GRACE_KK_ERROR("GRACE-2L/KK: L1_nradmax={} exceeds GRACE2L_MAX_NRADMAX={}", L1_nradmax, GRACE2L_MAX_NRADMAX);
    if (L2_nradmax > GRACE2L_MAX_NRADMAX)
      GRACE_KK_ERROR("GRACE-2L/KK: L2_nradmax={} exceeds GRACE2L_MAX_NRADMAX={}", L2_nradmax, GRACE2L_MAX_NRADMAX);
    if (I1_n_funcs > GRACE2L_MAX_I1_FUNCS)
      GRACE_KK_ERROR("GRACE-2L/KK: I1_n_funcs={} exceeds GRACE2L_MAX_I1_FUNCS={}", I1_n_funcs, GRACE2L_MAX_I1_FUNCS);
    if (I_n_funcs > GRACE2L_MAX_I_FUNCS)
      GRACE_KK_ERROR("GRACE-2L/KK: I_n_funcs={} exceeds GRACE2L_MAX_I_FUNCS={}", I_n_funcs, GRACE2L_MAX_I_FUNCS);
    if (n_yi_out_funcs > GRACE2L_MAX_YI_OUT_FUNCS)
      GRACE_KK_ERROR("GRACE-2L/KK: n_yi_out_funcs={} exceeds GRACE2L_MAX_YI_OUT_FUNCS={}", n_yi_out_funcs, GRACE2L_MAX_YI_OUT_FUNCS);
  }

  // Spherical harmonics
  grace::kk::MemKK::realloc_kokkos(d_idx_sph, "g2l:idx_sph", (lmax + 1) * (lmax + 1));
  grace::kk::MemKK::realloc_kokkos(alm, "g2l:alm", (lmax + 1) * (lmax + 1));
  grace::kk::MemKK::realloc_kokkos(blm, "g2l:blm", (lmax + 1) * (lmax + 1));
  grace::kk::MemKK::realloc_kokkos(cl, "g2l:cl", lmax + 1);
  grace::kk::MemKK::realloc_kokkos(dl, "g2l:dl", lmax + 1);
  precompute_harmonics();

  // l_from_lm lookup: for each lm index, store the corresponding l
  {
    int nlm = (lmax + 1) * (lmax + 1);
    grace::kk::MemKK::realloc_kokkos(d_l_from_lm, "g2l:l_from_lm", nlm);
    auto h_lfm = Kokkos::create_mirror_view(d_l_from_lm);
    for (int l = 0; l <= lmax; l++)
      for (int m = -l; m <= l; m++)
        h_lfm(l * (l + 1) + m) = l;
    Kokkos::deep_copy(d_l_from_lm, h_lfm);
  }

  // Copy all weights to device
  copy_weights_to_device();

#ifdef KOKKOS_ENABLE_CUDA
  // ---- create the cuBLAS handle once (CUDA device instantiation with
  // NNScalar=float, or with a UQ head), bind to the SAME stream the Kokkos parallel_fors launch
  // on, and force TRUE fp32 (CUBLAS_PEDANTIC_MATH disables the Ampere+ tf32
  // default). Same-stream binding => GEMMs and the surrounding Kokkos
  // assemble/act/scatter kernels execute in issue order, no race, no fences.
  // The if constexpr guards the CUDA-only cuda_stream() from the host
  // instantiation; the sizeof gate skips the fp64 engine unless it has a UQ head. ----
  // Handle is needed for the fp32 radial-MLP SGEMM path (NNScalar=float) AND for
  // the fp64 UQ DGEMM projection (UQ arrays are double, so the fp64 engine uses
  // cuBLAS for UQ even though its MLP path stays hand-written).
  if constexpr (is_cuda_v<DeviceType>) {
    if ((sizeof(NNScalar) == 4 || has_uq) && !blas_handle && blas::available()) {
      if (blas::create(&blas_handle) != blas::SUCCESS) {
#ifdef GRACE_KK_CUBLAS_DLOPEN
        blas_handle = nullptr;    // optional cuBLAS (e.g. no device memory left): the hand-written kernels
#else
        GRACE_KK_ERROR("GRACE-2L/KK: cublasCreate failed");
#endif
      }
      if (blas_handle) {
        blas_owner.reset(blas_handle, [](blas::handle_t h) { blas::destroy(h); });
        blas::set_stream(blas_handle, DeviceType());
        blas::set_pedantic_math(blas_handle);  // no tf32 (fp32 & fp64)
        if constexpr (sizeof(NNScalar) == 4)
          mlp_blas_selftest();   // tiny-case transpose/leading-dim gate (aborts on mismatch)
      }
    }
  }
  {
    if (blas_handle && sizeof(NNScalar) == 4)
      log.info(fmt("[GRACE-2L/KK] radial MLP: true-fp32 cuBLAS SGEMM enabled (CUDA)\n"));
    else
      log.info(fmt("[GRACE-2L/KK] radial MLP: hand-written Kokkos kernels "
                     "(cuBLAS is CUDA+fp32/Mixed only; fp64 & non-CUDA use the fallback)\n"));
    if (blas_handle && has_uq)
      log.info(fmt("[GRACE-2L/KK] UQ projection: fp64 cuBLAS DGEMM enabled (CUDA)\n"));
  }
#else
  log.info(fmt("[GRACE-2L/KK] radial MLP: hand-written Kokkos kernels "
                   "(cuBLAS is CUDA-only; default on non-CUDA backends, e.g. HIP/AMD)\n"));
#endif

  log.info(fmt("[GRACE-2L/KK] Initialization complete. comm_forward={}, L1_fc_n_out={}, L2_fc_n_out={}\n",
                   comm_width(), L1_fc_n_out, L2_fc_n_out));
  setup_done = true;
  check_kernel_launches<DeviceType>("GRACE-2L/KK");
}

// ======================================================================
// init_one
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::precompute_harmonics()
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
// copy_weights_to_device
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::copy_weights_to_device()
{
  auto &m = *grace_model;

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

  // Helper: copy MLP weights to 3D padded tensor
  auto copy_mlp = [&](const std::vector<GRACE2LModel::MLPLayer> &layers, int n_layers,
                       auto &d_W, auto &d_norms, auto &d_dims, int &max_dim,
                       const char *prefix) {
    std::vector<int> dims(n_layers + 1);
    dims[0] = layers[0].n_in;
    for (int i = 0; i < n_layers; i++) dims[i + 1] = layers[i].n_out;
    max_dim = 0;
    for (int i = 0; i < n_layers; i++)
      if (dims[i] > max_dim) max_dim = dims[i];
    int max_in = 0, max_out = 0;
    for (int i = 0; i < n_layers; i++) {
      if (layers[i].n_in > max_in) max_in = layers[i].n_in;
      if (layers[i].n_out > max_out) max_out = layers[i].n_out;
    }
    std::string s(prefix);
    grace::kk::MemKK::realloc_kokkos(d_W, (s+"_W").c_str(), n_layers, max_in, max_out);
    auto h_W = Kokkos::create_mirror_view(d_W);
    Kokkos::deep_copy(h_W, 0.0);
    for (int layer = 0; layer < n_layers; layer++) {
      auto &L = layers[layer];
      for (int i = 0; i < L.n_in; i++)
        for (int j = 0; j < L.n_out; j++)
          h_W(layer, i, j) = L.W[i * L.n_out + j];
    }
    Kokkos::deep_copy(d_W, h_W);
    grace::kk::MemKK::realloc_kokkos(d_norms, (s+"_norms").c_str(), n_layers);
    auto h_norms = Kokkos::create_mirror_view(d_norms);
    for (int i = 0; i < n_layers; i++) h_norms(i) = (NNScalar)layers[i].norm;
    Kokkos::deep_copy(d_norms, h_norms);
    grace::kk::MemKK::realloc_kokkos(d_dims, (s+"_dims").c_str(), n_layers + 1);
    auto h_dims = Kokkos::create_mirror_view(d_dims);
    for (int i = 0; i <= n_layers; i++) h_dims(i) = dims[i];
    Kokkos::deep_copy(d_dims, h_dims);
  };

  // Helper: copy FC weights
  auto copy_fc = [&](const GRACE2LModel::FCWeights &fc, const char *pfx,
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
    std::vector<double> nof_flat(fc.n_funcs_left);
    for (int i = 0; i < fc.n_funcs_left && i < (int)fc.norm_out_factor.size(); i++)
      nof_flat[i] = fc.norm_out_factor[i];
    copy_1d_double(nof_flat, dnof, (s+"_nof").c_str());
    nl = (NNScalar) fc.norm_left;
    nr = (NNScalar) fc.norm_right;
  };

  // Helper: copy CG product metadata
  auto copy_prod = [&](const GRACE2LModel::CGProduct &prod, const char *pfx,
                        auto &dli, auto &dri, auto &dsi, auto &dcg) {
    std::string s(pfx);
    copy_1d_int(prod.left_ind, dli, (s+"_li").c_str());
    copy_1d_int(prod.right_ind, dri, (s+"_ri").c_str());
    copy_1d_int(prod.m_sum_ind, dsi, (s+"_si").c_str());
    copy_1d_double(prod.cg_coeff, dcg, (s+"_cg").c_str());
  };

  // Helper: copy equivariant ReduceN (element-dependent, 4D W)
  auto copy_reduce_eq_elem = [&](const GRACE2LModel::ReduceWeights &rw, int nelem, int n_out,
                                  const char *pfx,
                                  auto &dW, NNScalar &norm, auto &dci, auto &dwlt, auto &dtsi, int &nc) {
    std::string s(pfx);
    copy_4d(rw.W, dW, nelem, n_out, rw.n_in, rw.w_shape, (s+"_W").c_str());
    norm = (NNScalar) rw.norm;
    copy_1d_int(rw.collect_ind, dci, (s+"_ci").c_str());
    copy_1d_int(rw.w_l_tile, dwlt, (s+"_wlt").c_str());
    copy_1d_int(rw.total_sum_ind, dtsi, (s+"_tsi").c_str());
    nc = rw.n_connections;
  };

  // Helper: copy equivariant ReduceN (NOT element-dependent, 3D W)
  auto copy_reduce_eq_noelem = [&](const GRACE2LModel::ReduceWeights &rw, int n_out,
                                    const char *pfx,
                                    auto &dW, NNScalar &norm, auto &dci, auto &dwlt, auto &dtsi, int &nc) {
    std::string s(pfx);
    copy_3d(rw.W, dW, n_out, rw.n_in, rw.w_shape, (s+"_W").c_str());
    norm = (NNScalar) rw.norm;
    copy_1d_int(rw.collect_ind, dci, (s+"_ci").c_str());
    copy_1d_int(rw.w_l_tile, dwlt, (s+"_wlt").c_str());
    copy_1d_int(rw.total_sum_ind, dtsi, (s+"_tsi").c_str());
    nc = rw.n_connections;
  };

  // Helper: copy scalar ReduceN (element-dependent, 4D W, uses w_l_tile as weight index)
  auto copy_reduce_scalar = [&](const GRACE2LModel::ReduceWeights &rw, int nelem, int n_out,
                                 const char *pfx,
                                 auto &dW, NNScalar &norm, auto &dci, auto &dwlt) {
    std::string s(pfx);
    copy_4d(rw.W, dW, nelem, n_out, rw.n_in, rw.w_shape, (s+"_W").c_str());
    norm = (NNScalar) rw.norm;
    copy_1d_int(rw.collect_ind, dci, (s+"_ci").c_str());
    copy_1d_int(rw.w_l_tile, dwlt, (s+"_wlt").c_str());
  };

  // ---- Chemical embedding ----
  copy_2d(m.chem_embedding, d_chem_embed, m.n_elements, m.embedding_size, "g2l:chem_embed");

  // ---- Layer 1 weights ----

  // A indicator transform
  copy_2d(m.A_lin_transform_W, d_A_lin_W, m.embedding_size, m.L1_nradmax, "g2l:A_lin_W");
  A_lin_norm = (NNScalar) m.A_lin_transform_norm;
  L1_inv_avg_n_neigh = (GeomScalar) m.A_inv_avg_n_neigh;

  // Precompute z_tr_A[mu][n] = Σ_e A_lin_W(e,n) * chem_embed(mu,e) * A_lin_norm
  {
    grace::kk::MemKK::realloc_kokkos(d_z_tr_A, "g2l:z_tr_A", m.n_elements, m.L1_nradmax);
    auto h = Kokkos::create_mirror_view(d_z_tr_A);
    for (int mu = 0; mu < m.n_elements; mu++)
      for (int n = 0; n < m.L1_nradmax; n++) {
        double val = 0.0;
        for (int e = 0; e < m.embedding_size; e++)
          val += m.A_lin_transform_W[e * m.L1_nradmax + n] * m.chem_embedding[mu * m.embedding_size + e];
        h(mu, n) = (NNScalar)(val * m.A_lin_transform_norm);
      }
    Kokkos::deep_copy(d_z_tr_A, h);
  }

  // MLP radial R
  mlp_rad_R_n_layers = m.L1_mlp_rad_n_layers;
  copy_mlp(m.L1_mlp_rad_layers, mlp_rad_R_n_layers,
           d_mlp_rad_R_W, d_mlp_rad_R_norms, d_mlp_rad_R_dims, mlp_rad_R_max_dim, "g2l:mlpR");
  if (mlp_rad_R_n_layers > GRACE2L_MAX_MLP_LAYERS)
    GRACE_KK_ERROR("GRACE-2L: R MLP depth ({}) exceeds GRACE2L_MAX_MLP_LAYERS ({})",
               mlp_rad_R_n_layers, GRACE2L_MAX_MLP_LAYERS);
  if (mlp_rad_R_max_dim > GRACE2L_MAX_MLP_HIDDEN)
    GRACE_KK_ERROR("GRACE-2L: R hidden dim ({}) exceeds GRACE2L_MAX_MLP_HIDDEN ({})",
               mlp_rad_R_max_dim, GRACE2L_MAX_MLP_HIDDEN);

  // Layer 1 FC
  copy_fc(m.fc_A1, "g2l:fc_A1", d_fc_A1_wl, d_fc_A1_wr, d_fc_A1_wtl, d_fc_A1_wtr,
          d_fc_A1_ct, d_fc_A1_cf, d_fc_A1_nof, fc_A1_nl, fc_A1_nr);
  copy_fc(m.fc_AA1, "g2l:fc_AA1", d_fc_AA1_wl, d_fc_AA1_wr, d_fc_AA1_wtl, d_fc_AA1_wtr,
          d_fc_AA1_ct, d_fc_AA1_cf, d_fc_AA1_nof, fc_AA1_nl, fc_AA1_nr);
  copy_fc(m.fc_AA2, "g2l:fc_AA2", d_fc_AA2_wl, d_fc_AA2_wr, d_fc_AA2_wtl, d_fc_AA2_wtr,
          d_fc_AA2_ct, d_fc_AA2_cf, d_fc_AA2_nof, fc_AA2_nl, fc_AA2_nr);

  // Layer 1 products
  copy_prod(m.prod_AA, "g2l:prod_AA", d_prod_AA_li, d_prod_AA_ri, d_prod_AA_si, d_prod_AA_cg);
  copy_prod(m.prod_AAA, "g2l:prod_AAA", d_prod_AAA_li, d_prod_AAA_ri, d_prod_AAA_si, d_prod_AAA_cg);
  copy_prod(m.prod_AAAA, "g2l:prod_AAAA", d_prod_AAAA_li, d_prod_AAAA_ri, d_prod_AAAA_si, d_prod_AAAA_cg);

  // I1 reducer (equivariant, element-dependent)
  copy_reduce_eq_elem(m.I1_reduce_A, m.n_elements, m.I1_n_out, "g2l:I1_A",
                       d_I1_reduce_A_W, I1_reduce_A_norm, d_I1_ci_A, d_I1_wlt_A, d_I1_tsi_A, I1_nc_A);
  copy_reduce_eq_elem(m.I1_reduce_AA, m.n_elements, m.I1_n_out, "g2l:I1_AA",
                       d_I1_reduce_AA_W, I1_reduce_AA_norm, d_I1_ci_AA, d_I1_wlt_AA, d_I1_tsi_AA, I1_nc_AA);
  copy_reduce_eq_elem(m.I1_reduce_AAA, m.n_elements, m.I1_n_out, "g2l:I1_AAA",
                       d_I1_reduce_AAA_W, I1_reduce_AAA_norm, d_I1_ci_AAA, d_I1_wlt_AAA, d_I1_tsi_AAA, I1_nc_AAA);
  copy_reduce_eq_elem(m.I1_reduce_AAAA, m.n_elements, m.I1_n_out, "g2l:I1_AAAA",
                       d_I1_reduce_AAAA_W, I1_reduce_AAAA_norm, d_I1_ci_AAAA, d_I1_wlt_AAAA, d_I1_tsi_AAAA, I1_nc_AAAA);

  // I reducer (equivariant, NOT element-dependent)
  copy_reduce_eq_noelem(m.I_reduce_I1, m.I_n_out, "g2l:I_I1",
                         d_I_reduce_I1_W, I_reduce_I1_norm, d_I_ci_I1, d_I_wlt_I1, d_I_tsi_I1, I_nc_I1);

  // rho reducer (scalar, element-dependent)
  copy_reduce_scalar(m.rho_reduce_A, m.n_elements, m.rho_n_out, "g2l:rho_A",
                      d_rho_reduce_A_W, rho_reduce_A_norm, d_rho_ci_A, d_rho_wlt_A);
  copy_reduce_scalar(m.rho_reduce_AA, m.n_elements, m.rho_n_out, "g2l:rho_AA",
                      d_rho_reduce_AA_W, rho_reduce_AA_norm, d_rho_ci_AA, d_rho_wlt_AA);
  copy_reduce_scalar(m.rho_reduce_AAA, m.n_elements, m.rho_n_out, "g2l:rho_AAA",
                      d_rho_reduce_AAA_W, rho_reduce_AAA_norm, d_rho_ci_AAA, d_rho_wlt_AAA);
  copy_reduce_scalar(m.rho_reduce_AAAA, m.n_elements, m.rho_n_out, "g2l:rho_AAAA",
                      d_rho_reduce_AAAA_W, rho_reduce_AAAA_norm, d_rho_ci_AAAA, d_rho_wlt_AAAA);

  // I_nl_LN scale
  copy_1d_double(m.I_nl_LN_scale, d_I_nl_LN_scale, "g2l:I_nl_LN_scale");

  // ---- Layer 2 weights ----

  // B0 indicator transform
  copy_2d(m.B0_lin_transform_W, d_B0_lin_W, m.embedding_size, m.L2_nradmax, "g2l:B0_lin_W");
  B0_lin_norm = (NNScalar) m.B0_lin_transform_norm;
  L2_inv_avg_n_neigh = (GeomScalar) m.B0_inv_avg_n_neigh;

  // Precompute z_tr_B0[mu][n] = Σ_e B0_lin_W(e,n) * chem_embed(mu,e) * B0_lin_norm
  {
    grace::kk::MemKK::realloc_kokkos(d_z_tr_B0, "g2l:z_tr_B0", m.n_elements, m.L2_nradmax);
    auto h = Kokkos::create_mirror_view(d_z_tr_B0);
    for (int mu = 0; mu < m.n_elements; mu++)
      for (int n = 0; n < m.L2_nradmax; n++) {
        double val = 0.0;
        for (int e = 0; e < m.embedding_size; e++)
          val += m.B0_lin_transform_W[e * m.L2_nradmax + n] * m.chem_embedding[mu * m.embedding_size + e];
        h(mu, n) = (NNScalar)(val * m.B0_lin_transform_norm);
      }
    Kokkos::deep_copy(d_z_tr_B0, h);
  }

  // MLP radial R1
  mlp_rad_R1_n_layers = m.L2_mlp_rad_n_layers;
  copy_mlp(m.L2_mlp_rad_layers, mlp_rad_R1_n_layers,
           d_mlp_rad_R1_W, d_mlp_rad_R1_norms, d_mlp_rad_R1_dims, mlp_rad_R1_max_dim, "g2l:mlpR1");
  if (mlp_rad_R1_n_layers > GRACE2L_MAX_MLP_LAYERS)
    GRACE_KK_ERROR("GRACE-2L: R1 MLP depth ({}) exceeds GRACE2L_MAX_MLP_LAYERS ({})",
               mlp_rad_R1_n_layers, GRACE2L_MAX_MLP_LAYERS);
  if (mlp_rad_R1_max_dim > GRACE2L_MAX_MLP_HIDDEN)
    GRACE_KK_ERROR("GRACE-2L: R1 hidden dim ({}) exceeds GRACE2L_MAX_MLP_HIDDEN ({})",
               mlp_rad_R1_max_dim, GRACE2L_MAX_MLP_HIDDEN);

#ifdef KOKKOS_ENABLE_CUDA
  // pack each radial-MLP layer's weights TIGHTLY row-major [n_in x
  // n_out] for cuBLAS (the padded LayoutLeft d_mlp_rad_R{,1}_W is not a
  // contiguous per-layer block). which: 0 = R (L1), 1 = R1 (L2). lda = n_out.
  // Guarded to the CUDA device float instantiation (the only one that uses SGEMM).
  if constexpr (is_cuda_v<DeviceType> && sizeof(NNScalar) == 4) {
    auto pack_mlp = [&](const std::vector<GRACE2LModel::MLPLayer> &layers, int which) {
      const int nl = (int) layers.size();
      for (int l = 0; l < nl; l++) {
        const int ni = layers[l].n_in, no = layers[l].n_out;
        grace::kk::MemKK::realloc_kokkos(d_mlp_Wg[which][l],
            (std::string("g2l:mlp_Wg_") + std::to_string(which) + "_" + std::to_string(l)).c_str(),
            std::max(ni, 1), std::max(no, 1));
        auto hWg = Kokkos::create_mirror_view(d_mlp_Wg[which][l]);
        for (int i = 0; i < ni; i++)
          for (int j = 0; j < no; j++)
            hWg(i, j) = (NNScalar) layers[l].W[(size_t) i * no + j];
        Kokkos::deep_copy(d_mlp_Wg[which][l], hWg);
      }
    };
    pack_mlp(m.L1_mlp_rad_layers, 0);
    pack_mlp(m.L2_mlp_rad_layers, 1);
  }
#endif

  // YI coupling metadata — sort CG terms by y_idx (= lr_inds[t*2]) to group by l-value.
  // This improves register reuse: R1(ii,jj,n,l) stays in a register for all terms with same l.
  {
    auto &yi_m = m.yi;
    const int ncg = yi_m.n_cg_terms;

    // Build sort permutation by out_f primary, y_idx secondary.
    // Out_f primary sort lets ComputeYI register-accumulate YI_out(ii,n,out_f)
    // across consecutive same-out_f t-entries (~877/190 = 4-5x reduction in
    // global RMWs), and lets ReverseYI/Derivative_L2 cache yi_adj(ii,n,out_f)
    // to a register across the same group (saves one global read per t-iter).
    std::vector<int> perm(ncg);
    for (int t = 0; t < ncg; t++) perm[t] = t;
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
      const int oa = yi_m.m_sum_ind[a], ob = yi_m.m_sum_ind[b];
      if (oa != ob) return oa < ob;
      return yi_m.lr_inds[a * 2] < yi_m.lr_inds[b * 2];
    });

    // Apply permutation to all parallel arrays
    grace::kk::MemKK::realloc_kokkos(d_yi_lr_inds, "g2l:yi_lr", ncg, 2);
    auto h_lr = Kokkos::create_mirror_view(d_yi_lr_inds);
    grace::kk::MemKK::realloc_kokkos(d_yi_m_sum_ind, "g2l:yi_msi", ncg);
    auto h_msi = Kokkos::create_mirror_view(d_yi_m_sum_ind);
    grace::kk::MemKK::realloc_kokkos(d_yi_cg_coeff, "g2l:yi_cg", ncg);
    auto h_cg = Kokkos::create_mirror_view(d_yi_cg_coeff);
    for (int t = 0; t < ncg; t++) {
      const int src = perm[t];
      h_lr(t, 0) = yi_m.lr_inds[src * 2];
      h_lr(t, 1) = yi_m.lr_inds[src * 2 + 1];
      h_msi(t) = yi_m.m_sum_ind[src];
      h_cg(t) = (NNScalar) yi_m.cg_coeff[src];
    }
    Kokkos::deep_copy(d_yi_lr_inds, h_lr);
    Kokkos::deep_copy(d_yi_m_sum_ind, h_msi);
    Kokkos::deep_copy(d_yi_cg_coeff, h_cg);
    yi_inv_avg_n_neigh = (GeomScalar) yi_m.inv_avg_n_neigh;

    // Build per-lm offsets into the sorted CG terms
    const int nlm = (m.lmax + 1) * (m.lmax + 1);
    grace::kk::MemKK::realloc_kokkos(d_yi_lm_offset, "g2l:yi_lm_off", nlm);
    grace::kk::MemKK::realloc_kokkos(d_yi_lm_count, "g2l:yi_lm_cnt", nlm);
    auto h_off = Kokkos::create_mirror_view(d_yi_lm_offset);
    auto h_cnt = Kokkos::create_mirror_view(d_yi_lm_count);
    for (int lm = 0; lm < nlm; lm++) { h_off(lm) = 0; h_cnt(lm) = 0; }
    for (int t = 0; t < ncg; t++) h_cnt(h_lr(t, 0))++;
    int acc = 0;
    for (int lm = 0; lm < nlm; lm++) { h_off(lm) = acc; acc += h_cnt(lm); }
    Kokkos::deep_copy(d_yi_lm_offset, h_off);
    Kokkos::deep_copy(d_yi_lm_count, h_cnt);

    // Build per-output offsets. Terms are sorted by output-function primary,
    // so each output's CG slice is contiguous.
    grace::kk::MemKK::realloc_kokkos(d_yi_out_offset, "g2l:yi_out_off", n_yi_out_funcs);
    grace::kk::MemKK::realloc_kokkos(d_yi_out_count, "g2l:yi_out_cnt", n_yi_out_funcs);
    auto h_out_off = Kokkos::create_mirror_view(d_yi_out_offset);
    auto h_out_cnt = Kokkos::create_mirror_view(d_yi_out_count);
    for (int f = 0; f < n_yi_out_funcs; f++) { h_out_off(f) = 0; h_out_cnt(f) = 0; }
    for (int t = 0; t < ncg; t++) h_out_cnt(h_msi(t))++;
    acc = 0;
    for (int f = 0; f < n_yi_out_funcs; f++) { h_out_off(f) = acc; acc += h_out_cnt(f); }
    Kokkos::deep_copy(d_yi_out_offset, h_out_off);
    Kokkos::deep_copy(d_yi_out_count, h_out_cnt);

    // Build unique (Y_lm, I_idx) pairs for a two-stage forward YI
    // contraction. The expensive neighbor loop computes each pair once, then
    // the sparse CG output reduction reuses those pair rows.
    std::vector<int> pair_yidx;
    std::vector<int> pair_iidx;
    std::vector<int> term_pair(ncg);
    pair_yidx.reserve(ncg);
    pair_iidx.reserve(ncg);
    for (int t = 0; t < ncg; t++) {
      const int y_idx = h_lr(t, 0);
      const int I_idx = h_lr(t, 1);
      int pair_idx = -1;
      for (int p = 0; p < (int)pair_yidx.size(); p++) {
        if (pair_yidx[p] == y_idx && pair_iidx[p] == I_idx) {
          pair_idx = p;
          break;
        }
      }
      if (pair_idx < 0) {
        pair_idx = (int)pair_yidx.size();
        pair_yidx.push_back(y_idx);
        pair_iidx.push_back(I_idx);
      }
      term_pair[t] = pair_idx;
    }
    n_yi_pairs = (int)pair_yidx.size();
    grace::kk::MemKK::realloc_kokkos(d_yi_pair_yidx, "g2l:yi_pair_yidx", n_yi_pairs);
    grace::kk::MemKK::realloc_kokkos(d_yi_pair_iidx, "g2l:yi_pair_iidx", n_yi_pairs);
    grace::kk::MemKK::realloc_kokkos(d_yi_pair_l, "g2l:yi_pair_l", n_yi_pairs);
    grace::kk::MemKK::realloc_kokkos(d_yi_term_pair, "g2l:yi_term_pair", ncg);
    grace::kk::MemKK::realloc_kokkos(d_yi_pair_term_offset, "g2l:yi_pair_term_off", n_yi_pairs);
    grace::kk::MemKK::realloc_kokkos(d_yi_pair_term_count, "g2l:yi_pair_term_cnt", n_yi_pairs);
    grace::kk::MemKK::realloc_kokkos(d_yi_pair_term_out, "g2l:yi_pair_term_out", ncg);
    grace::kk::MemKK::realloc_kokkos(d_yi_pair_term_cg, "g2l:yi_pair_term_cg", ncg);
    grace::kk::MemKK::realloc_kokkos(d_yi_i_pair_offset, "g2l:yi_i_pair_off", I_n_funcs);
    grace::kk::MemKK::realloc_kokkos(d_yi_i_pair_count, "g2l:yi_i_pair_cnt", I_n_funcs);
    grace::kk::MemKK::realloc_kokkos(d_yi_i_pair_index, "g2l:yi_i_pair_idx", n_yi_pairs);
    auto h_pair_yidx = Kokkos::create_mirror_view(d_yi_pair_yidx);
    auto h_pair_iidx = Kokkos::create_mirror_view(d_yi_pair_iidx);
    auto h_pair_l = Kokkos::create_mirror_view(d_yi_pair_l);
    auto h_term_pair = Kokkos::create_mirror_view(d_yi_term_pair);
    auto h_pair_term_off = Kokkos::create_mirror_view(d_yi_pair_term_offset);
    auto h_pair_term_cnt = Kokkos::create_mirror_view(d_yi_pair_term_count);
    auto h_pair_term_out = Kokkos::create_mirror_view(d_yi_pair_term_out);
    auto h_pair_term_cg = Kokkos::create_mirror_view(d_yi_pair_term_cg);
    auto h_i_pair_off = Kokkos::create_mirror_view(d_yi_i_pair_offset);
    auto h_i_pair_cnt = Kokkos::create_mirror_view(d_yi_i_pair_count);
    auto h_i_pair_idx = Kokkos::create_mirror_view(d_yi_i_pair_index);
    for (int f = 0; f < I_n_funcs; f++) { h_i_pair_off(f) = 0; h_i_pair_cnt(f) = 0; }
    for (int p = 0; p < n_yi_pairs; p++) {
      const int y_idx = pair_yidx[p];
      int l_val = 0;
      while ((l_val + 1) * (l_val + 1) <= y_idx) l_val++;
      h_pair_yidx(p) = y_idx;
      h_pair_iidx(p) = pair_iidx[p];
      h_pair_l(p) = l_val;
      h_pair_term_off(p) = 0;
      h_pair_term_cnt(p) = 0;
      h_i_pair_cnt(pair_iidx[p])++;
    }
    for (int t = 0; t < ncg; t++) h_pair_term_cnt(term_pair[t])++;
    acc = 0;
    for (int p = 0; p < n_yi_pairs; p++) {
      h_pair_term_off(p) = acc;
      acc += h_pair_term_cnt(p);
      h_pair_term_cnt(p) = 0;
    }
    for (int t = 0; t < ncg; t++) {
      const int p = term_pair[t];
      const int dst = h_pair_term_off(p) + h_pair_term_cnt(p);
      h_pair_term_out(dst) = h_msi(t);
      h_pair_term_cg(dst) = h_cg(t);
      h_pair_term_cnt(p)++;
      h_term_pair(t) = p;
    }
    acc = 0;
    for (int f = 0; f < I_n_funcs; f++) {
      h_i_pair_off(f) = acc;
      acc += h_i_pair_cnt(f);
      h_i_pair_cnt(f) = 0;
    }
    for (int p = 0; p < n_yi_pairs; p++) {
      const int I_idx = pair_iidx[p];
      const int dst = h_i_pair_off(I_idx) + h_i_pair_cnt(I_idx);
      h_i_pair_idx(dst) = p;
      h_i_pair_cnt(I_idx)++;
    }
    Kokkos::deep_copy(d_yi_pair_yidx, h_pair_yidx);
    Kokkos::deep_copy(d_yi_pair_iidx, h_pair_iidx);
    Kokkos::deep_copy(d_yi_pair_l, h_pair_l);
    Kokkos::deep_copy(d_yi_term_pair, h_term_pair);
    Kokkos::deep_copy(d_yi_pair_term_offset, h_pair_term_off);
    Kokkos::deep_copy(d_yi_pair_term_count, h_pair_term_cnt);
    Kokkos::deep_copy(d_yi_pair_term_out, h_pair_term_out);
    Kokkos::deep_copy(d_yi_pair_term_cg, h_pair_term_cg);
    Kokkos::deep_copy(d_yi_i_pair_offset, h_i_pair_off);
    Kokkos::deep_copy(d_yi_i_pair_count, h_i_pair_cnt);
    Kokkos::deep_copy(d_yi_i_pair_index, h_i_pair_idx);

    // Reverse dI groups by I channel. This lets each GPU lane own one
    // grad_I_global(j,I,n) update and reuse the already materialized Y_bond.
    std::vector<int> perm_i(ncg);
    for (int t = 0; t < ncg; t++) perm_i[t] = t;
    std::sort(perm_i.begin(), perm_i.end(), [&](int a, int b) {
      const int ia = yi_m.lr_inds[a * 2 + 1], ib = yi_m.lr_inds[b * 2 + 1];
      if (ia != ib) return ia < ib;
      return yi_m.lr_inds[a * 2] < yi_m.lr_inds[b * 2];
    });
    grace::kk::MemKK::realloc_kokkos(d_yi_i_yidx, "g2l:yi_i_yidx", ncg);
    grace::kk::MemKK::realloc_kokkos(d_yi_i_out, "g2l:yi_i_out", ncg);
    grace::kk::MemKK::realloc_kokkos(d_yi_i_cg, "g2l:yi_i_cg", ncg);
    auto h_i_yidx = Kokkos::create_mirror_view(d_yi_i_yidx);
    auto h_i_out = Kokkos::create_mirror_view(d_yi_i_out);
    auto h_i_cg = Kokkos::create_mirror_view(d_yi_i_cg);
    grace::kk::MemKK::realloc_kokkos(d_yi_i_offset, "g2l:yi_i_off", I_n_funcs);
    grace::kk::MemKK::realloc_kokkos(d_yi_i_count, "g2l:yi_i_cnt", I_n_funcs);
    auto h_i_off = Kokkos::create_mirror_view(d_yi_i_offset);
    auto h_i_cnt = Kokkos::create_mirror_view(d_yi_i_count);
    for (int f = 0; f < I_n_funcs; f++) { h_i_off(f) = 0; h_i_cnt(f) = 0; }
    for (int t = 0; t < ncg; t++) {
      const int src = perm_i[t];
      const int I_idx = yi_m.lr_inds[src * 2 + 1];
      h_i_yidx(t) = yi_m.lr_inds[src * 2];
      h_i_out(t) = yi_m.m_sum_ind[src];
      h_i_cg(t) = (NNScalar) yi_m.cg_coeff[src];
      h_i_cnt(I_idx)++;
    }
    acc = 0;
    for (int f = 0; f < I_n_funcs; f++) { h_i_off(f) = acc; acc += h_i_cnt(f); }
    Kokkos::deep_copy(d_yi_i_yidx, h_i_yidx);
    Kokkos::deep_copy(d_yi_i_out, h_i_out);
    Kokkos::deep_copy(d_yi_i_cg, h_i_cg);
    Kokkos::deep_copy(d_yi_i_offset, h_i_off);
    Kokkos::deep_copy(d_yi_i_count, h_i_cnt);
  }

  // B reducer (equivariant, NOT element-dependent)
  copy_reduce_eq_noelem(m.B_reduce_YI, m.B_n_out, "g2l:B_YI",
                         d_B_reduce_YI_W, B_reduce_YI_norm, d_B_ci_YI, d_B_wlt_YI, d_B_tsi_YI, B_nc_YI);
  copy_reduce_eq_noelem(m.B_reduce_B0, m.B_n_out, "g2l:B_B0",
                         d_B_reduce_B0_W, B_reduce_B0_norm, d_B_ci_B0, d_B_wlt_B0, d_B_tsi_B0, B_nc_B0);

  // Layer 2 FC
  copy_fc(m.fc_B1, "g2l:fc_B1", d_fc_B1_wl, d_fc_B1_wr, d_fc_B1_wtl, d_fc_B1_wtr,
          d_fc_B1_ct, d_fc_B1_cf, d_fc_B1_nof, fc_B1_nl, fc_B1_nr);
  copy_fc(m.fc_BB1, "g2l:fc_BB1", d_fc_BB1_wl, d_fc_BB1_wr, d_fc_BB1_wtl, d_fc_BB1_wtr,
          d_fc_BB1_ct, d_fc_BB1_cf, d_fc_BB1_nof, fc_BB1_nl, fc_BB1_nr);
  copy_fc(m.fc_BB2, "g2l:fc_BB2", d_fc_BB2_wl, d_fc_BB2_wr, d_fc_BB2_wtl, d_fc_BB2_wtr,
          d_fc_BB2_ct, d_fc_BB2_cf, d_fc_BB2_nof, fc_BB2_nl, fc_BB2_nr);

#ifdef KOKKOS_ENABLE_CUDA
  // build cuBLAS block-sparse FC metadata (CUDA device + float only).
  if constexpr (is_cuda_v<DeviceType> && sizeof(NNScalar) == 4) {
    if (uses_blas_ops()) {
      build_fc_blas_op(fc_op_A1,  m.fc_A1);
      build_fc_blas_op(fc_op_AA1, m.fc_AA1);
      build_fc_blas_op(fc_op_AA2, m.fc_AA2);
      build_fc_blas_op(fc_op_B1,  m.fc_B1);
      build_fc_blas_op(fc_op_BB1, m.fc_BB1);
      build_fc_blas_op(fc_op_BB2, m.fc_BB2);
      // element-independent B ReduceN (must run before grow() so the
      // shared FC/reduce scratch is sized for N = max #connections, e.g. 190 > FC).
      build_reduce_blas_op(reduce_op_B_YI, m.B_reduce_YI, m.B_n_out);
      build_reduce_blas_op(reduce_op_B_B0, m.B_reduce_B0, m.B_n_out);
    }
  }
#endif

  // Layer 2 products
  copy_prod(m.prod_BB, "g2l:prod_BB", d_prod_BB_li, d_prod_BB_ri, d_prod_BB_si, d_prod_BB_cg);
  copy_prod(m.prod_BBB, "g2l:prod_BBB", d_prod_BBB_li, d_prod_BBB_ri, d_prod_BBB_si, d_prod_BBB_cg);
  copy_prod(m.prod_BBBB, "g2l:prod_BBBB", d_prod_BBBB_li, d_prod_BBBB_ri, d_prod_BBBB_si, d_prod_BBBB_cg);

  // I2 reducer (scalar, element-dependent)
  copy_reduce_scalar(m.I2_reduce_B, m.n_elements, m.I2_n_out, "g2l:I2_B",
                      d_I2_reduce_B_W, I2_reduce_B_norm, d_I2_ci_B, d_I2_wlt_B);
  copy_reduce_scalar(m.I2_reduce_BB, m.n_elements, m.I2_n_out, "g2l:I2_BB",
                      d_I2_reduce_BB_W, I2_reduce_BB_norm, d_I2_ci_BB, d_I2_wlt_BB);
  copy_reduce_scalar(m.I2_reduce_BBB, m.n_elements, m.I2_n_out, "g2l:I2_BBB",
                      d_I2_reduce_BBB_W, I2_reduce_BBB_norm, d_I2_ci_BBB, d_I2_wlt_BBB);
  copy_reduce_scalar(m.I2_reduce_BBBB, m.n_elements, m.I2_n_out, "g2l:I2_BBBB",
                      d_I2_reduce_BBBB_W, I2_reduce_BBBB_norm, d_I2_ci_BBBB, d_I2_wlt_BBBB);

  // I_0_LN scale
  copy_1d_double(m.I_0_LN_scale, d_I_0_LN_scale, "g2l:I_0_LN_scale");

  // Energy MLP
  energy_n_layers = (int)m.energy_mlp_layers.size();
  if (m.energy_mlp_activation != 0 && m.energy_mlp_activation != 1)
    GRACE_KK_ERROR("GRACE-2L/KK: energy_mlp_activation={} not supported "
                      "(only 0=silu, 1=tanh)", m.energy_mlp_activation);
  energy_activation = m.energy_mlp_activation;
  copy_mlp(m.energy_mlp_layers, energy_n_layers,
           d_energy_W, d_energy_norms, d_energy_dims, energy_max_dim, "g2l:energy");
  if (energy_n_layers > GRACE2L_MAX_MLP_LAYERS)
    GRACE_KK_ERROR("GRACE-2L: energy MLP depth ({}) exceeds GRACE2L_MAX_MLP_LAYERS ({})",
               energy_n_layers, GRACE2L_MAX_MLP_LAYERS);
  if (energy_max_dim > GRACE2L_MAX_MLP_DIM)
    GRACE_KK_ERROR("GRACE-2L: energy hidden dim ({}) exceeds GRACE2L_MAX_MLP_DIM ({})",
               energy_max_dim, GRACE2L_MAX_MLP_DIM);

  // Shifts
  copy_1d_double(m.shift_values, d_shifts, "g2l:shifts");
  output_scale = static_cast<NNScalar>(m.output_scale);

  // Bond-specific cutoff map
  copy_2d(m.bond_cutoff_map, d_bond_cutoff, nelements, nelements, "g2l:bond_cutoff");

  // ---- UQ / extrapolation-grade artifacts (optional; schema v6 basis-RP) ----
  if (has_uq) {
    const int E = m.uq_n_elements;
    const int K = m.uq_max_clusters;
    const int D = m.uq_feature_dim;        // = rp_dim + n_density
    const int RP = m.uq_rp_dim;            // projection width (R cols)
    if (E != m.n_elements)
      GRACE_KK_ERROR("GRACE-2L/KK UQ: uq_n_elements ({}) != model n_elements ({})",
                 E, m.n_elements);
    if (D > GRACE2L_UQ_MAX_RP_DIM)
      GRACE_KK_ERROR("GRACE-2L/KK UQ: feature_dim ({}) exceeds GRACE2L_UQ_MAX_RP_DIM ({})",
                 D, GRACE2L_UQ_MAX_RP_DIM);
    // basis-RP reads the gathered scalar-reduce products (b_inv) in the exact
    // (source, n_out, collect) order the energy reduces iterate them, so the
    // R-row count D_basis must equal the L1(rho) + L2(I2) scalar-basis widths
    // [Σ nin·n_collected per source]. This guards against an artifact/model mismatch.
    const int d_basis_L1 =
        (int)d_rho_reduce_A_W.extent(2)    * (int)d_rho_ci_A.extent(0) +
        (int)d_rho_reduce_AA_W.extent(2)   * (int)d_rho_ci_AA.extent(0) +
        (int)d_rho_reduce_AAA_W.extent(2)  * (int)d_rho_ci_AAA.extent(0) +
        (int)d_rho_reduce_AAAA_W.extent(2) * (int)d_rho_ci_AAAA.extent(0);
    const int d_basis_L2 =
        (int)d_I2_reduce_B_W.extent(2)     * (int)d_I2_ci_B.extent(0) +
        (int)d_I2_reduce_BB_W.extent(2)    * (int)d_I2_ci_BB.extent(0) +
        (int)d_I2_reduce_BBB_W.extent(2)   * (int)d_I2_ci_BBB.extent(0) +
        (int)d_I2_reduce_BBBB_W.extent(2)  * (int)d_I2_ci_BBBB.extent(0);
    if (d_basis_L1 + d_basis_L2 != m.uq_d_basis)
      GRACE_KK_ERROR("GRACE-2L/KK UQ: scalar-basis width L1({})+L2({}) != uq_rp_matrix rows ({})",
                 d_basis_L1, d_basis_L2, m.uq_d_basis);
    uq_d_basis = m.uq_d_basis;
    uq_d_basis_L1 = d_basis_L1;
    copy_3d(m.uq_centroids, d_uq_centroids, E, K, D, "g2l:uq_centroids");
    copy_4d(m.uq_inv_cov, d_uq_inv_cov, E, K, D, D, "g2l:uq_inv_cov");
    copy_1d_int(m.uq_n_clusters, d_uq_n_clusters, "g2l:uq_n_clusters");
    copy_2d(m.uq_interp_thresholds, d_uq_interp_thresholds, E, K, "g2l:uq_interp_thr");
    copy_2d(m.uq_rp_matrix, d_uq_rp_matrix, m.uq_d_basis, RP, "g2l:uq_rp_matrix");
#ifdef KOKKOS_ENABLE_CUDA
    uq_sgemm = std::is_same<NNScalarT, float>::value;
    if (uq_sgemm) {
      d_uq_rp_matrix_f = Kokkos::View<float*, DeviceType>("g2l:uq_rp_matrix_f", m.uq_rp_matrix.size());
      auto h = Kokkos::create_mirror_view(d_uq_rp_matrix_f);
      for (size_t q = 0; q < m.uq_rp_matrix.size(); q++) h(q) = (float) m.uq_rp_matrix[q];
      Kokkos::deep_copy(d_uq_rp_matrix_f, h);
    }
#endif
  }
}

// ======================================================================
// grow: allocate per-chunk intermediate arrays
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::grow(int natom, int maxneigh_in)
{
  const int nlm = (lmax + 1) * (lmax + 1);  // 25

  if (realloc_all || (int)d_A.extent(0) < natom) {
    grace::kk::MemKK::realloc_kokkos(d_ncount, "g2l:ncount", natom);
    grace::kk::MemKK::realloc_kokkos(d_mu_i, "g2l:mu_i", natom);
    grace::kk::MemKK::realloc_kokkos(d_e_atom, "g2l:e_atom", natom);

    // Layer 1 per-atom
    grace::kk::MemKK::realloc_kokkos(d_A, "g2l:A", natom, L1_nradmax, nlm);
    grace::kk::MemKK::realloc_kokkos(d_A1, "g2l:A1", natom, L1_fc_n_out, n_funcs_A1);
    grace::kk::MemKK::realloc_kokkos(d_AA, "g2l:AA", natom, L1_fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_AA1, "g2l:AA1", natom, L1_fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_AAA, "g2l:AAA", natom, L1_fc_n_out, n_funcs_AAA);
    grace::kk::MemKK::realloc_kokkos(d_AA2, "g2l:AA2", natom, L1_fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_AAAA, "g2l:AAAA", natom, L1_fc_n_out, n_funcs_AAAA);
    grace::kk::MemKK::realloc_kokkos(d_I1, "g2l:I1", natom, I1_n_funcs, I1_n_out);
    grace::kk::MemKK::realloc_kokkos(d_rho, "g2l:rho", natom, rho_n_out);
    grace::kk::MemKK::realloc_kokkos(d_I_nl_LN, "g2l:I_nl_LN", natom, rho_n_out);

    // Layer 2 per-atom
    grace::kk::MemKK::realloc_kokkos(d_B0, "g2l:B0", natom, L2_nradmax, nlm);
    grace::kk::MemKK::realloc_kokkos(d_YI_pair, "g2l:YI_pair", need_bond_buffers() ? natom : 1, L2_nradmax, n_yi_pairs);
    grace::kk::MemKK::realloc_kokkos(d_YI, "g2l:YI", natom, L2_nradmax, n_yi_out_funcs);
    grace::kk::MemKK::realloc_kokkos(d_B, "g2l:B", natom, B_n_out, B_n_funcs);
    grace::kk::MemKK::realloc_kokkos(d_B1, "g2l:B1", natom, L2_fc_n_out, n_funcs_B1);
    grace::kk::MemKK::realloc_kokkos(d_BB, "g2l:BB", natom, L2_fc_n_out, n_funcs_BB);
    grace::kk::MemKK::realloc_kokkos(d_BB1, "g2l:BB1", natom, L2_fc_n_out, n_funcs_BB);
    grace::kk::MemKK::realloc_kokkos(d_BBB, "g2l:BBB", natom, L2_fc_n_out, n_funcs_BBB);
    grace::kk::MemKK::realloc_kokkos(d_BB2, "g2l:BB2", natom, L2_fc_n_out, n_funcs_BB);
    grace::kk::MemKK::realloc_kokkos(d_BBBB, "g2l:BBBB", natom, L2_fc_n_out, n_funcs_BBBB);
    grace::kk::MemKK::realloc_kokkos(d_I2, "g2l:I2", natom, I2_n_out);
    grace::kk::MemKK::realloc_kokkos(d_I_0_LN, "g2l:I_0_LN", natom, I2_n_out);

    // Adjoint views (Layer 1)
    grace::kk::MemKK::realloc_kokkos(d_rho_adj, "g2l:rho_adj", natom, rho_n_out);
    grace::kk::MemKK::realloc_kokkos(d_I_nl_LN_adj, "g2l:I_nl_LN_adj", natom, rho_n_out);
    grace::kk::MemKK::realloc_kokkos(d_I1_adj, "g2l:I1_adj", natom, I1_n_funcs, I1_n_out);
    grace::kk::MemKK::realloc_kokkos(d_AAAA_adj, "g2l:AAAA_adj", natom, L1_fc_n_out, n_funcs_AAAA);
    grace::kk::MemKK::realloc_kokkos(d_AA2_adj, "g2l:AA2_adj", natom, L1_fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_AAA_adj, "g2l:AAA_adj", natom, L1_fc_n_out, n_funcs_AAA);
    grace::kk::MemKK::realloc_kokkos(d_AA1_adj, "g2l:AA1_adj", natom, L1_fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_AA_adj, "g2l:AA_adj", natom, L1_fc_n_out, n_funcs_AA);
    grace::kk::MemKK::realloc_kokkos(d_A1_adj, "g2l:A1_adj", natom, L1_fc_n_out, n_funcs_A1);
    grace::kk::MemKK::realloc_kokkos(d_A_adj, "g2l:A_adj", natom, L1_nradmax, nlm);

    // Adjoint views (Layer 2)
    grace::kk::MemKK::realloc_kokkos(d_I_0_LN_adj, "g2l:I_0_LN_adj", natom, I2_n_out);
    grace::kk::MemKK::realloc_kokkos(d_emlp_work, "g2l:emlp_work", natom, (energy_n_layers + 1) * energy_max_dim);
    grace::kk::MemKK::realloc_kokkos(d_I2_adj, "g2l:I2_adj", natom, I2_n_out);
    grace::kk::MemKK::realloc_kokkos(d_BBBB_adj, "g2l:BBBB_adj", natom, L2_fc_n_out, n_funcs_BBBB);
    grace::kk::MemKK::realloc_kokkos(d_BB2_adj, "g2l:BB2_adj", natom, L2_fc_n_out, n_funcs_BB);
    grace::kk::MemKK::realloc_kokkos(d_BBB_adj, "g2l:BBB_adj", natom, L2_fc_n_out, n_funcs_BBB);
    grace::kk::MemKK::realloc_kokkos(d_BB1_adj, "g2l:BB1_adj", natom, L2_fc_n_out, n_funcs_BB);
    grace::kk::MemKK::realloc_kokkos(d_BB_adj, "g2l:BB_adj", natom, L2_fc_n_out, n_funcs_BB);
    grace::kk::MemKK::realloc_kokkos(d_B1_adj, "g2l:B1_adj", natom, L2_fc_n_out, n_funcs_B1);
    grace::kk::MemKK::realloc_kokkos(d_B_adj, "g2l:B_adj", natom, B_n_out, B_n_funcs);
    grace::kk::MemKK::realloc_kokkos(d_YI_adj, "g2l:YI_adj", natom, L2_nradmax, n_yi_out_funcs);
    grace::kk::MemKK::realloc_kokkos(d_B0_adj, "g2l:B0_adj", natom, L2_nradmax, nlm);
  }

  if (need_bond_buffers() &&
      (realloc_all || (int)d_radial_basis.extent(0) < natom || (int)d_radial_basis.extent(1) < maxneigh_in)) {
    grace::kk::MemKK::realloc_kokkos(d_nearest, "g2l:nearest", natom, maxneigh_in);
    grace::kk::MemKK::realloc_kokkos(d_rnorms, "g2l:rnorms", natom, maxneigh_in);
    grace::kk::MemKK::realloc_kokkos(d_rhats, "g2l:rhats", natom, maxneigh_in);
    grace::kk::MemKK::realloc_kokkos(d_mu_j, "g2l:mu_j", natom, maxneigh_in);
    grace::kk::MemKK::realloc_kokkos(d_radial_basis, "g2l:radial_basis", natom, maxneigh_in, nradbase);
    grace::kk::MemKK::realloc_kokkos(d_dradial_basis, "g2l:dradial_basis", natom, maxneigh_in, nradbase);
    grace::kk::MemKK::realloc_kokkos(d_R_nl, "g2l:R_nl", natom, maxneigh_in, L1_nradmax, lmax + 1);
    grace::kk::MemKK::realloc_kokkos(d_R1_nl, "g2l:R1_nl", natom, maxneigh_in, L2_nradmax, lmax + 1);
    grace::kk::MemKK::realloc_kokkos(d_Y_bond, "g2l:Y_bond", natom, maxneigh_in, (lmax + 1) * (lmax + 1));
    // Precision specialization: float precomputes the output-layer
    // radial derivative into d_DR{,1}_nl; fp64 keeps the raw last-hidden deriv in
    // d_dh2_R{,1} and forms DR inline (bit-identical, no fp64 regression). Only the
    // path actually used by this instantiation is allocated.
    if constexpr (sizeof(NNScalar) == 4) {
      grace::kk::MemKK::realloc_kokkos(d_DR_nl, "g2l:DR_nl", natom, maxneigh_in, L1_nradmax, lmax + 1);
      grace::kk::MemKK::realloc_kokkos(d_DR1_nl, "g2l:DR1_nl", natom, maxneigh_in, L2_nradmax, lmax + 1);
    } else {
      grace::kk::MemKK::realloc_kokkos(d_dh2_R, "g2l:dh2_R", natom, maxneigh_in, mlp_rad_R_max_dim);
      grace::kk::MemKK::realloc_kokkos(d_dh2_R1, "g2l:dh2_R1", natom, maxneigh_in, mlp_rad_R1_max_dim);
    }
    grace::kk::MemKK::realloc_kokkos(d_f_ij, "g2l:f_ij", natom, maxneigh_in);
    grace::kk::MemKK::realloc_kokkos(d_f_ij_L2, "g2l:f_ij_L2", natom, maxneigh_in);

#ifdef KOKKOS_ENABLE_CUDA
    // batched cuBLAS radial-MLP scratch (value + fwd-mode deriv).
    // M = natom*maxneigh bonds; widths = max over R and R1 of each layer's dim.
    // Only allocated for the CUDA device float instantiation (else unused).
    if constexpr (is_cuda_v<DeviceType> && sizeof(NNScalar) == 4) {
      const int64_t Mrows = (int64_t) natom * maxneigh_in;
      const int nin0 = nradbase;                                   // shared MLP input width (10)
      const int mh   = std::max(mlp_rad_R_max_dim, mlp_rad_R1_max_dim); // max hidden (64)
      const int mout = std::max(L1_nradmax * (lmax + 1), L2_nradmax * (lmax + 1)); // 210
      grace::kk::MemKK::realloc_kokkos(d_mlp_X,   "g2l:mlp_X",   std::max<int64_t>(Mrows, 1), std::max(nin0, 1));
      grace::kk::MemKK::realloc_kokkos(d_mlp_dX,  "g2l:mlp_dX",  std::max<int64_t>(Mrows, 1), std::max(nin0, 1));
      grace::kk::MemKK::realloc_kokkos(d_mlp_h0,  "g2l:mlp_h0",  std::max<int64_t>(Mrows, 1), std::max(mh, 1));
      grace::kk::MemKK::realloc_kokkos(d_mlp_h1,  "g2l:mlp_h1",  std::max<int64_t>(Mrows, 1), std::max(mh, 1));
      grace::kk::MemKK::realloc_kokkos(d_mlp_dh0, "g2l:mlp_dh0", std::max<int64_t>(Mrows, 1), std::max(mh, 1));
      grace::kk::MemKK::realloc_kokkos(d_mlp_dh1, "g2l:mlp_dh1", std::max<int64_t>(Mrows, 1), std::max(mh, 1));
      grace::kk::MemKK::realloc_kokkos(d_mlp_R,   "g2l:mlp_R",   std::max<int64_t>(Mrows, 1), std::max(mout, 1));

      // block-sparse FC cuBLAS scratch. G = gathered in-columns
      // [n_in x cs*N], C = per-tile SGEMM output [n_out x cs*N]; sized to the max
      // over all 6 FCs (n_in,n_out <= 64; N = max #entries per op). cs = natom.
      if (fc_blas_max_N > 0) {
        const size_t cols = (size_t) fc_blas_max_N * std::max(natom, 1);
        grace::kk::MemKK::realloc_kokkos(d_fc_G, "g2l:fc_G", (size_t) std::max(fc_blas_max_nin, 1) * cols);
        grace::kk::MemKK::realloc_kokkos(d_fc_C, "g2l:fc_C", (size_t) std::max(fc_blas_max_nout, 1) * cols);
      }
    }
#endif
  }
}

// ======================================================================
// grow_global: allocate global (nall-sized) arrays for inter-layer comm
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::grow_global(int nall)
{
  if (realloc_all || (int)d_I_global.extent(0) < nall) {
    grace::kk::MemKK::realloc_kokkos(d_I_global, "g2l:I_global", nall, I_n_funcs, I_n_out);
    grace::kk::MemKK::realloc_kokkos(d_I_nl_LN_global, "g2l:I_nl_LN_global", nall, rho_n_out);
    grace::kk::MemKK::realloc_kokkos(d_grad_I_global, "g2l:grad_I_global", nall, I_n_funcs, I_n_out);
    grace::kk::MemKK::realloc_kokkos(d_I_nl_LN_adj_global, "g2l:I_nl_LN_adj_global", nall, rho_n_out);
    // host copies: on first use (copy_rows_to_host, host comm only); the old ones go with the old device views
    grace::kk::MemKK::drop(h_I_global, h_grad_I_global);
  }
}

// ======================================================================
// compute_Y_bond_chunk: fill d_Y_bond from d_rhats for the current chunk.
// Must be called by Phase 1 (forward), and by Phase 2 / Phase 5 recompute
// branches when inum > chunksize — otherwise d_Y_bond holds stale values
// from Phase 1's last chunk and downstream kernels read wrong harmonics.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_Y_bond_chunk()
{
  auto Y_bond = d_Y_bond;
  auto rhats = d_rhats;
  auto nc_v = d_ncount;
  auto idx_s = d_idx_sph;
  auto alm_v = alm; auto blm_v = blm; auto cl_v = cl; auto dl_v = dl;
  int lmax_v = lmax;
  GeomScalar sq2_v = sq2;
  int cs = chunk_size;
  int mn = maxneigh;

  Kokkos::parallel_for("ComputeYBond",
    Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, mn}),
    KOKKOS_LAMBDA(const int ii, const int jj) {
      if (jj >= nc_v(ii)) return;
      const GeomScalar rx = rhats(ii, jj, 0);
      const GeomScalar ry = rhats(ii, jj, 1);
      const GeomScalar rz = rhats(ii, jj, 2);

      GeomScalar plm_arr[15];
      plm_arr[0] = GeomScalar(1.0);
      if (lmax_v > 0) {
        plm_arr[1] = GeomScalar(1.7320508075688772935) * rz;
        plm_arr[2] = GeomScalar(-1.2247448713915890491);
        for (int ll = 2; ll <= lmax_v; ll++) {
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
      for (int l = 0; l <= lmax_v; l++) {
        Y_bond(ii, jj, l*(l+1)) = plm_arr[l*(l+1)/2];
        GeomScalar pm_re = phase_re, pm_im = phase_im;
        for (int m = 1; m <= l; m++) {
          const int fac = (m % 2 == 0) ? 1 : -1;
          const GeomScalar pv = plm_arr[l*(l+1)/2 + m];
          Y_bond(ii, jj, l*(l+1)+m) = sq2_v * fac * pm_re * pv;
          Y_bond(ii, jj, l*(l+1)-m) = sq2_v * fac * pm_im * pv;
          const GeomScalar t_re = pm_re * phase_re - pm_im * phase_im;
          const GeomScalar t_im = pm_re * phase_im + pm_im * phase_re;
          pm_re = t_re; pm_im = t_im;
        }
      }
    });
}

// ======================================================================
// MPI Communication: pack/unpack for I features
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagPackForwardComm, const int& i) const
{
  const int j = d_sendlist(i);
  const int nf = I_n_funcs, no = I_n_out;
  int m = i * nf * no;
  for (int f = 0; f < nf; f++)
    for (int k = 0; k < no; k++)
      v_buf(m++) = d_I_global(j, f, k);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagUnpackForwardComm, const int& i) const
{
  const int j = comm_first + i;
  const int nf = I_n_funcs, no = I_n_out;
  int m = i * nf * no;
  for (int f = 0; f < nf; f++)
    for (int k = 0; k < no; k++)
      d_I_global(j, f, k) = v_buf(m++);
}

// Kokkos-native reverse communication for d_grad_I_global

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagPackReverseComm, const int& i) const
{
  const int j = comm_first + i;
  const int nf = I_n_funcs, no = I_n_out;
  int m = i * nf * no;
  for (int f = 0; f < nf; f++)
    for (int k = 0; k < no; k++)
      v_buf(m++) = d_grad_I_global(j, f, k);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagUnpackReverseComm, const int& i) const
{
  const int j = d_sendlist(i);
  const int nf = I_n_funcs, no = I_n_out;
  int m = i * nf * no;
  for (int f = 0; f < nf; f++)
    for (int k = 0; k < no; k++)
      Kokkos::atomic_add(&d_grad_I_global(j, f, k), v_buf(m++));
}

// ======================================================================
// compute: main 2-layer pipeline
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
typename Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::RunOutput
Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::run(const RunInput &in)
{
  if (!setup_done) GRACE_KK_ERROR("GRACE-2L/KK: run() before setup() (after set_model())");
  if (in.nlocal < in.inum || in.nall < in.nlocal)
    GRACE_KK_ERROR("GRACE-2L/KK: need inum <= nlocal <= nall (inum {}, nlocal {}, nall {})", in.inum, in.nlocal, in.nall);
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
  // energy-only requests (LAMMPS: ENERGY_ONLY in eflag, e.g. MC fixes, fix_numdiff, compute_fep) skip the backward
  bool do_energy_only = in.energy_only && !debug_no_energy_only_calc;
  d_eatom = in.eatom;
  d_vatom = in.vatom;
  d_cvatom = in.cvatom;
  x = in.x;
  couple_eq = in.couple_eq; couple_bit1 = in.couple_bit1; couple_bit2 = in.couple_bit2;
  couple_lambda = in.couple_lambda; d_mask = in.mask;
  if (couple_eq) {
    if (!couple_supported) GRACE_KK_ERROR("GRACE-2L/KK: two-group coupling needs a fused engine");
    if (couple_eq < 1 || couple_eq > 4) GRACE_KK_ERROR("GRACE-2L/KK: coupling equation {} (must be 1..4)", couple_eq);
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
      d_ilist_sorted = typename Traits::t_int_1d("g2l:ilist_sorted", inum);
      d_type_key = Kokkos::View<long long *, DeviceType>("g2l:type_key", inum);
    }
    auto src = in.ilist; auto ty = type; auto key = d_type_key; auto dst = d_ilist_sorted;
    const long long n = inum;
    Kokkos::parallel_for("grace_2l::type_key", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii) { key(ii) = (long long) ty(src(ii)) * n + ii; });
    Kokkos::sort(Kokkos::subview(key, std::make_pair(0, inum)));
    Kokkos::parallel_for("grace_2l::type_sort", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii) { dst(ii) = src((int) (key(ii) % n)); });
    d_ilist = d_ilist_sorted;
  }
  maxneigh = 0;
  {
    auto il = in.ilist; auto nn = in.numneigh;
    Kokkos::parallel_reduce("grace_2l::find_maxneigh", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii, int &mx) { const int c = nn[il[ii]]; if (mx < c) mx = c; },
        Kokkos::Max<int>(maxneigh));
  }
  maxneigh = std::max(maxneigh, 0);    // Kokkos::Max over no atoms (an empty rank) gives INT_MIN
  nall = in.nall;
  begin_run();
  choose_chunk();
  const int chunk_limit = chunk_cap();

  chunk_size = std::min(chunk_limit, inum);
  chunk_offset = 0;
  grow(chunk_size, maxneigh);

  // Allocate global arrays (nlocal + nghost)
  grow_global(nall);
  realloc_all = false;

  // ---- UQ / extrapolation grade: allocate per-atom outputs up front. The
  // basis-RP feature is assembled in two parts: the L1 part is projected in
  // Phase 1 (where the L1 products d_A.. are current) into d_uq_z, and Phase 3
  // adds the L2 part + runs the GMM. All UQ kernels are any_uq-gated, so the
  // energy path is byte-for-byte unchanged when UQ is not requested. ----
  const bool any_uq = in.uq && has_uq;
  uq_on = any_uq;
  if (any_uq) {
    // Also reallocate if d_uq_z's projection width no longer matches uq_rp_dim: a
    // mid-session pair_coeff that loads a model with a different rp_dim (while nlocal
    // has not grown) would otherwise leave d_uq_z sized to the OLD width, and the
    // L1basis kernel would write d_uq_z(i,d) out of bounds.
    if ((int)d_gamma.extent(0) < nlocal || (int)d_uq_z.extent(1) != uq_rp_dim) {
      grace::kk::MemKK::realloc_kokkos(d_gamma, "g2l:d_gamma", nlocal);
      grace::kk::MemKK::realloc_kokkos(d_sigma, "g2l:d_sigma", nlocal);
      grace::kk::MemKK::realloc_kokkos(d_gmm_cluster, "g2l:d_gmm_cluster", nlocal);
      grace::kk::MemKK::realloc_kokkos(d_uq_z, "g2l:d_uq_z", nlocal, uq_rp_dim);  // per-atom RAW L1 projection
      grace::kk::MemKK::realloc_kokkos(d_uq_n2L1, "g2l:d_uq_n2L1", nlocal);       // ||B^(L1)||^2 carry
    }
  }

  // ============ PHASE 1: Layer 1 forward (chunked) ============
  // Process all chunks to compute I and I_nl_LN, stored in global arrays

  Kokkos::deep_copy(d_I_global.base(), 0.0);
  Kokkos::deep_copy(d_I_nl_LN_global, 0.0);
  if (!do_energy_only) {
    Kokkos::deep_copy(d_grad_I_global.base(), 0.0);
    Kokkos::deep_copy(d_I_nl_LN_adj_global, 0.0);
  }

  chunk_offset = 0;
  while (chunk_offset < inum) {
    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;
    begin_L1_chunk(false);   // hook: Grace2LFusedKK keeps the L1 forward state per chunk

    // d_A zeroed inline inside ComputeAi kernel
    // d_rho zeroed inside FusedReduceN_rho kernel (uses = not +=)
    // d_I1 zeroed inside FusedReduceN_I1

    int team_size = 1;
    int vector_length = 1;
    if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
      team_size = 32;

    stage_L1_bond();

    // ---- Stages 5-10: FC A1, Product AA, FC AA1, Product AAA, FC AA2, Product AAAA ----
    // (Same pattern as 1L but with L1_fc_n_out=42 and L1_nradmax=42)

    // Stage 5: FC A1 = FC(A, A)
    {
      auto A_in = d_A; auto A1_out = d_A1;
      auto wl = d_fc_A1_wl; auto wr = d_fc_A1_wr;
      auto wtl = d_fc_A1_wtl; auto wtr = d_fc_A1_wtr;
      auto ct = d_fc_A1_ct; auto cf = d_fc_A1_cf;
      auto nof = d_fc_A1_nof;
      NNScalar nl = fc_A1_nl, nr = fc_A1_nr;
      int n_out = L1_fc_n_out, n_lm = n_funcs_A1;
      int n_collect = d_fc_A1_cf.extent(0), cs = chunk_size;

      if (!fc_forward_fast(FCLayer::A1, d_A, d_A, d_A1))
      Kokkos::parallel_for("FC_A1",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, n_out}),
        KOKKOS_LAMBDA(const int ii, const int k) {
          int n_in = A_in.extent(1);
          for (int lm = 0; lm < n_lm; lm++) {
            int tile = wtl(lm);
            NNScalar sum = 0.0;
            for (int n = 0; n < n_in; n++) sum += wl(k, n, tile) * A_in(ii, n, lm);
            A1_out(ii, k, lm) = sum * nof(lm) * nl;
          }
          for (int idx = 0; idx < n_collect; idx++) {
            int src_lm = cf(idx), tgt_lm = ct(idx), tile = wtr(idx);
            NNScalar sum = 0.0;
            for (int n = 0; n < n_in; n++) sum += wr(k, n, tile) * A_in(ii, n, src_lm);
            A1_out(ii, k, tgt_lm) += sum * nof(tgt_lm) * nr;
          }
        });
    }

    stage_prod_AA();

    // Stage 7: FC AA1 = FC(AA, A)
    {
      auto left_in = d_AA; auto right_in = d_A; auto out = d_AA1;
      auto wl = d_fc_AA1_wl; auto wr = d_fc_AA1_wr;
      auto wtl = d_fc_AA1_wtl; auto wtr = d_fc_AA1_wtr;
      auto ct = d_fc_AA1_ct; auto cf = d_fc_AA1_cf; auto nof = d_fc_AA1_nof;
      NNScalar nl = fc_AA1_nl, nr = fc_AA1_nr;
      int no = L1_fc_n_out, nlm = n_funcs_AA;
      int ncol = d_fc_AA1_cf.extent(0), cs = chunk_size;

      if (!fc_forward_fast(FCLayer::AA1, d_AA, d_A, d_AA1))
      Kokkos::parallel_for("FC_AA1",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, no}),
        KOKKOS_LAMBDA(const int ii, const int k) {
          int nin_l = left_in.extent(1);
          for (int lm = 0; lm < nlm; lm++) {
            int tile = wtl(lm);
            NNScalar sum = 0.0;
            for (int n = 0; n < nin_l; n++) sum += wl(k, n, tile) * left_in(ii, n, lm);
            out(ii, k, lm) = sum * nof(lm) * nl;
          }
          int nin_r = right_in.extent(1);
          for (int idx = 0; idx < ncol; idx++) {
            int src = cf(idx), tgt = ct(idx), tile = wtr(idx);
            NNScalar sum = 0.0;
            for (int n = 0; n < nin_r; n++) sum += wr(k, n, tile) * right_in(ii, n, src);
            out(ii, k, tgt) += sum * nof(tgt) * nr;
          }
        });
    }

    stage_prod_AAA();

    // Stage 9: FC AA2 = FC(AA, A)
    {
      auto left_in = d_AA; auto right_in = d_A; auto out = d_AA2;
      auto wl = d_fc_AA2_wl; auto wr = d_fc_AA2_wr;
      auto wtl = d_fc_AA2_wtl; auto wtr = d_fc_AA2_wtr;
      auto ct = d_fc_AA2_ct; auto cf = d_fc_AA2_cf; auto nof = d_fc_AA2_nof;
      NNScalar nl = fc_AA2_nl, nr = fc_AA2_nr;
      int no = L1_fc_n_out, nlm = n_funcs_AA;
      int ncol = d_fc_AA2_cf.extent(0), cs = chunk_size;

      if (!fc_forward_fast(FCLayer::AA2, d_AA, d_A, d_AA2))
      Kokkos::parallel_for("FC_AA2",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, no}),
        KOKKOS_LAMBDA(const int ii, const int k) {
          int nin_l = left_in.extent(1);
          for (int lm = 0; lm < nlm; lm++) {
            int tile = wtl(lm);
            NNScalar sum = 0.0;
            for (int n = 0; n < nin_l; n++) sum += wl(k, n, tile) * left_in(ii, n, lm);
            out(ii, k, lm) = sum * nof(lm) * nl;
          }
          int nin_r = right_in.extent(1);
          for (int idx = 0; idx < ncol; idx++) {
            int src = cf(idx), tgt = ct(idx), tile = wtr(idx);
            NNScalar sum = 0.0;
            for (int n = 0; n < nin_r; n++) sum += wr(k, n, tile) * right_in(ii, n, src);
            out(ii, k, tgt) += sum * nof(tgt) * nr;
          }
        });
    }

    stage_prod_AAAA();

    stage_reduce_I1();

    // ---- Stage 12: ReduceN I (equivariant, NOT element-dependent) ----
    // I[i, fo, k] = Σ_c W[k, n, tile] * I1[i, n, ci(c)] where fo = tsi(c)
    {
      auto I1_in = d_I1;
      auto I_global = d_I_global;
      auto W = d_I_reduce_I1_W;
      auto ci = d_I_ci_I1; auto wlt = d_I_wlt_I1; auto tsi = d_I_tsi_I1;
      NNScalar norm = I_reduce_I1_norm;
      int nc = I_nc_I1;
      int nIout = I_n_out;
      auto il = d_ilist;
      int co = chunk_offset, cs = chunk_size;

      Kokkos::parallel_for("ReduceN_I",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nIout}),
        KOKKOS_LAMBDA(const int ii, const int k) {
          const int i = il[ii + co];
          for (int c = 0; c < nc; c++) {
            const int fi = ci(c), fo = tsi(c), tile = wlt(c);
            const int nin = W.extent(1);
            NNScalar val = 0.0;
            for (int n = 0; n < nin; n++) val += W(k, n, tile) * I1_in(ii, fi, n);
            I_global(i, fo, k) += val * norm;
          }
        });
    }

    // ---- Stage 13: ReduceN rho (scalar, element-dependent) + RMSNorm I_nl_LN ----
    // First kernel uses = (not +=) to zero d_rho inline, eliminating deep_copy.
    {
      auto rho_out = d_rho;
      auto il = d_ilist; auto mp = d_map; auto tp = type;
      int co = chunk_offset, cs = chunk_size, nrho = rho_n_out;

      stage_reduce_rho();

      // RMSNorm I_nl_LN (type=only_nonlin): rho[0] pass-through, rho[1:] normalized
      {
        auto rho_in = d_rho;
        auto lnout = d_I_nl_LN;
        auto scale = d_I_nl_LN_scale;
        auto I_nl_glob = d_I_nl_LN_global;
        int n_out_ln = rho_n_out;

        Kokkos::parallel_for("RMSNorm_I_nl_LN", Kokkos::RangePolicy<DeviceType>(0, cs),
          KOKKOS_LAMBDA(const int ii) {
            const int i = il[ii + co];
            // Pass through linear term
            lnout(ii, 0) = rho_in(ii, 0);
            I_nl_glob(i, 0) = rho_in(ii, 0);
            // RMS of nonlinear part
            NNScalar sq_sum = 0.0;
            for (int k = 1; k < n_out_ln; k++)
              sq_sum += rho_in(ii, k) * rho_in(ii, k);
            NNScalar rms_inv = 1.0 / Kokkos::sqrt(sq_sum / (n_out_ln - 1) + 1e-8);
            for (int k = 1; k < n_out_ln; k++) {
              NNScalar val = rho_in(ii, k) * rms_inv * scale(k - 1);
              lnout(ii, k) = val;
              I_nl_glob(i, k) = val;
            }
          });
      }
    }

    // ---- UQ basis-RP: project the L1 scalar basis into d_uq_z for this chunk.
    // d_A..d_AAAA hold the current chunk's L1 products now (they are reused per
    // chunk, so the projection MUST happen here, not in the Phase-3 UQ kernel). ----
    if (any_uq) {
#ifdef KOKKOS_ENABLE_CUDA
      if (blas_handle && (uq_sgemm || blas::has_dgemm(blas_handle))) {    // fp64 UQ: dgemm
        compute_uq_blas_L1();   // gather rho block -> DGEMM -> raw L1 proj into d_uq_z
      } else
#endif
      {
      // Team-per-atom: lanes split the L1 (rho) B·R projection over rp_dim with a
      // register accumulator (no per-thread array => no local-memory spill).
      auto probe = Kokkos::TeamPolicy<DeviceType, TagComputeUQ_L1basis>(chunk_size, 1, 1);
      int ts = probe.team_size_max(*this, Kokkos::ParallelForTag());
      if (ts > 128) ts = 128; if (ts < 1) ts = 1;
      Kokkos::parallel_for("ComputeUQ_L1basis",
          Kokkos::TeamPolicy<DeviceType, TagComputeUQ_L1basis>(chunk_size, ts, 1), *this);
      }
    }

    Kokkos::fence();
    chunk_offset += chunk_size;
  }

  // ============ PHASE 2: Communicate I features to ghost atoms ============
  if (in.exchange) in.exchange->forward();

  // ============ PHASE 3: Layer 2 forward (chunked) ============

  chunk_size = std::min(chunk_limit, inum);
  chunk_offset = 0;

  while (chunk_offset < inum) {
    if (chunk_size > inum - chunk_offset)
      chunk_size = inum - chunk_offset;

    // d_B0 zeroed inline inside ComputeAi_B0 kernel
    // d_YI zeroed inline inside ComputeYI kernel
    // d_B zeroed inline inside FusedReduceN_B
    // d_I2 zeroed inside first ReduceN_I2 kernel (uses = not +=)

    int team_size = 1;
    int vector_length = 1;
    if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
      team_size = 32;

    stage_L2_bond_fwd();

    stage_reduce_B();

    // ---- L2 Stages 5-10: FC B1, Product BB, FC BB1, Product BBB, FC BB2, Product BBBB ----

    // Stage 5: FC B1 = FC(B, B) — fused L+R, no atomics
    {
      auto B_in = d_B; auto B1_out = d_B1;
      auto wl = d_fc_B1_wl; auto wr = d_fc_B1_wr;
      auto wtl = d_fc_B1_wtl; auto wtr = d_fc_B1_wtr;
      auto ct = d_fc_B1_ct; auto cf = d_fc_B1_cf; auto nof = d_fc_B1_nof;
      NNScalar nl = fc_B1_nl, nr = fc_B1_nr;
      int no = L2_fc_n_out, nlm = n_funcs_B1;
      int ncol = d_fc_B1_cf.extent(0), cs = chunk_size;

      if (!fc_forward_fast(FCLayer::B1, d_B, d_B, d_B1))
      Kokkos::parallel_for("FC_B1",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, no}),
        KOKKOS_LAMBDA(const int ii, const int k) {
          int nin = B_in.extent(1);
          // Left contribution: B1(ii,k,lm) = Σ_n wl(k,n,tile) * B(ii,n,lm)
          for (int lm = 0; lm < nlm; lm++) {
            int tile = wtl(lm);
            NNScalar sum = 0.0;
            for (int n = 0; n < nin; n++) sum += wl(k, n, tile) * B_in(ii, n, lm);
            B1_out(ii, k, lm) = sum * nof(lm) * nl;
          }
          // Right contribution: B1(ii,k,tgt) += Σ_n wr(k,n,tile) * B(ii,n,src)
          for (int idx = 0; idx < ncol; idx++) {
            int src = cf(idx), tgt = ct(idx), tile = wtr(idx);
            NNScalar sum = 0.0;
            for (int n = 0; n < nin; n++) sum += wr(k, n, tile) * B_in(ii, n, src);
            B1_out(ii, k, tgt) += sum * nof(tgt) * nr;
          }
        });
    }

    stage_prod_BB();

    // Stage 7: FC BB1 = FC(BB, B) — fused L+R, no atomics
    {
      auto left_in = d_BB; auto right_in = d_B; auto out = d_BB1;
      auto wl = d_fc_BB1_wl; auto wr = d_fc_BB1_wr;
      auto wtl = d_fc_BB1_wtl; auto wtr = d_fc_BB1_wtr;
      auto ct = d_fc_BB1_ct; auto cf = d_fc_BB1_cf; auto nof = d_fc_BB1_nof;
      NNScalar nl = fc_BB1_nl, nr = fc_BB1_nr;
      int no = L2_fc_n_out, nlm = n_funcs_BB;
      int ncol = d_fc_BB1_cf.extent(0), cs = chunk_size;

      if (!fc_forward_fast(FCLayer::BB1, d_BB, d_B, d_BB1))
      Kokkos::parallel_for("FC_BB1",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, no}),
        KOKKOS_LAMBDA(const int ii, const int k) {
          int nin_l = left_in.extent(1), nin_r = right_in.extent(1);
          for (int lm = 0; lm < nlm; lm++) {
            int tile = wtl(lm);
            NNScalar sum = 0.0;
            for (int n = 0; n < nin_l; n++) sum += wl(k, n, tile) * left_in(ii, n, lm);
            out(ii, k, lm) = sum * nof(lm) * nl;
          }
          for (int idx = 0; idx < ncol; idx++) {
            int src = cf(idx), tgt = ct(idx), tile = wtr(idx);
            NNScalar sum = 0.0;
            for (int n = 0; n < nin_r; n++) sum += wr(k, n, tile) * right_in(ii, n, src);
            out(ii, k, tgt) += sum * nof(tgt) * nr;
          }
        });
    }

    stage_prod_BBB();

    // Stage 9: FC BB2 = FC(BB, B) — fused L+R, no atomics
    {
      auto left_in = d_BB; auto right_in = d_B; auto out = d_BB2;
      auto wl = d_fc_BB2_wl; auto wr = d_fc_BB2_wr;
      auto wtl = d_fc_BB2_wtl; auto wtr = d_fc_BB2_wtr;
      auto ct = d_fc_BB2_ct; auto cf = d_fc_BB2_cf; auto nof = d_fc_BB2_nof;
      NNScalar nl = fc_BB2_nl, nr = fc_BB2_nr;
      int no = L2_fc_n_out, nlm = n_funcs_BB;
      int ncol = d_fc_BB2_cf.extent(0), cs = chunk_size;

      if (!fc_forward_fast(FCLayer::BB2, d_BB, d_B, d_BB2))
      Kokkos::parallel_for("FC_BB2",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, no}),
        KOKKOS_LAMBDA(const int ii, const int k) {
          int nin_l = left_in.extent(1), nin_r = right_in.extent(1);
          for (int lm = 0; lm < nlm; lm++) {
            int tile = wtl(lm);
            NNScalar sum = 0.0;
            for (int n = 0; n < nin_l; n++) sum += wl(k, n, tile) * left_in(ii, n, lm);
            out(ii, k, lm) = sum * nof(lm) * nl;
          }
          for (int idx = 0; idx < ncol; idx++) {
            int src = cf(idx), tgt = ct(idx), tile = wtr(idx);
            NNScalar sum = 0.0;
            for (int n = 0; n < nin_r; n++) sum += wr(k, n, tile) * right_in(ii, n, src);
            out(ii, k, tgt) += sum * nof(tgt) * nr;
          }
        });
    }

    stage_prod_BBBB();

    // ---- L2 Stage 11: ReduceN I2 (scalar, element-dependent) ----
    // First kernel uses = (not +=) to zero d_I2 inline, eliminating deep_copy.
    stage_reduce_I2();

    // ---- L2 Stage 12: RMSNorm I_0_LN (type=full) ----
    {
      auto I2_in = d_I2;
      auto lnout = d_I_0_LN;
      auto scale = d_I_0_LN_scale;
      int n_out_ln = I2_n_out, cs = chunk_size;

      Kokkos::parallel_for("RMSNorm_I_0_LN", Kokkos::RangePolicy<DeviceType>(0, cs),
        KOKKOS_LAMBDA(const int ii) {
          NNScalar sq_sum = 0.0;
          for (int k = 0; k < n_out_ln; k++)
            sq_sum += I2_in(ii, k) * I2_in(ii, k);
          NNScalar rms_inv = 1.0 / Kokkos::sqrt(sq_sum / n_out_ln + 1e-8);
          for (int k = 0; k < n_out_ln; k++)
            lnout(ii, k) = I2_in(ii, k) * rms_inv * scale(k);
        });
    }

    // ---- L2 Stage 13: ComputeMLPEnergy ----
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

    // ---- UQ / extrapolation grade (forward only; reads Phase-3 L2 features) ----
    // Team-per-atom: lanes split the I2 (L2) projection add over rp_dim, the cluster
    // search over the feature dim, and the D×D Mahalanobis over its quadratic form.
    if (any_uq) {
#ifdef KOKKOS_ENABLE_CUDA
      if (blas_handle && (uq_sgemm || blas::has_dgemm(blas_handle))) {    // fp64 UQ: dgemm
        compute_uq_blas_L2();   // gather I2 block -> DGEMM -> add L1 carry, normalize, GMM
      } else
#endif
      {
      // f_sh[D] + delta_sh[D] + (n2_L2, nrm_full) scalars (+ alignment slack).
      const int scratch_bytes = (2 * uq_D + 8) * (int)sizeof(double);
      auto probe = Kokkos::TeamPolicy<DeviceType, TagComputeUQ>(chunk_size, 1, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
      int ts = probe.team_size_max(*this, Kokkos::ParallelForTag());
      if (ts > 128) ts = 128; if (ts < 1) ts = 1;
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeUQ>(chunk_size, ts, 1)
          .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
      Kokkos::parallel_for("ComputeUQ", policy, *this);
      }
    }

    // ============ BACKWARD PASS — Layer 2 (Forces) ============
    if (!do_energy_only) {

    // Zero L2 adjoint arrays in one launch. At 1024 atoms these tiny memset
    // kernels are launch-bound, and the arrays are all per-atom chunk views.
    stage_zero_L2_adj();

    // ---- B2-1: ReverseMLPEnergy → d_I_nl_LN_adj, d_I_0_LN_adj ----
    {
      auto I_nl_in = d_I_nl_LN_global; auto I_0_in = d_I_0_LN;
      auto nl_adj = d_I_nl_LN_adj; auto i0_adj = d_I_0_LN_adj;
      auto nl_adj_glob = d_I_nl_LN_adj_global;
      auto eW = d_energy_W; auto enorms = d_energy_norms; auto edims = d_energy_dims;
      int en_layers = energy_n_layers;
      int emax = energy_max_dim;
      int cs = chunk_size;
      int eact = energy_activation;
      auto il = d_ilist; int co = chunk_offset;
      const NNScalar os = output_scale;
      auto wk = d_emlp_work;    // per atom: pre-activations [en_layers - 1][emax], then two adjoint rows [emax]

      Kokkos::parallel_for("ReverseMLPEnergy", Kokkos::RangePolicy<DeviceType>(0, cs),
        KOKKOS_LAMBDA(const int ii) {
          // the atom's scratch row wk(ii, :) (on the GPU LayoutLeft: one element of all atoms of a warp is contiguous,
          // as in the interleaved local memory it replaces)
          int adj_cur = (en_layers - 1) * emax, adj_nxt = adj_cur + emax;    // column offsets of the two rows

          // Input: I_nl_LN[:,1:] + I_0_LN[:,1:]
          const int nin0 = edims(0);

          // Forward pass: recompute hidden pre-activations
          {
            const int nout = edims(1);
            for (int j = 0; j < nout; j++) {
              NNScalar sum = 0.0;
              const int i = il[ii + co];
              for (int k = 0; k < nin0; k++)
                sum += eW(0, k, j) * (I_nl_in(i, k + 1) + I_0_in(ii, k + 1));
              wk(ii, 0 * emax + j) = sum * enorms(0);
            }
          }
          for (int layer = 1; layer < en_layers - 1; layer++) {
            const int nin = edims(layer); const int nout = edims(layer + 1);
            for (int j = 0; j < nout; j++) {
              NNScalar sum = 0.0;
              for (int k = 0; k < nin; k++) {
                NNScalar pa = wk(ii, (layer - 1) * emax + k);
                NNScalar act;
                if (eact == 1) act = Kokkos::tanh(pa);
                else { NNScalar sig = 1.0 / (1.0 + Kokkos::exp(-pa)); act = pa * sig; }
                sum += eW(layer, k, j) * act;
              }
              wk(ii, layer * emax + j) = sum * enorms(layer);
            }
          }

          // Backward pass.
          // Multiply seed by output_scale so the chain rule scales all forces.
          {
            const int last = en_layers - 1;
            const int nin = edims(last); const int nout = edims(last + 1);
            for (int k = 0; k < nin; k++) {
              NNScalar val = 0.0;
              for (int o = 0; o < nout; o++) val += eW(last, k, o) * enorms(last);
              wk(ii, adj_cur + k) = val * os;
            }
          }
          for (int layer = en_layers - 2; layer >= 0; layer--) {
            const int nin = edims(layer); const int nout = edims(layer + 1);
            for (int j = 0; j < nout; j++) {
              NNScalar pa = wk(ii, layer * emax + j);
              if (eact == 1) {
                NNScalar t = Kokkos::tanh(pa);
                wk(ii, adj_cur + j) *= (1.0 - t * t);  // tanh'
              } else {
                NNScalar sig = 1.0 / (1.0 + Kokkos::exp(-pa));
                wk(ii, adj_cur + j) *= sig * (1.0 + pa * (1.0 - sig));  // silu'
              }
            }
            for (int k = 0; k < nin; k++) {
              NNScalar val = 0.0;
              for (int j = 0; j < nout; j++) val += wk(ii, adj_cur + j) * eW(layer, k, j);
              wk(ii, adj_nxt + k) = val * enorms(layer);
            }
            const int tmp = adj_cur; adj_cur = adj_nxt; adj_nxt = tmp;
          }

          // adj_cur now holds d_output/d_input[k] for k=0..nin0-1
          // Both I_nl_LN and I_0_LN get the same gradient (sum rule).
          // Linear skip seeds scaled by output_scale (chain rule for E*output_scale).
          const int i = il[ii + co];
          nl_adj(ii, 0) = os;   // linear skip term, scaled
          i0_adj(ii, 0) = os;
          nl_adj_glob(i, 0) = os;
          for (int k = 0; k < nin0; k++) {
            nl_adj(ii, k + 1) = wk(ii, adj_cur + k);
            i0_adj(ii, k + 1) = wk(ii, adj_cur + k);
            nl_adj_glob(i, k + 1) = wk(ii, adj_cur + k);
          }
        });
    }

    // ---- B2-2: Reverse I_0_LN (RMSNorm, full type) → d_I2_adj ----
    {
      auto I2_in = d_I2; auto ln_adj = d_I_0_LN_adj; auto ln_out = d_I_0_LN;
      auto I2_adj_out = d_I2_adj;
      auto scale = d_I_0_LN_scale;
      int n_out_ln = I2_n_out, cs = chunk_size;

      Kokkos::parallel_for("ReverseRMSNorm_I_0_LN", Kokkos::RangePolicy<DeviceType>(0, cs),
        KOKKOS_LAMBDA(const int ii) {
          // Recompute rms_inv
          NNScalar sq_sum = 0.0;
          for (int k = 0; k < n_out_ln; k++)
            sq_sum += I2_in(ii, k) * I2_in(ii, k);
          NNScalar rms_inv = 1.0 / Kokkos::sqrt(sq_sum / n_out_ln + 1e-8);

          // dot = Σ_k d_ln[k] * ln_out[k]
          NNScalar dot = 0.0;
          for (int k = 0; k < n_out_ln; k++)
            dot += ln_adj(ii, k) * ln_out(ii, k);

          // dx[k] = rms_inv * (d_ln[k] * scale[k] - x[k] * dot * rms_inv / N)
          for (int k = 0; k < n_out_ln; k++)
            I2_adj_out(ii, k) = rms_inv * (ln_adj(ii, k) * scale(k) - I2_in(ii, k) * dot * rms_inv / n_out_ln);
        });
    }

    // ---- B2-3: ReverseReduceN_I2 → d_B_adj, d_BB_adj, d_BBB_adj, d_BBBB_adj ----
    stage_rev_reduce_I2();

    stage_rev_prod_BBBB();

    stage_rev_fc_BB2();

    stage_rev_prod_BBB();

    stage_rev_fc_BB1();

    stage_rev_prod_BB();

    stage_rev_fc_B1();

    stage_rev_reduce_B();

    stage_L2_bond_bwd();

    } // end L2 backward

    Kokkos::fence();
    chunk_offset += chunk_size;
  }

  // ============ PHASE 4: Reverse communicate d_grad_I_global ============
  if (!do_energy_only) {
    if (in.exchange) in.exchange->reverse();
  }

  // ============ PHASE 5: Layer 1 backward (recompute L1 forward + backward) ============
  if (!do_energy_only) {
    chunk_size = std::min(chunk_limit, inum);
    chunk_offset = 0;

    while (chunk_offset < inum) {
      if (chunk_size > inum - chunk_offset)
        chunk_size = inum - chunk_offset;
      begin_L1_chunk(true);

      int team_size = 1;
      int vector_length = 1;
      if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
        team_size = 32;

      if (inum > chunk_limit && need_L1_recompute()) {
      // --- Recompute L1 forward intermediates ---
      // Multi-chunk: L1 data was overwritten by later chunks + L2_ComputeNeigh.
      // d_A zeroed inline inside Recompute_ComputeAi kernel
      // d_rho assigned by stage_recompute_rho (= not +=)

      stage_L1_bond_recompute();
      // Recompute: FC A1
      {
        auto A_in = d_A; auto A1_out = d_A1;
        auto wl = d_fc_A1_wl; auto wr = d_fc_A1_wr;
        auto wtl = d_fc_A1_wtl; auto wtr = d_fc_A1_wtr;
        auto ct = d_fc_A1_ct; auto cf = d_fc_A1_cf; auto nof = d_fc_A1_nof;
        NNScalar nl = fc_A1_nl, nr = fc_A1_nr;
        int n_out = L1_fc_n_out, n_lm = n_funcs_A1;
        int n_collect = d_fc_A1_cf.extent(0), cs = chunk_size;
        if (!fc_forward_fast(FCLayer::A1, d_A, d_A, d_A1))
        Kokkos::parallel_for("Recompute_FC_A1",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, n_out}),
          KOKKOS_LAMBDA(const int ii, const int k) {
            int nin = A_in.extent(1);
            for (int lm = 0; lm < n_lm; lm++) {
              int tile = wtl(lm);
              NNScalar sum = 0.0;
              for (int n = 0; n < nin; n++) sum += wl(k, n, tile) * A_in(ii, n, lm);
              A1_out(ii, k, lm) = sum * nof(lm) * nl;
            }
            for (int idx = 0; idx < n_collect; idx++) {
              int src = cf(idx), tgt = ct(idx), tile = wtr(idx);
              NNScalar sum = 0.0;
              for (int n = 0; n < nin; n++) sum += wr(k, n, tile) * A_in(ii, n, src);
              A1_out(ii, k, tgt) += sum * nof(tgt) * nr;
            }
          });
      }
      stage_prod_AA();
      // Recompute: FC AA1
      {
        auto left_in = d_AA; auto right_in = d_A; auto out = d_AA1;
        auto wl = d_fc_AA1_wl; auto wr = d_fc_AA1_wr;
        auto wtl = d_fc_AA1_wtl; auto wtr = d_fc_AA1_wtr;
        auto ct = d_fc_AA1_ct; auto cf = d_fc_AA1_cf; auto nof = d_fc_AA1_nof;
        NNScalar nl = fc_AA1_nl, nr = fc_AA1_nr;
        int no = L1_fc_n_out, nlm = n_funcs_AA;
        int ncol = d_fc_AA1_cf.extent(0), cs = chunk_size;
        if (!fc_forward_fast(FCLayer::AA1, d_AA, d_A, d_AA1))
        Kokkos::parallel_for("Recompute_FC_AA1",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, no}),
          KOKKOS_LAMBDA(const int ii, const int k) {
            int nin_l = left_in.extent(1);
            for (int lm = 0; lm < nlm; lm++) {
              int tile = wtl(lm);
              NNScalar sum = 0.0;
              for (int n = 0; n < nin_l; n++) sum += wl(k, n, tile) * left_in(ii, n, lm);
              out(ii, k, lm) = sum * nof(lm) * nl;
            }
            int nin_r = right_in.extent(1);
            for (int idx = 0; idx < ncol; idx++) {
              int src = cf(idx), tgt = ct(idx), tile = wtr(idx);
              NNScalar sum = 0.0;
              for (int n = 0; n < nin_r; n++) sum += wr(k, n, tile) * right_in(ii, n, src);
              out(ii, k, tgt) += sum * nof(tgt) * nr;
            }
          });
      }
      stage_prod_AAA();
      // Recompute: FC AA2
      {
        auto left_in = d_AA; auto right_in = d_A; auto out = d_AA2;
        auto wl = d_fc_AA2_wl; auto wr = d_fc_AA2_wr;
        auto wtl = d_fc_AA2_wtl; auto wtr = d_fc_AA2_wtr;
        auto ct = d_fc_AA2_ct; auto cf = d_fc_AA2_cf; auto nof = d_fc_AA2_nof;
        NNScalar nl = fc_AA2_nl, nr = fc_AA2_nr;
        int no = L1_fc_n_out, nlm = n_funcs_AA;
        int ncol = d_fc_AA2_cf.extent(0), cs = chunk_size;
        if (!fc_forward_fast(FCLayer::AA2, d_AA, d_A, d_AA2))
        Kokkos::parallel_for("Recompute_FC_AA2",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, no}),
          KOKKOS_LAMBDA(const int ii, const int k) {
            int nin_l = left_in.extent(1);
            for (int lm = 0; lm < nlm; lm++) {
              int tile = wtl(lm);
              NNScalar sum = 0.0;
              for (int n = 0; n < nin_l; n++) sum += wl(k, n, tile) * left_in(ii, n, lm);
              out(ii, k, lm) = sum * nof(lm) * nl;
            }
            int nin_r = right_in.extent(1);
            for (int idx = 0; idx < ncol; idx++) {
              int src = cf(idx), tgt = ct(idx), tile = wtr(idx);
              NNScalar sum = 0.0;
              for (int n = 0; n < nin_r; n++) sum += wr(k, n, tile) * right_in(ii, n, src);
              out(ii, k, tgt) += sum * nof(tgt) * nr;
            }
          });
      }
      stage_prod_AAAA();
      stage_recompute_rho();    // rho for the RMSNorm backward (rms_inv)
      } // end if (inum > chunksize)

      // --- L1 Backward pass ---
      // Zero L1 adjoint arrays
      stage_zero_L1_adj();

      // ---- B1-1: Reverse I_nl_LN (RMSNorm, only_nonlin) → d_rho_adj ----
      {
        auto rho_in = d_rho;
        auto nl_adj = d_I_nl_LN_adj_global;
        auto I_nl_out = d_I_nl_LN_global;
        auto rho_adj_out = d_rho_adj;
        auto scale = d_I_nl_LN_scale;
        int n_out_ln = rho_n_out, cs = chunk_size;
        auto il = d_ilist; int co = chunk_offset;

        Kokkos::parallel_for("ReverseRMSNorm_I_nl_LN", Kokkos::RangePolicy<DeviceType>(0, cs),
          KOKKOS_LAMBDA(const int ii) {
            const int i = il[ii + co];
            // Pass-through for linear term
            rho_adj_out(ii, 0) = nl_adj(i, 0);  // gradient of linear skip = 1.0

            // Recompute rms_inv for nonlinear part
            NNScalar sq_sum = 0.0;
            for (int k = 1; k < n_out_ln; k++)
              sq_sum += rho_in(ii, k) * rho_in(ii, k);
            NNScalar rms_inv = 1.0 / Kokkos::sqrt(sq_sum / (n_out_ln - 1) + 1e-8);

            // dot = Σ_{k=1}^{n-1} d_ln[k] * ln_out[k]
            NNScalar dot = 0.0;
            for (int k = 1; k < n_out_ln; k++)
              dot += nl_adj(i, k) * I_nl_out(i, k);

            for (int k = 1; k < n_out_ln; k++)
              rho_adj_out(ii, k) = rms_inv * (nl_adj(i, k) * scale(k - 1) - rho_in(ii, k) * dot * rms_inv / (n_out_ln - 1));
          });
      }

      stage_rev_reduce_rho();

      // ---- B1-3: Reverse I → d_I1_adj (from d_grad_I_global) ----
      // I[i, fo, k] = Σ_c I_reduce_I1_W(k, n, tile_c) * I1[i, fi_c, n] * norm
      // d_I1[i, fi_c, n] += Σ_k I_reduce_I1_W(k, n, tile_c) * d_I[i, fo_c, k] * norm
      Kokkos::deep_copy(d_I1_adj, 0.0);
      {
        auto W = d_I_reduce_I1_W; auto ci = d_I_ci_I1; auto wlt = d_I_wlt_I1; auto tsi = d_I_tsi_I1;
        NNScalar norm = I_reduce_I1_norm; int nc = I_nc_I1;
        auto grad_I = d_grad_I_global;
        auto I1_adj_out = d_I1_adj;
        int cs = chunk_size, nIout = I_n_out;
        auto il = d_ilist; int co = chunk_offset;

        Kokkos::parallel_for("RevReduceN_I",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
          KOKKOS_LAMBDA(const int ii, const int c) {
            const int i = il[ii + co];
            int fi = ci(c), fo = tsi(c), tile = wlt(c);
            int nin = W.extent(1);
            for (int n = 0; n < nin; n++) {
              NNScalar val = 0.0;
              for (int k = 0; k < nIout; k++) val += W(k, n, tile) * grad_I(i, fo, k);
              Kokkos::atomic_add(&I1_adj_out(ii, fi, n), val * norm);
            }
          });
      }

      stage_rev_reduce_I1();

      stage_rev_prod_AAAA();
      stage_rev_fc_AA2();
      stage_rev_prod_AAA();
      stage_rev_fc_AA1();
      stage_rev_prod_AA();
      stage_rev_fc_A1();

      stage_L1_bond_bwd();
      chunk_offset += chunk_size;
    } // end L1 backward chunk loop
  } // end !do_energy_only

  RunOutput out;
  out.eng_vdwl = eng_vdwl;
  for (int k = 0; k < 6; k++) out.virial[k] = virial[k];
  check_kernel_launches<DeviceType>("GRACE-2L/KK");
  return out;
}

// ======================================================================
// Kernel: ComputeNeigh
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeNeigh,
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
// Kernel: ComputeRadialBasis - Chebyshev + cutoff
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeRadialBasis,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;
  const int jj = slot.neigh;
  if (jj >= d_ncount(ii)) return;

  const GeomScalar r = d_rnorms(ii, jj);
  const GeomScalar rcut_ij = d_bond_cutoff(d_mu_i(ii), d_mu_j(ii, jj));
  const GeomScalar x_norm = r / rcut_ij;

  const int p = radial_basis_p;
  GeomScalar xp = GeomScalar(1.0);
  for (int ip = 0; ip < p; ip++) xp *= x_norm;
  const GeomScalar xp1 = xp * x_norm, xp2 = xp1 * x_norm;
  const GeomScalar pp1 = GeomScalar(p) * (p + 1);
  const GeomScalar pp2 = GeomScalar(p) * (p + 2);
  const GeomScalar p1p2 = GeomScalar(p + 1) * (p + 2);
  const GeomScalar fcut = GeomScalar(1.0) - GeomScalar(0.5) * p1p2 * xp + pp2 * xp1 - GeomScalar(0.5) * pp1 * xp2;

  GeomScalar dfcut = GeomScalar(0.0);
  if (x_norm > GeomScalar(1e-14)) {
    const GeomScalar xp_m1 = xp / x_norm;
    const GeomScalar coeff = GeomScalar(p) * (p + 1) * (p + 2) * GeomScalar(0.5);
    dfcut = coeff * (-xp_m1 + GeomScalar(2.0) * xp - xp1) / rcut_ij;
  }

  const GeomScalar x_cheb = GeomScalar(2.0) * x_norm - GeomScalar(1.0);
  const GeomScalar dx_cheb_dr = GeomScalar(2.0) / rcut_ij;

  GeomScalar T_prev = GeomScalar(1.0), T_curr = x_cheb;
  GeomScalar dT_prev = GeomScalar(0.0), dT_curr = GeomScalar(1.0);

  d_radial_basis(ii, jj, 0) = T_curr * fcut;
  d_dradial_basis(ii, jj, 0) = dT_curr * dx_cheb_dr * fcut + T_curr * dfcut;

  for (int k = 1; k < nradbase; k++) {
    const GeomScalar T_next = GeomScalar(2.0) * x_cheb * T_curr - T_prev;
    const GeomScalar dT_next = GeomScalar(2.0) * (T_curr + x_cheb * dT_curr) - dT_prev;
    T_prev = T_curr; dT_prev = dT_curr;
    T_curr = T_next; dT_curr = dT_next;
    d_radial_basis(ii, jj, k) = T_curr * fcut;
    d_dradial_basis(ii, jj, k) = dT_curr * dx_cheb_dr * fcut + T_curr * dfcut;
  }
}

// ======================================================================
// Kernel: ComputeMLPRadial_R (Layer 1)
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeMLPRadial_R,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadial_R>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;
  const int jj = slot.neigh;
  if (jj >= d_ncount(ii)) return;

  const int n_layers = mlp_rad_R_n_layers;
  int mlp_dims[GRACE2L_MAX_MLP_LAYERS + 1];
  NNScalar mlp_norms[GRACE2L_MAX_MLP_LAYERS];
  for (int i = 0; i <= n_layers; i++) mlp_dims[i] = d_mlp_rad_R_dims(i);
  for (int i = 0; i < n_layers; i++) mlp_norms[i] = d_mlp_rad_R_norms(i);

  // Buffers sized to max(input, hidden) dims — NOT output dim (which writes to d_R_nl directly).
  // This reduces register pressure from 4*210=840 to 4*128=512 values (or less for typical models).
  NNScalar buf_a[GRACE2L_MAX_MLP_HIDDEN], buf_b[GRACE2L_MAX_MLP_HIDDEN];
  NNScalar dbuf_a[GRACE2L_MAX_MLP_HIDDEN], dbuf_b[GRACE2L_MAX_MLP_HIDDEN];
  NNScalar* h_cur = buf_a; NNScalar* h_nxt = buf_b;
  NNScalar* dh_cur = dbuf_a; NNScalar* dh_nxt = dbuf_b;

  const int n_input = mlp_dims[0];
  for (int k = 0; k < n_input; k++) {
    h_cur[k] = d_radial_basis(ii, jj, k);
    dh_cur[k] = d_dradial_basis(ii, jj, k);
  }

  for (int layer = 0; layer < n_layers - 1; layer++) {
    const int nin = mlp_dims[layer], nout = mlp_dims[layer + 1];
    const NNScalar norm = mlp_norms[layer];
    for (int j = 0; j < nout; j++) {
      NNScalar s = 0.0, ds = 0.0;
      for (int k = 0; k < nin; k++) {
        NNScalar w = d_mlp_rad_R_W(layer, k, j);
        s += w * h_cur[k]; ds += w * dh_cur[k];
      }
      s *= norm; ds *= norm;
      NNScalar sig = 1.0 / (1.0 + Kokkos::exp(-s));
      h_nxt[j] = s * sig;
      dh_nxt[j] = sig * (1.0 + s * (1.0 - sig)) * ds;
    }
    NNScalar* tmp;
    tmp = h_cur; h_cur = h_nxt; h_nxt = tmp;
    tmp = dh_cur; dh_cur = dh_nxt; dh_nxt = tmp;
  }

  const int n_last_hidden = mlp_dims[n_layers - 1];
  const int last_layer = n_layers - 1;
  const NNScalar out_norm = mlp_norms[last_layer];
  // Precision specialization: float precomputes DR = norm*(dh @ W2)
  // alongside the value R (shared W2 load) so ComputeDerivative_L1 reads it; fp64
  // keeps the original scheme (store the raw last-hidden deriv dh2, form DR inline
  // in ComputeDerivative_L1) which is bit-identical and avoids the fp64 regression.
  if constexpr (sizeof(NNScalar) == 4) {
    for (int n = 0; n < L1_nradmax; n++) {
      for (int l = 0; l <= lmax; l++) {
        const int j = n * (lmax + 1) + l;
        NNScalar sum = 0.0, dsum = 0.0;
        for (int k = 0; k < n_last_hidden; k++) {
          const NNScalar w = d_mlp_rad_R_W(last_layer, k, j);
          sum  += w * h_cur[k];
          dsum += w * dh_cur[k];
        }
        d_R_nl(ii, jj, n, l)  = sum * out_norm;
        d_DR_nl(ii, jj, n, l) = dsum * out_norm;
      }
    }
  } else {
    for (int j = 0; j < n_last_hidden; j++)
      d_dh2_R(ii, jj, j) = dh_cur[j];
    for (int n = 0; n < L1_nradmax; n++) {
      for (int l = 0; l <= lmax; l++) {
        const int j = n * (lmax + 1) + l;
        NNScalar sum = 0.0;
        for (int k = 0; k < n_last_hidden; k++)
          sum += d_mlp_rad_R_W(last_layer, k, j) * h_cur[k];
        d_R_nl(ii, jj, n, l) = sum * out_norm;
      }
    }
  }
}

// ======================================================================
// Kernel: ComputeMLPRadial_R1 (Layer 2)
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeMLPRadial_R1,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadial_R1>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;
  const int jj = slot.neigh;
  if (jj >= d_ncount(ii)) return;

  const int n_layers = mlp_rad_R1_n_layers;
  int mlp_dims[GRACE2L_MAX_MLP_LAYERS + 1];
  NNScalar mlp_norms[GRACE2L_MAX_MLP_LAYERS];
  for (int i = 0; i <= n_layers; i++) mlp_dims[i] = d_mlp_rad_R1_dims(i);
  for (int i = 0; i < n_layers; i++) mlp_norms[i] = d_mlp_rad_R1_norms(i);

  // Buffers sized to max(input, hidden) dims — NOT output dim
  NNScalar buf_a[GRACE2L_MAX_MLP_HIDDEN], buf_b[GRACE2L_MAX_MLP_HIDDEN];
  NNScalar dbuf_a[GRACE2L_MAX_MLP_HIDDEN], dbuf_b[GRACE2L_MAX_MLP_HIDDEN];
  NNScalar* h_cur = buf_a; NNScalar* h_nxt = buf_b;
  NNScalar* dh_cur = dbuf_a; NNScalar* dh_nxt = dbuf_b;

  const int n_input = mlp_dims[0];
  for (int k = 0; k < n_input; k++) {
    h_cur[k] = d_radial_basis(ii, jj, k);
    dh_cur[k] = d_dradial_basis(ii, jj, k);
  }

  for (int layer = 0; layer < n_layers - 1; layer++) {
    const int nin = mlp_dims[layer], nout = mlp_dims[layer + 1];
    const NNScalar norm = mlp_norms[layer];
    for (int j = 0; j < nout; j++) {
      NNScalar s = 0.0, ds = 0.0;
      for (int k = 0; k < nin; k++) {
        NNScalar w = d_mlp_rad_R1_W(layer, k, j);
        s += w * h_cur[k]; ds += w * dh_cur[k];
      }
      s *= norm; ds *= norm;
      NNScalar sig = 1.0 / (1.0 + Kokkos::exp(-s));
      h_nxt[j] = s * sig;
      dh_nxt[j] = sig * (1.0 + s * (1.0 - sig)) * ds;
    }
    NNScalar* tmp;
    tmp = h_cur; h_cur = h_nxt; h_nxt = tmp;
    tmp = dh_cur; dh_cur = dh_nxt; dh_nxt = tmp;
  }

  const int n_last_hidden = mlp_dims[n_layers - 1];
  const int last_layer = n_layers - 1;
  const NNScalar out_norm = mlp_norms[last_layer];
  // Precision specialization (see ComputeMLPRadial_R): float precomputes
  // DR1 alongside R1; fp64 stores raw dh2 and forms DR1 inline in ComputeDerivative_L2.
  if constexpr (sizeof(NNScalar) == 4) {
    for (int n = 0; n < L2_nradmax; n++) {
      for (int l = 0; l <= lmax; l++) {
        const int j = n * (lmax + 1) + l;
        NNScalar sum = 0.0, dsum = 0.0;
        for (int k = 0; k < n_last_hidden; k++) {
          const NNScalar w = d_mlp_rad_R1_W(last_layer, k, j);
          sum  += w * h_cur[k];
          dsum += w * dh_cur[k];
        }
        d_R1_nl(ii, jj, n, l)  = sum * out_norm;
        d_DR1_nl(ii, jj, n, l) = dsum * out_norm;
      }
    }
  } else {
    for (int j = 0; j < n_last_hidden; j++)
      d_dh2_R1(ii, jj, j) = dh_cur[j];
    for (int n = 0; n < L2_nradmax; n++) {
      for (int l = 0; l <= lmax; l++) {
        const int j = n * (lmax + 1) + l;
        NNScalar sum = 0.0;
        for (int k = 0; k < n_last_hidden; k++)
          sum += d_mlp_rad_R1_W(last_layer, k, j) * h_cur[k];
        d_R1_nl(ii, jj, n, l) = sum * out_norm;
      }
    }
  }
}

#ifdef KOKKOS_ENABLE_CUDA
// ======================================================================
// batched cuBLAS radial-MLP support kernels (CUDA + NNScalar=float).
// ======================================================================

// TagMLPAssemble: pack the tightly-packed row-major input matrices
// d_mlp_X[M x nradbase] (value) and d_mlp_dX[M x nradbase] (r-derivative) from
// the LayoutLeft d_radial_basis / d_dradial_basis, one bond per thread. Bond b
// decodes to (ii = b / bond_stride, jj = b % bond_stride), bond_stride ==
// d_radial_basis.extent(1) (allocated maxneigh). Padding rows (jj >= ncount) are
// zeroed so the batched SGEMM stays finite; their outputs are ignored downstream
// (consumers loop jj < ncount). The 2L radial MLP input is JUST the Chebyshev
// basis (no chem embedding, no bias) -> a straight per-bond copy.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPAssemble,
    const int b) const
{
  const int bond_stride = (int) d_radial_basis.extent(1);
  const int ii = b / bond_stride;
  const int jj = b - ii * bond_stride;
  const int n_in = nradbase;
  if (jj >= d_ncount(ii)) {
    for (int c = 0; c < n_in; c++) { d_mlp_X(b, c) = NNScalar(0.0); d_mlp_dX(b, c) = NNScalar(0.0); }
    return;
  }
  for (int c = 0; c < n_in; c++) {
    d_mlp_X(b, c)  = (NNScalar) d_radial_basis(ii, jj, c);
    d_mlp_dX(b, c) = (NNScalar) d_dradial_basis(ii, jj, c);
  }
}

// TagMLPActDeriv: fused value+deriv silu applied in place between GEMMs. Reads
// the raw value GEMM output (d_mlp_h{0,1}) and raw deriv GEMM output
// (d_mlp_dh{0,1}); forms s = raw*norm (2L MLP has NO bias) and ds = raw_deriv*norm,
// then writes silu(s) and silu'(s)*ds. silu'(s) = sig*(1 + s*(1-sig)) -- identical
// to the hand kernel TagComputeMLPRadial_R{,1}.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPActDeriv,
    const int64_t idx) const
{
  const int nout = mlp_act_n;
  const int b = (int) (idx / nout);
  const int j = (int) (idx - (int64_t) b * nout);
  const NNScalar norm = mlp_act_norm;
  if (mlp_act_layer == 0) {
    const NNScalar s   = d_mlp_h0(b, j) * norm;
    const NNScalar ds  = d_mlp_dh0(b, j) * norm;
    const NNScalar sig = NNScalar(1.0) / (NNScalar(1.0) + Kokkos::exp(-s));
    d_mlp_h0(b, j)  = s * sig;
    d_mlp_dh0(b, j) = sig * (NNScalar(1.0) + s * (NNScalar(1.0) - sig)) * ds;
  } else {
    const NNScalar s   = d_mlp_h1(b, j) * norm;
    const NNScalar ds  = d_mlp_dh1(b, j) * norm;
    const NNScalar sig = NNScalar(1.0) / (NNScalar(1.0) + Kokkos::exp(-s));
    d_mlp_h1(b, j)  = s * sig;
    d_mlp_dh1(b, j) = sig * (NNScalar(1.0) + s * (NNScalar(1.0) - sig)) * ds;
  }
}

// TagMLPScatterR: write the LayoutRight output-GEMM result d_mlp_R[b, j] (already
// scaled by the output norm via the SGEMM alpha) into the LayoutLeft 4D
// d_R_nl / d_R1_nl(ii, jj, n=j/(lmax+1), l=j%(lmax+1)). Padding bonds
// (jj >= ncount) are skipped so untouched entries match the hand kernel.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPScatterR,
    const int64_t idx) const
{
  const int lm1 = lmax + 1;
  const int n_rad = (mlp_which == 0) ? L1_nradmax : L2_nradmax;
  const int D3  = n_rad * lm1;
  const int bs  = (int) d_radial_basis.extent(1);
  const int b   = (int) (idx / D3);
  const int j   = (int) (idx - (int64_t) b * D3);
  const int ii  = b / bs;
  const int jj  = b - ii * bs;
  if (jj >= d_ncount(ii)) return;
  const int n = j / lm1, l = j - n * lm1;
  const NNScalar val = d_mlp_R(b, j);
  if (mlp_scatter_deriv == 0) {
    if (mlp_which == 0) d_R_nl(ii, jj, n, l)  = val;
    else                d_R1_nl(ii, jj, n, l) = val;
  } else {
    if (mlp_which == 0) d_DR_nl(ii, jj, n, l)  = val;
    else                d_DR1_nl(ii, jj, n, l) = val;
  }
}

// true-fp32 batched SGEMM. Row-major C[M x n_out] = alpha*In[M x n_in]*Wg[n_in x n_out].
// cuBLAS is column-major, so this equals the col-major product with OP_N/OP_N and
// leading dims = the row-major operands' physical row strides. CUBLAS_PEDANTIC_MATH
// (set on the handle) forces true fp32 (no tf32).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::mlp_sgemm(
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
    GRACE_KK_ERROR("GRACE-2L/KK: cublasSgemm failed (status {})", (int) st);
}

// Hard gate: verify the layer-0 SGEMM (transpose/leading-dim convention +
// true-fp32 path) reproduces a host fp64 reference on a tiny deterministic batch
// before the full chain is ever used; gate: blas::selftest_error <= 1.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::mlp_blas_selftest()
{
  const auto &layers = grace_model->L1_mlp_rad_layers;   // R
  if ((int) layers.size() < 1) return;
  const int n_in = layers[0].n_in, n_out = layers[0].n_out;
  const auto &Wg = d_mlp_Wg[0][0];
  const double err = blas::gemm_selftest<NNScalar, DeviceType>("g2l", n_in, n_out, layers[0].W.data(),
      [&](const NNScalar *X, NNScalar *C, int M) {
        mlp_sgemm(Wg.data(), X, C, n_out, M, n_in, (int) Wg.extent(1), n_in, n_out, NNScalar(1.0));
      });
  if (err > 1.0)
    GRACE_KK_ERROR("GRACE-2L/KK: {} SGEMM selftest failed (max error {:.3g} x the fp32 rounding bound)",
                   blas::name(), err);
}

// ======================================================================
// build tile-sorted metadata + repacked per-tile weights for one
// block-sparse FC. Host-side, from the model FCWeights. See FCBlasOp doc.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::build_fc_blas_op(
    FCBlasOp &op, const GRACE2LModel::FCWeights &fc)
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
  op.nof = t_nn_1d("g2l:fc_nof", std::max(N_l, 1));
  { auto h = Kokkos::create_mirror_view(op.nof);
    for (int i = 0; i < N_l; i++)
      h(i) = (NNScalar)(i < (int) fc.norm_out_factor.size() ? fc.norm_out_factor[i] : 0.0);
    Kokkos::deep_copy(op.nof, h); }

  // LEFT: stable-sort lm by tile wtl(lm) -> tile-contiguous columns
  op.L_src = t_int_1d("g2l:fc_Lsrc", std::max(N_l, 1));
  op.L_pos = t_int_1d("g2l:fc_Lpos", std::max(N_l, 1));
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
  op.R_src = t_int_1d("g2l:fc_Rsrc", std::max(N_r, 1));
  op.R_pos = t_int_1d("g2l:fc_Rpos", std::max(N_r, 1));
  op.R_tgt = t_int_1d("g2l:fc_Rtgt", std::max(N_r, 1));
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
    op.Wl_packed = t_nn_1d("g2l:fc_Wl", szl);
    auto h = Kokkos::create_mirror_view(op.Wl_packed);
    for (int k = 0; k < n_out; k++)
      for (int n = 0; n < n_in_l; n++)
        for (int t = 0; t < nt; t++)
          h((size_t) t * n_out * n_in_l + (size_t) n * n_out + k) =
              (NNScalar) fc.w_left[(size_t) k * n_in_l * nt + (size_t) n * nt + t];
    Kokkos::deep_copy(op.Wl_packed, h); }
  { const size_t szr = (size_t) nt * n_out * n_in_r;
    op.Wr_packed = t_nn_1d("g2l:fc_Wr", szr);
    auto h = Kokkos::create_mirror_view(op.Wr_packed);
    for (int k = 0; k < n_out; k++)
      for (int n = 0; n < n_in_r; n++)
        for (int t = 0; t < nt; t++)
          h((size_t) t * n_out * n_in_r + (size_t) n * n_out + k) =
              (NNScalar) fc.w_right[(size_t) k * n_in_r * nt + (size_t) n * nt + t];
    Kokkos::deep_copy(op.Wr_packed, h); }

  fc_blas_max_nin  = std::max(fc_blas_max_nin,  std::max(n_in_l, n_in_r));
  fc_blas_max_nout = std::max(fc_blas_max_nout, n_out);
  fc_blas_max_N    = std::max(fc_blas_max_N,    std::max(N_l, N_r));
  op.active = true;
}

// ======================================================================
// one forward FC via gather -> per-tile true-fp32 SGEMM -> scatter.
// LEFT half assigns out(ii,k,lm); RIGHT half accumulates onto out(ii,k,ct(idx)).
// The scatter is (ii,k)-parallel and loops functions in the same order as the
// hand kernel (no atomics) -> only the Σ_n contraction order changes.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::fc_forward_blas(
    FCBlasOp &op, const t_nn_3d &in_left, const t_nn_3d &in_right, const t_nn_3d &out)
{
  const int cs = chunk_size;
  const int n_out = op.n_out;
  const float one = 1.0f, zero = 0.0f;

  // ---------- LEFT: gather -> per-tile GEMM -> scatter (assign) ----------
  {
    const int nin = op.n_in_l, Nl = op.N_l;
    { auto G = d_fc_G; auto L_src = op.L_src; auto inL = in_left;
      Kokkos::parallel_for("FC_blas_gatherL",
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
        GRACE_KK_ERROR("GRACE-2L/KK: FC cublasSgemm(L) failed (status {})", (int) st);
    }
    { auto C = d_fc_C; auto L_pos = op.L_pos; auto nof = op.nof; auto outv = out;
      const NNScalar nl = op.nl; const int Nl2 = Nl, no = n_out, cs2 = cs;
      Kokkos::parallel_for("FC_blas_scatterL",
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
      Kokkos::parallel_for("FC_blas_gatherR",
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
        GRACE_KK_ERROR("GRACE-2L/KK: FC cublasSgemm(R) failed (status {})", (int) st);
    }
    { auto C = d_fc_C; auto R_pos = op.R_pos; auto R_tgt = op.R_tgt; auto nof = op.nof; auto outv = out;
      const NNScalar nr = op.nr; const int Nr2 = Nr, no = n_out, cs2 = cs;
      Kokkos::parallel_for("FC_blas_scatterR",
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
// build tile-sorted metadata + repacked per-tile weights for one
// element-independent ReduceN source (host-side, from the model ReduceWeights).
// Same repack as the FC (col-major [n_out x n_in] per tile). Also grows the
// shared FC/reduce scratch maxes (N can exceed the FC max, e.g. B_YI = 190).
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::build_reduce_blas_op(
    ReduceBlasOp &op, const GRACE2LModel::ReduceWeights &rw, int n_out)
{
  const int nt  = rw.w_shape;          // # tiles
  const int nin = rw.n_in;
  const int N   = rw.n_connections;
  if (n_out <= 0 || nt <= 0 || nin <= 0 || N <= 0) return;
  op.n_out = n_out; op.n_in = nin; op.N = N; op.ntiles = nt;
  op.norm = (NNScalar) rw.norm;

  // stable-sort connections by tile -> tile-contiguous gather columns.
  op.src = t_int_1d("g2l:red_src", N);
  op.pos = t_int_1d("g2l:red_pos", N);
  op.tgt = t_int_1d("g2l:red_tgt", N);
  op.pstart.assign(nt, 0); op.count.assign(nt, 0);
  { std::vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
        [&](int a, int b){ return rw.w_l_tile[a] < rw.w_l_tile[b]; });
    auto hsrc = Kokkos::create_mirror_view(op.src);
    auto hpos = Kokkos::create_mirror_view(op.pos);
    auto htgt = Kokkos::create_mirror_view(op.tgt);
    for (int p = 0; p < N; p++) {
      const int c = order[p];
      hsrc(p) = rw.collect_ind[c];     // fi at sorted col p
      hpos(c) = p;                     // orig connection c -> sorted col
      op.count[rw.w_l_tile[c]]++;
    }
    for (int c = 0; c < N; c++) htgt(c) = rw.total_sum_ind[c];  // fo, orig c order
    int acc = 0; for (int t = 0; t < nt; t++) { op.pstart[t] = acc; acc += op.count[t]; }
    Kokkos::deep_copy(op.src, hsrc); Kokkos::deep_copy(op.pos, hpos);
    Kokkos::deep_copy(op.tgt, htgt); }

  // repack weights col-major per tile: packed[t*n_out*n_in + n*n_out + k]
  // = rw.W[k*(n_in*nt) + n*nt + t]  (native flat [n_out][n_in][nt]).
  { const size_t sz = (size_t) nt * n_out * nin;
    op.W_packed = t_nn_1d("g2l:red_W", sz);
    auto h = Kokkos::create_mirror_view(op.W_packed);
    for (int k = 0; k < n_out; k++)
      for (int n = 0; n < nin; n++)
        for (int t = 0; t < nt; t++)
          h((size_t) t * n_out * nin + (size_t) n * n_out + k) =
              (NNScalar) rw.W[(size_t) k * nin * nt + (size_t) n * nt + t];
    Kokkos::deep_copy(op.W_packed, h); }

  fc_blas_max_nin  = std::max(fc_blas_max_nin,  nin);
  fc_blas_max_nout = std::max(fc_blas_max_nout, n_out);
  fc_blas_max_N    = std::max(fc_blas_max_N,    N);
  op.active = true;
}

// ======================================================================
// one element-independent ReduceN source X -> Bout. gather X cols by
// tile -> per-tile true-fp32 SGEMM (W_t[n_out x n_in] @ G[n_in x cs*N]) ->
// scatter-accumulate Bout(ii,k,fo) += norm * C. The scatter loops connections in
// ORIGINAL order (via pos) and is (ii,k)-parallel, matching the hand kernel's
// serial c-loop accumulation exactly (no atomics, only the Σ_n order changes).
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::reduce_forward_blas(
    ReduceBlasOp &op, const t_nn_3d &X, const t_nn_3d &Bout, bool zero_first, int nBf)
{
  const int cs = chunk_size;
  const int n_out = op.n_out, nin = op.n_in, N = op.N;
  const float one = 1.0f, zero = 0.0f;

  if (zero_first) {
    auto Bv = Bout; const int nf = nBf;
    Kokkos::parallel_for("ReduceB_blas_zero",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, n_out}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int f = 0; f < nf; f++) Bv(ii, k, f) = NNScalar(0.0);
      });
  }

  { auto G = d_fc_G; auto src = op.src; auto Xv = X; const int ni = nin;
    Kokkos::parallel_for("ReduceB_blas_gather",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {N*cs, ni}),
      KOKKOS_LAMBDA(const int col, const int n) {
        const int p = col / cs, ii = col - p*cs;
        G((size_t) col * ni + n) = Xv(ii, n, src(p));
      }); }

  for (int t = 0; t < op.ntiles; t++) {
    const int m = op.count[t]; if (!m) continue;
    const int colbase = op.pstart[t] * cs, Mt = m * cs;
    blas::status_t st = blas::sgemm(blas_handle, blas::OP_N, blas::OP_N,
        n_out, Mt, nin, &one,
        (const float*) op.W_packed.data() + (size_t) t * n_out * nin, n_out,
        (const float*) d_fc_G.data() + (size_t) colbase * nin, nin,
        &zero, (float*) d_fc_C.data() + (size_t) colbase * n_out, n_out);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-2L/KK: ReduceB cublasSgemm failed (status {})", (int) st);
  }

  { auto C = d_fc_C; auto pos = op.pos; auto tgt = op.tgt; auto Bv = Bout;
    const NNScalar nrm = op.norm; const int N2 = N, no = n_out, cs2 = cs;
    Kokkos::parallel_for("ReduceB_blas_scatter",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, n_out}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int c = 0; c < N2; c++) {
          const size_t col = (size_t)(pos(c) * cs2 + ii);
          Bv(ii, k, tgt(c)) += C(col * no + k) * nrm;
        }
      }); }
}

// ======================================================================
// cuBLAS DGEMM UQ projection drivers (2L). The B->R projection that dominated
// small-N UQ (each of the two team kernels re-walked its whole basis block per
// lane, a 128x-redundant indirect gather at low occupancy) becomes gather +
// one fp64 GEMM per block. R-row order is [I2(L2), rho(L1)]; the L1 (rho) block
// occupies rows [d_basis_L2, d_basis), the L2 (I2) block rows [0, d_basis_L2).
// Sub-blocked to <=8192 atoms to bound the Bmat buffer; fp64, CUBLAS_PEDANTIC.
// ======================================================================

// Ensure sub-block scratch is sized for uq_subN >= min(8192, chunk_size).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_uq_blas_L1()
{
  if (chunk_size <= 0) return;
  const int RP = uq_rp_dim;
  const int Db_L1 = uq_d_basis_L1;
  const int Db_L2 = uq_d_basis - uq_d_basis_L1;
  const int maxDb = Db_L1 > Db_L2 ? Db_L1 : Db_L2;

  // Build the rho (L1) gather map once (Phase 1: d_A.. are live here).
  if ((int) d_uq_mapL1_view.extent(0) == 0) {
    d_uq_mapL1_view = t_int_1d("g2l:uq_mapL1_view", Db_L1);
    d_uq_mapL1_n    = t_int_1d("g2l:uq_mapL1_n",    Db_L1);
    d_uq_mapL1_col  = t_int_1d("g2l:uq_mapL1_col",  Db_L1);
    auto mv = d_uq_mapL1_view; auto mn = d_uq_mapL1_n; auto mc = d_uq_mapL1_col;
    auto ciA = d_rho_ci_A; auto ciAA = d_rho_ci_AA;
    auto ciAAA = d_rho_ci_AAA; auto ciAAAA = d_rho_ci_AAAA;
    const int ninA = (int) d_A.extent(1),   ninAA = (int) d_AA.extent(1);
    const int ninAAA = (int) d_AAA.extent(1), ninAAAA = (int) d_AAAA.extent(1);
    const int Dbc = Db_L1;
    Kokkos::parallel_for("UQBuildMapL1", Kokkos::RangePolicy<DeviceType>(0, 1),
      KOKKOS_LAMBDA(const int) {
        int b = 0;
        #define GRACE2L_UQ_MAP(VID, NIN, CIV)                                   \
          { const int nc = (int)(CIV).extent(0);                               \
            for (int n = 0; n < (NIN); n++)                                    \
              for (int c = 0; c < nc; c++) {                                   \
                if (b < Dbc) { mv(b) = (VID); mn(b) = n; mc(b) = (CIV)(c); }   \
                b++;                                                           \
              } }
        GRACE2L_UQ_MAP(0, ninA,    ciA);
        GRACE2L_UQ_MAP(1, ninAA,   ciAA);
        GRACE2L_UQ_MAP(2, ninAAA,  ciAAA);
        GRACE2L_UQ_MAP(3, ninAAAA, ciAAAA);
        #undef GRACE2L_UQ_MAP
      });
  }

  const int want = std::min(8192, chunk_size);
  if (want > uq_subN) {
    uq_subN = want;
    grace::kk::MemKK::realloc_kokkos(d_uq_Bmat, "g2l:uq_Bmat", (size_t) maxDb * uq_subN);
    grace::kk::MemKK::realloc_kokkos(d_uq_proj, "g2l:uq_proj", (size_t) RP * uq_subN);
    if (uq_sgemm) {
      grace::kk::MemKK::realloc_kokkos(d_uq_Bmat_f, "g2l:uq_Bmat_f", (size_t) maxDb * uq_subN);
      grace::kk::MemKK::realloc_kokkos(d_uq_proj_f, "g2l:uq_proj_f", (size_t) RP * uq_subN);
    }
    grace::kk::MemKK::realloc_kokkos(d_uq_n2,   "g2l:uq_n2",   (size_t) uq_subN);
  }

  const double one = 1.0, zero = 0.0;
  uq_bmat_level = 1;
  for (int s0 = 0; s0 < chunk_size; s0 += uq_subN) {
    const int sN = std::min(uq_subN, chunk_size - s0);
    uq_bmat_s0 = s0;

    Kokkos::parallel_for("UQBmatL1",
        Kokkos::TeamPolicy<DeviceType, TagUQBmat>(sN, Kokkos::AUTO), *this);

    // zproj[RP x sN] = R_rho[RP x Db_L1] · Bmat_L1[Db_L1 x sN]; R_rho = rows
    // [Db_L2, d_basis) of the row-major [D_basis,RP] R -> offset Db_L2*RP, lda=RP.
    const float onef = 1.0f, zerof = 0.0f;
    blas::status_t st = uq_sgemm
        ? blas::sgemm(blas_handle, blas::OP_N, blas::OP_N, RP, sN, Db_L1, &onef,
                      d_uq_rp_matrix_f.data() + (size_t) Db_L2 * RP, RP, d_uq_Bmat_f.data(), Db_L1,
                      &zerof, d_uq_proj_f.data(), RP)
        : blas::dgemm(blas_handle, blas::OP_N, blas::OP_N, RP, sN, Db_L1, &one,
                      d_uq_rp_matrix.data() + (size_t) Db_L2 * RP, RP, d_uq_Bmat.data(), Db_L1,
                      &zero, d_uq_proj.data(), RP);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-2L/KK: UQ projection GEMM (L1) failed (status {})", (int) st);

    // Scatter the raw L1 projection into the per-global-atom carry d_uq_z(i,d).
    { auto proj = d_uq_proj; auto projf = d_uq_proj_f; auto z = d_uq_z; auto il = d_ilist;
      const bool sg = uq_sgemm;
      const int co = chunk_offset, s0c = s0, rp = RP;
      Kokkos::parallel_for("UQScatterZ",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {sN, RP}),
        KOKKOS_LAMBDA(const int loc, const int d) {
          const int gi = il(s0c + loc + co);
          z(gi, d) = sg ? (double) projf((size_t) rp * loc + d) : proj((size_t) rp * loc + d);
        }); }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_uq_blas_L2()
{
  if (chunk_size <= 0) return;
  const int RP = uq_rp_dim;
  const int Db_L2 = uq_d_basis - uq_d_basis_L1;

  // Build the I2 (L2) gather map once (Phase 3: d_B.. are live here).
  if ((int) d_uq_mapL2_view.extent(0) == 0) {
    d_uq_mapL2_view = t_int_1d("g2l:uq_mapL2_view", Db_L2);
    d_uq_mapL2_n    = t_int_1d("g2l:uq_mapL2_n",    Db_L2);
    d_uq_mapL2_col  = t_int_1d("g2l:uq_mapL2_col",  Db_L2);
    auto mv = d_uq_mapL2_view; auto mn = d_uq_mapL2_n; auto mc = d_uq_mapL2_col;
    auto ciB = d_I2_ci_B; auto ciBB = d_I2_ci_BB;
    auto ciBBB = d_I2_ci_BBB; auto ciBBBB = d_I2_ci_BBBB;
    const int ninB = (int) d_B.extent(1),   ninBB = (int) d_BB.extent(1);
    const int ninBBB = (int) d_BBB.extent(1), ninBBBB = (int) d_BBBB.extent(1);
    const int Dbc = Db_L2;
    Kokkos::parallel_for("UQBuildMapL2", Kokkos::RangePolicy<DeviceType>(0, 1),
      KOKKOS_LAMBDA(const int) {
        int b = 0;
        #define GRACE2L_UQ_MAP(VID, NIN, CIV)                                   \
          { const int nc = (int)(CIV).extent(0);                               \
            for (int n = 0; n < (NIN); n++)                                    \
              for (int c = 0; c < nc; c++) {                                   \
                if (b < Dbc) { mv(b) = (VID); mn(b) = n; mc(b) = (CIV)(c); }   \
                b++;                                                           \
              } }
        GRACE2L_UQ_MAP(0, ninB,    ciB);
        GRACE2L_UQ_MAP(1, ninBB,   ciBB);
        GRACE2L_UQ_MAP(2, ninBBB,  ciBBB);
        GRACE2L_UQ_MAP(3, ninBBBB, ciBBBB);
        #undef GRACE2L_UQ_MAP
      });
  }

  const int want = std::min(8192, chunk_size);
  if (want > uq_subN) {   // defensive: L1 sized this first in normal flow
    const int Db_L1 = uq_d_basis_L1;
    const int maxDb = Db_L1 > Db_L2 ? Db_L1 : Db_L2;
    uq_subN = want;
    grace::kk::MemKK::realloc_kokkos(d_uq_Bmat, "g2l:uq_Bmat", (size_t) maxDb * uq_subN);
    grace::kk::MemKK::realloc_kokkos(d_uq_proj, "g2l:uq_proj", (size_t) RP * uq_subN);
    if (uq_sgemm) {
      grace::kk::MemKK::realloc_kokkos(d_uq_Bmat_f, "g2l:uq_Bmat_f", (size_t) maxDb * uq_subN);
      grace::kk::MemKK::realloc_kokkos(d_uq_proj_f, "g2l:uq_proj_f", (size_t) RP * uq_subN);
    }
    grace::kk::MemKK::realloc_kokkos(d_uq_n2,   "g2l:uq_n2",   (size_t) uq_subN);
  }

  const double one = 1.0, zero = 0.0;
  const int scratch_bytes = (2 * uq_D + 8) * (int) sizeof(double);
  auto fprobe = Kokkos::TeamPolicy<DeviceType, TagUQFinish>(1, 1, 1)
      .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
  int fts = fprobe.team_size_max(*this, Kokkos::ParallelForTag());
  if (fts > 128) fts = 128;
  if (fts < 1)   fts = 1;

  uq_bmat_level = 2;
  for (int s0 = 0; s0 < chunk_size; s0 += uq_subN) {
    const int sN = std::min(uq_subN, chunk_size - s0);
    uq_bmat_s0 = s0;

    Kokkos::parallel_for("UQBmatL2",
        Kokkos::TeamPolicy<DeviceType, TagUQBmat>(sN, Kokkos::AUTO), *this);

    // proj[RP x sN] = R_I2[RP x Db_L2] · Bmat_L2[Db_L2 x sN]; R_I2 = rows
    // [0, Db_L2) of R -> offset 0, lda=RP.
    const float onef = 1.0f, zerof = 0.0f;
    blas::status_t st = uq_sgemm
        ? blas::sgemm(blas_handle, blas::OP_N, blas::OP_N, RP, sN, Db_L2, &onef,
                      d_uq_rp_matrix_f.data(), RP, d_uq_Bmat_f.data(), Db_L2, &zerof, d_uq_proj_f.data(), RP)
        : blas::dgemm(blas_handle, blas::OP_N, blas::OP_N, RP, sN, Db_L2, &one,
                      d_uq_rp_matrix.data(), RP, d_uq_Bmat.data(), Db_L2, &zero, d_uq_proj.data(), RP);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-2L/KK: UQ projection GEMM (L2) failed (status {})", (int) st);

    Kokkos::parallel_for("UQFinish",
        Kokkos::TeamPolicy<DeviceType, TagUQFinish>(sN, fts, 1)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)), *this);
  }
}
#endif  // KOKKOS_ENABLE_CUDA

// ======================================================================
// radial MLP forward (value R + output-layer radial derivative DR)
// for `which` (0 = R/L1, 1 = R1/L2). On CUDA with a live handle, NNScalar=float
// and the expected 3-layer architecture, runs the batched true-fp32 cuBLAS path
// (assemble -> {value,deriv} SGEMM/act x2 -> output value SGEMM -> scatter R ->
// output deriv SGEMM -> scatter DR). Otherwise falls back to the hand kernel
// TagComputeMLPRadial_R{,1}. Reproduces the hand kernel's d_R{,1}_nl AND
// d_DR{,1}_nl (up to fp32 accumulation order). No fences between steps: the
// handle is bound to the same Kokkos stream, so the whole chain serializes in
// issue order (race-free).
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_mlp_radial(int which)
{
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (is_cuda_v<DeviceType> && sizeof(NNScalar) == 4) {
    const int n_layers = (which == 0) ? mlp_rad_R_n_layers : mlp_rad_R1_n_layers;
    if (blas_handle && n_layers == 3) {
      mlp_which = which;
      const int bond_stride = (int) d_radial_basis.extent(1);
      const int M = chunk_size * bond_stride;
      mlp_M = M;
      const int nin0 = nradbase;                        // MLP input width (10)
      const int d1 = (int) d_mlp_Wg[which][0].extent(1); // hidden 0 (64)
      const int d2 = (int) d_mlp_Wg[which][1].extent(1); // hidden 1 (64)
      const int d3 = (int) d_mlp_Wg[which][2].extent(1); // output (210 / 160)
      const auto &layers = (which == 0) ? grace_model->L1_mlp_rad_layers
                                        : grace_model->L2_mlp_rad_layers;
      const NNScalar norm0 = (NNScalar) layers[0].norm;
      const NNScalar norm1 = (NNScalar) layers[1].norm;
      const NNScalar norm2 = (NNScalar) layers[2].norm;

      // 1. assemble X[M x nin0], dX[M x nin0] (zero padding rows)
      Kokkos::parallel_for("ComputeMLPRadial_assemble",
          Kokkos::RangePolicy<DeviceType, TagMLPAssemble>(0, M), *this);

      // 2. layer 0: h0 = X*W0 (raw), dh0 = dX*W0 (raw)
      mlp_sgemm(d_mlp_Wg[which][0].data(), d_mlp_X.data(),  d_mlp_h0.data(),
          d1, M, nin0, (int) d_mlp_Wg[which][0].extent(1), (int) d_mlp_X.extent(1),
          (int) d_mlp_h0.extent(1), NNScalar(1.0));
      mlp_sgemm(d_mlp_Wg[which][0].data(), d_mlp_dX.data(), d_mlp_dh0.data(),
          d1, M, nin0, (int) d_mlp_Wg[which][0].extent(1), (int) d_mlp_dX.extent(1),
          (int) d_mlp_dh0.extent(1), NNScalar(1.0));
      // 3. act 0: silu(h0*norm0), silu'(...)*ds
      mlp_act_layer = 0; mlp_act_n = d1; mlp_act_norm = norm0;
      Kokkos::parallel_for("ComputeMLPRadial_act",
          Kokkos::RangePolicy<DeviceType, TagMLPActDeriv, Kokkos::IndexType<int64_t>>(0, (int64_t) M * d1), *this);

      // 4. layer 1: h1 = h0*W1 (raw), dh1 = dh0*W1 (raw)
      mlp_sgemm(d_mlp_Wg[which][1].data(), d_mlp_h0.data(),  d_mlp_h1.data(),
          d2, M, d1, (int) d_mlp_Wg[which][1].extent(1), (int) d_mlp_h0.extent(1),
          (int) d_mlp_h1.extent(1), NNScalar(1.0));
      mlp_sgemm(d_mlp_Wg[which][1].data(), d_mlp_dh0.data(), d_mlp_dh1.data(),
          d2, M, d1, (int) d_mlp_Wg[which][1].extent(1), (int) d_mlp_dh0.extent(1),
          (int) d_mlp_dh1.extent(1), NNScalar(1.0));
      // 5. act 1: silu(h1*norm1), silu'(...)*ds -> d_mlp_h1, d_mlp_dh1 (last hidden)
      mlp_act_layer = 1; mlp_act_n = d2; mlp_act_norm = norm1;
      Kokkos::parallel_for("ComputeMLPRadial_act",
          Kokkos::RangePolicy<DeviceType, TagMLPActDeriv, Kokkos::IndexType<int64_t>>(0, (int64_t) M * d2), *this);

      // 6. output value: R = norm2 * (h1 * W2) into LayoutRight scratch d_mlp_R
      mlp_sgemm(d_mlp_Wg[which][2].data(), d_mlp_h1.data(), d_mlp_R.data(),
          d3, M, d2, (int) d_mlp_Wg[which][2].extent(1), (int) d_mlp_h1.extent(1),
          (int) d_mlp_R.extent(1), norm2);
      // 7. scatter R -> LayoutLeft d_R{,1}_nl
      mlp_scatter_deriv = 0;
      Kokkos::parallel_for("ComputeMLPRadial_scatterR",
          Kokkos::RangePolicy<DeviceType, TagMLPScatterR, Kokkos::IndexType<int64_t>>(0, (int64_t) M * d3), *this);

      // 8. output-layer radial derivative: DR = norm2 * (dh1 * W2) into scratch
      // d_mlp_R (reused; the value scatter above already consumed it in issue
      // order on the same stream). This replaces the inline W2^T*dh2 matvec that
      // ComputeDerivative_L1/L2 used to do per (n,l).
      mlp_sgemm(d_mlp_Wg[which][2].data(), d_mlp_dh1.data(), d_mlp_R.data(),
          d3, M, d2, (int) d_mlp_Wg[which][2].extent(1), (int) d_mlp_dh1.extent(1),
          (int) d_mlp_R.extent(1), norm2);
      // 9. scatter DR -> LayoutLeft d_DR{,1}_nl
      mlp_scatter_deriv = 1;
      Kokkos::parallel_for("ComputeMLPRadial_scatterDR",
          Kokkos::RangePolicy<DeviceType, TagMLPScatterR, Kokkos::IndexType<int64_t>>(0, (int64_t) M * d3), *this);
      return;
    }
  }
#endif
  // Fallback: original hand kernel (non-CUDA, fp64, or n_layers != 3).
  int team_size = 1, vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1) team_size = 32;
  int ts = team_size;
  if (which == 0) {
    ts = fit_team_size<DeviceType, TagComputeMLPRadial_R>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadial_R>(
        ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    Kokkos::parallel_for("ComputeMLPRadial_R", policy, *this);
  } else {
    ts = fit_team_size<DeviceType, TagComputeMLPRadial_R1>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadial_R1>(
        ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    Kokkos::parallel_for("ComputeMLPRadial_R1", policy, *this);
  }
}

// ======================================================================
// Kernel: ComputeAi (Layer 1) - A[i,n,lm] = Σ_j R_nl * Y_lm * Z_tr
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeAi,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeAi>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;
  const int jj = slot.neigh;
  if (jj >= d_ncount(ii)) return;

  const int mu_j = d_mu_j(ii, jj);
  const GeomScalar rx = d_rhats(ii, jj, 0), ry = d_rhats(ii, jj, 1), rz = d_rhats(ii, jj, 2);

  const GeomScalar Y00_v = Y00, sq3_v = sq3, sq3o2_v = sq3o2, sq2_v = sq2;

  int plm_size = (lmax + 1) * (lmax + 2) / 2;
  GeomScalar* plm = (GeomScalar*) team.thread_scratch(0).get_shmem(
      scratch_bytes<DeviceType, GeomScalar>(plm_size));

  plm[0] = Y00_v;
  if (lmax > 0) {
    plm[1] = Y00_v * sq3_v * rz;
    plm[2] = -sq3o2_v * Y00_v;
    for (int l = 2; l <= lmax; l++) {
      for (int m_val = 0; m_val < l - 1; m_val++) {
        const int idx = l*(l+1)/2 + m_val;
        const int idx_l1 = (l-1)*l/2 + m_val, idx_l2 = (l-2)*(l-1)/2 + m_val;
        const int ai = d_idx_sph(l*(l+1) + m_val);
        plm[idx] = alm(ai) * (rz * plm[idx_l1] + blm(ai) * plm[idx_l2]);
      }
      { const int idx = l*(l+1)/2+l-1;
        plm[idx] = dl(l) * plm[(l-1)*l/2+l-1] * rz; }
      { const int idx = l*(l+1)/2+l;
        plm[idx] = cl(l) * plm[(l-1)*l/2+l-1]; }
    }
  }

  NNScalar* z_tr = (NNScalar*) team.thread_scratch(0).get_shmem(L1_nradmax * sizeof(NNScalar));
  // A indicator transform — lookup precomputed z_tr, multiply by inv_avg
  for (int n = 0; n < L1_nradmax; n++)
    z_tr[n] = d_z_tr_A(mu_j, n) * L1_inv_avg_n_neigh;

  const GeomScalar phase_re = rx, phase_im = ry;

  // m = 0
  for (int l = 0; l <= lmax; l++) {
    const GeomScalar Y = plm[l*(l+1)/2];
    const int A_idx = l*(l+1);
    for (int n = 0; n < L1_nradmax; n++)
      Kokkos::atomic_add(&d_A(ii, n, A_idx), (NNScalar)(d_R_nl(ii, jj, n, l) * Y) * z_tr[n]);
  }

  // m >= 1
  GeomScalar pm_re = phase_re, pm_im = phase_im;
  for (int m_val = 1; m_val <= lmax; m_val++) {
    if (m_val >= 2) {
      GeomScalar tmp = pm_re * phase_re - pm_im * phase_im;
      pm_im = pm_re * phase_im + pm_im * phase_re;
      pm_re = tmp;
    }
    const int fac = (m_val % 2 == 0) ? 1 : -1;
    for (int l = m_val; l <= lmax; l++) {
      const GeomScalar pv = plm[l*(l+1)/2 + m_val];
      const GeomScalar rYp = sq2_v * fac * pm_re * pv;
      const GeomScalar rYn = sq2_v * fac * pm_im * pv;
      const int Ap = l*(l+1) + m_val, An = l*(l+1) - m_val;
      for (int n = 0; n < L1_nradmax; n++) {
        const NNScalar R = d_R_nl(ii, jj, n, l);
        Kokkos::atomic_add(&d_A(ii, n, Ap), R * (NNScalar)rYp * z_tr[n]);
        Kokkos::atomic_add(&d_A(ii, n, An), R * (NNScalar)rYn * z_tr[n]);
      }
    }
  }
}

// ======================================================================
// Kernel: ComputeAi_B0 (Layer 2) - B0[i,n,lm] = Σ_j R1_nl * Y_lm * Z_B0_tr
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeAi_B0,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeAi_B0>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;
  const int jj = slot.neigh;
  if (jj >= d_ncount(ii)) return;

  const int mu_j = d_mu_j(ii, jj);
  const GeomScalar rx = d_rhats(ii, jj, 0), ry = d_rhats(ii, jj, 1), rz = d_rhats(ii, jj, 2);

  const GeomScalar Y00_v = Y00, sq3_v = sq3, sq3o2_v = sq3o2, sq2_v = sq2;

  int plm_size = (lmax + 1) * (lmax + 2) / 2;
  GeomScalar* plm = (GeomScalar*) team.thread_scratch(0).get_shmem(
      scratch_bytes<DeviceType, GeomScalar>(plm_size));

  plm[0] = Y00_v;
  if (lmax > 0) {
    plm[1] = Y00_v * sq3_v * rz;
    plm[2] = -sq3o2_v * Y00_v;
    for (int l = 2; l <= lmax; l++) {
      for (int m_val = 0; m_val < l - 1; m_val++) {
        const int idx = l*(l+1)/2 + m_val;
        const int idx_l1 = (l-1)*l/2 + m_val, idx_l2 = (l-2)*(l-1)/2 + m_val;
        const int ai = d_idx_sph(l*(l+1) + m_val);
        plm[idx] = alm(ai) * (rz * plm[idx_l1] + blm(ai) * plm[idx_l2]);
      }
      { const int idx = l*(l+1)/2+l-1;
        plm[idx] = dl(l) * plm[(l-1)*l/2+l-1] * rz; }
      { const int idx = l*(l+1)/2+l;
        plm[idx] = cl(l) * plm[(l-1)*l/2+l-1]; }
    }
  }

  NNScalar* z_tr = (NNScalar*) team.thread_scratch(0).get_shmem(L2_nradmax * sizeof(NNScalar));
  // B0 indicator transform — lookup precomputed z_tr, multiply by inv_avg
  for (int n = 0; n < L2_nradmax; n++)
    z_tr[n] = d_z_tr_B0(mu_j, n) * L2_inv_avg_n_neigh;

  const GeomScalar phase_re = rx, phase_im = ry;

  for (int l = 0; l <= lmax; l++) {
    const GeomScalar Y = plm[l*(l+1)/2];
    const int idx = l*(l+1);
    for (int n = 0; n < L2_nradmax; n++)
      Kokkos::atomic_add(&d_B0(ii, n, idx), (NNScalar)(d_R1_nl(ii, jj, n, l) * Y) * z_tr[n]);
  }

  GeomScalar pm_re = phase_re, pm_im = phase_im;
  for (int m_val = 1; m_val <= lmax; m_val++) {
    if (m_val >= 2) {
      GeomScalar tmp = pm_re * phase_re - pm_im * phase_im;
      pm_im = pm_re * phase_im + pm_im * phase_re;
      pm_re = tmp;
    }
    const int fac = (m_val % 2 == 0) ? 1 : -1;
    for (int l = m_val; l <= lmax; l++) {
      const GeomScalar pv = plm[l*(l+1)/2 + m_val];
      const GeomScalar rYp = sq2_v * fac * pm_re * pv;
      const GeomScalar rYn = sq2_v * fac * pm_im * pv;
      const int Ap = l*(l+1) + m_val, An = l*(l+1) - m_val;
      for (int n = 0; n < L2_nradmax; n++) {
        const NNScalar R = d_R1_nl(ii, jj, n, l);
        Kokkos::atomic_add(&d_B0(ii, n, Ap), R * (NNScalar)rYp * z_tr[n]);
        Kokkos::atomic_add(&d_B0(ii, n, An), R * (NNScalar)rYn * z_tr[n]);
      }
    }
  }
}

// ======================================================================
// Kernel: ComputeYI - equivariant indicator basis
// YI[i, out_f, n] = inv_avg * Σ_j Σ_t cg[t] * R1[j,n,l(y_t)] * Y[y_t](r̂_ij) * I[j, I_t, n]
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeYI,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeYI>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;
  const int jj = slot.neigh;
  if (jj >= d_ncount(ii)) return;

  const int j_global = d_nearest(ii, jj);
  const GeomScalar rx = d_rhats(ii, jj, 0), ry = d_rhats(ii, jj, 1), rz = d_rhats(ii, jj, 2);

  const GeomScalar Y00_v = Y00, sq3_v = sq3, sq3o2_v = sq3o2, sq2_v = sq2;

  // Compute Y_lm for this bond direction
  int plm_size = (lmax + 1) * (lmax + 2) / 2;
  GeomScalar* plm = (GeomScalar*) team.thread_scratch(0).get_shmem(
      scratch_bytes<DeviceType, GeomScalar>(plm_size));

  plm[0] = Y00_v;
  if (lmax > 0) {
    plm[1] = Y00_v * sq3_v * rz;
    plm[2] = -sq3o2_v * Y00_v;
    for (int l = 2; l <= lmax; l++) {
      for (int m_val = 0; m_val < l - 1; m_val++) {
        const int idx = l*(l+1)/2 + m_val;
        const int idx_l1 = (l-1)*l/2 + m_val, idx_l2 = (l-2)*(l-1)/2 + m_val;
        const int ai = d_idx_sph(l*(l+1) + m_val);
        plm[idx] = alm(ai) * (rz * plm[idx_l1] + blm(ai) * plm[idx_l2]);
      }
      { const int idx = l*(l+1)/2+l-1;
        plm[idx] = dl(l) * plm[(l-1)*l/2+l-1] * rz; }
      { const int idx = l*(l+1)/2+l;
        plm[idx] = cl(l) * plm[(l-1)*l/2+l-1]; }
    }
  }

  // Precompute all real Y_lm values for this bond direction
  GeomScalar Y_vals[25];  // indexed by lm = l*(l+1)+m, max 25 for lmax=4
  {
    const GeomScalar phase_re = rx, phase_im = ry;
    for (int l = 0; l <= lmax; l++) {
      // m = 0
      Y_vals[l*(l+1)] = plm[l*(l+1)/2];
      // m >= 1: compute phase^m incrementally
      GeomScalar pm_re = phase_re, pm_im = phase_im;
      for (int m = 1; m <= l; m++) {
        const int fac = (m % 2 == 0) ? 1 : -1;
        const GeomScalar pv = plm[l*(l+1)/2 + m];
        Y_vals[l*(l+1)+m] = sq2_v * fac * pm_re * pv;   // cos-like (positive m)
        Y_vals[l*(l+1)-m] = sq2_v * fac * pm_im * pv;   // sin-like (negative m)
        const GeomScalar t_re = pm_re * phase_re - pm_im * phase_im;
        const GeomScalar t_im = pm_re * phase_im + pm_im * phase_re;
        pm_re = t_re; pm_im = t_im;
      }
    }
  }

  // Loop over CG terms with precomputed Y_lm lookup
  const GeomScalar inv_avg = yi_inv_avg_n_neigh;
  for (int t = 0; t < n_yi_cg; t++) {
    const int y_idx = d_yi_lr_inds(t, 0);
    const int I_idx = d_yi_lr_inds(t, 1);
    const int out_f = d_yi_m_sum_ind(t);
    const NNScalar c = d_yi_cg_coeff(t);
    const int l = d_l_from_lm(y_idx);
    const GeomScalar Y_val = Y_vals[y_idx];

    for (int n = 0; n < L2_nradmax; n++) {
      Kokkos::atomic_add(&d_YI(ii, n, out_f),
          (NNScalar)(c * d_R1_nl(ii, jj, n, l) * Y_val * inv_avg) * d_I_global(j_global, I_idx, n));
    }
  }
}

// ======================================================================
// Kernel: ComputeMLPEnergy
// E = MLP(I_nl_LN[:,1:] + I_0_LN[:,1:]) + I_nl_LN[:,0] + I_0_LN[:,0] + shift
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeMLPEnergyR, const int ii) const
{
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];
  const int mu_i = d_map(type(i));

  // Sum of linear terms from both layers
  // Use d_I_nl_LN_global(i, ...) not d_I_nl_LN(ii, ...) — the chunk-local
  // array is stale when L2 runs in different chunks than L1.
  const NNScalar linear_term = d_I_nl_LN_global(i, 0) + d_I_0_LN(ii, 0);

  NNScalar ebuf_a[GRACE2L_MAX_MLP_DIM], ebuf_b[GRACE2L_MAX_MLP_DIM];
  NNScalar* h_cur = ebuf_a;
  NNScalar* h_nxt = ebuf_b;

  // Input: I_nl_LN[:,1:] + I_0_LN[:,1:] (sum of nonlinear parts)
  const int nin0 = d_energy_dims(0);
  for (int k = 0; k < nin0; k++)
    h_cur[k] = d_I_nl_LN_global(i, k + 1) + d_I_0_LN(ii, k + 1);

  // Hidden layers with activation
  for (int layer = 0; layer < energy_n_layers - 1; layer++) {
    const int nin = d_energy_dims(layer);
    const int nout = d_energy_dims(layer + 1);
    const NNScalar norm = d_energy_norms(layer);
    for (int j = 0; j < nout; j++) {
      NNScalar sum = 0.0;
      for (int k = 0; k < nin; k++)
        sum += d_energy_W(layer, k, j) * h_cur[k];
      sum *= norm;
      // Apply activation: tanh for 2L, silu for 1L
      if (energy_activation == 1)
        h_nxt[j] = tanh_act(sum);
      else
        h_nxt[j] = silu(sum);
    }
    NNScalar* tmp = h_cur; h_cur = h_nxt; h_nxt = tmp;
  }

  // Output layer (no activation)
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

  // E_atom = (MLP_out + linear_term) * output_scale + shift, in fp64 (KKA) as on the CPU: an fp32 sum rounds E_atom
  // to ~1e-6 eV
  d_e_atom(ii) = ((KKA) e + (KKA) linear_term) * (KKA) output_scale + (KKA) d_shifts(mu_i);
}

// ======================================================================
// Kernel: ComputeUQ_L1basis (Phase 1) - project the L1 (rho) scalar B-basis into
// the RAW random-projection feature z (pre-normalization), and accumulate the L1
// block norm ||B^(L1)||^2. b_inv is the gathered products d_A..d_AAAA in the exact
// (source, n_out, collect) order the rho reduce iterates them — Python's reshape
// order — so the running R-row index b matches uq_rp_matrix's layout. z[d] = Σ_b
// b_inv[b]*R[b,d] over the uq_d_basis_L1 rows after the I2 (L2) ones; stored per global atom in
// d_uq_z (the Phase-3 ComputeUQ kernel adds the L2 part, then normalizes once both
// blocks are present — proj = (B·R)/||B|| by linearity). d_uq_n2L1 carries the L1
// block norm² across to Phase 3 for the density channels. Must run here: d_A.. are
// reused per chunk and are stale by Phase 3. Forward only, any_uq-gated.
// ======================================================================

// Stream one scalar-reduce source into the RAW projection feature: for each gathered
// product b_inv[b] = XVIEW(ii, n, CIVIEW(c)) (n outer, c inner — Python's reshape
// order), accumulate ACC[d] += b_inv[b]*R[b,d] over the RP projection columns, add
// b_inv[b]^2 into the block norm² N2, and advance the R-row cursor b. Shared by both
// UQ kernels below (L1 sources in Phase 1, L2 sources in Phase 3); expands against
// each kernel's local b/RP/ii, N2 accumulator and the d_uq_rp_matrix member.
// TODO(dedup): this macro + the schema-v6 load() validation are duplicated across
// the 1L / 2L engines (Kokkos and CPU). The R-row traversal order here is coupled
// to the exporter's reshape order and the load-time d_basis check guards width, not
// ordering — a shared helper would remove the drift risk. Follow-up refactor.
// Single projection column: for each gathered product b_inv[b]=XVIEW(ii,n,CIVIEW(c))
// (n outer, c inner — Python reshape order), accumulate ACC += b_inv[b]*R[b,d] for
// this lane's column d, add b_inv[b]^2 into the block norm² N2, advance R-row cursor b.
#define GRACE2L_UQ_WALK(XVIEW, CIVIEW, ACC, N2)                                    \
    { const int nin = (int)(XVIEW).extent(1); const int nc = (int)(CIVIEW).extent(0); \
      for (int n = 0; n < nin; n++)                                               \
        for (int c = 0; c < nc; c++) {                                            \
          const double bval = (double)(XVIEW)(ii, n, (CIVIEW)(c));                \
          (N2)  += bval * bval;                                                   \
          (ACC) += bval * d_uq_rp_matrix(b, d);                                   \
          b++;                                                                    \
        } }
// fp32 variant (uq_sgemm: the fp32 / mixed engines, the precision of their cuBLAS SGEMM path): fp32 products and
// sums against the fp32 copy of R (row-major [d_basis, RP]); fp64 is 1/64 of fp32 on consumer GPUs and the fp64 R
// does not fit their L2
#define GRACE2L_UQ_WALK_F(XVIEW, CIVIEW, ACC, N2)                                  \
    { const int nin = (int)(XVIEW).extent(1); const int nc = (int)(CIVIEW).extent(0); \
      for (int n = 0; n < nin; n++)                                               \
        for (int c = 0; c < nc; c++) {                                            \
          const float bval = (float)(XVIEW)(ii, n, (CIVIEW)(c));                  \
          (N2)  += bval * bval;                                                   \
          (ACC) += bval * d_uq_rp_matrix_f((size_t) b * RP + d);                  \
          b++;                                                                    \
        } }

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeUQ_L1basis,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeUQ_L1basis>::member_type& team) const
{
  const int ii = team.league_rank();
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];
  const int RP = uq_rp_dim;
  const int TS = team.team_size();
  const int lane = team.team_rank();

  // Each lane owns projection column(s) d = lane, lane+TS, ...; it walks the L1 (rho)
  // scalar-basis stream once per column, accumulating that column in a register (no
  // per-thread array => no local-memory spill). ||B^(L1)||² is recomputed per column
  // (identical value, cheap) and written once by lane 0.
  // R-row order is [I2-block, rho-block] (sorted by reduce name); the rho (L1) block
  // occupies rows [d_basis_L2, d_basis), so the cursor starts past I2.
  double n2_L1 = 0.0;
  for (int d = lane; d < RP; d += TS) {
    int b = uq_d_basis - uq_d_basis_L1;
    if (uq_sgemm) {
      float z = 0.0f, n2 = 0.0f;
      GRACE2L_UQ_WALK_F(d_A,    d_rho_ci_A,    z, n2);
      GRACE2L_UQ_WALK_F(d_AA,   d_rho_ci_AA,   z, n2);
      GRACE2L_UQ_WALK_F(d_AAA,  d_rho_ci_AAA,  z, n2);
      GRACE2L_UQ_WALK_F(d_AAAA, d_rho_ci_AAAA, z, n2);
      d_uq_z(i, d) = z;
      n2_L1 = n2;
      continue;
    }
    double z = 0.0;
    n2_L1 = 0.0;
    GRACE2L_UQ_WALK(d_A,    d_rho_ci_A,    z, n2_L1);
    GRACE2L_UQ_WALK(d_AA,   d_rho_ci_AA,   z, n2_L1);
    GRACE2L_UQ_WALK(d_AAA,  d_rho_ci_AAA,  z, n2_L1);
    GRACE2L_UQ_WALK(d_AAAA, d_rho_ci_AAAA, z, n2_L1);
    d_uq_z(i, d) = z;
  }
  if (lane == 0) d_uq_n2L1(i) = n2_L1;   // lane 0 always processes column 0
}

// ======================================================================
// Kernel: ComputeUQ (Phase 3) - finish the basis-RP feature: add the L2/I2 scalar
// basis to the Phase-1 RAW L1 projection in d_uq_z (R rows [0, uq_d_basis - uq_d_basis_L1);
// the L1 rows follow them), L2-normalize the combined projection by ||B|| = sqrt(n2_L1+n2_L2),
// append the n_density log-norm density channels [full, L2, L1], then run the GMM:
// nearest centroid, Mahalanobis sigma, gamma = sigma/threshold. d_B..d_BBBB hold the
// current chunk's L2 products here. Block order for density is [full, block0=I2(L2),
// block1=rho(L1)] — the sorted-by-reduce-name order that R's row layout follows.
// Forward only; gamma is the sole UQ signal. The energy hot path is untouched.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeUQ,
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
  // of scalars [n2_L2, nrm_full] carried from the projection loop to the density code.
  double* f     = (double*) team.team_shmem().get_shmem(D * sizeof(double));
  double* delta = (double*) team.team_shmem().get_shmem(D * sizeof(double));
  double* sc    = (double*) team.team_shmem().get_shmem(2 * sizeof(double));

  const double n2_L1 = d_uq_n2L1(i);   // rho (L1) block norm² from Phase 1

  // Combined projection proj = B·R: L1 part from Phase 1 (d_uq_z) + L2 (I2) added here.
  // Each lane owns column(s) d, accumulating in a register. R-row order is
  // [I2-block, rho-block]: the I2 (L2) block occupies rows [0, d_basis_L2).
  for (int d = lane; d < RP; d += TS) {
    double acc = d_uq_z(i, d);
    double n2_L2 = 0.0;
    int b = 0;
    if (uq_sgemm) {
      float a = 0.0f, n2 = 0.0f;
      GRACE2L_UQ_WALK_F(d_B,    d_I2_ci_B,    a, n2);
      GRACE2L_UQ_WALK_F(d_BB,   d_I2_ci_BB,   a, n2);
      GRACE2L_UQ_WALK_F(d_BBB,  d_I2_ci_BBB,  a, n2);
      GRACE2L_UQ_WALK_F(d_BBBB, d_I2_ci_BBBB, a, n2);
      acc += a;
      n2_L2 = n2;
    } else {
      GRACE2L_UQ_WALK(d_B,    d_I2_ci_B,    acc, n2_L2);
      GRACE2L_UQ_WALK(d_BB,   d_I2_ci_BB,   acc, n2_L2);
      GRACE2L_UQ_WALK(d_BBB,  d_I2_ci_BBB,  acc, n2_L2);
      GRACE2L_UQ_WALK(d_BBBB, d_I2_ci_BBBB, acc, n2_L2);
    }
    const double nrm_full = Kokkos::sqrt(n2_L1 + n2_L2);   // ||B||
    if (uq_normalize) acc /= (nrm_full + 1.0e-12);
    f[d] = acc;
    if (d == 0) { sc[0] = n2_L2; sc[1] = nrm_full; }
  }
  #undef GRACE2L_UQ_WALK
  #undef GRACE2L_UQ_WALK_F
  team.team_barrier();

  // Density channels [full, block0=I2(L2), block1=rho(L1)] — sorted-by-reduce-name
  // order (I2 sorts before rho), matching R's row layout. Cheap; lane 0 writes them.
  if (lane == 0) {
    if (uq_n_density > 0) f[RP + 0] = Kokkos::log(sc[1]               + 1.0e-12) * uq_density_scale;
    if (uq_n_density > 1) f[RP + 1] = Kokkos::log(Kokkos::sqrt(sc[0]) + 1.0e-12) * uq_density_scale;
    if (uq_n_density > 2) f[RP + 2] = Kokkos::log(Kokkos::sqrt(n2_L1) + 1.0e-12) * uq_density_scale;
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

#ifdef KOKKOS_ENABLE_CUDA
// ======================================================================
// Kernel: UQBmat - gather one block's dense invariant basis for a sub-block of
// atoms so its B->R projection can be a single cuBLAS DGEMM. uq_bmat_level picks
// the block: 1 = rho/L1 (views d_A..d_AAAA, rho gather map) written for Phase 1,
// 2 = I2/L2 (views d_B..d_BBBB, I2 gather map) for Phase 3. league = sub-block
// atom; team lanes split the Db basis rows, gather col-major into d_uq_Bmat and
// team-reduce the block norm ||B_blk||^2. Replaces the per-lane basis re-walk.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagUQBmat,
    const typename Kokkos::TeamPolicy<DeviceType, TagUQBmat>::member_type& team) const
{
  const int ii_local = team.league_rank();
  const int ii = uq_bmat_s0 + ii_local;
  if (ii >= chunk_size) return;
  const int i = d_ilist[ii + chunk_offset];
  const bool L2 = (uq_bmat_level == 2);
  const int Db = L2 ? (uq_d_basis - uq_d_basis_L1) : uq_d_basis_L1;
  const size_t base = (size_t) Db * ii_local;

  double n2 = 0.0;
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, Db),
    [&](const int b, double& s) {
      int vid, n, col;
      if (L2) { vid = d_uq_mapL2_view(b); n = d_uq_mapL2_n(b); col = d_uq_mapL2_col(b); }
      else    { vid = d_uq_mapL1_view(b); n = d_uq_mapL1_n(b); col = d_uq_mapL1_col(b); }
      double v;
      if (L2) {
        if      (vid == 0) v = (double) d_B(ii, n, col);
        else if (vid == 1) v = (double) d_BB(ii, n, col);
        else if (vid == 2) v = (double) d_BBB(ii, n, col);
        else               v = (double) d_BBBB(ii, n, col);
      } else {
        if      (vid == 0) v = (double) d_A(ii, n, col);
        else if (vid == 1) v = (double) d_AA(ii, n, col);
        else if (vid == 2) v = (double) d_AAA(ii, n, col);
        else               v = (double) d_AAAA(ii, n, col);
      }
      if (uq_sgemm) d_uq_Bmat_f(base + b) = (float) v;
      else d_uq_Bmat(base + b) = v;
      s += v * v;
    }, n2);
  Kokkos::single(Kokkos::PerTeam(team), [&]() {
    if (L2) d_uq_n2(ii_local) = n2;   // L2 norm carried to UQFinish (sub-block local)
    else    d_uq_n2L1(i) = n2;         // L1 norm carried per global atom to Phase 3
  });
}

// ======================================================================
// Kernel: UQFinish (Phase 3) - combine the Phase-1 raw L1 projection (d_uq_z)
// with the DGEMM L2 projection (d_uq_proj), L2-normalize by the combined ||B|| =
// sqrt(n2_L1+n2_L2), append density channels [full, ||B_L2||, ||B_L1||], and run
// the GMM. Identical math to ComputeUQ; only the projection source changes.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagUQFinish,
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

  const double n2_L1 = d_uq_n2L1(i);
  const double n2_L2 = d_uq_n2(ii_local);
  const double nrm_full = Kokkos::sqrt(n2_L1 + n2_L2);
  const double inv = uq_normalize ? (1.0 / (nrm_full + 1.0e-12)) : 1.0;
  const size_t pbase = (size_t) RP * ii_local;
  for (int d = lane; d < RP; d += TS)
    f[d] = (d_uq_z(i, d) + (uq_sgemm ? (double) d_uq_proj_f(pbase + d) : d_uq_proj(pbase + d))) * inv;

  if (lane == 0) {
    if (uq_n_density > 0) f[RP + 0] = Kokkos::log(nrm_full            + 1.0e-12) * uq_density_scale;
    if (uq_n_density > 1) f[RP + 1] = Kokkos::log(Kokkos::sqrt(n2_L2) + 1.0e-12) * uq_density_scale;
    if (uq_n_density > 2) f[RP + 2] = Kokkos::log(Kokkos::sqrt(n2_L1) + 1.0e-12) * uq_density_scale;
  }
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
// Kernel: ComputeDerivative_L1 — per-bond forces from d_A_adj
// Same structure as 1L ComputeDerivative but using L1 weights
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeDerivative_L1,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeDerivative_L1>::member_type& team) const
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

  const GeomScalar Y00_v = Y00, sq3_v = sq3, sq3o2_v = sq3o2, sq2_v = sq2;

  int plm_size = (lmax + 1) * (lmax + 2) / 2;
  GeomScalar* scratch_geom = (GeomScalar*) team.thread_scratch(0).get_shmem(
      2 * plm_size * sizeof(GeomScalar));
  NNScalar* z_tr = (NNScalar*) team.thread_scratch(0).get_shmem(
      L1_nradmax * sizeof(NNScalar));
  GeomScalar* plm = scratch_geom;
  GeomScalar* dplm = scratch_geom + plm_size;

  // A indicator transform — lookup precomputed z_tr
  for (int n = 0; n < L1_nradmax; n++)
    z_tr[n] = d_z_tr_A(mu_j, n);

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
        plm[idx] = cl(l) * plm[prev];  dplm[idx] = GeomScalar(0.0); }
    }
  }

  // Restructured loop: (l,m) outermost, n innermost
  // Angular derivatives depend only on (l, m, rhat) — computed once per (l,m)
  // DR_n[n] and R_n[n] precomputed per l value
  GeomScalar f_ji[3] = {GeomScalar(0.0), GeomScalar(0.0), GeomScalar(0.0)};
  const GeomScalar phase_re = rx, phase_im = ry;

  NNScalar DR_n[GRACE2L_MAX_NRADMAX];
  GeomScalar R_n[GRACE2L_MAX_NRADMAX];

  // Precision specialization: float reads the precomputed output-layer
  // derivative d_DR_nl (formed in compute_mlp_radial via cuBLAS SGEMM or the fused
  // hand kernel). fp64 keeps the original scheme — cache the raw last-hidden deriv
  // d_dh2_R and form DR inline via the W2^T*dh2 matvec — which is bit-identical to
  // the pre-cuBLAS baseline and avoids the fp64 global-traffic regression. The
  // fp64-only locals below are dead-code-eliminated in the float instantiation.
  NNScalar dh2_cache[GRACE2L_MAX_MLP_HIDDEN];
  int last_layer_R = 0, n_last_hidden = 0;
  NNScalar norm_last_R = NNScalar(0.0);
  if constexpr (sizeof(NNScalar) != 4) {
    last_layer_R = mlp_rad_R_n_layers - 1;
    n_last_hidden = d_mlp_rad_R_dims(last_layer_R);
    norm_last_R = d_mlp_rad_R_norms(last_layer_R);
    for (int k = 0; k < n_last_hidden; k++)
      dh2_cache[k] = d_dh2_R(ii, jj, k);
  }

  for (int l = 0; l <= lmax; l++) {
    // Load R_n[n] and DR_n[n] (output-layer radial value + deriv) for this l.
    for (int n = 0; n < L1_nradmax; n++) {
      R_n[n] = d_R_nl(ii, jj, n, l);
      if constexpr (sizeof(NNScalar) == 4) {
        DR_n[n] = d_DR_nl(ii, jj, n, l);
      } else {
        const int j_idx = n * (lmax + 1) + l;
        NNScalar dr = 0.0;
        for (int k = 0; k < n_last_hidden; k++)
          dr += d_mlp_rad_R_W(last_layer_R, k, j_idx) * dh2_cache[k];
        DR_n[n] = dr * norm_last_R;
      }
    }

    // m = 0: angular derivatives computed once
    {
      const GeomScalar Y = plm[l*(l+1)/2];
      const GeomScalar dp = dplm[l*(l+1)/2];
      const GeomScalar rdy = dp * rz;
      const GeomScalar DY_x = -rdy * rx, DY_y = -rdy * ry, DY_z = dp - rdy * rz;
      const int A_idx = l * (l + 1);

      for (int n = 0; n < L1_nradmax; n++) {
        const NNScalar w = d_A_adj(ii, n, A_idx) * z_tr[n] * L1_inv_avg_n_neigh;
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
      const GeomScalar Dpx_re = dyx_re - rdy_re*rx, Dpx_im = dyx_im - rdy_im*rx;
      const GeomScalar Dpy_re = dyy_re - rdy_re*ry, Dpy_im = dyy_im - rdy_im*ry;
      const GeomScalar Dpz_re = dyz_re - rdy_re*rz, Dpz_im = dyz_im - rdy_im*rz;
      const GeomScalar DYpx = sq2_v*fac*Dpx_re, DYpy = sq2_v*fac*Dpy_re, DYpz = sq2_v*fac*Dpz_re;
      const GeomScalar DYnx = sq2_v*fac*Dpx_im, DYny = sq2_v*fac*Dpy_im, DYnz = sq2_v*fac*Dpz_im;

      const int Ap = l*(l+1)+m, An = l*(l+1)-m;

      for (int n = 0; n < L1_nradmax; n++) {
        const NNScalar zn = z_tr[n] * L1_inv_avg_n_neigh;
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

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeForce<NEIGHFLAG,EVFLAG>, const int& /*ii*/) const
{
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
template<int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeForce<NEIGHFLAG,EVFLAG>, const int& /*ii*/, EV& /*ev*/) const
{
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
double Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::memory_usage()
{
  double bytes = 0;
  bytes += grace::kk::MemKK::memory_usage(d_A);
  bytes += grace::kk::MemKK::memory_usage(d_B0);
  bytes += grace::kk::MemKK::memory_usage(d_YI_pair);
  bytes += grace::kk::MemKK::memory_usage(d_R_nl);
  bytes += grace::kk::MemKK::memory_usage(d_R1_nl);
  bytes += grace::kk::MemKK::memory_usage(d_radial_basis);
  bytes += grace::kk::MemKK::memory_usage(d_I_global);
  bytes += grace::kk::MemKK::memory_usage(d_e_atom);
  if (has_uq) {
    bytes += grace::kk::MemKK::memory_usage(d_uq_centroids);
    bytes += grace::kk::MemKK::memory_usage(d_uq_inv_cov);
    bytes += grace::kk::MemKK::memory_usage(d_uq_interp_thresholds);
    bytes += grace::kk::MemKK::memory_usage(d_uq_rp_matrix);
    bytes += grace::kk::MemKK::memory_usage(d_uq_z);
    bytes += grace::kk::MemKK::memory_usage(d_uq_n2L1);
  }
  return bytes;
}

// ======================================================================
// Template instantiations
// ======================================================================

// ======================================================================
// Bond stages (virtual hooks; Grace2LFusedKK overrides them)
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_L1_bond()
{
  int team_size = 1;
  int vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
    team_size = 32;
  (void) team_size; (void) vector_length;
  // ---- Stage 1: ComputeNeigh ----
  {
    team_size = fit_team_size<DeviceType, TagComputeNeigh>(*this, chunk_size, team_size, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>(chunk_size, team_size, vector_length);
    Kokkos::parallel_for("ComputeNeigh", policy, *this);
  }

  // ---- Stage 1b: ComputeYBond — precompute real spherical harmonics per bond ----
  // Hoists plm + Y_vals out of ComputeAi/ComputeAi_B0/ComputeYI/ReverseYI_grad_I.
  // Each of those previously recomputed identical (plm,Y_vals) for the same
  // (ii,jj) bond up to L1_nradmax / L2_nradmax times. By materializing once
  // here we trade ~75 ops × maxneigh × cs (one-time) for ~75 × maxneigh × cs
  // × (L1_nradmax + L2_nradmax + L2_nradmax + L2_nradmax) saved ops downstream.
  compute_Y_bond_chunk();

  // ---- Stage 2: ComputeRadialBasis ----
  {
    int ts = team_size;
    ts = fit_team_size<DeviceType, TagComputeRadialBasis>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>(
        ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    Kokkos::parallel_for("ComputeRadialBasis", policy, *this);
  }

  // ---- Stage 3: ComputeMLPRadial_R (Layer 1) ----
  // cuBLAS SGEMM chain (CUDA+fp32/Mixed) or hand-kernel fallback.
  compute_mlp_radial(0);

  // ---- Stage 4: ComputeAi (Layer 1) — per-(ii,n) parallelization ----
  // Reads precomputed Y_vals from d_Y_bond instead of recomputing per (n).
  {
    auto A_out = d_A;
    auto R_nl = d_R_nl;
    auto Y_bond = d_Y_bond;
    auto nc_v = d_ncount;
    auto mu_j_v = d_mu_j;
    auto z_tr_A_v = d_z_tr_A;
    auto l_from_v = d_l_from_lm;
    NNScalar L1_inv_v = L1_inv_avg_n_neigh;
    int lmax_v = lmax;
    int L1_nr = L1_nradmax;
    int cs = chunk_size;

    Kokkos::parallel_for("ComputeAi",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, L1_nr}),
      KOKKOS_LAMBDA(const int ii, const int n) {
        // Zero A inline — eliminates deep_copy(d_A, 0.0)
        const int nlm = (lmax_v + 1) * (lmax_v + 1);
        for (int lm = 0; lm < nlm; lm++) A_out(ii, n, lm) = NNScalar(0.0);

        const int nct = nc_v(ii);
        for (int jj = 0; jj < nct; jj++) {
          const int mu_j = mu_j_v(ii, jj);

          // Lookup precomputed z_tr (includes A_lin_norm), multiply by inv_avg
          NNScalar z_n = z_tr_A_v(mu_j, n) * L1_inv_v;

          // Accumulate A — no atomics, this (ii,n) thread owns d_A(ii,n,*)
          for (int lm = 0; lm < nlm; lm++) {
            const int l = l_from_v(lm);
            A_out(ii, n, lm) += R_nl(ii, jj, n, l) * (NNScalar)Y_bond(ii, jj, lm) * z_n;
          }
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_L1_bond_bwd()
{
  int team_size = 1;
  int vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
    team_size = 32;
  (void) team_size; (void) vector_length;
  // ---- B1-11: ComputeDerivative_L1 → d_f_ij ----
  {
    int ts = team_size;
    ts = fit_team_size<DeviceType, TagComputeDerivative_L1>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    int plm_size = (lmax + 1) * (lmax + 2) / 2;
    int scratch_size = scratch_bytes<DeviceType, GeomScalar>(2 * plm_size) +
                        scratch_bytes<DeviceType, NNScalar>(L1_nradmax);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeDerivative_L1>(
        ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    policy = policy.set_scratch_size(0, Kokkos::PerThread(scratch_size));
    Kokkos::parallel_for("ComputeDerivative_L1", policy, *this);
  }

  // ---- B1-12: ComputeForce_L1 → atom forces + virial ----
  {
    auto fij = d_f_ij; auto nc = d_ncount; auto near = d_nearest;
    auto fout = f; auto il = d_ilist;
    auto rh = d_rhats; auto rn = d_rnorms;
    int co = chunk_offset, cs = chunk_size;
    bool do_v = (vflag_global != 0);
    bool do_va = (vflag_atom != 0);
    bool do_cva = (cvflag_atom != 0);
    auto va = d_vatom; auto cva = d_cvatom;

    EV ev_force;
    Kokkos::parallel_reduce("ComputeForce_L1", Kokkos::RangePolicy<DeviceType>(0, cs),
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
            const GeomScalar v0 = delx*fx, v1 = dely*fy, v2 = delz*fz;
            const GeomScalar v3 = delx*fy, v4 = delx*fz, v5 = dely*fz;
            if (do_v) { ev.v[0]+=v0; ev.v[1]+=v1; ev.v[2]+=v2; ev.v[3]+=v3; ev.v[4]+=v4; ev.v[5]+=v5; }
            if (do_va) {
              Kokkos::atomic_add(&va(i,0),(KKF)(GeomScalar(0.5)*v0)); Kokkos::atomic_add(&va(i,1),(KKF)(GeomScalar(0.5)*v1));
              Kokkos::atomic_add(&va(i,2),(KKF)(GeomScalar(0.5)*v2)); Kokkos::atomic_add(&va(i,3),(KKF)(GeomScalar(0.5)*v3));
              Kokkos::atomic_add(&va(i,4),(KKF)(GeomScalar(0.5)*v4)); Kokkos::atomic_add(&va(i,5),(KKF)(GeomScalar(0.5)*v5));
              Kokkos::atomic_add(&va(j,0),(KKF)(GeomScalar(0.5)*v0)); Kokkos::atomic_add(&va(j,1),(KKF)(GeomScalar(0.5)*v1));
              Kokkos::atomic_add(&va(j,2),(KKF)(GeomScalar(0.5)*v2)); Kokkos::atomic_add(&va(j,3),(KKF)(GeomScalar(0.5)*v3));
              Kokkos::atomic_add(&va(j,4),(KKF)(GeomScalar(0.5)*v4)); Kokkos::atomic_add(&va(j,5),(KKF)(GeomScalar(0.5)*v5));
            }
            if (do_cva) {
              const GeomScalar v6=dely*fx, v7=delz*fx, v8=delz*fy;
              Kokkos::atomic_add(&cva(i,0),(KKF)(GeomScalar(0.5)*v0)); Kokkos::atomic_add(&cva(i,1),(KKF)(GeomScalar(0.5)*v1));
              Kokkos::atomic_add(&cva(i,2),(KKF)(GeomScalar(0.5)*v2)); Kokkos::atomic_add(&cva(i,3),(KKF)(GeomScalar(0.5)*v3));
              Kokkos::atomic_add(&cva(i,4),(KKF)(GeomScalar(0.5)*v4)); Kokkos::atomic_add(&cva(i,5),(KKF)(GeomScalar(0.5)*v5));
              Kokkos::atomic_add(&cva(i,6),(KKF)(GeomScalar(0.5)*v6)); Kokkos::atomic_add(&cva(i,7),(KKF)(GeomScalar(0.5)*v7));
              Kokkos::atomic_add(&cva(i,8),(KKF)(GeomScalar(0.5)*v8));
              Kokkos::atomic_add(&cva(j,0),(KKF)(GeomScalar(0.5)*v0)); Kokkos::atomic_add(&cva(j,1),(KKF)(GeomScalar(0.5)*v1));
              Kokkos::atomic_add(&cva(j,2),(KKF)(GeomScalar(0.5)*v2)); Kokkos::atomic_add(&cva(j,3),(KKF)(GeomScalar(0.5)*v3));
              Kokkos::atomic_add(&cva(j,4),(KKF)(GeomScalar(0.5)*v4)); Kokkos::atomic_add(&cva(j,5),(KKF)(GeomScalar(0.5)*v5));
              Kokkos::atomic_add(&cva(j,6),(KKF)(GeomScalar(0.5)*v6)); Kokkos::atomic_add(&cva(j,7),(KKF)(GeomScalar(0.5)*v7));
              Kokkos::atomic_add(&cva(j,8),(KKF)(GeomScalar(0.5)*v8));
            }
          }
        }
      }, ev_force);
    if (vflag_global) {
      virial[0]+=ev_force.v[0]; virial[1]+=ev_force.v[1]; virial[2]+=ev_force.v[2];
      virial[3]+=ev_force.v[3]; virial[4]+=ev_force.v[4]; virial[5]+=ev_force.v[5];
    }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_L1_bond_recompute()
{
  int team_size = 1;
  int vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
    team_size = 32;
  (void) team_size; (void) vector_length;
  // Recompute: ComputeNeigh
  {
    team_size = fit_team_size<DeviceType, TagComputeNeigh>(*this, chunk_size, team_size, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>(chunk_size, team_size, vector_length);
    Kokkos::parallel_for("Recompute_ComputeNeigh", policy, *this);
  }
  // Recompute: ComputeRadialBasis
  {
    int ts = team_size;
    ts = fit_team_size<DeviceType, TagComputeRadialBasis>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>(
        ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
    Kokkos::parallel_for("Recompute_ComputeRadialBasis", policy, *this);
  }
  Kokkos::fence();
  // Recompute: Y_bond for this chunk. The forward-pass d_Y_bond is NOT valid:
  // it holds whatever the LAST forward chunk wrote, not the current chunk.
  compute_Y_bond_chunk();
  // Recompute: ComputeMLPRadial_R
  // cuBLAS SGEMM chain (CUDA+fp32/Mixed) or hand-kernel fallback.
  compute_mlp_radial(0);
  // Recompute: ComputeAi — reads d_R_nl + d_Y_bond (both refreshed above).
  {
    auto A_out = d_A;
    auto R_nl = d_R_nl;
    auto Y_bond = d_Y_bond;
    auto nc_v = d_ncount;
    auto mu_j_v = d_mu_j;
    auto z_tr_A_v = d_z_tr_A;
    auto l_from_v = d_l_from_lm;
    NNScalar L1_inv_v = L1_inv_avg_n_neigh;
    int lmax_v = lmax;
    int L1_nr = L1_nradmax;
    int cs = chunk_size;

    Kokkos::parallel_for("Recompute_ComputeAi",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, L1_nr}),
      KOKKOS_LAMBDA(const int ii, const int n) {
        // Zero A inline — eliminates deep_copy(d_A, 0.0)
        const int nlm = (lmax_v + 1) * (lmax_v + 1);
        for (int lm = 0; lm < nlm; lm++) A_out(ii, n, lm) = NNScalar(0.0);

        const int nct = nc_v(ii);
        for (int jj = 0; jj < nct; jj++) {
          const int mu_j = mu_j_v(ii, jj);

          // Lookup precomputed z_tr (includes A_lin_norm), multiply by inv_avg
          NNScalar z_n = z_tr_A_v(mu_j, n) * L1_inv_v;

          for (int lm = 0; lm < nlm; lm++) {
            const int l = l_from_v(lm);
            A_out(ii, n, lm) += R_nl(ii, jj, n, l) * (NNScalar)Y_bond(ii, jj, lm) * z_n;
          }
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_L2_bond_bwd()
{
  int team_size = 1;
  int vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
    team_size = 32;
  (void) team_size; (void) vector_length;
  // ---- B2-11a: precompute pair-basis YI adjoints ----
  // Reuse d_YI_pair as a temporary [atom, radial, pair] adjoint buffer.
  // This cuts both reverse dI and derivative inner loops from CG terms to
  // unique (Y_lm, I_idx) pairs for the large model.
  {
    auto pair_adj = d_YI_pair;
    auto yi_adj = d_YI_adj;
    auto pair_off = d_yi_pair_term_offset;
    auto pair_cnt = d_yi_pair_term_count;
    auto pair_out = d_yi_pair_term_out;
    auto pair_cg = d_yi_pair_term_cg;
    int cs = chunk_size, L2_nr = L2_nradmax, nPairs = n_yi_pairs;
    NNScalar yi_inv = (NNScalar) yi_inv_avg_n_neigh;

    Kokkos::parallel_for("ComputeYIAdj_pair",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, L2_nr, nPairs}),
      KOKKOS_LAMBDA(const int ii, const int n, const int pidx) {
        NNScalar acc = NNScalar(0.0);
        const int off = pair_off(pidx);
        const int cnt = pair_cnt(pidx);
        for (int q = 0; q < cnt; q++) {
          const int t = off + q;
          acc += pair_cg(t) * yi_adj(ii, n, pair_out(t));
        }
        pair_adj(ii, n, pidx) = acc * yi_inv;
      });
  }

  // ---- B2-11b: Accumulate d_grad_I_global — (ii, jj, n, I) parallelism ----
  // Group the already folded pair adjoints by I channel. This keeps the
  // high-parallelism reverse-YI launch shape but avoids re-walking all CG
  // terms for every bond/radial/I lane.
  {
    auto pair_adj = d_YI_pair;
    auto grad_I_g = d_grad_I_global;
    auto R1 = d_R1_nl;
    auto Y_bond = d_Y_bond;
    auto nc_v = d_ncount;
    auto near = d_nearest;
    auto pair_yidx = d_yi_pair_yidx;
    auto pair_l = d_yi_pair_l;
    auto pair_i_off = d_yi_i_pair_offset;
    auto pair_i_cnt = d_yi_i_pair_count;
    auto pair_i_idx = d_yi_i_pair_index;
    int I_nf = I_n_funcs;
    int L2_nr = L2_nradmax;
    int cs = chunk_size;

    Kokkos::parallel_for("ReverseYI_grad_I",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<4>>({0,0,0,0}, {cs, maxneigh, L2_nr, I_nf}),
      KOKKOS_LAMBDA(const int ii, const int jj, const int n, const int I_idx) {
        if (jj >= nc_v(ii)) return;
        const int j_global = near(ii, jj);

        NNScalar acc = NNScalar(0.0);
        const int off = pair_i_off(I_idx);
        const int cnt = pair_i_cnt(I_idx);
        for (int q = 0; q < cnt; q++) {
          const int pidx = pair_i_idx(off + q);
          const int y_idx = pair_yidx(pidx);
          const int l = pair_l(pidx);
          acc += pair_adj(ii, n, pidx) *
                 (NNScalar)Y_bond(ii, jj, y_idx) * R1(ii, jj, n, l);
        }
        if (acc != NNScalar(0.0))
          Kokkos::atomic_add(&grad_I_g(j_global, I_idx, n), acc);
      });
  }

  // ---- B2-11c: ComputeDerivative_L2 → d_f_ij_L2 (forces only) ----
  // Expose the radial axis as launch parallelism. Each (ii,jj,n) thread
  // computes one radial-channel force contribution and atomically accumulates
  // into the bond force. This trades some repeated angular setup for 32x more
  // parallel work in the dominant L2 derivative hotspot.
  {
    auto fij = d_f_ij_L2;
    int cs = chunk_size, mn = maxneigh, L2_nr = L2_nradmax;
    Kokkos::parallel_for("ZeroFijL2",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, mn, 3}),
      KOKKOS_LAMBDA(const int ii, const int jj, const int d) {
        fij(ii, jj, d) = GeomScalar(0.0);
      });

    auto nc_v = d_ncount;
    auto near = d_nearest;
    auto mu_j_v = d_mu_j;
    auto rnorms = d_rnorms;
    auto rhats = d_rhats;
    auto z_tr_B0_v = d_z_tr_B0;
    // Precision specialization: float reads precomputed DR1 = d_DR1_nl;
    // fp64 forms DR inline from the raw last-hidden deriv dh2 = d_dh2_R1 (original
    // bit-identical scheme). Both sets are captured; only the matching branch runs.
    auto DR1 = d_DR1_nl;
    auto dh2 = d_dh2_R1;
    auto mlp_dims = d_mlp_rad_R1_dims;
    auto mlp_W = d_mlp_rad_R1_W;
    auto mlp_norms = d_mlp_rad_R1_norms;
    int n_layers_R1 = mlp_rad_R1_n_layers;
    auto I_glob = d_I_global;
    auto pair_adj = d_YI_pair;
    auto pair_yidx = d_yi_pair_yidx;
    auto pair_iidx = d_yi_pair_iidx;
    auto R1 = d_R1_nl;
    auto B0_adj = d_B0_adj;
    auto idx_s = d_idx_sph;
    auto alm_v = alm; auto blm_v = blm; auto cl_v = cl; auto dl_v = dl;
    int lmax_v = lmax;
    int nPairs = n_yi_pairs;
    GeomScalar L2_inv = L2_inv_avg_n_neigh;
    GeomScalar Y00_v = Y00, sq3_v = sq3, sq3o2_v = sq3o2, sq2_v = sq2;

    Kokkos::parallel_for("ComputeDerivative_L2",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, mn, L2_nr}),
      KOKKOS_LAMBDA(const int ii, const int jj, const int n) {
        if (jj >= nc_v(ii)) return;

        const int j_global = near(ii, jj);
        const int mu_j = mu_j_v(ii, jj);
        const GeomScalar rinv = GeomScalar(1.0) / rnorms(ii, jj);
        const GeomScalar rx = rhats(ii, jj, 0);
        const GeomScalar ry = rhats(ii, jj, 1);
        const GeomScalar rz = rhats(ii, jj, 2);

        GeomScalar plm[15];
        GeomScalar dplm[15];
        plm[0] = Y00_v; dplm[0] = GeomScalar(0.0);
        if (lmax_v > 0) {
          plm[1] = Y00_v * sq3_v * rz; dplm[1] = Y00_v * sq3_v;
          plm[2] = -sq3o2_v * Y00_v; dplm[2] = GeomScalar(0.0);
          for (int ll = 2; ll <= lmax_v; ll++) {
            for (int mm = 0; mm < ll - 1; mm++) {
              const int idx = ll*(ll+1)/2 + mm;
              const int i1 = (ll-1)*ll/2 + mm, i2 = (ll-2)*(ll-1)/2 + mm;
              const int ai = idx_s(ll*(ll+1) + mm);
              const GeomScalar a = alm_v(ai), b = blm_v(ai);
              plm[idx] = a * (rz * plm[i1] + b * plm[i2]);
              dplm[idx] = a * (plm[i1] + rz * dplm[i1] + b * dplm[i2]);
            }
            { const int idx = ll*(ll+1)/2+ll-1, prev = (ll-1)*ll/2+ll-1;
              const GeomScalar t = dl_v(ll) * plm[prev];
              plm[idx] = t * rz; dplm[idx] = t; }
            { const int idx = ll*(ll+1)/2+ll, prev = (ll-1)*ll/2+ll-1;
              plm[idx] = cl_v(ll) * plm[prev]; dplm[idx] = GeomScalar(0.0); }
          }
        }

        NNScalar I_cache[GRACE2L_MAX_I_FUNCS];
        const int nIf_loc = (int) I_glob.extent(1);
        for (int idx = 0; idx < nIf_loc; idx++) I_cache[idx] = I_glob(j_global, idx, n);

        const int nlm = (lmax_v + 1) * (lmax_v + 1);
        NNScalar w_yi[25];
        for (int lm = 0; lm < nlm; lm++) w_yi[lm] = NNScalar(0.0);
        for (int p = 0; p < nPairs; p++) {
          w_yi[pair_yidx(p)] += pair_adj(ii, n, p) * I_cache[pair_iidx(p)];
        }

        // Precision specialization (fp64 only): cache the raw last-hidden
        // deriv so the inline DR matvec below matches the pre-cuBLAS baseline. DCE'd
        // in the float instantiation (which reads precomputed DR1 instead).
        // NOTE: plain `if` (not `if constexpr`) — nvcc forbids first-capturing a
        // variable inside a constexpr-if in an extended __host__ __device__ lambda.
        // sizeof(NNScalar) is a compile-time constant, so the dead branch is still
        // DCE'd (float keeps its register relief; fp64 keeps its exact arithmetic).
        NNScalar dh2_cache[GRACE2L_MAX_MLP_HIDDEN];
        int n_last_hidden_R1 = 0;
        if (sizeof(NNScalar) != 4) {
          n_last_hidden_R1 = mlp_dims(n_layers_R1 - 1);
          for (int k = 0; k < n_last_hidden_R1; k++) dh2_cache[k] = dh2(ii, jj, k);
        }

        const GeomScalar phase_re = rx, phase_im = ry;
        const NNScalar ztr_n = z_tr_B0_v(mu_j, n);
        GeomScalar f0 = GeomScalar(0.0), f1 = GeomScalar(0.0), f2 = GeomScalar(0.0);
        for (int l = 0; l <= lmax_v; l++) {
          const GeomScalar R = R1(ii, jj, n, l);
          NNScalar DR;
          if (sizeof(NNScalar) == 4) {   // plain if (see note above); constant-folded
            DR = DR1(ii, jj, n, l);
          } else {
            DR = NNScalar(0.0);
            const int j_idx = n * (lmax_v + 1) + l;
            const int last_layer = n_layers_R1 - 1;
            for (int k = 0; k < n_last_hidden_R1; k++)
              DR += mlp_W(last_layer, k, j_idx) * dh2_cache[k];
            DR *= mlp_norms(last_layer);
          }
          const GeomScalar R_over_r = R * rinv;

          {
            const int lm0 = l*(l+1);
            const GeomScalar Y = plm[l*(l+1)/2];
            const GeomScalar dp = dplm[l*(l+1)/2];
            const GeomScalar rdy = dp * rz;
            const GeomScalar DY_x = -rdy * rx, DY_y = -rdy * ry, DY_z = dp - rdy * rz;
            NNScalar w = B0_adj(ii, n, lm0) * ztr_n * L2_inv + w_yi[lm0];
            const GeomScalar YDR = Y * DR;
            f0 += w * (YDR * rx + DY_x * R_over_r);
            f1 += w * (YDR * ry + DY_y * R_over_r);
            f2 += w * (YDR * rz + DY_z * R_over_r);
          }

          GeomScalar pm_re = phase_re, pm_im = phase_im;
          for (int m = 1; m <= l; m++) {
            const int fac = (m % 2 == 0) ? 1 : -1;
            const GeomScalar pv = plm[l*(l+1)/2+m], dpv = dplm[l*(l+1)/2+m];
            const GeomScalar ylm_re = pm_re * pv, ylm_im = pm_im * pv;
            const GeomScalar rYp = sq2_v * fac * ylm_re, rYn = sq2_v * fac * ylm_im;

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
            const GeomScalar Dpx_re = dyx_re - rdy_re*rx, Dpx_im = dyx_im - rdy_im*rx;
            const GeomScalar Dpy_re = dyy_re - rdy_re*ry, Dpy_im = dyy_im - rdy_im*ry;
            const GeomScalar Dpz_re = dyz_re - rdy_re*rz, Dpz_im = dyz_im - rdy_im*rz;
            const GeomScalar DYpx = sq2_v*fac*Dpx_re, DYpy = sq2_v*fac*Dpy_re, DYpz = sq2_v*fac*Dpz_re;
            const GeomScalar DYnx = sq2_v*fac*Dpx_im, DYny = sq2_v*fac*Dpy_im, DYnz = sq2_v*fac*Dpz_im;

            const int Ap = l*(l+1)+m, An = l*(l+1)-m;
            NNScalar wp = B0_adj(ii, n, Ap) * ztr_n * L2_inv + w_yi[Ap];
            NNScalar wn = B0_adj(ii, n, An) * ztr_n * L2_inv + w_yi[An];

            { const GeomScalar YDR = rYp * DR;
              f0 += wp * (YDR*rx + DYpx*R_over_r);
              f1 += wp * (YDR*ry + DYpy*R_over_r);
              f2 += wp * (YDR*rz + DYpz*R_over_r); }
            { const GeomScalar YDR = rYn * DR;
              f0 += wn * (YDR*rx + DYnx*R_over_r);
              f1 += wn * (YDR*ry + DYny*R_over_r);
              f2 += wn * (YDR*rz + DYnz*R_over_r); }

            const GeomScalar t_re = pm_re*phase_re - pm_im*phase_im;
            const GeomScalar t_im = pm_re*phase_im + pm_im*phase_re;
            pm_re = t_re; pm_im = t_im;
          }
        }

        Kokkos::atomic_add(&fij(ii, jj, 0), f0);
        Kokkos::atomic_add(&fij(ii, jj, 1), f1);
        Kokkos::atomic_add(&fij(ii, jj, 2), f2);
      });
  }

  // ---- B2-12: ComputeForce_L2 → atom forces + virial ----
  {
    auto fij = d_f_ij_L2; auto nc = d_ncount; auto near = d_nearest;
    auto fout = f; auto il = d_ilist;
    auto rh = d_rhats; auto rn = d_rnorms;
    int co = chunk_offset, cs = chunk_size;
    bool do_v = (vflag_global != 0);
    bool do_va = (vflag_atom != 0);
    bool do_cva = (cvflag_atom != 0);
    auto va = d_vatom; auto cva = d_cvatom;

    EV ev_force;
    Kokkos::parallel_reduce("ComputeForce_L2", Kokkos::RangePolicy<DeviceType>(0, cs),
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
            const GeomScalar v0 = delx*fx, v1 = dely*fy, v2 = delz*fz;
            const GeomScalar v3 = delx*fy, v4 = delx*fz, v5 = dely*fz;
            if (do_v) { ev.v[0]+=v0; ev.v[1]+=v1; ev.v[2]+=v2; ev.v[3]+=v3; ev.v[4]+=v4; ev.v[5]+=v5; }
            if (do_va) {
              Kokkos::atomic_add(&va(i,0),(KKF)(GeomScalar(0.5)*v0)); Kokkos::atomic_add(&va(i,1),(KKF)(GeomScalar(0.5)*v1));
              Kokkos::atomic_add(&va(i,2),(KKF)(GeomScalar(0.5)*v2)); Kokkos::atomic_add(&va(i,3),(KKF)(GeomScalar(0.5)*v3));
              Kokkos::atomic_add(&va(i,4),(KKF)(GeomScalar(0.5)*v4)); Kokkos::atomic_add(&va(i,5),(KKF)(GeomScalar(0.5)*v5));
              Kokkos::atomic_add(&va(j,0),(KKF)(GeomScalar(0.5)*v0)); Kokkos::atomic_add(&va(j,1),(KKF)(GeomScalar(0.5)*v1));
              Kokkos::atomic_add(&va(j,2),(KKF)(GeomScalar(0.5)*v2)); Kokkos::atomic_add(&va(j,3),(KKF)(GeomScalar(0.5)*v3));
              Kokkos::atomic_add(&va(j,4),(KKF)(GeomScalar(0.5)*v4)); Kokkos::atomic_add(&va(j,5),(KKF)(GeomScalar(0.5)*v5));
            }
            if (do_cva) {
              const GeomScalar v6=dely*fx, v7=delz*fx, v8=delz*fy;
              Kokkos::atomic_add(&cva(i,0),(KKF)(GeomScalar(0.5)*v0)); Kokkos::atomic_add(&cva(i,1),(KKF)(GeomScalar(0.5)*v1));
              Kokkos::atomic_add(&cva(i,2),(KKF)(GeomScalar(0.5)*v2)); Kokkos::atomic_add(&cva(i,3),(KKF)(GeomScalar(0.5)*v3));
              Kokkos::atomic_add(&cva(i,4),(KKF)(GeomScalar(0.5)*v4)); Kokkos::atomic_add(&cva(i,5),(KKF)(GeomScalar(0.5)*v5));
              Kokkos::atomic_add(&cva(i,6),(KKF)(GeomScalar(0.5)*v6)); Kokkos::atomic_add(&cva(i,7),(KKF)(GeomScalar(0.5)*v7));
              Kokkos::atomic_add(&cva(i,8),(KKF)(GeomScalar(0.5)*v8));
              Kokkos::atomic_add(&cva(j,0),(KKF)(GeomScalar(0.5)*v0)); Kokkos::atomic_add(&cva(j,1),(KKF)(GeomScalar(0.5)*v1));
              Kokkos::atomic_add(&cva(j,2),(KKF)(GeomScalar(0.5)*v2)); Kokkos::atomic_add(&cva(j,3),(KKF)(GeomScalar(0.5)*v3));
              Kokkos::atomic_add(&cva(j,4),(KKF)(GeomScalar(0.5)*v4)); Kokkos::atomic_add(&cva(j,5),(KKF)(GeomScalar(0.5)*v5));
              Kokkos::atomic_add(&cva(j,6),(KKF)(GeomScalar(0.5)*v6)); Kokkos::atomic_add(&cva(j,7),(KKF)(GeomScalar(0.5)*v7));
              Kokkos::atomic_add(&cva(j,8),(KKF)(GeomScalar(0.5)*v8));
            }
          }
        }
      }, ev_force);
    if (vflag_global) {
      virial[0]+=ev_force.v[0]; virial[1]+=ev_force.v[1]; virial[2]+=ev_force.v[2];
      virial[3]+=ev_force.v[3]; virial[4]+=ev_force.v[4]; virial[5]+=ev_force.v[5];
    }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_L2_bond_fwd()
{
  const int chunk_limit = chunk_cap();
  int team_size = 1;
  int vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1)
    team_size = 32;
  (void) team_size; (void) vector_length;
  // ---- L2 Stage 0: Recompute neighbor data for this L2 chunk ----
  // Multi-chunk: d_ncount/d_nearest/d_rhats/d_radial_basis/d_Y_bond are stale
  // from Phase 1's last chunk. Single-chunk: Phase 1 data is still valid, skip.
  if (inum > chunk_limit) {
    {
      team_size = fit_team_size<DeviceType, TagComputeNeigh>(*this, chunk_size, team_size, vector_length);
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>(chunk_size, team_size, vector_length);
      Kokkos::parallel_for("L2_ComputeNeigh", policy, *this);
    }
    {
      int ts = team_size;
      ts = fit_team_size<DeviceType, TagComputeRadialBasis>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
      auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>(
          ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
      Kokkos::parallel_for("L2_ComputeRadialBasis", policy, *this);
    }
    Kokkos::fence();
    // Refresh d_Y_bond for the current chunk — ComputeAi_B0 / ComputeYI read it.
    compute_Y_bond_chunk();
  }

  // ---- L2 Stage 1: ComputeMLPRadial_R1 ----
  // cuBLAS SGEMM chain (CUDA+fp32/Mixed) or hand-kernel fallback.
  compute_mlp_radial(1);

  // ---- L2 Stage 2: ComputeAi_B0 — per-(ii,n) parallelization ----
  // Reads precomputed Y_vals from d_Y_bond instead of recomputing per (n).
  {
    auto B0_out = d_B0;
    auto R1_nl = d_R1_nl;
    auto Y_bond = d_Y_bond;
    auto nc_v = d_ncount;
    auto mu_j_v = d_mu_j;
    auto z_tr_B0_v = d_z_tr_B0;
    auto l_from_v = d_l_from_lm;
    NNScalar L2_inv_v = L2_inv_avg_n_neigh;
    int lmax_v = lmax;
    int L2_nr = L2_nradmax;
    int cs = chunk_size;

    Kokkos::parallel_for("ComputeAi_B0",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, L2_nr}),
      KOKKOS_LAMBDA(const int ii, const int n) {
        // Zero B0 inline — eliminates deep_copy(d_B0, 0.0)
        const int nlm = (lmax_v + 1) * (lmax_v + 1);
        for (int lm = 0; lm < nlm; lm++) B0_out(ii, n, lm) = NNScalar(0.0);

        const int nct = nc_v(ii);
        for (int jj = 0; jj < nct; jj++) {
          const int mu_j = mu_j_v(ii, jj);

          // Lookup precomputed z_tr (includes B0_lin_norm), multiply by inv_avg
          NNScalar z_n = z_tr_B0_v(mu_j, n) * L2_inv_v;

          // Accumulate B0 — no atomics, this (ii,n) thread owns d_B0(ii,n,*)
          for (int lm = 0; lm < nlm; lm++) {
            const int l = l_from_v(lm);
            B0_out(ii, n, lm) += R1_nl(ii, jj, n, l) * (NNScalar)Y_bond(ii, jj, lm) * z_n;
          }
        }
      });
  }

  // ---- L2 Stage 3: ComputeYI — pair-basis contraction ----
  // The 877 CG terms reuse only 385 unique (Y_lm, I_idx) pairs for this
  // model. Accumulate those pair rows over neighbors once, then reduce the
  // sparse CG terms into the 190 output functions.
  {
    auto YI_pair = d_YI_pair;
    auto YI_out = d_YI;
    auto I_glob = d_I_global;
    auto R1 = d_R1_nl;
    auto Y_bond = d_Y_bond;
    auto nc_v = d_ncount;
    auto near = d_nearest;
    auto pair_yidx = d_yi_pair_yidx;
    auto pair_iidx = d_yi_pair_iidx;
    auto pair_l = d_yi_pair_l;
    auto yi_out_off = d_yi_out_offset;
    auto yi_out_cnt = d_yi_out_count;
    auto yi_cg_v = d_yi_cg_coeff;
    auto term_pair = d_yi_term_pair;
    int L2_nr = L2_nradmax;
    GeomScalar inv_avg = yi_inv_avg_n_neigh;
    int cs = chunk_size, nYout = n_yi_out_funcs, nPairs = n_yi_pairs;

    Kokkos::parallel_for("ComputeYI_pair",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, L2_nr, nPairs}),
      KOKKOS_LAMBDA(const int ii, const int n, const int pidx) {
        NNScalar acc = NNScalar(0.0);
        const int y_idx = pair_yidx(pidx);
        const int I_idx = pair_iidx(pidx);
        const int l = pair_l(pidx);
        const int nct = nc_v(ii);
        for (int jj = 0; jj < nct; jj++) {
          const int j_global = near(ii, jj);
          acc += (NNScalar)Y_bond(ii, jj, y_idx) *
                 R1(ii, jj, n, l) * I_glob(j_global, I_idx, n);
        }
        YI_pair(ii, n, pidx) = acc * inv_avg;
      });

    Kokkos::parallel_for("ComputeYI",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, L2_nr, nYout}),
      KOKKOS_LAMBDA(const int ii, const int n, const int out_f) {
        NNScalar acc = NNScalar(0.0);
        const int off = yi_out_off(out_f);
        const int cnt = yi_out_cnt(out_f);
        for (int p = 0; p < cnt; p++) {
          const int t = off + p;
          acc += yi_cg_v(t) * YI_pair(ii, n, term_pair(t));
        }
        YI_out(ii, n, out_f) = acc;
      });
  }
}

// Reverse-FC stages (virtual hooks; Grace2LFusedKK overrides them)

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_fc_A1()
{
  // ---- B1-10: ReverseFC_A1 → d_A_adj += ----
  {
    auto oa = d_A1_adj; auto aa = d_A_adj;
    auto wl = d_fc_A1_wl; auto wr = d_fc_A1_wr;
    auto wtl = d_fc_A1_wtl; auto wtr = d_fc_A1_wtr;
    auto ct = d_fc_A1_ct; auto cf = d_fc_A1_cf; auto nof = d_fc_A1_nof;
    NNScalar nl = fc_A1_nl, nr = fc_A1_nr;
    int no = L1_fc_n_out, nlm = n_funcs_A1, ncol = (int)d_fc_A1_cf.extent(0), cs = chunk_size;
    { int ni = (int)aa.extent(1);
    Kokkos::parallel_for("RevFC_A1_L",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ni, nlm}),
      KOKKOS_LAMBDA(const int ii, const int n, const int lm) {
        NNScalar nf = nof(lm) * nl; int tile = wtl(lm);
        NNScalar val = 0.0;
        for (int k = 0; k < no; k++) val += wl(k, n, tile) * oa(ii, k, lm);
        aa(ii, n, lm) += val * nf;
      }); }
    Kokkos::parallel_for("RevFC_A1_R",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, ncol}),
      KOKKOS_LAMBDA(const int ii, const int idx) {
        int ni = aa.extent(1);
        int sl = cf(idx), tl = ct(idx), tile = wtr(idx);
        NNScalar nf = nof(tl) * nr;
        for (int n = 0; n < ni; n++) {
          NNScalar val = 0.0;
          for (int k = 0; k < no; k++) val += wr(k, n, tile) * oa(ii, k, tl);
          Kokkos::atomic_add(&aa(ii, n, sl), val * nf);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_fc_AA1()
{
  // ---- B1-8: ReverseFC_AA1 → d_AA_adj +=, d_A_adj += ----
  {
    auto oa = d_AA1_adj; auto la = d_AA_adj; auto ra = d_A_adj;
    auto wl = d_fc_AA1_wl; auto wr = d_fc_AA1_wr;
    auto wtl = d_fc_AA1_wtl; auto wtr = d_fc_AA1_wtr;
    auto ct = d_fc_AA1_ct; auto cf = d_fc_AA1_cf; auto nof = d_fc_AA1_nof;
    NNScalar nl = fc_AA1_nl, nr = fc_AA1_nr;
    int no = L1_fc_n_out, nlm = n_funcs_AA, ncol = (int)d_fc_AA1_cf.extent(0), cs = chunk_size;
    { int nil = (int)la.extent(1);
    Kokkos::parallel_for("RevFC_AA1_L",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nil, nlm}),
      KOKKOS_LAMBDA(const int ii, const int n, const int lm) {
        NNScalar nf = nof(lm) * nl; int tile = wtl(lm);
        NNScalar val = 0.0;
        for (int k = 0; k < no; k++) val += wl(k, n, tile) * oa(ii, k, lm);
        la(ii, n, lm) += val * nf;
      }); }
    { int nir = (int)ra.extent(1);
    Kokkos::parallel_for("RevFC_AA1_R",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ncol, nir}),
      KOKKOS_LAMBDA(const int ii, const int idx, const int n) {
        int sl = cf(idx), tl = ct(idx), tile = wtr(idx);
        NNScalar val = 0.0;
        for (int k = 0; k < no; k++) val += wr(k, n, tile) * oa(ii, k, tl);
        Kokkos::atomic_add(&ra(ii, n, sl), val * nof(tl) * nr);
      }); }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_fc_AA2()
{
  // ---- B1-6: ReverseFC_AA2 → d_AA_adj +=, d_A_adj += ----
  {
    auto oa = d_AA2_adj; auto la = d_AA_adj; auto ra = d_A_adj;
    auto wl = d_fc_AA2_wl; auto wr = d_fc_AA2_wr;
    auto wtl = d_fc_AA2_wtl; auto wtr = d_fc_AA2_wtr;
    auto ct = d_fc_AA2_ct; auto cf = d_fc_AA2_cf; auto nof = d_fc_AA2_nof;
    NNScalar nl = fc_AA2_nl, nr = fc_AA2_nr;
    int no = L1_fc_n_out, nlm = n_funcs_AA, ncol = (int)d_fc_AA2_cf.extent(0), cs = chunk_size;
    { int nil = (int)la.extent(1);
    Kokkos::parallel_for("RevFC_AA2_L",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nil, nlm}),
      KOKKOS_LAMBDA(const int ii, const int n, const int lm) {
        NNScalar nf = nof(lm) * nl; int tile = wtl(lm);
        NNScalar val = 0.0;
        for (int k = 0; k < no; k++) val += wl(k, n, tile) * oa(ii, k, lm);
        la(ii, n, lm) += val * nf;
      }); }
    { int nir = (int)ra.extent(1);
    Kokkos::parallel_for("RevFC_AA2_R",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ncol, nir}),
      KOKKOS_LAMBDA(const int ii, const int idx, const int n) {
        int sl = cf(idx), tl = ct(idx), tile = wtr(idx);
        NNScalar val = 0.0;
        for (int k = 0; k < no; k++) val += wr(k, n, tile) * oa(ii, k, tl);
        Kokkos::atomic_add(&ra(ii, n, sl), val * nof(tl) * nr);
      }); }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_fc_B1()
{
  // ---- B2-9: ReverseFC_B1 → d_B_adj += (both left and right are B) ----
  {
    auto oa = d_B1_adj; auto aa = d_B_adj;
    auto wl = d_fc_B1_wl; auto wr = d_fc_B1_wr;
    auto wtl = d_fc_B1_wtl; auto wtr = d_fc_B1_wtr;
    auto ct = d_fc_B1_ct; auto cf = d_fc_B1_cf; auto nof = d_fc_B1_nof;
    NNScalar nl = fc_B1_nl, nr = fc_B1_nr;
    int no = L2_fc_n_out, nlm = n_funcs_B1, ncol = (int)d_fc_B1_cf.extent(0), cs = chunk_size;
    { int ni = (int)aa.extent(1);
    Kokkos::parallel_for("RevFC_B1_L",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ni, nlm}),
      KOKKOS_LAMBDA(const int ii, const int n, const int lm) {
        NNScalar nf = nof(lm) * nl; int tile = wtl(lm);
        NNScalar val = 0.0;
        for (int k = 0; k < no; k++) val += wl(k, n, tile) * oa(ii, k, lm);
        aa(ii, n, lm) += val * nf;
      }); }
    Kokkos::parallel_for("RevFC_B1_R",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, ncol}),
      KOKKOS_LAMBDA(const int ii, const int idx) {
        int ni = aa.extent(1);
        int sl = cf(idx), tl = ct(idx), tile = wtr(idx);
        NNScalar nf = nof(tl) * nr;
        for (int n = 0; n < ni; n++) {
          NNScalar val = 0.0;
          for (int k = 0; k < no; k++) val += wr(k, n, tile) * oa(ii, k, tl);
          Kokkos::atomic_add(&aa(ii, n, sl), val * nf);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_fc_BB1()
{
  // ---- B2-7: ReverseFC_BB1 → d_BB_adj +=, d_B_adj += ----
  {
    auto oa = d_BB1_adj; auto la = d_BB_adj; auto ra = d_B_adj;
    auto wl = d_fc_BB1_wl; auto wr = d_fc_BB1_wr;
    auto wtl = d_fc_BB1_wtl; auto wtr = d_fc_BB1_wtr;
    auto ct = d_fc_BB1_ct; auto cf = d_fc_BB1_cf; auto nof = d_fc_BB1_nof;
    NNScalar nl = fc_BB1_nl, nr = fc_BB1_nr;
    int no = L2_fc_n_out, nlm = n_funcs_BB, ncol = (int)d_fc_BB1_cf.extent(0), cs = chunk_size;
    { int nil = (int)la.extent(1);
    Kokkos::parallel_for("RevFC_BB1_L",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nil, nlm}),
      KOKKOS_LAMBDA(const int ii, const int n, const int lm) {
        NNScalar nf = nof(lm) * nl; int tile = wtl(lm);
        NNScalar val = 0.0;
        for (int k = 0; k < no; k++) val += wl(k, n, tile) * oa(ii, k, lm);
        la(ii, n, lm) += val * nf;
      }); }
    { int nir = (int)ra.extent(1);
    Kokkos::parallel_for("RevFC_BB1_R",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ncol, nir}),
      KOKKOS_LAMBDA(const int ii, const int idx, const int n) {
        int sl = cf(idx), tl = ct(idx), tile = wtr(idx);
        NNScalar val = 0.0;
        for (int k = 0; k < no; k++) val += wr(k, n, tile) * oa(ii, k, tl);
        Kokkos::atomic_add(&ra(ii, n, sl), val * nof(tl) * nr);
      }); }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_fc_BB2()
{
  // ---- B2-5: ReverseFC_BB2 → d_BB_adj +=, d_B_adj += ----
  {
    auto oa = d_BB2_adj; auto la = d_BB_adj; auto ra = d_B_adj;
    auto wl = d_fc_BB2_wl; auto wr = d_fc_BB2_wr;
    auto wtl = d_fc_BB2_wtl; auto wtr = d_fc_BB2_wtr;
    auto ct = d_fc_BB2_ct; auto cf = d_fc_BB2_cf; auto nof = d_fc_BB2_nof;
    NNScalar nl = fc_BB2_nl, nr = fc_BB2_nr;
    int no = L2_fc_n_out, nlm = n_funcs_BB, ncol = (int)d_fc_BB2_cf.extent(0), cs = chunk_size;
    { int nil = (int)la.extent(1);
    Kokkos::parallel_for("RevFC_BB2_L",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nil, nlm}),
      KOKKOS_LAMBDA(const int ii, const int n, const int lm) {
        NNScalar nf = nof(lm) * nl; int tile = wtl(lm);
        NNScalar val = 0.0;
        for (int k = 0; k < no; k++) val += wl(k, n, tile) * oa(ii, k, lm);
        la(ii, n, lm) += val * nf;
      }); }
    { int nir = (int)ra.extent(1);
    Kokkos::parallel_for("RevFC_BB2_R",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, ncol, nir}),
      KOKKOS_LAMBDA(const int ii, const int idx, const int n) {
        int sl = cf(idx), tl = ct(idx), tile = wtr(idx);
        NNScalar val = 0.0;
        for (int k = 0; k < no; k++) val += wr(k, n, tile) * oa(ii, k, tl);
        Kokkos::atomic_add(&ra(ii, n, sl), val * nof(tl) * nr);
      }); }
  }
}

// CG-product stages (virtual hooks; Grace2LFusedKK overrides them)

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_prod_AA()
{
  // Stage 6: Product AA = A1 ⊗ A1
  {
    auto left = d_A1; auto right = d_A1; auto out = d_AA;
    auto li = d_prod_AA_li; auto ri = d_prod_AA_ri;
    auto si = d_prod_AA_si; auto cg = d_prod_AA_cg;
    int nt = n_cg_AA, nc = L1_fc_n_out, cs = chunk_size;
    int nf_AA = (int)d_AA.extent(2);
    Kokkos::parallel_for("Product_AA",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_AA; o++) out(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          int l = li(t), r = ri(t), o = si(t);
          out(ii, k, o) += cg(t) * left(ii, k, l) * right(ii, k, r);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_prod_AAA()
{
  // Stage 8: Product AAA = AA1 ⊗ A (NOTE: right=A not A1!)
  {
    auto left = d_AA1; auto right = d_A; auto out = d_AAA;
    auto li = d_prod_AAA_li; auto ri = d_prod_AAA_ri;
    auto si = d_prod_AAA_si; auto cg = d_prod_AAA_cg;
    int nt = n_cg_AAA, nc = L1_fc_n_out, cs = chunk_size;
    int nf_AAA = (int)d_AAA.extent(2);
    Kokkos::parallel_for("Product_AAA",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_AAA; o++) out(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          int l = li(t), r = ri(t), o = si(t);
          out(ii, k, o) += cg(t) * left(ii, k, l) * right(ii, k, r);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_prod_AAAA()
{
  // Stage 10: Product AAAA = AA2 ⊗ AA2
  {
    auto left = d_AA2; auto right = d_AA2; auto out = d_AAAA;
    auto li = d_prod_AAAA_li; auto ri = d_prod_AAAA_ri;
    auto si = d_prod_AAAA_si; auto cg = d_prod_AAAA_cg;
    int nt = n_cg_AAAA, nc = L1_fc_n_out, cs = chunk_size;
    int nf_AAAA = (int)d_AAAA.extent(2);
    Kokkos::parallel_for("Product_AAAA",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_AAAA; o++) out(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          int l = li(t), r = ri(t), o = si(t);
          out(ii, k, o) += cg(t) * left(ii, k, l) * right(ii, k, r);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_prod_BB()
{
  // Stage 6: Product BB = B1 ⊗ B1
  {
    auto left = d_B1; auto right = d_B1; auto out = d_BB;
    auto li = d_prod_BB_li; auto ri = d_prod_BB_ri;
    auto si = d_prod_BB_si; auto cg = d_prod_BB_cg;
    int nt = n_cg_BB, nc = L2_fc_n_out, cs = chunk_size;
    int nf_BB = (int)d_BB.extent(2);
    Kokkos::parallel_for("Product_BB",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_BB; o++) out(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          int l = li(t), r = ri(t), o = si(t);
          out(ii, k, o) += cg(t) * left(ii, k, l) * right(ii, k, r);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_prod_BBB()
{
  // Stage 8: Product BBB = BB1 ⊗ B
  {
    auto left = d_BB1; auto right = d_B; auto out = d_BBB;
    auto li = d_prod_BBB_li; auto ri = d_prod_BBB_ri;
    auto si = d_prod_BBB_si; auto cg = d_prod_BBB_cg;
    int nt = n_cg_BBB, nc = L2_fc_n_out, cs = chunk_size;
    int nf_BBB = (int)d_BBB.extent(2);
    Kokkos::parallel_for("Product_BBB",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_BBB; o++) out(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          int l = li(t), r = ri(t), o = si(t);
          out(ii, k, o) += cg(t) * left(ii, k, l) * right(ii, k, r);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_prod_BBBB()
{
  // Stage 10: Product BBBB = BB2 ⊗ BB2
  {
    auto left = d_BB2; auto right = d_BB2; auto out = d_BBBB;
    auto li = d_prod_BBBB_li; auto ri = d_prod_BBBB_ri;
    auto si = d_prod_BBBB_si; auto cg = d_prod_BBBB_cg;
    int nt = n_cg_BBBB, nc = L2_fc_n_out, cs = chunk_size;
    int nf_BBBB = (int)d_BBBB.extent(2);
    Kokkos::parallel_for("Product_BBBB",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_BBBB; o++) out(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          int l = li(t), r = ri(t), o = si(t);
          out(ii, k, o) += cg(t) * left(ii, k, l) * right(ii, k, r);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_prod_AA()
{
  // ---- B1-9: ReverseProduct_AA (A1 ⊗ A1 self) → d_A1_adj += ----
  {
    auto fwd = d_A1; auto oa = d_AA_adj; auto adj = d_A1_adj;
    auto li = d_prod_AA_li; auto ri = d_prod_AA_ri;
    auto si = d_prod_AA_si; auto cg = d_prod_AA_cg;
    int nt = n_cg_AA, nc = L1_fc_n_out, cs = chunk_size;
    int nf_A1 = (int)d_A1_adj.extent(2);
    Kokkos::parallel_for("RevProd_AA",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_A1; o++) adj(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          const int l = li(t), r = ri(t), o = si(t);
          const NNScalar d = cg(t) * oa(ii, k, o);
          adj(ii, k, l) += d * fwd(ii, k, r);
          adj(ii, k, r) += d * fwd(ii, k, l);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_prod_AAA()
{
  // ---- B1-7: ReverseProduct_AAA (AA1 ⊗ A) → d_AA1_adj, d_A_adj += ----
  {
    auto fl = d_AA1; auto fr = d_A; auto oa = d_AAA_adj;
    auto la = d_AA1_adj; auto ra = d_A_adj;  // right=A, so adjoint goes to d_A_adj
    auto li = d_prod_AAA_li; auto ri = d_prod_AAA_ri;
    auto si = d_prod_AAA_si; auto cg = d_prod_AAA_cg;
    int nt = n_cg_AAA, nc = L1_fc_n_out, cs = chunk_size;
    int nf_AA1 = (int)d_AA1_adj.extent(2);
    Kokkos::parallel_for("RevProd_AAA",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_AA1; o++) la(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          const int l = li(t), r = ri(t), o = si(t);
          const NNScalar d = cg(t) * oa(ii, k, o);
          la(ii, k, l) += d * fr(ii, k, r);
          ra(ii, k, r) += d * fl(ii, k, l);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_prod_AAAA()
{
  // ---- B1-5: ReverseProduct_AAAA (AA2 ⊗ AA2 self) → d_AA2_adj ----
  {
    auto fwd = d_AA2; auto out_adj = d_AAAA_adj; auto adj = d_AA2_adj;
    auto li = d_prod_AAAA_li; auto ri = d_prod_AAAA_ri;
    auto si = d_prod_AAAA_si; auto cg = d_prod_AAAA_cg;
    int nt = n_cg_AAAA, nc = L1_fc_n_out, cs = chunk_size;
    int nf_AA2 = (int)d_AA2_adj.extent(2);
    Kokkos::parallel_for("RevProd_AAAA",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_AA2; o++) adj(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          const int l = li(t), r = ri(t), o = si(t);
          const NNScalar d = cg(t) * out_adj(ii, k, o);
          adj(ii, k, l) += d * fwd(ii, k, r);
          adj(ii, k, r) += d * fwd(ii, k, l);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_prod_BB()
{
  // ---- B2-8: ReverseProduct_BB (B1 ⊗ B1 self-product) → d_B1_adj += ----
  {
    auto fwd = d_B1; auto oa = d_BB_adj; auto adj = d_B1_adj;
    auto li = d_prod_BB_li; auto ri = d_prod_BB_ri;
    auto si = d_prod_BB_si; auto cg = d_prod_BB_cg;
    int nt = n_cg_BB, nc = L2_fc_n_out, cs = chunk_size;
    int nf_B1 = (int)d_B1_adj.extent(2);
    Kokkos::parallel_for("RevProd_BB",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_B1; o++) adj(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          const int l = li(t), r = ri(t), o = si(t);
          const NNScalar d = cg(t) * oa(ii, k, o);
          adj(ii, k, l) += d * fwd(ii, k, r);
          adj(ii, k, r) += d * fwd(ii, k, l);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_prod_BBB()
{
  // ---- B2-6: ReverseProduct_BBB (BB1 ⊗ B) → d_BB1_adj, d_B_adj += ----
  {
    auto fl = d_BB1; auto fr = d_B; auto oa = d_BBB_adj;
    auto la = d_BB1_adj; auto ra = d_B_adj;
    auto li = d_prod_BBB_li; auto ri = d_prod_BBB_ri;
    auto si = d_prod_BBB_si; auto cg = d_prod_BBB_cg;
    int nt = n_cg_BBB, nc = L2_fc_n_out, cs = chunk_size;
    int nf_BB1 = (int)d_BB1_adj.extent(2);
    Kokkos::parallel_for("RevProd_BBB",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_BB1; o++) la(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          const int l = li(t), r = ri(t), o = si(t);
          const NNScalar d = cg(t) * oa(ii, k, o);
          la(ii, k, l) += d * fr(ii, k, r);
          ra(ii, k, r) += d * fl(ii, k, l);
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_prod_BBBB()
{
  // ---- B2-4: ReverseProduct_BBBB (BB2 ⊗ BB2 self-product) → d_BB2_adj ----
  {
    auto fwd = d_BB2; auto out_adj = d_BBBB_adj; auto adj = d_BB2_adj;
    auto li = d_prod_BBBB_li; auto ri = d_prod_BBBB_ri;
    auto si = d_prod_BBBB_si; auto cg = d_prod_BBBB_cg;
    int nt = n_cg_BBBB, nc = L2_fc_n_out, cs = chunk_size;
    int nf_BB2 = (int)d_BB2_adj.extent(2);
    Kokkos::parallel_for("RevProd_BBBB",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int o = 0; o < nf_BB2; o++) adj(ii, k, o) = NNScalar(0.0);
        for (int t = 0; t < nt; t++) {
          const int l = li(t), r = ri(t), o = si(t);
          const NNScalar d = cg(t) * out_adj(ii, k, o);
          adj(ii, k, l) += d * fwd(ii, k, r);
          adj(ii, k, r) += d * fwd(ii, k, l);
        }
      });
  }
}

// ReduceN stages (virtual hooks; Grace2LFusedKK overrides them)

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_reduce_B()
{
  // ---- L2 Stage 4: ReduceN B (equivariant, NOT element-dependent) ----
  {
    auto B_out = d_B;
    int cs = chunk_size, nBout = B_n_out, nBf = B_n_funcs;

    auto W_YI = d_B_reduce_YI_W; auto ci_YI = d_B_ci_YI;
    auto wlt_YI = d_B_wlt_YI; auto tsi_YI = d_B_tsi_YI;
    NNScalar norm_YI = B_reduce_YI_norm; int nc_YI = B_nc_YI; auto X_YI = d_YI;

    auto W_B0 = d_B_reduce_B0_W; auto ci_B0 = d_B_ci_B0;
    auto wlt_B0 = d_B_wlt_B0; auto tsi_B0 = d_B_tsi_B0;
    NNScalar norm_B0 = B_reduce_B0_norm; int nc_B0 = B_nc_B0; auto X_B0 = d_B0;

    bool reduce_done = false;
#ifdef KOKKOS_ENABLE_CUDA
    // element-independent B ReduceN via cuBLAS (float only). YI zeroes
    // all nBf output funcs, B0 accumulates on top. fp64/non-CUDA use the hand kernel.
    if constexpr (sizeof(NNScalar) == 4)
      if (blas_handle && reduce_op_B_YI.active) {
        reduce_forward_blas(reduce_op_B_YI, d_YI, d_B, /*zero_first=*/true, nBf);
        if (reduce_op_B_B0.active)
          reduce_forward_blas(reduce_op_B_B0, d_B0, d_B, /*zero_first=*/false, nBf);
        reduce_done = true;
      }
#endif
    if (!reduce_done)
    Kokkos::parallel_for("FusedReduceN_B",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nBout}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        for (int f = 0; f < nBf; f++) B_out(ii, k, f) = NNScalar(0.0);

        for (int c = 0; c < nc_YI; c++) {
          const int fi = ci_YI(c), fo = tsi_YI(c), tile = wlt_YI(c);
          const int nin = W_YI.extent(1);
          NNScalar val = NNScalar(0.0);
          for (int n = 0; n < nin; n++) val += W_YI(k, n, tile) * X_YI(ii, n, fi);
          B_out(ii, k, fo) += val * norm_YI;
        }

        for (int c = 0; c < nc_B0; c++) {
          const int fi = ci_B0(c), fo = tsi_B0(c), tile = wlt_B0(c);
          const int nin = W_B0.extent(1);
          NNScalar val = NNScalar(0.0);
          for (int n = 0; n < nin; n++) val += W_B0(k, n, tile) * X_B0(ii, n, fi);
          B_out(ii, k, fo) += val * norm_B0;
        }
      });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_reduce_I1()
{
  // ---- Stage 11: ReduceN I1 (equivariant, element-dependent) ----
  // The large 2L model has 1118 I1 connections. Keeping all connections serial
  // inside one (ii,k) thread under-exposes the A100; parallelize over
  // (ii,k,connection) and accumulate into I1 with atomics.
  {
    auto il = d_ilist; auto mp = d_map; auto tp = type;
    int co = chunk_offset, cs = chunk_size;
    int nI1 = I1_n_out;
    auto I1_out = d_I1;
    int nI1f = I1_n_funcs;

    Kokkos::parallel_for("ZeroI1",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nI1f, nI1}),
      KOKKOS_LAMBDA(const int ii, const int f, const int k) {
        I1_out(ii, f, k) = NNScalar(0.0);
      });

    #define REDUCE_I1_ATOMIC(SUFFIX, X_SRC) { \
      auto W = d_I1_reduce_##SUFFIX##_W; auto ci = d_I1_ci_##SUFFIX; \
      auto wlt = d_I1_wlt_##SUFFIX; auto tsi = d_I1_tsi_##SUFFIX; \
      NNScalar norm = I1_reduce_##SUFFIX##_norm; int nconn = I1_nc_##SUFFIX; \
      auto X = X_SRC; \
      Kokkos::parallel_for("ReduceN_I1_atomic_" #SUFFIX, \
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0,0,0}, {cs, nI1, nconn}), \
        KOKKOS_LAMBDA(const int ii, const int k, const int c) { \
          const int mu_i = mp(tp(il[ii + co])); \
          const int fi = ci(c), fo = tsi(c), tile = wlt(c); \
          const int nin = W.extent(2); \
          NNScalar val = NNScalar(0.0); \
          for (int n = 0; n < nin; n++) val += W(mu_i, k, n, tile) * X(ii, n, fi); \
          Kokkos::atomic_add(&I1_out(ii, fo, k), val * norm); \
        }); \
    }
    REDUCE_I1_ATOMIC(A, d_A)
    REDUCE_I1_ATOMIC(AA, d_AA)
    REDUCE_I1_ATOMIC(AAA, d_AAA)
    REDUCE_I1_ATOMIC(AAAA, d_AAAA)
    #undef REDUCE_I1_ATOMIC
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_reduce_B()
{
  // ---- B2-10: ReverseReduceN_B (equivariant, NOT elem-dep) → d_YI_adj, d_B0_adj ----
  {
    auto B_adj_v = d_B_adj;
    int cs = chunk_size, nBout = B_n_out;

    // Reverse B from YI
    {
      auto W = d_B_reduce_YI_W; auto ci = d_B_ci_YI; auto wlt = d_B_wlt_YI; auto tsi = d_B_tsi_YI;
      NNScalar norm = B_reduce_YI_norm; int nc = B_nc_YI; auto X_adj = d_YI_adj;
      Kokkos::parallel_for("RevReduceN_B_YI",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
        KOKKOS_LAMBDA(const int ii, const int c) {
          int fi = ci(c), fo = tsi(c), tile = wlt(c);
          int nin = W.extent(1);
          for (int n = 0; n < nin; n++) {
            NNScalar val = 0.0;
            for (int k = 0; k < nBout; k++) val += W(k, n, tile) * B_adj_v(ii, k, fo);
            Kokkos::atomic_add(&X_adj(ii, n, fi), val * norm);
          }
        });
    }
    // Reverse B from B0
    {
      auto W = d_B_reduce_B0_W; auto ci = d_B_ci_B0; auto wlt = d_B_wlt_B0; auto tsi = d_B_tsi_B0;
      NNScalar norm = B_reduce_B0_norm; int nc = B_nc_B0; auto X_adj = d_B0_adj;
      Kokkos::parallel_for("RevReduceN_B_B0",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nc}),
        KOKKOS_LAMBDA(const int ii, const int c) {
          int fi = ci(c), fo = tsi(c), tile = wlt(c);
          int nin = W.extent(1);
          for (int n = 0; n < nin; n++) {
            NNScalar val = 0.0;
            for (int k = 0; k < nBout; k++) val += W(k, n, tile) * B_adj_v(ii, k, fo);
            Kokkos::atomic_add(&X_adj(ii, n, fi), val * norm);
          }
        });
    }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_reduce_I1()
{
  // ---- B1-4: ReverseReduceN_I1 (equivariant, element-dependent) → d_A_adj +=, etc ----
  // I1[i, fo, k] = Σ_c W[mu_i, k, n, tile_c] * X[i, n, fi_c] * norm  where fo = tsi(c)
  // d_X[i, n, fi_c] += Σ_k W[mu_i, k, n, tile_c] * d_I1[i, fo_c, k] * norm
  {
    auto I1_adj_v = d_I1_adj;
    auto il = d_ilist; auto tp = type; auto mp = d_map;
    int co = chunk_offset, cs = chunk_size, nI1out = I1_n_out;

    #define REV_REDUCE_I1(SUFFIX, ADJ_TARGET) { \
      auto W = d_I1_reduce_##SUFFIX##_W; auto ci = d_I1_ci_##SUFFIX; \
      auto wlt = d_I1_wlt_##SUFFIX; auto tsi = d_I1_tsi_##SUFFIX; \
      NNScalar norm = I1_reduce_##SUFFIX##_norm; \
      int nconn = (int)ci.extent(0); auto X_adj = ADJ_TARGET; \
      Kokkos::parallel_for("RevReduceN_I1_" #SUFFIX, \
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nconn}), \
        KOKKOS_LAMBDA(const int ii, const int c) { \
          const int mu_i = mp(tp(il[ii + co])); \
          const int fi = ci(c), fo = tsi(c), tile = wlt(c); \
          const int nin = W.extent(2); \
          for (int n = 0; n < nin; n++) { \
            NNScalar val = 0.0; \
            for (int k = 0; k < nI1out; k++) val += W(mu_i, k, n, tile) * I1_adj_v(ii, fo, k); \
            Kokkos::atomic_add(&X_adj(ii, n, fi), val * norm); \
          } \
        }); \
    }
    REV_REDUCE_I1(A, d_A_adj)
    REV_REDUCE_I1(AA, d_AA_adj)
    REV_REDUCE_I1(AAA, d_AAA_adj)
    REV_REDUCE_I1(AAAA, d_AAAA_adj)
    #undef REV_REDUCE_I1
  }
}

// rho ReduceN stages (virtual hooks; Grace2LFusedKK fuses them with I1)

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_zero_L2_adj()
{

      auto B_adj = d_B_adj; auto BB_adj = d_BB_adj;
      auto BBB_adj = d_BBB_adj; auto BBBB_adj = d_BBBB_adj;
      auto YI_adj = d_YI_adj; auto B0_adj = d_B0_adj;
      const int cs = chunk_size;
      const int nB = B_adj.extent(1) * B_adj.extent(2);
      const int nBB = BB_adj.extent(1) * BB_adj.extent(2);
      const int nBBB = BBB_adj.extent(1) * BBB_adj.extent(2);
      const int nBBBB = BBBB_adj.extent(1) * BBBB_adj.extent(2);
      const int nYI = YI_adj.extent(1) * YI_adj.extent(2);
      const int nB0 = B0_adj.extent(1) * B0_adj.extent(2);
      const int nmax0 = nB > nBB ? nB : nBB;
      const int nmax1 = nBBB > nBBBB ? nBBB : nBBBB;
      const int nmax2 = nYI > nB0 ? nYI : nB0;
      const int nmax3 = nmax0 > nmax1 ? nmax0 : nmax1;
      const int nmax = nmax3 > nmax2 ? nmax3 : nmax2;
      Kokkos::parallel_for("ZeroL2Adjoints",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nmax}),
        KOKKOS_LAMBDA(const int ii, const int idx) {
          if (idx < nB) B_adj(ii, idx / B_adj.extent(2), idx % B_adj.extent(2)) = NNScalar(0.0);
          if (idx < nBB) BB_adj(ii, idx / BB_adj.extent(2), idx % BB_adj.extent(2)) = NNScalar(0.0);
          if (idx < nBBB) BBB_adj(ii, idx / BBB_adj.extent(2), idx % BBB_adj.extent(2)) = NNScalar(0.0);
          if (idx < nBBBB) BBBB_adj(ii, idx / BBBB_adj.extent(2), idx % BBBB_adj.extent(2)) = NNScalar(0.0);
          if (idx < nYI) YI_adj(ii, idx / YI_adj.extent(2), idx % YI_adj.extent(2)) = NNScalar(0.0);
          if (idx < nB0) B0_adj(ii, idx / B0_adj.extent(2), idx % B0_adj.extent(2)) = NNScalar(0.0);
        });
    }

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_reduce_I2()
{

      auto I2_out = d_I2;
      auto il = d_ilist; auto mp = d_map; auto tp = type;
      int co = chunk_offset, cs = chunk_size, nI2 = I2_n_out;

      // Fused ReduceN_I2: merge 4 separate kernel launches into 1
      {
        // Source B
        auto W_B = d_I2_reduce_B_W; auto ci_B = d_I2_ci_B; auto wlt_B = d_I2_wlt_B;
        NNScalar norm_B = I2_reduce_B_norm; int nc_B = (int)ci_B.extent(0); auto X_B = d_B;
        // Source BB
        auto W_BB = d_I2_reduce_BB_W; auto ci_BB = d_I2_ci_BB; auto wlt_BB = d_I2_wlt_BB;
        NNScalar norm_BB = I2_reduce_BB_norm; int nc_BB = (int)ci_BB.extent(0); auto X_BB = d_BB;
        // Source BBB
        auto W_BBB = d_I2_reduce_BBB_W; auto ci_BBB = d_I2_ci_BBB; auto wlt_BBB = d_I2_wlt_BBB;
        NNScalar norm_BBB = I2_reduce_BBB_norm; int nc_BBB = (int)ci_BBB.extent(0); auto X_BBB = d_BBB;
        // Source BBBB
        auto W_BBBB = d_I2_reduce_BBBB_W; auto ci_BBBB = d_I2_ci_BBBB; auto wlt_BBBB = d_I2_wlt_BBBB;
        NNScalar norm_BBBB = I2_reduce_BBBB_norm; int nc_BBBB = (int)ci_BBBB.extent(0); auto X_BBBB = d_BBBB;

        Kokkos::parallel_for("FusedReduceN_I2",
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nI2}),
          KOKKOS_LAMBDA(const int ii, const int k) {
            const int mu_i = mp(tp(il[ii + co]));
            NNScalar val = NNScalar(0.0);
            // From B
            { NNScalar s = NNScalar(0.0);
              for (int c = 0; c < nc_B; c++) {
                const int fi = ci_B(c), tile = wlt_B(c), nin = W_B.extent(2);
                for (int n = 0; n < nin; n++) s += W_B(mu_i, k, n, tile) * X_B(ii, n, fi);
              }
              val += s * norm_B; }
            // From BB
            { NNScalar s = NNScalar(0.0);
              for (int c = 0; c < nc_BB; c++) {
                const int fi = ci_BB(c), tile = wlt_BB(c), nin = W_BB.extent(2);
                for (int n = 0; n < nin; n++) s += W_BB(mu_i, k, n, tile) * X_BB(ii, n, fi);
              }
              val += s * norm_BB; }
            // From BBB
            { NNScalar s = NNScalar(0.0);
              for (int c = 0; c < nc_BBB; c++) {
                const int fi = ci_BBB(c), tile = wlt_BBB(c), nin = W_BBB.extent(2);
                for (int n = 0; n < nin; n++) s += W_BBB(mu_i, k, n, tile) * X_BBB(ii, n, fi);
              }
              val += s * norm_BBB; }
            // From BBBB
            { NNScalar s = NNScalar(0.0);
              for (int c = 0; c < nc_BBBB; c++) {
                const int fi = ci_BBBB(c), tile = wlt_BBBB(c), nin = W_BBBB.extent(2);
                for (int n = 0; n < nin; n++) s += W_BBBB(mu_i, k, n, tile) * X_BBBB(ii, n, fi);
              }
              val += s * norm_BBBB; }
            I2_out(ii, k) = val;
          });
      }
    }

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_reduce_I2()
{

      auto I2_adj_v = d_I2_adj;
      auto il = d_ilist; auto tp = type; auto mp = d_map;
      int co = chunk_offset, cs = chunk_size, n_I2 = I2_n_out;

      #define REV_REDUCE_I2(SUFFIX, ADJ_TARGET) { \
        auto W = d_I2_reduce_##SUFFIX##_W; auto ci = d_I2_ci_##SUFFIX; \
        auto wlt = d_I2_wlt_##SUFFIX; \
        NNScalar norm = I2_reduce_##SUFFIX##_norm; \
        int nconn = (int)ci.extent(0); auto X_adj = ADJ_TARGET; \
        int nin = (int)W.extent(2); \
        Kokkos::parallel_for("RevReduceN_I2_" #SUFFIX, \
          Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nin}), \
          KOKKOS_LAMBDA(const int ii, const int n) { \
            const int mu_i = mp(tp(il[ii + co])); \
            for (int c = 0; c < nconn; c++) { \
              const int fi = ci(c), tile = wlt(c); \
              NNScalar val = 0.0; \
              for (int k = 0; k < n_I2; k++) val += I2_adj_v(ii, k) * W(mu_i, k, n, tile); \
              X_adj(ii, n, fi) += val * norm; \
            } \
          }); \
      }
      REV_REDUCE_I2(B, d_B_adj)
      REV_REDUCE_I2(BB, d_BB_adj)
      REV_REDUCE_I2(BBB, d_BBB_adj)
      REV_REDUCE_I2(BBBB, d_BBBB_adj)
      #undef REV_REDUCE_I2
    }

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_zero_L1_adj()
{
  Kokkos::deep_copy(d_A_adj, 0.0);
  Kokkos::deep_copy(d_AA_adj, 0.0);
  Kokkos::deep_copy(d_AAA_adj, 0.0);
  Kokkos::deep_copy(d_AAAA_adj, 0.0);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_rev_reduce_rho()
{
  // ---- B1-2: ReverseReduceN_rho → d_A_adj, d_AA_adj, d_AAA_adj, d_AAAA_adj ----
  {
    auto rho_adj_v = d_rho_adj;
    auto il = d_ilist; auto tp = type; auto mp = d_map;
    int co = chunk_offset, cs = chunk_size, n_rho = rho_n_out;

    #define REV_REDUCE_RHO(SUFFIX, ADJ_TARGET) { \
      auto W = d_rho_reduce_##SUFFIX##_W; auto ci = d_rho_ci_##SUFFIX; \
      auto wlt = d_rho_wlt_##SUFFIX; \
      NNScalar norm = rho_reduce_##SUFFIX##_norm; \
      int nconn = (int)ci.extent(0); auto X_adj = ADJ_TARGET; \
      Kokkos::parallel_for("RevReduceN_rho_" #SUFFIX, \
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nconn}), \
        KOKKOS_LAMBDA(const int ii, const int c) { \
          const int mu_i = mp(tp(il[ii + co])); \
          const int fi = ci(c), tile = wlt(c); \
          const int nin = W.extent(2); \
          for (int n = 0; n < nin; n++) { \
            NNScalar val = 0.0; \
            for (int k = 0; k < n_rho; k++) val += rho_adj_v(ii, k) * W(mu_i, k, n, tile); \
            Kokkos::atomic_add(&X_adj(ii, n, fi), val * norm); \
          } \
        }); \
    }
    REV_REDUCE_RHO(A, d_A_adj)
    REV_REDUCE_RHO(AA, d_AA_adj)
    REV_REDUCE_RHO(AAA, d_AAA_adj)
    REV_REDUCE_RHO(AAAA, d_AAAA_adj)
    #undef REV_REDUCE_RHO
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::stage_reduce_rho()
{
  auto rho_out = d_rho;
  auto il = d_ilist; auto mp = d_map; auto tp = type;
  int co = chunk_offset, cs = chunk_size, nrho = rho_n_out;
  (void) nrho;
  // Fused ReduceN_rho: merge 4 separate kernel launches into 1
  // All 4 sources (A, AA, AAA, AAAA) accumulate in a single register per (ii,k)
  {
    // Source A
    auto W_A = d_rho_reduce_A_W; auto ci_A = d_rho_ci_A; auto wlt_A = d_rho_wlt_A;
    NNScalar norm_A = rho_reduce_A_norm; int nc_A = (int)ci_A.extent(0); auto X_A = d_A;
    // Source AA
    auto W_AA = d_rho_reduce_AA_W; auto ci_AA = d_rho_ci_AA; auto wlt_AA = d_rho_wlt_AA;
    NNScalar norm_AA = rho_reduce_AA_norm; int nc_AA = (int)ci_AA.extent(0); auto X_AA = d_AA;
    // Source AAA
    auto W_AAA = d_rho_reduce_AAA_W; auto ci_AAA = d_rho_ci_AAA; auto wlt_AAA = d_rho_wlt_AAA;
    NNScalar norm_AAA = rho_reduce_AAA_norm; int nc_AAA = (int)ci_AAA.extent(0); auto X_AAA = d_AAA;
    // Source AAAA
    auto W_AAAA = d_rho_reduce_AAAA_W; auto ci_AAAA = d_rho_ci_AAAA; auto wlt_AAAA = d_rho_wlt_AAAA;
    NNScalar norm_AAAA = rho_reduce_AAAA_norm; int nc_AAAA = (int)ci_AAAA.extent(0); auto X_AAAA = d_AAAA;

    Kokkos::parallel_for("FusedReduceN_rho",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0,0}, {cs, nrho}),
      KOKKOS_LAMBDA(const int ii, const int k) {
        const int mu_i = mp(tp(il[ii + co]));
        NNScalar val = NNScalar(0.0);
        // From A
        { NNScalar s = NNScalar(0.0);
          for (int c = 0; c < nc_A; c++) {
            const int fi = ci_A(c), tile = wlt_A(c), nin = W_A.extent(2);
            for (int n = 0; n < nin; n++) s += W_A(mu_i, k, n, tile) * X_A(ii, n, fi);
          }
          val += s * norm_A; }
        // From AA
        { NNScalar s = NNScalar(0.0);
          for (int c = 0; c < nc_AA; c++) {
            const int fi = ci_AA(c), tile = wlt_AA(c), nin = W_AA.extent(2);
            for (int n = 0; n < nin; n++) s += W_AA(mu_i, k, n, tile) * X_AA(ii, n, fi);
          }
          val += s * norm_AA; }
        // From AAA
        { NNScalar s = NNScalar(0.0);
          for (int c = 0; c < nc_AAA; c++) {
            const int fi = ci_AAA(c), tile = wlt_AAA(c), nin = W_AAA.extent(2);
            for (int n = 0; n < nin; n++) s += W_AAA(mu_i, k, n, tile) * X_AAA(ii, n, fi);
          }
          val += s * norm_AAA; }
        // From AAAA
        { NNScalar s = NNScalar(0.0);
          for (int c = 0; c < nc_AAAA; c++) {
            const int fi = ci_AAAA(c), tile = wlt_AAAA(c), nin = W_AAAA.extent(2);
            for (int n = 0; n < nin; n++) s += W_AAAA(mu_i, k, n, tile) * X_AAAA(ii, n, fi);
          }
          val += s * norm_AAAA; }
        rho_out(ii, k) = val;
      });
  }
}


// ======================================================================
// Engine API (replaces the LAMMPS constructor, coeff, init_style, init_one and the comm callbacks of
// Grace2LKK)
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::Grace2LKK()
{
  host_flag = Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename DeviceType::memory_space>::accessible ? 1 : 0;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::~Grace2LKK()
{
  // the BLAS handle is released by its shared owner (blas_owner) with the last engine copy
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::set_model(std::shared_ptr<GRACE2LModel> model,
                                                                       const std::vector<int> &type2mu)
{
  if (!model) GRACE_KK_ERROR("GRACE-2L/KK: set_model() without a model");
  grace_model = std::move(model);
  realloc_all = true;    // the per-atom buffers have the previous model's dims
  setup_done = false;
#ifdef KOKKOS_ENABLE_CUDA
  // the cuBLAS UQ gather maps and Bmat scratch belong to the previous model's basis (rebuilt when empty)
  d_uq_mapL1_view = t_int_1d();
  d_uq_mapL2_view = t_int_1d();
  uq_subN = 0;
#endif
  // A UQ head the build cannot use (schema other than 6, inconsistent arrays) leaves the engine usable without UQ
  has_uq = false;
  if (grace_model->has_uq) {
    try {
      grace_model->validate_uq();
      has_uq = true;
    } catch (const grace::Error &e) {
      log.info(fmt("[GRACE-2L/KK] UQ head not usable, running without UQ: {}\n", e.what()));
    }
  }
  uq_Kmax = grace_model->uq_max_clusters;
  uq_D = grace_model->uq_feature_dim;
  uq_rp_dim = grace_model->uq_rp_dim;
  uq_n_density = grace_model->uq_n_density;
  uq_normalize = grace_model->uq_normalize;
  uq_density_scale = grace_model->uq_density_scale;
  uq_d_basis = grace_model->uq_d_basis;
  // the message dims here, not in setup(): callers size their comm buffers with comm_width() before setup()
  I_n_out = grace_model->I_n_out;
  I_n_funcs = grace_model->I_n_funcs;
  if (has_uq)
    log.info(fmt("[GRACE-2L/KK] UQ artifacts loaded (uqv6 basis-RP): Kmax={}, D={}, rp_dim={}, "
                 "n_density={}, D_basis={}\n", uq_Kmax, uq_D, uq_rp_dim, uq_n_density, uq_d_basis));
  log.info(fmt("[GRACE-2L/KK] n_elements={}, lmax={}, nradbase={}, L1_nradmax={}, L2_nradmax={}, rcut={}\n",
               grace_model->n_elements, grace_model->lmax, grace_model->nradbase, grace_model->L1_nradmax,
               grace_model->L2_nradmax, grace_model->rcut));

  set_type_map(type2mu);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::set_type_map(const std::vector<int> &type2mu)
{
  // atom type -> model element, and the per-type-pair cutoffs (as LAMMPS init_one)
  const int n = (int) type2mu.size();
  check_type2mu(type2mu, n - 1, grace_model->n_elements, "GRACE-2L/KK");
  h_type2model = type2mu;
  grace::kk::MemKK::realloc_kokkos(d_map, "grace_2l:map", n);
  auto h_map = Kokkos::create_mirror_view(d_map);
  for (int t = 0; t < n; t++) h_map(t) = type2mu[t];
  Kokkos::deep_copy(d_map, h_map);
  grace::kk::MemKK::realloc_kokkos(k_cutsq, "grace_2l:cutsq", n, n);
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
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::set_types(const std::vector<int> &type2mu)
{
  if (!setup_done) GRACE_KK_ERROR("GRACE-2L/KK: set_types() before setup()");
  set_type_map(type2mu);
  retype();
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
double Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::cutoff(int mu_i, int mu_j) const
{
  return pair_cutoff(*grace_model, mu_i, mu_j);
}

// ---- message rows of the layer-1 exchange (LAMMPS pack/unpack callbacks of the pair style) ----

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
int Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::pack_forward(int n, const typename Traits::t_int_1d &sendlist,
                                                                        const typename Traits::t_double_1d &buf)
{
  d_sendlist = sendlist;
  v_buf = buf;
  if constexpr (is_hip_v<DeviceType>) comm_rows_hip(0, n);
  else Kokkos::parallel_for("PackForwardComm", Kokkos::RangePolicy<DeviceType, TagPackForwardComm>(0, n), *this);
  return comm_values(n, comm_width());
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::unpack_forward(int n, int first,
                                                                           const typename Traits::t_double_1d &buf)
{
  comm_first = first;
  v_buf = buf;
  if constexpr (is_hip_v<DeviceType>) comm_rows_hip(1, n);
  else Kokkos::parallel_for("UnpackForwardComm", Kokkos::RangePolicy<DeviceType, TagUnpackForwardComm>(0, n), *this);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
int Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::pack_reverse(int n, int first,
                                                                        const typename Traits::t_double_1d &buf)
{
  comm_first = first;
  v_buf = buf;
  if constexpr (is_hip_v<DeviceType>) comm_rows_hip(2, n);
  else Kokkos::parallel_for("PackReverseComm", Kokkos::RangePolicy<DeviceType, TagPackReverseComm>(0, n), *this);
  return comm_values(n, comm_width());
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::unpack_reverse(int n, const typename Traits::t_int_1d &sendlist,
                                                                           const typename Traits::t_double_1d &buf)
{
  d_sendlist = sendlist;
  v_buf = buf;
  if constexpr (is_hip_v<DeviceType>) comm_rows_hip(3, n);
  else Kokkos::parallel_for("UnpackReverseComm", Kokkos::RangePolicy<DeviceType, TagUnpackReverseComm>(0, n), *this);
}

// HIP: buffer row i (comm_width() doubles, element r = f * I_n_out + k) <-> the message (or its gradient) of atom j(i);
// a thread per atom (the Tag*Comm operators) wrote / read its row with a stride of comm_width() doubles between the
// lanes. Here 64 atoms x 64 row elements go through shared memory: atom-contiguous on the view side, row-contiguous
// on the buffer side. mode 0: pack forward (I of sendlist), 1: unpack forward (I of ghosts first + i), 2: pack
// reverse (dE/dI of the ghosts), 3: unpack reverse (added to dE/dI of sendlist).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::comm_rows_hip(int mode, int n)
{
  const int W = comm_width(), no = I_n_out, nti = (n + 63) / 64, ntr = (W + 63) / 64;
  if (nti == 0) return;
  auto I = d_I_global; auto gI = d_grad_I_global; auto buf = v_buf; auto sl = d_sendlist;
  const int first = comm_first;
  const bool to_buf = mode == 0 || mode == 2;
  using pol_t = Kokkos::TeamPolicy<DeviceType>;
  pol_t pol(nti * ntr, 256);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(64 * 65 * sizeof(NNScalar)));
  Kokkos::parallel_for("CommRows", pol, KOKKOS_LAMBDA(const typename pol_t::member_type &t) {
    const int tr = t.league_rank() / nti, i0 = (t.league_rank() - tr * nti) * 64, r0 = tr * 64;
    NNScalar *tile = (NNScalar *) t.team_shmem().get_shmem(64 * 65 * sizeof(NNScalar));
    for (int q = t.team_rank(); q < 64 * 64; q += 256) {    // coalesced source reads
      const int a = to_buf ? (q & 63) : (q >> 6), r = to_buf ? (q >> 6) : (q & 63);
      const int i = i0 + a, rr = r0 + r;
      if (i < n && rr < W) {
        const int f = rr / no, k = rr - f * no;
        tile[a * 65 + r] = mode == 0 ? I(sl(i), f, k) : mode == 2 ? gI(first + i, f, k)
                                                       : (NNScalar) buf((size_t) i * W + rr);
      }
    }
    t.team_barrier();
    for (int q = t.team_rank(); q < 64 * 64; q += 256) {    // coalesced destination writes
      const int a = to_buf ? (q >> 6) : (q & 63), r = to_buf ? (q & 63) : (q >> 6);
      const int i = i0 + a, rr = r0 + r;
      if (i < n && rr < W) {
        const int f = rr / no, k = rr - f * no;
        const NNScalar v = tile[a * 65 + r];
        if (to_buf) buf((size_t) i * W + rr) = v;
        else if (mode == 1) I(first + i, f, k) = v;
        else Kokkos::atomic_add(&gI(sl(i), f, k), v);
      }
    }
  });
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::copy_rows_to_host(bool reverse)
{
  grace::kk::MemKK::ensure_mirror(h_I_global, d_I_global);
  grace::kk::MemKK::ensure_mirror(h_grad_I_global, d_grad_I_global);
  if (reverse) Kokkos::deep_copy(h_grad_I_global.base(), d_grad_I_global.base());
  else Kokkos::deep_copy(h_I_global.base(), d_I_global.base());
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::copy_rows_from_host(bool reverse)
{
  if (reverse) Kokkos::deep_copy(d_grad_I_global.base(), h_grad_I_global.base());
  else Kokkos::deep_copy(d_I_global.base(), h_I_global.base());
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
int Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::pack_forward_host(int n, const int *list, double *buf)
{
  comm_values(n, comm_width());    // the int count LAMMPS gets
  int m = 0;
  const int nf = I_n_funcs, no = I_n_out;
  const int nmax = (int)h_I_global.extent(0);
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    if (j >= 0 && j < nmax) {
      for (int f = 0; f < nf; f++)
        for (int k = 0; k < no; k++)
          buf[m++] = h_I_global(j, f, k);
    } else {
      for (int f = 0; f < nf; f++)
        for (int k = 0; k < no; k++)
          buf[m++] = 0.0;
    }
  }
  return m;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::unpack_forward_host(int n, int first, const double *buf)
{
  int m = 0;
  const int nf = I_n_funcs, no = I_n_out;
  const int nmax = (int)h_I_global.extent(0);
  for (int i = 0; i < n; i++) {
    const int j = first + i;
    if (j >= 0 && j < nmax) {
      for (int f = 0; f < nf; f++)
        for (int k = 0; k < no; k++)
          h_I_global(j, f, k) = buf[m++];
    } else {
      m += nf * no;  // skip
    }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
int Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::pack_reverse_host(int n, int first, double *buf)
{
  comm_values(n, comm_width());    // the int count LAMMPS gets
  int m = 0;
  const int nf = I_n_funcs, no = I_n_out;
  for (int i = 0; i < n; i++) {
    const int j = first + i;
    for (int f = 0; f < nf; f++)
      for (int k = 0; k < no; k++)
        buf[m++] = h_grad_I_global(j, f, k);
  }
  return m;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace2LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::unpack_reverse_host(int n, const int *list, const double *buf)
{
  int m = 0;
  const int nf = I_n_funcs, no = I_n_out;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    for (int f = 0; f < nf; f++)
      for (int k = 0; k < no; k++)
        h_grad_I_global(j, f, k) += buf[m++];
  }
}

}    // namespace kk
}    // namespace grace

// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Part of grace::kk::Grace3LKK (grace_3l_kk.h), included by grace_3l_kk_impl.h:
// radial MLP (cuBLAS layers, derivative pass) and the per-pair cubic-Hermite radial tables.
// Moved from the LAMMPS style grace/3l/kk (lammps-ace 5f5ab57aa7, pair_grace_3l_kokkos_radial.cpp).
#pragma once

#include "grace/kk/grace_3l_kk.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace grace {
namespace kk {
using namespace sph_const;

// ======================================================================
// Kernel: TagComputeMLPRadial (Task 2.4 A1 / Task 2.8 A2) — generic radial
// MLP, one bond per thread, for the SPBF layer selected by the member
// radial_mlp_spbf (0=A1, idx_spbf_A2=A2). Reproduces
// numpy_forward_3l.radial_mlp EXACTLY:
//   x = concat([basis32, z_table[mu_i], z_table[mu_j]])         (width 138)
//   x = x @ (W_i*norm_i); hidden layers (i<n_layers-1) add b_i*norm_i, silu;
//   output layer (i==n_layers-1) is linear, no bias.
// Output reshaped [n_rad_max, lmax+1] (row-major n*[lmax+1]+l) and stored
// RAW (pre l_tile-gather) in d_R1_nl; the A*-assembly kernel gathers by
// l_tile when it forms a_nl. All arithmetic in NNScalar (fp32 Mixed
// default); d_radial_basis (GeomScalar) is cast down on read, d_chem_embed
// is already NNScalar. Team-policy (ii,jj) decode copied from
// TagComputeRadialBasis (Task 2.3). The A1/A2 MLPs share this code path but
// use their own weights (d_spbf[radial_mlp_spbf].mlp_*) and n_rad_max; the
// chem-embedding concat input is spbf-independent.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeMLPRadial,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadial>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;
  const int jj = slot.neigh;
  if (jj >= d_ncount(ii)) return;

  const DeviceSPBF &sp = d_spbf[radial_mlp_spbf];   // A1 or A2 (set on host)
  const int n_layers = sp.mlp_n_layers;
  int mlp_dims[GRACE3L_MAX_MLP_LAYERS + 1];
  NNScalar mlp_norms[GRACE3L_MAX_MLP_LAYERS];
  for (int i = 0; i <= n_layers; i++) mlp_dims[i] = sp.mlp_dims(i);
  for (int i = 0; i < n_layers; i++) mlp_norms[i] = sp.mlp_norms(i);

  NNScalar buf_a[GRACE3L_MAX_MLP_DIM], buf_b[GRACE3L_MAX_MLP_DIM];
  NNScalar* h_cur = buf_a; NNScalar* h_nxt = buf_b;

  // Input: concat([basis32, z_i32, z_j32]) — order pinned by the oracle.
  const int mu_i = d_mu_i(ii);
  const int mu_j = d_mu_j(ii, jj);
  int k = 0;
  for (int b = 0; b < nradbase; b++) h_cur[k++] = (NNScalar) d_radial_basis(ii, jj, b);
  for (int e = 0; e < embedding_size; e++) h_cur[k++] = d_chem_embed(mu_i, e);
  for (int e = 0; e < embedding_size; e++) h_cur[k++] = d_chem_embed(mu_j, e);

  for (int layer = 0; layer < n_layers - 1; layer++) {
    const int nin = mlp_dims[layer], nout = mlp_dims[layer + 1];
    const NNScalar norm = mlp_norms[layer];
    for (int j = 0; j < nout; j++) {
      NNScalar s = 0.0;
      for (int kk = 0; kk < nin; kk++)
        s += sp.mlp_W(layer, kk, j) * h_cur[kk];
      s = s * norm + sp.mlp_b(layer, j) * norm;
      h_nxt[j] = silu(s);
    }
    NNScalar* tmp = h_cur; h_cur = h_nxt; h_nxt = tmp;
  }

  // Output layer: linear, no bias.
  const int last_layer = n_layers - 1;
  const int n_last_hidden = mlp_dims[last_layer];
  const NNScalar out_norm = mlp_norms[last_layer];
  const int lmax1 = lmax + 1;
  for (int n = 0; n < sp.n_rad_max; n++) {
    for (int l = 0; l < lmax1; l++) {
      const int j = n * lmax1 + l;
      NNScalar sum = 0.0;
      for (int kk = 0; kk < n_last_hidden; kk++)
        sum += sp.mlp_W(last_layer, kk, j) * h_cur[kk];
      d_R1_nl(ii, jj, n, l) = sum * out_norm;
    }
  }
}

// ======================================================================
// Task 3.3: geometry backward (per-bond forces).
// ======================================================================

// ---- TagComputeMLPRadialDeriv: radial MLP forward + its r-derivative, one bond
// per thread, for the layer selected by radial_mlp_spbf. Reproduces
// TagComputeMLPRadial's value path EXACTLY (bit-identical R via the same raw
// Chebyshev basis read from d_radial_basis) and forward-propagates the r-derivative
// (only the Chebyshev inputs depend on r). Writes d_R1_nl and d_DR1_nl (=dR/dr); the envelope derivative
// d_denv comes from TagComputeRadialBasis. silu'(s) = sig*(1+s*(1-sig)).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeMLPRadialDeriv,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadialDeriv>::member_type& team) const
{
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;
  const int jj = slot.neigh;
  if (jj >= d_ncount(ii)) return;

  const DeviceSPBF &sp = d_spbf[radial_mlp_spbf];
  const int n_layers = sp.mlp_n_layers;
  int mlp_dims[GRACE3L_MAX_MLP_LAYERS + 1];
  NNScalar mlp_norms[GRACE3L_MAX_MLP_LAYERS];
  for (int i = 0; i <= n_layers; i++) mlp_dims[i] = sp.mlp_dims(i);
  for (int i = 0; i < n_layers; i++) mlp_norms[i] = sp.mlp_norms(i);

  // Geometry: bond length + per-element cutoff (== TagComputeRadialBasis).
  const int mu_i = d_mu_i(ii);
  const int mu_j = d_mu_j(ii, jj);
  const BondScalar r = d_rnorms(ii, jj);
  const BondScalar rcut_ij = d_bond_cutoff(mu_i, mu_j);
  const BondScalar x_norm = r / rcut_ij;
  const BondScalar x_cheb = BondScalar(2.0) * x_norm - BondScalar(1.0);
  const BondScalar dxcheb_dr = BondScalar(2.0) / rcut_ij;   // d x_cheb / dr

  // d_denv: TagComputeRadialBasis (once per bond, fp64)

  NNScalar buf_a[GRACE3L_MAX_MLP_DIM], buf_b[GRACE3L_MAX_MLP_DIM];
  NNScalar dbuf_a[GRACE3L_MAX_MLP_DIM], dbuf_b[GRACE3L_MAX_MLP_DIM];
  NNScalar* h_cur = buf_a;  NNScalar* h_nxt = buf_b;
  NNScalar* dh_cur = dbuf_a; NNScalar* dh_nxt = dbuf_b;

  // Input value path: read the raw Chebyshev basis (bit-identical to the forward).
  for (int b = 0; b < nradbase; b++) h_cur[b] = (NNScalar) d_radial_basis(ii, jj, b);
  // Input derivative path for the Chebyshev basis: basis[b] = T_{b+1}(x_cheb),
  // dT_1/dx=1, dT_{k+1}/dx = 2*T_k + 2*x*dT_k - dT_{k-1}; * dxcheb_dr for d/dr.
  {
    BondScalar dT_prev = BondScalar(0.0);   // dT_0/dx
    BondScalar dT_curr = BondScalar(1.0);   // dT_1/dx
    dh_cur[0] = (NNScalar)(dT_curr * dxcheb_dr);
    for (int kk = 1; kk < nradbase; kk++) {
      const BondScalar T_kk = (BondScalar) d_radial_basis(ii, jj, kk - 1);  // T_kk
      const BondScalar dT_next = BondScalar(2.0) * T_kk
                               + BondScalar(2.0) * x_cheb * dT_curr - dT_prev;
      dh_cur[kk] = (NNScalar)(dT_next * dxcheb_dr);
      dT_prev = dT_curr; dT_curr = dT_next;
    }
  }
  int k = nradbase;
  for (int e = 0; e < embedding_size; e++) { h_cur[k] = d_chem_embed(mu_i, e); dh_cur[k] = NNScalar(0.0); k++; }
  for (int e = 0; e < embedding_size; e++) { h_cur[k] = d_chem_embed(mu_j, e); dh_cur[k] = NNScalar(0.0); k++; }

  for (int layer = 0; layer < n_layers - 1; layer++) {
    const int nin = mlp_dims[layer], nout = mlp_dims[layer + 1];
    const NNScalar norm = mlp_norms[layer];
    for (int j = 0; j < nout; j++) {
      NNScalar s = 0.0, ds = 0.0;
      for (int kk = 0; kk < nin; kk++) {
        const NNScalar Wv = sp.mlp_W(layer, kk, j);
        s  += Wv * h_cur[kk];
        ds += Wv * dh_cur[kk];
      }
      s  = s * norm + sp.mlp_b(layer, j) * norm;
      ds = ds * norm;
      const NNScalar sig = NNScalar(1.0) / (NNScalar(1.0) + Kokkos::exp(-s));
      h_nxt[j]  = s * sig;                                          // silu(s)
      dh_nxt[j] = sig * (NNScalar(1.0) + s * (NNScalar(1.0) - sig)) * ds;  // silu'(s)*ds
    }
    NNScalar* t; t = h_cur; h_cur = h_nxt; h_nxt = t;
    t = dh_cur; dh_cur = dh_nxt; dh_nxt = t;
  }

  const int last_layer = n_layers - 1;
  const int n_last_hidden = mlp_dims[last_layer];
  const NNScalar out_norm = mlp_norms[last_layer];
  const int lmax1 = lmax + 1;
  for (int n = 0; n < sp.n_rad_max; n++) {
    for (int l = 0; l < lmax1; l++) {
      const int j = n * lmax1 + l;
      NNScalar R = 0.0, DR = 0.0;
      for (int kk = 0; kk < n_last_hidden; kk++) {
        const NNScalar Wv = sp.mlp_W(last_layer, kk, j);
        R  += Wv * h_cur[kk];
        DR += Wv * dh_cur[kk];
      }
      d_R1_nl(ii, jj, n, l)  = R * out_norm;
      d_DR1_nl(ii, jj, n, l) = DR * out_norm;
    }
  }
}

#ifdef KOKKOS_ENABLE_CUDA
#endif
#ifdef KOKKOS_ENABLE_CUDA
// ======================================================================
// Opt 15 (value path): support kernels for the batched cuBLAS radial MLP.
// TagMLPAssembleVal builds the tightly-packed row-major input matrix
// d_mlp_X[M x n_in0], one bond per thread. Bond b decodes to
// (ii = b / bond_stride, jj = b % bond_stride) with bond_stride =
// d_radial_basis.extent(1) (== allocated maxneigh). Padding rows (jj >=
// ncount) are zeroed so the batched SGEMM stays finite; their outputs are
// ignored downstream (consumers loop jj < ncount). Input order is pinned to
// the oracle: [Chebyshev basis, chem_embed(mu_i), chem_embed(mu_j)].
// TagMLPActVal applies the per-layer affine (s = raw*norm + bias*norm) and
// silu in place on d_mlp_h{0,1} (raw = X*W from the GEMM, no norm folded).
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPAssembleVal,
    const int b) const
{
  const int bond_stride = (int) d_radial_basis.extent(1);
  const int ii = b / bond_stride;
  const int jj = b - ii * bond_stride;
  const int n_in0 = nradbase + 2 * embedding_size;
  if (jj >= d_ncount(ii)) {
    for (int c = 0; c < n_in0; c++) d_mlp_X(b, c) = NNScalar(0.0);
    return;
  }
  const int mu_i = d_mu_i(ii);
  const int mu_j = d_mu_j(ii, jj);
  int k = 0;
  for (int c = 0; c < nradbase; c++)     d_mlp_X(b, k++) = (NNScalar) d_radial_basis(ii, jj, c);
  for (int e = 0; e < embedding_size; e++) d_mlp_X(b, k++) = d_chem_embed(mu_i, e);
  for (int e = 0; e < embedding_size; e++) d_mlp_X(b, k++) = d_chem_embed(mu_j, e);
}

#endif
#ifdef KOKKOS_ENABLE_CUDA
// TagMLPAssembleDeriv (Opt 15b): value input d_mlp_X[b, :n_in0] (identical to
// AssembleVal) + deriv input d_mlp_dX[b, :nradbase] (Chebyshev dT/dr recurrence;
// embeddings are r-independent so excluded, giving the K=nradbase input sub-GEMM). Reproduces the
// hand kernel TagComputeMLPRadialDeriv's input math EXACTLY. Padding bonds (jj>=ncount) zero X and dX
// (finite GEMM).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPAssembleDeriv,
    const int b) const
{
  const int bond_stride = (int) d_radial_basis.extent(1);
  const int ii = b / bond_stride;
  const int jj = b - ii * bond_stride;
  const int n_in0 = nradbase + 2 * embedding_size;
  if (jj >= d_ncount(ii)) {
    for (int c = 0; c < n_in0; c++)   d_mlp_X(b, c)  = NNScalar(0.0);
    for (int c = 0; c < nradbase; c++) d_mlp_dX(b, c) = NNScalar(0.0);
    return;
  }
  const int mu_i = d_mu_i(ii);
  const int mu_j = d_mu_j(ii, jj);

  // Geometry (== TagComputeRadialBasis / hand deriv kernel).
  const BondScalar r = d_rnorms(ii, jj);
  const BondScalar rcut_ij = d_bond_cutoff(mu_i, mu_j);
  const BondScalar x_norm = r / rcut_ij;
  const BondScalar x_cheb = BondScalar(2.0) * x_norm - BondScalar(1.0);
  const BondScalar dxcheb_dr = BondScalar(2.0) / rcut_ij;

  // d_denv: TagComputeRadialBasis (once per bond, fp64)

  // Value input: [Chebyshev basis, chem_embed(mu_i), chem_embed(mu_j)].
  int k = 0;
  for (int c = 0; c < nradbase; c++)     d_mlp_X(b, k++) = (NNScalar) d_radial_basis(ii, jj, c);
  for (int e = 0; e < embedding_size; e++) d_mlp_X(b, k++) = d_chem_embed(mu_i, e);
  for (int e = 0; e < embedding_size; e++) d_mlp_X(b, k++) = d_chem_embed(mu_j, e);

  // Deriv input: dT_{b+1}/dr for the Chebyshev basis (dT_1/dx=1;
  // dT_{k+1}/dx = 2*T_k + 2*x*dT_k - dT_{k-1}); *dxcheb_dr for d/dr.
  {
    BondScalar dT_prev = BondScalar(0.0);   // dT_0/dx
    BondScalar dT_curr = BondScalar(1.0);   // dT_1/dx
    d_mlp_dX(b, 0) = (NNScalar)(dT_curr * dxcheb_dr);
    for (int kk = 1; kk < nradbase; kk++) {
      const BondScalar T_kk = (BondScalar) d_radial_basis(ii, jj, kk - 1);  // T_kk
      const BondScalar dT_next = BondScalar(2.0) * T_kk
                               + BondScalar(2.0) * x_cheb * dT_curr - dT_prev;
      d_mlp_dX(b, kk) = (NNScalar)(dT_next * dxcheb_dr);
      dT_prev = dT_curr; dT_curr = dT_next;
    }
  }
}

#endif
#ifdef KOKKOS_ENABLE_CUDA
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPActVal,
    const int idx) const
{
  const int nout = mlp_act_n;
  const int b = idx / nout;
  const int j = idx - b * nout;
  const DeviceSPBF &sp = d_spbf[mlp_assemble_idx];
  const NNScalar norm = mlp_act_norm;
  if (mlp_act_layer == 0) {
    const NNScalar s = d_mlp_h0(b, j) * norm + sp.mlp_b(0, j) * norm;
    d_mlp_h0(b, j) = silu(s);
  } else {
    const NNScalar s = d_mlp_h1(b, j) * norm + sp.mlp_b(1, j) * norm;
    d_mlp_h1(b, j) = silu(s);
  }
}

#endif
#ifdef KOKKOS_ENABLE_CUDA
// TagMLPActDeriv (Opt 15b): fused value+deriv activation. Reads the raw value
// GEMM output (d_mlp_h*) and raw deriv GEMM output (d_mlp_dh*), forms
// s = raw*norm + bias*norm and ds = raw_deriv*norm, then writes silu(s) and
// silu'(s)*ds in place. silu'(s) = sig*(1 + s*(1-sig)) -- identical to the hand
// kernel TagComputeMLPRadialDeriv.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPActDeriv,
    const int idx) const
{
  const int nout = mlp_act_n;
  const int b = idx / nout;
  const int j = idx - b * nout;
  const DeviceSPBF &sp = d_spbf[mlp_assemble_idx];
  const NNScalar norm = mlp_act_norm;
  if (mlp_act_layer == 0) {
    const NNScalar s  = d_mlp_h0(b, j) * norm + sp.mlp_b(0, j) * norm;
    const NNScalar ds = d_mlp_dh0(b, j) * norm;
    const NNScalar sig = NNScalar(1.0) / (NNScalar(1.0) + Kokkos::exp(-s));
    d_mlp_h0(b, j)  = s * sig;
    d_mlp_dh0(b, j) = sig * (NNScalar(1.0) + s * (NNScalar(1.0) - sig)) * ds;
  } else {
    const NNScalar s  = d_mlp_h1(b, j) * norm + sp.mlp_b(1, j) * norm;
    const NNScalar ds = d_mlp_dh1(b, j) * norm;
    const NNScalar sig = NNScalar(1.0) / (NNScalar(1.0) + Kokkos::exp(-s));
    d_mlp_h1(b, j)  = s * sig;
    d_mlp_dh1(b, j) = sig * (NNScalar(1.0) + s * (NNScalar(1.0) - sig)) * ds;
  }
}

#endif
#ifdef KOKKOS_ENABLE_CUDA
// TagMLPScatterR: write the LayoutRight output-GEMM result into the (LayoutLeft)
// 4D target via the View operator() (layout-agnostic). mlp_scatter_which selects
// value (d_mlp_R -> d_R1_nl) vs deriv (d_mlp_dR -> d_DR1_nl). Padding bonds
// (jj >= ncount) are skipped so the untouched entries match the hand kernel.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPScatterR,
    const int idx) const
{
  const int lm1 = lmax + 1;
  const int D3  = d_spbf[mlp_assemble_idx].n_rad_max * lm1;
  const int bs  = (int) d_radial_basis.extent(1);
  const int b   = idx / D3;
  const int j   = idx - b * D3;
  const int ii  = b / bs;
  const int jj  = b - ii * bs;
  if (jj >= d_ncount(ii)) return;
  if (mlp_scatter_which == 0) d_R1_nl(ii, jj, j / lm1, j % lm1)  = d_mlp_R(b, j);
  else                        d_DR1_nl(ii, jj, j / lm1, j % lm1) = d_mlp_dR(b, j);
}

#endif
#ifdef KOKKOS_ENABLE_CUDA
// TagMLPLayer0Val: radial-MLP layer 0 + silu straight from the Chebyshev basis, one (bond, output)
// per thread: s = sum_k T_k W0c(k, j) + E(mu_i, j) + E(mu_j, d1 + j) (norm0 and bias folded in, see
// copy_weights_to_device). Replaces MLPAssembleVal + the K = 138 layer-0 GEMM + MLPActVal0.
// Padding bonds get h0 = 0 (their R is never read).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPLayer0Val,
    const int idx) const
{
  const int d1 = mlp_act_n;
  const int b = idx / d1, j = idx - b * d1;
  const int bs = (int) d_radial_basis.extent(1);
  const int ii = b / bs, jj = b - ii * bs;
  if (jj >= d_ncount(ii)) { d_mlp_h0(b, j) = NNScalar(0.0); return; }
  NNScalar s = d_mlp_E_cur(d_mu_i(ii), j) + d_mlp_E_cur(d_mu_j(ii, jj), d1 + j);
  for (int k = 0; k < nradbase; k++) s += (NNScalar) d_radial_basis(ii, jj, k) * d_mlp_W0c_cur(k, j);
  d_mlp_h0(b, j) = silu(s);
}

#endif
#ifdef KOKKOS_ENABLE_CUDA
// TagMLPLayer0Deriv: TagMLPLayer0Val plus the r-derivative of the layer-0 activation (Chebyshev dT/dr
// recurrence as TagMLPAssembleDeriv).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagMLPLayer0Deriv,
    const int idx) const
{
  const int d1 = mlp_act_n;
  const int b = idx / d1, j = idx - b * d1;
  const int bs = (int) d_radial_basis.extent(1);
  const int ii = b / bs, jj = b - ii * bs;
  if (jj >= d_ncount(ii)) { d_mlp_h0(b, j) = NNScalar(0.0); d_mlp_dh0(b, j) = NNScalar(0.0); return; }
  const int mu_i = d_mu_i(ii), mu_j = d_mu_j(ii, jj);
  const GeomScalar r = d_rnorms(ii, jj);
  const GeomScalar rcut_ij = d_bond_cutoff(mu_i, mu_j);
  const GeomScalar x_norm = r / rcut_ij;
  const GeomScalar x_cheb = GeomScalar(2.0) * x_norm - GeomScalar(1.0);
  const GeomScalar dxcheb_dr = GeomScalar(2.0) / rcut_ij;
  NNScalar s = d_mlp_E_cur(mu_i, j) + d_mlp_E_cur(mu_j, d1 + j);
  NNScalar ds = NNScalar(0.0);
  GeomScalar dT_prev = GeomScalar(0.0), dT_curr = GeomScalar(1.0);   // dT_0/dx, dT_1/dx
  for (int k = 0; k < nradbase; k++) {
    if (k > 0) {
      const GeomScalar dT_next = GeomScalar(2.0) * d_radial_basis(ii, jj, k - 1)
                               + GeomScalar(2.0) * x_cheb * dT_curr - dT_prev;
      dT_prev = dT_curr; dT_curr = dT_next;
    }
    const NNScalar w = d_mlp_W0c_cur(k, j);
    s += (NNScalar) d_radial_basis(ii, jj, k) * w;
    ds += (NNScalar) (dT_curr * dxcheb_dr) * w;
  }
  const NNScalar sig = NNScalar(1.0) / (NNScalar(1.0) + Kokkos::exp(-s));
  d_mlp_h0(b, j)  = s * sig;
  d_mlp_dh0(b, j) = sig * (NNScalar(1.0) + s * (NNScalar(1.0) - sig)) * ds;
}
#endif

// ======================================================================
// Radial tables. R_si(x; mu_i, mu_j) for x = r / rcut_ij in [0, 1], all outputs o = n (lmax+1) + l of SPBF si.
// Nodes x_k = k / N: value and d/dx by an fp64 forward-mode pass of the MLP on the device (team per node);
// per interval the cubic a0 + a1 t + a2 t^2 + a3 t^3 (t in [0, 1)) from the Hermite data in fp64.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::build_radial_splines()
{
  spline_on = 0;
  const auto &m = *grace_model;
  const int nrb = m.nradbase, ne = m.embedding_size;
  bool shape_ok = spline_mb != 0.0 && n_spbf > 0 && n_spbf <= 3;
  for (int si = 0; si < n_spbf && shape_ok; si++) {
    const auto &mlp = m.spbf[si].mlp;
    shape_ok = mlp.n_layers == 3 && mlp.n_in[0] == nrb + 2 * ne && mlp.n_out[2] <= L1_nradmax * (lmax + 1);
  }
  if (!shape_ok) return;
  std::vector<int> sp_of_mu(m.n_elements, -1), mus;
  for (int t = 1; t < (int) h_type2model.size(); t++) {
    const int mu = h_type2model[t];
    if (mu >= 0 && sp_of_mu[mu] < 0) { sp_of_mu[mu] = (int) mus.size(); mus.push_back(mu); }
  }
  n_sp = (int) mus.size();
  if (n_sp == 0) return;
  const int npair = n_sp * n_sp;
  spl_ld = L1_nradmax * (lmax + 1);
  int Nl[3] = {0, 0, 0}, off[3] = {0, 0, 0}, Ntot = 0, Nmax = 0;
  for (int si = 0; si < n_spbf; si++) {
    Nl[si] = grace::spline_n_3l(spline_n, spline_n_layer, si);
    off[si] = Ntot;
    Ntot += Nl[si];
    Nmax = std::max(Nmax, Nl[si]);
  }
  const double mb = (double) npair * Ntot * spl_ld * 4 * sizeof(float) / 1048576.0;
  // spline_mb < 0 (auto): a quarter of the device memory per rank, at least 2048 MB, at most what is free with the
  // engine's own tables counted as free (a re-setup, with the store and chunk buffers of the last run still
  // allocated, decides as the first one)
  double budget = spline_mb;
  if (budget < 0.0 && std::equal(Nl, Nl + 3, spl_n_built) && spl_mus == mus && d_splR.size() > 0) {
    budget = mb;    // auto: the tables of the last setup fit then, keep them (the last run's buffers are still held)
  } else if (budget < 0.0) {
    budget = 2048.0;
#ifdef KOKKOS_ENABLE_CUDA
    size_t fr = 0, tot = 0;
    if constexpr (is_cuda_v<DeviceType>)
      if (cudaMemGetInfo(&fr, &tot) == cudaSuccess) {
        const double held = (double) d_splR.size() * sizeof(float);
        budget = std::max(budget, std::min(0.25 * (double) tot, (double) fr + held) /
                                      std::max(1, ranks_per_device) / 1048576.0);
      }
#endif
  }
  if (mb > budget) {
      log.info(fmt("[GRACE-3L/KK] radial MLP by GEMM: tables for {} species pairs would need {:.0f} MB > "
                     "spline_mb {:.0f}\n", npair, mb, budget));
    return;
  }
  spl_mus = mus;
  for (int si = 0; si < 3; si++) {
    spl_n_built[si] = spl_N[si] = Nl[si];
    spl_off[si] = off[si];
  }
  spl_Ntot = Ntot;
  using DV = Kokkos::View<double *, DeviceType>;
  MemKK::realloc_kokkos(d_splR, "g3l:splR", (size_t) npair * Ntot * spl_ld * 4);
  DV nodes("g3l:spl_nodes", (size_t) (Nmax + 1) * spl_ld * 2);   // [k][o][value, d/dx] for one (pair, layer)
  auto tab = d_splR;
  for (int si = 0; si < n_spbf; si++) {
    const auto &mlp = m.spbf[si].mlp;
    const int N = Nl[si];
    const int d1 = mlp.n_out[0], d2 = mlp.n_out[1], d3 = mlp.n_out[2];
    const double n0 = mlp.norm[0], n1 = mlp.norm[1], n2 = mlp.norm[2];
    // fp64 weights: W0c [nrb][d1] (norm0), E [n_sp][2][d1] (norm0, bias), W1 [d1][d2], b1, W2 [d2][d3]
    std::vector<double> hW0c((size_t) nrb * d1), hE((size_t) n_sp * 2 * d1), hW1((size_t) d1 * d2), hb1(d2),
        hW2((size_t) d2 * d3);
    for (int k = 0; k < nrb; k++)
      for (int j = 0; j < d1; j++) hW0c[(size_t) k * d1 + j] = n0 * mlp.W[0][(size_t) k * d1 + j];
    for (int sp = 0; sp < n_sp; sp++)
      for (int j = 0; j < d1; j++) {
        double ei = mlp.b[0].empty() ? 0.0 : mlp.b[0][j], ej = 0.0;
        for (int e = 0; e < ne; e++) {
          const double z = m.chem_embedding[(size_t) mus[sp] * ne + e];
          ei += z * mlp.W[0][(size_t) (nrb + e) * d1 + j];
          ej += z * mlp.W[0][(size_t) (nrb + ne + e) * d1 + j];
        }
        hE[((size_t) sp * 2) * d1 + j] = n0 * ei;
        hE[((size_t) sp * 2 + 1) * d1 + j] = n0 * ej;
      }
    for (size_t q = 0; q < hW1.size(); q++) hW1[q] = n1 * mlp.W[1][q];
    for (int j = 0; j < d2; j++) hb1[j] = mlp.b[1].empty() ? 0.0 : n1 * mlp.b[1][j];
    for (size_t q = 0; q < hW2.size(); q++) hW2[q] = n2 * mlp.W[2][q];
    auto todev = [](const std::vector<double> &h, const char *nm) {
      DV d(std::string(nm), h.size());
      auto hd = Kokkos::create_mirror_view(d);
      for (size_t q = 0; q < h.size(); q++) hd(q) = h[q];
      Kokkos::deep_copy(d, hd);
      return d;
    };
    DV W0c = todev(hW0c, "g3l:spl_W0c"), E = todev(hE, "g3l:spl_E"), W1 = todev(hW1, "g3l:spl_W1"),
       b1 = todev(hb1, "g3l:spl_b1"), W2 = todev(hW2, "g3l:spl_W2");
    const int ld = spl_ld, nsp = n_sp;
    using team_policy = Kokkos::TeamPolicy<DeviceType>;
    using member = typename team_policy::member_type;
    const size_t shb = (size_t) (2 * d1 + 2 * d2 + 2 * nrb) * sizeof(double);
    // teams of 128 on GPUs; host execution spaces cap a team at their thread count
    int tsz = 128;
#ifdef KOKKOS_ENABLE_CUDA
    if (!std::is_same<typename DeviceType::execution_space, Kokkos::Cuda>::value)
#endif
      tsz = std::min(tsz, std::max(1, (int) typename DeviceType::execution_space().concurrency()));
    for (int p = 0; p < npair; p++) {
      const int spi = p / nsp, spj = p - spi * nsp;
      Kokkos::parallel_for("g3l:spl_nodes", team_policy(N + 1, tsz).set_scratch_size(0, Kokkos::PerTeam(shb)),
        KOKKOS_LAMBDA(const member &t) {
          double *sh = (double *) t.team_shmem().get_shmem(shb);
          double *h0 = sh, *g0 = h0 + d1, *h1 = g0 + d1, *g1 = h1 + d2, *T = g1 + d2, *dT = T + nrb;
          const int k = t.league_rank();
          const double x = (double) k / N, xc = 2.0 * x - 1.0;
          if (t.team_rank() == 0) {
            double Tp = 1.0, Tc = xc, dTp = 0.0, dTc = 2.0;   // d/dx (chain factor 2)
            for (int q = 0; q < nrb; q++) {
              T[q] = Tc; dT[q] = dTc;
              const double Tn = 2.0 * xc * Tc - Tp, dTn = 2.0 * (2.0 * Tc + xc * dTc) - dTp;
              Tp = Tc; dTp = dTc; Tc = Tn; dTc = dTn;
            }
          }
          t.team_barrier();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(t, d1), [&](const int j) {
            double sv = E(((size_t) spi * 2) * d1 + j) + E(((size_t) spj * 2 + 1) * d1 + j), dsv = 0.0;
            for (int q = 0; q < nrb; q++) { const double w = W0c((size_t) q * d1 + j); sv += T[q] * w; dsv += dT[q] * w; }
            const double sg = 1.0 / (1.0 + Kokkos::exp(-sv));
            h0[j] = sv * sg; g0[j] = sg * (1.0 + sv * (1.0 - sg)) * dsv;
          });
          t.team_barrier();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(t, d2), [&](const int j) {
            double sv = b1(j), dsv = 0.0;
            for (int q = 0; q < d1; q++) { const double w = W1((size_t) q * d2 + j); sv += h0[q] * w; dsv += g0[q] * w; }
            const double sg = 1.0 / (1.0 + Kokkos::exp(-sv));
            h1[j] = sv * sg; g1[j] = sg * (1.0 + sv * (1.0 - sg)) * dsv;
          });
          t.team_barrier();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(t, d3), [&](const int o) {
            double v = 0.0, dv = 0.0;
            for (int q = 0; q < d2; q++) { const double w = W2((size_t) q * d3 + o); v += h1[q] * w; dv += g1[q] * w; }
            nodes(((size_t) k * ld + o) * 2) = v;
            nodes(((size_t) k * ld + o) * 2 + 1) = dv;
          });
        });
      const size_t base = (size_t) p * Ntot + off[si];
      Kokkos::parallel_for("g3l:spl_coef", Kokkos::RangePolicy<DeviceType>(0, N * ld), KOKKOS_LAMBDA(const int q) {
        const int k = q / ld, o = q - k * ld;
        float *c = tab.data() + ((base + k) * ld + o) * 4;
        if (o >= d3) { c[0] = c[1] = c[2] = c[3] = 0.0f; return; }
        const double hh = 1.0 / N;
        const double f0 = nodes(((size_t) k * ld + o) * 2), f1 = nodes(((size_t) (k + 1) * ld + o) * 2);
        const double g0 = nodes(((size_t) k * ld + o) * 2 + 1) * hh, g1 = nodes(((size_t) (k + 1) * ld + o) * 2 + 1) * hh;
        c[0] = (float) f0; c[1] = (float) g0;
        c[2] = (float) (3.0 * (f1 - f0) - 2.0 * g0 - g1);
        c[3] = (float) (2.0 * (f0 - f1) + g0 + g1);
      });
    }
  }
  Kokkos::fence();
  MemKK::realloc_kokkos(d_sp_of_mu, "g3l:sp_of_mu", m.n_elements);
  auto hs = Kokkos::create_mirror_view(d_sp_of_mu);
  for (int mu = 0; mu < m.n_elements; mu++) hs(mu) = sp_of_mu[mu];
  Kokkos::deep_copy(d_sp_of_mu, hs);
  spline_on = 1;
    log.info(fmt("[GRACE-3L/KK] radial MLP as cubic tables: {} species pairs x {} layers, intervals {} {} {}, {:.0f} MB\n",
                   npair, n_spbf, Nl[0], Nl[1], Nl[2], mb));
}

// R (and dR/dr) of layer radial_mlp_spbf for every bond of the chunk from the tables;
// thread per (bond, output), one float4 of coefficients each.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::radial_from_splines(bool deriv)
{
  const int si = radial_mlp_spbf, N = spl_N[si], NT = spl_Ntot, off = spl_off[si], nsp = n_sp, ld = spl_ld;
  const int d3 = grace_model->spbf[si].mlp.n_out[2];
  const int bs = (int) d_R1_nl.extent(1), ldR = (int) (d_R1_nl.extent(2) * d_R1_nl.extent(3));
  auto tab = d_splR; auto spm = d_sp_of_mu; auto rn = d_rnorms; auto bc = d_bond_cutoff;
  auto mui = d_mu_i; auto muj = d_mu_j; auto nc = d_ncount;
  NNScalar *R = d_R1_nl.data(), *DR = d_DR1_nl.data();
  const bool dv = deriv;
  Kokkos::parallel_for("g3l:spl_radial", Kokkos::RangePolicy<DeviceType>(0, chunk_size * bs * d3), KOKKOS_LAMBDA(const int q) {
    const int b = q / d3, o = q - b * d3, ii = b / bs, jj = b - ii * bs;
    if (jj >= nc(ii)) return;
    const int mi = mui(ii), mj = muj(ii, jj);
    const int p = spm(mi) * nsp + spm(mj);
    const GeomScalar rc = bc(mi, mj), x = GeomScalar(rn(ii, jj)) / rc;
    const float s = (float) x * N;
    int k = (int) s;
    if (k > N - 1) k = N - 1;
    const float t = s - k;
    const G3F4 c = *reinterpret_cast<const G3F4 *>(tab.data() + (((size_t) p * NT + off + k) * ld + o) * 4);
    R[(size_t) b * ldR + o] = c.x + t * (c.y + t * (c.z + t * c.w));
    if (dv) {
      DR[(size_t) b * ldR + o] = (NNScalar) ((c.y + t * (2.0f * c.z + 3.0f * t * c.w)) * (float) N / (float) rc);
    }
  });
}


// ======================================================================
// Opt 15: value-only radial MLP forward for layer `radial_mlp_spbf`, writing
// d_R1_nl. On CUDA with a live handle and the expected 3-layer architecture,
// runs the batched true-fp32 cuBLAS path (assemble -> SGEMM/act x2 -> output
// SGEMM straight into d_R1_nl); otherwise falls back to the hand kernel
// TagComputeMLPRadial. Row-major operands throughout (see mlp_sgemm).
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_mlp_radial()
{
  if ((int) d_R1_nl.extent(0) < chunk_size)    // grow() sizes it by the fuse_radial conditions of compute()
    GRACE_KK_ERROR("GRACE-3L/KK: radial scratch not allocated for the radial MLP of layer {}", radial_mlp_spbf + 1);
  if (spline_on) { radial_from_splines(false); return; }
#ifdef KOKKOS_ENABLE_CUDA
  const auto &hmlp = grace_model->spbf[radial_mlp_spbf].mlp;
  if (cublas_handle && hmlp.n_layers == 3 && sizeof(NNScalar) == 4) {
    const int si = radial_mlp_spbf;
    const int bond_stride = (int) d_R1_nl.extent(1);   // == d_radial_basis.extent(1)
    const int M = chunk_size * bond_stride;
    mlp_M = M;
    const int n_in0 = hmlp.n_in[0];        // 138
    const int d1    = hmlp.n_out[0];       // 256
    const int d2    = hmlp.n_out[1];       // 128
    const int d3    = hmlp.n_out[2];       // n_rad_max*(lmax+1)

    // NOTE: no fences between these steps. The cuBLAS handle is bound (in
    // setup()) to DeviceType().cuda_stream() -- the SAME stream the Kokkos
    // parallel_fors below launch on -- so assemble -> GEMM -> act -> GEMM -> ...
    // -> scatter execute in strict issue order on one stream (race-free).

    mlp_assemble_idx = si;
    if (mlp_split[si]) {
      // 1-3. layer 0 + silu from the Chebyshev basis and the per-element embedding terms
      mlp_act_n = d1; d_mlp_W0c_cur = d_mlp_W0c[si]; d_mlp_E_cur = d_mlp_E[si];
      Kokkos::parallel_for("MLPLayer0Val",
          Kokkos::RangePolicy<DeviceType, TagMLPLayer0Val>(0, M * d1), *this);
    } else {
      // 1. assemble X[M x n_in0]
      Kokkos::parallel_for("MLPAssembleVal",
          Kokkos::RangePolicy<DeviceType, TagMLPAssembleVal>(0, M), *this);
      // 2. h0 = X * W0                              (raw; norm+bias in Act)
      mlp_sgemm(d_mlp_Wg[si][0].data(), d_mlp_X.data(), d_mlp_h0.data(),
          d1, M, n_in0, (int)d_mlp_Wg[si][0].extent(1), (int)d_mlp_X.extent(1),
          (int)d_mlp_h0.extent(1), NNScalar(1.0));
      // 3. Act0: silu(h0*norm0 + b0*norm0)
      mlp_act_layer = 0; mlp_act_n = d1; mlp_act_norm = (NNScalar) hmlp.norm[0];
      Kokkos::parallel_for("MLPActVal0",
          Kokkos::RangePolicy<DeviceType, TagMLPActVal>(0, M * d1), *this);
    }

    // 4. h1 = h0 * W1
    mlp_sgemm(d_mlp_Wg[si][1].data(), d_mlp_h0.data(), d_mlp_h1.data(),
        d2, M, d1, (int)d_mlp_Wg[si][1].extent(1), (int)d_mlp_h0.extent(1),
        (int)d_mlp_h1.extent(1), NNScalar(1.0));
    // 5. Act1: silu(h1*norm1 + b1*norm1)
    mlp_act_layer = 1; mlp_act_n = d2; mlp_act_norm = (NNScalar) hmlp.norm[1];
    Kokkos::parallel_for("MLPActVal1",
        Kokkos::RangePolicy<DeviceType, TagMLPActVal>(0, M * d2), *this);

    // 6. R = out_norm * (h1 * W2) (output layer: linear, no bias), written straight into the
    //    row-major d_R1_nl: row = bond, columns n * (lmax+1) + l, leading dim = its (n, l) extent
    //    (>= d3). Padding bonds get finite values that are never read.
    mlp_sgemm(d_mlp_Wg[si][2].data(), d_mlp_h1.data(), d_R1_nl.data(),
        d3, M, d2, (int)d_mlp_Wg[si][2].extent(1), (int)d_mlp_h1.extent(1),
        (int)(d_R1_nl.extent(2) * d_R1_nl.extent(3)), (NNScalar) hmlp.norm[2]);
    return;
  }
#endif
  // Fallback: original hand kernel.
  int team_size = 1, vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1) team_size = 32;
  int ts = team_size;
  ts = fit_team_size<DeviceType, TagComputeMLPRadial>(*this, ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
  auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadial>(
      ((chunk_size+ts-1)/ts)*maxneigh, ts, vector_length);
  Kokkos::parallel_for("ComputeMLPRadial", policy, *this);
}

#ifdef KOKKOS_ENABLE_CUDA
#endif
#ifdef KOKKOS_ENABLE_CUDA
// Opt 15: true-fp32 batched SGEMM. Row-major C[M x n_out] = alpha*In[M x n_in]*Wg[n_in x n_out].
// cuBLAS is column-major, so this equals the col-major product
// C^T[n_out x M] = alpha * Wg^T[n_out x n_in] * In^T[n_in x M], i.e. OP_N/OP_N with
// leading dims = the row-major operands' physical row strides. CUBLAS_PEDANTIC_MATH
// (set on the handle) forces true fp32 (no tf32).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::mlp_sgemm(
    const NNScalar* Wg, const NNScalar* In, NNScalar* C,
    int n_out, int M, int n_in, int ldW, int ldIn, int ldC, NNScalar alpha)
{
  const float a = (float) alpha, beta = 0.0f;
  cublasStatus_t st = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
      n_out, M, n_in, &a,
      (const float*) Wg, ldW,
      (const float*) In, ldIn,
      &beta, (float*) C, ldC);
  if (st != CUBLAS_STATUS_SUCCESS)
    GRACE_KK_ERROR("GRACE-3L/KK: cublasSgemm failed (status {})", (int) st);
}

#endif
#ifdef KOKKOS_ENABLE_CUDA
// Opt 15 hard gate: verify the layer-0 SGEMM (transpose/leading-dim convention +
// true-fp32 path) reproduces a host fp64 reference on a tiny deterministic batch
// before the full chain is ever used. Aborts on gross mismatch (transpose bug ->
// O(1) error); fp32 rounding over K~138 stays well under the 1e-4 gate.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::mlp_cublas_selftest()
{
  const auto &mlp = grace_model->spbf[0].mlp;   // A1
  if (mlp.n_layers < 1) return;
  const int n_in = mlp.n_in[0], n_out = mlp.n_out[0];
  const int Mt = 4;
  t_nn_2d_r X("g3l:selftest_X", Mt, n_in), C("g3l:selftest_C", Mt, n_out);
  auto hX = Kokkos::create_mirror_view(X);
  for (int b = 0; b < Mt; b++)
    for (int k = 0; k < n_in; k++)
      hX(b, k) = (NNScalar)(0.01 * (double)(((b * 7 + k * 3) % 13) - 6));
  Kokkos::deep_copy(X, hX);

  mlp_sgemm(d_mlp_Wg[0][0].data(), X.data(), C.data(), n_out, Mt, n_in,
      (int)d_mlp_Wg[0][0].extent(1), (int)X.extent(1), (int)C.extent(1), NNScalar(1.0));
  Kokkos::fence();

  auto hC = Kokkos::create_mirror_view(C);
  Kokkos::deep_copy(hC, C);
  double maxrel = 0.0;
  for (int b = 0; b < Mt; b++)
    for (int j = 0; j < n_out; j++) {
      double ref = 0.0;
      for (int k = 0; k < n_in; k++)
        ref += (double) hX(b, k) * mlp.W[0][(size_t) k * n_out + j];
      const double rel = std::fabs((double) hC(b, j) - ref) / std::max(1e-6, std::fabs(ref));
      maxrel = std::max(maxrel, rel);
    }
  // Threshold sized for the fp32 cublasSgemm this selftest exercises: correct
  // fp32-GEMM rounding is ~a few e-4 for realistic MLP widths, while a
  // transpose/leading-dim convention bug (what this gate guards against) yields
  // O(0.1-1) error. 7e-4 passes the former and still catches the latter.
  if (maxrel > 7e-4)
    GRACE_KK_ERROR("GRACE-3L/KK: cuBLAS layer-0 selftest FAILED (max rel err {})", maxrel);
    log.info(fmt("[GRACE-3L/KK] cuBLAS radial-MLP selftest OK (layer-0 max rel {:.2e})\n", maxrel));
}
#endif

// ---------------------- force launchers ----------------------

// Opt 15b: radial MLP forward value + r-derivative for layer `radial_mlp_spbf`,
// writing d_R1_nl (R) and d_DR1_nl (dR/dr). On CUDA with a live handle
// and the expected 3-layer architecture, runs the batched true-fp32 cuBLAS path
// (6 SGEMMs: 3 value + 3 deriv, deriv layer-0 is a K=nradbase sub-GEMM since only
// the Chebyshev inputs are r-dependent); otherwise the hand kernel
// TagComputeMLPRadialDeriv. Same-stream serialization => race-free, no fences.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_mlp_radial_deriv()
{
  if ((int) d_R1_nl.extent(0) < chunk_size)    // grow() sizes it by the fuse_radial conditions of compute()
    GRACE_KK_ERROR("GRACE-3L/KK: radial scratch not allocated for the radial MLP of layer {}", radial_mlp_spbf + 1);
  if (spline_on) { radial_from_splines(true); return; }
#ifdef KOKKOS_ENABLE_CUDA
  const auto &hmlp = grace_model->spbf[radial_mlp_spbf].mlp;
  if (cublas_handle && hmlp.n_layers == 3 && sizeof(NNScalar) == 4) {
    const int si = radial_mlp_spbf;
    const int bond_stride = (int) d_R1_nl.extent(1);
    const int M = chunk_size * bond_stride;
    mlp_M = M;
    const int n_in0 = hmlp.n_in[0];   // 138
    const int d1    = hmlp.n_out[0];  // 256
    const int d2    = hmlp.n_out[1];  // 128
    const int d3    = hmlp.n_out[2];  // n_rad_max*(lmax+1)
    mlp_assemble_idx = si;

    if (mlp_split[si]) {
      // 1-3. layer 0 value + r-derivative from the Chebyshev basis
      mlp_act_n = d1; d_mlp_W0c_cur = d_mlp_W0c[si]; d_mlp_E_cur = d_mlp_E[si];
      Kokkos::parallel_for("MLPLayer0Deriv",
          Kokkos::RangePolicy<DeviceType, TagMLPLayer0Deriv>(0, M * d1), *this);
    } else {
      // 1. assemble X[M x n_in0], dX[M x nradbase]
      Kokkos::parallel_for("MLPAssembleDeriv",
          Kokkos::RangePolicy<DeviceType, TagMLPAssembleDeriv>(0, M), *this);
      // 2. layer 0: value h0 = X*W0 (K=n_in0); deriv dh0 = dX*W0[0:nradbase] (K=nradbase)
      mlp_sgemm(d_mlp_Wg[si][0].data(), d_mlp_X.data(),  d_mlp_h0.data(),
          d1, M, n_in0,    (int)d_mlp_Wg[si][0].extent(1), (int)d_mlp_X.extent(1),
          (int)d_mlp_h0.extent(1),  NNScalar(1.0));
      mlp_sgemm(d_mlp_Wg[si][0].data(), d_mlp_dX.data(), d_mlp_dh0.data(),
          d1, M, nradbase, (int)d_mlp_Wg[si][0].extent(1), (int)d_mlp_dX.extent(1),
          (int)d_mlp_dh0.extent(1), NNScalar(1.0));
      // 3. act-deriv layer 0
      mlp_act_layer = 0; mlp_act_n = d1; mlp_act_norm = (NNScalar) hmlp.norm[0];
      Kokkos::parallel_for("MLPActDeriv0",
          Kokkos::RangePolicy<DeviceType, TagMLPActDeriv>(0, M * d1), *this);
    }

    // 4. layer 1: value h1 = h0*W1; deriv dh1 = dh0*W1
    mlp_sgemm(d_mlp_Wg[si][1].data(), d_mlp_h0.data(),  d_mlp_h1.data(),
        d2, M, d1, (int)d_mlp_Wg[si][1].extent(1), (int)d_mlp_h0.extent(1),
        (int)d_mlp_h1.extent(1),  NNScalar(1.0));
    mlp_sgemm(d_mlp_Wg[si][1].data(), d_mlp_dh0.data(), d_mlp_dh1.data(),
        d2, M, d1, (int)d_mlp_Wg[si][1].extent(1), (int)d_mlp_dh0.extent(1),
        (int)d_mlp_dh1.extent(1), NNScalar(1.0));
    // 5. act-deriv layer 1
    mlp_act_layer = 1; mlp_act_n = d2; mlp_act_norm = (NNScalar) hmlp.norm[1];
    Kokkos::parallel_for("MLPActDeriv1",
        Kokkos::RangePolicy<DeviceType, TagMLPActDeriv>(0, M * d2), *this);

    // 6. output layer (linear, no bias): value R=h1*W2*out_norm -> d_R1_nl;
    //    deriv DR=dh1*W2*out_norm -> d_DR1_nl (both row-major, written directly).
    const NNScalar out_norm = (NNScalar) hmlp.norm[2];
    const int ldR = (int) (d_R1_nl.extent(2) * d_R1_nl.extent(3));
    mlp_sgemm(d_mlp_Wg[si][2].data(), d_mlp_h1.data(),  d_R1_nl.data(),
        d3, M, d2, (int)d_mlp_Wg[si][2].extent(1), (int)d_mlp_h1.extent(1), ldR, out_norm);
    mlp_sgemm(d_mlp_Wg[si][2].data(), d_mlp_dh1.data(), d_DR1_nl.data(),
        d3, M, d2, (int)d_mlp_Wg[si][2].extent(1), (int)d_mlp_dh1.extent(1), ldR, out_norm);
    return;
  }
#endif
  int team_size = 1, vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1) team_size = 32;
  team_size = fit_team_size<DeviceType, TagComputeMLPRadialDeriv>(*this, ((chunk_size+team_size-1)/team_size)*maxneigh, team_size, vector_length);
  auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadialDeriv>(
      ((chunk_size+team_size-1)/team_size)*maxneigh, team_size, vector_length);
  Kokkos::parallel_for("ComputeMLPRadialDeriv", policy, *this);
}

// ======================================================================
// Template instantiations (members defined in this unit)
// ======================================================================

}    // namespace kk
}    // namespace grace

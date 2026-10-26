// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Part of grace::kk::Grace3LKK (grace_3l_kk.h), included by grace_3l_kk_impl.h:
// FunctionReduceN, FCRight2Left, equivariant / invariant RMS norms and the readout backward.
// Moved from the LAMMPS style grace/3l/kk (lammps-ace 5f5ab57aa7, pair_grace_3l_kokkos_dense.cpp).
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
// Kernel: TagReduceN (Task 2.6) — generic FunctionReduceN. Reproduces
// numpy_forward_3l.reduce_n EXACTLY for one reduce r over the current chunk:
//   for each instruction ins with n_conn connections:
//     for each connection c: gather A[a, n, collect_ind[c]] over n in
//     [0,n_in), dot with W[e, k, n, w_l_tile[c]] (e = elem_dep ? mu_i : 0;
//     W stored uniformly 4D so no branch on the read), * norm;
//     only_invar -> accumulate into f=0 (ignore total_sum_ind);
//     else       -> accumulate into f=total_sum_ind[c].
//   collection *= norm_map (if present).
// One thread owns the whole out[a,k,:] row (flat index over atom a, output
// channel k) -> the per-instruction/per-connection accumulation is serial,
// no atomics needed (same ownership pattern as TagCPLCouple).
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagReduceN,
    const int& idx) const
{
  const int n_out = reduceN_r.n_out;
  const int a = idx % chunk_size;
  const int k = idx / chunk_size;

  const bool only_invar = reduceN_r.only_invar != 0;
  const bool elem_dep = reduceN_r.elem_dep != 0;
  const int shape0 = only_invar ? 1 : reduceN_r.n_funcs;
  const int e = elem_dep ? d_mu_i(a) : 0;

  NNScalar acc[GRACE3L_MAX_REDUCE_NFUNCS];
  for (int f = 0; f < shape0; f++) acc[f] = NNScalar(0.0);

  for (int ins = 0; ins < reduceN_r.n_instr; ins++) {
    const auto &Wv = reduceN_r.W[ins];
    const auto &ci = reduceN_r.collect_ind[ins];
    const auto &wlt = reduceN_r.w_l_tile[ins];
    const auto &tsi = reduceN_r.total_sum_ind[ins];
    const int n_in = reduceN_r.n_in[ins];
    const int n_conn = reduceN_r.n_conn[ins];
    const NNScalar norm = reduceN_r.norm[ins];
    const auto &A = reduceN_inputs[ins];

    for (int c = 0; c < n_conn; c++) {
      const int lm = ci(c);
      const int tile = wlt(c);
      NNScalar s = NNScalar(0.0);
      for (int n = 0; n < n_in; n++)
        s += Wv(e, k, n, tile) * A(a, n, lm);
      s *= norm;
      if (only_invar) acc[0] += s;
      else acc[tsi(c)] += s;
    }
  }

  if (reduceN_r.has_norm_map)
    for (int f = 0; f < shape0; f++) acc[f] *= reduceN_r.norm_map(f);

  for (int f = 0; f < shape0; f++) reduceN_out(a, k, f) = acc[f];
}

// ======================================================================
// compute_reduceN (Task 2.6) — generic FunctionReduceN. Sets the per-call
// launch state (reduce metadata + per-instruction input views + output view)
// captured by TagReduceN, then launches one flat pass over (atom, n_out) for
// the current chunk. Reused by every A*_*_red / eq* / rho* reduce; the caller
// is responsible for matching inputs[j] to grace_model->reduce[i].instr[j]
// .name (device kernels have no strings to do that lookup themselves).
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_reduceN(
    const DeviceReduce &r, const t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR],
    const t_nn_3d &out)
{
#ifdef GRACE_KK_GEMM
  if (blas_handle && sizeof(NNScalar) == 4) {
    const DeviceReduce *rbase = d_reduce;
    const int idx = (int) (&r - rbase);
    if (idx >= 0 && idx < n_reduces && h_reduce[idx].cublas_ok && h_reduce[idx].grouped &&
        compute_reduceN_grouped(idx, inputs[0], out, false))
      return;
    if (idx >= 0 && idx < n_reduces && h_reduce[idx].cublas_ok &&
        compute_reduceN_cublas(r, idx, inputs, out))
      return;
  }
#endif
  if (r.blk_ok && sizeof(NNScalar) == 4 && !getenv("GRACE3L_GENERIC_SPBF")) { reduceN_blocked(r, inputs, out); return; }
  reduceN_r = r;
  for (int i = 0; i < GRACE3L_MAX_REDUCE_INSTR; i++) reduceN_inputs[i] = inputs[i];
  reduceN_out = out;
  Kokkos::parallel_for("ReduceN",
      Kokkos::RangePolicy<DeviceType, TagReduceN>(0, chunk_size * r.n_out), *this);
}

#ifdef GRACE_KK_GEMM
// Opt 17: cuBLAS batched-GEMM ReduceN forward. out(a,k,f) = Σ_{c:tsi(c)=f} Σ_n
// A(a,n,ci(c)) * Wsc[c](k,n), where Wsc already carries norm*norm_map(f). Zero `out`,
// then issue one cublasSgemmBatched per round (each round has DISTINCT f -> the beta=1
// accumulate into disjoint out(:,:,f) slabs is race-free). A/out LayoutLeft slabs are
// column-major (ld=extent(0), slab stride=extent(0)*extent(1)); the batch selects the
// gathered slabs via a device pointer array. OP_N/OP_T (K=n_in). True fp32 (PEDANTIC).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
const uintptr_t *Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::red_upload(int idx, int var, int n,
                                                                              const uintptr_t key[6])
{
  RedPtrs &c = red_ptrs[idx][var];
  bool same = (int) c.d.extent(0) >= n;
  for (int u = 0; u < 6; u++) same = same && c.key[u] == key[u];
  if (!same) {
    if ((int) c.d.extent(0) < n) c.d = Kokkos::View<uintptr_t *, DeviceType>("g3l:reduce_ptrs", n);
    Kokkos::deep_copy(Kokkos::subview(c.d, std::make_pair(0, n)), Kokkos::subview(h_redptr, std::make_pair(0, n)));
    for (int u = 0; u < 6; u++) c.key[u] = key[u];
  }
  return c.d.data();
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
bool Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_reduceN_cublas(
    const DeviceReduce &r, int idx, const t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR],
    const t_nn_3d &out)
{
  const ReduceHost &hr = h_reduce[idx];
  const t_nn_3d &A = inputs[0];
  const int nc = hr.n_conn, n_out = hr.n_out, n_in = hr.n_in;
  if (nc <= 0) return true;
  Kokkos::deep_copy(typename DeviceType::execution_space(), out, NNScalar(0));   // assignment (some f have no conn)
  const long strideA = (long) A.extent(0) * A.extent(1);
  const long strideC = (long) out.extent(0) * out.extent(1);
  const uintptr_t baseA = (uintptr_t) A.data();
  const uintptr_t baseW = (uintptr_t) r.Wsc.data();
  const uintptr_t baseC = (uintptr_t) out.data();
  const size_t szf = sizeof(NNScalar);
  for (int i = 0; i < nc; i++) {
    const int c = hr.fwd_order[i];
    h_redptr(i)          = baseA + (uintptr_t) (strideA * hr.ci[c]) * szf;
    h_redptr(nc + i)     = baseW + (uintptr_t) ((size_t) c * n_out * n_in) * szf;
    h_redptr(2 * nc + i) = baseC + (uintptr_t) (strideC * hr.tsi[c]) * szf;
  }
  const uintptr_t key[6] = {baseA, baseC, (uintptr_t) strideA, (uintptr_t) strideC, baseW, (uintptr_t) nc};
  const uintptr_t *P = red_upload(idx, 0, 3 * nc, key);
  const float** Aarr = (const float**) P;
  const float** Barr = (const float**) (P + nc);
  float** Carr       = (float**)       (P + 2 * nc);
  const float alpha = 1.0f, beta = 1.0f;
  const int ldA = (int) A.extent(0), ldC = (int) out.extent(0);
  for (int rr = 0; rr < hr.nrounds; rr++) {
    const int off = hr.round_ptr[rr];
    const int bs = hr.round_ptr[rr + 1] - off;
    if (bs <= 0) continue;
    blas::status_t st = blas::sgemm_batched(blas_handle, blas::OP_N, blas::OP_T,
        chunk_size, n_out, n_in, &alpha,
        Aarr + off, ldA, Barr + off, n_out, &beta,
        Carr + off, ldC, bs);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-3L/KK: reduceN fwd cublasSgemmBatched failed (status {})", (int) st);
  }
  return true;
}
#endif

// ======================================================================
// Kernel: TagFCRight2Left (Task 2.6) — generic FCRight2Left. Reproduces
// numpy_forward_3l.fc_right2left EXACTLY for one FC f over the current chunk:
//   left_t[lm,a,k] = left_coefs ? norm_left * Sum_n w_left[k,n,w_tile_left[lm]]
//                                            * left[a,n,lm]     (lm in [0,n_funcs_left))
//                               : left[a,k,lm]      (left already at this FC's n_out)
//   for each connection c in [0,n_funcs_right):
//     right_t = norm_right * Sum_n w_right[k,n,w_tile_right[c]] * right[a,n,collect_from[c]]
//     left_t[collect_to[c]] += right_t
//   left_t *= norm_out_factor   (loader requires it unconditionally -> always applied)
// One thread owns the whole out[a,k,:] row (flat index over atom a, output
// channel k) -> the scatter-add over connections is serial, no atomics needed
// (same ownership pattern as TagReduceN / TagCPLCouple).
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagFCRight2Left,
    const int& idx) const
{
  const int n_out = fc_p.n_out;
  const int a = idx % chunk_size;
  const int k = idx / chunk_size;

  const int n_funcs_left = fc_p.n_funcs_left;
  NNScalar acc[GRACE3L_MAX_FC_NFUNCS];

  if (fc_p.left_coefs) {
    const int n_in_left = fc_p.n_in_left;
    for (int w = 0; w < n_funcs_left; w++) {
      const int tile = fc_p.w_tile_left(w);
      NNScalar s = NNScalar(0.0);
      for (int n = 0; n < n_in_left; n++)
        s += fc_p.w_left(k, n, tile) * fc_in_left(a, n, w);
      acc[w] = s * fc_p.norm_left;
    }
  } else {
    for (int w = 0; w < n_funcs_left; w++)
      acc[w] = fc_in_left(a, k, w);
  }

  const int n_funcs_right = fc_p.n_funcs_right;
  const int n_in_right = fc_p.n_in_right;
  for (int c = 0; c < n_funcs_right; c++) {
    const int src = fc_p.collect_from(c);
    const int dst = fc_p.collect_to(c);
    const int tile = fc_p.w_tile_right(c);
    NNScalar s = NNScalar(0.0);
    for (int n = 0; n < n_in_right; n++)
      s += fc_p.w_right(k, n, tile) * fc_in_right(a, n, src);
    acc[dst] += s * fc_p.norm_right;
  }

  for (int w = 0; w < n_funcs_left; w++)
    acc[w] *= fc_p.norm_out_factor(w);

  for (int w = 0; w < n_funcs_left; w++)
    fc_out(a, k, w) = acc[w];
}

// ======================================================================
// compute_fc (Task 2.6) — generic FCRight2Left. Sets the per-call launch
// state (FC metadata + left/right input views + output view) captured by
// TagFCRight2Left, then launches one flat pass over (atom, n_out) for the
// current chunk. Reused by every A*_2a/2b FC.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_fc(
    const DeviceFC &f, const t_nn_3d &in_left, const t_nn_3d &in_right,
    const t_nn_3d &out)
{
#ifdef GRACE_KK_GEMM
  if (blas_handle && sizeof(NNScalar) == 4 && f.gemm_ok) {
    // out = nof * left, then one strided-batched GEMM over w accumulates right(:,:,w) Bw[w]
    const int cs = chunk_size, no = f.n_out, nf = f.n_funcs_left, ni = f.n_in_right;
    auto L = in_left; auto O = out; auto nof = f.norm_out_factor;
    Kokkos::parallel_for("FCLeftScale", Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0, 0, 0}, {cs, no, nf}),
      KOKKOS_LAMBDA(const int a, const int k, const int w) { O(a, k, w) = nof(w) * L(a, k, w); });
    const float one = 1.0f;
    blas::status_t st = blas::sgemm_strided_batched(blas_handle, blas::OP_N, blas::OP_N,
        cs, no, ni, &one,
        (const float*) in_right.data(), (int) in_right.extent(0), (long long) in_right.extent(0) * in_right.extent(1),
        (const float*) f.Bw.data(), ni, (long long) no * ni,
        &one, (float*) out.data(), (int) out.extent(0), (long long) out.extent(0) * out.extent(1), nf);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-3L/KK: FC forward cublas failed (status {})", (int) st);
    return;
  }
#endif
  fc_p = f;
  fc_in_left = in_left;
  fc_in_right = in_right;
  fc_out = out;
  Kokkos::parallel_for("FCRight2Left",
      Kokkos::RangePolicy<DeviceType, TagFCRight2Left>(0, chunk_size * f.n_out), *this);
}

// ======================================================================
// Kernel: TagEqNorm (Task 2.7) — generic EquivariantRMSNorm. Reproduces
// numpy_forward_3l.equivariant_rms_norm EXACTLY for one eqnorm e over the
// current chunk. x = eqnorm_in[a, feat, lm] with feat in [0,n_out=n_feat),
// lm in [0,M):
//   center_l0: l0_mean[lm] = mean_over_feat(x[a,:,lm]) * l0_mask[lm]
//              x <- x - l0_mean[lm]                      (broadcast over feat)
//   norm[a]  = mean_over_feat( sum_over_lm( dw[lm] * x^2 ) )   -- ONE scalar/atom
//   rms[a]   = 1/sqrt(norm[a] + eps)
//   scale[feat,lm] = affine_weight[expand_index[lm], feat]
//   out[a,feat,lm] = x[a,feat,lm] * rms[a] * scale[feat,lm]
// One thread owns an entire atom (all feat*lm entries never touched by any
// other thread) -> three sequential passes over the same (feat,lm) grid
// (mean, sum-of-squares, write), re-reading eqnorm_in each time instead of
// staging the centered x in a [feat,lm]-sized stack buffer (up to 64*25 —
// too large to keep as registers). l0_mean[lm] (<=25 lm) is the only
// intermediate kept in a stack array, matching the existing acc[25] idiom
// used by TagComputeA1 for the same lm-axis bound.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagEqNorm,
    const int& a) const
{
  const int n_feat = eqnorm_p.n_out;
  const int M = eqnorm_p.M;
  const bool center_l0 = eqnorm_p.center_l0 != 0;
  const NNScalar eps = eqnorm_p.eps;

  NNScalar l0mean[GRACE3L_MAX_SPBF_NLMIND];   // M (<=64) validated at load
  for (int lm = 0; lm < M; lm++) l0mean[lm] = NNScalar(0.0);

  if (center_l0) {
    for (int lm = 0; lm < M; lm++) {
      NNScalar s = NNScalar(0.0);
      for (int f = 0; f < n_feat; f++) s += eqnorm_in(a, f, lm);
      l0mean[lm] = (s / (NNScalar) n_feat) * eqnorm_p.l0_mask(lm);
    }
  }

  NNScalar sumsq = NNScalar(0.0);
  for (int f = 0; f < n_feat; f++)
    for (int lm = 0; lm < M; lm++) {
      const NNScalar xc = eqnorm_in(a, f, lm) - l0mean[lm];
      sumsq += xc * xc * eqnorm_p.degree_weights(lm);
    }
  const NNScalar norm = sumsq / (NNScalar) n_feat;
  const NNScalar rms = NNScalar(1.0) / Kokkos::sqrt(norm + eps);

  for (int f = 0; f < n_feat; f++)
    for (int lm = 0; lm < M; lm++) {
      const NNScalar xc = eqnorm_in(a, f, lm) - l0mean[lm];
      const int grp = eqnorm_p.expand_index(lm);
      const NNScalar scale = eqnorm_p.affine_weight(grp, f);
      eqnorm_out(a, f, lm) = xc * rms * scale;
    }
}

// ======================================================================
// compute_equiv_rms_norm (Task 2.7) — generic EquivariantRMSNorm. Sets the
// per-call launch state captured by TagEqNorm, then launches one thread per
// atom for the current chunk. Reused for eq1_norm/eq2_norm.
// ======================================================================

// The team EqNorm kernel sums over f with 32-lane warp shuffles: CUDA and HIP only (other backends use the per-atom
// kernels; the shuffle helpers abort outside CUDA / HIP device code).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
bool Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::eqnorm_team_ok(int M) const
{
#if defined(KOKKOS_ENABLE_CUDA)
  return std::is_same<typename DeviceType::execution_space, Kokkos::Cuda>::value && sizeof(NNScalar) == 4 &&
         M <= GRACE3L_MAX_SPBF_NLMIND;
#elif defined(KOKKOS_ENABLE_HIP)
  return is_hip_v<DeviceType> && sizeof(NNScalar) == 4 && M <= GRACE3L_MAX_SPBF_NLMIND;
#else
  (void) M;
  return false;
#endif
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_equiv_rms_norm(
    const DeviceEqNorm &e, const t_nn_3d &in, const t_nn_3d &out)
{
  static const bool per_atom = getenv("GRACE3L_EQNORM_ATOM") != nullptr;
  if (!per_atom && eqnorm_team_ok(e.M)) {
    eqnorm_team(e.center_l0, e.n_out, e.M, (float) e.eps, e.affine_weight, e.degree_weights, e.expand_index,
                e.l0_mask, in, out, out, out, false);
    return;
  }
  eqnorm_p = e;
  eqnorm_in = in;
  eqnorm_out = out;
  Kokkos::parallel_for("EqNorm",
      Kokkos::RangePolicy<DeviceType, TagEqNorm>(0, chunk_size), *this);
}

// ======================================================================
// Kernel: TagEqNormBwd (Task 3.2) — generic EquivariantRMSNorm VJP. Analytic
// transpose of TagEqNorm. Forward (per atom a; x=eqnorm_in[a,f,lm], f<n_feat,
// lm<M):
//   l0mean[lm] = mean_f(x[a,:,lm]) * l0_mask[lm]     (center_l0)
//   xc         = x - l0mean
//   S[a]       = (1/F) Σ_{f,lm} dw[lm]*xc^2          (F=n_feat, ONE scalar/atom)
//   rms[a]     = 1/sqrt(S[a]+eps)
//   out        = xc*rms*scale[f,lm],  scale[f,lm]=affine_weight[expand_index[lm],f]
// VJP w.r.t x (affine is a constant weight, no weight-grad needed):
//   adj_S   = -0.5*rms^3 * Σ_{f,lm} adj_out*xc*scale
//   adj_xc  = adj_out*rms*scale + adj_S*(2/F)*dw[lm]*xc
//   adj_x   = adj_xc - l0_mask[lm]*(1/F)*Σ_f adj_xc     (center_l0 transpose)
// One thread/atom (owns all f*lm). Recomputes xc/rms from the forward input
// (not cached). ASSIGNS adj_in (eq2 has a single consumer: eq2_norm). Stack
// arrays sized to the parity-doubled lm ceiling 50 (eq2_norm M=50).
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagEqNormBwd,
    const int& a) const
{
  const int n_feat = eqnorm_p.n_out;
  const int M = eqnorm_p.M;
  const bool center_l0 = eqnorm_p.center_l0 != 0;
  const NNScalar eps = eqnorm_p.eps;
  const NNScalar invF = NNScalar(1.0) / (NNScalar) n_feat;

  NNScalar l0mean[GRACE3L_MAX_SPBF_NLMIND];   // M (<=64) validated at load
  for (int lm = 0; lm < M; lm++) l0mean[lm] = NNScalar(0.0);
  if (center_l0) {
    for (int lm = 0; lm < M; lm++) {
      NNScalar s = NNScalar(0.0);
      for (int f = 0; f < n_feat; f++) s += eqnorm_in(a, f, lm);
      l0mean[lm] = s * invF * eqnorm_p.l0_mask(lm);
    }
  }

  // Recompute S/rms and adj_S = -0.5*rms^3 * Σ adj_out*xc*scale.
  NNScalar sumsq = NNScalar(0.0);
  NNScalar adot = NNScalar(0.0);
  for (int f = 0; f < n_feat; f++)
    for (int lm = 0; lm < M; lm++) {
      const NNScalar xc = eqnorm_in(a, f, lm) - l0mean[lm];
      sumsq += xc * xc * eqnorm_p.degree_weights(lm);
      const NNScalar scale = eqnorm_p.affine_weight(eqnorm_p.expand_index(lm), f);
      adot += eqnorm_adj_out(a, f, lm) * xc * scale;
    }
  const NNScalar S = sumsq * invF;
  const NNScalar rms = NNScalar(1.0) / Kokkos::sqrt(S + eps);
  const NNScalar adj_S = NNScalar(-0.5) * rms * rms * rms * adot;

  // colsum[lm] = Σ_f adj_xc[a,f,lm] (only needed for the center_l0 transpose).
  NNScalar colsum[GRACE3L_MAX_SPBF_NLMIND];   // M (<=64) validated at load
  for (int lm = 0; lm < M; lm++) colsum[lm] = NNScalar(0.0);
  if (center_l0) {
    for (int lm = 0; lm < M; lm++) {
      const NNScalar dw = eqnorm_p.degree_weights(lm);
      NNScalar cs = NNScalar(0.0);
      for (int f = 0; f < n_feat; f++) {
        const NNScalar xc = eqnorm_in(a, f, lm) - l0mean[lm];
        const NNScalar scale = eqnorm_p.affine_weight(eqnorm_p.expand_index(lm), f);
        const NNScalar adj_xc = eqnorm_adj_out(a, f, lm) * rms * scale
                              + adj_S * (NNScalar(2.0) * invF) * dw * xc;
        cs += adj_xc;
      }
      colsum[lm] = cs;
    }
  }

  for (int f = 0; f < n_feat; f++)
    for (int lm = 0; lm < M; lm++) {
      const NNScalar dw = eqnorm_p.degree_weights(lm);
      const NNScalar xc = eqnorm_in(a, f, lm) - l0mean[lm];
      const NNScalar scale = eqnorm_p.affine_weight(eqnorm_p.expand_index(lm), f);
      const NNScalar adj_xc = eqnorm_adj_out(a, f, lm) * rms * scale
                            + adj_S * (NNScalar(2.0) * invF) * dw * xc;
      // center_l0 transpose (colsum==0 when !center_l0 -> pure identity).
      eqnorm_adj_in(a, f, lm) = adj_xc - eqnorm_p.l0_mask(lm) * invF * colsum[lm];
    }
}

// compute_equiv_rms_norm_bwd (Task 3.2) — sets the per-call state captured by
// TagEqNormBwd, then launches one thread per atom. ASSIGNS adj_in.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_equiv_rms_norm_bwd(
    const DeviceEqNorm &e, const t_nn_3d &in, const t_nn_3d &adj_out,
    const t_nn_3d &adj_in)
{
  static const bool per_atom = getenv("GRACE3L_EQNORM_ATOM") != nullptr;
  if (!per_atom && eqnorm_team_ok(e.M)) {
    eqnorm_team(e.center_l0, e.n_out, e.M, (float) e.eps, e.affine_weight, e.degree_weights, e.expand_index,
                e.l0_mask, in, adj_in, adj_out, adj_in, true);
    return;
  }
  eqnorm_p = e;
  eqnorm_in = in;
  eqnorm_adj_out = adj_out;
  eqnorm_adj_in = adj_in;
  Kokkos::parallel_for("EqNormBwd",
      Kokkos::RangePolicy<DeviceType, TagEqNormBwd>(0, chunk_size), *this);
}

// ======================================================================
// Kernel: TagRMSNorm (Task 2.7) — generic InvariantLayerRMSNorm. Reproduces
// numpy_forward_3l.invariant_rms_norm EXACTLY for one rmsnorm r over the
// current chunk, at one (atom, lm) column x[a,:,lm]:
//   full        (scale_len==n_out):   rms = 1/sqrt(mean_over_ch(x^2)+eps);
//                                      out[ch] = x[ch]*rms*scale[ch]
//   only_nonlin (scale_len==n_out-1): out[0] = x[0] (passthrough); rms over
//                                      channels [1,n_out) only;
//                                      out[ch] = x[ch]*rms*scale[ch-1], ch>=1
// eps=1e-10 fixed (oracle hardcodes F32(1e-10), not an npz-loaded value).
// One thread owns the whole column x[a,:,lm] (flat index over atom a, lm) ->
// no atomics needed (same ownership pattern as TagReduceN/TagFCRight2Left).
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagRMSNorm,
    const int& idx) const
{
  const int n_lm = (int) rmsnorm_in.extent(2);
  const int a = idx / n_lm;
  const int lm = idx - a * n_lm;

  const int n_out = rmsnorm_p.n_out;
  const NNScalar eps = NNScalar(1e-10);

  if (rmsnorm_p.scale_len == n_out) {
    // full
    NNScalar sumsq = NNScalar(0.0);
    for (int c = 0; c < n_out; c++) {
      const NNScalar v = rmsnorm_in(a, c, lm);
      sumsq += v * v;
    }
    const NNScalar rms = NNScalar(1.0) / Kokkos::sqrt(sumsq / (NNScalar) n_out + eps);
    for (int c = 0; c < n_out; c++)
      rmsnorm_out(a, c, lm) = rmsnorm_in(a, c, lm) * rms * rmsnorm_p.scale(c);
  } else {
    // only_nonlin: scale_len == n_out - 1; channel 0 passthrough.
    rmsnorm_out(a, 0, lm) = rmsnorm_in(a, 0, lm);
    const int n_nl = n_out - 1;
    NNScalar sumsq = NNScalar(0.0);
    for (int c = 1; c < n_out; c++) {
      const NNScalar v = rmsnorm_in(a, c, lm);
      sumsq += v * v;
    }
    const NNScalar rms = NNScalar(1.0) / Kokkos::sqrt(sumsq / (NNScalar) n_nl + eps);
    for (int c = 1; c < n_out; c++)
      rmsnorm_out(a, c, lm) = rmsnorm_in(a, c, lm) * rms * rmsnorm_p.scale(c - 1);
  }
}

// ======================================================================
// compute_invariant_rms_norm (Task 2.7) — generic InvariantLayerRMSNorm.
// Sets the per-call launch state captured by TagRMSNorm, then launches one
// flat pass over (atom, lm) for the current chunk. Reused for
// rho1_norm/rho2_norm/rho3_norm.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_invariant_rms_norm(
    const DeviceRMSNorm &r, const t_nn_3d &in, const t_nn_3d &out)
{
  rmsnorm_p = r;
  rmsnorm_in = in;
  rmsnorm_out = out;
  const int n_lm = (int) in.extent(2);
  Kokkos::parallel_for("RMSNorm",
      Kokkos::RangePolicy<DeviceType, TagRMSNorm>(0, chunk_size * n_lm), *this);
}

// ======================================================================
// ============ Task 3.1: readout + Layer-3 backward (adjoints) ==========
// GENERIC reverse-mode VJP kernels + launchers (reused by Tasks 3.2/3.3 for
// L2/L1). Each is the analytic transpose of the matching forward routine.
// ======================================================================

// ---- TagReadoutBwd: VJP of TagComputeMLPEnergy. Seed dE/dE_atom=1. Recomputes
// the hidden pre-activations from the SUMMED 3-density input, backprops through
// the energy MLP, then writes d_rho{1,2,3}_norm_adj (all EQUAL — the 3 densities
// enter symmetrically): ch0 = output_scale (linear skip), ch (c+1) = adj_input[c],
// every lm>0 slot = 0. Mirrors GRACE-2L ReverseMLPEnergy generalized to 3 origins.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagReadoutBwd,
    const int& ii) const
{
  if (ii >= chunk_size) return;

  NNScalar pre_act_buf[GRACE3L_MAX_MLP_LAYERS * GRACE3L_MAX_ENERGY_MLP_DIM];
  NNScalar adj_a[GRACE3L_MAX_ENERGY_MLP_DIM], adj_b[GRACE3L_MAX_ENERGY_MLP_DIM];
  NNScalar* pre_act = pre_act_buf;
  NNScalar* adj_cur = adj_a;
  NNScalar* adj_nxt = adj_b;

  const int emax = energy_max_dim;
  const int nin0 = d_energy_dims(0);

  // Forward recompute: layer-0 pre-activations from the summed 3-density input.
  {
    const int nout = d_energy_dims(1);
    for (int j = 0; j < nout; j++) {
      NNScalar s = NNScalar(0.0);
      for (int k = 0; k < nin0; k++) {
        const NNScalar h = d_rho1_norm_full(ii + chunk_offset, k + 1, 0) + d_rho2_norm_full(ii + chunk_offset, k + 1, 0) + d_rho3_norm(ii, k + 1, 0);
        s += d_energy_W(0, k, j) * h;
      }
      pre_act[0 * emax + j] = s * d_energy_norms(0);
    }
  }
  for (int layer = 1; layer < energy_n_layers - 1; layer++) {
    const int nin = d_energy_dims(layer);
    const int nout = d_energy_dims(layer + 1);
    for (int j = 0; j < nout; j++) {
      NNScalar s = NNScalar(0.0);
      for (int k = 0; k < nin; k++) {
        const NNScalar pa = pre_act[(layer - 1) * emax + k];
        const NNScalar act = (energy_activation == 1) ? tanh_act(pa) : silu(pa);
        s += d_energy_W(layer, k, j) * act;
      }
      pre_act[layer * emax + j] = s * d_energy_norms(layer);
    }
  }

  // Backward: output layer, seeded by output_scale (E = (e+lin)*output_scale+shift).
  {
    const int last = energy_n_layers - 1;
    const int nin = d_energy_dims(last);
    const int nout = d_energy_dims(last + 1);
    for (int k = 0; k < nin; k++) {
      NNScalar v = NNScalar(0.0);
      for (int o = 0; o < nout; o++) v += d_energy_W(last, k, o) * d_energy_norms(last);
      adj_cur[k] = v * output_scale;
    }
  }
  for (int layer = energy_n_layers - 2; layer >= 0; layer--) {
    const int nin = d_energy_dims(layer);
    const int nout = d_energy_dims(layer + 1);
    for (int j = 0; j < nout; j++) {
      const NNScalar pa = pre_act[layer * emax + j];
      if (energy_activation == 1) {
        const NNScalar t = tanh_act(pa);
        adj_cur[j] *= (NNScalar(1.0) - t * t);            // tanh'
      } else {
        const NNScalar sig = NNScalar(1.0) / (NNScalar(1.0) + Kokkos::exp(-pa));
        adj_cur[j] *= sig * (NNScalar(1.0) + pa * (NNScalar(1.0) - sig));  // silu'
      }
    }
    for (int k = 0; k < nin; k++) {
      NNScalar v = NNScalar(0.0);
      for (int j = 0; j < nout; j++) v += adj_cur[j] * d_energy_W(layer, k, j);
      adj_nxt[k] = v * d_energy_norms(layer);
    }
    NNScalar* tmp = adj_cur; adj_cur = adj_nxt; adj_nxt = tmp;
  }

  // Write the shared readout adjoint into all 3 densities (identical). Zero every
  // channel/lm first (n_lm==1 for rho, but stay general), then set ch0 + ch1..nin0.
  const int nch = (int) d_rho3_norm_adj.extent(1);
  const int n_lm = (int) d_rho3_norm_adj.extent(2);
  // Task 3.5b: rho1/rho2 readout adjoints are natom-promoted -> write the dense
  // list-order row (ii+chunk_offset) so they survive until P5/P4; rho3 stays
  // chunk-local (ii, consumed within THIS P3 chunk).
  const int io = ii + chunk_offset;
  for (int c = 0; c < nch; c++)
    for (int lm = 0; lm < n_lm; lm++) {
      d_rho1_norm_adj(io, c, lm) = NNScalar(0.0);
      d_rho2_norm_adj(io, c, lm) = NNScalar(0.0);
      d_rho3_norm_adj(ii, c, lm) = NNScalar(0.0);
    }
  d_rho1_norm_adj(io, 0, 0) = output_scale;
  d_rho2_norm_adj(io, 0, 0) = output_scale;
  d_rho3_norm_adj(ii, 0, 0) = output_scale;
  for (int c = 0; c < nin0; c++) {
    d_rho1_norm_adj(io, c + 1, 0) = adj_cur[c];
    d_rho2_norm_adj(io, c + 1, 0) = adj_cur[c];
    d_rho3_norm_adj(ii, c + 1, 0) = adj_cur[c];
  }
}

// ---- TagRMSNormBwd: VJP of TagRMSNorm (InvariantLayerRMSNorm). Per (atom, lm)
// column, recompute rms from the forward input, dot = Σ_c adj_out·out (forward
// output), and dx[c] = rms*(adj_out[c]*scale[c] - x[c]*dot*rms/N). Both branches
// (full / only_nonlin, ch0 passthrough) selected by scale_len. eps=1e-10 (3L).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagRMSNormBwd,
    const int& idx) const
{
  const int n_lm = (int) rmsnorm_adj_in.extent(2);
  const int a = idx / n_lm;
  const int lm = idx - a * n_lm;

  const int n_out = rmsnorm_p.n_out;
  const NNScalar eps = NNScalar(1e-10);

  if (rmsnorm_p.scale_len == n_out) {
    // full
    NNScalar sumsq = NNScalar(0.0);
    for (int c = 0; c < n_out; c++) {
      const NNScalar v = rmsnorm_in(a, c, lm);
      sumsq += v * v;
    }
    const NNScalar rms = NNScalar(1.0) / Kokkos::sqrt(sumsq / (NNScalar) n_out + eps);
    NNScalar dot = NNScalar(0.0);
    for (int c = 0; c < n_out; c++)
      dot += rmsnorm_adj_out(a, c, lm) * rmsnorm_out(a, c, lm);
    for (int c = 0; c < n_out; c++) {
      const NNScalar x = rmsnorm_in(a, c, lm);
      rmsnorm_adj_in(a, c, lm) =
          rms * (rmsnorm_adj_out(a, c, lm) * rmsnorm_p.scale(c) - x * dot * rms / (NNScalar) n_out);
    }
  } else {
    // only_nonlin: channel 0 passthrough, RMS over channels [1,n_out).
    rmsnorm_adj_in(a, 0, lm) = rmsnorm_adj_out(a, 0, lm);
    const int n_nl = n_out - 1;
    NNScalar sumsq = NNScalar(0.0);
    for (int c = 1; c < n_out; c++) {
      const NNScalar v = rmsnorm_in(a, c, lm);
      sumsq += v * v;
    }
    const NNScalar rms = NNScalar(1.0) / Kokkos::sqrt(sumsq / (NNScalar) n_nl + eps);
    NNScalar dot = NNScalar(0.0);
    for (int c = 1; c < n_out; c++)
      dot += rmsnorm_adj_out(a, c, lm) * rmsnorm_out(a, c, lm);
    for (int c = 1; c < n_out; c++) {
      const NNScalar x = rmsnorm_in(a, c, lm);
      rmsnorm_adj_in(a, c, lm) =
          rms * (rmsnorm_adj_out(a, c, lm) * rmsnorm_p.scale(c - 1) - x * dot * rms / (NNScalar) n_nl);
    }
  }
}

// ---- TagReduceNBwd: VJP of TagReduceN, for ONE instruction (reduceN_bwd_instr).
// One thread per (a, n_in). adj_input[a,n,collect_ind[c]] += norm*norm_map[dst]*
// Σ_k adj_out[a,k,dst]*W[e,k,n,w_l_tile[c]], accumulated serially over c (same-fi
// collisions handled within the owning thread; += onto the pre-zeroed buffer to
// accumulate a tensor's fan-out across reduces/products).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagReduceNBwd,
    const int a, const int n) const
{
  const int ins = reduceN_bwd_instr;
  const bool only_invar = reduceN_r.only_invar != 0;
  const bool elem_dep = reduceN_r.elem_dep != 0;
  const int n_out = reduceN_r.n_out;
  const int e = elem_dep ? d_mu_i(a) : 0;

  const auto &Wv = reduceN_r.W[ins];
  const auto &ci = reduceN_r.collect_ind[ins];
  const auto &wlt = reduceN_r.w_l_tile[ins];
  const auto &tsi = reduceN_r.total_sum_ind[ins];
  const int n_conn = reduceN_r.n_conn[ins];
  const NNScalar norm = reduceN_r.norm[ins];
  const bool hnm = reduceN_r.has_norm_map;

  for (int c = 0; c < n_conn; c++) {
    const int fi = ci(c);
    const int tile = wlt(c);
    const int dst = only_invar ? 0 : tsi(c);
    NNScalar v = NNScalar(0.0);
    for (int k = 0; k < n_out; k++)
      v += reduceN_adj_out(a, k, dst) * Wv(e, k, n, tile);
    NNScalar factor = norm;
    if (hnm) factor *= reduceN_r.norm_map(dst);
    reduceN_adj_in(a, n, fi) += v * factor;
  }
}

// ---- TagFCBwdLeft: VJP of the FCRight2Left left path. left_coefs=false (identity,
// i==output channel k): adj_left[a,k,w] += adj_out[a,k,w]*norm_out_factor[w].
// left_coefs=true (i==input feature n): adj_left[a,n,w] += norm_left*norm_out_factor[w]
// * Σ_k w_left[k,n,w_tile_left[w]]*adj_out[a,k,w]. Owning (a,i,w) thread -> no atomic.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagFCBwdLeft,
    const int a, const int i, const int w) const
{
  const NNScalar nof = fc_p.norm_out_factor(w);
  if (!fc_p.left_coefs) {
    fc_adj_left(a, i, w) += fc_adj_out(a, i, w) * nof;
  } else {
    const int n_out = fc_p.n_out;
    const int tile = fc_p.w_tile_left(w);
    NNScalar s = NNScalar(0.0);
    for (int k = 0; k < n_out; k++)
      s += fc_p.w_left(k, i, tile) * fc_adj_out(a, k, w);
    fc_adj_left(a, i, w) += s * fc_p.norm_left * nof;
  }
}

// ---- TagFCBwdRight: VJP of the FCRight2Left right path. For connection c
// (src=collect_from[c], dst=collect_to[c]): adj_right[a,n,src] += norm_right*
// norm_out_factor[dst]*Σ_k w_right[k,n,w_tile_right[c]]*adj_out[a,k,dst]. Multiple
// c may share src -> atomic_add.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagFCBwdRight,
    const int a, const int c, const int n) const
{
  const int src = fc_p.collect_from(c);
  const int dst = fc_p.collect_to(c);
  const int tile = fc_p.w_tile_right(c);
  const NNScalar nof = fc_p.norm_out_factor(dst);
  const int n_out = fc_p.n_out;
  NNScalar s = NNScalar(0.0);
  for (int k = 0; k < n_out; k++)
    s += fc_p.w_right(k, n, tile) * fc_adj_out(a, k, dst);
  Kokkos::atomic_add(&fc_adj_right(a, n, src), s * fc_p.norm_right * nof);
}

// ---------------------- backward launchers ----------------------

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::readout_bwd()
{
  Kokkos::parallel_for("ReadoutBwd",
      Kokkos::RangePolicy<DeviceType, TagReadoutBwd>(0, chunk_size), *this);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_invariant_rms_norm_bwd(
    const DeviceRMSNorm &r, const t_nn_3d &in, const t_nn_3d &out,
    const t_nn_3d &adj_out, const t_nn_3d &adj_in)
{
  rmsnorm_p = r;
  rmsnorm_in = in;
  rmsnorm_out = out;
  rmsnorm_adj_out = adj_out;
  rmsnorm_adj_in = adj_in;
  const int n_lm = (int) in.extent(2);
  Kokkos::parallel_for("RMSNormBwd",
      Kokkos::RangePolicy<DeviceType, TagRMSNormBwd>(0, chunk_size * n_lm), *this);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_reduceN_bwd(
    const DeviceReduce &r, const t_nn_3d &adj_out,
    const t_nn_3d adj_inputs[GRACE3L_MAX_REDUCE_INSTR])
{
#ifdef GRACE_KK_GEMM
  if (blas_handle && sizeof(NNScalar) == 4) {
    const DeviceReduce *rbase = d_reduce;
    const int idx = (int) (&r - rbase);
    if (idx >= 0 && idx < n_reduces && h_reduce[idx].cublas_ok && h_reduce[idx].grouped &&
        compute_reduceN_grouped(idx, adj_inputs[0], adj_out, true))
      return;
    if (idx >= 0 && idx < n_reduces && h_reduce[idx].cublas_ok &&
        compute_reduceN_bwd_cublas(r, idx, adj_out, adj_inputs[0]))
      return;
  }
#endif
  if (r.blk_ok && sizeof(NNScalar) == 4 && !getenv("GRACE3L_GENERIC_SPBF")) {
    for (int ins = 0; ins < r.n_instr; ins++) reduceN_bwd_blocked(r, ins, adj_out, adj_inputs[ins]);
    return;
  }
  reduceN_r = r;
  reduceN_adj_out = adj_out;
  for (int ins = 0; ins < r.n_instr; ins++) {
    reduceN_bwd_instr = ins;
    reduceN_adj_in = adj_inputs[ins];
    const int n_in = r.n_in[ins];
    Kokkos::parallel_for("ReduceNBwd",
        Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>, TagReduceNBwd>({0, 0}, {chunk_size, n_in}), *this);
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_reduceN_bwd_set(
    const DeviceReduce &r, const t_nn_3d &adj_out, const t_nn_3d &adj_in)
{
#ifdef GRACE_KK_GEMM
  static const bool off = getenv("GRACE3L_REDUCE_BWD_ACC") != nullptr;
  const int idx = (int) (&r - d_reduce);
  if (!off && blas_handle && sizeof(NNScalar) == 4 && idx >= 0 && idx < n_reduces && r.n_instr == 1) {
    const ReduceHost &hr = h_reduce[idx];
    bool cover = hr.cublas_ok && hr.grouped && hr.n_conn == (int) adj_in.extent(2);
    for (size_t f = 0; cover && f < hr.gn.size(); f++)
      if (hr.gn[f] > 0 && hr.gstart[f] + hr.gn[f] > (int) adj_in.extent(2)) cover = false;
    // collect_ind is a bijection (cublas_ok), so n_conn columns in range are every column exactly once
    if (cover && compute_reduceN_grouped(idx, adj_in, adj_out, true, true)) return;
  }
#endif
  Kokkos::deep_copy(typename DeviceType::execution_space(), adj_in, NNScalar(0.0));
  t_nn_3d a[GRACE3L_MAX_REDUCE_INSTR];
  a[0] = adj_in;
  compute_reduceN_bwd(r, adj_out, a);
}

#ifdef GRACE_KK_GEMM
// Opt 17: cuBLAS batched-GEMM ReduceN backward (VJP). adj_in(a,n,ci(c)) += Σ_k
// adj_out(a,k,tsi(c)) * Wsc[c](k,n), Wsc carrying norm*norm_map(tsi(c)) (same fold as
// forward). collect_ind is a bijection (enforced at setup: cublas_ok is cleared for any
// reduce with duplicate collect_ind) -> every connection writes a DISTINCT adj_in slab,
// so a single beta=1 batch accumulates into the (pre-zeroed, fan-in) adj_in with no
// scatter and no race. OP_N/OP_N (K=n_out). True fp32 (PEDANTIC handle).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
bool Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_reduceN_bwd_cublas(
    const DeviceReduce &r, int idx, const t_nn_3d &adj_out, const t_nn_3d &adj_in)
{
  const ReduceHost &hr = h_reduce[idx];
  const int nc = hr.n_conn, n_out = hr.n_out, n_in = hr.n_in;
  if (nc <= 0) return true;
  const long strideAo = (long) adj_out.extent(0) * adj_out.extent(1);
  const long strideAi = (long) adj_in.extent(0) * adj_in.extent(1);
  const uintptr_t baseAo = (uintptr_t) adj_out.data();
  const uintptr_t baseW  = (uintptr_t) r.Wsc.data();
  const uintptr_t baseAi = (uintptr_t) adj_in.data();
  const size_t szf = sizeof(NNScalar);
  for (int c = 0; c < nc; c++) {
    h_redptr(c)          = baseAo + (uintptr_t) (strideAo * hr.tsi[c]) * szf;
    h_redptr(nc + c)     = baseW  + (uintptr_t) ((size_t) c * n_out * n_in) * szf;
    h_redptr(2 * nc + c) = baseAi + (uintptr_t) (strideAi * hr.ci[c]) * szf;
  }
  const uintptr_t key[6] = {baseAo, baseAi, (uintptr_t) strideAo, (uintptr_t) strideAi, baseW, (uintptr_t) nc};
  const uintptr_t *P = red_upload(idx, 1, 3 * nc, key);
  const float** Aarr = (const float**) P;
  const float** Barr = (const float**) (P + nc);
  float** Carr       = (float**)       (P + 2 * nc);
  const float alpha = 1.0f, beta = 1.0f;
  const int ldAo = (int) adj_out.extent(0), ldAi = (int) adj_in.extent(0);
  blas::status_t st = blas::sgemm_batched(blas_handle, blas::OP_N, blas::OP_N,
      chunk_size, n_in, n_out, &alpha,
      Aarr, ldAo, Barr, n_out, &beta,
      Carr, ldAi, nc);
  if (st != blas::SUCCESS)
    GRACE_KK_ERROR("GRACE-3L/KK: reduceN bwd cublasSgemmBatched failed (status {})", (int) st);
  return true;
}
#endif


#ifdef GRACE_KK_GEMM
// Grouped reduce with the input in d_spbf_T (row-major [ii][c * n_in + n], c the input column): block f is the
// row-major [cs x gn n_in] matrix at column gstart f * n_in, i.e. a column-major [gn n_in x cs] one with ld =
// row stride. Forward out(:,:,f) = block^T-op * Bg_f (OP_T); backward block^T = Bg_f * dout(:,:,f)^T, assigned
// (the blocks are disjoint and cover every column: spbf_rm_ok).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_reduceN_grouped_rm(int idx, const t_nn_3d &out, bool bwd)
{
  const ReduceHost &hr = h_reduce[idx];
  const int n_out = hr.n_out, n_in = hr.n_in;
  const int n_f = (int) hr.gn.size();
  if ((int) d_redptr.extent(0) < 3 * n_f || (int) out.extent(1) != n_out)
    GRACE_KK_ERROR("GRACE-3L/KK: row-major grouped reduce: buffer shape mismatch");
  if (!bwd && hr.has_empty) Kokkos::deep_copy(typename DeviceType::execution_space(), out, NNScalar(0));
  const long E0o = out.extent(0), ldT = (long) d_spbf_T.stride(0);
  const uintptr_t bi = (uintptr_t) d_spbf_T.data(), bo = (uintptr_t) out.data(), bw = (uintptr_t) d_Bg[idx].data();
  const size_t szf = sizeof(float);
  int q = 0;
  std::vector<int> start(hr.gsets.size());
  for (size_t g = 0; g < hr.gsets.size(); g++) {
    start[g] = q;
    for (int f : hr.gsets[g]) {
      h_redptr(q)           = bi + (uintptr_t) ((long) n_in * hr.gstart[f]) * szf;
      h_redptr(n_f + q)     = bw + (uintptr_t) hr.boff[f] * szf;
      h_redptr(2 * n_f + q) = bo + (uintptr_t) (E0o * n_out * f) * szf;
      q++;
    }
  }
  const uintptr_t key[6] = {bi, bo, bw, (uintptr_t) E0o, (uintptr_t) ldT, (uintptr_t) n_f};
  const uintptr_t *P = red_upload(idx, bwd ? 5 : 2, 3 * n_f, key);
  const float one = 1.0f, zero = 0.0f;
  for (size_t g = 0; g < hr.gsets.size(); g++) {
    const int cnt = (int) hr.gsets[g].size(), K = hr.gn[hr.gsets[g][0]] * n_in, o = start[g];
    float **Pin = (float **) (P + o), **Pw = (float **) (P + n_f + o);
    float **Pout = (float **) (P + 2 * n_f + o);
    blas::status_t st;
    if (!bwd)
      st = blas::sgemm_batched(blas_handle, blas::OP_T, blas::OP_N, chunk_size, n_out, K, &one,
                               (const float **) Pin, (int) ldT, (const float **) Pw, K, &zero, Pout, (int) E0o, cnt);
    else
      st = blas::sgemm_batched(blas_handle, blas::OP_N, blas::OP_T, K, chunk_size, n_out, &one,
                               (const float **) Pw, K, (const float **) Pout, (int) E0o, &zero, Pin, (int) ldT, cnt);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-3L/KK: row-major grouped reduce cublasSgemmBatched failed (status {})", (int) st);
  }
}

#endif
#ifdef GRACE_KK_GEMM
// Grouped reduce: forward out(:,:,f) = in(:, :, gstart f .. +gn f) [cs x gn n_in] * Bg_f [gn n_in x n_out];
// backward din(:, :, block f) += dout(:,:,f) * Bg_f^T. One cublasSgemmBatched per set of outputs with equal gn.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
bool Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_reduceN_grouped(
    int idx, const t_nn_3d &in, const t_nn_3d &out, bool bwd, bool assign)
{
  const ReduceHost &hr = h_reduce[idx];
  const int n_out = hr.n_out, n_in = hr.n_in;
  if ((int) in.extent(1) != n_in || (int) out.extent(1) != n_out) return false;
  const int n_f = (int) hr.gn.size();
  if ((int) d_redptr.extent(0) < 3 * n_f) return false;
  if (!bwd && hr.has_empty) Kokkos::deep_copy(typename DeviceType::execution_space(), out, NNScalar(0));
  const long E0i = in.extent(0), E0o = out.extent(0);
  const uintptr_t bi = (uintptr_t) in.data(), bo = (uintptr_t) out.data(), bw = (uintptr_t) d_Bg[idx].data();
  const size_t szf = sizeof(NNScalar);
  int q = 0;
  std::vector<int> start(hr.gsets.size());
  for (size_t g = 0; g < hr.gsets.size(); g++) {
    start[g] = q;
    for (int f : hr.gsets[g]) {
      h_redptr(q)           = bi + (uintptr_t) (E0i * n_in * hr.gstart[f]) * szf;   // in block (cols of f)
      h_redptr(n_f + q)     = bw + (uintptr_t) hr.boff[f] * szf;
      h_redptr(2 * n_f + q) = bo + (uintptr_t) (E0o * n_out * f) * szf;              // out slab f
      q++;
    }
  }
  const uintptr_t key[6] = {bi, bo, bw, (uintptr_t) E0i, (uintptr_t) E0o, (uintptr_t) n_f};
  const uintptr_t *P = red_upload(idx, bwd ? 4 : 3, 3 * n_f, key);
  const float one = 1.0f, zero = 0.0f;
  for (size_t g = 0; g < hr.gsets.size(); g++) {
    const int cnt = (int) hr.gsets[g].size(), K = hr.gn[hr.gsets[g][0]] * n_in, o = start[g];
    float **Pin = (float **) (P + o), **Pw = (float **) (P + n_f + o);
    float **Pout = (float **) (P + 2 * n_f + o);
    blas::status_t st;
    if (!bwd)
      st = blas::sgemm_batched(blas_handle, blas::OP_N, blas::OP_N, chunk_size, n_out, K, &one,
                               (const float **) Pin, (int) E0i, (const float **) Pw, K, &zero, Pout, (int) E0o, cnt);
    else
      st = blas::sgemm_batched(blas_handle, blas::OP_N, blas::OP_T, chunk_size, K, n_out, &one,
                               (const float **) Pout, (int) E0o, (const float **) Pw, K, assign ? &zero : &one, Pin,
                               (int) E0i, cnt);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-3L/KK: grouped reduce cublasSgemmBatched failed (status {})", (int) st);
  }
  return true;
}
#endif

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_fc_bwd(
    const DeviceFC &f, const t_nn_3d &adj_out, const t_nn_3d &adj_left,
    const t_nn_3d &adj_right)
{
#ifdef GRACE_KK_GEMM
  if (blas_handle && sizeof(NNScalar) == 4 && f.gemm_ok) {
    // adj_left += nof * adj_out; adj_right(:,:,w) += adj_out(:,:,w) Bw[w]^T (one strided-batched GEMM)
    const int cs = chunk_size, no = f.n_out, nf = f.n_funcs_left, ni = f.n_in_right;
    auto G = adj_out; auto dL = adj_left; auto nof = f.norm_out_factor;
    Kokkos::parallel_for("FCBwdLeftScale", Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>>({0, 0, 0}, {cs, no, nf}),
      KOKKOS_LAMBDA(const int a, const int k, const int w) { dL(a, k, w) += nof(w) * G(a, k, w); });
    const float one = 1.0f;
    blas::status_t st = blas::sgemm_strided_batched(blas_handle, blas::OP_N, blas::OP_T,
        cs, ni, no, &one,
        (const float*) adj_out.data(), (int) adj_out.extent(0), (long long) adj_out.extent(0) * adj_out.extent(1),
        (const float*) f.Bw.data(), ni, (long long) no * ni,
        &one, (float*) adj_right.data(), (int) adj_right.extent(0),
        (long long) adj_right.extent(0) * adj_right.extent(1), nf);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-3L/KK: FC backward cublas failed (status {})", (int) st);
    return;
  }
#endif
  fc_p = f;
  fc_adj_out = adj_out;
  fc_adj_left = adj_left;
  fc_adj_right = adj_right;
  const int dim1_left = f.left_coefs ? f.n_in_left : f.n_out;
  Kokkos::parallel_for("FCBwdLeft",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>, TagFCBwdLeft>(
          {0, 0, 0}, {chunk_size, dim1_left, f.n_funcs_left}), *this);
  Kokkos::parallel_for("FCBwdRight",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>, TagFCBwdRight>(
          {0, 0, 0}, {chunk_size, f.n_funcs_right, f.n_in_right}), *this);
}

// Hand ReduceN in blocks: forward thread = (atom a, 8 outputs k, output f), 8 FMA per input load, weights as float4
// rows along k; backward thread = (atom a, 8 inputs n) per instruction, 32 FMA per 4 dE/dout loads.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::reduceN_blocked(
    const DeviceReduce &r, const t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR], const t_nn_3d &out)
{
  const int cs = chunk_size, K = r.n_out, nkb = K / 8, shape0 = r.only_invar ? 1 : r.n_funcs;
  const bool elem = r.elem_dep != 0, hnm = r.has_norm_map != 0;
  t_nn_3d in[GRACE3L_MAX_REDUCE_INSTR];
  int nin[GRACE3L_MAX_REDUCE_INSTR], T[GRACE3L_MAX_REDUCE_INSTR];
  long off[GRACE3L_MAX_REDUCE_INSTR];
  for (int i = 0; i < GRACE3L_MAX_REDUCE_INSTR; i++) {
    in[i] = inputs[i]; nin[i] = r.n_in[i]; T[i] = r.w_shape[i]; off[i] = r.wb_off[i];
  }
  auto Wb = r.Wb; auto roff = r.roff; auto rins = r.rins; auto rci = r.rci; auto rtile = r.rtile; auto nm = r.norm_map;
  auto mui = d_mu_i; auto O = out;
  Kokkos::parallel_for("ReduceNB", Kokkos::RangePolicy<DeviceType>(0, cs * nkb * shape0), KOKKOS_LAMBDA(const int q) {
    const int a = q % cs, r2 = q / cs, kb = r2 % nkb, f = r2 / nkb;
    const int e = elem ? mui(a) : 0;
    float acc[8];
#pragma unroll
    for (int u = 0; u < 8; u++) acc[u] = 0.0f;
    for (int p = roff(f); p < roff(f + 1); p++) {
      const int ins = rins(p), ci = rci(p), t = rtile(p), ni = nin[ins];
      const float *w = Wb.data() + off[ins] + ((long) (e * T[ins] + t) * ni) * K + kb * 8;
      const t_nn_3d &A = in[ins];
      for (int n = 0; n < ni; n++) {
        const float x = A(a, n, ci);
        const G3F4 w0 = *reinterpret_cast<const G3F4 *>(w + (long) n * K);
        const G3F4 w1 = *reinterpret_cast<const G3F4 *>(w + (long) n * K + 4);
        acc[0] += x * w0.x; acc[1] += x * w0.y; acc[2] += x * w0.z; acc[3] += x * w0.w;
        acc[4] += x * w1.x; acc[5] += x * w1.y; acc[6] += x * w1.z; acc[7] += x * w1.w;
      }
    }
    const float sc = hnm ? nm(f) : 1.0f;
#pragma unroll
    for (int u = 0; u < 8; u++) O(a, kb * 8 + u, f) = acc[u] * sc;
  });
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::reduceN_bwd_blocked(
    const DeviceReduce &r, int ins, const t_nn_3d &adj_out, const t_nn_3d &adj_in)
{
  const int cs = chunk_size, K = r.n_out, ni = r.n_in[ins], nnb = ni / 8, T = r.w_shape[ins], nconn = r.n_conn[ins];
  const bool elem = r.elem_dep != 0, hnm = r.has_norm_map != 0, inv_only = r.only_invar != 0;
  const long off = r.wb_off[ins];
  auto Wb = r.Wb; auto ci = r.collect_ind[ins]; auto wlt = r.w_l_tile[ins]; auto tsi = r.total_sum_ind[ins];
  auto nm = r.norm_map; auto mui = d_mu_i; auto G = adj_out; auto D = adj_in;
  // distinct output columns per connection: the connections go to separate threads (thread per (a, nb, c));
  // otherwise one thread walks them all
  static const bool cloop = getenv("GRACE3L_REDUCE_BWD_CLOOP") != nullptr;
  const bool cpar = r.ci_unique[ins] != 0 && !cloop;
  const int ncp = cpar ? nconn : 1, cper = cpar ? 1 : nconn;
  Kokkos::parallel_for("ReduceNBwdB", Kokkos::RangePolicy<DeviceType>(0, (long) cs * nnb * ncp), KOKKOS_LAMBDA(const long qq) {
    const int q = (int) (qq % ((long) cs * nnb)), cb = (int) (qq / ((long) cs * nnb));
    const int a = q % cs, nb = q / cs;
    const int e = elem ? mui(a) : 0;
    for (int c = cb; c < cb + cper; c++) {
      const int f = inv_only ? 0 : tsi(c), t = wlt(c);
      const float *w = Wb.data() + off + ((long) (e * T + t) * ni + nb * 8) * K;
      float acc[8];
#pragma unroll
      for (int j = 0; j < 8; j++) acc[j] = 0.0f;
      for (int k4 = 0; k4 < K / 4; k4++) {
        const float g0 = G(a, 4 * k4, f), g1 = G(a, 4 * k4 + 1, f), g2 = G(a, 4 * k4 + 2, f), g3 = G(a, 4 * k4 + 3, f);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const G3F4 w4 = *reinterpret_cast<const G3F4 *>(w + (long) j * K + 4 * k4);
          acc[j] += g0 * w4.x + g1 * w4.y + g2 * w4.z + g3 * w4.w;
        }
      }
      const float sc = hnm ? nm(f) : 1.0f;
      const int col = ci(c);
#pragma unroll
      for (int j = 0; j < 8; j++) D(a, nb * 8 + j, col) += acc[j] * sc;
    }
  });
}

// EquivariantRMSNorm (same math as TagEqNorm / TagEqNormBwd) with a team per 8 consecutive atoms: 4 warps, lane =
// (atom lane & 7, f quarter lane >> 3), f-group g = 4 warp + quarter owns f = g, g + 16, ...; a warp load covers 8
// atoms x 4 f (full 32-byte sectors). Sums over f: two warp shuffles, then the 4 warps in fixed order through
// shared memory (deterministic).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::eqnorm_team(
    int center_l0, int n_out, int M_, float eps_, const t_nn_2d &aw, const t_nn_1d &dw, const t_int_1d &ex,
    const t_nn_1d &mk, const t_nn_3d &X, const t_nn_3d &Y, const t_nn_3d &G, const t_nn_3d &DX, bool bwd)
{
  constexpr int NA = 8, NW = 4, NG = 16, MM = GRACE3L_MAX_SPBF_NLMIND;
  constexpr size_t SH = (size_t) (NW * MM * NA + 2 * MM * NA + 2 * NW * NA) * sizeof(float);
  const int cs = chunk_size, F = n_out, M = M_;
  const bool c0 = center_l0 != 0;
  const float eps = eps_, invF = 1.0f / (float) F, Ff = (float) F;
  using team_policy = Kokkos::TeamPolicy<DeviceType, hip_lb_t<DeviceType, NW * 32>>;
  using member = typename team_policy::member_type;
  team_policy pol((cs + NA - 1) / NA, NW * 32);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(SH));
  Kokkos::parallel_for(bwd ? "EqNormBwdT" : "EqNormT", pol, KOKKOS_LAMBDA(const member &t) {
    float *red = (float *) t.team_shmem().get_shmem(SH);
    float *lm0 = red + NW * MM * NA, *csum = lm0 + MM * NA, *sred = csum + MM * NA;
    const int tid = t.team_rank(), w = tid >> 5, lane = tid & 31, ai = lane & 7, fq = lane >> 3, fg = w * 4 + fq;
    const int a = t.league_rank() * NA + ai;
    const bool ok = a < cs;
    auto wsum = [&](float v) {
      v += g3l_shfl_xor(v, 8);
      v += g3l_shfl_xor(v, 16);
      return v;
    };
    // per-lm sums over f of v(f, lm) -> dst[lm * NA + atom]
    auto lmsum = [&](float *dst, auto &&val, float post_scale, bool use_mask) {
      for (int lm = 0; lm < M; lm++) {
        float s = 0.0f;
        if (ok) for (int f = fg; f < F; f += NG) s += val(f, lm);
        s = wsum(s);
        if (fq == 0) red[(w * MM + lm) * NA + ai] = s;
      }
      t.team_barrier();
      for (int q = tid; q < M * NA; q += NW * 32) {
        const int lm = q / NA, i = q - lm * NA;
        float s = 0.0f;
        for (int ww = 0; ww < NW; ww++) s += red[(ww * MM + lm) * NA + i];
        dst[q] = s * post_scale * (use_mask ? mk(lm) : 1.0f);
      }
      t.team_barrier();
    };
    if (c0) lmsum(lm0, [&](int f, int lm) { return X(a, f, lm); }, invF, true);
    else {
      for (int q = tid; q < M * NA; q += NW * 32) lm0[q] = 0.0f;
      t.team_barrier();
    }
    float ss = 0.0f, ad = 0.0f;
    if (ok)
      for (int lm = 0; lm < M; lm++) {
        const float l0 = lm0[lm * NA + ai], d = dw(lm);
        const int g = ex(lm);
        for (int f = fg; f < F; f += NG) {
          const float xc = X(a, f, lm) - l0;
          ss += xc * xc * d;
          if (bwd) ad += G(a, f, lm) * xc * aw(g, f);
        }
      }
    ss = wsum(ss);
    ad = wsum(ad);
    if (fq == 0) { sred[w * NA + ai] = ss; sred[(NW + w) * NA + ai] = ad; }
    t.team_barrier();
    float S = 0.0f, A = 0.0f;
    for (int ww = 0; ww < NW; ww++) { S += sred[ww * NA + ai]; A += sred[(NW + ww) * NA + ai]; }
    const float rms = 1.0f / Kokkos::sqrt(S / Ff + eps);
    if (!bwd) {
      if (ok)
        for (int lm = 0; lm < M; lm++) {
          const float l0 = lm0[lm * NA + ai];
          const int g = ex(lm);
          for (int f = fg; f < F; f += NG) Y(a, f, lm) = (X(a, f, lm) - l0) * rms * aw(g, f);
        }
      return;
    }
    const float k2 = -0.5f * rms * rms * rms * A * (2.0f * invF);
    auto adjxc = [&](int f, int lm) {
      const float xc = X(a, f, lm) - lm0[lm * NA + ai];
      return G(a, f, lm) * rms * aw(ex(lm), f) + k2 * dw(lm) * xc;
    };
    if (c0) lmsum(csum, adjxc, 1.0f, false);
    if (ok)
      for (int lm = 0; lm < M; lm++) {
        const float cl = c0 ? mk(lm) * invF * csum[lm * NA + ai] : 0.0f;
        for (int f = fg; f < F; f += NG) DX(a, f, lm) = adjxc(f, lm) - cl;
      }
  });
}

// ======================================================================
// Template instantiations (members defined in this unit)
// ======================================================================

}    // namespace kk
}    // namespace grace

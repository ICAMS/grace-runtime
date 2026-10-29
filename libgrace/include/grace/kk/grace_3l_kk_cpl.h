// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Part of grace::kk::Grace3LKK (grace_3l_kk.h), included by grace_3l_kk_impl.h:
// cp_l products: rank projections, CG couple (coupling-path kernels) and their VJPs.
#pragma once

#include "grace/kk/grace_3l_kk.h"
#include "grace/kk/grace_3l_kk_cpl_path.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace grace {
namespace kk {
using namespace sph_const;

// ======================================================================
// Kernel: TagCPLProject — cp_l Pass A. Projects each input onto the
// rank axis per lm, reproducing the reference einsum("wrn,anw->arw", U[gl], left)
// * norm_u  (w=lm, n=feature, r=rank):
//   lproj[a,r,w] = norm_u * Sum_n U[group_left[w], r, n] * left[a, n, w]
//   rproj[a,r,w] = norm_v * Sum_n V[group_right[w], r, n] * right[a, n, w]
// Flat index over (atom a, rank r, lm w); one thread owns one (a,r,w) triple,
// so the scratch write is exclusive (no atomics). n_lm_left may differ from
// n_lm_right, so each projection is guarded by its own group_*.extent(0).
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagCPLProject,
    const int& idx) const
{
  const int rank = cpl_p.rank;
  const int nlm = cpl_nlm;
  const int a = idx / (rank * nlm);
  const int rem = idx - a * (rank * nlm);
  const int r = rem / nlm;
  const int w = rem - r * nlm;

  const int n_lm_left = (int) cpl_p.group_left.extent(0);
  if (w < n_lm_left) {
    const int gl = cpl_p.group_left(w);
    NNScalar s = NNScalar(0.0);
    const int nL = cpl_p.n_left;
    for (int nn = 0; nn < nL; nn++)
      s += cpl_p.U(gl, r, nn) * cpl_in_left(a, nn, w);
    d_cpl_lproj(a, r, w) = s * cpl_p.norm_u;
  }

  const int n_lm_right = (int) cpl_p.group_right.extent(0);
  if (w < n_lm_right) {
    const int gr = cpl_p.group_right(w);
    NNScalar s = NNScalar(0.0);
    const int nR = cpl_p.n_right;
    for (int nn = 0; nn < nR; nn++)
      s += cpl_p.V(gr, r, nn) * cpl_in_right(a, nn, w);
    d_cpl_rproj(a, r, w) = s * cpl_p.norm_v;
  }
}

// ======================================================================
// Kernel: TagCPLCouple — cp_l Pass B. Bilinear Clebsch-Gordan
// coupling on the rank axis + segment-sum over m_sum_ind:
//   out[a,r,:] = 0
//   for c: out[a,r,m_sum_ind[c]] += lproj[a,r,left_ind[c]]
//                                   * rproj[a,r,right_ind[c]] * cg[c]
// Flat index over (atom a, rank r); one thread owns the whole out[a,r,:] row,
// so the segment-sum accumulation is serial within a thread -> no atomics.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagCPLCouple,
    const int& idx) const
{
  const int rank = cpl_p.rank;
  // t_nn_3d is LayoutLeft on CUDA, so atom is the unit-stride dimension.
  // Map adjacent threads to adjacent atoms for coalesced lproj/rproj reads and
  // output writes. Each (a,r) still has one owner and scans CG entries in the
  // same order, so arithmetic and determinism are unchanged.
  const int a = idx % chunk_size;
  const int r = idx / chunk_size;

  const int nfunc = cpl_p.nfunc;
  for (int fcn = 0; fcn < nfunc; fcn++)
    cpl_out(a, r, fcn) = NNScalar(0.0);

  const int n_cg = cpl_p.n_cg;
  for (int c = 0; c < n_cg; c++) {
    const NNScalar v = d_cpl_lproj(a, r, cpl_p.left_ind(c)) *
                       d_cpl_rproj(a, r, cpl_p.right_ind(c)) *
                       cpl_p.cg(c);
    cpl_out(a, r, cpl_p.m_sum_ind(c)) += v;
  }
}

// ======================================================================
// Kernel: TagCPLFused — fused forward cp_l. ONE TEAM per (atom a, rank r).
// The two-pass Project+Couple, previously two global kernels round-tripping lproj/rproj
// [chunk,rank,nlm] through DRAM and doing an O(n_cg) global read-modify-write into the
// out[a,r,:] row, is fused: lproj/rproj and the out row live in TEAM SHARED memory. The
// couple's segment-sum uses SHARED atomics (fast) and out is written to global ONCE.
// Diagnosis (PROFILE.md): CPL kernels are memory-bandwidth-bound (REG:40/STACK:0, high
// occupancy), so cutting the lproj/rproj round-trip + the n_cg out-RMW is the lever.
// Shared layout: [ lproj_sh(nlm) | rproj_sh(nlm) | out_sh(nfunc) ] NNScalars.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagCPLFused,
    const typename Kokkos::TeamPolicy<DeviceType, TagCPLFused>::member_type& team) const
{
  const int rank = cpl_p.rank;
  const int a = team.league_rank() / rank;
  const int r = team.league_rank() - a * rank;
  const int nlm = cpl_nlm;
  const int nfunc = cpl_p.nfunc;
  const int n_lm_left = (int) cpl_p.group_left.extent(0);
  const int n_lm_right = (int) cpl_p.group_right.extent(0);
  const int nL = cpl_p.n_left;
  const int nR = cpl_p.n_right;

  NNScalar* sh = (NNScalar*) team.team_shmem().get_shmem((2 * nlm + nfunc) * sizeof(NNScalar), 0);
  NNScalar* lproj_sh = sh;
  NNScalar* rproj_sh = sh + nlm;
  NNScalar* out_sh   = sh + 2 * nlm;

  // Pass A (project) into shared: lproj_sh[w] = norm_u*Σ_n U[gl(w),r,n]*left[a,n,w].
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlm), [&] (const int w) {
    NNScalar sl = NNScalar(0.0);
    if (w < n_lm_left) {
      const int gl = cpl_p.group_left(w);
      for (int nn = 0; nn < nL; nn++) sl += cpl_p.U(gl, r, nn) * cpl_in_left(a, nn, w);
      sl *= cpl_p.norm_u;
    }
    lproj_sh[w] = sl;
    NNScalar sr = NNScalar(0.0);
    if (w < n_lm_right) {
      const int gr = cpl_p.group_right(w);
      for (int nn = 0; nn < nR; nn++) sr += cpl_p.V(gr, r, nn) * cpl_in_right(a, nn, w);
      sr *= cpl_p.norm_v;
    }
    rproj_sh[w] = sr;
  });
  // Zero the out row in shared.
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nfunc), [&] (const int fcn) {
    out_sh[fcn] = NNScalar(0.0);
  });
  team.team_barrier();

  // Pass B (couple): out_sh[m_sum_ind[c]] += lproj_sh[left_ind[c]]*rproj_sh[right_ind[c]]*cg[c]
  // via shared atomics (segment-sum over the CG connections, split across the team).
  // the three indices come packed in one int32 (8 B/entry with cg vs 16 B).
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, cpl_p.n_cg), [&] (const int c) {
    const int pk = cpl_p.cg_packed(c);
    const NNScalar v = lproj_sh[pk & 255] * rproj_sh[(pk >> 8) & 255] * cpl_p.cg(c);
    Kokkos::atomic_add(&out_sh[pk >> 16], v);
  });
  team.team_barrier();

  // Write the out row to global ONCE.
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nfunc), [&] (const int fcn) {
    cpl_out(a, r, fcn) = out_sh[fcn];
  });
}

// ======================================================================
// compute_cp_l — generic cp_l GeneralProductFunction product.
// Sets the per-call launch state (weights + in/out views) captured by the TagCPL*
// functors, then launches the fused team kernel (one team per (a,r)) over the chunk.
// Reused verbatim by every A*_* cp_l product (A1_2, A1_3, ... A3_2).
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_cp_l(
    const DeviceCPL &p, const t_nn_3d &in_left, const t_nn_3d &in_right,
    const t_nn_3d &out)
{
  cpl_p = p;
  cpl_in_left = in_left;
  cpl_in_right = in_right;
  cpl_out = out;
  const int n_lm_left = (int) p.group_left.extent(0);
  const int n_lm_right = (int) p.group_right.extent(0);
  cpl_nlm = (n_lm_left > n_lm_right) ? n_lm_left : n_lm_right;
  cpl_bind(p);

#ifdef GRACE_KK_GEMM
  // fp32 only: un-fuse. BLAS U/V rank-projection into the global lproj/rproj
  // scratch, then the couple reads them back. The projection is ~78% of the fused cp_l
  // cost and is the mirror of the backward projection GEMM (reuses p.Uw/p.Vw).
  // fp64 and builds without a BLAS handle keep the fused shared-memory kernel below (no SGEMM benefit, and
  // un-fusing reintroduces the lproj/rproj round-trip that the fused kernel avoids).
  if (blas_handle && sizeof(NNScalar) == 4) {
    compute_cp_l_project_blas(p, in_left, in_right);
    cpl_stamp[&p - d_prod] = cpl_stamp_now();
    cpl_couple_gather(p, out);
    return;
  }
#endif

  // Fused Project+Couple, one team per (a,r); lproj/rproj + out row in team shared.
  int team_size = 1, vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1) team_size = 128;
  const int league = chunk_size * p.rank;
  int scratch_size = scratch_bytes<DeviceType, NNScalar>(2 * cpl_nlm + p.nfunc);
  team_size = fit_team_size<DeviceType, TagCPLFused>(*this, league, team_size, vector_length);
  auto policy = Kokkos::TeamPolicy<DeviceType, TagCPLFused>(league, team_size, vector_length);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
  Kokkos::parallel_for("CPLFused", policy, *this);
}

#ifdef GRACE_KK_GEMM
// BLAS cp_l forward U/V rank-projection into the global lproj/rproj scratch.
//   lproj(a,r,w) = norm_u * Σ_n in_left(a,n,w) * U(gl(w),r,n)   (batch over w = lm)
//   rproj(a,r,w) = norm_v * Σ_n in_right(a,n,w) * V(gr(w),r,n)
// The LayoutLeft w-slabs in_left(:,:,w)/lproj(:,:,w) are column-major [chunk x n]/[chunk x
// rank] (ld=extent(0), w-stride=extent(0)*extent(1)); p.Uw[w] is the pre-gathered col-major
// [rank x n_left] weight (ldb=rank, w-stride=n_left*rank) built at setup. OP_N/OP_T so
// op(Uw)=[n_left x rank] gives C(a,r)=Σ_n in_left(a,n)*Uw(r,n). beta=0 (full overwrite).
// True fp32 (PEDANTIC handle).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_cp_l_project_blas(
    const DeviceCPL &p, const t_nn_3d &in_left, const t_nn_3d &in_right)
{
  const int n_lm_left  = (int) p.group_left.extent(0);
  const int n_lm_right = (int) p.group_right.extent(0);
  const float beta = 0.0f;
  const int ldc = (int) d_cpl_lproj.stride(1);
  const long long sc = (long long) d_cpl_lproj.stride(2);
  if (p.lr_ok && in_left.data() == in_right.data() && !getenv("GRACE3L_CPL_LR_SPLIT")) {
    // both sides from the same input: one GEMM into ranks 0 .. 2 rank - 1 of the slot
    const float alpha = (float) p.lr_alpha;
    blas::status_t st = blas::sgemm_strided_batched(blas_handle, blas::OP_N, blas::OP_T,
        chunk_size, 2 * p.rank, p.n_left, &alpha,
        (const float*) in_left.data(), (int) in_left.extent(0),
        (long long) in_left.extent(0) * in_left.extent(1),
        (const float*) p.Wlr.data(), 2 * p.rank, (long long) p.n_left * 2 * p.rank,
        &beta, (float*) d_cpl_lproj.data(), ldc, sc, n_lm_left);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-3L/KK: cp_l fwd merged projection GEMM ({}) failed (status {})", blas::name(), (int) st);
    return;
  }
  if (n_lm_left > 0 && p.n_left > 0) {
    const float alpha = (float) p.norm_u;
    blas::status_t st = blas::sgemm_strided_batched(blas_handle, blas::OP_N, blas::OP_T,
        chunk_size, p.rank, p.n_left, &alpha,
        (const float*) in_left.data(), (int) in_left.extent(0),
        (long long) in_left.extent(0) * in_left.extent(1),
        (const float*) p.Uw.data(), p.rank, (long long) p.n_left * p.rank,
        &beta, (float*) d_cpl_lproj.data(), ldc, sc, n_lm_left);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-3L/KK: cp_l fwd left projection GEMM ({}) failed (status {})", blas::name(), (int) st);
  }
  if (n_lm_right > 0 && p.n_right > 0) {
    const float alpha = (float) p.norm_v;
    blas::status_t st = blas::sgemm_strided_batched(blas_handle, blas::OP_N, blas::OP_T,
        chunk_size, p.rank, p.n_right, &alpha,
        (const float*) in_right.data(), (int) in_right.extent(0),
        (long long) in_right.extent(0) * in_right.extent(1),
        (const float*) p.Vw.data(), p.rank, (long long) p.n_right * p.rank,
        &beta, (float*) d_cpl_rproj.data(), ldc, sc, n_lm_right);
    if (st != blas::SUCCESS)
      GRACE_KK_ERROR("GRACE-3L/KK: cp_l fwd right projection GEMM ({}) failed (status {})", blas::name(), (int) st);
  }
}
#endif

// ---- TagCPLBwdCouple: VJP of the cp_l CG couple (Pass B). One thread per (a,r);
// recomputed lproj/rproj live in d_cpl_lproj/d_cpl_rproj (forward Pass A re-run
// just before). adj_p[c]=adj_out[a,r,m_sum_ind[c]]*cg[c]; adj_lproj[left_ind[c]] +=
// adj_p*rproj[right_ind[c]]; adj_rproj[right_ind[c]] += adj_p*lproj[left_ind[c]].
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagCPLBwdCouple,
    const int& idx) const
{
  const int rank = cpl_p.rank;
  const int a = idx / rank;
  const int r = idx - a * rank;
  const int nlm = cpl_nlm;

  for (int w = 0; w < nlm; w++) {
    d_cpl_lproj_adj(a, r, w) = NNScalar(0.0);
    d_cpl_rproj_adj(a, r, w) = NNScalar(0.0);
  }
  const int n_cg = cpl_p.n_cg;
  for (int c = 0; c < n_cg; c++) {
    const int li = cpl_p.left_ind(c);
    const int ri = cpl_p.right_ind(c);
    const int fsum = cpl_p.m_sum_ind(c);
    const NNScalar d = cpl_adj_out(a, r, fsum) * cpl_p.cg(c);
    d_cpl_lproj_adj(a, r, li) += d * d_cpl_rproj(a, r, ri);
    d_cpl_rproj_adj(a, r, ri) += d * d_cpl_lproj(a, r, li);
  }
}

// ---- TagCPLBwdProject: VJP of the cp_l U/V projection (Pass A). One thread per
// (a,n,w). adj_left[a,n,w] += norm_u*Σ_r U[gl[w],r,n]*adj_lproj[a,r,w]; adj_right
// analogous with V/gr/adj_rproj. For SELF-products adj_left==adj_right -> the two
// += accumulate into the one buffer (serial per (a,n,w) thread).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagCPLBwdProject,
    const int& idx) const
{
  const int nmax = cpl_bwd_nmax;
  const int nlm = cpl_nlm;
  const int rank = cpl_p.rank;
  const int a = idx / (nmax * nlm);
  const int rem = idx - a * (nmax * nlm);
  const int n = rem / nlm;
  const int w = rem - n * nlm;

  const int n_lm_left = (int) cpl_p.group_left.extent(0);
  if (w < n_lm_left && n < cpl_p.n_left) {
    const int gl = cpl_p.group_left(w);
    NNScalar s = NNScalar(0.0);
    for (int r = 0; r < rank; r++)
      s += cpl_p.U(gl, r, n) * d_cpl_lproj_adj(a, r, w);
    cpl_adj_left(a, n, w) += s * cpl_p.norm_u;
  }
  const int n_lm_right = (int) cpl_p.group_right.extent(0);
  if (w < n_lm_right && n < cpl_p.n_right) {
    const int gr = cpl_p.group_right(w);
    NNScalar s = NNScalar(0.0);
    for (int r = 0; r < rank; r++)
      s += cpl_p.V(gr, r, n) * d_cpl_rproj_adj(a, r, w);
    cpl_adj_right(a, n, w) += s * cpl_p.norm_v;
  }
}

// ======================================================================
// Kernel: TagCPLBwdFused — fused backward cp_l passes 1+2. ONE TEAM per (a,r).
// Replaces the separate CPLBwdRecomputeProject (global lproj/rproj) + CPLBwdCouple
// (n_cg re-reads of the big adj_out[a,r,:] row + n_cg global RMW into the small adj
// rows). Here lproj/rproj are RECOMPUTED into shared, adj_out[a,r,:] is loaded ONCE into
// shared, and the couple-VJP accumulates lproj_adj/rproj_adj in shared (shared atomics),
// written to global ONCE. (Pass 3, TagCPLBwdProject, stays separate — it reduces over r.)
// Shared: [ lproj_sh(nlm) | rproj_sh(nlm) | lproj_adj_sh(nlm) | rproj_adj_sh(nlm) | adj_out_sh(nfunc) ].
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagCPLBwdFused,
    const typename Kokkos::TeamPolicy<DeviceType, TagCPLBwdFused>::member_type& team) const
{
  const int rank = cpl_p.rank;
  const int a = team.league_rank() / rank;
  const int r = team.league_rank() - a * rank;
  const int nlm = cpl_nlm;
  const int nfunc = cpl_p.nfunc;
  const int n_lm_left = (int) cpl_p.group_left.extent(0);
  const int n_lm_right = (int) cpl_p.group_right.extent(0);
  const int nL = cpl_p.n_left;
  const int nR = cpl_p.n_right;

  NNScalar* sh = (NNScalar*) team.team_shmem().get_shmem((4 * nlm + nfunc) * sizeof(NNScalar), 0);
  NNScalar* lproj_sh     = sh;
  NNScalar* rproj_sh     = sh + nlm;
  NNScalar* lproj_adj_sh = sh + 2 * nlm;
  NNScalar* rproj_adj_sh = sh + 3 * nlm;
  NNScalar* adj_out_sh   = sh + 4 * nlm;

  // Load the BLAS projections when available; otherwise recompute forward
  // Pass A into shared. The VJP below remains fused/shared in both cases.
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlm), [&] (const int w) {
    NNScalar sl = NNScalar(0.0);
    if (w < n_lm_left) {
      if (cpl_bwd_projected) {
        sl = d_cpl_lproj(a, r, w);
      } else {
        const int gl = cpl_p.group_left(w);
        for (int nn = 0; nn < nL; nn++) sl += cpl_p.U(gl, r, nn) * cpl_in_left(a, nn, w);
        sl *= cpl_p.norm_u;
      }
    }
    lproj_sh[w] = sl; lproj_adj_sh[w] = NNScalar(0.0);
    NNScalar sr = NNScalar(0.0);
    if (w < n_lm_right) {
      if (cpl_bwd_projected) {
        sr = d_cpl_rproj(a, r, w);
      } else {
        const int gr = cpl_p.group_right(w);
        for (int nn = 0; nn < nR; nn++) sr += cpl_p.V(gr, r, nn) * cpl_in_right(a, nn, w);
        sr *= cpl_p.norm_v;
      }
    }
    rproj_sh[w] = sr; rproj_adj_sh[w] = NNScalar(0.0);
  });
  // Load the adj_out[a,r,:] row into shared once.
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nfunc), [&] (const int fcn) {
    adj_out_sh[fcn] = cpl_adj_out(a, r, fcn);
  });
  team.team_barrier();

  // Couple VJP: adj_p = adj_out_sh[m_sum_ind[c]]*cg[c];
  //   lproj_adj_sh[left_ind[c]] += adj_p*rproj_sh[right_ind[c]];
  //   rproj_adj_sh[right_ind[c]] += adj_p*lproj_sh[left_ind[c]]  (shared atomics).
  // the three indices come packed in one int32 (8 B/entry with cg vs 16 B).
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, cpl_p.n_cg), [&] (const int c) {
    const int pk = cpl_p.cg_packed(c);
    const int li = pk & 255;
    const int ri = (pk >> 8) & 255;
    const NNScalar d = adj_out_sh[pk >> 16] * cpl_p.cg(c);
    Kokkos::atomic_add(&lproj_adj_sh[li], d * rproj_sh[ri]);
    Kokkos::atomic_add(&rproj_adj_sh[ri], d * lproj_sh[li]);
  });
  team.team_barrier();

  // Write the adj rank-projections to global ONCE (consumed by TagCPLBwdProject).
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlm), [&] (const int w) {
    d_cpl_lproj_adj(a, r, w) = lproj_adj_sh[w];
    d_cpl_rproj_adj(a, r, w) = rproj_adj_sh[w];
  });
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_cp_l_bwd(
    const DeviceCPL &p, const t_nn_3d &in_left, const t_nn_3d &in_right,
    const t_nn_3d &adj_out, const t_nn_3d &adj_left, const t_nn_3d &adj_right)
{
  cpl_p = p;
  cpl_in_left = in_left;
  cpl_in_right = in_right;
  cpl_adj_out = adj_out;
  cpl_adj_left = adj_left;
  cpl_adj_right = adj_right;
  const int n_lm_left = (int) p.group_left.extent(0);
  const int n_lm_right = (int) p.group_right.extent(0);
  cpl_nlm = (n_lm_left > n_lm_right) ? n_lm_left : n_lm_right;
  cpl_bwd_nmax = (p.n_left > p.n_right) ? p.n_left : p.n_right;
  cpl_bwd_projected = 0;
  cpl_bind(p);

  // 1+2) fused recompute-project + couple-VJP (team per (a,r); lproj/rproj + adj_out row
  //      + adj accumulators in shared) -> d_cpl_lproj_adj/d_cpl_rproj_adj.
#ifdef GRACE_KK_GEMM
  // Use the same true-fp32 BLAS projections as the forward path, but keep the
  // VJP fused in shared memory. This avoids the hand projection's strided
  // LayoutLeft traversal without reintroducing global scattered VJP updates.
  if (blas_handle && sizeof(NNScalar) == 4) {
    // the forward projections of this chunk and evaluation are still in the slot unless recomputation is forced
    static const bool reproject = getenv("GRACE3L_CPL_REPROJECT") != nullptr;
    if (reproject || cpl_stamp[&p - d_prod] != cpl_stamp_now()) {
      compute_cp_l_project_blas(p, in_left, in_right);
      cpl_stamp[&p - d_prod] = cpl_stamp_now();
    }
    cpl_bwd_projected = 1;
    cpl_couple_vjp_gather(p, adj_out);
  }
#endif
  if (!cpl_bwd_projected) {
    int team_size = 1, vector_length = 1;
    if (Kokkos::DefaultExecutionSpace().concurrency() > 1) team_size = 128;
    const int league = chunk_size * p.rank;
    int scratch_size = scratch_bytes<DeviceType, NNScalar>(4 * cpl_nlm + p.nfunc);
    team_size = fit_team_size<DeviceType, TagCPLBwdFused>(*this, league, team_size, vector_length);
    auto policy = Kokkos::TeamPolicy<DeviceType, TagCPLBwdFused>(league, team_size, vector_length);
    policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
    Kokkos::parallel_for("CPLBwdFused", policy, *this);
  }
  // 3) project bwd -> adj_left/adj_right (accumulate; reduces over r, stays separate).
#ifdef GRACE_KK_GEMM
  // two batched col-major GEMMs (left with Uw, right with Vw), each beta=1
  // into the shared adj buffer. The LayoutLeft slabs A=lproj_adj(:,:,w) / C=adj(:,:,w)
  // are col-major with ld=extent(0), w-stride=extent(0)*extent(1) -> no scatter needed.
  // For SELF-products cpl_adj_left aliases cpl_adj_right; the two GEMMs run sequentially
  // on the SAME (default-instance) stream, so the beta=1 accumulation is race-free.
  if (blas_handle && sizeof(NNScalar) == 4) {
    const long strideA = (long) d_cpl_lproj_adj.stride(2);
    const int  ldA     = (int)  d_cpl_lproj_adj.stride(1);
    if (p.lr_ok && cpl_in_left.data() == cpl_in_right.data() && cpl_adj_left.data() == cpl_adj_right.data() &&
        !getenv("GRACE3L_CPL_LR_SPLIT")) {
      // self-product: adj_in += [dL | dR] [Uw; Vw] in one GEMM (k = 2 rank)
      cpl_project_sgemm_batched(
          d_cpl_lproj_adj.data(), strideA, ldA,
          p.Wlr.data(), p.n_left, 2 * p.rank,
          cpl_adj_left.data(),
          (long) cpl_adj_left.extent(0) * cpl_adj_left.extent(1),
          (int)  cpl_adj_left.extent(0),
          chunk_size, n_lm_left, p.lr_alpha);
      return;
    }
    cpl_project_sgemm_batched(
        d_cpl_lproj_adj.data(), strideA, ldA,
        p.Uw.data(), p.n_left, p.rank,
        cpl_adj_left.data(),
        (long) cpl_adj_left.extent(0) * cpl_adj_left.extent(1),
        (int)  cpl_adj_left.extent(0),
        chunk_size, n_lm_left, p.norm_u);
    cpl_project_sgemm_batched(
        d_cpl_rproj_adj.data(), strideA, ldA,
        p.Vw.data(), p.n_right, p.rank,
        cpl_adj_right.data(),
        (long) cpl_adj_right.extent(0) * cpl_adj_right.extent(1),
        (int)  cpl_adj_right.extent(0),
        chunk_size, n_lm_right, p.norm_v);
    return;
  }
#endif
  Kokkos::parallel_for("CPLBwdProject",
      Kokkos::RangePolicy<DeviceType, TagCPLBwdProject>(0, chunk_size * cpl_bwd_nmax * cpl_nlm), *this);
}

// ======================================================================
// fc_fold: the FC A*_2a / A*_2b folded into the cp_l projection of the product it feeds (see grace_3l_kk.h).
// Weights from the model's fp64 data, rounded once to fp32:
//   P[w](k, j) = norm_u U(gl(w), j, k)   (j < rank),   norm_v V(gr(w), j - rank, k)   (rank <= j < 2 rank, self)
//   UL[w](k, j) = nof[w] P[w](k, j),   UR[w](n, j) = norm_right nof[w] sum_k w_right[k, n, tile(w)] P[w](k, j)
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::build_fc_fold()
{
  for (int k = 0; k < 3; k++) fold_on[k] = 0;
  static const bool env = getenv("GRACE3L_FC_FOLD") && atoi(getenv("GRACE3L_FC_FOLD")) != 0;
  if (!(fc_fold || env)) return;
  bool ok = sizeof(NNScalar) == 4;
#ifdef GRACE_KK_GEMM
  ok = ok && blas_handle;
#else
  ok = false;
#endif
  if (!ok) {
    log.info(fmt("[GRACE-3L/KK] fc_fold needs the GPU BLAS fp32 path: FC not folded\n"));
    return;
  }
#ifdef GRACE_KK_GEMM
  const auto &m = *grace_model;
  auto build = [&](int pi, int fi, bool self) -> bool {
    if (pi < 0 || fi < 0) return false;
    const auto &c = m.prod[pi];
    const auto &f = m.fc[fi];
    const int nlm = (int) c.group_left.size();
    if (!d_fc[fi].gemm_ok || f.n_out != c.n_left || nlm != f.n_funcs_left || c.rank <= 0) return false;
    if (self && (!d_prod[pi].lr_ok || (int) c.group_right.size() != nlm || c.n_right != c.n_left)) return false;
    const int rk = c.rank, nr = self ? 2 * rk : rk, no = f.n_out, ni = f.n_in_right, nt = f.w_shape_right;
    MemKK::realloc_kokkos(d_fold_UL[pi], "g3l:fold_UL", nlm, no, nr);
    MemKK::realloc_kokkos(d_fold_UR[pi], "g3l:fold_UR", nlm, ni, nr);
    auto hL = Kokkos::create_mirror_view(d_fold_UL[pi]);
    auto hR = Kokkos::create_mirror_view(d_fold_UR[pi]);
    std::vector<double> P((size_t) no * nr);
    for (int w = 0; w < nlm; w++) {
      const int gl = c.group_left[w], gr = self ? c.group_right[w] : 0;
      for (int k = 0; k < no; k++)
        for (int j = 0; j < nr; j++)
          P[(size_t) k * nr + j] = j < rk ? c.norm_u * c.U[((size_t) gl * rk + j) * c.n_left + k]
                                          : c.norm_v * c.V[((size_t) gr * rk + j - rk) * c.n_right + k];
      const double nof = f.norm_out_factor[w], sr = f.norm_right * nof;
      const int t = f.w_tile_right[w];
      for (int k = 0; k < no; k++)
        for (int j = 0; j < nr; j++) hL(w, k, j) = (NNScalar) (nof * P[(size_t) k * nr + j]);
      for (int n = 0; n < ni; n++)
        for (int j = 0; j < nr; j++) {
          double s = 0.0;
          for (int k = 0; k < no; k++) s += f.w_right[((size_t) k * ni + n) * nt + t] * P[(size_t) k * nr + j];
          hR(w, n, j) = (NNScalar) (sr * s);
        }
    }
    Kokkos::deep_copy(d_fold_UL[pi], hL);
    Kokkos::deep_copy(d_fold_UR[pi], hR);
    return true;
  };
  const int pa[3] = {idx_prod_A1_3, idx_prod_A2_3, idx_prod_A3_3}, pb[3] = {idx_prod_A1_4, idx_prod_A2_4, idx_prod_A3_4};
  const int fa[3] = {idx_fc_A1_2a, idx_fc_A2_2a, idx_fc_A3_2a}, fb[3] = {idx_fc_A1_2b, idx_fc_A2_2b, idx_fc_A3_2b};
  for (int k = 0; k < 3; k++) fold_on[k] = build(pa[k], fa[k], false) && build(pb[k], fb[k], true);
  log.info(fmt("[GRACE-3L/KK] FC folded into the cp_l projections: L1 {} L2 {} L3 {}\n", fold_on[0], fold_on[1],
               fold_on[2]));
#endif
}

#ifdef GRACE_KK_GEMM
// lproj (or [lproj | rproj] of a self-product) = Lred UL^T + R UR^T; a non-self product projects its right input as
// compute_cp_l_project_blas does
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::cpl_fold_project(
    const DeviceCPL &p, const t_nn_3d &Lred, const t_nn_3d &R, const t_nn_3d &right)
{
  const int pi = (int) (&p - d_prod);
  const auto &UL = d_fold_UL[pi], &UR = d_fold_UR[pi];
  const int nlm = (int) UL.extent(0), no = (int) UL.extent(1), ni = (int) UR.extent(1), nr = (int) UL.extent(2);
  const int ldc = (int) d_cpl_lproj.stride(1);
  const long long sc = (long long) d_cpl_lproj.stride(2);
  const float one = 1.0f, zero = 0.0f;
  blas::status_t st = blas::sgemm_strided_batched(blas_handle, blas::OP_N, blas::OP_T, chunk_size, nr, no, &one,
      (const float *) Lred.data(), (int) Lred.extent(0), (long long) Lred.extent(0) * Lred.extent(1),
      (const float *) UL.data(), nr, (long long) no * nr, &zero, (float *) d_cpl_lproj.data(), ldc, sc, nlm);
  if (st == blas::SUCCESS)
    st = blas::sgemm_strided_batched(blas_handle, blas::OP_N, blas::OP_T, chunk_size, nr, ni, &one,
        (const float *) R.data(), (int) R.extent(0), (long long) R.extent(0) * R.extent(1),
        (const float *) UR.data(), nr, (long long) ni * nr, &one, (float *) d_cpl_lproj.data(), ldc, sc, nlm);
  if (st == blas::SUCCESS && nr == p.rank) {
    const int nlm_r = (int) p.group_right.extent(0);
    const float alpha = (float) p.norm_v;
    if (nlm_r > 0 && p.n_right > 0)
      st = blas::sgemm_strided_batched(blas_handle, blas::OP_N, blas::OP_T, chunk_size, p.rank, p.n_right, &alpha,
          (const float *) right.data(), (int) right.extent(0), (long long) right.extent(0) * right.extent(1),
          (const float *) p.Vw.data(), p.rank, (long long) p.n_right * p.rank, &zero, (float *) d_cpl_rproj.data(),
          ldc, sc, nlm_r);
  }
  if (st != blas::SUCCESS)
    GRACE_KK_ERROR("GRACE-3L/KK: cp_l folded projection GEMM ({}) failed (status {})", blas::name(), (int) st);
}
#endif

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_cp_l_fold(
    const DeviceCPL &p, const t_nn_3d &Lred, const t_nn_3d &R, const t_nn_3d &right, const t_nn_3d &out)
{
#ifdef GRACE_KK_GEMM
  cpl_p = p;
  cpl_out = out;
  cpl_nlm = std::max((int) p.group_left.extent(0), (int) p.group_right.extent(0));
  cpl_bind(p);
  cpl_fold_project(p, Lred, R, right);
  cpl_stamp[&p - d_prod] = cpl_stamp_now();
  cpl_couple_gather(p, out);
#else
  (void) p; (void) Lred; (void) R; (void) right; (void) out;
  GRACE_KK_ERROR("GRACE-3L/KK: fc_fold needs the GPU BLAS path");
#endif
}

// VJP of cp_l(fc(Lred, R), right) (right empty: the self-product cp_l(fc, fc)): adj_Lred += A UL, adj_R += A UR with
// A = lproj_adj ([lproj_adj | rproj_adj] for the self-product), adj_right += norm_v rproj_adj Vw otherwise
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_cp_l_fold_bwd(
    const DeviceCPL &p, const t_nn_3d &Lred, const t_nn_3d &R, const t_nn_3d &right, const t_nn_3d &adj_out,
    const t_nn_3d &adj_Lred, const t_nn_3d &adj_R, const t_nn_3d &adj_right)
{
#ifdef GRACE_KK_GEMM
  cpl_p = p;
  cpl_adj_out = adj_out;
  cpl_nlm = std::max((int) p.group_left.extent(0), (int) p.group_right.extent(0));
  cpl_bwd_nmax = std::max(p.n_left, p.n_right);
  cpl_bind(p);
  static const bool reproject = getenv("GRACE3L_CPL_REPROJECT") != nullptr;
  if (reproject || cpl_stamp[&p - d_prod] != cpl_stamp_now()) {
    cpl_fold_project(p, Lred, R, right);
    cpl_stamp[&p - d_prod] = cpl_stamp_now();
  }
  cpl_bwd_projected = 1;
  cpl_couple_vjp_gather(p, adj_out);
  const int pi = (int) (&p - d_prod);
  const auto &UL = d_fold_UL[pi], &UR = d_fold_UR[pi];
  const int nlm = (int) UL.extent(0), nr = (int) UL.extent(2);
  const long strideA = (long) d_cpl_lproj_adj.stride(2);
  const int ldA = (int) d_cpl_lproj_adj.stride(1);
  cpl_project_sgemm_batched(d_cpl_lproj_adj.data(), strideA, ldA, UL.data(), (int) UL.extent(1), nr, adj_Lred.data(),
                            (long) adj_Lred.extent(0) * adj_Lred.extent(1), (int) adj_Lred.extent(0), chunk_size, nlm,
                            NNScalar(1));
  cpl_project_sgemm_batched(d_cpl_lproj_adj.data(), strideA, ldA, UR.data(), (int) UR.extent(1), nr, adj_R.data(),
                            (long) adj_R.extent(0) * adj_R.extent(1), (int) adj_R.extent(0), chunk_size, nlm,
                            NNScalar(1));
  if (nr == p.rank)
    cpl_project_sgemm_batched(d_cpl_rproj_adj.data(), strideA, ldA, p.Vw.data(), p.n_right, p.rank, adj_right.data(),
                              (long) adj_right.extent(0) * adj_right.extent(1), (int) adj_right.extent(0), chunk_size,
                              (int) p.group_right.extent(0), p.norm_v);
#else
  (void) p; (void) Lred; (void) R; (void) right; (void) adj_out; (void) adj_Lred; (void) adj_R; (void) adj_right;
  GRACE_KK_ERROR("GRACE-3L/KK: fc_fold needs the GPU BLAS path");
#endif
}

// ======================================================================
// cp_l couple on the rank axis, thread per (atom a, rank r), 64 consecutive atoms per team (LayoutLeft rows
// coalesced); every thread keeps its own rows in a column of the team scratch (stride 64, no barriers):
//   forward  out[a,r,f] = sum_{t in f} cg_t lproj[a,r,l_t] rproj[a,r,r_t]     (plain thread per (a, r))
//   VJP      one pass in output order (adj_out read once, next value prefetched), accumulators in scratch:
//            lproj_adj[l_t] += cg_t adj_out[f_t] rproj[r_t],  rproj_adj[r_t] += cg_t adj_out[f_t] lproj[l_t]
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::cpl_couple_gather(const DeviceCPL &p, const t_nn_3d &out)
{
  if (cpl_path && p.path_n > 0 && sizeof(NNScalar) == 4) { cpl_couple_path(p, out); return; }
  // forward: rows read through L1 (a scratch copy of the rows lowered occupancy and was slower)
  const int cs = chunk_size, nfunc = p.nfunc;
  auto L = d_cpl_lproj; auto R = d_cpl_rproj; auto O = out;
  auto off = p.foff; auto ia = p.fa; auto ib = p.fb; auto cg = p.fcg; auto fcol = p.fcol;
  Kokkos::parallel_for("CPLCoupleG", Kokkos::RangePolicy<DeviceType, hip_lb_t<DeviceType, 256>>(0, cs * p.rank), KOKKOS_LAMBDA(const int q) {
    const int r = q / cs, a = q - r * cs;
    for (int f = 0; f < nfunc; f++) {
      NNScalar v = NNScalar(0.0);
      for (int e = off(f); e < off(f + 1); e++) v += cg(e) * L(a, r, ia(e)) * R(a, r, ib(e));
      O(a, r, fcol(f)) = v;
    }
  });
}

// Couple by coupling path (grace_3l_cpl_path.h): thread per (atom, rank), operand blocks and adjoint blocks in
// registers, the term pattern fixed at compile time per (l1, l2, L), cg values and the path lists from the model.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::cpl_couple_path(const DeviceCPL &p, const t_nn_3d &out)
{
  const int cs = chunk_size, nob = p.path_nob[0];
  auto L = d_cpl_lproj; auto R = d_cpl_rproj; auto O = out; auto fcol = p.fcol;
  auto ob = p.path_ob[0]; auto pr = p.path_pr[0]; auto pt = p.path_pt[0]; auto pcg = p.path_cg;
  const long sl = (long) L.stride(2), sr = (long) R.stride(2), so = (long) O.stride(2);
  const int rw = cpl_row_width();    // thread q -> (atom a < cs, rank r)
  Kokkos::parallel_for("CPLCouplePath", Kokkos::RangePolicy<DeviceType, hip_lb_t<DeviceType, 256>>(0, rw * p.rank), KOKKOS_LAMBDA(const int q) {
    const int r = q / rw, a = q - r * rw;
    if (a >= cs) return;
    const G3LPath::Tabs tb{ob.data(), pr.data(), pt.data(), reinterpret_cast<const float *>(pcg.data()), fcol.data(), nob};
    G3LPath::run<0>(tb, reinterpret_cast<const float *>(&L(a, r, 0)), sl, nullptr, 0,
                    reinterpret_cast<const float *>(&R(a, r, 0)), sr, reinterpret_cast<float *>(&O(a, r, 0)), so,
                    nullptr, 0);
  });
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::cpl_vjp_path(const DeviceCPL &p, const t_nn_3d &adj_out)
{
  const int cs = chunk_size;
  auto L = d_cpl_lproj; auto R = d_cpl_rproj; auto Dv = adj_out; auto fcol = p.fcol; auto pcg = p.path_cg;
  auto dL = d_cpl_lproj_adj; auto dR = d_cpl_rproj_adj;
  const long sl = (long) L.stride(2), sr = (long) R.stride(2), sd = (long) Dv.stride(2);
  const long sgl = (long) dL.stride(2), sgr = (long) dR.stride(2);
  const int rw = cpl_row_width();    // thread q -> (atom a < cs, rank r)
  // the one-pass form runs teams of 64 threads: CUDA only (host spaces cap a team at their thread count; on HIP /
  // MI250X it is slower than the two kernels: 3L W-bcc-5k 86.7 vs 75.5 us/atom); elsewhere the two-kernel form
  static const bool two_env = getenv("GRACE3L_CPL_VJP2") != nullptr;
  bool two = two_env;
#ifdef KOKKOS_ENABLE_CUDA
  if (!std::is_same<typename DeviceType::execution_space, Kokkos::Cuda>::value) two = true;
#else
  two = true;
#endif
  if (!two) {
    // both adjoints in one pass over D (left-outer tables): left accumulator in registers, right row in the team
    // scratch (column per thread), copied out at the end
    constexpr int TB = 64;
    const int nr = (int) p.group_right.extent(0), n = rw * p.rank, nob = p.path_nob[0];
    auto ob = p.path_ob[0]; auto pr = p.path_pr[0]; auto pt = p.path_pt[0];
    using team_policy = Kokkos::TeamPolicy<DeviceType>;
    using member = typename team_policy::member_type;
    team_policy pol((n + TB - 1) / TB, TB);
    pol = pol.set_scratch_size(0, Kokkos::PerTeam((size_t) nr * TB * sizeof(float)));
    Kokkos::parallel_for("CPLVjpPathB", pol, KOKKOS_LAMBDA(const member &t) {
      float *sh = (float *) t.team_shmem().get_shmem((size_t) nr * TB * sizeof(float));
      const int tid = t.team_rank(), q = t.league_rank() * TB + tid;
      const int r = q / rw, a = q - r * rw;
      if (q >= n || a >= cs) return;
      float *gr = sh + tid;
      for (int w = 0; w < nr; w++) gr[w * TB] = 0.0f;
      const G3LPath::Tabs tb{ob.data(), pr.data(), pt.data(), reinterpret_cast<const float *>(pcg.data()), fcol.data(), nob};
      G3LPath::run<3>(tb, reinterpret_cast<const float *>(&L(a, r, 0)), sl, reinterpret_cast<float *>(&dL(a, r, 0)), sgl,
                      reinterpret_cast<const float *>(&R(a, r, 0)), sr, nullptr, 0,
                      reinterpret_cast<const float *>(&Dv(a, r, 0)), sd, gr, TB);
      for (int w = 0; w < nr; w++) dR(a, r, w) = gr[w * TB];
    });
    return;
  }
  // gfx90a / gfx94x (512 registers per lane): at least 3 waves per SIMD (with M = 1 the left kernel took 172 VGPRs, i.e.
  // 2 waves): 3L W-bcc-5k both kernels 76.6 -> 65.5 ms, C-diamond-125k 1892 -> 1643 ms per run (4 waves: 128 VGPRs,
  // spills, 69.5 / 1696 ms). Other targets keep M = 1 (3 waves would leave about 84 VGPRs on gfx906 / gfx908). The
  // forward kernel (93 VGPRs, 5 waves) times the same at M = 1, 2 and 4.
  using vjp_lb = hip_lb_t<DeviceType, 256, GRACE_KK_GEMM_MFMA ? 3 : 1>;
  {
    const int nob = p.path_nob[0];
    auto ob = p.path_ob[0]; auto pr = p.path_pr[0]; auto pt = p.path_pt[0];
    Kokkos::parallel_for("CPLVjpPathL", Kokkos::RangePolicy<DeviceType, vjp_lb>(0, rw * p.rank), KOKKOS_LAMBDA(const int q) {
      const int r = q / rw, a = q - r * rw;
      if (a >= cs) return;
      const G3LPath::Tabs tb{ob.data(), pr.data(), pt.data(), reinterpret_cast<const float *>(pcg.data()), fcol.data(), nob};
      G3LPath::run<1>(tb, nullptr, 0, reinterpret_cast<float *>(&dL(a, r, 0)), sgl,
                      reinterpret_cast<const float *>(&R(a, r, 0)), sr, nullptr, 0,
                      reinterpret_cast<const float *>(&Dv(a, r, 0)), sd);
    });
  }
  {
    const int nob = p.path_nob[1];
    auto ob = p.path_ob[1]; auto pr = p.path_pr[1]; auto pt = p.path_pt[1];
    Kokkos::parallel_for("CPLVjpPathR", Kokkos::RangePolicy<DeviceType, vjp_lb>(0, rw * p.rank), KOKKOS_LAMBDA(const int q) {
      const int r = q / rw, a = q - r * rw;
      if (a >= cs) return;
      const G3LPath::Tabs tb{ob.data(), pr.data(), pt.data(), reinterpret_cast<const float *>(pcg.data()), fcol.data(), nob};
      G3LPath::run<2>(tb, nullptr, 0, reinterpret_cast<float *>(&dR(a, r, 0)), sgr,
                      reinterpret_cast<const float *>(&L(a, r, 0)), sl, nullptr, 0,
                      reinterpret_cast<const float *>(&Dv(a, r, 0)), sd);
    });
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
template<int TB, int SIDE>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::cpl_vjp_t(const DeviceCPL &p, const t_nn_3d &adj_out)
{
  // SIDE 0: both accumulators in the team scratch; 1: left only; 2: right only (two passes, half the scratch)
  const int cs = chunk_size, nfunc = p.nfunc, rank = p.rank;
  const int nl = (int) p.group_left.extent(0), nr = (int) p.group_right.extent(0);
  const int nacc = SIDE == 0 ? nl + nr : (SIDE == 1 ? nl : nr);
  const int nab = (cs + TB - 1) / TB;
  auto L = d_cpl_lproj; auto R = d_cpl_rproj; auto Dv = adj_out;
  auto dL = d_cpl_lproj_adj; auto dR = d_cpl_rproj_adj;
  auto off = p.foff; auto ia = p.fa; auto ib = p.fb; auto cg = p.fcg; auto fcol = p.fcol;
  using team_policy = Kokkos::TeamPolicy<DeviceType>;
  using member = typename team_policy::member_type;
  team_policy pol(nab * rank, TB);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam((size_t) nacc * TB * sizeof(NNScalar)));
  Kokkos::parallel_for("CPLVjpS", pol, KOKKOS_LAMBDA(const member &t) {
    NNScalar *sh = (NNScalar *) t.team_shmem().get_shmem((size_t) nacc * TB * sizeof(NNScalar));
    const int tid = t.team_rank(), r = t.league_rank() / nab, a = (t.league_rank() - r * nab) * TB + tid;
    if (a >= cs) return;
    NNScalar *gL = sh + tid, *gR = sh + (SIDE == 0 ? nl : 0) * TB + tid;
    for (int w = 0; w < nacc; w++) sh[w * TB + tid] = NNScalar(0.0);
    NNScalar dn = Dv(a, r, fcol(0));
    for (int f = 0; f < nfunc; f++) {
      const NNScalar d = dn;
      if (f + 1 < nfunc) dn = Dv(a, r, fcol(f + 1));
      if (d == NNScalar(0.0)) continue;
      for (int e = off(f); e < off(f + 1); e++) {
        const int l = ia(e), b = ib(e);
        const NNScalar c = cg(e) * d;
        if (SIDE != 2) gL[l * TB] += c * R(a, r, b);
        if (SIDE != 1) gR[b * TB] += c * L(a, r, l);
      }
    }
    if (SIDE != 2) for (int w = 0; w < nl; w++) dL(a, r, w) = gL[w * TB];
    if (SIDE != 1) for (int w = 0; w < nr; w++) dR(a, r, w) = gR[w * TB];
  });
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::cpl_couple_vjp_gather(const DeviceCPL &p, const t_nn_3d &adj_out)
{
  if (cpl_path && p.path_n > 0 && sizeof(NNScalar) == 4) { cpl_vjp_path(p, adj_out); return; }
  // two passes (left, right accumulators separately): half the scratch, twice the resident teams; faster than one
  // pass with both (3090: -17 %) or teams of 32 / 128
  cpl_vjp_t<64, 1>(p, adj_out);
  cpl_vjp_t<64, 2>(p, adj_out);
}

#ifdef GRACE_KK_GEMM
// batched projection SGEMM for CPLBwdProject. Standard col-major batched GEMM
// (NOT the row-major transpose trick): the LayoutLeft slabs A=lproj_adj(:,:,w) and
// C=adj(:,:,w) are already col-major [M x rank] / [M x n] with leading dim = extent(0)
// and per-batch (w) stride = extent(0)*extent(1). Bw (Uw/Vw) is packed col-major
// [rank x n], ldb=rank, w-stride=n*rank. beta=1 accumulates into the shared adj buffer
// (which already holds contributions from earlier backward stages). PEDANTIC = true fp32.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::cpl_project_sgemm_batched(
    const NNScalar* A, long strideA, int ldA,
    const NNScalar* Bw, int n, int rank, NNScalar* C, long strideC, int ldC,
    int M, int batch, NNScalar alpha)
{
  if (batch <= 0 || n <= 0 || M <= 0 || rank <= 0) return;
  const float a = (float) alpha, beta = 1.0f;
  blas::status_t st = blas::sgemm_strided_batched(blas_handle, blas::OP_N, blas::OP_N,
      M, n, rank, &a,
      (const float*) A,  ldA,  (long long) strideA,
      (const float*) Bw, rank, (long long) n * rank,
      &beta, (float*) C, ldC,  (long long) strideC,
      batch);
  if (st != blas::SUCCESS)
    GRACE_KK_ERROR("GRACE-3L/KK: cp_l projection strided-batched GEMM ({}) failed (status {})", blas::name(), (int) st);
}

#endif

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::cpl_bind(const DeviceCPL &p)
{
  const long ld = cpl_ld;
  const int R = std::max(p.rank, 1);
  const int nlm = std::max(std::max((int) p.group_left.extent(0), (int) p.group_right.extent(0)), 1);
  const Kokkos::LayoutStride ls(ld, 1, R, ld, nlm, ld * 2 * R);
  NNScalar *b = d_cpl_store.data() + cpl_slot_off[&p - d_prod];
  d_cpl_lproj = t_nn_3d_s(b, ls);
  d_cpl_rproj = t_nn_3d_s(b + ld * R, ls);
  NNScalar *g = d_cpl_adjbuf.data();
  d_cpl_lproj_adj = t_nn_3d_s(g, ls);
  d_cpl_rproj_adj = t_nn_3d_s(g + ld * R, ls);
}

// ======================================================================
// Template instantiations (members defined in this unit)
// ======================================================================

}    // namespace kk
}    // namespace grace

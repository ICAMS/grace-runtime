// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Part of grace::kk::Grace3LKK (grace_3l_kk.h), included by grace_3l_kk_impl.h:
// equivariant SPBF A2 / A3 (forward, backward) and the L1 / equivariant geometry forces.
// Moved from the LAMMPS style grace/3l/kk (lammps-ace 5f5ab57aa7, pair_grace_3l_kokkos_spbf.cpp).
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
// Kernel: TagComputeA2 (Task 2.8) — Layer-2 equivariant-indicator SPBF
// assembly, per (central-atom ii, radial-channel n). Reproduces
// numpy_forward_3l.spbf_equivariant EXACTLY for the A2 layer:
//   a_nl[lm_y] = R1_nl[n, l_tile[lm_y]] * Y_bond[lm_y] * env       (no z_tr)
//   bond_I[lm_ind] = indicator[ind_j][n, lm_ind]
//                    + z_proj[mu_j][n] * chem_l0_mask[lm_ind]      (L0 chem inject)
//   prod[lm_y, lm_ind] = inv_avg_n_neigh * Sum_j a_nl[lm_y] * bond_I[lm_ind]
//   A2[n, f] = Sum_p prod_flat[p] * cg_W[p, f]   (p = lm_y * n_lm_ind + lm_ind)
// indicator[ind_j] is read from the forward-comm'd global array d_I_global
// (layout [nall, n=I_n_funcs, lm_ind=I_n_out]); ind_j = d_nearest(ii,jj) is the
// LAMMPS local index of the neighbor (owned OR ghost) — ghost rows are filled
// by comm->forward_comm in compute() before this kernel runs.
// z_proj is the PRECOMPUTED per-element chem projection (chem_linear folded once
// at copy_weights_to_device). One thread owns the whole (ii,n,:) A2 row -> the
// per-atom 4-D product slice for this (ii,n) is a private [nlm_y, n_lm_ind]
// stack accumulator, so no atomics. Team-policy (ii,n) decode mirrors
// TagComputeA1, dividing by n_rad_max (A2's radial ceiling).
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeA2,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeA2>::member_type& team) const
{
  const DeviceSPBF &sp = d_spbf[idx_spbf_A2];   // A2 (equivariant)
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;
  const int n = slot.neigh;
  if (n >= sp.n_rad_max) return;

  const int nlm_y = (lmax + 1) * (lmax + 1);    // Y lm axis (<=25)
  const int n_lm_ind = sp.n_lm_ind;             // indicator lm axis (<=GRACE3L_MAX_SPBF_NLMIND)
  const int nfunc = sp.nfunc;                   // CG-couple output width
  const int P = nlm_y * n_lm_ind;               // flattened prod / cg_W rows

  // per-(ii,n) private 4-D product slice prod[lm_y, lm_ind], row-major.
  NNScalar prod_acc[25 * GRACE3L_MAX_SPBF_NLMIND];
  for (int p = 0; p < P; p++) prod_acc[p] = NNScalar(0.0);

  const int nct = d_ncount(ii);
  for (int jj = 0; jj < nct; jj++) {
    const int mu_j = d_mu_j(ii, jj);
    const int j_global = d_nearest(ii, jj);     // neighbor local index (owned OR ghost)
    const NNScalar env = (NNScalar) d_env(ii, jj);

    NNScalar a_nl[25];
    for (int ly = 0; ly < nlm_y; ly++) {
      const int l = sp.l_tile(ly);
      a_nl[ly] = d_R1_nl(ii, jj, n, l) * (NNScalar) d_Y_bond(ii, jj, ly) * env;
    }

    NNScalar bond_I[GRACE3L_MAX_SPBF_NLMIND];
    const NNScalar zpn = sp.z_proj(mu_j, n);
    for (int lr = 0; lr < n_lm_ind; lr++)
      bond_I[lr] = d_I_global(j_global, n, lr) + zpn * sp.chem_l0_mask(lr);

    for (int ly = 0; ly < nlm_y; ly++) {
      const NNScalar a = a_nl[ly];
      const int base = ly * n_lm_ind;
      for (int lr = 0; lr < n_lm_ind; lr++)
        prod_acc[base + lr] += a * bond_I[lr];
    }
  }

  const NNScalar inv_avg = (NNScalar) sp.inv_avg_n_neigh;
  for (int p = 0; p < P; p++) prod_acc[p] *= inv_avg;

  // dense CG couple: A2[n,f] = Sum_p prod_flat[p] * cg_W[p,f].
  for (int f = 0; f < nfunc; f++) {
    NNScalar s = NNScalar(0.0);
    for (int p = 0; p < P; p++)
      s += prod_acc[p] * sp.cg_W(p, f);
    d_A2(ii, n, f) = s;
  }
}

// ======================================================================
// Kernel: TagComputeA3 (Task 2.10) — Layer-3 equivariant-indicator SPBF
// assembly. IDENTICAL to TagComputeA2 except: (1) uses the A3 SPBF weights;
// (2) reads the indicator from the round-2 global buffer d_eq2_norm_global
// (n_lm_ind = A3.n_lm_ind = 50, parity-doubled vs A2's 25); (3) writes d_A3.
// The stack arrays are unchanged: n_lm_ind=50 <= GRACE3L_MAX_SPBF_NLMIND=64 and
// P = 25*50 = 1250 <= 25*GRACE3L_MAX_SPBF_NLMIND = 1600. Ghost rows of
// d_eq2_norm_global are filled by round-2 comm->forward_comm before this runs.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeA3,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeA3>::member_type& team) const
{
  const DeviceSPBF &sp = d_spbf[idx_spbf_A3];   // A3 (equivariant)
  const BondSlot slot = bond_slot(team, chunk_size);
  if (slot.atom >= chunk_size) return;
  int ii = slot.atom;
  const int n = slot.neigh;
  if (n >= sp.n_rad_max) return;

  const int nlm_y = (lmax + 1) * (lmax + 1);    // Y lm axis (<=25)
  const int n_lm_ind = sp.n_lm_ind;             // indicator lm axis (=50 for A3, <=GRACE3L_MAX_SPBF_NLMIND)
  const int nfunc = sp.nfunc;                   // CG-couple output width
  const int P = nlm_y * n_lm_ind;               // flattened prod / cg_W rows (=1250 for A3)

  // per-(ii,n) private 4-D product slice prod[lm_y, lm_ind], row-major.
  NNScalar prod_acc[25 * GRACE3L_MAX_SPBF_NLMIND];
  for (int p = 0; p < P; p++) prod_acc[p] = NNScalar(0.0);

  const int nct = d_ncount(ii);
  for (int jj = 0; jj < nct; jj++) {
    const int mu_j = d_mu_j(ii, jj);
    const int j_global = d_nearest(ii, jj);     // neighbor local index (owned OR ghost)
    const NNScalar env = (NNScalar) d_env(ii, jj);

    NNScalar a_nl[25];
    for (int ly = 0; ly < nlm_y; ly++) {
      const int l = sp.l_tile(ly);
      a_nl[ly] = d_R1_nl(ii, jj, n, l) * (NNScalar) d_Y_bond(ii, jj, ly) * env;
    }

    NNScalar bond_I[GRACE3L_MAX_SPBF_NLMIND];
    const NNScalar zpn = sp.z_proj(mu_j, n);
    for (int lr = 0; lr < n_lm_ind; lr++)
      bond_I[lr] = d_eq2_norm_global(j_global, n, lr) + zpn * sp.chem_l0_mask(lr);

    for (int ly = 0; ly < nlm_y; ly++) {
      const NNScalar a = a_nl[ly];
      const int base = ly * n_lm_ind;
      for (int lr = 0; lr < n_lm_ind; lr++)
        prod_acc[base + lr] += a * bond_I[lr];
    }
  }

  const NNScalar inv_avg = (NNScalar) sp.inv_avg_n_neigh;
  for (int p = 0; p < P; p++) prod_acc[p] *= inv_avg;

  // dense CG couple: A3[n,f] = Sum_p prod_flat[p] * cg_W[p,f].
  for (int f = 0; f < nfunc; f++) {
    NNScalar s = NNScalar(0.0);
    for (int p = 0; p < P; p++)
      s += prod_acc[p] * sp.cg_W(p, f);
    d_A3(ii, n, f) = s;
  }
}

// ======================================================================
// Kernel: TagComputeSPBFFused (Opt 6) — team+shared replacement for
// TagComputeA2/TagComputeA3. The old kernels gave one thread the whole (ii,n)
// row, so the [nlm_y, n_lm_ind] product slice (P=1250 for A3) lived in a
// per-thread stack array — REG:72 STACK:6768, i.e. a massive local-memory spill
// that made these the #1 hotspot (ComputeA3 ~25.6%). Here ONE TEAM owns one
// (ii,n): the product slice prod_sh[P] lives in TEAM SHARED, filled cooperatively
// over neighbours (a_nl/bond_I staged in shared per neighbour so global loads stay
// minimal — no atomics: each thread owns disjoint prod slots p), then the dense
// CG couple A[n,f]=Σ_p prod_sh[p]*cg_W[p,f] is split across the team (one f per
// thread). Parameterized by launch state so A2 (indicator d_I_global, out d_A2)
// and A3 (indicator d_eq2_norm_global, out d_A3) share one kernel.
// Shared layout: [ prod_sh(P) | anl_sh(nlm_y) | bondI_sh(n_lm_ind) ] NNScalars.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeSPBFFused,
    const typename Kokkos::TeamPolicy<DeviceType, TagComputeSPBFFused>::member_type& team) const
{
  const DeviceSPBF &sp = d_spbf[spbf_fused_idx];
  const int ii = team.league_rank() % chunk_size;   // league = chunk_size * n_rad_max
  const int n  = team.league_rank() / chunk_size;    // -> ii<chunk_size, n<n_rad_max always

  const int nlm_y = (lmax + 1) * (lmax + 1);         // Y lm axis (<=25)
  const int n_lm_ind = sp.n_lm_ind;                  // indicator lm axis (25 for A2, 50 for A3)
  const int nfunc = sp.nfunc;                        // CG-couple output width
  const int P = nlm_y * n_lm_ind;                    // flattened prod / cg_W rows

  NNScalar* sh = (NNScalar*) team.team_shmem().get_shmem((P + nlm_y + n_lm_ind) * sizeof(NNScalar), 0);
  NNScalar* prod_sh  = sh;
  NNScalar* anl_sh   = sh + P;
  NNScalar* bondI_sh = sh + P + nlm_y;

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, P), [&] (const int p) {
    prod_sh[p] = NNScalar(0.0);
  });
  team.team_barrier();

  const int nct = d_ncount(ii);
  for (int jj = 0; jj < nct; jj++) {
    const int mu_j = d_mu_j(ii, jj);
    const int j_global = d_nearest(ii, jj);          // neighbor local index (owned OR ghost)
    const NNScalar env = (NNScalar) d_env(ii, jj);
    const NNScalar zpn = sp.z_proj(mu_j, n);
    // Stage this neighbour's a_nl[ly] and bond_I[lr] into shared (disjoint regions).
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlm_y), [&] (const int ly) {
      const int l = sp.l_tile(ly);
      anl_sh[ly] = d_R1_nl(ii, jj, n, l) * (NNScalar) d_Y_bond(ii, jj, ly) * env;
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_lm_ind), [&] (const int lr) {
      bondI_sh[lr] = spbf_fused_indicator(j_global, n, lr) + zpn * sp.chem_l0_mask(lr);
    });
    team.team_barrier();
    // Outer-product accumulate into the shared prod slice; slot p=(ly,lr) is owned
    // by exactly one thread across the loop, so no atomics are needed.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, P), [&] (const int p) {
      const int ly = p / n_lm_ind;
      const int lr = p - ly * n_lm_ind;
      prod_sh[p] += anl_sh[ly] * bondI_sh[lr];
    });
    team.team_barrier();
  }

  // Sparse CG couple (Opt 9), one output function f per thread. cg_W is ~99%
  // zeros, so iterate only column f's nonzeros (CSC, ascending p = the dense
  // loop's accumulation order — numerically identical). inv_avg is folded into
  // the per-f result (A[n,f] = inv_avg·Σ_p prod·cg), sparing a whole
  // prod-scaling pass + barrier. All prod_sh writes are complete (barrier at
  // the end of the last neighbour iteration), so no extra barrier before this read.
  const NNScalar inv_avg = (NNScalar) sp.inv_avg_n_neigh;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nfunc), [&] (const int f) {
    NNScalar s = NNScalar(0.0);
    const int e1 = sp.cg_col_ptr(f + 1);
    for (int e = sp.cg_col_ptr(f); e < e1; e++)
      s += prod_sh[sp.cg_row_idx(e)] * sp.cg_col_val(e);
    spbf_fused_out(ii, n, f) = s * inv_avg;
  });
}

// ======================================================================
// compute_spbf_equiv (Opt 6) — launcher for the fused equiv-SPBF (A2/A3).
// Sets the per-call launch state captured by TagComputeSPBFFused, then launches
// one team per (ii,n) over the chunk with the prod slice + staging buffers in
// team shared. Replaces the TagComputeA2/TagComputeA3 launches.
// ======================================================================
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_spbf_equiv(
    int idx, const t_nn_3d_rg &indicator, const t_nn_3d &out)
{
  if (spbf_fast_ok(idx)) { spbf_fwd_fast(idx, indicator, out); return; }
  spbf_fused_idx = idx;
  spbf_fused_indicator = indicator;
  spbf_fused_out = out;

  const int nradmax = d_spbf[idx].n_rad_max;
  const int nlm_y = (lmax + 1) * (lmax + 1);
  const int n_lm_ind = d_spbf[idx].n_lm_ind;
  const int P = nlm_y * n_lm_ind;

  int team_size = 1, vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1) team_size = 128;
  const int league = chunk_size * nradmax;
  int scratch_size = scratch_bytes<DeviceType, NNScalar>(P + nlm_y + n_lm_ind);
  team_size = fit_team_size<DeviceType, TagComputeSPBFFused>(*this, league, team_size, vector_length);
  auto policy = Kokkos::TeamPolicy<DeviceType, TagComputeSPBFFused>(league, team_size, vector_length);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
  Kokkos::parallel_for("ComputeSPBFFused", policy, *this);
}

// ---- TagComputeAdjProd (Opt 4): compute adj_prod = inv_avg*Σ_f A_adj(ii,n,f)*cg_W(p,f)
// into the shared d_adj_prod buffer, ONE thread per OUTPUT element (ii,n,p). Unlike the
// previous per-(ii,n) computation (a register-spilling 1600-float local array in EACH of
// A{2,3}SPBFBwd and ForceEquiv), this holds a single accumulator per thread — no spill —
// and both consumers then just READ d_adj_prod. Opt 9: the f-reduction walks only row
// p's ~3-6 nonzeros (CSR) instead of the dense nfunc range (cg_W is ~99% zeros).
// force_spbf_idx/force_A_adj set by the launcher.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagComputeAdjProd,
    const int ii, const int n, const int p) const
{
  const DeviceSPBF &sp = d_spbf[force_spbf_idx];
  const NNScalar inv_avg = (NNScalar) sp.inv_avg_n_neigh;
  // Opt 9: iterate only row p's nonzeros (CSR, ascending f = the dense loop's
  // accumulation order — numerically identical; cg_W is ~99% zeros).
  NNScalar s = NNScalar(0.0);
  const int e1 = sp.cg_row_ptr(p + 1);
  for (int e = sp.cg_row_ptr(p); e < e1; e++)
    s += force_A_adj(ii, n, sp.cg_col_idx(e)) * sp.cg_row_val(e);
  d_adj_prod(ii, n, p) = s * inv_avg;
}

// ---- TagA3SPBFBwd: VJP of TagComputeA3 down to the L3 indicator eq2_norm. One
// thread per (owned atom ii, radial n). adj_prod[p] = inv_avg*Σ_f d_A3_adj[ii,n,f]
// *cg_W[p,f]; then for each neighbor j and lr: d_eq2_norm_adj[j,n,lr] += Σ_ly
// a_nl(ii,j,n)[ly]*adj_prod[ly*n_lm_ind+lr] (atomic; j owned OR ghost). The
// species-injection term (z_proj·chem_l0_mask) is constant -> no indicator adj.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagA3SPBFBwd,
    const int ii, const int n) const
{
  const DeviceSPBF &sp = d_spbf[idx_spbf_A3];
  if (n >= sp.n_rad_max) return;

  const int nlm_y = (lmax + 1) * (lmax + 1);
  const int n_lm_ind = sp.n_lm_ind;
  // Opt 4: adj_prod precomputed into d_adj_prod by compute_adj_prod(idx_spbf_A3).

  const int nct = d_ncount(ii);
  for (int jj = 0; jj < nct; jj++) {
    const int j_global = d_nearest(ii, jj);
    const NNScalar env = (NNScalar) d_env(ii, jj);
    NNScalar a_nl[25];
    for (int ly = 0; ly < nlm_y; ly++) {
      const int l = sp.l_tile(ly);
      a_nl[ly] = d_R1_nl(ii, jj, n, l) * (NNScalar) d_Y_bond(ii, jj, ly) * env;
    }
    for (int lr = 0; lr < n_lm_ind; lr++) {
      NNScalar s = NNScalar(0.0);
      for (int ly = 0; ly < nlm_y; ly++)
        s += a_nl[ly] * d_adj_prod(ii, n, ly * n_lm_ind + lr);
      Kokkos::atomic_add(&d_eq2_norm_adj(j_global, n, lr), s);
    }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_adj_prod(
    int spbf_idx, const t_nn_3d &A_adj)
{
  force_spbf_idx = spbf_idx;
  force_A_adj = A_adj;
  const int nradmax = d_spbf[spbf_idx].n_rad_max;
  const int nlm_y = (lmax + 1) * (lmax + 1);
  const int P = nlm_y * d_spbf[spbf_idx].n_lm_ind;
  Kokkos::parallel_for("ComputeAdjProd",
      Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<3>, TagComputeAdjProd>(
          {0, 0, 0}, {chunk_size, nradmax, P}), *this);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_A3_spbf_bwd()
{
  // Opt 10: fused team+shared version (old TagA3SPBFBwd kept for reference).
  compute_spbf_bwd_fused(idx_spbf_A3, d_eq2_norm_adj);
}

// ---- TagA2SPBFBwd (Task 3.2): VJP of TagComputeA2 down to the L2 indicator
// eq1_norm. IDENTICAL structure to TagA3SPBFBwd except: reads d_A2_adj, uses the
// A2 SPBF weights (idx_spbf_A2, n_lm_ind=25), and scatters into d_grad_I_global
// [nall, I_n_funcs=64, I_n_out=25] (the round-1 reverse buffer = d_eq1_norm_adj,
// owned partial). Requires d_R1_nl to hold A2's radial (caller recomputes it).
// The z_proj/chem_l0_mask species-injection term is constant -> no indicator adj.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagA2SPBFBwd,
    const int ii, const int n) const
{
  const DeviceSPBF &sp = d_spbf[idx_spbf_A2];
  if (n >= sp.n_rad_max) return;

  const int nlm_y = (lmax + 1) * (lmax + 1);
  const int n_lm_ind = sp.n_lm_ind;
  // Opt 4: adj_prod precomputed into d_adj_prod by compute_adj_prod(idx_spbf_A2).

  const int nct = d_ncount(ii);
  for (int jj = 0; jj < nct; jj++) {
    const int j_global = d_nearest(ii, jj);
    const NNScalar env = (NNScalar) d_env(ii, jj);
    NNScalar a_nl[25];
    for (int ly = 0; ly < nlm_y; ly++) {
      const int l = sp.l_tile(ly);
      a_nl[ly] = d_R1_nl(ii, jj, n, l) * (NNScalar) d_Y_bond(ii, jj, ly) * env;
    }
    for (int lr = 0; lr < n_lm_ind; lr++) {
      NNScalar s = NNScalar(0.0);
      for (int ly = 0; ly < nlm_y; ly++)
        s += a_nl[ly] * d_adj_prod(ii, n, ly * n_lm_ind + lr);
      Kokkos::atomic_add(&d_grad_I_global(j_global, n, lr), s);
    }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_A2_spbf_bwd()
{
  // Opt 10: fused team+shared version (old TagA2SPBFBwd kept for reference).
  compute_spbf_bwd_fused(idx_spbf_A2, d_grad_I_global);
}

// ---- TagSPBFBwdFused (Opt 10): team+shared equiv-SPBF backward, shared by A2
// (target d_grad_I_global) and A3 (target d_eq2_norm_adj). The old per-(ii,n)
// THREAD kernels re-read the full d_adj_prod row (P = nlm_y*n_lm_ind, 5 KB for
// A3) from global once PER NEIGHBOUR (~nct times) and walked the jj/lr/ly loops
// serially (~nct*n_lm_ind*nlm_y MACs per thread). Here ONE TEAM owns one (ii,n):
// the adj_prod row is staged into TEAM SHARED once, neighbours are processed in
// tiles of SPBF_BWD_TILE with each tile's a_nl[tn,nlm_y] staged cooperatively,
// and the (jj,lr) scatter pairs are split across the team. The atomic scatter
// into the nall-sized target is unchanged (same count, same nondeterministic
// accumulation order as before); the inner ly sum keeps ascending order.
// Shared layout: [ adjp_sh(P) | anl_sh(SPBF_BWD_TILE*nlm_y) ] NNScalars.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagSPBFBwdFused,
    const typename Kokkos::TeamPolicy<DeviceType, TagSPBFBwdFused>::member_type& team) const
{
  const DeviceSPBF &sp = d_spbf[spbf_bwd_idx];
  const int ii = team.league_rank() % chunk_size;   // league = chunk_size * n_rad_max
  const int n  = team.league_rank() / chunk_size;
  const int nlm_y = (lmax + 1) * (lmax + 1);
  const int nli = sp.n_lm_ind;
  const int P = nlm_y * nli;

  NNScalar* sh = (NNScalar*) team.team_shmem().get_shmem((P + (int) SPBF_BWD_TILE * nlm_y) * sizeof(NNScalar), 0);
  NNScalar* adjp_sh = sh;                 // [P] this (ii,n)'s adj_prod row
  NNScalar* anl_sh  = sh + P;             // [tn, nlm_y] staged tile of neighbour a_nl

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, P), [&] (const int p) {
    adjp_sh[p] = d_adj_prod(ii, n, p);
  });
  team.team_barrier();

  const int nct = d_ncount(ii);
  for (int t0 = 0; t0 < nct; t0 += (int) SPBF_BWD_TILE) {
    const int tn = (nct - t0 < (int) SPBF_BWD_TILE) ? (nct - t0) : (int) SPBF_BWD_TILE;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tn * nlm_y), [&] (const int q) {
      const int j = q / nlm_y, ly = q - j * nlm_y;
      const int l = sp.l_tile(ly);
      anl_sh[q] = d_R1_nl(ii, t0 + j, n, l) * (NNScalar) d_Y_bond(ii, t0 + j, ly)
                  * (NNScalar) d_env(ii, t0 + j);
    });
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, tn * nli), [&] (const int q) {
      const int j = q / nli, lr = q - j * nli;
      const NNScalar* an = anl_sh + j * nlm_y;
      NNScalar s = NNScalar(0.0);
      for (int ly = 0; ly < nlm_y; ly++)
        s += an[ly] * adjp_sh[ly * nli + lr];
      Kokkos::atomic_add(&spbf_bwd_target(d_nearest(ii, t0 + j), n, lr), s);
    });
    team.team_barrier();
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_spbf_bwd_fused(
    int idx, const t_nn_3d_rg &adj_target)
{
  spbf_bwd_idx = idx;
  spbf_bwd_target = adj_target;
  const int nradmax = d_spbf[idx].n_rad_max;
  const int nlm_y = (lmax + 1) * (lmax + 1);
  const int P = nlm_y * d_spbf[idx].n_lm_ind;

  int team_size = 1, vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1) team_size = 128;
  const int league = chunk_size * nradmax;
  int scratch_size = scratch_bytes<DeviceType, NNScalar>(P + (int) SPBF_BWD_TILE * nlm_y);
  team_size = fit_team_size<DeviceType, TagSPBFBwdFused>(*this, league, team_size, vector_length);
  auto policy = Kokkos::TeamPolicy<DeviceType, TagSPBFBwdFused>(league, team_size, vector_length);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
  Kokkos::parallel_for("SPBFBwdFused", policy, *this);
}

// ---- bond_geom_force: shared per-bond force chain (spec §4.1). Adapted VERBATIM
// from the validated GRACE-2L ComputeDerivative kernels (Y00==1 plm/dplm
// recurrence, tangential-gradient DY projection, f += w*(Y*DR*rhat + DY*R/r)),
// with the 3L twist that the envelope is a SEPARATE factor: Renv = R*env,
// DRenv = dR/dr*env + R*denv/dr. Geometry math is GeomScalar (fp64).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::bond_geom_force(
    const int ii, const int jj, const int n, const NNScalar* w,
    GeomScalar& f0, GeomScalar& f1, GeomScalar& f2) const
{
  const GeomScalar rinv = GeomScalar(1.0) / GeomScalar(d_rnorms(ii, jj));
  const GeomScalar rx = d_rhats(ii, jj, 0);
  const GeomScalar ry = d_rhats(ii, jj, 1);
  const GeomScalar rz = d_rhats(ii, jj, 2);
  const GeomScalar env  = d_env(ii, jj);
  const GeomScalar denv = d_denv(ii, jj);

  const GeomScalar Y00_v = GeomScalar(1.0), sq3_v = sq3, sq3o2_v = sq3o2, sq2_v = sq2;

  GeomScalar plm[15];
  GeomScalar dplm[15];
  plm[0] = Y00_v; dplm[0] = GeomScalar(0.0);
  if (lmax > 0) {
    plm[1] = Y00_v * sq3_v * rz; dplm[1] = Y00_v * sq3_v;
    plm[2] = -sq3o2_v * Y00_v;   dplm[2] = GeomScalar(0.0);
    for (int l = 2; l <= lmax; l++) {
      for (int m = 0; m < l - 1; m++) {
        const int idx = l*(l+1)/2 + m;
        const int i1 = (l-1)*l/2 + m, i2 = (l-2)*(l-1)/2 + m;
        const int ai = d_idx_sph(l*(l+1) + m);
        const GeomScalar a = alm(ai), b = blm(ai);
        plm[idx]  = a * (rz * plm[i1] + b * plm[i2]);
        dplm[idx] = a * (plm[i1] + rz * dplm[i1] + b * dplm[i2]);
      }
      { const int idx = l*(l+1)/2+l-1, prev = (l-1)*l/2+l-1;
        const GeomScalar t = dl(l) * plm[prev];
        plm[idx] = t * rz; dplm[idx] = t; }
      { const int idx = l*(l+1)/2+l, prev = (l-1)*l/2+l-1;
        plm[idx] = cl(l) * plm[prev]; dplm[idx] = GeomScalar(0.0); }
    }
  }

  const GeomScalar phase_re = rx, phase_im = ry;
  for (int l = 0; l <= lmax; l++) {
    const GeomScalar R_raw  = (GeomScalar) d_R1_nl(ii, jj, n, l);
    const GeomScalar DR_raw = (GeomScalar) d_DR1_nl(ii, jj, n, l);
    const GeomScalar Renv  = R_raw * env;
    const GeomScalar DRenv = DR_raw * env + R_raw * denv;   // d(R*env)/dr
    const GeomScalar R_over_r = Renv * rinv;

    // m = 0
    {
      const GeomScalar Y = plm[l*(l+1)/2];
      const GeomScalar dp = dplm[l*(l+1)/2];
      const GeomScalar rdy = dp * rz;
      const GeomScalar DY_x = -rdy * rx, DY_y = -rdy * ry, DY_z = dp - rdy * rz;
      const GeomScalar wv = (GeomScalar) w[l*(l+1)];
      const GeomScalar YDR = Y * DRenv;
      f0 += wv * (YDR * rx + DY_x * R_over_r);
      f1 += wv * (YDR * ry + DY_y * R_over_r);
      f2 += wv * (YDR * rz + DY_z * R_over_r);
    }

    // m >= 1
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
      const GeomScalar wp = (GeomScalar) w[Ap];
      const GeomScalar wn = (GeomScalar) w[An];

      { const GeomScalar YDR = rYp * DRenv;
        f0 += wp * (YDR*rx + DYpx*R_over_r);
        f1 += wp * (YDR*ry + DYpy*R_over_r);
        f2 += wp * (YDR*rz + DYpz*R_over_r); }
      { const GeomScalar YDR = rYn * DRenv;
        f0 += wn * (YDR*rx + DYnx*R_over_r);
        f1 += wn * (YDR*ry + DYny*R_over_r);
        f2 += wn * (YDR*rz + DYnz*R_over_r); }

      const GeomScalar t_re = pm_re*phase_re - pm_im*phase_im;
      const GeomScalar t_im = pm_re*phase_im + pm_im*phase_re;
      pm_re = t_re; pm_im = t_im;
    }
  }
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::bond_geom_force_ab(
    const int ii, const int jj, const GeomScalar* a, const GeomScalar* b,
    GeomScalar& f0, GeomScalar& f1, GeomScalar& f2) const
{
  const GeomScalar rinv = GeomScalar(1.0) / GeomScalar(d_rnorms(ii, jj));
  const GeomScalar rx = d_rhats(ii, jj, 0);
  const GeomScalar ry = d_rhats(ii, jj, 1);
  const GeomScalar rz = d_rhats(ii, jj, 2);

  const GeomScalar Y00_v = GeomScalar(1.0), sq3_v = sq3, sq3o2_v = sq3o2, sq2_v = sq2;

  GeomScalar plm[15];
  GeomScalar dplm[15];
  plm[0] = Y00_v; dplm[0] = GeomScalar(0.0);
  if (lmax > 0) {
    plm[1] = Y00_v * sq3_v * rz; dplm[1] = Y00_v * sq3_v;
    plm[2] = -sq3o2_v * Y00_v;   dplm[2] = GeomScalar(0.0);
    for (int l = 2; l <= lmax; l++) {
      for (int m = 0; m < l - 1; m++) {
        const int idx = l*(l+1)/2 + m;
        const int i1 = (l-1)*l/2 + m, i2 = (l-2)*(l-1)/2 + m;
        const int ai = d_idx_sph(l*(l+1) + m);
        const GeomScalar al = alm(ai), bl = blm(ai);
        plm[idx]  = al * (rz * plm[i1] + bl * plm[i2]);
        dplm[idx] = al * (plm[i1] + rz * dplm[i1] + bl * dplm[i2]);
      }
      { const int idx = l*(l+1)/2+l-1, prev = (l-1)*l/2+l-1;
        const GeomScalar t = dl(l) * plm[prev];
        plm[idx] = t * rz; dplm[idx] = t; }
      { const int idx = l*(l+1)/2+l, prev = (l-1)*l/2+l-1;
        plm[idx] = cl(l) * plm[prev]; dplm[idx] = GeomScalar(0.0); }
    }
  }

  const GeomScalar phase_re = rx, phase_im = ry;
  for (int l = 0; l <= lmax; l++) {
    // m = 0
    {
      const GeomScalar Y = plm[l*(l+1)/2];
      const GeomScalar dp = dplm[l*(l+1)/2];
      const GeomScalar rdy = dp * rz;
      const GeomScalar DY_x = -rdy * rx, DY_y = -rdy * ry, DY_z = dp - rdy * rz;
      const GeomScalar ay = a[l*(l+1)] * Y, br = b[l*(l+1)] * rinv;
      f0 += ay * rx + br * DY_x;
      f1 += ay * ry + br * DY_y;
      f2 += ay * rz + br * DY_z;
    }

    // m >= 1
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
      { const GeomScalar ay = a[Ap] * rYp, br = b[Ap] * rinv;
        f0 += ay*rx + br*DYpx; f1 += ay*ry + br*DYpy; f2 += ay*rz + br*DYpz; }
      { const GeomScalar ay = a[An] * rYn, br = b[An] * rinv;
        f0 += ay*rx + br*DYnx; f1 += ay*ry + br*DYny; f2 += ay*rz + br*DYnz; }

      const GeomScalar t_re = pm_re*phase_re - pm_im*phase_im;
      const GeomScalar t_im = pm_re*phase_im + pm_im*phase_re;
      pm_re = t_re; pm_im = t_im;
    }
  }
}

// ---- TagForceL1: L1 scalar-SPBF geometry backward. Team per owned atom ii, one bond
// per thread; per-lm weight w[n,lm] = d_A1_adj*z_tr*inv_avg (a_nl = R*Y*z_tr*env for L1),
// summed over the channels before the geometry (bond_geom_force_ab). Adds into d_f_ij.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagForceL1,
    const typename Kokkos::TeamPolicy<DeviceType, TagForceL1>::member_type& team) const
{
  // Team per atom ii, one bond per thread. w[n,lm] = A1_adj(ii,n,lm) * inv_avg (staged in shared) * z_tr(mu_j,n);
  // the channel sum is taken before the geometry: a/b as in bond_geom_force_ab.
  const DeviceSPBF &sp = d_spbf[0];   // A1
  const int ii = team.league_rank();
  const int NR = sp.n_rad_max;
  const int nlm = (lmax + 1) * (lmax + 1);
  const NNScalar inv_avg = (NNScalar) sp.inv_avg_n_neigh;
  NNScalar* wsh = (NNScalar*) team.team_shmem().get_shmem(NR * nlm * sizeof(NNScalar), 0);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NR * nlm), [&] (const int q) {
    const int n = q / nlm, lm = q - n * nlm;
    wsh[q] = d_A1_adj(ii, n, lm) * inv_avg;
  });
  team.team_barrier();
  const int nct = d_ncount(ii);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nct), [&] (const int jj) {
    const int mu_j = d_mu_j(ii, jj);
    const GeomScalar env = d_env(ii, jj), denv = d_denv(ii, jj);
    GeomScalar a[(GRACE3L_LMAX + 1) * (GRACE3L_LMAX + 1)], b[(GRACE3L_LMAX + 1) * (GRACE3L_LMAX + 1)];
    for (int lm = 0; lm < nlm; lm++) { a[lm] = GeomScalar(0.0); b[lm] = GeomScalar(0.0); }
    for (int n = 0; n < NR; n++) {
      const NNScalar z_n = sp.z_tr(mu_j, n);
      const NNScalar* wn = wsh + n * nlm;
      for (int l = 0; l <= lmax; l++) {
        const GeomScalar R = (GeomScalar) d_R1_nl(ii, jj, n, l);
        const GeomScalar DR = (GeomScalar) d_DR1_nl(ii, jj, n, l);
        const GeomScalar Renv = R * env, DRenv = DR * env + R * denv;
        for (int lm = l * l; lm < (l + 1) * (l + 1); lm++) {
          const GeomScalar w = (GeomScalar) (wn[lm] * z_n);
          a[lm] += w * DRenv;
          b[lm] += w * Renv;
        }
      }
    }
    GeomScalar f0 = GeomScalar(0.0), f1 = GeomScalar(0.0), f2 = GeomScalar(0.0);
    bond_geom_force_ab(ii, jj, a, b, f0, f1, f2);
    d_f_ij(ii, jj, 0) += f0;
    d_f_ij(ii, jj, 1) += f1;
    d_f_ij(ii, jj, 2) += f2;
  });
}

// ---- TagForceEquiv: L2/L3 equivariant-SPBF geometry backward. One thread per
// (owned atom ii, radial n); loops neighbors jj. adj_prod[p] = inv_avg*Σ_f
// force_A_adj*cg_W (same as compute_A{2,3}_spbf_bwd), the per-lm weight is
// w[ly] = Σ_lr adj_prod[ly*n_lm_ind+lr]*bond_I[lr] with bond_I the FULL indicator
// (indicator_global[j] + z_proj injection — a_nl multiplies both). Accumulates
// into d_f_ij (atomic).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagForceEquiv,
    const int ii, const int n) const
{
  const DeviceSPBF &sp = d_spbf[force_spbf_idx];
  if (n >= sp.n_rad_max) return;

  const int nlm_y = (lmax + 1) * (lmax + 1);
  const int n_lm_ind = force_n_lm_ind;
  // Opt 4: adj_prod precomputed into d_adj_prod by compute_adj_prod(force_spbf_idx).

  const int nct = d_ncount(ii);
  for (int jj = 0; jj < nct; jj++) {
    const int j_global = d_nearest(ii, jj);
    const int mu_j = d_mu_j(ii, jj);
    const NNScalar zpn = sp.z_proj(mu_j, n);

    NNScalar bond_I[GRACE3L_MAX_SPBF_NLMIND];
    if (force_ind_stage == 2) {
      for (int lr = 0; lr < n_lm_ind; lr++)
        bond_I[lr] = d_eq2_norm_global(j_global, n, lr) + zpn * sp.chem_l0_mask(lr);
    } else {
      for (int lr = 0; lr < n_lm_ind; lr++)
        bond_I[lr] = d_I_global(j_global, n, lr) + zpn * sp.chem_l0_mask(lr);
    }

    NNScalar w[25];
    for (int ly = 0; ly < nlm_y; ly++) {
      const int base = ly * n_lm_ind;
      NNScalar s = NNScalar(0.0);
      for (int lr = 0; lr < n_lm_ind; lr++)
        s += d_adj_prod(ii, n, base + lr) * bond_I[lr];
      w[ly] = s;
    }
    GeomScalar f0 = GeomScalar(0.0), f1 = GeomScalar(0.0), f2 = GeomScalar(0.0);
    bond_geom_force(ii, jj, n, w, f0, f1, f2);
    Kokkos::atomic_add(&d_f_ij(ii, jj, 0), f0);
    Kokkos::atomic_add(&d_f_ij(ii, jj, 1), f1);
    Kokkos::atomic_add(&d_f_ij(ii, jj, 2), f2);
  }
}

// ---- TagForceEquivFused (Opt 11): team-per-(ii,n) version of TagForceEquiv.
// The old kernel gave one THREAD the whole (ii,n) row and re-read the full
// d_adj_prod row (P = nlm_y*n_lm_ind, 5 KB for A3) from global once per
// neighbour (~nct times). Here ONE TEAM owns one (ii,n): the adj_prod row is
// staged into TEAM SHARED once, and the bonds are split across the team (one
// bond per thread, strided). The per-bond body — bond_I gather, w[ly] couple
// (ascending lr, unchanged accumulation order), bond_geom_force, d_f_ij atomic
// — is VERBATIM from TagForceEquiv, so per-bond arithmetic is identical.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
KOKKOS_INLINE_FUNCTION
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::operator() (TagForceEquivFused,
    const typename Kokkos::TeamPolicy<DeviceType, TagForceEquivFused>::member_type& team) const
{
  const DeviceSPBF &sp = d_spbf[force_spbf_idx];
  const int ii = team.league_rank() % chunk_size;   // league = chunk_size * n_rad_max
  const int n  = team.league_rank() / chunk_size;

  const int nlm_y = (lmax + 1) * (lmax + 1);
  const int n_lm_ind = force_n_lm_ind;
  const int P = nlm_y * n_lm_ind;

  NNScalar* adjp_sh = (NNScalar*) team.team_shmem().get_shmem(P * sizeof(NNScalar), 0);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, P), [&] (const int p) {
    adjp_sh[p] = d_adj_prod(ii, n, p);
  });
  team.team_barrier();

  const int nct = d_ncount(ii);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nct), [&] (const int jj) {
    const int j_global = d_nearest(ii, jj);
    const int mu_j = d_mu_j(ii, jj);
    const NNScalar zpn = sp.z_proj(mu_j, n);

    NNScalar bond_I[GRACE3L_MAX_SPBF_NLMIND];
    if (force_ind_stage == 2) {
      for (int lr = 0; lr < n_lm_ind; lr++)
        bond_I[lr] = d_eq2_norm_global(j_global, n, lr) + zpn * sp.chem_l0_mask(lr);
    } else {
      for (int lr = 0; lr < n_lm_ind; lr++)
        bond_I[lr] = d_I_global(j_global, n, lr) + zpn * sp.chem_l0_mask(lr);
    }

    NNScalar w[25];
    for (int ly = 0; ly < nlm_y; ly++) {
      const int base = ly * n_lm_ind;
      NNScalar s = NNScalar(0.0);
      for (int lr = 0; lr < n_lm_ind; lr++)
        s += adjp_sh[base + lr] * bond_I[lr];
      w[ly] = s;
    }
    GeomScalar f0 = GeomScalar(0.0), f1 = GeomScalar(0.0), f2 = GeomScalar(0.0);
    bond_geom_force(ii, jj, n, w, f0, f1, f2);
    Kokkos::atomic_add(&d_f_ij(ii, jj, 0), f0);
    Kokkos::atomic_add(&d_f_ij(ii, jj, 1), f1);
    Kokkos::atomic_add(&d_f_ij(ii, jj, 2), f2);
  });
}



// ======================================================================
// Equivariant SPBF (A2/A3) for lmax 4 (25 Y components) and 25 or 50 indicator components.
//   a[j, n, y]  = R(ii, j, n, l(y)) Y_y(j) env(j),   b[j, n, r] = I(j_global, n, r) + z_proj(mu_j, n) mask(r)
//   Q[n, y, r]  = sum_j a b,                         A[n, f] = inv_avg sum_(y,r) cg[(y,r), f] Q[n, y, r]
// Backward with M[n, y, r] = inv_avg sum_f cg[(y,r), f] dE/dA[n, f]:
//   dE/dI(j_global, n, r) += sum_y M a,   v[j, n, y] = sum_r M b  -> geometry force via bond_geom_force_ab.
// Team per (atom, block of 4 channels), one warp per channel.
// ======================================================================

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
bool Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_fast_ok(int idx) const
{
  static const bool off = getenv("GRACE3L_GENERIC_SPBF") != nullptr;
#ifdef KOKKOS_ENABLE_CUDA
  if (!std::is_same<typename DeviceType::execution_space, Kokkos::Cuda>::value) return false;
#else
  return false;
#endif
  if (sizeof(NNScalar) != 4) return false;
  if (off || lmax != 4) return false;
  const DeviceSPBF &sp = d_spbf[idx];
  if (sp.n_rad_max % 4 != 0) return false;
  if (sp.n_lm_ind != 25 && sp.n_lm_ind != 50) return false;
  if (!spbf_ltile_std[idx]) return false;
  if (sp.ell_w < 1 || sp.ell_w > GRACE3L_SPBF_ELL_MAX) return false;
  return Kokkos::DefaultExecutionSpace().concurrency() > 1;
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
bool Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_rm_ok(int si, int ridx) const
{
#ifdef KOKKOS_ENABLE_CUDA
  static const bool off = getenv("GRACE3L_SPBF_TRANSPOSE") != nullptr;
  if (off || !spbf_fast_ok(si) || !cublas_handle || ridx < 0 || ridx >= n_reduces) return false;
  const ReduceHost &hr = h_reduce[ridx];
  const DeviceSPBF &sp = d_spbf[si];
  if (!hr.cublas_ok || !hr.grouped || hr.n_in != sp.n_rad_max || hr.n_conn != sp.nfunc) return false;
  for (size_t f = 0; f < hr.gn.size(); f++)
    if (hr.gn[f] > 0 && hr.gstart[f] + hr.gn[f] > sp.nfunc) return false;
  return (int) d_spbf_T.extent(1) >= sp.n_rad_max * sp.nfunc && (int) d_spbf_T.extent(0) >= chunk_size;
#else
  (void) si; (void) ridx;
  return false;
#endif
}



// fuse_radial applies: the layer-1 tables exist and the CUDA fp32 fast kernels (ForceL1F) run
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
bool Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::l1_radial_fused() const
{
#ifdef KOKKOS_ENABLE_CUDA
  static const bool generic = getenv("GRACE3L_GENERIC_SPBF") != nullptr;
  return fuse_radial && spline_on && std::is_same<typename DeviceType::execution_space, Kokkos::Cuda>::value &&
         sizeof(NNScalar) == 4 && d_spbf[0].n_rad_max <= 64 && lmax <= 4 && spbf_ltile_std[0] && !generic;
#else
  return false;
#endif
}

// ComputeA1 with the layer-1 radial from the cubic tables (fuse_radial): thread per (atom, channel n), per bond the
// interval k and t of x = r / rcut once, the l = 0..lmax coefficients of output o = n (lmax + 1) + l as float4.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_A1_tables()
{
  const DeviceSPBF &sp = d_spbf[0];
  const int NR = L1_nradmax, lmx = lmax, N = spl_N[0], NT = spl_Ntot, nsp = n_sp, ld = spl_ld;
  const float inv = (float) sp.inv_avg_n_neigh;
  auto tab = d_splR; auto spm = d_sp_of_mu; auto bc = d_bond_cutoff; auto mui = d_mu_i; auto muj = d_mu_j;
  auto nc = d_ncount; auto rn = d_rnorms; auto Y = d_Y_bond; auto env = d_env; auto ztr = sp.z_tr; auto A1 = d_A1;
  Kokkos::parallel_for("ComputeA1T", Kokkos::RangePolicy<DeviceType>(0, chunk_size * NR), KOKKOS_LAMBDA(const int q) {
    const int ii = q / NR, n = q - ii * NR;
    float acc[25];
#pragma unroll
    for (int lm = 0; lm < 25; lm++) acc[lm] = 0.0f;
    const int mi = mui(ii), nct = nc(ii);
    const size_t pb = (size_t) spm(mi) * nsp;
    for (int jj = 0; jj < nct; jj++) {
      const int mj = muj(ii, jj);
      const float rc = bc(mi, mj), s = (float) rn(ii, jj) / rc * N;
      int k = (int) s;
      if (k > N - 1) k = N - 1;
      const float t = s - k, w = ztr(mj, n) * (float) env(ii, jj);
      const float *c4 = tab.data() + (((pb + spm(mj)) * NT + k) * ld + (size_t) n * (lmx + 1)) * 4;    // layer 1: offset 0
#pragma unroll
      for (int l = 0; l < 5; l++) {
        if (l > lmx) break;
        const G3F4 c = *reinterpret_cast<const G3F4 *>(c4 + 4 * l);
        const float R = (c.x + t * (c.y + t * (c.z + t * c.w))) * w;
#pragma unroll
        for (int m = 0; m < 2 * l + 1; m++) acc[l * l + m] += R * (float) Y(ii, jj, l * l + m);
      }
    }
    const int nlm = (lmx + 1) * (lmx + 1);
    for (int lm = 0; lm < nlm; lm++) A1(ii, n, lm) = acc[lm] * inv;
  });
}

// L1 geometry force, team per atom: warp per bond, lane = channels (lane, lane + 32) with their dE/dA1 rows in
// registers; f = sum_n sum_lm w z (d(R env)/dr Y rhat + (R env) DY / r) from d_Y_bond / d_dY_bond, the channel
// sum by warp shuffles. One warp per bond -> plain += into d_f_ij.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::force_L1_fast()
{
  const DeviceSPBF &sp = d_spbf[0];
  const int NR = sp.n_rad_max, lmx = lmax, nlm = (lmax + 1) * (lmax + 1);
  const float inv = (float) sp.inv_avg_n_neigh;
  auto A1a = d_A1_adj; auto ztr = sp.z_tr; auto R = d_R1_nl; auto DR = d_DR1_nl; auto Y = d_Y_bond; auto dY = d_dY_bond;
  auto env = d_env; auto denv = d_denv; auto rn = d_rnorms; auto rh = d_rhats; auto ncount = d_ncount; auto muj = d_mu_j;
  auto fij = d_f_ij;
  // fuse_radial: R and dR/dr of layer 1 from the cubic tables (as radial_from_splines), not from d_R1_nl / d_DR1_nl
  const bool tabr = l1_radial_fused();
  const int N = spl_N[0], NT = spl_Ntot, nsp = n_sp, ld = spl_ld;
  auto tab = d_splR; auto spm = d_sp_of_mu; auto bc = d_bond_cutoff; auto mui = d_mu_i;
  using team_policy = Kokkos::TeamPolicy<DeviceType>;
  using member = typename team_policy::member_type;
  Kokkos::parallel_for("ForceL1F", team_policy(chunk_size, 128), KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank(), tid = t.team_rank(), w = tid / 32, lane = tid - 32 * w;
    float wt[2][25];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int n = lane + 32 * h;
#pragma unroll
      for (int lm = 0; lm < 25; lm++) wt[h][lm] = (n < NR && lm < nlm) ? A1a(ii, n, lm) * inv : 0.0f;
    }
    const int nct = ncount(ii);
    for (int jj = w; jj < nct; jj += 4) {
      const int mu = muj(ii, jj);
      const GeomScalar ev = env(ii, jj), dev = denv(ii, jj);
      GeomScalar sA = GeomScalar(0.0), sBx = GeomScalar(0.0), sBy = GeomScalar(0.0), sBz = GeomScalar(0.0);
      const float *trow = nullptr;
      float tt = 0.0f, dxs = 0.0f;
      if (tabr) {
        const int mi = mui(ii);
        const float rc = bc(mi, mu), s = (float) rn(ii, jj) / rc * N;
        int k = (int) s;
        if (k > N - 1) k = N - 1;
        tt = s - k;
        dxs = (float) N / rc;
        trow = tab.data() + (((size_t) (spm(mi) * nsp + spm(mu)) * NT + k) * ld) * 4;    // layer 1: offset 0
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int n = lane + 32 * h;
        const float z = (n < NR) ? ztr(mu, n) : 0.0f;
#pragma unroll
        for (int l = 0; l < 5; l++) {
          if (l > lmx) break;
          float r = 0.0f, dr = 0.0f;
          if (n < NR) {
            if (tabr) {
              const G3F4 c = *reinterpret_cast<const G3F4 *>(trow + (size_t) (n * (lmx + 1) + l) * 4);
              r = c.x + tt * (c.y + tt * (c.z + tt * c.w));
              dr = (c.y + tt * (2.0f * c.z + 3.0f * tt * c.w)) * dxs;
            } else {
              r = R(ii, jj, n, l);
              dr = DR(ii, jj, n, l);
            }
          }
          const GeomScalar Re = (GeomScalar) r * ev, DRe = (GeomScalar) dr * ev + (GeomScalar) r * dev;
#pragma unroll
          for (int m = 0; m < 2 * l + 1; m++) {
            const int lm = l * l + m;
            const GeomScalar wz = (GeomScalar) (wt[h][lm] * z);
            sA += wz * DRe * Y(ii, jj, lm);
            const GeomScalar tz = wz * Re;
            sBx += tz * dY(ii, jj, lm, 0); sBy += tz * dY(ii, jj, lm, 1); sBz += tz * dY(ii, jj, lm, 2);
          }
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        sA += g3l_shfl_xor(sA, o);
        sBx += g3l_shfl_xor(sBx, o);
        sBy += g3l_shfl_xor(sBy, o);
        sBz += g3l_shfl_xor(sBz, o);
      }
      if (lane == 0) {
        const GeomScalar rinv = GeomScalar(1.0) / GeomScalar(rn(ii, jj));
        fij(ii, jj, 0) += sA * GeomScalar(rh(ii, jj, 0)) + rinv * sBx;
        fij(ii, jj, 1) += sA * GeomScalar(rh(ii, jj, 1)) + rinv * sBy;
        fij(ii, jj, 2) += sA * GeomScalar(rh(ii, jj, 2)) + rinv * sBz;
      }
    }
  });
}


// Transpose between a LayoutLeft [cs, NR, nfunc] view and the row-major scratch d_spbf_T[ii][f * NR + n]: both
// are [cs x NR nfunc] with column q = n + NR f, so this is a plain 32 x 32 tiled transpose through shared.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_transpose(const t_nn_3d &A, int NR, int nfunc, bool to_rowmajor)
{
  const int cs = chunk_size, Q = NR * nfunc;
  const int nti = (cs + 31) / 32, ntq = (Q + 31) / 32;
  float *a = (float *) A.data();
  const long lda = (long) A.extent(0);
  auto T = d_spbf_T;
  const bool fwd = to_rowmajor;
  using team_policy = Kokkos::TeamPolicy<DeviceType>;
  using member = typename team_policy::member_type;
  team_policy pol(nti * ntq, 256);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(32 * 33 * sizeof(float)));
  Kokkos::parallel_for("SPBFTranspose", pol, KOKKOS_LAMBDA(const member &t) {
    float *tile = (float *) t.team_shmem().get_shmem(32 * 33 * sizeof(float));
    const int lr = t.league_rank(), it = lr % nti, qt = lr / nti;
    const int i0 = it * 32, q0 = qt * 32;
    const int tx = t.team_rank() % 32, ty = t.team_rank() / 32;
    for (int k = 0; k < 4; k++) {
      const int r = ty + 8 * k;
      if (fwd) {              // read along ii (LayoutLeft): tile[q][ii]
        const int ii = i0 + tx, q = q0 + r;
        if (ii < cs && q < Q) tile[r * 33 + tx] = a[ii + lda * q];
      } else {                // read along q (row-major): tile[ii][q]
        const int ii = i0 + r, q = q0 + tx;
        if (ii < cs && q < Q) tile[r * 33 + tx] = T(ii, q);
      }
    }
    t.team_barrier();
    for (int k = 0; k < 4; k++) {
      const int r = ty + 8 * k;
      if (fwd) {              // write along q
        const int ii = i0 + r, q = q0 + tx;
        if (ii < cs && q < Q) T(ii, q) = tile[tx * 33 + r];
      } else {                // write along ii
        const int ii = i0 + tx, q = q0 + r;
        if (ii < cs && q < Q) a[ii + lda * q] = tile[tx * 33 + r];
      }
    }
  });
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_fwd_fast(
    int idx, const t_nn_3d_rg &indicator, const t_nn_3d &out, bool rm)
{
  if (d_spbf[idx].n_lm_ind == 25) spbf_fwd_fast_t<5>(idx, indicator, out, rm);
  else spbf_fwd_fast_t<10>(idx, indicator, out, rm);
}

// Forward: rounds of NB bonds staged in shared (a[b][c][y], b[b][c][r]); lane < 25 of the channel's warp
// owns a 5 (y) x TR (r) tile of Q in registers. Q then goes to shared for the sparse CG step.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
template<int TR>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_fwd_fast_t(
    int idx, const t_nn_3d_rg &indicator, const t_nn_3d &out, bool rm)
{
  // shared rows padded for 16-byte loads: a tile [5 -> 8], b tile [TR -> TRP]
  constexpr int CB = 4, NB = 16, NY = 25, TY = 5, NLI = 5 * TR, TRP = (TR + 3) / 4 * 4, TEAM = 128;
  constexpr int ASZ = NB * CB * 5 * 8, BSZ = NB * CB * 5 * TRP, QSZ = CB * NY * NLI;
  constexpr int SHF = (ASZ + BSZ > QSZ) ? ASZ + BSZ : QSZ;
  const DeviceSPBF &sp = d_spbf[idx];
  const int nblk = sp.n_rad_max / CB, nfunc = sp.nfunc;
  auto ltile = sp.l_tile; auto zproj = sp.z_proj; auto mask = sp.chem_l0_mask;
  auto cptr = sp.cg_col_ptr;
  const G3I2 *csc2 = reinterpret_cast<const G3I2 *>(sp.cg_csc2.data());
  const NNScalar inv = (NNScalar) sp.inv_avg_n_neigh;
  auto R = d_R1_nl; auto Y = d_Y_bond; auto env = d_env; auto ncount = d_ncount; auto nearest = d_nearest;
  auto muj = d_mu_j; auto ind = indicator; auto AT = d_spbf_T;
  const int NR = sp.n_rad_max;
  // fuse_radial: R of this layer from the cubic tables, staged per round in shared (Rs[b][cc][l]), not from d_R1_nl
  const bool tabr = fuse_radial && spline_on;
  const int TN = spl_N[idx], TNT = spl_Ntot, TOFF = spl_off[idx], tnsp = n_sp, tld = spl_ld, tlm = lmax + 1;
  auto tab = d_splR; auto spm = d_sp_of_mu; auto bc = d_bond_cutoff; auto mui = d_mu_i; auto rn = d_rnorms;

  using team_policy = Kokkos::TeamPolicy<DeviceType>;
  using member = typename team_policy::member_type;
  const size_t shb = (size_t) SHF * sizeof(float) + NB * (2 * sizeof(int) + sizeof(float)) +
                     (tabr ? NB * CB * 5 * sizeof(float) : 0) + 64;
  team_policy pol(chunk_size * nblk, TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(shb));
  Kokkos::parallel_for("SPBFFwdT", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank() / nblk, c0 = (t.league_rank() - ii * nblk) * CB;
    float *sh = (float *) t.team_shmem().get_shmem_aligned((size_t) SHF * sizeof(float), 16);
    int *jgs = (int *) t.team_shmem().get_shmem(NB * sizeof(int));
    int *mus = (int *) t.team_shmem().get_shmem(NB * sizeof(int));
    float *evs = (float *) t.team_shmem().get_shmem(NB * sizeof(float));
    float *Rs = tabr ? (float *) t.team_shmem().get_shmem(NB * CB * 5 * sizeof(float)) : nullptr;
    float *as = sh, *bs = sh + ASZ;
    const int tid = t.team_rank(), c = tid / 32, lane = tid - c * 32;
    const bool act = lane < 25;
    const int ty = act ? lane / 5 : 0, tr = act ? lane - 5 * ty : 0;
    float acc[TY][TR];
#pragma unroll
    for (int i = 0; i < TY; i++)
#pragma unroll
      for (int k = 0; k < TR; k++) acc[i][k] = 0.0f;
    const int nct = ncount(ii);
    for (int jb = 0; jb < nct; jb += NB) {
      const int nb = (nct - jb < NB) ? nct - jb : NB;
      // bonds past the last one repeat it with env 0: their a is an exact 0, and the staging loads below are
      // unconditional (so that they are issued together)
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NB), [&](const int b) {
        const int jq = jb + (b < nb ? b : nb - 1);
        jgs[b] = nearest(ii, jq); mus[b] = muj(ii, jq); evs[b] = (b < nb) ? (float) env(ii, jq) : 0.0f;
      });
      t.team_barrier();
      if (tabr) {
        const int mi = mui(ii);
        for (int pos = tid; pos < NB * CB * 5; pos += TEAM) {
          const int b = pos / (CB * 5), cl = pos - b * CB * 5, cc = cl / 5, l = cl - 5 * cc;
          const int jq = jb + (b < nb ? b : nb - 1), mj = mus[b];
          const float rc = bc(mi, mj), sx = (float) rn(ii, jq) / rc * TN;
          int k = (int) sx;
          if (k > TN - 1) k = TN - 1;
          const float tt = sx - k;
          const G3F4 c = *reinterpret_cast<const G3F4 *>(
              tab.data() + (((size_t) (spm(mi) * tnsp + spm(mj)) * TNT + TOFF + k) * tld +
                            (size_t) (c0 + cc) * tlm + l) * 4);
          Rs[pos] = c.x + tt * (c.y + tt * (c.z + tt * c.w));
        }
        t.team_barrier();
      }
      // staging: each thread owns fixed (channel, component) pairs and walks the NB bonds (independent loads)
      for (int pos = tid; pos < CB * NY; pos += TEAM) {
        const int cc = pos / NY, y = pos - cc * NY, yt = y / 5, yi = y - 5 * yt, l = ltile(y);
        float v[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) {
          const int jq = jb + (b < nb ? b : nb - 1);
          v[b] = (tabr ? Rs[(b * CB + cc) * 5 + l] : R(ii, jq, c0 + cc, l)) * (float) Y(ii, jq, y) * evs[b];
        }
#pragma unroll
        for (int b = 0; b < NB; b++) as[((b * CB + cc) * 5 + yt) * 8 + yi] = v[b];
      }
      for (int pos = tid; pos < CB * NLI; pos += TEAM) {
        const int cc = pos / NLI, lr = pos - cc * NLI, rt = lr / TR, rk = lr - TR * rt;
        const float mk = mask(lr);
        float v[NB];
#pragma unroll
        for (int b = 0; b < NB; b++) v[b] = ind(jgs[b], c0 + cc, lr) + zproj(mus[b], c0 + cc) * mk;
#pragma unroll
        for (int b = 0; b < NB; b++) bs[((b * CB + cc) * 5 + rt) * TRP + rk] = v[b];
      }
      t.team_barrier();
      if (act) {
        for (int b = 0; b < nb; b++) {
          const float *ab = as + ((b * CB + c) * 5 + ty) * 8;
          const float *bb = bs + ((b * CB + c) * 5 + tr) * TRP;
          float av[8], bv[TRP];
          const G3F4 a4 = *reinterpret_cast<const G3F4 *>(ab);
          av[0] = a4.x; av[1] = a4.y; av[2] = a4.z; av[3] = a4.w; av[4] = ab[4];
#pragma unroll
          for (int k4 = 0; k4 < TRP / 4; k4++) {
            const G3F4 b4 = reinterpret_cast<const G3F4 *>(bb)[k4];
            bv[4 * k4] = b4.x; bv[4 * k4 + 1] = b4.y; bv[4 * k4 + 2] = b4.z; bv[4 * k4 + 3] = b4.w;
          }
#pragma unroll
          for (int i = 0; i < TY; i++)
#pragma unroll
            for (int k = 0; k < TR; k++) acc[i][k] += av[i] * bv[k];
        }
      }
      t.team_barrier();
    }
    if (act)
#pragma unroll
      for (int i = 0; i < TY; i++)
#pragma unroll
        for (int k = 0; k < TR; k++) sh[(c * NY + ty * TY + i) * NLI + tr * TR + k] = acc[i][k];
    t.team_barrier();
    // CG step, thread per output function f for all CB channels (one CSC index load per entry); the CB channels
    // of f are contiguous in d_spbf_T (one 16-byte store)
    {
      float *Ao = &AT(ii, c0);
      for (int f = tid; f < nfunc; f += TEAM) {
        float s[CB];
#pragma unroll
        for (int cc = 0; cc < CB; cc++) s[cc] = 0.0f;
        const int e0 = cptr(f), e1 = cptr(f + 1);
        // entries in groups of 4 (packed (row, value) pairs loaded together), in CSC order
        for (int e = e0; e < e1; e += 4) {
          G3I2 en[4];
#pragma unroll
          for (int u = 0; u < 4; u++) en[u] = (e + u < e1) ? csc2[e + u] : G3I2{0, 0.0f};
#pragma unroll
          for (int u = 0; u < 4; u++)
            if (e + u < e1) {
#pragma unroll
              for (int cc = 0; cc < CB; cc++) s[cc] += sh[cc * NY * NLI + en[u].i] * en[u].v;
            }
        }
        *reinterpret_cast<G3F4 *>(Ao + (long) f * NR) = G3F4{s[0] * inv, s[1] * inv, s[2] * inv, s[3] * inv};
      }
    }
  });
  if (!rm) spbf_transpose(out, NR, nfunc, false);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_bwd_fast(
    int idx, const t_nn_3d &A_adj, const t_nn_3d_rg &indicator, const t_nn_3d_rg &target, bool rm)
{
  static const bool v1 = getenv("GRACE3L_SPBF_BWD1") != nullptr;
  const bool h1 = d_spbf[idx].n_lm_ind == 25;
  if (v1) {
    if (h1) spbf_bwd_fast_t<1>(idx, A_adj, indicator, target, rm);
    else spbf_bwd_fast_t<2>(idx, A_adj, indicator, target, rm);
  } else {
    // four teams per SM (registers capped at 128) where the opt-in shared memory per block allows it (A100, H100 /
    // H200: >= 160 KB; not the 99 KB of sm_86 / sm_89)
    static const bool big_smem = [] {
      int v = 0;
#ifdef KOKKOS_ENABLE_CUDA
      int dev = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
#endif
      return v >= 160 * 1024;
    }();
    if (big_smem) {
      if (h1) spbf_bwd_fast2_t<1, 4>(idx, A_adj, indicator, target, rm);
      else spbf_bwd_fast2_t<2, 4>(idx, A_adj, indicator, target, rm);
    } else {
      if (h1) spbf_bwd_fast2_t<1, 0>(idx, A_adj, indicator, target, rm);
      else spbf_bwd_fast2_t<2, 0>(idx, A_adj, indicator, target, rm);
    }
  }
}

// Backward + geometry force: M for the team's 4 channels in shared (from dE/dA through the CG rows);
// rounds of 32 / H bonds, lane = (bond, 1/H of the indicator components) holding b and dE/dI in registers.
// v goes to shared; then one thread per bond sums v over the block's channels and evaluates the force once.
// Expects d_R1_nl / d_DR1_nl / d_denv of this layer; d_f_ij accumulates (atomic: one team per channel block).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
template<int H>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_bwd_fast_t(
    int idx, const t_nn_3d &A_adj, const t_nn_3d_rg &indicator, const t_nn_3d_rg &target, bool rm)
{
  // M rows per (channel, y, half) padded 25 -> 28 for 16-byte loads. Lane = (half h, bond bl); for H = 2 the
  // two halves of v = sum_r M b are combined with a warp shuffle (lanes bl, bl + 16). Y and dY of the round's
  // bonds are staged in shared (one contiguous block); every lane adds its (bond, channel) force while it
  // walks y (l blocks unrolled: l_tile is the standard y -> l(y), checked in spbf_fast_ok).
  constexpr int CB = 4, NY = 25, NLH = 25, NLHP = 28, NLI = NLH * H, NBR = 32 / H, TEAM = 128;
  constexpr int MSZ = CB * NY * H * NLHP, GSZ = NBR * NY * 4, NP = NY * NLI, EW = GRACE3L_SPBF_ELL_MAX;
  const DeviceSPBF &sp = d_spbf[idx];
  const int nblk = sp.n_rad_max / CB;
  auto zproj = sp.z_proj; auto mask = sp.chem_l0_mask;
  auto ecol = sp.cg_ell_col; auto evl = sp.cg_ell_val; const int ew = sp.ell_w;
  const float inv = (float) sp.inv_avg_n_neigh;
  // dE/dA as row-major [ii][f * NR + n] (coalesced transpose of the LayoutLeft A_adj, or written there directly)
  const int NR = sp.n_rad_max, nfunc = sp.nfunc;
  if (!rm) spbf_transpose(A_adj, NR, nfunc, true);
  auto AT = d_spbf_T; auto ind = indicator; auto tgt = target;
  auto R = d_R1_nl; auto DR = d_DR1_nl; auto Y = d_Y_bond; auto dY = d_dY_bond; auto env = d_env; auto denv = d_denv;
  auto rn = d_rnorms; auto rh = d_rhats; auto ncount = d_ncount; auto nearest = d_nearest; auto muj = d_mu_j;
  auto fij = d_f_ij;
  const int mn = (int) d_Y_bond.extent(1);
  // fuse_radial: R and dR/dr of this layer from the cubic tables (as radial_from_splines), not from d_R1_nl / d_DR1_nl
  const bool tabr = fuse_radial && spline_on;
  const int TN = spl_N[idx], TNT = spl_Ntot, TOFF = spl_off[idx], tnsp = n_sp, tld = spl_ld, tlm = lmax + 1;
  auto tab = d_splR; auto spm = d_sp_of_mu; auto bc = d_bond_cutoff; auto mui = d_mu_i;
  // the dE/dA rows of cpp channels at a time are staged in the (then unused) Y / dY region
  const size_t gbytes = (size_t) GSZ * sizeof(GeomScalar);
  const int cpp = std::max(1, std::min(CB, (int) (gbytes / ((size_t) nfunc * sizeof(float)))));
  const size_t gsh = std::max(gbytes, (size_t) cpp * nfunc * sizeof(float));

  using team_policy = Kokkos::TeamPolicy<DeviceType>;
  using member = typename team_policy::member_type;
  team_policy pol(chunk_size * nblk, TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam((size_t) MSZ * sizeof(float) + gsh + 64));
  Kokkos::parallel_for("SPBFBwdT", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank() / nblk, c0 = (t.league_rank() - ii * nblk) * CB;
    float *Ms = (float *) t.team_shmem().get_shmem_aligned((size_t) MSZ * sizeof(float), 16);
    GeomScalar *Gs = (GeomScalar *) t.team_shmem().get_shmem_aligned(gsh, 16);
    const int tid = t.team_rank(), c = tid / 32, lane = tid - c * 32;
    const int h = lane / NBR, bl = lane - h * NBR, n = c0 + c;

    // M[c][y][h][k] = inv sum_{e in row (y, h NLH + k)} cg_e dE/dA[ii, c0 + c, f_e]; pad k = 25..27 with 0.
    // Thread per CG row p = (y, h NLH + k), its ELL entries loaded once for the pass's channels (ascending
    // columns as in the CSR; padded slots add an exact +0).
    for (int q = tid; q < CB * NY * H * (NLHP - NLH); q += TEAM) {
      const int r = q / (NLHP - NLH);
      Ms[r * NLHP + NLH + (q - r * (NLHP - NLH))] = 0.0f;
    }
    float *St = (float *) Gs;
    for (int cb = 0; cb < CB; cb += cpp) {
      const int nc = (CB - cb < cpp) ? CB - cb : cpp;
      const float *Ac = &AT(ii, c0);
      for (int f = tid; f < nfunc; f += TEAM) {
        const G3F4 a4 = *reinterpret_cast<const G3F4 *>(Ac + (long) f * NR);
        const float av[4] = {a4.x, a4.y, a4.z, a4.w};
        for (int cc = 0; cc < nc; cc++) St[cc * nfunc + f] = av[cb + cc];
      }
      t.team_barrier();
      for (int p = tid; p < NP; p += TEAM) {
        int col[EW]; float val[EW];
#pragma unroll
        for (int e = 0; e < EW; e++) {
          col[e] = 0; val[e] = 0.0f;
          if (e < ew) { col[e] = ecol(e * NP + p); val[e] = evl(e * NP + p); }
        }
        const int y = p / NLI, rem = p - y * NLI, hh = rem / NLH, k = rem - hh * NLH;
        for (int cc = 0; cc < nc; cc++) {
          const float *Sc = St + cc * nfunc;
          float s = 0.0f;
#pragma unroll
          for (int e = 0; e < EW; e++) if (e < ew) s += Sc[col[e]] * val[e];
          Ms[(((cb + cc) * NY + y) * H + hh) * NLHP + k] = s * inv;
        }
      }
      t.team_barrier();
    }

    const int nct = ncount(ii);
    for (int jb = 0; jb < nct; jb += NBR) {
      // stage [bond][y][Y, dYx, dYy, dYz] of this round (rows jb.. are contiguous in d_Y_bond / d_dY_bond)
      for (int q0 = tid; q0 < GSZ; q0 += TEAM * 4) {
        GeomScalar v[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const int q = q0 + u * TEAM;
          v[u] = GeomScalar(0.0);
          if (q < GSZ) {
            const int b = q / (NY * 4), r = q - b * (NY * 4), y = r / 4, k = r - 4 * y;
            const int jj = jb + b;
            if (jj < mn) v[u] = (k == 0) ? Y(ii, jj, y) : dY(ii, jj, y, k - 1);
          }
        }
#pragma unroll
        for (int u = 0; u < 4; u++) if (q0 + u * TEAM < GSZ) Gs[q0 + u * TEAM] = v[u];
      }
      t.team_barrier();
      const int jj = jb + bl;
      const bool valid = jj < nct;
      int jg = 0;
      float bv[NLHP], gI[NLHP], Rl[5];
      GeomScalar Re[5], DRe[5];
#pragma unroll
      for (int l = 0; l < 5; l++) { Rl[l] = 0.0f; Re[l] = GeomScalar(0.0); DRe[l] = GeomScalar(0.0); }
#pragma unroll
      for (int k = 0; k < NLHP; k++) { bv[k] = 0.0f; gI[k] = 0.0f; }
      if (valid) {
        jg = nearest(ii, jj);
        const GeomScalar ev = env(ii, jj), dev = denv(ii, jj);
        const float z = zproj(muj(ii, jj), n);
        const float *trow = nullptr;
        float tt = 0.0f, dxs = 0.0f;
        if (tabr) {
          const int mi = mui(ii), mj = muj(ii, jj);
          const float rc = bc(mi, mj), sx = (float) rn(ii, jj) / rc * TN;
          int k = (int) sx;
          if (k > TN - 1) k = TN - 1;
          tt = sx - k;
          dxs = (float) TN / rc;
          trow = tab.data() +
                 (((size_t) (spm(mi) * tnsp + spm(mj)) * TNT + TOFF + k) * tld + (size_t) n * tlm) * 4;
        }
#pragma unroll
        for (int l = 0; l < 5; l++) {
          float r, dr;
          if (tabr) {
            const G3F4 c = *reinterpret_cast<const G3F4 *>(trow + 4 * l);
            r = c.x + tt * (c.y + tt * (c.z + tt * c.w));
            dr = (c.y + tt * (2.0f * c.z + 3.0f * tt * c.w)) * dxs;
          } else {
            r = R(ii, jj, n, l);
            dr = DR(ii, jj, n, l);
          }
          Rl[l] = r * (float) ev;
          Re[l] = (GeomScalar) r * ev;
          DRe[l] = (GeomScalar) dr * ev + (GeomScalar) r * dev;
        }
#pragma unroll
        for (int k = 0; k < NLH; k++) bv[k] = ind(jg, n, h * NLH + k) + z * mask(h * NLH + k);
      }
      const GeomScalar *G = Gs + bl * NY * 4;
      GeomScalar sA = GeomScalar(0.0), sBx = GeomScalar(0.0), sBy = GeomScalar(0.0), sBz = GeomScalar(0.0);
#pragma unroll
      for (int l = 0; l < 5; l++) {
        for (int y = l * l; y < (l + 1) * (l + 1); y++) {
          // [Y, dYx, dYy, dYz] in one 16-byte load (fp32 geometry; scalar loads were 2-4-way bank conflicted)
          GeomScalar Yy, gx, gy, gz;
          if (sizeof(GeomScalar) == 4) {
            const G3F4 g4 = *reinterpret_cast<const G3F4 *>(G + 4 * y);
            Yy = (GeomScalar) g4.x; gx = (GeomScalar) g4.y; gy = (GeomScalar) g4.z; gz = (GeomScalar) g4.w;
          } else {
            Yy = G[4 * y]; gx = G[4 * y + 1]; gy = G[4 * y + 2]; gz = G[4 * y + 3];
          }
          const float aly = Rl[l] * (float) Yy;
          const G3F4 *My = reinterpret_cast<const G3F4 *>(Ms + ((c * NY + y) * H + h) * NLHP);
          // v = M b in four independent partial sums (one per float4 lane: no 28-long FMA chain)
          float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
#pragma unroll
          for (int k4 = 0; k4 < NLHP / 4; k4++) {
            const G3F4 m = My[k4];
            gI[4 * k4] += m.x * aly; gI[4 * k4 + 1] += m.y * aly; gI[4 * k4 + 2] += m.z * aly; gI[4 * k4 + 3] += m.w * aly;
            v0 += m.x * bv[4 * k4]; v1 += m.y * bv[4 * k4 + 1]; v2 += m.z * bv[4 * k4 + 2]; v3 += m.w * bv[4 * k4 + 3];
          }
          float v = (v0 + v1) + (v2 + v3);
          if (H == 2) v += g3l_shfl_xor(v, 16);
          const GeomScalar vg = (GeomScalar) v;
          sA += vg * DRe[l] * Yy;
          const GeomScalar vr = vg * Re[l];
          sBx += vr * gx; sBy += vr * gy; sBz += vr * gz;
        }
      }
      if (valid) {
#pragma unroll
        for (int k = 0; k < NLH; k++)
          if (gI[k] != 0.0f) Kokkos::atomic_add(&tgt(jg, n, h * NLH + k), gI[k]);
        if (h == 0) {
          const GeomScalar rinv = GeomScalar(1.0) / GeomScalar(rn(ii, jj));
          Kokkos::atomic_add(&fij(ii, jj, 0), sA * GeomScalar(rh(ii, jj, 0)) + rinv * sBx);
          Kokkos::atomic_add(&fij(ii, jj, 1), sA * GeomScalar(rh(ii, jj, 1)) + rinv * sBy);
          Kokkos::atomic_add(&fij(ii, jj, 2), sA * GeomScalar(rh(ii, jj, 2)) + rinv * sBz);
        }
      }
      t.team_barrier();
    }
  });
}

// As spbf_bwd_fast_t, with the L2 traffic of the per-lane gathers and atomics taken out:
//  - M from the ELL rows in one pass for the team's 4 channels (dE/dA of a column as one 16-byte load, no
//    shared staging of the dE/dA rows);
//  - b and dE/dI go through a per-warp slice of the Y / dY region, in two passes of 16 lanes (25 components
//    each): b rows are loaded and dE/dI rows added (atomic) with consecutive lanes on consecutive components,
//    instead of a lane walking its own row;
//  - the per-bond force is summed over the team's channels in shared before its atomic.
// M, b, v, dE/dI and the force partial of a channel are the same expressions as in spbf_bwd_fast_t.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
template<int H, int MB>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_bwd_fast2_t(
    int idx, const t_nn_3d &A_adj, const t_nn_3d_rg &indicator, const t_nn_3d_rg &target, bool rm)
{
  constexpr int CB = 4, NY = 25, NLH = 25, NLHP = 28, NLI = NLH * H, NBR = 32 / H, TEAM = 128;
  constexpr int MSZ = CB * NY * H * NLHP, GSZ = NBR * NY * 4, NP = NY * NLI, EW = GRACE3L_SPBF_ELL_MAX;
  constexpr int WS = 16 * NLH;   // per-warp slice: 16 lanes x 25 components
  static_assert(GSZ * sizeof(GeomScalar) >= CB * WS * sizeof(float), "Y / dY region too small for the slices");
  const DeviceSPBF &sp = d_spbf[idx];
  const int nblk = sp.n_rad_max / CB;
  auto zproj = sp.z_proj; auto mask = sp.chem_l0_mask;
  const G3I2 *ell2 = reinterpret_cast<const G3I2 *>(sp.cg_ell2.data()); const int ew = sp.ell_w;
  const float inv = (float) sp.inv_avg_n_neigh;
  const int NR = sp.n_rad_max, nfunc = sp.nfunc;
  if (!rm) spbf_transpose(A_adj, NR, nfunc, true);
  auto AT = d_spbf_T; auto ind = indicator; auto tgt = target;
  auto R = d_R1_nl; auto DR = d_DR1_nl; auto Y = d_Y_bond; auto dY = d_dY_bond; auto env = d_env; auto denv = d_denv;
  auto rn = d_rnorms; auto rh = d_rhats; auto ncount = d_ncount; auto nearest = d_nearest; auto muj = d_mu_j;
  auto fij = d_f_ij;
  // fuse_radial: R and dR/dr of this layer from the cubic tables (as radial_from_splines), not from d_R1_nl / d_DR1_nl
  const bool tabr = fuse_radial && spline_on;
  const int TN = spl_N[idx], TNT = spl_Ntot, TOFF = spl_off[idx], tnsp = n_sp, tld = spl_ld, tlm = lmax + 1;
  auto tab = d_splR; auto spm = d_sp_of_mu; auto bc = d_bond_cutoff; auto mui = d_mu_i;
  const size_t gbytes = (size_t) GSZ * sizeof(GeomScalar), fbytes = (size_t) CB * NBR * 4 * sizeof(GeomScalar);
  const bool stage_a = (size_t) nfunc * CB * sizeof(float) <= gbytes;
  // raw pointers and strides: the per-bond loads and atomics below without the views' index arithmetic. The Y / dY
  // rows of a round of bonds are contiguous ((lmax + 1)^2 = NY per bond, LayoutRight).
  if (Y.stride(1) != NY || dY.stride(1) != 3 * NY || ind.stride(2) != 1 || tgt.stride(2) != 1)
    GRACE_KK_ERROR("GRACE-3L/KK: spbf_bwd_fast2_t: unexpected Y / dY / indicator layout");
  if (!tabr && (R.stride(3) != 1 || DR.stride(0) != R.stride(0) || DR.stride(1) != R.stride(1) ||
                DR.stride(2) != R.stride(2)))
    GRACE_KK_ERROR("GRACE-3L/KK: spbf_bwd_fast2_t: unexpected R / dR layout");
  const float *Rp = R.data(), *DRp = DR.data();
  const long rsi = R.stride(0), rsj = R.stride(1), rsn = R.stride(2);
  const GeomScalar *Yp = Y.data(), *dYp = dY.data(), *envp = env.data(), *denvp = denv.data();
  const long ysi = Y.stride(0), dysi = dY.stride(0), es0 = env.stride(0), es1 = env.stride(1);
  const int *nrp = nearest.data(), *mjp = muj.data();
  const long ns0 = nearest.stride(0), ns1 = nearest.stride(1), ms0 = muj.stride(0), ms1 = muj.stride(1);
  const float *indp = ind.data();
  float *tgtp = tgt.data();
  const int isg = (int) ind.stride(0), isn = (int) ind.stride(1), tsg = (int) tgt.stride(0), tsn = (int) tgt.stride(1);

  using team_policy = Kokkos::TeamPolicy<DeviceType, Kokkos::LaunchBounds<MB ? TEAM : 0, MB>>;   // MB 0: default
  using member = typename team_policy::member_type;
  team_policy pol(chunk_size * nblk, TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam((size_t) MSZ * sizeof(float) + gbytes + fbytes +
                                                NLI * sizeof(float) + 64));
  auto body = KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank() / nblk, c0 = (t.league_rank() - ii * nblk) * CB;
    float *Ms = (float *) t.team_shmem().get_shmem_aligned((size_t) MSZ * sizeof(float), 16);
    GeomScalar *Gs = (GeomScalar *) t.team_shmem().get_shmem_aligned(gbytes, 16);
    GeomScalar *Fs = (GeomScalar *) t.team_shmem().get_shmem_aligned(fbytes, 16);
    float *mks = (float *) t.team_shmem().get_shmem(NLI * sizeof(float));
    const int tid = t.team_rank(), c = tid / 32, lane = tid - c * 32;
    const int h = lane / NBR, bl = lane - h * NBR, n = c0 + c;
    float *Wf = (float *) Gs + c * WS;

    for (int q = tid; q < CB * NY * H * (NLHP - NLH); q += TEAM) {
      const int r = q / (NLHP - NLH);
      Ms[r * NLHP + NLH + (q - r * (NLHP - NLH))] = 0.0f;
    }
    for (int q = tid; q < NLI; q += TEAM) mks[q] = mask(q);
    {
      // the atom's dE/dA columns of the team's channels first into the (still free) Y / dY region: the ELL walk
      // below then reads shared memory instead of a dependent global load per entry
      const float *Ac = &AT(ii, c0);
      float *As = (float *) Gs;
      if (stage_a) {
        for (int f = tid; f < nfunc; f += TEAM)
          *reinterpret_cast<G3F4 *>(As + 4 * f) = *reinterpret_cast<const G3F4 *>(Ac + (long) f * NR);
        t.team_barrier();
      }
      const float *Asrc = stage_a ? As : Ac;
      const long astr = stage_a ? 4 : NR;
      // two rows p per thread at a time, their packed ELL entries loaded together
      for (int p0 = tid; p0 < NP; p0 += 2 * TEAM) {
        G3I2 en[2][EW];
#pragma unroll
        for (int u = 0; u < 2; u++) {
          const int p = p0 + u * TEAM;
#pragma unroll
          for (int e = 0; e < EW; e++) en[u][e] = (e < ew && p < NP) ? ell2[e * NP + p] : G3I2{0, 0.0f};
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
          const int p = p0 + u * TEAM;
          if (p >= NP) break;
          float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
#pragma unroll
          for (int e = 0; e < EW; e++)
            if (e < ew) {
              const G3F4 a4 = *reinterpret_cast<const G3F4 *>(Asrc + en[u][e].i * astr);
              const float w = en[u][e].v;
              s0 += a4.x * w; s1 += a4.y * w; s2 += a4.z * w; s3 += a4.w * w;
            }
          const int y = p / NLI, rem = p - y * NLI, hh = rem / NLH, k = rem - hh * NLH;
          float *Mp = Ms + (y * H + hh) * NLHP + k;
          constexpr int CS = NY * H * NLHP;   // channel stride of Ms
          Mp[0] = s0 * inv; Mp[CS] = s1 * inv; Mp[2 * CS] = s2 * inv; Mp[3 * CS] = s3 * inv;
        }
      }
    }
    t.team_barrier();

    // lane's dE/dI components (sh NLH + lane, sh < H) with a nonzero M column for this channel: the others are
    // exact zeros for every bond, their atomics skipped (a per-lane predicate, no test on the values)
    const bool kl = lane < NLH;
    bool kuse[H];
#pragma unroll
    for (int sh = 0; sh < H; sh++) {
      bool u = false;
      if (kl)
        for (int y = 0; y < NY; y++) u = u || Ms[((c * NY + y) * H + sh) * NLHP + lane] != 0.0f;
      kuse[sh] = u;
    }
    const float *indn = indp + (long) n * isn + lane;
    float *tgtn = tgtp + (long) n * tsn + lane;

    const int nct = ncount(ii);
    for (int jb = 0; jb < nct; jb += NBR) {
      const int jj = jb + bl;
      const bool valid = jj < nct;
      const int jg = valid ? nrp[ii * ns0 + jj * ns1] : -1;
      float bv[NLHP], gI[NLHP], Rl[5];
      GeomScalar Re[5], DRe[5];
#pragma unroll
      for (int l = 0; l < 5; l++) { Rl[l] = 0.0f; Re[l] = GeomScalar(0.0); DRe[l] = GeomScalar(0.0); }
#pragma unroll
      for (int k = 0; k < NLHP; k++) { bv[k] = 0.0f; gI[k] = 0.0f; }
      float z = 0.0f;
      if (valid) {
        const GeomScalar ev = envp[ii * es0 + jj * es1], dev = denvp[ii * es0 + jj * es1];
        const int mj = mjp[ii * ms0 + jj * ms1];
        z = zproj(mj, n);
        const float *trow = nullptr;
        float tt = 0.0f, dxs = 0.0f;
        if (tabr) {
          const int mi = mui(ii);
          const float rc = bc(mi, mj), sx = (float) rn(ii, jj) / rc * TN;
          int k = (int) sx;
          if (k > TN - 1) k = TN - 1;
          tt = sx - k;
          dxs = (float) TN / rc;
          trow = tab.data() +
                 (((size_t) (spm(mi) * tnsp + spm(mj)) * TNT + TOFF + k) * tld + (size_t) n * tlm) * 4;
        }
#pragma unroll
        for (int l = 0; l < 5; l++) {
          float r, dr;
          if (tabr) {
            const G3F4 c = *reinterpret_cast<const G3F4 *>(trow + 4 * l);
            r = c.x + tt * (c.y + tt * (c.z + tt * c.w));
            dr = (c.y + tt * (2.0f * c.z + 3.0f * tt * c.w)) * dxs;
          } else {
            r = Rp[ii * rsi + jj * rsj + n * rsn + l];
            dr = DRp[ii * rsi + jj * rsj + n * rsn + l];
          }
          Rl[l] = r * (float) ev;
          Re[l] = (GeomScalar) r * ev;
          DRe[l] = (GeomScalar) dr * ev + (GeomScalar) r * dev;
        }
      }
      // b rows through the warp's slice: pass ps serves lanes 16 ps .. 16 ps + 15, slot s = lane - 16 ps, whose
      // (bond, half) is (bl, h) of that lane. Iteration s loads that row half with lane = component.
      for (int ps = 0; ps < 2; ps++) {
        float bw[16];   // all loads of the pass first, then the stores
#pragma unroll
        for (int s = 0; s < 16; s++) {
          const int ln = 16 * ps + s, sh = ln / NBR, sb = ln - sh * NBR;
          const int g = g3l_shfl(jg, sb);
          bw[s] = (kl && g >= 0) ? indn[(long) g * isg + sh * NLH] : 0.0f;
        }
#pragma unroll
        for (int s = 0; s < 16; s++)
          if (kl) Wf[s * NLH + lane] = bw[s];
        g3l_syncwarp();
        if ((lane >> 4) == ps) {
          const float *w = Wf + (lane - 16 * ps) * NLH;
#pragma unroll
          for (int k = 0; k < NLH; k++) bv[k] = w[k] + z * mks[h * NLH + k];
        }
        g3l_syncwarp();
      }
      t.team_barrier();
      // [Y, dYx, dYy, dYz] of (bond, y): thread per pair, rounds of TEAM pairs with their loads first. The round's
      // rows are contiguous in d_Y_bond / d_dY_bond (pair pr at offset pr, 3 pr); bonds past the count read as 0.
      const int npr = (nct - jb < NBR ? nct - jb : NBR) * NY;
      const GeomScalar *Yr = Yp + ii * ysi + (long) jb * NY, *dYr = dYp + ii * dysi + (long) jb * NY * 3;
      for (int p0 = 0; p0 < NBR * NY; p0 += 2 * TEAM) {
        GeomScalar v[2][4];
#pragma unroll
        for (int u = 0; u < 2; u++) {
          const int pr = p0 + tid + u * TEAM;
#pragma unroll
          for (int k = 0; k < 4; k++) v[u][k] = GeomScalar(0.0);
          if (pr < npr) {
            v[u][0] = Yr[pr]; v[u][1] = dYr[3 * pr]; v[u][2] = dYr[3 * pr + 1]; v[u][3] = dYr[3 * pr + 2];
          }
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
          const int pr = p0 + tid + u * TEAM;
          if (pr < NBR * NY) {
            GeomScalar *gp = Gs + 4 * pr;
            if (sizeof(GeomScalar) == 4)
              *reinterpret_cast<G3F4 *>(gp) = G3F4{(float) v[u][0], (float) v[u][1], (float) v[u][2], (float) v[u][3]};
            else { gp[0] = v[u][0]; gp[1] = v[u][1]; gp[2] = v[u][2]; gp[3] = v[u][3]; }
          }
        }
      }
      t.team_barrier();
      const GeomScalar *G = Gs + bl * NY * 4;
      GeomScalar sA = GeomScalar(0.0), sBx = GeomScalar(0.0), sBy = GeomScalar(0.0), sBz = GeomScalar(0.0);
#pragma unroll
      for (int l = 0; l < 5; l++) {
        for (int y = l * l; y < (l + 1) * (l + 1); y++) {
          GeomScalar Yy, gx, gy, gz;
          if (sizeof(GeomScalar) == 4) {
            const G3F4 g4 = *reinterpret_cast<const G3F4 *>(G + 4 * y);
            Yy = (GeomScalar) g4.x; gx = (GeomScalar) g4.y; gy = (GeomScalar) g4.z; gz = (GeomScalar) g4.w;
          } else {
            Yy = G[4 * y]; gx = G[4 * y + 1]; gy = G[4 * y + 2]; gz = G[4 * y + 3];
          }
          const float aly = Rl[l] * (float) Yy;
          const float *Myf = Ms + ((c * NY + y) * H + h) * NLHP;
          const G3F4 *My = reinterpret_cast<const G3F4 *>(Myf);
          float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
#pragma unroll
          for (int k4 = 0; k4 < NLH / 4; k4++) {
            const G3F4 m = My[k4];
            gI[4 * k4] += m.x * aly; gI[4 * k4 + 1] += m.y * aly; gI[4 * k4 + 2] += m.z * aly; gI[4 * k4 + 3] += m.w * aly;
            v0 += m.x * bv[4 * k4]; v1 += m.y * bv[4 * k4 + 1]; v2 += m.z * bv[4 * k4 + 2]; v3 += m.w * bv[4 * k4 + 3];
          }
          {   // component 24 (the zero-padded 25..27 of the 16-byte rows add exact zeros: skipped)
            static_assert(NLH % 4 == 1, "the tail below handles one component past the float4 loop");
            const float m = Myf[NLH - 1];
            gI[NLH - 1] += m * aly;
            v0 += m * bv[NLH - 1];
          }
          float v = (v0 + v1) + (v2 + v3);
          if (H == 2) v += g3l_shfl_xor(v, 16);
          const GeomScalar vg = (GeomScalar) v;
          sA += vg * DRe[l] * Yy;
          const GeomScalar vr = vg * Re[l];
          sBx += vr * gx; sBy += vr * gy; sBz += vr * gz;
        }
      }
      if (h == 0) {
        GeomScalar *F = Fs + (c * NBR + bl) * 4;
        F[0] = sA; F[1] = sBx; F[2] = sBy; F[3] = sBz;
      }
      t.team_barrier();
      // dE/dI rows through the slice (pass ps: lanes 16 ps .. 16 ps + 15 write theirs, the warp adds them)
      for (int ps = 0; ps < 2; ps++) {
        if ((lane >> 4) == ps) {
          float *w = Wf + (lane - 16 * ps) * NLH;
#pragma unroll
          for (int k = 0; k < NLH; k++) w[k] = gI[k];
        }
        g3l_syncwarp();
        float gw[16];
#pragma unroll
        for (int s = 0; s < 16; s++) gw[s] = kl ? Wf[s * NLH + lane] : 0.0f;
        g3l_syncwarp();
#pragma unroll
        for (int s = 0; s < 16; s++) {
          const int ln = 16 * ps + s, sh = ln / NBR, sb = ln - sh * NBR;
          const int g = g3l_shfl(jg, sb);
          if (g >= 0 && kuse[sh]) Kokkos::atomic_add(tgtn + (long) g * tsg + sh * NLH, gw[s]);
        }
      }
      // per-bond force: sum of the team's channels, then one atomic per component
      if (tid < NBR && jb + tid < nct) {
        const int jf = jb + tid;
        GeomScalar s4[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
          s4[u] = Fs[tid * 4 + u];
#pragma unroll
          for (int cc = 1; cc < CB; cc++) s4[u] += Fs[(cc * NBR + tid) * 4 + u];
        }
        const GeomScalar rinv = GeomScalar(1.0) / GeomScalar(rn(ii, jf));
        Kokkos::atomic_add(&fij(ii, jf, 0), s4[0] * GeomScalar(rh(ii, jf, 0)) + rinv * s4[1]);
        Kokkos::atomic_add(&fij(ii, jf, 1), s4[0] * GeomScalar(rh(ii, jf, 1)) + rinv * s4[2]);
        Kokkos::atomic_add(&fij(ii, jf, 2), s4[0] * GeomScalar(rh(ii, jf, 2)) + rinv * s4[3]);
      }
    }
  };
  // four teams per SM only when their scratch fits (a double Y / dY staging may not)
  if (MB && pol.team_size_max(body, Kokkos::ParallelForTag()) < TEAM) {
    spbf_bwd_fast2_t<H, 0>(idx, A_adj, indicator, target, true);    // dE/dA already row-major in d_spbf_T
    return;
  }
  Kokkos::parallel_for("SPBFBwdT2", pol, body);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_force_L1()
{
#ifdef KOKKOS_ENABLE_CUDA
  if (std::is_same<typename DeviceType::execution_space, Kokkos::Cuda>::value && sizeof(NNScalar) == 4 &&
      d_spbf[0].n_rad_max <= 64 && lmax <= 4 && spbf_ltile_std[0] && !getenv("GRACE3L_GENERIC_SPBF")) {
    force_L1_fast();
    return;
  }
#endif
  const int nradmax = d_spbf[0].n_rad_max;
  const int nlm = (lmax + 1) * (lmax + 1);
  int team_size = 1, vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1) team_size = 64;
  team_size = fit_team_size<DeviceType, TagForceL1>(*this, chunk_size, team_size, vector_length);
  auto policy = Kokkos::TeamPolicy<DeviceType, TagForceL1>(chunk_size, team_size, vector_length);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes<DeviceType, NNScalar>(nradmax * nlm)));
  Kokkos::parallel_for("ForceL1", policy, *this);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::compute_force_equiv(
    int spbf_idx, const t_nn_3d &A_adj, int ind_stage, int n_lm_ind)
{
  force_spbf_idx = spbf_idx;
  force_A_adj = A_adj;
  force_ind_stage = ind_stage;
  force_n_lm_ind = n_lm_ind;
  const int nradmax = d_spbf[spbf_idx].n_rad_max;
  // Opt 11: team-per-(ii,n) with the adj_prod row staged in team shared (old
  // per-thread TagForceEquiv kept for reference). team_size 64: the per-bond
  // body is register-heavy (bond_I[64] + w[25] + fp64 Plm/dPlm recurrence).
  const int nlm_y = (lmax + 1) * (lmax + 1);
  const int P = nlm_y * n_lm_ind;
  int team_size = 1, vector_length = 1;
  if (Kokkos::DefaultExecutionSpace().concurrency() > 1) team_size = 64;
  const int league = chunk_size * nradmax;
  int scratch_size = scratch_bytes<DeviceType, NNScalar>(P);
  team_size = fit_team_size<DeviceType, TagForceEquivFused>(*this, league, team_size, vector_length);
  auto policy = Kokkos::TeamPolicy<DeviceType, TagForceEquivFused>(league, team_size, vector_length);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
  Kokkos::parallel_for("ForceEquivFused", policy, *this);
}

// ======================================================================
// Template instantiations (members defined in this unit)
// ======================================================================

}    // namespace kk
}    // namespace grace

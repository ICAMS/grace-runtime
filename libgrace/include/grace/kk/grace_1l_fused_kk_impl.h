// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Implementation of grace::kk::Grace1LFusedKK (see grace_1l_fused_kk.h); included by one translation unit per consumer.
#pragma once

#include "grace/kk/grace_1l_fused_kk.h"
#include "grace/kk/grace_1l_kk_impl.h"
#include "grace/kk/fused_common.h"
#include "grace/kk/fused_dense.h"
#include "grace/kk/fused_hip_bonds.h"
#include "grace/kk/g1l_shape_large.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace grace {
namespace kk {
using namespace GraceFusedKK;

namespace g1lf_detail {

constexpr int G1LF_MAX_LMAX = GRACE_FUSED_MAX_LMAX;
constexpr int G1LF_TEAM = 256;      // threads per atom in the bond kernels
constexpr int G1LF_NB = 32;         // neighbours staged per batch (forward)
constexpr int G1LF_MAX_ACC = 16;    // per-thread accumulators: nradmax * nlm <= TEAM * MAX_ACC
constexpr int G1LF_NO = 64;         // FC widths compiled for: 64 (GRACE-1L large) and 32 (medium)
constexpr int G1LF_NO_M = 32;
constexpr int G1LF_NIN = 64;        // max channel width of the rho sources
constexpr int G1LF_NIR = 32;        // max right-input (A) width of the FCs
constexpr int G1LF_CC = 8;          // rho connections per thread (forward)

}    // namespace g1lf_detail
using namespace g1lf_detail;

/* ---------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::setup()
{
  if (spline_n < 16) GRACE_KK_ERROR("GRACE-1L/KK: spline N must be >= 16 (got {})", spline_n);
  this->couple_supported = true;    // two-group coupling in the bond kernels
  this->general_reduce_supported = true;    // its own reduce (build_red_meta): any collect_ind / w_l_tile layout
  Base::setup();
  const GRACE1LModel &m = *this->grace_model;
#ifdef KOKKOS_ENABLE_CUDA
  this->fc_blas_max_N = 0;    // the base cuBLAS FC gather/GEMM scratch is never used here
  this->fc_op_AA1.active = false;
  this->fc_op_AA2.active = false;
#endif
  const int nlm = (this->lmax + 1) * (this->lmax + 1);
  if (this->lmax > G1LF_MAX_LMAX)
    GRACE_KK_ERROR("GRACE-1L/KK: lmax {} > {} not supported", this->lmax, G1LF_MAX_LMAX);
  if (this->nradmax * nlm > G1LF_TEAM * G1LF_MAX_ACC || this->nradmax > G1LF_TEAM || G1LF_TEAM % this->nradmax)
    GRACE_KK_ERROR("GRACE-1L/KK: radial width {} outside the compiled bond kernels", this->nradmax);
  if (this->fc_n_out != G1LF_NO && this->fc_n_out != G1LF_NO_M)
    GRACE_KK_ERROR("GRACE-1L/KK: FC width {} not compiled (must be {} or {})", this->fc_n_out, G1LF_NO,
                     G1LF_NO_M);
  if (this->rho_n_out != 17) GRACE_KK_ERROR("GRACE-1L/KK: rho width must be 17");
  if (this->nradmax > G1LF_NIN || this->nradmax > G1LF_NIR)
    GRACE_KK_ERROR("GRACE-1L/KK: nradmax {} > {}", this->nradmax, G1LF_NIR);
  auto live = [](const GRACE1LModel::CGProduct &p, bool left, bool right) {
    std::vector<int> v;
    if (left) v.insert(v.end(), p.left_ind.begin(), p.left_ind.end());
    if (right) v.insert(v.end(), p.right_ind.begin(), p.right_ind.end());
    return v;
  };
  // A1 feeds AA (both operands) and AAA (right operand)
  std::vector<int> liveA1 = live(m.prod_AA, true, true), r = live(m.prod_AAA, false, true);
  liveA1.insert(liveA1.end(), r.begin(), r.end());
  build_fc_meta(fcm_A1, m.fc_A1, liveA1);
  build_fc_meta(fcm_AA1, m.fc_AA1, live(m.prod_AAA, true, false));
  build_fc_meta(fcm_AA2, m.fc_AA2, live(m.prod_AAAA, true, true));
  this->log.info(fmt("[GRACE-1L/KK] live FC outputs A1 {}/{} AA1 {}/{} AA2 {}/{}\n", fcm_A1.nlive,
                   fcm_A1.nf, fcm_AA1.nlive, fcm_AA1.nf, fcm_AA2.nlive, fcm_AA2.nf));
  build_prod_meta(pm_AA, m.prod_AA, true);
  build_prod_meta(pm_AAA, m.prod_AAA, false);
  build_prod_meta(pm_AAAA, m.prod_AAAA, true);
  build_red_meta();
  {   // shape-specialised CG products (generated): table match up to fp32 rounding
    auto same = [](const GRACE1LModel::CGProduct &p, int nt, const int (*tab)[3], const double *cg) {
      if ((int) p.cg_coeff.size() != nt) return false;
      for (int t = 0; t < nt; t++)
        if (p.left_ind[t] != tab[t][0] || p.right_ind[t] != tab[t][1] || p.m_sum_ind[t] != tab[t][2] ||
            std::fabs(p.cg_coeff[t] - cg[t]) > 1e-6 * std::max(1.0, std::fabs(cg[t]))) return false;
      return true;
    };
    namespace S = G1LShape_L;
    shape_id = (same(m.prod_AA, S::nt_AA, S::tab_AA, S::cg_AA) && same(m.prod_AAA, S::nt_AAA, S::tab_AAA, S::cg_AAA) &&
                same(m.prod_AAAA, S::nt_AAAA, S::tab_AAAA, S::cg_AAAA)) ? 1 : 0;
    if (no_shape) shape_id = 0;
    ng_AAA = 0;
    for (int x : m.prod_AAA.left_ind) ng_AAA = std::max(ng_AAA, x + 1);
    this->log.info(fmt("[GRACE-1L/KK] CG products {}\n",
                     shape_id ? "specialised: GRACE-1L large shape" : "generic (tables differ from the compiled shape)"));
  }
  {
    d_lfrom = V1<int>("g1lf:lfrom", nlm);
    auto h = Kokkos::create_mirror_view(d_lfrom);
    for (int l = 0; l <= this->lmax; l++)
      for (int lm = l * l; lm < (l + 1) * (l + 1); lm++) h(lm) = l;
    Kokkos::deep_copy(d_lfrom, h);
  }
  build_spline_tables();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::build_spline_tables()
{
  const GRACE1LModel &m = *this->grace_model;
  const int N = spline_n, p = this->radial_basis_p, nrb = this->nradbase;
  const double h = 1.0 / N;
  const auto &layers = m.mlp_rad_layers;
  const int nout = layers.back().n_out;
  if (nout != this->nradmax * (this->lmax + 1))
    GRACE_KK_ERROR("GRACE-1L/KK: radial MLP output width mismatch");
  std::vector<double> fv((size_t) (N + 1) * nout), dv((size_t) (N + 1) * nout);
  std::vector<double> g, dg, out, dout;
  for (int k = 0; k <= N; k++) {
    radial_basis_x(k * h, p, nrb, g, dg);
    mlp_x(layers, g, dg, out, dout);
    for (int o = 0; o < nout; o++) { fv[(size_t) k * nout + o] = out[o]; dv[(size_t) k * nout + o] = h * dout[o]; }
  }
  d_splR = V3<F>("g1lf:R", N, nout, 4);
  auto hc = Kokkos::create_mirror_view(d_splR);
  for (int k = 0; k < N; k++)
    for (int o = 0; o < nout; o++) {
      const double f0 = fv[(size_t) k * nout + o], f1 = fv[(size_t) (k + 1) * nout + o];
      const double d0 = dv[(size_t) k * nout + o], d1 = dv[(size_t) (k + 1) * nout + o];
      hc(k, o, 0) = (F) f0;
      hc(k, o, 1) = (F) d0;
      hc(k, o, 2) = (F) (-3.0 * f0 - 2.0 * d0 + 3.0 * f1 - d1);
      hc(k, o, 3) = (F) (2.0 * f0 + d0 - 2.0 * f1 + d1);
    }
  Kokkos::deep_copy(d_splR, hc);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, class Traits>
typename Grace1LKK<DeviceType, float, float, Traits>::RunOutput
Grace1LFusedKK<DeviceType, Traits>::run(const typename Base::RunInput &in)
{
  const int inum = in.inum;
  if ((int) d_vir.extent(0) < inum) d_vir = V2<double>("g1lf:vir", inum, 6);
  Kokkos::deep_copy(d_vir, 0.0);
  auto out = Base::run(in);
  bonds_ready = false;    // the compact bond list belongs to that run (begin_run)
  if (this->vflag_global) {
    for (int k = 0; k < 6; k++) {
      auto vir = d_vir;
      double s = 0.0;
      Kokkos::parallel_reduce("g1lf:virial", Kokkos::RangePolicy<DeviceType>(0, inum),
          KOKKOS_LAMBDA(const int i, double &acc) { acc += vir(i, k); }, s);
      out.virial[k] += s;
    }
  }
  if (this->couple_eq) Kokkos::deep_copy(out.dedl, this->d_dedl);
  check_kernel_launches<DeviceType>("GRACE-1L/KK");
  return out;
}

/* ----------------------------------------------------------------------
   Bond kernel: A[i, n, lm] = sum_j R_nl(x_ij) Y_lm(r_ij) z[mu_j, n]  (z includes 1/<N>)
   One team per atom; neighbours staged NB at a time (geometry, Y, spline R) in scratch.
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::fused_A()
{
#if GRACE_KK_GEMM_MFMA
  if constexpr (is_hip_v<DeviceType>) {
    if (a_mfma) {    // CDNA: matrix-core basis (begin_run)
      auto run = [&](auto cpl) {
        GraceFusedKK::a_fwd_mfma<DeviceType, Traits, decltype(cpl)::value>(
            "g1lf:A", this->chunk_size, this->chunk_offset, this->nradmax, 1.0f, spline_n, this->d_A, this->x,
            this->type, this->d_map, bond_list(), this->d_ilist, this->d_bond_cutoff, this->d_z_tr,
            this->couple_eq, this->couple_bit1, this->couple_bit2, this->couple_lambda, this->d_mask,
            this->d_idx_sph, this->alm, this->blm, this->cl, this->dl, d_splR);
      };
      if (this->couple_eq) run(std::true_type{});
      else run(std::false_type{});
      return;
    }
  }
#endif
  if (this->couple_eq) fused_A_c<true>();    // two-group coupling compiled only into its own kernels
  else fused_A_c<false>();
}

template<class DeviceType, class Traits>
template<bool CPL>
void Grace1LFusedKK<DeviceType, Traits>::fused_A_c()
{
  const int cs = this->chunk_size;
  const int lmax = this->lmax, nl = lmax + 1, nlm = nl * nl;
  const int NR = this->nradmax, nout = NR * nl, nA = NR * nlm;
  auto Af = this->d_A;
  const int co = this->chunk_offset;
  auto x = this->x; auto type = this->type; auto map = this->d_map;
  auto ilist = this->d_ilist;
  auto nbl = bond_list();    // the atom's bonds (GPU: the run's compact list)
  auto bc = this->d_bond_cutoff; auto ztr = this->d_z_tr; auto lfrom = d_lfrom;
  const int ceq = this->couple_eq, cb1 = this->couple_bit1, cb2 = this->couple_bit2;    // two-group coupling (CPL)
  const double clam = this->couple_lambda;
  auto cmask = this->d_mask;
  auto idx_s = this->d_idx_sph; auto alm = this->alm; auto blm = this->blm;
  auto cl = this->cl; auto dl = this->dl;
  auto sc = d_splR;
  const int N = spline_n;

  using team_policy = Kokkos::TeamPolicy<DeviceType, hip_lb_t<DeviceType, G1LF_TEAM, 2>>;
  using member = typename team_policy::member_type;
  // GPU: z[mu_j, n] folded into the staged radial (Rs = R z, zero for the bonds out of range and Y zeroed for them),
  // so the accumulation is one FMA per (output, bond) on two shared reads, without a branch on mus[b]
  constexpr bool gpu = is_gpu_v<DeviceType>;
  const size_t shmem = (size_t) G1LF_NB * (nlm + nout + 1) * sizeof(F) + G1LF_NB * sizeof(int);
  team_policy pol(cs, G1LF_TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(shmem));

  Kokkos::parallel_for("g1lf:A", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank();
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *Ys = (F *) t.team_shmem().get_shmem(G1LF_NB * nlm * sizeof(F));
    F *Rs = (F *) t.team_shmem().get_shmem(G1LF_NB * nout * sizeof(F));
    F *xs = (F *) t.team_shmem().get_shmem(G1LF_NB * sizeof(F));
    int *mus = (int *) t.team_shmem().get_shmem(G1LF_NB * sizeof(int));
    const int tid = t.team_rank(), ts = t.team_size();

    F acc[G1LF_MAX_ACC];
    for (int k = 0; k < G1LF_MAX_ACC; k++) acc[k] = 0.0f;

    const int jnum = nbl.count(i, ii + co);
    for (int jb = 0; jb < jnum; jb += G1LF_NB) {
      const int nb = (jnum - jb < G1LF_NB) ? jnum - jb : G1LF_NB;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, G1LF_NB), [&](const int b) {
        int mu = -1;
        if (b < nb) {
          const int j = nbl(i, ii + co, jb + b);
          const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi;    // bond geometry in fp64
          const double r2 = dx * dx + dy * dy + dz * dz;
          const int muj = map(type(j));
          const double rc = bc(mui, muj);
          if (r2 < rc * rc) {
            const double r = Kokkos::sqrt(r2), rinv = 1.0 / r;
            double rm = r, drm, dlm;    // the model's bond length: r' of a coupled bond (grace/coupling.h)
            if (CPL && coupling_cross(cmask(i), cmask(j), cb1, cb2)) coupling_shift(ceq, r, rc, clam, rm, drm, dlm);
            if (!CPL || rm < rc) {
              mu = muj;
              xs[b] = F(rm / rc);
              real_sph(lmax, F(dx * rinv), F(dy * rinv), F(dz * rinv), idx_s, alm, blm, cl, dl, Ys + b * nlm);
            }
          }
        }
        if (gpu && mu < 0)
          for (int lm = 0; lm < nlm; lm++) Ys[b * nlm + lm] = 0.0f;
        mus[b] = mu;
      });
      t.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, G1LF_NB * nout), [&](const int q) {
        const int b = q / nout, o = q - b * nout;
        if (gpu) Rs[q] = (mus[b] >= 0) ? spline_v(sc, N, xs[b], o) * ztr(mus[b], o / nl) : 0.0f;
        else Rs[q] = (mus[b] >= 0) ? spline_v(sc, N, xs[b], o) : 0.0f;
      });
      t.team_barrier();
#pragma unroll
      for (int k = 0; k < G1LF_MAX_ACC; k++) {
        const int a = tid + k * ts;
        if (a >= nA) continue;
        const int n = a / nlm, lm = a - n * nlm, o = n * nl + lfrom(lm);
        F s = acc[k];
        if (gpu) {
          for (int b = 0; b < nb; b++) s += Rs[b * nout + o] * Ys[b * nlm + lm];
        } else {
          for (int b = 0; b < nb; b++) {
            const int mu = mus[b];
            if (mu >= 0) s += Rs[b * nout + o] * Ys[b * nlm + lm] * ztr(mu, n);
          }
        }
        acc[k] = s;
      }
      t.team_barrier();
    }
#pragma unroll
    for (int k = 0; k < G1LF_MAX_ACC; k++) {
      const int a = tid + k * ts;
      if (a >= nA) continue;
      const int n = a / nlm, lm = a - n * nlm;
      Af(ii, n, lm) = acc[k];
    }
  });
}

/* ----------------------------------------------------------------------
   Bond backward: from dE/dA of atom i, per bond the force dE/dr_ij (atomics) and the virial.
   Rounds of TEAM / NR neighbours: thread (b, n), n = radial channel.
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::fused_bond_bwd()
{
  if constexpr (is_gpu_v<DeviceType>) {
    if (bwd_wave) {    // GPU: one 64-thread team per atom, thread = bond (begin_run)
      auto run = [&](auto cpl) {
        GraceFusedKK::bond_bwd_wave<DeviceType, Traits, decltype(cpl)::value>(
            "g1lf:bond_bwd", this->chunk_size, this->chunk_offset, this->nradmax, 1.0f, spline_n, this->d_A_adj,
            this->f, d_vir, this->x, this->type, this->d_map, bond_list(), this->d_ilist, this->d_bond_cutoff,
            this->d_z_tr, this->couple_eq, this->couple_bit1, this->couple_bit2, this->couple_lambda, this->d_mask,
            this->d_dedl, this->d_idx_sph, this->alm, this->blm, this->cl, this->dl, d_splR, this->vflag_global != 0,
            this->vflag_atom != 0, this->cvflag_atom != 0, this->d_vatom, this->d_cvatom);
      };
      if (this->couple_eq) run(std::true_type{});
      else run(std::false_type{});
      return;
    }
  }
  if (this->couple_eq) fused_bond_bwd_c<true>();
  else fused_bond_bwd_c<false>();
}

template<class DeviceType, class Traits>
template<bool CPL>
void Grace1LFusedKK<DeviceType, Traits>::fused_bond_bwd_c()
{
  const int cs = this->chunk_size, co = this->chunk_offset;
  const int lmax = this->lmax, nl = lmax + 1, nlm = nl * nl;
  const int NR = this->nradmax;
  const int NBR = G1LF_TEAM / NR;

  auto Aa = this->d_A_adj;
  auto f = this->f; auto vir = d_vir;
  auto x = this->x; auto type = this->type; auto map = this->d_map;
  auto ilist = this->d_ilist;
  auto nbl = bond_list();    // the atom's bonds (GPU: the run's compact list)
  auto bc = this->d_bond_cutoff; auto ztr = this->d_z_tr;
  const int ceq = this->couple_eq, cb1 = this->couple_bit1, cb2 = this->couple_bit2;    // two-group coupling (CPL)
  const double clam = this->couple_lambda;
  auto cmask = this->d_mask;
  auto dedl = this->d_dedl;
  auto idx_s = this->d_idx_sph; auto alm = this->alm; auto blm = this->blm;
  auto cl = this->cl; auto dl = this->dl;
  auto sc = d_splR;
  const int N = spline_n;
  const bool do_v = this->vflag_global != 0;
  const bool do_va = this->vflag_atom != 0, do_cva = this->cvflag_atom != 0;
  auto va = this->d_vatom; auto cva = this->d_cvatom;

  using member = typename Kokkos::TeamPolicy<DeviceType>::member_type;    // launch: fbwd_launch
  // GPU: dE/dA rows as [channel][lm] (LMP floats, 16-byte loads), Y and G of a bond as [lm][Y, Gx, Gy, Gz]
  constexpr bool gpu = is_gpu_v<DeviceType>;
  const int LMP = fused_lmp(nlm);
  // GPU: the bond geometry of NBG >= NBR bonds per step (more lanes busy, fewer barriers), then rounds of NBR
  const int NBG = gpu ? (NBR > 48 ? NBR : 48 / NBR * NBR) : NBR;
  const int nf = (gpu ? NR * LMP : nlm * NR) + NBG * (4 * nlm + 8) + 3 * G1LF_TEAM;
  const size_t shmem = (size_t) nf * sizeof(F) + 2 * NBG * sizeof(int) + (gpu ? 16 : 0);
  auto body = KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank();
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *Aa_s;
    if (gpu) Aa_s = (F *) t.team_shmem().get_shmem_aligned(nf * sizeof(F), 16);
    else Aa_s = (F *) t.team_shmem().get_shmem(nf * sizeof(F));
    int *mus = (int *) t.team_shmem().get_shmem(2 * NBG * sizeof(int));
    int *js = mus + NBG;
    F *Ys = Aa_s + (gpu ? NR * LMP : nlm * NR), *Gs = Ys + NBG * nlm, *geo = Gs + NBG * 3 * nlm, *Fs = geo + NBG * 8;
    const int tid = t.team_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nlm * NR), [&](const int q) {
      const int lm = q / NR, c = q - lm * NR;
      if (gpu) Aa_s[c * LMP + lm] = Aa(ii, c, lm);
      else Aa_s[q] = Aa(ii, c, lm);
    });
    double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dla = 0.0;    // dE/dlambda (coupling)
    t.team_barrier();

    const int jnum = nbl.count(i, ii + co);
    for (int jb = 0; jb < jnum; jb += NBG) {
      const int nb = (jnum - jb < NBG) ? jnum - jb : NBG;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NBG), [&](const int b) {
        int mu = -1;
        if (b < nb) {
          const int j = nbl(i, ii + co, jb + b);
          const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi;    // bond geometry in fp64
          const double r2 = dx * dx + dy * dy + dz * dz;
          const int muj = map(type(j));
          const double rc = bc(mui, muj);
          if (r2 < rc * rc) {
            const double r = Kokkos::sqrt(r2), rinv = 1.0 / r;
            double rm = r, drm = 1.0, dlm = 0.0;    // coupled bond (grace/coupling.h): r', dr'/dr, dr'/dlambda
            if (CPL && coupling_cross(cmask(i), cmask(j), cb1, cb2)) coupling_shift(ceq, r, rc, clam, rm, drm, dlm);
            if (!CPL || rm < rc) {
              mu = muj; js[b] = j;
              F *gb = geo + b * 8;    // radial part at r' (times dr'/dr), angular part and virial at the true r
              gb[0] = F(rm / rc); gb[1] = F(drm / rc); gb[2] = F(rinv);
              gb[3] = F(dx * rinv); gb[4] = F(dy * rinv); gb[5] = F(dz * rinv);
              gb[6] = F(r);
              if (CPL) gb[7] = F(dlm / rc);
              if (gpu)
                real_sph_grad<4, 4>(lmax, gb[3], gb[4], gb[5], idx_s, alm, blm, cl, dl, Ys + b * 4 * nlm, Ys + b * 4 * nlm + 1);
              else
                real_sph_grad(lmax, gb[3], gb[4], gb[5], idx_s, alm, blm, cl, dl, Ys + b * nlm, Gs + b * 3 * nlm);
            }
          }
        }
        mus[b] = mu;
      });
      t.team_barrier();
      for (int sb = 0; sb < nb; sb += NBR) {
      {
        const int b = sb + tid / NR, n = tid - (tid / NR) * NR;
        F fx = 0.0f, fy = 0.0f, fz = 0.0f;
        if (tid < NBR * NR && b < nb && mus[b] >= 0) {    // NBR NR < TEAM when NR does not divide it
          const int mu = mus[b];
          const F *gb = geo + b * 8, *Yb = Ys + b * nlm, *Gb = Gs + b * 3 * nlm;
          F R[G1LF_MAX_LMAX + 1], dR[G1LF_MAX_LMAX + 1], gR[G1LF_MAX_LMAX + 1];
#pragma unroll
          for (int l = 0; l <= G1LF_MAX_LMAX; l++) {
            R[l] = dR[l] = gR[l] = 0.0f;
            if (l < nl) spline_vd(sc, N, gb[0], n * nl + l, R[l], dR[l]);
          }
          F gx = 0.0f, gy = 0.0f, gz = 0.0f;
          const F z = ztr(mu, n);
          if (gpu) {
            // the same sums: dE/dA row of channel n as float4, [Y, G] of (bond, lm) as one float4
            constexpr int LMX = fused_lmp((G1LF_MAX_LMAX + 1) * (G1LF_MAX_LMAX + 1));
            F aa[LMX];
            const GF4 *a4 = reinterpret_cast<const GF4 *>(Aa_s + n * LMP);
#pragma unroll
            for (int q = 0; q < LMX / 4; q++) {
              GF4 v = {0.0f, 0.0f, 0.0f, 0.0f};
              if (4 * q < nlm) v = a4[q];
              aa[4 * q] = v.x; aa[4 * q + 1] = v.y; aa[4 * q + 2] = v.z; aa[4 * q + 3] = v.w;
            }
            const GF4 *yg = reinterpret_cast<const GF4 *>(Ys + b * 4 * nlm);
            if (is_hip_v<DeviceType>) {    // HIP-Clang unrolls the nested loops (the flat one was 1-2 % slower)
#pragma unroll
              for (int l = 0; l <= G1LF_MAX_LMAX; l++) {
                if (l >= nl) continue;
#pragma unroll
                for (int lm = l * l; lm < (l + 1) * (l + 1); lm++) {
                  const GF4 q = yg[lm];
                  const F w = aa[lm] * z;
                  gR[l] += w * q.x;
                  const F wr = w * R[l];
                  gx += wr * q.y; gy += wr * q.z; gz += wr * q.w;
                }
              }
            } else {    // one flat lm loop: nvcc keeps aa in registers (l_of_y)
#pragma unroll
              for (int lm = 0; lm < (G1LF_MAX_LMAX + 1) * (G1LF_MAX_LMAX + 1); lm++) {
                const int l = l_of_y(lm);
                if (l >= nl) continue;
                const GF4 q = yg[lm];
                const F w = aa[lm] * z;
                gR[l] += w * q.x;
                const F wr = w * R[l];
                gx += wr * q.y; gy += wr * q.z; gz += wr * q.w;
              }
            }
          } else {
#pragma unroll
          for (int l = 0; l <= G1LF_MAX_LMAX; l++) {
            if (l >= nl) continue;
            for (int lm = l * l; lm < (l + 1) * (l + 1); lm++) {
              const F w = Aa_s[lm * NR + n] * z;
              gR[l] += w * Yb[lm];
              const F wr = w * R[l];
              gx += wr * Gb[3 * lm]; gy += wr * Gb[3 * lm + 1]; gz += wr * Gb[3 * lm + 2];
            }
          }
          }
          F rad = 0.0f;
#pragma unroll
          for (int l = 0; l <= G1LF_MAX_LMAX; l++) rad += gR[l] * dR[l];
          if (CPL) dla += rad * gb[7];    // dE/dlambda: dE/dx' dx'/dlambda
          rad *= gb[1];
          const F rx = gb[3], ry = gb[4], rz = gb[5];
          const F rg = rx * gx + ry * gy + rz * gz;
          fx = rad * rx + (gx - rg * rx) * gb[2];
          fy = rad * ry + (gy - rg * ry) * gb[2];
          fz = rad * rz + (gz - rg * rz) * gb[2];
        }
        if (tid < NBR * NR) { Fs[3 * tid] = fx; Fs[3 * tid + 1] = fy; Fs[3 * tid + 2] = fz; }
      }
      t.team_barrier();
      if (tid < NBR && sb + tid < nb && mus[sb + tid] >= 0) {
        const int b = sb + tid, j = js[b];
        F fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int c = 0; c < NR; c++) {
          fx += Fs[3 * (tid * NR + c)]; fy += Fs[3 * (tid * NR + c) + 1]; fz += Fs[3 * (tid * NR + c) + 2];
        }
        Kokkos::atomic_add(&f(i, 0), (typename Traits::kk_acc_float) fx);
        Kokkos::atomic_add(&f(i, 1), (typename Traits::kk_acc_float) fy);
        Kokkos::atomic_add(&f(i, 2), (typename Traits::kk_acc_float) fz);
        Kokkos::atomic_add(&f(j, 0), (typename Traits::kk_acc_float) -fx);
        Kokkos::atomic_add(&f(j, 1), (typename Traits::kk_acc_float) -fy);
        Kokkos::atomic_add(&f(j, 2), (typename Traits::kk_acc_float) -fz);
        if (do_v) {
          const F *gb = geo + b * 8;
          const double dx = -gb[3] * gb[6], dy = -gb[4] * gb[6], dz = -gb[5] * gb[6];
          v[0] += dx * fx; v[1] += dy * fy; v[2] += dz * fz;
          v[3] += dx * fy; v[4] += dx * fz; v[5] += dy * fz;
        }
        if (do_va || do_cva) {
          const F *gb = geo + b * 8;
          tally_vatom(va, cva, do_va, do_cva, i, j, -gb[3] * gb[6], -gb[4] * gb[6], -gb[5] * gb[6], fx, fy, fz);
        }
      }
      t.team_barrier();
      }
    }
    if (do_v && tid < NBR)
      for (int k = 0; k < 6; k++) Kokkos::atomic_add(&vir(ii + co, k), v[k]);
    if (CPL) {    // dE/dlambda of this atom's bonds
      double s = dla;
      t.team_reduce(Kokkos::Sum<double>(s));
      if (tid == 0) Kokkos::atomic_add(&dedl(), s);
    }
  };
  GraceFusedKK::fbwd_launch<DeviceType, G1LF_TEAM>("g1lf:bond_bwd", cs, shmem, body);
}
// dense-chain kernels and metadata: shared with Grace2LFusedKK in grace/kk/fused_dense.h
template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::build_fc_meta(FCMeta &m, const GRACE1LModel::FCWeights &fc, const std::vector<int> &used)
{
  GraceFusedKK::build_fc_meta(m, fc, used, "GRACE-1L/KK");
}

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::build_prod_meta(ProdMeta &m, const GRACE1LModel::CGProduct &p, bool self)
{
  GraceFusedKK::build_prod_meta(m, p, self);
}

template<class DeviceType, class Traits>
template<int NO>
void Grace1LFusedKK<DeviceType, Traits>::fc_fwd(const FCMeta &m, const t_nn3 &in_l, const t_nn3 &in_r, const t_nn3 &out)
{
  GraceFusedKK::fc_fwd<DeviceType, NO>(m, this->chunk_size, in_l, in_r, out);
}

template<class DeviceType, class Traits>
template<int NO>
void Grace1LFusedKK<DeviceType, Traits>::fc_fwd4(const FCMeta &m, const t_nn3 &in_l, const t_nn3 &in_r, const t_nn3 &out)
{
  GraceFusedKK::fc_fwd4<DeviceType, NO>(m, this->chunk_size, in_l, in_r, out);
}

template<class DeviceType, class Traits>
template<int NO, int NIR>
void Grace1LFusedKK<DeviceType, Traits>::fc_bwd(const FCMeta &m, const t_nn3 &dout, const t_nn3 &din_l, const t_nn3 &din_r)
{
  GraceFusedKK::fc_bwd<DeviceType, NO, NIR>(m, this->chunk_size, dout, din_l, din_r);
}

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::prod_fwd(const ProdMeta &m, const t_nn3 &L, const t_nn3 &R, const t_nn3 &out)
{
  GraceFusedKK::prod_fwd(m, this->chunk_size, L, R, out);
}

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::prod_rev(const ProdMeta &m, const t_nn3 &L, const t_nn3 &R, const t_nn3 &D, const t_nn3 &dL,
                                                  const t_nn3 &dR, bool acc_left)
{
  GraceFusedKK::prod_rev(m, this->chunk_size, L, R, D, dL, dR, acc_left);
}



/* ----------------------------------------------------------------------
   rho = sum_s sum_c W_s[mu_i][:, n, tile c] X_s[i, n, ci c]   (s: A, AA, AAA, AAAA)
   forward: thread per (atom, block of CC connections, output quad), atomics into rho (17 per atom,
   zeroed per chunk by the base); reverse: thread per (atom, source function), assigns dX.
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::build_red_meta()
{
  const GRACE1LModel &m = *this->grace_model;
  const GRACE1LModel::ReduceWeights *rws[4] = {&m.reduce_A, &m.reduce_AA, &m.reduce_AAA, &m.reduce_AAAA};
  const int nfi[4] = {(this->lmax + 1) * (this->lmax + 1), this->n_funcs_AA, this->n_funcs_AAA, this->n_funcs_AAAA};
  const int ntypes = ((int) this->h_type2model.size() - 1), no = this->rho_n_out, nop = (no + 3) / 4 * 4;
  rm.nx = 4;
  int gtot = 0;
  std::vector<std::array<int, 3>> conns;                        // (x, fi, tile), sorted by x
  std::vector<std::vector<int>> by_g;
  for (int x = 0; x < 4; x++) {
    const auto &rw = *rws[x];
    const int nin = rw.n_in, nw = rw.w_shape > 0 ? rw.w_shape : 1;
    rm.nin[x] = nin; rm.nfi[x] = nfi[x]; rm.gbase[x] = gtot;
    if (nin > G1LF_NIN) GRACE_KK_ERROR("GRACE-1L/KK: rho source width {} > {}", nin, G1LF_NIN);
    by_g.resize(gtot + nfi[x]);
    for (size_t c = 0; c < rw.collect_ind.size(); c++) {
      const int fi = rw.collect_ind[c], tile = rw.w_l_tile[c];
      if (fi >= nfi[x] || tile >= nw) GRACE_KK_ERROR("GRACE-1L/KK: rho reduce index out of range");
      conns.push_back({x, fi, tile});
      by_g[gtot + fi].push_back(tile);
    }
    gtot += nfi[x];
    V4<F> W(std::string("g1lf:rhoW") + std::to_string(x), ntypes + 1, nw, nin, nop);
    auto hW = Kokkos::create_mirror_view(W);
    for (int ty = 0; ty <= ntypes; ty++) {
      const int mu = ty >= 1 ? this->h_type2model[ty] : -1;
      for (int t = 0; t < nw; t++)
        for (int n = 0; n < nin; n++)
          for (int k = 0; k < nop; k++) {
            double v = 0.0;
            if (k < no && mu >= 0) v = rw.norm * rw.W[(((size_t) mu * no + k) * nin + n) * nw + t];
            hW(ty, t, n, k) = (F) v;
          }
    }
    Kokkos::deep_copy(W, hW);
    rm.W[x] = W;
  }
  rm.gtot = gtot;
  const int nc = (int) conns.size();
  rm.nconn = nc;
  rm.tx = V1<int>("g1lf:rho_tx", std::max(nc, 1)); rm.tfi = V1<int>("g1lf:rho_tfi", std::max(nc, 1));
  rm.ttile = V1<int>("g1lf:rho_ttile", std::max(nc, 1));
  {
    auto hx = Kokkos::create_mirror_view(rm.tx); auto hf = Kokkos::create_mirror_view(rm.tfi);
    auto ht = Kokkos::create_mirror_view(rm.ttile);
    for (int c = 0; c < nc; c++) { hx(c) = conns[c][0]; hf(c) = conns[c][1]; ht(c) = conns[c][2]; }
    Kokkos::deep_copy(rm.tx, hx); Kokkos::deep_copy(rm.tfi, hf);
    Kokkos::deep_copy(rm.ttile, ht);
  }
  int tot = 0;
  for (auto &v : by_g) tot += v.size();
  rm.roff = V1<int>("g1lf:rho_roff", gtot + 1); rm.rtile = V1<int>("g1lf:rho_rtile", std::max(tot, 1));
  rm.rx = V1<int>("g1lf:rho_rx", 1);
  auto ho = Kokkos::create_mirror_view(rm.roff); auto ht = Kokkos::create_mirror_view(rm.rtile);
  int q = 0;
  for (int g = 0; g < gtot; g++) {
    ho(g) = q;
    for (int t : by_g[g]) ht(q++) = t;
  }
  ho(gtot) = q;
  Kokkos::deep_copy(rm.roff, ho); Kokkos::deep_copy(rm.rtile, ht);
}

template<class DeviceType, class Traits>
template<int NQ>
void Grace1LFusedKK<DeviceType, Traits>::red_fwd()
{
  const int cs = this->chunk_size, co = this->chunk_offset;
  const int nc = rm.nconn;
  const int nblk = (nc + G1LF_CC - 1) / G1LF_CC;
  auto W0 = rm.W[0]; auto W1 = rm.W[1]; auto W2 = rm.W[2]; auto W3 = rm.W[3];
  const int n0 = rm.nin[0], n1 = rm.nin[1], n2 = rm.nin[2], n3 = rm.nin[3];
  auto X0 = this->d_A; auto X1 = this->d_AA; auto X2 = this->d_AAA; auto X3 = this->d_AAAA;
  auto tx = rm.tx; auto tfi = rm.tfi; auto ttile = rm.ttile;
  auto rho = this->d_rho;
  auto type = this->type; auto ilist = this->d_ilist;
  const int no = this->rho_n_out;
  Kokkos::parallel_for("g1lf:Reduce", Kokkos::RangePolicy<DeviceType, hip_lb_t<DeviceType, 256, 2>>(0, cs * nblk * NQ), KOKKOS_LAMBDA(const int q) {
    const int ii = q % cs, r = q / cs, blk = r / NQ, kq = r - blk * NQ;
    const int ty = type(ilist[ii + co]);
    F4 acc = F4{0.f, 0.f, 0.f, 0.f};
    const int c1 = (blk + 1) * G1LF_CC < nc ? (blk + 1) * G1LF_CC : nc;
    for (int c = blk * G1LF_CC; c < c1; c++) {
      const int xs = tx(c), fi = tfi(c), t = ttile(c);
      const int nin = (xs == 0) ? n0 : (xs == 1) ? n1 : (xs == 2) ? n2 : n3;
      const auto &X = (xs == 0) ? X0 : (xs == 1) ? X1 : (xs == 2) ? X2 : X3;
      const auto &W = (xs == 0) ? W0 : (xs == 1) ? W1 : (xs == 2) ? W2 : W3;
      for (int n = 0; n < nin; n++) {
        const F xv = X(ii, n, fi);
        const F4 w = reinterpret_cast<const F4 *>(&W(ty, t, n, 0))[kq];
        fma4(acc, w, xv);
      }
    }
    const F v[4] = {acc.x, acc.y, acc.z, acc.w};
    for (int u = 0; u < 4; u++)
      if (4 * kq + u < no) Kokkos::atomic_add(&rho(ii, 4 * kq + u), v[u]);
  });
}

template<class DeviceType, class Traits>
template<int NQ, int NIN>
void Grace1LFusedKK<DeviceType, Traits>::red_bwd()
{
  const int cs = this->chunk_size, co = this->chunk_offset, gtot = rm.gtot;
  auto W0 = rm.W[0]; auto W1 = rm.W[1]; auto W2 = rm.W[2]; auto W3 = rm.W[3];
  const int n0 = rm.nin[0], n1 = rm.nin[1], n2 = rm.nin[2], n3 = rm.nin[3];
  const int gb1 = rm.gbase[1], gb2 = rm.gbase[2], gb3 = rm.gbase[3];
  auto dX0 = this->d_A_adj; auto dX1 = this->d_AA_adj; auto dX2 = this->d_AAA_adj; auto dX3 = this->d_AAAA_adj;
  auto roff = rm.roff; auto rtile = rm.rtile;
  auto gr = this->d_rho_adj;
  auto type = this->type; auto ilist = this->d_ilist;
  const int no = this->rho_n_out;
  // the weight / output views of a thread selected as raw pointers and strides, as the 2L reverse reducers (a view
  // selected by reference lives in scratch memory: MI250X 1L W-bcc-31k RevReduce 5.89 -> 4.30 ms per call)
  const F *wp0 = W0.data(), *wp1 = W1.data(), *wp2 = W2.data(), *wp3 = W3.data();
  const long a0 = W0.stride(0), a1 = W1.stride(0), a2 = W2.stride(0), a3 = W3.stride(0);
  const long b0 = W0.stride(1), b1 = W1.stride(1), b2 = W2.stride(1), b3 = W3.stride(1);
  const long e0 = W0.stride(2), e1 = W1.stride(2), e2 = W2.stride(2), e3 = W3.stride(2);
  F *xp0 = dX0.data(), *xp1 = dX1.data(), *xp2 = dX2.data(), *xp3 = dX3.data();
  const long xa0 = dX0.stride(0), xa1 = dX1.stride(0), xa2 = dX2.stride(0), xa3 = dX3.stride(0);
  const long xb0 = dX0.stride(1), xb1 = dX1.stride(1), xb2 = dX2.stride(1), xb3 = dX3.stride(1);
  const long xc0 = dX0.stride(2), xc1 = dX1.stride(2), xc2 = dX2.stride(2), xc3 = dX3.stride(2);
  Kokkos::parallel_for("g1lf:RevReduce", Kokkos::RangePolicy<DeviceType, hip_lb_t<DeviceType, 256, 2>>(0, cs * gtot), KOKKOS_LAMBDA(const int q) {
    const int g = q / cs, ii = q - g * cs;
    const int xs = (g >= gb3) ? 3 : (g >= gb2) ? 2 : (g >= gb1) ? 1 : 0;
    const int fi = g - ((xs == 0) ? 0 : (xs == 1) ? gb1 : (xs == 2) ? gb2 : gb3);
    const int nin = (xs == 0) ? n0 : (xs == 1) ? n1 : (xs == 2) ? n2 : n3;
    const int ty = type(ilist[ii + co]);
    const F *Wt = (xs == 0) ? wp0 + ty * a0 : (xs == 1) ? wp1 + ty * a1 : (xs == 2) ? wp2 + ty * a2 : wp3 + ty * a3;
    const long wt = (xs == 0) ? b0 : (xs == 1) ? b1 : (xs == 2) ? b2 : b3;
    const long wn = (xs == 0) ? e0 : (xs == 1) ? e1 : (xs == 2) ? e2 : e3;
    F4 gk[NQ];
    for (int k = 0; k < NQ; k++) {
      F v[4];
      for (int u = 0; u < 4; u++) v[u] = (4 * k + u < no) ? gr(ii, 4 * k + u) : 0.0f;
      gk[k] = F4{v[0], v[1], v[2], v[3]};
    }
    F acc[NIN];
    for (int n = 0; n < NIN; n++) acc[n] = 0.0f;
    for (int c = roff(g); c < roff(g + 1); c++) {
      const F *Wb = Wt + rtile(c) * wt;
#pragma unroll
      for (int n = 0; n < NIN; n++) {
        if (n >= nin) continue;
        const F4 *wr = reinterpret_cast<const F4 *>(Wb + n * wn);
        F s = 0.0f;
        for (int k = 0; k < NQ; k++) s += dot4(wr[k], gk[k]);
        acc[n] += s;
      }
    }
    F *Xb = (xs == 0) ? xp0 + ii * xa0 + fi * xc0 : (xs == 1) ? xp1 + ii * xa1 + fi * xc1 :
            (xs == 2) ? xp2 + ii * xa2 + fi * xc2 : xp3 + ii * xa3 + fi * xc3;
    const long xn = (xs == 0) ? xb0 : (xs == 1) ? xb1 : (xs == 2) ? xb2 : xb3;
#pragma unroll
    for (int n = 0; n < NIN; n++) if (n < nin) Xb[n * xn] = acc[n];
  });
}

/* ----------------------------------------------------------------------
   stage overrides
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::stage_bond_fwd(int, int) { fused_A(); }

// GPU: the run's compact bond list (every bond kernel of the run reads it) and the bond kernels for this model; one
// line on stderr per process: the kernels of the first run and why. The wave-per-atom backward is the default on HIP
// only: on CUDA the team-of-256 kernel is faster for 1L (A100, g1lf:bond_bwd 77 vs 88 ms per call, C-diamond-125k).
template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::begin_run()
{
  if constexpr (is_gpu_v<DeviceType>) {
    GraceFusedKK::compact_bonds<DeviceType, Traits>(this->inum, this->maxneigh, this->x, this->type, this->d_map,
        this->d_neighbors, this->d_numneigh, this->d_ilist, this->d_bond_cutoff, this->couple_eq, this->couple_bit1,
        this->couple_bit2, this->couple_lambda, this->d_mask, d_cnb, d_cnum);
    bonds_ready = true;
    const int lmax = this->lmax, NR = this->nradmax;
    const KernelChoice a = GraceFusedKK::fused_mfma(GraceFusedKK::a_fwd_mfma_fits<DeviceType>(lmax, NR));
    const KernelChoice w = GraceFusedKK::fused_wave_bwd(GraceFusedKK::bond_bwd_wave_fits<DeviceType>(lmax, NR),
                                                        is_hip_v<DeviceType>);
    a_mfma = a.on;
    bwd_wave = w.on;
    static std::once_flag logged;
    std::call_once(logged, [&] {
      std::fprintf(stderr, "grace kk: GRACE-1L bond kernels: basis A on the %s, bond backward on the %s kernel (%s); "
                   "dense FCs on the %s kernels\n", GraceFusedKK::fused_mfma_says(a).c_str(),
                   w.on ? "wave-per-atom" : "team-of-256", w.why.c_str(), fcm_A1.mfma_f ? "matrix-core" : "vector-ALU");
    });
  }
}

#define G1LF_SHAPE_FWD(P, Lv, Rv, Ov)                                                                  \
  {                                                                                                 \
    auto L_ = Lv; auto R_ = Rv; auto O_ = Ov;                                                       \
    const int cs = this->chunk_size, nc = (int) O_.extent(1);                                       \
    Kokkos::parallel_for("g1lf:ShapeProd_" #P, Kokkos::RangePolicy<DeviceType, hip_lb_t<DeviceType, 256, 2>>(0, cs * nc),         \
      KOKKOS_LAMBDA(const int q) {                                                                  \
        const int k = q / cs, ii = q - k * cs;                                                      \
        G1LShape_L::fwd_##P(L_, R_, O_, ii, k);                                                     \
      });                                                                                           \
  }
#define G1LF_SHAPE_REV(P, Lv, Rv, Dv, dLv, dRv, NG)                                                     \
  {                                                                                                 \
    auto L_ = Lv; auto R_ = Rv; auto D_ = Dv; auto dL_ = dLv; auto dR_ = dRv;                       \
    const int cs = this->chunk_size, nc = (int) D_.extent(1), nx = (int) dL_.extent(2), ng = NG;    \
    Kokkos::parallel_for("g1lf:ShapeRevProd_" #P, Kokkos::RangePolicy<DeviceType, hip_lb_t<DeviceType, 256, 2>>(0, cs * nc),      \
      KOKKOS_LAMBDA(const int q) {                                                                  \
        const int k = q / cs, ii = q - k * cs;                                                      \
        G1LShape_L::rev_##P(L_, R_, D_, dL_, dR_, ii, k);                                           \
        for (int x = ng; x < nx; x++) dL_(ii, k, x) = 0.0f;                                         \
      });                                                                                           \
  }

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::stage_dense_fwd()
{
  if (this->fc_n_out == G1LF_NO) fc_fwd4<G1LF_NO>(fcm_A1, this->d_A, this->d_A, this->d_A1);
  else fc_fwd4<G1LF_NO_M>(fcm_A1, this->d_A, this->d_A, this->d_A1);
  if (shape_id) G1LF_SHAPE_FWD(AA, this->d_A1, this->d_A1, this->d_AA)
  else prod_fwd(pm_AA, this->d_A1, this->d_A1, this->d_AA);
  if (this->fc_n_out == G1LF_NO) fc_fwd4<G1LF_NO>(fcm_AA1, this->d_AA, this->d_A, this->d_AA1);
  else fc_fwd4<G1LF_NO_M>(fcm_AA1, this->d_AA, this->d_A, this->d_AA1);
  if (shape_id) G1LF_SHAPE_FWD(AAA, this->d_AA1, this->d_A1, this->d_AAA)
  else prod_fwd(pm_AAA, this->d_AA1, this->d_A1, this->d_AAA);
  if (this->fc_n_out == G1LF_NO) fc_fwd4<G1LF_NO>(fcm_AA2, this->d_AA, this->d_A, this->d_AA2);
  else fc_fwd4<G1LF_NO_M>(fcm_AA2, this->d_AA, this->d_A, this->d_AA2);
  if (shape_id) G1LF_SHAPE_FWD(AAAA, this->d_AA2, this->d_AA2, this->d_AAAA)
  else prod_fwd(pm_AAAA, this->d_AA2, this->d_AA2, this->d_AAAA);
}

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::stage_reduce() { red_fwd<5>(); }

// the reverse reduce assigns A_adj .. AAAA_adj; A1_adj receives AAA (right) before AA (left, accumulated)
template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::stage_zero_adj() { Kokkos::deep_copy(this->d_A1_adj, 0.0f); }

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::stage_dense_bwd()
{
  red_bwd<5, G1LF_NIN>();
  if (shape_id) G1LF_SHAPE_REV(AAAA, this->d_AA2, this->d_AA2, this->d_AAAA_adj, this->d_AA2_adj, this->d_AA2_adj,
                               (int) this->d_AA2_adj.extent(2))
  else prod_rev(pm_AAAA, this->d_AA2, this->d_AA2, this->d_AAAA_adj, this->d_AA2_adj, this->d_AA2_adj, false);
  if (this->fc_n_out == G1LF_NO) fc_bwd<G1LF_NO, G1LF_NIR>(fcm_AA2, this->d_AA2_adj, this->d_AA_adj, this->d_A_adj);
  else fc_bwd<G1LF_NO_M, G1LF_NIR>(fcm_AA2, this->d_AA2_adj, this->d_AA_adj, this->d_A_adj);
  if (shape_id) G1LF_SHAPE_REV(AAA, this->d_AA1, this->d_A1, this->d_AAA_adj, this->d_AA1_adj, this->d_A1_adj, ng_AAA)
  else prod_rev(pm_AAA, this->d_AA1, this->d_A1, this->d_AAA_adj, this->d_AA1_adj, this->d_A1_adj, false);
  if (this->fc_n_out == G1LF_NO) fc_bwd<G1LF_NO, G1LF_NIR>(fcm_AA1, this->d_AA1_adj, this->d_AA_adj, this->d_A_adj);
  else fc_bwd<G1LF_NO_M, G1LF_NIR>(fcm_AA1, this->d_AA1_adj, this->d_AA_adj, this->d_A_adj);
  // AA reverse accumulates into A1_adj (generated with --acc AA); nothing to zero beyond the generated range
  if (shape_id) G1LF_SHAPE_REV(AA, this->d_A1, this->d_A1, this->d_AA_adj, this->d_A1_adj, this->d_A1_adj,
                               (int) this->d_A1_adj.extent(2))
  else prod_rev(pm_AA, this->d_A1, this->d_A1, this->d_AA_adj, this->d_A1_adj, this->d_A1_adj, true);
  if (this->fc_n_out == G1LF_NO) fc_bwd<G1LF_NO, G1LF_NIR>(fcm_A1, this->d_A1_adj, this->d_A_adj, this->d_A_adj);
  else fc_bwd<G1LF_NO_M, G1LF_NIR>(fcm_A1, this->d_A1_adj, this->d_A_adj, this->d_A_adj);
}

template<class DeviceType, class Traits>
void Grace1LFusedKK<DeviceType, Traits>::stage_bond_bwd(int, int) { fused_bond_bwd(); }

}    // namespace kk
}    // namespace grace

// the explicit instantiation of GRACE_KK_EXTERN_1L (grace/kk/grace_1l_fused_kk.h), in one translation unit
#define GRACE_KK_INSTANTIATE_1L(Device, Traits)                        \
  template class grace::kk::Grace1LKK<Device, float, float, Traits>; \
  template class grace::kk::Grace1LFusedKK<Device, Traits>

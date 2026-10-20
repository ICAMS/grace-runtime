// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Implementation of grace::kk::Grace2LFusedKK (see grace_2l_fused_kk.h); included by one translation unit per consumer.
#pragma once

#include "grace/kk/grace_2l_fused_kk.h"
#include "grace/kk/grace_2l_kk_impl.h"
#include "grace/kk/fused_common.h"
#include "grace/kk/fused_dense.h"
#include "grace/kk/g2l_shape_smaxl.h"
#include "grace/kk/g2l_shape_smaxm.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace grace {
namespace kk {
using namespace GraceFusedKK;

namespace g2lf_detail {

constexpr int G2LF_MAX_LMAX = GRACE_FUSED_MAX_LMAX;
constexpr int G2LF_TEAM = 256;      // threads per atom in the bond kernels
constexpr int G2LF_NB = 32;         // neighbours staged per batch
constexpr int G2LF_MAX_ACC = 16;    // per-thread accumulators: L1_nradmax * nlm <= TEAM * MAX_ACC
constexpr int G2LF_NB2 = 16;        // neighbours staged per batch, layer-2 forward
constexpr int G2LF_MAX_YS = 4;      // y-slots per thread in the dense L2 forward (nlm <= (TEAM / CB) * MAX_YS)      
constexpr int G2LF_MAX_B = 8;       // per-thread B0 accumulators
constexpr int G2LF_MAX_I = 16;      // I_n_funcs (16 = l <= 3 message of the large models)
constexpr int G2LF_NO1 = 42;        // FC widths compiled for (layer 1 / layer 2)
constexpr int G2LF_NO2 = 64;

}    // namespace g2lf_detail
using namespace g2lf_detail;

/* ---------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::setup()
{
  if (spline_n < 16) GRACE_KK_ERROR("grace/2l/fused/kk: spline N must be >= 16 (got {})", spline_n);
  this->couple_supported = true;    // two-group coupling in the bond kernels
  Base::setup();     // sets the model dimensions and device weights
#ifdef KOKKOS_ENABLE_CUDA
  this->fc_fused = true;    // fc_forward_cublas below runs the fused FC kernels (no cuBLAS)
#endif
  const GRACE2LModel &m = *this->grace_model;
  if (this->lmax > G2LF_MAX_LMAX)
    GRACE_KK_ERROR("grace/2l/fused/kk: lmax {} > {} not supported", this->lmax, G2LF_MAX_LMAX);
  if (m.L1_mlp_rad_has_chem_emb)
    GRACE_KK_ERROR("grace/2l/fused/kk: element-dependent radial MLP not supported");
  if (this->L1_nradmax * (this->lmax + 1) * (this->lmax + 1) > G2LF_TEAM * G2LF_MAX_ACC)
    GRACE_KK_ERROR("grace/2l/fused/kk: A basis too large for the bond kernel");
  const int nlm = (this->lmax + 1) * (this->lmax + 1);
  if (G2LF_TEAM % this->L2_nradmax != 0 || this->I_n_out != this->L2_nradmax ||
      this->L1_nradmax > G2LF_TEAM || this->I_n_funcs > G2LF_MAX_I ||
      nlm > (G2LF_TEAM / this->L2_nradmax) * G2LF_MAX_B)
    GRACE_KK_ERROR("grace/2l/fused/kk: model shape outside the compiled kernel limits");
  const GRACE2LModel &mm = *this->grace_model;
  for (const GRACE2LModel::FCWeights *w : {&mm.fc_A1, &mm.fc_AA1, &mm.fc_AA2})
    if (w->n_out != G2LF_NO1 || (int) (w->w_left.size() / ((size_t) w->n_out * w->w_shape_left)) != G2LF_NO1)
      GRACE_KK_ERROR("grace/2l/fused/kk: layer-1 FC width must be {}", G2LF_NO1);
  for (const GRACE2LModel::FCWeights *w : {&mm.fc_B1, &mm.fc_BB1, &mm.fc_BB2})
    if (w->n_out != G2LF_NO2 || (int) (w->w_left.size() / ((size_t) w->n_out * w->w_shape_left)) != G2LF_NO2)
      GRACE_KK_ERROR("grace/2l/fused/kk: layer-2 FC width must be {}", G2LF_NO2);
  // FC outputs that no CG product reads are not computed (forward) and have zero adjoint (reverse)
  auto live = [](const GRACE2LModel::CGProduct &p, bool left, bool right) {
    std::vector<int> v;
    if (left) v.insert(v.end(), p.left_ind.begin(), p.left_ind.end());
    if (right) v.insert(v.end(), p.right_ind.begin(), p.right_ind.end());
    return v;
  };
  build_fc_meta(fcm_A1, mm.fc_A1, live(mm.prod_AA, true, true));
  build_fc_meta(fcm_AA1, mm.fc_AA1, live(mm.prod_AAA, true, false));
  build_fc_meta(fcm_AA2, mm.fc_AA2, live(mm.prod_AAAA, true, true));
  build_fc_meta(fcm_B1, mm.fc_B1, live(mm.prod_BB, true, true));
  build_fc_meta(fcm_BB1, mm.fc_BB1, live(mm.prod_BBB, true, false));
  build_fc_meta(fcm_BB2, mm.fc_BB2, live(mm.prod_BBBB, true, true));
  this->log.info(fmt("grace/2l/fused/kk: live FC outputs A1 {}/{} AA1 {}/{} AA2 {}/{} B1 {}/{} BB1 {}/{} BB2 {}/{}\n",
                   fcm_A1.nlive, fcm_A1.nf, fcm_AA1.nlive, fcm_AA1.nf, fcm_AA2.nlive, fcm_AA2.nf,
                   fcm_B1.nlive, fcm_B1.nf, fcm_BB1.nlive, fcm_BB1.nf, fcm_BB2.nlive, fcm_BB2.nf));
  if (this->I1_n_out != 12 || this->B_n_out != G2LF_NO2 || this->L2_nradmax > 32 || this->L1_nradmax != G2LF_NO1)
    GRACE_KK_ERROR("grace/2l/fused/kk: ReduceN widths outside the compiled kernels (I1 12, B 64)");
  if (this->rho_n_out != 17) GRACE_KK_ERROR("grace/2l/fused/kk: rho width must be 17");
  if (this->I2_n_out != 17) GRACE_KK_ERROR("grace/2l/fused/kk: I2 width must be 17");
  build_type_metas();
  {
    std::vector<RedSpec> sp = {{&mm.B_reduce_YI, 0, -1, this->B_n_out}, {&mm.B_reduce_B0, 1, -1, this->B_n_out}};
    const int nlm = (this->lmax + 1) * (this->lmax + 1);
    build_red_meta(rm_B, sp, this->B_n_funcs, this->B_n_funcs, false, {(int) this->n_yi_out_funcs, nlm});
  }
  build_prod_meta(pm_AA, mm.prod_AA, true); build_prod_meta(pm_AAA, mm.prod_AAA, false);
  build_prod_meta(pm_AAAA, mm.prod_AAAA, true); build_prod_meta(pm_BB, mm.prod_BB, true);
  build_prod_meta(pm_BBB, mm.prod_BBB, false); build_prod_meta(pm_BBBB, mm.prod_BBBB, true);
  {   // shape-specialised CG products (generated per architecture): table match up to fp32 rounding
    auto same = [](const GRACE2LModel::CGProduct &p, int nt, const int (*tab)[3], const double *cg) {
      if ((int) p.cg_coeff.size() != nt) return false;
      for (int t = 0; t < nt; t++)
        if (p.left_ind[t] != tab[t][0] || p.right_ind[t] != tab[t][1] || p.m_sum_ind[t] != tab[t][2] ||
            std::fabs(p.cg_coeff[t] - cg[t]) > 1e-6 * std::max(1.0, std::fabs(cg[t]))) return false;   // fp32 literals
      return true;
    };
#define G2LF_SHAPE_MATCH(NS)                                                                         \
    (same(mm.prod_AA, NS::nt_AA, NS::tab_AA, NS::cg_AA) && same(mm.prod_AAA, NS::nt_AAA, NS::tab_AAA, NS::cg_AAA) && \
     same(mm.prod_AAAA, NS::nt_AAAA, NS::tab_AAAA, NS::cg_AAAA) && same(mm.prod_BB, NS::nt_BB, NS::tab_BB, NS::cg_BB) && \
     same(mm.prod_BBB, NS::nt_BBB, NS::tab_BBB, NS::cg_BBB) && same(mm.prod_BBBB, NS::nt_BBBB, NS::tab_BBBB, NS::cg_BBBB))
    shape_id = 0;
    if (G2LF_SHAPE_MATCH(G2LShape_SMAXM)) shape_id = 1;
    else if (G2LF_SHAPE_MATCH(G2LShape_SMAXL)) shape_id = 2;
#undef G2LF_SHAPE_MATCH
    if (no_shape) shape_id = 0;
    auto ngen = [](const GRACE2LModel::CGProduct &p, bool self) {
      int m = 0;
      for (size_t t = 0; t < p.left_ind.size(); t++) m = std::max(m, p.left_ind[t] + 1);
      if (self) for (size_t t = 0; t < p.right_ind.size(); t++) m = std::max(m, p.right_ind[t] + 1);
      return m;
    };
    ng_AA = ngen(mm.prod_AA, true); ng_AAA = ngen(mm.prod_AAA, false); ng_AAAA = ngen(mm.prod_AAAA, true);
    ng_BB = ngen(mm.prod_BB, true); ng_BBB = ngen(mm.prod_BBB, false); ng_BBBB = ngen(mm.prod_BBBB, true);
    this->log.info(fmt("grace/2l/fused/kk: CG products {}\n",
                     shape_id == 1 ? "specialised: medium shape (GRACE-2L-(SMAX-)OMAT-medium)" :
                     shape_id == 2 ? "specialised: large shape (GRACE-2L-(SMAX-)OMAT-large)" :
                     "generic (model tables differ from the compiled shapes)"));
  }
  {   // YI pairs grouped by (I channel, l) so the backward loops index registers statically
    const int P = this->n_yi_pairs, NG = G2LF_MAX_I * (G2LF_MAX_LMAX + 1);
    auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->d_yi_pair_iidx);
    auto hl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->d_yi_pair_l);
    d_yi_gperm = V1<int>("g2lf:yi_gperm", P); d_yi_goff = V1<int>("g2lf:yi_goff", NG + 1);
    auto hp = Kokkos::create_mirror_view(d_yi_gperm); auto ho = Kokkos::create_mirror_view(d_yi_goff);
    int q = 0;
    for (int g = 0; g < NG; g++) {
      ho(g) = q;
      const int I = g / (G2LF_MAX_LMAX + 1), l = g % (G2LF_MAX_LMAX + 1);
      for (int p = 0; p < P; p++) if (hy(p) == I && hl(p) == l) hp(q++) = p;
    }
    ho(NG) = q;
    if (q != P) GRACE_KK_ERROR("grace/2l/fused/kk: YI pair grouping incomplete");
    Kokkos::deep_copy(d_yi_gperm, hp); Kokkos::deep_copy(d_yi_goff, ho);
  }
  build_spline_tables();
}

/* ----------------------------------------------------------------------
   R(x), R1(x) tables from the npz MLP weights (host fp64 -> device fp32)
------------------------------------------------------------------------- */

// the reducers whose weights are element-dependent, packed per atom type (setup, and set_types on another species set)
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::build_type_metas()
{
  const GRACE2LModel &mm = *this->grace_model;
  {
    const int nfo1 = this->I1_n_funcs;
    std::vector<RedSpec> sp = {{&mm.I1_reduce_A, 0, -1, 12}, {&mm.I1_reduce_AA, 1, -1, 12},
                               {&mm.I1_reduce_AAA, 2, -1, 12}, {&mm.I1_reduce_AAAA, 3, -1, 12},
                               {&mm.rho_reduce_A, 0, nfo1, 17}, {&mm.rho_reduce_AA, 1, nfo1, 17},
                               {&mm.rho_reduce_AAA, 2, nfo1, 17}, {&mm.rho_reduce_AAAA, 3, nfo1, 17}};
    build_red_meta(rm_I1, sp, nfo1 + 1, nfo1, true,
                   {(int) this->n_funcs_A1, (int) this->n_funcs_AA, (int) this->n_funcs_AAA, (int) this->n_funcs_AAAA});
  }
  {
    std::vector<RedSpec> sp = {{&mm.I2_reduce_B, 0, 0, 17}, {&mm.I2_reduce_BB, 1, 0, 17},
                               {&mm.I2_reduce_BBB, 2, 0, 17}, {&mm.I2_reduce_BBBB, 3, 0, 17}};
    build_red_meta(rm_I2, sp, 1, 0, true,
                   {(int) this->B_n_funcs, (int) this->n_funcs_BB, (int) this->n_funcs_BBB, (int) this->n_funcs_BBBB});
  }
}

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::build_spline_tables()
{
  const GRACE2LModel &m = *this->grace_model;
  const int N = spline_n, p = this->radial_basis_p, nrb = this->nradbase;
  const double h = 1.0 / N;
  auto build = [&](const std::vector<GRACE2LModel::MLPLayer> &layers, V3<F> &dc, const char *name) {
    const int nout = layers.back().n_out;
    std::vector<double> fv((size_t) (N + 1) * nout), dv((size_t) (N + 1) * nout);
    std::vector<double> g, dg, out, dout;
    for (int k = 0; k <= N; k++) {
      radial_basis_x(k * h, p, nrb, g, dg);
      mlp_x(layers, g, dg, out, dout);
      for (int o = 0; o < nout; o++) { fv[(size_t) k * nout + o] = out[o]; dv[(size_t) k * nout + o] = h * dout[o]; }
    }
    dc = V3<F>(std::string("g2lf:") + name, N, nout, 4);
    auto hc = Kokkos::create_mirror_view(dc);
    for (int k = 0; k < N; k++)
      for (int o = 0; o < nout; o++) {
        const double f0 = fv[(size_t) k * nout + o], f1 = fv[(size_t) (k + 1) * nout + o];
        const double d0 = dv[(size_t) k * nout + o], d1 = dv[(size_t) (k + 1) * nout + o];
        hc(k, o, 0) = (F) f0;
        hc(k, o, 1) = (F) d0;
        hc(k, o, 2) = (F) (-3.0 * f0 - 2.0 * d0 + 3.0 * f1 - d1);
        hc(k, o, 3) = (F) (2.0 * f0 + d0 - 2.0 * f1 + d1);
      }
    Kokkos::deep_copy(dc, hc);
  };
  build(m.L1_mlp_rad_layers, d_splR, "R");
  build(m.L2_mlp_rad_layers, d_splR1, "R1");
  if ((int) d_splR.extent(1) != this->L1_nradmax * (this->lmax + 1) ||
      (int) d_splR1.extent(1) != this->L2_nradmax * (this->lmax + 1))
    GRACE_KK_ERROR("grace/2l/fused/kk: radial MLP output width mismatch");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType, class Traits>
typename Grace2LKK<DeviceType, float, float, Traits>::RunOutput
Grace2LFusedKK<DeviceType, Traits>::run(const typename Base::RunInput &in)
{
  const int inum = in.inum;
  if ((int) d_vir.extent(0) < inum) d_vir = V2<double>("g2lf:vir", inum, 6);
  Kokkos::deep_copy(d_vir, 0.0);
  auto out = Base::run(in);     // bond stages dispatch to the fused overrides below
  if (this->vflag_global) {
    double v[6];
    for (int k = 0; k < 6; k++) {
      auto vir = d_vir;
      double s = 0.0;
      Kokkos::parallel_reduce("g2lf:virial", Kokkos::RangePolicy<DeviceType>(0, inum),
          KOKKOS_LAMBDA(const int i, double &acc) { acc += vir(i, k); }, s);
      v[k] = s;
    }
    for (int k = 0; k < 6; k++) out.virial[k] += v[k];
  }
  if (this->couple_eq) Kokkos::deep_copy(out.dedl, this->d_dedl);
  return out;
}

/* ----------------------------------------------------------------------
   Layer 1 bond kernel: A[i, n, lm] = inv_avg * sum_j R_nl(x_ij) Y_lm(r_ij) z_A[mu_j, n]
   One team per atom; neighbours staged NB at a time (geometry, Y, spline R) in scratch;
   each thread owns up to MAX_ACC of the L1_nradmax * nlm outputs. No per-bond storage.
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer1_A()
{
  if (this->couple_eq) fused_layer1_A_c<true>();    // two-group coupling compiled only into its own kernels
  else fused_layer1_A_c<false>();
}

template<class DeviceType, class Traits>
template<bool CPL>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer1_A_c()
{
  const int cs = this->chunk_size;
  const int lmax = this->lmax, nl = lmax + 1, nlm = nl * nl;
  const int NR = this->L1_nradmax, nout = NR * nl, nA = NR * nlm;
  auto Af = this->d_A;                    // [chunk, NR, nlm] of the current chunk
  const int co = this->chunk_offset;
  auto x = this->x; auto type = this->type; auto map = this->d_map;
  auto neigh = this->d_neighbors; auto numneigh = this->d_numneigh; auto ilist = this->d_ilist;
  auto bc = this->d_bond_cutoff; auto ztr = this->d_z_tr_A; auto lfrom = this->d_l_from_lm;
  const int ceq = this->couple_eq, cb1 = this->couple_bit1, cb2 = this->couple_bit2;    // two-group coupling (CPL)
  const double clam = this->couple_lambda;
  auto cmask = this->d_mask;
  auto idx_s = this->d_idx_sph; auto alm = this->alm; auto blm = this->blm;
  auto cl = this->cl; auto dl = this->dl;
  auto sc = d_splR;
  const int N = spline_n;
  const F inv = this->L1_inv_avg_n_neigh;

  using team_policy = Kokkos::TeamPolicy<DeviceType>;
  using member = typename team_policy::member_type;
  const size_t shmem = (size_t) G2LF_NB * (nlm + nout + 1) * sizeof(F) + G2LF_NB * sizeof(int);
  team_policy pol(cs, G2LF_TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(shmem));

  Kokkos::parallel_for("g2lf:L1_A", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank();
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *Ys = (F *) t.team_shmem().get_shmem(G2LF_NB * nlm * sizeof(F));
    F *Rs = (F *) t.team_shmem().get_shmem(G2LF_NB * nout * sizeof(F));
    F *xs = (F *) t.team_shmem().get_shmem(G2LF_NB * sizeof(F));
    int *mus = (int *) t.team_shmem().get_shmem(G2LF_NB * sizeof(int));
    const int tid = t.team_rank(), ts = t.team_size();

    F acc[G2LF_MAX_ACC];
    for (int k = 0; k < G2LF_MAX_ACC; k++) acc[k] = 0.0f;

    const int jnum = numneigh(i);
    for (int jb = 0; jb < jnum; jb += G2LF_NB) {
      const int nb = (jnum - jb < G2LF_NB) ? jnum - jb : G2LF_NB;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, G2LF_NB), [&](const int b) {
        int mu = -1;
        if (b < nb) {
          const int j = neigh(i, jb + b) & Traits::neighmask;
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
        mus[b] = mu;
      });
      t.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, G2LF_NB * nout), [&](const int q) {
        const int b = q / nout, o = q - b * nout;
        Rs[q] = (mus[b] >= 0) ? spline_v(sc, N, xs[b], o) : 0.0f;
      });
      t.team_barrier();
#pragma unroll
      for (int k = 0; k < G2LF_MAX_ACC; k++) {
        const int a = tid + k * ts;
        if (a >= nA) continue;
        const int n = a / nlm, lm = a - n * nlm, o = n * nl + lfrom(lm);
        F s = acc[k];
        for (int b = 0; b < nb; b++) {
          const int mu = mus[b];
          if (mu >= 0) s += Rs[b * nout + o] * Ys[b * nlm + lm] * ztr(mu, n);
        }
        acc[k] = s;
      }
      t.team_barrier();
    }
#pragma unroll
    for (int k = 0; k < G2LF_MAX_ACC; k++) {
      const int a = tid + k * ts;
      if (a >= nA) continue;
      const int n = a / nlm, lm = a - n * nlm;
      Af(ii, n, lm) = acc[k] * inv;
    }
  });
}

/* ----------------------------------------------------------------------
   L1 forward state kept per chunk (A, A1, AA1, AA2, rho: all the L1 backward reads), so the
   multi-chunk L1 backward needs no forward recompute. The base per-chunk views are swapped
   for the stored ones at the start of every L1 forward / backward chunk.
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::begin_L1_chunk(bool backward)
{
  if (!l1_store_on()) return;
  const int cl = this->chunk_cap();
  if (this->inum <= cl) return;                       // one chunk: nothing is recomputed anyway
  const int c = this->chunk_offset / cl, nch = (this->inum + cl - 1) / cl;
  if (!backward && c == 0) {
    auto same = [](const auto &a, const auto &b) {    // lammps-ace 6e8e4cd0e7: every extent of every stored tensor
      for (int r = 0; r < (int) a.rank(); r++)
        if (a.extent(r) != b.extent(r)) return false;
      return true;
    };
    const bool fresh = (int) st_A.size() != nch || st_A.empty() || !same(st_A[0], this->d_A) ||
                       !same(st_A1[0], this->d_A1) || !same(st_AA1[0], this->d_AA1) ||
                       !same(st_AA2[0], this->d_AA2) || !same(st_rho[0], this->d_rho);
    if (fresh) {
      st_A.clear(); st_A1.clear(); st_AA1.clear(); st_AA2.clear(); st_rho.clear();
      for (int k = 0; k < nch; k++) {
        auto mk3 = [&](const t_nn3 &v, const char *nm) {
          return t_nn3(std::string("g2lf:st_") + nm, v.extent(0), v.extent(1), v.extent(2));
        };
        st_A.push_back(mk3(this->d_A, "A")); st_A1.push_back(mk3(this->d_A1, "A1"));
        st_AA1.push_back(mk3(this->d_AA1, "AA1")); st_AA2.push_back(mk3(this->d_AA2, "AA2"));
        st_rho.push_back(typename Base::t_nn_2d("g2lf:st_rho", this->d_rho.extent(0), this->d_rho.extent(1)));
      }
    }
  }
  this->d_A = st_A[c]; this->d_A1 = st_A1[c]; this->d_AA1 = st_AA1[c]; this->d_AA2 = st_AA2[c];
  this->d_rho = st_rho[c];
}

template<class DeviceType, class Traits>
bool Grace2LFusedKK<DeviceType, Traits>::need_L1_recompute() { return !l1_store_on(); }

/* ----------------------------------------------------------------------
   Hook overrides: the base pipeline calls these instead of its per-bond stages
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_L1_bond() { fused_layer1_A(); }

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_L1_bond_recompute() { fused_layer1_A(); }

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_L2_bond_fwd()
{
  // the message I (after the forward comm) in a neighbour-contiguous copy [nall][I * NR2 + n]
  if (this->chunk_offset == 0) {
    const int nall = this->nall, NI = this->I_n_funcs, NR2 = this->I_n_out;
    if ((int) d_IT.extent(0) < nall || (int) d_IT.extent(1) != NI * NR2) { d_IT = V2<F>("g2lf:IT", nall, NI * NR2); d_gIT = V2<F>("g2lf:gIT", nall, NI * NR2); }
    auto IT = d_IT; auto Ig = this->d_I_global;
    Kokkos::parallel_for("g2lf:I_transpose", Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0, 0}, {nall, NI * NR2}),
      KOKKOS_LAMBDA(const int j, const int q) { IT(j, q) = Ig(j, q / NR2, q % NR2); });
  }
  fused_layer2_fwd();
}

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_L2_bond_bwd()
{
  const int nall = this->nall, NI = this->I_n_funcs, NR2 = this->I_n_out;
  if (this->chunk_offset == 0) Kokkos::deep_copy(Kokkos::subview(d_gIT, std::make_pair(0, nall), Kokkos::ALL), 0.0f);
  fused_layer2_bwd();
  if (this->chunk_offset + this->chunk_size >= this->inum) {   // last chunk: hand dE/dI to the base for the reverse comm
    auto gIT = d_gIT; auto gIg = this->d_grad_I_global;
    Kokkos::parallel_for("g2lf:gI_add", Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0, 0}, {nall, NI * NR2}),
      KOKKOS_LAMBDA(const int j, const int q) { gIg(j, q / NR2, q % NR2) += gIT(j, q); });
  }
}

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_L1_bond_bwd() { fused_layer1_bwd(); }

template<class DeviceType, class Traits>
int Grace2LFusedKK<DeviceType, Traits>::l2_channel_block(bool bwd) const
{
  // channels per team: widest block whose dense per-atom [y][I][channel] pair matrix fits level-0 scratch
  const int NR2 = this->L2_nradmax, NI = this->I_n_funcs, nlm = (this->lmax + 1) * (this->lmax + 1);
  for (int CB = NR2; CB >= 8; CB /= 2) {
    if (NR2 % CB || G2LF_TEAM % CB) continue;
    const int NBR = G2LF_TEAM / CB;
    if (bwd) {
      const size_t b = (size_t) (nlm * NI * CB + nlm * CB + NBR * (4 * nlm + 8) + 3 * G2LF_TEAM) * sizeof(F)
                     + 2 * NBR * sizeof(int);
      if (b <= 46000) return CB;
    } else {
      const int ys = (nlm + NBR - 1) / NBR;
      const int stage = G2LF_NB2 * (nlm + CB * (this->lmax + 1) + NI * CB + 1);
      if (ys <= G2LF_MAX_YS && (size_t) std::max(stage, nlm * NI * CB) * sizeof(F) <= 46000) return CB;
    }
  }
  return 8;
}

/* ----------------------------------------------------------------------
   Layer 2 bond forward. Per neighbour j and channel n the pair contributions are the outer
   product w_y u_I, w_y = Y_y(r_ij) R1_{n,l(y)}(x_ij), u_I = I_j[I, n]; accumulated densely:
     Q[y, I, n] = sum_j w_y u_I            (only the (y, I) of the YI pairs are read later)
     B0[n, y]   = L2_inv * sum_j w_y z_B0[mu_j, n]
     YI[n, f]   = yi_inv * sum_{t in f} cg_t Q[y_t, I_t, n]
   Team per (atom, channel block of CB); thread (channel, y-group) keeps Q[y-slots][I] in registers.
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer2_fwd()
{
  if (this->couple_eq) {    // two-group coupling compiled only into its own kernels
    if (this->I_n_funcs <= 4) fused_layer2_fwd_t<4, true>();
    else fused_layer2_fwd_t<G2LF_MAX_I, true>();
  } else {
    if (this->I_n_funcs <= 4) fused_layer2_fwd_t<4, false>();
    else fused_layer2_fwd_t<G2LF_MAX_I, false>();
  }
}

template<class DeviceType, class Traits>
template<int NIM, bool CPL>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer2_fwd_t()
{
  const int cs = this->chunk_size, co = this->chunk_offset;
  const int lmax = this->lmax, nl = lmax + 1, nlm = nl * nl;
  const int NR2 = this->L2_nradmax;
  const int NI = this->I_n_funcs, NYo = this->n_yi_out_funcs;
  const int CB = l2_channel_block(false), nblk = NR2 / CB, G = G2LF_TEAM / CB, CBnl = CB * nl;
  const int ys = (nlm + G - 1) / G;

  auto B0 = this->d_B0; auto YI = this->d_YI;
  auto IT = d_IT;
  auto x = this->x; auto type = this->type; auto map = this->d_map;
  auto neigh = this->d_neighbors; auto numneigh = this->d_numneigh; auto ilist = this->d_ilist;
  auto bc = this->d_bond_cutoff; auto ztr = this->d_z_tr_B0; auto lfrom = this->d_l_from_lm;
  const int ceq = this->couple_eq, cb1 = this->couple_bit1, cb2 = this->couple_bit2;    // two-group coupling (CPL)
  const double clam = this->couple_lambda;
  auto cmask = this->d_mask;
  auto idx_s = this->d_idx_sph; auto alm = this->alm; auto blm = this->blm;
  auto cl = this->cl; auto dl = this->dl;
  auto py = this->d_yi_pair_yidx; auto pi_ = this->d_yi_pair_iidx;
  auto yoff = this->d_yi_out_offset; auto ycnt = this->d_yi_out_count;
  auto ycg = this->d_yi_cg_coeff; auto tpair = this->d_yi_term_pair;
  auto sc = d_splR1;
  const int N = spline_n;
  const F L2inv = this->L2_inv_avg_n_neigh, yinv = this->yi_inv_avg_n_neigh;

  using team_policy = Kokkos::TeamPolicy<DeviceType>;
  using member = typename team_policy::member_type;
  constexpr int NB = G2LF_NB2;
  const int stage = NB * (nlm + CBnl + NI * CB + 1);
  const int shf = (stage > nlm * NI * CB) ? stage : nlm * NI * CB;
  const size_t shmem = (size_t) shf * sizeof(F) + NB * sizeof(int);
  team_policy pol(cs * nblk, G2LF_TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(shmem));

  Kokkos::parallel_for("g2lf:L2_fwd", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank() / nblk, c0 = (t.league_rank() - ii * nblk) * CB;
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *sh = (F *) t.team_shmem().get_shmem(shf * sizeof(F));
    int *mus = (int *) t.team_shmem().get_shmem(NB * sizeof(int));
    F *Ys = sh, *Rs = sh + NB * nlm, *Is = Rs + NB * CBnl, *xs = Is + NB * NI * CB;
    const int tid = t.team_rank();
    const int nloc = tid % CB, yg = tid / CB, n = c0 + nloc;

    F acc[G2LF_MAX_YS][NIM], accB[G2LF_MAX_YS];
    int lys[G2LF_MAX_YS];
#pragma unroll
    for (int s = 0; s < G2LF_MAX_YS; s++) {
      const int y = yg + s * G;
      lys[s] = (s < ys && y < nlm) ? lfrom(y) : 0;
      accB[s] = 0.0f;
#pragma unroll
      for (int I = 0; I < NIM; I++) acc[s][I] = 0.0f;
    }

    const int jnum = numneigh(i);
    for (int jb = 0; jb < jnum; jb += NB) {
      const int nb = (jnum - jb < NB) ? jnum - jb : NB;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NB), [&](const int b) {
        int mu = -1;
        if (b < nb) {
          const int j = neigh(i, jb + b) & Traits::neighmask;
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
        mus[b] = mu;
      });
      t.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NB * CBnl), [&](const int q) {
        const int b = q / CBnl, o = q - b * CBnl;              // o = nloc * nl + l
        Rs[q] = (mus[b] >= 0) ? spline_v(sc, N, xs[b], c0 * nl + o) : 0.0f;
      });
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NB * NI * CB), [&](const int q) {
        const int b = q / (NI * CB), r_ = q - b * (NI * CB), I = r_ / CB, c = r_ - I * CB;
        F v = 0.0f;
        if (mus[b] >= 0) v = IT(neigh(i, jb + b) & Traits::neighmask, I * NR2 + c0 + c);
        Is[q] = v;
      });
      t.team_barrier();
      for (int b = 0; b < nb; b++) {
        const int mu = mus[b];
        if (mu < 0) continue;
        const F *Yb = Ys + b * nlm, *Rb = Rs + b * CBnl + nloc * nl, *Ib = Is + b * NI * CB + nloc;
        F u[NIM];
#pragma unroll
        for (int I = 0; I < NIM; I++) u[I] = (I < NI) ? Ib[I * CB] : 0.0f;
        const F z = ztr(mu, n);
#pragma unroll
        for (int s = 0; s < G2LF_MAX_YS; s++) {
          const int y = yg + s * G;
          if (s >= ys || y >= nlm) continue;
          const F w = Yb[y] * Rb[lys[s]];
          accB[s] += w * z;
#pragma unroll
          for (int I = 0; I < NIM; I++) acc[s][I] += w * u[I];
        }
      }
      t.team_barrier();
    }
    // B0 out; dense Q[y][I][c] to scratch, then the CG step to YI
#pragma unroll
    for (int s = 0; s < G2LF_MAX_YS; s++) {
      const int y = yg + s * G;
      if (s >= ys || y >= nlm) continue;
      B0(ii, n, y) = accB[s] * L2inv;
#pragma unroll
      for (int I = 0; I < NIM; I++) if (I < NI) sh[(y * NI + I) * CB + nloc] = acc[s][I];
    }
    t.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NYo * CB), [&](const int q) {
      const int f = q / CB, c = q - f * CB;
      const int off = yoff(f), cnt = ycnt(f);
      F s = 0.0f;
      for (int u = 0; u < cnt; u++) {
        const int p = tpair(off + u);
        s += ycg(off + u) * sh[(py(p) * NI + pi_(p)) * CB + c];
      }
      YI(ii, c0 + c, f) = s * yinv;
    });
  });
}

/* ----------------------------------------------------------------------
   Layer 2 bond backward. With M[y, I, n] = dE/dQ (dense, from dE/dYI through the CG step):
     dE/dI_j[I, n] = sum_y M[y, I, n] w_y,   v_y = sum_I M[y, I, n] u_I,
     dE/dR1_{n,l} += sum_{y in l} Y_y v_y,   dE/dY_y += R1_{n,l(y)} v_y   (+ the B0 terms).
   Team per (atom, channel block); rounds of TEAM / CB bonds, thread (bond, channel).
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer2_bwd()
{
  if (this->couple_eq) {    // two-group coupling compiled only into its own kernels
    if (this->I_n_funcs <= 4) fused_layer2_bwd_t<4, true>();
    else fused_layer2_bwd_t<G2LF_MAX_I, true>();
  } else {
    if (this->I_n_funcs <= 4) fused_layer2_bwd_t<4, false>();
    else fused_layer2_bwd_t<G2LF_MAX_I, false>();
  }
}

template<class DeviceType, class Traits>
template<int NIM, bool CPL>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer2_bwd_t()
{
  const int cs = this->chunk_size, co = this->chunk_offset;
  const int lmax = this->lmax, nl = lmax + 1, nlm = nl * nl;
  const int NR2 = this->L2_nradmax;
  const int P = this->n_yi_pairs, NI = this->I_n_funcs;
  const int CB = l2_channel_block(true), nblk = NR2 / CB;
  const int NBR = G2LF_TEAM / CB;             // bonds per round
  const int NM = nlm * NI;

  auto B0a = this->d_B0_adj; auto YIa = this->d_YI_adj;
  auto IT = d_IT; auto gIT = d_gIT;
  auto f = this->f; auto vir = d_vir;
  auto x = this->x; auto type = this->type; auto map = this->d_map;
  auto neigh = this->d_neighbors; auto numneigh = this->d_numneigh; auto ilist = this->d_ilist;
  auto bc = this->d_bond_cutoff; auto ztr = this->d_z_tr_B0;
  const int ceq = this->couple_eq, cb1 = this->couple_bit1, cb2 = this->couple_bit2;    // two-group coupling (CPL)
  const double clam = this->couple_lambda;
  auto cmask = this->d_mask;
  auto dedl = this->d_dedl;
  auto idx_s = this->d_idx_sph; auto alm = this->alm; auto blm = this->blm;
  auto cl = this->cl; auto dl = this->dl;
  auto py = this->d_yi_pair_yidx; auto pi_ = this->d_yi_pair_iidx;
  auto poff = this->d_yi_pair_term_offset; auto pcnt = this->d_yi_pair_term_count;
  auto pout = this->d_yi_pair_term_out; auto pcg = this->d_yi_pair_term_cg;
  auto sc = d_splR1;
  const int N = spline_n;
  const F L2inv = this->L2_inv_avg_n_neigh, yinv = this->yi_inv_avg_n_neigh;
  const bool do_v = this->vflag_global != 0;
  const bool do_va = this->vflag_atom != 0, do_cva = this->cvflag_atom != 0;
  auto va = this->d_vatom; auto cva = this->d_cvatom;

  using team_policy = Kokkos::TeamPolicy<DeviceType>;
  using member = typename team_policy::member_type;
  // scratch: M[nlm*NI][CB], Ba[nlm][CB], Y[NBR][nlm], G[NBR][3 nlm], geo[NBR][8], Fs[TEAM][3]; ints
  const int nf = NM * CB + nlm * CB + NBR * (4 * nlm + 8) + 3 * G2LF_TEAM;
  const size_t shmem = (size_t) nf * sizeof(F) + 2 * NBR * sizeof(int);
  team_policy pol(cs * nblk, G2LF_TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(shmem));

  Kokkos::parallel_for("g2lf:L2_bwd", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank() / nblk, c0 = (t.league_rank() - ii * nblk) * CB;
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *Ms = (F *) t.team_shmem().get_shmem(nf * sizeof(F));
    int *mus = (int *) t.team_shmem().get_shmem(2 * NBR * sizeof(int));
    int *js = mus + NBR;
    F *Ba = Ms + NM * CB, *Ys = Ba + nlm * CB, *Gs = Ys + NBR * nlm, *geo = Gs + NBR * 3 * nlm;
    F *Fs = geo + NBR * 8;
    const int tid = t.team_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NM * CB), [&](const int q) { Ms[q] = 0.0f; });
    t.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, P * CB), [&](const int q) {
      const int p = q / CB, c = q - p * CB;
      const int off = poff(p), cnt = pcnt(p);
      F s = 0.0f;
      for (int u = 0; u < cnt; u++) s += pcg(off + u) * YIa(ii, c0 + c, pout(off + u));
      Ms[(py(p) * NI + pi_(p)) * CB + c] = s * yinv;
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nlm * CB), [&](const int q) {
      const int lm = q / CB, c = q - lm * CB;
      Ba[q] = B0a(ii, c0 + c, lm) * L2inv;
    });
    double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dla = 0.0;    // dE/dlambda (coupling)
    t.team_barrier();

    const int jnum = numneigh(i);
    for (int jb = 0; jb < jnum; jb += NBR) {
      const int nb = (jnum - jb < NBR) ? jnum - jb : NBR;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NBR), [&](const int b) {
        int mu = -1;
        if (b < nb) {
          const int j = neigh(i, jb + b) & Traits::neighmask;
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
              real_sph_grad(lmax, gb[3], gb[4], gb[5], idx_s, alm, blm, cl, dl,
                            Ys + b * nlm, Gs + b * 3 * nlm);
            }
          }
        }
        mus[b] = mu;
      });
      t.team_barrier();
      {
        const int b = tid / CB, nloc = tid - b * CB, n = c0 + nloc;
        F fx = 0.0f, fy = 0.0f, fz = 0.0f;
        if (b < nb && mus[b] >= 0) {
          const int mu = mus[b], j = js[b];
          const F *gb = geo + b * 8, *Yb = Ys + b * nlm, *Gb = Gs + b * 3 * nlm;
          F R[G2LF_MAX_LMAX + 1], dR[G2LF_MAX_LMAX + 1], gR[G2LF_MAX_LMAX + 1];
#pragma unroll
          for (int l = 0; l <= G2LF_MAX_LMAX; l++) {
            R[l] = dR[l] = gR[l] = 0.0f;
            if (l < nl) spline_vd(sc, N, gb[0], n * nl + l, R[l], dR[l]);
          }
          F u[NIM], gI[NIM];
#pragma unroll
          for (int I = 0; I < NIM; I++) { u[I] = (I < NI) ? IT(j, I * NR2 + n) : 0.0f; gI[I] = 0.0f; }
          F gx = 0.0f, gy = 0.0f, gz = 0.0f;
          const F z = ztr(mu, n);
#pragma unroll
          for (int l = 0; l <= G2LF_MAX_LMAX; l++) {
            if (l >= nl) continue;
            for (int y = l * l; y < (l + 1) * (l + 1); y++) {
              const F *My = Ms + y * NI * CB + nloc;
              const F wy = Yb[y] * R[l];
              F vy = Ba[y * CB + nloc] * z;                 // B0 term: dE/dw_y
#pragma unroll
              for (int I = 0; I < NIM; I++) {
                if (I >= NI) continue;
                const F m = My[I * CB];
                vy += m * u[I];
                gI[I] += m * wy;
              }
              gR[l] += Yb[y] * vy;
              const F c = R[l] * vy;
              gx += c * Gb[3 * y]; gy += c * Gb[3 * y + 1]; gz += c * Gb[3 * y + 2];
            }
          }
#pragma unroll
          for (int I = 0; I < NIM; I++)
            if (I < NI && gI[I] != 0.0f) Kokkos::atomic_add(&gIT(j, I * NR2 + n), gI[I]);
          F rad = 0.0f;
#pragma unroll
          for (int l = 0; l <= G2LF_MAX_LMAX; l++) rad += gR[l] * dR[l];
          if (CPL) dla += rad * gb[7];    // dE/dlambda: dE/dx' dx'/dlambda
          rad *= gb[1];                                   // d/dr = d/dx / rc
          const F rx = gb[3], ry = gb[4], rz = gb[5];
          const F rg = rx * gx + ry * gy + rz * gz;
          fx = rad * rx + (gx - rg * rx) * gb[2];
          fy = rad * ry + (gy - rg * ry) * gb[2];
          fz = rad * rz + (gz - rg * rz) * gb[2];
        }
        Fs[3 * tid] = fx; Fs[3 * tid + 1] = fy; Fs[3 * tid + 2] = fz;
      }
      t.team_barrier();
      if (tid < nb && mus[tid] >= 0) {
        const int b = tid, j = js[b];
        F fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int c = 0; c < CB; c++) {
          fx += Fs[3 * (b * CB + c)]; fy += Fs[3 * (b * CB + c) + 1]; fz += Fs[3 * (b * CB + c) + 2];
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
    if (do_v && tid < NBR)
      for (int k = 0; k < 6; k++) Kokkos::atomic_add(&vir(ii + co, k), v[k]);
    if (CPL) {    // dE/dlambda of this atom's bonds
      double s = dla;
      t.team_reduce(Kokkos::Sum<double>(s));
      if (tid == 0) Kokkos::atomic_add(&dedl(), s);
    }
  });
}

/* ----------------------------------------------------------------------
   Layer 1 bond backward: from dE/dA of atom i, per bond the force dE/dr_ij.
   Rounds of TEAM / NR neighbours: thread (b, n), n = radial channel.
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer1_bwd()
{
  if (this->couple_eq) fused_layer1_bwd_c<true>();    // two-group coupling compiled only into its own kernels
  else fused_layer1_bwd_c<false>();
}

template<class DeviceType, class Traits>
template<bool CPL>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer1_bwd_c()
{
  const int cs = this->chunk_size, co = this->chunk_offset;
  const int lmax = this->lmax, nl = lmax + 1, nlm = nl * nl;
  const int NR = this->L1_nradmax;
  const int NBR = G2LF_TEAM / NR;

  auto Aa = this->d_A_adj;
  auto f = this->f; auto vir = d_vir;
  auto x = this->x; auto type = this->type; auto map = this->d_map;
  auto neigh = this->d_neighbors; auto numneigh = this->d_numneigh; auto ilist = this->d_ilist;
  auto bc = this->d_bond_cutoff; auto ztr = this->d_z_tr_A; auto lfrom = this->d_l_from_lm;
  const int ceq = this->couple_eq, cb1 = this->couple_bit1, cb2 = this->couple_bit2;    // two-group coupling (CPL)
  const double clam = this->couple_lambda;
  auto cmask = this->d_mask;
  auto dedl = this->d_dedl;
  auto idx_s = this->d_idx_sph; auto alm = this->alm; auto blm = this->blm;
  auto cl = this->cl; auto dl = this->dl;
  auto sc = d_splR;
  const int N = spline_n;
  const F L1inv = this->L1_inv_avg_n_neigh;
  const bool do_v = this->vflag_global != 0;
  const bool do_va = this->vflag_atom != 0, do_cva = this->cvflag_atom != 0;
  auto va = this->d_vatom; auto cva = this->d_cvatom;

  using team_policy = Kokkos::TeamPolicy<DeviceType>;
  using member = typename team_policy::member_type;
  const int nf = nlm * NR + NBR * (4 * nlm + 8) + 3 * G2LF_TEAM;
  const size_t shmem = (size_t) nf * sizeof(F) + 2 * NBR * sizeof(int);
  team_policy pol(cs, G2LF_TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(shmem));

  Kokkos::parallel_for("g2lf:L1_bwd", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank();
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *Aa_s = (F *) t.team_shmem().get_shmem(nf * sizeof(F));
    int *mus = (int *) t.team_shmem().get_shmem(2 * NBR * sizeof(int));
    int *js = mus + NBR;
    F *Ys = Aa_s + nlm * NR, *Gs = Ys + NBR * nlm, *geo = Gs + NBR * 3 * nlm, *Fs = geo + NBR * 8;
    const int tid = t.team_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nlm * NR), [&](const int q) {
      const int lm = q / NR, c = q - lm * NR;
      Aa_s[q] = Aa(ii, c, lm) * L1inv;
    });
    double v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dla = 0.0;    // dE/dlambda (coupling)
    t.team_barrier();

    const int jnum = numneigh(i);
    for (int jb = 0; jb < jnum; jb += NBR) {
      const int nb = (jnum - jb < NBR) ? jnum - jb : NBR;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NBR), [&](const int b) {
        int mu = -1;
        if (b < nb) {
          const int j = neigh(i, jb + b) & Traits::neighmask;
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
              real_sph_grad(lmax, gb[3], gb[4], gb[5], idx_s, alm, blm, cl, dl,
                            Ys + b * nlm, Gs + b * 3 * nlm);
            }
          }
        }
        mus[b] = mu;
      });
      t.team_barrier();
      {
        const int b = tid / NR, n = tid - b * NR;
        F fx = 0.0f, fy = 0.0f, fz = 0.0f;
        if (b < nb && mus[b] >= 0) {
          const int mu = mus[b];
          const F *gb = geo + b * 8, *Yb = Ys + b * nlm, *Gb = Gs + b * 3 * nlm;
          F R[G2LF_MAX_LMAX + 1], dR[G2LF_MAX_LMAX + 1], gR[G2LF_MAX_LMAX + 1];
#pragma unroll
          for (int l = 0; l <= G2LF_MAX_LMAX; l++) {
            R[l] = dR[l] = gR[l] = 0.0f;
            if (l < nl) spline_vd(sc, N, gb[0], n * nl + l, R[l], dR[l]);
          }
          F gx = 0.0f, gy = 0.0f, gz = 0.0f;
          const F z = ztr(mu, n);
#pragma unroll
          for (int l = 0; l <= G2LF_MAX_LMAX; l++) {
            if (l >= nl) continue;
            for (int lm = l * l; lm < (l + 1) * (l + 1); lm++) {
              const F w = Aa_s[lm * NR + n] * z;
              gR[l] += w * Yb[lm];
              const F wr = w * R[l];
              gx += wr * Gb[3 * lm]; gy += wr * Gb[3 * lm + 1]; gz += wr * Gb[3 * lm + 2];
            }
          }
          F rad = 0.0f;
#pragma unroll
          for (int l = 0; l <= G2LF_MAX_LMAX; l++) rad += gR[l] * dR[l];
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
      if (tid < nb && mus[tid] >= 0) {
        const int b = tid, j = js[b];
        F fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int c = 0; c < NR; c++) {
          fx += Fs[3 * (b * NR + c)]; fy += Fs[3 * (b * NR + c) + 1]; fz += Fs[3 * (b * NR + c) + 2];
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
    if (do_v && tid < NBR)
      for (int k = 0; k < 6; k++) Kokkos::atomic_add(&vir(ii + co, k), v[k]);
    if (CPL) {    // dE/dlambda of this atom's bonds
      double s = dla;
      t.team_reduce(Kokkos::Sum<double>(s));
      if (tid == 0) Kokkos::atomic_add(&dedl(), s);
    }
  });
}

/* ----------------------------------------------------------------------
   FC (FCRight2Left) directly on the [chunk, channel, func] views, no gather/scatter:
     out[a,k,f] = nof[f] * ( sum_n nl Wl[tl(f)][n][k] in_l[a,n,f]
                           + sum_{c: ct(c)=f} sum_n nr Wr[tr(c)][n][k] in_r[a,n,cf(c)] )
   One thread per (atom, f) with NO accumulators; the right half is a CSR by target,
   the reverse right half a CSR by source (no atomics).
------------------------------------------------------------------------- */

// dense-chain kernels and metadata: shared with grace/1l/fused/kk in grace_fused_kokkos_dense.h
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::build_fc_meta(FCMeta &m, const GRACE2LModel::FCWeights &fc, const std::vector<int> &used)
{
  GraceFusedKK::build_fc_meta(m, fc, used, "grace/2l/fused/kk");
}

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::build_prod_meta(ProdMeta &m, const GRACE2LModel::CGProduct &p, bool self)
{
  GraceFusedKK::build_prod_meta(m, p, self);
}

template<class DeviceType, class Traits>
template<int NO>
void Grace2LFusedKK<DeviceType, Traits>::fc_fwd(const FCMeta &m, const t_nn3 &in_l, const t_nn3 &in_r, const t_nn3 &out)
{
  GraceFusedKK::fc_fwd<DeviceType, NO>(m, this->chunk_size, in_l, in_r, out);
}

template<class DeviceType, class Traits>
template<int NO>
void Grace2LFusedKK<DeviceType, Traits>::fc_fwd4(const FCMeta &m, const t_nn3 &in_l, const t_nn3 &in_r, const t_nn3 &out)
{
  GraceFusedKK::fc_fwd4<DeviceType, NO>(m, this->chunk_size, in_l, in_r, out);
}

template<class DeviceType, class Traits>
template<int NO>
void Grace2LFusedKK<DeviceType, Traits>::fc_bwd(const FCMeta &m, const t_nn3 &dout, const t_nn3 &din_l, const t_nn3 &din_r)
{
  GraceFusedKK::fc_bwd<DeviceType, NO, NO>(m, this->chunk_size, dout, din_l, din_r);
}

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::prod_fwd(const ProdMeta &m, const t_nn3 &L, const t_nn3 &R, const t_nn3 &out)
{
  GraceFusedKK::prod_fwd(m, this->chunk_size, L, R, out);
}

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::prod_rev(const ProdMeta &m, const t_nn3 &L, const t_nn3 &R, const t_nn3 &D, const t_nn3 &dL,
                                                  const t_nn3 &dR)
{
  GraceFusedKK::prod_rev(m, this->chunk_size, L, R, D, dL, dR);
}

#ifdef KOKKOS_ENABLE_CUDA
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::fc_forward_cublas(typename Base::FCCublasOp &op,
    const t_nn3 &in_left, const t_nn3 &in_right, const t_nn3 &out)
{
  const FCMeta *m = nullptr;
  if (&op == &this->fc_op_A1) m = &fcm_A1;
  else if (&op == &this->fc_op_AA1) m = &fcm_AA1;
  else if (&op == &this->fc_op_AA2) m = &fcm_AA2;
  else if (&op == &this->fc_op_B1) m = &fcm_B1;
  else if (&op == &this->fc_op_BB1) m = &fcm_BB1;
  else if (&op == &this->fc_op_BB2) m = &fcm_BB2;
  if (!m) { Base::fc_forward_cublas(op, in_left, in_right, out); return; }
  if (m->no == this->L1_fc_n_out) fc_fwd<G2LF_NO1>(*m, in_left, in_right, out);
  else fc_fwd4<G2LF_NO2>(*m, in_left, in_right, out);
}
#endif

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_rev_fc_AA2() { fc_bwd<G2LF_NO1>(fcm_AA2, this->d_AA2_adj, this->d_AA_adj, this->d_A_adj); }
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_rev_fc_AA1() { fc_bwd<G2LF_NO1>(fcm_AA1, this->d_AA1_adj, this->d_AA_adj, this->d_A_adj); }
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_rev_fc_A1() { fc_bwd<G2LF_NO1>(fcm_A1, this->d_A1_adj, this->d_A_adj, this->d_A_adj); }
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_rev_fc_BB2() { fc_bwd<G2LF_NO2>(fcm_BB2, this->d_BB2_adj, this->d_BB_adj, this->d_B_adj); }
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_rev_fc_BB1() { fc_bwd<G2LF_NO2>(fcm_BB1, this->d_BB1_adj, this->d_BB_adj, this->d_B_adj); }
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_rev_fc_B1() { fc_bwd<G2LF_NO2>(fcm_B1, this->d_B1_adj, this->d_B_adj, this->d_B_adj); }

/* ----------------------------------------------------------------------
   CG products in gather form (one write per element, no per-term read-modify-write):
     forward  out[o] = sum_{t: o_t = o} cg_t L[l_t] R[r_t]          (CSR by output)
     reverse  dL[x]  = sum_{t: l_t = x} cg_t D[o_t] R[r_t]  (+ r-role terms for self products)
              dR[x] += sum_{t: r_t = x} cg_t D[o_t] L[l_t]          (mixed products only)
   Thread per (atom, channel); the CG product is channel-diagonal.
------------------------------------------------------------------------- */

#define G2LF_SHAPE_FWD(NS, P, Lv, Rv, Ov)                                                              \
  {                                                                                                 \
    auto L_ = Lv; auto R_ = Rv; auto O_ = Ov;                                                       \
    const int cs = this->chunk_size, nc = (int) O_.extent(1);                                       \
    Kokkos::parallel_for("g2lf:ShapeProd_" #P, Kokkos::RangePolicy<DeviceType>(0, cs * nc),         \
      KOKKOS_LAMBDA(const int q) {                                                                  \
        const int k = q / cs, ii = q - k * cs;                                                      \
        NS::fwd_##P(L_, R_, O_, ii, k);                                                 \
      });                                                                                           \
  }
#define G2LF_SHAPE_REV(NS, P, Lv, Rv, Dv, dLv, dRv, NG)                                                 \
  {                                                                                                 \
    auto L_ = Lv; auto R_ = Rv; auto D_ = Dv; auto dL_ = dLv; auto dR_ = dRv;                       \
    const int cs = this->chunk_size, nc = (int) D_.extent(1), nx = (int) dL_.extent(2), ng = NG;    \
    Kokkos::parallel_for("g2lf:ShapeRevProd_" #P, Kokkos::RangePolicy<DeviceType>(0, cs * nc),      \
      KOKKOS_LAMBDA(const int q) {                                                                  \
        const int k = q / cs, ii = q - k * cs;                                                      \
        NS::rev_##P(L_, R_, D_, dL_, dR_, ii, k);                                       \
        for (int x = ng; x < nx; x++) dL_(ii, k, x) = 0.0f;                                         \
      });                                                                                           \
  }

template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_prod_AA()
{ if (shape_id == 1) G2LF_SHAPE_FWD(G2LShape_SMAXM, AA, this->d_A1, this->d_A1, this->d_AA) else if (shape_id == 2) G2LF_SHAPE_FWD(G2LShape_SMAXL, AA, this->d_A1, this->d_A1, this->d_AA) else prod_fwd(pm_AA, this->d_A1, this->d_A1, this->d_AA); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_prod_AAA()
{ if (shape_id == 1) G2LF_SHAPE_FWD(G2LShape_SMAXM, AAA, this->d_AA1, this->d_A, this->d_AAA) else if (shape_id == 2) G2LF_SHAPE_FWD(G2LShape_SMAXL, AAA, this->d_AA1, this->d_A, this->d_AAA) else prod_fwd(pm_AAA, this->d_AA1, this->d_A, this->d_AAA); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_prod_AAAA()
{ if (shape_id == 1) G2LF_SHAPE_FWD(G2LShape_SMAXM, AAAA, this->d_AA2, this->d_AA2, this->d_AAAA) else if (shape_id == 2) G2LF_SHAPE_FWD(G2LShape_SMAXL, AAAA, this->d_AA2, this->d_AA2, this->d_AAAA) else prod_fwd(pm_AAAA, this->d_AA2, this->d_AA2, this->d_AAAA); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_prod_BB()
{ if (shape_id == 1) G2LF_SHAPE_FWD(G2LShape_SMAXM, BB, this->d_B1, this->d_B1, this->d_BB) else if (shape_id == 2) G2LF_SHAPE_FWD(G2LShape_SMAXL, BB, this->d_B1, this->d_B1, this->d_BB) else prod_fwd(pm_BB, this->d_B1, this->d_B1, this->d_BB); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_prod_BBB()
{ if (shape_id == 1) G2LF_SHAPE_FWD(G2LShape_SMAXM, BBB, this->d_BB1, this->d_B, this->d_BBB) else if (shape_id == 2) G2LF_SHAPE_FWD(G2LShape_SMAXL, BBB, this->d_BB1, this->d_B, this->d_BBB) else prod_fwd(pm_BBB, this->d_BB1, this->d_B, this->d_BBB); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_prod_BBBB()
{ if (shape_id == 1) G2LF_SHAPE_FWD(G2LShape_SMAXM, BBBB, this->d_BB2, this->d_BB2, this->d_BBBB) else if (shape_id == 2) G2LF_SHAPE_FWD(G2LShape_SMAXL, BBBB, this->d_BB2, this->d_BB2, this->d_BBBB) else prod_fwd(pm_BBBB, this->d_BB2, this->d_BB2, this->d_BBBB); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_prod_AA()
{ if (shape_id == 1) G2LF_SHAPE_REV(G2LShape_SMAXM, AA, this->d_A1, this->d_A1, this->d_AA_adj, this->d_A1_adj, this->d_A1_adj, ng_AA) else if (shape_id == 2) G2LF_SHAPE_REV(G2LShape_SMAXL, AA, this->d_A1, this->d_A1, this->d_AA_adj, this->d_A1_adj, this->d_A1_adj, ng_AA) else prod_rev(pm_AA, this->d_A1, this->d_A1, this->d_AA_adj, this->d_A1_adj, this->d_A1_adj); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_prod_AAA()
{ if (shape_id == 1) G2LF_SHAPE_REV(G2LShape_SMAXM, AAA, this->d_AA1, this->d_A, this->d_AAA_adj, this->d_AA1_adj, this->d_A_adj, ng_AAA) else if (shape_id == 2) G2LF_SHAPE_REV(G2LShape_SMAXL, AAA, this->d_AA1, this->d_A, this->d_AAA_adj, this->d_AA1_adj, this->d_A_adj, ng_AAA) else prod_rev(pm_AAA, this->d_AA1, this->d_A, this->d_AAA_adj, this->d_AA1_adj, this->d_A_adj); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_prod_AAAA()
{ if (shape_id == 1) G2LF_SHAPE_REV(G2LShape_SMAXM, AAAA, this->d_AA2, this->d_AA2, this->d_AAAA_adj, this->d_AA2_adj, this->d_AA2_adj, ng_AAAA) else if (shape_id == 2) G2LF_SHAPE_REV(G2LShape_SMAXL, AAAA, this->d_AA2, this->d_AA2, this->d_AAAA_adj, this->d_AA2_adj, this->d_AA2_adj, ng_AAAA) else prod_rev(pm_AAAA, this->d_AA2, this->d_AA2, this->d_AAAA_adj, this->d_AA2_adj, this->d_AA2_adj); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_prod_BB()
{ if (shape_id == 1) G2LF_SHAPE_REV(G2LShape_SMAXM, BB, this->d_B1, this->d_B1, this->d_BB_adj, this->d_B1_adj, this->d_B1_adj, ng_BB) else if (shape_id == 2) G2LF_SHAPE_REV(G2LShape_SMAXL, BB, this->d_B1, this->d_B1, this->d_BB_adj, this->d_B1_adj, this->d_B1_adj, ng_BB) else prod_rev(pm_BB, this->d_B1, this->d_B1, this->d_BB_adj, this->d_B1_adj, this->d_B1_adj); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_prod_BBB()
{ if (shape_id == 1) G2LF_SHAPE_REV(G2LShape_SMAXM, BBB, this->d_BB1, this->d_B, this->d_BBB_adj, this->d_BB1_adj, this->d_B_adj, ng_BBB) else if (shape_id == 2) G2LF_SHAPE_REV(G2LShape_SMAXL, BBB, this->d_BB1, this->d_B, this->d_BBB_adj, this->d_BB1_adj, this->d_B_adj, ng_BBB) else prod_rev(pm_BBB, this->d_BB1, this->d_B, this->d_BBB_adj, this->d_BB1_adj, this->d_B_adj); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_prod_BBBB()
{ if (shape_id == 1) G2LF_SHAPE_REV(G2LShape_SMAXM, BBBB, this->d_BB2, this->d_BB2, this->d_BBBB_adj, this->d_BB2_adj, this->d_BB2_adj, ng_BBBB) else if (shape_id == 2) G2LF_SHAPE_REV(G2LShape_SMAXL, BBBB, this->d_BB2, this->d_BB2, this->d_BBBB_adj, this->d_BB2_adj, this->d_BB2_adj, ng_BBBB) else prod_rev(pm_BBBB, this->d_BB2, this->d_BB2, this->d_BBBB_adj, this->d_BB2_adj, this->d_BB2_adj); }

/* ----------------------------------------------------------------------
   ReduceN without atomics, several reducers sharing input tensors in one pass:
     B  <- YI, B0                       (MODE 0: out[a, k, fo], dX += )
     I1 <- A, AA, AAA, AAAA  and  rho <- A, AA, AAA, AAAA
                                        (MODE 1: fo < nfo1 -> I1[a, fo, k], fo == nfo1 -> rho[a, k];
                                         reverse assigns dX = W_I1^T dI1 + W_rho^T drho)
   out[a, fo, k] = sum_{c -> fo} norm sum_n W[mu_a][k][n][tile(c)] X[a, n, ci(c)]
   forward: thread per (atom, fo), float4 rows over k; reverse: thread per (atom, input
   tensor, input function), CSR of every connection reading that function.
   Weights per reducer ("weight set"), packed per LAMMPS type when element-dependent.
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::build_red_meta(RedMeta &m, const std::vector<RedSpec> &specs,
                                                         int nfo, int nfo1, bool elem, const std::vector<int> &nfi_x)
{
  const int nws = specs.size(), nx = nfi_x.size();
  const int ntypes = ((int) this->h_type2model.size() - 1);
  int no_max = 0;
  for (auto &sp : specs) no_max = std::max(no_max, sp.n_out);
  const int nop = (no_max + 3) / 4 * 4;
  const int nt = elem ? ntypes + 1 : 1;
  m.nws = nws; m.nx = nx; m.nfo = nfo; m.nfo1 = nfo1; m.elem = elem; m.nop = nop;
  int gtot = 0;
  for (int x = 0; x < 4; x++) { m.nfi[x] = x < nx ? nfi_x[x] : 0; m.gbase[x] = gtot; if (x < nx) gtot += nfi_x[x]; }
  m.gtot = gtot;
  std::vector<std::vector<std::array<int, 3>>> by_fo(nfo);     // (ws, fi, tile)
  std::vector<std::vector<std::array<int, 3>>> by_g(std::max(gtot, 1));   // (fo, tile, ws)
  for (int w = 0; w < 8; w++) { m.ws_x[w] = 0; m.ws_nin[w] = 0; }
  for (int w = 0; w < nws; w++) {
    const auto &rw = *specs[w].rw;
    const int x = specs[w].x, nin = rw.n_in, nw = rw.w_shape > 0 ? rw.w_shape : 1, no = specs[w].n_out;
    m.ws_x[w] = x; m.ws_nin[w] = nin;
    for (int c = 0; c < rw.n_connections; c++) {
      const int fo = specs[w].fo >= 0 ? specs[w].fo : rw.total_sum_ind[c];
      const int fi = rw.collect_ind[c];
      if (fo >= nfo || fi >= m.nfi[x]) GRACE_KK_ERROR("grace/2l/fused/kk: reduce index out of range");
      by_fo[fo].push_back({w, fi, rw.w_l_tile[c]});
      by_g[m.gbase[x] + fi].push_back({fo, rw.w_l_tile[c], w});
    }
    V4<F> W(std::string("g2lf:redW") + std::to_string(w), nt, nw, nin, nop);
    auto hW = Kokkos::create_mirror_view(W);
    for (int ty = 0; ty < nt; ty++) {
      const int mu = elem ? (ty >= 1 ? this->h_type2model[ty] : -1) : 0;
      for (int t = 0; t < nw; t++)
        for (int n = 0; n < nin; n++)
          for (int k = 0; k < nop; k++) {
            double v = 0.0;
            if (k < no && mu >= 0) {
              const size_t off = elem ? ((((size_t) mu * no + k) * nin + n) * nw + t)
                                      : (((size_t) k * nin + n) * nw + t);
              v = rw.norm * rw.W[off];
            }
            hW(ty, t, n, k) = (F) v;
          }
    }
    Kokkos::deep_copy(W, hW);
    m.W[w] = W;
  }
  auto fill = [&](const std::vector<std::vector<std::array<int, 3>>> &lists, V1<int> &off, V1<int> &a, V1<int> &b,
                  V1<int> &cc, const char *nm) {
    const int nk = lists.size();
    int tot = 0; for (auto &v : lists) tot += v.size();
    off = V1<int>(std::string("g2lf:rd_off") + nm, nk + 1);
    a = V1<int>(std::string("g2lf:rd_a") + nm, std::max(tot, 1));
    b = V1<int>(std::string("g2lf:rd_b") + nm, std::max(tot, 1));
    cc = V1<int>(std::string("g2lf:rd_c") + nm, std::max(tot, 1));
    auto ho = Kokkos::create_mirror_view(off); auto ha = Kokkos::create_mirror_view(a);
    auto hb = Kokkos::create_mirror_view(b); auto hc = Kokkos::create_mirror_view(cc);
    int q = 0;
    for (int k = 0; k < nk; k++) {
      ho(k) = q;
      for (auto &e : lists[k]) { ha(q) = e[0]; hb(q) = e[1]; hc(q) = e[2]; q++; }
    }
    ho(nk) = q;
    Kokkos::deep_copy(off, ho); Kokkos::deep_copy(a, ha); Kokkos::deep_copy(b, hb); Kokkos::deep_copy(cc, hc);
  };
  fill(by_fo, m.toff, m.tws, m.tfi, m.ttile, "f");
  by_g.resize(gtot);
  fill(by_g, m.roff, m.rfo, m.rtile, m.rws, "r");
}

// MODE 0: out [a, k, fo] (B);  MODE 1: fo < nfo1 -> out[a, fo, k] (I1, NO1 outputs), fo == nfo1 -> out2[a, k] (rho, NO2)
template<class DeviceType, class Traits>
template<int NQ, int MODE, int NO1, int NO2>
void Grace2LFusedKK<DeviceType, Traits>::red_fwd(const RedMeta &m, const t_nn3 &X0, const t_nn3 &X1,
                                                 const t_nn3 &X2, const t_nn3 &X3, const t_nn3 &out,
                                                 const typename Base::t_nn_2d &out2)
{
  const int cs = this->chunk_size, co = this->chunk_offset, nfo = m.nfo, nfo1 = m.nfo1;
  const bool elem = m.elem;
  auto W0 = m.W[0]; auto W1 = m.W[1]; auto W2 = m.W[2]; auto W3 = m.W[3];
  auto W4 = m.W[4]; auto W5 = m.W[5]; auto W6 = m.W[6]; auto W7 = m.W[7];
  int wx[8], wn[8];
  for (int w = 0; w < 8; w++) { wx[w] = m.ws_x[w]; wn[w] = m.ws_nin[w]; }
  auto toff = m.toff; auto tws = m.tws; auto tfi = m.tfi; auto ttile = m.ttile;
  auto type = this->type; auto ilist = this->d_ilist;
  const int x0 = wx[0], x1 = wx[1], x2 = wx[2], x3 = wx[3], x4 = wx[4], x5 = wx[5], x6 = wx[6], x7 = wx[7];
  const int n0 = wn[0], n1 = wn[1], n2 = wn[2], n3 = wn[3], n4 = wn[4], n5 = wn[5], n6 = wn[6], n7 = wn[7];
  Kokkos::parallel_for("g2lf:Reduce", Kokkos::RangePolicy<DeviceType>(0, cs * nfo), KOKKOS_LAMBDA(const int q) {
    const int fo = q / cs, ii = q - fo * cs;
    const int ty = elem ? type(ilist[ii + co]) : 0;
    F4 acc[NQ];
    for (int k = 0; k < NQ; k++) acc[k] = F4{0.f, 0.f, 0.f, 0.f};
    for (int c = toff(fo); c < toff(fo + 1); c++) {
      const int w = tws(c), fi = tfi(c), t = ttile(c);
      const int xs = (w == 0) ? x0 : (w == 1) ? x1 : (w == 2) ? x2 : (w == 3) ? x3 : (w == 4) ? x4 : (w == 5) ? x5 : (w == 6) ? x6 : x7;
      const int nin = (w == 0) ? n0 : (w == 1) ? n1 : (w == 2) ? n2 : (w == 3) ? n3 : (w == 4) ? n4 : (w == 5) ? n5 : (w == 6) ? n6 : n7;
      const auto &X = (xs == 0) ? X0 : (xs == 1) ? X1 : (xs == 2) ? X2 : X3;
      const auto &W = (w == 0) ? W0 : (w == 1) ? W1 : (w == 2) ? W2 : (w == 3) ? W3 : (w == 4) ? W4 : (w == 5) ? W5 : (w == 6) ? W6 : W7;
      for (int n = 0; n < nin; n++) {
        const F xv = X(ii, n, fi);
        const F4 *wr = reinterpret_cast<const F4 *>(&W(ty, t, n, 0));
        for (int k = 0; k < NQ; k++) fma4(acc[k], wr[k], xv);
      }
    }
    for (int k = 0; k < NQ; k++) {
      const F v[4] = {acc[k].x, acc[k].y, acc[k].z, acc[k].w};
      for (int u = 0; u < 4; u++) {
        const int kk = 4 * k + u;
        if (MODE == 0) { if (kk < NO1) out(ii, kk, fo) = v[u]; }
        else if (fo < nfo1) { if (kk < NO1) out(ii, fo, kk) = v[u]; }
        else { if (kk < NO2) out2(ii, kk) = v[u]; }
      }
    }
  });
}

template<class DeviceType, class Traits>
template<int NQ, int NIN, int MODE, int NO1, int NO2, bool ASSIGN>
void Grace2LFusedKK<DeviceType, Traits>::red_bwd(const RedMeta &m, const t_nn3 &D, const typename Base::t_nn_2d &D2,
                                                 const t_nn3 &dX0, const t_nn3 &dX1, const t_nn3 &dX2, const t_nn3 &dX3)
{
  const int cs = this->chunk_size, co = this->chunk_offset, gtot = m.gtot, nx = m.nx, nfo1 = m.nfo1;
  const bool elem = m.elem;
  auto W0 = m.W[0]; auto W1 = m.W[1]; auto W2 = m.W[2]; auto W3 = m.W[3];
  auto W4 = m.W[4]; auto W5 = m.W[5]; auto W6 = m.W[6]; auto W7 = m.W[7];
  const int n0 = m.ws_nin[0], n1 = m.ws_nin[1], n2 = m.ws_nin[2], n3 = m.ws_nin[3];
  const int n4 = m.ws_nin[4], n5 = m.ws_nin[5], n6 = m.ws_nin[6], n7 = m.ws_nin[7];
  const int gb1 = m.gbase[1], gb2 = m.gbase[2], gb3 = m.gbase[3];
  auto roff = m.roff; auto rfo = m.rfo; auto rtile = m.rtile; auto rws = m.rws;
  if (ASSIGN)
    for (int x = 0; x < nx; x++) {
      const int e = (int) ((x == 0) ? dX0 : (x == 1) ? dX1 : (x == 2) ? dX2 : dX3).extent(1);
      if (e > NIN) GRACE_KK_ERROR("grace/2l/fused/kk: reduce input width exceeds NIN");
    }
  auto type = this->type; auto ilist = this->d_ilist;
  Kokkos::parallel_for("g2lf:RevReduce", Kokkos::RangePolicy<DeviceType>(0, cs * gtot), KOKKOS_LAMBDA(const int q) {
    const int g = q / cs, ii = q - g * cs;
    const int xs = (nx > 3 && g >= gb3) ? 3 : (nx > 2 && g >= gb2) ? 2 : (nx > 1 && g >= gb1) ? 1 : 0;
    const int fi = g - ((xs == 0) ? 0 : (xs == 1) ? gb1 : (xs == 2) ? gb2 : gb3);
    const int ty = elem ? type(ilist[ii + co]) : 0;
    F acc[NIN];
    for (int n = 0; n < NIN; n++) acc[n] = 0.0f;
    int nin = 0;
    for (int c = roff(g); c < roff(g + 1); c++) {
      const int fo = rfo(c), t = rtile(c), w = rws(c);
      nin = (w == 0) ? n0 : (w == 1) ? n1 : (w == 2) ? n2 : (w == 3) ? n3 : (w == 4) ? n4 : (w == 5) ? n5 : (w == 6) ? n6 : n7;
      const auto &W = (w == 0) ? W0 : (w == 1) ? W1 : (w == 2) ? W2 : (w == 3) ? W3 : (w == 4) ? W4 : (w == 5) ? W5 : (w == 6) ? W6 : W7;
      F4 gk[NQ];
      for (int k = 0; k < NQ; k++) {
        F v[4];
        for (int u = 0; u < 4; u++) {
          const int kk = 4 * k + u;
          if (MODE == 0) v[u] = (kk < NO1) ? D(ii, kk, fo) : 0.0f;
          else if (fo < nfo1) v[u] = (kk < NO1) ? D(ii, fo, kk) : 0.0f;
          else v[u] = (kk < NO2) ? D2(ii, kk) : 0.0f;
        }
        gk[k] = F4{v[0], v[1], v[2], v[3]};
      }
#pragma unroll
      for (int n = 0; n < NIN; n++) {
        if (n >= nin) continue;
        const F4 *wr = reinterpret_cast<const F4 *>(&W(ty, t, n, 0));
        F sacc = 0.0f;
        for (int k = 0; k < NQ; k++) sacc += dot4(wr[k], gk[k]);
        acc[n] += sacc;
      }
    }
    const auto &dX = (xs == 0) ? dX0 : (xs == 1) ? dX1 : (xs == 2) ? dX2 : dX3;
    const int nx_in = (int) dX.extent(1);      // <= NIN, checked on the host
    if (ASSIGN) {                                       // every (n, fi) of dX written: no zeroing pass
#pragma unroll
      for (int n = 0; n < NIN; n++) if (n < nx_in) dX(ii, n, fi) = acc[n];
    } else {
#pragma unroll
      for (int n = 0; n < NIN; n++) if (n < nin) dX(ii, n, fi) += acc[n];
    }
  });
}

template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_reduce_B()
{ red_fwd<G2LF_NO2 / 4, 0, G2LF_NO2, 0>(rm_B, this->d_YI, this->d_B0, this->d_YI, this->d_YI, this->d_B, this->d_rho); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_reduce_B()
{ red_bwd<G2LF_NO2 / 4, 32, 0, G2LF_NO2, 0, true>(rm_B, this->d_B_adj, this->d_rho_adj, this->d_YI_adj, this->d_B0_adj, this->d_YI_adj, this->d_YI_adj); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_reduce_I1()
{ red_fwd<5, 1, 12, 17>(rm_I1, this->d_A, this->d_AA, this->d_AAA, this->d_AAAA, this->d_I1, this->d_rho); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_reduce_I1()
{ red_bwd<5, G2LF_NO1, 1, 12, 17, true>(rm_I1, this->d_I1_adj, this->d_rho_adj, this->d_A_adj, this->d_AA_adj, this->d_AAA_adj, this->d_AAAA_adj); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_reduce_rho() {}        // done in stage_reduce_I1
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_reduce_rho() {}    // done in stage_rev_reduce_I1
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_zero_L1_adj() {}       // the I1+rho reverse assigns
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_zero_L2_adj() {}       // I2 and B reverses assign
// I2 <- B, BB, BBB, BBBB (scalar outputs, element-dependent): every connection to the single output fo = 0
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_reduce_I2()
{ red_fwd<5, 1, 0, 17>(rm_I2, this->d_B, this->d_BB, this->d_BBB, this->d_BBBB, this->d_B, this->d_I2); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_reduce_I2()
{ red_bwd<5, G2LF_NO2, 1, 0, 17, true>(rm_I2, this->d_B_adj, this->d_I2_adj, this->d_B_adj, this->d_BB_adj, this->d_BBB_adj, this->d_BBBB_adj); }

}    // namespace kk
}    // namespace grace

// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Implementation of grace::kk::Grace2LFusedKK (see grace_2l_fused_kk.h); included by one translation unit per consumer.
#pragma once

#include "grace/kk/grace_2l_fused_kk.h"
#include "grace/kk/grace_2l_kk_impl.h"
#include "grace/kk/fused_common.h"
#include "grace/kk/fused_dense.h"
#include "grace/kk/fused_hip_bonds.h"
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
// HIP: row stride of the layer-2 backward's M[y][channel][I] (I contiguous, 16-byte loads; 16 -> 20 floats so that
// the 16-byte reads of consecutive channels do not conflict)
constexpr int g2lf_nip(int nim) { return nim % 8 == 0 ? nim + 4 : nim; }
constexpr int G2LF_NO2 = 64;
// Level-0 scratch of the vector-ALU layer-2 kernels for a channel block CB (the kernels' own formulas; gpu: their GPU
// layouts, 16-byte rows). G2LF_SCRATCH: their budget (CUDA: below the 48 KB of a block); the GPU backward up to
// G2LF_SCRATCH_BWD (HIP: the 64 KB of a CU; CUDA with Kokkos 5.2+: dynamic shared memory above 48 KB, 2L-large CB 16
// with 58 KB, two teams per A100 SM). Always within the device limit (team_scratch_max), so a GPU with less shared
// memory per block gets fewer geometry rounds, a narrower channel block or the error of l2_channel_block, not a failed
// launch.
constexpr size_t G2LF_SCRATCH = 46000, G2LF_SCRATCH_BWD = 65536;
// fused_layer2_fwd_t<NIM>: staged geometry / radial / message of NB2 bonds, or Q [y][I][channel] for the CG step
inline size_t g2lf_fwd_t_shmem(bool gpu, int nim, int NI, int nl, int CB)
{
  const int nlm = nl * nl, nis = gpu ? g2lf_nip(nim) : NI;
  const int stage = G2LF_NB2 * (nlm + CB * nl + nis * CB + 1);
  return (size_t) std::max(stage, nlm * NI * CB) * sizeof(float) + G2LF_NB2 * sizeof(int) + (gpu ? 16 : 0);
}
// fused_layer2_bwd_t<NIM>: team size (HIP, 16-function message: 512, two waves per SIMD at the one team per CU that
// its 32 KB M allows; MI250X 31k: 86.3 -> 73.6 ms per call; else 256)
constexpr int g2lf_bwd_team(bool hip, int nim) { return hip && nim > 4 ? 512 : G2LF_TEAM; }
// Layer-2 backward, teams per SM (shared memory) with 3 geometry rounds / with one: one round when it turns one team
// into two (CUDA, 100 KB per SM, 2L-large: 1.3-2.1 % faster on L40S, 4-5 % on RTX 3060, 3 % on RTX 4070 SUPER); not
// generally the round count with more teams (A100, 164 KB: two teams with 3 rounds are 1-2 % faster than more with
// one). 0 teams (not known, e.g. HIP): the 3 rounds.
constexpr bool g2lf_one_round_better(int teams3, int teams1) { return teams3 == 1 && teams1 >= 2; }
// M [channel][y][I], Ba, the geometry (Y, G, 8 values) of NBG = rounds NBR bonds, the force partials of the team
// (TB threads); ints
inline size_t g2lf_bwd_t_shmem(bool gpu, int TB, int rounds, int nim, int NI, int nlm, int CB)
{
  const int NBR = TB / CB, NBG = rounds * NBR;
  const int nms = gpu ? nlm * g2lf_nip(nim) : nlm * NI;
  return (size_t) (nms * CB + nlm * CB + NBG * (4 * nlm + 8) + 3 * TB) * sizeof(float) + 2 * NBG * sizeof(int) +
         (gpu ? 16 : 0);
}
// the matrix-core layer-2 kernels: lmax 4, teams of 4 wavefronts x 2 channels, message rows of G2LF_MFMA_NIQ
constexpr int G2LF_MFMA_NIQ = 16;    // message functions per channel row of d_ITn / d_gIT (zero padded)
constexpr int G2LF_MFMA_CB = 8, G2LF_MFMA_TEAM = 256, G2LF_MFMA_NY = 25, G2LF_MFMA_NL = 5;
constexpr int G2LF_MFMA_MAXNEIGH = 384;    // forward: the geometry of an atom's bonds in shared memory (28 floats each)
// fused_layer2_fwd_m: geometry (Y, x', neighbour, type) of NBM bonds | later Q [CB][25][NI]; per wavefront R1 of a round
inline size_t g2lf_fwd_m_shmem(int maxneigh, int NI)
{
  const int nbm = (maxneigh + 15) / 16 * 16, geo = nbm * (G2LF_MFMA_NY + 3), qsz = G2LF_MFMA_CB * G2LF_MFMA_NY * NI;
  return (size_t) (std::max(geo, qsz) + G2LF_MFMA_CB * 16 * G2LF_MFMA_NL) * sizeof(float) + 16;
}
// fused_layer2_bwd_m: M [CB][25][17], Ba [CB][25], the team's dE/dYI rows [CB][NYo]
constexpr int G2LF_MFMA_MR = 17;    // M rows padded to 17 floats (V^T reads columns)
inline size_t g2lf_bwd_m_shmem(int NYo)
{
  return (size_t) (G2LF_MFMA_CB * G2LF_MFMA_NY * G2LF_MFMA_MR + G2LF_MFMA_CB * G2LF_MFMA_NY + G2LF_MFMA_CB * NYo) *
             sizeof(float) + 16;
}

constexpr int G2LF_RED_BW = 256;    // input rows (n) per block of the blocked forward reducers (GPU)
// GPU (HIP and CUDA): the forward reducers I1 / rho and I2 in connection blocks, the block partials summed by a second
// kernel in block order (deterministic): more teams for small systems (MI250X 2L-medium 5k atoms -10%, 128k -2%;
// A100 2L-medium / SMAX-medium 5k -14%, 2L-large 5k -8%, 125k-135k -1 to -2%, no cell slower). The fp32 summation
// order differs from the one-team-per-output kernel (forces up to 1.5e-5 eV/A apart on A100, closer to the CPU
// reference: SrTiO3-5k max |dF| 5.5-7.2e-6 against 1.25-1.3e-5). GRACE_RED_BLK=0: off.
template<class DeviceType, class M> inline bool red_blk_on(const M &m)
{
  static const int v = env_switch("GRACE_RED_BLK");
  return is_gpu_v<DeviceType> && v != 0 && m.nb > m.nfo;
}
}    // namespace g2lf_detail
using namespace g2lf_detail;

/* ---------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::setup()
{
  if (spline_n < 16) GRACE_KK_ERROR("GRACE-2L/KK: spline N must be >= 16 (got {})", spline_n);
  this->couple_supported = true;    // two-group coupling in the bond kernels
  Base::setup();     // sets the model dimensions and device weights
  if (this->lmax > G2LF_MAX_LMAX)
    GRACE_KK_ERROR("GRACE-2L/KK: lmax {} > {} not supported", this->lmax, G2LF_MAX_LMAX);
  if (this->L1_nradmax * (this->lmax + 1) * (this->lmax + 1) > G2LF_TEAM * G2LF_MAX_ACC)
    GRACE_KK_ERROR("GRACE-2L/KK: A basis too large for the bond kernel");
  const int nlm = (this->lmax + 1) * (this->lmax + 1);
  if (G2LF_TEAM % this->L2_nradmax != 0 || this->I_n_out != this->L2_nradmax ||
      this->L1_nradmax > G2LF_TEAM || this->I_n_funcs > G2LF_MAX_I ||
      nlm > (G2LF_TEAM / this->L2_nradmax) * G2LF_MAX_B)
    GRACE_KK_ERROR("GRACE-2L/KK: model shape outside the compiled kernel limits");
  const GRACE2LModel &mm = *this->grace_model;
  for (const GRACE2LModel::FCWeights *w : {&mm.fc_A1, &mm.fc_AA1, &mm.fc_AA2})
    if (w->n_out != G2LF_NO1 || (int) (w->w_left.size() / ((size_t) w->n_out * w->w_shape_left)) != G2LF_NO1)
      GRACE_KK_ERROR("GRACE-2L/KK: layer-1 FC width must be {}", G2LF_NO1);
  for (const GRACE2LModel::FCWeights *w : {&mm.fc_B1, &mm.fc_BB1, &mm.fc_BB2})
    if (w->n_out != G2LF_NO2 || (int) (w->w_left.size() / ((size_t) w->n_out * w->w_shape_left)) != G2LF_NO2)
      GRACE_KK_ERROR("GRACE-2L/KK: layer-2 FC width must be {}", G2LF_NO2);
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
  this->log.info(fmt("[GRACE-2L/KK] live FC outputs A1 {}/{} AA1 {}/{} AA2 {}/{} B1 {}/{} BB1 {}/{} BB2 {}/{}\n",
                   fcm_A1.nlive, fcm_A1.nf, fcm_AA1.nlive, fcm_AA1.nf, fcm_AA2.nlive, fcm_AA2.nf,
                   fcm_B1.nlive, fcm_B1.nf, fcm_BB1.nlive, fcm_BB1.nf, fcm_BB2.nlive, fcm_BB2.nf));
  if (this->I1_n_out != 12 || this->B_n_out != G2LF_NO2 || this->L2_nradmax > 32 || this->L1_nradmax != G2LF_NO1)
    GRACE_KK_ERROR("GRACE-2L/KK: ReduceN widths outside the compiled kernels (I1 12, B 64)");
  if (this->rho_n_out != 17) GRACE_KK_ERROR("GRACE-2L/KK: rho width must be 17");
  if (this->I2_n_out != 17) GRACE_KK_ERROR("GRACE-2L/KK: I2 width must be 17");
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
    this->log.info(fmt("[GRACE-2L/KK] CG products {}\n",
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
    if (q != P) GRACE_KK_ERROR("GRACE-2L/KK: YI pair grouping incomplete");
    Kokkos::deep_copy(d_yi_gperm, hp); Kokkos::deep_copy(d_yi_goff, ho);
  }
#if GRACE_KK_GEMM_MFMA    // read only by the matrix-core layer-2 forward (fused_layer2_fwd_m, CDNA builds)
  {   // the CG terms of each YI output as (Q index y * NI + I, cg), CSR by output
    const int NYo = this->n_yi_out_funcs, NI = this->I_n_funcs;
    auto hoff = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->d_yi_out_offset);
    auto hcnt = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->d_yi_out_count);
    auto htp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->d_yi_term_pair);
    auto hcg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->d_yi_cg_coeff);
    auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->d_yi_pair_yidx);
    auto hi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), this->d_yi_pair_iidx);
    int nt = 0;
    for (int f = 0; f < NYo; f++) nt += hcnt(f);
    d_cgf_off = V1<int>("g2lf:cgf_off", NYo + 1); d_cgf_t = V1<CGTerm>("g2lf:cgf_t", std::max(nt, 1));
    auto ho = Kokkos::create_mirror_view(d_cgf_off); auto ht = Kokkos::create_mirror_view(d_cgf_t);
    int q = 0;
    for (int f = 0; f < NYo; f++) {
      ho(f) = q;
      for (int u = 0; u < hcnt(f); u++, q++) {
        const int t = hoff(f) + u, p = htp(t);
        ht(q) = CGTerm{hy(p) * NI + hi(p), (float) hcg(t)};
      }
    }
    ho(NYo) = q;
    Kokkos::deep_copy(d_cgf_off, ho); Kokkos::deep_copy(d_cgf_t, ht);
  }
#endif
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
    GRACE_KK_ERROR("GRACE-2L/KK: radial MLP output width mismatch");
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
  bonds_ready = false;          // the compact bond list belongs to that run (begin_run)
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
  check_kernel_launches<DeviceType>("GRACE-2L/KK");
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
#if GRACE_KK_GEMM_MFMA
  if constexpr (is_hip_v<DeviceType>) {
    if (a_mfma) {    // CDNA: matrix-core basis (begin_run)
      auto run = [&](auto cpl) {
        GraceFusedKK::a_fwd_mfma<DeviceType, Traits, decltype(cpl)::value>(
            "g2lf:L1_A", this->chunk_size, this->chunk_offset, this->L1_nradmax, this->L1_inv_avg_n_neigh, spline_n,
            this->d_A, this->x, this->type, this->d_map, bond_list(), this->d_ilist, this->d_bond_cutoff,
            this->d_z_tr_A, this->couple_eq, this->couple_bit1, this->couple_bit2, this->couple_lambda, this->d_mask,
            this->d_idx_sph, this->alm, this->blm, this->cl, this->dl, d_splR);
      };
      if (this->couple_eq) run(std::true_type{});
      else run(std::false_type{});
      return;
    }
  }
#endif
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
  auto ilist = this->d_ilist;
  auto nbl = bond_list();    // the atom's bonds (GPU: the run's compact list)
  auto bc = this->d_bond_cutoff; auto ztr = this->d_z_tr_A; auto lfrom = this->d_l_from_lm;
  const int ceq = this->couple_eq, cb1 = this->couple_bit1, cb2 = this->couple_bit2;    // two-group coupling (CPL)
  const double clam = this->couple_lambda;
  auto cmask = this->d_mask;
  auto idx_s = this->d_idx_sph; auto alm = this->alm; auto blm = this->blm;
  auto cl = this->cl; auto dl = this->dl;
  auto sc = d_splR;
  const int N = spline_n;
  const F inv = this->L1_inv_avg_n_neigh;

  using team_policy = Kokkos::TeamPolicy<DeviceType, hip_lb_t<DeviceType, G2LF_TEAM, 2>>;
  using member = typename team_policy::member_type;
  // GPU: z_A[mu_j, n] folded into the staged radial (Rs = R z, zero for the bonds out of range and Y zeroed for them),
  // so the accumulation is one FMA per (output, bond) on two shared reads: no global z load, no branch on mus[b]
  constexpr bool gpu = is_gpu_v<DeviceType>;
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

    const int jnum = nbl.count(i, ii + co);
    for (int jb = 0; jb < jnum; jb += G2LF_NB) {
      const int nb = (jnum - jb < G2LF_NB) ? jnum - jb : G2LF_NB;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, G2LF_NB), [&](const int b) {
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
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, G2LF_NB * nout), [&](const int q) {
        const int b = q / nout, o = q - b * nout;
        if (gpu) Rs[q] = (mus[b] >= 0) ? spline_v(sc, N, xs[b], o) * ztr(mus[b], o / nl) : 0.0f;
        else Rs[q] = (mus[b] >= 0) ? spline_v(sc, N, xs[b], o) : 0.0f;
      });
      t.team_barrier();
#pragma unroll
      for (int k = 0; k < G2LF_MAX_ACC; k++) {
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
  const int cl = this->chunk_cap();
  if (this->inum <= cl) return;                       // one chunk: nothing is recomputed anyway
  const int c = this->chunk_offset / cl, nch = (this->inum + cl - 1) / cl;
  if (this->chunksize >= 0 && !backward && c == 0 && nch != l1_nofit_nch) l1_fits = true;    // decided again below
  if (!l1_store_on()) return;
  if (!backward && c == 0) {
    auto same = [](const auto &a, const auto &b) {    // every extent of every stored tensor
      for (int r = 0; r < (int) a.rank(); r++)
        if (a.extent(r) != b.extent(r)) return false;
      return true;
    };
    const bool fresh = (int) st_A.size() != nch || st_A.empty() || !same(st_A[0], this->d_A) ||
                       !same(st_A1[0], this->d_A1) || !same(st_AA1[0], this->d_AA1) ||
                       !same(st_AA2[0], this->d_AA2) || !same(st_rho[0], this->d_rho);
    if (fresh) {
      st_A.clear(); st_A1.clear(); st_AA1.clear(); st_AA2.clear(); st_rho.clear();
      // a fixed chunk (also the smaller chunks after an allocation failure of chunksize auto): the store only when it
      // fits 70 % of the free memory (as GRACE-3L store_fwd auto), else layer 1 is recomputed in the backward
      // (same results) instead of failing at allocation
      if (this->chunksize >= 0) {
        using M = grace::kk::MemKK;
        const double bytes = (double) nch * (M::memory_usage(this->d_A) + M::memory_usage(this->d_A1) +
                                             M::memory_usage(this->d_AA1) + M::memory_usage(this->d_AA2) +
                                             M::memory_usage(this->d_rho));
        size_t fr = 0, tot = 0;
        if (grace::kk::device_mem_info(fr, tot) && bytes > 0.7 * (double) fr / std::max(1, this->ranks_per_device)) {
          l1_fits = false;
          l1_nofit_nch = nch;
          this->log.info(grace::kk::fmt("[GRACE-2L/KK] {} chunks: layer 1 recomputed in the backward (store would need "
                                        "{:.0f} MB, free {:.0f} MB)\n", nch, bytes / 1048576.0, fr / 1048576.0));
          return;
        }
      }
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

// GPU: the run's compact bond list (every bond kernel of the run reads it) and the bond kernels for this model and
// maxneigh, one line on stderr per process (the kernels of the first run and why); the wave-per-atom L1 backward is the
// default on HIP and CUDA (A100 g2lf:L1_bwd 99 -> 66 ms per call, C-diamond-125k). All builds: the message buffers of
// the layer-2 kernels that run, nall rows (allocated before the chunk decision, which then sees them as used memory).
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::begin_run()
{
  l2p = L2Path{};
  if constexpr (is_gpu_v<DeviceType>) {
    GraceFusedKK::compact_bonds<DeviceType, Traits>(this->inum, this->maxneigh, this->x, this->type, this->d_map,
        this->d_neighbors, this->d_numneigh, this->d_ilist, this->d_bond_cutoff, this->couple_eq, this->couple_bit1,
        this->couple_bit2, this->couple_lambda, this->d_mask, d_cnb, d_cnum);
    bonds_ready = true;
    const int lmax = this->lmax, NR1 = this->L1_nradmax;
    const KernelChoice a = GraceFusedKK::fused_mfma(GraceFusedKK::a_fwd_mfma_fits<DeviceType>(lmax, NR1));
    const KernelChoice w = GraceFusedKK::fused_wave_bwd(GraceFusedKK::bond_bwd_wave_fits<DeviceType>(lmax, NR1));
    const KernelChoice lf = GraceFusedKK::fused_mfma(l2_fwd_mfma_fits());
    KernelChoice lb = GraceFusedKK::fused_mfma(lf.on && l2_bwd_mfma_fits());
    // the matrix-core backward only for more than 4 message functions (2L-medium, 4: the vector-ALU kernel is faster,
    // MI250X 31k 23.0 vs 32.2 ms per call)
    if (lb.on && this->I_n_funcs <= 4) lb = {false, "4 message functions: the vector-ALU kernel is faster"};
    a_mfma = a.on;
    bwd_wave = w.on;
    l2p.fwd_m = lf.on;
    l2p.bwd_m = lb.on;
    l2p.niq = l2p.need_ITn() ? G2LF_MFMA_NIQ : 0;
    static std::once_flag logged;
    std::call_once(logged, [&] {
      std::fprintf(stderr,
                   "grace kk: GRACE-2L bond kernels: layer-1 basis A on the %s, layer-2 forward on the %s, layer-2 "
                   "backward on the %s, layer-1 bond backward on the %s kernel (%s); dense FCs on the %s kernels\n",
                   GraceFusedKK::fused_mfma_says(a).c_str(), GraceFusedKK::fused_mfma_says(lf).c_str(),
                   GraceFusedKK::fused_mfma_says(lb).c_str(), w.on ? "wave-per-atom" : "team-of-256", w.why.c_str(),
                   fcm_A1.mfma_f ? "matrix-core" : "vector-ALU");
    });
  }
  // uninitialised (the transposes write every value, d_gIT is zeroed per run); the unused layout is freed
  const int nall = this->nall, NI = this->I_n_funcs, NR2 = this->I_n_out;
  auto fit = [&](grace::kk::View64<V2<F>> &v, const char *name, bool need, int w) {
    if (!need) MemKK::drop(v);
    else if ((int) v.extent(0) < nall || (int) v.extent(1) != w) MemKK::realloc_kokkos(v, name, nall, w);
  };
  fit(d_IT, "g2lf:IT", l2p.need_IT(), NI * NR2);
  fit(d_ITn, "g2lf:ITn", l2p.need_ITn(), l2p.niq * NR2);
  fit(d_gIT, "g2lf:gIT", true, l2p.bwd_m ? l2p.niq * NR2 : NI * NR2);
  // GPU blocked forward reducers: their block partials go into the buffer of YI_adj (free during the I1 / rho and I2
  // reductions) when its rows hold them (GRACE-2L OMAT / SMAX-OMAT, medium and large: 1299 and 2595 partial rows per
  // atom, YI_adj 1984 and 6080), else into d_redP. d_redP is allocated here, before the chunk decision (which counts
  // it as used memory), for all inum atoms although red_fwd uses nprow x chunk_size of it: more than multi-chunk runs
  // need
  int nprow = 0;
  if (red_blk_on<DeviceType>(rm_I1)) nprow = std::max(nprow, rm_I1.nprow);
  if (red_blk_on<DeviceType>(rm_I2)) nprow = std::max(nprow, rm_I2.nprow);
  if (nprow <= red_partials_in_yi_adj()) MemKK::drop(d_redP);
  else if ((size_t) d_redP.extent(0) < (size_t) nprow * this->inum) MemKK::realloc_kokkos(d_redP, "g2lf:redP", (size_t) nprow * this->inum);
}

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_L1_bond_recompute() { fused_layer1_A(); }

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_L2_bond_fwd()
{
  // the message I (after the forward comm) in the neighbour-contiguous layouts the run's kernels read (L2Path)
  if (this->chunk_offset == 0) {
    const int nall = this->nall, NI = this->I_n_funcs, NR2 = this->I_n_out;
    auto Ig = this->d_I_global;
    if (l2p.need_IT()) {    // [nall][I NR2 + n]
      auto IT = d_IT;
      Kokkos::parallel_for("g2lf:I_transpose", Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0, 0}, {nall, NI * NR2}),
        KOKKOS_LAMBDA(const int j, const int q) { IT(j, q) = Ig(j, q / NR2, q % NR2); });
    }
    if (l2p.need_ITn()) {    // [nall][niq n + I], I >= NI zero (16-byte aligned rows)
      const int niq = l2p.niq;
      auto ITn = d_ITn;
      Kokkos::parallel_for("g2lf:I_transpose_n", Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0, 0}, {nall, niq * NR2}),
        KOKKOS_LAMBDA(const int j, const int q) { ITn(j, q) = (q % niq < NI) ? Ig(j, q % niq, q / niq) : 0.0f; });
    }
  }
  fused_layer2_fwd();
  if (l2_rows()) {    // per-atom rows (in the adjoints' buffers) -> YI, B0
    rows_transpose(false, this->d_YI, this->d_YI_adj.data());
    rows_transpose(false, this->d_B0, this->d_B0_adj.data());
  }
}

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_L2_bond_bwd()
{
  const int nall = this->nall, NI = this->I_n_funcs, NR2 = this->I_n_out;
  if (this->chunk_offset == 0) Kokkos::deep_copy(Kokkos::subview(d_gIT.base(), std::make_pair(0, nall), Kokkos::ALL), 0.0f);
  if (l2_rows()) {    // dE/dYI, dE/dB0 -> per-atom rows (in the buffers of YI, B0)
    rows_transpose(true, this->d_YI_adj, this->d_YI.data());
    rows_transpose(true, this->d_B0_adj, this->d_B0.data());
  }
  fused_layer2_bwd();
  if (this->chunk_offset + this->chunk_size >= this->inum) {   // last chunk: hand dE/dI to the base for the reverse comm
    // d_gIT in the layout of the backward that ran (L2Path), read along its rows: q = I NR2 + n (vector-ALU) or
    // n NI + I (matrix-core; contiguous d_gIT reads: MI250X 2L-large 128k, 3.6 ms per call vs 7.9 with contiguous
    // d_grad_I_global rows instead)
    const bool m = l2p.bwd_m;
    const int nf = m ? NI : NR2, sI = m ? 1 : NR2, sN = m ? l2p.niq : 1;
    auto gIT = d_gIT; auto gIg = this->d_grad_I_global;
    Kokkos::parallel_for("g2lf:gI_add", Kokkos::MDRangePolicy<DeviceType, Kokkos::Rank<2>>({0, 0}, {nall, NI * NR2}),
      KOKKOS_LAMBDA(const int j, const int q) {
        const int a = q / nf, b = q - a * nf, I = m ? b : a, n = m ? a : b;
        gIg(j, I, n) += gIT(j, I * sI + n * sN);
      });
  }
}

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_L1_bond_bwd() { fused_layer1_bwd(); }

template<class DeviceType, class Traits>
int Grace2LFusedKK<DeviceType, Traits>::l2_channel_block(bool bwd, int *rounds) const
{
  // channels per team: the widest block whose shared memory (the kernel's formula) fits the budget: G2LF_SCRATCH, and
  // the device limit for the kernel's team; the GPU backward G2LF_SCRATCH_BWD (HIP teams of 512, 16-byte rows)
  constexpr bool hip = is_hip_v<DeviceType>, gpu = is_gpu_v<DeviceType>;
  const int NR2 = this->L2_nradmax, NI = this->I_n_funcs, nl = this->lmax + 1, nlm = nl * nl;
  const int nim = NI <= 4 ? 4 : G2LF_MAX_I;
  const int team = bwd ? g2lf_bwd_team(hip, nim) : G2LF_TEAM;
  const size_t lim = team_scratch_max<DeviceType>(team);
  const size_t budget = std::min(lim, (gpu && bwd) ? G2LF_SCRATCH_BWD : G2LF_SCRATCH);
  auto shmem = [&](int r, int CB) -> size_t {    // r: geometry rounds of the backward
    if (bwd) return g2lf_bwd_t_shmem(gpu, team, r, nim, NI, nlm, CB);
    const int ys = (nlm + G2LF_TEAM / CB - 1) / (G2LF_TEAM / CB);
    return ys <= G2LF_MAX_YS ? g2lf_fwd_t_shmem(gpu, nim, NI, nl, CB) : SIZE_MAX;
  };
  auto widest = [&](int r) {    // 0: none fits
    for (int CB = NR2; CB >= 8; CB /= 2)
      if (NR2 % CB == 0 && G2LF_TEAM % CB == 0 && shmem(r, CB) <= budget) return CB;
    return 0;
  };
  // backward, GPU teams of 256: the geometry of 3 rounds per step; one round where no channel block fits with 3
  // (CUDA with Kokkos 5.1, 48 KB: 2L-large CB 16 with 43 KB instead of 58 KB) or where it doubles the teams per SM
  // (g2lf_one_round_better)
  int r = (bwd && gpu && team <= 256) ? 3 : 1, CB = widest(r);
  if (r == 3) {
    const int CB1 = widest(1);
    if (CB == 0 || (CB1 > 0 && g2lf_one_round_better(teams_per_sm_by_scratch<DeviceType>(shmem(3, CB), team),
                                                     teams_per_sm_by_scratch<DeviceType>(shmem(1, CB1), team)))) {
      r = 1;
      CB = CB1;
    }
  }
  if (CB == 0)
    GRACE_KK_ERROR("GRACE-2L/KK: layer-2 {} kernel: no channel block of {} fits the team scratch memory",
                   bwd ? "backward" : "forward", NR2);
  if (rounds) *rounds = r;
  return CB;
}

/* ----------------------------------------------------------------------
   Layer 2 bond forward. Per neighbour j and channel n the pair contributions are the outer
   product w_y u_I, w_y = Y_y(r_ij) R1_{n,l(y)}(x_ij), u_I = I_j[I, n]; accumulated densely:
     Q[y, I, n] = sum_j w_y u_I            (only the (y, I) of the YI pairs are read later)
     B0[n, y]   = L2_inv * sum_j w_y z_B0[mu_j, n]
     YI[n, f]   = yi_inv * sum_{t in f} cg_t Q[y_t, I_t, n]
   Team per (atom, channel block of CB); thread (channel, y-group) keeps Q[y-slots][I] in registers.
------------------------------------------------------------------------- */

// the model (lmax 4, at most G2LF_MFMA_NIQ message functions, channels in blocks of 8) and maxneigh inside the limits of
// the matrix-core layer-2 kernels, their shared memory inside the device's
template<class DeviceType, class Traits>
bool Grace2LFusedKK<DeviceType, Traits>::l2_fwd_mfma_fits() const
{
  return this->lmax == 4 && this->I_n_funcs <= G2LF_MFMA_NIQ && this->L2_nradmax % G2LF_MFMA_CB == 0 &&
         this->maxneigh <= G2LF_MFMA_MAXNEIGH &&
         g2lf_fwd_m_shmem(this->maxneigh, this->I_n_funcs) <= team_scratch_max<DeviceType>(G2LF_MFMA_TEAM);
}

// HIP matrix-core forward: YI, B0 (forward) and dE/dYI, dE/dB0 (backward) of the bond kernels as per-atom rows
// T[ii][n * nf + f]: the team of an atom and 8 channels writes (reads) 8 nf contiguous floats instead of 8 nf single
// floats 4 chunk_size bytes apart (one cache line each: the YI stores were half of fused_layer2_fwd_m for 2L-large).
// A tiled transpose converts from / to the atom-contiguous views the reducers read and write. The rows live in the
// buffer of the other view (YI_adj, B0_adj are free in the forward, YI, B0 in the backward). GRACE_L2_ROWS=0: the
// direct strided accesses.
template<class DeviceType, class Traits>
bool Grace2LFusedKK<DeviceType, Traits>::l2_rows() const
{
  static const int env = env_switch("GRACE_L2_ROWS");
  return is_hip_v<DeviceType> && env != 0 && l2p.fwd_m;    // the adjoints have the rows of YI, B0 (allocated together)
}

// V(ii, n, f) (atom-contiguous) <-> T[ii][n * nf + f], ii < chunk_size: tiles of 64 atoms x 64 row floats through
// shared memory (tile rows padded to 65 floats)
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::rows_transpose(bool to_rows, const t_nn3 &V, F *T)
{
  const int cs = this->chunk_size, nf = (int) V.extent(2), RS = (int) V.extent(1) * nf;
  const int na = (cs + 63) / 64, nr = (RS + 63) / 64;
  using team_policy = Kokkos::TeamPolicy<DeviceType, hip_lb_t<DeviceType, 256, 2>>;
  using member = typename team_policy::member_type;
  team_policy pol(na * nr, 256);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(64 * 65 * sizeof(F)));
  Kokkos::parallel_for("g2lf:rows_transpose", pol, KOKKOS_LAMBDA(const member &t) {
    const int tr = t.league_rank() / na, a0 = (t.league_rank() - tr * na) * 64, r0 = tr * 64;
    F *tile = (F *) t.team_shmem().get_shmem(64 * 65 * sizeof(F));
    const int tid = t.team_rank();
    for (int k = tid; k < 64 * 64; k += 256) {    // coalesced reads: along the atoms of V, along the rows of T
      const int a = to_rows ? (k & 63) : (k >> 6), r = to_rows ? (k >> 6) : (k & 63);
      const int ii = a0 + a, rr = r0 + r;
      if (ii < cs && rr < RS) {
        const int n = rr / nf;
        tile[a * 65 + r] = to_rows ? V(ii, n, rr - n * nf) : T[(size_t) ii * RS + rr];
      }
    }
    t.team_barrier();
    for (int k = tid; k < 64 * 64; k += 256) {
      const int a = to_rows ? (k >> 6) : (k & 63), r = to_rows ? (k & 63) : (k >> 6);
      const int ii = a0 + a, rr = r0 + r;
      if (ii < cs && rr < RS) {
        const int n = rr / nf;
        if (to_rows) T[(size_t) ii * RS + rr] = tile[a * 65 + r];
        else V(ii, n, rr - n * nf) = tile[a * 65 + r];
      }
    }
  });
}

template<class DeviceType, class Traits>
bool Grace2LFusedKK<DeviceType, Traits>::l2_bwd_mfma_fits() const
{
  return this->lmax == 4 && this->I_n_funcs <= G2LF_MFMA_NIQ && this->L2_nradmax % G2LF_MFMA_CB == 0 &&
         g2lf_bwd_m_shmem(this->n_yi_out_funcs) <= team_scratch_max<DeviceType>(G2LF_MFMA_TEAM);
}

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer2_fwd()
{
#if GRACE_KK_GEMM_MFMA
  if constexpr (is_hip_v<DeviceType>) {
    if (l2p.fwd_m) {    // CDNA, at most 16 message functions: matrix-core kernel (begin_run)
      if (this->couple_eq) fused_layer2_fwd_m<true>();
      else fused_layer2_fwd_m<false>();
      return;
    }
  }
#endif
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
  auto ilist = this->d_ilist;
  auto nbl = bond_list();    // the atom's bonds (GPU: the run's compact list)
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

  using team_policy = Kokkos::TeamPolicy<DeviceType, hip_lb_t<DeviceType, G2LF_TEAM, 2>>;
  using member = typename team_policy::member_type;
  constexpr int NB = G2LF_NB2;
  // GPU: the staged message as [bond][channel][I] with rows of NIP (16-byte loads along I); elsewhere [bond][I][channel]
  constexpr bool gpu = is_gpu_v<DeviceType>;
  constexpr int NIP = g2lf_nip(NIM);
  const int NIS = gpu ? NIP : NI;
  const int stage = NB * (nlm + CBnl + NIS * CB + 1);
  const int shf = (stage > nlm * NI * CB) ? stage : nlm * NI * CB;
  team_policy pol(cs * nblk, G2LF_TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(g2lf_fwd_t_shmem(gpu, NIM, NI, nl, CB)));    // shf floats, NB ints

  Kokkos::parallel_for("g2lf:L2_fwd", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank() / nblk, c0 = (t.league_rank() - ii * nblk) * CB;
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *sh;
    if (gpu) sh = (F *) t.team_shmem().get_shmem_aligned(shf * sizeof(F), 16);
    else sh = (F *) t.team_shmem().get_shmem(shf * sizeof(F));
    int *mus = (int *) t.team_shmem().get_shmem(NB * sizeof(int));
    F *Ys = sh, *Rs = sh + NB * nlm, *Is = Rs + NB * CBnl, *xs = Is + NB * NIS * CB;
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

    const int jnum = nbl.count(i, ii + co);
    for (int jb = 0; jb < jnum; jb += NB) {
      const int nb = (jnum - jb < NB) ? jnum - jb : NB;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NB), [&](const int b) {
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
          for (int y = 0; y < nlm; y++) Ys[b * nlm + y] = 0.0f;
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
        if (mus[b] >= 0) v = IT(nbl(i, ii + co, jb + b), I * NR2 + c0 + c);
        if (gpu) Is[(b * CB + c) * NIP + I] = v;
        else Is[q] = v;
      });
      t.team_barrier();
      for (int b = 0; b < nb; b++) {
        int mu = mus[b];
        // GPU: no branch per bond (Y, R and the message of a bond out of range are zero in scratch, z of any type is
        // finite), so the loads of the next bonds can be issued early
        if (gpu) mu = mu < 0 ? 0 : mu;
        else if (mu < 0) continue;
        const F *Yb = Ys + b * nlm, *Rb = Rs + b * CBnl + nloc * nl;
        F u[NIM];
        if (gpu) {
          const GF4 *I4 = reinterpret_cast<const GF4 *>(Is + (b * CB + nloc) * NIP);
#pragma unroll
          for (int q = 0; q < NIM / 4; q++) {
            const GF4 v = I4[q];
            u[4 * q] = (4 * q < NI) ? v.x : 0.0f; u[4 * q + 1] = (4 * q + 1 < NI) ? v.y : 0.0f;
            u[4 * q + 2] = (4 * q + 2 < NI) ? v.z : 0.0f; u[4 * q + 3] = (4 * q + 3 < NI) ? v.w : 0.0f;
          }
        } else {
          const F *Ib = Is + b * NI * CB + nloc;
#pragma unroll
          for (int I = 0; I < NIM; I++) u[I] = (I < NI) ? Ib[I * CB] : 0.0f;
        }
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

#if GRACE_KK_GEMM_MFMA
/* ----------------------------------------------------------------------
   Layer 2 bond forward on the CDNA matrix cores (v_mfma_f32_16x16x4f32), lmax 4, at most 16 message functions.
   Per atom i and channel n, over the bonds j:  Q[y, I] = sum_j w[j, y] u[j, I]  (w = Y_y R1_{n,l(y)}, u = I_j[I, n]),
   B0[n, y] = L2_inv sum_j w[j, y] z_B0[mu_j, n], then the CG step to YI as fused_layer2_fwd_t.
   Team = one atom x 8 channels = 4 wavefronts (wavefront w: channels c0 + 2 w, + 1). The team first computes the geometry of
   all the atom's bonds (x', type, neighbour, Y in shared memory); each wavefront then runs rounds of 16 bonds, 4 MFMA
   steps per round with the lane group g = lane / 16 <-> bond jb + 4 g + s of step s: A = w^T (row y = li of tile 0,
   y = 16 + li of tile 1), B = u (column I = li). R1 of (bond li, l = g; group 0 also l = 4) once per round through
   wave-private shared memory. Bonds past the count have Y = 0 (exact zeros). Q of the 4 channels then reuses the
   geometry's shared memory for the CG step.
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
template<bool CPL>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer2_fwd_m()
{
  constexpr int CPW = 2, NW = 4, CB = G2LF_MFMA_CB, NY = G2LF_MFMA_NY, NL = G2LF_MFMA_NL, TEAM = G2LF_MFMA_TEAM;
  static_assert(CB == NW * CPW && TEAM == 64 * NW, "4 wavefronts x CPW channels");    // CPW channels per wavefront
  constexpr int NIQ = G2LF_MFMA_NIQ;
  const int cs = this->chunk_size, co = this->chunk_offset;
  const int NR2 = this->L2_nradmax, nblk = NR2 / CB;
  const int NI = this->I_n_funcs, NYo = this->n_yi_out_funcs;
  const int NBM = (this->maxneigh + 15) / 16 * 16;    // bonds of an atom <= neighbours, rounds of 16
  auto B0 = this->d_B0; auto YI = this->d_YI;
  auto ITn = d_ITn;
  auto x = this->x; auto type = this->type; auto map = this->d_map;
  auto ilist = this->d_ilist;
  auto nbl = bond_list();    // the run's compact bond list
  auto bc = this->d_bond_cutoff; auto ztr = this->d_z_tr_B0;
  const int ceq = this->couple_eq, cb1 = this->couple_bit1, cb2 = this->couple_bit2;
  const double clam = this->couple_lambda;
  auto cmask = this->d_mask;
  auto idx_s = this->d_idx_sph; auto alm = this->alm; auto blm = this->blm;
  auto cl = this->cl; auto dl = this->dl;
  auto cgo = d_cgf_off; auto cgt = d_cgf_t;
  // HIP: YI and B0 as per-atom rows in the buffers of YI_adj and B0_adj (l2_rows, rows_transpose)
  const bool rows = l2_rows();
  F *YT = this->d_YI_adj.data(), *BT = this->d_B0_adj.data();
  const int RS = NR2 * NYo;
  auto sc = d_splR1;
  const int N = spline_n;
  const F L2inv = this->L2_inv_avg_n_neigh, yinv = this->yi_inv_avg_n_neigh;

  using team_policy = Kokkos::TeamPolicy<DeviceType, Kokkos::LaunchBounds<TEAM, 2>>;
  using member = typename team_policy::member_type;
  // shared (g2lf_fwd_m_shmem): geometry (Y [NBM][25], x', neighbour, type) | later Q [CB][25][NI]; per wavefront R1
  // of a round [CPW][16][5]
  const int geo = NBM * (NY + 3), qsz = CB * NY * NI;
  const int sh1 = geo > qsz ? geo : qsz;
  team_policy pol(cs * nblk, TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(g2lf_fwd_m_shmem(this->maxneigh, NI)));

  Kokkos::parallel_for("g2lf:L2_fwd", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank() / nblk, c0 = (t.league_rank() - ii * nblk) * CB;
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *sh = (F *) t.team_shmem().get_shmem_aligned((size_t) sh1 * sizeof(F), 16);
    F *rws = (F *) t.team_shmem().get_shmem((size_t) NW * CPW * 16 * NL * sizeof(F));
    F *Yl = sh, *xl = sh + NBM * NY;
    int *jl = (int *) (xl + NBM), *ml = jl + NBM;
    const int tid = t.team_rank(), w = tid >> 6, lane = tid & 63, g = lane >> 4, li = lane & 15, n0 = c0 + CPW * w;
    const int nct = nbl.count(i, ii + co), nr = (nct + 15) / 16 * 16;

    // geometry of the atom's bonds (bonds past the count: Y = 0, the atom itself as neighbour)
    for (int b = tid; b < nr; b += TEAM) {
      F *Yb = Yl + b * NY;
      int j = i, mu = 0;
      F xb = 0.0f;
      if (b < nct) {    // a bond of the compact list (no cutoff test)
        j = nbl(i, ii + co, b);
        const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi;    // bond geometry in fp64
        const double r2 = dx * dx + dy * dy + dz * dz;
        const int muj = map(type(j));
        const double rc = bc(mui, muj);
        const double r = Kokkos::sqrt(r2), rinv = 1.0 / r;
        double rm = r, drm, dlm;    // the model's bond length: r' of a coupled bond (grace/coupling.h)
        if (CPL && coupling_cross(cmask(i), cmask(j), cb1, cb2)) coupling_shift(ceq, r, rc, clam, rm, drm, dlm);
        mu = muj; xb = F(rm / rc);
        real_sph(4, F(dx * rinv), F(dy * rinv), F(dz * rinv), idx_s, alm, blm, cl, dl, Yb);
      } else {
        for (int y = 0; y < NY; y++) Yb[y] = 0.0f;
      }
      xl[b] = xb; jl[b] = j; ml[b] = mu;
    }
    t.team_barrier();

    const int ly0 = l_of_y(li);    // l of y = li; y = 16 + li is l = 4
    const bool y1ok = li < NY - 16;
    F *rw = rws + w * CPW * 16 * NL;
    mfma_f4v acc0[CPW], acc1[CPW];
    F b00[CPW], b01[CPW];
#pragma unroll
    for (int k = 0; k < CPW; k++) {
      acc0[k] = mfma_f4v{0.0f, 0.0f, 0.0f, 0.0f}; acc1[k] = acc0[k];
      b00[k] = 0.0f; b01[k] = 0.0f;
    }
    for (int jb = 0; jb < nct; jb += 16) {
      F zl[CPW];
      {   // R1 of bond jb + li for l = g (group 0 also l = 4), and its z, the wavefront's channels
        const F xb = xl[jb + li];
        const int mb = ml[jb + li];
#pragma unroll
        for (int k = 0; k < CPW; k++) {
          rw[(k * 16 + li) * NL + g] = spline_v(sc, N, xb, (n0 + k) * NL + g);
          if (g == 0) rw[(k * 16 + li) * NL + 4] = spline_v(sc, N, xb, (n0 + k) * NL + 4);
          zl[k] = ztr(mb, n0 + k);
        }
      }
      wave_sync();
      F av0[CPW][4], av1[CPW][4], bv[CPW][4];
#pragma unroll
      for (int s = 0; s < 4; s++) {
        const int bl = 4 * g + s, b = jb + bl;
        const F *Yb = Yl + b * NY;
        const F y0v = Yb[li], y1v = y1ok ? Yb[16 + li] : 0.0f;
        const int jj = jl[b];
#pragma unroll
        for (int k = 0; k < CPW; k++) {
          av0[k][s] = rw[(k * 16 + bl) * NL + ly0] * y0v;
          av1[k][s] = rw[(k * 16 + bl) * NL + 4] * y1v;
          bv[k][s] = ITn(jj, NIQ * (n0 + k) + li);    // zero for li >= NI
          const F z = wave_shfl(zl[k], bl);
          b00[k] += av0[k][s] * z; b01[k] += av1[k][s] * z;
        }
      }
      wave_sync();
#pragma unroll
      for (int k = 0; k < CPW; k++)
#pragma unroll
        for (int s = 0; s < 4; s++) {
          mfma_16x16x4(av0[k][s], bv[k][s], acc0[k]);
          mfma_16x16x4(av1[k][s], bv[k][s], acc1[k]);
        }
    }
    // B0[n, y] for y = li, 16 + li: sum over the 4 lane groups
#pragma unroll
    for (int k = 0; k < CPW; k++) {
      F u0 = b00[k], u1 = b01[k];
      u0 += wave_shfl_xor(u0, 16); u0 += wave_shfl_xor(u0, 32);
      u1 += wave_shfl_xor(u1, 16); u1 += wave_shfl_xor(u1, 32);
      if (g == 0) {
        if (rows) {
          F *Bb = BT + (size_t) ii * NR2 * NY + (size_t) (n0 + k) * NY;
          Bb[li] = u0 * L2inv;
          if (y1ok) Bb[16 + li] = u1 * L2inv;
        } else {
          B0(ii, n0 + k, li) = u0 * L2inv;
          if (y1ok) B0(ii, n0 + k, 16 + li) = u1 * L2inv;
        }
      }
    }
    t.team_barrier();    // geometry no longer read: Q of the CB channels in its place, [c][y][I]
#pragma unroll
    for (int k = 0; k < CPW; k++) {
      F *Qw = sh + (CPW * w + k) * NY * NI;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int y0 = 4 * g + q, y1 = 16 + 4 * g + q;
        if (li < NI) {
          Qw[y0 * NI + li] = acc0[k][q];
          if (y1 < NY) Qw[y1 * NI + li] = acc1[k][q];
        }
      }
    }
    t.team_barrier();
    // CG step, thread per (output function, channel) on the team's 4 wavefronts; a term is one (Q index, cg) load (no
    // dependent pair-index loads). Per-atom rows: the team writes 8 NYo contiguous floats
    for (int q = tid; q < NYo * CB; q += TEAM) {
      int f, c;
      if (rows) { c = q / NYo; f = q - c * NYo; }
      else { f = q / CB; c = q - f * CB; }
      const int t1 = cgo(f + 1);
      const F *Qc = sh + c * NY * NI;
      F s = 0.0f;
      for (int u = cgo(f); u < t1; u++) {
        const CGTerm e = cgt(u);
        s += e.cg * Qc[e.q];
      }
      if (rows) YT[(size_t) ii * RS + (size_t) c0 * NYo + q] = s * yinv;
      else YI(ii, c0 + c, f) = s * yinv;
    }
  });
}
#endif

/* ----------------------------------------------------------------------
   Layer 2 bond backward. With M[y, I, n] = dE/dQ (dense, from dE/dYI through the CG step):
     dE/dI_j[I, n] = sum_y M[y, I, n] w_y,   v_y = sum_I M[y, I, n] u_I,
     dE/dR1_{n,l} += sum_{y in l} Y_y v_y,   dE/dY_y += R1_{n,l(y)} v_y   (+ the B0 terms).
   Team per (atom, channel block); rounds of TEAM / CB bonds, thread (bond, channel).
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer2_bwd()
{
#if GRACE_KK_GEMM_MFMA
  if constexpr (is_hip_v<DeviceType>) {
    if (l2p.bwd_m) {    // CDNA, 5..16 message functions: matrix-core kernel (begin_run; dE/dI as [nall][niq n + I])
      if (this->couple_eq) fused_layer2_bwd_m<true>();
      else fused_layer2_bwd_m<false>();
      return;
    }
  }
#endif
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
  constexpr bool hip = is_hip_v<DeviceType>, gpu = is_gpu_v<DeviceType>;
  constexpr int TB = g2lf_bwd_team(hip, NIM);    // team size: HIP 16-function message 512, else 256
  const int cs = this->chunk_size, co = this->chunk_offset;
  const int lmax = this->lmax, nl = lmax + 1, nlm = nl * nl;
  const int NR2 = this->L2_nradmax;
  const int P = this->n_yi_pairs, NI = this->I_n_funcs;
  int rounds = 1;
  const int CB = l2_channel_block(true, &rounds), nblk = NR2 / CB;
  const int NBR = TB / CB;                    // bonds per round
  const int NM = nlm * NI;
  // GPU: M as [y][channel][I] with rows of NIP (16-byte loads along I); host backends [y][I][channel]
  constexpr int NIP = g2lf_nip(NIM);
  const int NMS = gpu ? nlm * NIP : NM;    // M floats per channel

  auto B0a = this->d_B0_adj; auto YIa = this->d_YI_adj;
  const bool rows = l2_rows();    // HIP: dE/dYI, dE/dB0 as per-atom rows in the buffers of YI, B0 (rows_transpose)
  static const KernelChoice shf_sw = wave64_switch("GRACE_L2_BWD_SHFL");
  const bool shf = hip && shf_sw.on && CB <= 32;    // HIP CDNA (64-lane wavefronts): channel sums of the forces by shuffles
  if (hip) {    // one line per process, as the other kernel choices
    static std::once_flag logged;
    std::call_once(logged, [] {
      std::fprintf(stderr, "grace kk: GRACE-2L layer-2 backward channel sums by %s (%s)\n",
                   shf_sw.on ? "wavefront shuffles" : "shared memory", shf_sw.why.c_str());
    });
  }
  const F *YT = this->d_YI.data(), *BT = this->d_B0.data();
  const int NYo = this->n_yi_out_funcs, RS = NR2 * NYo;
  auto IT = d_IT; auto gIT = d_gIT;
  auto f = this->f; auto vir = d_vir;
  auto x = this->x; auto type = this->type; auto map = this->d_map;
  auto ilist = this->d_ilist;
  auto nbl = bond_list();    // the atom's bonds (GPU: the run's compact list)
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

  // HIP: two waves per SIMD (teams of 512: at the one team per CU of the 16-function message)
  using team_policy = Kokkos::TeamPolicy<DeviceType, hip_lb_t<DeviceType, TB, 2>>;
  using member = typename team_policy::member_type;
  // scratch (g2lf_bwd_t_shmem): M [NMS][CB], Ba [nlm][CB], Y [NBG][nlm], G [NBG][3 nlm], geo [NBG][8], Fs [TB][3]; ints.
  // GPU, teams of 256: the bond geometry of NBG = 3 NBR bonds per step, then three rounds of NBR (fewer geometry steps
  // and barriers; 7 KB more scratch: still two teams per CU for the 4-function message) where it fits
  // (l2_channel_block); teams of 512: NBG = NBR
  const int NBG = rounds * NBR;
  const int nf = NMS * CB + nlm * CB + NBG * (4 * nlm + 8) + 3 * TB;
  team_policy pol(cs * nblk, TB);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(g2lf_bwd_t_shmem(gpu, TB, rounds, NIM, NI, nlm, CB)));

  Kokkos::parallel_for("g2lf:L2_bwd", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank() / nblk, c0 = (t.league_rank() - ii * nblk) * CB;
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *Ms;
    if (gpu) Ms = (F *) t.team_shmem().get_shmem_aligned(nf * sizeof(F), 16);
    else Ms = (F *) t.team_shmem().get_shmem(nf * sizeof(F));
    int *mus = (int *) t.team_shmem().get_shmem(2 * NBG * sizeof(int));
    int *js = mus + NBG;
    F *Ba = Ms + NMS * CB, *Ys = Ba + nlm * CB, *Gs = Ys + NBG * nlm, *geo = Gs + NBG * 3 * nlm;
    F *Fs = geo + NBG * 8;
    const int tid = t.team_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NMS * CB), [&](const int q) { Ms[q] = 0.0f; });
    t.team_barrier();
    if (gpu) {
      // thread per (pair, 8 channels): the CG table read once per term for 8 channels
      const int nc8 = CB / 8;
      for (int q = tid; q < P * nc8; q += TB) {
        const int p = q / nc8, cb = (q - p * nc8) * 8;
        const int off = poff(p), cnt = pcnt(p), row = py(p) * CB, col = pi_(p);
        F s[8];
#pragma unroll
        for (int k = 0; k < 8; k++) s[k] = 0.0f;
        for (int u = 0; u < cnt; u++) {
          const F cg = pcg(off + u);
          const int fo = pout(off + u);
#pragma unroll
          for (int k = 0; k < 8; k++)
            s[k] += cg * (rows ? YT[(size_t) ii * RS + (size_t) (c0 + cb + k) * NYo + fo] : YIa(ii, c0 + cb + k, fo));
        }
#pragma unroll
        for (int k = 0; k < 8; k++) Ms[(row + cb + k) * NIP + col] = s[k] * yinv;
      }
    } else {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, P * CB), [&](const int q) {
      const int p = q / CB, c = q - p * CB;
      const int off = poff(p), cnt = pcnt(p);
      F s = 0.0f;
      for (int u = 0; u < cnt; u++) s += pcg(off + u) * YIa(ii, c0 + c, pout(off + u));
      Ms[(py(p) * NI + pi_(p)) * CB + c] = s * yinv;
    });
    }
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nlm * CB), [&](const int q) {
      if (rows) {    // contiguous row reads
        const int c = q / nlm, lm = q - c * nlm;
        Ba[lm * CB + c] = BT[(size_t) ii * NR2 * nlm + (size_t) c0 * nlm + q] * L2inv;
      } else {
        const int lm = q / CB, c = q - lm * CB;
        Ba[q] = B0a(ii, c0 + c, lm) * L2inv;
      }
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
              real_sph_grad(lmax, gb[3], gb[4], gb[5], idx_s, alm, blm, cl, dl,
                            Ys + b * nlm, Gs + b * 3 * nlm);
            }
          }
        }
        mus[b] = mu;
      });
      t.team_barrier();
      for (int sb = 0; sb < nb; sb += NBR) {
      {
        const int b = sb + tid / CB, nloc = tid - (tid / CB) * CB, n = c0 + nloc;
        F fx = 0.0f, fy = 0.0f, fz = 0.0f;
        if (tid < NBR * CB && b < nb && mus[b] >= 0) {
          const int mu = mus[b], j = js[b];
          const F *gb = geo + b * 8, *Yb = Ys + b * nlm, *Gb = Gs + b * 3 * nlm;
          F R[G2LF_MAX_LMAX + 1], dR[G2LF_MAX_LMAX + 1], gR[G2LF_MAX_LMAX + 1];
#pragma unroll
          for (int l = 0; l <= G2LF_MAX_LMAX; l++) {
            R[l] = dR[l] = gR[l] = 0.0f;
            if (l < nl) spline_vd(sc, N, gb[0], n * nl + l, R[l], dR[l]);
          }
          F u[NIM], gI[NIM];
          if (gpu) {
            // GPU: unconditional loads (the index clamped to the last message function; M of I >= NI is 0, so those
            // u never contribute) and no per-I test in the y loop below: straight FMA chains instead of a branch,
            // a multiply, a select and an add per (y, I) with a runtime NI
#pragma unroll
            for (int I = 0; I < NIM; I++) { u[I] = IT(j, (I < NI ? I : NI - 1) * NR2 + n); gI[I] = 0.0f; }
          } else {
#pragma unroll
            for (int I = 0; I < NIM; I++) { u[I] = (I < NI) ? IT(j, I * NR2 + n) : 0.0f; gI[I] = 0.0f; }
          }
          F gx = 0.0f, gy = 0.0f, gz = 0.0f;
          const F z = ztr(mu, n);
          // two y loops per layout (one loop with the inner M u product branched: MI250X L2_bwd 1.2 % slower)
          if (gpu) {
            // the row's I as 16-byte loads; M of I >= NI is 0 (zeroed above): no test on I. The y loop of an l is
            // not unrolled: the fully unrolled loop hoisted the shared-memory loads of all 25 rows
            // (256 VGPRs, spills for the 4-function message)
#pragma unroll
            for (int l = 0; l <= G2LF_MAX_LMAX; l++) {
              if (l >= nl) continue;
#pragma unroll 1
              for (int y = l * l; y < (l + 1) * (l + 1); y++) {
                const F wy = Yb[y] * R[l];
                F vy = Ba[y * CB + nloc] * z;                 // B0 term: dE/dw_y
                const GF4 *My4 = reinterpret_cast<const GF4 *>(Ms + (y * CB + nloc) * NIP);
#pragma unroll
                for (int I4 = 0; I4 < NIM / 4; I4++) {
                  const GF4 m4 = My4[I4];
                  const F mv[4] = {m4.x, m4.y, m4.z, m4.w};
#pragma unroll
                  for (int q = 0; q < 4; q++) {
                    const int I = 4 * I4 + q;
                    vy += mv[q] * u[I];
                    gI[I] += mv[q] * wy;
                  }
                }
                gR[l] += Yb[y] * vy;
                const F c = R[l] * vy;
                gx += c * Gb[3 * y]; gy += c * Gb[3 * y + 1]; gz += c * Gb[3 * y + 2];
              }
            }
          } else {
#pragma unroll
          for (int l = 0; l <= G2LF_MAX_LMAX; l++) {
            if (l >= nl) continue;
            for (int y = l * l; y < (l + 1) * (l + 1); y++) {
              const F wy = Yb[y] * R[l];
              F vy = Ba[y * CB + nloc] * z;                 // B0 term: dE/dw_y
              const F *My = Ms + y * NI * CB + nloc;
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
        if (shf) {
          // HIP: the bond's channel sums over its CB contiguous lanes (one wavefront) by xor shuffles; its first lane
          // adds the force (no shared-memory round trip, no barrier, no serial loop over the channels)
          for (int o = 1; o < CB; o *= 2) {
            fx += wave_shfl_xor(fx, o); fy += wave_shfl_xor(fy, o); fz += wave_shfl_xor(fz, o);
          }
          if (nloc == 0 && tid < NBR * CB && b < nb && mus[b] >= 0) {
            const int j = js[b];
            Kokkos::atomic_add(&f(i, 0), (typename Traits::kk_acc_float) fx);
            Kokkos::atomic_add(&f(i, 1), (typename Traits::kk_acc_float) fy);
            Kokkos::atomic_add(&f(i, 2), (typename Traits::kk_acc_float) fz);
            Kokkos::atomic_add(&f(j, 0), (typename Traits::kk_acc_float) -fx);
            Kokkos::atomic_add(&f(j, 1), (typename Traits::kk_acc_float) -fy);
            Kokkos::atomic_add(&f(j, 2), (typename Traits::kk_acc_float) -fz);
            const F *gb = geo + b * 8;
            if (do_v) {
              const double dx = -gb[3] * gb[6], dy = -gb[4] * gb[6], dz = -gb[5] * gb[6];
              v[0] += dx * fx; v[1] += dy * fy; v[2] += dz * fz;
              v[3] += dx * fy; v[4] += dx * fz; v[5] += dy * fz;
            }
            if (do_va || do_cva)
              tally_vatom(va, cva, do_va, do_cva, i, j, -gb[3] * gb[6], -gb[4] * gb[6], -gb[5] * gb[6], fx, fy, fz);
          }
        } else {
          Fs[3 * tid] = fx; Fs[3 * tid + 1] = fy; Fs[3 * tid + 2] = fz;
        }
      }
      if (shf) continue;
      t.team_barrier();
      if (tid < NBR && sb + tid < nb && mus[sb + tid] >= 0) {
        const int b = sb + tid, j = js[b];
        F fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int c = 0; c < CB; c++) {
          fx += Fs[3 * (tid * CB + c)]; fy += Fs[3 * (tid * CB + c) + 1]; fz += Fs[3 * (tid * CB + c) + 2];
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
      if (shf) t.team_barrier();    // the next step's geometry overwrites the bonds' shared memory
    }
    if (do_v && (shf ? (tid % CB == 0 && tid < NBR * CB) : tid < NBR))
      for (int k = 0; k < 6; k++) Kokkos::atomic_add(&vir(ii + co, k), v[k]);
    if (CPL) {    // dE/dlambda of this atom's bonds
      double s = dla;
      t.team_reduce(Kokkos::Sum<double>(s));
      if (tid == 0) Kokkos::atomic_add(&dedl(), s);
    }
  });
}

#if GRACE_KK_GEMM_MFMA
/* ----------------------------------------------------------------------
   Layer 2 bond backward on the CDNA matrix cores (v_mfma_f32_16x16x4f32), lmax 4, 5..16 message functions.
   Per atom i, channel n, bond j (M, Ba as fused_layer2_bwd_t, u = I_j[., n], w = Y R1_{n,l(y)}):
     G = W M        (bonds x I: dE/dI_j, atomics into gIT as [nall][16 n + I])
     V = U M^T      (bonds x y), v += Ba z: dE/dR1_{n,l} += sum_{y in l} Y_y v_y, g += R1 v dY/dr
   Team = one atom x 8 channels = 4 wavefronts; M and Ba of the 8 channels in shared memory. The wavefronts split the
   rounds of 16 bonds (wavefront w: rounds w, w + 4, ...); lane (li, g = lane / 16) has bond jb + li throughout and
   computes its geometry and Y / dY in registers (the 4 lane groups redundantly), keeping only the y it needs:
     G steps s = 0..6: k = g <-> y = 4 s + g, A = w (row = bond li), B = M[y][I = li]: D register q = (bond 4 g + q, I li)
     V^T steps s = 0..3: k = g <-> I = 4 g + s, A = M (row y = li, 16 + li), B = u[bond li][I]: D register q of tile t
       = (y = 16 t + 4 g + q, bond li)
   so the lane holds v for y = 4 g + q, 16 + 4 g + q of its bond; the force parts (dE/dx', g) are summed over the 8
   channels in registers, then over the 4 lane groups (two xor shuffles), one atomic per (bond, component) and team.
   Bonds past the count: zero geometry, z = 0 (exact zeros), no atomics.
------------------------------------------------------------------------- */

namespace g2lf_detail {
// the y a lane group needs: G operand y = 4 s + g (s < 7), V rows y = 4 g + q and 16 + 4 g + q (q < 4); 0 past y 24
template<int GG>
KOKKOS_INLINE_FUNCTION void g2lf_ysel(const float *Y, const float *G, float *yg, float *yv, float *gv)
{
#pragma unroll
  for (int s = 0; s < 7; s++) yg[s] = (4 * s + GG < 25) ? Y[4 * s + GG] : 0.0f;
#pragma unroll
  for (int q = 0; q < 8; q++) {
    const int y = (q < 4) ? 4 * GG + q : 16 + 4 * GG + q - 4;
    yv[q] = (y < 25) ? Y[y] : 0.0f;
    gv[3 * q] = (y < 25) ? G[3 * y] : 0.0f;
    gv[3 * q + 1] = (y < 25) ? G[3 * y + 1] : 0.0f;
    gv[3 * q + 2] = (y < 25) ? G[3 * y + 2] : 0.0f;
  }
}
}    // namespace g2lf_detail

template<class DeviceType, class Traits>
template<bool CPL>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer2_bwd_m()
{
  using KKA = typename Traits::kk_acc_float;
  constexpr int CB = G2LF_MFMA_CB, NY = G2LF_MFMA_NY, NL = G2LF_MFMA_NL, TEAM = G2LF_MFMA_TEAM;
  constexpr int MR = G2LF_MFMA_MR, NIQ = G2LF_MFMA_NIQ;
  const int cs = this->chunk_size, co = this->chunk_offset;
  const int NR2 = this->L2_nradmax, nblk = NR2 / CB;
  const int P = this->n_yi_pairs, NI = this->I_n_funcs;
  auto B0a = this->d_B0_adj; auto YIa = this->d_YI_adj;
  auto ITn = d_ITn; auto gIT = d_gIT;
  auto f = this->f; auto vir = d_vir;
  auto x = this->x; auto type = this->type; auto map = this->d_map;
  auto ilist = this->d_ilist;
  auto nbl = bond_list();    // the run's compact bond list
  auto bc = this->d_bond_cutoff; auto ztr = this->d_z_tr_B0;
  const int ceq = this->couple_eq, cb1 = this->couple_bit1, cb2 = this->couple_bit2;
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

  using team_policy = Kokkos::TeamPolicy<DeviceType, Kokkos::LaunchBounds<TEAM, 2>>;
  using member = typename team_policy::member_type;
  const int NYo = this->n_yi_out_funcs;
  const bool rows = l2_rows();    // dE/dYI, dE/dB0 as per-atom rows in the buffers of YI, B0 (rows_transpose)
  const F *YT = this->d_YI.data(), *BT = this->d_B0.data();
  const int RS = NR2 * NYo;
  // shared (g2lf_bwd_m_shmem): M [CB][25][MR], Ba [CB][25], the team's dE/dYI rows [CB][NYo] (contiguous loads)
  team_policy pol(cs * nblk, TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(g2lf_bwd_m_shmem(NYo)));

  Kokkos::parallel_for("g2lf:L2_bwd", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank() / nblk, c0 = (t.league_rank() - ii * nblk) * CB;
    const int i = ilist[ii + co];
    const int mui = map(type(i));
    const auto xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);    // positions' precision (double)
    F *Ms = (F *) t.team_shmem().get_shmem_aligned((size_t) (CB * NY * MR + CB * NY + CB * NYo) * sizeof(F), 16);
    F *Bas = Ms + CB * NY * MR, *Ya = Bas + CB * NY;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, CB * NY * MR), [&](const int q) { Ms[q] = 0.0f; });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, CB * NYo), [&](const int q) {
      const int c = q / NYo, fo = q - c * NYo;
      Ya[q] = rows ? YT[(size_t) ii * RS + (size_t) c0 * NYo + q] : YIa(ii, c0 + c, fo);
    });
    t.team_barrier();
    // M, thread per (y, I) pair for the team's channels (the CG table read once per term, not per channel)
    for (int p = t.team_rank(); p < P; p += TEAM) {
      const int off = poff(p), cnt = pcnt(p), row = py(p) * MR + pi_(p);
      F s[CB];
#pragma unroll
      for (int c = 0; c < CB; c++) s[c] = 0.0f;
      for (int u = 0; u < cnt; u++) {
        const F cg = pcg(off + u);
        const int fo = pout(off + u);
#pragma unroll
        for (int c = 0; c < CB; c++) s[c] += cg * Ya[c * NYo + fo];
      }
#pragma unroll
      for (int c = 0; c < CB; c++) Ms[c * NY * MR + row] = s[c] * yinv;
    }
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, NY * CB), [&](const int q) {
      const int c = q / NY, y = q - c * NY;
      Bas[q] = (rows ? BT[(size_t) ii * NR2 * NY + (size_t) c0 * NY + q] : B0a(ii, c0 + c, y)) * L2inv;
    });
    t.team_barrier();

    const int tid = t.team_rank(), w = tid >> 6, lane = tid & 63, g = lane >> 4, li = lane & 15;
    double fi[3] = {0.0, 0.0, 0.0}, v[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dla = 0.0;    // dE/dlambda (coupling)
    const int nct = nbl.count(i, ii + co);
    for (int jb = 16 * w; jb < nct; jb += 64) {
      const int b = jb + li;
      const bool valid = b < nct;
      int j = i, mu = 0;
      F xg = 0.0f, gdr = 0.0f, grinv = 0.0f, rx = 0.0f, ry = 0.0f, rz = 0.0f, gr = 0.0f, gdl = 0.0f;
      if (valid) {    // a bond of the compact list (no cutoff test)
        j = nbl(i, ii + co, b);
        const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi;    // bond geometry in fp64
        const double r2 = dx * dx + dy * dy + dz * dz;
        const int muj = map(type(j));
        const double rc = bc(mui, muj);
        const double r = Kokkos::sqrt(r2), rinv = 1.0 / r;
        double rm = r, drm = 1.0, dlm = 0.0;    // coupled bond (grace/coupling.h): r', dr'/dr, dr'/dlambda
        if (CPL && coupling_cross(cmask(i), cmask(j), cb1, cb2)) coupling_shift(ceq, r, rc, clam, rm, drm, dlm);
        mu = muj;
        xg = F(rm / rc); gdr = F(drm / rc); grinv = F(rinv);
        rx = F(dx * rinv); ry = F(dy * rinv); rz = F(dz * rinv);
        gr = F(r);
        if (CPL) gdl = F(dlm / rc);
      }
      F yg[7], yv[8], gv[24];
      {
        F Y[NY], Gd[3 * NY];
        real_sph_grad<1, 3, 4>(4, rx, ry, rz, idx_s, alm, blm, cl, dl, Y, Gd);
        if (!valid) {
#pragma unroll
          for (int k = 0; k < NY; k++) { Y[k] = 0.0f; Gd[3 * k] = Gd[3 * k + 1] = Gd[3 * k + 2] = 0.0f; }
        }
        if (g == 0) g2lf_detail::g2lf_ysel<0>(Y, Gd, yg, yv, gv);
        else if (g == 1) g2lf_detail::g2lf_ysel<1>(Y, Gd, yg, yv, gv);
        else if (g == 2) g2lf_detail::g2lf_ysel<2>(Y, Gd, yg, yv, gv);
        else g2lf_detail::g2lf_ysel<3>(Y, Gd, yg, yv, gv);
      }
      // neighbours of the bonds 4 g + q of the round (the G rows of this lane), and whether they are bonds
      int jq[4];
#pragma unroll
      for (int q = 0; q < 4; q++) jq[q] = wave_shfl(j, 4 * g + q);
      // l of the V rows: q = 0 -> lA, q = 1..3 -> lB (tile 0), l = 4 (tile 1)
      const int lA = (g == 0) ? 0 : (g == 3) ? 3 : 2, lB = (g == 0) ? 1 : (g == 1) ? 2 : 3;
      F RAD = 0.0f, GX = 0.0f, GY = 0.0f, GZ = 0.0f;
#pragma unroll 2
      for (int c = 0; c < CB; c++) {
        const int n = c0 + c;
        // R1, dR1 of the lane's bond: lane group g evaluates l = g (group 0 also l = 4), the lanes exchange them
        F R[NL], dR[NL];
        {
          F r0, d0, r4 = 0.0f, d4 = 0.0f;
          spline_vd(sc, N, xg, n * NL + g, r0, d0);
          if (g == 0) spline_vd(sc, N, xg, n * NL + 4, r4, d4);
#pragma unroll
          for (int l = 0; l < 4; l++) {
            R[l] = wave_shfl(r0, li + 16 * l);
            dR[l] = wave_shfl(d0, li + 16 * l);
          }
          R[4] = wave_shfl(r4, li);
          dR[4] = wave_shfl(d4, li);
        }
        const F z = valid ? ztr(mu, n) : 0.0f;
        const F *Mc = Ms + c * NY * MR;
        // V^T = M U^T, step s <-> I = 4 g + s: u[bond li][4 g .. 4 g + 3] as one 16-byte load
        mfma_f4v aV0 = {0.0f, 0.0f, 0.0f, 0.0f}, aV1 = {0.0f, 0.0f, 0.0f, 0.0f};
        GF4 u4 = {0.0f, 0.0f, 0.0f, 0.0f};
        if (valid) u4 = *reinterpret_cast<const GF4 *>(&ITn(j, NIQ * n + 4 * g));    // zero past NI
        const F uv[4] = {u4.x, u4.y, u4.z, u4.w};
#pragma unroll
        for (int s = 0; s < 4; s++) {
          const int I = 4 * g + s;
          mfma_16x16x4(Mc[li * MR + I], uv[s], aV0);
          mfma_16x16x4(li < NY - 16 ? Mc[(16 + li) * MR + I] : 0.0f, uv[s], aV1);
        }
        // G = W M
        mfma_f4v aG = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int s = 0; s < 7; s++) {
          const int y = 4 * s + g;
          // l of y and R[l] as explicit selects of the possible values (l_of_y / sel5 here: L2_bwd 2.6 % slower)
          const int l = (s == 0) ? (g == 0 ? 0 : 1) : (s == 1) ? 2 : (s == 2) ? (g == 0 ? 2 : 3) : (s == 3) ? 3 : 4;
          const F Rl = (l == 0) ? R[0] : (l == 1) ? R[1] : (l == 2) ? R[2] : (l == 3) ? R[3] : R[4];
          mfma_16x16x4(yg[s] * Rl, y < NY ? Mc[y * MR + li] : 0.0f, aG);
        }
        // dE/dI_j[I = li, n] of the bonds 4 g + q
        if (li < NI) {
#pragma unroll
          for (int q = 0; q < 4; q++)
            if (jb + 4 * g + q < nct && aG[q] != 0.0f) Kokkos::atomic_add(&gIT(jq[q], NIQ * n + li), aG[q]);
        }
        // force parts of the lane's bond from v of its rows
        const F *Bc = Bas + c * NY;
        const F RA = (lA == 0) ? R[0] : (lA == 2) ? R[2] : R[3], dRA = (lA == 0) ? dR[0] : (lA == 2) ? dR[2] : dR[3];
        const F RB = (lB == 1) ? R[1] : (lB == 2) ? R[2] : R[3], dRB = (lB == 1) ? dR[1] : (lB == 2) ? dR[2] : dR[3];
        F radc = 0.0f, gxa = 0.0f, gya = 0.0f, gza = 0.0f;
#pragma unroll
        for (int q = 0; q < 8; q++) {
          const int y = (q < 4) ? 4 * g + q : 16 + 4 * g + q - 4;
          const F vq = ((q < 4) ? aV0[q] : aV1[q - 4]) + ((y < NY) ? Bc[y] * z : 0.0f);
          const F Rq = (q == 0) ? RA : (q < 4) ? RB : R[4], dRq = (q == 0) ? dRA : (q < 4) ? dRB : dR[4];
          radc += dRq * yv[q] * vq;
          const F cr = Rq * vq;
          gxa += cr * gv[3 * q]; gya += cr * gv[3 * q + 1]; gza += cr * gv[3 * q + 2];
        }
        RAD += radc; GX += gxa; GY += gya; GZ += gza;
      }
      RAD += wave_shfl_xor(RAD, 16); RAD += wave_shfl_xor(RAD, 32);
      GX += wave_shfl_xor(GX, 16); GX += wave_shfl_xor(GX, 32);
      GY += wave_shfl_xor(GY, 16); GY += wave_shfl_xor(GY, 32);
      GZ += wave_shfl_xor(GZ, 16); GZ += wave_shfl_xor(GZ, 32);
      if (g == 0 && valid) {
        F rad = RAD;
        if (CPL) dla += rad * gdl;    // dE/dlambda: dE/dx' dx'/dlambda
        rad *= gdr;                   // d/dr = d/dx' dx'/dr
        const F rg = rx * GX + ry * GY + rz * GZ;
        const F fx = rad * rx + (GX - rg * rx) * grinv;
        const F fy = rad * ry + (GY - rg * ry) * grinv;
        const F fz = rad * rz + (GZ - rg * rz) * grinv;
        fi[0] += (KKA) fx; fi[1] += (KKA) fy; fi[2] += (KKA) fz;
        Kokkos::atomic_add(&f(j, 0), (KKA) -fx);
        Kokkos::atomic_add(&f(j, 1), (KKA) -fy);
        Kokkos::atomic_add(&f(j, 2), (KKA) -fz);
        const double dx = -rx * gr, dy = -ry * gr, dz = -rz * gr;
        if (do_v) {
          v[0] += dx * fx; v[1] += dy * fy; v[2] += dz * fz;
          v[3] += dx * fy; v[4] += dx * fz; v[5] += dy * fz;
        }
        if (do_va || do_cva) tally_vatom(va, cva, do_va, do_cva, i, j, dx, dy, dz, fx, fy, fz);
      }
    }
    for (int k = 0; k < 3; k++) {
      double sv = fi[k];
      t.team_reduce(Kokkos::Sum<double>(sv));
      if (tid == 0) Kokkos::atomic_add(&f(i, k), (KKA) sv);
    }
    if (do_v)
      for (int k = 0; k < 6; k++) {
        double sv = v[k];
        t.team_reduce(Kokkos::Sum<double>(sv));
        if (tid == 0) Kokkos::atomic_add(&vir(ii + co, k), sv);
      }
    if (CPL) {
      double sv = dla;
      t.team_reduce(Kokkos::Sum<double>(sv));
      if (tid == 0) Kokkos::atomic_add(&dedl(), sv);
    }
  });
}
#endif

/* ----------------------------------------------------------------------
   Layer 1 bond backward: from dE/dA of atom i, per bond the force dE/dr_ij.
   Rounds of TEAM / NR neighbours: thread (b, n), n = radial channel.
------------------------------------------------------------------------- */

template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::fused_layer1_bwd()
{
  if constexpr (is_gpu_v<DeviceType>) {
    if (bwd_wave) {    // GPU: one 64-thread team per atom, thread = bond (begin_run)
      auto run = [&](auto cpl) {
        GraceFusedKK::bond_bwd_wave<DeviceType, Traits, decltype(cpl)::value>(
            "g2lf:L1_bwd", this->chunk_size, this->chunk_offset, this->L1_nradmax, this->L1_inv_avg_n_neigh, spline_n,
            this->d_A_adj, this->f, d_vir, this->x, this->type, this->d_map, bond_list(), this->d_ilist,
            this->d_bond_cutoff, this->d_z_tr_A, this->couple_eq, this->couple_bit1, this->couple_bit2,
            this->couple_lambda, this->d_mask, this->d_dedl, this->d_idx_sph, this->alm, this->blm, this->cl, this->dl,
            d_splR, this->vflag_global != 0, this->vflag_atom != 0, this->cvflag_atom != 0, this->d_vatom,
            this->d_cvatom);
      };
      if (this->couple_eq) run(std::true_type{});
      else run(std::false_type{});
      return;
    }
  }
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
  auto ilist = this->d_ilist;
  auto nbl = bond_list();    // the atom's bonds (GPU: the run's compact list)
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

  using member = typename Kokkos::TeamPolicy<DeviceType>::member_type;    // launch: fbwd_launch
  // GPU: dE/dA rows as [channel][lm] (LMP floats, 16-byte loads), Y and G of a bond as [lm][Y, Gx, Gy, Gz]
  constexpr bool gpu = is_gpu_v<DeviceType>;
  const int LMP = fused_lmp(nlm);
  // GPU: the bond geometry of NBG >= NBR bonds per step (more lanes busy, fewer barriers), then rounds of NBR
  const int NBG = gpu ? (NBR > 48 ? NBR : 48 / NBR * NBR) : NBR;
  const int nf = (gpu ? NR * LMP : nlm * NR) + NBG * (4 * nlm + 8) + 3 * G2LF_TEAM;
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
      if (gpu) Aa_s[c * LMP + lm] = Aa(ii, c, lm) * L1inv;
      else Aa_s[q] = Aa(ii, c, lm) * L1inv;
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
                real_sph_grad(lmax, gb[3], gb[4], gb[5], idx_s, alm, blm, cl, dl,
                              Ys + b * nlm, Gs + b * 3 * nlm);
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
          F R[G2LF_MAX_LMAX + 1], dR[G2LF_MAX_LMAX + 1], gR[G2LF_MAX_LMAX + 1];
#pragma unroll
          for (int l = 0; l <= G2LF_MAX_LMAX; l++) {
            R[l] = dR[l] = gR[l] = 0.0f;
            if (l < nl) spline_vd(sc, N, gb[0], n * nl + l, R[l], dR[l]);
          }
          F gx = 0.0f, gy = 0.0f, gz = 0.0f;
          const F z = ztr(mu, n);
          if (gpu) {
            // the same sums: dE/dA row of channel n as float4, [Y, G] of (bond, lm) as one float4
            constexpr int LMX = fused_lmp((G2LF_MAX_LMAX + 1) * (G2LF_MAX_LMAX + 1));
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
              for (int l = 0; l <= G2LF_MAX_LMAX; l++) {
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
              for (int lm = 0; lm < (G2LF_MAX_LMAX + 1) * (G2LF_MAX_LMAX + 1); lm++) {
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
          for (int l = 0; l <= G2LF_MAX_LMAX; l++) {
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
  GraceFusedKK::fbwd_launch<DeviceType, G2LF_TEAM>("g2lf:L1_bwd", cs, shmem, body);
}

/* ----------------------------------------------------------------------
   FC (FCRight2Left) directly on the [chunk, channel, func] views, no gather/scatter:
     out[a,k,f] = nof[f] * ( sum_n nl Wl[tl(f)][n][k] in_l[a,n,f]
                           + sum_{c: ct(c)=f} sum_n nr Wr[tr(c)][n][k] in_r[a,n,cf(c)] )
   One thread per (atom, f) with NO accumulators; the right half is a CSR by target,
   the reverse right half a CSR by source (no atomics).
------------------------------------------------------------------------- */

// dense-chain kernels and metadata: shared with Grace1LFusedKK in grace/kk/fused_dense.h
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::build_fc_meta(FCMeta &m, const GRACE2LModel::FCWeights &fc, const std::vector<int> &used)
{
  GraceFusedKK::build_fc_meta(m, fc, used, "GRACE-2L/KK");
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

// GPU (CUDA and HIP): the fused FC kernels (no BLAS) through the backend-neutral hook of the base engine; host
// backends keep the plain FC kernels of the base engine
template<class DeviceType, class Traits>
bool Grace2LFusedKK<DeviceType, Traits>::fc_forward_own(typename Base::FCLayer which, const t_nn3 &in_left,
                                                        const t_nn3 &in_right, const t_nn3 &out)
{
  if constexpr (!is_gpu_v<DeviceType>) return false;
  using L = typename Base::FCLayer;
  const FCMeta &m = which == L::A1 ? fcm_A1 : which == L::AA1 ? fcm_AA1 : which == L::AA2 ? fcm_AA2
                  : which == L::B1 ? fcm_B1 : which == L::BB1 ? fcm_BB1 : fcm_BB2;
  if (m.no == this->L1_fc_n_out) fc_fwd<G2LF_NO1>(m, in_left, in_right, out);
  else fc_fwd4<G2LF_NO2>(m, in_left, in_right, out);
  return true;
}

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
    Kokkos::parallel_for("g2lf:ShapeProd_" #P, Kokkos::RangePolicy<DeviceType, hip_lb_t<DeviceType, 256, 2>>(0, cs * nc),         \
      KOKKOS_LAMBDA(const int q) {                                                                  \
        const int k = q / cs, ii = q - k * cs;                                                      \
        NS::fwd_##P(L_, R_, O_, ii, k);                                                 \
      });                                                                                           \
  }
#define G2LF_SHAPE_REV(NS, P, Lv, Rv, Dv, dLv, dRv, NG)                                                 \
  {                                                                                                 \
    auto L_ = Lv; auto R_ = Rv; auto D_ = Dv; auto dL_ = dLv; auto dR_ = dRv;                       \
    const int cs = this->chunk_size, nc = (int) D_.extent(1), nx = (int) dL_.extent(2), ng = NG;    \
    Kokkos::parallel_for("g2lf:ShapeRevProd_" #P, Kokkos::RangePolicy<DeviceType, hip_lb_t<DeviceType, 256, 2>>(0, cs * nc),      \
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
      if (fo >= nfo || fi >= m.nfi[x]) GRACE_KK_ERROR("GRACE-2L/KK: reduce index out of range");
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
  {   // blocks of the by-output connections of at most G2LF_RED_BW input rows (n) each
    std::vector<int> bfo, bc;
    int c = 0;
    for (int fo = 0; fo < nfo; fo++) {
      int rows = 0;
      for (size_t u = 0; u < by_fo[fo].size(); u++, c++) {
        const int nin = specs[by_fo[fo][u][0]].rw->n_in;
        if (u == 0 || rows + nin > G2LF_RED_BW) { bfo.push_back(fo); bc.push_back(c); rows = 0; }
        rows += nin;
      }
      if (by_fo[fo].empty()) { bfo.push_back(fo); bc.push_back(c); }
    }
    bc.push_back(c);
    m.nb = (int) bfo.size();
    std::vector<int> wfo(nfo, 0);    // output width of each function (the n_out of the specs writing it)
    for (int fo = 0; fo < nfo; fo++)
      for (auto &e : by_fo[fo]) wfo[fo] = std::max(wfo[fo], specs[e[0]].n_out);
    m.bfo = V1<int>("g2lf:rd_bfo", m.nb); m.bc = V1<int>("g2lf:rd_bc", m.nb + 1);
    m.fob = V1<int>("g2lf:rd_fob", nfo + 1); m.boff = V1<int>("g2lf:rd_boff", m.nb + 1);
    auto hf = Kokkos::create_mirror_view(m.bfo); auto hc = Kokkos::create_mirror_view(m.bc);
    auto ho = Kokkos::create_mirror_view(m.fob); auto hw = Kokkos::create_mirror_view(m.boff);
    for (int b = 0; b < m.nb; b++) hf(b) = bfo[b];
    for (int b = 0; b <= m.nb; b++) hc(b) = bc[b];
    hw(0) = 0;
    for (int b = 0; b < m.nb; b++) hw(b + 1) = hw(b) + wfo[bfo[b]];
    m.nprow = hw(m.nb);
    for (int fo = 0, b = 0; fo <= nfo; fo++) {    // the blocks are in output order
      while (b < m.nb && bfo[b] < fo) b++;
      ho(fo) = b;
    }
    m.brho = ho(nfo1);
    Kokkos::deep_copy(m.bfo, hf); Kokkos::deep_copy(m.bc, hc); Kokkos::deep_copy(m.fob, ho); Kokkos::deep_copy(m.boff, hw);
  }
  by_g.resize(gtot);
  fill(by_g, m.roff, m.rfo, m.rtile, m.rws, "r");
}

// MODE 0: out [a, k, fo] (B);  MODE 1: fo < nfo1 -> out[a, fo, k] (I1, NO1 outputs), fo == nfo1 -> out2[a, k] (rho, NO2)
// BLK (GPU, MODE 1): thread per (atom, block of connections of one output function) writing the block's partial sums
// (the I1 / rho and I2 reductions had a thread per (atom, output function) for 5 (2L-medium) / 1 output functions,
// the rho thread of an atom running every connection: latency-bound, one wavefront per SIMD); a second kernel sums
// each output over its blocks in block order (deterministic). The partials [row][atom] go into the buffer of YI_adj
// (not in use during the I1 / rho and I2 reductions) when it is large enough, else into d_redP.
// rho_only (MODE 1): the second-kind output (rho) alone, in the same order and with the same blocks as the full call.
template<class DeviceType, class Traits>
template<int NQ, int MODE, int NO1, int NO2, bool BLK>
void Grace2LFusedKK<DeviceType, Traits>::red_fwd(const RedMeta &m, const t_nn3 &X0, const t_nn3 &X1,
                                                 const t_nn3 &X2, const t_nn3 &X3, const t_nn3 &out,
                                                 const typename Base::t_nn_2d &out2, bool rho_only)
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
  auto bfo = m.bfo; auto bc = m.bc; auto fob = m.fob; auto boff = m.boff;
  static_assert(!BLK || MODE == 1, "the blocked forward reducers are the I1 / rho and I2 reductions (MODE 1)");
  F *P = nullptr;
  if (BLK) {    // the partials: YI_adj's buffer or d_redP (begin_run)
    P = m.nprow <= red_partials_in_yi_adj() ? this->d_YI_adj.data() : d_redP.data();
    if (m.nprow > red_partials_in_yi_adj() && (size_t) d_redP.extent(0) < (size_t) m.nprow * cs)
      GRACE_KK_ERROR("GRACE-2L/KK: blocked reducer partials without their buffer");
  }
  // the blocks (BLK) or output functions run: q0 .. q0 + nq - 1
  if (rho_only && MODE != 1) GRACE_KK_ERROR("GRACE-2L/KK: rho_only reduction outside the I1 / rho reduction");
  const int q0 = rho_only ? (BLK ? m.brho : nfo1) : 0, nq = (BLK ? m.nb : nfo) - q0;
  Kokkos::parallel_for("g2lf:Reduce", Kokkos::RangePolicy<DeviceType, hip_lb_t<DeviceType, 256, 2>>(0, cs * nq), KOKKOS_LAMBDA(const int q) {
    const int bq = q / cs, ii = q - bq * cs, b = q0 + bq;
    const int fo = BLK ? bfo(b) : b;
    const int ty = elem ? type(ilist[ii + co]) : 0;
    const int nqo = (MODE == 1 && fo < nfo1) ? (NO1 + 3) / 4 : NQ;    // BLK: the quads of the output width only
    F4 acc[NQ];
    for (int k = 0; k < NQ; k++) acc[k] = F4{0.f, 0.f, 0.f, 0.f};
    const int ce = BLK ? bc(b + 1) : toff(fo + 1);
    for (int c = BLK ? bc(b) : toff(fo); c < ce; c++) {
      const int w = tws(c), fi = tfi(c), t = ttile(c);
      const int xs = (w == 0) ? x0 : (w == 1) ? x1 : (w == 2) ? x2 : (w == 3) ? x3 : (w == 4) ? x4 : (w == 5) ? x5 : (w == 6) ? x6 : x7;
      const int nin = (w == 0) ? n0 : (w == 1) ? n1 : (w == 2) ? n2 : (w == 3) ? n3 : (w == 4) ? n4 : (w == 5) ? n5 : (w == 6) ? n6 : n7;
      const auto &X = (xs == 0) ? X0 : (xs == 1) ? X1 : (xs == 2) ? X2 : X3;
      const auto &W = (w == 0) ? W0 : (w == 1) ? W1 : (w == 2) ? W2 : (w == 3) ? W3 : (w == 4) ? W4 : (w == 5) ? W5 : (w == 6) ? W6 : W7;
      for (int n = 0; n < nin; n++) {
        const F xv = X(ii, n, fi);
        const F4 *wr = reinterpret_cast<const F4 *>(&W(ty, t, n, 0));
        for (int k = 0; k < NQ; k++)
          if (!BLK || k < nqo) fma4(acc[k], wr[k], xv);    // !BLK: no test (A100: g2lf:Reduce 8 % slower with it)
      }
    }
    const int r0 = BLK ? boff(b) : 0, nw = BLK ? boff(b + 1) - r0 : 0;
    for (int k = 0; k < NQ; k++) {
      const F v[4] = {acc[k].x, acc[k].y, acc[k].z, acc[k].w};
      for (int u = 0; u < 4; u++) {
        const int kk = 4 * k + u;
        if (BLK) { if (kk < nw) P[(size_t) (r0 + kk) * cs + ii] = v[u]; }
        else if (MODE == 0) { if (kk < NO1) out(ii, kk, fo) = v[u]; }
        else if (fo < nfo1) { if (kk < NO1) out(ii, fo, kk) = v[u]; }
        else { if (kk < NO2) out2(ii, kk) = v[u]; }
      }
    }
  });
  if (!BLK) return;
  // the outputs: sum of the function's block partials in block order (a block shorter than kk, e.g. the empty block
  // of a function without connections, adds nothing)
  constexpr int NO1D = NO1 > 0 ? NO1 : 1;    // I2 (NO1 = 0) has no first-kind outputs
  const int r0 = rho_only ? nfo1 * NO1 : 0, nr = nfo1 * NO1 + (nfo > nfo1 ? NO2 : 0) - r0;
  Kokkos::parallel_for("g2lf:ReduceSum", Kokkos::RangePolicy<DeviceType>(0, cs * nr), KOKKOS_LAMBDA(const int q) {
    const int rq = q / cs, ii = q - rq * cs, r = r0 + rq;
    const bool o1 = r < nfo1 * NO1;
    const int fo = o1 ? r / NO1D : nfo1, kk = o1 ? r - fo * NO1D : r - nfo1 * NO1;
    F s = 0.0f;
    for (int b = fob(fo); b < fob(fo + 1); b++)
      if (kk < boff(b + 1) - boff(b)) s += P[(size_t) (boff(b) + kk) * cs + ii];
    if (o1) out(ii, fo, kk) = s;
    else out2(ii, kk) = s;
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
      if (e > NIN) GRACE_KK_ERROR("GRACE-2L/KK: reduce input width exceeds NIN");
    }
  auto type = this->type; auto ilist = this->d_ilist;
  // GPU: the weight / output views of a thread selected as raw pointers and strides (selecting the view objects by
  // reference put them in scratch memory: 132 - 240 bytes per lane, reloaded in the loops; A100 within -1 .. +0.5 %)
  constexpr bool gpu = is_gpu_v<DeviceType>;
  const F *wp0 = W0.data(), *wp1 = W1.data(), *wp2 = W2.data(), *wp3 = W3.data(), *wp4 = W4.data(), *wp5 = W5.data(),
          *wp6 = W6.data(), *wp7 = W7.data();
  const long a0 = W0.stride(0), a1 = W1.stride(0), a2 = W2.stride(0), a3 = W3.stride(0), a4 = W4.stride(0),
             a5 = W5.stride(0), a6 = W6.stride(0), a7 = W7.stride(0);
  const long b0 = W0.stride(1), b1 = W1.stride(1), b2 = W2.stride(1), b3 = W3.stride(1), b4 = W4.stride(1),
             b5 = W5.stride(1), b6 = W6.stride(1), b7 = W7.stride(1);
  const long e0 = W0.stride(2), e1 = W1.stride(2), e2 = W2.stride(2), e3 = W3.stride(2), e4 = W4.stride(2),
             e5 = W5.stride(2), e6 = W6.stride(2), e7 = W7.stride(2);
  F *xp0 = dX0.data(), *xp1 = dX1.data(), *xp2 = dX2.data(), *xp3 = dX3.data();
  const long xa0 = dX0.stride(0), xa1 = dX1.stride(0), xa2 = dX2.stride(0), xa3 = dX3.stride(0);
  const long xb0 = dX0.stride(1), xb1 = dX1.stride(1), xb2 = dX2.stride(1), xb3 = dX3.stride(1);
  const long xc0 = dX0.stride(2), xc1 = dX1.stride(2), xc2 = dX2.stride(2), xc3 = dX3.stride(2);
  const int xe0 = dX0.extent(1), xe1 = dX1.extent(1), xe2 = dX2.extent(1), xe3 = dX3.extent(1);
  Kokkos::parallel_for("g2lf:RevReduce", Kokkos::RangePolicy<DeviceType, hip_lb_t<DeviceType, 256, 2>>(0, cs * gtot), KOKKOS_LAMBDA(const int q) {
    const int g = q / cs, ii = q - g * cs;
    const int xs = (nx > 3 && g >= gb3) ? 3 : (nx > 2 && g >= gb2) ? 2 : (nx > 1 && g >= gb1) ? 1 : 0;
    const int fi = g - ((xs == 0) ? 0 : (xs == 1) ? gb1 : (xs == 2) ? gb2 : gb3);
    const int ty = elem ? type(ilist[ii + co]) : 0;
    F acc[NIN];
#pragma unroll
    for (int n = 0; n < NIN; n++) acc[n] = 0.0f;
    int nin = 0;
    if (gpu) {
      for (int c = roff(g); c < roff(g + 1); c++) {
        const int fo = rfo(c), t = rtile(c), w = rws(c);
        nin = (w == 0) ? n0 : (w == 1) ? n1 : (w == 2) ? n2 : (w == 3) ? n3 : (w == 4) ? n4 : (w == 5) ? n5 : (w == 6) ? n6 : n7;
        const F *Wb = (w == 0) ? wp0 + ty * a0 + t * b0 : (w == 1) ? wp1 + ty * a1 + t * b1 : (w == 2) ? wp2 + ty * a2 + t * b2 :
                      (w == 3) ? wp3 + ty * a3 + t * b3 : (w == 4) ? wp4 + ty * a4 + t * b4 : (w == 5) ? wp5 + ty * a5 + t * b5 :
                      (w == 6) ? wp6 + ty * a6 + t * b6 : wp7 + ty * a7 + t * b7;
        const long wn = (w == 0) ? e0 : (w == 1) ? e1 : (w == 2) ? e2 : (w == 3) ? e3 : (w == 4) ? e4 : (w == 5) ? e5 : (w == 6) ? e6 : e7;
        F4 gk[NQ];
#pragma unroll
        for (int k = 0; k < NQ; k++) {
          F v[4];
#pragma unroll
          for (int u = 0; u < 4; u++) {
            const int kk = 4 * k + u;
            if (MODE == 0) v[u] = (kk < NO1) ? D(ii, kk, fo) : 0.0f;
            else if (fo < nfo1) v[u] = (kk < NO1) ? D(ii, fo, kk) : 0.0f;
            else v[u] = (kk < NO2) ? D2(ii, kk) : 0.0f;
          }
          gk[k] = F4{v[0], v[1], v[2], v[3]};
        }
        // the weight quads of the output function's width only (I1: 3 of the 5 quads of rho's 17; gk past it is 0)
        const int nqo = (MODE == 1 && fo < nfo1) ? (NO1 + 3) / 4 : NQ;
#pragma unroll
        for (int n = 0; n < NIN; n++) {
          if (n >= nin) continue;
          const F4 *wr = reinterpret_cast<const F4 *>(Wb + n * wn);
          F sacc = 0.0f;
#pragma unroll
          for (int k = 0; k < NQ; k++)
            if (k < nqo) sacc += dot4(wr[k], gk[k]);
          acc[n] += sacc;
        }
      }
      F *Xb = (xs == 0) ? xp0 + ii * xa0 + fi * xc0 : (xs == 1) ? xp1 + ii * xa1 + fi * xc1 :
              (xs == 2) ? xp2 + ii * xa2 + fi * xc2 : xp3 + ii * xa3 + fi * xc3;
      const long xn = (xs == 0) ? xb0 : (xs == 1) ? xb1 : (xs == 2) ? xb2 : xb3;
      const int nx_in = (xs == 0) ? xe0 : (xs == 1) ? xe1 : (xs == 2) ? xe2 : xe3;    // <= NIN, checked on the host
      if (ASSIGN) {
#pragma unroll
        for (int n = 0; n < NIN; n++) if (n < nx_in) Xb[n * xn] = acc[n];
      } else {
#pragma unroll
        for (int n = 0; n < NIN; n++) if (n < nin) Xb[n * xn] += acc[n];
      }
      return;
    }
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
// the I1 / rho reduction (rho_only: rho alone, same blocks and order)
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::reduce_I1(bool rho_only)
{
  auto A = this->d_A; auto AA = this->d_AA; auto AAA = this->d_AAA; auto AAAA = this->d_AAAA;
  if (red_blk_on<DeviceType>(rm_I1))
    red_fwd<5, 1, 12, 17, true>(rm_I1, A, AA, AAA, AAAA, this->d_I1, this->d_rho, rho_only);
  else red_fwd<5, 1, 12, 17>(rm_I1, A, AA, AAA, AAAA, this->d_I1, this->d_rho, rho_only);
}
template<class DeviceType, class Traits>
void Grace2LFusedKK<DeviceType, Traits>::stage_reduce_I1() { reduce_I1(false); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_reduce_I1()
{ red_bwd<5, G2LF_NO1, 1, 12, 17, true>(rm_I1, this->d_I1_adj, this->d_rho_adj, this->d_A_adj, this->d_AA_adj, this->d_AAA_adj, this->d_AAAA_adj); }
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_reduce_rho() {}        // done in stage_reduce_I1
// rho of stage_reduce_I1 alone (the base kernel sums in another fp32 order: forces about 1e-6 relative off the energy
// gradient in multi-chunk runs without the layer-1 store)
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_recompute_rho()
{
  reduce_I1(true);
}
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_reduce_rho() {}    // done in stage_rev_reduce_I1
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_zero_L1_adj() {}       // the I1+rho reverse assigns
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_zero_L2_adj() {}       // I2 and B reverses assign
// I2 <- B, BB, BBB, BBBB (scalar outputs, element-dependent): every connection to the single output fo = 0
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_reduce_I2()
{
  if (red_blk_on<DeviceType>(rm_I2)) red_fwd<5, 1, 0, 17, true>(rm_I2, this->d_B, this->d_BB, this->d_BBB, this->d_BBBB, this->d_B, this->d_I2);
  else red_fwd<5, 1, 0, 17>(rm_I2, this->d_B, this->d_BB, this->d_BBB, this->d_BBBB, this->d_B, this->d_I2);
}
template<class DeviceType, class Traits> void Grace2LFusedKK<DeviceType, Traits>::stage_rev_reduce_I2()
{ red_bwd<5, G2LF_NO2, 1, 0, 17, true>(rm_I2, this->d_B_adj, this->d_I2_adj, this->d_B_adj, this->d_BB_adj, this->d_BBB_adj, this->d_BBBB_adj); }

}    // namespace kk
}    // namespace grace

// the explicit instantiation of GRACE_KK_EXTERN_2L (grace/kk/grace_2l_fused_kk.h), in one translation unit
#define GRACE_KK_INSTANTIATE_2L(Device, Traits)                        \
  template class grace::kk::Grace2LKK<Device, float, float, Traits>; \
  template class grace::kk::Grace2LFusedKK<Device, Traits>

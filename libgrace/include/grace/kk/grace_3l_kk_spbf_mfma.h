// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Part of grace::kk::Grace3LKK (grace_3l_kk.h), included by grace_3l_kk_spbf.h: the equivariant SPBF forward and
// backward (A2 / A3, lmax 4, 25 or 50 indicator components) on the fp32 matrix cores of CDNA (gfx90a, gfx94x;
// v_mfma_f32_16x16x4f32: full fp32 products, fp32 accumulation), where spbf_fast_ok() holds; the other builds (CUDA,
// RDNA, host) do not compile it. Default on for gfx90a (MI250X: 1.24-1.32x faster GRACE-3L-OMAT-large), opt-in for
// gfx94x (MI300) until it has run there; the environment variable GRACE3L_SPBF_MFMA (0 / 1) overrides the default,
// see spbf_use_mfma().
//
// Per (atom ii, channel n), with j the neighbours, y = 0..24 the Y components, r the indicator components:
//   a[j, y] = R(j, n, l(y)) env(j) Y_y(j),   b[j, r] = I(j_global, n, r) + z_proj(mu_j, n) mask(r)
//   forward:   Q = a^T b            (25 x NLI, contraction over the bonds)
//   backward:  G = a M              (bonds x NLI: dE/dI rows, atomics into the target)
//              V = b M^T            (bonds x 25: the per-bond force through d a / d r_ij)
// with M[y, r] = inv_avg sum_f cg[(y, r), f] dE/dA[n, f] (sparse CG, as in spbf_bwd_fast2_t).
// Team = one atom x 4 channels = 4 wavefronts. 16x16x4 MFMA operand layout: A[i][k] in lane i + 16 k, B[k][j] in
// lane j + 16 k, D[i][j] in register i % 4 of lane j + 16 (i / 4); the lane group g = lane / 16 is the k index.
//   forward:  wavefront per channel, rounds of 16 bonds, step s: k = g <-> bond jb + 4 g + s; A = a^T (i = y),
//             B = b (j = r); the round's bond data, Y rows and radial table rows (a bond's 4 channels: one contiguous
//             320-byte block) loaded once per team with consecutive threads and shared.
//   backward: wavefront per round of 16 bonds (rounds split over the 4 wavefronts), the team's 4 channels in turn
//             (bond data and Y / dY loaded once for them); the lane's bond is jb + lane % 16 throughout:
//             G:   k = g <-> y in Yset(g) (8 slots), A = a (i = bond), B = M (j = r) from shared;
//             V^T: k = g <-> r = 16 (s / 4) + 4 g + s % 4 at step s (b of a lane group: 16-byte loads of 4 consecutive
//                  r), A = M (rows permuted so that output register q of tile t is y = Yset(g)[4 t + q]), B = b^T;
//             so a lane holds a, b, Y, dY and v of its bond's Yset(g) only, the force partial needs the sum over the
//             lane groups (two shuffles) and, summed over the 4 channels in registers, one atomic per bond and team.
//             A2 stages the round's R, dR of the 4 channels in the dE/dA staging area (free after the M build), like
//             the forward's table rows.
//             Yset groups the y by l (radial of at most two l per lane): g0 16..23 (l 4), g1 9..15 (l 3), 24 (l 4),
//             g2 1..3 (l 1), 4..8 (l 2), g3 0 (l 0) + 7 empty slots.
#pragma once

#include "grace/kk/grace_3l_kk.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <vector>

#if defined(GRACE_KK_GEMM_MFMA) && GRACE_KK_GEMM_MFMA
#define GRACE3L_SPBF_MFMA 1
#else
#define GRACE3L_SPBF_MFMA 0
#endif

namespace grace {
namespace kk {

#if GRACE3L_SPBF_MFMA
// 4 / 2 floats at 4-byte alignment (b rows of 25 / 50 floats; the CDNA global loads take dword-aligned vectors)
typedef float g3l_f4u __attribute__((ext_vector_type(4), aligned(4)));
typedef float g3l_f2u __attribute__((ext_vector_type(2), aligned(4)));
// (offset, width) of the slices of a sliced-ELL CG table, passed by value (kernel argument: scalar loads)
constexpr int G3L_SELL_MAXSL = 32;
struct G3SellSlices { int ow[2 * G3L_SELL_MAXSL]; };
// the MFMA kernels' limits: NFT = 3 dE/dA columns per thread (backward), G3L_SELL_MAXSL slices of the CG tables
inline bool g3l_spbf_mfma_fits(int nfunc, int nli) { return nfunc <= 3 * 256 && 25 * nli <= G3L_SELL_MAXSL * 64; }
struct alignas(8) G3F2 { float x, y; };
template<class T> struct alignas(4 * sizeof(T)) G3V4 { T x, y, z, w; };

// Yset(g)[s] of the backward (-1: empty slot); slots s < sw have l = la, the others l = lb
KOKKOS_INLINE_FUNCTION int g3l_yset(int g, int s)
{
  return g == 0 ? 16 + s : g == 1 ? (s < 7 ? 9 + s : 24) : g == 2 ? 1 + s : (s == 0 ? 0 : -1);
}
KOKKOS_INLINE_FUNCTION void g3l_yset_l(int g, int &la, int &lb, int &sw)
{
  la = g == 0 ? 4 : g == 1 ? 3 : g == 2 ? 1 : 0;
  lb = g == 0 ? 4 : g == 1 ? 4 : g == 2 ? 2 : 0;
  sw = g == 1 ? 7 : g == 2 ? 3 : 8;
}
#endif

#if defined(KOKKOS_ENABLE_HIP)
// The environment variable GRACE3L_SPBF_MFMA, read once per process (env_switch, support.h): 1 (MFMA kernels where
// compiled), 0 (the vector-ALU kernels), -1 when unset or empty (the build's default).
inline int g3l_spbf_mfma_env()
{
  static const int v = env_switch("GRACE3L_SPBF_MFMA");
  return v;
}
// one line on stderr per process (not per engine type): which SPBF kernels and why
inline void g3l_spbf_mfma_log(bool on, const char *why)
{
  static std::once_flag logged;
  std::call_once(logged, [&] {
    std::fprintf(stderr, "grace kk: GRACE-3L SPBF on the %s kernels (%s)\n", on ? "matrix-core (MFMA)" : "vector-ALU",
                 why);
  });
}
#endif

// Whether the SPBF of idx runs on the MFMA kernels: HIP engines of a CDNA build, GRACE3L_SPBF_MFMA=1 or the build's
// default (gfx90a) unless GRACE3L_SPBF_MFMA=0, and a model inside the kernels' limits. The first call in a process
// says on stderr which kernels are used and why (g3l_spbf_mfma_log). CUDA and host builds: false, silently.
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
bool Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_use_mfma(int idx) const
{
#if defined(KOKKOS_ENABLE_HIP)
  if (!is_hip_v<DeviceType>) return false;
#if GRACE3L_SPBF_MFMA
  const bool fits = g3l_spbf_mfma_fits(d_spbf[idx].nfunc, d_spbf[idx].n_lm_ind);
#else
  (void) idx;
  const bool fits = false;
#endif
  const KernelChoice s = mfma_switch("GRACE3L_SPBF_MFMA", g3l_spbf_mfma_env(), fits);
  g3l_spbf_mfma_log(s.on, s.why.c_str());
  return s.on;
#else
  (void) idx;
  return false;
#endif
}

#if GRACE3L_SPBF_MFMA
// The CG table as sliced ELL (rows of the CSR for the backward M build, columns of the CSC for the forward): the rows /
// columns sorted by length (longest first, ties in index order), slices of 64 of them (one per wavefront lane), each
// slice as wide as its longest member, entries [offset + e * 64 + lane] = (index, fp32 value bits) in the CSR / CSC
// order (ascending index), padded with (0, 0). Coalesced loads, and about 3.3 / 5.5 entries per row / column instead
// of the 6 of the padded ELL (GRACE-3L-OMAT-large).
template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_mfma_tables(int idx)
{
  const DeviceSPBF &sp = d_spbf[idx];
  auto build = [](SpbfSell &t, const t_int_1d &ptr, const t_int_1d &ix, const t_nn_1d &val, const char *nm) {
    if (t.src == ptr.data() && t.perm.extent(0) > 0) return;
    auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ptr);
    auto hi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ix);
    auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), val);
    const int n = (int) hp.extent(0) - 1, ns = (n + 63) / 64;
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return hp(a + 1) - hp(a) > hp(b + 1) - hp(b); });
    std::vector<int> perm(64 * ns, -1), len(64 * ns, 0), slc(2 * ns), ent;
    for (int k = 0; k < ns; k++) {
      int w = 0;
      for (int l = 0; l < 64 && 64 * k + l < n; l++) w = std::max(w, hp(ord[64 * k + l] + 1) - hp(ord[64 * k + l]));
      slc[2 * k] = (int) ent.size() / 2;
      slc[2 * k + 1] = w;
      const size_t o = ent.size();
      ent.resize(o + 2 * 64 * (size_t) w, 0);
      for (int l = 0; l < 64 && 64 * k + l < n; l++) {
        const int r = ord[64 * k + l], e0 = hp(r);
        perm[64 * k + l] = r;
        len[64 * k + l] = hp(r + 1) - e0;
        for (int e = 0; e < len[64 * k + l]; e++) {
          const float fv = (float) hv(e0 + e);
          ent[o + 2 * (e * 64 + l)] = hi(e0 + e);
          std::memcpy(&ent[o + 2 * (e * 64 + l) + 1], &fv, sizeof(float));
        }
      }
    }
    auto up = [&](const std::vector<int> &h, t_int_1d &d, const std::string &lab) {
      d = t_int_1d(Kokkos::view_alloc(Kokkos::WithoutInitializing, lab), std::max<size_t>(h.size(), 1));
      auto m = Kokkos::create_mirror_view(d);
      for (size_t q = 0; q < h.size(); q++) m(q) = h[q];
      Kokkos::deep_copy(d, m);
    };
    up(perm, t.perm, std::string(nm) + "_perm"); up(len, t.len, std::string(nm) + "_len");
    up(ent, t.ent, std::string(nm) + "_ent");
    if (ns > G3L_SELL_MAXSL) GRACE_KK_ERROR("GRACE-3L/KK: spbf_mfma_tables: {} slices > {}", ns, G3L_SELL_MAXSL);
    t.slc = slc;
    t.src = ptr.data();
  };
  build(d_spbf_rsell[idx], sp.cg_row_ptr, sp.cg_col_idx, sp.cg_row_val, "g3l:spbf_rsell");
  build(d_spbf_csell[idx], sp.cg_col_ptr, sp.cg_row_idx, sp.cg_col_val, "g3l:spbf_csell");
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
template<int NLI>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_fwd_mfma_t(
    int idx, const t_nn_3d_rg &indicator, const t_nn_3d &out, bool rm)
{
  // waves per SIMD of the register budget: A3 Q fills 20 KB of shared (2 teams per CU), A2 10 KB
  // NL: the l of lmax 4 (spbf_fast_ok)
  constexpr int CB = 4, NY = 25, NL = 5, TEAM = 256, TR = (NLI + 15) / 16, QSZ = CB * NY * NLI, MW = NLI > 32 ? 2 : 4;
  const DeviceSPBF &sp = d_spbf[idx];
  const int nblk = sp.n_rad_max / CB, nfunc = sp.nfunc, NR = sp.n_rad_max;
  auto zproj = sp.z_proj; auto mask = sp.chem_l0_mask;
  spbf_mfma_tables(idx);
  auto cperm = d_spbf_csell[idx].perm; auto clen = d_spbf_csell[idx].len;
  G3SellSlices cslc{};
  std::copy(d_spbf_csell[idx].slc.begin(), d_spbf_csell[idx].slc.end(), cslc.ow);
  const G3I2 *cent = reinterpret_cast<const G3I2 *>(d_spbf_csell[idx].ent.data());
  const int nslot = (int) cperm.extent(0);
  const float inv = (float) sp.inv_avg_n_neigh;
  auto R = d_R1_nl; auto env = d_env; auto ncount = d_ncount; auto nearest = d_nearest; auto muj = d_mu_j;
  auto AT = d_spbf_T;
  if (d_Y_bond.stride(1) != NY || indicator.stride(2) != 1)
    GRACE_KK_ERROR("GRACE-3L/KK: spbf_fwd_mfma_t: unexpected Y / indicator layout");
  const GeomScalar *Yp = d_Y_bond.data();
  const long ysi = d_Y_bond.stride(0), isg = indicator.stride(0), isn = indicator.stride(1);
  const float *indp = indicator.data();
  // fuse_radial: R of this layer from the cubic tables, not from d_R1_nl
  const bool tabr = fuse_radial && spline_on;
  const int tnsp = n_sp, tlm = lmax + 1;
  const G3LTab tab = spl_tab(idx);
  auto spm = d_sp_of_mu; auto bc = d_bond_cutoff; auto mui = d_mu_i; auto rn = d_rnorms;

  using team_policy = Kokkos::TeamPolicy<DeviceType, hip_lb_t<DeviceType, TEAM, MW>>;
  using member = typename team_policy::member_type;
  constexpr int NW = TEAM / 64;
  static_assert(CB == NW, "SPBFFwdM: wavefront per channel of the block");
  static_assert(16 * CB * NL > TEAM && 16 * CB * NL - TEAM + 16 * CB <= TEAM,
                "SPBFFwdM: z loaded by threads idle in the last pass of the table staging");
  // PAIR: b of the lane as 8-byte pairs, tile tr column li <-> r = 2 li + tr % 2 + 32 (tr / 2) (even NLI: rows 8-byte
  // aligned, checked below); otherwise r = li + 16 tr
  constexpr bool PAIR = NLI % 2 == 0;
  if (PAIR && (isg % 2 != 0 || isn % 2 != 0 || reinterpret_cast<std::uintptr_t>(indp) % 8 != 0))
    GRACE_KK_ERROR("GRACE-3L/KK: spbf_fwd_mfma_t: indicator rows not 8-byte aligned");
  // shared: Q of the 4 channels; the round's bond data and Y rows (16 bonds), staged once for the 4 wavefronts;
  // per wavefront z and R (5 l) of its channel
  const size_t shb = (size_t) QSZ * sizeof(float) + 16 * (sizeof(long) + 2 * sizeof(int) + 2 * sizeof(float)) +
                     16 * NY * sizeof(float) + NW * 16 * 6 * sizeof(float) + 64;
  team_policy pol(chunk_size * nblk, TEAM);
  pol = pol.set_scratch_size(0, Kokkos::PerTeam(shb));
  Kokkos::parallel_for("SPBFFwdM", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank() / nblk, c0 = (t.league_rank() - ii * nblk) * CB;
    float *Qs = (float *) t.team_shmem().get_shmem_aligned((size_t) QSZ * sizeof(float), 16);
    long *tbS = (long *) t.team_shmem().get_shmem_aligned(16 * sizeof(long), 8);
    int *jgS = (int *) t.team_shmem().get_shmem(16 * sizeof(int));
    int *mjS = (int *) t.team_shmem().get_shmem(16 * sizeof(int));
    float *evS = (float *) t.team_shmem().get_shmem(16 * sizeof(float));
    float *ttS = (float *) t.team_shmem().get_shmem(16 * sizeof(float));
    float *ysS = (float *) t.team_shmem().get_shmem(16 * NY * sizeof(float));
    float *zS = (float *) t.team_shmem().get_shmem(NW * 16 * sizeof(float));
    float *reS = (float *) t.team_shmem().get_shmem(NW * 16 * NL * sizeof(float));
    const int tid = t.team_rank(), w = tid >> 6, lane = tid & 63, g = lane >> 4, li = lane & 15, n = c0 + w;
    const int ly0 = l_of_y(li);           // l of y = li (A rows of tile 0); tile 1 (y = li + 16) is l = 4
    const bool y1ok = li < NY - 16;
    float mk[TR];
#pragma unroll
    for (int tr = 0; tr < TR; tr++) {
      const int r = PAIR ? 2 * li + (tr & 1) + 32 * (tr >> 1) : li + 16 * tr;
      mk[tr] = r < NLI ? (float) mask(r) : 0.0f;
    }
    mfma_f4v acc[2][TR];
#pragma unroll
    for (int ty = 0; ty < 2; ty++)
#pragma unroll
      for (int tr = 0; tr < TR; tr++) acc[ty][tr] = mfma_f4v{0.0f, 0.0f, 0.0f, 0.0f};
    const int nct = ncount(ii), mi = mui(ii);
    const GeomScalar *Yr = Yp + ii * ysi;
    float *zw = zS + 16 * w, *rew = reS + 16 * NL * w;
    for (int jb = 0; jb < nct; jb += 16) {
      t.team_barrier();    // the previous round's staging consumed
      // bond jb + bt (the last 16 threads): neighbour, species, env (0 past the count: a exact zeros), table row
      // offset of channel c0, t
      if (tid >= TEAM - 16) {
        const int bt = tid - (TEAM - 16), j = jb + bt;
        const bool valid = j < nct;
        const int jq = valid ? j : nct - 1;
        const int mj = muj(ii, jq);
        jgS[bt] = nearest(ii, jq);
        mjS[bt] = mj;
        evS[bt] = valid ? (float) env(ii, jq) : 0.0f;
        if (tabr) {
          const float rc = bc(mi, mj);
          const long r0 = tab.row(spm(mi) * tnsp + spm(mj), (float) rn(ii, jq), rc, ttS[bt]);
          tbS[bt] = (r0 + (long) c0 * tlm) * 4;
        }
      }
      // Y rows of the round's bonds, coalesced (past the count: the last bond's row)
      for (int q = tid; q < 16 * NY; q += TEAM) {
        const int b = q / NY, jq = jb + b < nct ? jb + b : nct - 1;
        ysS[q] = (float) Yr[jq * NY + q - b * NY];
      }
      t.team_barrier();
      if (tabr) {
        // R of the round's bonds for the 4 channels and 5 l: a bond's 4 cubic rows are contiguous (320 bytes),
        // consecutive threads load them; R of channel c0 + c to the R of wavefront c (CB == NW)
        for (int q = tid; q < 16 * CB * NL; q += TEAM) {
          const int b = q / (CB * NL), e = q - b * (CB * NL), c = e / NL;
          const G3F4 cf = *reinterpret_cast<const G3F4 *>(tab.c + tbS[b] + 4 * e);
          reS[(c * 16 + b) * NL + e - NL * c] = g3l_cubic(cf, ttS[b]);
        }
        // z of the 4 channels per bond, by threads idle in the loop's last pass
        if (tid >= 16 * CB * NL - TEAM && tid < 16 * CB * NL - TEAM + 16 * CB) {
          const int q = tid - (16 * CB * NL - TEAM);
          zS[q] = zproj(mjS[q & 15], c0 + (q >> 4));
        }
        t.team_barrier();
      } else {
        if (g == 0) zw[li] = zproj(mjS[li], n);
        // R of bond li for l = g (and l = 4 in group 0): once per (bond, l)
        const int jq = jb + li < nct ? jb + li : nct - 1;
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int l = h == 0 ? g : 4;
          if (h == 0 || g == 0) rew[li * NL + l] = R(ii, jq, n, l);
        }
        g3l_syncwarp();
      }
      // the lane group's 4 bonds of the round (k index of step s): a for y = li, li + 16; b for the r of the tiles
      float av[4][2], bv[4][TR];
#pragma unroll
      for (int s = 0; s < 4; s++) {
        const int b = 4 * g + s;
        const float ev = evS[b], z = zw[b];
        av[s][0] = rew[b * NL + ly0] * ysS[b * NY + li] * ev;
        av[s][1] = y1ok ? rew[b * NL + 4] * ysS[b * NY + li + 16] * ev : 0.0f;
        const float *ib = indp + (long) jgS[b] * isg + n * isn;
        if (PAIR) {
#pragma unroll
          for (int h = 0; h < TR / 2; h++) {
            const int r = 2 * li + 32 * h;
            G3F2 p{0.0f, 0.0f};
            if (r < NLI) p = *reinterpret_cast<const G3F2 *>(ib + r);
            bv[s][2 * h] = r < NLI ? p.x + z * mk[2 * h] : 0.0f;
            bv[s][2 * h + 1] = r < NLI ? p.y + z * mk[2 * h + 1] : 0.0f;
          }
        } else {
#pragma unroll
          for (int tr = 0; tr < TR; tr++) {
            const int r = li + 16 * tr;
            bv[s][tr] = (r < NLI) ? ib[r] + z * mk[tr] : 0.0f;
          }
        }
      }
#pragma unroll
      for (int s = 0; s < 4; s++)
#pragma unroll
        for (int ty = 0; ty < 2; ty++)
#pragma unroll
          for (int tr = 0; tr < TR; tr++) mfma_16x16x4(av[s][ty], bv[s][tr], acc[ty][tr]);
    }
    // Q[y][r] of the channel to shared: register q of tile (ty, tr) is y = 16 ty + 4 g + q, r of column li of tr
    float *Qw = Qs + w * NY * NLI;
#pragma unroll
    for (int ty = 0; ty < 2; ty++)
#pragma unroll
      for (int tr = 0; tr < TR; tr++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const int y = 16 * ty + 4 * g + q, r = PAIR ? 2 * li + (tr & 1) + 32 * (tr >> 1) : li + 16 * tr;
          if (y < NY && r < NLI) Qw[y * NLI + r] = acc[ty][tr][q];
        }
    t.team_barrier();
    // CG step: thread per output function f (column slot q of the sliced ELL) for the 4 channels, the CSC entries
    // of f in order (as spbf_fwd_fast_t)
    float *Ao = &AT(ii, c0);
    for (int q = tid; q < nslot; q += TEAM) {
      const int f = cperm(q), len = clen(q), k = q >> 6, w = cslc.ow[2 * k + 1];
      const G3I2 *ce = cent + cslc.ow[2 * k] + (q & 63);
      float s[CB];
#pragma unroll
      for (int cc = 0; cc < CB; cc++) s[cc] = 0.0f;
      for (int e = 0; e < w; e += 4) {
        G3I2 en[4];
#pragma unroll
        for (int u = 0; u < 4; u++) en[u] = (e + u < w) ? ce[(e + u) * 64] : G3I2{0, 0.0f};
#pragma unroll
        for (int u = 0; u < 4; u++)
          if (e + u < len) {
#pragma unroll
            for (int cc = 0; cc < CB; cc++) s[cc] += Qs[cc * NY * NLI + en[u].i] * en[u].v;
          }
      }
      if (f >= 0) *reinterpret_cast<G3F4 *>(Ao + (long) f * NR) = G3F4{s[0] * inv, s[1] * inv, s[2] * inv, s[3] * inv};
    }
  });
  if (!rm) spbf_transpose(out, NR, nfunc, false);
}

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
template<int NLI>
void Grace3LKK<DeviceType, NNScalarT, GeomScalarT, Traits>::spbf_bwd_mfma_t(
    int idx, const t_nn_3d &A_adj, const t_nn_3d_rg &indicator, const t_nn_3d_rg &target, bool rm)
{
  constexpr int CB = 4, NY = 25, TEAM = 256, NW = TEAM / 64, TR = (NLI + 15) / 16, NP = NY * NLI;
  // b: lane group g, k-step s <-> r = 16 (s / 4) + 4 g + s % 4 (16-byte loads); steps with r >= NLI for every g dropped
  constexpr int KS = 4 * (NLI / 16) + (NLI % 16 < 4 ? NLI % 16 : 4), EW = GRACE3L_SPBF_ELL_MAX;
  constexpr int MW = 2;    // 170 - 190 registers (3 waves per SIMD spilled for A2)
  // dE/dA of CPP channels at a time staged in shared for the M build (A3: two passes, so that M + staging stay
  // within the 2 teams per CU that M alone allows); the thread's rows: slots tid + TEAM u of the sliced ELL
  constexpr int CPP = NLI > 32 ? 2 : 4, NPT = (NP + TEAM - 1) / TEAM, NFT = 3;
  const DeviceSPBF &sp = d_spbf[idx];
  const int nblk = sp.n_rad_max / CB, NR = sp.n_rad_max, nfunc = sp.nfunc;
  auto zproj = sp.z_proj; auto mask = sp.chem_l0_mask;
  spbf_mfma_tables(idx);
  auto rperm = d_spbf_rsell[idx].perm;
  G3SellSlices rslc{};
  std::copy(d_spbf_rsell[idx].slc.begin(), d_spbf_rsell[idx].slc.end(), rslc.ow);
  const G3I2 *rent = reinterpret_cast<const G3I2 *>(d_spbf_rsell[idx].ent.data());
  const int nslot = (int) rperm.extent(0);
  if (nslot < NP || nslot > NPT * TEAM || nfunc > NFT * TEAM)
    GRACE_KK_ERROR("GRACE-3L/KK: spbf_bwd_mfma_t: unexpected CG table");
  const float inv = (float) sp.inv_avg_n_neigh;
  auto R = d_R1_nl; auto DR = d_DR1_nl; auto env = d_env; auto denv = d_denv;
  auto rn = d_rnorms; auto rh = d_rhats; auto ncount = d_ncount; auto nearest = d_nearest; auto muj = d_mu_j;
  auto fij = d_f_ij;
  const bool tabr = fuse_radial && spline_on;
  const int tnsp = n_sp, tlm = lmax + 1;
  const G3LTab tab = spl_tab(idx);
  auto spm = d_sp_of_mu; auto bc = d_bond_cutoff; auto mui = d_mu_i;
  if (d_Y_bond.stride(1) != NY || d_dY_bond.stride(1) != 3 * NY || d_dY_bond.stride(2) != 3 ||
      indicator.stride(2) != 1 || target.stride(2) != 1)
    GRACE_KK_ERROR("GRACE-3L/KK: spbf_bwd_mfma_t: unexpected Y / dY / indicator layout");
  if (!rm) spbf_transpose(A_adj, NR, nfunc, true);
  auto AT = d_spbf_T;
  // (Y, dY) of the chunk as 16-byte (fp32) records [atom][slot k][bond], slot k = 8 g + s of Yset(g)[s] (k = 24:
  // y = 0): one load per (bond, slot) below, the 16 bonds of a lane group on consecutive records
  const int cs = chunk_size, mn = (int) d_Y_bond.extent(1);
  if ((int) d_spbf_ydy.extent(0) < cs || (int) d_spbf_ydy.extent(1) != NY || (int) d_spbf_ydy.extent(2) != 4 * mn)
    d_spbf_ydy = Kokkos::View<GeomScalar ***, Kokkos::LayoutRight, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "grace3l:spbf_ydy"), cs, NY, 4 * mn);
  {
    auto Y = d_Y_bond; auto dY = d_dY_bond; auto P = d_spbf_ydy;
    Kokkos::parallel_for("SPBFPackYdY", Kokkos::RangePolicy<DeviceType>(0, (long) cs * mn * NY), KOKKOS_LAMBDA(const long q) {
      const int i = (int) (q / ((long) mn * NY)), rest = (int) (q - (long) i * mn * NY), k = rest / mn, j = rest - k * mn;
      const int y = g3l_yset(k >> 3, k & 7);
      *reinterpret_cast<G3V4<GeomScalar> *>(&P(i, k, 4 * j)) = G3V4<GeomScalar>{Y(i, j, y), dY(i, j, y, 0), dY(i, j, y, 1), dY(i, j, y, 2)};
    });
  }
  const GeomScalar *Pp = d_spbf_ydy.data();
  const long psi = d_spbf_ydy.stride(0);
  const float *indp = indicator.data();
  float *tgtp = target.data();
  const long isg = indicator.stride(0), isn = indicator.stride(1), tsg = target.stride(0), tsn = target.stride(1);

  using team_policy = Kokkos::TeamPolicy<DeviceType, hip_lb_t<DeviceType, TEAM, MW>>;
  using member = typename team_policy::member_type;
  team_policy pol(chunk_size * nblk, TEAM);
  // TST (A2): after the M build the staging area holds per wavefront the round's R, dR (16 bonds x 4 channels x 5 l)
  // and the bonds' table offset, t and d t / d r; not for A3, whose shared (with the HIP team scratch) would no longer
  // fit 2 teams per CU
  constexpr bool TST = NLI <= 32;
  constexpr int RWS = 16 * CB * 5 * 2 + 4 * 16;
  const int stn = TST && tabr && CPP * nfunc < NW * RWS ? NW * RWS : CPP * nfunc;
  pol = pol.set_scratch_size(0, Kokkos::PerTeam((size_t) (CB * NP + stn) * sizeof(float) + 64));
  Kokkos::parallel_for("SPBFBwdM", pol, KOKKOS_LAMBDA(const member &t) {
    const int ii = t.league_rank() / nblk, c0 = (t.league_rank() - ii * nblk) * CB;
    float *Ms = (float *) t.team_shmem().get_shmem_aligned((size_t) CB * NP * sizeof(float), 16);
    float *St = (float *) t.team_shmem().get_shmem_aligned((size_t) stn * sizeof(float), 16);
    const int tid = t.team_rank(), w = tid >> 6, lane = tid & 63, g = lane >> 4, li = lane & 15;

    // M[c][p = y NLI + r] = inv sum_{e in row p} cg_e dE/dA[ii, c0 + c, f_e] for the 4 channels (CSR rows in order,
    // as spbf_bwd_fast2_t; padding entries add exact zeros); the thread's entries loaded once, dE/dA of CPP
    // channels per pass from shared (staged with one 8- / 16-byte load per f)
    {
      G3I2 en[NPT][EW];
      int wq[NPT], pq[NPT];
#pragma unroll
      for (int u = 0; u < NPT; u++) {
        const int q = tid + u * TEAM, k = q >> 6, w = q < nslot ? rslc.ow[2 * k + 1] : 0;
        wq[u] = w;
        pq[u] = q < nslot ? rperm(q) : -1;
        const G3I2 *re = rent + (q < nslot ? rslc.ow[2 * k] : 0) + (q & 63);
#pragma unroll
        for (int e = 0; e < EW; e++) en[u][e] = e < w ? re[e * 64] : G3I2{0, 0.0f};
      }
      // dE/dA of the thread's f (at most NFT per thread) for all passes loaded up front: one global round trip
      const float *Ac = &AT(ii, c0);
      G3F4 dA[NFT];
#pragma unroll
      for (int v = 0; v < NFT; v++) {
        const int f = tid + v * TEAM;
        dA[v] = f < nfunc ? *reinterpret_cast<const G3F4 *>(Ac + (long) f * NR) : G3F4{0.0f, 0.0f, 0.0f, 0.0f};
      }
#pragma unroll
      for (int cp = 0; cp < CB / CPP; cp++) {
        if (cp > 0) t.team_barrier();
#pragma unroll
        for (int v = 0; v < NFT; v++) {
          const int f = tid + v * TEAM;
          if (f < nfunc) {
            if (CPP == 4) *reinterpret_cast<G3F4 *>(St + 4 * f) = dA[v];
            else *reinterpret_cast<G3F2 *>(St + 2 * f) = cp == 0 ? G3F2{dA[v].x, dA[v].y} : G3F2{dA[v].z, dA[v].w};
          }
        }
        t.team_barrier();
#pragma unroll
        for (int u = 0; u < NPT; u++) {
          const int w = wq[u], p = pq[u];
          if (p >= 0) {
            float sc[CPP];
#pragma unroll
            for (int c = 0; c < CPP; c++) sc[c] = 0.0f;
#pragma unroll
            for (int e = 0; e < EW; e++)
              if (e < w) {
                const float v = en[u][e].v;
                if (CPP == 4) {
                  const G3F4 a4 = *reinterpret_cast<const G3F4 *>(St + 4 * en[u][e].i);
                  sc[0] += a4.x * v; sc[1] += a4.y * v; sc[2 % CPP] += a4.z * v; sc[3 % CPP] += a4.w * v;
                } else {
                  const G3F2 a2 = *reinterpret_cast<const G3F2 *>(St + 2 * en[u][e].i);
                  sc[0] += a2.x * v; sc[1] += a2.y * v;
                }
              }
#pragma unroll
            for (int c = 0; c < CPP; c++) Ms[(cp * CPP + c) * NP + p] = sc[c] * inv;
          }
        }
      }
    }
    t.team_barrier();

    // lane constants: Yset(g) and its l, the M offsets of the G operand (row y_s, column li + 16 tr), of the
    // V^T operand (rows y of the output slots of lane li, columns r of the lane group) and the mask of those r
    int la, lb, sw;
    g3l_yset_l(g, la, lb, sw);
    int aoff[2];
#pragma unroll
    for (int ty = 0; ty < 2; ty++) {
      const int y = g3l_yset(li >> 2, 4 * ty + (li & 3));
      aoff[ty] = (y < 0 ? 0 : y) * NLI;
    }
    float mk[KS];
#pragma unroll
    for (int s = 0; s < KS; s++) {
      const int r = 16 * (s >> 2) + 4 * g + (s & 3);
      mk[s] = r < NLI ? (float) mask(r) : 0.0f;
    }

    const int nct = ncount(ii), mi = mui(ii);
    const GeomScalar *Pr = Pp + ii * psi;
    for (int jb = 16 * w; jb < nct; jb += 16 * NW) {
      // the lane's bond jb + li (past the count: the last one with env 0 -> a, R env, dR env exact zeros)
      const int j = jb + li;
      const bool valid = j < nct;
      const int jq = valid ? j : nct - 1;
      const int jg = nearest(ii, jq), mj = muj(ii, jq);
      const GeomScalar ev = valid ? (GeomScalar) env(ii, jq) : GeomScalar(0.0);
      const GeomScalar dev = valid ? (GeomScalar) denv(ii, jq) : GeomScalar(0.0);
      long tb = 0;
      float tt = 0.0f, dxs = 0.0f;
      if (tabr) {
        const float rc = bc(mi, mj);
        dxs = tab.dsdr(rc);
        tb = tab.row(spm(mi) * tnsp + spm(mj), (float) rn(ii, jq), rc, tt) * 4;
      }
      // TST: R, dR of the round's bonds for the 4 channels and 5 l; a bond's 4 cubic rows are contiguous (320 bytes),
      // consecutive lanes load them
      float *Rw = St + w * RWS;
      if (TST && tabr) {
        long *tbw = reinterpret_cast<long *>(Rw + 16 * CB * 5 * 2);
        float *ttw = Rw + 16 * CB * 5 * 2 + 32, *dxw = ttw + 16;
        g3l_syncwarp();
        if (g == 0) { tbw[li] = tb + (long) c0 * tlm * 4; ttw[li] = tt; dxw[li] = dxs; }
        g3l_syncwarp();
#pragma unroll
        for (int u = 0; u < 5; u++) {
          const int q = lane + 64 * u, b = q / 20, e = q - 20 * b;
          const G3F4 c = *reinterpret_cast<const G3F4 *>(tab.c + tbw[b] + 4 * e);
          const float tq = ttw[b];
          *reinterpret_cast<G3F2 *>(Rw + 2 * q) = G3F2{g3l_cubic(c, tq), g3l_cubic_dt(c, tq) * dxw[b]};
        }
        g3l_syncwarp();
      }
      // Y, dY of the lane's Yset(g) (empty slots 0)
      GeomScalar Yv[8], dYv[8][3];
#pragma unroll
      for (int s = 0; s < 8; s++) {
        const int y = g3l_yset(g, s);
        const bool ok = y >= 0;
        G3V4<GeomScalar> v{GeomScalar(0.0), GeomScalar(0.0), GeomScalar(0.0), GeomScalar(0.0)};
        if (ok) v = *reinterpret_cast<const G3V4<GeomScalar> *>(Pr + ((long) (g < 3 ? 8 * g + s : 24) * mn + jq) * 4);
        Yv[s] = ok ? v.x : GeomScalar(0.0);
        dYv[s][0] = ok ? v.y : GeomScalar(0.0); dYv[s][1] = ok ? v.z : GeomScalar(0.0); dYv[s][2] = ok ? v.w : GeomScalar(0.0);
      }
      // the dE/dI rows of the bonds 4 g + q (register q of the G tiles)
      long tq[4];
      bool vq[4];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        tq[q] = (long) wave_shfl(jg, 4 * g + q) * tsg + li;
        vq[q] = jb + 4 * g + q < nct;
      }
      const float *ib = indp + (long) jg * isg;
      GeomScalar sA = GeomScalar(0.0), sBx = GeomScalar(0.0), sBy = GeomScalar(0.0), sBz = GeomScalar(0.0);
      // channels in turn (not unrolled: the unrolled loop spilled)
#pragma unroll 1
      for (int cc = 0; cc < CB; cc++) {
        const int n = c0 + cc;
        float rA, drA, rB, drB;
        if (TST && tabr) {
          const G3F2 *Rb = reinterpret_cast<const G3F2 *>(Rw) + li * 20 + cc * 5;
          const G3F2 pa = Rb[la], pb = Rb[lb];
          rA = pa.x; drA = pa.y; rB = pb.x; drB = pb.y;
        } else if (tabr) {
          const float *trow = tab.c + tb + (long) n * tlm * 4;
          const G3F4 ca = *reinterpret_cast<const G3F4 *>(trow + 4 * la);
          const G3F4 cb = *reinterpret_cast<const G3F4 *>(trow + 4 * lb);
          rA = g3l_cubic(ca, tt);
          drA = g3l_cubic_dt(ca, tt) * dxs;
          rB = g3l_cubic(cb, tt);
          drB = g3l_cubic_dt(cb, tt) * dxs;
        } else {
          rA = R(ii, jq, n, la); drA = DR(ii, jq, n, la);
          rB = R(ii, jq, n, lb); drB = DR(ii, jq, n, lb);
        }
        const float RlA = rA * (float) ev, RlB = rB * (float) ev;
        const GeomScalar ReA = (GeomScalar) rA * ev, DReA = (GeomScalar) drA * ev + (GeomScalar) rA * dev;
        const GeomScalar ReB = (GeomScalar) rB * ev, DReB = (GeomScalar) drB * ev + (GeomScalar) rB * dev;
        // b of the lane's bond for the lane group's r (r >= NLI zero)
        float bvv[KS];
        {
          const float z = zproj(mj, n);
          const float *ibn = ib + n * isn;
#pragma unroll
          for (int m = 0; m < (KS + 3) / 4; m++) {
            const int r0 = 16 * m + 4 * g, nv = NLI - r0;
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            // the first term: rows full for every lane group, the branch folds at compile time; the tail (nv < 4)
            // is 1 or 2 floats for NLI 25 / 50
            if (16 * m + 16 <= NLI || nv >= 4) {
              const g3l_f4u q = *reinterpret_cast<const g3l_f4u *>(ibn + r0);
              v[0] = q.x; v[1] = q.y; v[2] = q.z; v[3] = q.w;
            } else if (nv >= 2) {
              const g3l_f2u q = *reinterpret_cast<const g3l_f2u *>(ibn + r0);
              v[0] = q.x; v[1] = q.y;
              if (nv == 3) v[2] = ibn[r0 + 2];
            } else if (nv == 1) {
              v[0] = ibn[r0];
            }
#pragma unroll
            for (int u = 0; u < 4; u++)
              if (4 * m + u < KS) bvv[4 * m + u] = v[u] + z * mk[4 * m + u];
          }
        }
        const float *Mc = Ms + cc * NP;
        // G[bond][r] = sum_y a[bond][y] M[y][r]: register q of tile tr is bond jb + 4 g + q, r = li + 16 tr
        mfma_f4v G[TR];
#pragma unroll
        for (int tr = 0; tr < TR; tr++) G[tr] = mfma_f4v{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int s = 0; s < 8; s++) {
          const float a = (s < sw ? RlA : RlB) * (float) Yv[s];
          const int ys = g3l_yset(g, s);
          const float *Mr = Mc + (ys < 0 ? 0 : ys) * NLI;
#pragma unroll
          for (int tr = 0; tr < TR; tr++) {
            const int r = li + 16 * tr, rc = r < NLI ? r : NLI - 1;
            mfma_16x16x4(a, Mr[rc], G[tr]);
          }
        }
        // V^T[y][bond] = sum_r M[y][r] b[bond][r]: register q of tile ty is y = Yset(g)[4 ty + q], bond jb + li
        mfma_f4v V[2];
#pragma unroll
        for (int ty = 0; ty < 2; ty++) V[ty] = mfma_f4v{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int s = 0; s < KS; s++) {
          const int r = 16 * (s >> 2) + 4 * g + (s & 3), rc = r < NLI ? r : NLI - 1;
#pragma unroll
          for (int ty = 0; ty < 2; ty++) mfma_16x16x4(Mc[aoff[ty] + rc], bvv[s], V[ty]);
        }
        // force partial of the lane's bond and slots (empty slots: Y = dY = 0)
#pragma unroll
        for (int s = 0; s < 8; s++) {
          const GeomScalar v = (GeomScalar) V[s >> 2][s & 3];
          sA += v * (s < sw ? DReA : DReB) * Yv[s];
          const GeomScalar vr = v * (s < sw ? ReA : ReB);
          sBx += vr * dYv[s][0]; sBy += vr * dYv[s][1]; sBz += vr * dYv[s][2];
        }
        // dE/dI rows: 16 consecutive components of 4 bonds per atomic instruction
        float *tn = tgtp + n * tsn;
#pragma unroll
        for (int q = 0; q < 4; q++)
          if (vq[q]) {
#pragma unroll
            for (int tr = 0; tr < TR; tr++)
              if (li + 16 * tr < NLI) Kokkos::atomic_add(tn + tq[q] + 16 * tr, G[tr][q]);
          }
      }
      // force of the bond: the 4 lane groups' slots, the 4 channels summed above
#pragma unroll
      for (int o = 16; o < 64; o <<= 1) {
        sA += wave_shfl_xor(sA, o);
        sBx += wave_shfl_xor(sBx, o); sBy += wave_shfl_xor(sBy, o); sBz += wave_shfl_xor(sBz, o);
      }
      if (g == 0 && valid) {
        const GeomScalar rinv = GeomScalar(1.0) / GeomScalar(rn(ii, j));
        Kokkos::atomic_add(&fij(ii, j, 0), sA * GeomScalar(rh(ii, j, 0)) + rinv * sBx);
        Kokkos::atomic_add(&fij(ii, j, 1), sA * GeomScalar(rh(ii, j, 1)) + rinv * sBy);
        Kokkos::atomic_add(&fij(ii, j, 2), sA * GeomScalar(rh(ii, j, 2)) + rinv * sBz);
      }
    }
  });
}
#endif    // GRACE3L_SPBF_MFMA

}    // namespace kk
}    // namespace grace

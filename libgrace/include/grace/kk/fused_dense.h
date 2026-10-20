// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from lammps-ace src/KOKKOS/grace_fused_kokkos_dense.h (07b8fb50d4); contributing author Yury Lysogorskiy (ICAMS)


/* ----------------------------------------------------------------------
   Contributing author: Yury Lysogorskiy (ICAMS)
   Dense-chain kernels shared by grace/1l/fused/kk and grace/2l/fused/kk: direct FC with liveness pruning
   (forward, reverse, reverse right part from transposed weights) and CG products, plus their metadata.
   Tensors are [chunk atom][channel][function] views; cs = atoms in the chunk.
------------------------------------------------------------------------- */

#pragma once

#include "grace/kk/fused_common.h"
#include "grace/kk/support.h"


#include <algorithm>
#include <array>
#include <vector>

namespace grace {
namespace kk {
namespace GraceFusedKK {

template<class T, class D> using GFV1 = Kokkos::View<T *, D>;
template<class T, class D> using GFV3 = Kokkos::View<T ***, Kokkos::LayoutRight, D>;    // float4 rows along the last index

template<class DeviceType>
struct FCMetaT {
  int no = 0, nf = 0, nil = 0, nir = 0, nsrc = 0, nlive = 0;
  int nirp = 0, hr = 1;                   // WrT: right input width padded to 4 hr; hr threads per (atom pair, source)
  GFV3<float, DeviceType> Wl, Wr, WrT;    // Wl/Wr [tile][n_in][n_out pad 4], norm folded in; WrT [tile][n_out][nirp]
  GFV1<int, DeviceType> flive, tl, toff, tsrc, ttile, soff, stgt, stile;
  GFV1<float, DeviceType> nof;
};

// CG product metadata: CSR by output (forward), by left / right operand (reverse)
template<class DeviceType>
struct ProdMetaT {
  int nout = 0, nl = 0, nr = 0;
  bool self = false;
  GFV1<int, DeviceType> foff, fa, fb, loff, la, lb, roff, ra, rb;
  GFV1<float, DeviceType> fc, lc, rc;
};

template<class DeviceType, class FCW>
void build_fc_meta(FCMetaT<DeviceType> &m, const FCW &fc, const std::vector<int> &used, const char *who)
{
  using F = float;
  const int no = fc.n_out, ntl = fc.w_shape_left, ntr = fc.w_shape_right;
  m.no = no; m.nf = fc.n_funcs_left;
  m.nil = (int) (fc.w_left.size() / ((size_t) no * ntl));
  m.nir = (int) (fc.w_right.size() / ((size_t) no * ntr));
  const int ncol = fc.n_funcs_right;
  int nsrc = 0;
  for (int c = 0; c < ncol; c++) nsrc = std::max(nsrc, fc.collect_from[c] + 1);
  m.nsrc = nsrc;
  const int nop = (no + 3) / 4 * 4;      // padded for float4 loads
  m.Wl = GFV3<float, DeviceType>("gfused:fcWl", ntl, m.nil, nop);
  m.Wr = GFV3<float, DeviceType>("gfused:fcWr", ntr, m.nir, nop);
  auto hWl = Kokkos::create_mirror_view(m.Wl);
  auto hWr = Kokkos::create_mirror_view(m.Wr);
  for (int k = 0; k < no; k++)
    for (int t = 0; t < ntl; t++)
      for (int n = 0; n < m.nil; n++)
        hWl(t, n, k) = (F) (fc.norm_left * fc.w_left[((size_t) k * m.nil + n) * ntl + t]);
  for (int k = 0; k < no; k++)
    for (int t = 0; t < ntr; t++)
      for (int n = 0; n < m.nir; n++)
        hWr(t, n, k) = (F) (fc.norm_right * fc.w_right[((size_t) k * m.nir + n) * ntr + t]);
  Kokkos::deep_copy(m.Wl, hWl); Kokkos::deep_copy(m.Wr, hWr);
  m.hr = m.nir > 48 ? 2 : 1;
  m.nirp = (m.nir + 4 * m.hr - 1) / (4 * m.hr) * (4 * m.hr);
  m.WrT = GFV3<float, DeviceType>("gfused:fcWrT", ntr, no, m.nirp);
  auto hWrT = Kokkos::create_mirror_view(m.WrT);
  for (int t = 0; t < ntr; t++)
    for (int k = 0; k < no; k++)
      for (int n = 0; n < m.nirp; n++) hWrT(t, k, n) = n < m.nir ? hWr(t, n, k) : (F) 0;
  Kokkos::deep_copy(m.WrT, hWrT);

  m.tl = GFV1<int, DeviceType>("gfused:fc_tl", m.nf); m.nof = GFV1<float, DeviceType>("gfused:fc_nof", m.nf);
  auto htl = Kokkos::create_mirror_view(m.tl); auto hnof = Kokkos::create_mirror_view(m.nof);
  for (int f = 0; f < m.nf; f++) { htl(f) = fc.w_tile_left[f]; hnof(f) = (F) fc.norm_out_factor[f]; }
  Kokkos::deep_copy(m.tl, htl); Kokkos::deep_copy(m.nof, hnof);
  std::vector<char> is_live(m.nf, 0);
  for (int f : used) if (f >= 0 && f < m.nf) is_live[f] = 1;
  std::vector<int> fl;
  for (int f = 0; f < m.nf; f++) if (is_live[f]) fl.push_back(f);
  m.nlive = fl.size();
  m.flive = GFV1<int, DeviceType>("gfused:fc_flive", std::max(m.nlive, 1));
  auto hfl = Kokkos::create_mirror_view(m.flive);
  for (int i = 0; i < m.nlive; i++) hfl(i) = fl[i];
  Kokkos::deep_copy(m.flive, hfl);

  // CSR by target (forward) and by source (reverse)
  std::vector<std::vector<int>> by_t(m.nf), by_s(nsrc);
  for (int c = 0; c < ncol; c++) {
    if (fc.collect_to[c] >= m.nf) GRACE_KK_ERROR("{}: FC collect_to outside the left function range", who);
    by_t[fc.collect_to[c]].push_back(c);
    if (is_live[fc.collect_to[c]]) by_s[fc.collect_from[c]].push_back(c);
  }
  m.toff = GFV1<int, DeviceType>("gfused:fc_toff", m.nf + 1); m.tsrc = GFV1<int, DeviceType>("gfused:fc_tsrc", std::max(ncol, 1));
  m.ttile = GFV1<int, DeviceType>("gfused:fc_ttile", std::max(ncol, 1));
  m.soff = GFV1<int, DeviceType>("gfused:fc_soff", nsrc + 1); m.stgt = GFV1<int, DeviceType>("gfused:fc_stgt", std::max(ncol, 1));
  m.stile = GFV1<int, DeviceType>("gfused:fc_stile", std::max(ncol, 1));
  auto htoff = Kokkos::create_mirror_view(m.toff), htsrc = Kokkos::create_mirror_view(m.tsrc);
  auto htt = Kokkos::create_mirror_view(m.ttile), hsoff = Kokkos::create_mirror_view(m.soff);
  auto hstgt = Kokkos::create_mirror_view(m.stgt), hst = Kokkos::create_mirror_view(m.stile);
  int q = 0;
  for (int f = 0; f < m.nf; f++) {
    htoff(f) = q;
    for (int c : by_t[f]) { htsrc(q) = fc.collect_from[c]; htt(q) = fc.w_tile_right[c]; q++; }
  }
  htoff(m.nf) = q;
  q = 0;
  for (int s = 0; s < nsrc; s++) {
    hsoff(s) = q;
    for (int c : by_s[s]) { hstgt(q) = fc.collect_to[c]; hst(q) = fc.w_tile_right[c]; q++; }
  }
  hsoff(nsrc) = q;
  Kokkos::deep_copy(m.toff, htoff); Kokkos::deep_copy(m.tsrc, htsrc); Kokkos::deep_copy(m.ttile, htt);
  Kokkos::deep_copy(m.soff, hsoff); Kokkos::deep_copy(m.stgt, hstgt); Kokkos::deep_copy(m.stile, hst);
}

template<class DeviceType>
void fill_csr(const std::vector<std::vector<std::array<double, 3>>> &lists, GFV1<int, DeviceType> &off,
              GFV1<int, DeviceType> &a, GFV1<int, DeviceType> &b, GFV1<float, DeviceType> &c)
{
  using F = float;
  const int nkey = lists.size();
  int tot = 0;
  for (auto &v : lists) tot += v.size();
  off = GFV1<int, DeviceType>("gfused:pr_off", nkey + 1); a = GFV1<int, DeviceType>("gfused:pr_a", std::max(tot, 1));
  b = GFV1<int, DeviceType>("gfused:pr_b", std::max(tot, 1)); c = GFV1<float, DeviceType>("gfused:pr_c", std::max(tot, 1));
  auto ho = Kokkos::create_mirror_view(off); auto ha = Kokkos::create_mirror_view(a);
  auto hb = Kokkos::create_mirror_view(b); auto hc = Kokkos::create_mirror_view(c);
  int q = 0;
  for (int k = 0; k < nkey; k++) {
    ho(k) = q;
    for (auto &e : lists[k]) { ha(q) = (int) e[0]; hb(q) = (int) e[1]; hc(q) = (F) e[2]; q++; }
  }
  ho(nkey) = q;
  Kokkos::deep_copy(off, ho); Kokkos::deep_copy(a, ha); Kokkos::deep_copy(b, hb); Kokkos::deep_copy(c, hc);
}

template<class DeviceType, class CGP>
void build_prod_meta(ProdMetaT<DeviceType> &m, const CGP &p, bool self)
{
  using F = float;
  const int nt = (int) p.cg_coeff.size();
  int nout = 0, nl = 0, nr = 0;
  for (int t = 0; t < nt; t++) {
    nout = std::max(nout, p.m_sum_ind[t] + 1);
    nl = std::max(nl, p.left_ind[t] + 1);
    nr = std::max(nr, p.right_ind[t] + 1);
  }
  if (self) nl = nr = std::max(nl, nr);
  m.nout = nout; m.nl = nl; m.nr = nr; m.self = self;
  // forward: key = output, a = left, b = right
  {
    std::vector<std::vector<std::array<double, 3>>> lists(nout);
    for (int t = 0; t < nt; t++) lists[p.m_sum_ind[t]].push_back({(double) p.left_ind[t], (double) p.right_ind[t], p.cg_coeff[t]});
    fill_csr<DeviceType>(lists, m.foff, m.fa, m.fb, m.fc);
  }
  // reverse, left role (+ right role for self): key = x, a = output, b = the other operand
  {
    std::vector<std::vector<std::array<double, 3>>> lists(nl);
    for (int t = 0; t < nt; t++) {
      lists[p.left_ind[t]].push_back({(double) p.m_sum_ind[t], (double) p.right_ind[t], p.cg_coeff[t]});
      if (self) lists[p.right_ind[t]].push_back({(double) p.m_sum_ind[t], (double) p.left_ind[t], p.cg_coeff[t]});
    }
    fill_csr<DeviceType>(lists, m.loff, m.la, m.lb, m.lc);
  }
  if (!self) {
    std::vector<std::vector<std::array<double, 3>>> lists(nr);
    for (int t = 0; t < nt; t++)
      lists[p.right_ind[t]].push_back({(double) p.m_sum_ind[t], (double) p.left_ind[t], p.cg_coeff[t]});
    fill_csr<DeviceType>(lists, m.roff, m.ra, m.rb, m.rc);
  }
}

// forward FC: two atoms per thread, weights as float4 along n_out
template<class DeviceType, int NO, class V3T>
void fc_fwd(const FCMetaT<DeviceType> &m, int cs, const V3T &in_l, const V3T &in_r, const V3T &out)
{
  using F = float;
  // two atoms (ii, ii + half) per thread, weights as float4 along n_out: 8 FMA per weight load
  constexpr int NQ = (NO + 3) / 4;
  const int nlive = m.nlive, nil = m.nil, nir = m.nir;
  const int half = (cs + 1) / 2;
  auto Wl = m.Wl; auto Wr = m.Wr; auto tl = m.tl; auto nof = m.nof; auto flive = m.flive;
  auto toff = m.toff; auto tsrc = m.tsrc; auto ttile = m.ttile;
  Kokkos::parallel_for("gfused:FC", Kokkos::RangePolicy<DeviceType>(0, half * nlive), KOKKOS_LAMBDA(const int q) {
    const int fi = q / half, i0 = q - fi * half, i1 = i0 + half, f = flive(fi);
    const bool two = i1 < cs;
    F4 a0[NQ], a1[NQ];
    for (int k = 0; k < NQ; k++) { a0[k] = F4{0.f, 0.f, 0.f, 0.f}; a1[k] = a0[k]; }
    const int t = tl(f);
    for (int n = 0; n < nil; n++) {
      const F x0 = in_l(i0, n, f), x1 = two ? in_l(i1, n, f) : 0.0f;
      const F4 *w = reinterpret_cast<const F4 *>(&Wl(t, n, 0));
      for (int k = 0; k < NQ; k++) { const F4 wk = w[k]; fma4(a0[k], wk, x0); fma4(a1[k], wk, x1); }
    }
    for (int c = toff(f); c < toff(f + 1); c++) {
      const int s = tsrc(c), tr = ttile(c);
      for (int n = 0; n < nir; n++) {
        const F x0 = in_r(i0, n, s), x1 = two ? in_r(i1, n, s) : 0.0f;
        const F4 *w = reinterpret_cast<const F4 *>(&Wr(tr, n, 0));
        for (int k = 0; k < NQ; k++) { const F4 wk = w[k]; fma4(a0[k], wk, x0); fma4(a1[k], wk, x1); }
      }
    }
    const F sc = nof(f);
    for (int k = 0; k < NQ; k++) {
      const F v0[4] = {a0[k].x, a0[k].y, a0[k].z, a0[k].w}, v1[4] = {a1[k].x, a1[k].y, a1[k].z, a1[k].w};
      for (int u = 0; u < 4; u++) {
        const int kk = 4 * k + u;
        if (kk < NO) { out(i0, kk, f) = v0[u] * sc; if (two) out(i1, kk, f) = v1[u] * sc; }
      }
    }
  });
}

// forward FC: four atoms per thread, half of the outputs each: 16 FMA per float4 weight load
template<class DeviceType, int NO, class V3T>
void fc_fwd4(const FCMetaT<DeviceType> &m, int cs, const V3T &in_l, const V3T &in_r, const V3T &out)
{
  using F = float;
  constexpr int NH = NO / 2, NQ = NH / 4;
  const int nlive = m.nlive, nil = m.nil, nir = m.nir;
  const int qa = (cs + 3) / 4;
  auto Wl = m.Wl; auto Wr = m.Wr; auto tl = m.tl; auto nof = m.nof; auto flive = m.flive;
  auto toff = m.toff; auto tsrc = m.tsrc; auto ttile = m.ttile;
  Kokkos::parallel_for("gfused:FC4", Kokkos::RangePolicy<DeviceType>(0, qa * 2 * nlive), KOKKOS_LAMBDA(const int q) {
    const int fh = q / qa, i0 = q - fh * qa, fi = fh >> 1, h = fh & 1, f = flive(fi);
    int ia[4];
    bool ok[4];
    for (int a = 0; a < 4; a++) { ia[a] = i0 + a * qa; ok[a] = ia[a] < cs; if (!ok[a]) ia[a] = i0; }
    F4 acc[4][NQ];
    for (int a = 0; a < 4; a++)
      for (int k = 0; k < NQ; k++) acc[a][k] = F4{0.f, 0.f, 0.f, 0.f};
    const int t = tl(f);
    for (int n = 0; n < nil; n++) {
      F x[4];
      for (int a = 0; a < 4; a++) x[a] = in_l(ia[a], n, f);
      const F4 *w = reinterpret_cast<const F4 *>(&Wl(t, n, 0)) + h * NQ;
      for (int k = 0; k < NQ; k++) {
        const F4 wk = w[k];
        for (int a = 0; a < 4; a++) fma4(acc[a][k], wk, x[a]);
      }
    }
    for (int c = toff(f); c < toff(f + 1); c++) {
      const int sr = tsrc(c), tr = ttile(c);
      for (int n = 0; n < nir; n++) {
        F x[4];
        for (int a = 0; a < 4; a++) x[a] = in_r(ia[a], n, sr);
        const F4 *w = reinterpret_cast<const F4 *>(&Wr(tr, n, 0)) + h * NQ;
        for (int k = 0; k < NQ; k++) {
          const F4 wk = w[k];
          for (int a = 0; a < 4; a++) fma4(acc[a][k], wk, x[a]);
        }
      }
    }
    const F sc = nof(f);
    for (int a = 0; a < 4; a++) {
      if (!ok[a]) continue;
      for (int k = 0; k < NQ; k++) {
        const int kk = h * NH + 4 * k;
        out(ia[a], kk, f) = acc[a][k].x * sc; out(ia[a], kk + 1, f) = acc[a][k].y * sc;
        out(ia[a], kk + 2, f) = acc[a][k].z * sc; out(ia[a], kk + 3, f) = acc[a][k].w * sc;
      }
    }
  });
}

// reverse FC, right part, from the transposed weights: thread = (two atoms, source, 1/H of the inputs);
// acc[n] += g[k] WrT[k][n] with g = dout scalars: 8 FMA per float4 weight load, no dout vector in registers
template<class DeviceType, int NO, int NIRP, int H, class V3T>
void fc_bwd_rt(const FCMetaT<DeviceType> &m, int cs, const V3T &dout, const V3T &din_r)
{
  using F = float;
  constexpr int NQR = NIRP / 4 / H;
  const int nsrc = m.nsrc, nir = m.nir;
  const int half = (cs + 1) / 2;
  auto WrT = m.WrT; auto nof = m.nof; auto soff = m.soff; auto stgt = m.stgt; auto stile = m.stile;
  Kokkos::parallel_for("gfused:RevFC_RT", Kokkos::RangePolicy<DeviceType>(0, half * nsrc * H), KOKKOS_LAMBDA(const int q) {
    const int sh = q / half, i0 = q - sh * half, i1 = i0 + half, sidx = sh / H, h = sh - sidx * H;
    if (soff(sidx) == soff(sidx + 1)) return;
    const bool two = i1 < cs;
    F4 a0[NQR], a1[NQR];
    for (int j = 0; j < NQR; j++) { a0[j] = F4{0.f, 0.f, 0.f, 0.f}; a1[j] = a0[j]; }
    for (int c = soff(sidx); c < soff(sidx + 1); c++) {
      const int tg = stgt(c), tr = stile(c);
      const F sc = nof(tg);
      const F4 *wt = reinterpret_cast<const F4 *>(&WrT(tr, 0, 0)) + h * NQR;
#pragma unroll 2
      for (int k = 0; k < NO; k++) {
        const F g0 = dout(i0, k, tg) * sc, g1 = two ? dout(i1, k, tg) * sc : 0.0f;
        const F4 *w = wt + k * (NIRP / 4);
        for (int j = 0; j < NQR; j++) { const F4 wj = w[j]; fma4(a0[j], wj, g0); fma4(a1[j], wj, g1); }
      }
    }
    for (int j = 0; j < NQR; j++) {
      const F v0[4] = {a0[j].x, a0[j].y, a0[j].z, a0[j].w}, v1[4] = {a1[j].x, a1[j].y, a1[j].z, a1[j].w};
      for (int u = 0; u < 4; u++) {
        const int n = (h * NQR + j) * 4 + u;
        if (n < nir) { din_r(i0, n, sidx) += v0[u]; if (two) din_r(i1, n, sidx) += v1[u]; }
      }
    }
  });
}

// reverse FC: left part (two atoms per thread), right part (fc_bwd_rt, or one atom per thread with up to NIRMAX inputs)
template<class DeviceType, int NO, int NIRMAX, class V3T>
void fc_bwd(const FCMetaT<DeviceType> &m, int cs, const V3T &dout, const V3T &din_l, const V3T &din_r)
{
  using F = float;
  constexpr int NQ = (NO + 3) / 4;
  const int nlive = m.nlive, nil = m.nil, nir = m.nir, nsrc = m.nsrc;
  const int half = (cs + 1) / 2;
  auto Wl = m.Wl; auto Wr = m.Wr; auto tl = m.tl; auto nof = m.nof; auto flive = m.flive;
  auto soff = m.soff; auto stgt = m.stgt; auto stile = m.stile;
  Kokkos::parallel_for("gfused:RevFC_L", Kokkos::RangePolicy<DeviceType>(0, half * nlive), KOKKOS_LAMBDA(const int q) {
    const int fi = q / half, i0 = q - fi * half, i1 = i0 + half, f = flive(fi);
    const bool two = i1 < cs;
    const F sc = nof(f);
    F4 g0[NQ], g1[NQ];
    for (int k = 0; k < NQ; k++) {
      F v0[4], v1[4];
      for (int u = 0; u < 4; u++) {
        const int kk = 4 * k + u;
        v0[u] = (kk < NO) ? dout(i0, kk, f) * sc : 0.0f;
        v1[u] = (kk < NO && two) ? dout(i1, kk, f) * sc : 0.0f;
      }
      g0[k] = F4{v0[0], v0[1], v0[2], v0[3]}; g1[k] = F4{v1[0], v1[1], v1[2], v1[3]};
    }
    const int t = tl(f);
    for (int n = 0; n < nil; n++) {
      const F4 *w = reinterpret_cast<const F4 *>(&Wl(t, n, 0));
      F s0 = 0.0f, s1 = 0.0f;
      for (int k = 0; k < NQ; k++) { const F4 wk = w[k]; s0 += dot4(wk, g0[k]); s1 += dot4(wk, g1[k]); }
      din_l(i0, n, f) += s0;
      if (two) din_l(i1, n, f) += s1;
    }
  });
  if (nsrc == 0) return;
  if (m.hr == 1 && m.nirp == 32) { fc_bwd_rt<DeviceType, NO, 32, 1>(m, cs, dout, din_r); return; }
  if (m.hr == 1 && m.nirp == 44) { fc_bwd_rt<DeviceType, NO, 44, 1>(m, cs, dout, din_r); return; }
  if (m.hr == 2 && m.nirp == 64) { fc_bwd_rt<DeviceType, NO, 64, 2>(m, cs, dout, din_r); return; }
  Kokkos::parallel_for("gfused:RevFC_R", Kokkos::RangePolicy<DeviceType>(0, cs * nsrc), KOKKOS_LAMBDA(const int q) {
    const int sidx = q / cs, ii = q - sidx * cs;
    if (soff(sidx) == soff(sidx + 1)) return;
    F acc[NIRMAX];
    for (int n = 0; n < NIRMAX; n++) acc[n] = 0.0f;
    for (int c = soff(sidx); c < soff(sidx + 1); c++) {
      const int tg = stgt(c), tr = stile(c);
      const F sc = nof(tg);
      F4 g[NQ];
      for (int k = 0; k < NQ; k++) {
        F v[4];
        for (int u = 0; u < 4; u++) { const int kk = 4 * k + u; v[u] = (kk < NO) ? dout(ii, kk, tg) * sc : 0.0f; }
        g[k] = F4{v[0], v[1], v[2], v[3]};
      }
#pragma unroll
      for (int n = 0; n < NIRMAX; n++) {
        if (n >= nir) continue;
        const F4 *w = reinterpret_cast<const F4 *>(&Wr(tr, n, 0));
        F s = 0.0f;
        for (int k = 0; k < NQ; k++) s += dot4(w[k], g[k]);
        acc[n] += s;
      }
    }
#pragma unroll
    for (int n = 0; n < NIRMAX; n++) if (n < nir) din_r(ii, n, sidx) += acc[n];
  });
}

template<class DeviceType, class V3T>
void prod_fwd(const ProdMetaT<DeviceType> &m, int cs, const V3T &L, const V3T &R, const V3T &out)
{
  using F = float;
  const int nc = (int) out.extent(1), nout = m.nout;
  auto off = m.foff; auto ia = m.fa; auto ib = m.fb; auto cg = m.fc;
  Kokkos::parallel_for("gfused:Prod", Kokkos::RangePolicy<DeviceType>(0, cs * nc), KOKKOS_LAMBDA(const int q) {
    const int k = q / cs, ii = q - k * cs;
    for (int o = 0; o < nout; o++) {
      F s = 0.0f;
      for (int e = off(o); e < off(o + 1); e++) s += cg(e) * L(ii, k, ia(e)) * R(ii, k, ib(e));
      out(ii, k, o) = s;
    }
  });
}

template<class DeviceType, class V3T>
void prod_rev(const ProdMetaT<DeviceType> &m, int cs, const V3T &L, const V3T &R, const V3T &D, const V3T &dL,
              const V3T &dR, bool acc_left = false)
{
  using F = float;
  // self: L == R, dL gets both roles. mixed: dL assigned (or accumulated: acc_left), dR accumulated.
  const int nc = (int) D.extent(1);
  const int nlx = (int) dL.extent(2), nl = m.nl;
  {
    auto off = m.loff; auto io = m.la; auto ib = m.lb; auto cg = m.lc;
    Kokkos::parallel_for("gfused:RevProd_L", Kokkos::RangePolicy<DeviceType>(0, cs * nc), KOKKOS_LAMBDA(const int q) {
      const int k = q / cs, ii = q - k * cs;
      for (int x = 0; x < nlx; x++) {
        F s = 0.0f;
        if (x < nl)
          for (int e = off(x); e < off(x + 1); e++) s += cg(e) * D(ii, k, io(e)) * R(ii, k, ib(e));
        if (acc_left) dL(ii, k, x) += s;
        else dL(ii, k, x) = s;
      }
    });
  }
  if (m.self) return;
  {
    const int nr = m.nr;
    auto off = m.roff; auto io = m.ra; auto ib = m.rb; auto cg = m.rc;
    Kokkos::parallel_for("gfused:RevProd_R", Kokkos::RangePolicy<DeviceType>(0, cs * nc), KOKKOS_LAMBDA(const int q) {
      const int k = q / cs, ii = q - k * cs;
      for (int x = 0; x < nr; x++) {
        if (off(x) == off(x + 1)) continue;
        F s = 0.0f;
        for (int e = off(x); e < off(x + 1); e++) s += cg(e) * D(ii, k, io(e)) * L(ii, k, ib(e));
        dR(ii, k, x) += s;
      }
    });
  }
}

}    // namespace GraceFusedKK
}    // namespace kk
}    // namespace grace


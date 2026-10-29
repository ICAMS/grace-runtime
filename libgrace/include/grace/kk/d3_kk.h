// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// DFT-D3(BJ) dispersion on the device (Kokkos; CUDA, HIP): the passes of grace/d3.h (CN, energy + direct forces +
// dE/dCN, CN-chain forces) in gather form, one team (warp / wavefront) per atom sharing out the bins, over the bins of
// a NeighborListKK (one structure) or a BatchNeighborListKK (many) built from earlier positions: the bins within reach
// of rc + skin around the build positions are visited and a pair is kept when its current distance is below rc (the
// Verlet condition of the list: no atom moved more than skin / 2 since the build). No stored long-range list and no
// ghost images: the lattice translation of each visited image is applied to the current positions, moved by the
// list's integer images first (x - img cell, near the cell). Bond vectors: the position difference in double, the
// lattice translation and everything after in float; CN, reference weights and all sums in double. Results per atom
// in the caller's order. The three-body term (s9 != 0): one team per first vertex of its triangles, the later
// neighbours within atm_cutoff gathered into team scratch, their pairs shared out over the threads in a fixed
// orientation; the other vertices' shares summed in scratch (energy in double, the rest in float) and added
// atomically, so the order of those sums varies: with the three-body term the last bits of energies, forces and the
// virial may differ between runs.
#pragma once

#include "grace/d3.h"
#include "grace/kk/neighbors_kk.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <type_traits>

namespace grace {
namespace kk {

// tables of one species set on the device
template<class DeviceType> struct D3TablesKK {
  int ns = 0;
  DispersionParams p;
  Kokkos::View<int *, DeviceType> nref;                  // [ns]
  Kokkos::View<double *, DeviceType> cnref;              // [ns * kMaxRef]
  Kokkos::View<float *, DeviceType> rcov, qq, r06, r08;  // [ns * ns]
  Kokkos::View<float *, DeviceType> c6ref;               // [ns * ns * kMaxRef * kMaxRef]
  Kokkos::View<float *, DeviceType> rvdw16;              // [ns * ns] (4/3 R0vdw)^(16/3)

  void set(const d3::Tables &t)
  {
    p = t.p;
    ns = t.ns;
    auto up = [](const char *name, const auto &v, auto conv) {
      using V = Kokkos::View<decltype(conv(v[0])) *, DeviceType>;
      V d(name, std::max<size_t>(v.size(), 1));
      auto h = Kokkos::create_mirror_view(d);
      for (size_t k = 0; k < v.size(); k++) h(k) = conv(v[k]);
      Kokkos::deep_copy(d, h);
      return d;
    };
    auto asf = [](double v) { return (float) v; };
    nref = up("d3:nref", t.nref, [](int v) { return v; });
    cnref = up("d3:cnref", t.cnref, [](double v) { return v; });
    rcov = up("d3:rcov", t.rcov, asf);
    qq = up("d3:qq", t.qq, asf);
    r06 = up("d3:r06", t.r06, asf);
    r08 = up("d3:r08", t.r08, asf);
    c6ref = up("d3:c6ref", t.c6ref, asf);
    rvdw16 = up("d3:rvdw16", t.rvdw16, asf);
  }
};

// per-atom work and results (caller's order)
template<class DeviceType> struct D3WorkKK {
  static constexpr int R = d3::kMaxRef;
  Kokkos::View<double *, DeviceType> dedcn, e;
  Kokkos::View<double *[R], Kokkos::LayoutRight, DeviceType> dw;
  Kokkos::View<float *[R], Kokkos::LayoutRight, DeviceType> wf;    // w in float, read for the neighbours
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> f;
  Kokkos::View<double *[6], Kokkos::LayoutRight, DeviceType> vir;
  Kokkos::View<double *[7], Kokkos::LayoutRight, DeviceType> sums;    // per structure: energy, virial [6]
  // positions moved by the list's integer image (x - img cell): near the cell, so bond vectors stay accurate in
  // float for unwrapped positions far from it (long MD)
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> xr;
  Kokkos::View<float *[9], Kokkos::LayoutRight, DeviceType> cells;    // float cell rows per structure (batch)

  void ensure(int64_t n)
  {
    if ((int64_t) e.extent(0) >= n && e.extent(0) > 0) return;
    const int64_t m = std::max<int64_t>(n + n / 8, 16);
    dedcn = Kokkos::View<double *, DeviceType>("d3:dedcn", m);
    e = Kokkos::View<double *, DeviceType>("d3:e", m);
    dw = decltype(dw)("d3:dw", m);
    wf = decltype(wf)("d3:wf", m);
    f = decltype(f)("d3:f", m);
    vir = decltype(vir)("d3:vir", m);
    xr = decltype(xr)("d3:xr", m);
  }
  void release() { *this = D3WorkKK(); }
};

struct D3Sums {
  double energy = 0.0;
  double virial[6] = {0, 0, 0, 0, 0, 0};    // xx yy zz xy xz yz, W = -dE/dstrain
};

// wk.sums [nsys][7] from the per-atom e (and vir unless energy_only: zeros) of the atoms start(s) .. start(s + 1) - 1,
// one team per structure
template<class DeviceType, class Start>
void d3_sums(D3WorkKK<DeviceType> &wk, int nsys, const Start &start, bool energy_only)
{
  using TP = Kokkos::TeamPolicy<typename DeviceType::execution_space>;
  if ((int) wk.sums.extent(0) < nsys) wk.sums = decltype(wk.sums)("d3:sums", nsys + nsys / 8 + 1);
  const auto e = wk.e;
  const auto vir = wk.vir;
  const auto sums = wk.sums;
  Kokkos::parallel_for("d3:sums", TP(nsys, Kokkos::AUTO), KOKKOS_LAMBDA(const typename TP::member_type &team) {
    const int s = team.league_rank();
    const int64_t a0 = start(s), a1 = start(s + 1);
    for (int m = 0; m < 7; m++) {
      double v = 0.0;
      if (m == 0 || !energy_only)
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, a0, a1),
                                [&](const int64_t a, double &acc) { acc += m == 0 ? e(a) : vir(a, m - 1); }, v);
      Kokkos::single(Kokkos::PerTeam(team), [&]() { sums(s, m) = v; });
    }
  });
}

// the atom range of one structure of n atoms, for d3_sums
struct D3OneRange {
  int64_t n;
  KOKKOS_INLINE_FUNCTION int64_t operator()(int s) const { return s == 0 ? 0 : n; }
};

// cell rows (float) of the structure of new atom q: one cell (kernel argument: uniform loads, no registers held), or
// the cell of q's structure in a batch
struct D3CellOne {
  float H[9];
  explicit D3CellOne(const Mat3 &m)
  {
    for (int k = 0; k < 9; k++) H[k] = (float) m.a[k];
  }
  KOKKOS_INLINE_FUNCTION float h(int, int k) const { return H[k]; }
};
template<class DeviceType> struct D3CellBatch {
  Kokkos::View<int *, DeviceType> sy;
  Kokkos::View<float *[9], Kokkos::LayoutRight, DeviceType> H;
  KOKKOS_INLINE_FUNCTION float h(int q, int k) const { return H(sy(q), k); }
};

// every atom of the bins within reach of a visitor (no distance test: the caller decides with the current positions),
// the bins shared out over the threads of a team: fn(p, s0, s1, s2) as the visitors; g = v.grid(q). Half: only the
// atoms after q in the order (bin offset, lexicographic; then slot in the own bin), as CellList::gather_half, so each
// pair of atoms and images comes from one of its two atoms
template<bool Half = false, class Team, template<class> class Visitor, class DeviceType, class F>
KOKKOS_INLINE_FUNCTION void team_visit(const Team &team, const Visitor<DeviceType> &v, const BinGrid &g, const int q,
                                       const F &fn)
{
  const int a = v.pm(q);
  // the grid as scalars captured by value (a local struct captured by reference may live in local memory)
  const int64_t r0 = g.r[0], r1 = g.r[1], r2 = g.r[2], n0 = g.nbk[0], n1 = g.nbk[1], n2 = g.nbk[2], b0 = g.base;
  const int p0 = g.pp[0], p1 = g.pp[1], p2 = g.pp[2];
  const int64_t w1 = 2 * r1 + 1, w2 = 2 * r2 + 1, nbin = (2 * r0 + 1) * w1 * w2, mid = nbin / 2;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, Half ? mid : (int64_t) 0, nbin),
                       [&, a, r0, r1, r2, n0, n1, n2, b0, p0, p1, p2, w1, w2, mid](const int64_t k) {
    const int64_t nb[3] = {n0, n1, n2};
    const int pb[3] = {p0, p1, p2};
    int64_t c[3] = {v.bb(a, 0) + k / (w1 * w2) - r0, v.bb(a, 1) + (k / w2) % w1 - r1, v.bb(a, 2) + k % w2 - r2};
    int64_t sh[3] = {0, 0, 0};
    for (int d = 0; d < 3; d++) {
      if (pb[d]) {
        sh[d] = NeighborVisitor<DeviceType>::fdiv(c[d], nb[d]);
        c[d] -= sh[d] * nb[d];
      } else if (c[d] < 0 || c[d] >= nb[d]) {
        return;
      }
    }
    const int64_t cb = b0 + (c[0] * n1 + c[1]) * n2 + c[2];
    for (int64_t p = v.bs(cb); p < v.bs(cb + 1); p++) {
      if (p == q && sh[0] == 0 && sh[1] == 0 && sh[2] == 0) continue;
      if (Half && k == mid && p <= q) continue;
      fn((int) p, (int) sh[0], (int) sh[1], (int) sh[2]);
    }
  });
}

// threads per atom: a warp (CUDA) / wavefront (HIP), 1 on the host
template<class DeviceType> int d3_team_size()
{
  if (std::is_same<typename DeviceType::execution_space, Kokkos::DefaultHostExecutionSpace>::value) return 1;
#if defined(KOKKOS_ENABLE_HIP)
  return 64;
#else
  return 32;
#endif
}

// the passes over n atoms (CN, pair terms, three-body terms with s9 != 0, CN-chain forces), one team per atom;
// per-atom results in wk (e; f and vir unless energy_only) (wk.xr filled by the caller); vis_atm: the visitor of
// atm_cutoff (vis_rc when the three-body term is off)
template<class DeviceType, class Visitor, class Cell>
void d3_passes(const D3TablesKK<DeviceType> &t, D3WorkKK<DeviceType> &wk,
               const Kokkos::View<const int *, DeviceType> &type, int type_base, int64_t n, const Visitor &vis_cn,
               const Visitor &vis_rc, const Visitor &vis_atm, const Cell &cell,
               const Kokkos::View<int *, DeviceType> &pm, bool energy_only)
{
  constexpr int R = d3::kMaxRef;
  using Exec = typename DeviceType::execution_space;
  using TP = Kokkos::TeamPolicy<Exec>;
  using Member = typename TP::member_type;
  using Sum = Kokkos::Sum<double>;
  const TP policy((int) n, d3_team_size<DeviceType>());
  wk.ensure(n);
  const int ns = t.ns, tb = type_base;
  const bool atm3 = t.p.s9 != 0.0;
  const float rc2 = (float) (t.p.cutoff * t.p.cutoff), rcn2 = (float) (t.p.cn_cutoff * t.p.cn_cutoff);
  const float s6 = (float) t.p.s6, s8 = (float) t.p.s8, rcf = (float) t.p.cutoff, ronf = (float) t.p.cutoff_on;
  const auto nref = t.nref;
  const auto cnref = t.cnref;
  const auto rcov = t.rcov, qq = t.qq, r06 = t.r06, r08 = t.r08, c6ref = t.c6ref;
  const auto dw = wk.dw;
  const auto wf = wk.wf;
  const auto e = wk.e, dedcn = wk.dedcn;
  const auto f = wk.f;
  const auto vir = wk.vir;
  const auto xr = wk.xr;
  // bond vector to the visited image (p, s) of atom b from atom a (s in the wrapped frame of the list), from the
  // current positions: the difference of the image-moved positions in double, the lattice translation in float
  auto bond = KOKKOS_LAMBDA(int q, int a, int b, int s0, int s1, int s2, float &dx, float &dy, float &dz) {
    const float t0 = (float) s0, t1 = (float) s1, t2 = (float) s2;
    dx = (float) (xr(b, 0) - xr(a, 0)) + t0 * cell.h(q, 0) + t1 * cell.h(q, 3) + t2 * cell.h(q, 6);
    dy = (float) (xr(b, 1) - xr(a, 1)) + t0 * cell.h(q, 1) + t1 * cell.h(q, 4) + t2 * cell.h(q, 7);
    dz = (float) (xr(b, 2) - xr(a, 2)) + t0 * cell.h(q, 2) + t1 * cell.h(q, 5) + t2 * cell.h(q, 8);
  };
  Kokkos::parallel_for("d3:cn", policy, KOKKOS_LAMBDA(const Member &team) {
    const int q = team.league_rank(), a = pm(q), si = type(a) - tb;
    const BinGrid g = vis_cn.grid(q);
    double c = 0.0;
    team_visit(team, vis_cn, g, q, [&](int p, int s0, int s1, int s2) {
      const int b = pm(p);
      float dx, dy, dz;
      bond(q, a, b, s0, s1, s2, dx, dy, dz);
      const float r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= rcn2 || r2 == 0.0f) return;
      float dfdr;
      c += (double) d3::count<float>(rcov(si * ns + type(b) - tb), Kokkos::sqrt(r2), dfdr);
    });
    team.team_reduce(Sum(c));
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      double wl[R], dwl[R];
      d3::weights(c, &cnref(si * R), nref(si), wl, energy_only ? nullptr : dwl);    // as the CPU path
      for (int k = 0; k < R; k++) {
        if (!energy_only) dw(a, k) = dwl[k];
        wf(a, k) = (float) wl[k];
      }
    });
  });
  Kokkos::parallel_for("d3:pair", policy, KOKKOS_LAMBDA(const Member &team) {
    const int q = team.league_rank(), a = pm(q), si = type(a) - tb, ni = nref(si);
    float wi[R], dwi[R];
    for (int k = 0; k < R; k++) {
      wi[k] = wf(a, k);
      dwi[k] = (float) dw(a, k);
    }
    const BinGrid g = vis_rc.grid(q);
    double ei = 0.0, dei = 0.0, fx = 0.0, fy = 0.0, fz = 0.0, v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;
    team_visit(team, vis_rc, g, q, [&](int p, int s0, int s1, int s2) {
      const int b = pm(p);
      float dx, dy, dz;
      bond(q, a, b, s0, s1, s2, dx, dy, dz);
      const float r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= rc2 || r2 == 0.0f) return;
      const int sj = type(b) - tb, nj = nref(sj), st = si * ns + sj;
      float wj[R];
      for (int k = 0; k < nj; k++) wj[k] = wf(b, k);
      float dc6i, dgdr;
      const float c = d3::c6<float>(wi, dwi, ni, wj, nj, &c6ref(st * R * R), dc6i);
      const float r = Kokkos::sqrt(r2);
      const float g = d3::pair<float>(r, r06(st), r08(st), qq(st), s6, s8, rcf, ronf, dgdr);
      ei += 0.5 * (double) (c * g);
      if (energy_only) return;
      dei += (double) (g * dc6i);
      const float h = c * dgdr / r;
      fx += (double) (h * dx);
      fy += (double) (h * dy);
      fz += (double) (h * dz);
      v0 -= (double) (0.5f * h * dx * dx);
      v1 -= (double) (0.5f * h * dy * dy);
      v2 -= (double) (0.5f * h * dz * dz);
      v3 -= (double) (0.5f * h * dx * dy);
      v4 -= (double) (0.5f * h * dx * dz);
      v5 -= (double) (0.5f * h * dy * dz);
    });
    team.team_reduce(Sum(ei));
    if (!energy_only) {
      team.team_reduce(Sum(dei));
      team.team_reduce(Sum(fx));
      team.team_reduce(Sum(fy));
      team.team_reduce(Sum(fz));
      team.team_reduce(Sum(v0));
      team.team_reduce(Sum(v1));
      team.team_reduce(Sum(v2));
      team.team_reduce(Sum(v3));
      team.team_reduce(Sum(v4));
      team.team_reduce(Sum(v5));
    }
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      e(a) = ei;
      if (energy_only) return;
      dedcn(a) = dei;
      f(a, 0) = fx;
      f(a, 1) = fy;
      f(a, 2) = fz;
      vir(a, 0) = v0;
      vir(a, 1) = v1;
      vir(a, 2) = v2;
      vir(a, 3) = v3;
      vir(a, 4) = v4;
      vir(a, 5) = v5;
    });
  });
  if (atm3) {
    // three-body term, each triangle once: from its first vertex in the order of team_visit<true> (bin offset, then
    // slot) over the pairs of the vertices after it ("later" neighbours); the shares of the two other vertices summed
    // per neighbour in team scratch and added atomically; the triangle's virial to the first vertex
    const auto rvdw16 = t.rvdw16;
    const float rc32 = (float) (t.p.atm_cutoff * t.p.atm_cutoff), rc3f = (float) t.p.atm_cutoff,
                ron3f = (float) t.p.atm_cutoff_on, s9u = (float) (t.p.s9 / std::sqrt(kHartree));
    // the most later neighbours of any atom: the scratch size, counted to a slightly larger cutoff (the gather below
    // may round r^2 differently, e.g. contracted to FMA), and the gather keeps at most that many
    const float rc32n = rc32 * (1.0f + 1e-5f);
    int mmax = 0;
    Kokkos::parallel_reduce("d3:atm_count", policy, KOKKOS_LAMBDA(const Member &team, int &mx) {
      const int q = team.league_rank(), a = pm(q);
      const BinGrid g = vis_atm.grid(q);
      int c = 0;
      team_visit<true>(team, vis_atm, g, q, [&](int p, int s0, int s1, int s2) {
        float dx, dy, dz;
        bond(q, a, pm(p), s0, s1, s2, dx, dy, dz);
        const float r2 = dx * dx + dy * dy + dz * dz;
        c += r2 < rc32n && r2 > 0.0f;
      });
      team.team_reduce(Kokkos::Sum<int>(c));
      Kokkos::single(Kokkos::PerTeam(team), [&]() { mx = c > mx ? c : mx; });
    }, Kokkos::Max<int>(mmax));
    // per later neighbour (float, [neighbour][field]): bond vector, C6 with the first vertex and the ratios
    // dC6/dCN / C6 there and at the neighbour, weights and their derivatives, the sums of its shares (dE/dCN, force),
    // its triangle side (d3::AtmSide); its atom; its energy sum (double)
    enum { X = 0, Y, Z, C, RQ, RB, W, DW = W + R, AD = DW + R, AX, AY, AZ, SU, SIU, SR, SQ, SSW, SDS, NF };
    using Scr = typename Exec::scratch_memory_space;
    using SF = Kokkos::View<float **, Kokkos::LayoutLeft, Scr, Kokkos::MemoryUnmanaged>;
    using SD = Kokkos::View<double *, Scr, Kokkos::MemoryUnmanaged>;
    using SI = Kokkos::View<int *, Scr, Kokkos::MemoryUnmanaged>;
    const int cap = std::max(mmax, 1);
    const size_t bytes = SD::shmem_size(cap) + SF::shmem_size(cap, NF) + SI::shmem_size(cap) + SI::shmem_size(1);
    const int level = bytes <= (size_t) TP::scratch_size_max(0) ? 0 : 1;
    // 256 threads per team: the scratch of one warp per centre left few teams per SM (RTX 3090, 4000 Cu atoms at
    // 10 / 12 A: 33 / 167 ms with 32 threads, 22 / 85 with 64, 14 / 48 with 128, 13 / 37 with 256, 15 / 39 with 512)
    TP pol((int) n, std::is_same<Exec, Kokkos::DefaultHostExecutionSpace>::value ? 1 : 256);
    pol.set_scratch_size(level, Kokkos::PerTeam(bytes));
    Kokkos::parallel_for("d3:atm", pol, KOKKOS_LAMBDA(const Member &team) {
      const int q = team.league_rank(), a = pm(q), si = type(a) - tb, ni = nref(si);
      SD sd(team.team_scratch(level), cap);
      SF sf(team.team_scratch(level), cap, NF);
      SI sb(team.team_scratch(level), cap), cnt(team.team_scratch(level), 1);
      float wi[R], dwi[R];
      for (int k = 0; k < R; k++) {
        wi[k] = wf(a, k);
        dwi[k] = (float) dw(a, k);
      }
      const BinGrid g = vis_atm.grid(q);
      Kokkos::single(Kokkos::PerTeam(team), [&]() { cnt(0) = 0; });
      team.team_barrier();
      team_visit<true>(team, vis_atm, g, q, [&](int p, int s0, int s1, int s2) {
        const int b = pm(p);
        float dx, dy, dz;
        bond(q, a, b, s0, s1, s2, dx, dy, dz);
        const float r2 = dx * dx + dy * dy + dz * dz;
        if (!(r2 < rc32 && r2 > 0.0f)) return;
        const int sj = type(b) - tb, nj = nref(sj);
        float wj[R], dwj[R];
        for (int k = 0; k < R; k++) {
          wj[k] = wf(b, k);
          dwj[k] = (float) dw(b, k);
        }
        float dq, db;
        const float c = d3::c6<float>(wi, dwi, ni, wj, dwj, nj, &c6ref((si * ns + sj) * R * R), dq, db);
        const int k = Kokkos::atomic_fetch_add(&cnt(0), 1);
        if (k >= cap) return;
        sb(k) = b;
        sf(k, X) = dx;
        sf(k, Y) = dy;
        sf(k, Z) = dz;
        sf(k, C) = c;
        sf(k, RQ) = dq / c;
        sf(k, RB) = db / c;
        for (int r = 0; r < R; r++) {
          sf(k, W + r) = wj[r];
          sf(k, DW + r) = dwj[r];
        }
        sd(k) = 0.0;
        for (int r = AD; r <= AZ; r++) sf(k, r) = 0.0f;
        const d3::AtmSide<float> sk = d3::atm_side<float>(r2, rc3f, ron3f);
        sf(k, SU) = sk.u;
        sf(k, SIU) = sk.iu;
        sf(k, SR) = sk.r;
        sf(k, SQ) = sk.q;
        sf(k, SSW) = sk.sw;
        sf(k, SDS) = sk.dsw;
      });
      team.team_barrier();
      // the m (m - 1) / 2 pairs (i, j): (i, i + d mod m) for d = 1 .. (m - 1) / 2, and (i, i + m / 2) for i < m / 2
      // with m even, consecutive threads on consecutive i
      const int m = cnt(0) < cap ? cnt(0) : cap, half = (m - 1) / 2, nsq = m * half;
      const int npair = nsq + (m % 2 == 0 ? m / 2 : 0);
      double ei = 0.0, dei = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
      double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, npair), [&](const int k) {
        int i = k < nsq ? k % m : k - nsq, j = k < nsq ? (i + 1 + k / m) % m : i + m / 2;
        // a fixed orientation of the pair (atom, then bond vector): the list order varies between runs (appended
        // atomically), the rounding of a triangle does not
        const bool swap = sb(i) != sb(j)         ? sb(i) > sb(j)
                          : sf(i, X) != sf(j, X) ? sf(i, X) > sf(j, X)
                          : sf(i, Y) != sf(j, Y) ? sf(i, Y) > sf(j, Y)
                                                 : sf(i, Z) > sf(j, Z);
        if (swap) {
          const int t = i;
          i = j;
          j = t;
        }
        const float ax = sf(i, X), ay = sf(i, Y), az = sf(i, Z), bx = sf(j, X), by = sf(j, Y), bz = sf(j, Z);
        const float ex = bx - ax, ey = by - ay, ez = bz - az;
        const float ujk = ex * ex + ey * ey + ez * ez;
        if (!(ujk < rc32 && ujk > 0.0f)) return;
        const int sj = type(sb(i)) - tb, sk = type(sb(j)) - tb;
        // C6 of the two neighbours from their weights and its CN derivatives on both sides
        const float *blk = &c6ref((sj * ns + sk) * R * R);
        float cjk = 0.0f, djj = 0.0f, djk = 0.0f;
        for (int r = 0; r < R; r++) {
          float bw = 0.0f, bd = 0.0f;
          for (int c = 0; c < R; c++) {
            bw += blk[r * R + c] * sf(j, W + c);
            bd += blk[r * R + c] * sf(j, DW + c);
          }
          cjk += sf(i, W + r) * bw;
          djj += sf(i, DW + r) * bw;
          djk += sf(i, W + r) * bd;
        }
        const float c9 = Kokkos::fabs(sf(i, C) * sf(j, C) * cjk);
        if (!(c9 > 0.0f)) return;
        const float cd3 = 6.0f * rvdw16(si * ns + sj) * rvdw16(si * ns + sk) * rvdw16(sj * ns + sk);
        float dg[3];
        const d3::AtmSide<float> si3 = {sf(i, SU), sf(i, SIU), sf(i, SR), sf(i, SQ), sf(i, SSW), sf(i, SDS)},
                                 sj3 = {sf(j, SU), sf(j, SIU), sf(j, SR), sf(j, SQ), sf(j, SSW), sf(j, SDS)};
        const float k9 = s9u * Kokkos::sqrt(c9),
                    e3 = k9 * d3::atm<float>(si3, sj3, d3::atm_side<float>(ujk, rc3f, ron3f), cd3,
                                             energy_only ? nullptr : dg);
        ei += (double) e3 / 3.0;
        Kokkos::atomic_add(&sd(i), (double) e3 / 3.0);
        Kokkos::atomic_add(&sd(j), (double) e3 / 3.0);
        if (energy_only) return;
        const float he = 0.5f * e3;
        const float hc = he / cjk;
        dei += (double) he * ((double) sf(i, RQ) + (double) sf(j, RQ));
        Kokkos::atomic_add(&sf(i, AD), he * sf(i, RB) + hc * djj);
        Kokkos::atomic_add(&sf(j, AD), he * sf(j, RB) + hc * djk);
        const float h0 = 2.0f * k9 * dg[0], h1 = 2.0f * k9 * dg[1], h2 = 2.0f * k9 * dg[2];
        fx += (double) (h0 * ax + h1 * bx);
        fy += (double) (h0 * ay + h1 * by);
        fz += (double) (h0 * az + h1 * bz);
        Kokkos::atomic_add(&sf(i, AX), h2 * ex - h0 * ax);
        Kokkos::atomic_add(&sf(i, AY), h2 * ey - h0 * ay);
        Kokkos::atomic_add(&sf(i, AZ), h2 * ez - h0 * az);
        Kokkos::atomic_add(&sf(j, AX), -(h1 * bx + h2 * ex));
        Kokkos::atomic_add(&sf(j, AY), -(h1 * by + h2 * ey));
        Kokkos::atomic_add(&sf(j, AZ), -(h1 * bz + h2 * ez));
        v0 -= (double) (h0 * ax * ax + h1 * bx * bx + h2 * ex * ex);
        v1 -= (double) (h0 * ay * ay + h1 * by * by + h2 * ey * ey);
        v2 -= (double) (h0 * az * az + h1 * bz * bz + h2 * ez * ez);
        v3 -= (double) (h0 * ax * ay + h1 * bx * by + h2 * ex * ey);
        v4 -= (double) (h0 * ax * az + h1 * bx * bz + h2 * ex * ez);
        v5 -= (double) (h0 * ay * az + h1 * by * bz + h2 * ey * ez);
      });
      team.team_reduce(Sum(ei));
      if (!energy_only) {
        team.team_reduce(Sum(dei));
        team.team_reduce(Sum(fx));
        team.team_reduce(Sum(fy));
        team.team_reduce(Sum(fz));
        team.team_reduce(Sum(v0));
        team.team_reduce(Sum(v1));
        team.team_reduce(Sum(v2));
        team.team_reduce(Sum(v3));
        team.team_reduce(Sum(v4));
        team.team_reduce(Sum(v5));
      }
      team.team_barrier();    // the scratch sums complete
      // the neighbours' shares (other teams add to the same atoms: atomics, also for the centre's own)
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, m), [&](const int k) {
        const int b = sb(k);
        Kokkos::atomic_add(&e(b), sd(k));
        if (energy_only) return;
        Kokkos::atomic_add(&dedcn(b), (double) sf(k, AD));
        Kokkos::atomic_add(&f(b, 0), (double) sf(k, AX));
        Kokkos::atomic_add(&f(b, 1), (double) sf(k, AY));
        Kokkos::atomic_add(&f(b, 2), (double) sf(k, AZ));
      });
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        Kokkos::atomic_add(&e(a), ei);
        if (energy_only) return;
        Kokkos::atomic_add(&dedcn(a), dei);
        Kokkos::atomic_add(&f(a, 0), fx);
        Kokkos::atomic_add(&f(a, 1), fy);
        Kokkos::atomic_add(&f(a, 2), fz);
        vir(a, 0) += v0;
        vir(a, 1) += v1;
        vir(a, 2) += v2;
        vir(a, 3) += v3;
        vir(a, 4) += v4;
        vir(a, 5) += v5;
      });
    });
  }
  if (energy_only) return;
  Kokkos::parallel_for("d3:cn_forces", policy, KOKKOS_LAMBDA(const Member &team) {
    const int q = team.league_rank(), a = pm(q), si = type(a) - tb;
    const double da = dedcn(a);
    const BinGrid g = vis_cn.grid(q);
    double fx = 0.0, fy = 0.0, fz = 0.0, v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0, v5 = 0.0;
    team_visit(team, vis_cn, g, q, [&](int p, int s0, int s1, int s2) {
      const int b = pm(p);
      float dx, dy, dz;
      bond(q, a, b, s0, s1, s2, dx, dy, dz);
      const float r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= rcn2 || r2 == 0.0f) return;
      const float r = Kokkos::sqrt(r2);
      float dfdr;
      d3::count<float>(rcov(si * ns + type(b) - tb), r, dfdr);
      const float h = (float) (da + dedcn(b)) * dfdr / r;
      fx += (double) (h * dx);
      fy += (double) (h * dy);
      fz += (double) (h * dz);
      v0 -= (double) (0.5f * h * dx * dx);
      v1 -= (double) (0.5f * h * dy * dy);
      v2 -= (double) (0.5f * h * dz * dz);
      v3 -= (double) (0.5f * h * dx * dy);
      v4 -= (double) (0.5f * h * dx * dz);
      v5 -= (double) (0.5f * h * dy * dz);
    });
    team.team_reduce(Sum(fx));
    team.team_reduce(Sum(fy));
    team.team_reduce(Sum(fz));
    team.team_reduce(Sum(v0));
    team.team_reduce(Sum(v1));
    team.team_reduce(Sum(v2));
    team.team_reduce(Sum(v3));
    team.team_reduce(Sum(v4));
    team.team_reduce(Sum(v5));
    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      f(a, 0) += fx;
      f(a, 1) += fy;
      f(a, 2) += fz;
      vir(a, 0) += v0;
      vir(a, 1) += v1;
      vir(a, 2) += v2;
      vir(a, 3) += v3;
      vir(a, 4) += v4;
      vir(a, 5) += v5;
    });
  });
}

// one structure. x [>= n][3]: current positions (caller's order); type [>= n]: species index + type_base (caller's
// order); nl: built from positions within skin / 2 of x (n > 0 atoms, same cell and pbc). Per-atom results in wk.e
// (and wk.f, wk.vir), sums returned (virial zero with energy_only).
template<class DeviceType>
D3Sums d3_compute(const D3TablesKK<DeviceType> &t, D3WorkKK<DeviceType> &wk,
                  const Kokkos::View<const double *[3], Kokkos::LayoutRight, DeviceType> &x,
                  const Kokkos::View<const int *, DeviceType> &type, int type_base, int n,
                  const NeighborListKK<DeviceType> &nl, double skin, bool energy_only)
{
  D3Sums out;
  if (n == 0) return out;
  wk.ensure(n);
  const auto xr = wk.xr;
  const auto im = nl.img;
  const Mat3 H = nl.grid_H;
  Kokkos::parallel_for("d3:xr", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int a) {
    for (int d = 0; d < 3; d++)
      xr(a, d) = x(a, d) - (im(a, 0) * H.a[d] + im(a, 1) * H.a[3 + d] + im(a, 2) * H.a[6 + d]);
  });
  const auto vrc = nl.visitor(t.p.cutoff + skin);
  d3_passes(t, wk, type, type_base, n, nl.visitor(t.p.cn_cutoff + skin), vrc,
            t.p.s9 != 0.0 ? nl.visitor(t.p.atm_cutoff + skin) : vrc, D3CellOne(nl.grid_H), nl.perm, energy_only);
  d3_sums(wk, 1, D3OneRange{n}, energy_only);
  double h[7];
  Kokkos::deep_copy(Kokkos::View<double[7], Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(h),
                    Kokkos::subview(wk.sums, 0, Kokkos::ALL));
  out.energy = h[0];
  std::copy(h + 1, h + 7, out.virial);
  return out;
}

// a batch: nl built from the current positions x (all structures, caller's order; bins_only is enough); per-atom
// results in wk, per structure in wk.sums [nsys][7]
template<class DeviceType>
void d3_compute_batch(const D3TablesKK<DeviceType> &t, D3WorkKK<DeviceType> &wk,
                      const Kokkos::View<const double *[3], Kokkos::LayoutRight, DeviceType> &x,
                      const Kokkos::View<const int *, DeviceType> &type, int type_base,
                      const BatchNeighborListKK<DeviceType> &nl, bool energy_only)
{
  if (nl.n == 0) return;
  wk.ensure(nl.n);
  if ((int) wk.cells.extent(0) < nl.nsys) wk.cells = decltype(wk.cells)("d3:cells", nl.nsys + nl.nsys / 8 + 1);
  const auto hf = wk.cells;
  const auto hd = nl.H;
  Kokkos::parallel_for("d3:cells", Kokkos::RangePolicy<DeviceType>(0, nl.nsys), KOKKOS_LAMBDA(const int s) {
    for (int k = 0; k < 9; k++) hf(s, k) = (float) hd(s, k);
  });
  const auto xr = wk.xr;
  const auto im = nl.img;
  const auto si = nl.sys_in;
  Kokkos::parallel_for("d3:xr", Kokkos::RangePolicy<DeviceType>(0, nl.n), KOKKOS_LAMBDA(const int a) {
    const int s = si(a);
    for (int d = 0; d < 3; d++)
      xr(a, d) = x(a, d) - (im(a, 0) * hd(s, d) + im(a, 1) * hd(s, 3 + d) + im(a, 2) * hd(s, 6 + d));
  });
  const auto vrc = nl.visitor(t.p.cutoff);
  d3_passes(t, wk, type, type_base, nl.n, nl.visitor(t.p.cn_cutoff), vrc,
            t.p.s9 != 0.0 ? nl.visitor(t.p.atm_cutoff) : vrc, D3CellBatch<DeviceType>{nl.sys, hf}, nl.perm,
            energy_only);
  d3_sums(wk, nl.nsys, nl.start_d, energy_only);
}

}    // namespace kk
}    // namespace grace

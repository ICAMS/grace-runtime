// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Cell-list neighbour search on the device: the algorithm of the CPU list of grace_runtime (python/src/neighbors.cpp)
// with the same bins, the same stable spatial order of the atoms and the same bonds (i, j, S) in the same row order:
// fractional coordinates, wrapped in periodic directions, bins of edge >= cutoff / 2 over the cell (periodic) or the
// occupied range (open), reach ceil(cutoff / bin width), counting pass + scan + fill pass. Output in the new order:
// perm (new -> caller index), CSR offset / j, and per bond the lattice translation S in the caller's frame,
// r_ij = x[perm j] - x[perm i] + S cell.
#pragma once

#include "grace/cell_list.h"
#include "grace/error.h"
#include "grace/kk/support.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace grace {
namespace kk {

struct Mat3 {
  double a[9];
};

// bin grid of the structure of one atom: bins per direction, periodic flags, reach in bins, first bin (batches)
struct BinGrid {
  int64_t nbk[3], r[3], base;
  int pp[3];
};

// every neighbour of new atom q within the cutoff, fn(p, S0, S1, S2) with S the image in the wrapped frame; the loop
// order of the CPU list (python/src/neighbors.cpp visit)
template<class DeviceType> struct NeighborVisitor {
  Kokkos::View<int *, DeviceType> pm;
  Kokkos::View<int *[3], Kokkos::LayoutRight, DeviceType> bb;
  Kokkos::View<int64_t *, DeviceType> bs;
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> x;
  Mat3 H;
  int64_t nbk[3];
  int pp[3];
  int64_t r[3];
  double rc2;

  KOKKOS_INLINE_FUNCTION static int64_t fdiv(int64_t v, int64_t d) { return v >= 0 ? v / d : -((-v + d - 1) / d); }

  KOKKOS_INLINE_FUNCTION BinGrid grid(int) const
  {
    BinGrid g;
    for (int k = 0; k < 3; k++) {
      g.nbk[k] = nbk[k];
      g.r[k] = r[k];
      g.pp[k] = pp[k];
    }
    g.base = 0;
    return g;
  }

  template<class F> KOKKOS_INLINE_FUNCTION void operator()(const int q, F &&fn) const
  {
    const int a = pm(q);
    for (int64_t o0 = -r[0]; o0 <= r[0]; o0++) {
      int64_t c0 = bb(a, 0) + o0, s0 = 0;
      if (pp[0]) { s0 = fdiv(c0, nbk[0]); c0 -= s0 * nbk[0]; }
      else if (c0 < 0 || c0 >= nbk[0]) continue;
      for (int64_t o1 = -r[1]; o1 <= r[1]; o1++) {
        int64_t c1 = bb(a, 1) + o1, s1 = 0;
        if (pp[1]) { s1 = fdiv(c1, nbk[1]); c1 -= s1 * nbk[1]; }
        else if (c1 < 0 || c1 >= nbk[1]) continue;
        for (int64_t o2 = -r[2]; o2 <= r[2]; o2++) {
          int64_t c2 = bb(a, 2) + o2, s2 = 0;
          if (pp[2]) { s2 = fdiv(c2, nbk[2]); c2 -= s2 * nbk[2]; }
          else if (c2 < 0 || c2 >= nbk[2]) continue;
          const double sx = s0 * H.a[0] + s1 * H.a[3] + s2 * H.a[6];
          const double sy = s0 * H.a[1] + s1 * H.a[4] + s2 * H.a[7];
          const double sz = s0 * H.a[2] + s1 * H.a[5] + s2 * H.a[8];
          const int64_t c = (c0 * nbk[1] + c1) * nbk[2] + c2;
          for (int64_t p = bs(c); p < bs(c + 1); p++) {
            if (p == q && s0 == 0 && s1 == 0 && s2 == 0) continue;
            const double dx = x(p, 0) + sx - x(q, 0);
            const double dy = x(p, 1) + sy - x(q, 1);
            const double dz = x(p, 2) + sz - x(q, 2);
            if (dx * dx + dy * dy + dz * dz < rc2) fn((int) p, (int) s0, (int) s1, (int) s2);
          }
        }
      }
    }
  }
};

template<class DeviceType> struct NeighborListKK {
  using V1i = Kokkos::View<int *, DeviceType>;
  using V3i = Kokkos::View<int *[3], Kokkos::LayoutRight, DeviceType>;
  using V3d = Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType>;
  int n = 0, maxneigh = 0;
  int64_t nb = 0;
  V1i perm;                                  // [n] new index -> caller index
  Kokkos::View<int64_t *, DeviceType> offset;    // [n + 1]
  V1i j;                                     // [nb] neighbour (new index)
  V3i S;                                     // [nb] lattice translation, caller's frame
  // scratch
  V3d frac, xw;
  V3i img, b3;
  Kokkos::View<int64_t *, DeviceType> key;
  Kokkos::View<int64_t *, DeviceType> bin_start;
  Kokkos::View<int64_t *, DeviceType> nn;
  // bin grid of the last build, for visitor()
  Mat3 grid_H{};
  int64_t grid_nbin[3] = {1, 1, 1};
  double grid_width[3] = {1.0, 1.0, 1.0};
  int grid_pbc[3] = {0, 0, 0};

  // every neighbour within any radius rc of the positions of the last build (wrapped xw, images in the wrapped frame;
  // img converts them to the caller's frame), over the bins of the last build; n > 0
  NeighborVisitor<DeviceType> visitor(double rc) const
  {
    int64_t r[3];
    for (int k = 0; k < 3; k++) r[k] = bin_reach(rc, grid_width[k], grid_nbin[k], grid_pbc[k]);
    return NeighborVisitor<DeviceType>{perm, b3, bin_start, xw, grid_H, {grid_nbin[0], grid_nbin[1], grid_nbin[2]},
                                       {grid_pbc[0], grid_pbc[1], grid_pbc[2]}, {r[0], r[1], r[2]}, rc * rc};
  }

  // pos [>= n][3] (device, caller's order), cell rows = vectors (host, full rank), pbc (host)
  void build(const Kokkos::View<const double *[3], Kokkos::LayoutRight, DeviceType> &pos, int n_, const double *cell,
             const bool *pbc, double cutoff)
  {
    if (!(cutoff > 0.0) || !std::isfinite(cutoff)) throw Error("neighbour search: cutoff must be positive");
    n = n_;
    nb = 0;
    maxneigh = 0;
    if ((int) perm.extent(0) < n || n == 0) {
      const int m = std::max(n, 1);
      perm = V1i("nl:perm", m);
      frac = V3d("nl:frac", m);
      xw = V3d("nl:xw", m);
      img = V3i("nl:img", m);
      b3 = V3i("nl:b3", m);
      key = Kokkos::View<int64_t *, DeviceType>("nl:key", m);
      nn = Kokkos::View<int64_t *, DeviceType>("nl:nn", m);
    }
    offset = Kokkos::View<int64_t *, DeviceType>("nl:offset", n + 1);
    if (n == 0) return;
    // frac = x inv(cell)
    Mat3 inv, H;
    double det;
    {
      const double *a = cell;
      det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
      if (!full_rank_cell(a, det)) throw Error("neighbour search needs a full-rank cell");
      const double d = 1.0 / det;
      inv.a[0] = (a[4] * a[8] - a[5] * a[7]) * d; inv.a[1] = (a[2] * a[7] - a[1] * a[8]) * d;
      inv.a[2] = (a[1] * a[5] - a[2] * a[4]) * d; inv.a[3] = (a[5] * a[6] - a[3] * a[8]) * d;
      inv.a[4] = (a[0] * a[8] - a[2] * a[6]) * d; inv.a[5] = (a[2] * a[3] - a[0] * a[5]) * d;
      inv.a[6] = (a[3] * a[7] - a[4] * a[6]) * d; inv.a[7] = (a[1] * a[6] - a[0] * a[7]) * d;
      inv.a[8] = (a[0] * a[4] - a[1] * a[3]) * d;
      for (int k = 0; k < 9; k++) H.a[k] = a[k];
    }
    double h[3];
    for (int k = 0; k < 3; k++)
      h[k] = 1.0 / std::sqrt(inv.a[k] * inv.a[k] + inv.a[3 + k] * inv.a[3 + k] + inv.a[6 + k] * inv.a[6 + k]);
    const int p0 = pbc[0], p1 = pbc[1], p2 = pbc[2];
    auto fr = frac;
    auto im = img;
    int bad = 0;
    Kokkos::parallel_reduce("nl:frac", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int a, int &nbad) {
      const int pp[3] = {p0, p1, p2};
      for (int k = 0; k < 3; k++) {
        const double f = pos(a, 0) * inv.a[k] + pos(a, 1) * inv.a[3 + k] + pos(a, 2) * inv.a[6 + k];
        if (!Kokkos::isfinite(f)) nbad++;
        const double fl = pp[k] ? Kokkos::floor(f) : 0.0;
        im(a, k) = (int) fl;
        fr(a, k) = f - fl;
      }
    }, bad);
    if (bad) throw Error("positions must be finite");
    double lo[3] = {0, 0, 0}, span[3] = {1, 1, 1};
    for (int k = 0; k < 3; k++)
      if (!pbc[k]) {    // open direction: bins over the occupied range
        double mn, mx;
        Kokkos::parallel_reduce("nl:min", Kokkos::RangePolicy<DeviceType>(0, n),
            KOKKOS_LAMBDA(const int a, double &v) { if (fr(a, k) < v) v = fr(a, k); }, Kokkos::Min<double>(mn));
        Kokkos::parallel_reduce("nl:max", Kokkos::RangePolicy<DeviceType>(0, n),
            KOKKOS_LAMBDA(const int a, double &v) { if (fr(a, k) > v) v = fr(a, k); }, Kokkos::Max<double>(mx));
        lo[k] = mn;
        span[k] = std::max(mx - mn, 1e-12);
      }
    int64_t nbin[3];
    for (int k = 0; k < 3; k++) nbin[k] = std::max<int64_t>(1, (int64_t) std::floor(span[k] * h[k] / (0.5 * cutoff)));
    while (nbin[0] * nbin[1] * nbin[2] > 8 * (int64_t) n + 64) {
      const int k = (int) (std::max_element(nbin, nbin + 3) - nbin);
      nbin[k] = std::max<int64_t>(1, nbin[k] / 2);
    }
    int64_t reach[3];
    for (int k = 0; k < 3; k++) {
      const double w = span[k] * h[k] / (double) nbin[k];
      reach[k] = bin_reach(cutoff, w, nbin[k], pbc[k]);
      grid_nbin[k] = nbin[k];
      grid_width[k] = w;
      grid_pbc[k] = pbc[k] ? 1 : 0;
    }
    grid_H = H;
    const int64_t nbt = nbin[0] * nbin[1] * nbin[2];
    const int64_t nb0 = nbin[0], nb1 = nbin[1], nb2 = nbin[2];
    const double lo0 = lo[0], lo1 = lo[1], lo2 = lo[2], sp0 = span[0], sp1 = span[1], sp2 = span[2];
    // bins; stable order by bin: sort of bin * n + a
    auto bb = b3;
    auto ky = key;
    const int64_t nn_ = n;
    Kokkos::parallel_for("nl:bin", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int a) {
      const double l[3] = {lo0, lo1, lo2}, s[3] = {sp0, sp1, sp2};
      const int64_t nbk[3] = {nb0, nb1, nb2};
      int64_t c[3];
      for (int k = 0; k < 3; k++) {
        int64_t b = (int64_t) Kokkos::floor((fr(a, k) - l[k]) / s[k] * (double) nbk[k]);
        b = b < 0 ? 0 : (b > nbk[k] - 1 ? nbk[k] - 1 : b);
        bb(a, k) = (int) b;
        c[k] = b;
      }
      ky(a) = ((c[0] * nb1 + c[1]) * nb2 + c[2]) * nn_ + a;
    });
    Kokkos::sort(Kokkos::subview(key, std::make_pair(0, n)));
    bin_start = Kokkos::View<int64_t *, DeviceType>("nl:bin_start", nbt + 1);
    auto bs = bin_start;
    auto pm = perm;
    Kokkos::parallel_for("nl:perm", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int q) {
      pm(q) = (int) (ky(q) % nn_);
      Kokkos::atomic_add(&bs(ky(q) / nn_ + 1), (int64_t) 1);
    });
    Kokkos::parallel_scan("nl:bin_scan", Kokkos::RangePolicy<DeviceType>(0, nbt + 1),
        KOKKOS_LAMBDA(const int64_t c, int64_t &acc, const bool fin) {
      acc += bs(c);
      if (fin) bs(c) = acc;
    });
    // wrapped Cartesian positions in the new order
    auto x = xw;
    Kokkos::parallel_for("nl:xw", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int q) {
      const int a = pm(q);
      for (int d = 0; d < 3; d++) x(q, d) = fr(a, 0) * H.a[d] + fr(a, 1) * H.a[3 + d] + fr(a, 2) * H.a[6 + d];
    });
    const double rc2 = cutoff * cutoff;
    const int64_t r0 = reach[0], r1 = reach[1], r2 = reach[2];
    const NeighborVisitor<DeviceType> visit{pm, bb, bs, x, H, {nb0, nb1, nb2}, {p0, p1, p2}, {r0, r1, r2}, rc2};
    auto cnt = nn;
    int zero = 0;    // pairs at distance 0 (two atoms, or an atom and an image, at the same position): an error
    Kokkos::parallel_reduce("nl:count", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int q, int &z) {
      int64_t c = 0;
      visit(q, [&](int p, int s0, int s1, int s2) {
        c++;
        double d2 = 0.0;
        for (int k = 0; k < 3; k++) {
          const double d = x(p, k) + s0 * H.a[k] + s1 * H.a[3 + k] + s2 * H.a[6 + k] - x(q, k);
          d2 += d * d;
        }
        if (d2 == 0.0) z++;
      });
      cnt(q) = c;
    }, zero);
    if (zero) throw Error("neighbour search: two atoms at the same position (a bond of length 0): not a physical structure");
    auto off = offset;
    int64_t total = 0;
    Kokkos::parallel_scan("nl:offset", Kokkos::RangePolicy<DeviceType>(0, n),
        KOKKOS_LAMBDA(const int q, int64_t &acc, const bool fin) {
      if (fin) off(q) = acc;
      acc += cnt(q);
      if (fin && q == nn_ - 1) off(q + 1) = acc;
    }, total);
    nb = total;
    Kokkos::parallel_reduce("nl:maxneigh", Kokkos::RangePolicy<DeviceType>(0, n),
        KOKKOS_LAMBDA(const int q, int &m) { if (cnt(q) > m) m = (int) cnt(q); }, Kokkos::Max<int>(maxneigh));
    maxneigh = std::max(maxneigh, 0);    // Kokkos::Max over no atoms gives INT_MIN
    if ((int64_t) j.extent(0) < nb) {
      j = V1i("nl:j", nb + nb / 8 + 16);
      S = V3i("nl:S", nb + nb / 8 + 16);
    }
    auto jj = j;
    auto SS = S;
    Kokkos::parallel_for("nl:fill", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int q) {
      int64_t w = off(q);
      const int a = pm(q);
      visit(q, [&](int p, int s0, int s1, int s2) {
        const int b = pm(p);
        // caller's (unwrapped) frame: S = S_wrapped + n_a - n_b
        jj(w) = p;
        SS(w, 0) = s0 - im(b, 0) + im(a, 0);
        SS(w, 1) = s1 - im(b, 1) + im(a, 1);
        SS(w, 2) = s2 - im(b, 2) + im(a, 2);
        w++;
      });
    });
  }
};

// every neighbour of new atom q of a batch (BatchNeighborListKK): the loop of NeighborVisitor over the bins of q's
// structure, its cell and grid read from per-structure views
template<class DeviceType> struct BatchNeighborVisitor {
  Kokkos::View<int *, DeviceType> pm, sy;
  Kokkos::View<int *[3], Kokkos::LayoutRight, DeviceType> bb;
  Kokkos::View<int64_t *, DeviceType> bs, base;
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> x;
  Kokkos::View<double *[9], Kokkos::LayoutRight, DeviceType> H;
  Kokkos::View<int64_t *[3], Kokkos::LayoutRight, DeviceType> nbk, r;
  Kokkos::View<int *[3], Kokkos::LayoutRight, DeviceType> pp;
  double rc2;

  KOKKOS_INLINE_FUNCTION BinGrid grid(int q) const
  {
    const int s = sy(q);
    BinGrid g;
    for (int k = 0; k < 3; k++) {
      g.nbk[k] = nbk(s, k);
      g.r[k] = r(s, k);
      g.pp[k] = pp(s, k);
    }
    g.base = base(s);
    return g;
  }

  template<class F> KOKKOS_INLINE_FUNCTION void operator()(const int q, F &&fn) const
  {
    using NV = NeighborVisitor<DeviceType>;
    const int a = pm(q), s = sy(q);
    const int64_t n0 = nbk(s, 0), n1 = nbk(s, 1), n2 = nbk(s, 2), b0 = base(s);
    for (int64_t o0 = -r(s, 0); o0 <= r(s, 0); o0++) {
      int64_t c0 = bb(a, 0) + o0, s0 = 0;
      if (pp(s, 0)) { s0 = NV::fdiv(c0, n0); c0 -= s0 * n0; }
      else if (c0 < 0 || c0 >= n0) continue;
      for (int64_t o1 = -r(s, 1); o1 <= r(s, 1); o1++) {
        int64_t c1 = bb(a, 1) + o1, s1 = 0;
        if (pp(s, 1)) { s1 = NV::fdiv(c1, n1); c1 -= s1 * n1; }
        else if (c1 < 0 || c1 >= n1) continue;
        for (int64_t o2 = -r(s, 2); o2 <= r(s, 2); o2++) {
          int64_t c2 = bb(a, 2) + o2, s2 = 0;
          if (pp(s, 2)) { s2 = NV::fdiv(c2, n2); c2 -= s2 * n2; }
          else if (c2 < 0 || c2 >= n2) continue;
          const double sx = s0 * H(s, 0) + s1 * H(s, 3) + s2 * H(s, 6);
          const double sy_ = s0 * H(s, 1) + s1 * H(s, 4) + s2 * H(s, 7);
          const double sz = s0 * H(s, 2) + s1 * H(s, 5) + s2 * H(s, 8);
          const int64_t c = b0 + (c0 * n1 + c1) * n2 + c2;
          for (int64_t p = bs(c); p < bs(c + 1); p++) {
            if (p == q && s0 == 0 && s1 == 0 && s2 == 0) continue;
            const double dx = x(p, 0) + sx - x(q, 0);
            const double dy = x(p, 1) + sy_ - x(q, 1);
            const double dz = x(p, 2) + sz - x(q, 2);
            if (dx * dx + dy * dy + dz * dz < rc2) fn((int) p, (int) s0, (int) s1, (int) s2);
          }
        }
      }
    }
  }
};

// The search of NeighborListKK for a batch of structures, atoms concatenated structure by structure (start [nsys + 1],
// host): one bin grid per structure (its cell, pbc and occupied range), all grids in one sorted key array, so the new
// order keeps the structures contiguous and in input order (new atoms [start[s], start[s + 1]) are structure s, each in
// the spatial order NeighborListKK gives it alone) and bonds stay inside a structure. Outputs as NeighborListKK plus
// sys (structure of each new atom) and H (cell rows of each structure, device) for the images.
template<class DeviceType> struct BatchNeighborListKK {
  using V1i = Kokkos::View<int *, DeviceType>;
  using V3i = Kokkos::View<int *[3], Kokkos::LayoutRight, DeviceType>;
  using V3d = Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType>;
  using V3l = Kokkos::View<int64_t *[3], Kokkos::LayoutRight, DeviceType>;
  using V9d = Kokkos::View<double *[9], Kokkos::LayoutRight, DeviceType>;
  using V1l = Kokkos::View<int64_t *, DeviceType>;
  int n = 0, nsys = 0, maxneigh = 0;
  int64_t nb = 0;
  V1i perm;    // [n] new index -> caller index
  V1i sys;     // [n] structure of each new atom
  V1l offset;  // [n + 1]
  V1i j;       // [nb] neighbour (new index)
  V3i S;       // [nb] lattice translation, caller's frame
  V9d H;       // [nsys] cell rows
  // scratch
  V9d inv;
  V3d frac, xw, lo, span, mn, mx;
  V3i img, b3, pp;
  V3l nbk, reach;
  V1i sys_in;
  V1l start_d, base, key, bin_start, nn;
  // bin grids of the last build per structure (host), for visitor()
  std::vector<double> grid_width;     // [nsys][3]
  std::vector<int64_t> grid_nbin;     // [nsys][3]
  std::vector<int> grid_pbc;          // [nsys][3]

  struct ReachEntry {
    double rc = 0.0;
    uint64_t gen = 0;
    V3l r;
  };
  uint64_t build_gen = 0;
  mutable std::vector<ReachEntry> reach_rc;    // reach views of visitor(), kept per radius

  // every neighbour within any radius rc of the positions of the last build (as NeighborListKK::visitor), with its
  // own reach per structure; n > 0
  BatchNeighborVisitor<DeviceType> visitor(double rc) const
  {
    auto it = std::find_if(reach_rc.begin(), reach_rc.end(), [&](const auto &e) { return e.rc == rc; });
    if (it == reach_rc.end() || (int) it->r.extent(0) < nsys) {
      if (it != reach_rc.end()) reach_rc.erase(it);
      reach_rc.push_back({rc, 0, V3l("bnl:reach_rc", std::max(nsys, 1))});
      it = reach_rc.end() - 1;
    }
    const V3l r = it->r;
    if (it->gen != build_gen) {
      auto hr = Kokkos::create_mirror_view(r);
      for (int s = 0; s < nsys; s++)
        for (int k = 0; k < 3; k++) {
          const size_t g = 3 * (size_t) s + k;
          hr(s, k) = bin_reach(rc, grid_width[g], grid_nbin[g], grid_pbc[g]);
        }
      Kokkos::deep_copy(r, hr);
      it->gen = build_gen;
    }
    return BatchNeighborVisitor<DeviceType>{perm, sys, b3, bin_start, base, xw, H, nbk, r, pp, rc * rc};
  }

  // pos [>= start[nsys]][3] (device, caller's order); start [nsys + 1], cell [nsys][9] rows = vectors (full rank),
  // pbc [nsys][3]: host. bins_only: the bins, order and images only, no CSR list (for visitor())
  void build(const Kokkos::View<const double *[3], Kokkos::LayoutRight, DeviceType> &pos, int nsys_,
             const int64_t *start, const double *cell, const bool *pbc, double cutoff, bool bins_only = false)
  {
    if (!(cutoff > 0.0) || !std::isfinite(cutoff)) throw Error("neighbour search: cutoff must be positive");
    if (nsys_ < 0) throw Error("neighbour search: negative number of structures");
    ++build_gen;
    nsys = nsys_;
    n = (int) start[nsys];
    nb = 0;
    maxneigh = 0;
    const int ms = std::max(nsys, 1), m = std::max(n, 1);
    if ((int) perm.extent(0) < n || perm.extent(0) == 0) {
      const int g = m + m / 8;
      perm = V1i("bnl:perm", g);
      sys = V1i("bnl:sys", g);
      sys_in = V1i("bnl:sys_in", g);
      frac = V3d("bnl:frac", g);
      xw = V3d("bnl:xw", g);
      img = V3i("bnl:img", g);
      b3 = V3i("bnl:b3", g);
      key = V1l("bnl:key", g);
      nn = V1l("bnl:nn", g);
    }
    if ((int) H.extent(0) < ms) {
      const int g = ms + ms / 8;
      H = V9d("bnl:H", g);
      inv = V9d("bnl:inv", g);
      lo = V3d("bnl:lo", g);
      span = V3d("bnl:span", g);
      mn = V3d("bnl:mn", g);
      mx = V3d("bnl:mx", g);
      pp = V3i("bnl:pbc", g);
      nbk = V3l("bnl:nbin", g);
      reach = V3l("bnl:reach", g);
      start_d = V1l("bnl:start", g + 1);
      base = V1l("bnl:base", g + 1);
    }
    offset = V1l("bnl:offset", n + 1);
    if (n == 0) return;
    // per structure on the host: inverse, plane spacings h (|rows of inv^T|^-1)
    auto hH = Kokkos::create_mirror_view(H);
    auto hinv = Kokkos::create_mirror_view(inv);
    auto hpp = Kokkos::create_mirror_view(pp);
    auto hst = Kokkos::create_mirror_view(start_d);
    std::vector<double> hsp(3 * (size_t) nsys);
    bool any_open = false;
    for (int s = 0; s < nsys; s++) {
      const double *a = cell + 9 * (size_t) s;
      const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                         a[2] * (a[3] * a[7] - a[4] * a[6]);
      if (start[s + 1] > start[s] && !full_rank_cell(a, det)) throw Error("neighbour search needs a full-rank cell");
      const double d = std::abs(det) > 0.0 ? 1.0 / det : 0.0;
      double v[9] = {(a[4] * a[8] - a[5] * a[7]) * d, (a[2] * a[7] - a[1] * a[8]) * d, (a[1] * a[5] - a[2] * a[4]) * d,
                     (a[5] * a[6] - a[3] * a[8]) * d, (a[0] * a[8] - a[2] * a[6]) * d, (a[2] * a[3] - a[0] * a[5]) * d,
                     (a[3] * a[7] - a[4] * a[6]) * d, (a[1] * a[6] - a[0] * a[7]) * d, (a[0] * a[4] - a[1] * a[3]) * d};
      for (int k = 0; k < 9; k++) {
        hH(s, k) = a[k];
        hinv(s, k) = v[k];
      }
      for (int k = 0; k < 3; k++) {
        const double r2 = v[k] * v[k] + v[3 + k] * v[3 + k] + v[6 + k] * v[6 + k];
        hsp[3 * s + k] = r2 > 0.0 ? 1.0 / std::sqrt(r2) : 1.0;
        hpp(s, k) = pbc[3 * (size_t) s + k] ? 1 : 0;
        if (!hpp(s, k) && start[s + 1] > start[s]) any_open = true;
      }
    }
    for (int s = 0; s <= nsys; s++) hst(s) = start[s];
    Kokkos::deep_copy(H, hH);
    Kokkos::deep_copy(inv, hinv);
    Kokkos::deep_copy(pp, hpp);
    Kokkos::deep_copy(start_d, hst);
    // structure of each caller atom (binary search in start), fractional coordinates, wrapped where periodic
    auto st = start_d;
    auto si = sys_in;
    auto iv = inv;
    auto ppv = pp;
    auto fr = frac;
    auto im = img;
    const int ns = nsys;
    int bad = 0;
    Kokkos::parallel_reduce("bnl:frac", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int a, int &nbad) {
      int lo_ = 0, hi_ = ns;    // largest s with start[s] <= a
      while (hi_ - lo_ > 1) {
        const int mid = (lo_ + hi_) / 2;
        if (st(mid) <= a) lo_ = mid;
        else hi_ = mid;
      }
      si(a) = lo_;
      for (int k = 0; k < 3; k++) {
        const double f = pos(a, 0) * iv(lo_, k) + pos(a, 1) * iv(lo_, 3 + k) + pos(a, 2) * iv(lo_, 6 + k);
        if (!Kokkos::isfinite(f)) nbad++;
        const double fl = ppv(lo_, k) ? Kokkos::floor(f) : 0.0;
        im(a, k) = (int) fl;
        fr(a, k) = f - fl;
      }
    }, bad);
    if (bad) throw Error("positions must be finite");
    // open directions: bins over each structure's occupied range
    std::vector<double> hmn(3 * (size_t) nsys, 0.0), hmx(3 * (size_t) nsys, 1.0);
    if (any_open) {
      auto mnv = mn;
      auto mxv = mx;
      Kokkos::parallel_for("bnl:range_init", Kokkos::RangePolicy<DeviceType>(0, nsys), KOKKOS_LAMBDA(const int s) {
        for (int k = 0; k < 3; k++) {
          mnv(s, k) = 1e300;
          mxv(s, k) = -1e300;
        }
      });
      Kokkos::parallel_for("bnl:range", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int a) {
        const int s = si(a);
        for (int k = 0; k < 3; k++)
          if (!ppv(s, k)) {
            Kokkos::atomic_min(&mnv(s, k), fr(a, k));
            Kokkos::atomic_max(&mxv(s, k), fr(a, k));
          }
      });
      auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mn);
      auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), mx);
      for (int s = 0; s < nsys; s++)
        for (int k = 0; k < 3; k++)
          if (!hpp(s, k) && start[s + 1] > start[s]) {
            hmn[3 * s + k] = hm(s, k);
            hmx[3 * s + k] = hx(s, k);
          }
    }
    // grids (the rules of NeighborListKK per structure) and their offsets in the global bin array
    auto hlo = Kokkos::create_mirror_view(lo);
    auto hspan = Kokkos::create_mirror_view(span);
    auto hnb = Kokkos::create_mirror_view(nbk);
    auto hre = Kokkos::create_mirror_view(reach);
    auto hbase = Kokkos::create_mirror_view(base);
    int64_t total_bins = 0;
    grid_width.assign(3 * (size_t) nsys, 1.0);
    grid_nbin.assign(3 * (size_t) nsys, 1);
    grid_pbc.assign(3 * (size_t) nsys, 0);
    for (int s = 0; s < nsys; s++) {
      const int64_t ns_at = start[s + 1] - start[s];
      int64_t nbin[3];
      for (int k = 0; k < 3; k++) {
        const bool open = !hpp(s, k);
        hlo(s, k) = open ? hmn[3 * s + k] : 0.0;
        hspan(s, k) = open ? std::max(hmx[3 * s + k] - hmn[3 * s + k], 1e-12) : 1.0;
        nbin[k] = std::max<int64_t>(1, (int64_t) std::floor(hspan(s, k) * hsp[3 * s + k] / (0.5 * cutoff)));
      }
      while (nbin[0] * nbin[1] * nbin[2] > 8 * ns_at + 64) {
        const int k = (int) (std::max_element(nbin, nbin + 3) - nbin);
        nbin[k] = std::max<int64_t>(1, nbin[k] / 2);
      }
      for (int k = 0; k < 3; k++) {
        const double w = hspan(s, k) * hsp[3 * s + k] / (double) nbin[k];
        hnb(s, k) = nbin[k];
        hre(s, k) = bin_reach(cutoff, w, nbin[k], hpp(s, k));
        grid_width[3 * (size_t) s + k] = w;
        grid_nbin[3 * (size_t) s + k] = nbin[k];
        grid_pbc[3 * (size_t) s + k] = hpp(s, k);
      }
      hbase(s) = total_bins;
      total_bins += nbin[0] * nbin[1] * nbin[2];
    }
    hbase(nsys) = total_bins;
    if (total_bins > (int64_t) 1 << 40 || (double) total_bins * (double) n > 9.0e18)
      throw Error("neighbour search: too many bins for one batch");
    Kokkos::deep_copy(lo, hlo);
    Kokkos::deep_copy(span, hspan);
    Kokkos::deep_copy(nbk, hnb);
    Kokkos::deep_copy(reach, hre);
    Kokkos::deep_copy(base, hbase);
    // bins; stable order by (structure's bin offset + bin, atom)
    auto bb = b3;
    auto ky = key;
    auto lov = lo;
    auto spv = span;
    auto nbv = nbk;
    auto bsv = base;
    const int64_t nn_ = n;
    Kokkos::parallel_for("bnl:bin", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int a) {
      const int s = si(a);
      int64_t c[3];
      for (int k = 0; k < 3; k++) {
        int64_t b = (int64_t) Kokkos::floor((fr(a, k) - lov(s, k)) / spv(s, k) * (double) nbv(s, k));
        b = b < 0 ? 0 : (b > nbv(s, k) - 1 ? nbv(s, k) - 1 : b);
        bb(a, k) = (int) b;
        c[k] = b;
      }
      ky(a) = (bsv(s) + (c[0] * nbv(s, 1) + c[1]) * nbv(s, 2) + c[2]) * nn_ + a;
    });
    Kokkos::sort(Kokkos::subview(key, std::make_pair(0, n)));
    bin_start = V1l("bnl:bin_start", total_bins + 1);
    auto bs = bin_start;
    auto pm = perm;
    auto sy = sys;
    Kokkos::parallel_for("bnl:perm", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int q) {
      const int a = (int) (ky(q) % nn_);
      pm(q) = a;
      sy(q) = si(a);
      Kokkos::atomic_add(&bs(ky(q) / nn_ + 1), (int64_t) 1);
    });
    Kokkos::parallel_scan("bnl:bin_scan", Kokkos::RangePolicy<DeviceType>(0, total_bins + 1),
        KOKKOS_LAMBDA(const int64_t c, int64_t &acc, const bool fin) {
      acc += bs(c);
      if (fin) bs(c) = acc;
    });
    // wrapped Cartesian positions in the new order
    auto x = xw;
    auto HH = H;
    Kokkos::parallel_for("bnl:xw", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int q) {
      const int a = pm(q), s = sy(q);
      for (int d = 0; d < 3; d++) x(q, d) = fr(a, 0) * HH(s, d) + fr(a, 1) * HH(s, 3 + d) + fr(a, 2) * HH(s, 6 + d);
    });
    if (bins_only) return;
    const BatchNeighborVisitor<DeviceType> visit{pm, sy, bb, bs, base, x, H, nbk, reach, pp, cutoff * cutoff};
    auto cnt = nn;
    int zero = 0;    // pairs at distance 0 (two atoms, or an atom and an image, at the same position): an error
    Kokkos::parallel_reduce("bnl:count", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int q, int &z) {
      int64_t c = 0;
      const int sq = sy(q);
      visit(q, [&](int p, int s0, int s1, int s2) {
        c++;
        double d2 = 0.0;
        for (int k = 0; k < 3; k++) {
          const double d = x(p, k) + s0 * HH(sq, k) + s1 * HH(sq, 3 + k) + s2 * HH(sq, 6 + k) - x(q, k);
          d2 += d * d;
        }
        if (d2 == 0.0) z++;
      });
      cnt(q) = c;
    }, zero);
    if (zero) throw Error("neighbour search: two atoms at the same position (a bond of length 0): not a physical structure");
    auto off = offset;
    int64_t total = 0;
    Kokkos::parallel_scan("bnl:offset", Kokkos::RangePolicy<DeviceType>(0, n),
        KOKKOS_LAMBDA(const int q, int64_t &acc, const bool fin) {
      if (fin) off(q) = acc;
      acc += cnt(q);
      if (fin && q == nn_ - 1) off(q + 1) = acc;
    }, total);
    nb = total;
    Kokkos::parallel_reduce("bnl:maxneigh", Kokkos::RangePolicy<DeviceType>(0, n),
        KOKKOS_LAMBDA(const int q, int &mm) { if (cnt(q) > mm) mm = (int) cnt(q); }, Kokkos::Max<int>(maxneigh));
    maxneigh = std::max(maxneigh, 0);    // Kokkos::Max over no atoms gives INT_MIN
    if ((int64_t) j.extent(0) < nb) {
      j = V1i("bnl:j", nb + nb / 8 + 16);
      S = V3i("bnl:S", nb + nb / 8 + 16);
    }
    auto jj = j;
    auto SS = S;
    Kokkos::parallel_for("bnl:fill", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int q) {
      int64_t w = off(q);
      const int a = pm(q);
      visit(q, [&](int p, int s0, int s1, int s2) {
        const int b = pm(p);
        jj(w) = p;
        SS(w, 0) = s0 - im(b, 0) + im(a, 0);
        SS(w, 1) = s1 - im(b, 1) + im(a, 1);
        SS(w, 2) = s2 - im(b, 2) + im(a, 2);
        w++;
      });
    });
  }
};

}    // namespace kk
}    // namespace grace

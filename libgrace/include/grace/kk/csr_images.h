// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// CSR bonds for the Kokkos engines without LAMMPS (the Python module): local atoms with their bonds (j = owner atom,
// rij = bond vector, as the CPU CsrBonds) are laid out as the engines read them under LAMMPS: one ghost image per
// (owner, lattice translation) != 0 at x(owner) + translation, every bond points at the local atom or its image.
// The engines run unchanged; forces and per-atom virial of the images are folded into their owners (fold), the
// GRACE-2L message rows go owner -> image and back through ImageExchangeKK, and BatchAccKK sums per-atom energy and
// virial per system of a batch.
#pragma once

#include "grace/error.h"
#include "grace/kk/neighbors_kk.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace grace {
namespace kk {

template<class DeviceType, class Traits> struct CsrImagesKK {
  using KKF = typename Traits::kk_float;
  int nlocal = 0, nall = 0, maxneigh = 0;
  Kokkos::View<KKF *[3], Kokkos::LayoutRight, DeviceType> x;    // [nall]
  typename Traits::t_int_1d type, ilist, numneigh;               // [nall], [nlocal], [nlocal]
  typename Traits::t_int_1d owner;                               // [nall - nlocal] local atom of each image
  Kokkos::View<double *[3], Kokkos::LayoutRight, DeviceType> tcart;    // [nall - nlocal] translation of each image
  typename Traits::t_neighbors_2d neighbors;                     // [nlocal][maxneigh]

  // n atoms: pos [n][3], typ [n] (engine atom types 1..ntypes), cell [nsys][3][3] lattice vectors as rows (nullptr:
  // no periodic images allowed), offset [n + 1], j [offset[n]], rij [offset[n]][3] = pos[j] - pos[i] + translation;
  // a batch of systems: system_id [n] (0..nsys-1, bonds stay inside a system), nullptr: one system
  void build(int n, const double *pos, const int *typ, const double *cell, const int64_t *offset, const int *j,
             const double *rij, const int *system_id = nullptr)
  {
    std::map<int, std::array<double, 9>> invs;    // system -> H^-1 (d = s H  ->  s = d H^-1)
    auto inverse = [&](int sys) -> const std::array<double, 9> & {
      auto it = invs.find(sys);
      if (it != invs.end()) return it->second;
      const double *h = cell + 9 * sys;
      const double det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                         h[2] * (h[3] * h[7] - h[4] * h[6]);
      if (!(std::fabs(det) > 0)) throw grace::Error("CSR bonds: singular cell");
      std::array<double, 9> v;
      v[0] = (h[4] * h[8] - h[5] * h[7]) / det; v[1] = (h[2] * h[7] - h[1] * h[8]) / det;
      v[2] = (h[1] * h[5] - h[2] * h[4]) / det; v[3] = (h[5] * h[6] - h[3] * h[8]) / det;
      v[4] = (h[0] * h[8] - h[2] * h[6]) / det; v[5] = (h[2] * h[3] - h[0] * h[5]) / det;
      v[6] = (h[3] * h[7] - h[4] * h[6]) / det; v[7] = (h[1] * h[6] - h[0] * h[7]) / det;
      v[8] = (h[0] * h[4] - h[1] * h[3]) / det;
      return invs.emplace(sys, v).first->second;
    };
    auto sys_of = [&](int a) { return system_id ? system_id[a] : 0; };
    nlocal = n;
    maxneigh = 0;
    for (int i = 0; i < n; i++) maxneigh = std::max(maxneigh, (int) (offset[i + 1] - offset[i]));
    const int64_t nb = offset[n];
    std::vector<int> nbr(nb);
    std::map<std::array<int, 4>, int> img;    // (owner, translation) -> image index
    std::vector<std::array<int, 4>> imgs;
    for (int i = 0; i < n; i++)
      for (int64_t b = offset[i]; b < offset[i + 1]; b++) {
        const int a = j[b];
        if (a < 0 || a >= n) throw grace::Error("CSR bonds: neighbour index out of range");
        if (sys_of(a) != sys_of(i)) throw grace::Error("CSR bonds: bond between two systems of a batch");
        double d[3];
        for (int k = 0; k < 3; k++) d[k] = rij[3 * b + k] - (pos[3 * a + k] - pos[3 * i + k]);
        int s[3] = {0, 0, 0};
        if (cell) {
          const auto &inv = inverse(sys_of(i));
          for (int k = 0; k < 3; k++) {
            const double sk = d[0] * inv[k] + d[1] * inv[3 + k] + d[2] * inv[6 + k];
            s[k] = (int) std::lround(sk);
            if (std::fabs(sk - s[k]) > 1e-4) throw grace::Error("CSR bonds: rij is not pos[j] - pos[i] + a lattice translation");
          }
        } else if (std::fabs(d[0]) + std::fabs(d[1]) + std::fabs(d[2]) > 1e-8) {
          throw grace::Error("CSR bonds: rij differs from pos[j] - pos[i] and no cell was given");
        }
        if (s[0] == 0 && s[1] == 0 && s[2] == 0) {
          nbr[b] = a;
          continue;
        }
        const std::array<int, 4> key = {a, s[0], s[1], s[2]};
        auto it = img.find(key);
        if (it == img.end()) {
          it = img.emplace(key, n + (int) imgs.size()).first;
          imgs.push_back(key);
        }
        nbr[b] = it->second;
      }
    nall = n + (int) imgs.size();
    x = decltype(x)("csr:x", nall);
    type = typename Traits::t_int_1d("csr:type", nall);
    ilist = typename Traits::t_int_1d("csr:ilist", n);
    numneigh = typename Traits::t_int_1d("csr:numneigh", n);
    owner = typename Traits::t_int_1d("csr:owner", nall - n);
    tcart = decltype(tcart)("csr:tcart", nall - n);
    neighbors = typename Traits::t_neighbors_2d("csr:neighbors", n, std::max(maxneigh, 1));
    auto hx = Kokkos::create_mirror_view(x);
    auto ht = Kokkos::create_mirror_view(type);
    auto hi = Kokkos::create_mirror_view(ilist);
    auto hn = Kokkos::create_mirror_view(numneigh);
    auto ho = Kokkos::create_mirror_view(owner);
    auto htc = Kokkos::create_mirror_view(tcart);
    auto hb = Kokkos::create_mirror_view(neighbors);
    for (int i = 0; i < n; i++) {
      for (int k = 0; k < 3; k++) hx(i, k) = pos[3 * i + k];
      ht(i) = typ[i];
      hi(i) = i;
      hn(i) = (int) (offset[i + 1] - offset[i]);
      for (int64_t b = offset[i]; b < offset[i + 1]; b++) hb(i, (int) (b - offset[i])) = nbr[b];
    }
    for (int g = 0; g < (int) imgs.size(); g++) {
      const auto &q = imgs[g];
      const int a = q[0];
      const double *h = cell + 9 * sys_of(a);
      for (int k = 0; k < 3; k++) {
        htc(g, k) = q[1] * h[k] + q[2] * h[3 + k] + q[3] * h[6 + k];
        hx(n + g, k) = pos[3 * a + k] + htc(g, k);
      }
      ht(n + g) = typ[a];
      ho(g) = a;
    }
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(type, ht);
    Kokkos::deep_copy(ilist, hi);
    Kokkos::deep_copy(numneigh, hn);
    Kokkos::deep_copy(owner, ho);
    Kokkos::deep_copy(tcart, htc);
    Kokkos::deep_copy(neighbors, hb);
  }

  // images from a device neighbour list (grace/kk/neighbors_kk.h NeighborListKK: CSR offset / j in its new atom
  // order, lattice translation S per bond); types [n] (device, the list's order); cell rows (host, one system; unused
  // for a BatchNeighborListKK, whose per-structure cells give each image's translation); positions follow with
  // set_positions(). One image per (j, S != 0), keys sorted on the device.
  KOKKOS_INLINE_FUNCTION static uint64_t pack_key(int j, int s0, int s1, int s2)
  {
    return ((uint64_t) j << 30) | ((uint64_t) (s0 + 512) << 20) | ((uint64_t) (s1 + 512) << 10) | (uint64_t) (s2 + 512);
  }
  template<class NL>
  void build_device(const NL &nl, const Kokkos::View<const int *, DeviceType> &types_new, const double *cell)
  {
    const int n = nl.n;
    const int64_t nbd = nl.nb;
    nlocal = n;
    maxneigh = nl.maxneigh;
    auto off = nl.offset;
    auto jj = nl.j;
    auto SS = nl.S;
    int bad = 0;
    Kokkos::parallel_reduce("csr:check_S", Kokkos::RangePolicy<DeviceType>(0, nbd), KOKKOS_LAMBDA(const int64_t b, int &v) {
      for (int k = 0; k < 3; k++)
        if (SS(b, k) <= -512 || SS(b, k) >= 512) v++;
    }, bad);
    if (bad) throw grace::Error("CSR images: lattice translation beyond +-511 cells");
    Kokkos::View<uint64_t *, DeviceType> keys("csr:keys", std::max<int64_t>(nbd, 1));
    int64_t nk = 0;
    Kokkos::parallel_scan("csr:keys", Kokkos::RangePolicy<DeviceType>(0, nbd),
        KOKKOS_LAMBDA(const int64_t b, int64_t &acc, const bool fin) {
      if (SS(b, 0) != 0 || SS(b, 1) != 0 || SS(b, 2) != 0) {
        if (fin) keys(acc) = pack_key(jj(b), SS(b, 0), SS(b, 1), SS(b, 2));
        acc++;
      }
    }, nk);
    if (nk > 1) Kokkos::sort(Kokkos::subview(keys, std::make_pair((int64_t) 0, nk)));
    Kokkos::View<uint64_t *, DeviceType> uniq("csr:uniq", std::max<int64_t>(nk, 1));
    int64_t ng = 0;
    Kokkos::parallel_scan("csr:unique", Kokkos::RangePolicy<DeviceType>(0, nk),
        KOKKOS_LAMBDA(const int64_t i, int64_t &acc, const bool fin) {
      if (i == 0 || keys(i) != keys(i - 1)) {
        if (fin) uniq(acc) = keys(i);
        acc++;
      }
    }, ng);
    nall = n + (int) ng;
    x = decltype(x)("csr:x", std::max(nall, 1));
    type = typename Traits::t_int_1d("csr:type", std::max(nall, 1));
    ilist = typename Traits::t_int_1d("csr:ilist", std::max(n, 1));
    numneigh = typename Traits::t_int_1d("csr:numneigh", std::max(n, 1));
    owner = typename Traits::t_int_1d("csr:owner", std::max<int64_t>(ng, 1));
    tcart = decltype(tcart)("csr:tcart", std::max<int64_t>(ng, 1));
    neighbors = typename Traits::t_neighbors_2d("csr:neighbors", std::max(n, 1), std::max(maxneigh, 1));
    auto ty = type;
    auto ow = owner;
    auto tc = tcart;
    auto il = ilist;
    auto nn = numneigh;
    auto nbr = neighbors;
    if constexpr (requires { nl.H; nl.sys; }) {    // a batch (BatchNeighborListKK): the cell of the owner's structure
      auto HH = nl.H;
      auto sy = nl.sys;
      Kokkos::parallel_for("csr:images", Kokkos::RangePolicy<DeviceType>(0, ng), KOKKOS_LAMBDA(const int64_t g) {
        const uint64_t k = uniq(g);
        const int a = (int) (k >> 30), c = sy(a);
        const int s[3] = {(int) ((k >> 20) & 1023) - 512, (int) ((k >> 10) & 1023) - 512, (int) (k & 1023) - 512};
        ow(g) = a;
        ty(n + g) = types_new(a);
        for (int d = 0; d < 3; d++) tc(g, d) = s[0] * HH(c, d) + s[1] * HH(c, 3 + d) + s[2] * HH(c, 6 + d);
      });
    } else {
      Mat3 H;
      for (int k = 0; k < 9; k++) H.a[k] = cell[k];
      Kokkos::parallel_for("csr:images", Kokkos::RangePolicy<DeviceType>(0, ng), KOKKOS_LAMBDA(const int64_t g) {
        const uint64_t k = uniq(g);
        const int a = (int) (k >> 30);
        const int s[3] = {(int) ((k >> 20) & 1023) - 512, (int) ((k >> 10) & 1023) - 512, (int) (k & 1023) - 512};
        ow(g) = a;
        ty(n + g) = types_new(a);
        for (int d = 0; d < 3; d++) tc(g, d) = s[0] * H.a[d] + s[1] * H.a[3 + d] + s[2] * H.a[6 + d];
      });
    }
    Kokkos::parallel_for("csr:rows", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int i) {
      ty(i) = types_new(i);
      il(i) = i;
      nn(i) = (int) (off(i + 1) - off(i));
      for (int64_t b = off(i); b < off(i + 1); b++) {
        int v = jj(b);
        if (SS(b, 0) != 0 || SS(b, 1) != 0 || SS(b, 2) != 0) {
          const uint64_t k = pack_key(jj(b), SS(b, 0), SS(b, 1), SS(b, 2));
          int64_t lo = 0, hi = ng;    // lower bound in the sorted unique keys
          while (lo < hi) {
            const int64_t mid = (lo + hi) / 2;
            if (uniq(mid) < k) lo = mid + 1;
            else hi = mid;
          }
          v = n + (int) lo;
        }
        nbr(i, (int) (b - off(i))) = v;
      }
    });
  }

  // new positions of the local atoms [nlocal][3] (device) -> x of the atoms and of their images (same bonds, same
  // translations: the caller rebuilds when atoms moved too far)
  void set_positions(const Kokkos::View<const double *[3], Kokkos::LayoutRight, DeviceType> &pos) const
  {
    auto xx = x;
    auto ow = owner;
    auto tc = tcart;
    const int nl = nlocal, ng = nall - nlocal;
    Kokkos::parallel_for("csr:set_x", Kokkos::RangePolicy<DeviceType>(0, nl), KOKKOS_LAMBDA(const int i) {
      for (int k = 0; k < 3; k++) xx(i, k) = pos(i, k);
    });
    if (ng)
      Kokkos::parallel_for("csr:set_x_images", Kokkos::RangePolicy<DeviceType>(0, ng), KOKKOS_LAMBDA(const int g) {
        for (int k = 0; k < 3; k++) xx(nl + g, k) = pos(ow(g), k) + tc(g, k);
      });
  }

  // rows [nlocal, nall) of a [nall][w] view added to the rows of their owners
  template<class V> void fold(const V &v) const
  {
    const int ng = nall - nlocal, w = (int) v.extent(1), nl = nlocal;
    if (!ng) return;
    auto ow = owner;
    Kokkos::parallel_for("csr:fold", Kokkos::RangePolicy<DeviceType>(0, ng), KOKKOS_LAMBDA(const int g) {
      for (int k = 0; k < w; k++) Kokkos::atomic_add(&v(ow(g), k), v(nl + g, k));
    });
  }
};

// the images of `img` in blocks of at most `block`: f(n, first image index, owners of the block's images); the
// exchanges pass one block at a time through their buffer, so it is sized by the block, not by the number of images
// (GRACE-3L: 25 KB per image)
template<class Traits, class Images, class F> void for_image_blocks(const Images &img, int block, F &&f)
{
  const int ng = img.nall - img.nlocal;
  for (int off = 0; off < ng; off += block) {
    const int n = std::min(block, ng - off);
    const typename Traits::t_int_1d owners = Kokkos::subview(img.owner, std::make_pair(off, off + n));
    f(n, img.nlocal + off, owners);
  }
}

// message rows of the GRACE-2L engines: owner -> image (forward), image summed into owner (reverse), with the
// engine's own pack / unpack kernels
template<class Engine, class Images> struct ImageExchangeKK : Engine::Exchange {
  using Traits = typename Engine::traits_type;
  Engine *eng = nullptr;
  const Images *img = nullptr;
  int block = 8192;    // images per pass through buf
  typename Traits::t_double_1d buf;
  void ensure()
  {
    const size_t need = (size_t) std::min(block, img->nall - img->nlocal) * eng->comm_width();
    if (buf.extent(0) < need) buf = typename Traits::t_double_1d("csr:xbuf", need);
  }
  void forward() override
  {
    ensure();
    for_image_blocks<Traits>(*img, block, [&](int n, int first, const typename Traits::t_int_1d &owners) {
      eng->pack_forward(n, owners, buf);
      eng->unpack_forward(n, first, buf);
    });
  }
  void reverse() override
  {
    ensure();
    for_image_blocks<Traits>(*img, block, [&](int n, int first, const typename Traits::t_int_1d &owners) {
      eng->pack_reverse(n, first, buf);
      eng->unpack_reverse(n, owners, buf);
    });
  }
};

// message rows of the GRACE-3L engine: two layers (0: I, 1: eq2 norm) of different widths, layer by layer as the
// engine asks (forward 0, 1; reverse 1, 0)
template<class Engine, class Images> struct ImageExchange3KK : Engine::Exchange3 {
  using Traits = typename Engine::traits_type;
  Engine *eng = nullptr;
  const Images *img = nullptr;
  int block = 8192;    // images per pass through buf
  typename Traits::t_double_1d buf;
  void ensure(int layer)
  {
    const size_t need = (size_t) std::min(block, img->nall - img->nlocal) * eng->comm_width(layer);
    if (buf.extent(0) < need) buf = typename Traits::t_double_1d("csr:xbuf3", need);
  }
  void forward(int layer) override
  {
    ensure(layer);
    for_image_blocks<Traits>(*img, block, [&](int n, int first, const typename Traits::t_int_1d &owners) {
      eng->pack_forward(layer, n, owners, buf);
      eng->unpack_forward(layer, n, first, buf);
    });
  }
  void reverse(int layer) override
  {
    ensure(layer);
    for_image_blocks<Traits>(*img, block, [&](int n, int first, const typename Traits::t_int_1d &owners) {
      eng->pack_reverse(layer, n, first, buf);
      eng->unpack_reverse(layer, n, owners, buf);
    });
  }
};

// per-system energy and virial of a batch from the per-atom energy [n] and the folded per-atom virial [n][6]
template<class DeviceType> struct BatchAccKK {
  Kokkos::View<double *[7], Kokkos::LayoutRight, DeviceType> acc;    // [nsys]: energy, virial xx yy zz xy xz yz
  template<class VE, class VV, class VS>
  void sum(int n, int nsys, const VE &eatom, const VV &vatom, const VS &system_id)
  {
    if ((int) acc.extent(0) != nsys) acc = decltype(acc)("csr:batch", nsys);
    else Kokkos::deep_copy(acc, 0.0);
    auto a = acc;
    Kokkos::parallel_for("csr:batch_sum", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(const int i) {
      const int s = system_id(i);
      Kokkos::atomic_add(&a(s, 0), (double) eatom(i));
      for (int k = 0; k < 6; k++) Kokkos::atomic_add(&a(s, 1 + k), (double) vatom(i, k));
    });
  }
};

}    // namespace kk
}    // namespace grace

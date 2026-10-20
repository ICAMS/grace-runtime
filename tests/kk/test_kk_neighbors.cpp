// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The device neighbour list (grace/kk/neighbors_kk.h) against the CPU list of grace_runtime (python/src/neighbors.cpp):
// the same spatial permutation, the same CSR rows with the same bonds (j, S) in the same order, on periodic, small
// triclinic, shifted (atoms outside the cell), slab (mixed pbc) and open cells, 0 and 1 atoms; and the device ghost
// images (CsrImagesKK::build_device) resolve every bond to the same atom and translation as the host build; all cases
// as one batch (BatchNeighborListKK + batch images) equal to each case alone.
#include "grace/kk/csr_images.h"
#include "grace/kk/neighbors_kk.h"
#include "grace/kk/traits.h"
#include "grace_cnpy/cnpy.h"
#include "neighbors.h"
#include "../test_util.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using Dev = Kokkos::DefaultExecutionSpace;
using T = grace::kk::DefaultTraits<Dev>;

namespace {
struct Case {
  std::string name;
  std::vector<double> pos, cell;
  bool pbc[3];
  double cutoff;
};

Case from_fixture(const std::string &fx, const std::string &name, double cutoff)
{
  auto npz = grace::cnpy::npz_load(fx + "/" + name + ".npz");
  Case c;
  c.name = name;
  const auto &p = npz["positions"];
  c.pos.assign(p.data<double>(), p.data<double>() + p.num_vals);
  c.cell.assign(npz["cell"].data<double>(), npz["cell"].data<double>() + 9);
  c.pbc[0] = c.pbc[1] = c.pbc[2] = true;
  c.cutoff = cutoff;
  return c;
}

int check(const Case &c)
{
  const int n = (int) c.pos.size() / 3;
  const auto ref = grace_runtime::neighbor_search(c.pos.data(), n, c.cell.data(), c.pbc, c.cutoff, 1);
  Kokkos::View<double *[3], Kokkos::LayoutRight, Dev> pos("pos", std::max(n, 1));
  auto hp = Kokkos::create_mirror_view(pos);
  for (int a = 0; a < n; a++)
    for (int k = 0; k < 3; k++) hp(a, k) = c.pos[3 * a + k];
  Kokkos::deep_copy(pos, hp);
  grace::kk::NeighborListKK<Dev> nl;
  nl.build(pos, n, c.cell.data(), c.pbc, c.cutoff);
  CHECK(nl.n == n && nl.nb == (int64_t) ref.j.size());
  auto perm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nl.perm);
  auto off = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nl.offset);
  auto j = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nl.j);
  auto S = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nl.S);
  int bad = 0;
  for (int q = 0; q < n; q++) {
    if (perm(q) != ref.perm[q]) bad++;
    if (off(q + 1) != ref.offset[q + 1]) bad++;
  }
  // inverse cell: S of the CPU list from its Cartesian shift
  double inv[9];
  {
    const double *a = c.cell.data();
    const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                       a[2] * (a[3] * a[7] - a[4] * a[6]);
    inv[0] = (a[4] * a[8] - a[5] * a[7]) / det; inv[1] = (a[2] * a[7] - a[1] * a[8]) / det;
    inv[2] = (a[1] * a[5] - a[2] * a[4]) / det; inv[3] = (a[5] * a[6] - a[3] * a[8]) / det;
    inv[4] = (a[0] * a[8] - a[2] * a[6]) / det; inv[5] = (a[2] * a[3] - a[0] * a[5]) / det;
    inv[6] = (a[3] * a[7] - a[4] * a[6]) / det; inv[7] = (a[1] * a[6] - a[0] * a[7]) / det;
    inv[8] = (a[0] * a[4] - a[1] * a[3]) / det;
  }
  if (!bad)
    for (size_t b = 0; b < ref.j.size(); b++) {
      if (j(b) != ref.j[b]) { bad++; continue; }
      for (int k = 0; k < 3; k++) {
        const double sk = ref.shift[3 * b] * inv[k] + ref.shift[3 * b + 1] * inv[3 + k] + ref.shift[3 * b + 2] * inv[6 + k];
        if ((int) std::lround(sk) != S(b, k)) bad++;
      }
    }
  // device images vs the host build from (pos, rij)
  int ibad = 0, nimg_h = 0, nimg_d = 0;
  if (n > 0 && !bad) {
    std::vector<double> pos_new(3 * (size_t) n), rij(3 * ref.j.size());
    std::vector<int> ty(n, 1);
    for (int q = 0; q < n; q++)
      for (int k = 0; k < 3; k++) pos_new[3 * q + k] = c.pos[3 * ref.perm[q] + k];
    for (size_t b = 0; b < ref.j.size(); b++)
      for (int k = 0; k < 3; k++) rij[3 * b + k] = c.pos[3 * ref.jo[b] + k] - c.pos[3 * ref.io[b] + k] + ref.shift[3 * b + k];
    std::vector<int> jj(ref.j.begin(), ref.j.end());
    grace::kk::CsrImagesKK<Dev, T> hi, di;
    hi.build(n, pos_new.data(), ty.data(), c.cell.data(), ref.offset.data(), jj.data(), rij.data());
    Kokkos::View<int *, Dev> tyd("ty", n);
    Kokkos::deep_copy(tyd, 1);
    di.build_device(nl, tyd, c.cell.data());
    nimg_h = hi.nall - n;
    nimg_d = di.nall - n;
    auto resolve = [&](const grace::kk::CsrImagesKK<Dev, T> &im, std::vector<int> &owner, std::vector<double> &tr) {
      auto nb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), im.neighbors);
      auto ow = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), im.owner);
      auto tc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), im.tcart);
      for (int q = 0; q < n; q++)
        for (int64_t b = ref.offset[q]; b < ref.offset[q + 1]; b++) {
          const int v = nb(q, (int) (b - ref.offset[q]));
          owner.push_back(v < n ? v : ow(v - n));
          for (int k = 0; k < 3; k++) tr.push_back(v < n ? 0.0 : tc(v - n, k));
        }
    };
    std::vector<int> oh, od;
    std::vector<double> th, td;
    resolve(hi, oh, th);
    resolve(di, od, td);
    for (size_t b = 0; b < oh.size(); b++) {
      if (oh[b] != od[b]) ibad++;
      for (int k = 0; k < 3; k++)
        if (std::fabs(th[3 * b + k] - td[3 * b + k]) > 1e-9) ibad++;
    }
  }
  std::printf("%-14s n %5d bonds %7zu  maxneigh %3d  mismatches %d  images host %d device %d  image mismatches %d\n",
              c.name.c_str(), n, ref.j.size(), nl.maxneigh, bad, nimg_h, nimg_d, ibad);
  CHECK(nimg_h == nimg_d);
  return bad + ibad;
}

// all cases in one batch (one cutoff) against each case alone: structure s keeps its atoms [start[s], start[s + 1])
// in the order its own list gives them, the same rows, neighbours (offset by start[s]) and translations, and the batch
// images resolve every bond to the same owner and Cartesian translation as the single-structure images
int check_batch(const std::vector<Case> &cases, double cutoff)
{
  const int ns = (int) cases.size();
  std::vector<int64_t> start(ns + 1, 0);
  std::vector<double> pos, cell;
  std::vector<char> pbc;
  for (int s = 0; s < ns; s++) {
    start[s + 1] = start[s] + (int64_t) cases[s].pos.size() / 3;
    pos.insert(pos.end(), cases[s].pos.begin(), cases[s].pos.end());
    cell.insert(cell.end(), cases[s].cell.begin(), cases[s].cell.end());
    for (int k = 0; k < 3; k++) pbc.push_back(cases[s].pbc[k]);
  }
  const int n = (int) start[ns];
  Kokkos::View<double *[3], Kokkos::LayoutRight, Dev> dpos("pos", std::max(n, 1));
  auto hp = Kokkos::create_mirror_view(dpos);
  for (int a = 0; a < n; a++)
    for (int k = 0; k < 3; k++) hp(a, k) = pos[3 * (size_t) a + k];
  Kokkos::deep_copy(dpos, hp);
  grace::kk::BatchNeighborListKK<Dev> bl;
  bl.build(dpos, ns, start.data(), cell.data(), reinterpret_cast<const bool *>(pbc.data()), cutoff);
  Kokkos::View<int *, Dev> tyb("tyb", std::max(n, 1));
  Kokkos::deep_copy(tyb, 1);
  grace::kk::CsrImagesKK<Dev, T> bi;
  bi.build_device(bl, tyb, nullptr);
  auto bperm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bl.perm);
  auto bsys = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bl.sys);
  auto boff = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bl.offset);
  auto bj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bl.j);
  auto bS = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bl.S);
  auto bnb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bi.neighbors);
  auto bow = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bi.owner);
  auto btc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bi.tcart);
  int bad = 0, nimg = 0;
  for (int s = 0; s < ns; s++) {
    const Case &c = cases[s];
    const int m = (int) (start[s + 1] - start[s]);
    const int64_t a0 = start[s];
    Kokkos::View<double *[3], Kokkos::LayoutRight, Dev> p1("p1", std::max(m, 1));
    Kokkos::deep_copy(Kokkos::subview(p1, std::make_pair(0, m), Kokkos::ALL),
                      Kokkos::subview(dpos, std::make_pair((int) a0, (int) a0 + m), Kokkos::ALL));
    grace::kk::NeighborListKK<Dev> nl;
    nl.build(p1, m, c.cell.data(), c.pbc, cutoff);
    if (m == 0) continue;
    Kokkos::View<int *, Dev> ty1("ty1", m);
    Kokkos::deep_copy(ty1, 1);
    grace::kk::CsrImagesKK<Dev, T> si;
    si.build_device(nl, ty1, c.cell.data());
    nimg += si.nall - m;
    auto perm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nl.perm);
    auto off = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nl.offset);
    auto j = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nl.j);
    auto S = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nl.S);
    auto snb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), si.neighbors);
    auto sow = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), si.owner);
    auto stc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), si.tcart);
    const int64_t b0 = boff(a0);
    int sbad = 0;
    for (int q = 0; q < m; q++) {
      const int64_t qb = a0 + q;
      if (bperm(qb) != a0 + perm(q) || bsys(qb) != s) sbad++;
      if (boff(qb + 1) - b0 != off(q + 1)) sbad++;
    }
    if (!sbad)
      for (int q = 0; q < m; q++)
        for (int64_t b = off(q); b < off(q + 1); b++) {
          if (bj(b0 + b) != a0 + j(b)) sbad++;
          for (int k = 0; k < 3; k++)
            if (bS(b0 + b, k) != S(b, k)) sbad++;
          // images: same owner (offset by a0) and translation
          const int vs = snb(q, (int) (b - off(q))), vb = bnb(a0 + q, (int) (b - off(q)));
          const int os = vs < m ? vs : sow(vs - m), ob = vb < n ? vb : bow(vb - n);
          if (ob != a0 + os) sbad++;
          for (int k = 0; k < 3; k++) {
            const double ts = vs < m ? 0.0 : stc(vs - m, k), tb = vb < n ? 0.0 : btc(vb - n, k);
            if (std::fabs(ts - tb) > 1e-9) sbad++;
          }
        }
    bad += sbad;
  }
  std::printf("batch of %d structures: n %d bonds %ld images %d (alone %d) mismatches %d\n", ns, n, (long) bl.nb,
              bi.nall - n, nimg, bad);
  CHECK(bi.nall - n == nimg);
  return bad;
}
}    // namespace

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  std::vector<Case> cases;
  cases.push_back(from_fixture(fx, "W54", 6.5));
  cases.push_back(from_fixture(fx, "WMo54", 7.0));
  cases.push_back(from_fixture(fx, "W2tri", 7.0));    // 2-atom triclinic cell, cutoff >> cell: many self-images
  {
    Case c = from_fixture(fx, "W54", 6.5);    // atoms outside the cell (shifted by 1.3 of every lattice vector)
    c.name = "W54-shifted";
    for (size_t a = 0; a < c.pos.size() / 3; a++)
      for (int k = 0; k < 3; k++) c.pos[3 * a + k] += 1.3 * (c.cell[k] + c.cell[3 + k] + c.cell[6 + k]) - 0.7 * c.cell[3 * (a % 3) + k];
    cases.push_back(c);
  }
  {
    Case c = from_fixture(fx, "W54", 6.5);    // slab: open in z, box padded as the calculator does
    c.name = "W54-slab";
    c.pbc[2] = false;
    for (int k = 0; k < 3; k++) c.cell[6 + k] *= 4.0;
    cases.push_back(c);
  }
  {
    Case c;    // open cluster in a padded box
    c.name = "cluster";
    c.pbc[0] = c.pbc[1] = c.pbc[2] = false;
    c.cutoff = 6.0;
    c.cell = {40, 0, 0, 0, 40, 0, 0, 0, 40};
    for (int a = 0; a < 27; a++) {
      c.pos.push_back(10 + 2.7 * (a % 3) + 0.1 * std::sin(a));
      c.pos.push_back(12 + 2.7 * ((a / 3) % 3) + 0.1 * std::cos(a));
      c.pos.push_back(9 + 2.7 * (a / 9));
    }
    cases.push_back(c);
  }
  {
    Case c;
    c.name = "one-atom";
    c.pbc[0] = c.pbc[1] = c.pbc[2] = true;
    c.cutoff = 5.0;
    c.cell = {3.2, 0, 0, 0, 3.2, 0, 0, 0, 3.2};
    c.pos = {0.1, 0.2, 0.3};
    cases.push_back(c);
  }
  {
    Case c;
    c.name = "empty";
    c.pbc[0] = c.pbc[1] = c.pbc[2] = true;
    c.cutoff = 5.0;
    c.cell = {3.2, 0, 0, 0, 3.2, 0, 0, 0, 3.2};
    cases.push_back(c);
  }
  int bad = 0;
  for (const auto &c : cases) bad += check(c);
  bad += check_batch(cases, 6.5);
  {
    std::vector<Case> rev(cases.rbegin(), cases.rend());    // the empty and one-atom cases first
    bad += check_batch(rev, 6.0);
  }
  CHECK(bad == 0);
  std::printf("kk neighbors ok\n");
  return 0;
}

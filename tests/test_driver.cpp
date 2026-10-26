// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Drivers on the CsrBonds path (what the Python calculator will call) against the LAMMPS styles (pre-move
// binary, 1 rank x 1 thread) on the same cells: energy, per-atom energy, forces, pressure.
// Not bit-identical: LAMMPS gathers the bonds in its neighbour-list order and folds ghost forces back by
// reverse comm; the CSR path sums in the bond order of the CSR directly onto the atoms.
//   env GRACE_TEST_REF_DIR (default tests/fixtures/ref): <cell>_<1l|2l>.dump / .thermo from LAMMPS runs of grace/{1l,2l,3l}/fused
#include "grace/fused_1l.h"
#include "grace/fused_2l.h"
#include "grace/fused_3l.h"
#include "grace/fused_cpu_types.h"
#include "grace_cnpy/cnpy.h"
#include "test_util.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {
struct Cell {
  int n = 0;
  std::vector<int> type;     // LAMMPS types: W = 1, Mo = 2 (fixtures' specorder)
  std::vector<int64_t> offset;
  std::vector<int> j;
  std::vector<double> rij;
  double volume = 0;
};
Cell load_cell(const std::string &path)
{
  auto npz = grace::cnpy::npz_load(path);
  Cell c;
  const auto &num = npz["numbers"];
  c.n = (int) num.shape[0];
  for (int a = 0; a < c.n; a++) c.type.push_back(num.data<int>()[a] == 74 ? 1 : 2);
  const auto &off = npz["offset"];
  c.offset.assign(off.data<int64_t>(), off.data<int64_t>() + off.shape[0]);
  const auto &jj = npz["j"];
  c.j.assign(jj.data<int>(), jj.data<int>() + jj.shape[0]);
  const auto &r = npz["rij"];
  c.rij.assign(r.data<double>(), r.data<double>() + r.num_vals);
  const double *h = npz["cell"].data<double>();
  c.volume = std::fabs(h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                       h[2] * (h[3] * h[7] - h[4] * h[6]));
  return c;
}
struct Ref {
  std::vector<double> f, pe;    // [n][3], [n] by id
  double etot = 0, p[6] = {0, 0, 0, 0, 0, 0};
};
Ref load_ref(const std::string &stem, int n)
{
  Ref r;
  r.f.assign(3 * n, 0); r.pe.assign(n, 0);
  std::ifstream d(stem + ".dump");
  CHECK(d.good());
  std::string line;
  for (int k = 0; k < 9; k++) std::getline(d, line);
  for (int a = 0; a < n; a++) {
    int id; double fx, fy, fz, pe;
    d >> id >> fx >> fy >> fz >> pe;
    r.f[3 * (id - 1)] = fx; r.f[3 * (id - 1) + 1] = fy; r.f[3 * (id - 1) + 2] = fz; r.pe[id - 1] = pe;
  }
  std::ifstream t(stem + ".thermo");
  CHECK(t.good());
  double step;
  t >> step >> r.etot >> r.p[0] >> r.p[1] >> r.p[2] >> r.p[3] >> r.p[4] >> r.p[5];
  return r;
}
std::vector<int> type2mu_for(const std::vector<std::string> &names)
{
  std::vector<int> t2m(3, -1);
  for (int e = 0; e < (int) names.size(); e++) {
    if (names[e] == "W") t2m[1] = e;
    if (names[e] == "Mo") t2m[2] = e;
  }
  CHECK(t2m[1] >= 0 && t2m[2] >= 0);
  return t2m;
}
struct Result {
  double e = 0, vir[6] = {0, 0, 0, 0, 0, 0};
  std::vector<double> f, eatom;
};
template<class D>
Result run(D &drv, const Cell &c, int nthreads, bool two_layer, int nmax_nodes = 0, std::vector<int> ilist = {})
{
  if (nmax_nodes == 0) nmax_nodes = c.n;
  Result r;
  r.f.assign(3 * c.n, 0); r.eatom.assign(c.n, 0);
  grace::CsrBonds src{c.type.data(), c.offset.data(), c.j.data(), c.rij.data()};
  grace::Outputs out;
  out.f = r.f.data();
  out.eatom = r.eatom.data();
  grace::ComputeFlags fl;
  fl.eflag_atom = true;
  fl.virial = true;
  if (ilist.empty()) {
    ilist.resize(c.n);
    std::iota(ilist.begin(), ilist.end(), 0);
  }
  const int inum = (int) ilist.size();
  if constexpr (std::is_same_v<D, grace::Grace2LFused> || std::is_same_v<D, grace::Grace3LFused>) {
    grace::NoExchange ex;
    drv.compute(inum, ilist.data(), c.n, nmax_nodes, src, ex, out, fl, nthreads);
  } else {
    drv.compute(inum, ilist.data(), c.n, nmax_nodes, src, out, fl, nthreads);
  }
  (void) two_layer;
  r.e = out.energy;
  for (int k = 0; k < 6; k++) r.vir[k] = out.virial[k];
  return r;
}
void compare(const char *what, const Result &r, const Ref &ref, const Cell &c)
{
  const double nktv2p = 1.6021765e6;    // eV/A^3 -> bar (LAMMPS units metal)
  double df = 0, dpe = 0, dp = 0, pscale = 1;
  for (int q = 0; q < 3 * c.n; q++) df = std::max(df, std::fabs(r.f[q] - ref.f[q]));
  for (int a = 0; a < c.n; a++) dpe = std::max(dpe, std::fabs(r.eatom[a] - ref.pe[a]));
  for (int k = 0; k < 6; k++) {
    dp = std::max(dp, std::fabs(r.vir[k] / c.volume * nktv2p - ref.p[k]));
    pscale = std::max(pscale, std::fabs(ref.p[k]));
  }
  std::printf("%-14s dE/n %.3e  max|de_i| %.3e  max|dF| %.3e  max|dP|/max|P| %.3e\n", what,
              std::fabs(r.e - ref.etot) / c.n, dpe, df, dp / pscale);
  CHECK(std::fabs(r.e - ref.etot) / c.n <= 1e-6);
  CHECK(dpe <= 1e-5);
  CHECK(df <= 2e-5);
  CHECK(dp / pscale <= 1e-5);
}
// the atoms of type t of c and the bonds among them, in the original order; idx[i]: new index of atom i, -1 if dropped
Cell subcell(const Cell &c, int t, std::vector<int> &idx)
{
  Cell s;
  idx.assign(c.n, -1);
  for (int i = 0; i < c.n; i++)
    if (c.type[i] == t) idx[i] = s.n++;
  s.type.assign(s.n, t);
  s.offset = {0};
  s.volume = c.volume;
  for (int i = 0; i < c.n; i++) {
    if (idx[i] < 0) continue;
    for (int64_t q = c.offset[i]; q < c.offset[i + 1]; q++)
      if (idx[c.j[q]] >= 0) {
        s.j.push_back(idx[c.j[q]]);
        s.rij.insert(s.rij.end(), &c.rij[3 * q], &c.rij[3 * q + 3]);
      }
    s.offset.push_back((int64_t) s.j.size());
  }
  return s;
}
// test_unmapped_type (pair_style hybrid NULL: type2mu -1, no bonds to or from that type), on WMo54 against the
// driver ref set up for W = 1, Mo = 2:
//   (a) an unused unmapped type between W and Mo changes nothing;
//   (b) Mo unmapped and out of the list: the W atoms give what the W atoms alone (Mo removed) give, and no force
//       reaches the Mo atoms
template<class D, class M> void check_unmapped(D &ref, const std::shared_ptr<M> &m, const Cell &c, const char *tag)
{
  const std::vector<int> t2m = type2mu_for(m->element_names);    // {-1, W, Mo}
  D gap(m, {}, {}), nomo(m, {}, {});
  gap.setup_types(3, {-1, t2m[1], -1, t2m[2]});
  nomo.setup_types(2, {-1, t2m[1], -1});
  CHECK(nomo.cutoff(1, 2) == 0.0 && nomo.cutoff(2, 1) == 0.0 && nomo.cutoff(1, 1) > 0.0);
  bool threw = false;    // an unmapped atom in the atom list is an error, not garbage
  try {
    run(nomo, c, 1, false);
  } catch (const grace::Error &) {
    threw = true;
  }
  CHECK(threw);

  Cell c3 = c;
  for (int &t : c3.type)
    if (t == 2) t = 3;
  const Result a = run(ref, c, 1, false), b = run(gap, c3, 1, false);
  CHECK(a.e == b.e && a.f == b.f && a.eatom == b.eatom);

  std::vector<int> idx, wlist;
  const Cell w = subcell(c, 1, idx);
  for (int i = 0; i < c.n; i++)
    if (idx[i] >= 0) wlist.push_back(i);
  CHECK(w.n > 0 && w.n < c.n);
  const Result x = run(nomo, c, 1, false, 0, wlist), y = run(nomo, w, 1, false);
  double df = 0, dpe = 0;
  for (int i = 0; i < c.n; i++) {
    if (idx[i] < 0) {
      for (int k = 0; k < 3; k++) CHECK(x.f[3 * i + k] == 0.0);
      continue;
    }
    for (int k = 0; k < 3; k++) df = std::max(df, std::fabs(x.f[3 * i + k] - y.f[3 * idx[i] + k]));
    dpe = std::max(dpe, std::fabs(x.eatom[i] - y.eatom[idx[i]]));
  }
  std::printf("%s unmapped Mo vs W only: dE %.3e  max|de_i| %.3e  max|dF| %.3e\n", tag, std::fabs(x.e - y.e), dpe, df);
  CHECK(std::fabs(x.e - y.e) / w.n <= 1e-9 && dpe <= 1e-9 && df <= 1e-8);
}
}    // namespace

int main()
{
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  const std::string rd = env_or("GRACE_TEST_REF_DIR", "tests/fixtures/ref");
  auto m2 = std::make_shared<grace::GRACE2LModel>();
  m2->load(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"));
  auto m1 = std::make_shared<grace::GRACE1LModel>();
  m1->load(test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"));
  grace::Grace2LFused d2(m2, {}, {});
  grace::Grace1LFused d1(m1, {}, {});
  d2.setup_types(2, type2mu_for(m2->element_names));
  d1.setup_types(2, type2mu_for(m1->element_names));
  auto m3 = std::make_shared<grace::GRACE3LModel>();
  m3->load(test_model("GRACE_TEST_M3", "GRACE-3L-OMAT-large"));
  grace::Grace3LFused d3(m3, {}, {});
  d3.setup_types(2, type2mu_for(m3->element_names));
  for (const char *cell : {"W54", "WMo54", "W2tri"}) {
    const Cell c = load_cell(fx + "/" + cell + ".npz");
    compare((std::string(cell) + " 2L").c_str(), run(d2, c, 1, true), load_ref(rd + "/" + cell + "_2l", c.n), c);
    compare((std::string(cell) + " 1L").c_str(), run(d1, c, 1, false), load_ref(rd + "/" + cell + "_1l", c.n), c);
    compare((std::string(cell) + " 3L").c_str(), run(d3, c, 1, false), load_ref(rd + "/" + cell + "_3l", c.n), c);
  }
  // test_3l_radial_mlp_by_gemm: the path above spline_mb (GRACE-3L-OMAT-large with >= 12 species), forced by
  // spline_mb = 0, against the same LAMMPS references (which use the tables)
  {
    grace::FusedSettings fs;
    fs.spline_mb = 0;
    grace::Grace3LFused d3m(m3, fs, {});
    d3m.setup_types(2, type2mu_for(m3->element_names));
    for (const char *cell : {"W54", "WMo54"}) {
      const Cell c = load_cell(fx + "/" + cell + ".npz");
      compare((std::string(cell) + " 3L MLP").c_str(), run(d3m, c, 2, false), load_ref(rd + "/" + cell + "_3l", c.n), c);
    }
  }
  // test_3l_fc_fold: the FCs folded into the cp_l projections (the default) against the unfolded chain
  {
    grace::FusedSettings fs;
    fs.fc_fold = false;
    grace::Grace3LFused d3u(m3, fs, {});
    d3u.setup_types(2, type2mu_for(m3->element_names));
    const Cell c = load_cell(fx + "/WMo54.npz");
    const Result a = run(d3, c, 1, false), b = run(d3u, c, 1, false);
    double df = 0;
    for (int q = 0; q < 3 * c.n; q++) df = std::max(df, std::fabs(a.f[q] - b.f[q]));
    std::printf("3L fc_fold vs unfolded: dE %.3e  max|dF| %.3e\n", std::fabs(a.e - b.e), df);
    CHECK(std::fabs(a.e - b.e) / c.n <= 1e-6 && df <= 1e-5);
  }
  // thread count changes the summation order only (document the size, require it to be tiny)
  {
    const Cell c = load_cell(fx + "/WMo54.npz");
    const Result a = run(d2, c, 1, true), b = run(d2, c, 4, true);
    double df = 0;
    for (int q = 0; q < 3 * c.n; q++) df = std::max(df, std::fabs(a.f[q] - b.f[q]));
    std::printf("2L 1 vs 4 threads: dE %.3e  max|dF| %.3e\n", std::fabs(a.e - b.e), df);
    CHECK(std::fabs(a.e - b.e) / c.n <= 1e-9 && df <= 1e-6);
    const Result a3 = run(d3, c, 1, false), b3 = run(d3, c, 4, false), c3 = run(d3, c, 2, false);
    double df3 = 0;
    for (int q = 0; q < 3 * c.n; q++) df3 = std::max({df3, std::fabs(a3.f[q] - b3.f[q]), std::fabs(a3.f[q] - c3.f[q])});
    std::printf("3L 1 vs 4 vs 2 threads: dE %.3e  max|dF| %.3e\n", std::fabs(a3.e - b3.e), df3);
    CHECK(std::fabs(a3.e - b3.e) / c.n <= 1e-9 && std::fabs(a3.e - c3.e) / c.n <= 1e-9 && df3 <= 1e-6);
  }
  // test_isolated_atom: one atom, no bonds -> finite energy, zero force (Review Focus 2)
  {
    Cell c;
    c.n = 1; c.type = {1}; c.offset = {0, 0}; c.volume = 8000;
    const Result r2 = run(d2, c, 1, true), r1 = run(d1, c, 1, false), r3 = run(d3, c, 1, false);
    CHECK(std::isfinite(r2.e) && std::isfinite(r1.e) && std::isfinite(r3.e));
    for (int q = 0; q < 3; q++) CHECK(r2.f[q] == 0.0 && r1.f[q] == 0.0 && r3.f[q] == 0.0);
    std::printf("isolated W: E(2L) %.6f  E(1L) %.6f\n", r2.e, r1.e);
  }
  {
    const Cell c = load_cell(fx + "/WMo54.npz");
    check_unmapped(d1, m1, c, "1L");
    check_unmapped(d2, m2, c, "2L");
    check_unmapped(d3, m3, c, "3L");
  }
  // test_usable_after_failed_growth: a growth that cannot be allocated (nmax_nodes = INT_MAX: terabytes) throws
  // std::bad_alloc; the next call, with more atoms than before the failure, grows again and gives the same result
  {
    const Cell c = load_cell(fx + "/WMo54.npz");
    const Result a2 = run(d2, c, 1, true), a3 = run(d3, c, 1, false);
    for (int k = 0; k < 2; k++) {
      bool threw = false;
      try {
        if (k == 0) run(d2, c, 1, true, INT_MAX);
        else run(d3, c, 1, false, INT_MAX);
      } catch (const std::bad_alloc &) {
        threw = true;
      }
      CHECK(threw);
    }
    const Result b2 = run(d2, c, 1, true, 4 * c.n), b3 = run(d3, c, 1, false, 4 * c.n);
    CHECK(a2.e == b2.e && a2.f == b2.f && a3.e == b3.e && a3.f == b3.f);
    grace::fused::ThreadForces tf;
    tf.ensure(2, 10);
    bool threw = false;
    try {
      tf.ensure(1 << 16, 1 << 20);    // 26 TB
    } catch (const std::bad_alloc &) {
      threw = true;
    }
    CHECK(threw);
    tf.ensure(1 << 16, 100);
    CHECK(tf.buf.size() == (size_t) (1 << 16) * 100 * 3 && tf.of((1 << 16) - 1)[299] == 0.0);
    std::printf("usable after a failed growth: 2L, 3L, ThreadForces\n");
  }
  return 0;
}

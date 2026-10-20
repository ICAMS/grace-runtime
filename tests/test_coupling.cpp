// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Two-group coupling (grace/coupling.h) of the fused CPU drivers on WMo54 (group 1: W, group 2: Mo):
//   - eq 2 at lambda = 0 (r' = r) equals the uncoupled run bit for bit;
//   - forces, dE/dlambda and the virial are the derivatives of the coupled energy (central differences in the atom
//     positions, lambda and a homogeneous strain of the bond vectors), eq 1..4 at lambda = 0.35 (eq 1 drops the
//     bonds with r > 0.65 rc); the same differences of the uncoupled energy give the fp32 noise level;
//   - an energy-only call gives the same energy;
//   - energy, dE/dlambda, virial and forces equal the TF reference (coupling_ref.h): scattered groups, eq 1..4 at
//     lambda 0.3 / 0.7.
#include "grace/fused_1l.h"
#include "grace/fused_2l.h"
#include "grace/fused_3l.h"
#include "grace/fused_cpu_types.h"
#include "coupling_ref.h"
#include "grace_cnpy/cnpy.h"
#include "test_util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {
struct Cell {
  int n = 0;
  std::vector<int> type;     // LAMMPS types: W = 1, Mo = 2
  std::vector<int64_t> offset;
  std::vector<int> j;
  std::vector<double> rij;
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
  return c;
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
  double e = 0, dedl = 0, vir[6] = {0, 0, 0, 0, 0, 0};
  std::vector<double> f;
};
// eq 0: uncoupled
// mask: 1 group 1, 2 group 2, 0 neither (default: W group 1, Mo group 2)
template<class D> Result run(D &drv, const Cell &c, int eq, double lam, bool eonly = false,
                             const std::vector<int> *gmask = nullptr)
{
  Result r;
  r.f.assign(3 * c.n, 0);
  std::vector<int> mask(c.n);
  for (int a = 0; a < c.n; a++) mask[a] = gmask ? (*gmask)[a] : c.type[a] == 1 ? 1 : 2;
  grace::CsrBonds src{c.type.data(), c.offset.data(), c.j.data(), c.rij.data()};
  grace::Outputs out;
  out.f = r.f.data();
  grace::ComputeFlags fl;
  fl.virial = true;
  fl.energy_only = eonly;
  fl.couple.eq = eq; fl.couple.lambda = lam; fl.couple.mask = mask.data(); fl.couple.bit1 = 1; fl.couple.bit2 = 2;
  std::vector<int> ilist(c.n);
  std::iota(ilist.begin(), ilist.end(), 0);
  if constexpr (std::is_same_v<D, grace::Grace1LFused>) {
    drv.compute(c.n, ilist.data(), c.n, c.n, src, out, fl, 1);
  } else {
    grace::NoExchange ex;
    drv.compute(c.n, ilist.data(), c.n, c.n, src, ex, out, fl, 1);
  }
  r.e = out.energy;
  r.dedl = out.dedl;
  for (int k = 0; k < 6; k++) r.vir[k] = out.virial[k];
  return r;
}
Cell displaced(const Cell &c, int atom, int d, double h)
{
  Cell m = c;
  for (int i = 0; i < c.n; i++)
    for (int64_t q = c.offset[i]; q < c.offset[i + 1]; q++)
      m.rij[3 * q + d] += (c.j[q] == atom ? h : 0.0) - (i == atom ? h : 0.0);
  return m;
}
Cell strained(const Cell &c, int a, int b, double e)    // r_b += e r_a (and r_a += e r_b off the diagonal)
{
  Cell m = c;
  for (size_t q = 0; q < c.rij.size() / 3; q++) {
    m.rij[3 * q + b] += e * c.rij[3 * q + a];
    if (a != b) m.rij[3 * q + a] += e * c.rij[3 * q + b];
  }
  return m;
}
struct FDErr {
  double f = 0, v = 0;    // max |F - F_fd|, max |vir - vir_fd| / max |vir|
};
template<class D> FDErr fd_errors(D &drv, const Cell &c, int eq, double lam)
{
  FDErr err;
  const Result r = run(drv, c, eq, lam);
  const double h = 5e-3;
  for (int atom : {0, 7, 20, 41})
    for (int d = 0; d < 3; d++) {
      const double ep = run(drv, displaced(c, atom, d, h), eq, lam).e, em = run(drv, displaced(c, atom, d, -h), eq, lam).e;
      err.f = std::max(err.f, std::fabs(r.f[3 * atom + d] + (ep - em) / (2 * h)));
    }
  // virial xx yy zz xy xz yz = -dE/d(strain); the off-diagonal strain is symmetric: -(v_ab + v_ba) = -2 v_ab
  const int ab[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
  const double e = 1e-3;
  double vs = 1.0;
  for (int k = 0; k < 6; k++) vs = std::max(vs, std::fabs(r.vir[k]));
  for (int k = 0; k < 6; k++) {
    const double ep = run(drv, strained(c, ab[k][0], ab[k][1], e), eq, lam).e;
    const double em = run(drv, strained(c, ab[k][0], ab[k][1], -e), eq, lam).e;
    const double vfd = -(ep - em) / (2 * e) / (k < 3 ? 1.0 : 2.0);
    err.v = std::max(err.v, std::fabs(r.vir[k] - vfd) / vs);
  }
  return err;
}
template<class D> void check_driver(D &drv, const char *tag, const Cell &c)
{
  // lambda = 0, eq 2: r' = r, no bond dropped -> the uncoupled result bit for bit
  const Result off = run(drv, c, 0, 0.0), z = run(drv, c, 2, 0.0);
  CHECK(off.e == z.e && off.f == z.f);
  for (int k = 0; k < 6; k++) CHECK(off.vir[k] == z.vir[k]);
  const FDErr base = fd_errors(drv, c, 0, 0.0);
  std::printf("%s uncoupled: FD max|dF| %.2e  max|dvir|/max|vir| %.2e (fp32 noise level)\n", tag, base.f, base.v);
  for (int eq = 1; eq <= 4; eq++) {
    const double lam = 0.35, dl = 1e-3;
    const Result r = run(drv, c, eq, lam);
    const FDErr fe = fd_errors(drv, c, eq, lam);
    const double dedl_fd = (run(drv, c, eq, lam + dl).e - run(drv, c, eq, lam - dl).e) / (2 * dl);
    const Result eo = run(drv, c, eq, lam, true);
    std::printf("%s eq %d lambda %.2f: E - E(uncoupled) %+.4f  dE/dlambda %.5f (FD %.5f)  FD max|dF| %.2e  "
                "max|dvir|/max|vir| %.2e  energy-only dE %.1e\n", tag, eq, lam, r.e - off.e, r.dedl, dedl_fd, fe.f,
                fe.v, std::fabs(eo.e - r.e));
    CHECK(std::fabs(r.e - off.e) > 1e-2);    // the coupling changes the energy
    CHECK(fe.f <= std::max(3 * base.f, 1e-3));
    CHECK(fe.v <= std::max(3 * base.v, 1e-4));
    CHECK(std::fabs(r.dedl - dedl_fd) <= 1e-3 * std::fabs(r.dedl) + 1e-2);
    CHECK(eo.e == r.e && eo.dedl == 0.0);
  }
}
template<class D> void check_tf(D &drv, const char *tag, const Cell &c, const std::string &ref)
{
  const cplref::Ref r = cplref::load(ref, c.n);
  for (const cplref::Case &k : r.cases) {
    const Result g = run(drv, c, k.eq, k.lambda, false, &r.mask);
    cplref::compare(tag, k, g.e, g.dedl, g.vir, g.f);
  }
}
}    // namespace

int main()
{
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  const Cell c = load_cell(fx + "/WMo54.npz");
  auto m1 = std::make_shared<grace::GRACE1LModel>();
  m1->load(test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"));
  grace::Grace1LFused d1(m1, {}, {});
  d1.setup_types(2, type2mu_for(m1->element_names));
  check_driver(d1, "1L", c);
  check_tf(d1, "1L", c, fx + "/ref_cpl/WMo54_1l.txt");
  auto m2 = std::make_shared<grace::GRACE2LModel>();
  m2->load(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"));
  grace::Grace2LFused d2(m2, {}, {});
  d2.setup_types(2, type2mu_for(m2->element_names));
  check_driver(d2, "2L", c);
  check_tf(d2, "2L", c, fx + "/ref_cpl/WMo54_2l.txt");
  auto m3 = std::make_shared<grace::GRACE3LModel>();
  m3->load(test_model("GRACE_TEST_M3", "GRACE-3L-OMAT-large"));
  grace::Grace3LFused d3(m3, {}, {});
  d3.setup_types(2, type2mu_for(m3->element_names));
  check_driver(d3, "3L", c);
  check_tf(d3, "3L", c, fx + "/ref_cpl/WMo54_3l.txt");
  std::printf("coupling ok\n");
  return 0;
}

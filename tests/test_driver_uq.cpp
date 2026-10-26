// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// UQ (uqv6 gamma_only) of the drivers on the CsrBonds path: errors, energies/forces unchanged by UQ, results
// independent of the thread count, finite values for an isolated atom; ComputeFlags::energy_only gives the same
// energy, per-atom energy and UQ without touching forces / virial. Values against TPCalculator: python/tests.
// Exit 77 = skipped (the schema-6 models are not in the repository).
#include "grace/fused_1l.h"
#include "grace/fused_2l.h"
#include "grace/fused_3l.h"
#include "grace_cnpy/cnpy.h"
#include "test_util.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
struct Cell {
  int n = 0;
  std::vector<int> type;
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
  return t2m;
}
struct Res {
  double e = 0;
  double vir[6] = {0, 0, 0, 0, 0, 0};
  std::vector<double> f, eatom, gamma, sigma, cluster;
};
template<class D> Res run(D &drv, const Cell &c, int nth, bool uq, bool energy_only = false)
{
  Res r;
  r.f.assign(3 * c.n, 0);
  r.eatom.assign(c.n, 0);
  r.gamma.assign(c.n, -1); r.sigma.assign(c.n, -1); r.cluster.assign(c.n, -1);
  grace::CsrBonds src{c.type.data(), c.offset.data(), c.j.data(), c.rij.data()};
  grace::Outputs out;
  out.f = r.f.data();
  out.eatom = r.eatom.data();
  out.gamma = r.gamma.data(); out.atomic_sigma = r.sigma.data(); out.gmm_cluster = r.cluster.data();
  grace::ComputeFlags fl;
  fl.uq = uq;
  fl.eflag_atom = true;
  fl.virial = true;
  fl.energy_only = energy_only;
  if constexpr (std::is_same_v<D, grace::Grace2LFused> || std::is_same_v<D, grace::Grace3LFused>) {
    grace::NoExchange ex;
    drv.compute(c.n, nullptr, c.n, c.n, src, ex, out, fl, nth);
  } else {
    drv.compute(c.n, nullptr, c.n, c.n, src, out, fl, nth);
  }
  r.e = out.energy;
  for (int k = 0; k < 6; k++) r.vir[k] = out.virial[k];
  return r;
}
template<class D> void check_driver(const char *name, D &drv, const Cell &c)
{
  const Res off = run(drv, c, 1, false), on1 = run(drv, c, 1, true), on3 = run(drv, c, 3, true),
            on8 = run(drv, c, 8, true);
  CHECK(off.e == on1.e);
  for (int q = 0; q < 3 * c.n; q++) CHECK(off.f[q] == on1.f[q]);
  double gmax = 0;
  for (int a = 0; a < c.n; a++) {
    CHECK(std::isfinite(on1.gamma[a]) && on1.gamma[a] > 0 && std::isfinite(on1.sigma[a]));
    CHECK(on1.cluster[a] >= 0 && on1.cluster[a] < 8 && on1.cluster[a] == std::floor(on1.cluster[a]));
    CHECK(on1.gamma[a] == on3.gamma[a] && on1.gamma[a] == on8.gamma[a]);    // one thread per atom, fixed order
    CHECK(on1.sigma[a] == on8.sigma[a] && on1.cluster[a] == on8.cluster[a]);
    gmax = std::max(gmax, on1.gamma[a]);
  }
  // energy only (LAMMPS ENERGY_ONLY, MC trial moves): same energy, per-atom energy and UQ, forces / virial untouched
  const Res eo = run(drv, c, 1, true, true), eo4 = run(drv, c, 4, true, true);
  CHECK(eo.e == on1.e);
  for (int a = 0; a < c.n; a++)
    CHECK(eo.eatom[a] == on1.eatom[a] && eo.gamma[a] == on1.gamma[a] && eo.sigma[a] == on1.sigma[a] &&
          eo.cluster[a] == on1.cluster[a] && eo4.gamma[a] == on1.gamma[a]);
  for (int q = 0; q < 3 * c.n; q++) CHECK(eo.f[q] == 0.0 && eo4.f[q] == 0.0);
  for (int k = 0; k < 6; k++) CHECK(eo.vir[k] == 0.0 && on1.vir[k] != 0.0);
  CHECK(std::fabs(eo4.e - on1.e) <= 1e-9 * std::fabs(on1.e));
  std::printf("%-10s max gamma %.4f  gamma[0] %.6f sigma[0] %.6f cluster[0] %g\n", name, gmax, on1.gamma[0],
              on1.sigma[0], on1.cluster[0]);
}
bool exists(const char *p) { return std::ifstream(p).good(); }
}    // namespace

int main()
{
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  const char *p1 = test_model("GRACE_TEST_UQ1", "GRACE-1L-OMAT-large-base");
  const char *p2 = test_model("GRACE_TEST_UQ2", "GRACE-2L-OMAT-medium-ft-E");
  // a 2L model without a UQ head: GRACE_TEST_NOUQ (python/tests/conftest.py strip_uq writes one); unset: not checked
  const char *p2n = env_or("GRACE_TEST_NOUQ", "");
  if (!exists(p1) || !exists(p2)) { std::printf("v6 models absent: skipped\n"); return 77; }
  auto m1 = std::make_shared<grace::GRACE1LModel>();
  m1->load(p1);
  auto m2 = std::make_shared<grace::GRACE2LModel>();
  m2->load(p2);
  grace::Grace1LFused d1(m1, {}, {});
  grace::Grace2LFused d2(m2, {}, {});
  d1.setup_types(2, type2mu_for(m1->element_names));
  d2.setup_types(2, type2mu_for(m2->element_names));
  const Cell c = load_cell(fx + "/WMo54.npz");

  bool threw = false;    // UQ before enable_uq()
  try { run(d1, c, 1, true); } catch (const grace::Error &) { threw = true; }
  CHECK(threw);
  d1.enable_uq();
  d2.enable_uq();
  CHECK(d1.uq_enabled() && d2.uq_enabled());
  check_driver("1L WMo54", d1, c);
  check_driver("2L WMo54", d2, c);
  auto m3 = std::make_shared<grace::GRACE3LModel>();
  m3->load(test_model("GRACE_TEST_M3", "GRACE-3L-OMAT-large"));
  grace::Grace3LFused d3(m3, {}, {});
  d3.setup_types(2, type2mu_for(m3->element_names));
  threw = false;    // UQ before enable_uq()
  try { run(d3, c, 1, true); } catch (const grace::Error &) { threw = true; }
  CHECK(threw);
  d3.enable_uq();
  check_driver("3L WMo54", d3, c);
  {    // 3L without the stored forward tensors (the reverse recomputes layers 1-2): same energy, forces and UQ
    grace::FusedSettings s;
    s.store_fwd = false;
    grace::Grace3LFused d3r(m3, s, {});
    d3r.setup_types(2, type2mu_for(m3->element_names));
    d3r.enable_uq();
    // one thread: with several, each thread's force rows hold the blocks it got (dynamic schedule), last bits vary
    const Res a = run(d3, c, 1, true), b = run(d3r, c, 1, true);
    CHECK(a.e == b.e);
    for (int q = 0; q < c.n; q++) CHECK(a.gamma[q] == b.gamma[q] && a.cluster[q] == b.cluster[q]);
    for (int q = 0; q < 3 * c.n; q++) CHECK(a.f[q] == b.f[q]);
  }
  {    // 2L without stored layer 1 (P5 recomputes layer 1): same UQ
    grace::FusedSettings s;
    s.store_l1 = 0;
    grace::Grace2LFused d2r(m2, s, {});
    d2r.setup_types(2, type2mu_for(m2->element_names));
    d2r.enable_uq();
    const Res a = run(d2, c, 2, true), b = run(d2r, c, 2, true);
    for (int q = 0; q < c.n; q++) CHECK(a.gamma[q] == b.gamma[q] && a.cluster[q] == b.cluster[q]);
  }
  {    // isolated atom: zero basis except the A self term (no bonds): finite
    Cell iso;
    iso.n = 1; iso.type = {1}; iso.offset = {0, 0};
    const Res r1 = run(d1, iso, 1, true), r2 = run(d2, iso, 1, true), r3 = run(d3, iso, 1, true);
    CHECK(std::isfinite(r1.gamma[0]) && std::isfinite(r2.gamma[0]) && std::isfinite(r3.gamma[0]));
    std::printf("isolated W: gamma 1L %.4g 2L %.4g 3L %.4g\n", r1.gamma[0], r2.gamma[0], r3.gamma[0]);
  }
  {    // no atoms (a LAMMPS rank without local atoms): UQ on with null outputs is fine, nothing is written
    Cell none;
    none.offset = {0};
    grace::Outputs out;    // all outputs null
    std::vector<double> f(3, 0.0);
    out.f = f.data();
    grace::ComputeFlags fl;
    fl.uq = true;
    grace::CsrBonds src{nullptr, none.offset.data(), nullptr, nullptr};
    grace::NoExchange ex;
    out.energy = 0.0;    // compute adds to it: nothing to add without atoms
    d1.compute(0, nullptr, 0, 1, src, out, fl, 1);
    CHECK(out.energy == 0.0);
    out.energy = 0.0;
    d2.compute(0, nullptr, 0, 1, src, ex, out, fl, 1);
    CHECK(out.energy == 0.0);
    out.energy = 0.0;
    d3.compute(0, nullptr, 0, 1, src, ex, out, fl, 1);
    CHECK(out.energy == 0.0);
    // with atoms the null outputs are an error (1L, 2L, 3L)
    const Cell cw = load_cell(fx + "/WMo54.npz");
    grace::CsrBonds csrc{cw.type.data(), cw.offset.data(), cw.j.data(), cw.rij.data()};
    std::vector<double> fc(3 * cw.n, 0.0);
    out.f = fc.data();
    int threw_null = 0;
    try { d1.compute(cw.n, nullptr, cw.n, cw.n, csrc, out, fl, 1); } catch (const grace::Error &) { threw_null++; }
    try { d2.compute(cw.n, nullptr, cw.n, cw.n, csrc, ex, out, fl, 1); } catch (const grace::Error &) { threw_null++; }
    try { d3.compute(cw.n, nullptr, cw.n, cw.n, csrc, ex, out, fl, 1); } catch (const grace::Error &) { threw_null++; }
    CHECK(threw_null == 3);
    std::printf("UQ with null outputs: no atoms fine, with atoms an error (1L/2L/3L)\n");
  }
  if (*p2n && exists(p2n)) {    // a model without a UQ head
    auto mn = std::make_shared<grace::GRACE2LModel>();
    mn->load(p2n);
    grace::Grace2LFused dn(mn, {}, {});
    dn.setup_types(2, type2mu_for(mn->element_names));
    threw = false;
    try { dn.enable_uq(); } catch (const grace::Error &) { threw = true; }
    CHECK(threw && !dn.uq_enabled());
  }
  std::printf("driver uq ok\n");
  return 0;
}

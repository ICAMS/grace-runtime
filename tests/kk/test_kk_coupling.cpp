// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Two-group coupling (grace/coupling.h) of the fused Kokkos engines against the fused CPU drivers (whose forces,
// dE/dlambda and virial test_coupling checks against finite differences): WMo54, group 1 W, group 2 Mo, eq 1..4
// at lambda = 0.35, GRACE-3L also in chunks of 16 atoms; eq 2 at lambda = 0 against the uncoupled run; the TF
// reference (coupling_ref.h: scattered groups, eq 1..4 at lambda 0.3 / 0.7).
#include "grace/fused_1l.h"
#include "grace/fused_2l.h"
#include "grace/fused_3l.h"
#include "grace/kk/csr_images.h"
#include "grace/kk/grace_1l_fused_kk_impl.h"
#include "grace/kk/grace_2l_fused_kk_impl.h"
#include "grace/kk/grace_3l_fused_kk.h"
#include "grace/kk/grace_3l_kk_impl.h"
#include "grace/kk/traits.h"
#include "coupling_ref.h"

#include <numeric>

using Dev = Kokkos::DefaultExecutionSpace;
using T = grace::kk::DefaultTraits<Dev>;
#include "kk_csr_util.h"
using E1F = grace::kk::Grace1LFusedKK<Dev, T>;
using E2F = grace::kk::Grace2LFusedKK<Dev, T>;
using E3F = grace::kk::Grace3LFusedKK<Dev, T>;
template class grace::kk::Grace1LFusedKK<Dev, T>;
template class grace::kk::Grace2LFusedKK<Dev, T>;
template class grace::kk::Grace3LKK<Dev, float, float, T>;

namespace {
using namespace kkt;

template<class D> Result cpu_run(D &drv, const Cell &c, const Couple &cp)
{
  Result r;
  r.f.assign(3 * c.n, 0);
  grace::CsrBonds src{c.type.data(), c.offset.data(), c.j.data(), c.rij.data()};
  grace::Outputs out;
  out.f = r.f.data();
  grace::ComputeFlags fl;
  fl.virial = true;
  fl.couple.eq = cp.eq; fl.couple.lambda = cp.lambda; fl.couple.mask = cp.mask.data();
  fl.couple.bit1 = 1; fl.couple.bit2 = 2;
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

void compare(const std::string &what, const Result &g, const Result &c, int n)
{
  double df = 0, dv = 0, vs = 1;
  for (int q = 0; q < 3 * n; q++) df = std::max(df, std::fabs(g.f[q] - c.f[q]));
  for (int k = 0; k < 6; k++) {
    dv = std::max(dv, std::fabs(g.vir[k] - c.vir[k]));
    vs = std::max(vs, std::fabs(c.vir[k]));
  }
  const double dd = std::fabs(g.dedl - c.dedl);
  std::printf("%-22s dE/n %.2e  max|dF| %.2e  max|dvir|/max|vir| %.2e  dE/dlambda %.5f vs CPU %.5f (diff %.1e)\n",
              what.c_str(), std::fabs(g.e - c.e) / n, df, dv / vs, g.dedl, c.dedl, dd);
  CHECK(std::fabs(g.e - c.e) / n <= 2e-5);
  CHECK(df <= 2e-5);
  CHECK(dv / vs <= 2e-5);
  CHECK(dd <= 1e-4 * std::fabs(c.dedl) + 1e-3);
}

template<class Eng> void check_tf(Eng &eng, const char *tag, const Cell &c, const std::string &ref)
{
  const cplref::Ref r = cplref::load(ref, c.n);
  Couple cp;
  cp.mask = r.mask;
  for (const cplref::Case &k : r.cases) {
    cp.eq = k.eq; cp.lambda = k.lambda;
    const Result g = run(eng, c, 1, 0, true, &cp);
    cplref::compare(tag, k, g.e, g.dedl, g.vir, g.f);
  }
}

template<class Eng, class D> void check(Eng &eng, D &drv, const char *tag, const Cell &c)
{
  Couple cp;
  for (int a = 0; a < c.n; a++) cp.mask.push_back(c.type[a] == 1 ? 1 : 2);
  {    // eq 2 at lambda = 0: r' = r -> the uncoupled run (fp32 run-to-run level on the GPU)
    cp.eq = 2; cp.lambda = 0.0;
    const Result a = run(eng, c), b = run(eng, c, 1, 0, true, &cp);
    double df = 0;
    for (int q = 0; q < 3 * c.n; q++) df = std::max(df, std::fabs(a.f[q] - b.f[q]));
    std::printf("%-22s eq 2 lambda 0 vs uncoupled: dE/n %.2e  max|dF| %.2e\n", tag, std::fabs(a.e - b.e) / c.n, df);
    CHECK(std::fabs(a.e - b.e) / c.n <= 1e-6 && df <= 1e-5);
  }
  for (int eq = 1; eq <= 4; eq++) {
    cp.eq = eq; cp.lambda = 0.35;
    compare(std::string(tag) + " eq " + std::to_string(eq), run(eng, c, 1, 0, true, &cp), cpu_run(drv, c, cp), c.n);
  }
}
}    // namespace

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  const Cell c = load_cell(fx + "/WMo54.npz");
  {
    auto m1 = std::make_shared<grace::GRACE1LModel>();
    m1->load(test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"));
    E1F e; e.set_model(m1, type2mu_for(m1->element_names)); e.setup();
    grace::Grace1LFused d(m1, {}, {});
    d.setup_types(2, type2mu_for(m1->element_names));
    check(e, d, "1L fused", c);
    check_tf(e, "1L fused", c, fx + "/ref_cpl/WMo54_1l.txt");
  }
  {
    auto m2 = std::make_shared<grace::GRACE2LModel>();
    m2->load(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"));
    E2F e; e.set_model(m2, type2mu_for(m2->element_names)); e.setup();
    grace::Grace2LFused d(m2, {}, {});
    d.setup_types(2, type2mu_for(m2->element_names));
    check(e, d, "2L fused", c);
    check_tf(e, "2L fused", c, fx + "/ref_cpl/WMo54_2l.txt");
  }
  {
    auto m3 = std::make_shared<grace::GRACE3LModel>();
    m3->load(test_model("GRACE_TEST_M3", "GRACE-3L-OMAT-large"));
    grace::Grace3LFused d(m3, {}, {});
    d.setup_types(2, type2mu_for(m3->element_names));
    E3F e; e.set_model(m3, type2mu_for(m3->element_names)); e.setup();
    check(e, d, "3L fused", c);
    check_tf(e, "3L fused", c, fx + "/ref_cpl/WMo54_3l.txt");
    E3F ec; ec.chunksize = 16; ec.set_model(m3, type2mu_for(m3->element_names)); ec.setup();
    check(ec, d, "3L fused chunks of 16", c);
    check_tf(ec, "3L fused chunks of 16", c, fx + "/ref_cpl/WMo54_3l.txt");
  }
  std::printf("kk coupling ok\n");
  return 0;
}

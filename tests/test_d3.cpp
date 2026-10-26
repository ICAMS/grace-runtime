// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// DFT-D3(BJ) dispersion, CPU (grace/d3.h, grace/cell_list.h): energy, per-atom energies, forces and virial against
// simple-dftd3 (tests/fixtures/d3_ref.npz, two- and three-body cases), forces against finite differences,
// thread-count and energy-only consistency, the formula functions (switch, weights far from the references), input
// errors.
#include "d3_ref.h"
#include "test_util.h"

#include <algorithm>
#include <string>
#include <vector>

namespace {

using d3_ref::Case;
using d3_ref::maxabs;
using d3_ref::Out;

Out run(const Case &c, const std::vector<double> &x, int threads, bool energy_only = false)
{
  return d3_ref::run_cpu(c, x, threads, energy_only);
}

void test_formulas()
{
  // switch: value and slope continuous at cutoff_on and cutoff
  const double r06 = 1e3, r08 = 1e4, qq = 20.0, rc = 15.0, ron = 12.0;
  double d1, d2;
  const double g1 = grace::d3::pair(ron - 1e-9, r06, r08, qq, 1.0, 0.8, rc, ron, d1);
  const double g2 = grace::d3::pair(ron + 1e-9, r06, r08, qq, 1.0, 0.8, rc, ron, d2);
  CHECK_NEAR(g1, g2, 1e-8 * std::fabs(g1));    // the value moves by slope * 2e-9 across ron
  CHECK_NEAR(d1, d2, 1e-6 * std::fabs(d1));
  const double g3 = grace::d3::pair(rc - 1e-6, r06, r08, qq, 1.0, 0.8, rc, ron, d1);
  CHECK(std::fabs(g3) < 1e-20 && std::fabs(d1) < 1e-14);
  // hard cutoff (cutoff_on = cutoff) and r rounded to the cutoff: zero, not 0 / 0
  const double g4 = grace::d3::pair(rc, r06, r08, qq, 1.0, 0.8, rc, rc, d1);
  CHECK(g4 == 0.0 && d1 == 0.0);
  const grace::d3::AtmSide<double> s4 = grace::d3::atm_side<double>(rc * rc, rc, rc);
  CHECK(s4.sw == 0.0 && s4.dsw == 0.0);
  // dg/dr against a central difference inside the switch
  const double r = 13.1, h = 1e-6;
  double dum;
  grace::d3::pair(r, r06, r08, qq, 1.0, 0.8, rc, ron, d1);
  const double fd = (grace::d3::pair(r + h, r06, r08, qq, 1.0, 0.8, rc, ron, dum) -
                     grace::d3::pair(r - h, r06, r08, qq, 1.0, 0.8, rc, ron, dum)) / (2 * h);
  CHECK_NEAR(d1, fd, 1e-6 * std::fabs(fd));
  // counting function derivative
  double dfdr;
  grace::d3::count(2.0, 1.9, dfdr);
  const double fdc = (grace::d3::count(2.0, 1.9 + h, dum) - grace::d3::count(2.0, 1.9 - h, dum)) / (2 * h);
  CHECK_NEAR(dfdr, fdc, 1e-6 * std::fabs(fdc));
  // weights: far from all references (C: CNref 0..4) -> the largest reference, finite derivative
  double w[grace::d3::kMaxRef], dw[grace::d3::kMaxRef];
  const double *cref = grace::d3_data::kCNRef[6];
  grace::d3::weights(60.0, cref, grace::d3_data::kNRef[6], w, dw);
  for (int a = 0; a < grace::d3::kMaxRef; a++) {
    CHECK(std::isfinite(w[a]) && std::isfinite(dw[a]));
    CHECK(w[a] == (a == 4 ? 1.0 : 0.0));
  }
  grace::d3::weights(2.2, cref, grace::d3_data::kNRef[6], w, dw);
  double sw = 0.0, sdw = 0.0;
  for (int a = 0; a < grace::d3::kMaxRef; a++) {
    sw += w[a];
    sdw += dw[a];
  }
  CHECK_NEAR(sw, 1.0, 1e-14);
  CHECK_NEAR(sdw, 0.0, 1e-13);
  // dw/dCN against a central difference, near the references and ~10 from all of them (Gaussians ~1e-174: the
  // largest reference has w = 1 to round-off, so dw ~ 0; the form g dnorm / norm^2 gave -80 there)
  for (const double cn : {2.7, 7.0, 14.0}) {
    double wp[grace::d3::kMaxRef], wm[grace::d3::kMaxRef], dwx[grace::d3::kMaxRef];
    grace::d3::weights(cn, cref, grace::d3_data::kNRef[6], w, dw);
    grace::d3::weights(cn + h, cref, grace::d3_data::kNRef[6], wp, dwx);
    grace::d3::weights(cn - h, cref, grace::d3_data::kNRef[6], wm, dwx);
    for (int a = 0; a < grace::d3::kMaxRef; a++)
      CHECK_NEAR(dw[a], (wp[a] - wm[a]) / (2 * h), 1e-6 * std::max(1.0, std::fabs(dw[a])));
  }
}

void test_errors()
{
  grace::DispersionParams p;
  p.a1 = 0.4;
  p.a2 = 2.0;
  const int bad[1] = {104};
  bool thrown = false;
  try { grace::d3::make_tables(bad, 1, p); } catch (const std::invalid_argument &) { thrown = true; }
  CHECK(thrown);
  thrown = false;
  try { grace::d3bj_preset("PBE0-unknown"); } catch (const std::invalid_argument &) { thrown = true; }
  CHECK(thrown);
  p.cutoff_on = p.cutoff + 1.0;
  thrown = false;
  try { grace::validate(p); } catch (const std::invalid_argument &) { thrown = true; }
  CHECK(thrown);
  p.cutoff_on = 12.0;
  p.kind = 7;
  thrown = false;
  try { grace::validate(p); } catch (const std::invalid_argument &) { thrown = true; }
  CHECK(thrown);
  // the presets are the published numbers (a2 in Bohr)
  const auto &q = grace::d3bj_preset("pbe");
  CHECK(q.a1 == 0.4289 && q.s8 == 0.7875 && q.a2_bohr == 4.4407 && q.s6 == 1.0);
  CHECK(grace::d3bj_preset("r2scan").a1 == 0.49484001);
}

}    // namespace

int main()
{
  test_formulas();
  test_errors();
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  auto npz = grace::cnpy::npz_load(fx + "/d3_ref.npz");
  const int ncase = (int) *npz["ncase"].data<int64_t>();
  CHECK(ncase >= 15);
  for (int k = 0; k < ncase; k++) {
    const Case c = d3_ref::load_case(npz, k);
    const Out o = run(c, c.x, 4);
    const double ae = std::fabs(c.energy);
    double dv = 0.0;
    for (int m = 0; m < 6; m++) dv = std::max(dv, std::fabs(o.virial[m] - c.virial[m]));
    std::printf("case %2d: n %3zu  E %+.12f  dE %.1e  dE_i %.1e  dF %.1e  dW %.1e\n", k, c.z.size(), o.energy,
                o.energy - c.energy, maxabs(o.energies, c.energies), maxabs(o.forces, c.forces), dv);
    CHECK_NEAR(o.energy, c.energy, 1e-11 * ae);
    CHECK(maxabs(o.energies, c.energies) <= 1e-11 * ae);
    CHECK(maxabs(o.forces, c.forces) <= 1e-11);
    CHECK(dv <= 1e-10 * std::max(1.0, ae));
    // the same call repeats bit for bit; thread count and energy-only give the same energy
    const Out o4 = run(c, c.x, 4);
    CHECK(o4.energy == o.energy && o4.forces == o.forces && o4.energies == o.energies);
    for (int m = 0; m < 6; m++) CHECK(o4.virial[m] == o.virial[m]);
    const Out o1 = run(c, c.x, 1), oe = run(c, c.x, 3, true);
    CHECK_NEAR(o1.energy, o.energy, 1e-13 * ae);
    CHECK(maxabs(o1.forces, o.forces) <= 1e-14);
    CHECK_NEAR(oe.energy, o.energy, 1e-13 * ae);
    CHECK(maxabs(oe.energies, o.energies) <= 1e-14 * ae);
  }
  // forces against a central difference of the energy (water, switch 12-15 A; Cu with the three-body term, switch
  // 6-8 A); a few atoms and directions
  for (const int k : {0, 19}) {
    const Case c = d3_ref::load_case(npz, k);
    CHECK((k == 19) == (c.p.s9 != 0.0));
    const Out o = run(c, c.x, 4);
    const double h = 1e-4;
    for (int a : {0, 1, 2, 17, 31})
      for (int d = 0; d < 3; d++) {
        std::vector<double> xp = c.x, xm = c.x;
        xp[3 * a + d] += h;
        xm[3 * a + d] -= h;
        const double fd = -(run(c, xp, 4, true).energy - run(c, xm, 4, true).energy) / (2 * h);
        CHECK_NEAR(o.forces[3 * a + d], fd, 1e-7);
      }
  }
  std::printf("test_d3: OK\n");
  return 0;
}

// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// DFT-D3(BJ) dispersion on the device (grace/kk/d3_kk.h): over the bins of a NeighborListKK built at a short model
// cutoff, against simple-dftd3 (tests/fixtures/d3_ref.npz) and against the CPU driver (grace/d3.h) after the atoms
// moved by up to skin / 2 since the build (the Verlet condition of the list), energy-only consistency.
#include "grace/kk/d3_kk.h"
#include "grace/kk/neighbors_kk.h"
#include "../d3_ref.h"
#include "../test_util.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using Dev = Kokkos::DefaultExecutionSpace;

namespace {

using d3_ref::Case;
using d3_ref::maxabs;
using d3_ref::Out;

// device: list built at x_build with the model-like cutoff rlist (+ skin), D3 evaluated at x
Out run_device(const Case &c, const std::vector<double> &x_build, const std::vector<double> &x, double rlist,
               double skin, bool energy_only)
{
  std::vector<int> type;
  const auto sp = grace::d3::species_of(c.z.data(), (int64_t) c.z.size(), type);
  const int n = (int) c.z.size();
  grace::kk::D3TablesKK<Dev> tk;
  tk.set(grace::d3::make_tables(sp.data(), (int) sp.size(), c.p));
  Kokkos::View<double *[3], Kokkos::LayoutRight, Dev> xb("xb", n), xc("xc", n);
  Kokkos::View<int *, Dev> ty("ty", n);
  auto hb = Kokkos::create_mirror_view(xb), hc = Kokkos::create_mirror_view(xc);
  auto ht = Kokkos::create_mirror_view(ty);
  for (int a = 0; a < n; a++) {
    for (int k = 0; k < 3; k++) {
      hb(a, k) = x_build[3 * a + k];
      hc(a, k) = x[3 * a + k];
    }
    ht(a) = type[a] + 1;    // 1-based, as the System's types
  }
  Kokkos::deep_copy(xb, hb);
  Kokkos::deep_copy(xc, hc);
  Kokkos::deep_copy(ty, ht);
  grace::kk::NeighborListKK<Dev> nl;
  nl.build(xb, n, c.cell.data(), c.pbc, rlist + skin);
  grace::kk::D3WorkKK<Dev> wk;
  const auto s = grace::kk::d3_compute<Dev>(tk, wk, xc, ty, 1, n, nl, skin, energy_only);
  Out o;
  o.energy = s.energy;
  std::copy_n(s.virial, 6, o.virial);
  auto he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), wk.e);
  o.energies.assign(he.data(), he.data() + n);
  if (!energy_only) {
    auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), wk.f);
    o.forces.assign(hf.data(), hf.data() + 3 * n);
  }
  return o;
}

double vdiff(const double *a, const double *b)
{
  double m = 0.0;
  for (int k = 0; k < 6; k++) m = std::max(m, std::fabs(a[k] - b[k]));
  return m;
}

// fp32 pair terms: tolerances relative to the scale of the quantity
void compare(const char *what, int k, const Out &o, double energy, const std::vector<double> &energies,
             const std::vector<double> &forces, const double *virial, size_t n)
{
  double fmax = 0.0, wmax = 0.0;
  for (double v : forces) fmax = std::max(fmax, std::fabs(v));
  for (int m = 0; m < 6; m++) wmax = std::max(wmax, std::fabs(virial[m]));
  const double ae = std::fabs(energy);
  const double de = std::fabs(o.energy - energy) / ae, dei = maxabs(o.energies, energies) / (ae / (double) n);
  const double df = maxabs(o.forces, forces), dw = vdiff(o.virial, virial) / std::max(wmax, 1e-3);
  std::printf("%s case %2d: n %3zu  dE/E %.1e  dE_i/(E/n) %.1e  dF %.1e (max|F| %.1e)  dW/W %.1e\n", what, k, n, de,
              dei, df, fmax, dw);
  CHECK(de < 2e-6);
  CHECK(dei < 2e-5);
  CHECK(df < 5e-6 * std::max(fmax, 5e-2));    // fp32 pair terms: ~1e-6 of the (partly cancelling) pair forces
  CHECK(dw < 1e-5);
}

}    // namespace

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  auto npz = grace::cnpy::npz_load(fx + "/d3_ref.npz");
  const int ncase = (int) *npz["ncase"].data<int64_t>();
  const double rlist = 5.0, skin = 0.5;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int k = 0; k < ncase; k++) {
    const Case c = d3_ref::load_case(npz, k);
    const size_t n = c.z.size();
    const Out o = run_device(c, c.x, c.x, rlist, skin, false);
    compare("ref", k, o, c.energy, c.energies, c.forces, c.virial, n);
    const Out oe = run_device(c, c.x, c.x, rlist, skin, true);
    CHECK_NEAR(oe.energy, o.energy, 1e-12 * std::fabs(o.energy));
    // atoms moved by up to skin / 2 (along a random direction) since the list was built: the CPU driver on the moved
    // positions is the reference
    std::vector<double> xm = c.x;
    for (size_t a = 0; a < n; a++) {
      double d[3] = {u(rng), u(rng), u(rng)};
      const double nd = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) + 1e-12;
      for (int m = 0; m < 3; m++) xm[3 * a + m] += 0.499 * skin * d[m] / nd * std::fabs(u(rng));
    }
    const Out om = run_device(c, c.x, xm, rlist, skin, false), oc = d3_ref::run_cpu(c, xm, 4);
    compare("moved", k, om, oc.energy, oc.energies, oc.forces, oc.virial, n);
    // unwrapped positions far from the cell (long MD): atoms moved by up to ~1000 cells, same result
    if (c.pbc[0] && c.pbc[1] && c.pbc[2]) {
      std::vector<double> xs = c.x;
      for (size_t a = 0; a < n; a++)
        for (int m = 0; m < 3; m++) {
          const double shift = (double) ((int) ((a * 7 + m * 3) % 2001) - 1000);
          for (int d = 0; d < 3; d++) xs[3 * a + d] += shift * c.cell[3 * m + d];
        }
      const Out os = run_device(c, xs, xs, rlist, skin, false);
      compare("shifted", k, os, c.energy, c.energies, c.forces, c.virial, n);
    }
  }
  std::printf("test_kk_d3: OK\n");
  return 0;
}

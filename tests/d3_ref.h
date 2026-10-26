// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The DFT-D3(BJ) references of tests/fixtures/d3_ref.npz (simple-dftd3, two-body and three-body cases) and the CPU
// driver on them, shared by test_d3 and test_kk_d3.
#pragma once

#include "grace/cell_list.h"
#include "grace/d3.h"
#include "grace_cnpy/cnpy.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace d3_ref {

struct Case {
  std::vector<int> z;
  std::vector<double> x, cell;    // cell full rank: open directions without a vector padded (molecules)
  bool pbc[3];
  grace::DispersionParams p;
  double energy, virial[6];
  std::vector<double> energies, forces;
};

inline Case load_case(grace::cnpy::npz_t &npz, int k)
{
  const std::string s = std::to_string(k) + "/";
  auto vec = [&](const char *key) {
    const auto &a = npz[s + key];
    return std::vector<double>(a.data<double>(), a.data<double>() + a.num_vals);
  };
  Case c;
  const auto &num = npz[s + "numbers"];
  c.z.assign(num.data<int>(), num.data<int>() + num.shape[0]);
  c.x = vec("positions");
  c.cell = vec("cell");
  for (int d = 0; d < 3; d++) c.pbc[d] = npz[s + "pbc"].data<bool>()[d];
  const double *pr = npz[s + "params"].data<double>(), *cu = npz[s + "cutoffs"].data<double>();
  c.p.a1 = pr[0];
  c.p.a2 = pr[1] * grace::kBohr;
  c.p.s6 = pr[2];
  c.p.s8 = pr[3];
  c.p.cutoff = cu[0];
  c.p.cutoff_on = cu[0] - cu[1];
  c.p.cn_cutoff = cu[2];
  const double *at = npz[s + "atm"].data<double>();    // s9, rc3, width3
  c.p.s9 = at[0];
  c.p.atm_cutoff = at[1];
  c.p.atm_cutoff_on = at[1] - at[2];
  c.energy = *npz[s + "energy"].data<double>();
  std::copy_n(npz[s + "virial"].data<double>(), 6, c.virial);
  c.energies = vec("energies");
  c.forces = vec("forces");
  // open directions without a cell vector: one longer than the extent plus 2 (reach + 1) A, as the calculator pads
  const double reach = std::max({c.p.cutoff, c.p.cn_cutoff, c.p.s9 != 0.0 ? c.p.atm_cutoff : 0.0});
  for (int d = 0; d < 3; d++) {
    const double *v = &c.cell[3 * d];
    if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] > 0.0) continue;
    double lo = 1e300, hi = -1e300;
    for (size_t a = 0; a < c.z.size(); a++) {
      lo = std::min(lo, c.x[3 * a + d]);
      hi = std::max(hi, c.x[3 * a + d]);
    }
    for (int e = 0; e < 3; e++) c.cell[3 * d + e] = e == d ? hi - lo + 2.0 * (reach + 1.0) + 1.0 : 0.0;
  }
  return c;
}

struct Out {
  double energy = 0.0, virial[6] = {0, 0, 0, 0, 0, 0};
  std::vector<double> energies, forces;
};

// the CPU driver at positions x
inline Out run_cpu(const Case &c, const std::vector<double> &x, int threads, bool energy_only = false)
{
  std::vector<int> type;
  const auto sp = grace::d3::species_of(c.z.data(), (int64_t) c.z.size(), type);
  const auto t = grace::d3::make_tables(sp.data(), (int) sp.size(), c.p);
  grace::CellList cl;
  cl.build(x.data(), (int64_t) c.z.size(), c.cell.data(), c.pbc, 0.5 * std::min(c.p.cutoff, c.p.cn_cutoff));
  Out o;
  o.energies.assign(c.z.size(), 0.0);
  o.forces.assign(3 * c.z.size(), 0.0);
  grace::d3::compute_cpu(t, type.data(), (int64_t) c.z.size(), cl, threads, energy_only, o.energy, o.energies.data(),
                         energy_only ? nullptr : o.forces.data(), energy_only ? nullptr : o.virial);
  return o;
}

inline double maxabs(const std::vector<double> &a, const std::vector<double> &b)
{
  double m = 0.0;
  for (size_t k = 0; k < a.size(); k++) m = std::max(m, std::fabs(a[k] - b[k]));
  return m;
}

}    // namespace d3_ref

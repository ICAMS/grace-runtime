// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// TF reference of the two-group coupling (tests/fixtures/ref_cpl/WMo54_<1l|2l|3l>.txt, from Pablo Pena Cano's
// TensorFlow pair_style grace with "couple" in LAMMPS):
// group mask, then per case (eq 0 = uncoupled, eq 1..4 at lambda 0.3 / 0.7) energy, dE/dlambda, virial, forces.
// Unlike the finite-difference checks, it pins the shift equations themselves (r', which bonds are dropped).
#pragma once
#include "test_util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace cplref {
struct Case {
  int eq = 0;
  double lambda = 0, e = 0, dedl = 0, vir[6] = {0, 0, 0, 0, 0, 0};
  std::vector<double> f;
};
struct Ref {
  std::vector<int> mask;
  std::vector<Case> cases;
};
inline Ref load(const std::string &path, int n)
{
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "coupling reference %s not found\n", path.c_str());
    test_exit(1);
  }
  Ref r;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    if (key == "mask") {
      r.mask.resize(n);
      for (int a = 0; a < n; a++) ss >> r.mask[a];
    } else if (key == "case") {
      Case c;
      ss >> c.eq >> c.lambda >> c.e >> c.dedl;
      for (double &v : c.vir) ss >> v;
      c.f.resize(3 * n);
      for (int a = 0; a < n; a++) {
        std::getline(in, line);
        std::istringstream fs(line);
        fs >> c.f[3 * a] >> c.f[3 * a + 1] >> c.f[3 * a + 2];
      }
      r.cases.push_back(c);
    }
  }
  CHECK((int) r.mask.size() == n && r.cases.size() == 9);
  return r;
}
// Tolerances above the TF vs libgrace (fp32) level, which is the same with and without coupling. Measured on this cell
// (E ~ -640 eV), CPU fused / fused kk on an L40S: |dE| <= 2.6e-5 / 4.8e-5 eV, max|dF| <= 2.5e-6 / 3.7e-6 eV/A,
// max|dvir| <= 1.4e-4 / 2.4e-4 eV, dE/dlambda <= 3e-7 relative. A wrong shift equation moves E by eV.
struct Tol {
  double e = 2e-4, f = 2e-5, vir = 1e-3, dedl_rel = 1e-5, dedl_abs = 1e-4;
};
inline void compare(const char *tag, const Case &r, double e, double dedl, const double *vir,
                    const std::vector<double> &f, const Tol &t = {})
{
  double df = 0, dv = 0;
  for (size_t q = 0; q < f.size(); q++) df = std::max(df, std::fabs(f[q] - r.f[q]));
  for (int k = 0; k < 6; k++) dv = std::max(dv, std::fabs(vir[k] - r.vir[k]));
  std::printf("%-22s vs TF eq %d lambda %.1f: dE %.1e  max|dF| %.1e  max|dvir| %.1e  dE/dlambda %.6f vs %.6f\n", tag,
              r.eq, r.lambda, e - r.e, df, dv, dedl, r.dedl);
  CHECK(std::fabs(e - r.e) <= t.e);
  CHECK(df <= t.f);
  CHECK(dv <= t.vir);
  CHECK(std::fabs(dedl - r.dedl) <= t.dedl_rel * std::fabs(r.dedl) + t.dedl_abs);
}
}    // namespace cplref

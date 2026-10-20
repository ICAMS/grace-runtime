// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// the kernels header compiles on its own (no LAMMPS headers) and its basic helpers work
#include "grace/fused_cpu.h"
#include "test_util.h"
int main()
{
  CHECK(grace::fused::pad16(17) == 32);
  grace::fused::fvec x(32, 1.f), y(32, 0.f);
  grace::fused::axpy(y.data(), x.data(), 2.f, 32);
  CHECK_NEAR(y[31], 2.0, 0);
  CHECK_NEAR(grace::fused::dot(x.data(), y.data(), 32), 64.0, 0);
  std::vector<double> a, b, c, d;
  grace::fused::sh_coefs(2, a, b, c, d);
  CHECK(!a.empty());
  return 0;
}

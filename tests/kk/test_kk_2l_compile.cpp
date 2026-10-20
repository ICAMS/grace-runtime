// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The GRACE-2L Kokkos engines build without LAMMPS and set a model up on the default device.
#include "grace/kk/grace_2l_kk_impl.h"
#include "grace/kk/grace_2l_fused_kk_impl.h"
#include "grace/kk/traits.h"
#include "../test_util.h"

using Dev = Kokkos::DefaultExecutionSpace;    // as LAMMPS LMPDeviceType: an execution space
template class grace::kk::Grace2LKK<Dev, float, float, grace::kk::DefaultTraits<Dev>>;
template class grace::kk::Grace2LKK<Dev, double, double, grace::kk::DefaultTraits<Dev>>;
template class grace::kk::Grace2LFusedKK<Dev, grace::kk::DefaultTraits<Dev>>;

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  auto m = std::make_shared<grace::GRACE2LModel>();
  m->load(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"));
  grace::kk::Grace2LKK<Dev, float, float, grace::kk::DefaultTraits<Dev>> e;
  std::vector<int> type2mu = {-1, 0};
  e.set_model(m, type2mu);
  e.setup();
  CHECK(e.cutoff(0, 0) > 4.0);
  CHECK(e.comm_width() > 0);
  grace::kk::Grace2LFusedKK<Dev, grace::kk::DefaultTraits<Dev>> ef;
  ef.set_model(m, type2mu);
  ef.setup();
  std::printf("kk 2l setup ok\n");
  return 0;
}

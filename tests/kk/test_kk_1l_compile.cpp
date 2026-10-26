// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The GRACE-1L Kokkos engine builds without LAMMPS and sets a model up on the default device.
#include "grace/kk/grace_1l_kk_impl.h"
#include "grace/kk/grace_1l_fused_kk_impl.h"
#include "grace/kk/traits.h"
#include "../test_util.h"

using Dev = Kokkos::DefaultExecutionSpace;    // as LAMMPS LMPDeviceType: an execution space
template class grace::kk::Grace1LKK<Dev, float, float, grace::kk::DefaultTraits<Dev>>;
template class grace::kk::Grace1LKK<Dev, double, double, grace::kk::DefaultTraits<Dev>>;
template class grace::kk::Grace1LFusedKK<Dev, grace::kk::DefaultTraits<Dev>>;

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  auto m = std::make_shared<grace::GRACE1LModel>();
  m->load(test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"));
  grace::kk::Grace1LKK<Dev, float, float, grace::kk::DefaultTraits<Dev>> e;
  std::vector<int> type2mu = {-1, 0};
  e.set_model(m, type2mu);
  e.setup();
  CHECK(e.cutoff(0, 0) > 4.0);
  grace::kk::Grace1LFusedKK<Dev, grace::kk::DefaultTraits<Dev>> ef;
  ef.set_model(m, type2mu);
  ef.setup();
  // the matrix-core basis A covers lmax 4 and ceil(NR / 16) * 5 <= 16 output blocks: radial widths up to 48; the
  // engines use the vector-ALU kernel for the others
  namespace G = grace::kk::GraceFusedKK;
  CHECK(G::a_fwd_mfma_fits<Dev>(4, 32) && G::a_fwd_mfma_fits<Dev>(4, 42) && G::a_fwd_mfma_fits<Dev>(4, 48));
  CHECK(!G::a_fwd_mfma_fits<Dev>(4, 49) && !G::a_fwd_mfma_fits<Dev>(4, 64) && !G::a_fwd_mfma_fits<Dev>(3, 32));
  std::printf("kk 1l setup ok\n");
  return 0;
}

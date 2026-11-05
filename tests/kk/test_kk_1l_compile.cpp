// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The GRACE-1L Kokkos engine builds without LAMMPS and sets a model up on the default device.
#include "grace/kk/grace_1l_kk_impl.h"
#include "grace/kk/grace_1l_fused_kk_impl.h"
#include "grace/kk/traits.h"
#include "../test_util.h"

using Dev = Kokkos::DefaultExecutionSpace;    // as LAMMPS LMPDeviceType: an execution space
GRACE_KK_EXTERN_1L(Dev, grace::kk::DefaultTraits<Dev>);         // as LAMMPS does: declared in its header,
GRACE_KK_INSTANTIATE_1L(Dev, grace::kk::DefaultTraits<Dev>);    // defined in one translation unit
template class grace::kk::Grace1LKK<Dev, double, double, grace::kk::DefaultTraits<Dev>>;

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
  struct Test1LFused : grace::kk::Grace1LFusedKK<Dev, grace::kk::DefaultTraits<Dev>> {
    using Grace1LFusedKK::rm;
    using Grace1LFusedKK::cgt_AA;
    using Grace1LFusedKK::cgt_AAA;
    using Grace1LFusedKK::cgt_AAAA;
  };
  Test1LFused ef;
  ef.set_model(m, type2mu);
  ef.setup();
  CHECK(ef.rm.nconn > 0);
  CHECK(ef.cgt_AA >= 0 && ef.cgt_AAA >= 0 && ef.cgt_AAAA >= 0);    // compiled CG tables (cg_tables.h), each product
  {   // lmax 3, max_order 3 (as GRACE-OFF-1L-small): AA and AAA compiled, the loader's dead AAAA stage generic
    auto m3 = std::make_shared<grace::GRACE1LModel>();
    m3->load(std::string(env_or("GRACE_TEST_FIXTURES", "tests/fixtures")) + "/grace_1l_l3.npz");
    Test1LFused e3;
    e3.set_model(m3, type2mu);
    e3.setup();
    CHECK(e3.cgt_AA >= 0 && e3.cgt_AAA >= 0 && e3.cgt_AAAA < 0);
    Test1LFused eg;    // 'noshape': all generic
    eg.no_shape = 1;
    eg.set_model(m3, type2mu);
    eg.setup();
    CHECK(eg.cgt_AA < 0 && eg.cgt_AAA < 0 && eg.cgt_AAAA < 0);
  }
  // the matrix-core basis A covers lmax 3 and 4 and ceil(NR / 16) (lmax + 1) <= 16 output blocks: radial widths up to
  // 48 (lmax 4) / 64 (lmax 3); the engines use the vector-ALU kernel for the others. The wave-per-atom backward: lmax 3
  // and 4.
  namespace G = grace::kk::GraceFusedKK;
  CHECK(G::a_fwd_mfma_fits<Dev>(4, 32) && G::a_fwd_mfma_fits<Dev>(4, 42) && G::a_fwd_mfma_fits<Dev>(4, 48));
  CHECK(G::a_fwd_mfma_fits<Dev>(3, 20) && G::a_fwd_mfma_fits<Dev>(3, 64));
  CHECK(!G::a_fwd_mfma_fits<Dev>(4, 49) && !G::a_fwd_mfma_fits<Dev>(4, 64) && !G::a_fwd_mfma_fits<Dev>(2, 32));
  CHECK(G::bond_bwd_wave_fits<Dev>(3, 20) && G::bond_bwd_wave_fits<Dev>(4, 64) && !G::bond_bwd_wave_fits<Dev>(2, 32));
  CHECK(grace::kk::wave64_switch("GRACE_L2_BWD_SHFL", true).on == (bool) GRACE_KK_WAVE64);
  CHECK(!grace::kk::wave64_switch("GRACE_L2_BWD_SHFL", false).on);
  std::printf("kk 1l setup ok\n");
  return 0;
}

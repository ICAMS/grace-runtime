// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The GRACE-2L Kokkos engines build without LAMMPS and set a model up on the default device.
#include "grace/kk/grace_2l_kk_impl.h"
#include "grace/kk/grace_2l_fused_kk_impl.h"
#include "grace/kk/traits.h"
#include "../test_util.h"

using Dev = Kokkos::DefaultExecutionSpace;    // as LAMMPS LMPDeviceType: an execution space
GRACE_KK_EXTERN_2L(Dev, grace::kk::DefaultTraits<Dev>);         // as LAMMPS does: declared in its header,
GRACE_KK_INSTANTIATE_2L(Dev, grace::kk::DefaultTraits<Dev>);    // defined in one translation unit
template class grace::kk::Grace2LKK<Dev, double, double, grace::kk::DefaultTraits<Dev>>;

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  auto m = std::make_shared<grace::GRACE2LModel>();
  m->load(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"));
  struct Test2L : grace::kk::Grace2LKK<Dev, float, float, grace::kk::DefaultTraits<Dev>> {
    using Grace2LKK::grow_global;
    using Grace2LKK::copy_rows_to_host;
    using Grace2LKK::d_I_global;
    using Grace2LKK::h_I_global;
    using Grace2LKK::h_grad_I_global;
  };
  Test2L e;
  std::vector<int> type2mu = {-1, 0};
  e.set_model(m, type2mu);
  const int width = e.comm_width();    // valid before setup(): callers size their comm buffers with it
  CHECK(width > 0);
  e.setup();
  CHECK(e.cutoff(0, 0) > 4.0);
  CHECK(e.comm_width() == width);
  e.grow_global(16);
  CHECK(e.h_I_global.data() == nullptr && e.h_grad_I_global.data() == nullptr);
  Kokkos::deep_copy(e.d_I_global.base(), 0.0f);
  e.copy_rows_to_host(false);
  CHECK(e.h_I_global.data() != nullptr && (int) e.h_I_global.extent(0) == 16);
  CHECK(e.h_grad_I_global.data() != nullptr && (int) e.h_grad_I_global.extent(0) == 16);
  e.release_buffers();
  CHECK(e.h_I_global.data() == nullptr && e.h_grad_I_global.data() == nullptr);
  struct Test2LFused : grace::kk::Grace2LFusedKK<Dev, grace::kk::DefaultTraits<Dev>> {
    using Grace2LFusedKK::cgt_AA;
    using Grace2LFusedKK::cgt_AAA;
    using Grace2LFusedKK::cgt_AAAA;
    using Grace2LFusedKK::cgt_BB;
    using Grace2LFusedKK::cgt_BBB;
    using Grace2LFusedKK::cgt_BBBB;
  };
  Test2LFused ef;
  ef.set_model(m, type2mu);
  ef.setup();
  // compiled CG tables (cg_tables.h), product by product
  CHECK(ef.cgt_AA >= 0 && ef.cgt_AAA >= 0 && ef.cgt_AAAA >= 0 && ef.cgt_BB >= 0 && ef.cgt_BBB >= 0 && ef.cgt_BBBB >= 0);
  {   // lmax 3, max_order 3, layer 2 as GRACE-2L-SMAX-OMAT-medium: all but the loader's dead AAAA / BBBB stages
    auto m3 = std::make_shared<grace::GRACE2LModel>();
    m3->load(std::string(env_or("GRACE_TEST_FIXTURES", "tests/fixtures")) + "/grace_2l_l3.npz");
    Test2LFused e3;
    e3.set_model(m3, type2mu);
    e3.setup();
    CHECK(e3.cgt_AA >= 0 && e3.cgt_AAA >= 0 && e3.cgt_AAAA < 0 && e3.cgt_BB >= 0 && e3.cgt_BBB >= 0 && e3.cgt_BBBB < 0);
    Test2LFused eg;    // 'noshape': all generic
    eg.no_shape = 1;
    eg.set_model(m3, type2mu);
    eg.setup();
    CHECK(eg.cgt_AA < 0 && eg.cgt_AAA < 0 && eg.cgt_AAAA < 0 && eg.cgt_BB < 0 && eg.cgt_BBB < 0 && eg.cgt_BBBB < 0);
  }
  std::printf("kk 2l setup ok\n");
  return 0;
}

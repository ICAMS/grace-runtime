// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// GRACE-2L with the large message (16 functions per channel): the fused engine against the fp64 engine of the same
// model on W54, WMo54, W2tri, also in 3 chunks. Its layer-2 backward has the widest team scratch of the fused kernels
// (CUDA: 58 KB with Kokkos 5.2+, one geometry round within the 48 KB of Kokkos 5.1); the medium models of test_kk_csr
// do not reach it.
//   env GRACE_TEST_M2L: kokkos.npz of the model (default GRACE-2L-SMAX-OMAT-large of the test cache)
#include "grace/kk/csr_images.h"
#include "grace/kk/grace_2l_fused_kk_impl.h"
#include "grace/kk/grace_2l_kk_impl.h"
#include "grace/kk/traits.h"
#include "../test_util.h"

#include <cstdio>
#include <string>

using Dev = Kokkos::DefaultExecutionSpace;
using T = grace::kk::DefaultTraits<Dev>;
#include "kk_csr_util.h"
using E2 = grace::kk::Grace2LKK<Dev, double, double, T>;
using E2F = grace::kk::Grace2LFusedKK<Dev, T>;
template class grace::kk::Grace2LKK<Dev, double, double, T>;
template class grace::kk::Grace2LFusedKK<Dev, T>;

using namespace kkt;

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  auto m = std::make_shared<grace::GRACE2LModel>();
  m->load(test_model("GRACE_TEST_M2L", "GRACE-2L-SMAX-OMAT-large"));
  E2 e64;
  e64.set_model(m, type2mu_for(m->element_names));
  e64.setup();
  E2F ef;
  ef.set_model(m, type2mu_for(m->element_names));
  ef.setup();
  for (const char *cell : {"W54", "WMo54", "W2tri"}) {
    const Cell c = load_cell(fx + "/" + cell + ".npz");
    const Ref ref = to_ref(run(e64, c), c.volume);
    compare(std::string(cell) + " 2L-large fused", run(ef, c), ref, c, tol_fp32_vs_fp64);
    if (c.n >= 3) {
      E2F ec;
      ec.chunksize = (c.n + 2) / 3;
      ec.set_model(m, type2mu_for(m->element_names));
      ec.setup();
      compare(std::string(cell) + " 2L-large 3 chunks", run(ec, c), ref, c, tol_fp32_vs_fp64);
    }
  }
  std::printf("kk 2l large ok\n");
  return 0;
}

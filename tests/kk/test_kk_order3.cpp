// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// GRACE-1L/2L models with max_order 3 on the Kokkos engines (generic fp64 and fused): the loader adds a dead AAAA /
// BBBB stage (zero weights; one CG term, one output), the engines must run it and give the model's results: tensorpotential's
// TPCalculator on the SavedModels (fp64): energy, per-atom energies, forces.
//   env GRACE_TEST_FIXTURES (default tests/fixtures)
#include "grace/kk/csr_images.h"
#include "grace/kk/grace_1l_fused_kk_impl.h"
#include "grace/kk/grace_1l_kk_impl.h"
#include "grace/kk/grace_2l_fused_kk_impl.h"
#include "grace/kk/grace_2l_kk_impl.h"
#include "grace/kk/traits.h"
#include "grace_cnpy/cnpy.h"
#include "../test_util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using Dev = Kokkos::DefaultExecutionSpace;
using T = grace::kk::DefaultTraits<Dev>;
#include "kk_csr_util.h"
using E1 = grace::kk::Grace1LKK<Dev, double, double, T>;
using E1F = grace::kk::Grace1LFusedKK<Dev, T>;
using E2 = grace::kk::Grace2LKK<Dev, double, double, T>;
using E2F = grace::kk::Grace2LFusedKK<Dev, T>;
template class grace::kk::Grace1LKK<Dev, double, double, T>;
template class grace::kk::Grace1LFusedKK<Dev, T>;
template class grace::kk::Grace2LKK<Dev, double, double, T>;
template class grace::kk::Grace2LFusedKK<Dev, T>;

namespace {
using namespace kkt;

grace::cnpy::npz_t tf_ref;

// the fixture model on one engine, on the cells, against the TensorFlow reference
template<class Eng, class M>
void check(const char *tag, const std::string &name, const std::string &fx, const Tol &t)
{
  auto m = std::make_shared<M>();
  m->load(fx + "/" + name + ".npz");
  CHECK(m->max_order == 3);
  Eng e;
  e.set_model(m, type2mu_for(m->element_names));
  e.setup();
  for (const char *cell : {"W54", "WMo54", "W2tri"}) {
    const Cell c = load_cell(fx + "/" + cell + ".npz");
    const Result r = run(e, c);
    const std::string k = name + "/" + cell;
    const double e_ref = *tf_ref.at(k + "/energy").data<double>();
    const double *f_ref = tf_ref.at(k + "/forces").data<double>(), *ei_ref = tf_ref.at(k + "/energies").data<double>();
    double df = 0, dpe = 0, fmax = 0;
    for (int q = 0; q < 3 * c.n; q++) {
      df = std::max(df, std::fabs(r.f[q] - f_ref[q]));
      fmax = std::max(fmax, std::fabs(f_ref[q]));
    }
    for (int a = 0; a < c.n; a++) dpe = std::max(dpe, std::fabs(r.eatom[a] - ei_ref[a]));
    std::printf("%-22s %-14s %-6s dE/n %.3e  max|de_i| %.3e  max|dF| %.3e (max|F| %.3f)\n", tag, name.c_str(), cell,
                std::fabs(r.e - e_ref) / c.n, dpe, df, fmax);
    CHECK(std::fabs(r.e - e_ref) / c.n <= t.e);
    CHECK(dpe <= t.ei);
    CHECK(df <= t.f);
    CHECK(fmax > 1e-2);
  }
}
}    // namespace

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  tf_ref = grace::cnpy::npz_load(fx + "/ref_tf/order3_tf.npz");
  const Tol tol_generic = {1e-8, 1e-8, 1e-8, 0};    // fp64 engine against fp64 TensorFlow
  // the generic engines refuse the legacy layouts (the fused ones run them)
  for (const char *name : {"legacy_1l_order3", "grace_1l_order3"}) {
    if (name[0] == 'g') check<E1, grace::GRACE1LModel>("1L kk fp64", name, fx, tol_generic);
    check<E1F, grace::GRACE1LModel>("1L fused", name, fx, tol_fp32);
  }
  for (const char *name : {"legacy_2l_order3", "grace_2l_order3"}) {
    if (name[0] == 'g') check<E2, grace::GRACE2LModel>("2L kk fp64", name, fx, tol_generic);
    check<E2F, grace::GRACE2LModel>("2L fused", name, fx, tol_fp32_2l);
  }
  std::printf("kk order3 ok\n");
  return 0;
}

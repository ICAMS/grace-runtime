// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The Kokkos engines on CSR bonds (grace/kk/csr_images.h: ghost images, image exchange, batch sums) against the
// LAMMPS GPU references of the same styles (tests/fixtures/ref_gpu from ref_lammps_gpu.sh): energy, per-atom energy,
// forces, pressure within fp32 parity (GPU runs are not reproducible bit for bit); the per-atom virial sums to the
// global one; two cells in one batch give each cell's result.
//   env GRACE_TEST_FIXTURES (default tests/fixtures), GRACE_TEST_REF_DIR (default tests/fixtures/ref_gpu)
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
#include <fstream>
#include <optional>
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

// engine guards for callers without LAMMPS: run() before setup() and nall < nlocal are errors; an engine
// copy outlives the original (shared cuBLAS handle); a second model on one engine with several chunks gives the
// result of a fresh engine
void check_guards(const std::shared_ptr<grace::GRACE2LModel> &m2, const std::shared_ptr<grace::GRACE2LModel> &m2b,
                  const std::string &fx, const std::string &rd)
{
  const Cell c = load_cell(fx + "/WMo54.npz");
  const Ref ref = load_ref(rd + "/WMo54_2l_fused", c.n);
  {
    E2F e;
    e.set_model(m2, type2mu_for(m2->element_names));
    typename E2F::RunInput in;
    bool threw = false;
    try { e.run(in); } catch (const grace::Error &) { threw = true; }
    std::printf("guard run before setup: %s\n", threw ? "throws" : "NO ERROR");
    CHECK(threw);
  }
  {
    E2F e;
    e.set_model(m2, type2mu_for(m2->element_names));
    e.setup();
    grace::kk::CsrImagesKK<Dev, T> img;
    img.build(c.n, c.pos.data(), c.type.data(), c.cell.data(), c.offset.data(), c.j.data(), c.rij.data());
    typename T::t_kkacc_1d_3 f("f", img.nall);
    typename E2F::RunInput in;
    in.x = img.x; in.f = f; in.type = img.type; in.ilist = img.ilist; in.numneigh = img.numneigh;
    in.neighbors = img.neighbors; in.inum = c.n; in.nlocal = c.n; in.ntypes = 2; in.eflag_global = 1;
    in.nall = c.n - 1;
    bool threw = false;
    try { e.run(in); } catch (const grace::Error &) { threw = true; }
    std::printf("guard nall < nlocal: %s\n", threw ? "throws" : "NO ERROR");
    CHECK(threw);
  }
  {
    std::optional<E2F> orig;
    orig.emplace();
    orig->set_model(m2, type2mu_for(m2->element_names));
    orig->setup();
    E2F copy(*orig);
    orig.reset();    // the original goes first; the copy keeps a valid cuBLAS handle
    compare("WMo54 2L copy", run(copy, c), ref, c, tol_fp32);
  }
  {
    E2F a, b;
    a.chunksize = b.chunksize = 16;    // 4 chunks of WMo54: the layer-1 store per chunk
    a.set_model(m2b, type2mu_for(m2b->element_names));
    a.setup();
    run(a, c);
    a.set_model(m2, type2mu_for(m2->element_names));
    a.setup();
    b.set_model(m2, type2mu_for(m2->element_names));
    b.setup();
    const Result ra = run(a, c), rb = run(b, c);
    double df = 0;
    for (int q = 0; q < 3 * c.n; q++) df = std::max(df, std::fabs(ra.f[q] - rb.f[q]));
    std::printf("second model, 4 chunks vs fresh engine: dE/n %.3e  max|dF| %.3e\n", std::fabs(ra.e - rb.e) / c.n, df);
    CHECK(std::fabs(ra.e - rb.e) / c.n <= tol_fp32.e && df <= tol_fp32.f);
    compare("WMo54 2L 4 chunks", rb, ref, c, tol_fp32);
  }
}
}    // namespace

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  const std::string rd = env_or("GRACE_TEST_REF_DIR", "tests/fixtures/ref_gpu");
  auto m1 = std::make_shared<grace::GRACE1LModel>();
  m1->load(test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"));
  auto m2 = std::make_shared<grace::GRACE2LModel>();
  m2->load(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"));
  {
    E1 e; e.set_model(m1, type2mu_for(m1->element_names)); e.setup();
    check_engine(e, "1L kk fp64", "1l_kk", tol_fp64, fx, rd);
  }
  {
    E1F e; e.set_model(m1, type2mu_for(m1->element_names)); e.setup();
    check_engine(e, "1L fused", "1l_fused", tol_fp32, fx, rd);
    check_vs_fp64(e, "1L fused", "1l_kk", fx, rd);
    check_energy_sum(e, "1L fused", fx);
  }
  {
    E2 e; e.set_model(m2, type2mu_for(m2->element_names)); e.setup();
    check_engine(e, "2L kk fp64", "2l_kk", tol_fp64, fx, rd);
  }
  {
    E2F e; e.set_model(m2, type2mu_for(m2->element_names)); e.setup();
    check_engine(e, "2L fused", "2l_fused", tol_fp32, fx, rd);
    check_vs_fp64(e, "2L fused", "2l_kk", fx, rd);
    check_energy_sum(e, "2L fused", fx);
  }
  auto m2b = std::make_shared<grace::GRACE2LModel>();
  m2b->load(test_model("GRACE_TEST_UQ2", "GRACE-2L-OMAT-medium-ft-E"));
  check_guards(m2, m2b, fx, rd);
  std::printf("kk csr ok\n");
  return 0;
}

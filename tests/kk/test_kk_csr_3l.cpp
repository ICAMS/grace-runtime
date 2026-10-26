// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The GRACE-3L Kokkos engine (Grace3LKK <float,double>, as the former LAMMPS grace/3l/kk/mixed) on CSR bonds with ghost images
// and the layered owner <-> image exchange (ImageExchange3KK) against the LAMMPS GPU references
// (tests/fixtures/ref_gpu/<cell>_3l_kk from ref_lammps_gpu.sh): energy, per-atom energy, forces, pressure; batch ==
// alone; isolated atom; another species set on the same engine (set_types) == a fresh engine, and back with the
// saved type tables (use_type_tables) == the first run; a second model on the engine == a fresh engine.
//   env GRACE_TEST_FIXTURES (default tests/fixtures), GRACE_TEST_REF_DIR (default tests/fixtures/ref_gpu)
#include "grace/kk/grace_3l_kk_impl.h"
#include "grace/kk/traits.h"

using Dev = Kokkos::DefaultExecutionSpace;
using T = grace::kk::DefaultTraits<Dev>;
#include "kk_csr_util.h"

using E3 = grace::kk::Grace3LKK<Dev, float, double, T>;
template class grace::kk::Grace3LKK<Dev, float, double, T>;

namespace {
using namespace kkt;

double maxdf(const Result &a, const Result &b)
{
  double d = 0;
  for (size_t q = 0; q < a.f.size(); q++) d = std::max(d, std::fabs(a.f[q] - b.f[q]));
  return d;
}

// type tables: WMo54 with the type map W/Mo and with Mo/W (another species assignment of the same cell)
void check_types(const std::shared_ptr<grace::GRACE3LModel> &m, const std::string &fx)
{
  const Cell c = load_cell(fx + "/WMo54.npz");
  const std::vector<int> t2m = type2mu_for(m->element_names), swapped = {-1, t2m[2], t2m[1]};
  E3 e;
  e.set_model(m, t2m);
  e.setup();
  const Result r0 = run(e, c);
  const auto tt = e.type_tables();
  e.set_types(swapped);
  const Result rs = run(e, c);
  E3 f;
  f.set_model(m, swapped);
  f.setup();
  const Result rf = run(f, c);
  e.use_type_tables(tt);
  const Result rb = run(e, c);
  std::printf("3L set_types(Mo/W) vs fresh: dE/n %.3e max|dF| %.3e; use_type_tables back vs first: dE/n %.3e "
              "max|dF| %.3e (swapped - first: dE/n %.3e)\n", std::fabs(rs.e - rf.e) / c.n, maxdf(rs, rf),
              std::fabs(rb.e - r0.e) / c.n, maxdf(rb, r0), std::fabs(rs.e - r0.e) / c.n);
  CHECK(std::fabs(rs.e - rf.e) / c.n <= 1e-6 && maxdf(rs, rf) <= 1e-5);
  CHECK(std::fabs(rb.e - r0.e) / c.n <= 1e-6 && maxdf(rb, r0) <= 1e-5);
  CHECK(std::fabs(rs.e - r0.e) / c.n > 1e-4);    // the swap is a different structure (7.6e-4 eV/atom)
}

// without the global virial (its own force-scatter kernel): the same energy, forces and per-atom virials
void check_no_global_virial(const std::shared_ptr<grace::GRACE3LModel> &m, const std::string &fx)
{
  const Cell c = load_cell(fx + "/WMo54.npz");
  E3 e;
  e.set_model(m, type2mu_for(m->element_names));
  e.setup();
  const Result rv = run(e, c), rn = run(e, c, 1, 0, false);
  double dva = 0;
  for (int k = 0; k < 6; k++) dva = std::max(dva, std::fabs(rv.vir_atom[k] - rn.vir_atom[k]));
  std::printf("3L without the global virial vs with: dE/n %.3e max|dF| %.3e max|d sum vatom| %.3e\n",
              std::fabs(rv.e - rn.e) / c.n, maxdf(rv, rn), dva);
  CHECK(std::fabs(rv.e - rn.e) / c.n <= 1e-6 && maxdf(rv, rn) <= 1e-5 && dva <= 1e-4);
  CHECK(rn.vir[0] == 0.0 && rn.vir[5] == 0.0);
}

// a second model on one engine (chunks of 16) == a fresh engine
void check_second_model(const std::shared_ptr<grace::GRACE3LModel> &m, const std::shared_ptr<grace::GRACE3LModel> &mb,
                        const std::string &fx)
{
  const Cell c = load_cell(fx + "/WMo54.npz");
  E3 e;
  e.chunksize = 16;
  e.set_model(m, type2mu_for(m->element_names));
  e.setup();
  run(e, c);
  e.set_model(mb, type2mu_for(mb->element_names));
  e.setup();
  const Result ra = run(e, c);
  E3 f;
  f.chunksize = 16;
  f.set_model(mb, type2mu_for(mb->element_names));
  f.setup();
  const Result rb = run(f, c);
  std::printf("3L second model on the engine vs fresh: dE/n %.3e max|dF| %.3e\n", std::fabs(ra.e - rb.e) / c.n,
              maxdf(ra, rb));
  CHECK(std::fabs(ra.e - rb.e) / c.n <= 1e-6 && maxdf(ra, rb) <= 1e-5);
}
}    // namespace

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  const std::string rd = env_or("GRACE_TEST_REF_DIR", "tests/fixtures/ref_gpu");
  auto m3 = std::make_shared<grace::GRACE3LModel>();
  m3->load(test_model("GRACE_TEST_M3", "GRACE-3L-OMAT-large"));
  {
    E3 e;
    e.set_model(m3, type2mu_for(m3->element_names));
    e.setup();
    check_engine(e, "3L kk", "3l_kk", tol_fp32, fx, rd);
  }
  check_types(m3, fx);
  check_no_global_virial(m3, fx);
  auto m3b = std::make_shared<grace::GRACE3LModel>();
  m3b->load(test_model("GRACE_TEST_M3B", "GRACE-3L-OMAT-large-ft-AM"));
  check_second_model(m3, m3b, fx);
  check_empty_list<E3>(m3, "3L kk", fx);
  std::printf("kk csr 3l ok\n");
  return 0;
}

// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Large systems on the fused Kokkos engines (1L, 2L, 3L as the Python module and LAMMPS run them): W54 (bcc W) copied
// 2^k times, each copy its own periodic cell with its own images, so every copy must give the result of W54 alone.
// Per-atom energies and forces of all copies against one W54 run catch an index that wraps at large nall (local atoms
// + images): GRACE-3L exchanges 3200 floats per atom (nall x 3200 > 2^31 from nall 671089).
//   env GRACE_TEST_LARGE_K: copies 2^k (default 4; 11: 110592 atoms, nall 860160, about 40 GB with GRACE-3L)
//       GRACE_TEST_LARGE_MODELS: engines to run, default "123"
//       GRACE_TEST_LARGE_CHUNK: chunksize of the engines (default auto)
#include "grace/kk/csr_images.h"
#include "grace/kk/grace_1l_fused_kk_impl.h"
#include "grace/kk/grace_2l_fused_kk_impl.h"
#include "grace/kk/grace_3l_fused_kk.h"
#include "grace/kk/grace_3l_kk_impl.h"
#include "grace/kk/traits.h"
#include "../test_util.h"

#include <cstdlib>
#include <string>
#include <type_traits>

using Dev = Kokkos::DefaultExecutionSpace;
using T = grace::kk::DefaultTraits<Dev>;
#include "kk_csr_util.h"
using E1F = grace::kk::Grace1LFusedKK<Dev, T>;
using E2F = grace::kk::Grace2LFusedKK<Dev, T>;
using E3F = grace::kk::Grace3LFusedKK<Dev, T>;
template class grace::kk::Grace1LFusedKK<Dev, T>;
template class grace::kk::Grace2LFusedKK<Dev, T>;
template class grace::kk::Grace3LKK<Dev, float, float, T>;

namespace {
using namespace kkt;

// a View64 does not pass as a plain view (whose int indices would wrap again): only through .base()
using PlainV = Kokkos::View<float ***, Kokkos::LayoutRight, Dev>;
static_assert(!std::is_convertible_v<grace::kk::View64<PlainV>, const PlainV &>);
static_assert(std::is_convertible_v<PlainV, grace::kk::View64<PlainV>>);

// the 2^31-element guards without large allocations: a plain view above 2^31 elements is refused before it is
// allocated; chunksize auto keeps a work buffer of 10000 values per atom below 2^31 elements for 10^6 atoms
void check_guards()
{
  Kokkos::View<char **, Dev> v;
  bool threw = false;
  try {
    grace::kk::MemKK::realloc_kokkos(v, "test:big", 1 << 16, 1 << 16);
  } catch (const grace::Error &) {
    threw = true;
  }
  CHECK(threw && v.data() == nullptr);
  Kokkos::View<float **, Dev> w;
  const auto grow = [&](int c, int) { grace::kk::MemKK::realloc_kokkos(w, "test:work", c, 10000); };
  const int c = grace::kk::auto_chunk<Dev>(1000000, 1, 1, grow, [] { return 0.0; }, grace::Logger{}, "test");
  std::printf("guards: refused a 2^32-element view; chunksize auto %d for 10000 values per atom\n", c);
  CHECK((double) c * 10000 <= (double) INT32_MAX);
}

template<class Eng> void check_copies(Eng &eng, const char *tag, const Cell &w, const Cell &big)
{
  const int chunk = std::atoi(env_or("GRACE_TEST_LARGE_CHUNK", "0"));
  if (chunk > 0) eng.chunksize = chunk;
  const Result r0 = run(eng, w);
  run(eng, big);    // warm-up: buffers, chunksize auto
  double sec = 0;
  const Result rb = run(eng, big, 1, 0, true, nullptr, &sec);
  const int copies = big.n / w.n;
  double de = 0, df = 0;
  for (int c = 0; c < copies; c++)
    for (int a = 0; a < w.n; a++) {
      de = std::max(de, std::fabs(rb.eatom[(size_t) c * w.n + a] - r0.eatom[a]));
      for (int k = 0; k < 3; k++) df = std::max(df, std::fabs(rb.f[3 * ((size_t) c * w.n + a) + k] - r0.f[3 * a + k]));
    }
  const double dE = std::fabs(rb.e / copies - r0.e) / w.n;
  std::printf("%s %d copies (%d atoms, %d images): dE/n %.3e max|de_i| %.3e max|dF| %.3e, run %.3f s\n", tag, copies,
              big.n, rb.nimages, dE, de, df, sec);
  CHECK(std::isfinite(rb.e) && dE <= tol_fp32.e && de <= tol_fp32.ei && df <= tol_fp32.f);
}
}    // namespace

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  const std::string fx = env_or("GRACE_TEST_FIXTURES", "tests/fixtures");
  const std::string which = env_or("GRACE_TEST_LARGE_MODELS", "123");
  const int k = std::atoi(env_or("GRACE_TEST_LARGE_K", "4"));
  check_guards();
  const Cell w = load_cell(fx + "/W54.npz");
  Cell big = w;
  for (int q = 0; q < k; q++) big = concat(big, big);
  big.sys.assign(big.n, 0);    // one system: all copies have the same cell, and no bond joins two copies
  if (which.find('1') != std::string::npos) {
    auto m = std::make_shared<grace::GRACE1LModel>();
    m->load(test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"));
    E1F e;
    e.set_model(m, type2mu_for(m->element_names));
    e.setup();
    check_copies(e, "1L fused", w, big);
  }
  if (which.find('2') != std::string::npos) {
    auto m = std::make_shared<grace::GRACE2LModel>();
    m->load(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"));
    E2F e;
    e.set_model(m, type2mu_for(m->element_names));
    e.setup();
    check_copies(e, "2L fused", w, big);
  }
  if (which.find('3') != std::string::npos) {
    auto m = std::make_shared<grace::GRACE3LModel>();
    m->load(test_model("GRACE_TEST_M3", "GRACE-3L-OMAT-large"));
    E3F e;
    e.set_model(m, type2mu_for(m->element_names));
    e.setup();
    check_copies(e, "3L fused", w, big);
  }
  return 0;
}

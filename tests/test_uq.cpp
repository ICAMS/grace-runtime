// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// UQ (uqv6 gamma_only): basis layout of the deployed v6 models and the loader cross-checks. Exit 77 = skipped (the
// v6 models are not in the repository).
#include "grace/model_1l.h"
#include "grace/model_2l.h"
#include "grace/uq.h"
#include "test_util.h"

#include <fstream>
#include <vector>

static bool exists(const char *p) { return std::ifstream(p).good(); }

static std::vector<int64_t> widths(const grace::UqBlock &b)
{
  std::vector<int64_t> w;
  for (auto &s : b.src) w.push_back((int64_t) s.nin * (int64_t) s.ci.size());
  return w;
}

template<class M> static bool validate_throws(const M &m)
{
  try { m.validate_uq(); } catch (const grace::Error &) { return true; }
  return false;
}

int main()
{
  const char *p1 = test_model("GRACE_TEST_UQ1", "GRACE-1L-OMAT-large-base");
  const char *p2 = test_model("GRACE_TEST_UQ2", "GRACE-2L-OMAT-medium-ft-E");
  if (!exists(p1) || !exists(p2)) { std::printf("v6 models absent: skipped\n"); return 77; }

  grace::GRACE1LModel m1;
  m1.load(p1);
  m1.validate_uq();
  const auto L1 = m1.uq_layout();
  CHECK(L1.size() == 1 && L1[0].name == "rho");
  CHECK((widths(L1[0]) == std::vector<int64_t>{32, 320, 896, 4416}));
  CHECK(L1[0].off == 0 && L1[0].width == 5664 && m1.uq_d_basis == 5664);
  CHECK(L1[0].src[0].off == 0 && L1[0].src[1].off == 32 && L1[0].src[3].off == 32 + 320 + 896);

  grace::GRACE2LModel m2;
  m2.load(p2);
  m2.validate_uq();
  const auto L2 = m2.uq_layout();
  CHECK(L2.size() == 2 && L2[0].name == "I2" && L2[1].name == "rho");     // reduces sorted by name
  CHECK((widths(L2[0]) == std::vector<int64_t>{64, 256, 704, 2432}));
  CHECK((widths(L2[1]) == std::vector<int64_t>{42, 210, 588, 2898}));
  CHECK(L2[0].off == 0 && L2[0].width == 3456 && L2[1].off == 3456 && L2[1].width == 3738);
  CHECK(L2[1].src[0].off == 3456 && m2.uq_d_basis == 7194);

  // cross-checks of validate_uq: block count vs density channels, basis width vs R, element count
  { auto b = m1; b.uq_n_density = 3; CHECK(validate_throws(b)); }
  { auto b = m2; b.uq_d_basis = 7193; CHECK(validate_throws(b)); }
  { auto b = m2; b.uq_n_elements = m2.n_elements - 1; CHECK(validate_throws(b)); }
  { auto b = m2; b.I2_instruction_names = {"B", "BBB", "BB", "BBBB"}; CHECK(validate_throws(b)); }
  { auto b = m1; b.rho_instruction_names = {"AA", "A", "AAA", "AAAA"}; CHECK(validate_throws(b)); }
  // a model loaded into an object that held another one (pair_coeff twice in LAMMPS) keeps nothing of the first
  if (const char *nouq = std::getenv("GRACE_TEST_NOUQ")) {
    grace::GRACE2LModel r;
    r.load(p2);
    r.load(nouq);
    CHECK(!r.has_uq && r.uq_rp_matrix.empty() && r.uq_n_clusters.empty());
  }
  std::printf("uq layout ok\n");
  return 0;
}

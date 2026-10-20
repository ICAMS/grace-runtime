// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// loaders: the foundation 1L / 2L / 3L kokkos.npz load; an old UQ schema loads (no UQ) and fails only validate_uq()
#include "grace/model_1l.h"
#include "grace/model_2l.h"
#include "grace/model_3l.h"
#include "test_util.h"
#include <cstdio>
int main()
{
  grace::GRACE2LModel m2;
  m2.load(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"));
  std::printf("m2 n_elements %d embedding_size %d lmax %d has_uq %d\n", m2.n_elements, m2.embedding_size, m2.lmax, (int) m2.has_uq);
  CHECK(m2.n_elements > 80);
  CHECK(m2.lmax > 0);
  grace::GRACE1LModel m1;
  m1.load(test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"));
  std::printf("m1 n_elements %d lmax %d has_uq %d\n", m1.n_elements, m1.lmax, (int) m1.has_uq);
  CHECK(m1.n_elements > 80);
  if (const char *old = std::getenv("GRACE_TEST_OLD_UQ")) {     // test_uq_schema_lazy
    grace::GRACE2LModel mo;
    mo.load(old);                                                // must not throw
    CHECK(mo.has_uq);
    bool threw = false;
    try { mo.validate_uq(); } catch (const grace::Error &) { threw = true; }
    CHECK(threw);
  }
  if (const char *v6 = std::getenv("GRACE_TEST_UQ6")) {          // a schema-6 model validates
    grace::GRACE2LModel mu;
    mu.load(v6);
    mu.validate_uq();
    CHECK(mu.has_uq);
  }
  bool threw = false;
  try { grace::GRACE2LModel bad; bad.load("/nonexistent.npz"); } catch (const std::exception &) { threw = true; }
  CHECK(threw);
  grace::GRACE3LModel m3;    // GRACE-3L: three SPBF layers, 9 products, the UQ head loaded and checked by load()
  m3.load(test_model("GRACE_TEST_M3", "GRACE-3L-OMAT-large"));
  std::printf("m3 n_elements %d lmax %d spbf %zu prod %zu reduce %zu has_uq %d rp_dim %d d_basis %d\n", m3.n_elements,
              m3.lmax, m3.spbf.size(), m3.prod.size(), m3.reduce.size(), (int) m3.has_uq, m3.uq_rp_dim, m3.uq_d_basis);
  CHECK(m3.n_elements == 89 && m3.element_names.size() == 89);
  CHECK(m3.spbf.size() == 3 && m3.prod.size() == 9 && m3.fc.size() == 6);
  CHECK(m3.has_uq && m3.uq_rp_dim > 0 && m3.uq_d_basis > 0);
  threw = false;
  try { grace::GRACE3LModel bad; bad.load(test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium")); }
  catch (const grace::Error &) { threw = true; }
  CHECK(threw);    // a 2L kokkos.npz is not a GRACE-3L model: a grace::Error, not a crash
  return 0;
}

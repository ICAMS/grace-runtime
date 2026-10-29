// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// loaders: the foundation 1L / 2L / 3L kokkos.npz load; an old UQ schema loads (no UQ) and fails only validate_uq()
#include "grace/model_1l.h"
#include "grace/model_2l.h"
#include "grace/model_3l.h"
#include "grace_cnpy/cnpy.h"
#include "test_util.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>
int main()
{
  // cnpy hardening: (a) a zip member name shorter than 4 bytes (no ".npy" suffix) does not underflow string::erase;
  // (b) a .npy header whose shape product overflows size_t is rejected before allocating a wrapped buffer size
  // (model-independent: before the model loads, which skip the test without the cache; files in the temp directory)
  bool threw = false;    // also for the model checks further down
  const std::string tmp = (std::filesystem::temp_directory_path() / ("grace_test_cnpy_" + std::to_string(getpid())))
                              .string();
  {
    const std::string p = tmp + "_short_name.npz";
    const std::vector<char> npy = grace::cnpy::create_npy_header<double>({1});
    const double val = 3.5;
    const uint32_t sz = (uint32_t) (npy.size() + sizeof(val));
    std::vector<char> zip;
    auto u16 = [&](uint16_t v) { zip.push_back((char) (v & 0xff)); zip.push_back((char) (v >> 8)); };
    auto u32 = [&](uint32_t v) { u16((uint16_t) (v & 0xffff)); u16((uint16_t) (v >> 16)); };
    zip.insert(zip.end(), {'P', 'K', 3, 4});
    u16(20); u16(0); u16(0); u16(0); u16(0); u32(0); u32(sz); u32(sz); u16(1); u16(0);
    zip.push_back('a');
    zip.insert(zip.end(), npy.begin(), npy.end());
    zip.insert(zip.end(), reinterpret_cast<const char *>(&val), reinterpret_cast<const char *>(&val) + sizeof(val));
    zip.insert(zip.end(), {'P', 'K', 1, 2});
    zip.resize(zip.size() + 26, 0);
    std::ofstream(p, std::ios::binary).write(zip.data(), (std::streamsize) zip.size());
    auto all = grace::cnpy::npz_load(p);
    CHECK(all.count("a") == 1 && *all["a"].data<double>() == 3.5);
    auto one = grace::cnpy::npz_load(p, "a");
    CHECK(*one.data<double>() == 3.5);
    std::remove(p.c_str());
  }
  {
    const std::string p = tmp + "_overflow.npy";
    const std::vector<char> buf = grace::cnpy::create_npy_header<double>({(size_t) 1 << 32, (size_t) 1 << 32});
    std::ofstream(p, std::ios::binary).write(buf.data(), (std::streamsize) buf.size());
    try { grace::cnpy::npy_load(p); } catch (const std::runtime_error &) { threw = true; }
    std::remove(p.c_str());
    CHECK(threw);
  }

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
  threw = false;
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

  // grace_model_kind / grace_format_version (exports since format 1): a 1L export with them loads; a newer format or
  // another kind is refused with a message naming it
  {
    auto npz = grace::cnpy::npz_load(test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"));
    const auto set = [&](const char *kind, int64_t version) {
      grace::cnpy::NpyArray k({1}, 8, false), v({1}, 8, false);
      k.descr = "|S8";
      std::copy(kind, kind + 8, k.data<char>());
      v.descr = "<i8";
      *v.data<int64_t>() = version;
      npz["grace_model_kind"] = k;
      npz["grace_format_version"] = v;
    };
    const auto refused = [&](const char *what) {
      try {
        grace::GRACE1LModel m;
        m.load_arrays(npz);
      } catch (const grace::Error &e) {
        return std::string(e.what()).find(what) != std::string::npos;
      }
      return false;
    };
    set("GRACE-1L", 1);
    grace::GRACE1LModel ok;
    ok.load_arrays(npz);
    CHECK(ok.n_elements == m1.n_elements);
    set("GRACE-1L", 2);
    CHECK(refused("update grace-runtime"));
    set("GRACE-2L", 1);
    CHECK(refused("holds a GRACE-2L model"));
  }

  return 0;
}

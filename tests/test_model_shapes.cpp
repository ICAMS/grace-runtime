// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The model loaders against corrupted copies of the foundation kokkos.npz (1L, 2L, 3L): every array a loader reads,
// truncated (one row or value fewer), an integer one with a value of -1 or 2^20, must be a grace::Error naming the
// array (UQ arrays: at the load or at validate_uq(), which reports a broken UQ head) - never a crash, an allocation
// from a corrupt size, or a model that loads. A NaN goes into every 8th float array only: its rejection is one generic
// check of npz_get_double, the same for every array.
//   env GRACE_TEST_SHAPES_STRIDE: every k-th key only (default 1: all keys); GRACE_TEST_SHAPES_MODELS: default "123";
//       GRACE_TEST_M1 / M2 / M3: the models
#include "grace/model_1l.h"
#include "grace/model_2l.h"
#include "grace/model_3l.h"
#include "npz3.h"
#include "test_util.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using grace::cnpy::NpyArray;
using grace::cnpy::npz_t;

namespace {

enum Mode { TRUNCATE, MINUS_ONE, HUGE_INT, NAN_FLOAT };
const char *mode_name[] = {"truncated", "-1", "2^20", "NaN"};

// a copy of a with its own data, the modification applied; false when the mode does not apply to a
bool modified(const NpyArray &a, Mode m, NpyArray &out)
{
  const char k = a.kind();
  const bool is_int = (k == 'i' || k == 'u') && a.num_vals > 0, is_float = k == 'f' && a.num_vals > 0;
  if ((m == MINUS_ONE || m == HUGE_INT) && !is_int) return false;
  if (m == NAN_FLOAT && !is_float) return false;
  if (m == TRUNCATE && a.num_vals == 0) return false;
  std::vector<size_t> shape = a.shape;
  if (m == TRUNCATE) {
    if (shape.empty()) shape = {0};             // a scalar becomes an empty array
    else if (shape[0] > 1) shape[0]--;          // one row fewer
    else shape.back()--;                        // [1, ..., n]: one value fewer in the last dimension
  }
  out = NpyArray(shape, a.word_size, a.fortran_order);
  out.descr = a.descr;
  std::memcpy(out.data<char>(), a.data<char>(), std::min(a.num_bytes(), out.num_bytes()));
  if (m == MINUS_ONE || m == HUGE_INT) {
    const long long v = m == MINUS_ONE ? -1 : (1LL << 20);
    switch (a.word_size) {
    case 1: out.data<int8_t>()[0] = (int8_t) (m == MINUS_ONE ? -1 : 127); break;
    case 2: out.data<int16_t>()[0] = (int16_t) (m == MINUS_ONE ? -1 : 32767); break;
    case 4: out.data<int32_t>()[0] = (int32_t) v; break;
    default: out.data<int64_t>()[0] = (int64_t) v;
    }
  } else if (m == NAN_FLOAT) {
    if (a.word_size == 4) out.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
    else out.data<double>()[0] = std::numeric_limits<double>::quiet_NaN();
  }
  return true;
}

template<class M> void validate_uq(M &m) { m.validate_uq(); }
template<> void validate_uq(grace::GRACE3LModel &) {}    // the 3L loader checks its UQ head itself

// the load of npz is refused (UQ arrays: also at validate_uq)
template<class M> bool refused(const npz_t &npz, const std::string &key, std::string &why)
{
  M m;
  try {
    m.load_arrays(npz);
  } catch (const grace::Error &e) {
    why = e.what();
    return true;
  } catch (const std::exception &e) {    // not a validation error (e.g. std::bad_alloc from a corrupt size)
    why = std::string("not a grace::Error: ") + e.what();
    return false;
  }
  if (key.rfind("uq_", 0) == 0) {
    try {
      validate_uq(m);
    } catch (const grace::Error &e) {
      why = e.what();
      return true;
    }
  }
  why = "loaded";
  return false;
}

template<class M> int sweep(const char *tag, const char *path)
{
  npz_t npz = grace::npz::npz_open(path);
  std::set<std::string> keys;
  grace::npz::read_keys = &keys;
  {
    M m;
    m.load_arrays(npz);    // the untouched model loads
  }
  grace::npz::read_keys = nullptr;
  const int stride = std::max(1, std::atoi(env_or("GRACE_TEST_SHAPES_STRIDE", "1")));
  int checked = 0, failed = 0, q = 0, nf = 0;
  for (const std::string &key : keys) {
    auto it = npz.find(key);
    if (it == npz.end() || q++ % stride) continue;
    const bool nan = it->second.kind() == 'f' && nf++ % 8 == 0;
    for (Mode mode : {TRUNCATE, MINUS_ONE, HUGE_INT, NAN_FLOAT}) {
      NpyArray mod;
      if ((mode == NAN_FLOAT && !nan) || !modified(it->second, mode, mod)) continue;
      std::swap(it->second, mod);    // the corrupted array in, the original back after the load
      std::string why;
      checked++;
      const bool ok = refused<M>(npz, key, why);
      std::swap(it->second, mod);
      if (!ok) {
        failed++;
        std::printf("%s: '%s' %s (%s): %s\n", tag, key.c_str(), mode_name[mode], it->second.descr.c_str(),
                    why.c_str());
      }
    }
  }
  std::printf("%s: %zu arrays read, %d corrupted copies, %d not refused\n", tag, keys.size(), checked, failed);
  return failed;
}

}    // namespace

int main()
{
  const std::string which = env_or("GRACE_TEST_SHAPES_MODELS", "123");
  int failed = 0;
  if (which.find('1') != std::string::npos)
    failed += sweep<grace::GRACE1LModel>("1L", test_model("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"));
  if (which.find('2') != std::string::npos)
    failed += sweep<grace::GRACE2LModel>("2L", test_model("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"));
  if (which.find('3') != std::string::npos)
    failed += sweep<grace::GRACE3LModel>("3L", test_model("GRACE_TEST_M3", "GRACE-3L-OMAT-large"));
  CHECK(failed == 0);
  return 0;
}

// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from lammps-ace src/ML-PACE/grace_npz.h (feature/grace-runtime a73b8b17ee): npz access of the GRACE-1L / 2L / 3L
// loaders and the UQ head (private to libgrace: cnpy is not part of the public headers). Every array is read by its
// stored dtype: a value is converted or rejected, never reinterpreted.
#ifndef GRACE_SRC_NPZ3_H
#define GRACE_SRC_NPZ3_H
#include "grace/error.h"
#include "grace_cnpy/cnpy.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace grace {
namespace npz {

// all arrays of a .npz file; a file cnpy cannot read (missing, truncated, compressed, big-endian, ...) is a
// grace::Error naming the file
inline cnpy::npz_t npz_open(const std::string &path)
{
  try {
    return cnpy::npz_load(path);
  } catch (const Error &) {
    throw;
  } catch (const std::exception &e) {
    throw Error("cannot read the model file " + path + ": " + e.what());
  }
}

namespace detail {
// the array stored under key, nullptr when absent. The loaders read arrays flat in C (row-major) order, so an array
// with more than one non-trivial dimension in Fortran order is rejected rather than read transposed.
inline const cnpy::NpyArray *find_array(const cnpy::npz_t &npz, const std::string &key)
{
  auto it = npz.find(key);
  if (it == npz.end()) return nullptr;
  const auto &arr = it->second;
  if (arr.fortran_order) {
    int dims = 0;
    for (size_t d : arr.shape) dims += d > 1;
    if (dims > 1) throw Error("model array '" + key + "' is stored in Fortran order; re-export it in C order");
  }
  return &arr;
}

inline Error dtype_error(const std::string &key, const cnpy::NpyArray &arr, const char *expected)
{
  return Error("model array '" + key + "' has dtype '" + arr.descr + "', expected " + expected + "; re-export the model");
}

inline bool is_integer(const cnpy::NpyArray &arr)
{
  const char k = arr.kind();
  return (k == 'i' || k == 'u') && (arr.word_size == 1 || arr.word_size == 2 || arr.word_size == 4 || arr.word_size == 8);
}

// element q of an integer array (is_integer) as int64; false when an unsigned value does not fit
inline bool integer_at(const cnpy::NpyArray &arr, size_t q, int64_t &v)
{
  const bool u = arr.kind() == 'u';
  switch (arr.word_size) {
  case 1: v = u ? (int64_t) arr.data<uint8_t>()[q] : (int64_t) arr.data<int8_t>()[q]; return true;
  case 2: v = u ? (int64_t) arr.data<uint16_t>()[q] : (int64_t) arr.data<int16_t>()[q]; return true;
  case 4: v = u ? (int64_t) arr.data<uint32_t>()[q] : (int64_t) arr.data<int32_t>()[q]; return true;
  default:
    if (!u) { v = arr.data<int64_t>()[q]; return true; }
    const uint64_t x = arr.data<uint64_t>()[q];
    v = (int64_t) x;
    return x <= (uint64_t) INT64_MAX;
  }
}
}    // namespace detail

// float32 (UQ, fp32 exports) and float64 as stored; integer arrays (e.g. an export that copied `cutoff: 6` or
// `output_scale: 1` as-is) converted by value, never reinterpreted. Any other dtype is an error. Empty if the key is
// absent.
inline std::vector<double> npz_get_double(const cnpy::npz_t &npz, const std::string &key)
{
  const cnpy::NpyArray *arr = detail::find_array(npz, key);
  if (!arr) return {};
  if (arr->kind() == 'f' && arr->word_size == 4) {
    const float *p = arr->data<float>();
    return std::vector<double>(p, p + arr->num_vals);
  }
  if (arr->kind() == 'f' && arr->word_size == 8) {
    const double *p = arr->data<double>();
    return std::vector<double>(p, p + arr->num_vals);
  }
  if (detail::is_integer(*arr)) {
    std::vector<double> v(arr->num_vals);
    for (size_t q = 0; q < v.size(); q++) {
      int64_t x;
      if (!detail::integer_at(*arr, q, x)) throw detail::dtype_error(key, *arr, "a float32 or float64 array");
      v[q] = (double) x;
    }
    return v;
  }
  throw detail::dtype_error(key, *arr, "a float32 or float64 array");
}

// integer (signed or unsigned, 1 to 8 bytes) or bool arrays, each value within the range of int. Floating-point and
// other dtypes are an error. Empty if the key is absent.
inline std::vector<int> npz_get_int(const cnpy::npz_t &npz, const std::string &key)
{
  const cnpy::NpyArray *arr = detail::find_array(npz, key);
  if (!arr) return {};
  std::vector<int> v(arr->num_vals);
  if (arr->kind() == 'b' && arr->word_size == 1) {
    for (size_t q = 0; q < v.size(); q++) v[q] = arr->data<uint8_t>()[q] != 0;
    return v;
  }
  if (!detail::is_integer(*arr)) throw detail::dtype_error(key, *arr, "an integer array");
  for (size_t q = 0; q < v.size(); q++) {
    int64_t x;
    if (!detail::integer_at(*arr, q, x) || x < INT_MIN || x > INT_MAX)
      throw Error("model array '" + key + "' has a value outside the range of int; re-export the model");
    v[q] = (int) x;
  }
  return v;
}

inline int npz_get_int_scalar(const cnpy::npz_t &npz, const std::string &key, int default_val = 0)
{
  auto v = npz_get_int(npz, key);
  return v.empty() ? default_val : v[0];
}

inline double npz_get_double_scalar(const cnpy::npz_t &npz, const std::string &key, double default_val = 0.0)
{
  auto v = npz_get_double(npz, key);
  return v.empty() ? default_val : v[0];
}

// finite and non-negative; integer arrays are read by value (npz_get_double)
inline std::vector<double> npz_get_cutoff(const cnpy::npz_t &npz, const std::string &key, const char *style)
{
  auto v = npz_get_double(npz, key);
  for (double x : v)
    if (!std::isfinite(x) || x < 0.0)
      throw Error(std::string(style) + ": invalid cutoff in '" + key + "'");
  return v;
}

inline double npz_get_cutoff_scalar(const cnpy::npz_t &npz, const std::string &key, double default_val,
                                    const char *style)
{
  auto v = npz_get_cutoff(npz, key, style);
  if (v.empty()) return default_val;
  if (!(v[0] > 0.0)) throw Error(std::string(style) + ": cutoff '" + key + "' is not positive");
  return v[0];
}

// require-form: throw if a key the pair style cannot run without is absent. Catches the silent-zero failure
// mode where an optional npz_get_* returns an empty vector that would become zero-filled weights (and hence
// zero forces). Read keys by EXACT name only, never by prefix: e.g. the reduce `rho1`'s `rho1_norm_map` shares
// a prefix with the RMS-norm `rho1_norm`'s `rho1_norm_scale`; they are distinct keys.
inline std::vector<double> npz_require_double(const cnpy::npz_t &npz, const std::string &key)
{
  auto v = npz_get_double(npz, key);
  if (v.empty())
    throw Error("required weight '" + key + "' missing from the .npz file (an empty load would "
                             "give zero forces); re-export the model");
  return v;
}

inline std::vector<int> npz_require_int(const cnpy::npz_t &npz, const std::string &key)
{
  auto v = npz_get_int(npz, key);
  if (v.empty())
    throw Error("required index array '" + key + "' missing from the .npz file; re-export the model");
  return v;
}

// dims of a stored array (empty if absent)
inline std::vector<size_t> npz_shape(const cnpy::npz_t &npz, const std::string &key)
{
  auto it = npz.find(key);
  if (it == npz.end()) return {};
  return it->second.shape;
}

// decode an S-dtype names array (element_names, *_names) to strings
inline std::vector<std::string> npz_names(const cnpy::npz_t &npz, const std::string &key)
{
  std::vector<std::string> out;
  const cnpy::NpyArray *p = detail::find_array(npz, key);
  if (!p) return out;
  const auto &arr = *p;
  if (arr.kind() != 'S' || arr.shape.size() > 1) throw detail::dtype_error(key, arr, "a 1-d byte-string array (|S)");
  const int n = arr.shape.empty() ? 0 : (int) arr.shape[0];
  const int wsize = arr.word_size;
  const char *raw = arr.data<char>();
  out.resize(n);
  for (int i = 0; i < n; i++) {
    std::string s(raw + (size_t) i * wsize, wsize);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    out[i] = s;
  }
  return out;
}

}    // namespace npz

struct UqHead;
// reads and validates the uq_* arrays when the model has them (uq_schema_version present) and sets has_uq;
// max_density: density channels the evaluating kernels write (0: any); who: message prefix
void load_uq_head(UqHead &h, const cnpy::npz_t &npz, int n_elements, int max_density, const std::string &who);

}    // namespace grace
#endif

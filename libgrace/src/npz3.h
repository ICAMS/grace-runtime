// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from lammps-ace src/ML-PACE/grace_npz.h (feature/grace-runtime a73b8b17ee): npz access of the GRACE-1L / 2L / 3L
// loaders and the UQ head (private to libgrace: cnpy is not part of the public headers). Every array is read by its
// stored dtype: a value is converted or rejected, never reinterpreted.
#ifndef GRACE_SRC_NPZ3_H
#define GRACE_SRC_NPZ3_H
#include "grace/error.h"
#include "grace_cnpy/cnpy.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <set>
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

// tests: when set, the keys the loaders look up (present or not) are added to it
inline thread_local std::set<std::string> *read_keys = nullptr;

// sanity bounds of what the loaders read from a file, checked before a value sizes or indexes anything: any size or
// count (the largest of the GRACE exports is ~1e4), an angular momentum ((kMaxL + 1)^2 = kMaxDim functions) and the
// envelope order p (the engines form p (p + 2) in int)
constexpr int kMaxDim = 1 << 16;
constexpr int kMaxL = 255;
constexpr int kMaxP = 46339;

// a model array that does not fit the model: an error naming it
inline Error bad_array(const std::string &key, const std::string &what)
{
  return Error("model array '" + key + "' " + what + "; re-export the model");
}

// a * b, saturated (never equal to the size of a stored array)
inline size_t mul_sat(size_t a, size_t b)
{
  size_t r;
  return __builtin_mul_overflow(a, b, &r) ? SIZE_MAX : r;
}

namespace detail {
// the array stored under key, nullptr when absent. The loaders read arrays flat in C (row-major) order, so an array
// with more than one non-trivial dimension in Fortran order is rejected rather than read transposed.
inline const cnpy::NpyArray *find_array(const cnpy::npz_t &npz, const std::string &key)
{
  if (read_keys) read_keys->insert(key);
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
  return bad_array(key, "has dtype '" + arr.descr + "', expected " + expected);
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
  std::vector<double> v(arr->num_vals);
  auto put = [&](size_t q, double x) {
    if (!std::isfinite(x)) throw bad_array(key, "has a non-finite value (NaN / Inf) at " + std::to_string(q));
    v[q] = x;
  };
  if (arr->kind() == 'f' && arr->word_size == 4) {
    for (size_t q = 0; q < v.size(); q++) put(q, arr->data<float>()[q]);
  } else if (arr->kind() == 'f' && arr->word_size == 8) {
    for (size_t q = 0; q < v.size(); q++) put(q, arr->data<double>()[q]);
  } else if (detail::is_integer(*arr)) {
    for (size_t q = 0; q < v.size(); q++) {
      int64_t x;
      if (!detail::integer_at(*arr, q, x)) throw detail::dtype_error(key, *arr, "a float32 or float64 array");
      v[q] = (double) x;
    }
  } else {
    throw detail::dtype_error(key, *arr, "a float32 or float64 array");
  }
  return v;
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
      throw bad_array(key, "has a value outside the range of int");
    v[q] = (int) x;
  }
  return v;
}

namespace detail {
inline void require_scalar(const cnpy::npz_t &npz, const std::string &key)
{
  const auto it = npz.find(key);
  if (it != npz.end() && it->second.num_vals != 1)
    throw bad_array(key, "has " + std::to_string(it->second.num_vals) + " values, expected one");
}

template<class T> std::string dims_str(const std::vector<T> &d)
{
  std::string o = "(";
  for (size_t i = 0; i < d.size(); i++) o += (i ? ", " : "") + std::to_string(d[i]);
  return o + ")";
}
}    // namespace detail

// one value; default_val when the key is absent (present with no or several values: an error)
inline int npz_get_int_scalar(const cnpy::npz_t &npz, const std::string &key, int default_val = 0)
{
  detail::require_scalar(npz, key);
  auto v = npz_get_int(npz, key);
  return v.empty() ? default_val : v[0];
}

inline double npz_get_double_scalar(const cnpy::npz_t &npz, const std::string &key, double default_val = 0.0)
{
  detail::require_scalar(npz, key);
  auto v = npz_get_double(npz, key);
  return v.empty() ? default_val : v[0];
}

// non-negative (and finite: npz_get_double); integer arrays are read by value
inline std::vector<double> npz_get_cutoff(const cnpy::npz_t &npz, const std::string &key, const char *style)
{
  auto v = npz_get_double(npz, key);
  for (double x : v)
    if (x < 0.0)
      throw Error(std::string(style) + ": invalid cutoff in '" + key + "'");
  return v;
}

inline double npz_get_cutoff_scalar(const cnpy::npz_t &npz, const std::string &key, double default_val,
                                    const char *style)
{
  detail::require_scalar(npz, key);
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

// an int scalar in [lo, hi]; def when absent (a def outside [lo, hi]: the key is required)
inline int npz_get_int_in(const cnpy::npz_t &npz, const std::string &key, int def, int lo, int hi)
{
  if (!npz.count(key) && (def < lo || def > hi)) throw bad_array(key, "is missing from the .npz file");
  const int v = npz_get_int_scalar(npz, key, def);
  if (v < lo || v > hi)
    throw bad_array(key, "is " + std::to_string(v) + ", expected a value in [" + std::to_string(lo) + ", " +
                             std::to_string(hi) + "]");
  return v;
}

// the same, required (npz_get_int_in with a default outside [lo, hi] also requires the key)
inline int npz_require_int_in(const cnpy::npz_t &npz, const std::string &key, int lo, int hi)
{
  if (!npz.count(key)) throw bad_array(key, "is missing from the .npz file");
  return npz_get_int_in(npz, key, 0, lo, hi);
}

// 0 or 1; def when absent
inline bool npz_get_flag(const cnpy::npz_t &npz, const std::string &key, int def)
{
  return npz_get_int_in(npz, key, def, 0, 1) != 0;
}

// shape checks of the loaders: an array that does not fit the model's declared sizes is an error naming it, before
// any execution buffer is built from it

// the stored shape of key: exactly rank dimensions, each in [1, kMaxDim] (an absent key has none)
inline std::vector<size_t> npz_shape_rank(const cnpy::npz_t &npz, const std::string &key, size_t rank)
{
  const std::vector<size_t> d = npz_shape(npz, key);
  bool ok = d.size() == rank;
  for (size_t x : d) ok = ok && x >= 1 && x <= (size_t) kMaxDim;
  if (!ok)
    throw bad_array(key, "has shape " + detail::dims_str(d) + ", expected " + std::to_string(rank) +
                             " dimensions in [1, " + std::to_string(kMaxDim) + "]");
  return d;
}

// the stored shape is exactly `shape`, a negative dimension matching none (an absent key passes: the require / get
// accessors decide about absence)
inline void require_shape(const cnpy::npz_t &npz, const std::string &key, const std::vector<int64_t> &shape)
{
  const cnpy::NpyArray *arr = detail::find_array(npz, key);
  if (!arr) return;
  bool ok = arr->shape.size() == shape.size();
  for (size_t i = 0; ok && i < shape.size(); i++) ok = shape[i] >= 0 && arr->shape[i] == (size_t) shape[i];
  if (!ok)
    throw bad_array(key, "has shape " + detail::dims_str(arr->shape) + ", expected " + detail::dims_str(shape));
}

template<class V> inline void require_size(const std::string &key, const V &v, size_t n)
{
  if (v.size() != n)
    throw bad_array(key, "has " + std::to_string(v.size()) + " values, expected " + std::to_string(n));
}

// got == want (why: where want comes from); noun: "has <got> <noun>" (a dimension of the array), else "is <got>"
inline void require_eq(const std::string &key, int64_t got, int64_t want, const std::string &why,
                       const char *noun = nullptr)
{
  if (got != want)
    throw bad_array(key, (noun ? "has " + std::to_string(got) + " " + noun : "is " + std::to_string(got)) +
                             ", expected " + std::to_string(want) + " (" + why + ")");
}

// every value an index into n things: in [0, n)
inline void require_index(const std::string &key, const std::vector<int> &v, int64_t n)
{
  for (size_t q = 0; q < v.size(); q++)
    if (v[q] < 0 || v[q] >= n)
      throw bad_array(key, "has " + std::to_string(v[q]) + " at " + std::to_string(q) + ", outside [0, " +
                               std::to_string(n) + ")");
}

// 1 + the largest value (0 for none): the functions an index array addresses
inline int64_t index_extent(const std::vector<int> &v)
{
  int64_t n = 0;
  for (int x : v) n = std::max(n, (int64_t) x + 1);
  return n;
}

// the names got (of key) are the set want, in any order
inline void require_name_set(const std::string &key, std::vector<std::string> got, std::vector<std::string> want)
{
  auto join = [](const std::vector<std::string> &v) {
    std::string o = "{";
    for (size_t i = 0; i < v.size(); i++) o += (i ? ", " : "") + v[i];
    return o + "}";
  };
  const std::string g = join(got), w = join(want);
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  if (got != want) throw bad_array(key, "lists " + g + ", expected " + w);
}

struct MlpLayer {
  std::vector<double> W;    // [n_in, n_out]
  double norm = 1.0;
  int n_in = 0, n_out = 0;
};

// the n_layers layers prefix_W{i} [n_in, n_out] with scalar norms prefix_norm{i} (default 1) of an MLP, chained (the
// rows of a layer are the columns of the one before): n_in rows first (in_why), n_out columns last (out_why). A
// layer stored beyond n_layers (prefix_n_layers) is an error.
inline std::vector<MlpLayer> npz_mlp(const cnpy::npz_t &npz, const std::string &prefix, int n_layers, int64_t n_in,
                                     int64_t n_out, const std::string &in_why, const std::string &out_why)
{
  auto wkey = [&](int i) { return prefix + "_W" + std::to_string(i); };
  if (npz.count(wkey(n_layers)))
    throw bad_array(prefix + "_n_layers", "is " + std::to_string(n_layers) + ", but the file has " + wkey(n_layers));
  std::vector<MlpLayer> L;
  for (int i = 0; i < n_layers; i++) {
    MlpLayer l;
    l.W = npz_require_double(npz, wkey(i));
    l.norm = npz_get_double_scalar(npz, prefix + "_norm" + std::to_string(i), 1.0);
    const auto d = npz_shape_rank(npz, wkey(i), 2);
    l.n_in = (int) d[0];
    l.n_out = (int) d[1];
    require_eq(wkey(i), l.n_in, i ? L.back().n_out : n_in, i ? "the columns of " + wkey(i - 1) : in_why, "rows");
    L.push_back(std::move(l));
  }
  require_eq(wkey(n_layers - 1), L.back().n_out, n_out, out_why, "columns");
  return L;
}

}    // namespace npz

struct UqHead;
// reads and validates the uq_* arrays when the model has them (uq_schema_version present) and sets has_uq;
// max_density: density channels the evaluating kernels write (0: any); who: message prefix
void load_uq_head(UqHead &h, const cnpy::npz_t &npz, int n_elements, int max_density, const std::string &who);

}    // namespace grace
#endif

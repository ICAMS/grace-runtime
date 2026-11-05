// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Contributing author: Yury Lysogorskiy (ICAMS)
// GRACE-1L / 2L models with max_order 3 have no AAAA (layer 1) and BBBB (layer 2) stage. The engines compute A .. AAAA
// and B .. BBBB, so the loaders add a dead stage to the archive they read: AA2 = AA1's FC, one CG term, one output
// function, and one connection with weight zero in every reduce that reads it. The results are those of the model
// without the stage. The engines that prune what no connection reads (CPU fused, Kokkos fused) pay one FC output, one
// CG term and one reduce row per atom; the generic Kokkos engines compute the copied FC in full.
#ifndef GRACE_MAX_ORDER3_H
#define GRACE_MAX_ORDER3_H

#include "npz3.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace grace {
namespace npz {

namespace detail {
inline const cnpy::NpyArray &get_array(const cnpy::npz_t &npz, const std::string &key)
{
  const cnpy::NpyArray *p = find_array(npz, key);
  if (!p) throw bad_array(key, "is missing");
  return *p;
}

template <typename T>
inline cnpy::NpyArray make_array(const char *descr, std::vector<size_t> shape, std::initializer_list<T> values)
{
  cnpy::NpyArray a(shape, sizeof(T), false);
  a.descr = descr;
  size_t q = 0;
  for (T v : values) a.data<T>()[q++] = v;
  return a;
}
}    // namespace detail

// One reduce of a stage: the arrays of `from` (a template for the shapes) become a dead `to`: W zero with one weight
// column, one connection to function 0.
inline void add_dead_reduce(cnpy::npz_t &npz, const std::string &prefix, const std::string &from, const std::string &to)
{
  using namespace detail;
  const std::string w = prefix + "_reduce_" + from + "_W";
  const cnpy::NpyArray &W = get_array(npz, w);
  if (W.shape.empty() || W.shape.back() == 0) throw bad_array(w, "has no weight column");
  cnpy::NpyArray dead_w = W;
  dead_w.shape.back() = 1;
  dead_w.num_vals = W.num_vals / W.shape.back();
  dead_w.data_holder = std::make_shared<std::vector<char>>(dead_w.num_vals * W.word_size, 0);
  npz[prefix + "_reduce_" + to + "_W"] = dead_w;
  const auto copy_scalar = [&](const std::string &src, const std::string &dst) {    // as the template's (if present)
    if (const cnpy::NpyArray *a = find_array(npz, src)) npz[dst] = *a;
  };
  copy_scalar(prefix + "_reduce_" + from + "_norm", prefix + "_reduce_" + to + "_norm");
  copy_scalar(prefix + "_n_in_" + from, prefix + "_n_in_" + to);
  npz[prefix + "_collect_ind_" + to] = make_array<int32_t>("<i4", {1}, {0});
  npz[prefix + "_w_l_tile_" + to] = make_array<int32_t>("<i4", {1}, {0});
  npz[prefix + "_total_sum_ind_" + to] = make_array<int32_t>("<i4", {1, 1}, {0});
  npz[prefix + "_w_shape_" + to] = make_array<int64_t>("<i8", {1}, {1});
  const std::string names = prefix + "_instruction_names";    // a source list: the stage is one more source
  if (const cnpy::NpyArray *n = find_array(npz, names)) {
    if (n->kind() != 'S' || n->shape.size() != 1) throw dtype_error(names, *n, "a 1-d byte-string array (|S)");
    if (to.size() > n->word_size) throw bad_array(names, "has names too short for " + to);
    cnpy::NpyArray a({n->shape[0] + 1}, n->word_size, false);
    a.descr = n->descr;
    std::copy(n->data<char>(), n->data<char>() + n->num_bytes(), a.data<char>());
    std::copy(to.begin(), to.end(), a.data<char>() + n->num_bytes());
    npz[names] = a;
  }
}

// The stage `to` (AAAA / BBBB) of a layer: FC `fc_to` (= the FC of `fc_from`), the product prod_<to>, and the reduces
// `prefixes` (each reads `from` = AAA / BBB as the shape template).
struct DeadStage {
  std::string from, to, fc_from, fc_to;
  std::vector<std::string> reduce_prefixes;
};

inline bool has_stage(const cnpy::npz_t &npz, const DeadStage &s) { return npz.count("prod_" + s.to + "_n_cg_terms") > 0; }

inline void add_dead_stage(cnpy::npz_t &npz, const DeadStage &s)
{
  using namespace detail;
  for (const auto &kv : std::vector<std::pair<std::string, cnpy::NpyArray>>(npz.begin(), npz.end()))
    if (kv.first.rfind(s.fc_from + "_", 0) == 0) npz[s.fc_to + kv.first.substr(s.fc_from.size())] = kv.second;
  const std::string p = "prod_" + s.to + "_";
  npz[p + "left_ind"] = make_array<int32_t>("<i4", {1}, {0});
  npz[p + "right_ind"] = make_array<int32_t>("<i4", {1}, {0});
  npz[p + "m_sum_ind"] = make_array<int32_t>("<i4", {1}, {0});
  npz[p + "cg_coeff"] = make_array<double>("<f8", {1}, {1.0});
  npz[p + "n_output_funcs"] = make_array<int64_t>("<i8", {1}, {1});
  npz[p + "n_cg_terms"] = make_array<int64_t>("<i8", {1}, {1});
  for (const std::string &r : s.reduce_prefixes) add_dead_reduce(npz, r, s.from, s.to);
}

// The arrays the dead stage is built from: without them the file is not a model of max_order 3, and the loader's own
// checks report what is wrong with it (in their order).
inline bool has_templates(const cnpy::npz_t &npz, const DeadStage &s)
{
  if (!npz.count(s.fc_from + "_w_left")) return false;
  for (const std::string &r : s.reduce_prefixes)
    if (!npz.count(r + "_reduce_" + s.from + "_W")) return false;
  return true;
}

// A file with some arrays of a stage but not its product is broken, not an order-3 model.
inline void require_no_stage_arrays(const cnpy::npz_t &npz, const DeadStage &s)
{
  for (const auto &kv : npz)
    if (kv.first.find(s.to) != std::string::npos || kv.first.rfind(s.fc_to + "_", 0) == 0)
      throw bad_array(kv.first, "is present, but prod_" + s.to + "_n_cg_terms is not: the " + s.to + " stage is incomplete");
}

// Adds the dead stages of `stages` to a file without them (max_order 3); returns the max_order of the file: 4 with all
// stages present, 3 with none (4 also if the arrays to build the stages from are missing). Some but not all of the
// stages is an error.
inline int complete_to_order4(cnpy::npz_t &npz, const std::vector<DeadStage> &stages)
{
  std::string have, lack;
  for (const DeadStage &s : stages) (has_stage(npz, s) ? have : lack) += " " + s.to;
  if (lack.empty()) return 4;
  if (!have.empty())
    throw Error("the file has the stage(s)" + have + " but not" + lack + " (a GRACE model has all or none of them)");
  for (const DeadStage &s : stages) require_no_stage_arrays(npz, s);
  for (const DeadStage &s : stages)
    if (!has_templates(npz, s)) return 4;
  for (const DeadStage &s : stages) add_dead_stage(npz, s);
  return 3;
}

}    // namespace npz
}    // namespace grace

#endif

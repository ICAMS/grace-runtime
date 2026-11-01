// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// zero padding of weight arrays (GRACE1LModel::padded, GRACE2LModel::padded; private to libgrace)
#pragma once

#include "grace/error.h"

#include <cstddef>
#include <vector>

namespace grace {

// a row-major array of dims `from` zero-padded to dims `to` (to[k] >= from[k])
inline std::vector<double> zero_pad(const std::vector<double> &a, const std::vector<int> &from,
                                    const std::vector<int> &to)
{
  size_t n = 1, n0 = 1;
  for (int t : to) n *= (size_t) t;
  for (int f : from) n0 *= (size_t) f;
  if (n0 != a.size()) throw Error("padding an array of unexpected size");
  std::vector<double> out(n, 0.0);
  std::vector<int> idx(from.size(), 0);
  for (double v : a) {
    size_t o = 0;
    for (size_t k = 0; k < to.size(); k++) o = o * to[k] + idx[k];
    out[o] = v;
    for (int k = (int) from.size() - 1; k >= 0 && ++idx[k] == from[k]; k--) idx[k] = 0;
  }
  return out;
}

// the last radial-MLP layer (outputs o = n * nl + l) zero-padded from n to `to` channels
template<class Layer> void pad_radial_out(Layer &L, int n, int to, int nl)
{
  L.W = zero_pad(L.W, {L.n_in, n, nl}, {L.n_in, to, nl});
  L.n_out = to * nl;
}

// an FC (w_left / w_right [n_out, n_in, w_shape]) zero-padded to `no` outputs and to_left / to_right inputs
template<class FC> void pad_fc(FC &w, int no, int to_left, int to_right)
{
  const int nl_in = (int) (w.w_left.size() / ((size_t) w.n_out * w.w_shape_left));
  const int nr_in = (int) (w.w_right.size() / ((size_t) w.n_out * w.w_shape_right));
  w.w_left = zero_pad(w.w_left, {w.n_out, nl_in, w.w_shape_left}, {no, to_left, w.w_shape_left});
  w.w_right = zero_pad(w.w_right, {w.n_out, nr_in, w.w_shape_right}, {no, to_right, w.w_shape_right});
  w.n_out = no;
}

// a reduce (W [ne, n_out, n_in, w_shape]) zero-padded to to_out outputs and to_in inputs
template<class Reduce> void pad_reduce(Reduce &w, int ne, int n_out, int to_out, int to_in)
{
  const int ws = w.w_shape > 0 ? w.w_shape : 1, n_in = (int) (w.W.size() / ((size_t) ne * n_out * ws));
  w.W = zero_pad(w.W, {ne, n_out, n_in, ws}, {ne, to_out, to_in, ws});
  w.n_in = to_in;
}

}    // namespace grace

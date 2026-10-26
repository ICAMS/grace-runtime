// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
#include "neighbors.h"

#include "grace/cell_list.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace grace_runtime {

NeighborResult neighbor_search(const double *pos, int64_t n, const double *cell, const bool *pbc, double cutoff,
                               int num_threads)
{
  if (!(cutoff > 0.0) || !std::isfinite(cutoff)) throw std::invalid_argument("cutoff must be positive");
  if (n < 0 || n >= INT32_MAX) throw std::invalid_argument("atom count out of range");
  NeighborResult r;
  r.offset.assign(n + 1, 0);
  if (n == 0) return r;
  // bins of edge >= cutoff / 2, atoms in bin order (the spatial order the 2L driver wants)
  grace::CellList cl;
  cl.build(pos, n, cell, pbc, 0.5 * cutoff);
  r.perm.resize(n);
  for (int64_t q = 0; q < n; q++) r.perm[q] = cl.perm(q);
  // visit(q, fn): every neighbour of new atom q, fn(p, S0, S1, S2) with S the image in the wrapped frame
  auto visit = [&](int64_t q, auto &&fn) {
    cl.visit_slot(q, cutoff, [&](int64_t p, int64_t s0, int64_t s1, int64_t s2, double, double, double, double) {
      fn(p, s0, s1, s2);
    });
  };
  std::vector<int64_t> nn(n);
  const int nth = std::max(1, num_threads);
#pragma omp parallel for schedule(dynamic, 64) num_threads(nth)
  for (int64_t q = 0; q < n; q++) {
    int64_t c = 0;
    visit(q, [&](int64_t, int64_t, int64_t, int64_t) { c++; });
    nn[q] = c;
  }
  for (int64_t q = 0; q < n; q++) r.offset[q + 1] = r.offset[q] + nn[q];
  const int64_t nb = r.offset[n];
  r.j.resize(nb);
  r.io.resize(nb);
  r.jo.resize(nb);
  r.shift.resize(3 * nb);
#pragma omp parallel for schedule(dynamic, 64) num_threads(nth)
  for (int64_t q = 0; q < n; q++) {
    int64_t w = r.offset[q];
    const int64_t a = r.perm[q];
    visit(q, [&](int64_t p, int64_t s0, int64_t s1, int64_t s2) {
      const int64_t b = r.perm[p];
      // unwrapped frame: S = S_wrapped + n_a - n_b (only periodic directions carry images)
      const int64_t *ia = cl.image(a), *ib = cl.image(b);
      const double t0 = (double) (s0 - ib[0] + ia[0]);
      const double t1 = (double) (s1 - ib[1] + ia[1]);
      const double t2 = (double) (s2 - ib[2] + ia[2]);
      r.j[w] = (int32_t) p;
      r.io[w] = a;
      r.jo[w] = (int32_t) b;
      for (int d = 0; d < 3; d++) r.shift[3 * w + d] = t0 * cell[d] + t1 * cell[3 + d] + t2 * cell[6 + d];
      w++;
    });
  }
  return r;
}

}    // namespace grace_runtime

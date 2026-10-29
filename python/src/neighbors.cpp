// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
#include "neighbors.h"

#include "grace/cell_list.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

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
  // the image of a bond in the unwrapped frame, S = S_wrapped + n_a - n_b (only periodic directions carry images),
  // packed into one key; |S| < 2^20 per direction (atoms up to a million cells apart)
  constexpr int64_t kB = int64_t(1) << 20;
  std::atomic<bool> far{false};
  auto key = [&](int64_t a, int64_t b, int64_t s0, int64_t s1, int64_t s2) {
    const int64_t *ia = cl.image(a), *ib = cl.image(b);
    const int64_t t0 = s0 - ib[0] + ia[0], t1 = s1 - ib[1] + ia[1], t2 = s2 - ib[2] + ia[2];
    if (std::max({std::abs(t0), std::abs(t1), std::abs(t2)}) >= kB) {
      far.store(true, std::memory_order_relaxed);    // checked after the parallel region
      return int64_t(0);
    }
    return ((t0 + kB) << 42) | ((t1 + kB) << 21) | (t2 + kB);
  };
  std::vector<int64_t> nn(n), keys;    // keys: the distinct images, sorted
  const int nth = std::max(1, num_threads);
#pragma omp parallel num_threads(nth)
  {
    std::unordered_set<int64_t> seen;
    int64_t last = -1;
#pragma omp for schedule(dynamic, 64)
    for (int64_t q = 0; q < n; q++) {
      const int64_t a = r.perm[q];
      int64_t c = 0;
      visit(q, [&](int64_t p, int64_t s0, int64_t s1, int64_t s2) {
        c++;
        const int64_t k = key(a, r.perm[p], s0, s1, s2);
        if (k != last) seen.insert(last = k);
      });
      nn[q] = c;
    }
#pragma omp critical
    keys.insert(keys.end(), seen.begin(), seen.end());
  }
  if (far) throw std::invalid_argument("atoms a million cells or more apart: wrap the positions");
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  r.shifts.resize(3 * keys.size());
  for (size_t u = 0; u < keys.size(); u++) {
    const int64_t m = (int64_t(1) << 21) - 1, k = keys[u];
    const double t0 = (double) (((k >> 42) & m) - kB), t1 = (double) (((k >> 21) & m) - kB),
                 t2 = (double) ((k & m) - kB);
    for (int d = 0; d < 3; d++) r.shifts[3 * u + d] = t0 * cell[d] + t1 * cell[3 + d] + t2 * cell[6 + d];
  }
  for (int64_t q = 0; q < n; q++) r.offset[q + 1] = r.offset[q] + nn[q];
  const int64_t nb = r.offset[n];
  r.j.resize(nb);
  r.image.resize(nb);
#pragma omp parallel for schedule(dynamic, 64) num_threads(nth)
  for (int64_t q = 0; q < n; q++) {
    int64_t w = r.offset[q], last = -1;
    int32_t u = 0;
    const int64_t a = r.perm[q];
    visit(q, [&](int64_t p, int64_t s0, int64_t s1, int64_t s2) {
      const int64_t k = key(a, r.perm[p], s0, s1, s2);
      if (k != last) u = (int32_t) (std::lower_bound(keys.begin(), keys.end(), last = k) - keys.begin());
      r.j[w] = (int32_t) p;
      r.image[w] = u;
      w++;
    });
  }
  return r;
}

}    // namespace grace_runtime

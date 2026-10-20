// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
#include "batch_csr.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace grace_runtime {

namespace {
using Part = BatchSearch::Part;

void one_structure(Part &p, int64_t n, const double *pos, const double *h, const bool *pbc, double cutoff, int nth)
{
  if (n == 0) return;
  p.nl = neighbor_search(pos, n, h, pbc, cutoff, nth);
  const double det = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                     h[2] * (h[3] * h[7] - h[4] * h[6]);
  const double inv[9] = {(h[4] * h[8] - h[5] * h[7]) / det, (h[2] * h[7] - h[1] * h[8]) / det,
                         (h[1] * h[5] - h[2] * h[4]) / det, (h[5] * h[6] - h[3] * h[8]) / det,
                         (h[0] * h[8] - h[2] * h[6]) / det, (h[2] * h[3] - h[0] * h[5]) / det,
                         (h[3] * h[7] - h[4] * h[6]) / det, (h[1] * h[6] - h[0] * h[7]) / det,
                         (h[0] * h[4] - h[1] * h[3]) / det};
  const int64_t nb = p.nl.offset[n];
  p.slot.assign(nb, -1);
  std::unordered_map<uint64_t, int32_t> seen;
  for (int64_t q = 0; q < n; q++) {
    p.maxneigh = std::max(p.maxneigh, (int) (p.nl.offset[q + 1] - p.nl.offset[q]));
    for (int64_t b = p.nl.offset[q]; b < p.nl.offset[q + 1]; b++) {
      const double *t = &p.nl.shift[3 * b];
      int s[3];
      for (int k = 0; k < 3; k++) {
        const double sk = std::nearbyint(t[0] * inv[k] + t[1] * inv[3 + k] + t[2] * inv[6 + k]);
        if (!(std::fabs(sk) <= 511)) throw std::invalid_argument("CSR images: lattice translation beyond +-511 cells");
        s[k] = (int) sk;
      }
      if (s[0] == 0 && s[1] == 0 && s[2] == 0) continue;
      const int32_t j = p.nl.j[b];
      const uint64_t key = ((uint64_t) j << 30) | ((uint64_t) (s[0] + 512) << 20) | ((uint64_t) (s[1] + 512) << 10) |
                           (uint64_t) (s[2] + 512);
      auto it = seen.find(key);
      if (it == seen.end()) {
        it = seen.emplace(key, (int32_t) p.img_j.size()).first;
        p.img_j.push_back(j);
        p.img_t.insert(p.img_t.end(), t, t + 3);
      }
      p.slot[b] = it->second;
    }
  }
}
}    // namespace

BatchSearch batch_search(int nsys, const int32_t *natoms, const double *pos, const double *cell, const bool *pbc,
                         double cutoff, int num_threads)
{
  if (nsys < 0) throw std::invalid_argument("nsys must be >= 0");
  BatchSearch bs;
  bs.start.assign(nsys + 1, 0);
  for (int s = 0; s < nsys; s++) {
    if (natoms[s] < 0) throw std::invalid_argument("natoms must be >= 0");
    bs.start[s + 1] = bs.start[s] + natoms[s];
  }
  bs.parts.resize(nsys);
  std::vector<std::string> err(nsys);
  const int nth = std::max(1, num_threads);
  auto run = [&](int s, int inner) {
    try {
      one_structure(bs.parts[s], natoms[s], pos + 3 * bs.start[s], cell + 9 * s, pbc + 3 * s, cutoff, inner);
    } catch (const std::exception &e) {
      err[s] = e.what();
    }
  };
  if (nsys >= nth) {    // many structures: one thread each; few (large) ones: threads inside the cell list
#pragma omp parallel for schedule(dynamic, 1) num_threads(nth)
    for (int s = 0; s < nsys; s++) run(s, 1);
  } else {
    for (int s = 0; s < nsys; s++) run(s, nth);
  }
  for (const auto &e : err)
    if (!e.empty()) throw std::invalid_argument(e);
  return bs;
}

BatchCsr batch_layout(const BatchSearch &bs, const int32_t *types, const double *pos, int s0, int s1, int num_threads)
{
  const int64_t base = bs.start[s0];
  std::vector<int64_t> istart(s1 - s0 + 1, bs.start[s1] - base);
  int maxneigh = 0;
  for (int s = s0; s < s1; s++) {
    istart[s - s0 + 1] = istart[s - s0] + bs.images(s);
    maxneigh = std::max(maxneigh, bs.parts[s].maxneigh);
  }
  if (istart[s1 - s0] > INT_MAX) throw std::invalid_argument("batch pass too large: lower max_nall");
  const int nth = std::max(1, num_threads);
  BatchCsr r;
  r.n = (int) (bs.start[s1] - base);
  r.nall = (int) istart[s1 - s0];
  r.maxneigh = maxneigh;
  const size_t mn = std::max(maxneigh, 1);
  r.perm.resize(r.n);
  r.sys.resize(r.n);
  r.type.resize(r.nall);
  r.x.resize(3 * (size_t) r.nall);
  r.numneigh.resize(r.n);
  r.neighbors.assign((size_t) r.n * mn, 0);
  r.owner.resize(r.nall - r.n);
  r.tcart.resize(3 * (size_t) (r.nall - r.n));
#pragma omp parallel for schedule(dynamic, 1) num_threads(nth)
  for (int s = s0; s < s1; s++) {
    const Part &p = bs.parts[s];
    const int64_t a0 = bs.start[s] - base, in0 = bs.start[s], g0 = istart[s - s0];
    const int64_t n = bs.start[s + 1] - bs.start[s];
    for (int64_t q = 0; q < n; q++) {
      const int64_t g = a0 + q, a = in0 + p.nl.perm[q];
      r.perm[g] = (int32_t) a;
      r.sys[g] = s - s0;
      r.type[g] = types[a];
      for (int k = 0; k < 3; k++) r.x[3 * g + k] = pos[3 * a + k];
      r.numneigh[g] = (int32_t) (p.nl.offset[q + 1] - p.nl.offset[q]);
      int32_t *row = &r.neighbors[(size_t) g * mn];
      for (int64_t b = p.nl.offset[q]; b < p.nl.offset[q + 1]; b++)
        row[b - p.nl.offset[q]] = (int32_t) (p.slot[b] < 0 ? a0 + p.nl.j[b] : g0 + p.slot[b]);
    }
    for (size_t i = 0; i < p.img_j.size(); i++) {
      const int64_t gi = g0 + (int64_t) i, ow = a0 + p.img_j[i], a = in0 + p.nl.perm[p.img_j[i]];
      r.owner[gi - r.n] = (int32_t) ow;
      r.type[gi] = types[a];
      for (int k = 0; k < 3; k++) {
        r.tcart[3 * (gi - r.n) + k] = p.img_t[3 * i + k];
        r.x[3 * gi + k] = pos[3 * a + k] + p.img_t[3 * i + k];
      }
    }
  }
  return r;
}

}    // namespace grace_runtime

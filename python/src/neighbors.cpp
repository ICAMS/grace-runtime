// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
#include "neighbors.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace grace_runtime {

namespace {

void inverse3(const double *a, double *inv, double &det)
{
  det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
  if (!(std::abs(det) > 0.0)) throw std::invalid_argument("neighbour search needs a full-rank cell");
  const double d = 1.0 / det;
  inv[0] = (a[4] * a[8] - a[5] * a[7]) * d;
  inv[1] = (a[2] * a[7] - a[1] * a[8]) * d;
  inv[2] = (a[1] * a[5] - a[2] * a[4]) * d;
  inv[3] = (a[5] * a[6] - a[3] * a[8]) * d;
  inv[4] = (a[0] * a[8] - a[2] * a[6]) * d;
  inv[5] = (a[2] * a[3] - a[0] * a[5]) * d;
  inv[6] = (a[3] * a[7] - a[4] * a[6]) * d;
  inv[7] = (a[1] * a[6] - a[0] * a[7]) * d;
  inv[8] = (a[0] * a[4] - a[1] * a[3]) * d;
}

int64_t floordiv(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

}    // namespace

NeighborResult neighbor_search(const double *pos, int64_t n, const double *cell, const bool *pbc, double cutoff,
                               int num_threads)
{
  if (!(cutoff > 0.0) || !std::isfinite(cutoff)) throw std::invalid_argument("cutoff must be positive");
  if (n < 0 || n >= INT32_MAX) throw std::invalid_argument("atom count out of range");
  NeighborResult r;
  r.offset.assign(n + 1, 0);
  if (n == 0) return r;
  // frac = x inv(cell): rows of cell are the vectors a_k, x = frac cell
  double inv[9], det;
  inverse3(cell, inv, det);
  // perpendicular heights h_k = 1 / |column k of inv(cell)|; a cutoff sphere spans at most rc / h_k in frac_k
  double h[3];
  for (int k = 0; k < 3; k++)
    h[k] = 1.0 / std::sqrt(inv[k] * inv[k] + inv[3 + k] * inv[3 + k] + inv[6 + k] * inv[6 + k]);
  std::vector<double> frac(3 * n);
  std::vector<int64_t> img(3 * n);    // integer image n_i: wrapped frac = frac - n_i (periodic directions)
  double lo[3], span[3];
  for (int k = 0; k < 3; k++) {
    lo[k] = 0.0;
    span[k] = 1.0;
  }
  for (int64_t a = 0; a < n; a++)
    for (int k = 0; k < 3; k++) {
      const double f = pos[3 * a] * inv[k] + pos[3 * a + 1] * inv[3 + k] + pos[3 * a + 2] * inv[6 + k];
      if (!std::isfinite(f)) throw std::invalid_argument("positions must be finite");
      const double fl = pbc[k] ? std::floor(f) : 0.0;
      img[3 * a + k] = (int64_t) fl;
      frac[3 * a + k] = f - fl;
    }
  for (int k = 0; k < 3; k++)
    if (!pbc[k]) {    // open direction: bins over the occupied range
      double mn = frac[k], mx = frac[k];
      for (int64_t a = 1; a < n; a++) {
        mn = std::min(mn, frac[3 * a + k]);
        mx = std::max(mx, frac[3 * a + k]);
      }
      lo[k] = mn;
      span[k] = std::max(mx - mn, 1e-12);
    }
  // bins of edge >= cutoff / 2 (span * h / nbin >= rc / 2); total capped so empty bins stay cheap
  int64_t nbin[3];
  for (int k = 0; k < 3; k++) nbin[k] = std::max<int64_t>(1, (int64_t) std::floor(span[k] * h[k] / (0.5 * cutoff)));
  while (nbin[0] * nbin[1] * nbin[2] > 8 * n + 64) {
    const int k = (int) (std::max_element(nbin, nbin + 3) - nbin);
    nbin[k] = std::max<int64_t>(1, nbin[k] / 2);
  }
  // reach in bins per direction; periodic directions may wrap several times (cells smaller than the cutoff)
  int64_t reach[3];
  for (int k = 0; k < 3; k++) {
    const double w = span[k] * h[k] / (double) nbin[k];    // bin width along the height
    reach[k] = (int64_t) std::ceil(cutoff / w - 1e-12);
    if (!pbc[k]) reach[k] = std::min(reach[k], nbin[k] - 1);
  }
  const int64_t nbt = nbin[0] * nbin[1] * nbin[2];
  std::vector<int64_t> bin(n), count(nbt + 1, 0);
  std::vector<int64_t> b3(3 * n);
  for (int64_t a = 0; a < n; a++) {
    for (int k = 0; k < 3; k++) {
      int64_t b = (int64_t) std::floor((frac[3 * a + k] - lo[k]) / span[k] * (double) nbin[k]);
      b3[3 * a + k] = std::min(std::max<int64_t>(b, 0), nbin[k] - 1);
    }
    bin[a] = (b3[3 * a] * nbin[1] + b3[3 * a + 1]) * nbin[2] + b3[3 * a + 2];
    count[bin[a] + 1]++;
  }
  for (int64_t b = 0; b < nbt; b++) count[b + 1] += count[b];
  r.perm.resize(n);
  {
    std::vector<int64_t> fill(count.begin(), count.end() - 1);
    for (int64_t a = 0; a < n; a++) {    // stable counting sort by bin
      const int64_t q = fill[bin[a]]++;
      r.perm[q] = a;
    }
  }
  // wrapped Cartesian positions in the new order
  std::vector<double> xw(3 * n);
  for (int64_t q = 0; q < n; q++) {
    const int64_t a = r.perm[q];
    for (int d = 0; d < 3; d++)
      xw[3 * q + d] = frac[3 * a] * cell[d] + frac[3 * a + 1] * cell[3 + d] + frac[3 * a + 2] * cell[6 + d];
  }
  const double rc2 = cutoff * cutoff;
  // visit(q, fn): every neighbour of new atom q, fn(p, S0, S1, S2, dx, dy, dz) with S the image in the wrapped frame
  auto visit = [&](int64_t q, auto &&fn) {
    const int64_t a = r.perm[q];
    const int64_t *bq = &b3[3 * a];
    for (int64_t o0 = -reach[0]; o0 <= reach[0]; o0++) {
      int64_t c0 = bq[0] + o0, s0 = 0;
      if (pbc[0]) { s0 = floordiv(c0, nbin[0]); c0 -= s0 * nbin[0]; }
      else if (c0 < 0 || c0 >= nbin[0]) continue;
      for (int64_t o1 = -reach[1]; o1 <= reach[1]; o1++) {
        int64_t c1 = bq[1] + o1, s1 = 0;
        if (pbc[1]) { s1 = floordiv(c1, nbin[1]); c1 -= s1 * nbin[1]; }
        else if (c1 < 0 || c1 >= nbin[1]) continue;
        for (int64_t o2 = -reach[2]; o2 <= reach[2]; o2++) {
          int64_t c2 = bq[2] + o2, s2 = 0;
          if (pbc[2]) { s2 = floordiv(c2, nbin[2]); c2 -= s2 * nbin[2]; }
          else if (c2 < 0 || c2 >= nbin[2]) continue;
          const double sx = s0 * cell[0] + s1 * cell[3] + s2 * cell[6];
          const double sy = s0 * cell[1] + s1 * cell[4] + s2 * cell[7];
          const double sz = s0 * cell[2] + s1 * cell[5] + s2 * cell[8];
          const int64_t c = (c0 * nbin[1] + c1) * nbin[2] + c2;
          for (int64_t p = count[c]; p < count[c + 1]; p++) {
            if (p == q && s0 == 0 && s1 == 0 && s2 == 0) continue;
            const double dx = xw[3 * p] + sx - xw[3 * q];
            const double dy = xw[3 * p + 1] + sy - xw[3 * q + 1];
            const double dz = xw[3 * p + 2] + sz - xw[3 * q + 2];
            if (dx * dx + dy * dy + dz * dz < rc2) fn(p, s0, s1, s2);
          }
        }
      }
    }
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
      const double t0 = (double) (s0 - img[3 * b] + img[3 * a]);
      const double t1 = (double) (s1 - img[3 * b + 1] + img[3 * a + 1]);
      const double t2 = (double) (s2 - img[3 * b + 2] + img[3 * a + 2]);
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

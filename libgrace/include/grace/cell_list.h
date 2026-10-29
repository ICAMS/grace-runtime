// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Host cell list that visits the neighbours of an atom within any radius on the fly, without a stored list (the
// long-range dispersion terms reach ~1500 neighbours per atom: gather_half); the neighbour search of the Python
// package (python/src/neighbors.cpp) lists its visits (visit_slot). Binning as the device list
// (grace/kk/neighbors_kk.h): fractional coordinates, wrapped in periodic directions, bins over the occupied range in
// open directions, bin edge >= bin_size along each height, atoms in the stable bin order.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace grace {

// bins to visit on each side for a radius rc, bins of width w (along the height) and nbin bins in one direction;
// periodic directions may wrap several times (cells smaller than rc)
inline int64_t bin_reach(double rc, double w, int64_t nbin, bool pbc)
{
  const int64_t r = (int64_t) std::ceil(rc / w - 1e-12);
  return pbc ? r : std::min(r, nbin - 1);
}

// a cell (rows a, b, c = the three cell vectors) the neighbour search can use: |det| above 1e-12 |a| |b| |c|, so a
// nearly flat cell is rejected (its inverse and bin counts blow up) as well as a singular one
inline bool full_rank_cell(const double *a, double det)
{
  const double na = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  const double nb = std::sqrt(a[3] * a[3] + a[4] * a[4] + a[5] * a[5]);
  const double nc = std::sqrt(a[6] * a[6] + a[7] * a[7] + a[8] * a[8]);
  return std::abs(det) > 1e-12 * na * nb * nc;
}

class CellList {
 public:
  // pos [n][3] in the caller's order, cell [3][3] (rows = vectors, full rank; open directions padded by the caller),
  // pbc [3]; bins of edge >= bin_size
  void build(const double *pos, int64_t n, const double *cell, const bool *pbc, double bin_size)
  {
    if (!(bin_size > 0.0) || !std::isfinite(bin_size))
      throw std::invalid_argument("cell list: bin size must be positive");
    n_ = n;
    max_bin_ = 0;
    for (int k = 0; k < 9; k++) cell_[k] = cell[k];
    for (int k = 0; k < 3; k++) {
      pbc_[k] = pbc[k];
      nbin_[k] = 1;
      width_[k] = 1.0;
    }
    perm_.assign(n, 0);
    slot_.clear();
    start_.clear();
    xw_.assign(3 * n, 0.0);
    b3_.assign(3 * n, 0);
    img_.assign(3 * n, 0);
    if (n == 0) return;
    const double *a = cell;
    const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                       a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (!std::isfinite(det) || !full_rank_cell(a, det))
      throw std::invalid_argument("neighbour search needs a full-rank cell");
    const double d = 1.0 / det;
    const double inv[9] = {(a[4] * a[8] - a[5] * a[7]) * d, (a[2] * a[7] - a[1] * a[8]) * d,
                           (a[1] * a[5] - a[2] * a[4]) * d, (a[5] * a[6] - a[3] * a[8]) * d,
                           (a[0] * a[8] - a[2] * a[6]) * d, (a[2] * a[3] - a[0] * a[5]) * d,
                           (a[3] * a[7] - a[4] * a[6]) * d, (a[1] * a[6] - a[0] * a[7]) * d,
                           (a[0] * a[4] - a[1] * a[3]) * d};
    double h[3], lo[3] = {0.0, 0.0, 0.0}, span[3] = {1.0, 1.0, 1.0};
    for (int k = 0; k < 3; k++)
      h[k] = 1.0 / std::sqrt(inv[k] * inv[k] + inv[3 + k] * inv[3 + k] + inv[6 + k] * inv[6 + k]);
    std::vector<double> frac(3 * n);
    for (int64_t i = 0; i < n; i++)
      for (int k = 0; k < 3; k++) {
        const double f = pos[3 * i] * inv[k] + pos[3 * i + 1] * inv[3 + k] + pos[3 * i + 2] * inv[6 + k];
        if (!std::isfinite(f)) throw std::invalid_argument("positions must be finite");
        const double fl = pbc[k] ? std::floor(f) : 0.0;
        img_[3 * i + k] = (int64_t) fl;
        frac[3 * i + k] = f - fl;
      }
    for (int k = 0; k < 3; k++)
      if (!pbc[k]) {
        double mn = frac[k], mx = frac[k];
        for (int64_t i = 1; i < n; i++) {
          mn = std::min(mn, frac[3 * i + k]);
          mx = std::max(mx, frac[3 * i + k]);
        }
        lo[k] = mn;
        span[k] = std::max(mx - mn, 1e-12);
      }
    const int64_t max_bins = 8 * n + 64;
    for (int k = 0; k < 3; k++) {
      const double nb = std::floor(span[k] * h[k] / bin_size);
      nbin_[k] = (int64_t) std::min<double>(std::max(nb, 1.0), (double) max_bins);
    }
    while ((double) nbin_[0] * (double) nbin_[1] * (double) nbin_[2] > (double) max_bins) {
      const int k = (int) (std::max_element(nbin_, nbin_ + 3) - nbin_);
      nbin_[k] = std::max<int64_t>(1, nbin_[k] / 2);
    }
    for (int k = 0; k < 3; k++) width_[k] = span[k] * h[k] / (double) nbin_[k];
    const int64_t nbt = nbin_[0] * nbin_[1] * nbin_[2];
    std::vector<int64_t> bin(n);
    start_.assign(nbt + 1, 0);
    for (int64_t i = 0; i < n; i++) {
      for (int k = 0; k < 3; k++) {
        const int64_t b = (int64_t) std::floor((frac[3 * i + k] - lo[k]) / span[k] * (double) nbin_[k]);
        b3_[3 * i + k] = std::min(std::max<int64_t>(b, 0), nbin_[k] - 1);
      }
      bin[i] = (b3_[3 * i] * nbin_[1] + b3_[3 * i + 1]) * nbin_[2] + b3_[3 * i + 2];
      start_[bin[i] + 1]++;
    }
    max_bin_ = 0;
    for (int64_t b = 0; b < nbt; b++) {
      max_bin_ = std::max(max_bin_, start_[b + 1]);
      start_[b + 1] += start_[b];
    }
    std::vector<int64_t> fill(start_.begin(), start_.end() - 1);
    std::vector<int64_t> slot(n);
    for (int64_t i = 0; i < n; i++) {    // stable counting sort by bin
      slot[i] = fill[bin[i]]++;
      perm_[slot[i]] = i;
    }
    for (int64_t i = 0; i < n; i++)
      for (int dd = 0; dd < 3; dd++)
        xw_[3 * slot[i] + dd] =
            frac[3 * i] * cell[dd] + frac[3 * i + 1] * cell[3 + dd] + frac[3 * i + 2] * cell[6 + dd];
    slot_ = std::move(slot);
  }

  int64_t size() const { return n_; }
  int64_t perm(int64_t q) const { return perm_[q]; }    // slot (bin order) -> caller index
  // integer image of atom i (caller's index) in periodic directions: wrapped position = x_i - image cell
  const int64_t *image(int64_t i) const { return &img_[3 * i]; }

  // every neighbour of the atom in slot q within rc (periodic images and self-images included):
  // fn(p, s0, s1, s2, dx, dy, dz, r2) with p a slot, S the image in the wrapped frame and d = xw_p + S cell - xw_q
  template<class F> void visit_slot(int64_t q, double rc, F &&fn) const
  {
    const double rc2 = rc * rc;
    const double *xq = &xw_[3 * q];
    for_bins(q, rc, false, [&](int64_t c, int64_t s0, int64_t s1, int64_t s2, double sx, double sy, double sz,
                               bool own) {
      for (int64_t p = start_[c]; p < start_[c + 1]; p++) {
        if (own && p == q) continue;
        const double dx = xw_[3 * p] + sx - xq[0];
        const double dy = xw_[3 * p + 1] + sy - xq[1];
        const double dz = xw_[3 * p + 2] + sz - xq[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < rc2) fn(p, s0, s1, s2, dx, dy, dz, r2);
      }
    });
  }

  // neighbours as arrays (gather_half)
  struct Hits {
    std::vector<int64_t> p;    // slot
    std::vector<double> dx, dy, dz, r2;
    void grow(size_t size)
    {
      p.resize(size);
      dx.resize(size);
      dy.resize(size);
      dz.resize(size);
      r2.resize(size);
    }
    void reserve(size_t size)
    {
      p.reserve(size);
      dx.reserve(size);
      dy.reserve(size);
      dz.reserve(size);
      r2.reserve(size);
    }
  };
  // an upper bound of gather_half's count for radius rc (size Hits with it to keep gather_half from allocating)
  int64_t max_gather(double rc) const
  {
    int64_t bins = 1;
    for (int k = 0; k < 3; k++) bins *= 2 * bin_reach(rc, width_[k], nbin_[k], pbc_[k]) + 1;
    return bins * max_bin_;
  }
  // half list: the neighbours p (slots) of the atom in slot q at 0 < r < rc such that each pair (q, p, S) of the
  // structure comes from one of its two atoms (bin offsets after zero in lexicographic order, in the own bin the later
  // slots), into h (grown as needed): p, d = xw_p + S cell - xw_q, r2; returns their number. Coincident atoms and the
  // atom itself are left out. No branch per candidate (the hot loop of the dispersion terms).
  int64_t gather_half(int64_t q, double rc, Hits &h) const
  {
    const double rc2 = rc * rc;
    const double *xq = &xw_[3 * q];
    int64_t m = 0;
    for_bins(q, rc, true, [&](int64_t c, int64_t, int64_t, int64_t, double sx, double sy, double sz, bool own) {
      const int64_t p0 = own ? q + 1 : start_[c], p1 = start_[c + 1];
      if (m + (p1 - p0) > (int64_t) h.p.size()) h.grow(2 * (size_t) (m + p1 - p0) + 64);
      int64_t *hp = h.p.data();
      double *hx = h.dx.data(), *hy = h.dy.data(), *hz = h.dz.data(), *hr = h.r2.data();
      for (int64_t p = p0; p < p1; p++) {
        const double dx = xw_[3 * p] + sx - xq[0];
        const double dy = xw_[3 * p + 1] + sy - xq[1];
        const double dz = xw_[3 * p + 2] + sz - xq[2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        hp[m] = p;
        hx[m] = dx;
        hy[m] = dy;
        hz[m] = dz;
        hr[m] = r2;
        m += (r2 < rc2) & (r2 > 0.0);
      }
    });
    return m;
  }

 private:
  // the bins within reach of rc around the atom in slot q: fn(c, s0, s1, s2, sx, sy, sz, own) with c the bin, S the
  // image in the wrapped frame, s = S cell and own for offset zero. half: only the offsets from zero on in
  // lexicographic order (the offsets of a pair seen from its two atoms are opposite, so one of them is visited)
  template<class F> void for_bins(int64_t q, double rc, bool half, F &&fn) const
  {
    int64_t reach[3];
    for (int k = 0; k < 3; k++) reach[k] = bin_reach(rc, width_[k], nbin_[k], pbc_[k]);
    const int64_t *bq = &b3_[3 * perm_[q]];
    for (int64_t o0 = half ? 0 : -reach[0]; o0 <= reach[0]; o0++) {
      int64_t c0 = bq[0] + o0, s0 = 0;
      if (pbc_[0]) { s0 = floordiv(c0, nbin_[0]); c0 -= s0 * nbin_[0]; }
      else if (c0 < 0 || c0 >= nbin_[0]) continue;
      for (int64_t o1 = half && o0 == 0 ? 0 : -reach[1]; o1 <= reach[1]; o1++) {
        int64_t c1 = bq[1] + o1, s1 = 0;
        if (pbc_[1]) { s1 = floordiv(c1, nbin_[1]); c1 -= s1 * nbin_[1]; }
        else if (c1 < 0 || c1 >= nbin_[1]) continue;
        for (int64_t o2 = half && o0 == 0 && o1 == 0 ? 0 : -reach[2]; o2 <= reach[2]; o2++) {
          int64_t c2 = bq[2] + o2, s2 = 0;
          if (pbc_[2]) { s2 = floordiv(c2, nbin_[2]); c2 -= s2 * nbin_[2]; }
          else if (c2 < 0 || c2 >= nbin_[2]) continue;
          fn((c0 * nbin_[1] + c1) * nbin_[2] + c2, s0, s1, s2, s0 * cell_[0] + s1 * cell_[3] + s2 * cell_[6],
             s0 * cell_[1] + s1 * cell_[4] + s2 * cell_[7], s0 * cell_[2] + s1 * cell_[5] + s2 * cell_[8],
             o0 == 0 && o1 == 0 && o2 == 0);
        }
      }
    }
  }

  static int64_t floordiv(int64_t a, int64_t b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

  int64_t n_ = 0, max_bin_ = 0;    // atoms, most atoms in one bin
  double cell_[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  bool pbc_[3] = {false, false, false};
  int64_t nbin_[3] = {1, 1, 1};
  double width_[3] = {1.0, 1.0, 1.0};
  std::vector<int64_t> perm_;     // bin order -> caller index
  std::vector<int64_t> slot_;     // caller index -> bin order
  std::vector<int64_t> start_;    // bin -> first slot, nbins + 1
  std::vector<int64_t> b3_;       // caller index -> bin coordinates [3]
  std::vector<int64_t> img_;      // caller index -> integer image [3]
  std::vector<double> xw_;        // wrapped positions in bin order
};

}    // namespace grace

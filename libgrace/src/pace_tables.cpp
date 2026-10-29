// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
#include "grace/pace_tables.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace grace {

PaceTables::PaceTables(const PaceModel &m, int spline)
{
  nel = m.n_elements;
  for (const PaceBond &b : m.bonds)
    if (b.defined) {
      nr = std::max(nr, b.nradmax);
      lmax = std::max(lmax, b.lmax);
    }
  nlm = (lmax + 1) * (lmax + 1);
  for (int i = 0; i < nel; i++)
    for (int j = 0; j < nel; j++)
      if (m.bond(i, j).defined)
        nf = std::max(nf, m.elements[i].ndensity + m.bond(i, j).nradmax * (m.bond(i, j).lmax + 1) + 1);
  hc.resize((size_t) nel * nel);
  for (int i = 0; i < nel; i++)
    for (int j = 0; j < nel; j++)
      if (m.bond(i, j).defined) hc[(size_t) i * nel + j].assign((size_t) m.elements[i].ndensity * m.bond(i, j).nradbase, 0.0);
  na = (size_t) nel * nr * nlm;
  el.resize(nel);
  for (int mu = 0; mu < nel; mu++) {
    const PaceElement &e = m.elements[mu];
    auto &o = el[mu];
    o.nd = e.ndensity;
    o.off.push_back(0);
    for (int r = 0; r < e.nrows; r++) {
      const size_t q0 = (size_t) r * e.rankmax;
      const double *c = &e.ctildes[(size_t) r * e.ndensity];
      const auto uq_row = [&](std::vector<int> &off, std::vector<int> &func, std::vector<double> &cg) {
        if (!m.has_uq) return;
        for (int q = e.uq_off[r]; q < e.uq_off[r + 1]; q++) {
          func.push_back(e.uq_func[q]);
          cg.push_back(e.uq_cg[q]);
        }
        off.push_back((int) func.size());
      };
      if (e.rank[r] == 1) {    // into the pair's h_p
        if (m.has_uq) {
          o.r1_mu.push_back(e.mus[q0]);
          o.r1_n.push_back(e.ns[q0] - 1);
          uq_row(o.r1_off, o.r1_func, o.r1_cg);
        }
        const int muj = e.mus[q0];
        std::vector<double> &h = hc[(size_t) mu * nel + muj];
        const int nbj = m.bond(mu, muj).nradbase;
        for (int p = 0; p < e.ndensity; p++) h[(size_t) p * nbj + e.ns[q0] - 1] += c[p];
        continue;
      }
      for (int t = 0; t < e.rank[r]; t++) {
        const int mu_t = e.mus[q0 + t], n0 = e.ns[q0 + t] - 1, l = e.ls[q0 + t], mm = e.ms[q0 + t];
        o.fidx.push_back((int) a(mu_t, n0, l, mm));
        o.ftgt.push_back((int) a(mu_t, n0, l, std::abs(mm)));
        const double sg = mm % 2 == 0 ? 1.0 : -1.0;
        o.fre.push_back(mm > 0 ? 1.0 : mm == 0 ? 2.0 : sg);
        o.fim.push_back(mm > 0 ? 1.0 : mm == 0 ? 0.0 : -sg);
      }
      o.off.push_back((int) o.fidx.size());
      o.c.insert(o.c.end(), c, c + e.ndensity);
      uq_row(o.uq_off, o.uq_func, o.uq_cg);
    }
  }
  alm.assign((size_t) (lmax + 1) * (lmax + 1), 0.0);
  blm.assign(alm.size(), 0.0);
  cl.assign(lmax + 1, 0.0);
  dl.assign(lmax + 1, 0.0);
  for (int l = 1; l <= lmax; l++) {
    const double lsq = l * l, ld = 2 * l, l1 = 4 * lsq - 1, l2 = lsq - ld + 1;
    for (int m = 0; m < l - 1; m++) {
      const double msq = m * m;
      alm[l * (lmax + 1) + m] = std::sqrt(l1 / (lsq - msq));
      blm[l * (lmax + 1) + m] = -std::sqrt((l2 - msq) / (4 * l2 - 1));
    }
    cl[l] = -std::sqrt(1.0 + 0.5 / l);
    dl[l] = std::sqrt(2.0 * (l - 1) + 3);
  }
  if (spline > 0) {
    spl.resize((size_t) nel * nel);
    for (int i = 0; i < nel; i++)
      for (int j = 0; j < nel; j++)
        if (m.bond(i, j).defined)
          spl[(size_t) i * nel + j].build(m.bond(i, j), m.inner_cutoff_type, std::max(1, spline / 2), spline,
                                          kPaceExactBelow * m.bond(i, j).rcut, kPaceTableSplit * m.bond(i, j).rcut,
                                          hc[(size_t) i * nel + j].data(), m.elements[i].ndensity);
  }
}

}    // namespace grace

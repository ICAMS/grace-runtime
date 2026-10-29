// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// PACE CPU driver: OpenMP over groups of 8 atoms of one element; per atom the neighbour work (radial tables, Y_lm, the
// A-basis, the forces) in SIMD across its neighbours, per group the C-tilde products and their adjoint in SIMD across
// the 8 atoms. In fp64: in fp32 the products and their adjoint miss the gate by up to 1e-3 eV/atom (large terms
// cancel in the densities). The evaluation follows
// ACECTildeEvaluator::compute_atom of lammps-user-pace (ML-PACE/ace-evaluator/ace_evaluator.cpp) and its spherical
// harmonics (ace_spherical_cart.cpp, Y_00 = 1), embedding (ace_abstract_basis.cpp) and radial functions
// (grace/pace_radial.h).
#include "grace/pace_cpu.h"
#include "grace/bonds.h"    // throw_zero_bond
#include "grace/cpu_simd.h"
#include "grace/fused_cpu.h"    // add_vatom
#include "grace/pace_embedding.h"
#include "grace/pace_radial.h"
#include "grace/type_map.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace grace {

namespace {

// GRACE_PACE_PROF=1 (environment): wall time per stage on OpenMP thread 0, printed to stderr after each compute()
enum { P_NEIGH, P_PROD, P_EMB, P_ADJ, P_FORCE, P_N };
const char *const P_NAME[P_N] = {"neighbours (radial, Ylm, A)", "products", "embedding", "adjoint", "forces"};
struct Prof {
  bool on = false;
  double t[P_N] = {}, t0 = 0.0;
  void start() { if (on) t0 = now(); }
  void lap(int k) { if (on) { const double t1 = now(); t[k] += t1 - t0; t0 = t1; } }
  static double now()
  {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return 0.0;
#endif
  }
};

inline double hsum(simd::d8 v)
{
  alignas(64) double t[8];
  simd::store(t, v);
  return ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
}

// the bonds of central atom i: f(j, r_ij) with r_ij = x_j - x_i
template<class Src> struct Bonds;
template<> struct Bonds<CsrBonds> {
  template<class F> static void each(const CsrBonds &s, int i, F &&f)
  {
    double d[3];
    for (int64_t q = s.offset[i]; q < s.offset[i + 1]; q++) f(s.j[q], s.bond(i, q, d));
  }
};
template<> struct Bonds<LammpsBonds> {
  template<class F> static void each(const LammpsBonds &s, int i, F &&f)
  {
    const int *jl = s.firstneigh[i];
    const double *xi = s.x[i];
    for (int jj = 0; jj < s.numneigh[i]; jj++) {
      const int j = jl[jj] & s.neighmask;
      const double d[3] = {s.x[j][0] - xi[0], s.x[j][1] - xi[1], s.x[j][2] - xi[2]};
      f(j, d);
    }
  }
};

// complex Y_lm (m >= 0, Y_00 = 1) and their gradients on the unit sphere (as ACECartesianSphericalHarmonics)
struct Ylm {
  int lmax = 0;
  const std::vector<double> &alm, &blm, &cl, &dl;    // PaceTables: [l * (lmax + 1) + m], [l]
  explicit Ylm(const PaceTables &t) : lmax(t.lmax), alm(t.alm), blm(t.blm), cl(t.cl), dl(t.dl) {}
  static int idx(int l, int m) { return l * (l + 1) / 2 + m; }    // m >= 0
  static int size(int lmax) { return (lmax + 1) * (lmax + 2) / 2; }
  // Y_lm of 8 directions at once (SIMD over neighbours): y*[q * ld], dy*[(3 q + d) * ld] (q = idx(l, m)), offset by
  // the caller
  void compute8(simd::d8 rx, simd::d8 ry, simd::d8 rz, double *yre, double *yim, double *dyre, double *dyim,
                size_t ld) const
  {
    using simd::d8;
    using simd::store;
    const double sq3 = 1.73205080756887719318, sq3o2 = 1.22474487139158894067;    // sqrt(3), sqrt(3 / 2)
    d8 P[(kPaceMaxLmax + 1) * (kPaceMaxLmax + 2) / 2], dP[(kPaceMaxLmax + 1) * (kPaceMaxLmax + 2) / 2];
    const d8 zero = rx * 0.0, one = zero + 1.0;
    P[idx(0, 0)] = one;
    dP[idx(0, 0)] = zero;
    if (lmax > 0) {
      P[idx(1, 0)] = sq3 * rz;
      dP[idx(1, 0)] = zero + sq3;
      P[idx(1, 1)] = zero + (-sq3o2);    // d8 - double is not defined for the AVX2 halves (cpu_simd.h)
      dP[idx(1, 1)] = zero;
      for (int l = 2; l <= lmax; l++) {
        for (int m = 0; m < l - 1; m++) {
          const double a = alm[l * (lmax + 1) + m], b = blm[l * (lmax + 1) + m];
          P[idx(l, m)] = a * (rz * P[idx(l - 1, m)] + b * P[idx(l - 2, m)]);
          dP[idx(l, m)] = a * (P[idx(l - 1, m)] + rz * dP[idx(l - 1, m)] + b * dP[idx(l - 2, m)]);
        }
        const d8 t = dl[l] * P[idx(l - 1, l - 1)];
        P[idx(l, l - 1)] = t * rz;
        dP[idx(l, l - 1)] = t;
        P[idx(l, l)] = cl[l] * P[idx(l - 1, l - 1)];
        dP[idx(l, l)] = zero;
      }
    }
    auto put = [&](int q, d8 yr, d8 yi, d8 x0r, d8 x0i, d8 x1r, d8 x1i, d8 x2r, d8 x2i) {
      store(yre + q * ld, yr);
      store(yim + q * ld, yi);
      store(dyre + (3 * q + 0) * ld, x0r);
      store(dyim + (3 * q + 0) * ld, x0i);
      store(dyre + (3 * q + 1) * ld, x1r);
      store(dyim + (3 * q + 1) * ld, x1i);
      store(dyre + (3 * q + 2) * ld, x2r);
      store(dyim + (3 * q + 2) * ld, x2i);
    };
    for (int l = 0; l <= lmax; l++) {    // m = 0
      const d8 p = P[idx(l, 0)], dp = dP[idx(l, 0)], rdy = dp * rz;
      put(idx(l, 0), p, zero, zero - rdy * rx, zero, zero - rdy * ry, zero, dp - rdy * rz, zero);
    }
    for (int l = 1; l <= lmax; l++) {    // m = 1
      const d8 p = P[idx(l, 1)], dp = dP[idx(l, 1)];
      const d8 dzr = rx * dp, dzi = ry * dp;
      const d8 rr = rx * p + rz * dzr, ri = ry * p + rz * dzi;
      put(idx(l, 1), rx * p, ry * p, p - rr * rx, zero - ri * rx, zero - rr * ry, p - ri * ry, dzr - rr * rz,
          dzi - ri * rz);
    }
    d8 pr = rx, pi = ry;    // phase^m
    for (int m = 2; m <= lmax; m++) {
      const d8 mr = pr * (double) m, mi = pi * (double) m;
      const d8 nr = pr * rx - pi * ry;
      pi = pr * ry + pi * rx;
      pr = nr;
      for (int l = m; l <= lmax; l++) {
        const d8 p = P[idx(l, m)], dp = dP[idx(l, m)];
        const d8 xr = mr * p, xi = mi * p;      // dyx
        const d8 yr_ = zero - xi, yi_ = xr;     // dyy
        const d8 zr = pr * dp, zi = pi * dp;    // dyz
        const d8 rr = rx * xr + ry * yr_ + rz * zr, ri = rx * xi + ry * yi_ + rz * zi;
        put(idx(l, m), pr * p, pi * p, xr - rr * rx, xi - ri * rx, yr_ - rr * ry, yi_ - ri * ry, zr - rr * rz,
            zi - ri * rz);
      }
    }
  }
};

}    // namespace

// atoms per group: the products of one element's atoms run in SIMD across them (8 doubles: one AVX-512 register)
constexpr int kLanes = 8;

struct PaceCpu::Work {
  int zero_i = -1, zero_j = -1;    // the first bond of length 0 its atoms met (throw_zero_bond)
  struct Atom {    // one atom of the group: its neighbours inside the pair cutoffs, by element, as arrays over them
    int i = -1;
    int np = 0;                                  // neighbours incl. padding: element blocks of multiples of 8
    std::vector<int> nj;                         // atom index of each neighbour, -1: padding
    std::vector<std::array<int, 3>> blocks;      // (element, first, count incl. padding)
    std::vector<double> nr, rx, ry, rz;          // [neighbour]
    std::vector<double> Rv, Rd;                  // [f][neighbour]: h_p, R_nl (pair layout), core repulsion (f = nf - 1)
    std::vector<double> Yre, Yim, dYre, dYim;    // [q][neighbour], [3 q + d][neighbour]
    double rho_core = 0.0, dmin = 0.0, dF_drho_core = 0.0, dF_dfcut = 0.0;
    double delta_in = 0.0;    // ZBL: delta_in of the closest neighbour's pair
    double rho1[kPaceMaxDensity];    // the rank-1 part of the densities
    int jjmin = -1;
  } at[kLanes];
  std::vector<double> Are, Aim, Wre, Wim;    // [A index][lane]
  std::vector<double> are, aim;              // one atom's A-basis, contiguous
  std::vector<double> ev, ed;                // [lane][nf]: radial blocks evaluated directly
  std::vector<double> uv, ua, r1;            // UQ: projections [function][lane], gamma sums [k][lane], rank-1 rows
  double vir[6] = {0, 0, 0, 0, 0, 0}, energy = 0.0;
};

PaceCpu::PaceCpu(std::shared_ptr<const PaceModel> model, int spline, Logger log) :
    m_(std::move(model)), log_(log), lay_(std::make_unique<PaceTables>(*m_, spline))
{
}

PaceCpu::~PaceCpu() = default;

void PaceCpu::setup_types(int ntypes, const std::vector<int> &type2mu)
{
  check_type2mu(type2mu, ntypes, m_->n_elements, "PACE");
  ntypes_ = ntypes;
  type2mu_ = type2mu;
}

double PaceCpu::cutoff(int t1, int t2) const
{
  const int a = type2mu_.at(t1), b = type2mu_.at(t2);
  if (a < 0 || b < 0) return 0.0;
  return m_->cutoff(a, b);
}

template<class Src>
void PaceCpu::compute(int inum, const int *ilist, int n_nodes, int nmax_nodes, const Src &bonds, Outputs &out,
                      const ComputeFlags &flags, int nthreads)
{
  using simd::d8;
  using simd::load;
  using simd::store;
  const PaceModel &m = *m_;
  const PaceTables &L = *lay_;
  const bool forces = !flags.energy_only;
  const Ylm ylm(L);
  const int nyl = Ylm::size(L.lmax);
  const int ict = m.inner_cutoff_type;
  const bool zbl = ict == kInnerZbl;
  double cutmax = 0.0;
  for (const PaceBond &b : m.bonds)
    if (b.defined) cutmax = std::max(cutmax, b.rcut);
#ifdef _OPENMP
  const int nth = std::max(1, nthreads);
#else
  const int nth = 1;
  (void) nthreads;
#endif
  if (flags.couple.eq) throw Error("PACE: two-group coupling (pair_modify decouple) is not supported");
  const bool uq = flags.uq;
  if (uq && !m.has_uq) throw Error("PACE: the model has no active set (.asi) for the extrapolation grade");
  if (uq && inum > 0 && !out.gamma) throw Error("PACE: the extrapolation grade requested without the gamma output");
  const bool vatom = forces && flags.vflag_atom && out.vatom;
  const auto central = [&](int q) { return ilist ? ilist[q] : q; };
  for (int q = 0; q < inum; q++) {    // before the parallel region: an exception must not leave it
    const int i = central(q);
    if (bonds.type[i] < 1 || bonds.type[i] > ntypes_ || type2mu_[bonds.type[i]] < 0)
      throw Error("PACE: atom " + std::to_string(i) + ": its type has no element of the model");
  }
  // groups of up to kLanes atoms of one element
  std::vector<int> ord(inum);
  std::vector<std::pair<int, int>> groups;    // (start in ord, count)
  {
    std::vector<std::vector<int>> by(L.nel);
    for (int q = 0; q < inum; q++) by[type2mu_[bonds.type[central(q)]]].push_back(central(q));
    int q = 0;
    for (auto &v : by)
      for (size_t g = 0; g < v.size(); g += kLanes) {
        const int c = (int) std::min<size_t>(kLanes, v.size() - g);
        for (int k = 0; k < c; k++) ord[q + k] = v[g + k];
        groups.emplace_back(q, c);
        q += c;
      }
  }
  if ((int) works_.size() != nth) {    // kept between computes: the buffers grow to the largest neighbourhoods once
    works_.clear();
    works_.resize(nth);
    for (Work &w : works_) {
      w.Are.resize(L.na * kLanes);
      w.Aim.resize(L.na * kLanes);
      w.Wre.resize(L.na * kLanes);
      w.Wim.resize(L.na * kLanes);
      w.are.resize(L.na);
      w.aim.resize(L.na);
      w.ev.resize((size_t) kLanes * L.nf);
      w.ed.resize((size_t) kLanes * L.nf);
      if (m.has_uq) {
        size_t nf = 0, n = 0, n1 = 0;
        for (int mu = 0; mu < L.nel; mu++) {
          const PaceElement &e = m.elements[mu];
          nf = std::max(nf, (size_t) e.uq_nfunc);
          n = std::max(n, (size_t) e.uq_nfunc * (m.uq_grade ? e.ndensity : 1));
          n1 = std::max(n1, L.el[mu].r1_mu.size());
        }
        w.uv.resize(std::max<size_t>(nf, 1) * kLanes);
        w.ua.resize(std::max<size_t>(n, 1) * kLanes);
        w.r1.resize(std::max<size_t>(n1, 1) * kLanes);
      }
    }
  }
  for (Work &w : works_) {
    w.energy = 0.0;
    std::fill(w.vir, w.vir + 6, 0.0);
  }
  if (forces) fthr_.ensure(nth, std::max(nmax_nodes, std::max(n_nodes, 1)));
  static const bool prof_env = [] { const char *e = std::getenv("GRACE_PACE_PROF"); return e && e[0] == '1'; }();
  Prof prof;
  prof.on = prof_env;
  const int ngroups = (int) groups.size();

#pragma omp parallel for num_threads(nth) schedule(dynamic, 4)
  for (int g = 0; g < ngroups; g++) {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
    Prof *pf = tid == 0 ? &prof : nullptr;
#else
    const int tid = 0;
    Prof *pf = &prof;
#endif
    Work &w = works_[tid];
    double *ft = forces ? fthr_.of(tid) : nullptr;    // this thread's forces [n_nodes][3]
    if (pf) pf->start();
    const int g0 = groups[g].first, nl = groups[g].second;
    const int mui = type2mu_[bonds.type[ord[g0]]];
    const auto &E = L.el[mui];
    const int nd = E.nd;
    const PaceElement &pe = m.elements[mui];
    std::fill(w.Are.begin(), w.Are.end(), 0.0);
    std::fill(w.Aim.begin(), w.Aim.end(), 0.0);
    // per atom: neighbours by element, radial functions, Y_lm, the A-basis (SIMD over neighbours) into its lane
    for (int ln = 0; ln < nl; ln++) {
      Work::Atom &a = w.at[ln];
      const int i = a.i = ord[g0 + ln];
      int cnt[kPaceMaxElements] = {0};
      Bonds<Src>::each(bonds, i, [&](int j, const double *) {
        const int muj = type2mu_[bonds.type[j]];    // -1: a type without an element (pair_style hybrid)
        if (muj >= 0) cnt[muj]++;
      });
      int start[kPaceMaxElements];
      a.blocks.clear();
      a.np = 0;
      for (int mu = 0; mu < L.nel; mu++) {    // an upper bound per element; trimmed below
        start[mu] = a.np;
        a.np += (cnt[mu] + kLanes - 1) / kLanes * kLanes;
      }
      a.nj.assign(a.np, -1);
      a.nr.assign(a.np, 1.0);
      a.rx.assign(a.np, 0.0);
      a.ry.assign(a.np, 0.0);
      a.rz.assign(a.np, 1.0);
      int fill[kPaceMaxElements];
      for (int mu = 0; mu < L.nel; mu++) fill[mu] = start[mu];
      Bonds<Src>::each(bonds, i, [&](int j, const double *d) {
        const int muj = type2mu_[bonds.type[j]];
        if (muj < 0) return;
        const PaceBond &b = m.bond(mui, muj);
        if (!b.defined) return;
        const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (r >= b.rcut) return;
        if (r == 0.0) {    // not physical: an error after the parallel region (throw_zero_bond)
          if (w.zero_i < 0) { w.zero_i = i; w.zero_j = j; }
          return;
        }
        const int k = fill[muj]++;
        a.nj[k] = j;
        a.nr[k] = r;
        a.rx[k] = d[0] / r;
        a.ry[k] = d[1] / r;
        a.rz[k] = d[2] / r;
      });
      for (int mu = 0; mu < L.nel; mu++) {
        const int c = (fill[mu] - start[mu] + kLanes - 1) / kLanes * kLanes;
        if (c > 0) a.blocks.push_back({mu, start[mu], c});
      }
      const size_t np = a.np;
      a.Rv.resize((size_t) L.nf * np);    // every column of a block is written below (padding: zeros)
      a.Rd.resize((size_t) L.nf * np);
      a.Yre.resize(nyl * np);
      a.Yim.resize(nyl * np);
      a.dYre.resize(3 * nyl * np);
      a.dYim.resize(3 * nyl * np);
      a.rho_core = 0.0;
      a.dmin = cutmax;
      a.jjmin = -1;
      std::fill(w.are.begin(), w.are.end(), 0.0);
      std::fill(w.aim.begin(), w.aim.end(), 0.0);
      for (int p = 0; p < nd; p++) a.rho1[p] = 0.0;
      for (const auto &blk : a.blocks) {
        const int muj = blk[0], b0 = blk[1], bn = blk[2];
        const PaceBond &b = m.bond(mui, muj);
        const int bl1 = b.lmax + 1, qc = nd + b.nradmax * bl1;    // block: h_p, R_nl, core repulsion
        const double *hc = L.hc[(size_t) mui * L.nel + muj].data();
        // radial functions, 8 neighbours at a time: from the table (one function's 8 values a cache line) or, at
        // close range, directly
        for (int k0 = b0; k0 < b0 + bn; k0 += kLanes) {
          const double *cf[kLanes];
          double x[kLanes], ih[kLanes];
          const auto ev = [&](int t) { return w.ev.data() + (size_t) t * L.nf; };
          const auto ed = [&](int t) { return w.ed.data() + (size_t) t * L.nf; };
          for (int t = 0; t < kLanes; t++) {
            const int k = k0 + t;
            cf[t] = nullptr;
            x[t] = ih[t] = 0.0;
            if (a.nj[k] < 0) {    // padding
              std::fill(ev(t), ev(t) + qc + 1, 0.0);
              std::fill(ed(t), ed(t) + qc + 1, 0.0);
              continue;
            }
            const double r = a.nr[k];
            if (zbl) {
              const double d = r - (b.r_in - b.delta_in);
              if (d < a.dmin) {
                a.dmin = d;
                a.jjmin = k;
                a.delta_in = b.delta_in;
              }
            }
            if (L.spl.empty() || r < kPaceExactBelow * b.rcut)
              pace::radial_h(b, ict, r, hc, nd, ev(t), ed(t));
            else
              cf[t] = L.spl[(size_t) mui * L.nel + muj].at(r, x[t], ih[t]);
          }
          for (int f = 0; f <= qc; f++) {
            double *rv = &a.Rv[(size_t) (f < qc ? f : L.nf - 1) * np + k0], *rd = &a.Rd[(size_t) (f < qc ? f : L.nf - 1) * np + k0];
            for (int t = 0; t < kLanes; t++) {
              if (cf[t]) {
                const double *c0 = cf[t] + f, xx = x[t];
                const double c1 = c0[qc + 1], c2 = c0[2 * (qc + 1)], c3 = c0[3 * (qc + 1)];
                rv[t] = c0[0] + xx * (c1 + xx * (c2 + xx * c3));
                rd[t] = (c1 + xx * (2 * c2 + 3 * xx * c3)) * ih[t];
              } else {
                rv[t] = ev(t)[f];
                rd[t] = ed(t)[f];
              }
            }
          }
        }
        for (int k = b0; k < b0 + bn; k++) a.rho_core += a.Rv[(size_t) (L.nf - 1) * np + k];
        for (int k = b0; k < b0 + bn; k += kLanes)    // Y_lm, SIMD over the neighbours
          ylm.compute8(load(&a.rx[k]), load(&a.ry[k]), load(&a.rz[k]), &a.Yre[k], &a.Yim[k], &a.dYre[k],
                       &a.dYim[k], np);
        // A_{mu_j n l m} = sum_j R_nl(r_j) Y_lm(rhat_j): reductions over the block
        for (int p = 0; p < nd; p++) {    // rank 1: sum_j h_p(r_j)
          const double *h = &a.Rv[(size_t) p * np + b0];
          d8 sh = load(h);
          for (int k = kLanes; k < bn; k += kLanes) sh += load(h + k);
          a.rho1[p] += hsum(sh);
        }
        for (int nq = 0; nq < b.nradmax; nq++)
          for (int l = 0; l < bl1; l++) {
            const double *R = &a.Rv[(size_t) (nd + nq * bl1 + l) * np + b0];
            const size_t q0 = L.a(muj, nq, l, 0);
            for (int mm = 0; mm <= l; mm++) {
              const double *yr = &a.Yre[(size_t) Ylm::idx(l, mm) * np + b0], *yi = &a.Yim[(size_t) Ylm::idx(l, mm) * np + b0];
              d8 sr = load(R) * load(yr), si = load(R) * load(yi);
              for (int k = kLanes; k < bn; k += kLanes) {
                const d8 rk = load(R + k);
                sr += rk * load(yr + k);
                si += rk * load(yi + k);
              }
              w.are[q0 + mm] += hsum(sr);
              w.aim[q0 + mm] += hsum(si);
            }
          }
      }
      if (uq) {    // the rank-1 rows of the extrapolation grade: sum_j g_n(r_ij) of the neighbours of their element
        const auto &r1mu = E.r1_mu, &r1n = E.r1_n;
        for (size_t q = 0; q < r1mu.size(); q++) w.r1[q * kLanes + ln] = 0.0;
        for (const auto &blk : a.blocks) {
          const int muj = blk[0], b0 = blk[1], bn = blk[2];
          const PaceBond &b = m.bond(mui, muj);
          double g[kPaceMaxNradbase], dg[kPaceMaxNradbase];
          for (int k = b0; k < b0 + bn; k++) {
            if (a.nj[k] < 0) continue;
            pace::radbase(b, ict, a.nr[k], g, dg);
            for (size_t q = 0; q < r1mu.size(); q++)
              if (r1mu[q] == muj) w.r1[q * kLanes + ln] += g[r1n[q]];
          }
        }
      }
      for (size_t q = 0; q < L.na; q++) {    // into the atom's lane
        w.Are[q * kLanes + ln] = w.are[q];
        w.Aim[q * kLanes + ln] = w.aim[q];
      }
    }
    if (pf) pf->lap(P_NEIGH);
    for (int mu = 0; mu < L.nel; mu++)    // A_{l,-m} = (-1)^m conj(A_{l,m}), all lanes
      for (int nq = 0; nq < L.nr; nq++)
        for (int l = 0; l <= L.lmax; l++)
          for (int mm = 1; mm <= l; mm++) {
            const size_t p = L.a(mu, nq, l, mm) * kLanes, q = L.a(mu, nq, l, -mm) * kLanes;
            const double sg = mm % 2 == 0 ? 1.0 : -1.0;
            store(&w.Are[q], sg * load(&w.Are[p]));
            store(&w.Aim[q], -sg * load(&w.Aim[p]));
          }
    // densities: rank 1, then the C-tilde products in SIMD across the lanes
    d8 rho[kPaceMaxDensity];
    for (int p = 0; p < nd; p++) {
      alignas(64) double t[kLanes] = {0};
      for (int ln = 0; ln < nl; ln++) t[ln] = w.at[ln].rho1[p];
      rho[p] = load(t);
    }
    const int nrows = (int) E.off.size() - 1;
    if (uq) std::fill(w.uv.begin(), w.uv.end(), 0.0);
    for (int k = 0; k < nrows; k++) {
      const int q0 = E.off[k], q1 = E.off[k + 1];
      d8 br = load(&w.Are[(size_t) E.fidx[q0] * kLanes]), bi = load(&w.Aim[(size_t) E.fidx[q0] * kLanes]);
      for (int q = q0 + 1; q < q1; q++) {
        const d8 ar = load(&w.Are[(size_t) E.fidx[q] * kLanes]), ai = load(&w.Aim[(size_t) E.fidx[q] * kLanes]);
        const d8 t = br * ar - bi * ai;
        bi = br * ai + bi * ar;
        br = t;
      }
      for (int p = 0; p < nd; p++) rho[p] += E.c[(size_t) k * nd + p] * br;
      if (uq)    // projections: Re of the row's product times the generalized CG of each of its B functions
        for (int q = E.uq_off[k]; q < E.uq_off[k + 1]; q++) {
          double *v = &w.uv[(size_t) E.uq_func[q] * kLanes];
          store(v, load(v) + E.uq_cg[q] * br);
        }
    }
    if (pf) pf->lap(P_PROD);
    // embedding, inner cutoff / core repulsion, per atom
    alignas(64) double dF[kPaceMaxDensity][kLanes], dFfs[kPaceMaxDensity][kLanes];
    for (int ln = 0; ln < kLanes; ln++)
      for (int p = 0; p < nd; p++) dF[p][ln] = dFfs[p][ln] = 0.0;
    for (int ln = 0; ln < nl; ln++) {
      Work::Atom &a = w.at[ln];
      double rl[kPaceMaxDensity], dl[kPaceMaxDensity], dfs[kPaceMaxDensity];
      for (int p = 0; p < nd; p++) rl[p] = rho[p][ln];
      const pace::Embedded em = pace::embed(pe.npoti, pe.fs_parameters.data(), nd, rl, dl, a.rho_core, pe.rho_core_cut,
                                            pe.drho_core_cut, zbl, a.jjmin != -1, a.dmin, a.delta_in, dfs);
      for (int p = 0; p < nd; p++) {
        dF[p][ln] = dl[p];
        dFfs[p][ln] = dfs[p];
      }
      a.dF_drho_core = em.dF_drho_core;
      a.dF_dfcut = em.dF_dfcut;
      const double ecut = em.energy;
      w.energy += ecut;
      if (flags.eflag_atom && out.eatom) out.eatom[a.i] += ecut;    // own index: no other thread writes it
    }
    if (pf) pf->lap(P_EMB);
    if (uq) {    // gamma_i = max_k |sum_j u_j A[j][k]|, u: the projections (linear) or dF/drho_p times them, [p][f]
      const int nfun = pe.uq_nfunc;
      for (size_t q = 0; q < E.r1_mu.size(); q++)
        for (int t = E.r1_off[q]; t < E.r1_off[q + 1]; t++) {
          double *v = &w.uv[(size_t) E.r1_func[t] * kLanes];
          store(v, load(v) + E.r1_cg[t] * load(&w.r1[q * kLanes]));
        }
      const int np = m.uq_grade ? nd : 1, n = nfun * np;
      const double *A = pe.asi.data();
      std::fill(w.ua.begin(), w.ua.begin() + (size_t) n * kLanes, 0.0);
      for (int p = 0; p < np; p++)
        for (int f = 0; f < nfun; f++) {
          d8 u = load(&w.uv[(size_t) f * kLanes]);
          if (m.uq_grade) u = u * load(&dFfs[p][0]);
          const double *row = A + (size_t) (p * nfun + f) * n;
          for (int k = 0; k < n; k++) {
            double *acc = &w.ua[(size_t) k * kLanes];
            store(acc, load(acc) + row[k] * u);
          }
        }
      alignas(64) double gmax[kLanes] = {0};
      for (int k = 0; k < n; k++)
        for (int ln = 0; ln < kLanes; ln++) gmax[ln] = std::max(gmax[ln], std::abs(w.ua[(size_t) k * kLanes + ln]));
      for (int ln = 0; ln < nl; ln++) out.gamma[w.at[ln].i] = gmax[ln];
    }
    if (!forces) continue;
    // adjoint dE/dA in SIMD across the lanes: the forward products again, then backwards
    std::fill(w.Wre.begin(), w.Wre.end(), 0.0);
    std::fill(w.Wim.begin(), w.Wim.end(), 0.0);
    d8 fr_[kPaceMaxRank + 1], fi_[kPaceMaxRank + 1];
    for (int k = 0; k < nrows; k++) {
      d8 th = load(&dF[0][0]) * 0.0;
      for (int p = 0; p < nd; p++) th += E.c[(size_t) k * nd + p] * load(&dF[p][0]);
      th = 0.5 * th;
      const int q0 = E.off[k], r = E.off[k + 1] - q0;
      fr_[0] = th;    // theta times the products of the factors before t
      fi_[0] = th * 0.0;
      for (int t = 0; t < r - 1; t++) {
        const size_t q = (size_t) E.fidx[q0 + t] * kLanes;
        const d8 ar = load(&w.Are[q]), ai = load(&w.Aim[q]);
        fr_[t + 1] = fr_[t] * ar - fi_[t] * ai;
        fi_[t + 1] = fr_[t] * ai + fi_[t] * ar;
      }
      d8 br = th * 0.0 + 1.0, bi = th * 0.0;    // products of the factors after t
      for (int t = r - 1; t >= 0; t--) {
        const d8 gr = fr_[t] * br - fi_[t] * bi, gi = fr_[t] * bi + fi_[t] * br;    // theta dB_t
        const int qq = q0 + t;
        double *wr = &w.Wre[(size_t) E.ftgt[qq] * kLanes], *wi = &w.Wim[(size_t) E.ftgt[qq] * kLanes];
        store(wr, load(wr) + E.fre[qq] * gr);
        store(wi, load(wi) + E.fim[qq] * gi);
        if (t > 0) {
          const size_t q = (size_t) E.fidx[qq] * kLanes;
          const d8 ar = load(&w.Are[q]), ai = load(&w.Aim[q]);
          const d8 nb_ = br * ar - bi * ai;
          bi = br * ai + bi * ar;
          br = nb_;
        }
      }
    }
    if (pf) pf->lap(P_ADJ);
    // forces on the bonds, per atom, SIMD over its neighbours
    for (int ln = 0; ln < nl; ln++) {
      const Work::Atom &a = w.at[ln];
      const int i = a.i;
      const size_t np = a.np;
      for (size_t q = 0; q < L.na; q++) {    // the atom's dE/dA, contiguous
        w.are[q] = w.Wre[q * kLanes + ln];
        w.aim[q] = w.Wim[q * kLanes + ln];
      }
      for (const auto &blk : a.blocks) {
        const int muj = blk[0], b0 = blk[1], bn = blk[2];
        const PaceBond &b = m.bond(mui, muj);
        const int bl1 = b.lmax + 1;
        for (int k = b0; k < b0 + bn; k += kLanes) {
          // rad = dE/dr along rhat, ang = the tangential part times r:
          // sum_p dE/drho_p dh_p + sum_lm c_m (Re(S1 Y) rhat + Re(S2 dY) / r), S1 = sum_n W_nlm dR_nl,
          // S2 = sum_n W_nlm R_nl
          d8 rad = load(&a.Rd[(size_t) (L.nf - 1) * np + k]) * a.dF_drho_core;
          for (int p = 0; p < nd; p++) rad += dF[p][ln] * load(&a.Rd[(size_t) p * np + k]);    // rank 1
          const d8 zero = rad * 0.0;
          d8 ang0 = zero, ang1 = zero, ang2 = zero;
          for (int l = 0; l < bl1; l++)
            for (int mm = 0; mm <= l; mm++) {
              d8 s1r = zero, s1i = zero, s2r = zero, s2i = zero;
              for (int nq = 0; nq < b.nradmax; nq++) {
                const size_t q = L.a(muj, nq, l, mm);
                const double wr = w.are[q], wi = w.aim[q];
                const size_t f = (size_t) (nd + nq * bl1 + l) * np + k;
                const d8 R = load(&a.Rv[f]), dR = load(&a.Rd[f]);
                s1r += wr * dR;
                s1i += wi * dR;
                s2r += wr * R;
                s2i += wi * R;
              }
              const double c = mm > 0 ? 2.0 : 1.0;
              const size_t q = (size_t) Ylm::idx(l, mm);
              rad += c * (s1r * load(&a.Yre[q * np + k]) - s1i * load(&a.Yim[q * np + k]));
              ang0 += c * (s2r * load(&a.dYre[(3 * q + 0) * np + k]) - s2i * load(&a.dYim[(3 * q + 0) * np + k]));
              ang1 += c * (s2r * load(&a.dYre[(3 * q + 1) * np + k]) - s2i * load(&a.dYim[(3 * q + 1) * np + k]));
              ang2 += c * (s2r * load(&a.dYre[(3 * q + 2) * np + k]) - s2i * load(&a.dYim[(3 * q + 2) * np + k]));
            }
          alignas(64) double rd_[kLanes], a0_[kLanes], a1_[kLanes], a2_[kLanes];
          store(rd_, rad);
          store(a0_, ang0);
          store(a1_, ang1);
          store(a2_, ang2);
          for (int t = 0; t < kLanes; t++) {
            const int kk = k + t, j = a.nj[kk];
            if (j < 0) continue;
            const double rr = rd_[t] + (zbl && kk == a.jjmin ? a.dF_dfcut : 0.0), inv_r = 1.0 / a.nr[kk];
            const double rh[3] = {a.rx[kk], a.ry[kk], a.rz[kk]};
            const double f[3] = {rr * rh[0] + a0_[t] * inv_r, rr * rh[1] + a1_[t] * inv_r, rr * rh[2] + a2_[t] * inv_r};
            for (int d = 0; d < 3; d++) {
              ft[3 * (size_t) i + d] += f[d];
              ft[3 * (size_t) j + d] -= f[d];
            }
            if (vatom) fused::add_vatom(out.vatom, i, j, rh, a.nr[kk], f);
            if (flags.virial) {    // del = x_i - x_j = -r rhat, f = force on i
              const double r = a.nr[kk], dx = -rh[0] * r, dy = -rh[1] * r, dz = -rh[2] * r;
              w.vir[0] += dx * f[0];
              w.vir[1] += dy * f[1];
              w.vir[2] += dz * f[2];
              w.vir[3] += dx * f[1];
              w.vir[4] += dx * f[2];
              w.vir[5] += dy * f[2];
            }
          }
        }
      }
    }
    if (pf) pf->lap(P_FORCE);
  }
  {
    std::vector<Work *> zero(works_.size());
    for (size_t t = 0; t < works_.size(); t++) zero[t] = &works_[t];
    throw_zero_bond(zero);
  }
  if (prof.on) {
    double tot = 0.0;
    for (double x : prof.t) tot += x;
    std::fprintf(stderr, "PACE profile (thread 0, %d atoms, %.3f s):", inum, tot);
    for (int k = 0; k < P_N; k++) std::fprintf(stderr, " %s %.1f%%", P_NAME[k], 100.0 * prof.t[k] / std::max(tot, 1e-30));
    std::fprintf(stderr, "\n");
  }
  for (const Work &w : works_) {
    out.energy += w.energy;
    if (forces && flags.virial)
      for (int k = 0; k < 6; k++) out.virial[k] += w.vir[k];
  }
  if (forces) fthr_.reduce(out.f, n_nodes);
}

template void PaceCpu::compute<CsrBonds>(int, const int *, int, int, const CsrBonds &, Outputs &, const ComputeFlags &,
                                         int);
template void PaceCpu::compute<LammpsBonds>(int, const int *, int, int, const LammpsBonds &, Outputs &,
                                            const ComputeFlags &, int);

}    // namespace grace

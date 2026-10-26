// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// DFT-D3 with Becke-Johnson (rational) damping, two-body terms (C6, C8) and the Axilrod-Teller-Muto three-body term;
// S. Grimme, J. Antony, S. Ehrlich, H. Krieg, J. Chem. Phys. 132, 154104 (2010); S. Grimme, S. Ehrlich, L. Goerigk,
// J. Comput. Chem. 32, 1456 (2011). Reference data: grace/d3_params.h. Eqs. as simple-dftd3, in eV and Angstrom:
//   CN_i = sum_j f(r_ij), f = 1 / (1 + exp(-k1 (rcov_ij / r - 1))), rcov_ij = 4/3 (R_i + R_j), r < cn_cutoff
//   w_ia = exp(-wf (CN_i - CNref_ia)^2) / sum_b (...); C6_ij = sum_ab w_ia w_jb C6ref_ab
//   e_ij = -C6_ij [s6 / (r^6 + R0^6) + s8 qq / (r^8 + R0^8)] sw(r), qq = 3 r4r2_i r4r2_j, R0 = a1 sqrt(qq) + a2,
//   sw = 1 below cutoff_on, x^3 (10 - 15 x + 6 x^2) with x = (cutoff - r) / (cutoff - cutoff_on) above
//   E = 1/2 sum_i sum_j e_ij (all periodic images), E_i = 1/2 sum_j e_ij
// Forces (the device kernels in gather form: each pass writes only to the centre atom i, every pair is visited from
// both sides; the CPU driver visits each pair once and adds the opposite share to j):
//   direct  F_i += de_ij/dr r_ij / r                       (C6 fixed)
//   CN      F_i += (dE/dCN_i + dE/dCN_j) df/dr r_ij / r,   dE/dCN_i = sum_j de_ij/dC6 dC6_ij/dCN_i
// virial W = -dE/dstrain (LAMMPS sign, xx yy zz xy xz yz): -g r_ij r_ij^T / r per pair term with dE/dr g.
// Three-body (s9 != 0; simple-dftd3 get_atm_dispersion with rs9 = 4/3, alp = 16): per triangle of atoms and images
// with all sides below atm_cutoff
//   E_ijk = s9 sqrt|C6_ij C6_ik C6_jk| / sqrt(Hartree) atm(u_ij, u_ik, u_jk)   (C9 = sqrt(C6^3) holds in atomic units)
// (atm below; u the squared sides); each vertex takes E/3 as its energy (as simple-dftd3). The drivers visit each
// triangle once, from its first vertex i in a total order of atoms and images over pairs (j, k) of i's neighbours.
// The formula functions are host and device code (CUDA / HIP); the pair precision is a template parameter, the
// per-atom quantities (CN, weights) are double. The CPU driver (OpenMP, half lists) is d3::compute_cpu.
#pragma once

#include "grace/cell_list.h"
#include "grace/d3_params.h"
#include "grace/dispersion.h"
#include "grace/hd.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace grace {
namespace d3 {

inline constexpr int kMaxRef = d3_data::kMaxRef;
inline constexpr double kK1 = 16.0;    // steepness of the counting function
inline constexpr double kWF = 4.0;     // Gaussian weighting factor of the reference coordination numbers

GRACE_HD float fexp(float x) { return ::expf(x); }
GRACE_HD double fexp(double x) { return ::exp(x); }
GRACE_HD float fsqrt(float x) { return ::sqrtf(x); }
GRACE_HD double fsqrt(double x) { return ::sqrt(x); }
GRACE_HD bool finite(double x) { return ::fabs(x) <= 1.7976931348623157e308; }    // false for NaN and inf

// counting function f(r) and df/dr; rcov = 4/3 (R_i + R_j)
template<class T> GRACE_HD T count(T rcov, T r, T &dfdr)
{
  const T e = fexp(-T(kK1) * (rcov / r - T(1)));
  const T f = T(1) / (T(1) + e);
  dfdr = -T(kK1) * rcov / (r * r) * f * f * e;    // -k1 rcov / r^2 f (1 - f)
  return f;
}

// reference weights w[a] and dw[a]/dCN of an atom with coordination number cn (entries nref..kMaxRef-1 set to 0); a
// non-finite weight becomes 1 for the reference with the largest CNref and 0 otherwise, a non-finite derivative 0
// (as simple-dftd3 for CN far from all references). dw is written as w (2 kWF dc - dnorm / norm), which does not
// underflow when norm is tiny (CN ~10 from every reference)
GRACE_HD void weights(double cn, const double *cnref, int nref, double *w, double *dw)
{
  double g[kMaxRef], norm = 0.0, dnorm = 0.0, cmax = -1.0;
  for (int a = 0; a < nref; a++) {
    const double dc = cnref[a] - cn;
    g[a] = ::exp(-kWF * dc * dc);
    norm += g[a];
    dnorm += 2.0 * kWF * dc * g[a];
    cmax = cnref[a] > cmax ? cnref[a] : cmax;
  }
  const double ninv = 1.0 / norm;
  for (int a = 0; a < kMaxRef; a++) {
    if (a >= nref) {
      w[a] = dw[a] = 0.0;
      continue;
    }
    double wa = g[a] * ninv;
    if (!finite(wa)) wa = cnref[a] == cmax ? 1.0 : 0.0;
    double da = wa * (2.0 * kWF * (cnref[a] - cn) - dnorm * ninv);
    if (!finite(da)) da = 0.0;
    w[a] = wa;
    dw[a] = da;
  }
}

// C6 of a pair from the weights of i (wi, dwi) and j (wj) and the reference block blk[a kMaxRef + b] (a: references
// of i, b: of j); dc6i = dC6/dCN_i
template<class T> GRACE_HD T c6(const T *wi, const T *dwi, int ni, const T *wj, int nj, const T *blk, T &dc6i)
{
  T c = T(0), d = T(0);
  for (int a = 0; a < ni; a++) {
    T s = T(0);
    for (int b = 0; b < nj; b++) s += blk[a * kMaxRef + b] * wj[b];
    c += wi[a] * s;
    d += dwi[a] * s;
  }
  dc6i = d;
  return c;
}

// the cutoff switch sw(r): 1 below ron, x^3 (10 - 15 x + 6 x^2) with x = (rc - r) / (rc - ron) above, 0 from rc;
// dsw/dr
template<class T> GRACE_HD T smooth(T r, T rc, T ron, T &dsw)
{
  dsw = T(0);
  if (!(r < rc)) return T(0);    // r rounded up to rc (fp32 on the device, sqrt), also for a hard cutoff (ron = rc)
  if (!(r > ron)) return T(1);
  const T wdt = rc - ron, x = (rc - r) / wdt;
  dsw = T(-30) * x * x * (T(1) - x) * (T(1) - x) / wdt;
  return x * x * x * (T(10) + x * (T(-15) + T(6) * x));
}

// C6 of a pair and its CN derivatives at both atoms (dc6i = dC6/dCN_i, dc6j = dC6/dCN_j); blk as c6
template<class T>
GRACE_HD T c6(const T *wi, const T *dwi, int ni, const T *wj, const T *dwj, int nj, const T *blk, T &dc6i, T &dc6j)
{
  T c = T(0), di = T(0), dj = T(0);
  for (int a = 0; a < ni; a++) {
    T s = T(0), sd = T(0);
    for (int b = 0; b < nj; b++) {
      s += blk[a * kMaxRef + b] * wj[b];
      sd += blk[a * kMaxRef + b] * dwj[b];
    }
    c += wi[a] * s;
    di += dwi[a] * s;
    dj += wi[a] * sd;
  }
  dc6i = di;
  dc6j = dj;
  return c;
}

// pair term per unit C6, g = -(s6 / (r^6 + r06) + s8 qq / (r^8 + r08)) sw(r), and dg/dr; r < rc
template<class T> GRACE_HD T pair(T r, T r06, T r08, T qq, T s6, T s8, T rc, T ron, T &dgdr)
{
  const T r2 = r * r, r6 = r2 * r2 * r2, r8 = r6 * r2;
  const T t6 = T(1) / (r6 + r06), t8 = T(1) / (r8 + r08);
  const T u = s6 * t6 + s8 * qq * t8;
  const T du = -(T(6) * s6 * r6 * t6 * t6 + T(8) * s8 * qq * r8 * t8 * t8) / r;
  T dsw;
  const T sw = smooth(r, rc, ron, dsw);
  dgdr = -(du * sw + u * dsw);
  return -u * sw;
}

// x^(-1/3) for normal x > 0 without libm's cbrt on the host (a guess from the float bits, 3.5e-2 relative, and four
// Newton steps: 5.6e-16, checked over 1e-3 .. 1e4; libm's cbrt was a fifth of the CPU three-body time); rcbrt on the
// device
inline double inv_cbrt_host(double x)
{
  const float xf = (float) x;
  uint32_t i;
  std::memcpy(&i, &xf, 4);
  i = 0x54a21d2au - i / 3u;
  float yf;
  std::memcpy(&yf, &i, 4);
  double y = yf;
  for (int k = 0; k < 4; k++) y = y * (4.0 - x * y * y * y) * (1.0 / 3.0);
  return y;
}

GRACE_HD float finv_cbrt(float x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return ::rcbrtf(x);
#else
  return (float) inv_cbrt_host(x);
#endif
}
GRACE_HD double finv_cbrt(double x)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return ::rcbrt(x);
#else
  return inv_cbrt_host(x);
#endif
}

// a side of a triangle for the three-body term: u = r^2 (< rc^2) and what atm needs of it (computed once per
// neighbour for the two sides at the first vertex): 1/u, r, u^(-1/3), sw(r) and dsw/du
template<class T> struct AtmSide {
  T u, iu, r, q, sw, dsw;
};
template<class T> GRACE_HD AtmSide<T> atm_side(T u, T rc, T ron)
{
  AtmSide<T> s;
  s.u = u;
  s.iu = T(1) / u;
  s.r = fsqrt(u);
  s.q = finv_cbrt(u);
  T d;
  s.sw = smooth(s.r, rc, ron, d);
  s.dsw = d / (T(2) * s.r);
  return s;
}

// three-body term per unit sqrt|C6 C6 C6| / sqrt(Hartree) of a triangle with sides s0, s1, s2:
//   atm = (3/8 prod_p (S - 2 u_p) / P^(5/2) + 1 / P^(3/2)) fd sw(r_0) sw(r_1) sw(r_2),  S = sum u, P = prod u
//   (= (3 cos a cos b cos c + 1) / (r_0 r_1 r_2)^3), fd = 1 / (1 + c P^(-8/3)) with c = 6 prod_pairs (4/3 R0)^(16/3)
//   (= 1 / (1 + 6 (R0 / r)^(16/3)), R0 and r the products over the sides); dg[p] = d atm / du_p
template<class T>
GRACE_HD T atm(const AtmSide<T> &s0, const AtmSide<T> &s1, const AtmSide<T> &s2, T c, T *dg)
{
  const AtmSide<T> *sd[3] = {&s0, &s1, &s2};
  const T l0 = s1.u + s2.u - s0.u, l1 = s0.u + s2.u - s1.u, l2 = s0.u + s1.u - s2.u;
  const T m0 = l1 * l2, m1 = l0 * l2, m2 = l0 * l1, nn = l0 * m0;
  const T ip = s0.iu * s1.iu * s2.iu;
  const T r3 = ip / (s0.r * s1.r * s2.r), r5 = r3 * ip;    // P^(-3/2), P^(-5/2)
  const T a = T(0.375) * nn * r5 + r3;
  const T q = s0.q * s1.q * s2.q, q2 = q * q, q4 = q2 * q2, t = c * q4 * q4;    // c P^(-8/3)
  const T fd = T(1) / (T(1) + t);
  const T s3 = s0.sw * s1.sw * s2.sw, af = a * fd;
  const T msum = m0 + m1 + m2, mm[3] = {m0, m1, m2};
  const T so[3] = {s1.sw * s2.sw, s0.sw * s2.sw, s0.sw * s1.sw};
  for (int p = 0; p < 3; p++) {
    const T iu = sd[p]->iu;
    const T da = T(0.375) * ((msum - T(2) * mm[p]) * r5 - T(2.5) * nn * r5 * iu) - T(1.5) * r3 * iu;
    const T dfd = fd * fd * t * T(8.0 / 3.0) * iu;
    dg[p] = (da * fd + a * dfd) * s3 + af * so[p] * sd[p]->dsw;
  }
  return af * s3;
}

// tables for the species of one system (eV, Angstrom): species s = 0..ns-1 with atomic numbers z[s]
struct Tables {
  DispersionParams p;    // validated
  int ns = 0;
  std::vector<int> nref;        // [ns]
  std::vector<double> cnref;    // [ns][kMaxRef] (-1 past nref)
  std::vector<double> rcov;     // [ns][ns] 4/3 (R_s + R_t)
  std::vector<double> qq;       // [ns][ns] 3 r4r2_s r4r2_t (Angstrom^2)
  std::vector<double> r06;      // [ns][ns] R0^6, R0 = a1 sqrt(qq) + a2
  std::vector<double> r08;      // [ns][ns] R0^8
  std::vector<double> c6ref;    // [ns][ns][kMaxRef][kMaxRef] eV Angstrom^6, zero padded; block (s, t): a over s, b over t
  std::vector<double> rvdw16;   // [ns][ns] (4/3 R0vdw_st)^(16/3) in Angstrom^(16/3) (three-body damping)
};

inline Tables make_tables(const int *z, int ns, const DispersionParams &p)
{
  validate(p);
  Tables t;
  t.p = p;
  t.ns = ns;
  t.nref.resize(ns);
  t.cnref.assign((size_t) ns * kMaxRef, -1.0);
  t.rcov.resize((size_t) ns * ns);
  t.qq.resize((size_t) ns * ns);
  t.r06.resize((size_t) ns * ns);
  t.r08.resize((size_t) ns * ns);
  t.c6ref.assign((size_t) ns * ns * kMaxRef * kMaxRef, 0.0);
  t.rvdw16.resize((size_t) ns * ns);
  std::vector<double> r4r2(ns);
  for (int s = 0; s < ns; s++) {
    if (z[s] < 1 || z[s] > d3_data::kMaxZ)
      throw std::invalid_argument("dispersion: no DFT-D3 reference data for atomic number " + std::to_string(z[s]) +
                                  " (elements 1.." + std::to_string(d3_data::kMaxZ) + ")");
    t.nref[s] = d3_data::kNRef[z[s]];
    for (int a = 0; a < t.nref[s]; a++) t.cnref[(size_t) s * kMaxRef + a] = d3_data::kCNRef[z[s]][a];
    r4r2[s] = std::sqrt(0.5 * d3_data::kR4OverR2[z[s]] * std::sqrt((double) z[s])) * kBohr;
  }
  const double c6unit = kHartree * std::pow(kBohr, 6);
  for (int s = 0; s < ns; s++)
    for (int u = 0; u < ns; u++) {
      const size_t st = (size_t) s * ns + u;
      t.rcov[st] = 4.0 / 3.0 * (d3_data::kCovRad[z[s]] + d3_data::kCovRad[z[u]]);
      t.qq[st] = 3.0 * r4r2[s] * r4r2[u];
      const double r0 = p.a1 * std::sqrt(t.qq[st]) + p.a2, r02 = r0 * r0;
      t.r06[st] = r02 * r02 * r02;
      t.r08[st] = t.r06[st] * r02;
      const int zi = std::min(z[s], z[u]), zj = std::max(z[s], z[u]);
      t.rvdw16[st] = std::pow(4.0 / 3.0 * d3_data::kRVdW[zj * (zj + 1) / 2 + zi], 16.0 / 3.0);
      const double *blk = &d3_data::kC6[d3_data::kC6Offset[zj * (zj + 1) / 2 + zi]];
      for (int a = 0; a < t.nref[s]; a++)
        for (int b = 0; b < t.nref[u]; b++)    // stored (zi <= zj): row over the references of zi
          t.c6ref[st * kMaxRef * kMaxRef + a * kMaxRef + b] =
              c6unit * (z[s] <= z[u] ? blk[a * d3_data::kNRef[zj] + b] : blk[b * d3_data::kNRef[zj] + a]);
    }
  return t;
}

// species of a structure: returns the distinct atomic numbers (ascending), type [n] = each atom's index into them
inline std::vector<int> species_of(const int *numbers, int64_t n, std::vector<int> &type)
{
  std::vector<int> z(numbers, numbers + n);
  std::sort(z.begin(), z.end());
  z.erase(std::unique(z.begin(), z.end()), z.end());
  type.resize(n);
  for (int64_t i = 0; i < n; i++) type[i] = (int) (std::lower_bound(z.begin(), z.end(), numbers[i]) - z.begin());
  return z;
}


// a pair term of the CPU driver with dE/dr / r = hh along d = r_j - r_i: force on i (fi), on j (fj), virial v
inline void add_pair(double hh, double dx, double dy, double dz, double *fi, double *fj, double *v)
{
  fi[0] += hh * dx;
  fi[1] += hh * dy;
  fi[2] += hh * dz;
  fj[0] -= hh * dx;
  fj[1] -= hh * dy;
  fj[2] -= hh * dz;
  v[0] -= hh * dx * dx;
  v[1] -= hh * dy * dy;
  v[2] -= hh * dz * dz;
  v[3] -= hh * dx * dy;
  v[4] -= hh * dx * dz;
  v[5] -= hh * dy * dz;
}

// CPU evaluation (fp64), added into the outputs. type [n] in 0..ns-1; cl: a cell list of the positions (bins of any
// size; coincident atoms have no terms); energy += E; eat [n] (nullable) += E_i; f [n][3] and vir [6] (nullable with
// energy_only) += forces, virial. num_threads: OpenMP threads (<= 0: 1). Atoms in the cell list's (spatial) order;
// each pair once (half lists gathered into arrays: CellList::gather_half), the share of the other atom into
// per-thread arrays summed after the pass; C6 from per-atom rows sum_a w_ia C6ref_ab, so a pair is a 7-term product
// with j's weights. (Explicit SIMD of the pair loops with a polynomial exp was tried: no gain, the gather and the
// scatter dominate.) The three-body term (s9 != 0) visits each triangle once, from its first vertex over the pairs of
// that vertex's half list, after the pair terms.
inline void compute_cpu(const Tables &t, const int *type, int64_t n, const CellList &cl, int num_threads,
                        bool energy_only, double &energy, double *eat, double *f, double *vir)
{
  constexpr int R = kMaxRef;
  const DispersionParams &p = t.p;
  const int nth = num_threads > 0 ? num_threads : 1;
  const int ns = t.ns;
  const double rc = p.cutoff, ron = p.cutoff_on, rcn = p.cn_cutoff, rc3 = p.atm_cutoff, ron3 = p.atm_cutoff_on;
  const bool atm3 = p.s9 != 0.0;
  const double s9u = p.s9 / std::sqrt(kHartree);
  if (cl.size() != n) throw std::invalid_argument("dispersion: the cell list is not of these atoms");
  // per slot (cell-list order): species, weights, dE/dCN; with the three-body term the C6 rows of every atom
  // sum_a w_qa C6ref(s_q, s)_ab and their CN derivatives [n][ns][R] (zero past the references; the triangles need the
  // rows of the neighbours), else one atom's rows per thread
  std::vector<int> ty(n);
  for (int64_t q = 0; q < n; q++) ty[q] = type[cl.perm(q)];
  std::vector<double> w((size_t) n * R), dw((size_t) n * R), dedcn(n);
  const size_t nrow = atm3 ? (size_t) n * ns * R : 0;
  std::vector<double> row(nrow), drow(energy_only ? 0 : nrow);
  // per thread, allocated here (an exception must not leave the parallel region; the arrays are zeroed, so first
  // touched, by their thread): per slot [n] CN, then energy; [n] dE/dCN; [3n] forces; the neighbour arrays; one
  // atom's C6 rows; three-body per neighbour: its triangle side, C6 with the first vertex, the ratios dC6/dCN / C6
  // there and at the neighbour, the sums of its shares (energy, dE/dCN, force)
  struct Work {
    std::vector<double> part, rows, nb;
    std::vector<AtmSide<double>> side;
    CellList::Hits h;
    double e = 0.0, v[6] = {0, 0, 0, 0, 0, 0};    // the thread's energy and virial
  };
  enum { C6N, RQN, RAN, AEN, ADN, AXN, AYN, AZN, NNB };    // nb fields, [field][neighbour]
  std::vector<Work> work(nth);
  const int64_t hmax = cl.max_gather(std::max(rc, rcn)), h3 = atm3 ? cl.max_gather(rc3) : 0;
  const size_t psize = energy_only ? (size_t) n : 5 * (size_t) n;
  for (auto &wk : work) {
    wk.part.reserve(psize);
    wk.h.reserve(std::max(hmax, h3));
    wk.rows.reserve(2 * (size_t) ns * R);
    wk.nb.reserve(NNB * (size_t) h3);
    wk.side.reserve(h3);
  }
  // schedule(static, 32) throughout and the thread sums added in thread order: the same atoms on the same thread in
  // every call, so the results repeat bit for bit at a given thread count
#pragma omp parallel num_threads(nth)
  {
#if defined(_OPENMP)
    Work &wk = work[omp_get_thread_num()];
    const int nt = omp_get_num_threads();
#else
    Work &wk = work[0];
    const int nt = 1;
#endif
    auto summed = [&](size_t o) {    // the thread arrays at offset o
      double s = 0.0;
      for (int th = 0; th < nt; th++) s += work[th].part[o];
      return s;
    };
    wk.part.assign(psize, 0.0);    // within the capacity: no allocation
    wk.h.grow(std::max(hmax, h3));
    wk.rows.resize(2 * (size_t) ns * R);
    wk.nb.resize(NNB * (size_t) h3);
    wk.side.resize(h3);
    CellList::Hits &h = wk.h;
    double *ce = wk.part.data(), *cd = energy_only ? nullptr : ce + n, *cf = energy_only ? nullptr : ce + 2 * n;
    // coordination numbers
#pragma omp for schedule(static, 32)
    for (int64_t q = 0; q < n; q++) {
      const double *rcov = &t.rcov[(size_t) ty[q] * ns];
      const int64_t m = cl.gather_half(q, rcn, h);
      double c = 0.0, dfdr;
      for (int64_t k = 0; k < m; k++) {
        const double fk = count(rcov[ty[h.p[k]]], std::sqrt(h.r2[k]), dfdr);
        c += fk;
        ce[h.p[k]] += fk;
      }
      ce[q] += c;
    }
    // the C6 rows of slot q into r and dr ([ns][R]; dr nullable)
    auto rows_of = [&](int64_t q, double *r, double *dr) {
      const int si = ty[q];
      const double *wi = &w[(size_t) q * R], *dwi = &dw[(size_t) q * R];
      for (int s = 0; s < ns; s++)
        for (int b = 0; b < R; b++) {
          const double *blk = &t.c6ref[((size_t) si * ns + s) * R * R + b];
          double cr = 0.0, dd = 0.0;
          for (int a = 0; a < R; a++) {
            cr += wi[a] * blk[a * R];
            dd += dwi[a] * blk[a * R];
          }
          r[s * R + b] = cr;
          if (dr) dr[s * R + b] = dd;
        }
    };
#pragma omp for schedule(static)
    for (int64_t q = 0; q < n; q++) {
      const int si = ty[q];
      weights(summed(q), &t.cnref[(size_t) si * R], t.nref[si], &w[(size_t) q * R], &dw[(size_t) q * R]);
    }
    if (atm3) {
#pragma omp for schedule(static)
      for (int64_t q = 0; q < n; q++)
        rows_of(q, &row[(size_t) q * ns * R], energy_only ? nullptr : &drow[(size_t) q * ns * R]);
    }
    std::fill(ce, ce + n, 0.0);    // (after the barrier above: every CN read)
    // pair terms and dE/dCN
#pragma omp for schedule(static, 32)
    for (int64_t q = 0; q < n; q++) {
      const int si = ty[q];
      const double *rq = wk.rows.data(), *drq = rq + ns * R;
      if (atm3) {
        rq = &row[(size_t) q * ns * R];
        drq = energy_only ? nullptr : &drow[(size_t) q * ns * R];
      } else {
        rows_of(q, wk.rows.data(), energy_only ? nullptr : wk.rows.data() + ns * R);
      }
      const int64_t m = cl.gather_half(q, rc, h);
      double ei = 0.0, dei = 0.0, vq[6] = {0, 0, 0, 0, 0, 0};
      for (int64_t k = 0; k < m; k++) {
        const int64_t j = h.p[k];
        const int sj = ty[j];
        const size_t st = (size_t) si * ns + sj;
        const double *wj = &w[(size_t) j * R], *dwj = &dw[(size_t) j * R], *ri = &rq[(size_t) sj * R];
        double c = 0.0, dgdr;
        for (int b = 0; b < R; b++) c += ri[b] * wj[b];
        const double r = std::sqrt(h.r2[k]);
        const double g = pair(r, t.r06[st], t.r08[st], t.qq[st], p.s6, p.s8, rc, ron, dgdr);
        ei += 0.5 * c * g;
        ce[j] += 0.5 * c * g;
        if (energy_only) continue;
        const double *dri = &drq[(size_t) sj * R];
        double dci = 0.0, dcj = 0.0;
        for (int b = 0; b < R; b++) {
          dci += dri[b] * wj[b];
          dcj += ri[b] * dwj[b];
        }
        dei += g * dci;
        cd[j] += g * dcj;
        add_pair(c * dgdr / r, h.dx[k], h.dy[k], h.dz[k], &cf[3 * q], &cf[3 * j], vq);
      }
      ce[q] += ei;
      if (energy_only) continue;
      cd[q] += dei;
      for (int k = 0; k < 6; k++) wk.v[k] += vq[k];
    }
    if (atm3) {
      // three-body term, each triangle once: from its first vertex q over the pairs (a, b) of q's half list (which
      // orders atoms and images totally: bin coordinate, then slot), with the shares of all three vertices; the
      // triangle's virial to q. The shares of the neighbours summed per neighbour, added after q's triangles
      const double rc32 = rc3 * rc3;
      const size_t H = (size_t) h3;
      double *c6n = &wk.nb[C6N * H], *rqn = &wk.nb[RQN * H], *ran = &wk.nb[RAN * H], *ae = &wk.nb[AEN * H],
             *ad = &wk.nb[ADN * H], *axn = &wk.nb[AXN * H], *ayn = &wk.nb[AYN * H], *azn = &wk.nb[AZN * H];
      AtmSide<double> *side = wk.side.data();
#pragma omp for schedule(static, 32)
      for (int64_t q = 0; q < n; q++) {
        const int si = ty[q];
        const double *rq = &row[(size_t) q * ns * R], *drq = energy_only ? nullptr : &drow[(size_t) q * ns * R];
        const int64_t m = cl.gather_half(q, rc3, h);
        for (int64_t a = 0; a < m; a++) {    // C6 of q with each neighbour and the ratios dC6/dCN / C6 at both ends
          const int64_t j = h.p[a];
          const int sj = ty[j];
          const double *wj = &w[(size_t) j * R], *dwj = &dw[(size_t) j * R], *ri = &rq[(size_t) sj * R];
          double c = 0.0, dq = 0.0, da = 0.0;
          for (int b = 0; b < R; b++) c += ri[b] * wj[b];
          if (!energy_only)
            for (int b = 0; b < R; b++) {
              dq += drq[(size_t) sj * R + b] * wj[b];
              da += ri[b] * dwj[b];
            }
          c6n[a] = c;
          rqn[a] = dq / c;
          ran[a] = da / c;
          ae[a] = ad[a] = axn[a] = ayn[a] = azn[a] = 0.0;
          side[a] = atm_side(h.r2[a], rc3, ron3);
        }
        double ei = 0.0, dei = 0.0, fq[3] = {0, 0, 0}, vq[6] = {0, 0, 0, 0, 0, 0};
        for (int64_t a = 0; a < m; a++) {
          const int64_t ja = h.p[a];
          const int sj = ty[ja];
          const double *rj = &row[(size_t) ja * ns * R], *drj = energy_only ? nullptr : &drow[(size_t) ja * ns * R];
          const double rij = t.rvdw16[(size_t) si * ns + sj];
          const double ax = h.dx[a], ay = h.dy[a], az = h.dz[a];
          for (int64_t b = a + 1; b < m; b++) {
            const double bx = h.dx[b], by = h.dy[b], bz = h.dz[b];
            const double ex = bx - ax, ey = by - ay, ez = bz - az, ujk = ex * ex + ey * ey + ez * ez;
            if (!(ujk < rc32 && ujk > 0.0)) continue;
            const int64_t kb = h.p[b];
            const int sk = ty[kb];
            const double *rjk = &rj[(size_t) sk * R], *wk3 = &w[(size_t) kb * R];
            double cjk = 0.0;
            for (int c = 0; c < R; c++) cjk += rjk[c] * wk3[c];
            const double c9 = std::fabs(c6n[a] * c6n[b] * cjk);
            if (!(c9 > 0.0)) continue;
            const double cd3 = 6.0 * rij * t.rvdw16[(size_t) si * ns + sk] * t.rvdw16[(size_t) sj * ns + sk];
            double dg[3];
            const double k9 = s9u * std::sqrt(c9);
            const double e = k9 * atm(side[a], side[b], atm_side(ujk, rc3, ron3), cd3, dg), e3 = e / 3.0;
            ei += e3;
            ae[a] += e3;
            ae[b] += e3;
            if (energy_only) continue;
            // dE/dCN through the three C6 (dE/dC6 = E / (2 C6))
            const double *drjk = &drj[(size_t) sk * R], *dwk = &dw[(size_t) kb * R];
            double djj = 0.0, djk = 0.0;    // dC6_jk/dCN_j, dC6_jk/dCN_k
            for (int c = 0; c < R; c++) {
              djj += drjk[c] * wk3[c];
              djk += rjk[c] * dwk[c];
            }
            const double he = 0.5 * e, hc = he / cjk;
            dei += he * (rqn[a] + rqn[b]);
            ad[a] += he * ran[a] + hc * djj;
            ad[b] += he * ran[b] + hc * djk;
            // dE/dr / r of the sides q-j, q-k, j-k (2 dE/du): forces on the three vertices
            const double h0 = 2.0 * k9 * dg[0], h1 = 2.0 * k9 * dg[1], h2 = 2.0 * k9 * dg[2];
            fq[0] += h0 * ax + h1 * bx;
            fq[1] += h0 * ay + h1 * by;
            fq[2] += h0 * az + h1 * bz;
            axn[a] += h2 * ex - h0 * ax;
            ayn[a] += h2 * ey - h0 * ay;
            azn[a] += h2 * ez - h0 * az;
            axn[b] -= h1 * bx + h2 * ex;
            ayn[b] -= h1 * by + h2 * ey;
            azn[b] -= h1 * bz + h2 * ez;
            vq[0] -= h0 * ax * ax + h1 * bx * bx + h2 * ex * ex;
            vq[1] -= h0 * ay * ay + h1 * by * by + h2 * ey * ey;
            vq[2] -= h0 * az * az + h1 * bz * bz + h2 * ez * ez;
            vq[3] -= h0 * ax * ay + h1 * bx * by + h2 * ex * ey;
            vq[4] -= h0 * ax * az + h1 * bx * bz + h2 * ex * ez;
            vq[5] -= h0 * ay * az + h1 * by * bz + h2 * ey * ez;
          }
        }
        for (int64_t a = 0; a < m; a++) {    // the neighbours' shares
          const int64_t j = h.p[a];
          ce[j] += ae[a];
          if (energy_only) continue;
          cd[j] += ad[a];
          cf[3 * j] += axn[a];
          cf[3 * j + 1] += ayn[a];
          cf[3 * j + 2] += azn[a];
        }
        ce[q] += ei;
        if (energy_only) continue;
        cd[q] += dei;
        for (int d = 0; d < 3; d++) cf[3 * q + d] += fq[d];
        for (int k = 0; k < 6; k++) wk.v[k] += vq[k];
      }
    }
#pragma omp for schedule(static)
    for (int64_t q = 0; q < n; q++) {
      const double ea = summed(q);
      wk.e += ea;
      if (eat) eat[cl.perm(q)] += ea;
      if (!energy_only) dedcn[q] = summed(n + q);
    }
    if (!energy_only) {
      // forces through the coordination numbers: (dE/dCN_i + dE/dCN_j) df/dr per pair
#pragma omp for schedule(static, 32)
      for (int64_t q = 0; q < n; q++) {
        const double *rcov = &t.rcov[(size_t) ty[q] * ns];
        const int64_t m = cl.gather_half(q, rcn, h);
        double dfdr, vq[6] = {0, 0, 0, 0, 0, 0};
        for (int64_t k = 0; k < m; k++) {
          const int64_t j = h.p[k];
          const double r = std::sqrt(h.r2[k]);
          count(rcov[ty[j]], r, dfdr);
          add_pair((dedcn[q] + dedcn[j]) * dfdr / r, h.dx[k], h.dy[k], h.dz[k], &cf[3 * q], &cf[3 * j], vq);
        }
        for (int k = 0; k < 6; k++) wk.v[k] += vq[k];
      }
#pragma omp for schedule(static)
      for (int64_t q = 0; q < n; q++)
        for (int d = 0; d < 3; d++) f[3 * cl.perm(q) + d] += summed(2 * n + 3 * q + d);
    }
  }
  for (const auto &wk : work) energy += wk.e;
  if (energy_only) return;
  for (const auto &wk : work)
    for (int k = 0; k < 6; k++) vir[k] += wk.v[k];
}

}    // namespace d3
}    // namespace grace

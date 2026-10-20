// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from lammps-ace src/ML-PACE/grace_3l_fused_cpu.h (feature/grace-runtime a73b8b17ee); contributing author: Yury
// Lysogorskiy (ICAMS). Block operations of the GRACE-3L CPU driver. Internal: include only from src/fused_3l.cpp.
//
// A block holds nb <= NB atoms of one type. A block tensor of nf functions and C channels is stored
// [f][NB][C]: row (f, a) at ((size_t) f * NB + a) * C, so the rows of one function over the block's atoms are
// contiguous and every dense contraction is a GEMM with M = rows.
#ifndef GRACE_FUSED_3L_CPU_H
#define GRACE_FUSED_3L_CPU_H

#include "grace/cpu_gemm.h"
#include "grace/cpu_simd.h"
#include "grace/fused_cpu_types.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __linux__
#include <sys/mman.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace grace {
namespace f3l {

using fused::fvec;

// large buffers (block tensors, per-atom arrays, radial tables) on 2 MB transparent huge pages where the kernel
// only grants them on request (THP "madvise" mode): their rows are visited with page-sized strides
template<class T> struct AlignHuge {
  using value_type = T;
  static constexpr size_t HP = size_t(1) << 21;
  AlignHuge() = default;
  template<class U> AlignHuge(const AlignHuge<U> &) {}
  T *allocate(std::size_t n)
  {
    const size_t bytes = n * sizeof(T);
    if (bytes < HP) return static_cast<T *>(::operator new(bytes, std::align_val_t(64)));
    const size_t b = (bytes + HP - 1) / HP * HP;
    void *p = std::aligned_alloc(HP, b);
    if (!p) throw std::bad_alloc();
#ifdef __linux__
    madvise(p, b, MADV_HUGEPAGE);
#endif
    return static_cast<T *>(p);
  }
  void deallocate(T *p, std::size_t n)
  {
    if (n * sizeof(T) < HP)
      ::operator delete(p, std::align_val_t(64));
    else
      std::free(p);
  }
  template<class U> bool operator==(const AlignHuge<U> &) const { return true; }
  template<class U> bool operator!=(const AlignHuge<U> &) const { return false; }
};
using hvec = std::vector<float, AlignHuge<float>>;

typedef simd::f16 sv16;
inline sv16 sld(const float *p) { return simd::load(p); }
inline void sst(float *p, const sv16 v) { simd::store(p, v); }

// profiling (G2LF_PROF): prof_on is set on OpenMP thread 0 only; prof_couple is the time spent producing /
// reversing couple rows inside the fused reduces
inline thread_local bool prof_on = false;
inline thread_local double prof_couple = 0.0;
inline double prof_now()
{
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return 0.0;
#endif
}

/* out[dst][a][k] += sum_n W[tile][n][k] in[src][a][n] over connections (src, dst, tile); the weights carry their
   norm. Connections are grouped by tile, so a group is one GEMM with M = (connections of the tile) x nb. */
struct Contract {
  int nin = 0, nout = 0, nt = 0, maxg = 0;
  std::vector<int> goff, src, dst;    // group t = connections [goff[t], goff[t+1])
  std::vector<char> csrc, cdst;       // group has consecutive src / dst (no gather / scatter when the block is full)
  fvec W, WT;                         // [sp][t][nin][nout], [sp][t][nout][nin]
  const float *w(int sp, int t) const { return W.data() + ((size_t) sp * nt + t) * nin * nout; }
  const float *wt(int sp, int t) const { return WT.data() + ((size_t) sp * nt + t) * nin * nout; }
};

// conns (src, dst, tile) and a weight getter wget(sp, tile, n, k) (norm folded by the caller)
template<class WGet>
inline void build_contract(Contract &C, int nin, int nout, int nt, int nsp, const std::vector<int> &src,
                           const std::vector<int> &dst, const std::vector<int> &tile, WGet wget)
{
  C.nin = nin;
  C.nout = nout;
  C.nt = nt;
  C.goff.assign(nt + 1, 0);
  for (int t : tile) C.goff[t + 1]++;
  for (int t = 0; t < nt; t++) C.goff[t + 1] += C.goff[t];
  std::vector<int> pos(C.goff.begin(), C.goff.end() - 1);
  C.src.assign(src.size(), 0);
  C.dst.assign(src.size(), 0);
  for (size_t c = 0; c < src.size(); c++) {
    const int q = pos[tile[c]]++;
    C.src[q] = src[c];
    C.dst[q] = dst[c];
  }
  C.maxg = 0;
  C.csrc.assign(nt, 0);
  C.cdst.assign(nt, 0);
  for (int t = 0; t < nt; t++) {
    const int g0 = C.goff[t], ng = C.goff[t + 1] - g0;
    C.maxg = std::max(C.maxg, ng);
    bool cs = ng > 0, cd = ng > 0;
    for (int q = 1; q < ng; q++) {
      cs = cs && C.src[g0 + q] == C.src[g0] + q;
      cd = cd && C.dst[g0 + q] == C.dst[g0] + q;
    }
    C.csrc[t] = cs;
    C.cdst[t] = cd;
  }
  C.W.assign((size_t) nsp * nt * nin * nout, 0.0f);
  C.WT.assign(C.W.size(), 0.0f);
  for (int sp = 0; sp < nsp; sp++)
    for (int t = 0; t < nt; t++) {
      float *w = C.W.data() + ((size_t) sp * nt + t) * nin * nout;
      float *wt = C.WT.data() + ((size_t) sp * nt + t) * nin * nout;
      for (int n = 0; n < nin; n++)
        for (int k = 0; k < nout; k++) {
          const float v = (float) wget(sp, t, n, k);
          w[(size_t) n * nout + k] = v;
          wt[(size_t) k * nin + n] = v;
        }
    }
}

// scratch of the contractions: X [maxg NB nin], Y [maxg NB nout]
struct Scratch {
  fvec X, Y;
  void ensure(size_t nx, size_t ny)
  {
    if (X.size() < nx) X.resize(nx);
    if (Y.size() < ny) Y.resize(ny);
  }
};

inline void vadd(float *__restrict y, const float *__restrict x, size_t n)
{
#pragma omp simd
  for (size_t q = 0; q < n; q++) y[q] += x[q];
}

inline void contract_fwd(const Contract &C, int sp, const float *in, float *out, int NB, int nb, Scratch &s)
{
  const size_t ri = (size_t) NB * C.nin, ro = (size_t) NB * C.nout;
  for (int t = 0; t < C.nt; t++) {
    const int g0 = C.goff[t], ng = C.goff[t + 1] - g0;
    if (ng == 0) continue;
    const float *w = C.w(sp, t);
    if (ng == 1) {
      gemm::gemm(nb, C.nout, C.nin, 1.0f, in + C.src[g0] * ri, C.nin, w, C.nout, 1.0f, out + C.dst[g0] * ro,
                      C.nout);
      continue;
    }
    const bool full = nb == NB;
    const float *X;
    if (full && C.csrc[t])
      X = in + C.src[g0] * ri;
    else {
      s.ensure((size_t) C.maxg * NB * C.nin, (size_t) C.maxg * NB * C.nout);
      for (int q = 0; q < ng; q++)
        std::memcpy(s.X.data() + (size_t) q * nb * C.nin, in + C.src[g0 + q] * ri, (size_t) nb * C.nin * sizeof(float));
      X = s.X.data();
    }
    if (full && C.cdst[t]) {
      gemm::gemm(ng * nb, C.nout, C.nin, 1.0f, X, C.nin, w, C.nout, 1.0f, out + C.dst[g0] * ro, C.nout);
    } else {
      s.ensure((size_t) C.maxg * NB * C.nin, (size_t) C.maxg * NB * C.nout);
      gemm::gemm(ng * nb, C.nout, C.nin, 1.0f, X, C.nin, w, C.nout, 0.0f, s.Y.data(), C.nout);
      for (int q = 0; q < ng; q++) vadd(out + C.dst[g0 + q] * ro, s.Y.data() + (size_t) q * nb * C.nout, (size_t) nb * C.nout);
    }
  }
}

// din[src][a][n] += sum_k W[tile][n][k] dout[dst][a][k]
inline void contract_bwd(const Contract &C, int sp, const float *dout, float *din, int NB, int nb, Scratch &s)
{
  const size_t ri = (size_t) NB * C.nin, ro = (size_t) NB * C.nout;
  for (int t = 0; t < C.nt; t++) {
    const int g0 = C.goff[t], ng = C.goff[t + 1] - g0;
    if (ng == 0) continue;
    const float *wt = C.wt(sp, t);
    if (ng == 1) {
      gemm::gemm(nb, C.nin, C.nout, 1.0f, dout + C.dst[g0] * ro, C.nout, wt, C.nin, 1.0f, din + C.src[g0] * ri,
                      C.nin);
      continue;
    }
    const bool full = nb == NB;
    const float *Y;
    if (full && C.cdst[t])
      Y = dout + C.dst[g0] * ro;
    else {
      s.ensure((size_t) C.maxg * NB * C.nin, (size_t) C.maxg * NB * C.nout);
      for (int q = 0; q < ng; q++)
        std::memcpy(s.Y.data() + (size_t) q * nb * C.nout, dout + C.dst[g0 + q] * ro,
                    (size_t) nb * C.nout * sizeof(float));
      Y = s.Y.data();
    }
    if (full && C.csrc[t]) {
      gemm::gemm(ng * nb, C.nin, C.nout, 1.0f, Y, C.nout, wt, C.nin, 1.0f, din + C.src[g0] * ri, C.nin);
    } else {
      s.ensure((size_t) C.maxg * NB * C.nin, (size_t) C.maxg * NB * C.nout);
      gemm::gemm(ng * nb, C.nin, C.nout, 1.0f, Y, C.nout, wt, C.nin, 0.0f, s.X.data(), C.nin);
      for (int q = 0; q < ng; q++) vadd(din + C.src[g0 + q] * ri, s.X.data() + (size_t) q * nb * C.nin, (size_t) nb * C.nin);
    }
  }
}

// cp_l couple: out[f][a][r] = sum_{t: m_sum(t) = f} cg_t Lp[li_t][a][r] Rp[ri_t][a][r]
struct Couple {
  int nfunc = 0, rank = 0;
  std::vector<int> off, li, ri;
  fvec cg;
};

// one couple function f for the nb atoms of a block: o[a][r] = sum_t cg_t Lp[li_t][a][r] Rp[ri_t][a][r] (o: nb rows);
// accumulators in registers over the terms, 64 floats at a time (nb * rank is a multiple of 16)
inline void couple_row_fwd(const Couple &c, int f, const float *Lp, const float *Rp, float *o, int NB, int nb)
{
  const size_t rs = (size_t) NB * c.rank;
  const int n = nb * c.rank, t0 = c.off[f], t1 = c.off[f + 1];
  int q = 0;
  for (; q + 64 <= n; q += 64) {
    sv16 a0 = sv16{}, a1 = sv16{}, a2 = sv16{}, a3 = sv16{};
    for (int t = t0; t < t1; t++) {
      const float *l = Lp + c.li[t] * rs + q, *r = Rp + c.ri[t] * rs + q;
      const float g = c.cg[t];
      a0 += g * sld(l) * sld(r);
      a1 += g * sld(l + 16) * sld(r + 16);
      a2 += g * sld(l + 32) * sld(r + 32);
      a3 += g * sld(l + 48) * sld(r + 48);
    }
    sst(o + q, a0); sst(o + q + 16, a1); sst(o + q + 32, a2); sst(o + q + 48, a3);
  }
  for (; q < n; q += 16) {
    sv16 a0 = sv16{};
    for (int t = t0; t < t1; t++) a0 += c.cg[t] * sld(Lp + c.li[t] * rs + q) * sld(Rp + c.ri[t] * rs + q);
    sst(o + q, a0);
  }
}

// adjoint of couple_row_fwd for the rows d of function f
inline void couple_row_bwd(const Couple &c, int f, const float *Lp, const float *Rp, const float *d, float *dLp,
                           float *dRp, int NB, int nb)
{
  const size_t rs = (size_t) NB * c.rank;
  const int n = nb * c.rank;
  for (int t = c.off[f]; t < c.off[f + 1]; t++) {
    const float *__restrict l = Lp + c.li[t] * rs, *__restrict r = Rp + c.ri[t] * rs;
    float *__restrict dl = dLp + c.li[t] * rs, *__restrict dr = dRp + c.ri[t] * rs;
    const float g = c.cg[t];
#pragma omp simd
    for (int q = 0; q < n; q++) {
      const float gd = g * d[q];
      dl[q] += gd * r[q];
      dr[q] += gd * l[q];
    }
  }
}

/* cp_l couple fused with the reduce that consumes it: per weight tile the couple rows of the tile's connections
   are produced straight into the GEMM operand, so the couple output (hundreds of functions) is never stored.
   out[dst] += W[tile] couple(src); backward: d couple = dout W^T, then the couple adjoint into dLp / dRp. */
inline void couple_contract_fwd(const Couple &cp, const Contract &C, const float *Lp, const float *Rp, float *out,
                                int NB, int nb, Scratch &s)
{
  const size_t ro = (size_t) NB * C.nout;
  s.ensure((size_t) C.maxg * NB * C.nin, (size_t) C.maxg * NB * C.nout);
  for (int t = 0; t < C.nt; t++) {
    const int g0 = C.goff[t], ng = C.goff[t + 1] - g0;
    if (ng == 0) continue;
    const double t0 = prof_on ? prof_now() : 0.0;
    for (int q = 0; q < ng; q++) couple_row_fwd(cp, C.src[g0 + q], Lp, Rp, s.X.data() + (size_t) q * nb * C.nin, NB, nb);
    if (prof_on) prof_couple += prof_now() - t0;
    const float *w = C.w(0, t);
    if (ng == 1 || (nb == NB && C.cdst[t])) {
      gemm::gemm(ng * nb, C.nout, C.nin, 1.0f, s.X.data(), C.nin, w, C.nout, 1.0f, out + C.dst[g0] * ro, C.nout);
    } else {
      gemm::gemm(ng * nb, C.nout, C.nin, 1.0f, s.X.data(), C.nin, w, C.nout, 0.0f, s.Y.data(), C.nout);
      for (int q = 0; q < ng; q++) vadd(out + C.dst[g0 + q] * ro, s.Y.data() + (size_t) q * nb * C.nout, (size_t) nb * C.nout);
    }
  }
}

inline void couple_contract_bwd(const Couple &cp, const Contract &C, const float *Lp, const float *Rp, const float *dout,
                                float *dLp, float *dRp, int NB, int nb, Scratch &s)
{
  const size_t ro = (size_t) NB * C.nout;
  s.ensure((size_t) C.maxg * NB * C.nin, (size_t) C.maxg * NB * C.nout);
  for (int t = 0; t < C.nt; t++) {
    const int g0 = C.goff[t], ng = C.goff[t + 1] - g0;
    if (ng == 0) continue;
    const float *Y;
    if (ng == 1 || (nb == NB && C.cdst[t]))
      Y = dout + C.dst[g0] * ro;
    else {
      for (int q = 0; q < ng; q++)
        std::memcpy(s.Y.data() + (size_t) q * nb * C.nout, dout + C.dst[g0 + q] * ro, (size_t) nb * C.nout * sizeof(float));
      Y = s.Y.data();
    }
    gemm::gemm(ng * nb, C.nin, C.nout, 1.0f, Y, C.nout, C.wt(0, t), C.nin, 0.0f, s.X.data(), C.nin);
    const double t0 = prof_on ? prof_now() : 0.0;
    for (int q = 0; q < ng; q++)
      couple_row_bwd(cp, C.src[g0 + q], Lp, Rp, s.X.data() + (size_t) q * nb * C.nin, dLp, dRp, NB, nb);
    if (prof_on) prof_couple += prof_now() - t0;
  }
}

// scale the rows of function f by s[f]
inline void scale_rows(float *x, const float *s, int nf, int C, int NB, int nb)
{
  const size_t rs = (size_t) NB * C;
  for (int f = 0; f < nf; f++) {
    float *__restrict p = x + f * rs;
    const float v = s[f];
#pragma omp simd
    for (int q = 0; q < nb * C; q++) p[q] *= v;
  }
}

/* EquivariantRMSNorm of one atom: x[f][n] at x + f * rs (n < C):
     xc = x - l0_mask * mean_n x;  r = 1 / sqrt(mean_n sum_f dw[f] xc^2 + eps);  y = xc r aff[e(f)][n] */
struct EqNorm {
  int nf = 0, C = 0;
  double eps = 0.0;
  bool center = true;
  std::vector<double> dw, mask;
  std::vector<int> e;
  fvec aff;    // [n_groups][C]
};

inline void eqnorm_fwd(const EqNorm &E, const float *x, float *y, size_t rs, float *xc_buf)
{
  double S = 0.0;
  for (int f = 0; f < E.nf; f++) {
    const float *xf = x + f * rs;
    float *c = xc_buf + (size_t) f * E.C;
    double m = 0.0;
    if (E.center && E.mask[f] != 0.0) {
      for (int n = 0; n < E.C; n++) m += xf[n];
      m = m / E.C * E.mask[f];
    }
    double s2 = 0.0;
    for (int n = 0; n < E.C; n++) {
      c[n] = (float) (xf[n] - m);
      s2 += (double) c[n] * c[n];
    }
    S += E.dw[f] * s2;
  }
  const double r = 1.0 / std::sqrt(S / E.C + E.eps);
  for (int f = 0; f < E.nf; f++) {
    const float *c = xc_buf + (size_t) f * E.C, *a = E.aff.data() + (size_t) E.e[f] * E.C;
    float *yf = y + f * rs;
    for (int n = 0; n < E.C; n++) yf[n] = (float) (c[n] * r) * a[n];
  }
}

// dx[f][n] += d/dx of sum dy . y
inline void eqnorm_bwd(const EqNorm &E, const float *x, const float *dy, float *dx, size_t rs, float *xc_buf)
{
  double S = 0.0;
  for (int f = 0; f < E.nf; f++) {
    const float *xf = x + f * rs;
    float *c = xc_buf + (size_t) f * E.C;
    double m = 0.0;
    if (E.center && E.mask[f] != 0.0) {
      for (int n = 0; n < E.C; n++) m += xf[n];
      m = m / E.C * E.mask[f];
    }
    double s2 = 0.0;
    for (int n = 0; n < E.C; n++) {
      c[n] = (float) (xf[n] - m);
      s2 += (double) c[n] * c[n];
    }
    S += E.dw[f] * s2;
  }
  const double r = 1.0 / std::sqrt(S / E.C + E.eps);
  // g = dy aff;  dxc = r g - r^3 / C (sum_f,n g xc) dw[f] xc
  double gx = 0.0;
  for (int f = 0; f < E.nf; f++) {
    const float *c = xc_buf + (size_t) f * E.C, *a = E.aff.data() + (size_t) E.e[f] * E.C, *d = dy + f * rs;
    for (int n = 0; n < E.C; n++) gx += (double) d[n] * a[n] * c[n];
  }
  const double k3 = r * r * r * gx / E.C;
  for (int f = 0; f < E.nf; f++) {
    const float *c = xc_buf + (size_t) f * E.C, *a = E.aff.data() + (size_t) E.e[f] * E.C, *d = dy + f * rs;
    float *o = dx + f * rs;
    const double kf = k3 * E.dw[f];
    double mean = 0.0;
    // dxc into the scratch (overwrites xc of this f, no longer needed)
    float *g = xc_buf + (size_t) f * E.C;
    for (int n = 0; n < E.C; n++) {
      const double v = r * d[n] * a[n] - kf * c[n];
      g[n] = (float) v;
      mean += v;
    }
    mean = (E.center && E.mask[f] != 0.0) ? mean / E.C * E.mask[f] : 0.0;
    for (int n = 0; n < E.C; n++) o[n] += (float) (g[n] - mean);
  }
}

// InvariantLayerRMSNorm of one atom, x[n] (n < C): full (scale length C) or only_nonlin (channel 0 passes)
struct RMSNorm {
  int C = 0;
  bool only_nonlin = false;
  fvec scale;
};

inline void rmsnorm_fwd(const RMSNorm &R, const float *x, float *y)
{
  const int n0 = R.only_nonlin ? 1 : 0, nn = R.C - n0;
  if (n0) y[0] = x[0];
  double s = 0.0;
  for (int n = n0; n < R.C; n++) s += (double) x[n] * x[n];
  const double r = 1.0 / std::sqrt(s / nn + 1e-10);
  for (int n = n0; n < R.C; n++) y[n] = (float) (x[n] * r) * R.scale[n - n0];
}

inline void rmsnorm_bwd(const RMSNorm &R, const float *x, const float *dy, float *dx)
{
  const int n0 = R.only_nonlin ? 1 : 0, nn = R.C - n0;
  if (n0) dx[0] += dy[0];
  double s = 0.0, gx = 0.0;
  for (int n = n0; n < R.C; n++) {
    s += (double) x[n] * x[n];
    gx += (double) dy[n] * R.scale[n - n0] * x[n];
  }
  const double r = 1.0 / std::sqrt(s / nn + 1e-10), k = r * r * r * gx / nn;
  for (int n = n0; n < R.C; n++) dx[n] += (float) (r * dy[n] * R.scale[n - n0] - k * x[n]);
}

/* Products of 16-lane vector matrices (one channel vector cv of the equivariant SPBF bond sums):
     O[r][c] = sum_k A[r][k] B[k][c],  A at A + r ra + k ka,  B at B + k kb + c cb,  O at O + r ro + c co
   (strides in floats, every element a 16-float vector), in MR x MC register tiles. */

template<int MR, int MC>
inline void vtile(int K, const float *A, size_t ra, size_t ka, const float *B, size_t kb, size_t cb, float *O, size_t ro,
                  size_t co)
{
  sv16 acc[MR][MC];
  for (int r = 0; r < MR; r++)
    for (int c = 0; c < MC; c++) acc[r][c] = sv16{};
  for (int k = 0; k < K; k++) {
    sv16 a[MR];
    for (int r = 0; r < MR; r++) a[r] = sld(A + r * ra + k * ka);
    for (int c = 0; c < MC; c++) {    // one B vector live at a time: MR x MC + MR + 1 registers
      const sv16 b = sld(B + k * kb + c * cb);
      for (int r = 0; r < MR; r++) acc[r][c] += a[r] * b;
    }
  }
  for (int r = 0; r < MR; r++)
    for (int c = 0; c < MC; c++) sst(O + r * ro + c * co, acc[r][c]);
}

// MR rows of the product in MR x TC tiles; the remainder columns use a tile of the remainder width
template<int MR, int TC>
inline void vrows(int Cc, int K, const float *A, size_t ra, size_t ka, const float *B, size_t kb, size_t cb, float *O,
                  size_t ro, size_t co)
{
  int c0 = 0;
  for (; c0 + TC <= Cc; c0 += TC) vtile<MR, TC>(K, A, ra, ka, B + c0 * cb, kb, cb, O + c0 * co, ro, co);
  const int rem = Cc - c0;
  const float *b = B + c0 * cb;
  float *o = O + c0 * co;
  if constexpr (TC > 1) if (rem == 1) vtile<MR, 1>(K, A, ra, ka, b, kb, cb, o, ro, co);
  if constexpr (TC > 2) if (rem == 2) vtile<MR, 2>(K, A, ra, ka, b, kb, cb, o, ro, co);
  if constexpr (TC > 3) if (rem == 3) vtile<MR, 3>(K, A, ra, ka, b, kb, cb, o, ro, co);
  if constexpr (TC > 4) if (rem == 4) vtile<MR, 4>(K, A, ra, ka, b, kb, cb, o, ro, co);
  if constexpr (TC > 5) if (rem == 5) vtile<MR, 5>(K, A, ra, ka, b, kb, cb, o, ro, co);
}

// the same product on one lane group of native-width vectors (simd::fn): on AVX2 an MR x MC tile of 16-lane
// values needs 2 MR MC accumulator registers and spills; MR x MC + MR + 1 = 16 native registers for 5 x 2
template<int MR, int MC>
inline void ntile(int K, const float *A, size_t ra, size_t ka, const float *B, size_t kb, size_t cb, float *O, size_t ro,
                  size_t co)
{
  using simd::fn;
  fn acc[MR][MC];
  for (int r = 0; r < MR; r++)
    for (int c = 0; c < MC; c++) acc[r][c] = fn{};
  for (int k = 0; k < K; k++) {
    fn a[MR];
    for (int r = 0; r < MR; r++) a[r] = simd::loadn(A + r * ra + k * ka);
    for (int c = 0; c < MC; c++) {
      const fn b = simd::loadn(B + k * kb + c * cb);
      for (int r = 0; r < MR; r++) acc[r][c] += a[r] * b;
    }
  }
  for (int r = 0; r < MR; r++)
    for (int c = 0; c < MC; c++) simd::storen(O + r * ro + c * co, acc[r][c]);
}

template<int MR, int MC>
inline void nrows(int Cc, int K, const float *A, size_t ra, size_t ka, const float *B, size_t kb, size_t cb, float *O,
                  size_t ro, size_t co)
{
  int c0 = 0;
  for (; c0 + MC <= Cc; c0 += MC) ntile<MR, MC>(K, A, ra, ka, B + c0 * cb, kb, cb, O + c0 * co, ro, co);
  for (; c0 < Cc; c0++) ntile<MR, 1>(K, A, ra, ka, B + c0 * cb, kb, cb, O + c0 * co, ro, co);
}

// the product in TR x TC tiles; the remainder rows use tiles of the remainder height. Narrower than AVX-512: 5 x 2
// tiles of native vectors, lane groups outermost (2.0x the 16-lane tiles on Zen 2, 1.3x with SSE)
template<int TR, int TC>
inline void vmm_t(int R, int Cc, int K, const float *A, size_t ra, size_t ka, const float *B, size_t kb, size_t cb,
                  float *O, size_t ro, size_t co)
{
  if constexpr (simd::NF < 16) {
    for (int h = 0; h < 16; h += simd::NF) {
      int r0 = 0;
      for (; r0 + 5 <= R; r0 += 5) nrows<5, 2>(Cc, K, A + r0 * ra + h, ra, ka, B + h, kb, cb, O + r0 * ro + h, ro, co);
      for (; r0 < R; r0++) nrows<1, 2>(Cc, K, A + r0 * ra + h, ra, ka, B + h, kb, cb, O + r0 * ro + h, ro, co);
    }
    return;
  }
  int r0 = 0;
  for (; r0 + TR <= R; r0 += TR) vrows<TR, TC>(Cc, K, A + r0 * ra, ra, ka, B, kb, cb, O + r0 * ro, ro, co);
  const int rem = R - r0;
  const float *a = A + r0 * ra;
  float *o = O + r0 * ro;
  if constexpr (TR > 1) if (rem == 1) vrows<1, TC>(Cc, K, a, ra, ka, B, kb, cb, o, ro, co);
  if constexpr (TR > 2) if (rem == 2) vrows<2, TC>(Cc, K, a, ra, ka, B, kb, cb, o, ro, co);
  if constexpr (TR > 3) if (rem == 3) vrows<3, TC>(Cc, K, a, ra, ka, B, kb, cb, o, ro, co);
  if constexpr (TR > 4) if (rem == 4) vrows<4, TC>(Cc, K, a, ra, ka, B, kb, cb, o, ro, co);
}

}    // namespace f3l
}    // namespace grace

#endif

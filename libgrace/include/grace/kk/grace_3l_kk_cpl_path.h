/* ----------------------------------------------------------------------
   SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
   Moved from lammps-ace src/KOKKOS/grace_3l_cpl_path.h (5f5ab57aa7), used by grace::kk::Grace3LKK.
   GRACE-3L cp_l couple by coupling path, operands in registers.

   The couple out[f] = sum_t cg_t L[l_t] R[r_t] of a cp_l product splits into coupling paths (l1,l2)L: an output
   function belongs to one path (a left lm block of size 2 l1 + 1, a right block of size 2 l2 + 1, 2L + 1 consecutive
   output functions). Within a path the nonzero real-spherical-harmonic CG terms (m1, m2, M) (m = -l..l in block
   order, m < 0 sine-like) satisfy
     |M| in {|m1| + |m2|, ||m1| - |m2||}   and   sine-like(M) = sine-like(m1) ^ sine-like(m2) ^ (l1 + l2 + L odd),
   so the term pattern is known at compile time for every (l1, l2, L) with l <= 4; only the cg values (one per
   candidate, 0 where the model has none) and the path lists come from the model. The style checks every term
   against this rule at load time and uses the table kernels otherwise.

   Loop structure (thread per (atom, rank); MODE 0 forward, 1 left adjoint, 2 right adjoint, 3 both adjoints with the
   right one in a caller-provided (shared) row):
     outer blocks (left blocks for modes 0 / 1, right blocks for mode 2): the outer operand block (mode 0) or the
       outer adjoint accumulator (modes 1 / 2) in registers, l_outer a template parameter;
     pairs (outer block, inner block): the inner operand block loaded once into registers;
     paths of the pair (one per L): the terms, fully unrolled.
   Tables per side: ob [n_ob][4] = (l, off, pair0, pair1), pr [n_pr][4] = (l, off, path0, path1),
   pt [n_pt][3] = (L, f0, cg offset); cg blocks 16-byte aligned, padded to a multiple of 4.
------------------------------------------------------------------------- */

#pragma once
#include <Kokkos_Core.hpp>

namespace grace {
namespace kk {
namespace G3LPath {

static constexpr int LMAX = 4;

KOKKOS_INLINE_FUNCTION constexpr int iabs(int x) { return x < 0 ? -x : x; }

KOKKOS_INLINE_FUNCTION constexpr bool cand(int l1, int l2, int L, int m1, int m2, int M)
{
  const int a = iabs(m1), b = iabs(m2), s = iabs(M);
  if (s != a + b && s != iabs(a - b)) return false;
  return (((m1 < 0) != (m2 < 0)) != (((l1 + l2 + L) & 1) == 1)) == (M < 0);
}

constexpr bool valid(int l1, int l2, int L) { return iabs(l1 - l2) <= L && L <= l1 + l2 && L <= LMAX; }

// Table reads (ob / pr / pt, cg, fcol): the same address in every lane and read-only in the kernel. HIP reads them
// through the constant address space, i.e. as scalar loads into SGPRs; as generic loads they were one vector load per
// lane through the texture path (CPLCouplePath: 882 vector, 17 scalar loads).
template<class T>
KOKKOS_FORCEINLINE_FUNCTION T uld(const T *p)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return *(const __attribute__((address_space(4))) T *) p;
#else
  return *p;
#endif
}

struct alignas(16) F4 { float v[4]; };
// HIP: per float (clang does not copy a struct with an array member out of address space 4); the scalar loads merge
KOKKOS_FORCEINLINE_FUNCTION F4 uld(const F4 *p)
{
#if defined(__HIP_DEVICE_COMPILE__)
  return F4{{uld(p->v), uld(p->v + 1), uld(p->v + 2), uld(p->v + 3)}};
#else
  return *p;
#endif
}
#define G3L_PATH_CG(k) (((k) & 3) == 0 ? (c4 = uld(cg4 + ((k) >> 2)), c4.v[0]) : c4.v[(k) & 3])

// the table pointers point into global memory (uld reads them through the constant address space on HIP)
struct Tabs {
  const int *ob, *pr, *pt;
  const float *cg;
  const int *fcol;
  int nob;
};

// one path, l1 = left, l2 = right
template<int l1, int l2, int L>
KOKKOS_FORCEINLINE_FUNCTION void p_fwd(const float *x, const float *y, float *O, long so, const float *cg, const int *fcol)
{
  if constexpr (valid(l1, l2, L)) {
    const F4 *cg4 = reinterpret_cast<const F4 *>(cg); F4 c4;
    int k = 0;
#pragma unroll
    for (int M = -L; M <= L; M++) {
      float v = 0.0f;
#pragma unroll
      for (int m1 = -l1; m1 <= l1; m1++)
#pragma unroll
        for (int m2 = -l2; m2 <= l2; m2++)
          if (cand(l1, l2, L, m1, m2, M)) { v += G3L_PATH_CG(k) * x[m1 + l1] * y[m2 + l2]; k++; }
      O[uld(fcol + M + L) * so] = v;
    }
  }
}

template<int l1, int l2, int L>
KOKKOS_FORCEINLINE_FUNCTION void p_vl(float *g, const float *y, const float *D, long sd, const float *cg, const int *fcol)
{
  if constexpr (valid(l1, l2, L)) {
    const F4 *cg4 = reinterpret_cast<const F4 *>(cg); F4 c4;
    int k = 0;
#pragma unroll
    for (int M = -L; M <= L; M++) {
      const float d = D[uld(fcol + M + L) * sd];
#pragma unroll
      for (int m1 = -l1; m1 <= l1; m1++)
#pragma unroll
        for (int m2 = -l2; m2 <= l2; m2++)
          if (cand(l1, l2, L, m1, m2, M)) { g[m1 + l1] += (G3L_PATH_CG(k) * d) * y[m2 + l2]; k++; }
    }
  }
}

template<int l1, int l2, int L>
KOKKOS_FORCEINLINE_FUNCTION void p_vr(float *g, const float *x, const float *D, long sd, const float *cg, const int *fcol)
{
  if constexpr (valid(l1, l2, L)) {
    const F4 *cg4 = reinterpret_cast<const F4 *>(cg); F4 c4;
    int k = 0;
#pragma unroll
    for (int M = -L; M <= L; M++) {
      const float d = D[uld(fcol + M + L) * sd];
#pragma unroll
      for (int m1 = -l1; m1 <= l1; m1++)
#pragma unroll
        for (int m2 = -l2; m2 <= l2; m2++)
          if (cand(l1, l2, L, m1, m2, M)) { g[m2 + l2] += (G3L_PATH_CG(k) * d) * x[m1 + l1]; k++; }
    }
  }
}

// both adjoints at once (mode 3): gl (left block) and gr (right block) accumulators, x / y the operand blocks
template<int l1, int l2, int L>
KOKKOS_FORCEINLINE_FUNCTION void p_vb(float *gl, float *gr, const float *x, const float *y, const float *D, long sd,
                                      const float *cg, const int *fcol)
{
  if constexpr (valid(l1, l2, L)) {
    const F4 *cg4 = reinterpret_cast<const F4 *>(cg); F4 c4;
    int k = 0;
#pragma unroll
    for (int M = -L; M <= L; M++) {
      const float d = D[uld(fcol + M + L) * sd];
#pragma unroll
      for (int m1 = -l1; m1 <= l1; m1++)
#pragma unroll
        for (int m2 = -l2; m2 <= l2; m2++)
          if (cand(l1, l2, L, m1, m2, M)) {
            const float c = G3L_PATH_CG(k) * d;
            gl[m1 + l1] += c * y[m2 + l2];
            gr[m2 + l2] += c * x[m1 + l1];
            k++;
          }
    }
  }
}

#define G3L_PATH_SWL(F, A, B, ...)                                                                         \
  switch (L) {                                                                                             \
    case 0: F<A, B, 0>(__VA_ARGS__); break; case 1: F<A, B, 1>(__VA_ARGS__); break;                        \
    case 2: F<A, B, 2>(__VA_ARGS__); break; case 3: F<A, B, 3>(__VA_ARGS__); break;                        \
    case 4: F<A, B, 4>(__VA_ARGS__); break; default: break;                                                \
  }

// pair (outer block LO, inner block LI): inner operand block in registers, then its paths
template<int MODE, int LO, int LI>
KOKKOS_FORCEINLINE_FUNCTION void pair_run(const Tabs &tb, float *o, const float *xo, const float *Xi, long sxi,
                                          float *Gi, long sgi, int t0, int t1, float *O, long so, const float *D,
                                          long sd)
{
  float in[2 * LI + 1], gi[2 * LI + 1];
#pragma unroll
  for (int i = 0; i < 2 * LI + 1; i++) in[i] = Xi[i * sxi];
  if (MODE == 3) {
#pragma unroll
    for (int i = 0; i < 2 * LI + 1; i++) gi[i] = Gi[i * sgi];
  }
  for (int t = t0; t < t1; t++) {
    const int *q = tb.pt + 3 * t;
    const int L = uld(q);
    const float *cg = tb.cg + uld(q + 2);
    const int *fc = tb.fcol + uld(q + 1);
    if (MODE == 0) { G3L_PATH_SWL(p_fwd, LO, LI, o, in, O, so, cg, fc) }
    else if (MODE == 1) { G3L_PATH_SWL(p_vl, LO, LI, o, in, D, sd, cg, fc) }
    else if (MODE == 2) { G3L_PATH_SWL(p_vr, LI, LO, o, in, D, sd, cg, fc) }
    else { G3L_PATH_SWL(p_vb, LO, LI, o, gi, xo, in, D, sd, cg, fc) }
  }
  if (MODE == 3) {
#pragma unroll
    for (int i = 0; i < 2 * LI + 1; i++) Gi[i * sgi] = gi[i];
  }
}

template<int MODE, int LO>
KOKKOS_FORCEINLINE_FUNCTION void block_run(const Tabs &tb, int b, const float *Xo, long sxo, float *Go, long sgo,
                                           const float *Xi, long sxi, float *Gi, long sgi, float *O, long so,
                                           const float *D, long sd)
{
  const int *bo = tb.ob + 4 * b;
  float o[2 * LO + 1], xo[2 * LO + 1];
#pragma unroll
  for (int i = 0; i < 2 * LO + 1; i++) {
    xo[i] = (MODE == 0 || MODE == 3) ? Xo[(uld(bo + 1) + i) * sxo] : 0.0f;
    o[i] = MODE == 0 ? xo[i] : 0.0f;
  }
  for (int p = uld(bo + 2); p < uld(bo + 3); p++) {
    const int *q = tb.pr + 4 * p;
    const float *xi = Xi + uld(q + 1) * sxi;
    float *gi = MODE == 3 ? Gi + uld(q + 1) * sgi : nullptr;
    switch (uld(q)) {
      case 0: pair_run<MODE, LO, 0>(tb, o, xo, xi, sxi, gi, sgi, uld(q + 2), uld(q + 3), O, so, D, sd); break;
      case 1: pair_run<MODE, LO, 1>(tb, o, xo, xi, sxi, gi, sgi, uld(q + 2), uld(q + 3), O, so, D, sd); break;
      case 2: pair_run<MODE, LO, 2>(tb, o, xo, xi, sxi, gi, sgi, uld(q + 2), uld(q + 3), O, so, D, sd); break;
      case 3: pair_run<MODE, LO, 3>(tb, o, xo, xi, sxi, gi, sgi, uld(q + 2), uld(q + 3), O, so, D, sd); break;
      case 4: pair_run<MODE, LO, 4>(tb, o, xo, xi, sxi, gi, sgi, uld(q + 2), uld(q + 3), O, so, D, sd); break;
      default: break;
    }
  }
  if (MODE != 0) {
#pragma unroll
    for (int i = 0; i < 2 * LO + 1; i++) Go[(uld(bo + 1) + i) * sgo] = o[i];
  }
}

// one (atom, rank): Xo / Go outer operand / adjoint row, Xi inner operand row, O output row (mode 0), D adjoint
// of the output row (modes 1 - 3); mode 3 also accumulates the inner adjoint in Gi (stride sgi, zeroed by the caller)
template<int MODE>
KOKKOS_INLINE_FUNCTION void run(const Tabs &tb, const float *Xo, long sxo, float *Go, long sgo, const float *Xi,
                                long sxi, float *O, long so, const float *D, long sd, float *Gi = nullptr,
                                long sgi = 0)
{
  for (int b = 0; b < tb.nob; b++) {
    switch (uld(tb.ob + 4 * b)) {
      case 0: block_run<MODE, 0>(tb, b, Xo, sxo, Go, sgo, Xi, sxi, Gi, sgi, O, so, D, sd); break;
      case 1: block_run<MODE, 1>(tb, b, Xo, sxo, Go, sgo, Xi, sxi, Gi, sgi, O, so, D, sd); break;
      case 2: block_run<MODE, 2>(tb, b, Xo, sxo, Go, sgo, Xi, sxi, Gi, sgi, O, so, D, sd); break;
      case 3: block_run<MODE, 3>(tb, b, Xo, sxo, Go, sgo, Xi, sxi, Gi, sgi, O, so, D, sd); break;
      case 4: block_run<MODE, 4>(tb, b, Xo, sxo, Go, sgo, Xi, sxi, Gi, sgi, O, so, D, sd); break;
      default: break;
    }
  }
}

#undef G3L_PATH_SWL
#undef G3L_PATH_CG
}   // namespace G3LPath
}    // namespace kk
}    // namespace grace

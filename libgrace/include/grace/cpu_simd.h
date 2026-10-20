// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Ported from lammps-ace feature/grace-fused src/ML-PACE/grace_cpu_simd.h (26c83c526a).
//   64-byte vectors (16 floats / 8 doubles) of the CPU GRACE kernels. With AVX-512 they are GCC / Clang vector
//   types, one register each. On narrower ISAs GCC keeps arrays of 64-byte vector types (the register tiles of the
//   kernels) on the stack and every operation round-trips memory (4-5x slower kernels with AVX2 on Zen 2, 3x with
//   SSE2), so there they are structs of native parts, which it does keep in registers: two 32-byte halves with
//   AVX / AVX2, four 16-byte parts otherwise (SSE2, the x86-64 default; also NEON). Unaligned loads and stores.

#ifndef GRACE_CPU_SIMD_H
#define GRACE_CPU_SIMD_H

#include <cstring>

namespace grace {
namespace simd {

#if defined(__AVX512F__)

typedef float f16 __attribute__((vector_size(64)));
typedef double d8 __attribute__((vector_size(64)));
inline f16 load(const float *p) { f16 v; std::memcpy(&v, p, sizeof(v)); return v; }
inline d8 load(const double *p) { d8 v; std::memcpy(&v, p, sizeof(v)); return v; }
inline void store(float *p, const f16 v) { std::memcpy(p, &v, sizeof(v)); }
inline void store(double *p, const d8 v) { std::memcpy(p, &v, sizeof(v)); }

#elif defined(__AVX__)

// two named 32-byte halves (an array member of two parts compiled 1.3x slower in situ with GCC 11)
template<class V, class S> struct Pair {
  V lo, hi;
  Pair &operator+=(const Pair &b) { lo += b.lo; hi += b.hi; return *this; }
  Pair &operator-=(const Pair &b) { lo -= b.lo; hi -= b.hi; return *this; }
  Pair &operator*=(const Pair &b) { lo *= b.lo; hi *= b.hi; return *this; }
  Pair &operator*=(S s) { lo *= s; hi *= s; return *this; }
  S operator[](int i) const
  {
    constexpr int h = sizeof(V) / sizeof(S);
    return i < h ? lo[i] : hi[i - h];
  }
};
template<class V, class S> inline Pair<V, S> operator+(const Pair<V, S> &a, const Pair<V, S> &b)
{
  return {a.lo + b.lo, a.hi + b.hi};
}
template<class V, class S> inline Pair<V, S> operator+(const Pair<V, S> &a, S s) { return {a.lo + s, a.hi + s}; }
template<class V, class S> inline Pair<V, S> operator+(S s, const Pair<V, S> &a) { return {s + a.lo, s + a.hi}; }
template<class V, class S> inline Pair<V, S> operator-(const Pair<V, S> &a, const Pair<V, S> &b)
{
  return {a.lo - b.lo, a.hi - b.hi};
}
template<class V, class S> inline Pair<V, S> operator*(const Pair<V, S> &a, const Pair<V, S> &b)
{
  return {a.lo * b.lo, a.hi * b.hi};
}
template<class V, class S> inline Pair<V, S> operator*(S s, const Pair<V, S> &a) { return {s * a.lo, s * a.hi}; }
template<class V, class S> inline Pair<V, S> operator*(const Pair<V, S> &a, S s) { return {a.lo * s, a.hi * s}; }
template<class V, class S> inline Pair<V, S> operator-(const Pair<V, S> &a) { return {-a.lo, -a.hi}; }

typedef float f8 __attribute__((vector_size(32)));
typedef double d4 __attribute__((vector_size(32)));
typedef Pair<f8, float> f16;
typedef Pair<d4, double> d8;
inline f16 load(const float *p)
{
  f16 v;
  std::memcpy(&v.lo, p, 32);
  std::memcpy(&v.hi, p + 8, 32);
  return v;
}
inline d8 load(const double *p)
{
  d8 v;
  std::memcpy(&v.lo, p, 32);
  std::memcpy(&v.hi, p + 4, 32);
  return v;
}
inline void store(float *p, const f16 v)
{
  std::memcpy(p, &v.lo, 32);
  std::memcpy(p + 8, &v.hi, 32);
}
inline void store(double *p, const d8 v)
{
  std::memcpy(p, &v.lo, 32);
  std::memcpy(p + 4, &v.hi, 32);
}

#else

// four 16-byte parts (SSE2, the x86-64 default; NEON)
constexpr int PARTS = 4;

// 64 bytes as N native vectors V of scalars S
template<class V, class S, int N> struct Parts {
  static constexpr int W = 64 / N / sizeof(S);    // scalars per part
  V p[N];
  Parts &operator+=(const Parts &b) { for (int i = 0; i < N; i++) p[i] += b.p[i]; return *this; }
  Parts &operator-=(const Parts &b) { for (int i = 0; i < N; i++) p[i] -= b.p[i]; return *this; }
  Parts &operator*=(const Parts &b) { for (int i = 0; i < N; i++) p[i] *= b.p[i]; return *this; }
  Parts &operator*=(S s) { for (int i = 0; i < N; i++) p[i] *= s; return *this; }
  S operator[](int i) const { return p[i / W][i % W]; }
};
template<class V, class S, int N> inline Parts<V, S, N> operator+(const Parts<V, S, N> &a, const Parts<V, S, N> &b)
{
  Parts<V, S, N> r;
  for (int i = 0; i < N; i++) r.p[i] = a.p[i] + b.p[i];
  return r;
}
template<class V, class S, int N> inline Parts<V, S, N> operator-(const Parts<V, S, N> &a, const Parts<V, S, N> &b)
{
  Parts<V, S, N> r;
  for (int i = 0; i < N; i++) r.p[i] = a.p[i] - b.p[i];
  return r;
}
template<class V, class S, int N> inline Parts<V, S, N> operator*(const Parts<V, S, N> &a, const Parts<V, S, N> &b)
{
  Parts<V, S, N> r;
  for (int i = 0; i < N; i++) r.p[i] = a.p[i] * b.p[i];
  return r;
}
template<class V, class S, int N> inline Parts<V, S, N> operator*(S s, const Parts<V, S, N> &a)
{
  Parts<V, S, N> r;
  for (int i = 0; i < N; i++) r.p[i] = s * a.p[i];
  return r;
}
template<class V, class S, int N> inline Parts<V, S, N> operator*(const Parts<V, S, N> &a, S s)
{
  Parts<V, S, N> r;
  for (int i = 0; i < N; i++) r.p[i] = a.p[i] * s;
  return r;
}
template<class V, class S, int N> inline Parts<V, S, N> operator+(S s, const Parts<V, S, N> &a)
{
  Parts<V, S, N> r;
  for (int i = 0; i < N; i++) r.p[i] = s + a.p[i];
  return r;
}
template<class V, class S, int N> inline Parts<V, S, N> operator+(const Parts<V, S, N> &a, S s)
{
  Parts<V, S, N> r;
  for (int i = 0; i < N; i++) r.p[i] = a.p[i] + s;
  return r;
}
template<class V, class S, int N> inline Parts<V, S, N> operator-(const Parts<V, S, N> &a)
{
  Parts<V, S, N> r;
  for (int i = 0; i < N; i++) r.p[i] = -a.p[i];
  return r;
}

typedef float fpart __attribute__((vector_size(16)));
typedef double dpart __attribute__((vector_size(16)));
typedef Parts<fpart, float, PARTS> f16;
typedef Parts<dpart, double, PARTS> d8;
template<class T, class P> inline T load_parts(const P *q)
{
  T v;
  for (int i = 0; i < PARTS; i++) std::memcpy(&v.p[i], q + T::W * i, sizeof(v.p[i]));
  return v;
}
template<class T, class P> inline void store_parts(P *q, const T &v)
{
  for (int i = 0; i < PARTS; i++) std::memcpy(q + T::W * i, &v.p[i], sizeof(v.p[i]));
}
inline f16 load(const float *p) { return load_parts<f16>(p); }
inline d8 load(const double *p) { return load_parts<d8>(p); }
inline void store(float *p, const f16 v) { store_parts(p, v); }
inline void store(double *p, const d8 v) { store_parts(p, v); }

#endif

// one hardware register of floats: a kernel whose register tile of f16 values would spill on the narrower ISAs
// runs there in lane groups of NF floats (ported from lammps-ace 72299ce02c, GRACE-3L SPBF tiles)
#if defined(__AVX512F__)
typedef f16 fn;
#elif defined(__AVX__)
typedef f8 fn;
#else
typedef fpart fn;
#endif
constexpr int NF = sizeof(fn) / sizeof(float);
inline fn loadn(const float *p) { fn v; std::memcpy(&v, p, sizeof(v)); return v; }
inline void storen(float *p, const fn v) { std::memcpy(p, &v, sizeof(v)); }

// the wider x86 vector extension this CPU supports but the code was compiled without, or nullptr
// (ported from lammps-ace ef190b09fb; the LAMMPS glue warns with it)
inline const char *isa_hint()
{
#if defined(__x86_64__) || defined(__i386__)
#if !defined(__AVX512F__)
  if (__builtin_cpu_supports("avx512f")) return "AVX-512";
#endif
#if !defined(__AVX2__)
  if (__builtin_cpu_supports("avx2")) return "AVX2";
#endif
#endif
  return nullptr;
}

}    // namespace simd
}    // namespace grace

#endif
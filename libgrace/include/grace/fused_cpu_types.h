// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Contributing author: Yury Lysogorskiy (ICAMS)
#ifndef GRACE_FUSED_CPU_TYPES_H
#define GRACE_FUSED_CPU_TYPES_H

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace grace {
namespace fused {

constexpr int NB = 4;    // atoms per batch: per-atom tensors are [function][NB][channel]

// 64-byte aligned storage: the kernels load whole cache lines (v16); std::allocator gives 16-byte alignment,
// i.e. a large buffer (mmap: page + 16) makes every 64-byte load a cache-line split
template<class T> struct Align64 {
  using value_type = T;
  Align64() = default;
  template<class U> Align64(const Align64<U> &) {}
  T *allocate(std::size_t n) { return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t(64))); }
  void deallocate(T *p, std::size_t) { ::operator delete(p, std::align_val_t(64)); }
  template<class U> bool operator==(const Align64<U> &) const { return true; }
  template<class U> bool operator!=(const Align64<U> &) const { return false; }
};
using fvec = std::vector<float, Align64<float>>;

// Shared per-atom buffer (trivial T) filled by the OpenMP threads in static chunks, so that its pages are first
// touched by the threads that use them: std::vector::assign touches them all on the calling thread, which puts
// them on one NUMA node when a rank spans several. A static schedule gives thread t the t-th contiguous chunk,
// i.e. its own slice of a [nthreads][...] buffer.
template<class T, class A = Align64<T>> class SpreadBuf {
  static_assert(std::is_trivially_copyable<T>::value, "SpreadBuf holds trivial types");
  A al;
  T *p = nullptr;
  size_t n = 0;
  void release()
  {
    if (p) al.deallocate(p, n);
    p = nullptr;
    n = 0;
  }

 public:
  SpreadBuf() = default;
  SpreadBuf(const SpreadBuf &) = delete;
  SpreadBuf &operator=(const SpreadBuf &) = delete;
  ~SpreadBuf() { release(); }
  void assign(size_t m, T v, int nthreads)
  {
    if (m != n) {
      release();
      if (m) p = al.allocate(m);
      n = m;
    }
    T *q = p;
#ifdef _OPENMP
    const int nt = std::min(nthreads > 0 ? nthreads : omp_get_max_threads(), std::max(1, omp_get_num_procs()));
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (size_t i = 0; i < m; i++) q[i] = v;
  }
  void clear() { release(); }
  T *data() { return p; }
  const T *data() const { return p; }
  size_t size() const { return n; }
  T &operator[](size_t i) { return p[i]; }
  const T &operator[](size_t i) const { return p[i]; }
};

// dst[dst_sel] + out_off[q] * NB + a * ostr  +=  sc[q] * sum_k src[src_sel][in_off[q] * NB + a * istr + k] * M[k][:]
struct MVG {
  const float *M = nullptr;
  size_t tystride = 0;
  int nk = 0, ldm = 0, src = 0, dst = 0;
  int istr = 0, ostr = 0;
  std::vector<int> in_off, out_off;
  fvec sc;
};
struct FC {
  int nil = 0, nir = 0, nout = 0, nfl = 0, ntl = 0, ntr = 0;
  std::vector<int> tile_l;
  std::vector<int> rc_to, rc_from, rc_tile;
  fvec nof;
  fvec Wl, Wr, WlT, WrT;
  std::vector<char> live;
  int pinl = 0, pinr = 0, pout = 0;
  std::vector<MVG> gf, gb;
};
struct Prod {
  int nout = 0, nl = 0, nr = 0;
  std::vector<int> l, r, o, off;
  fvec c;
};
struct Red {
  int nout = 0, nfo = 0;
  bool elem = false;
  struct Src {
    int nin = 0, nt = 0;
    std::vector<int> ci, tile, fo;
    fvec W, WT;
    int nout = 0, pin = 0;
  };
  std::vector<Src> src;
  int pout = 0;
  std::vector<MVG> gf, gb;
};

// rows of one reduce source by input function (CSR): the product producing that source is fused with the
// reduction row by row (prod_red_fwd / prod_red_bwd), so the source tensor is never stored
struct RowRed {
  int src = -1;
  std::vector<int> off, t, fo;
};

// lock of one per-atom row that several threads add into (dE/dI of the message rows)
struct SpinLock {
  std::atomic<int> *l;
  explicit SpinLock(std::atomic<int> *l_) : l(l_)
  {
    while (l->exchange(1, std::memory_order_acquire))
      while (l->load(std::memory_order_relaxed)) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
      }
  }
  ~SpinLock() { l->store(0, std::memory_order_release); }
};

// one lock per per-atom row (SpinLock lk(locks[j]))
struct RowLocks {
  std::unique_ptr<std::atomic<int>[]> l;
  void reset(size_t n)
  {
    l.reset(new std::atomic<int>[n]);
    for (size_t i = 0; i < n; i++) l[i].store(0);
  }
  std::atomic<int> *operator[](size_t i) const { return &l[i]; }
};

// per-thread force rows [nmax][3] (no atomics: neighbours of one thread's atoms are often owned by another).
// Kept all-zero between computes: reduce() adds them to atom->f in contiguous chunks and clears them.
// Each thread first touches its own rows (SpreadBuf), which it then writes all compute long.
struct ThreadForces {
  SpreadBuf<double> buf;
  int nth = 1, nmax = 0;
  void ensure(int nthreads, int nmax_)
  {
    if (nthreads != nth || nmax_ > nmax) {
      nth = 0;    // committed after the allocation (a failed one leaves the next call to allocate again)
      nmax = 0;
      buf.assign((size_t) nthreads * nmax_ * 3, 0.0, nthreads);
      nth = nthreads;
      nmax = nmax_;
    }
  }
  double *of(int tid) { return buf.data() + (size_t) tid * nmax * 3; }
  void reduce(double *f, int nall)    // f: atom->f[0] (contiguous [nmax][3])
  {
    constexpr int CH = 512;
    const int nch = (nall + CH - 1) / CH;
#ifdef _OPENMP
    const int nt = std::min(std::max(1, nth), std::max(1, omp_get_num_procs()));
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (int c = 0; c < nch; c++) {
      const size_t q0 = (size_t) 3 * c * CH, q1 = (size_t) 3 * std::min(nall, (c + 1) * CH);
      for (int t = 0; t < nth; t++) {
        double *src = of(t);
#pragma omp simd
        for (size_t q = q0; q < q1; q++) f[q] += src[q];
        std::memset(src + q0, 0, (q1 - q0) * sizeof(double));
      }
    }
  }
};

}    // namespace fused
}    // namespace grace

#endif

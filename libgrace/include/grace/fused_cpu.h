// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Contributing author: Yury Lysogorskiy (ICAMS)
#ifndef GRACE_FUSED_CPU_H
#define GRACE_FUSED_CPU_H

#include "grace/fused_cpu_types.h"
#include "grace/cpu_simd.h"

#include "grace/error.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace grace {
namespace fused {

// model-independent context of the layout builders
struct Ctx {
  int ntypes;
  const std::vector<int> *type2mu;     // LAMMPS type -> model element
  int n_elements;
};


constexpr int MAX_LMAX = 6;
constexpr int MAX_NLM = (MAX_LMAX + 1) * (MAX_LMAX + 1);

inline int pad16(int n) { return (n + 15) / 16 * 16; }

// the memory limit of the process's cgroup in MB (Slurm --mem, containers): the smallest limit of the cgroup and its
// ancestors (total) and the smallest limit - usage (avail, at least 1 MB; the usage without the inactive page cache,
// the "working set" of container tools); false, and total / avail unchanged, when no level has a limit. proc_cgroup
// and root: /proc/self/cgroup and /sys/fs/cgroup (the tests pass a fake tree). cgroup v2: the smaller of memory.max
// and memory.high, memory.current, memory.stat inactive_file; v1: the memory controller's memory.limit_in_bytes,
// memory.usage_in_bytes, memory.stat total_inactive_file (a limit near 2^63: none). Only the usual mounts are looked at
// (/sys/fs/cgroup, v1 memory under /sys/fs/cgroup/memory).
inline bool cgroup_memory_mb(const char *proc_cgroup, const std::string &root, double &total, double &avail)
{
  std::FILE *f = std::fopen(proc_cgroup, "r");
  if (!f) return false;
  std::string path;
  bool v1 = false;
  char buf[4096];
  while (std::fgets(buf, sizeof buf, f)) {    // "<id>:<controllers>:<path>"; v2: "0::<path>"
    std::string line(buf);
    while (!line.empty() && line.back() == '\n') line.pop_back();
    const size_t a = line.find(':'), b = a == std::string::npos ? a : line.find(':', a + 1);
    if (b == std::string::npos) continue;
    const std::string ctrl = "," + line.substr(a + 1, b - a - 1) + ",";
    if (ctrl.find(",memory,") != std::string::npos) {
      path = line.substr(b + 1);
      v1 = true;
    } else if (!v1 && ctrl == ",," && line.compare(0, a, "0") == 0) {
      path = line.substr(b + 1);
    }
  }
  std::fclose(f);
  // no cgroup, or a ".." component (outside the cgroup namespace)
  if (path.empty() || path[0] != '/' || (path + "/").find("/../") != std::string::npos) return false;
  const auto bytes = [](const std::string &file, double &v) {
    std::FILE *g = std::fopen(file.c_str(), "r");
    if (!g) return false;
    char w[64];
    const bool ok = std::fscanf(g, "%63s", w) == 1;
    std::fclose(g);
    const double x = ok ? std::strtod(w, nullptr) : 0.0;    // "max" -> 0
    if (!(x > 0.0 && x < 4e18)) return false;
    v = x;
    return true;
  };
  static const struct {
    const char *sub, *limit, *high, *usage, *inactive;
  } files[2] = {{"", "/memory.max", "/memory.high", "/memory.current", "inactive_file"},
                {"/memory", "/memory.limit_in_bytes", nullptr, "/memory.usage_in_bytes", "total_inactive_file"}};
  const auto &c = files[v1];
  const auto cache = [&](const std::string &dir) {    // the inactive page cache from memory.stat, 0 when absent
    double v = 0.0;
    if (std::FILE *g = std::fopen((dir + "/memory.stat").c_str(), "r")) {
      char key[64];
      double x;
      while (std::fscanf(g, "%63s %lf", key, &x) == 2)
        if (!std::strcmp(key, c.inactive)) v = x;
      std::fclose(g);
    }
    return v;
  };
  constexpr double none = 1e300;
  double t = none, a = none;
  for (std::string p = path;;) {    // the cgroup, then its ancestors up to the root
    const std::string dir = root + c.sub + (p == "/" ? "" : p);
    double lim = none, x = 0.0;
    if (bytes(dir + c.limit, x)) lim = x;
    if (c.high && bytes(dir + c.high, x)) lim = std::min(lim, x);
    if (lim < none) {
      t = std::min(t, lim);
      a = std::min(a, bytes(dir + c.usage, x) ? lim - std::max(0.0, x - cache(dir)) : lim);
    }
    if (p == "/") break;
    const size_t k = p.rfind('/');
    p = k == 0 ? "/" : p.substr(0, k);
  }
  if (t >= none) return false;
  total = t / 1048576.0;
  avail = std::max(a / 1048576.0, 1.0);    // a full cgroup: 1 MB, not 0 (host_memory_mb's "unreadable")
  return true;
}

// host memory in MB (Linux: /proc/meminfo MemTotal / MemAvailable, capped by the cgroup limit; macOS: sysctl +
// host_statistics64); 0 when unreadable
inline void host_memory_mb(double &total, double &avail)
{
  total = avail = 0.0;
#if defined(__linux__)
  if (FILE *f = std::fopen("/proc/meminfo", "r")) {
    char key[64];
    double kb;
    while (std::fscanf(f, "%63s %lf kB", key, &kb) == 2) {
      if (!std::strcmp(key, "MemTotal:")) total = kb / 1024.0;
      if (!std::strcmp(key, "MemAvailable:")) avail = kb / 1024.0;
    }
    std::fclose(f);
  }
  double ct = 0.0, ca = 0.0;
  if (cgroup_memory_mb("/proc/self/cgroup", "/sys/fs/cgroup", ct, ca)) {
    if (total <= 0.0 || ct < total) total = ct;
    if (avail <= 0.0 || ca < avail) avail = ca;
  }
#elif defined(__APPLE__)
  uint64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) total = (double) mem / 1048576.0;
  vm_size_t page_size = 0;
  mach_port_t host = mach_host_self();
  vm_statistics64_data_t vmstat;
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_page_size(host, &page_size) == KERN_SUCCESS &&
      host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vmstat), &count) == KERN_SUCCESS) {
    const uint64_t pages = (uint64_t) vmstat.free_count + vmstat.inactive_count + vmstat.speculative_count;
    avail = (double) pages * (double) page_size / 1048576.0;
  }
  mach_port_deallocate(mach_task_self(), host);
#endif
}

// size limit (GB) of a per-atom store: max_gb when >= 0, else half the available host memory per process of the node
// (8 GB where the memory query fails)
inline double store_budget_gb(double max_gb, int ranks_per_node)
{
  if (max_gb >= 0.0) return max_gb;
  double total = 0.0, avail = 0.0;
  host_memory_mb(total, avail);
  if (avail <= 0.0) return 8.0;
  return 0.5 * avail * 1048576.0 / 1e9 / std::max(1, ranks_per_node);
}

// G2LF_PROF=1 (environment): wall time per category, measured on OpenMP thread 0 only
enum { T_BOND, T_RAD1, T_FC1, T_PROD1, T_RED1, T_RAD1B, T_FC1B, T_PROD1B, T_RED1B, T_RAD2, T_FC2, T_PROD2,
       T_RED2, T_EN, T_RAD2B, T_FC2B, T_PROD2B, T_RED2B, T_P1, T_P2, T_P3, T_P4, T_P5, T_P6,
       T_UQPROD, T_UQPUSH, T_UQGEMM, T_UQGMM, T_N };
inline const char *const T_NAME[T_N] = {"bonds+Y", "L1 bond fwd", "L1 FC", "L1 prod", "L1 red", "L1 bond bwd", "L1 FC bwd",
                           "L1 prod bwd", "L1 red bwd", "L2 bond fwd", "L2 FC", "L2 prod", "L2 red", "energy",
                           "L2 bond bwd", "L2 FC bwd", "L2 prod bwd", "L2 red bwd", "P1 total", "P2 comm",
                           "P3 total", "P4 reduce+comm", "P5 total", "P6 force reduce",
                           "UQ products", "UQ gather", "UQ projection", "UQ GMM"};
inline double g_prof[T_N] = {};
inline bool g_prof_on = false;
inline double now_s()
{
#ifdef _OPENMP
  return omp_get_wtime();
#else
  return 0.0;
#endif
}
inline bool prof_thread()
{
#ifdef _OPENMP
  return g_prof_on && omp_get_thread_num() == 0;
#else
  return g_prof_on;
#endif
}
inline double g_flops[T_N] = {};    // matrix-vector flops (mv_batch) per category, thread 0
inline int g_cur = -1;
struct ProfScope {
  int id, prev; double t0; bool on;
  ProfScope(int i) : id(i), prev(g_cur), t0(0.0), on(prof_thread())
  {
    if (on) { t0 = now_s(); g_cur = id; }
  }
  void stop()
  {
    if (on) { g_prof[id] += now_s() - t0; g_cur = prev; }
    on = false;
  }
  ~ProfScope() { stop(); }
};
#define PROF(id) ProfScope _ps(id)

inline void axpy(float *__restrict y, const float *__restrict x, const float a, const int P)
{
#pragma omp simd
  for (int k = 0; k < P; k++) y[k] += a * x[k];
}

inline void axpy2(float *__restrict y, const float *__restrict x1, const float *__restrict x2, const float a,
                  const int P)
{
#pragma omp simd
  for (int k = 0; k < P; k++) y[k] += a * x1[k] * x2[k];
}

inline float dot(const float *__restrict x, const float *__restrict y, const int P)
{
  float s = 0.0f;
#pragma omp simd reduction(+ : s)
  for (int k = 0; k < P; k++) s += x[k] * y[k];
  return s;
}

inline float dot3(const float *__restrict x, const float *__restrict y, const float *__restrict z, const int P)
{
  float s = 0.0f;
#pragma omp simd reduction(+ : s)
  for (int k = 0; k < P; k++) s += x[k] * y[k] * z[k];
  return s;
}

inline void vzero(float *y, const size_t n) { std::memset(y, 0, n * sizeof(float)); }

// fp64: Chebyshev T_1..T_nrb of (2x-1) times fcut_p(x), and d/dx
inline void radial_basis_x(double x, int p, int nrb, std::vector<double> &g, std::vector<double> &dg)
{
  const double xp = std::pow(x, p), xp1 = xp * x, xp2 = xp1 * x;
  const double fc = 1.0 - 0.5 * (p + 1) * (p + 2) * xp + p * (p + 2) * xp1 - 0.5 * p * (p + 1) * xp2;
  const double dfc = 0.5 * p * (p + 1) * (p + 2) * (-std::pow(x, p - 1) + 2.0 * xp - xp1);
  const double t = 2.0 * x - 1.0;
  double Tp = 1.0, Tc = t, dTp = 0.0, dTc = 2.0;
  g.resize(nrb);
  dg.resize(nrb);
  for (int k = 0; k < nrb; k++) {
    g[k] = Tc * fc;
    dg[k] = dTc * fc + Tc * dfc;
    const double Tn = 2.0 * t * Tc - Tp, dTn = 2.0 * (2.0 * Tc + t * dTc) - dTp;
    Tp = Tc; dTp = dTc; Tc = Tn; dTc = dTn;
  }
}

// fp64 forward-mode radial MLP (silu hidden layers, linear output)
template<class Layers>
void mlp_x(const Layers &layers, std::vector<double> h, std::vector<double> dh,
           std::vector<double> &out, std::vector<double> &dout)
{
  const int n = layers.size();
  for (int il = 0; il < n; il++) {
    const auto &L = layers[il];
    std::vector<double> s(L.n_out, 0.0), ds(L.n_out, 0.0);
    for (int k = 0; k < L.n_in; k++)
      for (int j = 0; j < L.n_out; j++) {
        const double w = L.W[(size_t) k * L.n_out + j];
        s[j] += w * h[k];
        ds[j] += w * dh[k];
      }
    for (int j = 0; j < L.n_out; j++) {
      s[j] *= L.norm;
      ds[j] *= L.norm;
      if (il < n - 1) {
        const double sg = 1.0 / (1.0 + std::exp(-s[j]));
        ds[j] = sg * (1.0 + s[j] * (1.0 - sg)) * ds[j];
        s[j] = s[j] * sg;
      }
    }
    h.swap(s);
    dh.swap(ds);
  }
  out.swap(h);
  dout.swap(dh);
}



// 16-float vectors (one AVX-512 register, two halves on AVX2: grace_cpu_simd.h). Unaligned access.
typedef simd::f16 v16;
inline v16 ld(const float *p) { return simd::load(p); }
inline void st(float *p, const v16 v) { simd::store(p, v); }
typedef float v8f __attribute__((vector_size(32)));
typedef float v4f __attribute__((vector_size(16)));
// pairwise tree (the plain loop is a 15-add serial chain without -ffast-math)
inline float hsum(const v16 v)
{
  v8f a, b;
  std::memcpy(&a, &v, 32); std::memcpy(&b, reinterpret_cast<const char *>(&v) + 32, 32);
  a += b;
  v4f c, d;
  std::memcpy(&c, &a, 16); std::memcpy(&d, reinterpret_cast<const char *>(&a) + 16, 16);
  c += d;
  return (c[0] + c[2]) + (c[1] + c[3]);
}

#if defined(__aarch64__)
// NEON (aarch64): a v16 is 4 of the 32 registers and the v16 tiles below spill. o[a] += sc M^T x[a] for the NB atoms,
// 16 columns at a time (NB x 16 / NF accumulators)
template<int NV>
inline void mv_bands(const float *M, int ldm, int nk, const float *x, int is, float *o, int os, float sc)
{
  constexpr int P = 16 / simd::NF;
  for (int v = 0; v < NV; v++) {
    simd::fn acc[NB][P] = {};
    for (int k = 0; k < nk; k++) {
      const float *row = M + (size_t) k * ldm + 16 * v;
      simd::fn w[P];
      for (int p = 0; p < P; p++) w[p] = simd::loadn(row + simd::NF * p);
      for (int a = 0; a < NB; a++) {
        const float xa = x[a * is + k];
        for (int p = 0; p < P; p++) acc[a][p] += xa * w[p];
      }
    }
    for (int a = 0; a < NB; a++) {
      float *oa = o + a * os + 16 * v;
      for (int p = 0; p < P; p++) simd::storen(oa + simd::NF * p, simd::loadn(oa + simd::NF * p) + sc * acc[a][p]);
    }
  }
}
#endif

/* Matrix-vector items over a batch: for every item q and atom slot a
     out[out_off[q] * NB + a * ostr] += sc[q] * sum_k in[in_off[q] * NB + a * istr + k] * M[k][:]
   Each matrix row is loaded once per SB atoms (registers: SB * NV accumulators). */
template<int NV>
inline void mv_batch(const MVG &g, int ty, const float *const *src, float *const *dst)
{
  constexpr int SB0 = NV <= 2 ? 8 : (NV <= 4 ? 4 : 2);
  constexpr int SB = SB0 < NB ? SB0 : NB;
  const float *M = g.M + (size_t) ty * g.tystride;
  const float *S = src[g.src];
  float *D = dst[g.dst];
  const int ni = g.in_off.size(), nk = g.nk, ldm = g.ldm, is = g.istr, os = g.ostr;
  if (g_prof_on && g_cur >= 0 && prof_thread()) g_flops[g_cur] += 2.0 * ni * nk * 16 * NV * NB;
  for (int q = 0; q < ni; q++) {
    const float *x = S + (size_t) g.in_off[q] * NB;
    float *o = D + (size_t) g.out_off[q] * NB;
    const float sc = g.sc[q];
#if defined(__aarch64__)
    mv_bands<NV>(M, ldm, nk, x, is, o, os, sc);
    continue;
#endif
    for (int a0 = 0; a0 < NB; a0 += SB) {
      v16 acc[SB][NV];
      for (int a = 0; a < SB; a++)
        for (int v = 0; v < NV; v++) acc[a][v] = v16{};
      for (int k = 0; k < nk; k++) {
        const float *row = M + (size_t) k * ldm;
        v16 w[NV];
        for (int v = 0; v < NV; v++) w[v] = ld(row + 16 * v);
        for (int a = 0; a < SB; a++) {
          const float xa = x[(a0 + a) * is + k];
          for (int v = 0; v < NV; v++) acc[a][v] += xa * w[v];
        }
      }
      for (int a = 0; a < SB; a++) {
        float *oa = o + (a0 + a) * os;
        for (int v = 0; v < NV; v++) st(oa + 16 * v, ld(oa + 16 * v) + sc * acc[a][v]);
      }
    }
  }
}

inline void mv_groups(const std::vector<MVG> &gs, int ty, const float *const *src, float *const *dst)
{
  for (const auto &g : gs) {
    switch (g.ldm / 16) {
      case 1: mv_batch<1>(g, ty, src, dst); break;
      case 2: mv_batch<2>(g, ty, src, dst); break;
      case 3: mv_batch<3>(g, ty, src, dst); break;
      case 4: mv_batch<4>(g, ty, src, dst); break;
      case 5: mv_batch<5>(g, ty, src, dst); break;
      case 6: mv_batch<6>(g, ty, src, dst); break;
      case 7: mv_batch<7>(g, ty, src, dst); break;
      case 8: mv_batch<8>(g, ty, src, dst); break;
      default: break;    // rejected at init
    }
  }
}

// out[f] = nof[f] (Wl[tile_l f] . Xl[f] + sum_c->f Wr[tile c] . Xr[from c])   (out zeroed here)
inline void fc_fwd(const FC &F, const float *Xl, const float *Xr, float *out)
{
  vzero(out, (size_t) F.nfl * NB * F.pout);
  const float *src[2] = {Xl, Xr};
  float *dst[1] = {out};
  mv_groups(F.gf, 0, src, dst);
}

// gXl += ..., gXr += ...
inline void fc_bwd(const FC &F, const float *gout, float *gXl, float *gXr)
{
  const float *src[1] = {gout};
  float *dst[2] = {gXl, gXr};
  mv_groups(F.gb, 0, src, dst);
}

#if defined(__aarch64__)
// NEON: the products in chunks of PROD_CH v16 values (NR is a multiple of 4: G2LF_NR_DISPATCH)
constexpr int PROD_CH = 4;

// dst = sum_t c_t L[l t] R[r t] over the terms t of output o
template<int NR>
inline void prod_fwd_chunks(const Prod &p, int o, const float *L, const float *R, float *dst)
{
  static_assert(NR % PROD_CH == 0, "whole chunks");
  constexpr int RL = 16 * NR;
  for (int v0 = 0; v0 < NR; v0 += PROD_CH) {
    v16 acc[PROD_CH] = {};
    for (int t = p.off[o]; t < p.off[o + 1]; t++) {
      const float *l = L + (size_t) p.l[t] * RL + 16 * v0, *r = R + (size_t) p.r[t] * RL + 16 * v0;
      const float c = p.c[t];
      for (int v = 0; v < PROD_CH; v++) acc[v] += c * ld(l + 16 * v) * ld(r + 16 * v);
    }
    for (int v = 0; v < PROD_CH; v++) st(dst + 16 * (v0 + v), acc[v]);
  }
}

// gL[l t] += c_t d * R[r t], gR[r t] += c_t d * L[l t] for the terms t of output o, d = the row's gradient
template<int NR>
inline void prod_bwd_chunks(const Prod &p, int o, const float *L, const float *R, const float *d0, float *gL, float *gR)
{
  static_assert(NR % PROD_CH == 0, "whole chunks");
  constexpr int RL = 16 * NR;
  for (int v0 = 0; v0 < NR; v0 += PROD_CH) {
    v16 d[PROD_CH];
    for (int v = 0; v < PROD_CH; v++) d[v] = ld(d0 + 16 * (v0 + v));
    for (int t = p.off[o]; t < p.off[o + 1]; t++) {
      const size_t lo = (size_t) p.l[t] * RL + 16 * v0, ro = (size_t) p.r[t] * RL + 16 * v0;
      const float c = p.c[t];
      for (int v = 0; v < PROD_CH; v++) {
        const v16 cd = c * d[v];
        st(gL + lo + 16 * v, ld(gL + lo + 16 * v) + cd * ld(R + ro + 16 * v));
        st(gR + ro + 16 * v, ld(gR + ro + 16 * v) + cd * ld(L + lo + 16 * v));
      }
    }
  }
}
#endif

// CG products on batch rows of length RL = NB * P = 16 * NR: whole rows per term (term indices read once; NEON:
// once per chunk)
template<int NR>
inline void prod_fwd_t(const Prod &p, const float *L, const float *R, float *out)
{
  constexpr int RL = 16 * NR;
  for (int o = 0; o < p.nout; o++) {
#if defined(__aarch64__)
    prod_fwd_chunks<NR>(p, o, L, R, out + (size_t) o * RL);
    continue;
#endif
    v16 acc[NR];
    for (int v = 0; v < NR; v++) acc[v] = v16{};
    for (int t = p.off[o]; t < p.off[o + 1]; t++) {
      const float *l = L + (size_t) p.l[t] * RL, *r = R + (size_t) p.r[t] * RL;
      const float c = p.c[t];
      for (int v = 0; v < NR; v++) acc[v] += c * ld(l + 16 * v) * ld(r + 16 * v);
    }
    for (int v = 0; v < NR; v++) st(out + (size_t) o * RL + 16 * v, acc[v]);
  }
}

template<int NR>
inline void prod_bwd_t(const Prod &p, const float *L, const float *R, const float *g, float *gL, float *gR)
{
  constexpr int RL = 16 * NR;
  for (int o = 0; o < p.nout; o++) {
    if (p.off[o] == p.off[o + 1]) continue;
#if defined(__aarch64__)
    prod_bwd_chunks<NR>(p, o, L, R, g + (size_t) o * RL, gL, gR);
    continue;
#endif
    v16 d[NR];
    for (int v = 0; v < NR; v++) d[v] = ld(g + (size_t) o * RL + 16 * v);
    for (int t = p.off[o]; t < p.off[o + 1]; t++) {
      const size_t lo = (size_t) p.l[t] * RL, ro = (size_t) p.r[t] * RL;
      const float c = p.c[t];
      for (int v = 0; v < NR; v++) {
        const v16 cd = c * d[v];
        st(gL + lo + 16 * v, ld(gL + lo + 16 * v) + cd * ld(R + ro + 16 * v));
        st(gR + ro + 16 * v, ld(gR + ro + 16 * v) + cd * ld(L + lo + 16 * v));
      }
    }
  }
}

#define G2LF_NR_DISPATCH(RL, CALL)                                                                 \
  switch ((RL) / 16) {                                                                             \
    case 4: CALL(4); break;   case 8: CALL(8); break;   case 12: CALL(12); break;                 \
    case 16: CALL(16); break; case 20: CALL(20); break; case 24: CALL(24); break;                 \
    case 28: CALL(28); break; case 32: CALL(32); break;                                           \
    default: break;                                                                                \
  }

inline void prod_fwd(const Prod &p, const float *L, const float *R, float *out, int P)
{
#define CALL(N) prod_fwd_t<N>(p, L, R, out)
  G2LF_NR_DISPATCH(NB * P, CALL)
#undef CALL
}

inline void prod_bwd(const Prod &p, const float *L, const float *R, const float *g, float *gL, float *gR, int P)
{
#define CALL(N) prod_bwd_t<N>(p, L, R, g, gL, gR)
  G2LF_NR_DISPATCH(NB * P, CALL)
#undef CALL
}

// o[a] += M^T x[a] for the NB atoms of one row (x: stride is, o: stride os), M [nk][16 NV]
template<int NV>
inline void mv_item(const float *M, int nk, const float *x, int is, float *o, int os)
{
  constexpr int SB0 = NV <= 4 ? 4 : 2;
  constexpr int SB = SB0 < NB ? SB0 : NB;
  constexpr int ldm = 16 * NV;
#if defined(__aarch64__)
  mv_bands<NV>(M, ldm, nk, x, is, o, os, 1.0f);
  return;
#endif
  for (int a0 = 0; a0 < NB; a0 += SB) {
    v16 acc[SB][NV];
    for (int a = 0; a < SB; a++)
      for (int v = 0; v < NV; v++) acc[a][v] = v16{};
    for (int k = 0; k < nk; k++) {
      const float *row = M + (size_t) k * ldm;
      v16 w[NV];
      for (int v = 0; v < NV; v++) w[v] = ld(row + 16 * v);
      for (int a = 0; a < SB; a++) {
        const float xa = x[(a0 + a) * is + k];
        for (int v = 0; v < NV; v++) acc[a][v] += xa * w[v];
      }
    }
    for (int a = 0; a < SB; a++) {
      float *oa = o + (a0 + a) * os;
      for (int v = 0; v < NV; v++) st(oa + 16 * v, ld(oa + 16 * v) + acc[a][v]);
    }
  }
}

inline void mv_item_d(int ldm, const float *M, int nk, const float *x, int is, float *o, int os)
{
  switch (ldm / 16) {
    case 1: mv_item<1>(M, nk, x, is, o, os); break;
    case 2: mv_item<2>(M, nk, x, is, o, os); break;
    case 3: mv_item<3>(M, nk, x, is, o, os); break;
    case 4: mv_item<4>(M, nk, x, is, o, os); break;
    case 5: mv_item<5>(M, nk, x, is, o, os); break;
    case 6: mv_item<6>(M, nk, x, is, o, os); break;
    case 7: mv_item<7>(M, nk, x, is, o, os); break;
    case 8: mv_item<8>(M, nk, x, is, o, os); break;
    default: break;
  }
}

// moves source s of R from the grouped mv path to a row CSR (nrow input functions)
inline void build_rowred(RowRed &rr, Red &R, int s, int nrow)
{
  const auto &S = R.src[s];
  rr.src = s;
  rr.off.assign(nrow + 1, 0);
  for (int ci : S.ci) rr.off[ci + 1]++;
  for (int q = 0; q < nrow; q++) rr.off[q + 1] += rr.off[q];
  rr.t.assign(S.ci.size(), 0);
  rr.fo.assign(S.ci.size(), 0);
  std::vector<int> pos(rr.off.begin(), rr.off.end() - 1);
  for (size_t c = 0; c < S.ci.size(); c++) {
    const int q = pos[S.ci[c]]++;
    rr.t[q] = S.tile[c];
    rr.fo[q] = S.fo[c];
  }
  std::vector<MVG> gf, gb;
  for (auto &g : R.gf) if (g.src != s) gf.push_back(g);
  for (auto &g : R.gb) if (g.dst != s) gb.push_back(g);
  R.gf.swap(gf);
  R.gb.swap(gb);
}

// product rows computed one at a time and reduced at once: outs[k] += W_k X_row for the reduces k
template<int NR>
inline void prod_red_fwd_t(const Prod &p, const float *L, const float *Rt, int ty, const Red *const *reds,
                           const RowRed *const *rrs, int nred, float *const *outs, float *row)
{
  constexpr int RL = 16 * NR;
  for (int o = 0; o < p.nout; o++) {
    if (p.off[o] == p.off[o + 1]) continue;
    bool any = false;
    for (int k = 0; k < nred; k++) any |= rrs[k]->off[o] < rrs[k]->off[o + 1];
    if (!any) continue;
#if defined(__aarch64__)
    prod_fwd_chunks<NR>(p, o, L, Rt, row);
#else
    v16 acc[NR];
    for (int v = 0; v < NR; v++) acc[v] = v16{};
    for (int t = p.off[o]; t < p.off[o + 1]; t++) {
      const float *l = L + (size_t) p.l[t] * RL, *r = Rt + (size_t) p.r[t] * RL;
      const float c = p.c[t];
      for (int v = 0; v < NR; v++) acc[v] += c * ld(l + 16 * v) * ld(r + 16 * v);
    }
    for (int v = 0; v < NR; v++) st(row + 16 * v, acc[v]);
#endif
    for (int k = 0; k < nred; k++) {
      const Red &R = *reds[k];
      const RowRed &rr = *rrs[k];
      const auto &S = R.src[rr.src];
      const size_t tyo = R.elem ? (size_t) ty : 0;
      for (int c = rr.off[o]; c < rr.off[o + 1]; c++)
        mv_item_d(R.pout, S.W.data() + (tyo * S.nt + rr.t[c]) * S.nin * R.pout, S.nin, row, S.pin,
                  outs[k] + (size_t) rr.fo[c] * NB * R.pout, R.pout);
    }
  }
}

// reverse: row gradient from the reduces (W^T gOut), then the product reverse of that row
template<int NR>
inline void prod_red_bwd_t(const Prod &p, const float *L, const float *Rt, int ty, const Red *const *reds,
                           const RowRed *const *rrs, int nred, const float *const *gouts, float *gL, float *gR,
                           float *row)
{
  constexpr int RL = 16 * NR;
  for (int o = 0; o < p.nout; o++) {
    if (p.off[o] == p.off[o + 1]) continue;
    bool any = false;
    for (int k = 0; k < nred; k++) any |= rrs[k]->off[o] < rrs[k]->off[o + 1];
    if (!any) continue;
    vzero(row, RL);
    for (int k = 0; k < nred; k++) {
      const Red &R = *reds[k];
      const RowRed &rr = *rrs[k];
      const auto &S = R.src[rr.src];
      const size_t tyo = R.elem ? (size_t) ty : 0;
      for (int c = rr.off[o]; c < rr.off[o + 1]; c++)
        mv_item_d(S.pin, S.WT.data() + (tyo * S.nt + rr.t[c]) * R.pout * S.pin, S.nout,
                  gouts[k] + (size_t) rr.fo[c] * NB * R.pout, R.pout, row, S.pin);
    }
#if defined(__aarch64__)
    prod_bwd_chunks<NR>(p, o, L, Rt, row, gL, gR);
    continue;
#endif
    v16 d[NR];
    for (int v = 0; v < NR; v++) d[v] = ld(row + 16 * v);
    for (int t = p.off[o]; t < p.off[o + 1]; t++) {
      const size_t lo = (size_t) p.l[t] * RL, ro = (size_t) p.r[t] * RL;
      const float c = p.c[t];
      for (int v = 0; v < NR; v++) {
        const v16 cd = c * d[v];
        st(gL + lo + 16 * v, ld(gL + lo + 16 * v) + cd * ld(Rt + ro + 16 * v));
        st(gR + ro + 16 * v, ld(gR + ro + 16 * v) + cd * ld(L + lo + 16 * v));
      }
    }
  }
}

inline void prod_red_fwd(const Prod &p, const float *L, const float *R, int P, int ty, const Red *const *reds,
                         const RowRed *const *rrs, int nred, float *const *outs, float *row)
{
#define CALL(N) prod_red_fwd_t<N>(p, L, R, ty, reds, rrs, nred, outs, row)
  G2LF_NR_DISPATCH(NB * P, CALL)
#undef CALL
}

inline void prod_red_bwd(const Prod &p, const float *L, const float *R, int P, int ty, const Red *const *reds,
                         const RowRed *const *rrs, int nred, const float *const *gouts, float *gL, float *gR, float *row)
{
#define CALL(N) prod_red_bwd_t<N>(p, L, R, ty, reds, rrs, nred, gouts, gL, gR, row)
  G2LF_NR_DISPATCH(NB * P, CALL)
#undef CALL
}

// Out[nfo][NB][pout] = sum_s sum_c W_s[ty][tile c] X_s[ci c]   (Out zeroed here)
inline void red_fwd(const Red &R, int ty, const float *const *X, float *Out)
{
  vzero(Out, (size_t) R.nfo * NB * R.pout);
  float *dst[1] = {Out};
  mv_groups(R.gf, R.elem ? ty : 0, X, dst);
}

inline void red_bwd(const Red &R, int ty, const float *gOut, float *const *gX)
{
  const float *src[1] = {gOut};
  mv_groups(R.gb, R.elem ? ty : 0, src, gX);
}

#define G2LF_NVD(P, ...)                                                   \
  switch ((P) / 16) {                                                      \
    case 1: { constexpr int NV = 1; __VA_ARGS__; } break;                  \
    case 2: { constexpr int NV = 2; __VA_ARGS__; } break;                  \
    case 3: { constexpr int NV = 3; __VA_ARGS__; } break;                  \
    case 4: { constexpr int NV = 4; __VA_ARGS__; } break;                  \
    case 5: { constexpr int NV = 5; __VA_ARGS__; } break;                  \
    case 6: { constexpr int NV = 6; __VA_ARGS__; } break;                  \
    case 7: { constexpr int NV = 7; __VA_ARGS__; } break;                  \
    case 8: { constexpr int NV = 8; __VA_ARGS__; } break;                  \
    default: break;                                                        \
  }

// ---- bond geometry: neighbours within the cutoff, unit vectors, real spherical harmonics Y[lm] and the
// gradient G[lm][3] of their polynomial extension (dual numbers). Shared by the CPU GRACE-1L and 2L drivers.

inline void sh_coefs(int lmax, std::vector<double> &a, std::vector<double> &b, std::vector<double> &c,
                     std::vector<double> &d)
{
  a.assign((MAX_LMAX + 1) * (MAX_LMAX + 2) / 2, 0.0);
  b.assign(a.size(), 0.0);
  c.assign(MAX_LMAX + 1, 0.0);
  d.assign(MAX_LMAX + 1, 0.0);
  for (int l = 2; l <= lmax; l++) {
    for (int mm = 0; mm < l - 1; mm++) {
      const double l2 = l * l, m2 = mm * mm, lm1 = l - 1;
      a[l * (l + 1) / 2 + mm] = std::sqrt((4.0 * l2 - 1.0) / (l2 - m2));
      b[l * (l + 1) / 2 + mm] = -std::sqrt((lm1 * lm1 - m2) / (4.0 * lm1 * lm1 - 1.0));
    }
    c[l] = -std::sqrt(1.0 + 0.5 / l);
    d[l] = std::sqrt(2.0 * l + 1.0);
  }
}

// spherical-harmonics coefficients of sh_block (bond gathering itself lives in grace/bonds.h)
struct SHCoefs {
  int lmax, nlm;
  const double *sh_a, *sh_b, *sh_c, *sh_d;
};

// SH of 8 bonds at a time (double lanes, k innermost), written to Y [nb][nlm] (and G [nb][nlm][3])
template<bool GRAD>
inline void sh_block(const SHCoefs &g, int n, const double *ux, const double *uy, const double *uz, float *Y, float *G)
{
  constexpr int B = 8;
  constexpr int NP = (MAX_LMAX + 1) * (MAX_LMAX + 2) / 2;
  const int nlm = g.nlm;
  alignas(64) double rx[B], ry[B], rz[B], plm[NP][B], dplm[NP][B];
  alignas(64) float Yt[MAX_NLM][B], Gt[GRAD ? MAX_NLM : 1][3][B];
  for (int k = 0; k < B; k++) {
    rx[k] = k < n ? ux[3 * k] : 0.0; ry[k] = k < n ? uy[3 * k] : 0.0; rz[k] = k < n ? uz[3 * k] : 1.0;
  }
#pragma omp simd
  for (int k = 0; k < B; k++) { plm[0][k] = 1.0; dplm[0][k] = 0.0; }
  if (g.lmax > 0) {
    const double s3 = std::sqrt(3.0), s15 = -std::sqrt(1.5);
#pragma omp simd
    for (int k = 0; k < B; k++) {
      plm[1][k] = s3 * rz[k]; dplm[1][k] = s3;
      plm[2][k] = s15;        dplm[2][k] = 0.0;
    }
    for (int l = 2; l <= g.lmax; l++) {
      for (int mm = 0; mm < l - 1; mm++) {
        const int id = l * (l + 1) / 2 + mm, i1 = (l - 1) * l / 2 + mm, i2 = (l - 2) * (l - 1) / 2 + mm;
        const double ca = g.sh_a[id], cb = g.sh_b[id];
#pragma omp simd
        for (int k = 0; k < B; k++) {
          plm[id][k] = ca * (rz[k] * plm[i1][k] + cb * plm[i2][k]);
          if (GRAD) dplm[id][k] = ca * (plm[i1][k] + rz[k] * dplm[i1][k] + cb * dplm[i2][k]);
        }
      }
      const int prev = (l - 1) * l / 2 + l - 1, i0 = l * (l + 1) / 2 + l - 1, i1 = l * (l + 1) / 2 + l;
      const double sd = g.sh_d[l], sc = g.sh_c[l];
#pragma omp simd
      for (int k = 0; k < B; k++) {
        plm[i0][k] = sd * plm[prev][k] * rz[k];
        plm[i1][k] = sc * plm[prev][k];
        if (GRAD) {
          dplm[i0][k] = sd * (dplm[prev][k] * rz[k] + plm[prev][k]);
          dplm[i1][k] = sc * dplm[prev][k];
        }
      }
    }
  }
  const double sq2 = std::sqrt(2.0);
  for (int l = 0; l <= g.lmax; l++) {
    const int c = l * (l + 1), p0 = l * (l + 1) / 2;
    alignas(64) double pr[B], pi[B], prx[B], pry[B], pix[B], piy[B];
#pragma omp simd
    for (int k = 0; k < B; k++) {
      Yt[c][k] = (float) plm[p0][k];
      if (GRAD) { Gt[c][0][k] = 0.0f; Gt[c][1][k] = 0.0f; Gt[c][2][k] = (float) dplm[p0][k]; }
      pr[k] = rx[k]; pi[k] = ry[k]; prx[k] = 1.0; pry[k] = 0.0; pix[k] = 0.0; piy[k] = 1.0;
    }
    for (int mm = 1; mm <= l; mm++) {
      const double sg = (mm % 2 == 0) ? sq2 : -sq2;
#pragma omp simd
      for (int k = 0; k < B; k++) {
        const double pv = plm[p0 + mm][k];
        Yt[c + mm][k] = (float) (sg * pr[k] * pv);
        Yt[c - mm][k] = (float) (sg * pi[k] * pv);
        if (GRAD) {
          const double dpv = dplm[p0 + mm][k];
          Gt[c + mm][0][k] = (float) (sg * prx[k] * pv);
          Gt[c + mm][1][k] = (float) (sg * pry[k] * pv);
          Gt[c + mm][2][k] = (float) (sg * pr[k] * dpv);
          Gt[c - mm][0][k] = (float) (sg * pix[k] * pv);
          Gt[c - mm][1][k] = (float) (sg * piy[k] * pv);
          Gt[c - mm][2][k] = (float) (sg * pi[k] * dpv);
          const double nrx = prx[k] * rx[k] + pr[k] - pix[k] * ry[k], nry = pry[k] * rx[k] - piy[k] * ry[k] - pi[k];
          const double nix = prx[k] * ry[k] + pix[k] * rx[k] + pi[k], niy = pry[k] * ry[k] + pr[k] + piy[k] * rx[k];
          prx[k] = nrx; pry[k] = nry; pix[k] = nix; piy[k] = niy;
        }
        const double nr = pr[k] * rx[k] - pi[k] * ry[k], ni = pr[k] * ry[k] + pi[k] * rx[k];
        pr[k] = nr; pi[k] = ni;
      }
    }
  }
  for (int k = 0; k < n; k++) {
    float *y = Y + (size_t) k * nlm;
    for (int lm = 0; lm < nlm; lm++) y[lm] = Yt[lm][k];
    if (GRAD) {
      float *gg = G + (size_t) k * nlm * 3;
      for (int lm = 0; lm < nlm; lm++) {
        gg[3 * lm] = Gt[lm][0][k]; gg[3 * lm + 1] = Gt[lm][1][k]; gg[3 * lm + 2] = Gt[lm][2][k];
      }
    }
  }
}


// ---- per-bond kernels, NV = channels / 16. R, dR: [nl][16 NV] spline value and d/dx; z: chemical
// channel weight; batch tensors are addressed at the atom's slot with row stride rs = NB * 16 NV.

// X[lm] += Y[lm] (R[l] z)   (A for layer 1, B0 for layer 2)
template<int NV>
inline void bond_scatter_t(const float *R, const float *z, const float *Y, int nl, float *X, int rs)
{
  constexpr int P = 16 * NV;
  v16 zz[NV];
  for (int v = 0; v < NV; v++) zz[v] = ld(z + 16 * v);
  for (int l = 0; l < nl; l++) {
    v16 rz[NV];
    for (int v = 0; v < NV; v++) rz[v] = ld(R + l * P + 16 * v) * zz[v];
    for (int lm = l * l; lm < (l + 1) * (l + 1); lm++) {
      const float y = Y[lm];
      float *x = X + (size_t) lm * rs;
      for (int v = 0; v < NV; v++) st(x + 16 * v, ld(x + 16 * v) + y * rz[v]);
    }
  }
}

// YIp[p] += Y[y_p] R[l_p] I_j[i_p]
template<int NV>
inline void bond_yi_fwd_t(const float *R, const float *Y, const float *Ij, int npair, const int *py, const int *pl,
                   const int *pi, float *YIp, int rs)
{
  constexpr int P = 16 * NV;
  for (int p = 0; p < npair; p++) {
    const float y = Y[py[p]];
    const float *r = R + pl[p] * P, *ij = Ij + pi[p] * P;
    float *o = YIp + (size_t) p * rs;
    for (int v = 0; v < NV; v++) st(o + 16 * v, ld(o + 16 * v) + y * ld(r + 16 * v) * ld(ij + 16 * v));
  }
}

// reverse of bond_scatter: gY[lm] += gX[lm].(R z), gR[l] += z sum_lm Y[lm] gX[lm]
template<int NV>
inline void bond_scatter_bwd_t(const float *R, const float *z, const float *Y, int nl, const float *gX, int rs, v16 *gYv,
                        v16 *gR)
{
  constexpr int P = 16 * NV;
  v16 zz[NV];
  for (int v = 0; v < NV; v++) zz[v] = ld(z + 16 * v);
  for (int l = 0; l < nl; l++) {
    v16 rz[NV], acc[NV];
    for (int v = 0; v < NV; v++) { rz[v] = ld(R + l * P + 16 * v) * zz[v]; acc[v] = v16{}; }
    for (int lm = l * l; lm < (l + 1) * (l + 1); lm++) {
      const float y = Y[lm];
      const float *g = gX + (size_t) lm * rs;
      v16 s = v16{};
      for (int v = 0; v < NV; v++) {
        const v16 gv = ld(g + 16 * v);
        s += gv * rz[v];
        acc[v] += y * gv;
      }
      gYv[lm] += s;
    }
    for (int v = 0; v < NV; v++) gR[l * NV + v] += acc[v] * zz[v];
  }
}

// reverse of bond_yi_fwd: gY[y] += gYIp[p].(R[l] I_j[i]), gR[l] += Y gYIp[p] I_j[i], gI_j[i] += Y gYIp[p] R[l]
template<int NV>
inline void bond_yi_bwd_t(const float *R, const float *Y, const float *Ij, int npair, const int *py, const int *pl,
                   const int *pi, const float *gYIp, int rs, v16 *gYv, v16 *gR, float *gIj)
{
  constexpr int P = 16 * NV;
  for (int p = 0; p < npair; p++) {
    const int yy = py[p], l = pl[p];
    const float y = Y[yy];
    const float *r = R + l * P, *ij = Ij + pi[p] * P, *g = gYIp + (size_t) p * rs;
    float *gi = gIj + pi[p] * P;
    v16 s = v16{};
    for (int v = 0; v < NV; v++) {
      const v16 gv = ld(g + 16 * v), rv = ld(r + 16 * v), iv = ld(ij + 16 * v);
      s += gv * rv * iv;
      gR[l * NV + v] += y * gv * iv;
      st(gi + 16 * v, ld(gi + 16 * v) + y * gv * rv);
    }
    gYv[yy] += s;
  }
}

// grouped by (l, I): YIp[p] += Y[y_p] (R[l] I_j[I])
template<int NV>
inline void bond_yi_fwd_g(const float *R, const float *Y, const float *Ij, int ng, const int *goff, const int *gl,
                   const int *gi, const int *py, float *YIp, int rs)
{
  constexpr int P = 16 * NV;
  for (int g = 0; g < ng; g++) {
    v16 ri[NV];
    for (int v = 0; v < NV; v++) ri[v] = ld(R + gl[g] * P + 16 * v) * ld(Ij + gi[g] * P + 16 * v);
    for (int p = goff[g]; p < goff[g + 1]; p++) {
      const float y = Y[py[p]];
      float *o = YIp + (size_t) p * rs;
      for (int v = 0; v < NV; v++) st(o + 16 * v, ld(o + 16 * v) + y * ri[v]);
    }
  }
}

// reverse, per group S = sum_p Y[y_p] gYIp[p]:  gY[y_p] += gYIp[p].(R[l] I_j), gR[l] += S I_j, dIg[g] = S R[l]
// (bond_yi_commit adds dIg into gI_j)
template<int NV>
inline void bond_yi_bwd_g(const float *R, const float *Y, const float *Ij, int ng, const int *goff, const int *gl,
                   const int *gi, const int *py, const float *gYIp, int rs, v16 *gYv, v16 *gR, float *dIg)
{
  constexpr int P = 16 * NV;
  for (int g = 0; g < ng; g++) {
    const int l = gl[g];
    const float *r = R + l * P, *ij = Ij + gi[g] * P;
    v16 rv[NV], iv[NV], ri[NV], S[NV];
    for (int v = 0; v < NV; v++) {
      rv[v] = ld(r + 16 * v); iv[v] = ld(ij + 16 * v);
      ri[v] = rv[v] * iv[v]; S[v] = v16{};
    }
    for (int p = goff[g]; p < goff[g + 1]; p++) {
      const float y = Y[py[p]];
      const float *gp = gYIp + (size_t) p * rs;
      v16 s = v16{};
      for (int v = 0; v < NV; v++) {
        const v16 gv = ld(gp + 16 * v);
        s += gv * ri[v];
        S[v] += y * gv;
      }
      gYv[py[p]] += s;
    }
    for (int v = 0; v < NV; v++) {
      gR[l * NV + v] += S[v] * iv[v];
      st(dIg + (size_t) g * P + 16 * v, S[v] * rv[v]);
    }
  }
}

// gI_j[I] += dIg[g] of bond_yi_bwd_g, group by group
template<int NV> inline void bond_yi_commit(int ng, const int *gi, const float *dIg, float *gIj)
{
  constexpr int P = 16 * NV;
  for (int g = 0; g < ng; g++) {
    float *gI = gIj + gi[g] * P;
    for (int v = 0; v < NV; v++) st(gI + 16 * v, ld(gI + 16 * v) + ld(dIg + (size_t) g * P + 16 * v));
  }
}

// ---- atom-level forward accumulation, bonds innermost, outputs of one (l) or (l, I) group in registers.
// Rb: [nb][nl][P] radial values of the atom's bonds, Yb: [nb][nlm].

// X[l*l + m] = sum_b Y_b[l*l + m] (R_b[l] z_b),  m < K = 2l + 1   (assigns)
template<int NV, int K>
inline void atom_scatter_l(int nb, int l, int nl, int nlm, const float *Rb, const float *const *zb, const float *Yb,
                    float *X, int rs)
{
  constexpr int P = 16 * NV;
  v16 acc[K][NV];
  for (int m = 0; m < K; m++)
    for (int v = 0; v < NV; v++) acc[m][v] = v16{};
  for (int b = 0; b < nb; b++) {
    v16 rz[NV];
    for (int v = 0; v < NV; v++) rz[v] = ld(Rb + ((size_t) b * nl + l) * P + 16 * v) * ld(zb[b] + 16 * v);
    const float *y = Yb + (size_t) b * nlm + l * l;
    for (int m = 0; m < K; m++)
      for (int v = 0; v < NV; v++) acc[m][v] += y[m] * rz[v];
  }
  for (int m = 0; m < K; m++)
    for (int v = 0; v < NV; v++) st(X + (size_t) (l * l + m) * rs + 16 * v, acc[m][v]);
}

template<int NV>
inline void atom_scatter(int nb, int nl, int nlm, const float *Rb, const float *const *zb, const float *Yb, float *X, int rs)
{
  for (int l = 0; l < nl; l++) {
    switch (2 * l + 1) {
      case 1: atom_scatter_l<NV, 1>(nb, l, nl, nlm, Rb, zb, Yb, X, rs); break;
      case 3: atom_scatter_l<NV, 3>(nb, l, nl, nlm, Rb, zb, Yb, X, rs); break;
      case 5: atom_scatter_l<NV, 5>(nb, l, nl, nlm, Rb, zb, Yb, X, rs); break;
      case 7: atom_scatter_l<NV, 7>(nb, l, nl, nlm, Rb, zb, Yb, X, rs); break;
      case 9: atom_scatter_l<NV, 9>(nb, l, nl, nlm, Rb, zb, Yb, X, rs); break;
      case 11: atom_scatter_l<NV, 11>(nb, l, nl, nlm, Rb, zb, Yb, X, rs); break;
      case 13: atom_scatter_l<NV, 13>(nb, l, nl, nlm, Rb, zb, Yb, X, rs); break;
      default: break;
    }
  }
}

// YIp[p0 + m] = sum_b Y_b[y_m] (R_b[l] I_jb[I]),  m < K   (assigns)
template<int NV, int K>
inline void atom_yi_group(int nb, int l, int I, int nl, int nlm, const float *Rb, const float *const *Ijb, const float *Yb,
                   const int *py, float *YIp, int rs)
{
  constexpr int P = 16 * NV;
  v16 acc[K][NV];
  for (int m = 0; m < K; m++)
    for (int v = 0; v < NV; v++) acc[m][v] = v16{};
  for (int b = 0; b < nb; b++) {
    v16 ri[NV];
    for (int v = 0; v < NV; v++)
      ri[v] = ld(Rb + ((size_t) b * nl + l) * P + 16 * v) * ld(Ijb[b] + (size_t) I * P + 16 * v);
    const float *y = Yb + (size_t) b * nlm;
    for (int m = 0; m < K; m++) {
      const float ym = y[py[m]];
      for (int v = 0; v < NV; v++) acc[m][v] += ym * ri[v];
    }
  }
  for (int m = 0; m < K; m++)
    for (int v = 0; v < NV; v++) st(YIp + (size_t) m * rs + 16 * v, acc[m][v]);
}

template<int NV>
inline void atom_yi(int nb, int nl, int nlm, const float *Rb, const float *const *Ijb, const float *Yb, int ng,
             const int *goff, const int *gl, const int *gi, const int *py, float *YIp, int rs)
{
  for (int g = 0; g < ng; g++) {
    const int p0 = goff[g], k = goff[g + 1] - p0;
    float *o = YIp + (size_t) p0 * rs;
#define G2LF_YIG(K) atom_yi_group<NV, K>(nb, gl[g], gi[g], nl, nlm, Rb, Ijb, Yb, py + p0, o, rs)
    switch (k) {
      case 1: G2LF_YIG(1); break;   case 2: G2LF_YIG(2); break;   case 3: G2LF_YIG(3); break;
      case 4: G2LF_YIG(4); break;   case 5: G2LF_YIG(5); break;   case 6: G2LF_YIG(6); break;
      case 7: G2LF_YIG(7); break;   case 8: G2LF_YIG(8); break;   case 9: G2LF_YIG(9); break;
      case 10: G2LF_YIG(10); break; case 11: G2LF_YIG(11); break; case 12: G2LF_YIG(12); break;
      case 13: G2LF_YIG(13); break;
      default: break;
    }
#undef G2LF_YIG
  }
}

template<int NV>
inline double bond_dEdx_t(const v16 *gR, const float *dR, int nl)
{
  constexpr int P = 16 * NV;
  v16 s = v16{};
  for (int l = 0; l < nl; l++)
    for (int v = 0; v < NV; v++) s += gR[l * NV + v] * ld(dR + l * P + 16 * v);
  return hsum(s);
}

// force of one bond from dE/dx (x = r / rcut) and the vector dE/dY: tangential projection of the gradient
inline void bond_force(double dEdx, const v16 *gYv, const float *G, int nlm, double r, double rc, const double *rh,
                       double *fv)
{
  // sum_lm hsum(gY[lm]) G[lm] = hsum(sum_lm gY[lm] G[lm]): three reductions instead of nlm
  v16 a0 = v16{}, a1 = v16{}, a2 = v16{};
  for (int lm = 0; lm < nlm; lm++) {
    a0 += gYv[lm] * G[3 * lm]; a1 += gYv[lm] * G[3 * lm + 1]; a2 += gYv[lm] * G[3 * lm + 2];
  }
  const double gv[3] = {hsum(a0), hsum(a1), hsum(a2)};
  const double dEdr = dEdx / rc;
  const double gr = gv[0] * rh[0] + gv[1] * rh[1] + gv[2] * rh[2];
  for (int d = 0; d < 3; d++) fv[d] = dEdr * rh[d] + (gv[d] - gr * rh[d]) / r;
}

inline void spline_eval(const float *tab, int N, int nl, int P, double x, float *R, float *dR)
{
  double s = x * N;
  int k = (int) s;
  if (k > N - 1) k = N - 1;
  if (k < 0) k = 0;
  const float t = (float) (s - k);
  const size_t blk = (size_t) nl * P;
  const float *c0 = tab + (size_t) k * 4 * blk, *c1 = c0 + blk, *c2 = c1 + blk, *c3 = c2 + blk;
  const int n = nl * P;
  if (dR) {
    const float fN = (float) N;
#pragma omp simd
    for (int q = 0; q < n; q++) {
      R[q] = c0[q] + t * (c1[q] + t * (c2[q] + t * c3[q]));
      dR[q] = (c1[q] + t * (2.0f * c2[q] + t * 3.0f * c3[q])) * fN;
    }
  } else {
#pragma omp simd
    for (int q = 0; q < n; q++) R[q] = c0[q] + t * (c1[q] + t * (c2[q] + t * c3[q]));
  }
}

// per-atom virial of one bond (del = x_i - x_j, f = force on i): half to each atom, as Grace2LKK
inline void add_vatom(double **vatom, int i, int j, const double *rh, double r, const double *fv)
{
  const double dx = -rh[0] * r, dy = -rh[1] * r, dz = -rh[2] * r;
  const double v[6] = {0.5 * dx * fv[0], 0.5 * dy * fv[1], 0.5 * dz * fv[2],
                       0.5 * dx * fv[1], 0.5 * dx * fv[2], 0.5 * dy * fv[2]};
  for (int k = 0; k < 6; k++) {
#pragma omp atomic
    vatom[i][k] += v[k];
#pragma omp atomic
    vatom[j][k] += v[k];
  }
}

inline void add_force(double *ft, int i, int j, const double *fv)
{
  for (int d = 0; d < 3; d++) {
    ft[3 * i + d] += fv[d];
    ft[3 * j + d] -= fv[d];
  }
}


/* ----------------------------------------------------------------------
   model -> CPU layout
------------------------------------------------------------------------- */

template<class FCW>
void build_fc(FC &f, const FCW &w, int nil, int nir, int pinl, int pinr, int pout, const std::vector<char> &live,
              const Ctx &cx)
{
  f.nil = nil;
  f.nir = nir;
  f.nout = w.n_out;
  f.nfl = w.n_funcs_left;
  f.tile_l = w.w_tile_left;
  f.ntl = 0;
  for (int t : w.w_tile_left) f.ntl = std::max(f.ntl, t + 1);
  f.ntl = std::max(f.ntl, w.w_shape_left);
  f.ntr = 0;
  for (int t : w.w_tile_right) f.ntr = std::max(f.ntr, t + 1);
  f.ntr = std::max(f.ntr, w.w_shape_right);
  if ((size_t) f.nout * nil * f.ntl != w.w_left.size() || (size_t) f.nout * nir * f.ntr != w.w_right.size())
    throw Error("GRACE CPU engine: FC weight shape mismatch");
  f.nof.resize(f.nfl);
  for (int q = 0; q < f.nfl; q++) f.nof[q] = (float) w.norm_out_factor[q];
  // right connections sorted by target
  std::vector<int> ord(w.collect_from.size());
  for (size_t c = 0; c < ord.size(); c++) ord[c] = c;
  std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return w.collect_to[a] < w.collect_to[b]; });
  f.rc_to.clear(); f.rc_from.clear(); f.rc_tile.clear();
  for (int c : ord) {
    f.rc_to.push_back(w.collect_to[c]);
    f.rc_from.push_back(w.collect_from[c]);
    f.rc_tile.push_back(w.w_tile_right[c]);
  }
  auto pack = [&](const std::vector<double> &W, int nt, double norm, int nin, int pin, fvec &Wo,
                  fvec &WT) {
    Wo.assign((size_t) nt * nin * pout, 0.0f);
    WT.assign((size_t) nt * f.nout * pin, 0.0f);
    for (int k = 0; k < f.nout; k++)
      for (int n = 0; n < nin; n++)
        for (int t = 0; t < nt; t++) {
          const float v = (float) (norm * W[((size_t) k * nin + n) * nt + t]);
          Wo[((size_t) t * nin + n) * pout + k] = v;
          WT[((size_t) t * f.nout + k) * pin + n] = v;
        }
  };
  pack(w.w_left, f.ntl, w.norm_left, nil, pinl, f.Wl, f.WlT);
  pack(w.w_right, f.ntr, w.norm_right, nir, pinr, f.Wr, f.WrT);
  f.live = live.empty() ? std::vector<char>(f.nfl, 1) : live;
  f.pinl = pinl;
  f.pinr = pinr;
  f.pout = pout;
  if (pinl % 16 || pinr % 16 || pout % 16 || pinl > 128 || pinr > 128 || pout > 128) throw Error("GRACE CPU engine: FC width > 128");
  f.gf.clear();
  f.gb.clear();
  for (int t = 0; t < f.ntl; t++) {
    MVG gF, gB;
    gF.M = f.Wl.data() + (size_t) t * nil * pout; gF.istr = pinl; gF.ostr = pout; gF.nk = nil; gF.ldm = pout; gF.src = 0; gF.dst = 0;
    gB.M = f.WlT.data() + (size_t) t * f.nout * pinl; gB.istr = pout; gB.ostr = pinl; gB.nk = f.nout; gB.ldm = pinl; gB.src = 0; gB.dst = 0;
    for (int q = 0; q < f.nfl; q++)
      if (f.tile_l[q] == t && f.live[q]) {
        gF.in_off.push_back(q * pinl); gF.out_off.push_back(q * pout); gF.sc.push_back(f.nof[q]);
        gB.in_off.push_back(q * pout); gB.out_off.push_back(q * pinl); gB.sc.push_back(f.nof[q]);
      }
    if (!gF.in_off.empty()) { f.gf.push_back(gF); f.gb.push_back(gB); }
  }
  for (int t = 0; t < f.ntr; t++) {
    MVG gF, gB;
    gF.M = f.Wr.data() + (size_t) t * nir * pout; gF.istr = pinr; gF.ostr = pout; gF.nk = nir; gF.ldm = pout; gF.src = 1; gF.dst = 0;
    gB.M = f.WrT.data() + (size_t) t * f.nout * pinr; gB.istr = pout; gB.ostr = pinr; gB.nk = f.nout; gB.ldm = pinr; gB.src = 0; gB.dst = 1;
    for (size_t c = 0; c < f.rc_to.size(); c++)
      if (f.rc_tile[c] == t && f.live[f.rc_to[c]]) {
        const int to = f.rc_to[c], fr = f.rc_from[c];
        gF.in_off.push_back(fr * pinr); gF.out_off.push_back(to * pout); gF.sc.push_back(f.nof[to]);
        gB.in_off.push_back(to * pout); gB.out_off.push_back(fr * pinr); gB.sc.push_back(f.nof[to]);
      }
    if (!gF.in_off.empty()) { f.gf.push_back(gF); f.gb.push_back(gB); }
  }
}

template<class CGP>
void build_prod(Prod &p, const CGP &cg)
{
  const int nt = cg.cg_coeff.size();
  std::vector<int> ord(nt);
  for (int t = 0; t < nt; t++) ord[t] = t;
  std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return cg.m_sum_ind[a] < cg.m_sum_ind[b]; });
  p.nout = 0; p.nl = 0; p.nr = 0;
  p.l.clear(); p.r.clear(); p.o.clear(); p.c.clear();
  for (int t : ord) {
    p.l.push_back(cg.left_ind[t]);
    p.r.push_back(cg.right_ind[t]);
    p.o.push_back(cg.m_sum_ind[t]);
    p.c.push_back((float) cg.cg_coeff[t]);
    p.nout = std::max(p.nout, cg.m_sum_ind[t] + 1);
    p.nl = std::max(p.nl, cg.left_ind[t] + 1);
    p.nr = std::max(p.nr, cg.right_ind[t] + 1);
  }
  p.off.assign(p.nout + 1, 0);
  for (int t = 0; t < nt; t++) p.off[p.o[t] + 1]++;
  for (int q = 0; q < p.nout; q++) p.off[q + 1] += p.off[q];
}

// rw[s]: ReduceWeights of source s; fo_fixed[s] >= 0 sends every connection of s to that output function;
// nouts per source come from the weight size. pin[s]: channel stride of the source tensor.
template<class RW>
void build_red(Red &R, const std::vector<const RW *> &rw, const std::vector<int> &fo_fixed, int nout_max, int nfo,
               bool elem, int pout, const std::vector<int> &pin, const Ctx &cx)
{
  const int ntypes = cx.ntypes;
  const int nty = elem ? ntypes + 1 : 1;
  R.nout = nout_max;
  R.nfo = nfo;
  R.elem = elem;
  R.src.clear();
  for (size_t s = 0; s < rw.size(); s++) {
    const auto &w = *rw[s];
    Red::Src S;
    S.nin = w.n_in;
    S.nt = std::max(1, w.w_shape);
    const size_t per = (size_t) S.nin * S.nt;
    const size_t nel = elem ? (size_t) cx.n_elements : 1;
    if (w.W.size() % (per * nel) != 0) throw Error("GRACE CPU engine: reduce weight shape mismatch");
    const int nout = w.W.size() / (per * nel);
    if (nout > pout) throw Error("GRACE CPU engine: reduce width exceeds padding");
    for (int c = 0; c < (int) w.collect_ind.size(); c++) {
      S.ci.push_back(w.collect_ind[c]);
      S.tile.push_back(w.w_l_tile[c]);
      const int fo = fo_fixed[s] >= 0 ? fo_fixed[s] : w.total_sum_ind[c];
      if (fo >= nfo) throw Error("GRACE CPU engine: reduce output index out of range");
      S.fo.push_back(fo);
    }
    S.W.assign((size_t) nty * S.nt * S.nin * pout, 0.0f);
    S.WT.assign((size_t) nty * S.nt * pout * pin[s], 0.0f);
    for (int ty = 0; ty < nty; ty++) {
      const int mu = elem ? (ty >= 1 ? (*cx.type2mu)[ty] : -1) : 0;
      if (mu < 0) continue;
      for (int k = 0; k < nout; k++)
        for (int n = 0; n < S.nin; n++)
          for (int t = 0; t < S.nt; t++) {
            const size_t off = elem ? ((((size_t) mu * nout + k) * S.nin + n) * S.nt + t)
                                    : (((size_t) k * S.nin + n) * S.nt + t);
            const float v = (float) (w.norm * w.W[off]);
            S.W[(((size_t) ty * S.nt + t) * S.nin + n) * pout + k] = v;
            S.WT[(((size_t) ty * S.nt + t) * pout + k) * pin[s] + n] = v;
          }
    }
    S.nout = nout;
    S.pin = pin[s];
    R.src.push_back(std::move(S));
  }
  R.pout = pout;
  if (pout % 16 || pout > 128) throw Error("GRACE CPU engine: reduce width > 128");
  R.gf.clear();
  R.gb.clear();
  for (size_t s = 0; s < R.src.size(); s++) {
    const auto &S = R.src[s];
    if (S.pin % 16 || S.pin > 128) throw Error("GRACE CPU engine: reduce input width > 128");
    for (int t = 0; t < S.nt; t++) {
      MVG gF, gB;
      gF.M = S.W.data() + (size_t) t * S.nin * pout;
      gF.istr = S.pin; gF.ostr = pout;
      gF.tystride = elem ? (size_t) S.nt * S.nin * pout : 0;
      gF.nk = S.nin; gF.ldm = pout; gF.src = s; gF.dst = 0;
      gB.M = S.WT.data() + (size_t) t * pout * S.pin;
      gB.istr = pout; gB.ostr = S.pin;
      gB.tystride = elem ? (size_t) S.nt * pout * S.pin : 0;
      gB.nk = S.nout; gB.ldm = S.pin; gB.src = 0; gB.dst = s;
      for (size_t c = 0; c < S.ci.size(); c++)
        if (S.tile[c] == t) {
          gF.in_off.push_back(S.ci[c] * S.pin); gF.out_off.push_back(S.fo[c] * pout); gF.sc.push_back(1.0f);
          gB.in_off.push_back(S.fo[c] * pout); gB.out_off.push_back(S.ci[c] * S.pin); gB.sc.push_back(1.0f);
        }
      if (!gF.in_off.empty()) { R.gf.push_back(gF); R.gb.push_back(gB); }
    }
  }
}

// ---- UQ (schema v6 basis-RP as Grace1LKK / Grace2LKK; projection in fp32, norms and GMM in fp64)
// The basis rows of the projection matrix R are the collected functions of a scalar reduce's sources in model
// order (source, input channel, collected function).
// The projection matrix (3-4 MB in fp32) would be streamed from L3 once per atom: atoms are queued per thread
// (basis vectors as rows of Bq [MA][db], written contiguously) and projected up to UQ_MA at a time, Z = Bq R by a
// register-tiled fp32 GEMM (2.3x the fp64 one; gamma within ~1e-5 of the fp64 projection)
constexpr int UQ_MA = 64;
constexpr int UQ_ROWS = 72;    // UQ_MA rounded up to the GEMM atom tiles (6 and 3)
struct UQQueue {
  int na = 0;
  int atom[UQ_MA], ty[UQ_MA];
  double n2[UQ_MA];
  std::vector<float> Bq, Z;    // [UQ_ROWS][db], [UQ_ROWS][RP]
  // queue atom a of a batch (sources X as passed to red_fwd), LAMMPS index i, type t
  void push(const Red &R, const float *const *X, int a, int i, int t, int db)
  {
    if (Bq.size() < (size_t) db * UQ_ROWS) Bq.assign((size_t) db * UQ_ROWS, 0.0f);
    float *row = Bq.data() + (size_t) na * db;
    double s2 = 0.0;
    for (size_t s = 0; s < R.src.size(); s++) {
      const Red::Src &S = R.src[s];
      const int nc = S.ci.size();
      for (int c = 0; c < nc; c++) {    // contiguous reads of one function's channels, writes strided by nc
        const float *x = X[s] + ((size_t) S.ci[c] * NB + a) * S.pin;
        for (int n = 0; n < S.nin; n++) {
          const double bv = x[n];
          s2 += bv * bv;
          row[(size_t) n * nc + c] = x[n];
        }
      }
      row += (size_t) S.nin * nc;
    }
    atom[na] = i;
    ty[na] = t;
    n2[na] = s2;
    na++;
  }
};

// Z [na][RP] = Bq R, Bq [na][db], R [db][RP] (na rounded up to TA <= UQ_ROWS): k blocks of R kept in L1 across the
// atom tiles, TA atoms x 16 NV columns of accumulators in registers (transposed variants with R as broadcast
// operands were not faster in fp64)
template <int TA, int NV>
inline void uq_gemm_t(const float *Bq, int na, int db, const float *R, int RP, float *Z)
{
  constexpr int KB = 256, TD = 16 * NV;
  const int ma = (na + TA - 1) / TA * TA;
  std::fill(Z, Z + (size_t) ma * RP, 0.0f);
  for (int b0 = 0; b0 < db; b0 += KB) {
    const int b1 = std::min(db, b0 + KB);
    for (int d0 = 0; d0 + TD <= RP; d0 += TD)
      for (int a0 = 0; a0 < ma; a0 += TA) {
        v16 acc[TA][NV];
        for (int a = 0; a < TA; a++)
          for (int v = 0; v < NV; v++) acc[a][v] = ld(Z + (size_t) (a0 + a) * RP + d0 + 16 * v);
        for (int b = b0; b < b1; b++) {
          v16 r[NV];
          for (int v = 0; v < NV; v++) r[v] = ld(R + (size_t) b * RP + d0 + 16 * v);
          for (int a = 0; a < TA; a++) {
            const float x = Bq[(size_t) (a0 + a) * db + b];    // rows >= na: stale but finite, unused
            for (int v = 0; v < NV; v++) acc[a][v] += x * r[v];
          }
        }
        for (int a = 0; a < TA; a++)
          for (int v = 0; v < NV; v++) st(Z + (size_t) (a0 + a) * RP + d0 + 16 * v, acc[a][v]);
      }
    for (int d = RP - RP % TD; d < RP; d++)    // columns left over by the tile
      for (int a = 0; a < ma; a++) {
        float t = 0.0f;
        for (int b = b0; b < b1; b++) t += Bq[(size_t) a * db + b] * R[(size_t) b * RP + d];
        Z[(size_t) a * RP + d] += t;
      }
  }
}

// projection of the queued atoms: q.Z [UQ_ROWS][RP] (atom c: row c)
inline void uq_gemm(UQQueue &q, int db, const float *R, int RP)
{
  if (q.Z.size() < (size_t) UQ_ROWS * RP) q.Z.assign((size_t) UQ_ROWS * RP, 0.0f);
#ifdef __AVX512F__
  uq_gemm_t<6, 4>(q.Bq.data(), q.na, db, R, RP, q.Z.data());
#else
  uq_gemm_t<3, 2>(q.Bq.data(), q.na, db, R, RP, q.Z.data());
#endif
}

inline int uq_width(const Red &R)
{
  int w = 0;
  for (const auto &S : R.src) w += S.nin * (int) S.ci.size();
  return w;
}

typedef simd::d8 vd8;
inline vd8 ldd(const double *p) { return simd::load(p); }

// f[0, RP): raw projection; nrm = ||b|| over all blocks; dens[0, n_density): density channel norms. Normalise,
// append log(dens), nearest centroid of element mu, Mahalanobis distance: sigma, gamma = sigma / threshold
// dot product with eight vector accumulators (fp reductions are not vectorised without -ffast-math)
inline double uq_dot(const double *x, const double *y, size_t n)
{
  vd8 acc = vd8{};
  size_t k = 0;
  for (; k + 8 <= n; k += 8) acc += ldd(x + k) * ldd(y + k);
  double t = 0.0;
  for (int v = 0; v < 8; v++) t += acc[v];
  for (; k < n; k++) t += x[k] * y[k];
  return t;
}

template <class M>
inline void uq_gmm(const M &m, int mu, double *f, double *delta, double nrm, const double *dens, double &gamma,
                   double &sigma, double &cluster)
{
  // sizes unsigned, as the loaders hold them: they check uq_rp_dim >= 1 and uq_feature_dim >= uq_rp_dim (uq_head.cpp,
  // model_1l / model_2l.cpp), which an int D does not show the compiler (GCC 16 -Wstringop-overflow on std::fill)
  const size_t RP = m.uq_rp_dim, D = m.uq_feature_dim, K = m.uq_max_clusters;
  if (m.uq_normalize)
    for (size_t d = 0; d < RP; d++) f[d] /= nrm + 1.0e-12;
  for (int p = 0; p < m.uq_n_density; p++) f[RP + p] = std::log(dens[p] + 1.0e-12) * m.uq_density_scale;
  int kstar = 0;
  double best = 1.0e300;
  for (int q = 0; q < m.uq_n_clusters[mu]; q++) {
    const double *cen = m.uq_centroids.data() + ((size_t) mu * K + q) * D;
    for (size_t p = 0; p < D; p++) delta[p] = f[p] - cen[p];
    const double d2 = uq_dot(delta, delta, D);
    if (d2 < best) { best = d2; kstar = q; }
  }
  const double *cen = m.uq_centroids.data() + ((size_t) mu * K + kstar) * D;
  for (size_t p = 0; p < D; p++) delta[p] = f[p] - cen[p];
  // sig2 = delta^T S delta = delta^T S^T delta: y = S^T delta accumulated row by row of S (vectorises), f reused for y
  const double *icov = m.uq_inv_cov.data() + ((size_t) mu * K + kstar) * D * D;
  std::fill(f, f + D, 0.0);
  for (size_t p = 0; p < D; p++) {
    const double dp = delta[p], *row = icov + p * D;
    for (size_t q = 0; q < D; q++) f[q] += dp * row[q];
  }
  const double sig2 = uq_dot(delta, f, D);
  sigma = std::sqrt(sig2 + 1.0e-8);
  gamma = sigma / std::max(m.uq_interp_thresholds[(size_t) mu * K + kstar], 1.0e-10);
  cluster = (double) kstar;
}

}    // namespace fused
}    // namespace grace

#endif

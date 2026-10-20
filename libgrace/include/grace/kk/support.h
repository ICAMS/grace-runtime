// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Helpers of the Kokkos engines that LAMMPS used to provide: view reallocation (MemoryKokkos::realloc_kokkos),
// view memory usage, and errors with {} placeholders (LAMMPS error->all) thrown as grace::Error.
#pragma once

#include "grace/error.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace grace {
namespace kk {

// the device runs CUDA (LAMMPS: std::is_same_v<DeviceType, LMPDeviceType> inside KOKKOS_ENABLE_CUDA blocks)
#ifdef KOKKOS_ENABLE_CUDA
template<class D> inline constexpr bool is_cuda_v = std::is_same_v<typename D::execution_space, Kokkos::Cuda>;
#else
template<class D> inline constexpr bool is_cuda_v = false;
#endif

// real spherical harmonics normalisation constants of the engines (shared: one translation unit may hold several)
namespace sph_const {
inline constexpr double Y00 = 1.0;
inline constexpr double sq3 = 1.7320508075688772935;
inline constexpr double sq3o2 = 1.2247448713915890491;
inline constexpr double sq2 = 1.4142135623730950488;
}    // namespace sph_const

// Tables of the associated Legendre recurrences behind the engines' real spherical harmonics, l <= lmax:
//   P(l, m)     = A(l, m) * (z P(l-1, m) + B(l, m) P(l-2, m))    m <= l - 2
//   P(l, l - 1) = D(l) * z P(l-1, l-1)
//   P(l, l)     = C(l) * P(l-1, l-1)
// A(l, m) = sqrt((4 l^2 - 1) / (l^2 - m^2)), B(l, m) = -sqrt(((l-1)^2 - m^2) / (4 (l-1)^2 - 1)),
// C(l) = -sqrt(1 + 1 / (2 l)), D(l) = sqrt(2 l + 1).
// A and B are packed by m, then l (m = 0: l = 0..lmax, m = 1: l = 1..lmax, ...), zero where m >= l - 1;
// slot[l (l + 1) + m] is the packed position of (l, m) and -1 for m < 0. The numerators and denominators are
// integers, exact in double, so every entry is one correctly rounded division and square root.
struct SphTables {
  std::vector<double> slot, a, b, c, d;
  int npacked = 0;
};

inline SphTables sph_tables(int lmax)
{
  SphTables t;
  const int nlm = (lmax + 1) * (lmax + 1);
  t.slot.assign(nlm, -1.0);
  t.a.reserve(nlm);
  t.b.reserve(nlm);
  for (int m = 0; m <= lmax; m++)
    for (int l = m; l <= lmax; l++) {
      t.slot[l * (l + 1) + m] = double(t.a.size());
      const bool rec = l >= 2 && m <= l - 2;
      t.a.push_back(rec ? std::sqrt(double(4 * l * l - 1) / double(l * l - m * m)) : 0.0);
      t.b.push_back(rec ? -std::sqrt(double((l - 1) * (l - 1) - m * m) / double(4 * (l - 1) * (l - 1) - 1)) : 0.0);
    }
  t.npacked = int(t.a.size());
  t.c.assign(lmax + 1, 0.0);
  t.d.assign(lmax + 1, 0.0);
  for (int l = 1; l <= lmax; l++) {
    t.c[l] = -std::sqrt(1.0 + 0.5 / l);
    t.d[l] = std::sqrt(double(2 * l + 1));
  }
  return t;
}

// the leading values of a device view from a host vector (converted to the view's value type)
template<class View> void upload_prefix(const View &v, const std::vector<double> &h)
{
  auto m = Kokkos::create_mirror_view(v);
  for (size_t k = 0; k < h.size(); k++) m(k) = typename View::non_const_value_type(h[k]);
  Kokkos::deep_copy(v, m);
}

// team size of a launch of functor f with tag Tag over `league` teams: the requested one, lowered when
// team size * vector length exceeds the device limit for that kernel
template<class DeviceType, class Tag, class F> int fit_team_size(const F &f, int league, int team_size, int vector_length)
{
  const int lim = Kokkos::TeamPolicy<DeviceType, Tag>(league, Kokkos::AUTO).team_size_max(f, Kokkos::ParallelForTag());
  return team_size * vector_length > lim ? lim / vector_length : team_size;
}

// bytes of team scratch holding n values of type T
template<class DeviceType, class T> KOKKOS_INLINE_FUNCTION int scratch_bytes(int n)
{
  return int(Kokkos::View<T *, typename DeviceType::scratch_memory_space, Kokkos::MemoryUnmanaged>::shmem_size(n));
}

// The bond launches run ceil(chunk / team size) * maxneigh teams: the neighbour slot is the league rank divided by
// ceil(chunk / team size), and the atoms of one neighbour slot are spread over that many consecutive teams, one per
// thread. atom >= chunk: a padding thread of the last team.
struct BondSlot {
  int atom, neigh;
};
template<class Member> KOKKOS_INLINE_FUNCTION BondSlot bond_slot(const Member &team, int chunk)
{
  const int ts = team.team_size(), teams_per_slot = (chunk + ts - 1) / ts;
  const int lr = team.league_rank();
  return {team.team_rank() + ts * (lr % teams_per_slot), lr / teams_per_slot};
}

// A View whose element access computes the offset in 64 bits. Kokkos 5 View::operator() computes the offset in the
// common type of the indices (an optimisation of Kokkos 5), so v(i, j, k) with int indices wraps once the view holds
// more than 2^31 elements: the GRACE-3L [nall][64][50] message rows from nall = 671089 (issue #1). The views with a
// row per atom of nall (local + ghost) are View64; every other view stays below 2^31 elements (MemKK::realloc_kokkos
// checks it, chunksize auto picks the chunk below that).
template<class V> struct View64 : V {
  using V::V;
  View64() = default;
  View64(const V &v) : V(v) {}
  template<class... I>
  KOKKOS_FORCEINLINE_FUNCTION typename V::reference_type operator()(const size_t i0, const I... i) const
  {
    return V::operator()(i0, static_cast<size_t>(i)...);
  }
};
template<class T> struct is_view64 : std::false_type {};
template<class V> struct is_view64<View64<V>> : std::true_type {};

struct MemKK {
  // bytes of the views (re)allocated by realloc_kokkos while set (chunksize auto: work buffer bytes per atom)
  static inline thread_local double *count = nullptr;
  // while count is set: the most elements of one of those views (chunksize auto: the chunk keeps it below 2^31)
  static inline thread_local double max_span = 0.0;
  // as LAMMPS MemoryKokkos::realloc_kokkos: drop the old allocation first, then allocate uninitialised. A view of more
  // than 2^31 elements must be a View64 (above): Kokkos 5 indexes the others with int arithmetic.
  template<typename TYPE, typename... Indices>
  static std::enable_if_t<TYPE::rank_dynamic == sizeof...(Indices), void> realloc_kokkos(TYPE &data, const char *name,
                                                                                        Indices... ns)
  {
    data = TYPE();
    double span = 1.0;    // elements: the runtime extents and the compile-time ones ([3], [6], ...)
    ((span *= (double) ns), ...);
    using Shape = Kokkos::View<typename TYPE::data_type, Kokkos::HostSpace>;    // TYPE may be a DualView
    for (unsigned r = Shape::rank_dynamic; r < Shape::rank; r++) span *= (double) Shape::static_extent(r);
    if constexpr (!is_view64<TYPE>::value)
      if (span > (double) INT32_MAX)
        throw grace::Error(std::string(name) + ": " + std::to_string((long long) span) +
                           " elements, more than the 2^31 a Kokkos view indexes with int; use a smaller chunksize");
    data = TYPE(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(name)), ns...);
    if (count) {
      *count += (double) data.span() * sizeof(typename TYPE::value_type);
      max_span = std::max(max_span, (double) data.span());
    }
  }
  template<typename TYPE> static double memory_usage(const TYPE &data)
  {
    return (double) data.span() * sizeof(typename TYPE::value_type);
  }
};

namespace detail {
inline void fmt_into(std::ostringstream &o, const char *f)
{
  o << f;
}
template<class A, class... R> void fmt_into(std::ostringstream &o, const char *f, A &&a, R &&...r)
{
  for (; *f; f++) {
    if (f[0] == '{' && (f[1] == '}' || f[1] == ':')) {    // "{}" or "{:spec}" (spec ignored)
      const char *e = f + 1;
      while (*e && *e != '}') e++;
      if (*e) {
        o << a;
        fmt_into(o, e + 1, std::forward<R>(r)...);
        return;
      }
    }
    o << *f;
  }
}
}    // namespace detail

// "{}" / "{:spec}" placeholders filled in order, specs ignored (the subset of fmt the engines' messages use)
template<class... A> std::string fmt(const char *f, A &&...a)
{
  std::ostringstream o;
  detail::fmt_into(o, f, std::forward<A>(a)...);
  return o.str();
}

// values of one exchange call, n message rows of w values: an int (the LAMMPS comm buffers and the pack / unpack
// kernels index them with int)
inline int comm_values(int n, int w)
{
  const int64_t v = (int64_t) n * w;
  if (v > INT32_MAX) throw grace::Error(fmt("exchange of {} atoms x {} values exceeds 2^31 values", n, w));
  return (int) v;
}

// chunksize auto: atoms per chunk for n atoms with up to mn neighbours. grow(c0, mn) re-grows every chunk work buffer
// for c0 <= 1024 atoms, counted (bytes per atom); the chunk is all n atoms when they fit 70 % of this process's share
// of the free device memory (free + the counted buffers, over ranks_per_device processes on the GPU), else the largest
// power of two that fits, at least 1024 (a structure too large even then fails at allocation as before). Host memory:
// one chunk.
// stored(): bytes per atom the engine keeps for all n atoms when there is more than one chunk (2L fused store_l1).
// kept_fits (optional): false when the kept bytes do not fit the budget next to the chosen chunk.
template<class DeviceType, class Grow, class Stored>
int auto_chunk(int n, int mn, int ranks_per_device, Grow &&grow, Stored &&stored, const grace::Logger &log,
               const char *tag, bool *kept_fits = nullptr)
{
  const int c0 = std::max(1, std::min(n, 1024));
  double bytes = 0.0;
  MemKK::count = &bytes;
  MemKK::max_span = 0.0;
  try {
    grow(c0, mn);
  } catch (...) {
    MemKK::count = nullptr;
    throw;
  }
  MemKK::count = nullptr;
  size_t free_b = 0, total_b = 0;
#ifdef KOKKOS_ENABLE_CUDA
  if (std::is_same<typename DeviceType::execution_space, Kokkos::Cuda>::value) cudaMemGetInfo(&free_b, &total_b);
#endif
  if (kept_fits) *kept_fits = true;
  // the largest work buffer stays below 2^31 elements (MemKK::realloc_kokkos)
  const double span_atom = MemKK::max_span / c0, cmax = span_atom > 0 ? (double) INT32_MAX / span_atom : (double) n;
  if (total_b == 0) return n > cmax ? std::max(1, (int) cmax) : n;    // no device memory budget (host backends)
  const double per_atom = bytes / c0, kept = stored();
  const double budget = 0.7 * ((double) free_b + bytes) / std::max(1, ranks_per_device);
  int c = n;
  if (per_atom * n > budget || n > cmax) {
    c = std::min(n, 1024);
    while (2.0 * c < n && per_atom * 2.0 * c + kept * n <= budget && 2.0 * c <= cmax) c *= 2;
    if (c > cmax) c = std::max(1, (int) cmax);
    if (kept_fits) *kept_fits = per_atom * c + kept * n <= budget;
  }
  log.info(fmt("[{}] chunksize auto: {} atoms per chunk ({} atoms, {} MB work buffers + {} MB kept per 1000 atoms, {} MB "
               "GPU memory free, {} rank(s) on this GPU)\n",
               tag, c, n, (long) (per_atom * 1000 / 1048576), (long) (kept * 1000 / 1048576),
               (long) (free_b / 1048576), ranks_per_device));
  return c;
}

// chunksize auto with optional kept state (2L fused store_l1, 3L store_fwd): the chunk with the kept state when it fits
// and is at least half of min(the chunk without it, sat) - sat: the chunk at which the engine's speed saturates (a
// much smaller chunk below that is slower than recomputing the kept part; above it the chunk size hardly matters) -
// else the chunk without it. Returns {chunk, keep}; only the decision is logged. RTX 3090 size scan: 2L sat 16384
// (2L 250k atoms: store with chunk 2048 6.9 us/atom/step, no store with 32768 5.6), 3L sat 2048 (3L 65k: store with
// 2048 52.8, no store with 4096 65.0).
template<class DeviceType, class Grow, class Stored>
std::pair<int, bool> auto_chunk_keep(int n, int mn, int ranks_per_device, Grow &&grow, Stored &&stored,
                                     const grace::Logger &log, const char *tag, int sat)
{
  bool fits = true;
  const int with = auto_chunk<DeviceType>(n, mn, ranks_per_device, grow, stored, grace::Logger{}, tag, &fits);
  const int without = auto_chunk<DeviceType>(n, mn, ranks_per_device, grow, [] { return 0.0; }, grace::Logger{}, tag);
  const bool keep = fits && 2.0 * with >= std::min(without, sat);
  const int c = auto_chunk<DeviceType>(n, mn, ranks_per_device, grow, [&] { return keep ? stored() : 0.0; }, log, tag);
  return {c, keep};
}

}    // namespace kk
}    // namespace grace

#define GRACE_KK_ERROR(...) throw ::grace::Error(::grace::kk::fmt(__VA_ARGS__))

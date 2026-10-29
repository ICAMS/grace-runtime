// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Helpers of the Kokkos engines that LAMMPS provides to its own styles: view reallocation
// (MemoryKokkos::realloc_kokkos), view memory usage, and errors with {} placeholders (LAMMPS error->all) thrown as
// grace::Error.
#pragma once

#include "grace/error.h"

#include <Kokkos_Core.hpp>
#ifdef KOKKOS_ENABLE_HIP
#include <hip/hip_runtime_api.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#ifdef KOKKOS_ENABLE_HIP
template<class D> inline constexpr bool is_hip_v = std::is_same_v<typename D::execution_space, Kokkos::HIP>;
#else
template<class D> inline constexpr bool is_hip_v = false;
#endif
// the device is a GPU whose free memory device_mem_info() reports
template<class D> inline constexpr bool is_gpu_v = is_cuda_v<D> || is_hip_v<D>;
// An on / off kernel switch from the environment (GRACE3L_SPBF_MFMA, GRACE_FUSED_MFMA, ...): 1, 0, or -1 when unset or
// empty (the build's default). Other values: a note on stderr, then -1. The callers read it once per process.
inline int env_switch(const char *name)
{
  const char *e = std::getenv(name);
  if (e == nullptr || *e == '\0') return -1;
  if (std::strcmp(e, "0") == 0) return 0;
  if (std::strcmp(e, "1") == 0) return 1;
  std::fprintf(stderr, "grace kk: %s=%s ignored (expected 0 or 1)\n", name, e);
  return -1;
}
// Launch bounds of a kernel launched with blocks of N threads: on HIP Kokkos::LaunchBounds<N, 1>, elsewhere the default
// (none, as before). Without them Kokkos compiles every HIP kernel for 1024-thread blocks, i.e. at most 128 VGPRs per
// lane, and the register-heavy fused kernels spilled 0.2 - 1 KB per lane to scratch on gfx90a.
// M: minimum number of waves per SIMD the compiler keeps registers for (HIP __launch_bounds__): 1 lets a kernel use
// all 512 registers per lane (one wave per SIMD), 2 caps it at 256, 3 at 168 (gfx90a / gfx94x; gfx906 / gfx908 have
// 256 per lane).
template<class D, unsigned N, unsigned M = 1>
using hip_lb_t = std::conditional_t<is_hip_v<D>, Kokkos::LaunchBounds<N, M>, Kokkos::LaunchBounds<>>;
// A team launch with launch bounds and a fallback launch (the body without them, or another kernel): whether Kokkos
// admits teams of `team` threads with the bounded policy (team_size_max, as checked at launch: the registers and the
// shared memory of the bounded number of teams per SM). GRACE_LB_FALLBACK=1: never, i.e. always the fallback (tests).
inline bool lb_fallback_forced()
{
  static const bool forced = env_switch("GRACE_LB_FALLBACK") == 1;
  return forced;
}
template<class Policy, class Body> bool launch_bounds_fit(const Policy &pol, const Body &body, int team)
{
  return !lb_fallback_forced() && pol.team_size_max(body, Kokkos::ParallelForTag()) >= team;
}

// GPU: the level-0 team scratch (bytes) a launch with teams of `team` threads can have, as Kokkos checks it at launch:
// the shared memory of a block less the (team + 2) doubles of the team reductions. CUDA: Kokkos' own limit,
// TeamPolicy::scratch_size_max(0) (it assumes teams of 1024): 48 KB per block with Kokkos 5.1 (LAMMPS' bundled one),
// the opt-in above that from Kokkos 5.2 on. Host backends: no limit here.
template<class D> size_t team_scratch_max(int team)
{
#if defined(KOKKOS_ENABLE_HIP)
  if constexpr (is_hip_v<D>) return Kokkos::HIP::hip_device_prop().sharedMemPerBlock - (size_t) (team + 2) * sizeof(double);
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (is_cuda_v<D>)
    return (size_t) Kokkos::TeamPolicy<D>::scratch_size_max(0) + (size_t) (1024 - team) * sizeof(double);
#endif
  (void) team;
  return SIZE_MAX;
}
// CUDA: teams of `team` threads with `shmem` bytes of team scratch each that the shared memory of an SM holds: per
// block the scratch, Kokkos' team-reduction doubles and its static 8-byte team id, rounded to the 128-byte allocation
// unit, plus the driver's reservation per block. Shared memory only (the caller's kernel must not be limited by
// registers first). 0: not known (other backends, a failed query) or not even one team.
template<class D> int teams_per_sm_by_scratch(size_t shmem, int team)
{
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (is_cuda_v<D>) {
    static const std::pair<int, int> sm = [] {
      const int dev = Kokkos::Cuda().cuda_device();
      int per_sm = 0, reserved = 0;
      if (cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev) != cudaSuccess ||
          cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev) != cudaSuccess) {
        (void) cudaGetLastError();    // not left pending for the next launch check
        return std::make_pair(0, 0);
      }
      return std::make_pair(per_sm, reserved);
    }();
    const size_t block = (shmem + (size_t) (team + 2) * sizeof(double) + sizeof(int64_t) + 127) / 128 * 128;
    return (int) ((size_t) sm.first / (block + (size_t) sm.second));
  }
#endif
  (void) shmem;
  (void) team;
  return 0;
}

// (free, total) bytes of the current GPU of this build (cudaMemGetInfo / hipMemGetInfo); false (both 0 unless the
// runtime wrote them) without a GPU backend or when the query fails
inline bool device_mem_info(size_t &free_b, size_t &total_b)
{
#if defined(KOKKOS_ENABLE_CUDA)
  return cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
#elif defined(KOKKOS_ENABLE_HIP)
  return hipMemGetInfo(&free_b, &total_b) == hipSuccess;
#else
  (void) free_b;
  (void) total_b;
  return false;
#endif
}

// Identity of the GPU of this process's Kokkos device instance, unique on the node: "uuid-<32 hex digits>" (the
// device UUID: CUDA cudaDeviceProp::uuid, which also tells MIG instances apart; HIP hipDeviceGetUuid, one per GCD of an
// MI250X), or "pci-<PCI bus id>" when the UUID query fails or returns zeros. Processes on the same GPU get the same
// string whatever their CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES; compare the strings exactly. The device is the
// runtime ordinal of the default Kokkos instance (Kokkos::Cuda().cuda_device(), Kokkos::HIP().hip_device()), not
// Kokkos::device_id() (an index into KOKKOS_VISIBLE_DEVICES). Empty without a GPU backend or when both queries fail.
// Only a failed call's own error is cleared (cudaGetLastError / hipGetLastError right after it), so an error pending
// from elsewhere stays pending unless that call failed. Call after Kokkos::initialize.
inline std::string device_key()
{
  auto hex = [](const char *bytes) {
    static const char digits[] = "0123456789abcdef";
    std::string s = "uuid-";
    bool nonzero = false;
    for (int i = 0; i < 16; i++) {
      const unsigned char b = (unsigned char) bytes[i];
      nonzero = nonzero || b != 0;
      s += digits[b >> 4];
      s += digits[b & 15];
    }
    return nonzero ? s : std::string();
  };
  char bus[64] = {0};
#if defined(KOKKOS_ENABLE_CUDA)
  const int dev = Kokkos::Cuda().cuda_device();
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
    const std::string key = hex(prop.uuid.bytes);
    if (!key.empty()) return key;
  } else {
    (void) cudaGetLastError();
  }
  if (cudaDeviceGetPCIBusId(bus, (int) sizeof(bus), dev) == cudaSuccess) return std::string("pci-") + bus;
  (void) cudaGetLastError();
  return std::string();
#elif defined(KOKKOS_ENABLE_HIP)
  const int dev = Kokkos::HIP().hip_device();
  hipDevice_t hd;
  hipUUID uuid;
  if (hipDeviceGet(&hd, dev) == hipSuccess && hipDeviceGetUuid(&uuid, hd) == hipSuccess) {
    const std::string key = hex(uuid.bytes);
    if (!key.empty()) return key;
  } else {
    (void) hipGetLastError();
  }
  if (hipDeviceGetPCIBusId(bus, (int) sizeof(bus), dev) == hipSuccess) return std::string("pci-") + bus;
  (void) hipGetLastError();
  return std::string();
#else
  (void) hex;
  (void) bus;
  return std::string();
#endif
}

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

// A RangePolicy with 64-bit indices, for ranges that can pass 2^31 (bonds of a whole system). Kokkos >= 5.2 indexes a
// RangePolicy without IndexType with int on CUDA (before: unsigned int) and aborts the process on a larger bound.
template<class D, class... P> using Range64 = Kokkos::RangePolicy<D, P..., Kokkos::IndexType<int64_t>>;

// A View whose element access computes the offset in 64 bits. Kokkos 5 View::operator() computes the offset in the
// common type of the indices (an optimisation of Kokkos 5), so v(i, j, k) with int indices wraps once the view holds
// more than 2^31 elements: the GRACE-3L [nall][64][50] message rows from nall = 671089. The views with a
// row per atom of nall (local + ghost) are View64; every other view stays below 2^31 elements (MemKK::realloc_kokkos
// checks it, chunksize auto picks the chunk below that).
template<class V> class View64 : V {    // private base: a View64 never passes as a plain V (int offsets again)
 public:
  using view_type = V;
  using typename V::data_type;
  using typename V::value_type;
  using typename V::non_const_value_type;
  using typename V::reference_type;
  using typename V::memory_space;
  using typename V::execution_space;
  using typename V::host_mirror_type;
  using V::rank;
  using V::rank_dynamic;
  using V::V;
  View64() = default;
  View64(const V &v) : V(v) {}
  // the plain view, for what takes one by design: deep_copy, subview, create_mirror_view
  KOKKOS_FORCEINLINE_FUNCTION const V &base() const { return *this; }
  using V::data;
  using V::extent;
  using V::span;
  using V::size;
  using V::label;
  using V::stride;
  template<class... I>
  KOKKOS_FORCEINLINE_FUNCTION reference_type operator()(const size_t i0, const I... i) const
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
  // the views emptied: an allocation goes with its last reference
  template<class... V> static void drop(V &...v) { ((v = V()), ...); }
  // host copy h of the rank-3 View64 d, made on first use (and again when d's extents changed)
  template<class H, class D> static void ensure_mirror(H &h, const D &d)
  {
    if (h.extent(0) != d.extent(0) || h.extent(1) != d.extent(1) || h.extent(2) != d.extent(2))
      h = Kokkos::create_mirror_view(d.base());
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

// the smallest chunk of chunksize auto and of its smaller chunks after an allocation failure (python/kk): GRACE-3L-
// large on an RTX 2070 8 GB fits ~10 % more atoms with 512 than with 1024 at the same speed (RTX 3090, 65k atoms: 512
// is 18 % slower than 1024, 256 62 %; the floor is reached only when the larger chunks do not fit)
inline constexpr int kChunkMin = 512;

// chunksize auto: atoms per chunk for n atoms with up to mn neighbours. grow(c0, mn) re-grows every chunk work buffer
// for c0 <= 1024 atoms, counted (bytes per atom); the chunk is all n atoms when they fit 70 % of this process's share
// of the free device memory (free + the counted buffers, over ranks_per_device processes on the GPU), else the largest
// power of two that fits, at least kChunkMin (a structure too large even then fails at allocation as before). Host
// memory: one chunk.
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
  if (is_gpu_v<DeviceType>) device_mem_info(free_b, total_b);
  if (kept_fits) *kept_fits = true;
  // the largest work buffer stays below 2^31 elements (MemKK::realloc_kokkos)
  // (256 rows of margin: the HIP 3L engine pads the chunk's rows to a multiple of 128 that is not a power of two)
  const double span_atom = MemKK::max_span / c0, cmax = span_atom > 0 ? (double) INT32_MAX / span_atom - 256 : (double) n;
  if (total_b == 0) return n > cmax ? std::max(1, (int) cmax) : n;    // no device memory budget (host backends)
  const double per_atom = bytes / c0, kept = stored();
  const double budget = 0.7 * ((double) free_b + bytes) / std::max(1, ranks_per_device);
  int c = n;
  if (per_atom * n > budget || n > cmax) {
    c = std::min(n, kChunkMin);
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

// tests (the GPU module's _debug_fail_launch): the nth check_kernel_launches from now on reports an out-of-memory
// launch failure (0: off)
inline std::atomic<int> g_fail_launch{0};
inline bool fail_launch_now()
{
  int n = g_fail_launch.load();
  while (n > 0 && !g_fail_launch.compare_exchange_weak(n, n - 1)) {}
  return n == 1;
}

// Throws grace::Error when a GPU kernel launch (or another CUDA / HIP runtime call) failed since the last check.
// Kokkos checks kernel launches only with KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK: in a release build a launch that fails,
// e.g. out of memory for the kernel's code (lazy module loading) or its local memory when the GPU is nearly full,
// leaves the kernel's outputs unwritten without any error, and the next fence still succeeds (such errors are not
// sticky). The engines call this at the end of setup() and run(): one cudaGetLastError per step.
template<class DeviceType> void check_kernel_launches(const char *who)
{
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (is_cuda_v<DeviceType>) {
    cudaError_t e = cudaGetLastError();
    if (fail_launch_now()) e = cudaErrorMemoryAllocation;
    if (e != cudaSuccess)
      throw grace::Error(fmt("{}: a GPU kernel launch or runtime call failed ({}: {}); the results of this step are "
                             "invalid. With 'out of memory': free GPU memory (other processes, fewer MPI ranks per "
                             "GPU, smaller chunksize)", who, cudaGetErrorName(e), cudaGetErrorString(e)));
  }
#elif defined(KOKKOS_ENABLE_HIP)
  if constexpr (is_hip_v<DeviceType>) {
    hipError_t e = hipGetLastError();
    if (fail_launch_now()) e = hipErrorOutOfMemory;
    if (e != hipSuccess)
      throw grace::Error(fmt("{}: a GPU kernel launch or runtime call failed ({}: {}); the results of this step are "
                             "invalid. With 'out of memory': free GPU memory (other processes, fewer MPI ranks per "
                             "GPU, smaller chunksize)", who, hipGetErrorName(e), hipGetErrorString(e)));
  }
#else
  (void) who;
#endif
}

}    // namespace kk
}    // namespace grace

#define GRACE_KK_ERROR(...) throw ::grace::Error(::grace::kk::fmt(__VA_ARGS__))

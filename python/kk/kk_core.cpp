// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// GPU backend of grace_runtime (kk_api.h): the fused GRACE-1L / 2L and the GRACE-3L Kokkos engines of libgrace on CSR
// bonds with ghost images (grace/kk/csr_images.h). Compiled through the Kokkos launch compiler; kk_module.cpp binds it to Python.
#include "kk_api.h"

#include "batch_csr.h"

#include "grace/kk/csr_images.h"
#include "grace/kk/neighbors_kk.h"
#include "grace/kk/grace_1l_fused_kk_impl.h"
#include "grace/kk/grace_2l_fused_kk_impl.h"
#include "grace/kk/grace_3l_fused_kk.h"
#include "grace/kk/traits.h"
#include "grace/model_1l.h"
#include "grace/model_2l.h"
#include "grace/model_3l.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <chrono>
#include <limits>
#include <sched.h>
#include <mutex>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

#ifndef GRACE_KK_ARCH
#define GRACE_KK_ARCH "unknown"
#endif

namespace grace_runtime_kk {

using Dev = Kokkos::DefaultExecutionSpace;
using T = grace::kk::DefaultTraits<Dev>;
using E1 = grace::kk::Grace1LFusedKK<Dev, T>;
using E2 = grace::kk::Grace2LFusedKK<Dev, T>;
using E3 = grace::kk::Grace3LFusedKK<Dev, T>;    // as LAMMPS grace/3l/fused/kk (fp32 with fp64 bonds / envelope)
}    // namespace grace_runtime_kk
extern template class grace::kk::Grace3LKK<Kokkos::DefaultExecutionSpace, float, float,
                                           grace::kk::DefaultTraits<Kokkos::DefaultExecutionSpace>>;    // kk_core_3l.cpp
namespace grace_runtime_kk {

namespace {
std::mutex g_mu;
// Kokkos dispatch to the default execution space instance is not thread-safe (the CUDA backend stages large functors
// in a buffer of the instance): every entry point that launches, allocates or frees device work holds this lock, one
// thread at a time (two calculators in two Python threads crashed with illegal addresses). Recursive: the last holder
// of a Model may release it from inside a System / Batch call. Taken before g_mu.
std::recursive_mutex g_device_mu;
int g_device = -1;
bool g_finalized = false;
// live Systems / Batches: finalize() releases their device memory first
struct DeviceState {
  virtual void release() = 0;

 protected:
  ~DeviceState() = default;
};
std::set<DeviceState *> g_states;

int affinity_threads()
{
  cpu_set_t c;
  if (sched_getaffinity(0, sizeof(c), &c) == 0) return std::max(1, CPU_COUNT(&c));
  return 1;
}

std::string detect_layers(const std::string &path)
{
  // the archive's member names tell the generation (as python/src/core.cpp detect_arch)
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::invalid_argument("cannot open " + path);
  std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (data.find("mlp_rad_W0.npy") != std::string::npos) return "1l";
  if (data.find("B0_lin_transform_W.npy") != std::string::npos) return "2l";
  if (data.find("prod_names.npy") != std::string::npos) return "3l";
  throw std::invalid_argument("not a GRACE-1L/2L/3L kokkos.npz: " + path);
}
}    // namespace

void initialize(int dev)
{
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_finalized) throw std::runtime_error("the GPU backend was finalized; restart the process");
  if (Kokkos::is_initialized()) {
    if (dev != g_device)
      throw std::invalid_argument("one CUDA device per process: already using cuda:" + std::to_string(g_device));
    return;
  }
  Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(dev).set_disable_warnings(true));
  g_device = dev;
}

bool initialized() { return Kokkos::is_initialized(); }
int device() { return g_device; }
const char *arch() { return GRACE_KK_ARCH; }
bool cublas() { return grace::kk::detail::cublas_ok(); }
std::pair<size_t, size_t> memory_info()
{
  size_t fr = 0, tot = 0;
  if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
  return {fr, tot};
}

// allocation failure injection (tests): a Kokkos Tools allocate callback that throws once, after the allocation (the
// view being assigned keeps what it had; realloc_kokkos has already dropped the old one: a half-grown engine)
namespace {
std::mutex g_fail_mu;
std::string g_fail_label;
int g_fail_left = 0;
void fail_alloc_cb(const Kokkos_Profiling_SpaceHandle, const char *name, const void *, const uint64_t)
{
  std::lock_guard<std::mutex> lk(g_fail_mu);
  if (g_fail_left <= 0 || std::strstr(name, g_fail_label.c_str()) == nullptr || --g_fail_left > 0) return;
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  throw std::runtime_error(std::string("failed to allocate ") + name + " (injected)");
}
}    // namespace
// live device allocations by label (tests / diagnostics): Kokkos Tools allocate / deallocate callbacks
namespace {
std::mutex g_live_mu;
std::map<const void *, std::pair<std::string, uint64_t>> g_live;
uint64_t g_live_bytes = 0, g_peak_bytes = 0;    // bytes allocated since tracking started, their high-water mark
void live_alloc_cb(const Kokkos_Profiling_SpaceHandle, const char *name, const void *ptr, const uint64_t size)
{
  std::lock_guard<std::mutex> lk(g_live_mu);
  g_live[ptr] = {name, size};
  g_live_bytes += size;
  g_peak_bytes = std::max(g_peak_bytes, g_live_bytes);
}
void live_dealloc_cb(const Kokkos_Profiling_SpaceHandle, const char *, const void *ptr, const uint64_t)
{
  std::lock_guard<std::mutex> lk(g_live_mu);
  const auto it = g_live.find(ptr);
  if (it == g_live.end()) return;    // allocated before tracking started
  g_live_bytes -= it->second.second;
  g_live.erase(it);
}
}    // namespace
void debug_track_alloc(bool on)
{
  std::lock_guard<std::mutex> lk(g_live_mu);
  g_live.clear();
  g_live_bytes = g_peak_bytes = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(on ? live_alloc_cb : nullptr);
  Kokkos::Tools::Experimental::set_deallocate_data_callback(on ? live_dealloc_cb : nullptr);
}
std::vector<std::pair<std::string, uint64_t>> debug_live_alloc()
{
  std::lock_guard<std::mutex> lk(g_live_mu);
  std::map<std::string, uint64_t> by;
  for (auto &kv : g_live) by[kv.second.first] += kv.second.second;
  std::vector<std::pair<std::string, uint64_t>> v(by.begin(), by.end());
  std::sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.second > b.second; });
  return v;
}

uint64_t debug_peak_alloc()
{
  std::lock_guard<std::mutex> lk(g_live_mu);
  return g_peak_bytes;
}

void debug_fail_alloc(const std::string &label, int nth)
{
  std::lock_guard<std::mutex> lk(g_fail_mu);
  g_fail_label = label;
  g_fail_left = nth;
  Kokkos::Tools::Experimental::set_allocate_data_callback(nth > 0 ? fail_alloc_cb : nullptr);
}

// ---------------------------------------------------------------- Model

// the fused engine of a Model for one (spline, chunksize): weights, spline tables, cuBLAS and work buffers set up once;
// Systems / Batches of any species set use it in turn, switching its type tables (set_types) under mu
struct SharedEngine {
  std::mutex mu;
  std::unique_ptr<E1> e1;
  std::unique_ptr<E2> e2;
  std::unique_ptr<E3> e3;
  uint64_t owner = 0;    // Engines whose type tables are installed (Engines::id)
  // a device allocation failed in this engine: its work buffers may be half grown (realloc drops the old view first,
  // growth checks one view per block), so no compute uses it again; every holder attaches a fresh engine
  std::atomic<bool> poisoned{false};

  ~SharedEngine()
  {
    if (!Kokkos::is_initialized()) {    // after finalize() the views are already empty: drop without touching Kokkos
      e1.release();
      e2.release();
      e3.release();
    }
  }
};

struct Model::Impl : DeviceState {
  std::string layers;
  std::shared_ptr<grace::GRACE1LModel> m1;
  std::shared_ptr<grace::GRACE2LModel> m2;
  std::shared_ptr<grace::GRACE3LModel> m3;
  std::mutex emu;
  std::map<std::pair<int, int>, std::shared_ptr<SharedEngine>> engines;    // (spline, chunksize)
  int64_t setups = 0;

  Impl()
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_states.insert(this);
  }
  ~Impl()
  {
    std::lock_guard<std::recursive_mutex> dev(g_device_mu);    // the engines free device memory
    std::lock_guard<std::mutex> lk(g_mu);
    g_states.erase(this);
    engines.clear();
  }
  void release() override
  {
    std::lock_guard<std::mutex> lk(emu);
    for (auto &kv : engines) {
      kv.second->e1.reset();
      kv.second->e2.reset();
      kv.second->e3.reset();
    }
    engines.clear();
  }

  // forget the engine of (spline, chunksize) if it is `se` (its memory goes with the last user)
  void drop_engine(int spline, int chunksize, const std::shared_ptr<SharedEngine> &se)
  {
    std::lock_guard<std::mutex> lk(emu);
    auto it = engines.find({spline, chunksize});
    if (it != engines.end() && it->second == se) engines.erase(it);
  }

  // the shared engine for (spline, chunksize), set up with type2mu when new; grace::Error -> invalid_argument
  std::shared_ptr<SharedEngine> engine(int spline, int chunksize, const std::vector<int> &type2mu, uint64_t owner)
  {
    std::lock_guard<std::mutex> lk(emu);
    auto &se = engines[{spline, chunksize}];
    if (se) return se;
    auto e = std::make_shared<SharedEngine>();
    try {
      if (m1) {
        e->e1 = std::make_unique<E1>();
        e->e1->spline_n = spline;
        e->e1->chunksize = chunksize;
        e->e1->set_model(m1, type2mu);
        e->e1->setup();
      } else if (m3) {
        e->e3 = std::make_unique<E3>();
        e->e3->spline_n = spline;
        e->e3->chunksize = chunksize;
        e->e3->set_model(m3, type2mu);
        e->e3->setup();
      } else {
        e->e2 = std::make_unique<E2>();
        e->e2->spline_n = spline;
        e->e2->chunksize = chunksize;
        e->e2->set_model(m2, type2mu);
        e->e2->setup();
      }
    } catch (const grace::Error &err) {
      engines.erase({spline, chunksize});
      throw std::invalid_argument(std::string("this model is not supported on device='cuda' by the fused GPU kernels (") +
                                  err.what() + "); use device='cpu' or TPCalculator");
    } catch (...) {    // e.g. out of device memory: no half-set-up engine stays cached
      engines.erase({spline, chunksize});
      throw;
    }
    e->owner = owner;
    setups++;
    se = e;
    return se;
  }
};

Model::Model(const std::string &path, int /*spline*/) : impl(std::make_shared<Impl>())
{
  impl->layers = detect_layers(path);
  try {
    if (impl->layers == "1l") {
      impl->m1 = std::make_shared<grace::GRACE1LModel>();
      impl->m1->load(path);
    } else if (impl->layers == "3l") {
      impl->m3 = std::make_shared<grace::GRACE3LModel>();
      impl->m3->load(path);
    } else {
      impl->m2 = std::make_shared<grace::GRACE2LModel>();
      impl->m2->load(path);
    }
  } catch (const std::exception &e) {
    throw std::invalid_argument(std::string("cannot load ") + path + ": " + e.what());
  }
}
Model::~Model() = default;
int Model::layers() const { return impl->m1 ? 1 : impl->m3 ? 3 : 2; }
bool Model::has_uq() const { return impl->m1 ? impl->m1->has_uq : impl->m3 ? impl->m3->has_uq : impl->m2->has_uq; }
void Model::validate_uq() const
{
  try {
    if (impl->m1) impl->m1->validate_uq();
    else if (impl->m3) {    // the 3L loader checks its UQ head itself
      if (!impl->m3->has_uq) throw grace::Error("the model has no UQ head (kokkos.npz exported without UQ artifacts)");
    } else impl->m2->validate_uq();
  } catch (const std::exception &e) {
    throw std::invalid_argument(e.what());
  }
}
int64_t Model::engine_setups() const
{
  std::lock_guard<std::mutex> lk(impl->emu);
  return impl->setups;
}
std::vector<std::string> Model::element_names() const
{
  return impl->m1 ? impl->m1->element_names : impl->m3 ? impl->m3->element_names : impl->m2->element_names;
}

// ---------------------------------------------------------------- System

// per-atom value v [n] in the neighbour list's order -> o in the caller's order (o1: device scratch)
template<class VIn>
void scatter_back(int n, const Kokkos::View<int *, Dev> &pm, const VIn &v, const Kokkos::View<double *, Dev> &o1,
                  std::vector<double> &o)
{
  Kokkos::parallel_for("kk:atom_out", Kokkos::RangePolicy<Dev>(0, n), KOKKOS_LAMBDA(const int q) { o1(pm(q)) = v(q); });
  o.assign(n, 0.0);
  Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(o.data(), n);
  Kokkos::deep_copy(h, Kokkos::subview(o1, std::make_pair(0, n)));
}

// a species set (types 1..len(species)) on the Model's shared engine: its type table, the engine's type tables built
// for it (kept, so switching back costs nothing) and its largest pair cutoff
struct Engines {
  static inline std::atomic<uint64_t> next_id{1};
  const uint64_t id = next_id++;
  std::vector<int> type2mu;
  double rc = 0.0;
  std::shared_ptr<SharedEngine> se;
  E1 *e1 = nullptr;    // se's engines (valid while se lives and before finalize)
  E2 *e2 = nullptr;
  E3 *e3 = nullptr;
  bool have_tables = false;
  typename E1::TypeTables t1;
  typename E2::TypeTables t2;
  typename E3::TypeTables t3;
  std::shared_ptr<Model::Impl> mimpl;
  int spline = 1024, chunk = -1;
  // chunksize < 0: the engine's auto chunk (chunk -1: from the free GPU memory at each larger structure); after a
  // device allocation failure (the memory taken in between) fixed chunks, halving
  bool auto_chunk = false;
  int64_t last_n = 0;    // atoms of the last compute (chunksize(): one pass or the engine's auto chunk)

  static bool is_oom(const std::exception &e)
  {
    return dynamic_cast<const std::bad_alloc *>(&e) != nullptr ||
           std::string(e.what()).find("failed to allocate") != std::string::npos;
  }
  // the Model's engine for (spline, chunk), switched to this species set's tables
  void attach()
  {
    se = mimpl->engine(spline, chunk, type2mu, id);
    e1 = se->e1.get();
    e2 = se->e2.get();
    e3 = se->e3.get();
    std::lock_guard<std::mutex> lk(se->mu);
    if (se->owner == id) save_tables();    // this species set made the engine: its tables are installed
  }
  // the engine's auto chunk (0: one pass) for chunk -1, else chunk
  int effective_chunk() const
  {
    if (chunk >= 0) return chunk;
    const int c = e1 ? e1->auto_chunk_size() : e3 ? e3->auto_chunk_size() : e2 ? e2->auto_chunk_size() : 0;
    return c > 0 && c < last_n ? c : 0;
  }
  // auto chunk: the engine of the next smaller chunk (the failed one and its memory dropped): half the chunk in use,
  // from one pass the power of two below natoms / 2; down to 1024; false: none left
  bool shrink(int64_t natoms)
  {
    if (!auto_chunk) return false;
    const int cur = effective_chunk();
    int64_t next = cur / 2;
    if (cur == 0)
      for (next = 1; next * 2 <= natoms / 2; next *= 2) {}
    while (next >= 1024) {
      if (se) mimpl->drop_engine(spline, chunk, se);
      drop();
      chunk = (int) next;
      try {
        attach();
        return true;
      } catch (const std::exception &e) {
        if (!is_oom(e)) throw;
      }
      next = chunk / 2;
    }
    return false;
  }
  // the engine failed an allocation: never used again, gone from the Model (its memory goes with the last holder)
  void poison()
  {
    if (!se) return;
    se->poisoned = true;
    mimpl->drop_engine(spline, chunk, se);
  }
  // in a catch handler of a compute: retry (return) after an allocation failure with a smaller chunk, else throw (the
  // next compute attaches a fresh engine of the same chunk)
  void after_failure(const std::exception &e, int64_t natoms)
  {
    if (!is_oom(e)) throw;
    poison();
    if (shrink(natoms)) return;
    drop();
    throw std::runtime_error("the structure (" + std::to_string(natoms) + " atoms) does not fit the GPU memory " +
                             (auto_chunk ? std::string("even with smaller chunks (down to 1024 atoms)")
                                         : "with chunksize " + std::to_string(chunk) +
                                               " (chunksize='auto' tries smaller ones)") +
                             "; use device='cpu' or a GPU with more memory (" + e.what() + ")");
  }

  // chunksize < 0: auto (the engine's chunk from the free GPU memory, smaller fixed ones after a failure)
  void setup(const Model &model, const std::vector<std::string> &species, int spline_, int chunksize)
  {
    mimpl = model.impl;
    spline = spline_;
    auto_chunk = chunksize < 0;
    chunk = auto_chunk ? -1 : chunksize;
    const auto names = model.element_names();
    type2mu.assign(species.size() + 1, -1);
    for (size_t t = 0; t < species.size(); t++) {
      auto it = std::find(names.begin(), names.end(), species[t]);
      if (it == names.end()) throw std::invalid_argument("element " + species[t] + " is not in the model");
      type2mu[t + 1] = (int) (it - names.begin());
    }
    try {
      attach();
    } catch (const std::exception &e) {    // engine setup: weights, tables, first buffers
      if (!is_oom(e)) throw;
      poison();
      if (!shrink(0)) {
        drop();
        throw std::runtime_error(std::string("the GPU engine of this model (weights, tables, work buffers) does not "
                                             "fit the free GPU memory (another process using the GPU?); use "
                                             "device='cpu' (") + e.what() + ")");
      }
    }
    for (size_t a = 1; a <= species.size(); a++)
      for (size_t b = 1; b <= species.size(); b++) rc = std::max(rc, cutoff_pair((int) a, (int) b));
  }
  void save_tables()
  {
    if (e1) t1 = e1->type_tables();
    else if (e3) t3 = e3->type_tables();
    else t2 = e2->type_tables();
    have_tables = true;
  }
  void drop()
  {
    e1 = nullptr;
    e2 = nullptr;
    e3 = nullptr;
    t1 = typename E1::TypeTables();
    t2 = typename E2::TypeTables();
    t3 = typename E3::TypeTables();
    have_tables = false;
    se.reset();
  }
  // the shared engine, switched to this species set, held until the lock goes; a fresh one when it was dropped or
  // poisoned by an allocation failure (here or in another holder)
  std::unique_lock<std::mutex> lock_engine()
  {
    if (!Kokkos::is_initialized()) throw std::runtime_error("the GPU backend was finalized");
    if (!se || !(e1 || e2 || e3) || se->poisoned) {
      drop();
      attach();
    }
    std::unique_lock<std::mutex> lk(se->mu);
    if (se->poisoned) {    // poisoned by another holder between the check and the lock
      lk.unlock();
      return lock_engine();
    }
    if (se->owner != id) {
      se->owner = 0;    // tables in between while switching
      try {
        if (have_tables) {
          if (e1) e1->use_type_tables(t1);
          else if (e3) e3->use_type_tables(t3);
          else e2->use_type_tables(t2);
        } else {
          if (e1) e1->set_types(type2mu);
          else if (e3) e3->set_types(type2mu);
          else e2->set_types(type2mu);
          save_tables();
        }
      } catch (const grace::Error &err) {
        throw std::runtime_error(err.what());
      }
      se->owner = id;
    }
    return lk;
  }
  double cutoff_pair(int a, int b) const
  {
    return e1 ? e1->cutoff(type2mu[a], type2mu[b]) : e3 ? e3->cutoff(type2mu[a], type2mu[b])
              : e2->cutoff(type2mu[a], type2mu[b]);
  }
  bool uq_available() const { return e1 ? e1->uq_available() : e3 ? e3->uq_available() : e2->uq_available(); }
  // the engine's per-atom buffers back to the device (weights and tables stay); under its lock
  void release_engine_buffers()
  {
    if (!se || se->poisoned || !Kokkos::is_initialized()) return;
    std::lock_guard<std::mutex> lk(se->mu);
    try {
      if (e1) e1->release_buffers();
      else if (e3) e3->release_buffers();
      else if (e2) e2->release_buffers();
    } catch (const std::exception &) {    // a shrink that fails leaves a half-grown engine: not used again
      se->poisoned = true;
      mimpl->drop_engine(spline, chunk, se);
    }
  }
  // the per-atom UQ outputs of the engine in use: f(gamma, sigma, cluster)
  template<class F> void uq_out(F &&f) const
  {
    if (e1) f(e1->uq_gamma(), e1->uq_sigma(), e1->uq_cluster());
    else if (e3) f(e3->uq_gamma(), e3->uq_sigma(), e3->uq_cluster());
    else f(e2->uq_gamma(), e2->uq_sigma(), e2->uq_cluster());
  }
};

struct System::Impl : Engines, DeviceState {
  std::mutex mu;    // one compute at a time
  std::shared_ptr<Model::Impl> model;
  std::vector<std::string> species;
  double skin = 0.5;
  grace::kk::CsrImagesKK<Dev, T> img;
  grace::kk::ImageExchangeKK<E2, grace::kk::CsrImagesKK<Dev, T>> ex;
  grace::kk::ImageExchange3KK<E3, grace::kk::CsrImagesKK<Dev, T>> ex3;
  // neighbour list state: host copies of what forces a rebuild, device positions of the last build
  int n = -1;
  std::vector<int32_t> types;
  std::vector<double> cell;
  std::vector<char> pbc;
  int64_t nbuild = 0;
  grace::kk::NeighborListKK<Dev> nl;
  using V3 = Kokkos::View<double *[3], Kokkos::LayoutRight, Dev>;
  V3 d_pos, d_pos_build, d_pos_new, d_out3;
  V3::host_mirror_type h_pos;
  Kokkos::View<int *, Dev> d_types, d_types_new;
  Kokkos::View<double *, Dev> d_out1;
  typename T::t_kkacc_1d_3 f;
  typename T::t_kkacc_1d eatom;

  void release() override    // device memory goes before Kokkos::finalize
  {
    drop();
    reset_buffers();
  }
  // the System's own device buffers and list dropped (after an allocation failure some may be half grown)
  void reset_buffers()
  {
    img = grace::kk::CsrImagesKK<Dev, T>();
    ex.buf = typename T::t_double_1d();
    ex3.buf = typename T::t_double_1d();
    nl = grace::kk::NeighborListKK<Dev>();
    d_pos = d_pos_build = d_pos_new = d_out3 = V3();
    h_pos = V3::host_mirror_type();
    d_types = d_types_new = Kokkos::View<int *, Dev>();
    d_out1 = Kokkos::View<double *, Dev>();
    f = typename T::t_kkacc_1d_3();
    eatom = typename T::t_kkacc_1d();
    n = -1;
  }

  // positions of this call already in d_pos (caller's order)
  bool need_build(int n_, const int32_t *ty, const double *c, const bool *p) const
  {
    if (n_ != n || std::memcmp(ty, types.data(), sizeof(int32_t) * n_) != 0 ||
        std::memcmp(c, cell.data(), 9 * sizeof(double)) != 0)
      return true;
    for (int k = 0; k < 3; k++)
      if ((bool) pbc[k] != p[k]) return true;
    if (n_ == 0) return false;
    auto x = d_pos;
    auto x0 = d_pos_build;
    double d2 = 0.0;
    Kokkos::parallel_reduce("kk:skin", Kokkos::RangePolicy<Dev>(0, n_), KOKKOS_LAMBDA(const int a, double &m) {
      const double dx = x(a, 0) - x0(a, 0), dy = x(a, 1) - x0(a, 1), dz = x(a, 2) - x0(a, 2);
      const double v = dx * dx + dy * dy + dz * dz;
      if (v > m) m = v;
    }, Kokkos::Max<double>(d2));
    return d2 > 0.25 * skin * skin;
  }

  void ensure(int n_)
  {
    if ((int) d_pos.extent(0) >= std::max(n_, 1)) return;
    const int m = std::max(n_, 1) + std::max(n_, 1) / 8;
    d_pos = V3("kk:pos", m);
    d_pos_build = V3("kk:pos_build", m);
    d_pos_new = V3("kk:pos_new", m);
    d_out3 = V3("kk:out3", m);
    h_pos = Kokkos::create_mirror_view(d_pos);
    d_types = Kokkos::View<int *, Dev>("kk:types", m);
    d_types_new = Kokkos::View<int *, Dev>("kk:types_new", m);
    d_out1 = Kokkos::View<double *, Dev>("kk:out1", m);
  }

  void build(int n_, const int32_t *ty, const double *c, const bool *p)
  {
    n = -1;    // the list is overwritten below: a throw on the way must force a rebuild on the next call
    auto hty = Kokkos::create_mirror_view(d_types);
    for (int a = 0; a < n_; a++) hty(a) = ty[a];
    Kokkos::deep_copy(d_types, hty);
    nl.build(d_pos, n_, c, p, rc + skin);
    auto pm = nl.perm;
    auto tin = d_types;
    auto tnew = d_types_new;
    Kokkos::parallel_for("kk:types_new", Kokkos::RangePolicy<Dev>(0, n_), KOKKOS_LAMBDA(const int q) { tnew(q) = tin(pm(q)); });
    if (n_ > 0) img.build_device(nl, d_types_new, c);
    else img = grace::kk::CsrImagesKK<Dev, T>();
    Kokkos::deep_copy(d_pos_build, d_pos);
    n = n_;
    types.assign(ty, ty + n_);
    cell.assign(c, c + 9);
    pbc.assign(p, p + 3);
    nbuild++;
  }
};

System::System(const Model &model, const std::vector<std::string> &species, double skin, int spline, int chunksize) :
    impl(std::make_unique<Impl>())
{
  std::lock_guard<std::recursive_mutex> dev(g_device_mu);
  if (!Kokkos::is_initialized()) throw std::runtime_error("the GPU backend is not initialised");
  impl->model = model.impl;
  impl->species = species;
  impl->skin = skin;
  impl->setup(model, species, spline, chunksize);
  std::lock_guard<std::mutex> lk(g_mu);
  g_states.insert(impl.get());
}

System::~System()
{
  if (!impl) return;
  std::lock_guard<std::recursive_mutex> dev(g_device_mu);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_states.erase(impl.get());
  }
  if (Kokkos::is_initialized()) impl->release();
  else impl->drop();    // after finalize() the views are already empty (the shared engine drops itself)
}

double System::cutoff() const { return impl->rc; }
int64_t System::rebuilds() const { return impl->nbuild; }
int System::chunksize() const { return impl->effective_chunk(); }
void System::release_device_buffers()
{
  std::lock_guard<std::recursive_mutex> dev(g_device_mu);
  std::lock_guard<std::mutex> lk(impl->mu);
  if (!Kokkos::is_initialized()) return;
  impl->reset_buffers();
  impl->release_engine_buffers();
}

static Result system_compute_once(System::Impl &s, int n, const int32_t *types, const double *pos, const double *cell,
                                  const bool *pbc, bool energies, bool uq, bool energy_only);

Result System::compute(int n, const int32_t *types, const double *pos, const double *cell, const bool *pbc,
                       bool energies, bool uq, bool energy_only)
{
  std::lock_guard<std::recursive_mutex> dev(g_device_mu);
  std::lock_guard<std::mutex> lk(impl->mu);
  for (;;) {
    try {
      impl->last_n = n;
      return system_compute_once(*impl, n, types, pos, cell, pbc, energies, uq, energy_only);
    } catch (const std::exception &e) {
      if (Engines::is_oom(e)) impl->reset_buffers();
      impl->after_failure(e, n);    // a smaller chunk to retry with, else throws
    }
  }
}

static Result system_compute_once(System::Impl &s, int n, const int32_t *types, const double *pos, const double *cell,
                                  const bool *pbc, bool energies, bool uq, bool energy_only)
{
  if (!Kokkos::is_initialized()) throw std::runtime_error("the GPU backend was finalized");
  auto elk = s.lock_engine();
  const int ns = (int) s.species.size();
  for (int a = 0; a < n; a++)
    if (types[a] < 1 || types[a] > ns) throw std::invalid_argument("types must be 1 .. len(species)");
  if (uq && !s.uq_available())
    throw std::invalid_argument("the model has no usable UQ head (uqv6 gamma_only)");
  s.ensure(n);
  for (int a = 0; a < n; a++)
    for (int k = 0; k < 3; k++) s.h_pos(a, k) = pos[3 * a + k];
  Kokkos::deep_copy(Kokkos::subview(s.d_pos, std::make_pair(0, std::max(n, 0)), Kokkos::ALL),
                    Kokkos::subview(s.h_pos, std::make_pair(0, std::max(n, 0)), Kokkos::ALL));
  try {
    if (s.need_build(n, types, cell, pbc)) s.build(n, types, cell, pbc);
  } catch (const grace::Error &e) {
    throw std::invalid_argument(e.what());
  }
  Result res;
  if (!energy_only) res.forces.assign(3 * (size_t) n, 0.0);
  if (energies) res.energies.assign(n, 0.0);
  if (n == 0) return res;
  {    // positions in the list's order -> atoms and images
    auto pm = s.nl.perm;
    auto x = s.d_pos;
    auto xn = s.d_pos_new;
    Kokkos::parallel_for("kk:pos_new", Kokkos::RangePolicy<Dev>(0, n), KOKKOS_LAMBDA(const int q) {
      for (int k = 0; k < 3; k++) xn(q, k) = x(pm(q), k);
    });
  }
  s.img.set_positions(s.d_pos_new);
  const int nall = s.img.nall;
  if ((int) s.f.extent(0) < nall) s.f = typename T::t_kkacc_1d_3("kk:f", nall);
  else Kokkos::deep_copy(s.f, 0.0);
  if (energies) {
    if ((int) s.eatom.extent(0) < nall) s.eatom = typename T::t_kkacc_1d("kk:eatom", nall);
    else Kokkos::deep_copy(s.eatom, 0.0);
  }
  auto fill = [&](auto &in) {
    in.x = s.img.x;
    in.f = s.f;
    in.type = s.img.type;
    in.ilist = s.img.ilist;
    in.numneigh = s.img.numneigh;
    in.neighbors = s.img.neighbors;
    in.inum = n;
    in.nlocal = n;
    in.ntypes = ns;
    in.eflag_global = 1;
    in.eflag_atom = energies ? 1 : 0;
    in.vflag_global = energy_only ? 0 : 1;
    in.energy_only = energy_only;
    in.uq = uq;
    if (energies) in.eatom = s.eatom;
  };
  try {
    if (s.e1) {
      typename E1::RunInput in;
      fill(in);
      auto out = s.e1->run(in);
      res.energy = out.eng_vdwl;
      std::copy(out.virial, out.virial + 6, res.virial);
    } else if (s.e3) {
      typename E3::RunInput in;
      fill(in);
      s.ex3.eng = s.e3;
      s.ex3.img = &s.img;
      in.nall = nall;
      in.ex = &s.ex3;
      auto out = s.e3->run(in);
      res.energy = out.eng_vdwl;
      std::copy(out.virial, out.virial + 6, res.virial);
    } else {
      typename E2::RunInput in;
      fill(in);
      s.ex.eng = s.e2;
      s.ex.img = &s.img;
      in.nall = nall;
      in.exchange = &s.ex;
      auto out = s.e2->run(in);
      res.energy = out.eng_vdwl;
      std::copy(out.virial, out.virial + 6, res.virial);
    }
  } catch (const grace::Error &e) {
    throw std::runtime_error(e.what());
  }
  auto pm = s.nl.perm;
  if (!energy_only) {
    s.img.fold(s.f);
    auto ff = s.f;
    auto o3 = s.d_out3;
    Kokkos::parallel_for("kk:forces_out", Kokkos::RangePolicy<Dev>(0, n), KOKKOS_LAMBDA(const int q) {
      for (int k = 0; k < 3; k++) o3(pm(q), k) = ff(q, k);
    });
    Kokkos::View<double *[3], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> hf(res.forces.data(), n);
    Kokkos::deep_copy(hf, Kokkos::subview(o3, std::make_pair(0, n), Kokkos::ALL));
  }
  auto back = [&](const auto &v, std::vector<double> &o) { scatter_back(n, pm, v, s.d_out1, o); };
  if (energies) back(s.eatom, res.energies);
  if (uq)
    s.uq_out([&](const auto &g, const auto &sg, const auto &cl) {
      back(g, res.gamma); back(sg, res.sigma); back(cl, res.cluster);
    });
  return res;
}

// ---------------------------------------------------------------- Batch

struct Batch::Impl : Engines, DeviceState {
  std::shared_ptr<Model::Impl> model;
  int ns = 0, npasses = 0;
  double host_s = 0.0;    // last compute: seconds of the neighbour search and pass layouts (host or device path)
  bool keep = false;      // keep the device buffers between computes (an MD loop over the same batch)
  // atoms + images per pass that fit after an allocation failure (later computes start there)
  int64_t nall_cap = std::numeric_limits<int64_t>::max();
  std::mutex mu;    // one compute at a time (shared views and engine buffers)
  grace::kk::CsrImagesKK<Dev, T> img;
  grace::kk::ImageExchangeKK<E2, grace::kk::CsrImagesKK<Dev, T>> ex;
  grace::kk::ImageExchange3KK<E3, grace::kk::CsrImagesKK<Dev, T>> ex3;
  grace::kk::BatchAccKK<Dev> acc;
  typename T::t_kkacc_1d_3 f;
  typename T::t_kkacc_1d eatom;
  typename T::t_kkacc_1d_6 vatom;
  Kokkos::View<int *, Dev> d_sys;
  // the device path (compute_device): batch neighbour list, types and positions of a pass in its order
  grace::kk::BatchNeighborListKK<Dev> bnl;
  Kokkos::View<int *, Dev> d_types_new;
  Kokkos::View<double *[3], Kokkos::LayoutRight, Dev> d_pos_new;

  void release() override
  {
    drop();
    release_buffers();
  }
  void release_buffers()    // the batch's own device buffers (after an allocation failure some may be half grown)
  {
    img = grace::kk::CsrImagesKK<Dev, T>();
    ex.buf = typename T::t_double_1d();
    ex3.buf = typename T::t_double_1d();
    acc = grace::kk::BatchAccKK<Dev>();
    f = typename T::t_kkacc_1d_3();
    eatom = typename T::t_kkacc_1d();
    vatom = typename T::t_kkacc_1d_6();
    d_sys = Kokkos::View<int *, Dev>();
    bnl = grace::kk::BatchNeighborListKK<Dev>();
    d_types_new = Kokkos::View<int *, Dev>();
    d_pos_new = Kokkos::View<double *[3], Kokkos::LayoutRight, Dev>();
  }

  // one pass (batch_csr.h layout) into the image views the engines read
  void upload(const grace_runtime::BatchCsr &c)
  {
    const int n = c.n, nall = c.nall, ng = nall - n, mn = std::max(c.maxneigh, 1);
    img.nlocal = n;
    img.nall = nall;
    img.maxneigh = c.maxneigh;
    img.x = decltype(img.x)("batch:x", nall);
    img.type = typename T::t_int_1d("batch:type", nall);
    img.ilist = typename T::t_int_1d("batch:ilist", n);
    img.numneigh = typename T::t_int_1d("batch:numneigh", n);
    img.owner = typename T::t_int_1d("batch:owner", std::max(ng, 1));
    img.tcart = decltype(img.tcart)("batch:tcart", std::max(ng, 1));
    img.neighbors = typename T::t_neighbors_2d("batch:neighbors", n, mn);
    auto hx = Kokkos::create_mirror_view(img.x);
    auto ht = Kokkos::create_mirror_view(img.type);
    auto hi = Kokkos::create_mirror_view(img.ilist);
    auto hn = Kokkos::create_mirror_view(img.numneigh);
    auto ho = Kokkos::create_mirror_view(img.owner);
    auto htc = Kokkos::create_mirror_view(img.tcart);
    auto hb = Kokkos::create_mirror_view(img.neighbors);
    for (int a = 0; a < nall; a++) {
      for (int k = 0; k < 3; k++) hx(a, k) = c.x[3 * (size_t) a + k];
      ht(a) = c.type[a];
    }
    for (int i = 0; i < n; i++) {
      hi(i) = i;
      hn(i) = c.numneigh[i];
      for (int q = 0; q < c.numneigh[i]; q++) hb(i, q) = c.neighbors[(size_t) i * mn + q];
    }
    for (int g = 0; g < ng; g++) {
      ho(g) = c.owner[g];
      for (int k = 0; k < 3; k++) htc(g, k) = c.tcart[3 * (size_t) g + k];
    }
    Kokkos::deep_copy(img.x, hx);
    Kokkos::deep_copy(img.type, ht);
    Kokkos::deep_copy(img.ilist, hi);
    Kokkos::deep_copy(img.numneigh, hn);
    Kokkos::deep_copy(img.owner, ho);
    Kokkos::deep_copy(img.tcart, htc);
    Kokkos::deep_copy(img.neighbors, hb);
    d_sys = Kokkos::View<int *, Dev>("batch:sys", n);
    Kokkos::deep_copy(d_sys, Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(c.sys.data(), n));
    auto grow = [&](auto &v, int m, const char *name) {
      using V = std::decay_t<decltype(v)>;
      if ((int) v.extent(0) < m) v = V(name, m);
      else Kokkos::deep_copy(v, 0.0);
    };
    grow(f, nall, "batch:f");
    grow(eatom, nall, "batch:eatom");
    grow(vatom, nall, "batch:vatom");
  }

  // one engine pass over structures [s0, s0 + nsys) of the call; results into res at the input indices (c.perm)
  // the engine over the pass in img (f, eatom, vatom sized to it), images folded into their owners
  void run_engine(int n, bool uq, bool energy_only)
  {
    auto fill = [&](auto &in) {
      in.x = img.x;
      in.f = f;
      in.type = img.type;
      in.ilist = img.ilist;
      in.numneigh = img.numneigh;
      in.neighbors = img.neighbors;
      in.inum = n;
      in.nlocal = n;
      in.ntypes = ns;
      in.eflag_global = 1;
      in.eflag_atom = 1;    // per-structure energies: sums of the per-atom ones
      in.vflag_global = energy_only ? 0 : 1;
      in.vflag_atom = energy_only ? 0 : 1;
      in.energy_only = energy_only;
      in.uq = uq;
      in.eatom = eatom;
      in.vatom = vatom;
    };
    try {
      if (e1) {
        typename E1::RunInput in;
        fill(in);
        e1->run(in);
      } else if (e3) {
        typename E3::RunInput in;
        fill(in);
        ex3.eng = e3;
        ex3.img = &img;
        in.nall = img.nall;
        in.ex = &ex3;
        e3->run(in);
      } else {
        typename E2::RunInput in;
        fill(in);
        ex.eng = e2;
        ex.img = &img;
        in.nall = img.nall;
        in.exchange = &ex;
        e2->run(in);
      }
    } catch (const grace::Error &e) {
      throw std::runtime_error(e.what());
    }
    if (!energy_only) {
      img.fold(f);
      img.fold(vatom);
    }
  }

  void run_pass(const grace_runtime::BatchCsr &c, int s0, int nsys, bool energies, bool uq, bool energy_only,
                BatchResult &res)
  {
    const int n = c.n;
    upload(c);
    run_engine(n, uq, energy_only);
    acc.sum(n, nsys, eatom, vatom, d_sys);
    auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), acc.acc);
    for (int s = 0; s < nsys; s++) {
      res.energy[s0 + s] = ha(s, 0);
      if (!energy_only)
        for (int k = 0; k < 6; k++) res.virial[6 * (size_t) (s0 + s) + k] = ha(s, 1 + k);
    }
    const auto &pm = c.perm;
    if (!energy_only) {
      Kokkos::View<double *[3], Kokkos::LayoutRight, Dev> t("batch:f_out", n);    // contiguous, any layout of f
      auto ff = f;
      Kokkos::parallel_for("batch:f_out", Kokkos::RangePolicy<Dev>(0, n), KOKKOS_LAMBDA(const int q) {
        for (int k = 0; k < 3; k++) t(q, k) = ff(q, k);
      });
      auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), t);
      for (int q = 0; q < n; q++)
        for (int k = 0; k < 3; k++) res.forces[3 * (size_t) pm[q] + k] = hf(q, k);
    }
    auto back = [&](const auto &v, std::vector<double> &o) {
      auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(v, std::make_pair(0, n)));
      for (int q = 0; q < n; q++) o[pm[q]] = h(q);
    };
    if (energies) back(eatom, res.energies);
    if (uq)
      uq_out([&](const auto &g, const auto &sg, const auto &cl) {
        back(g, res.gamma); back(sg, res.sigma); back(cl, res.cluster);
      });
  }

  using PosView = Kokkos::View<const double *[3], Kokkos::LayoutRight, Dev, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using TypeView = Kokkos::View<const int *, Dev, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  // structures [s0, s1) of a device call laid out on the device (img, d_sys, f / eatom / vatom); returns atoms + images
  int64_t build_device_pass(const PosView &pos, const TypeView &typ, int s0, int s1, const int64_t *start,
                            const double *cell, const bool *pbc)
  {
    const int64_t a0 = start[s0];
    const int n = (int) (start[s1] - a0);
    std::vector<int64_t> st(s1 - s0 + 1);
    for (int k = 0; k <= s1 - s0; k++) st[k] = start[s0 + k] - a0;
    const Kokkos::View<const double *[3], Kokkos::LayoutRight, Dev> sub =
        Kokkos::subview(pos, std::make_pair(a0, a0 + n), Kokkos::ALL);
    bnl.build(sub, s1 - s0, st.data(), cell + 9 * (size_t) s0, pbc + 3 * (size_t) s0, rc);
    if ((int) d_types_new.extent(0) < std::max(n, 1)) {
      const int m = std::max(n, 1) + std::max(n, 1) / 8;
      d_types_new = Kokkos::View<int *, Dev>("batch:types_new", m);
      d_pos_new = Kokkos::View<double *[3], Kokkos::LayoutRight, Dev>("batch:pos_new", m);
    }
    auto pm = bnl.perm;
    auto tn = d_types_new;
    auto pn = d_pos_new;
    Kokkos::parallel_for("batch:new_order", Kokkos::RangePolicy<Dev>(0, n), KOKKOS_LAMBDA(const int q) {
      const int64_t a = a0 + pm(q);
      tn(q) = typ(a);
      for (int k = 0; k < 3; k++) pn(q, k) = pos(a, k);
    });
    if (n > 0) {
      img.build_device(bnl, d_types_new, nullptr);
      img.set_positions(d_pos_new);
    } else {
      img = grace::kk::CsrImagesKK<Dev, T>();
    }
    d_sys = bnl.sys;
    auto grow = [&](auto &v, int m, const char *name) {
      using V = std::decay_t<decltype(v)>;
      if ((int) v.extent(0) < m) v = V(name, m);
      else Kokkos::deep_copy(v, 0.0);
    };
    grow(f, std::max(img.nall, 1), "batch:f");
    grow(eatom, std::max(img.nall, 1), "batch:eatom");
    grow(vatom, std::max(img.nall, 1), "batch:vatom");
    return img.nall;
  }

  // the engine over the pass built by build_device_pass, results into the caller's device outputs
  void run_device_pass(int s0, int nsys, int64_t a0, bool energies, bool uq, bool energy_only, const DeviceOut &o)
  {
    const int n = img.nlocal;
    using U = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    if (n > 0) {
      run_engine(n, uq, energy_only);
      acc.sum(n, nsys, eatom, vatom, d_sys);
    } else {
      acc.acc = decltype(acc.acc)("csr:batch", nsys);
    }
    auto ac = acc.acc;
    Kokkos::View<double *, Dev, U> e_out(o.energy + s0, nsys);
    const bool want_v = !energy_only && o.virial;
    Kokkos::View<double *[6], Kokkos::LayoutRight, Dev, U> v_out(want_v ? o.virial + 6 * (size_t) s0 : nullptr,
                                                                 want_v ? nsys : 0);
    Kokkos::parallel_for("batch:sys_out", Kokkos::RangePolicy<Dev>(0, nsys), KOKKOS_LAMBDA(const int s) {
      e_out(s) = ac(s, 0);
      if (want_v)
        for (int k = 0; k < 6; k++) v_out(s, k) = ac(s, 1 + k);
    });
    if (n == 0) return;
    auto pm = bnl.perm;
    if (!energy_only && o.forces) {
      Kokkos::View<double *[3], Kokkos::LayoutRight, Dev, U> f_out(o.forces, a0 + n);
      auto ff = f;
      Kokkos::parallel_for("batch:f_out_dev", Kokkos::RangePolicy<Dev>(0, n), KOKKOS_LAMBDA(const int q) {
        for (int k = 0; k < 3; k++) f_out(a0 + pm(q), k) = ff(q, k);
      });
    }
    if (energies) atom_out(eatom, o.energies, n, a0);
    if (uq)
      uq_out([&](const auto &g, const auto &sg, const auto &cl) {
        atom_out(g, o.gamma, n, a0); atom_out(sg, o.sigma, n, a0); atom_out(cl, o.cluster, n, a0);
      });
  }

  // per-atom values of the pass [n] (its order) -> dst (device, caller's order from atom a0 of the call)
  template<class V> void atom_out(const V &v, double *dst, int n, int64_t a0) const
  {
    if (!dst) return;
    Kokkos::View<double *, Dev, Kokkos::MemoryTraits<Kokkos::Unmanaged>> out(dst, a0 + n);
    auto pm = bnl.perm;
    Kokkos::parallel_for("batch:atom_out_dev", Kokkos::RangePolicy<Dev>(0, n),
                         KOKKOS_LAMBDA(const int q) { out(a0 + pm(q)) = (double) v(q); });
  }
};

Batch::Batch(const Model &model, const std::vector<std::string> &species, int spline, int chunksize) :
    impl(std::make_unique<Impl>())
{
  std::lock_guard<std::recursive_mutex> dev(g_device_mu);
  if (!Kokkos::is_initialized()) throw std::runtime_error("the GPU backend is not initialised");
  impl->model = model.impl;
  impl->ns = (int) species.size();
  impl->setup(model, species, spline, chunksize);
  std::lock_guard<std::mutex> lk(g_mu);
  g_states.insert(impl.get());
}

Batch::~Batch()
{
  if (!impl) return;
  std::lock_guard<std::recursive_mutex> dev(g_device_mu);
  {
    std::lock_guard<std::mutex> lk(g_mu);
    g_states.erase(impl.get());
  }
  if (Kokkos::is_initialized()) impl->release();
  else impl->drop();    // after finalize() the views are already empty (the shared engine drops itself)
}

double Batch::cutoff() const { return impl->rc; }
int Batch::passes() const { return impl->npasses; }
double Batch::host_seconds() const { return impl->host_s; }
bool Batch::keep_buffers() const { return impl->keep; }
void Batch::release_device_buffers()
{
  std::lock_guard<std::recursive_mutex> dev(g_device_mu);
  std::lock_guard<std::mutex> lk(impl->mu);
  if (!Kokkos::is_initialized()) return;
  impl->release_buffers();
  impl->release_engine_buffers();
}
void Batch::set_keep_buffers(bool on)
{
  std::lock_guard<std::mutex> lk(impl->mu);
  impl->keep = on;
}
int Batch::chunksize() const { return impl->effective_chunk(); }

static BatchResult batch_compute_once(Batch::Impl &b, int nsys, const int32_t *natoms, int64_t ntot,
                                      const int32_t *types, const double *pos, const double *cell, const bool *pbc,
                                      bool energies, bool uq, bool energy_only, int num_threads, int64_t max_nall);

BatchResult Batch::compute(int nsys, const int32_t *natoms, int64_t ntot, const int32_t *types, const double *pos,
                           const double *cell, const bool *pbc, bool energies, bool uq, bool energy_only,
                           int num_threads, int64_t max_nall)
{
  std::lock_guard<std::recursive_mutex> dev(g_device_mu);
  std::lock_guard<std::mutex> lk(impl->mu);
  max_nall = std::min(max_nall, impl->nall_cap);
  for (;;) {
    try {
      impl->last_n = ntot;
      BatchResult r = batch_compute_once(*impl, nsys, natoms, ntot, types, pos, cell, pbc, energies, uq, energy_only,
                                         num_threads, max_nall);
      // the device memory of the passes back (a batch is a job, unless keep_buffers: an MD loop): the batch's
      // buffers and the engine's per-atom ones; the model's weights and tables stay cached
      if (!impl->keep) {
        impl->release_buffers();
        impl->release_engine_buffers();
      }
      return r;
    } catch (const std::exception &e) {
      // atoms + images per pass first (the engines keep per-image rows beyond the chunk: GRACE-3L ~64 KB), then the
      // chunk; the engine stays: release_buffers reallocates all its work buffers (a failed one cannot recover and is
      // dropped, lock_engine attaches a fresh one)
      if (Engines::is_oom(e) && max_nall > 16384) {
        impl->release_buffers();
        impl->release_engine_buffers();
        max_nall = impl->nall_cap = std::max<int64_t>(16384, max_nall / 2);
        continue;
      }
      impl->after_failure(e, ntot);
    }
  }
}

static BatchResult batch_compute_once(Batch::Impl &b, int nsys, const int32_t *natoms, int64_t ntot,
                                      const int32_t *types, const double *pos, const double *cell, const bool *pbc,
                                      bool energies, bool uq, bool energy_only, int num_threads, int64_t max_nall)
{
  if (!Kokkos::is_initialized()) throw std::runtime_error("the GPU backend was finalized");
  auto elk = b.lock_engine();
  if (max_nall < 1) throw std::invalid_argument("max_nall must be >= 1");
  int64_t sum = 0;
  for (int s = 0; s < nsys; s++) {
    if (natoms[s] < 0) throw std::invalid_argument("natoms must be >= 0");
    sum += natoms[s];
  }
  if (sum != ntot) throw std::invalid_argument("natoms must sum to the number of atoms");
  for (int64_t a = 0; a < ntot; a++)
    if (types[a] < 1 || types[a] > b.ns) throw std::invalid_argument("types must be 1 .. len(species)");
  if (uq && !b.uq_available()) throw std::invalid_argument("the model has no usable UQ head (uqv6 gamma_only)");
  const int nth = num_threads > 0 ? num_threads : affinity_threads();
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  const auto bs = grace_runtime::batch_search(nsys, natoms, pos, cell, pbc, b.rc, nth);
  b.host_s = std::chrono::duration<double>(clk::now() - t0).count();
  BatchResult res;
  res.energy.assign(nsys, 0.0);
  if (!energy_only) {
    res.virial.assign(6 * (size_t) nsys, 0.0);
    res.forces.assign(3 * (size_t) ntot, 0.0);
  }
  if (energies) res.energies.assign(ntot, 0.0);
  if (uq) {
    res.gamma.assign(ntot, 0.0);
    res.sigma.assign(ntot, 0.0);
    res.cluster.assign(ntot, 0.0);
  }
  b.npasses = 0;
  for (int s0 = 0; s0 < nsys;) {    // whole structures per pass, at most max_nall atoms + images (at least one)
    int s1 = s0;
    int64_t na = 0, nall = 0;
    while (s1 < nsys && (s1 == s0 || nall + natoms[s1] + bs.images(s1) <= max_nall)) {
      na += natoms[s1];
      nall += natoms[s1] + bs.images(s1);
      s1++;
    }
    if (na > 0) {
      const auto t1 = clk::now();
      auto layout = grace_runtime::batch_layout(bs, types, pos, s0, s1, nth);
      b.host_s += std::chrono::duration<double>(clk::now() - t1).count();
      b.run_pass(layout, s0, s1 - s0, energies, uq, energy_only, res);
      b.npasses++;
    }
    s0 = s1;
  }
  return res;
}

static void batch_compute_device_once(Batch::Impl &b, int nsys, const int32_t *natoms, int64_t ntot,
                                      const int32_t *types, const double *pos, const double *cell, const bool *pbc,
                                      bool energies, bool uq, bool energy_only, int64_t max_nall, const DeviceOut &o)
{
  if (!Kokkos::is_initialized()) throw std::runtime_error("the GPU backend was finalized");
  auto elk = b.lock_engine();
  if (max_nall < 1) throw std::invalid_argument("max_nall must be >= 1");
  if (!o.energy) throw std::invalid_argument("compute_device needs the energy output");
  std::vector<int64_t> start(nsys + 1, 0);
  for (int s = 0; s < nsys; s++) {
    if (natoms[s] < 0) throw std::invalid_argument("natoms must be >= 0");
    start[s + 1] = start[s] + natoms[s];
  }
  if (start[nsys] != ntot) throw std::invalid_argument("natoms must sum to the number of atoms");
  if (ntot > INT_MAX) throw std::invalid_argument("too many atoms for one call");
  if (uq && !b.uq_available()) throw std::invalid_argument("the model has no usable UQ head (uqv6 gamma_only)");
  const Batch::Impl::PosView p(pos, ntot);
  const Batch::Impl::TypeView t(types, ntot);
  int bad = 0;
  const int ns = b.ns;
  Kokkos::parallel_reduce("batch:check_types", Kokkos::RangePolicy<Dev>(0, ntot),
      KOKKOS_LAMBDA(const int64_t a, int &v) { if (t(a) < 1 || t(a) > ns) v++; }, bad);
  if (bad) throw std::invalid_argument("types must be 1 .. len(species)");
  using clk = std::chrono::steady_clock;
  b.npasses = 0;
  b.host_s = 0.0;
  auto timed_build = [&](int s0, int s1) {
    const auto t0 = clk::now();
    const int64_t nall = b.build_device_pass(p, t, s0, s1, start.data(), cell, pbc);
    Kokkos::fence();
    b.host_s += std::chrono::duration<double>(clk::now() - t0).count();
    return nall;
  };
  if (nsys == 0) return;
  const int64_t nall = timed_build(0, nsys);
  if (nall <= max_nall || nsys == 1) {
    b.run_device_pass(0, nsys, 0, energies, uq, energy_only, o);
    b.npasses = 1;
  } else {
    // whole structures per pass, at most max_nall atoms + images (at least one): image counts of this build
    Kokkos::View<int64_t *, Dev> cnt("batch:images_per_sys", nsys);
    auto ow = b.img.owner;
    auto sy = b.bnl.sys;
    const int nl = b.img.nlocal;
    Kokkos::parallel_for("batch:count_images", Kokkos::RangePolicy<Dev>(0, b.img.nall - nl),
                         KOKKOS_LAMBDA(const int g) { Kokkos::atomic_add(&cnt(sy(ow(g))), (int64_t) 1); });
    auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cnt);
    for (int s0 = 0; s0 < nsys;) {
      int s1 = s0;
      int64_t na = 0;
      while (s1 < nsys && (s1 == s0 || na + natoms[s1] + hc(s1) <= max_nall)) {
        na += natoms[s1] + hc(s1);
        s1++;
      }
      timed_build(s0, s1);
      b.run_device_pass(s0, s1 - s0, start[s0], energies, uq, energy_only, o);
      b.npasses++;
      s0 = s1;
    }
  }
  Kokkos::fence();
}

void Batch::compute_device(int nsys, const int32_t *natoms, int64_t ntot, const int32_t *types, const double *pos,
                           const double *cell, const bool *pbc, bool energies, bool uq, bool energy_only,
                           int64_t max_nall, const DeviceOut &out)
{
  std::lock_guard<std::recursive_mutex> dev(g_device_mu);
  std::lock_guard<std::mutex> lk(impl->mu);
  max_nall = std::min(max_nall, impl->nall_cap);
  for (;;) {
    try {
      impl->last_n = ntot;
      batch_compute_device_once(*impl, nsys, natoms, ntot, types, pos, cell, pbc, energies, uq, energy_only, max_nall,
                                out);
      if (!impl->keep) {
        impl->release_buffers();
        impl->release_engine_buffers();
      }
      return;
    } catch (const std::exception &e) {
      if (Engines::is_oom(e) && max_nall > 16384) {    // as compute: smaller passes first, then the chunk
        impl->release_buffers();
        impl->release_engine_buffers();
        max_nall = impl->nall_cap = std::max<int64_t>(16384, max_nall / 2);
        continue;
      }
      impl->after_failure(e, ntot);
    }
  }
}

void finalize()
{
  std::lock_guard<std::recursive_mutex> dev(g_device_mu);
  std::lock_guard<std::mutex> lk(g_mu);
  if (!Kokkos::is_initialized()) return;
  for (auto *s : g_states) s->release();
  Kokkos::finalize();
  g_finalized = true;
}

}    // namespace grace_runtime_kk

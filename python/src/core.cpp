// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// grace_runtime._core: a GRACE-1L/2L model (kokkos.npz) evaluated on CPU from bonds in CSR form.
//
// A Model holds the weights once and one driver per species list (created on first use); every call names its
// species, so several calculators can share one Model from several threads. The GIL is released while drivers are
// created and while they compute; each driver has its own mutex (its buffers are per call state).
#include "grace/cell_list.h"
#include "grace/d3.h"
#include "grace/fused_1l.h"
#include "grace/fused_2l.h"
#include "grace/fused_3l.h"
#include "dispersion_args.h"
#include "neighbors.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <set>
#include <string>
#include <vector>
#ifdef __linux__
#include <sched.h>
#else
#include <thread>
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace {

int affinity_threads()
{
  if (const char *e = std::getenv("OMP_NUM_THREADS")) {
    const int v = std::atoi(e);
    if (v > 0) return v;
  }
#ifdef __linux__
  cpu_set_t s;
  if (sched_getaffinity(0, sizeof(s), &s) == 0) return std::max(1, CPU_COUNT(&s));
  return 1;
#else
  return std::max(1u, std::thread::hardware_concurrency());   // no CPU affinity API (macOS)
#endif
}

// member names of a zip archive from its central directory (only the end of the file is read)
std::set<std::string> zip_names(const std::string &path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw nb::value_error(("cannot open " + path).c_str());
  const int64_t size = f.tellg();
  const int64_t tail = std::min<int64_t>(size, 65536 + 22);
  std::vector<unsigned char> b(tail);
  f.seekg(size - tail);
  f.read(reinterpret_cast<char *>(b.data()), tail);
  auto u16 = [&](const unsigned char *p) { return (uint32_t) p[0] | (uint32_t) p[1] << 8; };
  auto u32 = [&](const unsigned char *p) { return u16(p) | u16(p + 2) << 16; };
  int64_t eocd = -1;
  for (int64_t q = tail - 22; q >= 0; q--)
    if (u32(&b[q]) == 0x06054b50u) { eocd = q; break; }
  if (eocd < 0) throw nb::value_error(("not a zip/npz file: " + path).c_str());
  const uint32_t cd_size = u32(&b[eocd + 12]), cd_off = u32(&b[eocd + 16]);
  std::vector<unsigned char> cd(cd_size);
  f.seekg(cd_off);
  f.read(reinterpret_cast<char *>(cd.data()), cd_size);
  if (!f) throw nb::value_error(("cannot read the zip directory of " + path).c_str());
  std::set<std::string> names;
  for (size_t q = 0; q + 46 <= cd.size() && u32(&cd[q]) == 0x02014b50u;) {
    const uint32_t nl = u16(&cd[q + 28]), xl = u16(&cd[q + 30]), cl = u16(&cd[q + 32]);
    names.emplace(reinterpret_cast<const char *>(&cd[q + 46]), nl);
    q += 46 + nl + xl + cl;
  }
  return names;
}

std::string detect_arch(const std::string &path)
{
  const auto n = zip_names(path);
  if (n.count("mlp_rad_W0.npy")) return "1l";
  if (n.count("B0_lin_transform_W.npy")) return "2l";
  if (n.count("prod_names.npy")) return "3l";
  throw nb::value_error(("not a GRACE-1L/2L/3L kokkos.npz: " + path).c_str());
}

template<class T> nb::ndarray<nb::numpy, double> owned(std::unique_ptr<T[]> p, std::initializer_list<size_t> shape)
{
  T *raw = p.release();
  nb::capsule own(raw, [](void *q) noexcept { delete[] static_cast<T *>(q); });
  return nb::ndarray<nb::numpy, double>(raw, shape, own);
}

// input arrays: C-contiguous in host memory. A DLPack tensor on another device (CUDA) is rejected by the argument
// conversion (TypeError) before any host access to its data.
template<class T, class Shape> using HostArray = nb::ndarray<T, Shape, nb::c_contig, nb::device::cpu>;

// a std::vector handed to numpy without a copy
template<class T> nb::ndarray<nb::numpy, T> from_vec(std::vector<T> &&v, std::initializer_list<size_t> shape)
{
  auto *p = new std::vector<T>(std::move(v));
  nb::capsule own(p, [](void *q) noexcept { delete static_cast<std::vector<T> *>(q); });
  return nb::ndarray<nb::numpy, T>(p->data(), shape, own);
}

// cell-list neighbour search (NeighborList backend "native"): perm, offset, j, io, jo, shift
auto neighbor_search(HostArray<const double, nb::shape<-1, 3>> x,
                     HostArray<const double, nb::shape<3, 3>> cell, std::array<bool, 3> pbc, double cutoff,
                     int num_threads)
{
  grace_runtime::NeighborResult r;
  const int64_t n = (int64_t) x.shape(0);
  {
    nb::gil_scoped_release nogil;
    try {
      r = grace_runtime::neighbor_search(x.data(), n, cell.data(), pbc.data(), cutoff,
                                        num_threads > 0 ? num_threads : affinity_threads());
    } catch (const std::invalid_argument &e) {
      throw grace::Error(e.what());
    }
  }
  const size_t nb_ = r.j.size();
  return std::make_tuple(from_vec(std::move(r.perm), {(size_t) n}), from_vec(std::move(r.offset), {(size_t) n + 1}),
                         from_vec(std::move(r.j), {nb_}), from_vec(std::move(r.io), {nb_}),
                         from_vec(std::move(r.jo), {nb_}), from_vec(std::move(r.shift), {nb_, 3}));
}

// r_ij = x[j] - x[i] + shift, one row per bond (the calculator's per-step bond vectors)
nb::ndarray<nb::numpy, double> bond_vectors(HostArray<const double, nb::shape<-1, 3>> x,
                                            HostArray<const int64_t, nb::ndim<1>> i,
                                            HostArray<const int32_t, nb::ndim<1>> j,
                                            HostArray<const double, nb::shape<-1, 3>> shift, int num_threads)
{
  const int64_t nb_ = (int64_t) i.shape(0), n = (int64_t) x.shape(0);
  if ((int64_t) j.shape(0) != nb_ || (int64_t) shift.shape(0) != nb_)
    throw nb::value_error("i, j and shift differ in length");
  const double *xp = x.data(), *sp = shift.data();
  const int64_t *ip = i.data();
  const int32_t *jp = j.data();
  for (int64_t q = 0; q < nb_; q++)
    if (ip[q] < 0 || ip[q] >= n || jp[q] < 0 || jp[q] >= n) throw nb::value_error("i and j must index rows of x");
  std::unique_ptr<double[]> r(new double[(size_t) 3 * std::max<int64_t>(nb_, 1)]);
  double *rp = r.get();
  {
    nb::gil_scoped_release nogil;
    const int nth = num_threads > 0 ? num_threads : affinity_threads();
#pragma omp parallel for schedule(static) num_threads(nth)
    for (int64_t q = 0; q < nb_; q++)
      for (int d = 0; d < 3; d++) rp[3 * q + d] = xp[3 * jp[q] + d] - xp[3 * ip[q] + d] + sp[3 * q + d];
  }
  return owned(std::move(r), {(size_t) nb_, 3});
}

// dispersion correction alone (grace/dispersion.h; the calculator adds it to the model): params a dict
// (dispersion_args.h); numbers [n] atomic numbers, x [n, 3] in the caller's order, cell full rank (open directions
// padded). energy, energies [n] and, unless energy_only, forces [n, 3], virial [6]
// (xx yy zz xy xz yz, W = -dE/dstrain)
nb::dict dispersion_compute(const nb::dict &params, HostArray<const int32_t, nb::ndim<1>> numbers,
                            HostArray<const double, nb::shape<-1, 3>> x, HostArray<const double, nb::shape<3, 3>> cell,
                            std::array<bool, 3> pbc, bool energy_only, int num_threads)
{
  const int64_t n = (int64_t) numbers.shape(0);
  if ((int64_t) x.shape(0) != n) throw nb::value_error("positions and numbers differ in length");
  const grace::DispersionParams p = dispersion_params(params);
  std::vector<double> e(n, 0.0), f(energy_only ? 0 : 3 * n, 0.0);
  double energy = 0.0, vir[6] = {0, 0, 0, 0, 0, 0};
  {
    nb::gil_scoped_release nogil;
    try {
      std::vector<int> type;
      const auto species = grace::d3::species_of(numbers.data(), n, type);
      const auto t = grace::d3::make_tables(species.data(), (int) species.size(), p);
      grace::CellList cl;
      cl.build(x.data(), n, cell.data(), pbc.data(), 0.5 * std::min(p.cutoff, p.cn_cutoff));
      grace::d3::compute_cpu(t, type.data(), n, cl, num_threads > 0 ? num_threads : affinity_threads(), energy_only,
                             energy, e.data(), energy_only ? nullptr : f.data(), energy_only ? nullptr : vir);
    } catch (const std::invalid_argument &ex) {
      throw grace::Error(ex.what());
    }
  }
  nb::dict d;
  d["energy"] = energy;
  d["energies"] = from_vec(std::move(e), {(size_t) n});
  if (!energy_only) {
    d["forces"] = from_vec(std::move(f), {(size_t) n, 3});
    d["virial"] = from_vec(std::vector<double>(vir, vir + 6), {6});
  }
  return d;
}

// the published D3(BJ) parameters of a functional (lower case), a2 in Bohr
std::map<std::string, double> d3bj_preset(const std::string &name)
{
  try {
    const auto &q = grace::d3bj_preset(name);
    return {{"s6", q.s6}, {"a1", q.a1}, {"s8", q.s8}, {"a2_bohr", q.a2_bohr}};
  } catch (const std::invalid_argument &ex) {
    throw nb::value_error(ex.what());
  }
}

std::vector<std::string> d3bj_presets()
{
  std::vector<std::string> v;
  for (const auto &q : grace::kD3BJPresets) v.emplace_back(q.name);
  return v;
}

}    // namespace

class Model {
 public:
  Model(const std::string &path, int spline) : path_(path)
  {
    if (spline < 16) throw nb::value_error("spline must be >= 16 (intervals of the radial tables)");
    arch_ = detect_arch(path);
    fs_.spline_n = spline;
    nb::gil_scoped_release nogil;
    if (arch_ == "1l") {
      m1_ = std::make_shared<grace::GRACE1LModel>();
      m1_->load(path);
      names_ = m1_->element_names;
    } else if (arch_ == "3l") {
      m3_ = std::make_shared<grace::GRACE3LModel>();
      m3_->load(path);
      names_ = m3_->element_names;
    } else {
      m2_ = std::make_shared<grace::GRACE2LModel>();
      m2_->load(path);
      names_ = m2_->element_names;
    }
  }

  nb::ndarray<nb::numpy, double> cutoff_matrix(const std::vector<std::string> &species)
  {
    std::shared_ptr<Entry> e;
    {
      nb::gil_scoped_release nogil;
      e = entry(species);
    }
    const size_t k = species.size();
    std::unique_ptr<double[]> c(new double[k * k]);
    for (size_t a = 0; a < k; a++)
      for (size_t b = 0; b < k; b++)
        c[a * k + b] = e->d1 ? e->d1->cutoff((int) a + 1, (int) b + 1)
                     : e->d3 ? e->d3->cutoff((int) a + 1, (int) b + 1) : e->d2->cutoff((int) a + 1, (int) b + 1);
    return owned(std::move(c), {k, k});
  }

  nb::dict compute(const std::vector<std::string> &species, HostArray<const int32_t, nb::ndim<1>> types,
                   HostArray<const int64_t, nb::ndim<1>> offset,
                   HostArray<const int32_t, nb::ndim<1>> j,
                   HostArray<const double, nb::shape<-1, 3>> rij, bool energies, int num_threads,
                   bool uq, bool energy_only)
  {
    if (types.shape(0) > (size_t) INT_MAX) throw nb::value_error("more atoms than an int can index");
    const int n = (int) types.shape(0);
    const int ns = (int) species.size();
    if ((int64_t) offset.shape(0) != (int64_t) n + 1) throw nb::value_error("offset must have len(types) + 1 entries");
    if (j.shape(0) != rij.shape(0)) throw nb::value_error("j and rij differ in length");
    const int32_t *ty = types.data();
    const int64_t *off = offset.data();
    const int32_t *jj = j.data();
    for (int a = 0; a < n; a++)
      if (ty[a] < 1 || ty[a] > ns) throw nb::value_error("types must be 1 .. len(species)");
    if (off[0] != 0 || off[n] != (int64_t) j.shape(0)) throw nb::value_error("offset[0] must be 0 and offset[-1] == len(j)");
    for (int a = 0; a < n; a++)
      if (off[a + 1] < off[a]) throw nb::value_error("offset must be non-decreasing");
    for (int64_t q = 0; q < off[n]; q++)
      if (jj[q] < 0 || jj[q] >= n) throw nb::value_error("j must index atoms 0 .. len(types)-1");
    std::unique_ptr<double[]> f(new double[(size_t) 3 * std::max(n, 1)]());
    std::unique_ptr<double[]> e(energies ? new double[std::max(n, 1)]() : nullptr);
    std::unique_ptr<double[]> v(new double[6]());
    std::unique_ptr<double[]> ug(uq ? new double[std::max(n, 1)]() : nullptr), us(uq ? new double[std::max(n, 1)]() : nullptr),
        uc(uq ? new double[std::max(n, 1)]() : nullptr);
    double energy = 0.0;
    {
      nb::gil_scoped_release nogil;
      auto ent = entry(species);
      std::lock_guard<std::mutex> lk(ent->mu);
      grace::CsrBonds src{ty, off, jj, rij.data()};
      grace::Outputs out;
      out.f = f.get();
      out.eatom = e.get();
      grace::ComputeFlags fl;
      fl.eflag_atom = energies;
      fl.virial = !energy_only;
      fl.energy_only = energy_only;    // no backward pass: forces / virial untouched
      if (uq) {    // the driver's UQ head is set up once (validate_uq + widths); grace::Error -> ValueError
        if (ent->d1 && !ent->d1->uq_enabled()) ent->d1->enable_uq();
        if (ent->d2 && !ent->d2->uq_enabled()) ent->d2->enable_uq();
        if (ent->d3 && !ent->d3->uq_enabled()) ent->d3->enable_uq();
        fl.uq = true;
        out.gamma = ug.get();
        out.atomic_sigma = us.get();
        out.gmm_cluster = uc.get();
      }
      const int nth = num_threads > 0 ? num_threads : affinity_threads();
      grace::NoExchange ex;
      if (ent->d1)
        ent->d1->compute(n, nullptr, n, n, src, out, fl, nth);
      else if (ent->d3)
        ent->d3->compute(n, nullptr, n, n, src, ex, out, fl, nth);
      else
        ent->d2->compute(n, nullptr, n, n, src, ex, out, fl, nth);
      energy = out.energy;
      std::copy(out.virial, out.virial + 6, v.get());
    }
    nb::dict r;
    r["energy"] = energy;
    if (!energy_only) {
      r["forces"] = owned(std::move(f), {(size_t) n, 3});
      r["virial"] = owned(std::move(v), {6});
    }
    if (energies) r["energies"] = owned(std::move(e), {(size_t) n});
    if (uq) {
      double ts = 0.0;
      for (int a = 0; a < n; a++) ts += us[a];
      r["total_sigma"] = ts;
      r["gamma"] = owned(std::move(ug), {(size_t) n});
      r["atomic_sigma"] = owned(std::move(us), {(size_t) n});
      r["gmm_cluster"] = owned(std::move(uc), {(size_t) n});
    }
    return r;
  }

  bool has_uq() const { return m1_ ? m1_->has_uq : m3_ ? m3_->has_uq : m2_->has_uq; }
  int spline() const { return fs_.spline_n; }
  // throws ValueError unless the model carries a usable uqv6 head
  void validate_uq() const
  {
    if (m1_) m1_->validate_uq();
    else if (m3_) {    // the 3L loader checks the UQ head itself; the driver checks it against its basis
      if (!m3_->has_uq) throw grace::Error("the model has no UQ head (kokkos.npz exported without UQ artifacts)");
    } else m2_->validate_uq();
  }

  size_t cached_drivers()
  {
    std::lock_guard<std::mutex> lk(map_mu_);
    return drivers_.size();
  }

  std::string arch_, path_;
  std::vector<std::string> names_;

 private:
  struct Entry {
    std::unique_ptr<grace::Grace1LFused> d1;
    std::unique_ptr<grace::Grace2LFused> d2;
    std::unique_ptr<grace::Grace3LFused> d3;
    std::mutex mu;
    uint64_t used = 0;    // entry() call stamp (LRU)
  };
  // drivers kept per species set: each holds radial tables (GRACE-3L: ~15 MiB per species pair) and per-atom buffers
  // of the largest structure it has seen; a running compute keeps an evicted one alive through its shared_ptr
  static constexpr size_t kKeepDrivers = 4;
  // called without the GIL
  std::shared_ptr<Entry> entry(const std::vector<std::string> &species)
  {
    std::string key;
    for (auto &s : species) key += s + ",";
    std::lock_guard<std::mutex> lk(map_mu_);
    auto it = drivers_.find(key);
    if (it != drivers_.end()) {
      it->second->used = ++stamp_;
      return it->second;
    }
    if (species.empty()) throw grace::Error("species must not be empty");
    std::vector<int> t2m(species.size() + 1, -1);
    for (size_t t = 0; t < species.size(); t++) {
      auto el = std::find(names_.begin(), names_.end(), species[t]);
      if (el == names_.end()) {
        std::string all;
        for (auto &x : names_) all += x + " ";
        throw grace::Error("element " + species[t] + " is not in the model (model elements: " + all + ")");
      }
      t2m[t + 1] = (int) (el - names_.begin());
    }
    auto e = std::make_shared<Entry>();
    if (arch_ == "1l") {
      e->d1 = std::make_unique<grace::Grace1LFused>(m1_, fs_, grace::Logger{});
      e->d1->setup_types((int) species.size(), t2m);
    } else if (arch_ == "3l") {
      e->d3 = std::make_unique<grace::Grace3LFused>(m3_, fs_, grace::Logger{});
      e->d3->setup_types((int) species.size(), t2m);
    } else {
      e->d2 = std::make_unique<grace::Grace2LFused>(m2_, fs_, grace::Logger{});
      e->d2->setup_types((int) species.size(), t2m);
    }
    e->used = ++stamp_;
    while (drivers_.size() >= kKeepDrivers) {
      auto old = std::min_element(drivers_.begin(), drivers_.end(),
                                  [](const auto &a, const auto &b) { return a.second->used < b.second->used; });
      drivers_.erase(old);
    }
    drivers_[key] = e;
    return e;
  }
  grace::FusedSettings fs_;
  std::shared_ptr<grace::GRACE1LModel> m1_;
  std::shared_ptr<grace::GRACE2LModel> m2_;
  std::shared_ptr<grace::GRACE3LModel> m3_;
  std::map<std::string, std::shared_ptr<Entry>> drivers_;
  uint64_t stamp_ = 0;
  std::mutex map_mu_;
};

NB_MODULE(GRACE_RUNTIME_MODULE, m)    // _core_x86_64_v3 / _core_x86_64_v4 / _core_native (python/CMakeLists.txt)
{
  nb::register_exception_translator([](const std::exception_ptr &p, void *) {
    try {
      std::rethrow_exception(p);
    } catch (const grace::Error &e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    }
  });
  m.def("neighbor_search", &neighbor_search, "x"_a, "cell"_a, "pbc"_a, "cutoff"_a, "num_threads"_a = 0);
  m.def("bond_vectors", &bond_vectors, "x"_a, "i"_a, "j"_a, "shift"_a, "num_threads"_a = 0);
  m.def("dispersion_compute", &dispersion_compute, "params"_a, "numbers"_a, "x"_a, "cell"_a, "pbc"_a,
        "energy_only"_a = false, "num_threads"_a = 0);
  m.def("d3bj_preset", &d3bj_preset, "name"_a);
  m.def("d3bj_presets", &d3bj_presets);
  m.attr("BOHR") = grace::kBohr;
  nb::class_<Model>(m, "Model")
      .def(nb::init<const std::string &, int>(), "path"_a, "spline"_a = 1024)
      .def_ro("arch", &Model::arch_)
      .def_ro("path", &Model::path_)
      .def_ro("elements", &Model::names_)
      .def_prop_ro("spline", &Model::spline)
      .def("cutoff_matrix", &Model::cutoff_matrix, "species"_a)
      .def_prop_ro("has_uq", &Model::has_uq)
      .def_prop_ro("cached_drivers", &Model::cached_drivers)
      .def("validate_uq", &Model::validate_uq)
      .def("compute", &Model::compute, "species"_a, "types"_a, "offset"_a, "j"_a, "rij"_a, "energies"_a = false,
           "num_threads"_a = 0, "uq"_a = false, "energy_only"_a = false);
}

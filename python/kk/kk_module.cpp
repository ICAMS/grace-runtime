// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// grace_runtime._kk_<arch>: Python binding of the GPU backend (kk_api.h). Host compiler only, no Kokkos headers.
#include "kk_api.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
namespace gk = grace_runtime_kk;

namespace {
nb::ndarray<nb::numpy, double> from_vec(std::vector<double> &&v, std::initializer_list<size_t> shape)
{
  auto *p = new std::vector<double>(std::move(v));
  nb::capsule own(p, [](void *q) noexcept { delete static_cast<std::vector<double> *>(q); });
  return nb::ndarray<nb::numpy, double>(p->data(), shape, own);
}

nb::dict compute(gk::System &s, nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> types,
                 nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> pos,
                 nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu> cell, std::array<bool, 3> pbc, bool energies,
                 bool uq, bool energy_only)
{
  const int n = (int) types.shape(0);
  if ((int) pos.shape(0) != n) throw nb::value_error("positions and types differ in length");
  gk::Result r;
  {
    nb::gil_scoped_release nogil;
    r = s.compute(n, types.data(), pos.data(), cell.data(), pbc.data(), energies, uq, energy_only);
  }
  nb::dict d;
  d["energy"] = r.energy;
  if (!energy_only) {
    d["virial"] = from_vec(std::vector<double>(r.virial, r.virial + 6), {6});
    d["forces"] = from_vec(std::move(r.forces), {(size_t) n, 3});
  }
  if (energies) d["energies"] = from_vec(std::move(r.energies), {(size_t) n});
  if (uq) {
    double ts = 0.0;
    for (double v : r.sigma) ts += v;
    d["total_sigma"] = ts;
    d["gamma"] = from_vec(std::move(r.gamma), {(size_t) n});
    d["atomic_sigma"] = from_vec(std::move(r.sigma), {(size_t) n});
    d["gmm_cluster"] = from_vec(std::move(r.cluster), {(size_t) n});
  }
  return d;
}

nb::dict compute_batch(gk::Batch &b, nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> natoms,
                       nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> types,
                       nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> pos,
                       nb::ndarray<const double, nb::shape<-1, 3, 3>, nb::c_contig, nb::device::cpu> cells,
                       nb::ndarray<const bool, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> pbc, bool energies, bool uq,
                       bool energy_only, int num_threads, int64_t max_nall)
{
  const size_t nsys = natoms.shape(0), n = types.shape(0);
  if (pos.shape(0) != n) throw nb::value_error("positions and types differ in length");
  if (cells.shape(0) != nsys || pbc.shape(0) != nsys) throw nb::value_error("natoms, cells and pbc differ in length");
  gk::BatchResult r;
  {
    nb::gil_scoped_release nogil;
    r = b.compute((int) nsys, natoms.data(), (int64_t) n, types.data(), pos.data(), cells.data(), pbc.data(),
                  energies, uq, energy_only, num_threads, max_nall);
  }
  nb::dict d;
  d["energy"] = from_vec(std::move(r.energy), {nsys});
  if (!energy_only) {
    d["virial"] = from_vec(std::move(r.virial), {nsys, 6});
    d["forces"] = from_vec(std::move(r.forces), {n, 3});
  }
  if (energies) d["energies"] = from_vec(std::move(r.energies), {n});
  if (uq) {
    d["gamma"] = from_vec(std::move(r.gamma), {n});
    d["atomic_sigma"] = from_vec(std::move(r.sigma), {n});
    d["gmm_cluster"] = from_vec(std::move(r.cluster), {n});
  }
  return d;
}
template<class A> void on_device(const A &a, const char *name)
{
  if (a.device_type() != nb::device::cuda::value || a.device_id() != gk::device())
    throw nb::value_error((std::string(name) + " must be on the module's CUDA device").c_str());
}

using DevD1 = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cuda>;
using DevD2 = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cuda>;

// device inputs / outputs (DLPack: torch / CuPy arrays of the module's GPU, no copies); natoms, cells, pbc on the host
void compute_batch_device(gk::Batch &b, nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> natoms,
                          nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cuda> types,
                          nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cuda> pos,
                          nb::ndarray<const double, nb::shape<-1, 3, 3>, nb::c_contig, nb::device::cpu> cells,
                          nb::ndarray<const bool, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> pbc, DevD1 energy,
                          std::optional<DevD2> forces, std::optional<DevD2> virial, std::optional<DevD1> energies,
                          std::optional<DevD1> gamma, std::optional<DevD1> sigma, std::optional<DevD1> cluster,
                          bool energy_only, int64_t max_nall)
{
  const size_t nsys = natoms.shape(0), n = types.shape(0);
  if (pos.shape(0) != n) throw nb::value_error("positions and types differ in length");
  if (cells.shape(0) != nsys || pbc.shape(0) != nsys) throw nb::value_error("natoms, cells and pbc differ in length");
  on_device(types, "types");
  on_device(pos, "positions");
  on_device(energy, "energy");
  if (energy.shape(0) != nsys) throw nb::value_error("energy must have one value per structure");
  gk::DeviceOut o;
  o.energy = energy.data();
  auto out2 = [&](std::optional<DevD2> &a, size_t rows, size_t cols, const char *name) -> double * {
    if (!a) return nullptr;
    on_device(*a, name);
    if (a->shape(0) != rows || a->shape(1) != cols) throw nb::value_error((std::string(name) + ": wrong shape").c_str());
    return a->data();
  };
  auto out1 = [&](std::optional<DevD1> &a, const char *name) -> double * {
    if (!a) return nullptr;
    on_device(*a, name);
    if (a->shape(0) != n) throw nb::value_error((std::string(name) + " must have one value per atom").c_str());
    return a->data();
  };
  o.forces = out2(forces, n, 3, "forces");
  o.virial = out2(virial, nsys, 6, "virial");
  o.energies = out1(energies, "energies");
  o.gamma = out1(gamma, "gamma");
  o.sigma = out1(sigma, "atomic_sigma");
  o.cluster = out1(cluster, "gmm_cluster");
  const bool want_uq = o.gamma || o.sigma || o.cluster;
  if (want_uq && !(o.gamma && o.sigma && o.cluster)) throw nb::value_error("uq needs gamma, atomic_sigma and gmm_cluster");
  nb::gil_scoped_release nogil;
  b.compute_device((int) nsys, natoms.data(), (int64_t) n, types.data(), pos.data(), cells.data(), pbc.data(),
                   o.energies != nullptr, want_uq, energy_only, max_nall, o);
}
}    // namespace

#ifndef GRACE_KK_MODULE
#define GRACE_KK_MODULE _kk_unknown
#endif

NB_MODULE(GRACE_KK_MODULE, m)
{
  nb::register_exception_translator([](const std::exception_ptr &p, void *) {
    try {
      std::rethrow_exception(p);
    } catch (const std::invalid_argument &e) {
      PyErr_SetString(PyExc_ValueError, e.what());
    }
  });
  m.def("initialize", &gk::initialize, "device"_a);
  m.def("finalize", &gk::finalize);
  m.def("initialized", &gk::initialized);
  m.def("device", &gk::device);
  m.def("arch", [] { return std::string(gk::arch()); });
  m.def("cublas", &gk::cublas);
  m.def("memory_info", &gk::memory_info, "(free, total) bytes of the GPU (cudaMemGetInfo)");
  m.def("_debug_fail_alloc", &gk::debug_fail_alloc, "label"_a, "nth"_a);
  m.def("_debug_track_alloc", &gk::debug_track_alloc, "on"_a);
  m.def("_debug_live_alloc", &gk::debug_live_alloc);
  m.def("_debug_peak_alloc", &gk::debug_peak_alloc, "high-water mark (bytes) of the device allocations made since "
        "_debug_track_alloc(True)");
  nb::class_<gk::Model>(m, "Model")
      .def(nb::init<const std::string &, int>(), "path"_a, "spline"_a = 1024)
      .def_prop_ro("layers", &gk::Model::layers)
      .def_prop_ro("has_uq", &gk::Model::has_uq)
      .def_prop_ro("element_names", &gk::Model::element_names)
      .def_prop_ro("engine_setups", &gk::Model::engine_setups)
      .def("validate_uq", &gk::Model::validate_uq);
  nb::class_<gk::System>(m, "System")
      .def(nb::init<const gk::Model &, const std::vector<std::string> &, double, int, int>(), "model"_a, "species"_a,
           "skin"_a = 0.5, "spline"_a = 1024, "chunksize"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("cutoff", &gk::System::cutoff)
      .def_prop_ro("rebuilds", &gk::System::rebuilds)
      .def_prop_ro("chunksize", &gk::System::chunksize)
      .def("release_buffers", &gk::System::release_device_buffers, nb::call_guard<nb::gil_scoped_release>(),
           "return the system's and its engine's per-atom device buffers now (they grow again on the next compute)")
      .def("compute", &compute, "types"_a, "positions"_a, "cell"_a, "pbc"_a, "energies"_a = false, "uq"_a = false,
           "energy_only"_a = false);
  nb::class_<gk::Batch>(m, "Batch")
      .def(nb::init<const gk::Model &, const std::vector<std::string> &, int, int>(), "model"_a, "species"_a,
           "spline"_a = 1024, "chunksize"_a = -1, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("cutoff", &gk::Batch::cutoff)
      .def_prop_ro("passes", &gk::Batch::passes)
      .def_prop_ro("last_host_seconds", &gk::Batch::host_seconds)
      .def_prop_rw("keep_buffers", &gk::Batch::keep_buffers, &gk::Batch::set_keep_buffers)
      .def("release_buffers", &gk::Batch::release_device_buffers, nb::call_guard<nb::gil_scoped_release>(),
           "return the batch's and its engine's per-atom device buffers now (they grow again on the next compute)")
      // device arrays without implicit conversion: a converted output would be a temporary copy (results lost), a
      // converted input a hidden host -> device copy per call
      .def("compute_device", &compute_batch_device, "natoms"_a, "types"_a.noconvert(), "positions"_a.noconvert(),
           "cells"_a, "pbc"_a, "energy"_a.noconvert(), "forces"_a.noconvert().none() = nb::none(),
           "virial"_a.noconvert().none() = nb::none(), "energies"_a.noconvert().none() = nb::none(),
           "gamma"_a.noconvert().none() = nb::none(), "atomic_sigma"_a.noconvert().none() = nb::none(),
           "gmm_cluster"_a.noconvert().none() = nb::none(), "energy_only"_a = false,
           "max_nall"_a = (int64_t) 1 << 21,
           "compute with device inputs and outputs: types int32 [N], positions float64 [N, 3] and the outputs "
           "(float64; energy [S], forces [N, 3], virial [S, 6], per-atom [N]) on the module's GPU (DLPack); natoms, "
           "cells, pbc on the host")
      .def_prop_ro("chunksize", &gk::Batch::chunksize)
      .def("compute", &compute_batch, "natoms"_a, "types"_a, "positions"_a, "cells"_a, "pbc"_a, "energies"_a = false,
           "uq"_a = false, "energy_only"_a = false, "num_threads"_a = 0,
           "max_nall"_a = 2097152);
}

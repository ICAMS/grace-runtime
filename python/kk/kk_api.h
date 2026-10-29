// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Plain C++ interface of the GPU backend (grace_runtime._kk_<arch>): implemented with Kokkos in kk_core.cpp (compiled
// through the Kokkos launch compiler), bound to Python in kk_module.cpp (host compiler, no Kokkos headers).
#pragma once

#include "grace/dispersion.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace grace_runtime_kk {

// Kokkos on one GPU per process; finalize() releases every live System / Batch first
void initialize(int device);
void finalize();
bool initialized();
int device();
const char *arch();        // GPU architecture this module was built for, e.g. "sm86", "gfx90a"
const char *platform();    // "cuda" (NVIDIA, Kokkos CUDA) or "rocm" (AMD, Kokkos HIP)
const char *blas();        // GEMM library of the engines: "cublas", "rocblas", "kkgemm" (hand-written Kokkos GEMMs), ""
bool cublas();             // cuBLAS in use (dense GEMMs); false: Kokkos GEMMs or hand-written kernels
std::pair<size_t, size_t> memory_info();    // (free, total) bytes of the device (cudaMemGetInfo / hipMemGetInfo)
void debug_track_alloc(bool on);                                  // diagnostics: live allocations by label
std::vector<std::pair<std::string, uint64_t>> debug_live_alloc();
uint64_t debug_peak_alloc();    // high-water mark of the bytes allocated since debug_track_alloc(true)
// tests: the nth device allocation from now on whose label contains `label` fails as out of memory (nth <= 0: off)
void debug_fail_alloc(const std::string &label, int nth);
// tests: the nth engine launch check from now on reports a kernel launch that ran out of device memory (0: off)
void debug_fail_launch(int nth);

struct Result {
  double energy = 0.0;
  double virial[6] = {0, 0, 0, 0, 0, 0};    // xx yy zz xy xz yz (LAMMPS order), sum over bonds
  std::vector<double> forces;               // [n][3], caller's atom order
  std::vector<double> energies;             // [n] when requested
  std::vector<double> gamma, sigma, cluster;    // [n] when uq
  // dispersion (System::set_dispersion; dispersion true): its share of energy, per-atom energies, virial and forces
  // (not with energy_only), all included in the totals above
  bool dispersion = false;
  double dispersion_energy = 0.0;
  double dispersion_virial[6] = {0, 0, 0, 0, 0, 0};
  std::vector<double> dispersion_forces;      // [n][3]
  std::vector<double> dispersion_energies;    // [n]
};

// a loaded GRACE-1L / 2L / 3L kokkos.npz or PACE npz (host); its Systems / Batches share one engine on the device (weights, spline
// tables), each with its own species set (type tables switched per call)
class Model {
 public:
  Model(const std::string &path, int spline, const std::string &asi = "");
  ~Model();
  int layers() const;    // 1, 2 or 3 (GRACE); 0: a PACE model
  bool has_uq() const;
  void validate_uq() const;    // throws std::invalid_argument unless the UQ head is usable
  std::vector<std::string> element_names() const;
  int64_t engine_setups() const;    // full GPU engine setups so far (one per spline / chunksize, any species)
  struct Impl;
  std::shared_ptr<Impl> impl;
};

// one structure on the device (compute calls are serialised per System): neighbour list (cutoff + skin, rebuilt when an atom moved more than skin / 2 or the
// atoms / cell / pbc change), ghost images, the fused engine for one species set
class System {
 public:
  // species: element names of the atom types 1..len(species); throws std::invalid_argument for an element the model
  // lacks or a model outside the fused kernels' limits
  // chunksize: atoms per engine pass (bounds the GPU memory of large structures; 0: all atoms in one pass; < 0: auto:
  // the engine's chunk from the free GPU memory, as LAMMPS chunksize auto - one pass while it fits; on a device
  // allocation failure half that chunk, halved down to 512, then a 'does not fit' error)
  System(const Model &model, const std::vector<std::string> &species, double skin, int spline, int chunksize);
  ~System();
  double cutoff() const;    // largest pair cutoff of the species set
  // types [n] (1..len(species)), pos [n][3], cell [3][3] (full rank, rows = vectors), pbc [3]
  // energy_only: no backward pass (Result::forces left empty, virial zero), as LAMMPS eflag_only
  Result compute(int n, const int32_t *types, const double *pos, const double *cell, const bool *pbc, bool energies,
                 bool uq, bool energy_only = false);
  // a dispersion correction added to every compute from now on (copied; nullptr: off); atomic_numbers: Z of the types
  // 1..len(species). Throws std::invalid_argument for invalid parameters or an element without reference data.
  void set_dispersion(const grace::DispersionParams *p, const std::vector<int> &atomic_numbers);
  int64_t rebuilds() const;    // neighbour list builds so far
  int chunksize() const;       // atoms per engine pass in the last compute (0: one pass)
  void release_device_buffers();    // the System's (list, images, outputs) and its engine's per-atom buffers back now
  struct Impl;
  std::unique_ptr<Impl> impl;
};

struct BatchResult {
  std::vector<double> energy;    // [nsys]
  std::vector<double> virial;    // [nsys][6] xx yy zz xy xz yz
  std::vector<double> forces;    // [N][3] input order (empty with energy_only)
  std::vector<double> energies;  // [N] when requested
  std::vector<double> gamma, sigma, cluster;    // [N] when uq
  // dispersion (Batch::set_dispersion; dispersion true): its energy per structure, and with dispersion_parts its share
  // of the virial, forces (not with energy_only) and per-atom energies, all included in the totals above
  bool dispersion = false;
  std::vector<double> dispersion_energy;      // [nsys]
  std::vector<double> dispersion_virial;      // [nsys][6]
  std::vector<double> dispersion_forces;      // [N][3]
  std::vector<double> dispersion_energies;    // [N]
};

// outputs of Batch::compute_device: device pointers of the module's GPU (caller's atom order), nullptr where not wanted
struct DeviceOut {
  double *energy = nullptr;      // [nsys]
  double *virial = nullptr;      // [nsys][6] xx yy zz xy xz yz (not with energy_only)
  double *forces = nullptr;      // [N][3] (not with energy_only)
  double *energies = nullptr;    // [N]
  double *gamma = nullptr, *sigma = nullptr, *cluster = nullptr;    // [N] (uq)
  // dispersion (Batch::set_dispersion), its share of the outputs above (which include it); overwritten (left
  // untouched without set_dispersion)
  double *dispersion_energy = nullptr;      // [nsys]
  double *dispersion_virial = nullptr;      // [nsys][6] (not with energy_only)
  double *dispersion_forces = nullptr;      // [N][3] (not with energy_only)
  double *dispersion_energies = nullptr;    // [N]
};

// many structures per engine pass (compute calls are serialised per Batch): neighbour lists built on the device, one
// CSR with each structure's ghost images, per-structure energy and virial summed on the device
// (grace/kk/csr_images.h BatchAccKK)
class Batch {
 public:
  // species, spline, chunksize as System; throws std::invalid_argument as System
  Batch(const Model &model, const std::vector<std::string> &species, int spline, int chunksize);
  ~Batch();
  double cutoff() const;    // largest pair cutoff of the species set
  // natoms [nsys]; types [N] (1..len(species)); pos [N][3]; cell [nsys][3][3] full rank (zero vectors padded by the
  // caller); pbc [nsys][3]; max_nall: atoms + ghost images per engine pass (device memory scales with it; whole
  // structures per pass); types and positions copied to the device, neighbour lists and images built there as in
  // compute_device. dispersion_parts: also the dispersion's own forces / virial / per-atom energies (with
  // set_dispersion)
  BatchResult compute(int nsys, const int32_t *natoms, int64_t ntot, const int32_t *types, const double *pos,
                      const double *cell, const bool *pbc, bool energies, bool uq, bool energy_only, int64_t max_nall,
                      bool dispersion_parts = false);
  // as System::set_dispersion
  void set_dispersion(const grace::DispersionParams *p, const std::vector<int> &atomic_numbers);
  // compute with device inputs and outputs (no host copies of atoms): types [N] (1..len(species)) and pos [N][3] are
  // device pointers of the module's GPU; natoms, cell (zero vectors padded), pbc on the host as in compute; neighbour
  // lists and images built on the device (grace/kk/neighbors_kk.h BatchNeighborListKK)
  void compute_device(int nsys, const int32_t *natoms, int64_t ntot, const int32_t *types, const double *pos,
                      const double *cell, const bool *pbc, bool energies, bool uq, bool energy_only, int64_t max_nall,
                      const DeviceOut &out);
  int passes() const;    // engine passes of the last compute
  double host_seconds() const;    // last compute: neighbour search + pass layouts on the device (s)
  // keep the device buffers between computes (MD loops; default false: compute returns them, a batch is a job)
  bool keep_buffers() const;
  void set_keep_buffers(bool on);
  void release_device_buffers();    // the batch's and its engine's per-atom buffers back now
  int chunksize() const;    // atoms per engine pass in the last compute (0: one pass)
  struct Impl;
  std::unique_ptr<Impl> impl;
};

}    // namespace grace_runtime_kk

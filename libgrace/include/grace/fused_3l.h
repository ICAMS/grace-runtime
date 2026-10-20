// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from LAMMPS src/ML-PACE/pair_grace_3l_fused (lammps-ace, feature/grace-runtime a73b8b17ee); contributing author:
// Yury Lysogorskiy (ICAMS).
//
// GRACE-3L fused CPU driver (GRACE-3L-OMAT-large architecture), fp32 blocks of up to FusedSettings::block same-type
// atoms per OpenMP thread, GEMMs through grace/cpu_gemm.h, readout in fp64. Per compute():
//   S1  layer 1 forward -> I1, rho1          S2  Exchange::forward(0): I1 to ghost rows
//   S3  layer 2 forward -> I2, rho2          S4  Exchange::forward(1): I2 to ghost rows
//   S5  layer 3 forward + readout + reverse: bond forces of layer 3, dE/dI2 of the neighbours
//   S6  Exchange::reverse(1): ghost rows of dE/dI2 summed into their owners
//   S7  layer 2 (restored or recomputed) reverse: bond forces, dE/dI1      S8  Exchange::reverse(0)
//   S9  layer 1 reverse: bond forces
// ComputeFlags::energy_only stops after S5's readout (no reverse, no exchanges after S4).
#pragma once

#include "grace/error.h"
#include "grace/model_3l.h"
#include "grace/system.h"

#include <memory>
#include <vector>

namespace grace {

class Grace3LFused {
 public:
  Grace3LFused(std::shared_ptr<const GRACE3LModel> m, FusedSettings s, Logger log);
  ~Grace3LFused();
  Grace3LFused(const Grace3LFused &) = delete;
  Grace3LFused &operator=(const Grace3LFused &) = delete;

  // atom types 1..ntypes; type2mu[t] = model element index for t = 1..ntypes, -1 unmapped (pair_style hybrid NULL: no
  // bonds to or from that type); throws Error for a model outside the GRACE-3L-OMAT-large architecture
  void setup_types(int ntypes, const std::vector<int> &type2mu);
  double cutoff(int t1, int t2) const;

  // UQ (uqv6 gamma_only): throws Error when the model has no UQ head; afterwards ComputeFlags::uq fills
  // Outputs::gamma / atomic_sigma / gmm_cluster
  void enable_uq();
  bool uq_enabled() const { return uq_ready; }

  // central atoms ilist[0..inum) (nullptr: 0 .. inum-1, indices < n_nodes), n_nodes = central + ghost atoms,
  // nmax_nodes >= n_nodes (row capacity of the per-atom buffers; LAMMPS passes atom->nmax), nthreads OpenMP threads
  template<class Src>
  void compute(int inum, const int *ilist, int n_nodes, int nmax_nodes, const Src &bonds, Exchange &ex, Outputs &out,
               const ComputeFlags &fl, int nthreads);

  // message layer k = 0 (I1) and 1 (I2) for the LAMMPS forward / reverse comm: rows of message_width(k) floats
  int message_width(int k) const;
  float *message(int k);
  float *message_grad(int k);
  double memory_bytes() const;
  const char *gemm_backend() const;

  struct Net;
  struct Glob;
  struct Work;

 private:
  Logger log;
  std::shared_ptr<const GRACE3LModel> model;
  FusedSettings set;
  int ntypes = 0;
  std::vector<int> type2mu;
  std::vector<double> cutsq_flat;
  std::vector<double *> cutsq_rows;
  std::unique_ptr<Net> net;
  std::unique_ptr<Glob> g;
  std::vector<std::unique_ptr<Work>> work;
  int nmax = 0, nthreads = 0;
  bool uq_ready = false, uq_on = false;

  void build_net();
  void grow(int nmax_new);
  void ensure_threads(int nth);
  template<class Src> void run_layer(int k, bool bwd, const Src &src, Outputs &out, const ComputeFlags &fl);
  void uq_layer(int k, Work &w, Outputs &out);
};

}    // namespace grace

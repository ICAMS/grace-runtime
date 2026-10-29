// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// CPU driver of a PACE model (grace/pace_model.h) on CsrBonds (Python) or LammpsBonds, with the interface of the GRACE drivers: per atom the
// A-basis from its neighbours, the C-tilde products, the Finnis-Sinclair embedding with the inner cutoff / core
// repulsion of ML-PACE's ACECTildeEvaluator (lammps-user-pace ML-PACE/ace-evaluator/ace_evaluator.cpp), then the
// forces from the adjoint of the products. All in fp64 (fp32 misses the gate).
#ifndef GRACE_PACE_CPU_H
#define GRACE_PACE_CPU_H

#include "grace/error.h"
#include "grace/fused_cpu_types.h"
#include "grace/pace_model.h"
#include "grace/pace_tables.h"
#include "grace/system.h"

#include <memory>
#include <vector>

namespace grace {

class PaceCpu {
 public:
  // spline: intervals of the cubic radial tables per element pair; 0: the radial functions evaluated directly
  PaceCpu(std::shared_ptr<const PaceModel> model, int spline = 1024, Logger log = Logger{});
  ~PaceCpu();
  // LAMMPS-style types 1 .. ntypes -> model elements (type2mu[t], -1: not in the model)
  void setup_types(int ntypes, const std::vector<int> &type2mu);
  double cutoff(int t1, int t2) const;
  // the central atoms ilist[0 .. inum) (nullptr: 0 .. inum-1) with their bonds (full list, r_ij = x_j - x_i; Src:
  // CsrBonds or LammpsBonds), neighbours j among n_nodes atoms (LAMMPS: local and ghost); forces into out.f
  // [n_nodes][3] (accumulated), the virial from the bonds (flags.virial), per-atom energies (flags.eflag_atom) and
  // virials (flags.vflag_atom: half of each bond's to each of its atoms); flags.energy_only: no forces / virials.
  // Atoms go in groups of 8 of one element, so the working memory per thread is bounded by the neighbours of 8 atoms,
  // apart from its force rows [nmax_nodes][3]. Not reentrant: the workspace is the object's.
  template<class Src>
  void compute(int inum, const int *ilist, int n_nodes, int nmax_nodes, const Src &bonds, Outputs &out,
               const ComputeFlags &flags, int nthreads);

 private:
  struct Work;
  std::shared_ptr<const PaceModel> m_;
  Logger log_;
  int ntypes_ = 0;
  std::vector<int> type2mu_;
  std::unique_ptr<PaceTables> lay_;
  std::vector<Work> works_;      // per thread, kept between computes
  fused::ThreadForces fthr_;    // per-thread force rows, all zero between computes
};

}    // namespace grace

#endif

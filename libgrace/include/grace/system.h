// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// What a driver call reads and writes: the bond sources, the per-layer message exchange and the outputs.
#pragma once

#include "grace/coupling.h"

#include <algorithm>
#include <cstdint>

namespace grace {

// LAMMPS: positions of owned and ghost atoms and the full neighbour list (neighbour indices may carry the
// special-bond bits, removed with neighmask). r_ij = x[j] - x[i] is formed inside the gatherer, as before.
struct LammpsBonds {
  const double *const *x;
  const int *type;
  const int *numneigh;
  int *const *firstneigh;
  int neighmask;
};

// Python: bonds of central atom i are [offset[i], offset[i+1]); j is always a central atom (periodic
// images are folded into r_ij = x_j - x_i + S cell). r_ij comes from the caller (rij, [nbonds][3]) or, with rij
// null, is formed here from the positions x [n][3] (the order of type) and the shift of each bond, a row of the table
// shifts (S cell, [nimages][3]) given by image [nbonds].
struct CsrBonds {
  const int *type;
  const int64_t *offset;
  const int *j;
  const double *rij;
  const double *x = nullptr;
  const int *image = nullptr;
  const double *shifts = nullptr;
  // r_ij of bond q of central atom i: a row of rij, or d filled as the bond_vectors of the Python module does
  // (x_j - x_i + shift, the same operations: the same bits)
  const double *bond(int i, int64_t q, double *d) const
  {
    if (rij) return rij + 3 * q;
    const double *xi = x + 3 * (int64_t) i, *xj = x + 3 * (int64_t) j[q], *s = shifts + 3 * (int64_t) image[q];
    for (int k = 0; k < 3; k++) d[k] = xj[k] - xi[k] + s[k];
    return d;
  }
};

// per type-pair cutoffs: a bond is kept when r^2 < cutsq[ti][tj]; rcut_type[ti * ntp + tj] is its cutoff
struct BondFilter {
  const double *const *cutsq;
  const double *rcut_type;
  int ntp;
  const Coupling *cpl = nullptr;    // two-group coupling (grace/coupling.h), nullptr: off
};

// message exchange of one message-passing layer: forward copies owner rows to ghost rows of the layer's
// message array, reverse sums ghost rows of the gradient array into their owners. LAMMPS: comm; Python:
// nothing to do (no ghosts).
struct Exchange {
  virtual ~Exchange() = default;
  virtual void forward(int layer) = 0;
  virtual void reverse(int layer) = 0;
};
struct NoExchange final : Exchange {
  void forward(int) override {}
  void reverse(int) override {}
};

struct Outputs {
  double energy = 0.0;
  double virial[6] = {0, 0, 0, 0, 0, 0};    // xx yy zz xy xz yz; filled on the CsrBonds path (LAMMPS: fdotr)
  double *f = nullptr;                      // [n_nodes][3], accumulated (+=)
  double *eatom = nullptr;                  // [n] or nullptr, accumulated
  double **vatom = nullptr;                 // [n_nodes][6] or nullptr, accumulated
  // UQ (ComputeFlags::uq, after enable_uq()): per central atom, indexed by atom index, written (not accumulated)
  double *gamma = nullptr, *atomic_sigma = nullptr, *gmm_cluster = nullptr;
  double dedl = 0.0;                        // two-group coupling: dE/dlambda (with the backward pass), accumulated
};

struct FusedSettings {
  int spline_n = 1024;              // spline intervals of the radial MLPs
  int store_l1 = 1;                 // keep A, A1, AA1, AA2 per atom from P1 instead of recomputing layer 1 in P5
  double store_l1_max_gb = -1.0;    // store_l1 only up to this size (GB), else recompute; < 0 (auto):
                                    // fused::store_budget_gb (half the available host memory per process, within
                                    // the cgroup memory limit: Slurm --mem, containers)
  // GRACE-3L (Grace3LFused) only
  int block = 16;                   // atoms per block (1..64)
  double spline_mb = -1.0;          // radial tables above this size (MB): the radial MLP by GEMM instead; < 0 (auto):
                                    // a quarter of the host memory per process of the node, at least 2048 MB, at most
                                    // the available memory per process (both within the cgroup memory limit)
  int spline_n_layer[3] = {0, 0, 0};  // per-layer intervals; 0: spline_n_3l (spline_n / 2 for layers 1-2)
  int ranks_per_node = 1;           // auto spline_mb and store limits: processes of this host (LAMMPS: MPI ranks)
  bool store_fwd = true;            // keep the layer-1/2 forward tensors for the reverse pass (no recompute)
  double store_fwd_max_gb = -1.0;   // store_fwd only up to this size (GB), else recompute; < 0 (auto) as store_l1
  bool fc_fold = true;              // fold the FCs into the cp_l projections they feed
};

// GRACE-3L radial table intervals of layer k (0..2): an explicit per-layer value, else spline_n / 2 for layers 1-2
// (their radial functions are smoother: the same interpolation error as layer 3 at spline_n / 4) and spline_n for
// layer 3
inline int spline_n_3l(int spline_n, const int *per_layer, int k)
{
  if (per_layer && per_layer[k] > 0) return per_layer[k];
  return k < 2 ? std::max(16, spline_n / 2) : spline_n;
}
inline int spline_n_of(const FusedSettings &s, int k) { return spline_n_3l(s.spline_n, s.spline_n_layer, k); }

// the per-atom forward store of the reverse pass (2L store_l1, 3L store_fwd), decided when the per-atom arrays grow:
// kept when need_gb <= limit_gb, else the reverse pass recomputes (same results, slower); need_gb 0: not decided yet
// or the store is off
struct StoreInfo {
  bool kept = false;
  double need_gb = 0.0, limit_gb = 0.0;
};

struct ComputeFlags {
  bool eflag_atom = false;          // accumulate per-atom energies into Outputs::eatom
  bool vflag_atom = false;          // accumulate per-atom virials into Outputs::vatom
  bool virial = false;              // CsrBonds only: global virial from the bonds into Outputs::virial
  bool uq = false;                  // UQ (uqv6 gamma_only) into Outputs::gamma / atomic_sigma / gmm_cluster
  bool energy_only = false;         // energy, per-atom energy and UQ only: no backward pass, Outputs::f / vatom /
                                    // virial untouched (LAMMPS ENERGY_ONLY, e.g. Monte Carlo trial moves)
  Coupling couple;                  // two-group coupling (grace/coupling.h): couple.eq 0 off; mask over all atoms
};

}    // namespace grace

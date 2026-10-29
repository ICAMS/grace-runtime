// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Contributing author: Yury Lysogorskiy (ICAMS).
//
// GRACE-2L fused CPU driver, fp32, one batch of up to NB same-type atoms per OpenMP thread at a time with
// every per-atom tensor in a thread-private workspace, SIMD over channels. Per compute():
//   P1  layer 1 forward for every central atom -> I_i (message), rho_i
//   P2  Exchange::forward(0): I to ghost rows (LAMMPS) / nothing (CSR bonds: j is a central atom)
//   P3  layer 2 forward + energy + layer-2 reverse per central atom: bond forces of layer 2, dE/dI_j
//       added into its row of the shared dE/dI under the row's lock, dE/drho_i kept
//   P4  Exchange::reverse(0): ghost rows of dE/dI summed into their owners
//   P5  layer 1 recomputed (or restored) and reversed from (dE/dI_i, dE/drho_i): bond forces of layer 1
#pragma once

#include "grace/error.h"
#include "grace/fused_cpu_types.h"
#include "grace/model_2l.h"
#include "grace/system.h"

#include <memory>
#include <string>
#include <vector>

namespace grace {

class Grace2LFused {
 public:
  static constexpr int NB = fused::NB;
  using MVG = fused::MVG;
  using FC = fused::FC;
  using Prod = fused::Prod;
  using Red = fused::Red;

  Grace2LFused(std::shared_ptr<const GRACE2LModel> m, FusedSettings s, Logger log);
  ~Grace2LFused();

  // atom types 1..ntypes; type2mu[t] = model element index for t = 1..ntypes, -1 unmapped (pair_style hybrid NULL: no
  // bonds to or from that type)
  void setup_types(int ntypes, const std::vector<int> &type2mu);
  double cutoff(int t1, int t2) const { return rcut_type[t1 * (ntypes + 1) + t2]; }

  // UQ (uqv6 gamma_only, as Grace2LKK): validates the model's UQ head against the built reduces; throws Error when
  // the model has none or an unusable one. Afterwards compute() with ComputeFlags::uq fills Outputs::gamma etc.
  void enable_uq();
  bool uq_enabled() const { return uq_ready; }

  // central atoms ilist[0..inum) (indices < n_nodes), n_nodes = central + ghost atoms, nmax_nodes >= n_nodes
  // (row capacity of the per-atom buffers; LAMMPS passes atom->nmax), nthreads OpenMP threads
  template<class Src>
  void compute(int inum, const int *ilist, int n_nodes, int nmax_nodes, const Src &bonds, Exchange &ex,
               Outputs &out, const ComputeFlags &fl, int nthreads);

  // message rows of layer 0, for the LAMMPS forward / reverse comm: nI rows of PR floats per atom, CR used
  int message_rows() const { return nI; }
  int message_channels() const { return CR; }
  int message_stride() const { return PR; }
  float *message() { return I_glob.data(); }
  float *message_grad() { return gI_sum.data(); }
  double memory_bytes() const;
  const StoreInfo &store_info() const { return store; }

  struct Work;

 protected:
  Logger log;
  int ntypes = 0;
  std::vector<double> cutsq_flat;                   // [ntypes+1][ntypes+1]
  std::vector<double *> cutsq_rows;
  std::vector<std::unique_ptr<Work>> work;
  bool vflag_atom = false;          // per call (layer1_backward / layer2 read them)
  double **vatom = nullptr;

  std::shared_ptr<const GRACE2LModel> model;
  int spline_n = 1024;
  int store_l1 = 1;                 // keep A, A1, AA1, AA2 per atom from P1 instead of recomputing layer 1 in P5
  double store_l1_max_gb = -1.0;    // < 0: auto (fused::store_budget_gb)
  int ranks_per_node = 1;
  fused::SpreadBuf<float> l1_store;      // [nmax][nlm + nlm + nAA + nAA][P1]
  StoreInfo store;                       // store_l1: kept or recomputed (decided when nmax grows)

  // dimensions
  int lmax = 0, nlm = 0, nl = 0;
  int C1 = 0, P1 = 0;          // layer-1 channels (A, A1, AA..AAAA), padded to 16
  int CR = 0, PR = 0;          // layer-2 radial channels (B0, YI, I)
  int C2 = 0, P2 = 0;          // layer-2 FC channels (B..BBBB)
  int nI = 0;                  // I functions
  int nI1 = 0, PI1 = 0;        // I1 functions, I1 width (padded)
  int PRHO = 0, PO2 = 0;       // rho and I2 widths (padded)
  int nrho = 0, nI2 = 0;
  int nA1 = 0, nAA = 0, nAAA = 0, nAAAA = 0;
  int nB = 0, nBB = 0, nBBB = 0, nBBBB = 0, nYI = 0;

  // per-type data
  std::vector<int> type2mu;                         // [ntypes+1]
  fused::fvec zA, zB0;                       // [ntypes+1][P1 / PR], inv_avg_n_neigh folded in
  std::vector<double> rcut_type;                    // [ntypes+1][ntypes+1]
  std::vector<double> shift_type;                   // [ntypes+1]

  // radial splines: [N][4][nl][P] cubic coefficients in t (per interval)
  fused::fvec splR, splR1;

  // spherical harmonics coefficients
  std::vector<double> sh_a, sh_b;                   // indexed l*(l+1)/2+m
  std::vector<double> sh_c, sh_d;                   // indexed l

  FC fcA1, fcAA1, fcAA2, fcB1, fcBB1, fcBB2;
  Prod pAA, pAAA, pAAAA, pBB, pBBB, pBBBB;
  Red rI1, rRho, rI, rB, rI2;  // I1, rho (from A..AAAA); I from I1; B from YI, B0; I2 (B..BBBB)
  bool rowred = true;          // AAA, AAAA, BBB, BBBB never stored: products fused with the I1 / rho reductions row by row
  fused::RowRed rrI1_3, rrRho_3, rrI1_4, rrRho_4, rrI2_3, rrI2_4;    // (BBB, BBBB with I2 likewise)

  // YI: unique (y, I) pairs and the CG terms over them
  std::vector<int> yp_y, yp_i, yp_l;
  std::vector<int> yg_off, yg_l, yg_i;             // pairs grouped by (l, I)
  std::vector<int> yt_pair, yt_o, yt_off;           // terms sorted by output, CSR yt_off
  fused::fvec yt_c;
  float yi_inv_avg = 1.0f;

  // RMSNorms + energy MLP (double)
  std::vector<double> lnI_scale, ln0_scale;
  std::vector<std::vector<double>> eW;
  std::vector<double> e_norm;
  std::vector<int> e_nin, e_nout;
  int e_act = 1;
  double out_scale = 1.0;

  // per-atom globals
  int nmax = 0;
  fused::SpreadBuf<float> I_glob;        // [nmax][nI][PR]
  fused::SpreadBuf<float> gI_sum;        // [nmax][nI][PR]  dE/dI (reverse comm, layer-1 seeds)
  fused::RowLocks gI_lock;               // [nmax]  row locks of gI_sum: the layer-2 reverse adds into it
  fused::SpreadBuf<float> rho_glob;      // [nlocal][nrho]
  fused::SpreadBuf<double> grho_glob;    // [nlocal][nrho]
  int nthreads = 1;
  fused::ThreadForces fthr;   // per-thread force rows

  void build_model();
  void build_splines();
  void ensure_threads(int nth);

  std::vector<int> order;           // central atoms sorted by type
  std::vector<int> bstart, bcount;  // batches (type-homogeneous) into order

  void layer1_forward(Work &);
  void layer1_backward(Work &);
  void layer2(Work &);
  void uq_l1(Work &);
  void uq_l2(Work &);
  void uq_flush(Work &, int layer);
  // UQ: projection rows [I2 block (layer 2), rho block (layer 1)]; raw layer-1 projection and ||b_L1||^2 per atom
  bool uq_ready = false, uq_on = false;
  std::vector<float> uq_rp;    // fp32 projection matrix [d_basis][rp_dim] (enable_uq)
  int uq_row_l1 = 0;
  Outputs *uq_out = nullptr;
  bool eonly = false;                   // this compute(): ComputeFlags::energy_only (layer 2 stops after the energy)
  fused::SpreadBuf<double> uq_z, uq_n2l1;
  // the atom's energy from rho and I2; grho = dE/drho and gI2 = dE/dI2, both skipped when grho is nullptr (energy only:
  // pass both or neither)
  double energy_atom(const float *rho, const float *I2, int ty, double *grho, float *gI2) const;
  template<class Src> void bonds(Work &, const Src &, bool grad);
};

}    // namespace grace

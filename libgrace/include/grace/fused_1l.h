// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from LAMMPS src/ML-PACE/pair_grace_1l_fused (lammps-ace, feature/grace-fused); contributing author:
// Yury Lysogorskiy (ICAMS).
//
// GRACE-1L fused CPU driver, fp32, type-homogeneous batches of NB atoms per OpenMP thread, tensors
// [function][atom][channel] in a thread-private workspace, SIMD over channels; one pass (forward, energy,
// backward) per batch.
#pragma once

#include "grace/error.h"
#include "grace/fused_cpu_types.h"
#include "grace/model_1l.h"
#include "grace/system.h"

#include <memory>
#include <string>
#include <vector>

namespace grace {

class Grace1LFused {
 public:
  static constexpr int NB = fused::NB;
  using MVG = fused::MVG;
  using FC = fused::FC;
  using Prod = fused::Prod;
  using Red = fused::Red;

  Grace1LFused(std::shared_ptr<const GRACE1LModel> m, FusedSettings s, Logger log);
  ~Grace1LFused();

  void setup_types(int ntypes, const std::vector<int> &type2mu);
  double cutoff(int t1, int t2) const { return rcut_type[t1 * (ntypes + 1) + t2]; }

  // UQ (uqv6 gamma_only, as grace/1l/kk): validates the model's UQ head against the built reduces; throws Error when
  // the model has none or an unusable one. Afterwards compute() with ComputeFlags::uq fills Outputs::gamma etc.
  void enable_uq();
  bool uq_enabled() const { return uq_ready; }

  // as Grace2LFused::compute, without the message exchange
  template<class Src>
  void compute(int inum, const int *ilist, int n_nodes, int nmax_nodes, const Src &bonds, Outputs &out,
               const ComputeFlags &fl, int nthreads);

  struct Work;

 protected:
  Logger log;
  int ntypes = 0;
  std::vector<double> cutsq_flat;
  std::vector<double *> cutsq_rows;
  std::vector<std::unique_ptr<Work>> work;
  bool vflag_atom = false;
  double **vatom = nullptr;

  std::shared_ptr<const GRACE1LModel> model;
  int spline_n = 1024;

  int lmax = 0, nlm = 0, nl = 0;
  int CA = 0, PA = 0;          // A channels (nradmax), padded to 16
  int CF = 0, PF = 0;          // FC channels (A1 .. AAAA)
  int nrho = 0, PRHO = 0;
  int nAA = 0, nAAA = 0, nAAAA = 0;

  std::vector<int> type2mu;
  fused::fvec zA;                            // [ntypes+1][PA], 1/<N> folded in
  std::vector<double> rcut_type, shift_type;
  fused::fvec splR;                          // [N][4][nl][PA]
  std::vector<double> sh_a, sh_b, sh_c, sh_d;

  FC fcA1, fcAA1, fcAA2;
  Prod pAA, pAAA, pAAAA;
  Red rRho;
  bool rowred = true;          // AAA, AAAA never stored: products fused with the rho reduction row by row
  fused::RowRed rrRho_3, rrRho_4;

  std::vector<std::vector<double>> eW;
  std::vector<double> e_norm;
  std::vector<int> e_nin, e_nout;
  int e_act = 0;
  double out_scale = 1.0;

  std::vector<int> order, bstart, bcount;
  int nthreads = 1;
  fused::ThreadForces fthr;   // per-thread force rows
  bool uq_ready = false;      // enable_uq() passed
  std::vector<float> uq_rp;    // fp32 projection matrix [d_basis][rp_dim] (enable_uq)

  void build_model();
  void build_splines();
  void ensure_threads(int nth);

  template<class Src> void bonds(Work &, const Src &, bool grad);
  void forward(Work &);
  void backward(Work &);
  void uq_batch(Work &, Outputs &);
  void uq_flush(Work &, Outputs &);
  double energy_atom(const float *rho, int ty, float *grho) const;
};

}    // namespace grace

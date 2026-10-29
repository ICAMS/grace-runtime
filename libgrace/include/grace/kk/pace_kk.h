// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// PACE Kokkos engine (CUDA, HIP, host backends), fp64 throughout as the CPU engine (grace/pace_cpu.h): the same tables
// (grace/pace_tables.h) and functions (grace/pace_radial.h, grace/pace_embedding.h). Input as the GRACE Kokkos
// engines (local atoms and images, a 2D neighbour list, forces added to f). Per chunk of atoms ordered by element:
//   1. A-basis: one team per atom, threads over its neighbours: radial functions (cubic tables, directly below
//      kPaceExactBelow rcut, grace/pace_tables.h) and Y_lm (m >= 0) summed in team scratch; then A [A index][atom] (all m), the rank-1
//      densities, the core repulsion sum, the ZBL switch's closest neighbour;
//   2. products: threads over (slice of C-tilde rows, atom), atoms fastest, densities summed per atom;
//   3. embedding per atom (energies);
//   4. adjoint: threads over (slice, atom), dE/dA into W [A index][atom] (m >= 0);
//   5. forces: one team per atom, threads over its neighbours: radial and Y_lm derivatives again, contracted with W.
// Atomic sums (scratch A, densities and W across slices, forces) make the last bits vary between runs.
#pragma once

#include "grace/error.h"
#include "grace/hd.h"
#include "grace/kk/support.h"
#include "grace/kk/traits.h"
#include "grace/pace_embedding.h"
#include "grace/pace_model.h"
#include "grace/pace_radial.h"
#include "grace/pace_tables.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <type_traits>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace grace {
namespace kk {

// a pair on the device: the PaceBond fields the radial functions read, crad / hc and its table as device pointers
struct PaceBondKK {
  int defined = 0, radbase = 0, nradmax = 0, lmax = 0, nradbase = 0;
  double lambda = 0, rcut = 0, dcut = 0, r_in = 0, delta_in = 0, prehc = 0, lambdahc = 0;
  int zi = 0, zj = 0;
  const double *crad = nullptr;    // [nradmax][lmax + 1][nradbase]
  const double *hc = nullptr;      // [nd(mu_i)][nradbase]
  const double *spl = nullptr;     // [n0 + n1][nf][4] (pace::RadialSpline transposed); null: no table
  int n0 = 0, n1 = 0, nf = 0;      // intervals of the segments, functions of the block (nd + nradmax (lmax + 1) + 1)
  double r0 = 0, r1 = 0, inv_h0 = 0, inv_h1 = 0;
};

namespace pace_kk {

KOKKOS_INLINE_FUNCTION int lmh(int l, int m) { return l * (l + 1) / 2 + m; }

// associated Legendre parts P_lm(z), dP_lm/dz (m >= 0, l <= lmax <= LM) of ACECartesianSphericalHarmonics; alm,
// blm: [l ld + m] (ld: the row length of the tables, the model's lmax + 1). LM: compile-time bound (the loops unroll,
// P / dP stay in registers)
template<int LM, class AB>
KOKKOS_INLINE_FUNCTION void legendre(int lmax, int ld, double z, const AB &alm, const AB &blm, const AB &cl,
                                     const AB &dl, double *P, double *dP)
{
  P[0] = 1;
  dP[0] = 0;
  if (LM == 0 || lmax == 0) return;
  P[lmh(1, 0)] = sph_const::sq3 * z;
  dP[lmh(1, 0)] = sph_const::sq3;
  P[lmh(1, 1)] = -sph_const::sq3o2;
  dP[lmh(1, 1)] = 0;
#pragma unroll
  for (int l = 2; l <= LM; l++) {
    if (l > lmax) break;
#pragma unroll
    for (int m = 0; m < LM - 1; m++) {
      if (m >= l - 1) break;
      const double a = alm(l * ld + m), b = blm(l * ld + m);
      P[lmh(l, m)] = a * (z * P[lmh(l - 1, m)] + b * P[lmh(l - 2, m)]);
      dP[lmh(l, m)] = a * (P[lmh(l - 1, m)] + z * dP[lmh(l - 1, m)] + b * dP[lmh(l - 2, m)]);
    }
    const double t = dl(l) * P[lmh(l - 1, l - 1)];
    P[lmh(l, l - 1)] = t * z;
    dP[lmh(l, l - 1)] = t;
    P[lmh(l, l)] = cl(l) * P[lmh(l - 1, l - 1)];
    dP[lmh(l, l)] = 0;
  }
}

// Y_lm and its gradient on the unit sphere from p = P_lm, dp, the phases (x + i y)^m (pr, pi) and ^(m-1) (qr, qi; 0
// for m = 0), as Ylm::compute8 of pace_cpu.cpp
KOKKOS_INLINE_FUNCTION void ylm_grad(int m, double p, double dp, double pr, double pi, double qr, double qi, double x,
                                     double y, double z, double &yr, double &yi, double *gr, double *gi)
{
  yr = pr * p;
  yi = pi * p;
  const double xr = m * qr * p, xi = m * qi * p;    // d/dx
  const double y_r = -xi, y_i = xr;                     // d/dy
  const double zr = pr * dp, zi = pi * dp;              // d/dz
  const double rr = x * xr + y * y_r + z * zr, ri = x * xi + y * y_i + z * zi;
  gr[0] = xr - rr * x;
  gi[0] = xi - ri * x;
  gr[1] = y_r - rr * y;
  gi[1] = y_i - ri * y;
  gr[2] = zr - rr * z;
  gi[2] = zi - ri * z;
}

// function f of a pair's radial block evaluated directly (below the table or without one): kept out of line, so its
// buffers do not weigh on the registers and stack of the kernels that call it (it is rare: close pairs)
KOKKOS_FUNCTION GRACE_NOINLINE inline void radial_direct(const PaceBondKK &b, int nd, int ict, double r, int f,
                                                            double &v, double &d)
{
  const int qc = nd + b.nradmax * (b.lmax + 1);
  if (f == qc) {
    pace::core(b, ict, r, v, d);
    return;
  }
  double g[kPaceMaxNradbase], dg[kPaceMaxNradbase];
  pace::radbase(b, ict, r, g, dg);
  const double *w = f < nd ? b.hc + (size_t) f * b.nradbase : b.crad + (size_t) (f - nd) * b.nradbase;
  double s = 0, ds = 0;
  for (int k = 0; k < b.nradbase; k++) {
    s += w[k] * g[k];
    ds += w[k] * dg[k];
  }
  v = s;
  d = ds;
}

// the radial block of one pair at r: from its table or, below r0 or without one, directly
struct Radial {
  const PaceBondKK *b = nullptr;
  int nd = 0, qc = 0, ict = 0;
  bool tab = false;
  const double *c = nullptr;
  double x = 0, r = 0, inv_h = 0;
  KOKKOS_INLINE_FUNCTION Radial(const PaceBondKK &bond, int nd_, int ict_, double r_) : b(&bond), nd(nd_), ict(ict_), r(r_)
  {
    qc = nd + bond.nradmax * (bond.lmax + 1);
    tab = bond.spl && r >= bond.r0;
    if (tab) {    // as pace::RadialSpline::at
      const bool near = r < bond.r1;
      inv_h = near ? bond.inv_h0 : bond.inv_h1;
      const double t = (r - (near ? bond.r0 : bond.r1)) * inv_h;
      const int k = Kokkos::min((int) t, (near ? bond.n0 : bond.n1) - 1);
      x = t - k;
      c = bond.spl + (size_t) ((near ? 0 : bond.n0) + k) * 4 * bond.nf;    // [nf][4] of the interval
    }
  }
  // value and d/dr of function f of the block (h_p, R_nl at nd + n (lmax + 1) + l, the core repulsion at qc)
  KOKKOS_INLINE_FUNCTION void at(int f, double &v, double &d) const
  {
    if (!tab) {
      radial_direct(*b, nd, ict, r, f, v, d);
      return;
    }
    const double c0 = c[4 * f], c1 = c[4 * f + 1], c2 = c[4 * f + 2], c3 = c[4 * f + 3];
    v = c0 + x * (c1 + x * (c2 + x * c3));
    d = (c1 + x * (2 * c2 + 3 * x * c3)) * inv_h;
  }
  KOKKOS_INLINE_FUNCTION double val(int f) const
  {
    double v, d;
    at(f, v, d);
    return v;
  }
};

}    // namespace pace_kk

template<class DeviceType, class Traits = DefaultTraits<DeviceType>>
class PaceKK {
 public:
  using device_type = DeviceType;
  using model_type = PaceModel;
  template<class T> using V1 = Kokkos::View<T *, DeviceType>;
  template<class T> using V2 = Kokkos::View<T **, Kokkos::LayoutRight, DeviceType>;
  using team_policy = Kokkos::TeamPolicy<DeviceType>;
  using member_type = typename team_policy::member_type;
  // the neighbour kernels (A-basis, forces): teams of kNbTeam threads per atom (forces on CUDA: team size AUTO). HIP
  // launch bounds of kNbTeam threads, M = 1 (up to 512 registers per lane, hip_lb_t in support.h): under the default
  // 1024-thread bound forces spilled 5 KB per lane on gfx90a. One such team always fits, so no unbounded fallback
  static constexpr int kNbTeam = 64;       // one wavefront on CDNA (wave64)
  static constexpr int kBatch = kNbTeam;   // A-basis: neighbours per batch, one per thread
  using nb_policy = Kokkos::TeamPolicy<DeviceType, hip_lb_t<DeviceType, kNbTeam, 1>>;

  using ScratchV = Kokkos::View<double *, typename DeviceType::execution_space::scratch_memory_space,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  // the RunInput / RunOutput of the GRACE engines (the LAMMPS glue and the GPU module fill them); nlocal, ntypes:
  // unused; uq, cvflag_atom and couple_eq: refused (run() throws)
  struct RunInput {
    typename Traits::t_kkfloat_1d_3_lr_randomread x;
    typename Traits::t_kkacc_1d_3 f;
    typename Traits::t_int_1d_randomread type, ilist, numneigh;
    typename Traits::t_neighbors_2d neighbors;
    int inum = 0, nlocal = 0, ntypes = 0;
    int eflag_global = 0, eflag_atom = 0, vflag_global = 0, vflag_atom = 0, cvflag_atom = 0;
    bool energy_only = false, uq = false;
    int couple_eq = 0, couple_bit1 = 0, couple_bit2 = 0;
    double couple_lambda = 0.0;
    typename Traits::t_int_1d_randomread mask;
    typename Traits::t_kkacc_1d eatom;
    typename Traits::t_kkacc_1d_6 vatom;    // vflag_atom: half of each bond's virial to each of its atoms
    typename Traits::t_kkacc_1d_9 cvatom;
  };
  struct RunOutput {
    double eng_vdwl = 0.0;
    double virial[6] = {0, 0, 0, 0, 0, 0};
    double dedl = 0.0;    // two-group coupling: not supported, always 0
  };
  struct TypeTables {
    std::vector<int> type2model;
    V1<int> map;
  };

  grace::Logger log;
  int spline_n = 1024;      // cubic-Hermite intervals of the radial tables, 0: none
  int chunksize = -1;       // atoms per chunk: > 0 fixed, 0 one chunk, -1 auto (from the free GPU memory)
  int ranks_per_device = 1;
  bool debug_no_energy_only_calc = false;    // energy-only requests still run the forces (debug)

  void set_model(std::shared_ptr<const PaceModel> model, const std::vector<int> &type2mu)
  {
    m_ = std::move(model);
    h_type2model = type2mu;
    setup_done = false;
  }

  // the device tables of the model (PaceTables); again only after set_model() or a new spline_n (LAMMPS calls it at
  // every run)
  void setup()
  {
    if (!m_) throw Error("PACE/KK: setup() without a model");
    if (setup_done && setup_spline == spline_n) return;
    const PaceModel &m = *m_;
    const PaceTables t(m, spline_n);
    nel = t.nel;
    nr = t.nr;
    lmax = t.lmax;
    nlm = t.nlm;
    na = (int) t.na;
    ict = m.inner_cutoff_type;
    ndmax = 1;
    maxrows = 0;
    for (const auto &e : m.elements) ndmax = std::max(ndmax, e.ndensity);
    cutmax = 0.0;
    for (const PaceBond &b : m.bonds)
      if (b.defined) cutmax = std::max(cutmax, b.rcut);
    // products: the rows of all elements, their factors, coefficients [row][ndmax]
    std::vector<int> erow0(nel + 1, 0), off(1, 0), fidx, ftgt;
    std::vector<double> fre, fim, c;
    // extrapolation grade: rows -> B functions (global row index), the rank-1 rows per element, the active sets
    std::vector<int> uqoff{0}, uqf, r1e0{0}, r1mu, r1n, r1off{0}, r1f, nfun(nel, 0), asioff(nel + 1, 0);
    std::vector<double> uqcg, r1cg, asi;
    has_uq_ = m.has_uq;
    grade_ = m.uq_grade;
    nfmax_ = 1;
    nqmax_ = 1;
    for (int mu = 0; mu < nel; mu++) {
      const auto &E = t.el[mu];
      const int nrows = (int) E.off.size() - 1;
      maxrows = std::max(maxrows, nrows);
      for (int k = 0; k < nrows; k++) {
        for (int q = E.off[k]; q < E.off[k + 1]; q++) {
          fidx.push_back(E.fidx[q]);
          ftgt.push_back(E.ftgt[q]);
          fre.push_back(E.fre[q]);
          fim.push_back(E.fim[q]);
        }
        off.push_back((int) fidx.size());
        for (int p = 0; p < ndmax; p++) c.push_back(p < E.nd ? E.c[(size_t) k * E.nd + p] : 0.0);
      }
      erow0[mu + 1] = erow0[mu] + nrows;
      if (has_uq_) {
        for (int k = 0; k < nrows; k++) {
          for (int q = E.uq_off[k]; q < E.uq_off[k + 1]; q++) {
            uqf.push_back(E.uq_func[q]);
            uqcg.push_back(E.uq_cg[q]);
          }
          uqoff.push_back((int) uqf.size());
        }
        for (size_t q = 0; q < E.r1_mu.size(); q++) {
          r1mu.push_back(E.r1_mu[q]);
          r1n.push_back(E.r1_n[q]);
          for (int t = E.r1_off[q]; t < E.r1_off[q + 1]; t++) {
            r1f.push_back(E.r1_func[t]);
            r1cg.push_back(E.r1_cg[t]);
          }
          r1off.push_back((int) r1f.size());
        }
        const auto &pe = m.elements[mu];
        nfun[mu] = pe.uq_nfunc;
        nfmax_ = std::max(nfmax_, pe.uq_nfunc);
        nqmax_ = std::max(nqmax_, pe.uq_nfunc * (grade_ ? pe.ndensity : 1));
        asi.insert(asi.end(), pe.asi.begin(), pe.asi.end());
      }
      r1e0.push_back((int) r1mu.size());
      asioff[mu + 1] = (int) asi.size();
    }
    d_erow0 = up("pace_kk:erow0", erow0);
    if (has_uq_) {
      d_uqoff = up("pace_kk:uqoff", uqoff);
      d_uqf = up("pace_kk:uqf", uqf);
      d_uqcg = up("pace_kk:uqcg", uqcg);
      d_r1e0 = up("pace_kk:r1e0", r1e0);
      d_r1mu = up("pace_kk:r1mu", r1mu);
      d_r1n = up("pace_kk:r1n", r1n);
      d_r1off = up("pace_kk:r1off", r1off);
      d_r1f = up("pace_kk:r1f", r1f);
      d_r1cg = up("pace_kk:r1cg", r1cg);
      d_nfun = up("pace_kk:nfun", nfun);
      d_asioff = up("pace_kk:asioff", asioff);
      d_asi = up("pace_kk:asi", asi);
    }
    d_off = up("pace_kk:off", off);
    d_fidx = up("pace_kk:fidx", fidx);
    d_ftgt = up("pace_kk:ftgt", ftgt);
    d_fre = up("pace_kk:fre", fre);
    d_fim = up("pace_kk:fim", fim);
    d_c = up("pace_kk:c", c);
    // elements
    std::vector<int> nd(nel), npoti(nel);
    std::vector<double> fs((size_t) nel * 2 * ndmax, 0.0), rcc(nel), drcc(nel);
    for (int mu = 0; mu < nel; mu++) {
      const auto &e = m.elements[mu];
      nd[mu] = e.ndensity;
      npoti[mu] = e.npoti;
      std::copy(e.fs_parameters.begin(), e.fs_parameters.end(), fs.begin() + (size_t) mu * 2 * ndmax);
      rcc[mu] = e.rho_core_cut;
      drcc[mu] = e.drho_core_cut;
    }
    d_nd = up("pace_kk:nd", nd);
    d_npoti = up("pace_kk:npoti", npoti);
    d_fs = up("pace_kk:fs", fs);
    d_rcc = up("pace_kk:rcc", rcc);
    d_drcc = up("pace_kk:drcc", drcc);
    // pairs: crad, hc and the tables concatenated, the PaceBondKK with pointers into them
    std::vector<double> crad, hc, spl;
    std::vector<size_t> ocrad(nel * nel), ohc(nel * nel), ospl(nel * nel);
    for (int ij = 0; ij < nel * nel; ij++) {
      const PaceBond &b = m.bonds[ij];
      ocrad[ij] = crad.size();
      ohc[ij] = hc.size();
      ospl[ij] = spl.size();
      if (!b.defined) continue;
      crad.insert(crad.end(), b.crad.begin(), b.crad.end());
      hc.insert(hc.end(), t.hc[ij].begin(), t.hc[ij].end());
      if (!t.spl.empty()) {    // [interval][4][nf] -> [interval][nf][4]: one function's cubic in one 32-byte sector
        const auto &sp = t.spl[ij];
        const size_t ni = sp.c.size() / (4 * sp.nf);
        for (size_t k = 0; k < ni; k++)
          for (int f = 0; f < sp.nf; f++)
            for (int e = 0; e < 4; e++) spl.push_back(sp.c[(k * 4 + e) * sp.nf + f]);
      }
    }
    d_crad = up("pace_kk:crad", crad);
    d_hc = up("pace_kk:hc", hc);
    d_spl = up("pace_kk:spl", spl);
    d_bonds = V1<PaceBondKK>("pace_kk:bonds", nel * nel);
    auto hb = Kokkos::create_mirror_view(d_bonds);
    for (int ij = 0; ij < nel * nel; ij++) {
      const PaceBond &b = m.bonds[ij];
      PaceBondKK &o = hb(ij);
      o = PaceBondKK();
      if (!b.defined) continue;
      o.defined = 1;
      o.radbase = b.radbase;
      o.nradmax = b.nradmax;
      o.lmax = b.lmax;
      o.nradbase = b.nradbase;
      o.lambda = b.lambda;
      o.rcut = b.rcut;
      o.dcut = b.dcut;
      o.r_in = b.r_in;
      o.delta_in = b.delta_in;
      o.prehc = b.prehc;
      o.lambdahc = b.lambdahc;
      o.zi = b.zi;
      o.zj = b.zj;
      o.crad = d_crad.data() + ocrad[ij];
      o.hc = d_hc.data() + ohc[ij];
      if (!t.spl.empty()) {
        const auto &s = t.spl[ij];
        o.spl = d_spl.data() + ospl[ij];
        o.n0 = s.n0;
        o.n1 = s.n1;
        o.nf = s.nf;
        o.r0 = s.r0;
        o.r1 = s.r1;
        o.inv_h0 = s.inv_h0;
        o.inv_h1 = s.inv_h1;
      }
    }
    Kokkos::deep_copy(d_bonds, hb);
    d_alm = up("pace_kk:alm", t.alm);
    d_blm = up("pace_kk:blm", t.blm);
    d_cl = up("pace_kk:cl", t.cl);
    d_dl = up("pace_kk:dl", t.dl);
    d_sums = decltype(d_sums)("pace_kk:sums");
    h_sums = decltype(h_sums)("pace_kk:sums_host");
    set_types(h_type2model);
    setup_done = true;
    setup_spline = spline_n;
    check_kernel_launches<DeviceType>("PACE/KK setup");
  }

  void set_types(const std::vector<int> &type2mu)
  {
    for (size_t t = 1; t < type2mu.size(); t++)
      if (type2mu[t] < -1 || type2mu[t] >= nel)
        throw Error("PACE/KK: atom type " + std::to_string(t) + " maps to model element " + std::to_string(type2mu[t]));
    h_type2model = type2mu;
    d_map = up("pace_kk:map", type2mu);
  }
  TypeTables type_tables() const { return {h_type2model, d_map}; }
  void use_type_tables(const TypeTables &t)
  {
    h_type2model = t.type2model;
    d_map = t.map;
  }
  double cutoff(int mu_i, int mu_j) const { return mu_i < 0 || mu_j < 0 ? 0.0 : m_->cutoff(mu_i, mu_j); }
  // the extrapolation grade: the model's active set (.asi, or in the npz); uq_gamma: per atom index of the last run
  bool uq_available() const { return m_ && m_->has_uq; }    // after set_model (LAMMPS asks before setup)
  const V1<double> &uq_gamma() const { return d_gamma; }
  const V1<double> &uq_sigma() const { return d_none; }
  const V1<double> &uq_cluster() const { return d_none; }
  double memory_usage() const
  {
    double b = 0;
    for (const auto *v : {&d_Are, &d_Aim, &d_Wre, &d_Wim, &d_rho, &d_dF, &d_vir}) b += MemKK::memory_usage(*v);
    for (const auto *v : {&d_core, &d_dmin, &d_din, &d_dFc, &d_dFcut, &d_ea, &d_spl, &d_c, &d_fre, &d_fim})
      b += MemKK::memory_usage(*v);
    return b;
  }
  int auto_chunk_size() const { return chunk_auto; }
  void release_buffers()
  {
    MemKK::drop(d_Are, d_Aim, d_Wre, d_Wim, d_rho, d_dF, d_vir, d_core, d_dmin, d_din, d_jj, d_dFc, d_dFcut, d_ea,
                d_mu, d_order, d_key);
    cap = 0;
    chunk_auto = 0;
  }

  RunOutput run(const RunInput &in)
  {
    if (!setup_done) throw Error("PACE/KK: run() before setup()");
    if (in.uq && !has_uq_)
      throw Error("PACE/KK: the model has no active set (.asi) for the extrapolation grade");
    if (in.couple_eq) throw Error("PACE/KK: two-group coupling (pair_modify decouple) is not supported");
    if (in.cvflag_atom) throw Error("PACE/KK: per-atom centroid virials are not supported");
    RunOutput out;
    const int inum = in.inum;
    if (inum <= 0) return out;
    const int nmapped = sort_atoms(in);    // atoms of types not in the model (-1, LAMMPS NULL) sorted last, skipped
    choose_chunk(nmapped);
    const int chunk = chunksize > 0 ? std::min(chunksize, std::max(nmapped, 1)) : chunksize == 0 ? std::max(nmapped, 1)
                                                                                                 : std::max(1, chunk_auto);
    grow(chunk);
    const bool forces_on = !in.energy_only || debug_no_energy_only_calc;
    uq_run_ = in.uq;
    if (uq_run_) {    // gamma of every atom index the list can hold (unmapped and absent atoms: 0)
      const int ng = std::max(in.nlocal, inum);
      if ((int) d_gamma.extent(0) < ng) MemKK::realloc_kokkos(d_gamma, "pace_kk:gamma", ng);
      Kokkos::deep_copy(d_gamma, 0.0);
    }
    Kokkos::deep_copy(d_sums, 0.0);
    for (int c0 = 0; c0 < nmapped; c0 += chunk) {    // no host synchronisation until the sums are read
      const int nc = std::min(chunk, nmapped - c0);
      basis(in, c0, nc);
      if (uq_run_) uq_rank1(in, c0, nc);
      densities_embedding(in, c0, nc);
      if (uq_run_) uq_grade(c0, nc);
      if (forces_on) {
        adjoint(nc);
        forces(in, c0, nc);
      }
      add_sums(nc, forces_on);
    }
    Kokkos::deep_copy(h_sums, d_sums);
    out.eng_vdwl = h_sums(0);
    if (in.vflag_global && forces_on)
      for (int k = 0; k < 6; k++) out.virial[k] = h_sums(1 + k);
    check_kernel_launches<DeviceType>("PACE/KK run");
    return out;
  }

  // ---- stages (public: device lambdas) ----

  // atoms ordered by element (stable: key = element * inum + position), the unmapped ones last; their number
  int sort_atoms(const RunInput &in)
  {
    const int inum = in.inum;
    if ((int) d_order.extent(0) < inum) {
      MemKK::realloc_kokkos(d_order, "pace_kk:order", inum);
      MemKK::realloc_kokkos(d_key, "pace_kk:key", inum);
    }
    auto il = in.ilist;
    auto ty = in.type;
    auto map = d_map;
    auto key = d_key;
    auto ord = d_order;
    const long long n = inum;
    const int nel_ = nel;
    int nmapped = 0;
    Kokkos::parallel_reduce(
        "pace_kk:key", Kokkos::RangePolicy<DeviceType>(0, inum),
        KOKKOS_LAMBDA(const int ii, int &c) {
          const int mu = map(ty(il(ii)));
          key(ii) = (long long) (mu < 0 ? nel_ : mu) * n + ii;
          c += mu >= 0;
        },
        nmapped);
    if (nel > 1 || nmapped < inum) Kokkos::sort(Kokkos::subview(key, std::make_pair(0, inum)));
    Kokkos::parallel_for("pace_kk:order", Kokkos::RangePolicy<DeviceType>(0, inum), KOKKOS_LAMBDA(const int ii) {
      ord(ii) = il((int) (key(ii) % n));
    });
    return nmapped;
  }

  // 1. the A-basis, rank-1 densities, core repulsion sums, ZBL closest neighbours of atoms c0 .. c0 + nc. Per team
  // (one atom) its neighbours in batches of kBatch: each thread evaluates one neighbour's R_nl and Y_lm (m >= 0) into
  // scratch, then each thread sums its own A entries over the batch (no atomics)
  static constexpr int kTile = 256;    // atoms per tile of the products / adjoint kernels
  // thread id of the products / adjoint kernels -> atom a of the chunk and its rows [rb, re) of slice s: tiles of kTile
  // atoms with their S slices next to each other (the tile's A / W stay cached); false: a padding thread
  template<class VI>
  KOKKOS_INLINE_FUNCTION static bool slice_rows(size_t id, int S, int nc, const VI &mu_, const VI &erow0, int &a, int &rb,
                                                int &re)
  {
    const int s = (int) (id / kTile % S);
    a = (int) (id / kTile / S * kTile + id % kTile);
    if (a >= nc) return false;
    const int mu = mu_(a), r0 = erow0(mu), nrw = erow0(mu + 1) - r0, per = (nrw + S - 1) / S;
    rb = r0 + s * per;
    re = Kokkos::min(r0 + nrw, rb + per);
    return true;
  }
  static size_t slice_threads(int S, int nc) { return (size_t) S * ((nc + kTile - 1) / kTile) * kTile; }
  void basis(const RunInput &in, int c0, int nc)
  {
    with_lmax([&](auto lm) { basis_t<decltype(lm)::value>(in, c0, nc); });
  }
  template<int LM> void basis_t(const RunInput &in, int c0, int nc)
  {
    const int lmh_n = (lmax + 1) * (lmax + 2) / 2, nrl = nr * (lmax + 1), nah = nel * nr * lmh_n;
    const int nsc = 2 * nah + kBatch + ndmax + 1 + kBatch * (nrl + 2 * lmh_n);    // sums, then the batch
    const size_t bytes = ScratchV::shmem_size(nsc);
    const int level = bytes <= 40000 ? 0 : 1;
    auto x = in.x;
    auto ty = in.type;
    auto nn = in.numneigh;
    auto nb = in.neighbors;
    auto ord = d_order;
    auto map = d_map;
    auto bonds = d_bonds;
    auto nd_ = d_nd;
    auto Are = d_Are, Aim = d_Aim, rho = d_rho;
    auto core = d_core, dmin = d_dmin, din = d_din;
    auto jjm = d_jj, mu_ = d_mu;
    auto alm = d_alm, blm = d_blm, cl = d_cl, dl = d_dl;
    const int nel_ = nel, nr_ = nr, nlm_ = nlm, na_ = na, ndm = ndmax, ict_ = ict, L_ = lmax + 1, ld = L_, B = kBatch;
    const double cutmax_ = cutmax;
    Kokkos::parallel_for(
        "pace_kk:basis", nb_policy(nc, is_gpu_v<DeviceType> ? kBatch : 1).set_scratch_size(level, Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const member_type &tm) {
          const int a = tm.league_rank(), i = ord(c0 + a), mui = map(ty(i)), nd = nd_(mui), jnum = nn(i);
          ScratchV s(tm.team_scratch(level), nsc);
          double *sre = s.data(), *sim = sre + nah;                 // A (m >= 0) [element][n][lm]
          double *bmu = sim + nah;                                  // [batch] element of the neighbour, -1: none
          double *srho = bmu + B, *score = srho + ndm;
          double *bR = score + 1;                                   // [batch][n L + l]
          double *bYr = bR + B * nrl, *bYi = bYr + B * lmh_n;       // [batch][lm]
          Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, 2 * nah), [&](const int q) { sre[q] = 0.0; });
          Kokkos::single(Kokkos::PerTeam(tm), [&]() {
            for (int p = 0; p <= ndm; p++) srho[p] = 0.0;    // and score
          });
          const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
          double bval = cutmax_;    // ZBL: this thread's closest neighbour
          int bloc = -1;
          for (int k0 = 0; k0 < jnum; k0 += B) {
            tm.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, B), [&](const int t) {
              const int jj = k0 + t;
              double *R = bR + t * nrl, *Yr = bYr + t * lmh_n, *Yi = bYi + t * lmh_n;
              bmu[t] = -1;
              if (jj >= jnum) return;
              const int j = nb(i, jj) & Traits::neighmask, muj = map(ty(j));
              if (muj < 0) return;
              const PaceBondKK &b = bonds(mui * nel_ + muj);
              const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi, r2 = dx * dx + dy * dy + dz * dz;
              if (!b.defined || r2 >= b.rcut * b.rcut || r2 == 0.0) return;    // r = 0: skipped, as on the CPU
              const double r = Kokkos::sqrt(r2), ux = dx / r, uy = dy / r, uz = dz / r;
              if (ict_ == kInnerZbl) {
                const double d = r - (b.r_in - b.delta_in);
                if (d < bval) {
                  bval = d;
                  bloc = jj;
                }
              }
              const pace_kk::Radial Rf(b, nd, ict_, r);
              for (int p = 0; p < nd; p++) Kokkos::atomic_add(&srho[p], Rf.val(p));
              Kokkos::atomic_add(score, Rf.val(Rf.qc));
              const int Lb = b.lmax + 1;
              for (int n = 0; n < nr_; n++)
                for (int l = 0; l < L_; l++) R[n * L_ + l] = n < b.nradmax && l < Lb ? Rf.val(nd + n * Lb + l) : 0.0;
              constexpr int NLH = (LM + 1) * (LM + 2) / 2;
              double P[NLH], dP[NLH];
              pace_kk::legendre<LM>(L_ - 1, ld, uz, alm, blm, cl, dl, P, dP);
              double pr = 1, pi = 0;
#pragma unroll
              for (int m = 0; m <= LM; m++) {
                if (m >= L_) break;
#pragma unroll
                for (int l = 0; l <= LM; l++) {
                  if (l < m) continue;
                  if (l >= L_) break;
                  Yr[pace_kk::lmh(l, m)] = P[pace_kk::lmh(l, m)] * pr;
                  Yi[pace_kk::lmh(l, m)] = P[pace_kk::lmh(l, m)] * pi;
                }
                const double t2 = pr * ux - pi * uy;
                pi = pr * uy + pi * ux;
                pr = t2;
              }
              bmu[t] = muj;
            });
            tm.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nah), [&](const int e) {
              const int lm = e % lmh_n, mun = e / lmh_n, n = mun % nr_, mu = mun / nr_;
              int l = 0;
              while ((l + 1) * (l + 2) / 2 <= lm) l++;
              double ar = 0, ai = 0;
              const int kn = Kokkos::min(B, jnum - k0);
              for (int t = 0; t < kn; t++) {
                if ((int) bmu[t] != mu) continue;
                const double Rv = bR[t * nrl + n * L_ + l];
                ar += Rv * bYr[t * lmh_n + lm];
                ai += Rv * bYi[t * lmh_n + lm];
              }
              sre[e] += ar;
              sim[e] += ai;
            });
          }
          using ML = Kokkos::MinFirstLoc<double, int, DeviceType>;    // ties: the first neighbour, as the CPU
          typename ML::value_type best;
          best.val = bval;
          best.loc = bloc;
          if (ict_ == kInnerZbl) tm.team_reduce(ML(best));
          tm.team_barrier();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, na_), [&](const int q) {
            const int lm = q % nlm_, mun = q / nlm_;
            const int l = (int) Kokkos::sqrt((double) lm + 0.5), m = lm - l * l - l;
            const int h = mun * lmh_n + pace_kk::lmh(l, m < 0 ? -m : m);
            const double sg = (m < 0 && (m % 2 != 0)) ? -1.0 : 1.0;
            const double vr = m < 0 ? sg * sre[h] : sre[h], vi = m < 0 ? -sg * sim[h] : sim[h];
            Are(q, a) = vr;
            Aim(q, a) = vi;
          });
          Kokkos::single(Kokkos::PerTeam(tm), [&]() {
            for (int p = 0; p < ndm; p++) rho(p, a) = p < nd ? srho[p] : 0.0;
            core(a) = *score;
            mu_(a) = mui;
            jjm(a) = -1;
            dmin(a) = cutmax_;
            din(a) = 0.0;
            if (ict_ == kInnerZbl && best.loc >= 0 && best.val < cutmax_) {
              const int j = nb(i, best.loc) & Traits::neighmask;
              jjm(a) = best.loc;
              dmin(a) = best.val;
              din(a) = bonds(mui * nel_ + map(ty(j))).delta_in;
            }
          });
        });
  }

  // the neighbour kernels with lmax as a compile-time bound LM >= lmax (Legendre recursion unrolled, P / dP in registers)
  template<class F> void with_lmax(F &&f) const
  {
    switch (lmax) {
      case 0: return f(std::integral_constant<int, 0>());
      case 1: return f(std::integral_constant<int, 1>());
      case 2: return f(std::integral_constant<int, 2>());
      case 3: return f(std::integral_constant<int, 3>());
      case 4: return f(std::integral_constant<int, 4>());
      case 5: return f(std::integral_constant<int, 5>());
      case 6: return f(std::integral_constant<int, 6>());
      default: return f(std::integral_constant<int, kPaceMaxLmax>());
    }
  }

  // densities per thread in registers: a compile-time bound ND >= ndmax (a run-time bound puts the arrays into local
  // memory / scratch)
  template<class F> void with_nd(F &&f) const
  {
    if (ndmax <= 1) f(std::integral_constant<int, 1>());
    else if (ndmax <= 2) f(std::integral_constant<int, 2>());
    else if (ndmax <= 4) f(std::integral_constant<int, 4>());
    else if (ndmax <= 8) f(std::integral_constant<int, 8>());
    else if (ndmax <= 16) f(std::integral_constant<int, 16>());
    else f(std::integral_constant<int, kPaceMaxDensity>());
  }

  // 2. + 3. products into the densities, the embedding; the energy of the chunk
  void densities_embedding(const RunInput &in, int c0, int nc)
  {
    with_nd([&](auto nd_c) { products<decltype(nd_c)::value>(nc); });
    embedding(in, c0, nc);
  }
  template<int ND> void products(int nc)
  {
    const int S = slices_products(nc);
    const bool uq = uq_run_;
    auto uv = d_uv;
    auto uqoff = d_uqoff, uqf = d_uqf;
    auto uqcg = d_uqcg;
    auto Are = d_Are, Aim = d_Aim, rho = d_rho;
    auto mu_ = d_mu, nd_ = d_nd;
    auto erow0 = d_erow0, off = d_off, fidx = d_fidx;
    auto c = d_c;
    const int ndm = ndmax;
    Kokkos::parallel_for("pace_kk:products", Kokkos::RangePolicy<DeviceType>(0, slice_threads(S, nc)),
                         KOKKOS_LAMBDA(const size_t id) {
                           int a, rb, re;
                           if (!slice_rows(id, S, nc, mu_, erow0, a, rb, re)) return;
                           const int mu = mu_(a), nd = nd_(mu);
                           if (rb >= re) return;
                           double acc[ND];
#pragma unroll
                           for (int p = 0; p < ND; p++) acc[p] = 0.0;
                           for (int k = rb; k < re; k++) {
                             const int q0 = off(k), q1 = off(k + 1);
                             double br = Are(fidx(q0), a), bi = Aim(fidx(q0), a);
                             for (int q = q0 + 1; q < q1; q++) {
                               const double ar = Are(fidx(q), a), ai = Aim(fidx(q), a), t = br * ar - bi * ai;
                               bi = br * ai + bi * ar;
                               br = t;
                             }
#pragma unroll
                             for (int p = 0; p < ND; p++)
                               if (p < nd) acc[p] += c((size_t) k * ndm + p) * br;
                             if (uq)    // projections: Re of the row's product times the CG of its B functions
                               for (int q = uqoff(k); q < uqoff(k + 1); q++) Kokkos::atomic_add(&uv(uqf(q), a), uqcg(q) * br);
                           }
#pragma unroll
                           for (int p = 0; p < ND; p++)
                             if (p < nd) Kokkos::atomic_add(&rho(p, a), acc[p]);
                         });
  }
  void embedding(const RunInput &in, int c0, int nc)
  {
    auto mu_ = d_mu, nd_ = d_nd;
    auto rho = d_rho;
    const int ndm = ndmax;
    auto ord = d_order;
    auto npoti = d_npoti;
    auto fs = d_fs;
    auto rcc = d_rcc, drcc = d_drcc;
    auto core = d_core, dmin = d_dmin, din = d_din;
    auto jjm = d_jj;
    auto dF = d_dF;
    auto dFc = d_dFc, dFcut = d_dFcut;
    auto eatom = in.eatom;
    const bool eatom_on = in.eflag_atom && eatom.extent(0) > 0;
    const bool zbl = ict == kInnerZbl;
    auto ea = d_ea;
    const bool uq = uq_run_;
    auto dFfs = d_dFfs;
    Kokkos::parallel_for(
        "pace_kk:embedding", Kokkos::RangePolicy<DeviceType>(0, nc),
        KOKKOS_LAMBDA(const int a) {
          const int mu = mu_(a), nd = nd_(mu);
          double r[kPaceMaxDensity], d[kPaceMaxDensity], dfs[kPaceMaxDensity];
          for (int p = 0; p < nd; p++) r[p] = rho(p, a);
          // dfs always: `uq ? dfs : nullptr` left it unwritten on HIP (ROCm 6.3, LAMMPS build): the non-linear grade 0
          const pace::Embedded em = pace::embed(npoti(mu), &fs((size_t) mu * 2 * ndm), nd, r, d, core(a), rcc(mu),
                                                drcc(mu), zbl, jjm(a) >= 0, dmin(a), din(a), dfs);
          for (int p = 0; p < nd; p++) dF(p, a) = d[p];
          if (uq)    // the non-linear grade takes dF/drho of the embedding alone
            for (int p = 0; p < nd; p++) dFfs(p, a) = dfs[p];
          dFc(a) = em.dF_drho_core;
          dFcut(a) = em.dF_dfcut;
          if (eatom_on) eatom(ord(c0 + a)) += em.energy;
          ea(a) = em.energy;
        });
  }

  // extrapolation grade, the rank-1 rows: sum_j g_n(r_ij) over the neighbours of the row's element, into the
  // projections of its B functions; zeroes the chunk's projections first (the products add theirs after)
  void uq_rank1(const RunInput &in, int c0, int nc)
  {
    Kokkos::deep_copy(Kokkos::subview(d_uv, Kokkos::ALL, std::make_pair(0, nc)), 0.0);
    auto x = in.x;
    auto ty = in.type;
    auto nn = in.numneigh;
    auto nb = in.neighbors;
    auto ord = d_order;
    auto map = d_map;
    auto bonds = d_bonds;
    auto uv = d_uv;
    auto r1e0 = d_r1e0, r1mu = d_r1mu, r1n = d_r1n, r1off = d_r1off, r1f = d_r1f;
    auto r1cg = d_r1cg;
    const int nel_ = nel, ict_ = ict;
    Kokkos::parallel_for(
        "pace_kk:uq_rank1", team_policy(nc, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type &tm) {
          const int a = tm.league_rank(), i = ord(c0 + a), mui = map(ty(i));
          const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nn(i)), [&](const int jj) {
            const int j = nb(i, jj) & Traits::neighmask, muj = map(ty(j));
            if (muj < 0) return;
            const PaceBondKK &b = bonds(mui * nel_ + muj);
            const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi, r2 = dx * dx + dy * dy + dz * dz;
            if (!b.defined || r2 >= b.rcut * b.rcut || r2 == 0.0) return;
            double g[kPaceMaxNradbase], dg[kPaceMaxNradbase];
            pace::radbase(b, ict_, Kokkos::sqrt(r2), g, dg);
            for (int q = r1e0(mui); q < r1e0(mui + 1); q++)
              if (r1mu(q) == muj)
                for (int t = r1off(q); t < r1off(q + 1); t++) Kokkos::atomic_add(&uv(r1f(t), a), r1cg(t) * g[r1n(q)]);
          });
        });
  }

  // extrapolation grade gamma_i = max_k |sum_j u_j A[j][k]| (u: the projections, or dF/drho_p times them for the
  // non-linear grade, [p][f]); one team per atom, u in team scratch, the threads over k
  void uq_grade(int c0, int nc)
  {
    auto ord = d_order;
    auto mu_ = d_mu, nd_ = d_nd;
    auto uv = d_uv, dFfs = d_dFfs;
    auto nfun = d_nfun, asioff = d_asioff;
    auto asi = d_asi;
    auto gamma = d_gamma;
    const int grade = grade_;
    const size_t bytes = ScratchV::shmem_size(nqmax_);
    const int level = bytes <= 32768 ? 0 : 1;
    Kokkos::parallel_for(
        "pace_kk:uq_grade", team_policy(nc, Kokkos::AUTO).set_scratch_size(level, Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const member_type &tm) {
          const int a = tm.league_rank(), mu = mu_(a), nf = nfun(mu), n = nf * (grade ? nd_(mu) : 1);
          ScratchV u(tm.team_scratch(level), n);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int j) {
            const int p = j / nf, f = j % nf;
            u(j) = grade ? dFfs(p, a) * uv(f, a) : uv(f, a);
          });
          tm.team_barrier();
          const double *A = asi.data() + asioff(mu);
          double gmax = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(tm, n),
              [&](const int k, double &mx) {
                double acc = 0.0;
                for (int j = 0; j < n; j++) acc += u(j) * A[(size_t) j * n + k];
                mx = Kokkos::max(mx, Kokkos::abs(acc));
              },
              Kokkos::Max<double>(gmax));
          Kokkos::single(Kokkos::PerTeam(tm), [&]() { gamma(ord(c0 + a)) = gmax; });
        });
  }

  // the chunk's energy and virial (per atom in d_ea, d_vir) into d_sums, on the device: teams of 256 atoms, one atomic
  // add per team and value
  void add_sums(int nc, bool virial)
  {
    auto ea = d_ea;
    auto va = d_vir;
    auto sums = d_sums;
    const int nv = virial ? 7 : 1;
    Kokkos::parallel_for(
        "pace_kk:sums", team_policy((nc + 255) / 256, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type &tm) {
          const int a0 = tm.league_rank() * 256, a1 = Kokkos::min(nc, a0 + 256);
          for (int k = 0; k < nv; k++) {
            double acc = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(tm, a0, a1), [&](const int a, double &x) { x += k == 0 ? ea(a) : va(a, k - 1); },
                acc);
            Kokkos::single(Kokkos::PerTeam(tm), [&]() { Kokkos::atomic_add(&sums(k), acc); });
          }
        });
  }

  // one C-tilde row of rank R in the adjoint: its factors, the products before t (times theta) and after t in registers;
  // dE/dA_t = theta prod_{s != t} A_s into W (atomics when several threads share the atom)
  template<int R, class VA, class VI, class VD>
  KOKKOS_INLINE_FUNCTION static void adjoint_row(const VA &Are, const VA &Aim, const VA &Wre, const VA &Wim,
                                                 const VI &fidx, const VI &ftgt, const VD &fre, const VD &fim, int q0,
                                                 int a, double th, bool atomic)
  {
    double ar[R], ai[R], fr[R], fi[R];
#pragma unroll
    for (int t = 0; t < R; t++) {
      const int q = fidx(q0 + t);
      ar[t] = Are(q, a);
      ai[t] = Aim(q, a);
    }
    fr[0] = th;    // theta times the products of the factors before t
    fi[0] = 0.0;
#pragma unroll
    for (int t = 0; t < R - 1; t++) {
      fr[t + 1] = fr[t] * ar[t] - fi[t] * ai[t];
      fi[t + 1] = fr[t] * ai[t] + fi[t] * ar[t];
    }
    double br = 1.0, bi = 0.0;    // products of the factors after t
#pragma unroll
    for (int t = R - 1; t >= 0; t--) {
      const double gr = fr[t] * br - fi[t] * bi, gi = fr[t] * bi + fi[t] * br;
      const int qq = q0 + t, tg = ftgt(qq);
      if (atomic) {
        Kokkos::atomic_add(&Wre(tg, a), fre(qq) * gr);
        Kokkos::atomic_add(&Wim(tg, a), fim(qq) * gi);
      } else {    // one thread per atom: its own column
        Wre(tg, a) += fre(qq) * gr;
        Wim(tg, a) += fim(qq) * gi;
      }
      if (t > 0) {
        const double nb_ = br * ar[t] - bi * ai[t];
        bi = br * ai[t] + bi * ar[t];
        br = nb_;
      }
    }
  }

  static_assert(kPaceMaxRank == 6, "adjoint_t: one case per rank");
  // 4. dE/dA (m >= 0) into W
  void adjoint(int nc)
  {
    with_nd([&](auto nd_c) { adjoint_t<decltype(nd_c)::value>(nc); });
  }
  template<int ND> void adjoint_t(int nc)
  {
    const int S = slices_adjoint(nc);
    auto Are = d_Are, Aim = d_Aim, Wre = d_Wre, Wim = d_Wim, dF = d_dF;
    auto mu_ = d_mu, nd_ = d_nd;
    auto erow0 = d_erow0, off = d_off, fidx = d_fidx, ftgt = d_ftgt;
    auto fre = d_fre, fim = d_fim, c = d_c;
    const int ndm = ndmax;
    Kokkos::deep_copy(Kokkos::subview(Wre, Kokkos::ALL, std::make_pair(0, nc)), 0.0);
    Kokkos::deep_copy(Kokkos::subview(Wim, Kokkos::ALL, std::make_pair(0, nc)), 0.0);
    Kokkos::parallel_for("pace_kk:adjoint", Kokkos::RangePolicy<DeviceType>(0, slice_threads(S, nc)),
                         KOKKOS_LAMBDA(const size_t id) {
                           int a, rb, re;
                           if (!slice_rows(id, S, nc, mu_, erow0, a, rb, re)) return;
                           const int mu = mu_(a), nd = nd_(mu);
                           double df[ND];
#pragma unroll
                           for (int p = 0; p < ND; p++) df[p] = p < nd ? dF(p, a) : 0.0;
                           const bool atomic = S > 1;
                           for (int k = rb; k < re; k++) {
                             double th = 0.0;
#pragma unroll
                             for (int p = 0; p < ND; p++)
                               if (p < nd) th += c((size_t) k * ndm + p) * df[p];
                             th *= 0.5;
                             const int q0 = off(k);
                             switch (off(k + 1) - q0) {    // rows are sorted by rank: the same branch for the warp
                               case 1: adjoint_row<1>(Are, Aim, Wre, Wim, fidx, ftgt, fre, fim, q0, a, th, atomic); break;
                               case 2: adjoint_row<2>(Are, Aim, Wre, Wim, fidx, ftgt, fre, fim, q0, a, th, atomic); break;
                               case 3: adjoint_row<3>(Are, Aim, Wre, Wim, fidx, ftgt, fre, fim, q0, a, th, atomic); break;
                               case 4: adjoint_row<4>(Are, Aim, Wre, Wim, fidx, ftgt, fre, fim, q0, a, th, atomic); break;
                               case 5: adjoint_row<5>(Are, Aim, Wre, Wim, fidx, ftgt, fre, fim, q0, a, th, atomic); break;
                               default: adjoint_row<kPaceMaxRank>(Are, Aim, Wre, Wim, fidx, ftgt, fre, fim, q0, a, th, atomic);
                             }
                           }
                         });
  }

  // 5. forces on the bonds of atoms c0 .. c0 + nc, the virial per atom (d_vir)
  void forces(const RunInput &in, int c0, int nc)
  {
    with_lmax([&](auto lm) { forces_t<decltype(lm)::value>(in, c0, nc); });
  }
  template<int LM> void forces_t(const RunInput &in, int c0, int nc)
  {
    auto x = in.x;
    auto f = in.f;
    auto ty = in.type;
    auto nn = in.numneigh;
    auto nb = in.neighbors;
    auto ord = d_order;
    auto map = d_map;
    auto bonds = d_bonds;
    auto nd_ = d_nd;
    auto Wre = d_Wre, Wim = d_Wim, dF = d_dF;
    auto dFc = d_dFc, dFcut = d_dFcut;
    auto jjm = d_jj;
    auto alm = d_alm, blm = d_blm, cl = d_cl, dl = d_dl;
    const int nel_ = nel, nr_ = nr, nlm_ = nlm, ict_ = ict;
    const int ld = lmax + 1;
    const bool zbl = ict == kInnerZbl;
    auto va = d_vir;
    auto vatom = in.vatom;
    const bool do_va = in.vflag_atom && vatom.extent(0) > 0;
    const int nlmh = (lmax + 1) * (lmax + 2) / 2, nah = nel * nr * nlmh;    // the atom's dE/dA (m >= 0) in scratch
    const size_t bytes = ScratchV::shmem_size(2 * nah);
    const int level = bytes <= 32768 ? 0 : 1;
    Kokkos::parallel_for(
        "pace_kk:forces",
        (is_hip_v<DeviceType> ? nb_policy(nc, kNbTeam) : nb_policy(nc, Kokkos::AUTO))
            .set_scratch_size(level, Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const member_type &tm) {
          const int a = tm.league_rank(), i = ord(c0 + a), mui = map(ty(i)), nd = nd_(mui);
          const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2), dfc = dFc(a), dfcut = dFcut(a);
          const int jmin = jjm(a);
          ScratchV sw(tm.team_scratch(level), 2 * nah);
          double *swr = sw.data(), *swi = swr + nah;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nah), [&](const int h) {
            const int lm = h % nlmh, mun = h / nlmh;
            int l = 0;
            while ((l + 1) * (l + 2) / 2 <= lm) l++;
            const int q = mun * nlm_ + l * l + l + (lm - l * (l + 1) / 2);
            swr[h] = Wre(q, a);
            swi[h] = Wim(q, a);
          });
          tm.team_barrier();
          double fx = 0, fy = 0, fz = 0, v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(tm, nn(i)),
              [&](const int jj) {
                const int j = nb(i, jj) & Traits::neighmask, muj = map(ty(j));
                if (muj < 0) return;
                const PaceBondKK &b = bonds(mui * nel_ + muj);
                const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi, r2 = dx * dx + dy * dy + dz * dz;
                if (!b.defined || r2 >= b.rcut * b.rcut || r2 == 0.0) return;
                const double r = Kokkos::sqrt(r2), ux = dx / r, uy = dy / r, uz = dz / r;
                const pace_kk::Radial R(b, nd, ict_, r);
                double v, d;
                R.at(R.qc, v, d);
                double rad = d * dfc;    // dE/dr along rhat
                for (int p = 0; p < nd; p++) {
                  R.at(p, v, d);
                  rad += dF(p, a) * d;
                }
                if (zbl && jj == jmin) rad += dfcut;
                double radn = 0, ang[3] = {0, 0, 0};    // the A-basis part of rad; the tangential part times r
                constexpr int NLH = (LM + 1) * (LM + 2) / 2;
                double P[NLH], dP[NLH];
                pace_kk::legendre<LM>(b.lmax, ld, uz, alm, blm, cl, dl, P, dP);
                const int L = b.lmax + 1;
#pragma unroll
                for (int l = 0; l <= LM; l++) {
                  if (l >= L) break;
                  // s1 = sum_n W_nlm dR_nl, s2 = sum_n W_nlm R_nl, per m
                  double s1r[LM + 1], s1i[LM + 1], s2r[LM + 1], s2i[LM + 1];
#pragma unroll
                  for (int m = 0; m <= LM; m++) s1r[m] = s1i[m] = s2r[m] = s2i[m] = 0;
                  for (int n = 0; n < b.nradmax; n++) {
                    double Rv, Rd;
                    R.at(nd + n * L + l, Rv, Rd);
                    const int h0 = (muj * nr_ + n) * nlmh + l * (l + 1) / 2;
#pragma unroll
                    for (int m = 0; m <= LM; m++) {
                      if (m > l) break;
                      const double wr = swr[h0 + m], wi = swi[h0 + m];
                      s1r[m] += wr * Rd;
                      s1i[m] += wi * Rd;
                      s2r[m] += wr * Rv;
                      s2i[m] += wi * Rv;
                    }
                  }
                  double pr = 1, pi = 0, qr = 0, qi = 0;    // (x + i y)^m, ^(m-1)
#pragma unroll
                  for (int m = 0; m <= LM; m++) {
                    if (m > l) break;
                    double yr, yi_, gr[3], gi[3];
                    pace_kk::ylm_grad(m, P[pace_kk::lmh(l, m)], dP[pace_kk::lmh(l, m)], pr, pi, qr, qi, ux, uy, uz,
                                      yr, yi_, gr, gi);
                    const double cm = m > 0 ? 2.0 : 1.0;
                    radn += cm * (s1r[m] * yr - s1i[m] * yi_);
                    for (int k = 0; k < 3; k++) ang[k] += cm * (s2r[m] * gr[k] - s2i[m] * gi[k]);
                    qr = pr;
                    qi = pi;
                    const double t = pr * ux - pi * uy;
                    pi = pr * uy + pi * ux;
                    pr = t;
                  }
                }
                rad += radn;
                const double inv_r = 1.0 / r;
                const double fv[3] = {rad * ux + ang[0] * inv_r, rad * uy + ang[1] * inv_r, rad * uz + ang[2] * inv_r};
                for (int k = 0; k < 3; k++) Kokkos::atomic_add(&f(j, k), -fv[k]);
                fx += fv[0];
                fy += fv[1];
                fz += fv[2];
                // del (x) f, del = x_i - x_j, f = force on i (xx yy zz xy xz yz)
                const double w[6] = {-dx * fv[0], -dy * fv[1], -dz * fv[2], -dx * fv[1], -dx * fv[2], -dy * fv[2]};
                if (do_va)    // half to each atom of the bond (atom i's half once per team, below)
                  for (int k = 0; k < 6; k++) Kokkos::atomic_add(&vatom(j, k), 0.5 * w[k]);
                v0 += w[0];
                v1 += w[1];
                v2 += w[2];
                v3 += w[3];
                v4 += w[4];
                v5 += w[5];
              });
          using Sum = Kokkos::Sum<double>;
          tm.team_reduce(Sum(fx));
          tm.team_reduce(Sum(fy));
          tm.team_reduce(Sum(fz));
          tm.team_reduce(Sum(v0));
          tm.team_reduce(Sum(v1));
          tm.team_reduce(Sum(v2));
          tm.team_reduce(Sum(v3));
          tm.team_reduce(Sum(v4));
          tm.team_reduce(Sum(v5));
          Kokkos::single(Kokkos::PerTeam(tm), [&]() {
            Kokkos::atomic_add(&f(i, 0), fx);
            Kokkos::atomic_add(&f(i, 1), fy);
            Kokkos::atomic_add(&f(i, 2), fz);
            if (do_va) {
              const double vi[6] = {v0, v1, v2, v3, v4, v5};
              for (int k = 0; k < 6; k++) Kokkos::atomic_add(&vatom(i, k), 0.5 * vi[k]);
            }
            va(a, 0) = v0;
            va(a, 1) = v1;
            va(a, 2) = v2;
            va(a, 3) = v3;
            va(a, 4) = v4;
            va(a, 5) = v5;
          });
        });
  }

 private:
  std::shared_ptr<const PaceModel> m_;
  std::vector<int> h_type2model;
  bool setup_done = false;
  int setup_spline = -1;    // spline_n of the tables on the device
  int nel = 0, nr = 1, lmax = 0, nlm = 1, na = 0, ict = 0, ndmax = 1, maxrows = 0;
  double cutmax = 0.0;
  int cap = 0, chunk_auto = 0, auto_inum = 0;
  V1<int> d_erow0, d_off, d_fidx, d_ftgt, d_nd, d_npoti, d_map;
  V1<double> d_fre, d_fim, d_c, d_fs, d_rcc, d_drcc, d_crad, d_hc, d_spl, d_alm, d_blm, d_cl, d_dl;
  V1<PaceBondKK> d_bonds;
  V2<double> d_Are, d_Aim, d_Wre, d_Wim, d_rho, d_dF;    // [A index or density][atom of the chunk]
  V2<double> d_vir;                                      // [atom of the chunk][6]
  V1<double> d_core, d_dmin, d_din, d_dFc, d_dFcut, d_ea;
  Kokkos::View<double[7], DeviceType> d_sums;    // energy, virial [6] of a run
  Kokkos::View<double[7], Kokkos::HostSpace> h_sums;
  V1<int> d_jj, d_mu, d_order;
  V1<double> d_none;
  // extrapolation grade (has_uq_)
  bool has_uq_ = false, uq_run_ = false;
  int grade_ = 0, nfmax_ = 1, nqmax_ = 1;
  V1<int> d_uqoff, d_uqf, d_r1e0, d_r1mu, d_r1n, d_r1off, d_r1f, d_nfun, d_asioff;
  V1<double> d_uqcg, d_r1cg, d_asi, d_gamma;
  V2<double> d_uv, d_dFfs;    // [function][atom of the chunk], [density][atom of the chunk]
  Kokkos::View<long long *, DeviceType> d_key;

  template<class T> static V1<T> up(const char *name, const std::vector<T> &v)
  {
    V1<T> d(std::string(name), std::max<size_t>(v.size(), 1));
    auto h = Kokkos::create_mirror_view(d);
    for (size_t k = 0; k < v.size(); k++) h(k) = v[k];
    Kokkos::deep_copy(d, h);
    return d;
  }
  // slices of C-tilde rows per atom: about thr threads over the chunk, at least rmin rows a slice
  int slices(int nc, long long thr, int rmin) const
  {
    const int s = (int) std::min<long long>((thr + nc - 1) / nc, std::max(1, maxrows / rmin));
    return std::max(1, s);
  }
  // thread targets, measured (carbon / water, 32k atoms): the products want many threads (latency of the A gathers);
  // the adjoint on NVIDIA fewer (A100: 2^18 best, 2^22 30 % slower); on AMD (MI250X) both about 2^22
  static constexpr bool kHip =
#if defined(KOKKOS_ENABLE_HIP)
      true;
#else
      false;
#endif
  int slices_products(int nc) const { return kHip ? slices(nc, 1LL << 22, 16) : slices(nc, 1LL << 20, 4); }
  int slices_adjoint(int nc) const { return kHip ? slices(nc, 1LL << 22, 16) : slices(nc, 1LL << 18, 16); }
  void grow(int c)
  {
    if (c <= cap) return;
    MemKK::realloc_kokkos(d_Are, "pace_kk:Are", na, c);
    MemKK::realloc_kokkos(d_Aim, "pace_kk:Aim", na, c);
    if (has_uq_) {
      MemKK::realloc_kokkos(d_uv, "pace_kk:uv", nfmax_, c);
      MemKK::realloc_kokkos(d_dFfs, "pace_kk:dFfs", ndmax, c);
    }
    MemKK::realloc_kokkos(d_Wre, "pace_kk:Wre", na, c);
    MemKK::realloc_kokkos(d_Wim, "pace_kk:Wim", na, c);
    MemKK::realloc_kokkos(d_rho, "pace_kk:rho", ndmax, c);
    MemKK::realloc_kokkos(d_dF, "pace_kk:dF", ndmax, c);
    MemKK::realloc_kokkos(d_core, "pace_kk:core", c);
    MemKK::realloc_kokkos(d_dmin, "pace_kk:dmin", c);
    MemKK::realloc_kokkos(d_din, "pace_kk:din", c);
    MemKK::realloc_kokkos(d_dFc, "pace_kk:dFc", c);
    MemKK::realloc_kokkos(d_dFcut, "pace_kk:dFcut", c);
    MemKK::realloc_kokkos(d_ea, "pace_kk:ea", c);
    MemKK::realloc_kokkos(d_jj, "pace_kk:jj", c);
    MemKK::realloc_kokkos(d_mu, "pace_kk:mu", c);
    MemKK::realloc_kokkos(d_vir, "pace_kk:vir", c, 6);
    cap = c;
  }
  void choose_chunk(int inum)
  {
    if (chunksize >= 0 || (chunk_auto > 0 && inum <= auto_inum)) return;
    auto_inum = inum + inum / 10;
    chunk_auto = auto_chunk<DeviceType>(auto_inum, 1, ranks_per_device, [this](int a, int) { cap = 0; grow(a); },
                                        [] { return 0.0; }, log, "PACE/KK");
    cap = 0;
  }
};

}    // namespace kk
}    // namespace grace

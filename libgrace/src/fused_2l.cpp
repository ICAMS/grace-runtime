// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from LAMMPS src/ML-PACE/pair_grace_2l_fused.cpp (lammps-ace, feature/grace-fused); contributing author:
// Yury Lysogorskiy (ICAMS). Model layout, layers and compute phases are unchanged; LAMMPS calls are replaced
// by the bond source, Exchange, Outputs, grace::Error and the Logger.

#include "grace/fused_2l.h"

#include "grace/bonds.h"
#include "grace/fused_cpu.h"
#include "grace/type_map.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace grace {

using namespace fused;

namespace {
// "{}" placeholders filled in order (the log lines of the former LAMMPS style)
inline void braces_rec(std::ostringstream &o, const char *f) { o << f; }
template<class T, class... R> void braces_rec(std::ostringstream &o, const char *f, const T &v, const R &...r)
{
  const char *p = std::strstr(f, "{}");
  if (!p) { o << f; return; }
  o.write(f, p - f);
  o << v;
  braces_rec(o, p + 2, r...);
}
template<class... A> std::string braces(const char *f, const A &...a)
{
  std::ostringstream o;
  braces_rec(o, f, a...);
  return o.str();
}
// bond (i, j), del = x_i - x_j = -r rhat, fv = force on i: virial += del (x) fv (xx yy zz xy xz yz, as LAMMPS)
inline void add_bond_virial(double *v, const double *rh, double r, const double *fv)
{
  const double dx = -rh[0] * r, dy = -rh[1] * r, dz = -rh[2] * r;
  v[0] += dx * fv[0]; v[1] += dy * fv[1]; v[2] += dz * fv[2];
  v[3] += dx * fv[1]; v[4] += dx * fv[2]; v[5] += dy * fv[2];
}
}    // namespace

/* ---------------------------------------------------------------------- */

struct Grace2LFused::Work {
  // batch: up to NB atoms of one type
  int nbat = 0, ty = 0;
  int atom[NB];
  double E[NB];
  // bonds of the batch atoms (atom a: boff[a] .. boff[a+1])
  std::vector<int> boff, bj, bt;
  std::vector<double> rh, rr, rc;     // [nb][3], [nb], [nb]
  fvec Y, G;            // [nb][nlm], [nb][nlm][3]
  fvec R, dR;           // radial value, d/dx: [nl][P]
  fvec Rb;              // radial values of one atom's bonds [nb][nl][P]
  fvec row;             // one product row [NB][P] (fused product + reduction)
  std::vector<const float *> pz, pI;  // per bond: chemical weight, neighbour message
  // layer 1, [function][NB][channel]
  fvec A, A1, AA, AA1, AAA, AA2, AAAA, O1, Rho, Iv;
  fvec gA, gA1, gAA1, gAA2, gO1, gRho, gIb;    // gAA, gAAA, gAAAA live in AA, AAA, AAAA
  // layer 2
  fvec B0, YIp, YI, B, B1, BB, BB1, BBB, BB2, BBBB, O2;
  fvec dIg;                         // dE/dI_j of one bond, per (l, I) group [ngy][PR]
  fvec gB, gB1, gBB1, gBB2, gO2;    // the other layer-2 gradients live in the value buffers
  fvec u3, u4;                      // UQ with rowred: AAA / BBB, AAAA / BBBB materialised
  fused::UQQueue uqq;               // UQ: atoms waiting for the projection (layer 1 in P1, layer 2 in P3)
  std::vector<double> uqf, uqd;     // UQ feature, Mahalanobis delta
  double *ft = nullptr;                           // this thread's force rows
  double vir[6] = {0, 0, 0, 0, 0, 0};             // CsrBonds: this thread's bond virial
  bool want_vir = false;
  const Coupling *cpl = nullptr;   // two-group coupling (ComputeFlags::couple), nullptr: off
  std::vector<double> cp;           // coupling: per bond true r, dr'/dr, dr'/dlambda (bonds.h)
  double dedl = 0.0;                // coupling: this thread's dE/dlambda
};

/* ---------------------------------------------------------------------- */







/* ---------------------------------------------------------------------- */



/* ---------------------------------------------------------------------- */



/* ---------------------------------------------------------------------- */





/* ----------------------------------------------------------------------
   model -> CPU layout
------------------------------------------------------------------------- */





void Grace2LFused::build_splines()
{
  const GRACE2LModel &m = *model;
  const int N = spline_n;
  const double h = 1.0 / N;
  auto build = [&](const std::vector<GRACE2LModel::MLPLayer> &layers, int nrad, int P, fvec &tab) {
    const int nout = layers.back().n_out;
    if (nout != nrad * nl) throw Error("pair grace/2l/fused: radial MLP output width mismatch");
    std::vector<double> fv((size_t) (N + 1) * nout), dv((size_t) (N + 1) * nout), g, dg, out, dout;
    for (int k = 0; k <= N; k++) {
      radial_basis_x(k * h, m.radial_basis_p, m.nradbase, g, dg);
      mlp_x(layers, g, dg, out, dout);
      for (int o = 0; o < nout; o++) {
        fv[(size_t) k * nout + o] = out[o];
        dv[(size_t) k * nout + o] = h * dout[o];
      }
    }
    tab.assign((size_t) N * 4 * nl * P, 0.0f);
    for (int k = 0; k < N; k++)
      for (int n = 0; n < nrad; n++)
        for (int l = 0; l < nl; l++) {
          const int o = n * nl + l;
          const double f0 = fv[(size_t) k * nout + o], f1 = fv[(size_t) (k + 1) * nout + o];
          const double d0 = dv[(size_t) k * nout + o], d1 = dv[(size_t) (k + 1) * nout + o];
          const double c[4] = {f0, d0, -3.0 * f0 - 2.0 * d0 + 3.0 * f1 - d1, 2.0 * f0 + d0 - 2.0 * f1 + d1};
          for (int q = 0; q < 4; q++) tab[(((size_t) k * 4 + q) * nl + l) * P + n] = (float) c[q];
        }
  };
  build(m.L1_mlp_rad_layers, C1, P1, splR);
  build(m.L2_mlp_rad_layers, CR, PR, splR1);
}

void Grace2LFused::build_model()
{
  const GRACE2LModel &m = *model;
  const Ctx cx{ntypes, &type2mu, m.n_elements};
  lmax = m.lmax;
  if (lmax > MAX_LMAX) throw Error(braces("pair grace/2l/fused: lmax > {} not supported", MAX_LMAX));
  nl = lmax + 1;
  nlm = nl * nl;
  C1 = m.L1_nradmax;
  if (m.L1_fc_n_out != C1) throw Error("pair grace/2l/fused: needs L1_nradmax == L1_fc_n_out");
  CR = m.L2_nradmax;
  if (m.I_n_out != CR) throw Error("pair grace/2l/fused: needs I_n_out == L2_nradmax");
  C2 = m.L2_fc_n_out;
  if (m.B_n_out != C2) throw Error("pair grace/2l/fused: needs B_n_out == L2_fc_n_out");
  P1 = pad16(C1);
  PR = pad16(CR);
  P2 = pad16(C2);
  nI = m.I_n_funcs;
  nI1 = m.I1_n_funcs;
  nrho = m.rho_n_out;
  nI2 = m.I2_n_out;
  nB = m.B_n_funcs;

  // per-type tables
  rcut_type.assign((size_t) (ntypes + 1) * (ntypes + 1), 0.0);
  for (int a = 1; a <= ntypes; a++)
    for (int b = 1; b <= ntypes; b++) rcut_type[a * (ntypes + 1) + b] = pair_cutoff(m, type2mu[a], type2mu[b]);
  zA.assign((size_t) (ntypes + 1) * P1, 0.0f);
  zB0.assign((size_t) (ntypes + 1) * PR, 0.0f);
  shift_type.assign(ntypes + 1, 0.0);
  for (int t = 1; t <= ntypes; t++) {
    const int mu = type2mu[t];
    if (mu < 0) continue;    // unmapped: zero rows, never read
    for (int n = 0; n < C1; n++) {
      double v = 0.0;
      for (int e = 0; e < m.embedding_size; e++)
        v += m.A_lin_transform_W[e * C1 + n] * m.chem_embedding[mu * m.embedding_size + e];
      zA[t * P1 + n] = (float) (v * m.A_lin_transform_norm * m.A_inv_avg_n_neigh);
    }
    for (int n = 0; n < CR; n++) {
      double v = 0.0;
      for (int e = 0; e < m.embedding_size; e++)
        v += m.B0_lin_transform_W[e * CR + n] * m.chem_embedding[mu * m.embedding_size + e];
      zB0[t * PR + n] = (float) (v * m.B0_lin_transform_norm * m.B0_inv_avg_n_neigh);
    }
    shift_type[t] = mu < (int) m.shift_values.size() ? m.shift_values[mu] : 0.0;
  }

  // spherical harmonics coefficients
  sh_coefs(lmax, sh_a, sh_b, sh_c, sh_d);

  build_splines();

  // FCs and products (layer 1)
  build_fc(fcA1, m.fc_A1, C1, C1, P1, P1, P1, {}, cx);
  build_prod(pAA, m.prod_AA);
  nA1 = fcA1.nfl;
  nAA = pAA.nout;
  build_fc(fcAA1, m.fc_AA1, C1, C1, P1, P1, P1, {}, cx);
  build_prod(pAAA, m.prod_AAA);
  nAAA = pAAA.nout;
  build_fc(fcAA2, m.fc_AA2, C1, C1, P1, P1, P1, {}, cx);
  build_prod(pAAAA, m.prod_AAAA);
  nAAAA = pAAAA.nout;
  if (nA1 != nlm || fcAA1.nfl != nAA || fcAA2.nfl != nAA)
    throw Error("pair grace/2l/fused: unexpected layer-1 function counts");

  // I1 (fo = total_sum_ind), rho (one output function), I from I1
  PI1 = pad16(m.I1_n_out);
  PRHO = pad16(nrho);
  PO2 = pad16(nI2);
  build_red<GRACE2LModel::ReduceWeights>(rI1, {&m.I1_reduce_A, &m.I1_reduce_AA, &m.I1_reduce_AAA, &m.I1_reduce_AAAA}, {-1, -1, -1, -1},
            m.I1_n_out, nI1, true, PI1, {P1, P1, P1, P1}, cx);
  build_red<GRACE2LModel::ReduceWeights>(rRho, {&m.rho_reduce_A, &m.rho_reduce_AA, &m.rho_reduce_AAA, &m.rho_reduce_AAAA}, {0, 0, 0, 0}, nrho,
            1, true, PRHO, {P1, P1, P1, P1}, cx);
  build_red<GRACE2LModel::ReduceWeights>(rI, {&m.I_reduce_I1}, {-1}, CR, nI, false, PR, {PI1}, cx);

  // layer 2
  {
    // unique (y, I) pairs
    const auto &yi = m.yi;
    // pairs ordered by (l(y), I, y): groups of equal (l, I) share R[l] I_j in the bond kernels
    auto l_of = [](int y) { return (int) std::lround(std::floor(std::sqrt((double) y) + 1e-9)); };
    std::map<std::array<int, 3>, int> pid;
    for (int t = 0; t < yi.n_cg_terms; t++) {
      const int y = yi.lr_inds[2 * t], ii = yi.lr_inds[2 * t + 1];
      pid[{l_of(y), ii, y}] = 0;
    }
    yp_y.clear(); yp_i.clear(); yp_l.clear();
    yg_off.clear(); yg_l.clear(); yg_i.clear();
    for (auto &kv : pid) {
      kv.second = yp_y.size();
      const int l = kv.first[0], ii = kv.first[1];
      if (yg_l.empty() || yg_l.back() != l || yg_i.back() != ii) {
        yg_off.push_back(yp_y.size());
        yg_l.push_back(l);
        yg_i.push_back(ii);
      }
      yp_l.push_back(l);
      yp_i.push_back(ii);
      yp_y.push_back(kv.first[2]);
    }
    yg_off.push_back(yp_y.size());
    std::vector<std::array<int, 3>> terms;    // (pair, o, idx)
    for (int t = 0; t < yi.n_cg_terms; t++) {
      const int y = yi.lr_inds[2 * t], ii = yi.lr_inds[2 * t + 1];
      terms.push_back({pid[{l_of(y), ii, y}], yi.m_sum_ind[t], t});
    }
    std::stable_sort(terms.begin(), terms.end(), [](const auto &a, const auto &b) { return a[1] < b[1]; });
    nYI = 0;
    yt_pair.clear(); yt_o.clear(); yt_c.clear();
    for (auto &q : terms) {
      yt_pair.push_back(q[0]);
      yt_o.push_back(q[1]);
      yt_c.push_back((float) yi.cg_coeff[q[2]]);
      nYI = std::max(nYI, q[1] + 1);
    }
    yt_off.assign(nYI + 1, 0);
    for (int o : yt_o) yt_off[o + 1]++;
    for (int q = 0; q < nYI; q++) yt_off[q + 1] += yt_off[q];
    yi_inv_avg = (float) yi.inv_avg_n_neigh;
    for (int y : yp_y)
      if (y >= nlm) throw Error("pair grace/2l/fused: YI harmonic index out of range");
    for (int ii : yp_i)
      if (ii >= nI) throw Error("pair grace/2l/fused: YI message index out of range");
  }
  build_red<GRACE2LModel::ReduceWeights>(rB, {&m.B_reduce_YI, &m.B_reduce_B0}, {-1, -1}, C2, nB, false, P2, {PR, PR}, cx);
  build_fc(fcB1, m.fc_B1, C2, C2, P2, P2, P2, {}, cx);
  build_prod(pBB, m.prod_BB);
  nBB = pBB.nout;
  build_fc(fcBB1, m.fc_BB1, C2, C2, P2, P2, P2, {}, cx);
  build_prod(pBBB, m.prod_BBB);
  nBBB = pBBB.nout;
  build_fc(fcBB2, m.fc_BB2, C2, C2, P2, P2, P2, {}, cx);
  build_prod(pBBBB, m.prod_BBBB);
  nBBBB = pBBBB.nout;
  if (fcB1.nfl != nB || fcBB1.nfl != nBB || fcBB2.nfl != nBB)
    throw Error("pair grace/2l/fused: unexpected layer-2 function counts");
  build_red<GRACE2LModel::ReduceWeights>(rI2, {&m.I2_reduce_B, &m.I2_reduce_BB, &m.I2_reduce_BBB, &m.I2_reduce_BBBB}, {0, 0, 0, 0}, nI2, 1,
            true, PO2, {P2, P2, P2, P2}, cx);

  // ---- liveness: outputs never read on the way to the energy are not computed (FC outputs, CG-product terms)
  {
    auto ci_of = [](const Red &R, int src, std::vector<char> &need) {
      for (int c : R.src[src].ci) need[c] = 1;
    };
    auto prune = [](Prod &p, const std::vector<char> &need) {
      Prod q = p;
      q.l.clear(); q.r.clear(); q.o.clear(); q.c.clear();
      for (size_t t = 0; t < p.c.size(); t++)
        if (need[p.o[t]]) { q.l.push_back(p.l[t]); q.r.push_back(p.r[t]); q.o.push_back(p.o[t]); q.c.push_back(p.c[t]); }
      q.off.assign(q.nout + 1, 0);
      for (int o : q.o) q.off[o + 1]++;
      for (int o = 0; o < q.nout; o++) q.off[o + 1] += q.off[o];
      p = q;
    };
    auto operands = [](const Prod &p, int n, bool left, bool right) {
      std::vector<char> live(n, 0);
      for (size_t t = 0; t < p.c.size(); t++) {
        if (left) live[p.l[t]] = 1;
        if (right) live[p.r[t]] = 1;
      }
      return live;
    };
    auto count = [](const std::vector<char> &v) { int n = 0; for (char c : v) n += c; return n; };
    // layer 1
    std::vector<char> nAAAv(nAAA, 0), nAAAAv(nAAAA, 0), nAAv(nAA, 0);
    ci_of(rI1, 2, nAAAv); ci_of(rRho, 2, nAAAv);
    ci_of(rI1, 3, nAAAAv); ci_of(rRho, 3, nAAAAv);
    prune(pAAA, nAAAv);
    prune(pAAAA, nAAAAv);
    const auto liveAA1 = operands(pAAA, nAA, true, false);
    const auto liveAA2 = operands(pAAAA, nAA, true, true);
    for (int q = 0; q < nAA; q++) nAAv[q] = liveAA1[q] || liveAA2[q];
    ci_of(rI1, 1, nAAv); ci_of(rRho, 1, nAAv);
    prune(pAA, nAAv);
    const auto liveA1 = operands(pAA, nlm, true, true);
    build_fc(fcA1, m.fc_A1, C1, C1, P1, P1, P1, liveA1, cx);
    build_fc(fcAA1, m.fc_AA1, C1, C1, P1, P1, P1, liveAA1, cx);
    build_fc(fcAA2, m.fc_AA2, C1, C1, P1, P1, P1, liveAA2, cx);
    // layer 2
    std::vector<char> nBBBv(nBBB, 0), nBBBBv(nBBBB, 0), nBBv(nBB, 0);
    ci_of(rI2, 2, nBBBv);
    ci_of(rI2, 3, nBBBBv);
    prune(pBBB, nBBBv);
    prune(pBBBB, nBBBBv);
    const auto liveBB1 = operands(pBBB, nBB, true, false);
    const auto liveBB2 = operands(pBBBB, nBB, true, true);
    for (int q = 0; q < nBB; q++) nBBv[q] = liveBB1[q] || liveBB2[q];
    ci_of(rI2, 1, nBBv);
    prune(pBB, nBBv);
    const auto liveB1 = operands(pBB, nB, true, true);
    build_fc(fcB1, m.fc_B1, C2, C2, P2, P2, P2, liveB1, cx);
    build_fc(fcBB1, m.fc_BB1, C2, C2, P2, P2, P2, liveBB1, cx);
    build_fc(fcBB2, m.fc_BB2, C2, C2, P2, P2, P2, liveBB2, cx);
    log.info(braces("[grace/2l/fused] live outputs: A1 {}/{} AA {}/{} AA1 {}/{} AAA {}/{} AA2 {}/{} AAAA {}/{} | "
                     "B1 {}/{} BB {}/{} BB1 {}/{} BBB {}/{} BB2 {}/{} BBBB {}/{}",
                     count(liveA1), nlm, count(nAAv), nAA, count(liveAA1), nAA, count(nAAAv), nAAA, count(liveAA2),
                     nAA, count(nAAAAv), nAAAA, count(liveB1), nB, count(nBBv), nBB, count(liveBB1), nBB,
                     count(nBBBv), nBBB, count(liveBB2), nBB, count(nBBBBv), nBBBB));
  }

  rowred = !getenv("G2LF_NOROWRED");
  if (rowred) {
    build_rowred(rrI1_3, rI1, 2, nAAA); build_rowred(rrRho_3, rRho, 2, nAAA);
    build_rowred(rrI1_4, rI1, 3, nAAAA); build_rowred(rrRho_4, rRho, 3, nAAAA);
    build_rowred(rrI2_3, rI2, 2, nBBB); build_rowred(rrI2_4, rI2, 3, nBBBB);
  }

  // norms + energy MLP
  lnI_scale = m.I_nl_LN_scale;
  ln0_scale = m.I_0_LN_scale;
  if ((int) lnI_scale.size() != nrho - 1 || (int) ln0_scale.size() != nI2 || nrho != nI2)
    throw Error("pair grace/2l/fused: unexpected RMSNorm widths");
  eW.clear(); e_norm.clear(); e_nin.clear(); e_nout.clear();
  for (auto &L : m.energy_mlp_layers) {
    eW.push_back(L.W);
    e_norm.push_back(L.norm);
    e_nin.push_back(L.n_in);
    e_nout.push_back(L.n_out);
  }
  if (e_nin.empty() || e_nin[0] != nrho - 1 || e_nout.back() != 1)
    throw Error("pair grace/2l/fused: unexpected energy MLP shape");
  // energy_atom keeps the activations in fixed stack buffers act/pre[8][256] (it runs inside the parallel region,
  // where it cannot throw): at most 7 layers and every width at most 256
  if (e_nin.size() > 7) throw Error(braces("pair grace/2l/fused: energy MLP has {} layers, at most 7 supported", e_nin.size()));
  for (size_t L = 0; L < e_nin.size(); L++)
    if (e_nin[L] > 256 || e_nout[L] > 256 || e_nin[L] < 1 || e_nout[L] < 1 || L + 1 < e_nin.size() && e_nout[L] != e_nin[L + 1] ||
        eW[L].size() != (size_t) e_nin[L] * e_nout[L])
      throw Error(braces("pair grace/2l/fused: energy MLP layer {} is {} x {} (widths 1..256, consecutive layers must match)",
                         L, e_nin[L], e_nout[L]));
  e_act = m.energy_mlp_activation;
  out_scale = m.output_scale;

  log.info(braces("[grace/2l/fused] lmax {} | L1 {} ch, A1/AA/AAA/AAAA {}/{}/{}/{} | I {}x{} | L2 {} / {} ch, "
                   "YI {} ({} pairs), B/BB/BBB/BBBB {}/{}/{}/{} | spline N {}",
                   lmax, C1, nA1, nAA, nAAA, nAAAA, nI, CR, CR, C2, nYI, (int) yp_y.size(), nB, nBB, nBBB, nBBBB,
                   spline_n));
}

/* ----------------------------------------------------------------------
   kernels on [function][channel] tensors
------------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
   bonds of the batch atoms: rhat, r, rcut_ij, Y_lm and dY/d(rhat) (polynomial extension)
------------------------------------------------------------------------- */



/* ----------------------------------------------------------------------
   layer 1 forward of a batch (bonds gathered): A .. AAAA, I1 (O1), rho (Rho), I (Iv)
------------------------------------------------------------------------- */

void Grace2LFused::layer1_forward(Work &w)
{
  const int ty = w.ty, RL1 = NB * P1;
  float *A = w.A.data();
  vzero(A, (size_t) nlm * RL1);
  {
    PROF(T_RAD1);
    for (int a = 0; a < w.nbat; a++) {
      const int b0 = w.boff[a], nb = w.boff[a + 1] - b0;
      if ((int) w.Rb.size() < nb * nl * P1) w.Rb.resize((size_t) nb * nl * std::max({P1, PR, P2}));
      if ((int) w.pz.size() < nb) { w.pz.resize(nb); w.pI.resize(nb); }
      for (int b = 0; b < nb; b++) {
        spline_eval(splR.data(), spline_n, nl, P1, w.rr[b0 + b] / w.rc[b0 + b], w.Rb.data() + (size_t) b * nl * P1,
                    nullptr);
        w.pz[b] = zA.data() + (size_t) w.bt[b0 + b] * P1;
      }
      G2LF_NVD(P1, atom_scatter<NV>(nb, nl, nlm, w.Rb.data(), w.pz.data(), w.Y.data() + (size_t) b0 * nlm, A + a * P1,
                                    RL1))
    }
  }
  { PROF(T_FC1); fc_fwd(fcA1, A, A, w.A1.data()); }
  { PROF(T_PROD1); prod_fwd(pAA, w.A1.data(), w.A1.data(), w.AA.data(), P1); }
  { PROF(T_FC1); fc_fwd(fcAA1, w.AA.data(), A, w.AA1.data()); }
  { PROF(T_FC1); fc_fwd(fcAA2, w.AA.data(), A, w.AA2.data()); }
  if (rowred) {
    const float *X[4] = {A, w.AA.data(), nullptr, nullptr};
    { PROF(T_RED1); red_fwd(rI1, ty, X, w.O1.data()); }
    { PROF(T_RED1); red_fwd(rRho, ty, X, w.Rho.data()); }
    const Red *reds[2] = {&rI1, &rRho};
    float *outs[2] = {w.O1.data(), w.Rho.data()};
    { const RowRed *rrs[2] = {&rrI1_3, &rrRho_3};
      PROF(T_PROD1); prod_red_fwd(pAAA, w.AA1.data(), A, P1, ty, reds, rrs, 2, outs, w.row.data()); }
    { const RowRed *rrs[2] = {&rrI1_4, &rrRho_4};
      PROF(T_PROD1); prod_red_fwd(pAAAA, w.AA2.data(), w.AA2.data(), P1, ty, reds, rrs, 2, outs, w.row.data()); }
  } else {
    { PROF(T_PROD1); prod_fwd(pAAA, w.AA1.data(), A, w.AAA.data(), P1); }
    { PROF(T_PROD1); prod_fwd(pAAAA, w.AA2.data(), w.AA2.data(), w.AAAA.data(), P1); }
    const float *X[4] = {A, w.AA.data(), w.AAA.data(), w.AAAA.data()};
    { PROF(T_RED1); red_fwd(rI1, ty, X, w.O1.data()); }
    { PROF(T_RED1); red_fwd(rRho, ty, X, w.Rho.data()); }
  }
  const float *XI[1] = {w.O1.data()};
  { PROF(T_RED1); red_fwd(rI, 0, XI, w.Iv.data()); }
}

/* ----------------------------------------------------------------------
   layer 1 reverse of a batch (forward state in w): seeds dE/dI (gI_sum) and dE/drho (grho_glob)
------------------------------------------------------------------------- */

void Grace2LFused::layer1_backward(Work &w)
{
  const int ty = w.ty, RL1 = NB * P1;
  float *gO1 = w.gO1.data(), *gRho = w.gRho.data(), *gIb = w.gIb.data();
  vzero(gO1, (size_t) nI1 * NB * PI1);
  vzero(gRho, (size_t) NB * PRHO);
  vzero(gIb, (size_t) nI * NB * PR);
  for (int a = 0; a < w.nbat; a++) {
    const int i = w.atom[a];
    const float *gI = gI_sum.data() + (size_t) i * nI * PR;
    for (int fn = 0; fn < nI; fn++) std::memcpy(gIb + ((size_t) fn * NB + a) * PR, gI + fn * PR, PR * sizeof(float));
    const double *gr = grho_glob.data() + (size_t) i * nrho;
    for (int k = 0; k < nrho; k++) gRho[a * PRHO + k] = (float) gr[k];
  }
  {
    float *gX[1] = {gO1};
    PROF(T_RED1B);
    red_bwd(rI, 0, gIb, gX);
  }
  // AA, AAA, AAAA values are not needed by the reverse: their buffers hold the gradients
  float *gA = w.gA.data(), *gAA = w.AA.data(), *gAAA = w.AAA.data(), *gAAAA = w.AAAA.data();
  vzero(gA, (size_t) nlm * RL1);
  vzero(w.gA1.data(), (size_t) nlm * RL1);
  vzero(gAA, (size_t) nAA * RL1);
  vzero(w.gAA1.data(), (size_t) nAA * RL1);
  vzero(w.gAA2.data(), (size_t) nAA * RL1);
  if (!rowred) {
    vzero(gAAA, (size_t) nAAA * RL1);
    vzero(gAAAA, (size_t) nAAAA * RL1);
  }
  {
    float *gX[4] = {gA, gAA, gAAA, gAAAA};
    PROF(T_RED1B);
    red_bwd(rI1, ty, gO1, gX);
    red_bwd(rRho, ty, gRho, gX);
  }
  const float *A = w.A.data();
  const Red *reds[2] = {&rI1, &rRho};
  const float *gouts[2] = {gO1, gRho};
  if (rowred) {
    const RowRed *rrs[2] = {&rrI1_4, &rrRho_4};
    PROF(T_PROD1B);
    prod_red_bwd(pAAAA, w.AA2.data(), w.AA2.data(), P1, ty, reds, rrs, 2, gouts, w.gAA2.data(), w.gAA2.data(),
                 w.row.data());
  } else {
    PROF(T_PROD1B);
    prod_bwd(pAAAA, w.AA2.data(), w.AA2.data(), gAAAA, w.gAA2.data(), w.gAA2.data(), P1);
  }
  { PROF(T_FC1B); fc_bwd(fcAA2, w.gAA2.data(), gAA, gA); }
  if (rowred) {
    const RowRed *rrs[2] = {&rrI1_3, &rrRho_3};
    PROF(T_PROD1B);
    prod_red_bwd(pAAA, w.AA1.data(), A, P1, ty, reds, rrs, 2, gouts, w.gAA1.data(), gA, w.row.data());
  } else {
    PROF(T_PROD1B);
    prod_bwd(pAAA, w.AA1.data(), A, gAAA, w.gAA1.data(), gA, P1);
  }
  { PROF(T_FC1B); fc_bwd(fcAA1, w.gAA1.data(), gAA, gA); }
  { PROF(T_PROD1B); prod_bwd(pAA, w.A1.data(), w.A1.data(), gAA, w.gA1.data(), w.gA1.data(), P1); }
  { PROF(T_FC1B); fc_bwd(fcA1, w.gA1.data(), gA, gA); }

  // bonds
  PROF(T_RAD1B);
  v16 gRv[(MAX_LMAX + 1) * 8], gYv[MAX_NLM];
  for (int a = 0; a < w.nbat; a++) {
    const int i = w.atom[a];
    for (int b = w.boff[a]; b < w.boff[a + 1]; b++) {
      spline_eval(splR.data(), spline_n, nl, P1, w.rr[b] / w.rc[b], w.R.data(), w.dR.data());
      const float *z = zA.data() + (size_t) w.bt[b] * P1;
      const float *Y = w.Y.data() + (size_t) b * nlm;
      for (int q = 0; q < nlm; q++) gYv[q] = v16{};
      for (int q = 0; q < nl * (P1 / 16); q++) gRv[q] = v16{};
      double dEdx = 0.0;
      G2LF_NVD(P1, bond_scatter_bwd_t<NV>(w.R.data(), z, Y, nl, gA + a * P1, RL1, gYv, gRv);
               dEdx = bond_dEdx_t<NV>(gRv, w.dR.data(), nl))
      double fv[3];
      bond_force(dEdx, gYv, w.G.data() + (size_t) b * nlm * 3, nlm, w.rr[b], w.rc[b], &w.rh[3 * b], fv);
      const double rv = w.cpl ? couple_force(&w.cp[3 * b], w.rr[b], &w.rh[3 * b], fv, w.dedl) : w.rr[b];
      add_force(w.ft, i, w.bj[b], fv);
      if (w.want_vir) add_bond_virial(w.vir, &w.rh[3 * b], rv, fv);
      if (vflag_atom) add_vatom(vatom, i, w.bj[b], &w.rh[3 * b], rv, fv);
    }
  }
}

/* ----------------------------------------------------------------------
   energy of one atom from rho and I2 (RMSNorms + MLP, fp64) and dE/drho, dE/dI2
------------------------------------------------------------------------- */

double Grace2LFused::energy_atom(const float *rho, const float *I2, int ty, double *grho, float *gI2) const
{
  const int nr = nrho;
  double ms1 = 0.0, ms0 = 0.0;
  for (int k = 1; k < nr; k++) ms1 += (double) rho[k] * rho[k];
  for (int k = 0; k < nr; k++) ms0 += (double) I2[k] * I2[k];
  const double s1 = std::sqrt(ms1 / (nr - 1) + 1e-8), s0 = std::sqrt(ms0 / nr + 1e-8);
  constexpr int MAXW = 256;
  const int nlay = eW.size();
  double act[8][MAXW], pre[8][MAXW];
  for (int k = 1; k < nr; k++) act[0][k - 1] = rho[k] / s1 * lnI_scale[k - 1] + I2[k] / s0 * ln0_scale[k];
  for (int L = 0; L < nlay; L++) {
    for (int c = 0; c < e_nout[L]; c++) pre[L][c] = 0.0;
    for (int a = 0; a < e_nin[L]; a++) {
      const double x = act[L][a];
      const double *Wr = eW[L].data() + (size_t) a * e_nout[L];
      for (int c = 0; c < e_nout[L]; c++) pre[L][c] += x * Wr[c];
    }
    for (int c = 0; c < e_nout[L]; c++) {
      pre[L][c] *= e_norm[L];
      const double v = pre[L][c];
      act[L + 1][c] = (L < nlay - 1) ? ((e_act == 1) ? std::tanh(v) : v / (1.0 + std::exp(-v))) : v;
    }
  }
  const double Ei = (act[nlay][0] + rho[0] + I2[0] / s0 * ln0_scale[0]) * out_scale + shift_type[ty];
  // reverse through the MLP
  double g[MAXW], gn[MAXW];
  g[0] = out_scale;
  for (int L = nlay - 1; L >= 0; L--) {
    if (L < nlay - 1)
      for (int c = 0; c < e_nout[L]; c++) {
        const double v = pre[L][c];
        if (e_act == 1) {
          const double t = std::tanh(v);
          g[c] *= 1.0 - t * t;
        } else {
          const double sg = 1.0 / (1.0 + std::exp(-v));
          g[c] *= sg * (1.0 + v * (1.0 - sg));
        }
      }
    for (int a = 0; a < e_nin[L]; a++) {
      const double *Wr = eW[L].data() + (size_t) a * e_nout[L];
      double s = 0.0;
      for (int c = 0; c < e_nout[L]; c++) s += g[c] * Wr[c];
      gn[a] = s * e_norm[L];
    }
    for (int a = 0; a < e_nin[L]; a++) g[a] = gn[a];
  }
  // g = dE/dx_in; RMSNorm reverses (y_k = x_k s_k / sigma)
  grho[0] = out_scale;
  double t1 = 0.0;
  for (int k = 1; k < nr; k++) t1 += g[k - 1] * lnI_scale[k - 1] * rho[k];
  for (int k = 1; k < nr; k++) grho[k] = g[k - 1] * lnI_scale[k - 1] / s1 - rho[k] * t1 / ((nr - 1) * s1 * s1 * s1);
  double gy[MAXW];
  gy[0] = out_scale;
  for (int k = 1; k < nr; k++) gy[k] = g[k - 1];
  double t0 = 0.0;
  for (int k = 0; k < nr; k++) t0 += gy[k] * ln0_scale[k] * I2[k];
  for (int k = 0; k < nr; k++) gI2[k] = (float) (gy[k] * ln0_scale[k] / s0 - I2[k] * t0 / (nr * s0 * s0 * s0));
  return Ei;
}

/* ----------------------------------------------------------------------
   layer 2 forward, energy and layer-2 reverse of a batch; energies in w.E
------------------------------------------------------------------------- */

void Grace2LFused::layer2(Work &w)
{
  const int ty = w.ty, RLR = NB * PR, RL2 = NB * P2;
  const int npair = yp_y.size(), ngy = yg_l.size();
  float *B0 = w.B0.data(), *YIp = w.YIp.data();
  vzero(B0, (size_t) nlm * RLR);
  vzero(YIp, (size_t) npair * RLR);
  {
    PROF(T_RAD2);
    for (int a = 0; a < w.nbat; a++) {
      const int b0 = w.boff[a], nb = w.boff[a + 1] - b0;
      if ((int) w.Rb.size() < nb * nl * PR) w.Rb.resize((size_t) nb * nl * std::max({P1, PR, P2}));
      if ((int) w.pz.size() < nb) { w.pz.resize(nb); w.pI.resize(nb); }
      for (int b = 0; b < nb; b++) {
        spline_eval(splR1.data(), spline_n, nl, PR, w.rr[b0 + b] / w.rc[b0 + b], w.Rb.data() + (size_t) b * nl * PR,
                    nullptr);
        w.pz[b] = zB0.data() + (size_t) w.bt[b0 + b] * PR;
        w.pI[b] = I_glob.data() + (size_t) w.bj[b0 + b] * nI * PR;
      }
      const float *Yb = w.Y.data() + (size_t) b0 * nlm;
      G2LF_NVD(PR, atom_scatter<NV>(nb, nl, nlm, w.Rb.data(), w.pz.data(), Yb, B0 + a * PR, RLR);
               atom_yi<NV>(nb, nl, nlm, w.Rb.data(), w.pI.data(), Yb, ngy, yg_off.data(), yg_l.data(), yg_i.data(),
                           yp_y.data(), YIp + a * PR, RLR))
    }
  }
  float *YI = w.YI.data();
  for (int o = 0; o < nYI; o++) {
    float *y = YI + (size_t) o * RLR;
    vzero(y, RLR);
    for (int t = yt_off[o]; t < yt_off[o + 1]; t++) axpy(y, YIp + (size_t) yt_pair[t] * RLR, yi_inv_avg * yt_c[t], RLR);
  }
  {
    const float *X[2] = {YI, B0};
    PROF(T_RED2);
    red_fwd(rB, 0, X, w.B.data());
  }
  const float *B = w.B.data();
  { PROF(T_FC2); fc_fwd(fcB1, B, B, w.B1.data()); }
  { PROF(T_PROD2); prod_fwd(pBB, w.B1.data(), w.B1.data(), w.BB.data(), P2); }
  { PROF(T_FC2); fc_fwd(fcBB1, w.BB.data(), B, w.BB1.data()); }
  { PROF(T_FC2); fc_fwd(fcBB2, w.BB.data(), B, w.BB2.data()); }
  if (rowred) {
    {
      const float *X[4] = {B, w.BB.data(), nullptr, nullptr};
      PROF(T_RED2);
      red_fwd(rI2, ty, X, w.O2.data());
    }
    const Red *reds[1] = {&rI2};
    float *outs[1] = {w.O2.data()};
    { const RowRed *rrs[1] = {&rrI2_3};
      PROF(T_PROD2); prod_red_fwd(pBBB, w.BB1.data(), B, P2, ty, reds, rrs, 1, outs, w.row.data()); }
    { const RowRed *rrs[1] = {&rrI2_4};
      PROF(T_PROD2); prod_red_fwd(pBBBB, w.BB2.data(), w.BB2.data(), P2, ty, reds, rrs, 1, outs, w.row.data()); }
  } else {
    { PROF(T_PROD2); prod_fwd(pBBB, w.BB1.data(), B, w.BBB.data(), P2); }
    { PROF(T_PROD2); prod_fwd(pBBBB, w.BB2.data(), w.BB2.data(), w.BBBB.data(), P2); }
    const float *X[4] = {B, w.BB.data(), w.BBB.data(), w.BBBB.data()};
    PROF(T_RED2);
    red_fwd(rI2, ty, X, w.O2.data());
  }

  if (uq_on) uq_l2(w);

  // energy and its gradient wrt rho_i, I2_i
  float *gO2 = w.gO2.data();
  vzero(gO2, (size_t) NB * PO2);
  {
    PROF(T_EN);
    for (int a = 0; a < w.nbat; a++) {
      const int i = w.atom[a];
      w.E[a] = energy_atom(rho_glob.data() + (size_t) i * nrho, w.O2.data() + a * PO2, ty,
                           grho_glob.data() + (size_t) i * nrho, gO2 + a * PO2);
    }
  }
  if (eonly) return;

  // layer 2 reverse
  // BB, BBB, BBBB, YI, B0, YIp values are not needed by the reverse: their buffers hold the gradients
  float *gB = w.gB.data(), *gBB = w.BB.data(), *gBBB = w.BBB.data(), *gBBBB = w.BBBB.data();
  vzero(gB, (size_t) nB * RL2);
  vzero(w.gB1.data(), (size_t) nB * RL2);
  vzero(gBB, (size_t) nBB * RL2);
  vzero(w.gBB1.data(), (size_t) nBB * RL2);
  vzero(w.gBB2.data(), (size_t) nBB * RL2);
  if (!rowred) {
    vzero(gBBB, (size_t) nBBB * RL2);
    vzero(gBBBB, (size_t) nBBBB * RL2);
  }
  {
    float *gX[4] = {gB, gBB, gBBB, gBBBB};
    PROF(T_RED2B);
    red_bwd(rI2, ty, gO2, gX);
  }
  const Red *reds2[1] = {&rI2};
  const float *gouts2[1] = {gO2};
  if (rowred) {
    const RowRed *rrs[1] = {&rrI2_4};
    PROF(T_PROD2B);
    prod_red_bwd(pBBBB, w.BB2.data(), w.BB2.data(), P2, ty, reds2, rrs, 1, gouts2, w.gBB2.data(), w.gBB2.data(),
                 w.row.data());
  } else {
    PROF(T_PROD2B);
    prod_bwd(pBBBB, w.BB2.data(), w.BB2.data(), gBBBB, w.gBB2.data(), w.gBB2.data(), P2);
  }
  { PROF(T_FC2B); fc_bwd(fcBB2, w.gBB2.data(), gBB, gB); }
  if (rowred) {
    const RowRed *rrs[1] = {&rrI2_3};
    PROF(T_PROD2B);
    prod_red_bwd(pBBB, w.BB1.data(), B, P2, ty, reds2, rrs, 1, gouts2, w.gBB1.data(), gB, w.row.data());
  } else {
    PROF(T_PROD2B);
    prod_bwd(pBBB, w.BB1.data(), B, gBBB, w.gBB1.data(), gB, P2);
  }
  { PROF(T_FC2B); fc_bwd(fcBB1, w.gBB1.data(), gBB, gB); }
  { PROF(T_PROD2B); prod_bwd(pBB, w.B1.data(), w.B1.data(), gBB, w.gB1.data(), w.gB1.data(), P2); }
  { PROF(T_FC2B); fc_bwd(fcB1, w.gB1.data(), gB, gB); }
  float *gYI = w.YI.data(), *gB0 = w.B0.data(), *gYIp = w.YIp.data();
  vzero(gYI, (size_t) nYI * RLR);
  vzero(gB0, (size_t) nlm * RLR);
  {
    float *gX[2] = {gYI, gB0};
    PROF(T_RED2B);
    red_bwd(rB, 0, gB, gX);
  }
  vzero(gYIp, (size_t) npair * RLR);
  for (int o = 0; o < nYI; o++)
    for (int t = yt_off[o]; t < yt_off[o + 1]; t++)
      axpy(gYIp + (size_t) yt_pair[t] * RLR, gYI + (size_t) o * RLR, yi_inv_avg * yt_c[t], RLR);

  // bonds: forces of layer 2 and dE/dI_j
  PROF(T_RAD2B);
  v16 gRv[(MAX_LMAX + 1) * 8], gYv[MAX_NLM];
  for (int a = 0; a < w.nbat; a++) {
    const int i = w.atom[a];
    for (int b = w.boff[a]; b < w.boff[a + 1]; b++) {
      spline_eval(splR1.data(), spline_n, nl, PR, w.rr[b] / w.rc[b], w.R.data(), w.dR.data());
      const float *z = zB0.data() + (size_t) w.bt[b] * PR;
      const float *Y = w.Y.data() + (size_t) b * nlm;
      const int j = w.bj[b];
      const float *Ij = I_glob.data() + (size_t) j * nI * PR;
      float *gIj = gI_sum.data() + (size_t) j * nI * PR;
      for (int q = 0; q < nlm; q++) gYv[q] = v16{};
      for (int q = 0; q < nl * (PR / 16); q++) gRv[q] = v16{};
      double dEdx = 0.0;
      G2LF_NVD(PR, bond_scatter_bwd_t<NV>(w.R.data(), z, Y, nl, gB0 + a * PR, RLR, gYv, gRv);
               bond_yi_bwd_g<NV>(w.R.data(), Y, Ij, ngy, yg_off.data(), yg_l.data(), yg_i.data(), yp_y.data(),
                                 gYIp + a * PR, RLR, gYv, gRv, w.dIg.data());
               {
                 SpinLock lk(gI_lock[j]);    // row j of the shared dE/dI (one row per atom, not per thread)
                 bond_yi_commit<NV>(ngy, yg_i.data(), w.dIg.data(), gIj);
               }
               dEdx = bond_dEdx_t<NV>(gRv, w.dR.data(), nl))
      double fv[3];
      bond_force(dEdx, gYv, w.G.data() + (size_t) b * nlm * 3, nlm, w.rr[b], w.rc[b], &w.rh[3 * b], fv);
      const double rv = w.cpl ? couple_force(&w.cp[3 * b], w.rr[b], &w.rh[3 * b], fv, w.dedl) : w.rr[b];
      add_force(w.ft, i, j, fv);
      if (w.want_vir) add_bond_virial(w.vir, &w.rh[3 * b], rv, fv);
      if (vflag_atom) add_vatom(vatom, i, j, &w.rh[3 * b], rv, fv);
    }
  }
}

/* ----------------------------------------------------------------------
   setup (was the LAMMPS style's constructor, coeff(), init_style(), init_one())
------------------------------------------------------------------------- */

Grace2LFused::Grace2LFused(std::shared_ptr<const GRACE2LModel> m, FusedSettings s, Logger log_) :
    log(log_), model(std::move(m))
{
  spline_n = s.spline_n;
  store_l1 = s.store_l1;
  store_l1_max_gb = s.store_l1_max_gb;
  if (spline_n < 16) throw Error("pair grace/2l/fused: spline N must be >= 16");
  g_prof_on = getenv("G2LF_PROF") != nullptr;
}

Grace2LFused::~Grace2LFused() = default;

void Grace2LFused::setup_types(int ntypes_, const std::vector<int> &type2mu_)
{
  ntypes = ntypes_;
  type2mu = type2mu_;
  check_type2mu(type2mu, ntypes, model->n_elements, "pair grace/2l/fused");
  build_model();
  const int ntp = ntypes + 1;
  cutsq_flat.assign((size_t) ntp * ntp, 0.0);
  cutsq_rows.resize(ntp);
  for (int i = 0; i < ntp; i++) cutsq_rows[i] = cutsq_flat.data() + (size_t) i * ntp;
  for (int i = 1; i <= ntypes; i++)
    for (int j = 1; j <= ntypes; j++) {
      const double rc = rcut_type[i * ntp + j];
      cutsq_rows[i][j] = rc * rc;
    }
  work.clear();
  nthreads = 0;
  nmax = 0;
}

void Grace2LFused::ensure_threads(int nth)
{
  if (nth < 1) nth = 1;
  if (nth == nthreads && (int) work.size() == nth) return;
  nthreads = nth;
  work.clear();
  for (int t = 0; t < nthreads; t++) work.emplace_back(new Work());
  nmax = 0;    // per-atom buffers are placed for nthreads (SpreadBuf first touch): rebuild on the next compute
}

double Grace2LFused::memory_bytes() const
{
  return (double) (I_glob.size() + gI_sum.size() + rho_glob.size() + l1_store.size()) * sizeof(float) +
         (double) grho_glob.size() * sizeof(double);
}

/* ----------------------------------------------------------------------
   bonds of the batch atoms: rhat, r, rcut_ij, Y_lm and dY/d(rhat) (polynomial extension)
------------------------------------------------------------------------- */

template<class Src> void Grace2LFused::bonds(Work &w, const Src &src, bool grad)
{
  const BondFilter f{cutsq_rows.data(), rcut_type.data(), ntypes + 1, w.cpl};
  const SHCoefs sh{lmax, nlm, sh_a.data(), sh_b.data(), sh_c.data(), sh_d.data()};
  BondGather<Src>::run(w, src, f, sh, grad);
}

/* ---------------------------------------------------------------------- */

template<class Src>
void Grace2LFused::compute(int inum, const int *ilist, int n_nodes, int nmax_nodes, const Src &src, Exchange &ex,
                           Outputs &out, const ComputeFlags &fl, int nthreads_req)
{
  if (ntypes == 0) throw Error("pair grace/2l/fused: setup_types() not called");
  if (fl.couple.eq && (fl.couple.eq < 1 || fl.couple.eq > 4 || !fl.couple.mask))
    throw Error("grace fused: two-group coupling needs eq 1..4 and the atom mask");
  const int nall = n_nodes;
  const int *type = src.type;
  const bool eflag_atom = fl.eflag_atom && out.eatom;
  vflag_atom = fl.vflag_atom && out.vatom;
  vatom = out.vatom;
  ensure_threads(nthreads_req);
#ifdef _OPENMP
  const int nth = nthreads;
#endif

  if (nmax_nodes > nmax) {
    nmax = 0;    // committed after the allocations: a failed one (std::bad_alloc) leaves the next call to grow again
    const size_t n = (size_t) nmax_nodes;
    I_glob.assign(n * nI * PR, 0.0f);
    gI_sum.assign(n * nI * PR, 0.0f);
    gI_lock.reset(n);
    rho_glob.assign(n * nrho, 0.0f);
    grho_glob.assign(n * nrho, 0.0);
    l1_store.clear();
    nmax = nmax_nodes;
  }
  const size_t l1_rows = (size_t) 2 * nlm + 2 * nAA;
  l1_stored = false;
  if (store_l1) {
    const double gb = (double) nmax * l1_rows * P1 * sizeof(float) / 1e9;
    if (gb <= store_l1_max_gb) {
      if (l1_store.size() != (size_t) nmax * l1_rows * P1) l1_store.assign((size_t) nmax * l1_rows * P1, 0.0f);
      l1_stored = true;
    } else if (!warned_store) {
      char msg[160];
      std::snprintf(msg, sizeof(msg), "pair grace/2l/fused: store_l1 would need %.1f GB, recomputing layer 1 instead", gb);
      log.warn(msg);
      warned_store = true;
    }
  }
  // per-atom row block of the layer-1 store <-> batch slot a of the workspace
  auto l1_copy = [&](Work &w, int a, bool save) {
    float *blk = l1_store.data() + (size_t) w.atom[a] * l1_rows * P1;
    float *bufs[4] = {w.A.data(), w.A1.data(), w.AA1.data(), w.AA2.data()};
    const int rows[4] = {nlm, nlm, nAA, nAA};
    for (int t = 0; t < 4; t++)
      for (int r = 0; r < rows[t]; r++) {
        float *ws = bufs[t] + ((size_t) r * NB + a) * P1;
        if (save) std::memcpy(blk, ws, P1 * sizeof(float));
        else std::memcpy(ws, blk, P1 * sizeof(float));
        blk += P1;
      }
  };
  // type-homogeneous batches of NB atoms
  if (ilist) order.assign(ilist, ilist + inum);
  else {    // nullptr: the central atoms are 0 .. inum-1 (CSR callers)
    order.resize(inum);
    for (int q = 0; q < inum; q++) order[q] = q;
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return type[a] < type[b]; });
  bstart.clear();
  bcount.clear();
  for (int q = 0; q < inum;) {
    if (type2mu[type[order[q]]] < 0)    // its per-type rows are not built (pair_style hybrid keeps it out of the list)
      throw Error(braces("pair grace/2l/fused: atom {} of unmapped type {} in the atom list", order[q], type[order[q]]));
    int n = 1;
    while (n < NB && q + n < inum && type[order[q + n]] == type[order[q]]) n++;
    bstart.push_back(q);
    bcount.push_back(n);
    q += n;
  }
  const int nbatch = bstart.size();

  // per-thread workspaces
  for (auto &wp : work) {
    Work &w = *wp;
    const size_t s1 = (size_t) NB * P1, s2 = (size_t) NB * P2, sr = (size_t) NB * PR;
    const int npair = yp_y.size();
    const size_t nlP = (size_t) nl * std::max({P1, P2, PR});
    if (w.A.size() != nlm * s1) {
      w.boff.assign(NB + 1, 0);
      w.R.assign(nlP, 0.0f); w.dR.assign(nlP, 0.0f);
      for (auto *v : {&w.A, &w.A1, &w.gA, &w.gA1}) v->assign(nlm * s1, 0.0f);
      for (auto *v : {&w.AA, &w.AA1, &w.AA2, &w.gAA1, &w.gAA2}) v->assign(nAA * s1, 0.0f);
      w.AAA.assign(rowred ? 0 : nAAA * s1, 0.0f);
      w.AAAA.assign(rowred ? 0 : nAAAA * s1, 0.0f);
      w.row.assign((size_t) NB * std::max({P1, PR, P2}), 0.0f);
      for (auto *v : {&w.O1, &w.gO1}) v->assign((size_t) nI1 * NB * PI1, 0.0f);
      for (auto *v : {&w.Rho, &w.gRho}) v->assign((size_t) NB * PRHO, 0.0f);
      for (auto *v : {&w.Iv, &w.gIb}) v->assign((size_t) nI * sr, 0.0f);
      w.B0.assign(nlm * sr, 0.0f);
      w.YIp.assign(npair * sr, 0.0f);
      w.YI.assign(nYI * sr, 0.0f);
      w.dIg.assign((size_t) yg_l.size() * PR, 0.0f);
      for (auto *v : {&w.B, &w.B1, &w.gB, &w.gB1}) v->assign(nB * s2, 0.0f);
      for (auto *v : {&w.BB, &w.BB1, &w.BB2, &w.gBB1, &w.gBB2}) v->assign(nBB * s2, 0.0f);
      w.BBB.assign(rowred ? 0 : nBBB * s2, 0.0f);
      w.BBBB.assign(rowred ? 0 : nBBBB * s2, 0.0f);
      for (auto *v : {&w.O2, &w.gO2}) v->assign((size_t) NB * PO2, 0.0f);
    }
    w.want_vir = fl.virial;
    w.cpl = fl.couple.eq ? &fl.couple : nullptr;
    w.dedl = 0.0;
    for (int k = 0; k < 6; k++) w.vir[k] = 0.0;
  }
  auto setup = [&](Work &w, int q) {
    w.nbat = bcount[q];
    for (int a = 0; a < w.nbat; a++) w.atom[a] = order[bstart[q] + a];
    w.ty = type[w.atom[0]];
  };

  uq_on = fl.uq;
  uq_out = &out;
  eonly = fl.energy_only;
  if (uq_on) {
    if (!uq_ready) throw Error("grace/2l/fused: UQ requested before enable_uq()");
    if (inum > 0 && !(out.gamma && out.atomic_sigma && out.gmm_cluster))    // no atoms: nothing written
      throw Error("grace/2l/fused: UQ requested without the gamma / atomic_sigma / gmm_cluster outputs");
    int imax = 0;
    for (int i : order) imax = std::max(imax, i + 1);
    uq_z.assign((size_t) imax * model->uq_rp_dim, 0.0);
    uq_n2l1.assign(imax, 0.0);
  }

  // P1: layer 1 forward -> I, rho
  double tph = now_s();
#pragma omp parallel num_threads(nth)
  {
#pragma omp for schedule(dynamic, 1) nowait
  for (int q = 0; q < nbatch; q++) {
#ifdef _OPENMP
    Work &w = *work[omp_get_thread_num()];
#else
    Work &w = *work[0];
#endif
    setup(w, q);
    { PROF(T_BOND); bonds(w, src, false); }
    layer1_forward(w);
    if (uq_on) uq_l1(w);
    for (int a = 0; a < w.nbat; a++) {
      const int i = w.atom[a];
      for (int fn = 0; fn < nI; fn++)
        std::memcpy(I_glob.data() + ((size_t) i * nI + fn) * PR, w.Iv.data() + ((size_t) fn * NB + a) * PR,
                    PR * sizeof(float));
      std::memcpy(rho_glob.data() + (size_t) i * nrho, w.Rho.data() + a * PRHO, nrho * sizeof(float));
      if (l1_stored && !eonly) l1_copy(w, a, true);
    }
  }
#ifdef _OPENMP
  if (uq_on) uq_flush(*work[omp_get_thread_num()], 1);
#else
  if (uq_on) uq_flush(*work[0], 1);
#endif
  }
  if (g_prof_on) { g_prof[T_P1] += now_s() - tph; tph = now_s(); }

  // P2: I to ghosts
  ex.forward(0);
  if (g_prof_on) { g_prof[T_P2] += now_s() - tph; tph = now_s(); }

  fthr.ensure(nthreads, nmax_nodes);

  // P3: layer 2 + energy + layer-2 reverse, adding dE/dI into the rows of gI_sum under their locks
  // static schedule: a thread's batches are contiguous in (spatially sorted) atom order, so two threads seldom add
  // into the same row at the same time
  if (!eonly) {
    const size_t W = (size_t) nI * PR;
#pragma omp parallel for schedule(static) num_threads(nth)
    for (int j = 0; j < nall; j++) vzero(gI_sum.data() + j * W, W);
  }
  double eng = 0.0;
#pragma omp parallel reduction(+ : eng) num_threads(nth)
  {
#pragma omp for schedule(static) nowait
  for (int q = 0; q < nbatch; q++) {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    Work &w = *work[tid];
    setup(w, q);
    w.ft = fthr.of(tid);
    { PROF(T_BOND); bonds(w, src, !eonly); }
    layer2(w);
    for (int a = 0; a < w.nbat; a++) {
      eng += w.E[a];
      if (eflag_atom) out.eatom[w.atom[a]] += w.E[a];
    }
  }
#ifdef _OPENMP
  if (uq_on) uq_flush(*work[omp_get_thread_num()], 2);
#else
  if (uq_on) uq_flush(*work[0], 2);
#endif
  }
  out.energy += eng;
  if (g_prof_on) { g_prof[T_P3] += now_s() - tph; tph = now_s(); }
  if (eonly) return;    // energy only: no layer-2 reverse, no reverse exchange, no layer-1 backward

  // P4: dE/dI of the ghosts to their owners
  ex.reverse(0);
  if (g_prof_on) { g_prof[T_P4] += now_s() - tph; tph = now_s(); }

  // P5: layer 1 recompute + reverse
#pragma omp parallel for schedule(dynamic, 1) num_threads(nth)
  for (int q = 0; q < nbatch; q++) {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    Work &w = *work[tid];
    setup(w, q);
    w.ft = fthr.of(tid);
    { PROF(T_BOND); bonds(w, src, true); }
    if (l1_stored)
      for (int a = 0; a < w.nbat; a++) l1_copy(w, a, false);
    else
      layer1_forward(w);
    layer1_backward(w);
  }
  if (g_prof_on) { g_prof[T_P5] += now_s() - tph; tph = now_s(); }
  if (nall > 0) fthr.reduce(out.f, nall);
  if (g_prof_on) g_prof[T_P6] += now_s() - tph;

  if (fl.virial)
    for (auto &wp : work)
      for (int k = 0; k < 6; k++) out.virial[k] += wp->vir[k];
  if (fl.couple.eq && !fl.energy_only)
    for (auto &wp : work) out.dedl += wp->dedl;
}

void Grace2LFused::enable_uq()
{
  const GRACE2LModel &m = *model;
  m.validate_uq();
  const int w1 = fused::uq_width(rRho), w2 = fused::uq_width(rI2);
  if (w1 + w2 != m.uq_d_basis)
    throw Error(braces("grace/2l/fused UQ: basis width {} + {} != uq_rp_matrix rows {}", w1, w2, m.uq_d_basis));
  if (m.uq_feature_dim != m.uq_rp_dim + m.uq_n_density || m.uq_n_density > 3)
    throw Error("grace/2l/fused UQ: inconsistent UQ head");
  uq_row_l1 = w2;
  uq_rp.assign(m.uq_rp_matrix.begin(), m.uq_rp_matrix.end());
  uq_ready = true;
}

// UQ (as grace/2l/kk): layer 1 streams the rho reduce sources into the raw projection after the P1 forward
void Grace2LFused::uq_l1(Work &w)
{
  const GRACE2LModel &m = *model;
  const float *X[4] = {w.A.data(), w.AA.data(), w.AAA.data(), w.AAAA.data()};
  if (rowred) {
    const size_t s1 = (size_t) NB * P1;
    if (w.u3.size() < nAAA * s1) w.u3.assign(nAAA * s1, 0.0f);
    if (w.u4.size() < nAAAA * s1) w.u4.assign(nAAAA * s1, 0.0f);
    { PROF(T_UQPROD); prod_fwd(pAAA, w.AA1.data(), w.A.data(), w.u3.data(), P1);
    prod_fwd(pAAAA, w.AA2.data(), w.AA2.data(), w.u4.data(), P1); }
    X[2] = w.u3.data();
    X[3] = w.u4.data();
  }
  for (int a = 0; a < w.nbat; a++) {
    { PROF(T_UQPUSH); w.uqq.push(rRho, X, a, w.atom[a], w.ty, m.uq_d_basis - uq_row_l1); }
    if (w.uqq.na == fused::UQ_MA) uq_flush(w, 1);
  }
}

// layer 2 (after the I2 reduce of the batch): I2 reduce sources, then normalisation, density [full, L2, L1], GMM
void Grace2LFused::uq_l2(Work &w)
{
  const float *X[4] = {w.B.data(), w.BB.data(), w.BBB.data(), w.BBBB.data()};
  if (rowred) {
    const size_t s2 = (size_t) NB * P2;
    if (w.u3.size() < nBBB * s2) w.u3.assign(nBBB * s2, 0.0f);
    if (w.u4.size() < nBBBB * s2) w.u4.assign(nBBBB * s2, 0.0f);
    { PROF(T_UQPROD); prod_fwd(pBBB, w.BB1.data(), w.B.data(), w.u3.data(), P2);
    prod_fwd(pBBBB, w.BB2.data(), w.BB2.data(), w.u4.data(), P2); }
    X[2] = w.u3.data();
    X[3] = w.u4.data();
  }
  for (int a = 0; a < w.nbat; a++) {
    { PROF(T_UQPUSH); w.uqq.push(rI2, X, a, w.atom[a], w.ty, uq_row_l1); }
    if (w.uqq.na == fused::UQ_MA) uq_flush(w, 2);
  }
}

// projection of the queued atoms; layer 1: keep it (and ||b_L1||^2) per atom; layer 2: add the layer-1 part,
// normalise, density [full, L2, L1], GMM
void Grace2LFused::uq_flush(Work &w, int layer)
{
  fused::UQQueue &q = w.uqq;
  if (q.na == 0) return;
  const GRACE2LModel &m = *model;
  const int RP = m.uq_rp_dim, D = m.uq_feature_dim;
  if (layer == 1) {
    { PROF(T_UQGEMM); fused::uq_gemm(q, m.uq_d_basis - uq_row_l1, uq_rp.data() + (size_t) uq_row_l1 * RP, RP); }
    for (int c = 0; c < q.na; c++) {
      const int i = q.atom[c];
      double *z = uq_z.data() + (size_t) i * RP;
      for (int d = 0; d < RP; d++) z[d] = q.Z[(size_t) c * RP + d];
      uq_n2l1[i] = q.n2[c];
    }
  } else {
    Outputs &out = *uq_out;
    { PROF(T_UQGEMM); fused::uq_gemm(q, uq_row_l1, uq_rp.data(), RP); }
    if ((int) w.uqf.size() < D) { w.uqf.assign(D, 0.0); w.uqd.assign(D, 0.0); }
    for (int c = 0; c < q.na; c++) {
      const int i = q.atom[c];
      double *f = w.uqf.data();
      const double *z1 = uq_z.data() + (size_t) i * RP;
      for (int d = 0; d < RP; d++) f[d] = z1[d] + q.Z[(size_t) c * RP + d];
      const double n2l2 = q.n2[c], nrm = std::sqrt(uq_n2l1[i] + n2l2);
      const double dens[3] = {nrm, std::sqrt(n2l2), std::sqrt(uq_n2l1[i])};
      { PROF(T_UQGMM); fused::uq_gmm(m, type2mu[q.ty[c]], f, w.uqd.data(), nrm, dens, out.gamma[i],
                                     out.atomic_sigma[i], out.gmm_cluster[i]); }
    }
  }
  q.na = 0;
}

template void Grace2LFused::compute<LammpsBonds>(int, const int *, int, int, const LammpsBonds &, Exchange &, Outputs &,
                                                 const ComputeFlags &, int);
template void Grace2LFused::compute<CsrBonds>(int, const int *, int, int, const CsrBonds &, Exchange &, Outputs &,
                                              const ComputeFlags &, int);

}    // namespace grace

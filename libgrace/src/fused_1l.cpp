// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Contributing author: Yury Lysogorskiy (ICAMS). Input through the bond source, results through Outputs, errors
// through grace::Error and the Logger.

#include "grace/fused_1l.h"

#include "grace/bonds.h"
#include "grace/fused_cpu.h"
#include "grace/type_map.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace grace {

using namespace fused;

namespace {
// "{}" placeholders filled in order
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

struct Grace1LFused::Work {
  int nbat = 0, ty = 0;
  int zero_i = -1, zero_j = -1;    // the first bond of length 0 its gathers met (throw_zero_bond)
  int atom[NB];
  double E[NB];
  std::vector<int> boff, bj, bt;
  std::vector<double> rh, rr, rc;
  fvec Y, G;
  fvec R, dR, Rb;
  fvec row;    // one product row [NB][PF] (fused product + reduction)
  std::vector<const float *> pz;
  // [function][NB][channel]; the AA, AAA, AAAA buffers hold their gradients in the reverse
  fvec A, A1, AA, AA1, AAA, AA2, AAAA, Rho;
  fvec gA, gA1, gAA1, gAA2, gRho;
  fvec uA3, uA4;                  // UQ with rowred: AAA, AAAA materialised
  fused::UQQueue uqq;             // UQ: atoms waiting for the projection
  std::vector<double> uqf, uqd;   // UQ feature, Mahalanobis delta
  double *ft = nullptr;    // this thread's force rows
  double vir[6] = {0, 0, 0, 0, 0, 0};             // CsrBonds: this thread\'s bond virial
  bool want_vir = false;
  const Coupling *cpl = nullptr;   // two-group coupling (ComputeFlags::couple), nullptr: off
  std::vector<double> cp;           // coupling: per bond true r, dr'/dr, dr'/dlambda (bonds.h)
  double dedl = 0.0;                // coupling: this thread's dE/dlambda
};

/* ---------------------------------------------------------------------- */















/* ----------------------------------------------------------------------
   model -> CPU layout
------------------------------------------------------------------------- */




void Grace1LFused::build_splines()
{
  const GRACE1LModel &m = *model;
  const int N = spline_n;
  const double h = 1.0 / N;
  const auto &layers = m.mlp_rad_layers;
  const int nout = layers.back().n_out;
  if (nout != CA * nl) throw Error("GRACE-1L: radial MLP output width mismatch");
  std::vector<double> fv((size_t) (N + 1) * nout), dv((size_t) (N + 1) * nout), g, dg, out, dout;
  for (int k = 0; k <= N; k++) {
    radial_basis_x(k * h, m.radial_basis_p, m.nradbase, g, dg);
    mlp_x(layers, g, dg, out, dout);
    for (int o = 0; o < nout; o++) {
      fv[(size_t) k * nout + o] = out[o];
      dv[(size_t) k * nout + o] = h * dout[o];
    }
  }
  splR.assign((size_t) N * 4 * nl * PA, 0.0f);
  for (int k = 0; k < N; k++)
    for (int n = 0; n < CA; n++)
      for (int l = 0; l < nl; l++) {
        const int o = n * nl + l;
        const double f0 = fv[(size_t) k * nout + o], f1 = fv[(size_t) (k + 1) * nout + o];
        const double d0 = dv[(size_t) k * nout + o], d1 = dv[(size_t) (k + 1) * nout + o];
        const double c[4] = {f0, d0, -3.0 * f0 - 2.0 * d0 + 3.0 * f1 - d1, 2.0 * f0 + d0 - 2.0 * f1 + d1};
        for (int q = 0; q < 4; q++) splR[(((size_t) k * 4 + q) * nl + l) * PA + n] = (float) c[q];
      }
}

void Grace1LFused::build_model()
{
  const GRACE1LModel &m = *model;
  const Ctx cx{ntypes, &type2mu, m.n_elements};
  lmax = m.lmax;
  if (lmax > MAX_LMAX) throw Error(braces("GRACE-1L: lmax > {} not supported", MAX_LMAX));
  nl = lmax + 1;
  nlm = nl * nl;
  CA = m.nradmax;
  CF = m.fc_A1.n_out;
  if (m.fc_AA1.n_out != CF || m.fc_AA2.n_out != CF) throw Error("GRACE-1L: FC widths differ");
  PA = pad16(CA);
  PF = pad16(CF);
  nrho = m.rho_n_out;
  PRHO = pad16(nrho);

  rcut_type.assign((size_t) (ntypes + 1) * (ntypes + 1), 0.0);
  for (int a = 1; a <= ntypes; a++)
    for (int b = 1; b <= ntypes; b++) rcut_type[a * (ntypes + 1) + b] = pair_cutoff(m, type2mu[a], type2mu[b]);
  zA.assign((size_t) (ntypes + 1) * PA, 0.0f);
  shift_type.assign(ntypes + 1, 0.0);
  for (int t = 1; t <= ntypes; t++) {
    const int mu = type2mu[t];
    if (mu < 0) continue;    // unmapped: zero rows, never read
    for (int n = 0; n < CA; n++) {
      double v = 0.0;
      for (int e = 0; e < m.embedding_size; e++)
        v += m.A_lin_transform_W[e * CA + n] * m.chem_embedding[mu * m.embedding_size + e];
      zA[t * PA + n] = (float) (v * m.A_lin_transform_norm * m.A_inv_avg_n_neigh);
    }
    shift_type[t] = mu < (int) m.shift_values.size() ? m.shift_values[mu] : 0.0;
  }
  sh_coefs(lmax, sh_a, sh_b, sh_c, sh_d);
  build_splines();

  build_prod(pAA, m.prod_AA);
  build_prod(pAAA, m.prod_AAA);
  build_prod(pAAAA, m.prod_AAAA);
  nAA = pAA.nout;
  nAAA = pAAA.nout;
  nAAAA = pAAAA.nout;
  build_red<GRACE1LModel::ReduceWeights>(rRho, {&m.reduce_A, &m.reduce_AA, &m.reduce_AAA, &m.reduce_AAAA}, {0, 0, 0, 0}, nrho, 1, true, PRHO,
            {PA, PF, PF, PF}, cx);

  // liveness (backwards from rho): CG terms and FC outputs never read are not computed
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
  std::vector<char> nAAAv(nAAA, 0), nAAAAv(nAAAA, 0), nAAv(nAA, 0);
  ci_of(rRho, 2, nAAAv);
  ci_of(rRho, 3, nAAAAv);
  prune(pAAA, nAAAv);
  prune(pAAAA, nAAAAv);
  const auto liveAA1 = operands(pAAA, nAA, true, false);
  const auto liveAA2 = operands(pAAAA, nAA, true, true);
  for (int q = 0; q < nAA; q++) nAAv[q] = liveAA1[q] || liveAA2[q];
  ci_of(rRho, 1, nAAv);
  prune(pAA, nAAv);
  auto liveA1 = operands(pAA, nlm, true, true);
  const auto liveA1r = operands(pAAA, nlm, false, true);
  for (int q = 0; q < nlm; q++) liveA1[q] = liveA1[q] || liveA1r[q];
  build_fc(fcA1, m.fc_A1, CA, CA, PA, PA, PF, liveA1, cx);
  build_fc(fcAA1, m.fc_AA1, CF, CA, PF, PA, PF, liveAA1, cx);
  build_fc(fcAA2, m.fc_AA2, CF, CA, PF, PA, PF, liveAA2, cx);
  if (fcA1.nfl != nlm || fcAA1.nfl != nAA || fcAA2.nfl != nAA)
    throw Error("GRACE-1L: unexpected function counts");
  rowred = !getenv("G2LF_NOROWRED");
  if (rowred) {
    build_rowred(rrRho_3, rRho, 2, nAAA);
    build_rowred(rrRho_4, rRho, 3, nAAAA);
  }

  eW.clear(); e_norm.clear(); e_nin.clear(); e_nout.clear();
  for (auto &Lr : m.energy_mlp_layers) {
    eW.push_back(Lr.W);
    e_norm.push_back(Lr.norm);
    e_nin.push_back(Lr.n_in);
    e_nout.push_back(Lr.n_out);
  }
  if (e_nin.empty() || e_nin[0] != nrho - 1 || e_nout.back() != 1)
    throw Error("GRACE-1L: unexpected energy MLP shape");
  // energy_atom keeps the activations in fixed stack buffers act/pre[8][256] (it runs inside the parallel region,
  // where it cannot throw): at most 7 layers and every width at most 256
  if (e_nin.size() > 7) throw Error(braces("GRACE-1L: energy MLP has {} layers, at most 7 supported", e_nin.size()));
  for (size_t L = 0; L < e_nin.size(); L++)
    if (e_nin[L] > 256 || e_nout[L] > 256 || e_nin[L] < 1 || e_nout[L] < 1 || L + 1 < e_nin.size() && e_nout[L] != e_nin[L + 1] ||
        eW[L].size() != (size_t) e_nin[L] * e_nout[L])
      throw Error(braces("GRACE-1L: energy MLP layer {} is {} x {} (widths 1..256, consecutive layers must match)",
                         L, e_nin[L], e_nout[L]));
  e_act = m.energy_mlp_activation;
  out_scale = m.output_scale;
  log.info(braces("[GRACE-1L] lmax {} | A {} ch, FC {} ch | live A1 {}/{} AA {}/{} AA1 {}/{} AAA {}/{} AA2 {}/{} "
                   "AAAA {}/{} | rho {} | spline N {}",
                   lmax, CA, CF, count(liveA1), nlm, count(nAAv), nAA, count(liveAA1), nAA, count(nAAAv), nAAA,
                   count(liveAA2), nAA, count(nAAAAv), nAAAA, nrho, spline_n));
}


/* ----------------------------------------------------------------------
   bonds of the batch atoms: rhat, r, rcut_ij, Y_lm and dY/d(rhat) (polynomial extension)
------------------------------------------------------------------------- */



/* ----------------------------------------------------------------------
   forward of a batch (bonds gathered): A .. AAAA, rho
------------------------------------------------------------------------- */

void Grace1LFused::forward(Work &w)
{
  const int ty = w.ty, RLA = NB * PA;
  float *A = w.A.data();
  vzero(A, (size_t) nlm * RLA);
  {
    PROF(T_RAD1);
    for (int a = 0; a < w.nbat; a++) {
      const int b0 = w.boff[a], nb = w.boff[a + 1] - b0;
      if ((int) w.Rb.size() < nb * nl * PA) w.Rb.resize((size_t) nb * nl * PA);
      if ((int) w.pz.size() < nb) w.pz.resize(nb);
      for (int b = 0; b < nb; b++) {
        spline_eval(splR.data(), spline_n, nl, PA, w.rr[b0 + b] / w.rc[b0 + b], w.Rb.data() + (size_t) b * nl * PA,
                    nullptr);
        w.pz[b] = zA.data() + (size_t) w.bt[b0 + b] * PA;
      }
      G2LF_NVD(PA, atom_scatter<NV>(nb, nl, nlm, w.Rb.data(), w.pz.data(), w.Y.data() + (size_t) b0 * nlm, A + a * PA,
                                    RLA))
    }
  }
  { PROF(T_FC1); fc_fwd(fcA1, A, A, w.A1.data()); }
  { PROF(T_PROD1); prod_fwd(pAA, w.A1.data(), w.A1.data(), w.AA.data(), PF); }
  { PROF(T_FC1); fc_fwd(fcAA1, w.AA.data(), A, w.AA1.data()); }
  { PROF(T_FC1); fc_fwd(fcAA2, w.AA.data(), A, w.AA2.data()); }
  if (rowred) {
    const float *X[4] = {A, w.AA.data(), nullptr, nullptr};
    { PROF(T_RED1); red_fwd(rRho, ty, X, w.Rho.data()); }
    const Red *reds[1] = {&rRho};
    float *outs[1] = {w.Rho.data()};
    { const RowRed *rrs[1] = {&rrRho_3};
      PROF(T_PROD1); prod_red_fwd(pAAA, w.AA1.data(), w.A1.data(), PF, ty, reds, rrs, 1, outs, w.row.data()); }
    { const RowRed *rrs[1] = {&rrRho_4};
      PROF(T_PROD1); prod_red_fwd(pAAAA, w.AA2.data(), w.AA2.data(), PF, ty, reds, rrs, 1, outs, w.row.data()); }
  } else {
    { PROF(T_PROD1); prod_fwd(pAAA, w.AA1.data(), w.A1.data(), w.AAA.data(), PF); }
    { PROF(T_PROD1); prod_fwd(pAAAA, w.AA2.data(), w.AA2.data(), w.AAAA.data(), PF); }
    const float *X[4] = {A, w.AA.data(), w.AAA.data(), w.AAAA.data()};
    { PROF(T_RED1); red_fwd(rRho, ty, X, w.Rho.data()); }
  }
}

/* ----------------------------------------------------------------------
   reverse of a batch from dE/drho (w.gRho): forces
------------------------------------------------------------------------- */

void Grace1LFused::backward(Work &w)
{
  const int ty = w.ty, RLA = NB * PA, RLF = NB * PF;
  // AA, AAA, AAAA values are not needed by the reverse: their buffers hold the gradients
  float *gA = w.gA.data(), *gA1 = w.gA1.data(), *gAA = w.AA.data(), *gAAA = w.AAA.data(), *gAAAA = w.AAAA.data();
  vzero(gA, (size_t) nlm * RLA);
  vzero(gA1, (size_t) nlm * RLF);
  vzero(gAA, (size_t) nAA * RLF);
  vzero(w.gAA1.data(), (size_t) nAA * RLF);
  vzero(w.gAA2.data(), (size_t) nAA * RLF);
  if (!rowred) {
    vzero(gAAA, (size_t) nAAA * RLF);
    vzero(gAAAA, (size_t) nAAAA * RLF);
  }
  {
    float *gX[4] = {gA, gAA, gAAA, gAAAA};
    PROF(T_RED1B);
    red_bwd(rRho, ty, w.gRho.data(), gX);
  }
  const Red *reds[1] = {&rRho};
  const float *gouts[1] = {w.gRho.data()};
  if (rowred) {
    const RowRed *rrs[1] = {&rrRho_4};
    PROF(T_PROD1B);
    prod_red_bwd(pAAAA, w.AA2.data(), w.AA2.data(), PF, ty, reds, rrs, 1, gouts, w.gAA2.data(), w.gAA2.data(),
                 w.row.data());
  } else {
    PROF(T_PROD1B);
    prod_bwd(pAAAA, w.AA2.data(), w.AA2.data(), gAAAA, w.gAA2.data(), w.gAA2.data(), PF);
  }
  { PROF(T_FC1B); fc_bwd(fcAA2, w.gAA2.data(), gAA, gA); }
  if (rowred) {
    const RowRed *rrs[1] = {&rrRho_3};
    PROF(T_PROD1B);
    prod_red_bwd(pAAA, w.AA1.data(), w.A1.data(), PF, ty, reds, rrs, 1, gouts, w.gAA1.data(), gA1, w.row.data());
  } else {
    PROF(T_PROD1B);
    prod_bwd(pAAA, w.AA1.data(), w.A1.data(), gAAA, w.gAA1.data(), gA1, PF);
  }
  { PROF(T_FC1B); fc_bwd(fcAA1, w.gAA1.data(), gAA, gA); }
  { PROF(T_PROD1B); prod_bwd(pAA, w.A1.data(), w.A1.data(), gAA, gA1, gA1, PF); }
  { PROF(T_FC1B); fc_bwd(fcA1, gA1, gA, gA); }

  PROF(T_RAD1B);
  v16 gRv[(MAX_LMAX + 1) * 8], gYv[MAX_NLM];
  for (int a = 0; a < w.nbat; a++) {
    const int i = w.atom[a];
    for (int b = w.boff[a]; b < w.boff[a + 1]; b++) {
      spline_eval(splR.data(), spline_n, nl, PA, w.rr[b] / w.rc[b], w.R.data(), w.dR.data());
      const float *z = zA.data() + (size_t) w.bt[b] * PA;
      const float *Y = w.Y.data() + (size_t) b * nlm;
      for (int q = 0; q < nlm; q++) gYv[q] = v16{};
      for (int q = 0; q < nl * (PA / 16); q++) gRv[q] = v16{};
      double dEdx = 0.0;
      G2LF_NVD(PA, bond_scatter_bwd_t<NV>(w.R.data(), z, Y, nl, gA + a * PA, RLA, gYv, gRv);
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
   E_i = (MLP(rho[1:]) + rho[0]) * output_scale + shift[mu_i]  (fp64) and dE/drho
------------------------------------------------------------------------- */

double Grace1LFused::energy_atom(const float *rho, int ty, float *grho) const
{
  constexpr int MAXW = 256;
  const int nlay = eW.size();
  double act[8][MAXW], pre[8][MAXW];
  for (int k = 1; k < nrho; k++) act[0][k - 1] = rho[k];
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
  const double Ei = (act[nlay][0] + rho[0]) * out_scale + shift_type[ty];
  if (!grho) return Ei;
  double g[MAXW], gn[MAXW];
  g[0] = out_scale;
  for (int L = nlay - 1; L >= 0; L--) {
    if (L < nlay - 1)
      for (int c = 0; c < e_nout[L]; c++) {
        const double v = pre[L][c];
        if (e_act == 1) {
          const double t = act[L + 1][c];
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
  grho[0] = (float) out_scale;
  for (int k = 1; k < nrho; k++) grho[k] = (float) g[k - 1];
  return Ei;
}

Grace1LFused::Grace1LFused(std::shared_ptr<const GRACE1LModel> m, FusedSettings s, Logger log_) :
    log(log_), model(std::move(m))
{
  spline_n = s.spline_n;
  if (spline_n < 16) throw Error("GRACE-1L: spline N must be >= 16");
  g_prof_on = getenv("G2LF_PROF") != nullptr;
}

Grace1LFused::~Grace1LFused() = default;

void Grace1LFused::setup_types(int ntypes_, const std::vector<int> &type2mu_)
{
  ntypes = ntypes_;
  type2mu = type2mu_;
  check_type2mu(type2mu, ntypes, model->n_elements, "GRACE-1L");
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
}

void Grace1LFused::ensure_threads(int nth)
{
  if (nth < 1) nth = 1;
  if (nth == nthreads && (int) work.size() == nth) return;
  nthreads = nth;
  work.clear();
  for (int t = 0; t < nthreads; t++) work.emplace_back(new Work());
}

template<class Src> void Grace1LFused::bonds(Work &w, const Src &src, bool grad)
{
  const BondFilter f{cutsq_rows.data(), rcut_type.data(), ntypes + 1, w.cpl};
  const SHCoefs sh{lmax, nlm, sh_a.data(), sh_b.data(), sh_c.data(), sh_d.data()};
  BondGather<Src>::run(w, src, f, sh, grad);
}

template<class Src>
void Grace1LFused::compute(int inum, const int *ilist, int n_nodes, int nmax_nodes, const Src &src, Outputs &out,
                           const ComputeFlags &fl, int nthreads_req)
{
  if (ntypes == 0) throw Error("GRACE-1L: setup_types() not called");
  if (fl.couple.eq && (fl.couple.eq < 1 || fl.couple.eq > 4 || !fl.couple.mask))
    throw Error("GRACE-1L: two-group coupling needs eq 1..4 and the atom mask");
  const int *type = src.type;
  const bool eflag_atom = fl.eflag_atom && out.eatom;
  vflag_atom = fl.vflag_atom && out.vatom;
  vatom = out.vatom;
  ensure_threads(nthreads_req);
#ifdef _OPENMP
  const int nth = nthreads;
#endif

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
      throw Error(braces("GRACE-1L: atom {} of unmapped type {} in the atom list", order[q], type[order[q]]));
    int n = 1;
    while (n < NB && q + n < inum && type[order[q + n]] == type[order[q]]) n++;
    bstart.push_back(q);
    bcount.push_back(n);
    q += n;
  }
  const int nbatch = bstart.size();

  for (auto &wp : work) {
    Work &w = *wp;
    const size_t sa = (size_t) NB * PA, sf = (size_t) NB * PF;
    if (w.A.size() != nlm * sa) {
      w.boff.assign(NB + 1, 0);
      w.R.assign((size_t) nl * PA, 0.0f);
      w.dR.assign((size_t) nl * PA, 0.0f);
      for (auto *v : {&w.A, &w.gA}) v->assign(nlm * sa, 0.0f);
      for (auto *v : {&w.A1, &w.gA1}) v->assign(nlm * sf, 0.0f);
      for (auto *v : {&w.AA, &w.AA1, &w.AA2, &w.gAA1, &w.gAA2}) v->assign(nAA * sf, 0.0f);
      w.AAA.assign(rowred ? 0 : nAAA * sf, 0.0f);
      w.AAAA.assign(rowred ? 0 : nAAAA * sf, 0.0f);
      w.row.assign((size_t) NB * PF, 0.0f);
      for (auto *v : {&w.Rho, &w.gRho}) v->assign((size_t) NB * PRHO, 0.0f);
    }
    w.want_vir = fl.virial;
    w.cpl = fl.couple.eq ? &fl.couple : nullptr;
    w.dedl = 0.0;
    for (int k = 0; k < 6; k++) w.vir[k] = 0.0;
  }

  const double t0 = now_s();
  const int nall = n_nodes;
  const bool uq_on = fl.uq;
  const bool eonly = fl.energy_only;    // no backward pass: no forces, no virials
  if (uq_on && !uq_ready) throw Error("GRACE-1L: UQ requested before enable_uq()");
  if (uq_on && inum > 0 && !(out.gamma && out.atomic_sigma && out.gmm_cluster))    // no atoms: nothing written
    throw Error("GRACE-1L: UQ requested without the gamma / atomic_sigma / gmm_cluster outputs");
  if (!eonly) fthr.ensure(nthreads, nmax_nodes);
  double eng = 0.0;
#pragma omp parallel reduction(+ : eng) num_threads(nth)
  {
#pragma omp for schedule(dynamic, 1) nowait
  for (int q = 0; q < nbatch; q++) {
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    Work &w = *work[tid];
    w.ft = eonly ? nullptr : fthr.of(tid);
    w.nbat = bcount[q];
    for (int a = 0; a < w.nbat; a++) w.atom[a] = order[bstart[q] + a];
    w.ty = type[w.atom[0]];
    { PROF(T_BOND); bonds(w, src, !eonly); }
    forward(w);
    if (uq_on) uq_batch(w, out);
    if (!eonly) vzero(w.gRho.data(), (size_t) NB * PRHO);
    {
      PROF(T_EN);
      for (int a = 0; a < w.nbat; a++) {
        const double Ei = energy_atom(w.Rho.data() + a * PRHO, w.ty, eonly ? nullptr : w.gRho.data() + a * PRHO);
        eng += Ei;
        if (eflag_atom) out.eatom[w.atom[a]] += Ei;
      }
    }
    if (!eonly) backward(w);
  }
#ifdef _OPENMP
  if (uq_on) uq_flush(*work[omp_get_thread_num()], out);
#else
  if (uq_on) uq_flush(*work[0], out);
#endif
  }
  throw_zero_bond(work);
  if (nall > 0 && !eonly) fthr.reduce(out.f, nall);
  out.energy += eng;
  if (g_prof_on) g_prof[T_P3] += now_s() - t0;
  if (fl.virial && !eonly)
    for (auto &wp : work)
      for (int k = 0; k < 6; k++) out.virial[k] += wp->vir[k];
  if (fl.couple.eq && !eonly)
    for (auto &wp : work) out.dedl += wp->dedl;
}

void Grace1LFused::enable_uq()
{
  const GRACE1LModel &m = *model;
  m.validate_uq();
  if (fused::uq_width(rRho) != m.uq_d_basis)
    throw Error(braces("GRACE-1L UQ: basis width {} != uq_rp_matrix rows {}", fused::uq_width(rRho), m.uq_d_basis));
  if (m.uq_feature_dim != m.uq_rp_dim + m.uq_n_density) throw Error("GRACE-1L UQ: inconsistent UQ head");
  uq_rp.assign(m.uq_rp_matrix.begin(), m.uq_rp_matrix.end());
  uq_ready = true;
}

// UQ of a batch after the forward (as Grace1LKK): one invariant block, the rho reduce sources
void Grace1LFused::uq_batch(Work &w, Outputs &out)
{
  const GRACE1LModel &m = *model;
  const float *X[4] = {w.A.data(), w.AA.data(), w.AAA.data(), w.AAAA.data()};
  if (rowred) {
    const size_t sf = (size_t) NB * PF;
    if (w.uA3.size() < nAAA * sf) w.uA3.assign(nAAA * sf, 0.0f);
    if (w.uA4.size() < nAAAA * sf) w.uA4.assign(nAAAA * sf, 0.0f);
    { PROF(T_UQPROD); prod_fwd(pAAA, w.AA1.data(), w.A1.data(), w.uA3.data(), PF);
    prod_fwd(pAAAA, w.AA2.data(), w.AA2.data(), w.uA4.data(), PF); }
    X[2] = w.uA3.data();
    X[3] = w.uA4.data();
  }
  for (int a = 0; a < w.nbat; a++) {
    { PROF(T_UQPUSH); w.uqq.push(rRho, X, a, w.atom[a], w.ty, m.uq_d_basis); }
    if (w.uqq.na == fused::UQ_MA) uq_flush(w, out);
  }
}

// projection of the queued atoms, normalisation, density channels (one block: all equal ||b||), GMM
void Grace1LFused::uq_flush(Work &w, Outputs &out)
{
  fused::UQQueue &q = w.uqq;
  if (q.na == 0) return;
  const GRACE1LModel &m = *model;
  const int RP = m.uq_rp_dim, D = m.uq_feature_dim;
  { PROF(T_UQGEMM); fused::uq_gemm(q, m.uq_d_basis, uq_rp.data(), RP); }
  if ((int) w.uqf.size() < D) { w.uqf.assign(D, 0.0); w.uqd.assign(D, 0.0); }
  std::vector<double> dens(m.uq_n_density);
  for (int c = 0; c < q.na; c++) {
    double *f = w.uqf.data();
    for (int d = 0; d < RP; d++) f[d] = q.Z[(size_t) c * RP + d];
    const double nrm = std::sqrt(q.n2[c]);
    std::fill(dens.begin(), dens.end(), nrm);
    const int i = q.atom[c];
    { PROF(T_UQGMM); fused::uq_gmm(m, type2mu[q.ty[c]], f, w.uqd.data(), nrm, dens.data(), out.gamma[i],
                                   out.atomic_sigma[i], out.gmm_cluster[i]); }
  }
  q.na = 0;
}

template void Grace1LFused::compute<LammpsBonds>(int, const int *, int, int, const LammpsBonds &, Outputs &,
                                                 const ComputeFlags &, int);
template void Grace1LFused::compute<CsrBonds>(int, const int *, int, int, const CsrBonds &, Outputs &,
                                              const ComputeFlags &, int);

}    // namespace grace

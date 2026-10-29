// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Contributing author: Yury Lysogorskiy (ICAMS). Input through the bond source (BondGather<Src>), the messages of
// the two layers through Exchange, results through Outputs.
#include "grace/fused_3l.h"

#include "grace/bonds.h"
#include "grace/cpu_gemm.h"
#include "grace/fused_3l_cpu.h"
#include "grace/fused_cpu.h"
#include "grace/type_map.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace grace {

using namespace f3l;


namespace {

// block tensors of one layer
enum { T_A, T_X, T_L2, T_R2p, T_P2, T_R2, T_FA, T_L3, T_R3p, T_P3, T_R3, T_FB, T_L4, T_R4p, T_P4, T_R4, T_EQ, T_I,
       T_RHO, T_RHON, T_NT };
// forward tensors the reverse pass reads (store_fwd): norm inputs and cp_l projections
constexpr int STORED[] = {T_EQ, T_RHO, T_L2, T_R2p, T_L3, T_R3p, T_L4, T_R4p};

struct Prod {
  Contract U, V;
  Couple cp;
  int tl = 0, tr = 0, tlp = 0, trp = 0;
  // fc_fold: the left input (both inputs of a self product) is an FC output; its projections are taken from the
  // FC inputs instead (UL / VL on the FC's left input fl, UR / VR on its right input fr)
  bool fold = false, self = false;
  int fl = 0, fr = 0;
  Contract UL, UR, VL, VR;
};
struct Red {
  std::vector<Contract> ins;
  std::vector<int> role;
  std::vector<float> nm;    // per output function, empty: none
  std::vector<std::vector<int>> ci;    // per instruction: collect_ind in model order (UQ feature walk)
  int out = 0;
  bool elem = false;
};
struct FCL {
  Contract right;
  std::vector<float> nof;
  int tleft = 0, tright = 0, tout = 0;
};

// phase timers (G2LF_PROF)
enum { P_S1, P_S2, P_S3, P_S4, P_S5, P_S6, P_S7, P_S8, P_S9, P_FRED, P_N };
const char *const P_NAME[P_N] = {"S1 L1 fwd", "S2 comm I1", "S3 L2 fwd", "S4 comm I2", "S5 L3 fwd+bwd",
                                 "S6 comm dI2", "S7 L2 fwd+bwd", "S8 comm dI1", "S9 L1 fwd+bwd", "force reduce"};
double g3_prof[P_N] = {};
// categories, OpenMP thread 0 only, per layer
enum { C_BOND, C_RAD, C_SPBF, C_CHAIN, C_CHAINB, C_SPBFB, C_B_DP, C_B_VEC, C_B_VMM, C_B_BOND, C_N };
const char *const C_NAME[C_N] = {"bonds+Y", "radial", "SPBF fwd", "chain fwd", "chain bwd", "SPBF bwd+force",
                                 " .. dP (CG^T)", " .. bond vectors", " .. vmm", " .. bond loops"};
double g3_cat[3][C_N] = {};
// chain operations, thread 0 only: reduces (incl. the fused couple), cp_l projections, FCs
enum { O_RED, O_PROJ, O_FC, O_N };
const char *const O_NAME[O_N] = {"reduce", "cp_l project", "FC"};
double g3_op[2][O_N] = {};
double g3_couple_tot = 0.0;
struct OpT {
  int f, o;
  double t;
  OpT(int f_, int o_) : f(f_), o(o_), t(f3l::prof_on ? prof_now() : 0.0) {}
  ~OpT()
  {
    if (f3l::prof_on) g3_op[f][o] += prof_now() - t;
  }
};
bool g3_prof_on = false;

inline int thread_id()
{
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

// string from its parts (messages)
template<class... A> std::string str(A &&...a)
{
  std::ostringstream o;
  (o << ... << a);
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

struct Grace3LFused::Net {
  int nsp = 0;
  std::vector<int> sp_of_type, mu_of_sp;
  int lmax = 0, nl = 0, nlm = 0;
  std::vector<double> sh_a, sh_b, sh_c, sh_d, rcut_type;
  struct Layer {
    bool equiv = false;
    int C = 0;                  // radial channels
    std::vector<int> lt;        // [nlm] l of each Y function
    int nfA = 0, nlmi = 0;
    std::vector<int> cg_off, cg_p, cgt_off, cgt_f;
    std::vector<float> cg_v, cgt_v;
    fvec zproj;                 // [nsp][C]
    std::vector<float> l0mask;  // [nlmi]
    hvec spl;                   // [pair][N][4][nl][C]
    int N = 0;                  // spline intervals of this layer
    size_t splpair = 0;
    // radial MLP by GEMM (tables above spline_mb): layer weights with norms, species terms of layer 0
    bool use_mlp = false;
    int nrb = 0, d1 = 0, d2 = 0, d3 = 0, penv = 0;
    double inv_avg = 1.0;
    fvec W0c, Ei, Ej, W1, b1, W2, fac;    // [nrb][d1], [nsp][d1] x2, [d1][d2], [d2], [d2][d3], [nsp][C]
    bool has_redA = false, has_eq = false;
    Red redA, r2, r3, r4, eq, rho;
    Prod p2, p3, p4;
    FCL fa, fb;
    EqNorm eqn;
    RMSNorm rhon;
    int nf[T_NT] = {}, nc[T_NT] = {};
  } L[3];
  int e_nin = 0, e_nh = 0, e_act = 0;
  std::vector<double> eW0, eW1;    // readout in fp64, as grace/{1,2}l/fused
  double en0 = 1.0, en1 = 1.0;
  double oscale = 1.0;
  std::vector<double> shift_sp;
  size_t maxten = 0, maxP = 0;
  size_t tmax[T_NT] = {};    // per tensor: max over layers of nf * nc
  int folded[3] = {0, 0, 0};    // fc_fold: products per layer whose FC is folded into the projections (0..2)
  // UQ: first row of the projection matrix for the rho1 / rho2 / rho3 basis blocks (and total width in uq_off[3])
  bool has_uq = false;
  int uq_off[4] = {0, 0, 0, 0};
  std::vector<float> uq_rp;    // fp32 projection matrix [d_basis][rp_dim]
};

struct Grace3LFused::Glob {
  // per-atom arrays shared by all threads: pages first touched by the threads (NUMA)
  using sbuf = fused::SpreadBuf<float, AlignHuge<float>>;
  sbuf I1, I2, dI1, dI2, rho1, rho2, dRn;
  fused::SpreadBuf<float> uqz;      // UQ: per local atom raw projection [RP] carried over layers (fp32)
  fused::SpreadBuf<double> uqn2;    // block norms^2 [3]
  // store_fwd: per atom the forward tensors the reverse of layers 1 and 2 needs (norm inputs, cp_l projections)
  sbuf st[2];
  int stw[2] = {0, 0};
  int w1 = 0, w2 = 0, wr = 0;
  fused::RowLocks lock;
  fused::ThreadForces fthr;
  std::vector<int> order, bstart, blen;
};

struct Grace3LFused::Work {
  int nbat = 0, ty = 0, sp = 0;
  int zero_i = -1, zero_j = -1;    // the first bond of length 0 its gathers met (throw_zero_bond)
  std::vector<int> atom, boff;
  std::vector<int> bj, bt;
  std::vector<double> rh, rr, rc;
  fvec Y, G, Rb, dRb;
  hvec val[T_NT], grad[T_NT], tmp;
  fvec P, dP, xc, Uc, Ic, dUc, dIc, dAc;
  std::vector<sv16, fused::Align64<sv16>> sy;
  fvec mT, mdT, mS0, mdS0, mS1, mdS1, mO, mdO;    // radial MLP by GEMM
  Scratch s;
  std::vector<double> uqf, uqd;    // UQ feature and Mahalanobis delta [D]
  bool want_vir = false;           // CsrBonds: this thread's bond virial
  const Coupling *cpl = nullptr;   // two-group coupling (ComputeFlags::couple), nullptr: off
  std::vector<double> cp;           // coupling: per bond true r, dr'/dr, dr'/dlambda (bonds.h)
  double dedl = 0.0;                // coupling: this thread's dE/dlambda
  double vir[6] = {0, 0, 0, 0, 0, 0};
  double E[64];
  double *ft = nullptr;
};

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

Grace3LFused::Grace3LFused(std::shared_ptr<const GRACE3LModel> m, FusedSettings s, Logger lg) :
    log(lg), model(std::move(m)), set(s), g(std::make_unique<Glob>())
{
  if (set.block < 1 || set.block > 64) throw Error("GRACE-3L: block must be in 1..64");
  for (int k = 0; k < 3; k++)
    if (spline_n_of(set, k) < 16) throw Error(str("GRACE-3L: spline N of layer ", k + 1, " must be >= 16"));
  if ((int) model->shift_values.size() != model->n_elements && !model->shift_values.empty())
    throw Error(str("GRACE-3L: shift_values has ", model->shift_values.size(), " entries, expected n_elements = ",
                    model->n_elements));
  g3_prof_on = getenv("G2LF_PROF") != nullptr;
}

Grace3LFused::~Grace3LFused()
{
  if (g3_prof_on) {
    double tot = 0.0;
    for (int q = 0; q < P_N; q++) tot += g3_prof[q];
    std::ostringstream o;
    o << "[GRACE-3L] profile (wall), s:\n";
    for (int q = 0; q < P_N; q++) {
      char b[160];
      std::snprintf(b, sizeof(b), "  %-16s %10.4f  %5.1f %%\n", P_NAME[q], g3_prof[q], 100.0 * g3_prof[q] / std::max(tot, 1e-30));
      o << b;
    }
    o << "[GRACE-3L] thread 0 categories, s (L1 L2 L3):\n";
    for (int c = 0; c < C_N; c++) {
      char b[160];
      std::snprintf(b, sizeof(b), "  %-16s %9.4f %9.4f %9.4f\n", C_NAME[c], g3_cat[0][c], g3_cat[1][c], g3_cat[2][c]);
      o << b;
    }
    o << "[GRACE-3L] thread 0 chain operations, s (fwd bwd):\n";
    for (int q = 0; q < O_N; q++) {
      char b[160];
      std::snprintf(b, sizeof(b), "  %-16s %9.4f %9.4f\n", O_NAME[q], g3_op[0][q], g3_op[1][q]);
      o << b;
    }
    char b[160];
    std::snprintf(b, sizeof(b), "  %-16s %9.4f (fwd + bwd, part of reduce)\n", "couple rows", g3_couple_tot);
    o << b;
    log.info(o.str());
  }
}

void Grace3LFused::setup_types(int ntypes_, const std::vector<int> &t2m)
{
  ntypes = ntypes_;
  type2mu = t2m;
  check_type2mu(type2mu, ntypes, model->n_elements, "GRACE-3L");
  const int ntp = ntypes + 1;
  cutsq_flat.assign((size_t) ntp * ntp, 0.0);
  cutsq_rows.resize(ntp);
  for (int i = 0; i < ntp; i++) cutsq_rows[i] = cutsq_flat.data() + (size_t) i * ntp;
  for (int i = 1; i <= ntypes; i++)
    for (int j = 1; j <= ntypes; j++) {
      const double rc = cutoff(i, j);
      cutsq_rows[i][j] = rc * rc;
    }
  build_net();
  uq_ready = false;
}

double Grace3LFused::cutoff(int i, int j) const
{
  return pair_cutoff(*model, type2mu[i], type2mu[j]);
}

void Grace3LFused::enable_uq()
{
  if (!net) throw Error("GRACE-3L: setup_types() before enable_uq()");
  if (!net->has_uq) throw Error("GRACE-3L: the model has no UQ head (kokkos.npz exported without UQ artifacts)");
  uq_ready = true;
}

void Grace3LFused::ensure_threads(int nth)
{
  if (nth == nthreads && (int) work.size() == nth) return;
  nthreads = nth;
  work.clear();
  for (int t = 0; t < nthreads; t++) work.emplace_back(new Work());
  if (gemm::blas_thread_unsafe() && nthreads > 1 && !gemm::force_builtin()) {
    gemm::force_builtin() = true;
    log.warn("GRACE-3L: the linked OpenBLAS uses pthreads and is not safe to call from several OpenMP threads; "
             "using the built-in GEMM. Link a sequential or OpenMP BLAS (e.g. BLA_VENDOR=Intel10_64lp_seq) to use it");
  }
}

const char *Grace3LFused::gemm_backend() const { return gemm::backend(); }
int Grace3LFused::message_width(int k) const { return k == 0 ? g->w1 : g->w2; }
float *Grace3LFused::message(int k) { return k == 0 ? g->I1.data() : g->I2.data(); }
float *Grace3LFused::message_grad(int k) { return k == 0 ? g->dI1.data() : g->dI2.data(); }

double Grace3LFused::memory_bytes() const
{
  double b = 0.0;
  if (net)
    for (int k = 0; k < 3; k++) b += net->L[k].spl.size() * sizeof(float);
  b += (double) nmax * (2.0 * g->w1 + 2.0 * g->w2 + 3.0 * g->wr) * sizeof(float);
  if (net)
    for (int t = 0; t < T_NT; t++) b += (double) nthreads * 2.0 * net->tmax[t] * set.block * sizeof(float);
  return b;
}
namespace {

// fp64 radial MLP with forward-mode d/dx: input [T_1..T_nrb (2x - 1), z_i, z_j]; hidden layers bias + silu
void radial_mlp_x(const GRACE3LModel::MLP &mlp, int nrb, const double *zi, const double *zj, int ne, double x,
                  std::vector<double> &out, std::vector<double> &dout)
{
  std::vector<double> h(nrb + 2 * ne, 0.0), dh(nrb + 2 * ne, 0.0);
  const double t = 2.0 * x - 1.0;
  double Tp = 1.0, Tc = t, dTp = 0.0, dTc = 2.0;
  for (int k = 0; k < nrb; k++) {
    h[k] = Tc;
    dh[k] = dTc;
    const double Tn = 2.0 * t * Tc - Tp, dTn = 2.0 * (2.0 * Tc + t * dTc) - dTp;
    Tp = Tc; dTp = dTc; Tc = Tn; dTc = dTn;
  }
  for (int e = 0; e < ne; e++) {
    h[nrb + e] = zi[e];
    h[nrb + ne + e] = zj[e];
  }
  for (int il = 0; il < mlp.n_layers; il++) {
    const int ni = mlp.n_in[il], no = mlp.n_out[il];
    std::vector<double> s(no, 0.0), ds(no, 0.0);
    const std::vector<double> &W = mlp.W[il];
    for (int k = 0; k < ni; k++) {
      const double hk = h[k], dk = dh[k];
      const double *w = W.data() + (size_t) k * no;
      for (int j = 0; j < no; j++) {
        s[j] += w[j] * hk;
        ds[j] += w[j] * dk;
      }
    }
    const double nrm = mlp.norm[il];
    for (int j = 0; j < no; j++) {
      s[j] *= nrm;
      ds[j] *= nrm;
      if (il < mlp.n_layers - 1) {
        if (!mlp.b[il].empty()) s[j] += nrm * mlp.b[il][j];
        const double sg = 1.0 / (1.0 + std::exp(-s[j]));
        ds[j] = sg * (1.0 + s[j] * (1.0 - sg)) * ds[j];
        s[j] = s[j] * sg;
      }
    }
    h.swap(s);
    dh.swap(ds);
  }
  out.swap(h);
  dout.swap(dh);
}

int find_name(const std::vector<std::string> &names, const std::string &n)
{
  for (size_t i = 0; i < names.size(); i++)
    if (names[i] == n) return (int) i;
  return -1;
}

}    // namespace

void Grace3LFused::build_net()
{
  const GRACE3LModel &m = *model;
  const double spline_mb = set.spline_mb;
  const bool store_fwd = set.store_fwd;
  auto arch = [&](bool ok, const std::string &why) {
    if (!ok) throw Error("GRACE-3L: the CPU engine supports only the GRACE-3L-OMAT-large architecture: " + why);
  };
  net = std::make_unique<Net>();
  Net &n = *net;

  arch(m.spbf.size() == 3 && m.prod.size() == 9 && m.fc.size() == 6 && m.eqnorm.size() == 2 && m.rmsnorm.size() == 3,
       "expected 3 SPBF, 9 products, 6 FCs, 2 eqnorms, 3 rmsnorms");
  arch(!m.spbf[0].equivariant && m.spbf[1].equivariant && m.spbf[2].equivariant, "layer 1 scalar, layers 2-3 equivariant");
  n.lmax = m.lmax;
  arch(n.lmax <= fused::MAX_LMAX, "lmax");
  n.nl = n.lmax + 1;
  n.nlm = n.nl * n.nl;
  fused::sh_coefs(n.lmax, n.sh_a, n.sh_b, n.sh_c, n.sh_d);

  // species present
  n.sp_of_type.assign(ntypes + 1, -1);
  n.mu_of_sp.clear();
  for (int t = 1; t <= ntypes; t++) {
    const int mu = type2mu[t];
    if (mu < 0) continue;    // unmapped: sp_of_type -1, never read
    int sp = -1;
    for (size_t q = 0; q < n.mu_of_sp.size(); q++)
      if (n.mu_of_sp[q] == mu) sp = q;
    if (sp < 0) {
      sp = n.mu_of_sp.size();
      n.mu_of_sp.push_back(mu);
    }
    n.sp_of_type[t] = sp;
  }
  n.nsp = n.mu_of_sp.size();
  n.shift_sp.resize(n.nsp);
  for (int sp = 0; sp < n.nsp; sp++) n.shift_sp[sp] = m.shift_values.empty() ? 0.0 : m.shift_values[n.mu_of_sp[sp]];
  n.oscale = m.output_scale;
  n.rcut_type.assign((size_t) (ntypes + 1) * (ntypes + 1), 0.0);
  for (int a = 1; a <= ntypes; a++)
    for (int b = 1; b <= ntypes; b++) n.rcut_type[a * (ntypes + 1) + b] = pair_cutoff(m, type2mu[a], type2mu[b]);

  int nsum = 0;
  for (int k = 0; k < 3; k++) nsum += spline_n_of(set, k);
  const double table_mb = (double) n.nsp * n.nsp * nsum * 4 * n.nl * m.spbf[0].n_rad_max * 4 / 1048576.0;
  double budget = spline_mb;
  if (budget < 0.0) {    // auto: a quarter of the host memory per process, within what is available
    double total = 0.0, avail = 0.0;
    fused::host_memory_mb(total, avail);
    budget = std::max(2048.0, std::min(0.25 * total, avail) / std::max(1, set.ranks_per_node));
  }
  const bool use_mlp = table_mb > budget;
  if (use_mlp)
    log.info(str("[GRACE-3L] radial MLP by GEMM: tables for ", n.nsp * n.nsp, " species pairs would need ",
                 (long) table_mb, " MB > spline_mb ", (long) budget, "\n"));

  std::vector<std::string> prod_names, red_names, fc_names;
  for (auto &p : m.prod) prod_names.push_back(p.name);
  for (auto &r : m.reduce) red_names.push_back(r.name);
  for (auto &f : m.fc) fc_names.push_back(f.name);
  const int ne = m.embedding_size;

  for (int k = 0; k < 3; k++) {
    Net::Layer &L = n.L[k];
    const GRACE3LModel::SPBF &sp = m.spbf[k];
    const std::string K = "A" + std::to_string(k + 1);
    arch(sp.name == K, "SPBF " + K);
    arch(sp.lmax == n.lmax, K + " lmax");
    L.equiv = sp.equivariant;
    L.C = sp.n_rad_max;
    arch(L.C % 16 == 0, K + " n_rad_max multiple of 16");
    L.lt = sp.l_tile;
    arch((int) L.lt.size() == n.nlm, K + " l_tile");
    arch(sp.mlp.n_layers >= 2 && sp.mlp.n_in[0] == m.nradbase + 2 * ne && sp.mlp.n_out.back() == L.C * n.nl,
         K + " radial MLP shape");

    // radial tables: G[l][n](x) = MLP_o(x) env(x) inv_avg (z_tr[sp_j][n] in layer 1), o = n (lmax+1) + l
    std::vector<double> ztr((size_t) n.nsp * L.C, 1.0);
    if (!L.equiv) {
      arch((int) sp.lin_transform_W.size() == ne * L.C, K + " lin_transform");
      for (int s = 0; s < n.nsp; s++)
        for (int c = 0; c < L.C; c++) {
          double v = 0.0;
          for (int e = 0; e < ne; e++)
            v += m.chem_embedding[(size_t) n.mu_of_sp[s] * ne + e] * sp.lin_transform_W[(size_t) e * L.C + c];
          ztr[(size_t) s * L.C + c] = v * sp.lin_transform_norm;
        }
    }
    const int N = spline_n_of(set, k), nl = n.nl, C = L.C, p = sp.p;
    L.N = N;
    L.use_mlp = use_mlp;
    L.penv = p;
    L.inv_avg = sp.inv_avg_n_neigh;
    L.fac.assign((size_t) n.nsp * C, 1.0f);
    for (int s2 = 0; s2 < n.nsp; s2++)
      for (int c = 0; c < C; c++)
        L.fac[(size_t) s2 * C + c] = (float) (sp.inv_avg_n_neigh * (L.equiv ? 1.0 : ztr[(size_t) s2 * C + c]));
    if (use_mlp) {
      const auto &M = sp.mlp;
      arch(M.n_layers == 3, K + " radial MLP with 3 layers for the GEMM path");
      L.nrb = m.nradbase;
      L.d1 = M.n_out[0];
      L.d2 = M.n_out[1];
      L.d3 = M.n_out[2];
      const double n0 = M.norm[0], n1 = M.norm[1], n2 = M.norm[2];
      L.W0c.resize((size_t) L.nrb * L.d1);
      for (int k = 0; k < L.nrb; k++)
        for (int j = 0; j < L.d1; j++) L.W0c[(size_t) k * L.d1 + j] = (float) (n0 * M.W[0][(size_t) k * L.d1 + j]);
      L.Ei.assign((size_t) n.nsp * L.d1, 0.0f);
      L.Ej.assign((size_t) n.nsp * L.d1, 0.0f);
      for (int s2 = 0; s2 < n.nsp; s2++)
        for (int j = 0; j < L.d1; j++) {
          double ei = M.b[0].empty() ? 0.0 : M.b[0][j], ej = 0.0;
          for (int e = 0; e < ne; e++) {
            const double z = m.chem_embedding[(size_t) n.mu_of_sp[s2] * ne + e];
            ei += z * M.W[0][(size_t) (L.nrb + e) * L.d1 + j];
            ej += z * M.W[0][(size_t) (L.nrb + ne + e) * L.d1 + j];
          }
          L.Ei[(size_t) s2 * L.d1 + j] = (float) (n0 * ei);
          L.Ej[(size_t) s2 * L.d1 + j] = (float) (n0 * ej);
        }
      L.W1.resize((size_t) L.d1 * L.d2);
      for (size_t q = 0; q < L.W1.size(); q++) L.W1[q] = (float) (n1 * M.W[1][q]);
      L.b1.assign(L.d2, 0.0f);
      for (int j = 0; j < L.d2; j++) L.b1[j] = M.b[1].empty() ? 0.0f : (float) (n1 * M.b[1][j]);
      L.W2.resize((size_t) L.d2 * L.d3);
      for (size_t q = 0; q < L.W2.size(); q++) L.W2[q] = (float) (n2 * M.W[2][q]);
    }
    L.splpair = use_mlp ? 0 : (size_t) N * 4 * nl * C;
    L.spl.assign((size_t) n.nsp * n.nsp * L.splpair, 0.0f);
    // species pairs in parallel when there are enough of them (25 species: 625 pairs, 9 GB), else the nodes of a pair
    const int npair = use_mlp ? 0 : n.nsp * n.nsp;
#ifdef _OPENMP
    const bool par_pairs = npair >= omp_get_max_threads();
#else
    const bool par_pairs = false;
#endif
#pragma omp parallel for schedule(dynamic) if (par_pairs)
    for (int pr = 0; pr < npair; pr++) {
        const int si = pr / n.nsp, sj = pr % n.nsp;
        const double *zi = m.chem_embedding.data() + (size_t) n.mu_of_sp[si] * ne;
        const double *zj = m.chem_embedding.data() + (size_t) n.mu_of_sp[sj] * ne;
        std::vector<double> fv((size_t) (N + 1) * nl * C), dv((size_t) (N + 1) * nl * C);
        const double h = 1.0 / N;
#pragma omp parallel for schedule(static) if (!par_pairs)
        for (int q = 0; q <= N; q++) {
          std::vector<double> out, dout;
          const double x = q * h;
          radial_mlp_x(sp.mlp, m.nradbase, zi, zj, ne, x, out, dout);
          const double xp = std::pow(x, p), xp1 = xp * x, xp2 = xp1 * x;
          const double env = 1.0 - 0.5 * (p + 1) * (p + 2) * xp + p * (p + 2) * xp1 - 0.5 * p * (p + 1) * xp2;
          const double denv = p == 0 ? 0.0 :
              0.5 * p * (p + 1) * (p + 2) * (-std::pow(x, p - 1) + 2.0 * xp - xp1);
          for (int c = 0; c < C; c++)
            for (int l = 0; l < nl; l++) {
              const int o = c * nl + l;
              const double fac = sp.inv_avg_n_neigh * (L.equiv ? 1.0 : ztr[(size_t) sj * C + c]);
              fv[((size_t) q * nl + l) * C + c] = out[o] * env * fac;
              dv[((size_t) q * nl + l) * C + c] = h * (dout[o] * env + out[o] * denv) * fac;
            }
        }
        float *tab = L.spl.data() + (size_t) (si * n.nsp + sj) * L.splpair;
        for (int q = 0; q < N; q++)
          for (int l = 0; l < nl; l++)
            for (int c = 0; c < C; c++) {
              const size_t i0 = ((size_t) q * nl + l) * C + c, i1 = ((size_t) (q + 1) * nl + l) * C + c;
              const double f0 = fv[i0], f1 = fv[i1], d0 = dv[i0], d1 = dv[i1];
              const double cf[4] = {f0, d0, -3.0 * f0 - 2.0 * d0 + 3.0 * f1 - d1, 2.0 * f0 + d0 - 2.0 * f1 + d1};
              for (int r = 0; r < 4; r++) tab[(((size_t) q * 4 + r) * nl + l) * C + c] = (float) cf[r];
            }
      }

    // equivariant SPBF: CG couple of (Y lm_y) x (message lm_i), and the species term of the message
    if (L.equiv) {
      L.nfA = sp.nfunc;
      L.nlmi = sp.n_lm_ind;
      arch((int) sp.cg_W.size() == n.nlm * L.nlmi * L.nfA, K + " cg_W shape");
      const int np = n.nlm * L.nlmi;
      L.cg_off.assign(L.nfA + 1, 0);
      L.cgt_off.assign(np + 1, 0);
      for (int q = 0; q < np; q++)
        for (int f = 0; f < L.nfA; f++)
          if (sp.cg_W[(size_t) q * L.nfA + f] != 0.0) {
            L.cg_off[f + 1]++;
            L.cgt_off[q + 1]++;
          }
      for (int f = 0; f < L.nfA; f++) L.cg_off[f + 1] += L.cg_off[f];
      for (int q = 0; q < np; q++) L.cgt_off[q + 1] += L.cgt_off[q];
      L.cg_p.resize(L.cg_off[L.nfA]);
      L.cg_v.resize(L.cg_off[L.nfA]);
      L.cgt_f.resize(L.cgt_off[np]);
      L.cgt_v.resize(L.cgt_off[np]);
      std::vector<int> pf(L.cg_off.begin(), L.cg_off.end() - 1), pq(L.cgt_off.begin(), L.cgt_off.end() - 1);
      for (int q = 0; q < np; q++)
        for (int f = 0; f < L.nfA; f++) {
          const double v = sp.cg_W[(size_t) q * L.nfA + f];
          if (v == 0.0) continue;
          L.cg_p[pf[f]] = q;
          L.cg_v[pf[f]++] = (float) v;
          L.cgt_f[pq[q]] = f;
          L.cgt_v[pq[q]++] = (float) v;
        }
      arch((int) sp.chem_linear_W.size() == ne * L.C && (int) sp.chem_l0_mask.size() == L.nlmi, K + " chem_linear");
      L.zproj.assign((size_t) n.nsp * L.C, 0.0f);
      for (int s = 0; s < n.nsp; s++)
        for (int c = 0; c < L.C; c++) {
          double v = 0.0;
          for (int e = 0; e < ne; e++)
            v += m.chem_embedding[(size_t) n.mu_of_sp[s] * ne + e] * sp.chem_linear_W[(size_t) e * L.C + c];
          L.zproj[(size_t) s * L.C + c] = (float) (v * sp.chem_linear_norm);
        }
      L.l0mask.assign(sp.chem_l0_mask.begin(), sp.chem_l0_mask.end());
      L.nf[T_A] = L.nfA;
      L.nc[T_A] = L.C;
    } else {
      L.nf[T_X] = n.nlm;
      L.nc[T_X] = L.C;
    }

    // tensors by model name
    std::map<std::string, int> role;
    role[K] = L.equiv ? T_A : T_X;
    role[K + "_red"] = T_X;
    role[K + "_2"] = T_P2;
    role[K + "_2_red"] = T_R2;
    role[K + "_3"] = T_P3;
    role[K + "_3_red"] = T_R3;
    role[K + "_4"] = T_P4;
    role[K + "_4_red"] = T_R4;

    auto build_red = [&](Red &R, const std::string &name, int tout, bool need) {
      const int ri = find_name(red_names, name);
      if (ri < 0) {
        arch(!need, "missing reduce " + name);
        return false;
      }
      const GRACE3LModel::Reduce &r = m.reduce[ri];
      R.out = tout;
      R.elem = r.elem_dep;
      const int nsp = R.elem ? n.nsp : 1;
      const int nfo = r.only_invar ? 1 : r.n_funcs;
      L.nf[tout] = nfo;
      L.nc[tout] = r.n_out;
      R.ins.clear();
      R.role.clear();
      R.ci.clear();
      for (const auto &ins : r.instr) {
        auto it = role.find(ins.name);
        arch(it != role.end(), "reduce " + name + " input " + ins.name);
        const int src = it->second;
        arch(L.nc[src] == ins.n_in, "reduce " + name + " input width " + ins.name);
        std::vector<int> s(ins.n_conn), d(ins.n_conn), t(ins.n_conn);
        for (int c = 0; c < ins.n_conn; c++) {
          s[c] = ins.collect_ind[c];
          d[c] = r.only_invar ? 0 : ins.total_sum_ind[c];
          t[c] = ins.w_l_tile[c];
          arch(s[c] >= 0 && s[c] < L.nf[src] && d[c] >= 0 && d[c] < nfo && t[c] >= 0 && t[c] < ins.w_shape,
               "reduce " + name + " indices");
        }
        const size_t per = (size_t) r.n_out * ins.n_in * ins.w_shape;
        arch(ins.W.size() == per * (R.elem ? m.n_elements : 1), "reduce " + name + " weight shape");
        Contract c;
        build_contract(c, ins.n_in, r.n_out, ins.w_shape, nsp, s, d, t, [&](int spx, int tt, int nn, int kk) {
          const size_t mu = R.elem ? n.mu_of_sp[spx] : 0;
          return ins.norm * ins.W[mu * per + ((size_t) kk * ins.n_in + nn) * ins.w_shape + tt];
        });
        R.ins.push_back(std::move(c));
        R.role.push_back(src);
        R.ci.push_back(s);
      }
      R.nm.clear();
      if (r.has_norm_map) {
        arch((int) r.norm_map.size() == nfo, "reduce " + name + " norm_map");
        R.nm.assign(r.norm_map.begin(), r.norm_map.end());
      }
      return true;
    };
    auto build_prod = [&](Prod &P, const std::string &name, int tl, int tr, int tlp, int trp, int tout) {
      const int pi = find_name(prod_names, name);
      arch(pi >= 0, "missing product " + name);
      const GRACE3LModel::CPL &c = m.prod[pi];
      arch(c.n_left == L.nc[tl] && c.n_right == L.nc[tr], "product " + name + " widths");
      arch((int) c.group_left.size() == L.nf[tl] && (int) c.group_right.size() == L.nf[tr], "product " + name + " lm");
      P.tl = tl; P.tr = tr; P.tlp = tlp; P.trp = trp;
      L.nf[tlp] = L.nf[tl]; L.nc[tlp] = c.rank;
      L.nf[trp] = L.nf[tr]; L.nc[trp] = c.rank;
      L.nf[tout] = c.nfunc; L.nc[tout] = c.rank;
      std::vector<int> s(L.nf[tl]);
      for (int q = 0; q < L.nf[tl]; q++) s[q] = q;
      build_contract(P.U, c.n_left, c.rank, c.n_groups_left, 1, s, s, c.group_left, [&](int, int t, int nn, int kk) {
        return c.norm_u * c.U[((size_t) t * c.rank + kk) * c.n_left + nn];
      });
      std::vector<int> s2(L.nf[tr]);
      for (int q = 0; q < L.nf[tr]; q++) s2[q] = q;
      build_contract(P.V, c.n_right, c.rank, c.n_groups_right, 1, s2, s2, c.group_right, [&](int, int t, int nn, int kk) {
        return c.norm_v * c.V[((size_t) t * c.rank + kk) * c.n_right + nn];
      });
      Couple &cp = P.cp;
      cp.nfunc = c.nfunc;
      cp.rank = c.rank;
      cp.off.assign(c.nfunc + 1, 0);
      for (int t = 0; t < c.n_cg; t++) {
        arch(c.m_sum_ind[t] >= 0 && c.m_sum_ind[t] < c.nfunc && c.left_ind[t] < L.nf[tl] && c.right_ind[t] < L.nf[tr],
             "product " + name + " indices");
        cp.off[c.m_sum_ind[t] + 1]++;
      }
      for (int f = 0; f < c.nfunc; f++) cp.off[f + 1] += cp.off[f];
      std::vector<int> pos(cp.off.begin(), cp.off.end() - 1);
      cp.li.resize(c.n_cg);
      cp.ri.resize(c.n_cg);
      cp.cg.resize(c.n_cg);
      for (int t = 0; t < c.n_cg; t++) {
        const int q = pos[c.m_sum_ind[t]]++;
        cp.li[q] = c.left_ind[t];
        cp.ri[q] = c.right_ind[t];
        cp.cg[q] = (float) c.cg[t];
      }
    };
    auto build_fc = [&](FCL &F, const std::string &name, int tleft, int tright, int tout) {
      const int fi = find_name(fc_names, name);
      arch(fi >= 0, "missing FC " + name);
      const GRACE3LModel::FC &f = m.fc[fi];
      arch(!f.left_coefs, "FC " + name + " left_coefs");
      arch(f.n_funcs_left == L.nf[tleft] && f.n_out == L.nc[tleft] && f.n_in_right == L.nc[tright],
           "FC " + name + " shapes");
      F.tleft = tleft; F.tright = tright; F.tout = tout;
      L.nf[tout] = f.n_funcs_left;
      L.nc[tout] = f.n_out;
      std::vector<int> s(f.n_funcs_right), d(f.n_funcs_right), t(f.n_funcs_right);
      for (int q = 0; q < f.n_funcs_right; q++) {
        s[q] = f.collect_from[q];
        d[q] = f.collect_to[q];
        t[q] = f.w_tile_right[q];
        arch(s[q] < L.nf[tright] && d[q] < f.n_funcs_left && t[q] < f.w_shape_right, "FC " + name + " indices");
      }
      build_contract(F.right, f.n_in_right, f.n_out, f.w_shape_right, 1, s, d, t, [&](int, int tt, int nn, int kk) {
        return f.norm_right * f.w_right[((size_t) kk * f.n_in_right + nn) * f.w_shape_right + tt];
      });
      F.nof.assign(f.norm_out_factor.begin(), f.norm_out_factor.end());
      arch((int) F.nof.size() == f.n_funcs_left, "FC " + name + " norm_out_factor");
    };

    // fc_fold: with P = norm_u U (norm_v V) of an lm group g and the FC out[d] = nof[d] (left[d] + W_t right[s]),
    // the projection P out = (nof P) left + (nof W_t P) right; folded only when all functions of a group share the
    // FC tile and nof, so the folded weights stay one per group
    auto build_fold = [&](Prod &P, const FCL &F, const std::string &pname, const std::string &fname, bool self) {
      P.fold = false;
      if (!set.fc_fold) return false;
      const GRACE3LModel::CPL &c = m.prod[find_name(prod_names, pname)];
      const GRACE3LModel::FC &f = m.fc[find_name(fc_names, fname)];
      const int nf = L.nf[F.tout], no = f.n_out, ni = f.n_in_right, rk = c.rank, nt = f.w_shape_right;
      if (c.n_left != no || (self && c.n_right != no) || nf != L.nf[F.tleft]) return false;
      std::vector<int> gt[2];
      std::vector<double> gn[2];
      for (int side = 0; side < (self ? 2 : 1); side++) {
        const std::vector<int> &grp = side ? c.group_right : c.group_left;
        const int ng = side ? c.n_groups_right : c.n_groups_left;
        if ((int) grp.size() != nf) return false;
        gt[side].assign(ng, -1);
        gn[side].assign(ng, 0.0);
        std::vector<char> seen(ng, 0);
        for (int d = 0; d < nf; d++) {
          const int g = grp[d];
          if (seen[g] && gn[side][g] != f.norm_out_factor[d]) return false;
          seen[g] = 1;
          gn[side][g] = f.norm_out_factor[d];
        }
        for (int q = 0; q < f.n_funcs_right; q++) {
          const int g = grp[f.collect_to[q]];
          if (gt[side][g] >= 0 && gt[side][g] != f.w_tile_right[q]) return false;
          gt[side][g] = f.w_tile_right[q];
        }
      }
      std::vector<int> id(nf), rs(f.n_funcs_right), rd(f.n_funcs_right);
      for (int d = 0; d < nf; d++) id[d] = d;
      for (int q = 0; q < f.n_funcs_right; q++) {
        rs[q] = f.collect_from[q];
        rd[q] = f.collect_to[q];
      }
      for (int side = 0; side < (self ? 2 : 1); side++) {
        const std::vector<int> &grp = side ? c.group_right : c.group_left;
        const int ng = side ? c.n_groups_right : c.n_groups_left;
        const std::vector<double> &PM = side ? c.V : c.U;
        const double pn = side ? c.norm_v : c.norm_u;
        auto pw = [&](int g, int k, int j) { return pn * PM[((size_t) g * rk + j) * no + k]; };
        std::vector<int> rt(f.n_funcs_right);
        for (int q = 0; q < f.n_funcs_right; q++) rt[q] = grp[f.collect_to[q]];
        build_contract(side ? P.VL : P.UL, no, rk, ng, 1, id, id, grp,
                       [&](int, int g, int k, int j) { return gn[side][g] * pw(g, k, j); });
        build_contract(side ? P.VR : P.UR, ni, rk, ng, 1, rs, rd, rt, [&](int, int g, int n, int j) {
          if (gt[side][g] < 0) return 0.0;
          double v = 0.0;
          for (int k = 0; k < no; k++) v += f.w_right[((size_t) k * ni + n) * nt + gt[side][g]] * pw(g, k, j);
          return gn[side][g] * f.norm_right * v;
        });
      }
      P.fold = true;
      P.self = self;
      P.fl = F.tleft;
      P.fr = F.tright;
      return true;
    };

    L.has_redA = L.equiv;
    if (L.equiv) build_red(L.redA, K + "_red", T_X, true);
    build_prod(L.p2, K + "_2", T_X, T_X, T_L2, T_R2p, T_P2);
    build_red(L.r2, K + "_2_red", T_R2, true);
    build_fc(L.fa, K + "_2a", T_R2, T_X, T_FA);
    build_prod(L.p3, K + "_3", T_FA, T_X, T_L3, T_R3p, T_P3);
    build_red(L.r3, K + "_3_red", T_R3, true);
    build_fc(L.fb, K + "_2b", T_R2, T_X, T_FB);
    build_prod(L.p4, K + "_4", T_FB, T_FB, T_L4, T_R4p, T_P4);
    build_red(L.r4, K + "_4_red", T_R4, true);
    n.folded[k] = build_fold(L.p3, L.fa, K + "_3", K + "_2a", false) + build_fold(L.p4, L.fb, K + "_4", K + "_2b", true);
    for (const Red *R : {&L.r2, &L.r3, &L.r4})
      arch(R->ins.size() == 1 && !R->elem && R->ins[0].nin == L.nc[R->role[0]] &&
               (R->role[0] == T_P2 || R->role[0] == T_P3 || R->role[0] == T_P4),
           K + " product reduces: one element-independent instruction on the product");
    const std::string kk = std::to_string(k + 1);
    L.has_eq = build_red(L.eq, "eq" + kk, T_EQ, k < 2);
    arch(L.has_eq == (k < 2), "eq" + kk);
    build_red(L.rho, "rho" + kk, T_RHO, true);
    arch(L.rho.elem && L.nf[T_RHO] == 1, "rho" + kk + " invariant, element-dependent");
    if (L.has_eq) {
      const GRACE3LModel::EqNorm &e = m.eqnorm[k];
      arch(e.name == "eq" + kk + "_norm" && e.M == L.nf[T_EQ] && e.n_out == L.nc[T_EQ], "eq" + kk + "_norm");
      EqNorm &E = L.eqn;
      E.nf = e.M;
      E.C = e.n_out;
      E.eps = e.eps;
      E.center = e.center_l0 != 0;
      E.dw = e.degree_weights;
      E.mask = e.l0_mask;
      E.e = e.expand_index;
      E.aff.assign(e.affine_weight.begin(), e.affine_weight.end());
      L.nf[T_I] = E.nf;
      L.nc[T_I] = E.C;
    }
    const GRACE3LModel::RMSNorm &rn = m.rmsnorm[k];
    arch(rn.name == "rho" + kk + "_norm" && rn.n_out == L.nc[T_RHO], "rho" + kk + "_norm");
    L.rhon.C = rn.n_out;
    L.rhon.only_nonlin = (int) rn.scale.size() == rn.n_out - 1;
    arch(L.rhon.only_nonlin || (int) rn.scale.size() == rn.n_out, "rho" + kk + "_norm scale");
    L.rhon.scale.assign(rn.scale.begin(), rn.scale.end());
    L.nf[T_RHON] = 1;
    L.nc[T_RHON] = rn.n_out;
  }
  // messages: I1 feeds layer 2, I2 layer 3
  arch(n.L[0].nf[T_I] == n.L[1].nlmi && n.L[0].nc[T_I] == n.L[1].C, "I1 -> A2 shape");
  arch(n.L[1].nf[T_I] == n.L[2].nlmi && n.L[1].nc[T_I] == n.L[2].C, "I2 -> A3 shape");
  g->w1 = n.L[0].nf[T_I] * n.L[0].nc[T_I];
  g->w2 = n.L[1].nf[T_I] * n.L[1].nc[T_I];
  arch(g->w1 % 2 == 0 && g->w2 % 2 == 0, "message widths");
  g->wr = n.L[0].nc[T_RHON];
  arch(n.L[1].nc[T_RHON] == g->wr && n.L[2].nc[T_RHON] == g->wr, "rho widths");

  // readout
  const GRACE3LModel::MLP &e = m.energy_mlp;
  arch(e.n_layers == 2 && e.n_in[0] == g->wr - 1 && e.n_out[1] == 1, "energy MLP shape");
  arch(e.n_in[0] <= 128 && e.n_out[0] <= 256, "energy MLP width (readout buffers 128 / 256)");
  n.e_nin = e.n_in[0];
  n.e_nh = e.n_out[0];
  n.eW0.assign(e.W[0].begin(), e.W[0].end());
  n.eW1.assign(e.W[1].begin(), e.W[1].end());
  n.en0 = e.norm[0];
  n.en1 = e.norm[1];
  n.e_act = m.energy_mlp_activation;

  n.maxten = 0;
  for (int t = 0; t < T_NT; t++) n.tmax[t] = 0;
  for (int k = 0; k < 3; k++)
    for (int t = 0; t < T_NT; t++) {
      if (t == T_P2 || t == T_P3 || t == T_P4) continue;    // couple outputs are fused into their reduces
      n.tmax[t] = std::max(n.tmax[t], (size_t) n.L[k].nf[t] * n.L[k].nc[t]);
      n.maxten = std::max(n.maxten, n.tmax[t]);
    }
  n.maxP = 0;
  for (int k = 1; k < 3; k++) n.maxP = std::max(n.maxP, (size_t) n.nlm * n.L[k].nlmi * n.L[k].C);
  for (int k = 0; k < 2; k++) {
    g->stw[k] = 0;
    for (int t : STORED) g->stw[k] += n.L[k].nf[t] * n.L[k].nc[t];
  }

  // UQ: the basis is the concatenated inputs of the rho1, rho2, rho3 reduces (projection matrix rows in that order)
  n.has_uq = m.has_uq;
  if (n.has_uq) {
    n.uq_rp.assign(m.uq_rp_matrix.begin(), m.uq_rp_matrix.end());
    int off = 0;
    for (int k = 0; k < 3; k++) {
      n.uq_off[k] = off;
      const Red &R = n.L[k].rho;
      for (size_t j = 0; j < R.ins.size(); j++) {
        arch(R.role[j] != T_P2 && R.role[j] != T_P3 && R.role[j] != T_P4, "UQ: rho input is a cp_l product");
        off += n.L[k].nc[R.role[j]] * (int) R.ci[j].size();
      }
    }
    n.uq_off[3] = off;
    if (off != m.uq_d_basis)
      throw Error(str("GRACE-3L UQ: basis width ", off, " != uq_rp_matrix rows ", m.uq_d_basis));
    if (m.uq_n_elements != m.n_elements || m.uq_feature_dim != m.uq_rp_dim + m.uq_n_density || m.uq_n_density > 4)
      throw Error(str("GRACE-3L UQ: inconsistent UQ head (elements ", m.uq_n_elements, " / ", m.n_elements,
                      ", D ", m.uq_feature_dim, ", RP ", m.uq_rp_dim, ", density ", m.uq_n_density, ")"));
  }
  nmax = 0;    // reallocate the per-atom arrays with the new widths
  log.info(str("[GRACE-3L] ", n.nsp, " species, radial tables ", (long) table_mb, " MB (N = ", spline_n_of(set, 0),
               " ", spline_n_of(set, 1), " ", spline_n_of(set, 2), "), largest block tensor ",
               n.maxten, " floats/atom, store_fwd ", store_fwd ? "yes" : "no", " (",
               store_fwd ? (g->stw[0] + g->stw[1]) * 4 / 1024 : 0, " KB/atom); GEMM ", gemm::backend(),
               "; FCs folded into the cp_l projections per layer ", n.folded[0], " ", n.folded[1], " ", n.folded[2],
               "\n"));
}

/* ---------------------------------------------------------------------- */

namespace {

using Layer = Grace3LFused::Net::Layer;

inline float *tv(Grace3LFused::Work &w, int t) { return w.val[t].data(); }
inline float *tg(Grace3LFused::Work &w, int t) { return w.grad[t].data(); }

void zero_t(const Layer &L, hvec *b, int t, int NB) { std::memset(b[t].data(), 0, (size_t) L.nf[t] * NB * L.nc[t] * sizeof(float)); }

// grad[t] with the rows of function f scaled by sc[f], into w.tmp
const float *scaled_grad(const Layer &L, Grace3LFused::Work &w, int t, const float *sc, int NB)
{
  std::memcpy(w.tmp.data(), tg(w, t), (size_t) L.nf[t] * NB * L.nc[t] * sizeof(float));
  scale_rows(w.tmp.data(), sc, L.nf[t], L.nc[t], NB, w.nbat);
  return w.tmp.data();
}

// store_fwd: the STORED tensors of block atom a <-> its per-atom row (to_row: block -> row, else row -> block)
void copy_stored(const Layer &L, Grace3LFused::Work &w, int NB, int a, float *row, bool to_row)
{
  for (int t : STORED) {
    const int nf = L.nf[t], nc = L.nc[t];
    for (int f = 0; f < nf; f++) {
      float *blk = tv(w, t) + ((size_t) f * NB + a) * nc, *r = row + (size_t) f * nc;
      if (to_row)
        std::memcpy(r, blk, nc * sizeof(float));
      else
        std::memcpy(blk, r, nc * sizeof(float));
    }
    row += (size_t) nf * nc;
  }
}

void red_fwd(const Layer &L, const Red &R, Grace3LFused::Work &w, int NB)
{
  OpT ot(0, O_RED);
  zero_t(L, w.val, R.out, NB);
  const int sp = R.elem ? w.sp : 0;
  for (size_t j = 0; j < R.ins.size(); j++)
    contract_fwd(R.ins[j], sp, tv(w, R.role[j]), tv(w, R.out), NB, w.nbat, w.s);
  if (!R.nm.empty()) scale_rows(tv(w, R.out), R.nm.data(), L.nf[R.out], L.nc[R.out], NB, w.nbat);
}

void red_bwd(const Layer &L, const Red &R, Grace3LFused::Work &w, int NB)
{
  OpT ot(1, O_RED);
  const int sp = R.elem ? w.sp : 0;
  const float *g = R.nm.empty() ? tg(w, R.out) : scaled_grad(L, w, R.out, R.nm.data(), NB);
  for (size_t j = 0; j < R.ins.size(); j++) contract_bwd(R.ins[j], sp, g, tg(w, R.role[j]), NB, w.nbat, w.s);
}

// cp_l product P with the reduce R that consumes it (R: one instruction on the product, not element-dependent)
void prod_red_fwd(const Layer &L, const Prod &P, const Red &R, Grace3LFused::Work &w, int NB)
{
  {
    OpT ot(0, O_PROJ);
    zero_t(L, w.val, P.tlp, NB);
    zero_t(L, w.val, P.trp, NB);
    if (P.fold) {
      contract_fwd(P.UL, 0, tv(w, P.fl), tv(w, P.tlp), NB, w.nbat, w.s);
      contract_fwd(P.UR, 0, tv(w, P.fr), tv(w, P.tlp), NB, w.nbat, w.s);
    } else
      contract_fwd(P.U, 0, tv(w, P.tl), tv(w, P.tlp), NB, w.nbat, w.s);
    if (P.fold && P.self) {
      contract_fwd(P.VL, 0, tv(w, P.fl), tv(w, P.trp), NB, w.nbat, w.s);
      contract_fwd(P.VR, 0, tv(w, P.fr), tv(w, P.trp), NB, w.nbat, w.s);
    } else
      contract_fwd(P.V, 0, tv(w, P.tr), tv(w, P.trp), NB, w.nbat, w.s);
  }
  OpT ot(0, O_RED);
  zero_t(L, w.val, R.out, NB);
  couple_contract_fwd(P.cp, R.ins[0], tv(w, P.tlp), tv(w, P.trp), tv(w, R.out), NB, w.nbat, w.s);
  if (!R.nm.empty()) scale_rows(tv(w, R.out), R.nm.data(), L.nf[R.out], L.nc[R.out], NB, w.nbat);
}

void prod_red_bwd(const Layer &L, const Prod &P, const Red &R, Grace3LFused::Work &w, int NB)
{
  {
    OpT ot(1, O_RED);
    const float *g = R.nm.empty() ? tg(w, R.out) : scaled_grad(L, w, R.out, R.nm.data(), NB);
    zero_t(L, w.grad, P.tlp, NB);
    zero_t(L, w.grad, P.trp, NB);
    couple_contract_bwd(P.cp, R.ins[0], tv(w, P.tlp), tv(w, P.trp), g, tg(w, P.tlp), tg(w, P.trp), NB, w.nbat, w.s);
  }
  OpT ot(1, O_PROJ);
  if (P.fold) {
    contract_bwd(P.UL, 0, tg(w, P.tlp), tg(w, P.fl), NB, w.nbat, w.s);
    contract_bwd(P.UR, 0, tg(w, P.tlp), tg(w, P.fr), NB, w.nbat, w.s);
  } else
    contract_bwd(P.U, 0, tg(w, P.tlp), tg(w, P.tl), NB, w.nbat, w.s);
  if (P.fold && P.self) {
    contract_bwd(P.VL, 0, tg(w, P.trp), tg(w, P.fl), NB, w.nbat, w.s);
    contract_bwd(P.VR, 0, tg(w, P.trp), tg(w, P.fr), NB, w.nbat, w.s);
  } else
    contract_bwd(P.V, 0, tg(w, P.trp), tg(w, P.tr), NB, w.nbat, w.s);
}

void fc_fwd(const Layer &L, const FCL &F, Grace3LFused::Work &w, int NB)
{
  OpT ot(0, O_FC);
  const size_t n = (size_t) L.nf[F.tout] * NB * L.nc[F.tout];
  std::memcpy(tv(w, F.tout), tv(w, F.tleft), n * sizeof(float));
  contract_fwd(F.right, 0, tv(w, F.tright), tv(w, F.tout), NB, w.nbat, w.s);
  scale_rows(tv(w, F.tout), F.nof.data(), L.nf[F.tout], L.nc[F.tout], NB, w.nbat);
}

void fc_bwd(const Layer &L, const FCL &F, Grace3LFused::Work &w, int NB)
{
  OpT ot(1, O_FC);
  const float *g = scaled_grad(L, w, F.tout, F.nof.data(), NB);
  vadd(tg(w, F.tleft), g, (size_t) L.nf[F.tout] * NB * L.nc[F.tout]);
  contract_bwd(F.right, 0, g, tg(w, F.tright), NB, w.nbat, w.s);
}

// layer chain after the SPBF (X, or A for equivariant layers, filled)
void chain_fwd(const Layer &L, Grace3LFused::Work &w, int NB)
{
  if (L.has_redA) red_fwd(L, L.redA, w, NB);
  prod_red_fwd(L, L.p2, L.r2, w, NB);
  if (!L.p3.fold) fc_fwd(L, L.fa, w, NB);
  prod_red_fwd(L, L.p3, L.r3, w, NB);
  if (!L.p4.fold) fc_fwd(L, L.fb, w, NB);
  prod_red_fwd(L, L.p4, L.r4, w, NB);
  if (L.has_eq) {
    red_fwd(L, L.eq, w, NB);
    const size_t rs = (size_t) NB * L.nc[T_EQ];
    for (int a = 0; a < w.nbat; a++)
      eqnorm_fwd(L.eqn, tv(w, T_EQ) + (size_t) a * L.nc[T_EQ], tv(w, T_I) + (size_t) a * L.nc[T_I], rs, w.xc.data());
  }
  red_fwd(L, L.rho, w, NB);
  for (int a = 0; a < w.nbat; a++)
    rmsnorm_fwd(L.rhon, tv(w, T_RHO) + (size_t) a * L.nc[T_RHO], tv(w, T_RHON) + (size_t) a * L.nc[T_RHON]);
}

// from grad[T_I] (equivariant message adjoint) and grad[T_RHON] to grad[T_X] (layer 1) or grad[T_A]
void chain_bwd(const Layer &L, Grace3LFused::Work &w, int NB)
{
  for (int t : {T_A, T_X, T_R2, T_FA, T_R3, T_FB, T_R4, T_EQ, T_RHO})
    if (L.nf[t] > 0) zero_t(L, w.grad, t, NB);
  if (L.has_eq) {
    const size_t rs = (size_t) NB * L.nc[T_EQ];
    for (int a = 0; a < w.nbat; a++)
      eqnorm_bwd(L.eqn, tv(w, T_EQ) + (size_t) a * L.nc[T_EQ], tg(w, T_I) + (size_t) a * L.nc[T_I],
                 tg(w, T_EQ) + (size_t) a * L.nc[T_EQ], rs, w.xc.data());
  }
  for (int a = 0; a < w.nbat; a++)
    rmsnorm_bwd(L.rhon, tv(w, T_RHO) + (size_t) a * L.nc[T_RHO], tg(w, T_RHON) + (size_t) a * L.nc[T_RHON],
                tg(w, T_RHO) + (size_t) a * L.nc[T_RHO]);
  red_bwd(L, L.rho, w, NB);
  if (L.has_eq) red_bwd(L, L.eq, w, NB);
  prod_red_bwd(L, L.p4, L.r4, w, NB);
  if (!L.p4.fold) fc_bwd(L, L.fb, w, NB);
  prod_red_bwd(L, L.p3, L.r3, w, NB);
  if (!L.p3.fold) fc_bwd(L, L.fa, w, NB);
  prod_red_bwd(L, L.p2, L.r2, w, NB);
  if (L.has_redA) red_bwd(L, L.redA, w, NB);
}

// force of one bond from dE/dx (x = r / rc) and dE/dY[lm]; G = gradient of Y on the unit sphere's extension
inline void bond_fv(double dEdx, const double *gY, const float *G, int nlm, double r, double rc, const double *rh,
                    double *fv)
{
  double gv[3] = {0.0, 0.0, 0.0};
  for (int lm = 0; lm < nlm; lm++) {
    gv[0] += gY[lm] * G[3 * lm];
    gv[1] += gY[lm] * G[3 * lm + 1];
    gv[2] += gY[lm] * G[3 * lm + 2];
  }
  const double dEdr = dEdx / rc, gr = gv[0] * rh[0] + gv[1] * rh[1] + gv[2] * rh[2];
  for (int d = 0; d < 3; d++) fv[d] = dEdr * rh[d] + (gv[d] - gr * rh[d]) / r;
}

// bond vectors of one atom and one channel vector cv for the equivariant SPBF, layout [lm][bond][16]:
// Uc = Y[ly] R[l(ly)], Ic = neighbour message I_j[li] + chem_l0_mask[li] z_proj[species j]. The global message
// rows (and their adjoints) are stored [cv][li][16], so a channel vector of a neighbour is one contiguous run.
constexpr int BPAD = 5;    // bond counts padded to the vmm_t tile width (zero bonds in the pad)
inline int bond_pad(int nba) { return (nba + BPAD - 1) / BPAD * BPAD; }

void bond_vectors(const Layer &L, Grace3LFused::Work &w, const Grace3LFused::Net &n, const float *Iin, int b0,
                  int nba, int nlm, int nlmi, int C, int cv)
{
  const int nl = n.nl, wI = nlmi * C, nbp = bond_pad(nba);
  const size_t nu = (size_t) nbp * nlm * 16, ni = (size_t) nbp * nlmi * 16;
  if (w.Uc.size() < nu) w.Uc.resize(nu);
  if (w.Ic.size() < ni) w.Ic.resize(ni);
  float *Uc = w.Uc.data(), *Ic = w.Ic.data();
  for (int bb = nba; bb < nbp; bb++) {
    for (int ly = 0; ly < nlm; ly++) sst(Uc + ((size_t) ly * nbp + bb) * 16, sv16{});
    for (int li = 0; li < nlmi; li++) sst(Ic + ((size_t) li * nbp + bb) * 16, sv16{});
  }
  for (int bb = 0; bb < nba; bb++) {
    const int b = b0 + bb;
    if (bb + 2 < nba) {    // the neighbour message slice two bonds ahead (random rows of the global array)
      const char *nx = reinterpret_cast<const char *>(Iin + (size_t) w.bj[b + 2] * wI + (size_t) cv * nlmi * 16);
      for (int q = 0; q < nlmi * 64; q += 64) __builtin_prefetch(nx + q);
    }
    const float *R = w.Rb.data() + (size_t) b * nl * C + 16 * cv, *Y = w.Y.data() + (size_t) b * nlm;
    for (int ly = 0; ly < nlm; ly++)
      sst(Uc + ((size_t) ly * nbp + bb) * 16, (sv16{} + Y[ly]) * sld(R + (size_t) L.lt[ly] * C));
    const float *Ij = Iin + (size_t) w.bj[b] * wI + (size_t) cv * nlmi * 16;    // global rows [cv][li][16]
    const sv16 zp = sld(L.zproj.data() + (size_t) n.sp_of_type[w.bt[b]] * C + 16 * cv);
    for (int li = 0; li < nlmi; li++)
      sst(Ic + ((size_t) li * nbp + bb) * 16, sld(Ij + (size_t) li * 16) + L.l0mask[li] * zp);
  }
}

// radial functions of all bonds of a block by the radial MLP (GEMMs over chunks of bonds), with d/dx when deriv:
// Rb[b][l][c] = MLP_{c (lmax+1) + l}(x) env(x) fac[species j][c], x = r / rc
void radial_mlp_block(const Layer &L, Grace3LFused::Work &w, const Grace3LFused::Net &n, int nbond, bool deriv)
{
  constexpr int CH = 256;
  const int nrb = L.nrb, d1 = L.d1, d2 = L.d2, d3 = L.d3, nl = n.nl, C = L.C, p = L.penv;
  auto ens = [](fvec &v, size_t sz) {
    if (v.size() < sz) v.resize(sz);
  };
  ens(w.mT, (size_t) CH * nrb); ens(w.mS0, (size_t) CH * d1); ens(w.mS1, (size_t) CH * d2); ens(w.mO, (size_t) CH * d3);
  if (deriv) { ens(w.mdT, (size_t) CH * nrb); ens(w.mdS0, (size_t) CH * d1); ens(w.mdS1, (size_t) CH * d2); ens(w.mdO, (size_t) CH * d3); }
  for (int c0 = 0; c0 < nbond; c0 += CH) {
    const int nc = std::min(CH, nbond - c0);
    for (int q = 0; q < nc; q++) {
      const double x = w.rr[c0 + q] / w.rc[c0 + q], t = 2.0 * x - 1.0;
      double Tp = 1.0, Tc = t, dTp = 0.0, dTc = 2.0;
      for (int k = 0; k < nrb; k++) {
        w.mT[(size_t) q * nrb + k] = (float) Tc;
        if (deriv) w.mdT[(size_t) q * nrb + k] = (float) dTc;
        const double Tn = 2.0 * t * Tc - Tp, dTn = 2.0 * (2.0 * Tc + t * dTc) - dTp;
        Tp = Tc; dTp = dTc; Tc = Tn; dTc = dTn;
      }
    }
    gemm::gemm(nc, d1, nrb, 1.0f, w.mT.data(), nrb, L.W0c.data(), d1, 0.0f, w.mS0.data(), d1);
    if (deriv) gemm::gemm(nc, d1, nrb, 1.0f, w.mdT.data(), nrb, L.W0c.data(), d1, 0.0f, w.mdS0.data(), d1);
    for (int q = 0; q < nc; q++) {
      const float *ei = L.Ei.data() + (size_t) w.sp * d1, *ej = L.Ej.data() + (size_t) n.sp_of_type[w.bt[c0 + q]] * d1;
      float *s0 = w.mS0.data() + (size_t) q * d1, *ds0 = deriv ? w.mdS0.data() + (size_t) q * d1 : nullptr;
      for (int j = 0; j < d1; j++) {
        const float v = s0[j] + ei[j] + ej[j], sg = 1.0f / (1.0f + std::exp(-v));
        s0[j] = v * sg;
        if (deriv) ds0[j] *= sg * (1.0f + v * (1.0f - sg));
      }
    }
    gemm::gemm(nc, d2, d1, 1.0f, w.mS0.data(), d1, L.W1.data(), d2, 0.0f, w.mS1.data(), d2);
    if (deriv) gemm::gemm(nc, d2, d1, 1.0f, w.mdS0.data(), d1, L.W1.data(), d2, 0.0f, w.mdS1.data(), d2);
    for (int q = 0; q < nc; q++) {
      float *s1 = w.mS1.data() + (size_t) q * d2, *ds1 = deriv ? w.mdS1.data() + (size_t) q * d2 : nullptr;
      for (int j = 0; j < d2; j++) {
        const float v = s1[j] + L.b1[j], sg = 1.0f / (1.0f + std::exp(-v));
        s1[j] = v * sg;
        if (deriv) ds1[j] *= sg * (1.0f + v * (1.0f - sg));
      }
    }
    gemm::gemm(nc, d3, d2, 1.0f, w.mS1.data(), d2, L.W2.data(), d3, 0.0f, w.mO.data(), d3);
    if (deriv) gemm::gemm(nc, d3, d2, 1.0f, w.mdS1.data(), d2, L.W2.data(), d3, 0.0f, w.mdO.data(), d3);
    for (int q = 0; q < nc; q++) {
      const int b = c0 + q;
      const double x = w.rr[b] / w.rc[b];
      const double xp = std::pow(x, p), xp1 = xp * x, xp2 = xp1 * x;
      const float env = (float) (1.0 - 0.5 * (p + 1) * (p + 2) * xp + p * (p + 2) * xp1 - 0.5 * p * (p + 1) * xp2);
      const float denv = p == 0 ? 0.0f : (float) (0.5 * p * (p + 1) * (p + 2) * (-std::pow(x, p - 1) + 2.0 * xp - xp1));
      const float *fac = L.fac.data() + (size_t) n.sp_of_type[w.bt[b]] * C;
      const float *o = w.mO.data() + (size_t) q * d3, *dO = deriv ? w.mdO.data() + (size_t) q * d3 : nullptr;
      float *R = w.Rb.data() + (size_t) b * nl * C, *dR = deriv ? w.dRb.data() + (size_t) b * nl * C : nullptr;
      for (int c = 0; c < C; c++)
        for (int l = 0; l < nl; l++) {
          const float v = o[c * nl + l];
          R[l * C + c] = v * env * fac[c];
          if (deriv) dR[l * C + c] = (dO[c * nl + l] * env + v * denv) * fac[c];
        }
    }
  }
}

}    // namespace

/* ----------------------------------------------------------------------
   one layer over all local blocks: k = 0, 1, 2; bwd: reverse pass (forces, message adjoints)
------------------------------------------------------------------------- */

// UQ (schema v6 as Grace3LFusedKK; projection in fp32, carried in fp32 over the layers; norms and GMM in fp64):
// stream the rho reduce inputs of layer k (model order: instruction, channel, collected function) into the raw
// projection z = B R and the block norm^2; after layer 3 the shared GMM (uq_gmm: normalise, density channels
// [full, rho1, rho2, rho3], nearest centroid, Mahalanobis distance)
void Grace3LFused::uq_layer(int k, Work &w, Outputs &out)
{
  const Net &n = *net;
  const GRACE3LModel &m = *model;
  const Layer &L = n.L[k];
  const Red &R = L.rho;
  const int NB = set.block, RP = m.uq_rp_dim, D = m.uq_feature_dim,
            kb = n.uq_off[k + 1] - n.uq_off[k];    // basis width of layer k
  w.s.ensure((size_t) w.nbat * kb, (size_t) w.nbat * RP);
  for (int a = 0; a < w.nbat; a++) {
    float *bx = w.s.X.data() + (size_t) a * kb;
    double n2 = 0.0;
    int col = 0;
    for (size_t j = 0; j < R.ins.size(); j++) {
      const int src = R.role[j], C = L.nc[src];
      const float *T = tv(w, src);
      for (int c = 0; c < C; c++)
        for (int f : R.ci[j]) {
          const float bv = T[((size_t) f * NB + a) * C + c];
          bx[col++] = bv;
          n2 += (double) bv * bv;
        }
    }
    g->uqn2[(size_t) w.atom[a] * 3 + k] = n2;
  }
  // the built-in kernel, not a linked BLAS: the same z (so gamma / cluster) on every platform and thread count
  gemm::gemm_builtin(w.nbat, RP, kb, 1.0f, w.s.X.data(), kb, n.uq_rp.data() + (size_t) n.uq_off[k] * RP, RP, 0.0f,
                     w.s.Y.data(), RP);
  for (int a = 0; a < w.nbat; a++) {
    const int i = w.atom[a];
    float *z = g->uqz.data() + (size_t) i * RP;
    const float *zy = w.s.Y.data() + (size_t) a * RP;
    vadd(z, zy, RP);
    if (k < 2) continue;

    if ((int) w.uqf.size() < D) { w.uqf.assign(D, 0.0); w.uqd.assign(D, 0.0); }
    double *fe = w.uqf.data();
    for (int d = 0; d < RP; d++) fe[d] = z[d];
    const double *n2b = g->uqn2.data() + (size_t) i * 3;
    const double nrm = std::sqrt(n2b[0] + n2b[1] + n2b[2]);
    const double dens[4] = {nrm, std::sqrt(n2b[0]), std::sqrt(n2b[1]), std::sqrt(n2b[2])};
    fused::uq_gmm(m, n.mu_of_sp[w.sp], fe, w.uqd.data(), nrm, dens, out.gamma[i], out.atomic_sigma[i], out.gmm_cluster[i]);
  }
}

template<class Src>
void Grace3LFused::run_layer(int k, bool bwd, const Src &src, Outputs &out, const ComputeFlags &fl)
{
  Net &n = *net;
  const Layer &L = n.L[k];
  const int NB = set.block, nl = n.nl, nlm = n.nlm, C = L.C;
  const bool store_fwd = store.kept && !fl.energy_only;    // no reverse pass reads the store under energy_only
  const int nblk = g->bstart.size();
  const BondFilter filt{cutsq_rows.data(), n.rcut_type.data(), ntypes + 1, fl.couple.eq ? &fl.couple : nullptr};
  const fused::SHCoefs shc{n.lmax, nlm, n.sh_a.data(), n.sh_b.data(), n.sh_c.data(), n.sh_d.data()};
  double *const eatom = fl.eflag_atom ? out.eatom : nullptr;
  double **const vatom = fl.vflag_atom ? out.vatom : nullptr;
  const float *Iin = k == 1 ? g->I1.data() : (k == 2 ? g->I2.data() : nullptr);
  float *dIin = k == 1 ? g->dI1.data() : (k == 2 ? g->dI2.data() : nullptr);
  const float *dIout = k == 0 ? g->dI1.data() : (k == 1 ? g->dI2.data() : nullptr);
  const int nlmi = L.nlmi, wI = nlmi * C;
  double eng = 0.0;

#pragma omp parallel for schedule(dynamic, 1) reduction(+ : eng) num_threads(nthreads)
  for (int q = 0; q < nblk; q++) {
    const int tid = thread_id();
    Work &w = *work[tid];
    // workspace
    if ((int) w.atom.size() < NB) {
      w.atom.assign(NB, 0);
      w.boff.assign(NB + 1, 0);
    }
    for (int t = 0; t < T_NT; t++) {
      const size_t tsz = n.tmax[t] * NB;
      if (w.val[t].size() < tsz) w.val[t].assign(tsz, 0.0f);
      if (w.grad[t].size() < tsz) w.grad[t].assign(tsz, 0.0f);
    }
    if (w.tmp.size() < n.maxten * NB) w.tmp.assign(n.maxten * NB, 0.0f);
    if (w.xc.size() < n.maxten) w.xc.assign(n.maxten, 0.0f);
    if (w.P.size() < n.maxP) { w.P.assign(n.maxP, 0.0f); w.dP.assign(n.maxP, 0.0f); }
    w.nbat = g->blen[q];
    for (int a = 0; a < w.nbat; a++) w.atom[a] = g->order[g->bstart[q] + a];
    w.ty = src.type[w.atom[0]];
    w.sp = n.sp_of_type[w.ty];
    w.ft = bwd ? g->fthr.of(tid) : nullptr;
    const bool pt = g3_prof_on && tid == 0;
    f3l::prof_on = pt;
    double tc = pt ? prof_now() : 0.0;
    auto cat = [&](int c) {
      if (!pt) return;
      const double t = prof_now();
      g3_cat[k][c] += t - tc;
      tc = t;
    };
    BondGather<Src>::run(w, src, filt, shc, bwd);
    cat(C_BOND);
    const int nbond = w.boff[w.nbat];
    // radial tables
    if (w.Rb.size() < (size_t) nbond * nl * C) w.Rb.resize((size_t) nbond * nl * C);
    if (bwd && w.dRb.size() < (size_t) nbond * nl * C) w.dRb.resize((size_t) nbond * nl * C);
    if (L.use_mlp)
      radial_mlp_block(L, w, n, nbond, bwd);
    else
      for (int b = 0; b < nbond; b++) {
        const int pr = w.sp * n.nsp + n.sp_of_type[w.bt[b]];
        fused::spline_eval(L.spl.data() + pr * L.splpair, L.N, nl, C, w.rr[b] / w.rc[b], w.Rb.data() + (size_t) b * nl * C,
                        bwd ? w.dRb.data() + (size_t) b * nl * C : nullptr);
      }
    cat(C_RAD);

    const bool restore = bwd && k < 2 && store_fwd;
    float *const strow = k < 2 ? g->st[k].data() : nullptr;    // store_fwd rows of this layer
    if (restore) {
      cat(C_SPBF);
      for (int a = 0; a < w.nbat; a++) copy_stored(L, w, NB, a, strow + (size_t) w.atom[a] * g->stw[k], false);
      cat(C_CHAIN);
    } else {
    // ---- SPBF forward
    if (!L.equiv) {
      float *X = tv(w, T_X);
      zero_t(L, w.val, T_X, NB);
      for (int a = 0; a < w.nbat; a++)
        for (int b = w.boff[a]; b < w.boff[a + 1]; b++) {
          const float *R = w.Rb.data() + (size_t) b * nl * C, *Y = w.Y.data() + (size_t) b * nlm;
          for (int lm = 0; lm < nlm; lm++) {
            const float y = Y[lm], *r = R + (size_t) L.lt[lm] * C;
            float *x = X + ((size_t) lm * NB + a) * C;
#pragma omp simd
            for (int c = 0; c < C; c++) x[c] += y * r[c];
          }
        }
    } else {
      float *A = tv(w, T_A);
      const int NV = C / 16;
      for (int a = 0; a < w.nbat; a++) {
        const int b0 = w.boff[a], nba = w.boff[a + 1] - b0;
        float *P = w.P.data();    // one channel vector: [ly][li][16]
        for (int cv = 0; cv < NV; cv++) {
          if (nba > 0) {
            bond_vectors(L, w, n, Iin, b0, nba, nlm, nlmi, C, cv);
            const int nbp = bond_pad(nba);
            vmm_t<5, 4>(nlm, nlmi, nba, w.Uc.data(), (size_t) nbp * 16, 16, w.Ic.data(), 16, (size_t) nbp * 16, P,
                        (size_t) nlmi * 16, 16);
          } else
            std::memset(P, 0, (size_t) nlm * nlmi * 16 * sizeof(float));
          for (int f = 0; f < L.nfA; f++) {
            sv16 o = sv16{};
            for (int t = L.cg_off[f]; t < L.cg_off[f + 1]; t++) o += L.cg_v[t] * sld(P + (size_t) L.cg_p[t] * 16);
            sst(A + ((size_t) f * NB + a) * C + 16 * cv, o);
          }
        }
      }
    }
    cat(C_SPBF);
    chain_fwd(L, w, NB);
    if (uq_on && (!bwd || k == 2)) uq_layer(k, w, out);
    cat(C_CHAIN);
    if (!bwd && k < 2 && store_fwd)
      for (int a = 0; a < w.nbat; a++) copy_stored(L, w, NB, a, strow + (size_t) w.atom[a] * g->stw[k], true);
    }

    // ---- layer outputs
    const int wr = g->wr;
    if (k < 2 && !bwd) {
      float *Iout = k == 0 ? g->I1.data() : g->I2.data(), *rho = k == 0 ? g->rho1.data() : g->rho2.data();
      const int wo = k == 0 ? g->w1 : g->w2, nfI = L.nf[T_I], cI = L.nc[T_I];
      for (int a = 0; a < w.nbat; a++) {
        const int i = w.atom[a];
        float *Io = Iout + (size_t) i * wo;
        for (int f = 0; f < nfI; f++)
          for (int cv = 0; cv < cI / 16; cv++)
            std::memcpy(Io + ((size_t) cv * nfI + f) * 16, tv(w, T_I) + ((size_t) f * NB + a) * cI + 16 * cv, 16 * sizeof(float));
        std::memcpy(rho + (size_t) i * wr, tv(w, T_RHON) + (size_t) a * wr, wr * sizeof(float));
      }
    }
    if (k == 2) {
      // readout (fp64): E = (MLP(h) + lin) oscale + shift; dE/drho_norm (same for all three layers) -> dRn
      for (int a = 0; a < w.nbat; a++) {
        const int i = w.atom[a];
        const float *r1 = g->rho1.data() + (size_t) i * wr, *r2 = g->rho2.data() + (size_t) i * wr;
        const float *r3 = tv(w, T_RHON) + (size_t) a * wr;
        const int nin = n.e_nin, nh = n.e_nh;
        double hin[128], s[256], act[256];
        const double lin = (double) r1[0] + r2[0] + r3[0];
        for (int c = 0; c < nin; c++) hin[c] = (double) r1[c + 1] + r2[c + 1] + r3[c + 1];
        for (int j = 0; j < nh; j++) s[j] = 0.0;
        for (int c = 0; c < nin; c++) {
          const double x = hin[c];
          const double *row = n.eW0.data() + (size_t) c * nh;
          for (int j = 0; j < nh; j++) s[j] += x * row[j];
        }
        double e = 0.0;
        for (int j = 0; j < nh; j++) {
          const double v = s[j] * n.en0;
          s[j] = v;
          act[j] = n.e_act == 1 ? std::tanh(v) : v / (1.0 + std::exp(-v));
          e += n.eW1[j] * act[j];
        }
        e *= n.en1;
        const double Ea = (e + lin) * n.oscale + n.shift_sp[w.sp];
        w.E[a] = Ea;
        eng += Ea;
        if (eatom) eatom[i] += Ea;
        if (!bwd) continue;
        float *d = g->dRn.data() + (size_t) i * wr;
        d[0] = (float) n.oscale;
        double ds[256];
        for (int j = 0; j < nh; j++) {
          double dact;
          if (n.e_act == 1) {
            const double t = act[j];
            dact = 1.0 - t * t;
          } else {
            const double sg = 1.0 / (1.0 + std::exp(-s[j]));
            dact = sg * (1.0 + s[j] * (1.0 - sg));
          }
          ds[j] = n.oscale * n.en1 * n.eW1[j] * dact * n.en0;
        }
        for (int c = 0; c < nin; c++) {
          double v = 0.0;
          for (int j = 0; j < nh; j++) v += n.eW0[(size_t) c * nh + j] * ds[j];
          d[c + 1] = (float) v;
        }
      }
    }
    if (!bwd) continue;

    // ---- reverse
    for (int a = 0; a < w.nbat; a++) {
      const int i = w.atom[a];
      std::memcpy(tg(w, T_RHON) + (size_t) a * wr, g->dRn.data() + (size_t) i * wr, wr * sizeof(float));
      if (L.has_eq) {
        const int nfI = L.nf[T_I], cI = L.nc[T_I];
        for (int f = 0; f < nfI; f++)
          for (int cv = 0; cv < cI / 16; cv++)
            std::memcpy(tg(w, T_I) + ((size_t) f * NB + a) * cI + 16 * cv,
                        dIout + (size_t) i * nfI * cI + ((size_t) cv * nfI + f) * 16, 16 * sizeof(float));
      }
    }
    cat(C_CHAIN);
    chain_bwd(L, w, NB);
    cat(C_CHAINB);

    double gY[fused::MAX_NLM], fv[3];
    if (!L.equiv) {
      const float *dX = tg(w, T_X);
      for (int a = 0; a < w.nbat; a++) {
        const int i = w.atom[a];
        for (int b = w.boff[a]; b < w.boff[a + 1]; b++) {
          const float *R = w.Rb.data() + (size_t) b * nl * C, *dR = w.dRb.data() + (size_t) b * nl * C;
          const float *Y = w.Y.data() + (size_t) b * nlm;
          double dEdx = 0.0;
          for (int lm = 0; lm < nlm; lm++) {
            const float *d = dX + ((size_t) lm * NB + a) * C, *r = R + (size_t) L.lt[lm] * C, *dr = dR + (size_t) L.lt[lm] * C;
            float sy = 0.0f, sx = 0.0f;
#pragma omp simd reduction(+ : sy, sx)
            for (int c = 0; c < C; c++) {
              sy += d[c] * r[c];
              sx += d[c] * dr[c];
            }
            gY[lm] = sy;
            dEdx += (double) Y[lm] * sx;
          }
          bond_fv(dEdx, gY, w.G.data() + (size_t) b * nlm * 3, nlm, w.rr[b], w.rc[b], &w.rh[3 * b], fv);
          const double rv = w.cpl ? couple_force(&w.cp[3 * b], w.rr[b], &w.rh[3 * b], fv, w.dedl) : w.rr[b];
          fused::add_force(w.ft, i, w.bj[b], fv);
          if (vatom) fused::add_vatom(vatom, i, w.bj[b], &w.rh[3 * b], rv, fv);
          if (w.want_vir) add_bond_virial(w.vir, &w.rh[3 * b], rv, fv);
        }
      }
    } else {
      const float *dA = tg(w, T_A);
      const int np = nlm * nlmi, NV = C / 16;
      for (int a = 0; a < w.nbat; a++) {
        const int i = w.atom[a];
        float *dP = w.dP.data();    // one channel vector: [ly][li][16]
        const int b0 = w.boff[a], nba = w.boff[a + 1] - b0;
        if (nba == 0) continue;
        const int nbp = bond_pad(nba);
        if (w.dUc.size() < (size_t) nbp * nlm * 16) w.dUc.resize((size_t) nbp * nlm * 16);
        if (w.dIc.size() < (size_t) nbp * nlmi * 16) w.dIc.resize((size_t) nbp * nlmi * 16);
        if (w.sy.size() < (size_t) 2 * nba * nlm) w.sy.resize((size_t) 2 * nba * nlm);
        sv16 *sy = w.sy.data(), *sx = sy + (size_t) nba * nlm;
        for (size_t q = 0; q < (size_t) 2 * nba * nlm; q++) sy[q] = sv16{};
        for (int cv = 0; cv < NV; cv++) {
          {
            // the atom's dA slice, contiguous [f][16], then dP[p] = sum_{(f, v) in column p} v dA[f]
            if (w.dAc.size() < (size_t) L.nfA * 16) w.dAc.resize((size_t) L.nfA * 16);
            float *dAc = w.dAc.data();
            for (int f = 0; f < L.nfA; f++) sst(dAc + (size_t) f * 16, sld(dA + ((size_t) f * NB + a) * C + 16 * cv));
            for (int p = 0; p < np; p++) {
              sv16 o = sv16{};
              for (int t = L.cgt_off[p]; t < L.cgt_off[p + 1]; t++) o += L.cgt_v[t] * sld(dAc + (size_t) L.cgt_f[t] * 16);
              sst(dP + (size_t) p * 16, o);
            }
          }
          cat(C_B_DP);
          bond_vectors(L, w, n, Iin, b0, nba, nlm, nlmi, C, cv);
          cat(C_B_VEC);
          float *dUc = w.dUc.data(), *dIc = w.dIc.data();
          // dI[li][b] = sum_ly dP[ly][li] U[ly][b],  dU[ly][b] = sum_li dP[ly][li] I[li][b]
          vmm_t<5, 5>(nlmi, nbp, nlm, dP, 16, (size_t) nlmi * 16, w.Uc.data(), (size_t) nbp * 16, 16, dIc, (size_t) nbp * 16, 16);
          vmm_t<5, 5>(nlm, nbp, nlmi, dP, (size_t) nlmi * 16, 16, w.Ic.data(), (size_t) nbp * 16, 16, dUc, (size_t) nbp * 16, 16);
          cat(C_B_VMM);
          for (int bb = 0; bb < nba; bb++) {
            const int b = b0 + bb, j = w.bj[b];
            const float *R = w.Rb.data() + (size_t) b * nl * C + 16 * cv, *dR = w.dRb.data() + (size_t) b * nl * C + 16 * cv;
            for (int ly = 0; ly < nlm; ly++) {
              const sv16 du = sld(dUc + ((size_t) ly * nbp + bb) * 16);
              sy[(size_t) bb * nlm + ly] += du * sld(R + (size_t) L.lt[ly] * C);
              sx[(size_t) bb * nlm + ly] += du * sld(dR + (size_t) L.lt[ly] * C);
            }
            fused::SpinLock lk(g->lock[j]);
            float *dst = dIin + (size_t) j * wI + (size_t) cv * nlmi * 16;
            for (int li = 0; li < nlmi; li++) {
              float *d = dst + (size_t) li * 16;
              sst(d, sld(d) + sld(dIc + ((size_t) li * nbp + bb) * 16));
            }
          }
          cat(C_B_BOND);
        }
        for (int bb = 0; bb < nba; bb++) {
          const int b = b0 + bb, j = w.bj[b];
          const float *Y = w.Y.data() + (size_t) b * nlm;
          double dEdx = 0.0;
          for (int ly = 0; ly < nlm; ly++) {
            const sv16 vy = sy[(size_t) bb * nlm + ly], vx = sx[(size_t) bb * nlm + ly];
            double ty = 0.0, tx = 0.0;
            for (int q = 0; q < 16; q++) {
              ty += vy[q];
              tx += vx[q];
            }
            gY[ly] = ty;
            dEdx += (double) Y[ly] * tx;
          }
          bond_fv(dEdx, gY, w.G.data() + (size_t) b * nlm * 3, nlm, w.rr[b], w.rc[b], &w.rh[3 * b], fv);
          const double rv = w.cpl ? couple_force(&w.cp[3 * b], w.rr[b], &w.rh[3 * b], fv, w.dedl) : w.rr[b];
          fused::add_force(w.ft, i, j, fv);
          if (vatom) fused::add_vatom(vatom, i, j, &w.rh[3 * b], rv, fv);
          if (w.want_vir) add_bond_virial(w.vir, &w.rh[3 * b], rv, fv);
        }
      }
    }
    cat(C_SPBFB);
  }
  if (k == 2) out.energy += eng;
  if (g3_prof_on) {
#pragma omp parallel num_threads(nthreads)
    {
      if (thread_id() == 0) {
        g3_couple_tot += f3l::prof_couple;
        f3l::prof_couple = 0.0;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void Grace3LFused::grow(int nmax_new)
{
  if (nmax_new <= nmax) return;
  nmax = 0;    // committed after the allocations: a failed one (std::bad_alloc) leaves the next call to grow again
  const size_t n = (size_t) nmax_new;
  g->I1.assign(n * g->w1, 0.0f, nthreads);
  g->dI1.assign(n * g->w1, 0.0f, nthreads);
  g->I2.assign(n * g->w2, 0.0f, nthreads);
  g->dI2.assign(n * g->w2, 0.0f, nthreads);
  g->rho1.assign(n * g->wr, 0.0f, nthreads);
  g->rho2.assign(n * g->wr, 0.0f, nthreads);
  g->dRn.assign(n * g->wr, 0.0f, nthreads);
  // store_fwd: decided here, once per size (the available memory changes between calls, the code path should not)
  for (int k = 0; k < 2; k++) g->st[k].clear();
  store = StoreInfo{};
  if (set.store_fwd) {
    store.need_gb = (double) n * (g->stw[0] + g->stw[1]) * sizeof(float) / 1e9;
    store.limit_gb = fused::store_budget_gb(set.store_fwd_max_gb, set.ranks_per_node);
    if (store.need_gb <= store.limit_gb) {
      for (int k = 0; k < 2; k++) g->st[k].assign(n * g->stw[k], 0.0f, nthreads);
      store.kept = true;
    } else {
      auto gb = [](double x) { return std::round(x * 10) / 10; };
      log.warn(str("GRACE-3L: store_fwd would need ", gb(store.need_gb), " GB > limit ", gb(store.limit_gb),
                   " GB; recomputing layers 1-2 in the reverse pass instead"));
    }
  }
  g->lock.reset(n);
  nmax = nmax_new;
}

/* ---------------------------------------------------------------------- */

template<class Src>
void Grace3LFused::compute(int inum, const int *ilist, int n_nodes, int nmax_nodes, const Src &src, Exchange &ex,
                           Outputs &out, const ComputeFlags &fl, int nth)
{
  if (!net) throw Error("GRACE-3L: setup_types() before compute()");
  if (fl.couple.eq && (fl.couple.eq < 1 || fl.couple.eq > 4 || !fl.couple.mask))
    throw Error("GRACE-3L: two-group coupling needs eq 1..4 and the atom mask");
  if (fl.uq && !uq_ready) throw Error("GRACE-3L: enable_uq() before a compute with ComputeFlags::uq");
  ensure_threads(std::max(nth, 1));
  grow(nmax_nodes);
  const int NB = set.block;
  double t0 = prof_now();
  auto tick = [&](int q) {
    if (!g3_prof_on) return;
    const double t = prof_now();
    g3_prof[q] += t - t0;
    t0 = t;
  };

  // blocks: central atoms sorted by type, the given order kept within a type
  const int *type = src.type;
  g->order.resize(inum);
  for (int p = 0; p < inum; p++) g->order[p] = ilist ? ilist[p] : p;
  std::stable_sort(g->order.begin(), g->order.end(), [&](int a, int b) { return type[a] < type[b]; });
  g->bstart.clear();
  g->blen.clear();
  for (int p = 0; p < inum;) {
    const int t = type[g->order[p]];
    if (type2mu[t] < 0)    // its per-type rows are not built (pair_style hybrid keeps it out of the list)
      throw Error(str("GRACE-3L: atom ", g->order[p], " of unmapped type ", t, " in the atom list"));
    int e = p;
    while (e < inum && e - p < NB && type[g->order[e]] == t) e++;
    g->bstart.push_back(p);
    g->blen.push_back(e - p);
    p = e;
  }
  if (!fl.energy_only) g->fthr.ensure(nthreads, nmax_nodes);
  uq_on = fl.uq;
  if (uq_on && inum > 0 && !(out.gamma && out.atomic_sigma && out.gmm_cluster))    // no atoms: nothing written
    throw Error("GRACE-3L: UQ requested without the gamma / atomic_sigma / gmm_cluster outputs");
  if (uq_on) {
    g->uqz.assign((size_t) n_nodes * model->uq_rp_dim, 0.0f, nthreads);
    g->uqn2.assign((size_t) n_nodes * 3, 0.0, nthreads);
  }
  for (auto &wp : work) {
    wp->want_vir = fl.virial && !fl.energy_only;
    wp->cpl = fl.couple.eq ? &fl.couple : nullptr;
    wp->dedl = 0.0;
    for (int k = 0; k < 6; k++) wp->vir[k] = 0.0;
  }

  run_layer(0, false, src, out, fl);
  tick(P_S1);
  ex.forward(0);
  tick(P_S2);
  run_layer(1, false, src, out, fl);
  tick(P_S3);
  ex.forward(1);
  tick(P_S4);
  if (fl.energy_only) {    // readout and UQ of layer 3 only: no reverse pass, forces and virials untouched
    run_layer(2, false, src, out, fl);
    tick(P_S5);
    throw_zero_bond(work);
    return;
  }
#pragma omp parallel for schedule(static) num_threads(nthreads)
  for (int i = 0; i < n_nodes; i++) std::memset(g->dI2.data() + (size_t) i * g->w2, 0, g->w2 * sizeof(float));
  run_layer(2, true, src, out, fl);
  tick(P_S5);
  ex.reverse(1);
  tick(P_S6);
#pragma omp parallel for schedule(static) num_threads(nthreads)
  for (int i = 0; i < n_nodes; i++) std::memset(g->dI1.data() + (size_t) i * g->w1, 0, g->w1 * sizeof(float));
  run_layer(1, true, src, out, fl);
  tick(P_S7);
  ex.reverse(0);
  tick(P_S8);
  run_layer(0, true, src, out, fl);
  tick(P_S9);
  throw_zero_bond(work);
  if (n_nodes > 0) g->fthr.reduce(out.f, n_nodes);
  tick(P_FRED);
  if (fl.virial)
    for (auto &wp : work)
      for (int k = 0; k < 6; k++) out.virial[k] += wp->vir[k];
  if (fl.couple.eq && !fl.energy_only)
    for (auto &wp : work) out.dedl += wp->dedl;
}

template void Grace3LFused::compute<LammpsBonds>(int, const int *, int, int, const LammpsBonds &, Exchange &, Outputs &,
                                                 const ComputeFlags &, int);
template void Grace3LFused::compute<CsrBonds>(int, const int *, int, int, const CsrBonds &, Exchange &, Outputs &,
                                              const ComputeFlags &, int);

}    // namespace grace

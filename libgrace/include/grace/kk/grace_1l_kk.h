// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// GRACE-1L Kokkos engine (from the LAMMPS style grace/1l/kk, lammps-ace feature/grace-runtime 07b8fb50d4;
// contributing author Yury Lysogorskiy, ICAMS). Independent of LAMMPS: the consumer supplies positions, types, a
// full neighbour list and the output views through RunInput, and the Kokkos view types through Traits
// (grace/kk/traits.h). Implementation: grace/kk/grace_1l_kk_impl.h, included by one translation unit per consumer.
#pragma once

#include "grace/coupling.h"
#include "grace/error.h"
#include "grace/model_1l.h"
#include "grace/type_map.h"
#include "grace/kk/support.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_ScatterView.hpp>
#ifdef KOKKOS_ENABLE_CUDA
#include <cublas_v2.h>
#include "grace/kk/cublas_guard.h"
#endif

#include <memory>
#include <type_traits>
#include <vector>

namespace grace {
namespace kk {

// Compile-time max dimensions for stack arrays in CUDA kernels
static constexpr int GRACE1L_MAX_MLP_DIM = 64;   // max hidden dim for MLP layers
static constexpr int GRACE1L_MAX_NRADMAX = 32;    // max radial basis functions
static constexpr int GRACE1L_MAX_MLP_LAYERS = 8;  // max MLP depth
static constexpr int GRACE1L_UQ_MAX_RP_DIM = 132;   // bounds the FULL UQ feature dim D = rp_dim + n_density (z/f/delta stack arrays); 132 = 128 rp_dim + up to 3 density channels + 1 slack; a larger model fails the D > MAX guard at load

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
class Grace1LKK {
 public:
  // Kernel tags
  struct TagComputeNeigh{};
  struct TagComputeRadialBasis{};
  struct TagComputeMLPRadial{};
  struct TagComputeAi{};
  // FC, Product, ReduceN dispatched via inline lambdas in compute()
  struct TagComputeMLPEnergy{};
  struct TagComputeDerivative{};
  struct TagComputeUQ{};   // extrapolation grade (gamma) + error model — forward only
#ifdef KOKKOS_ENABLE_CUDA
  struct TagMLPAssemble{};
  struct TagMLPActDeriv{};
  struct TagMLPScatterR{};
  struct TagUQBmat{};    // materialize dense Bmat[D_basis x subN] + ||B||^2 (cuBLAS UQ)
  struct TagUQFinish{};  // normalize proj + density + GMM from cuBLAS projection
#endif

  // Force computation tags (Stage 2)
  template<int NEIGHFLAG, int EVFLAG>
  struct TagComputeForce{};

  typedef DeviceType device_type;
  using EV = typename Traits::ev_type;    // energy / virial reduction type (LAMMPS: EV_FLOAT)
  using KKF = typename Traits::kk_float;   // position precision (LAMMPS: KK_FLOAT)
  using KKA = typename Traits::kk_acc_float;    // force / energy accumulation precision (LAMMPS: KK_ACC_FLOAT)
  typedef EV value_type;
  using model_type = GRACE1LModel;
  using traits_type = Traits;

  // Precision typedefs
  using GeomScalar = GeomScalarT;
  using NNScalar = NNScalarT;

  Grace1LKK();
  virtual ~Grace1LKK();
  Grace1LKK(const Grace1LKK &) = default;    // Kokkos copies the engine as the kernel functor

  // ---- engine API (the LAMMPS glue and the Python module call these) ----
  // model: loaded GRACE-1L model; type2mu[t] (t = 1..ntypes): model element of atom type t, -1 unmapped
  void set_model(std::shared_ptr<GRACE1LModel> model, const std::vector<int> &type2mu);
  virtual void setup();                   // scalars, harmonics, device weights (LAMMPS: init_style)
  // another species set on a set-up engine: only the type tables (atom type -> model element, pair cutoffs and the
  // fused engines' per-type packed weights) are rebuilt; weights, spline tables and buffers stay
  void set_types(const std::vector<int> &type2mu);
  double cutoff(int mu_i, int mu_j) const;    // pair cutoff of model elements
  struct RunInput {
    typename Traits::t_kkfloat_1d_3_lr_randomread x;
    typename Traits::t_kkacc_1d_3 f;
    typename Traits::t_int_1d_randomread type, ilist, numneigh;
    typename Traits::t_neighbors_2d neighbors;
    int inum = 0, nlocal = 0, ntypes = 0;
    int eflag_global = 0, eflag_atom = 0, vflag_global = 0, vflag_atom = 0, cvflag_atom = 0;
    bool energy_only = false, uq = false;
    // two-group coupling (grace/coupling.h): couple_eq 0 off; mask [nall] (local atoms then ghosts) with couple_eq
    int couple_eq = 0, couple_bit1 = 0, couple_bit2 = 0;
    double couple_lambda = 0.0;
    typename Traits::t_int_1d_randomread mask;
    typename Traits::t_kkacc_1d eatom;
    typename Traits::t_kkacc_1d_6 vatom;
    typename Traits::t_kkacc_1d_9 cvatom;
  };
  struct RunOutput {
    double eng_vdwl = 0.0;          // energy of the inum atoms (summed in KKA, as LAMMPS eng_vdwl)
    double virial[6] = {0, 0, 0, 0, 0, 0};
    double dedl = 0.0;              // coupling: dE/dlambda of the inum atoms' bonds (with the backward pass)
  };
  // forces added to f, per-atom energy / virial added to eatom / vatom / cvatom; global energy and virial returned;
  // with uq (and a usable UQ head) the per-atom gamma / sigma / cluster in uq_gamma() ... [nlocal]
  virtual RunOutput run(const RunInput &in);
  const Kokkos::View<double *, DeviceType> &uq_gamma() const { return d_gamma; }
  const Kokkos::View<double *, DeviceType> &uq_sigma() const { return d_sigma; }
  const Kokkos::View<double *, DeviceType> &uq_cluster() const { return d_gmm_cluster; }
  bool uq_available() const { return has_uq; }
  grace::Logger log;
  // atoms per chunk (pair_style keyword chunksize): > 0 fixed, 0 one chunk, -1 auto (chunk_auto, from the free GPU
  // memory: one chunk while it fits)
  int chunksize = -1;
  int ranks_per_device = 1;    // chunksize auto: processes sharing this GPU (each takes its share of the free memory)
  int auto_chunk_size() const { return chunk_auto; }    // chunksize auto: the chunk chosen last (0: none yet)
  bool debug_no_energy_only_calc = false;    // energy-only requests still run the backward pass (debug)
  // two-group coupling (RunInput::couple_*), set by run(); couple_supported: engines that apply it
  int couple_eq = 0, couple_bit1 = 0, couple_bit2 = 0;
  double couple_lambda = 0.0;
  typename Traits::t_int_1d_randomread d_mask;
  Kokkos::View<double, DeviceType> d_dedl;
  bool couple_supported = false;
  double memory_usage();

  // Kernel operators (to be implemented)
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeNeigh, const typename Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeRadialBasis, const typename Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeMLPRadial, const typename Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadial>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeAi, const typename Kokkos::TeamPolicy<DeviceType, TagComputeAi>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeMLPEnergy, const typename Kokkos::TeamPolicy<DeviceType, TagComputeMLPEnergy>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeDerivative, const typename Kokkos::TeamPolicy<DeviceType, TagComputeDerivative>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeUQ, const typename Kokkos::TeamPolicy<DeviceType, TagComputeUQ>::member_type& team) const;

#ifdef KOKKOS_ENABLE_CUDA
  KOKKOS_INLINE_FUNCTION
  void operator() (TagUQBmat, const typename Kokkos::TeamPolicy<DeviceType, TagUQBmat>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagUQFinish, const typename Kokkos::TeamPolicy<DeviceType, TagUQFinish>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagMLPAssemble, const int b) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagMLPActDeriv, const int idx) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagMLPScatterR, const int idx) const;
#endif

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeForce<NEIGHFLAG,EVFLAG>, const int& ii) const;

  template<int NEIGHFLAG, int EVFLAG>
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeForce<NEIGHFLAG,EVFLAG>, const int& ii, EV&) const;

 protected:
  int inum, maxneigh, chunk_size, chunk_offset;
  // chunksize auto: the chunk chosen for up to auto_inum atoms and auto_maxneigh neighbours (10 % headroom, so a
  // slowly growing system does not re-decide every step)
  int chunk_auto = 0, auto_inum = 0, auto_maxneigh = 0;
  int chunk_cap() const { const int c = chunksize < 0 ? chunk_auto : chunksize; return c > 0 ? c : inum; }
  void choose_chunk()
  {
    if (chunksize >= 0 || (chunk_auto > 0 && inum <= auto_inum && maxneigh <= auto_maxneigh)) return;
    auto_inum = inum + inum / 10;
    auto_maxneigh = maxneigh + maxneigh / 10 + 1;
    realloc_all = true;    // the buffers of the calibration size go; run() grows them for the chunk
    chunk_auto = grace::kk::auto_chunk<DeviceType>(auto_inum, auto_maxneigh, ranks_per_device,
                                                   [this](int a, int m) { grow(a, m); }, [] { return 0.0; }, log,
                                                   "GRACE-1L/KK");
  }
  int host_flag;
  int eflag = 0, vflag = 0;
  int eflag_global = 0, eflag_atom = 0, vflag_global = 0, vflag_atom = 0, cvflag_atom = 0, nlocal = 0;
  double eng_vdwl = 0.0, virial[6] = {0, 0, 0, 0, 0, 0};
  bool setup_done = false;    // setup() after the last set_model(): run() refuses otherwise
  void set_type_map(const std::vector<int> &type2mu);    // h_type2model, d_map, cutsq
  virtual void retype() {}    // type-dependent parts of the derived engines (after set_type_map)
  int neighflag;

  // Architecture params (copied from GRACE1LModel for device access)
  int nelements, lmax, nradmax, nradbase, embedding_size;
  int radial_basis_p;
  GeomScalar rcut;

  // LAMMPS arrays
  typename Traits::t_neighbors_2d d_neighbors;
  typename Traits::t_int_1d_randomread d_ilist;
  typename Traits::t_int_1d d_ilist_sorted;      // d_ilist ordered by atom type (multi-element)
  Kokkos::View<long long *, DeviceType> d_type_key;
  typename Traits::t_int_1d_randomread d_numneigh;

  typename Traits::t_kkacc_1d d_eatom;
  typename Traits::t_kkacc_1d_6 d_vatom;
  typename Traits::t_kkacc_1d_9 d_cvatom;

  typename Traits::t_kkfloat_1d_3_lr_randomread x;
  typename Traits::t_kkacc_1d_3 f;
  typename Traits::t_int_1d_randomread type;

  typedef Kokkos::DualView<KKF**, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq, k_scale;
  typedef Kokkos::View<KKF**, DeviceType> t_fparams;
  t_fparams d_cutsq, d_scale;
  typename Traits::t_int_1d d_map;

 public:
  // the type tables of the current species set (views shared, not copied): a caller keeps them and hands them back
  // with use_type_tables() to switch to that species set again without rebuilding (set_types builds new ones)
  struct TypeTables {
    std::vector<int> type2model;
    typename Traits::t_int_1d map;
    tdual_fparams cutsq;
    std::shared_ptr<void> derived;    // the derived engine's per-type tables
  };
  TypeTables type_tables() const { return {h_type2model, d_map, k_cutsq, save_derived_types()}; }
  // the per-atom buffers shrunk to one atom; weights and type tables stay, the next run() grows the buffers again
  void release_buffers()
  {
    if (!setup_done) return;
    realloc_all = true;
    grow(1, 1);
    realloc_all = true;
    chunk_auto = 0;
  }
  void use_type_tables(const TypeTables &t)
  {
    h_type2model = t.type2model;
    d_map = t.map;
    k_cutsq = t.cutsq;
    d_cutsq = k_cutsq.template view<DeviceType>();
    use_derived_types(t.derived);
  }

 protected:
  virtual std::shared_ptr<void> save_derived_types() const { return nullptr; }
  virtual void use_derived_types(const std::shared_ptr<void> &) {}

  int need_dup;
  using KKDeviceType = typename Traits::kk_device;

  template<typename DataType, typename Layout>
  using DupScatterView = Kokkos::Experimental::ScatterView<DataType, Layout, KKDeviceType, Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>;
  template<typename DataType, typename Layout>
  using NonDupScatterView = Kokkos::Experimental::ScatterView<DataType, Layout, KKDeviceType, Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>;

  DupScatterView<KKA*[3], typename Traits::layout_f> dup_f;
  DupScatterView<KKA*[6], typename Traits::layout_vatom> dup_vatom;
  DupScatterView<KKA*[9], typename Traits::layout_cvatom> dup_cvatom;

  NonDupScatterView<KKA*[3], typename Traits::layout_f> ndup_f;
  NonDupScatterView<KKA*[6], typename Traits::layout_vatom> ndup_vatom;
  NonDupScatterView<KKA*[9], typename Traits::layout_cvatom> ndup_cvatom;


  // ---------- Kokkos View typedefs ----------
  typedef Kokkos::View<int*, DeviceType> t_int_1d;
  typedef Kokkos::View<int**, DeviceType> t_int_2d;
  typedef Kokkos::View<GeomScalar*, DeviceType> t_geom_1d;
  typedef Kokkos::View<GeomScalar**, DeviceType> t_geom_2d;
  typedef Kokkos::View<GeomScalar***, DeviceType> t_geom_3d;
  typedef Kokkos::View<GeomScalar**[3], DeviceType> t_geom_3d3;
  typedef Kokkos::View<GeomScalar****, DeviceType> t_geom_4d;
  typedef Kokkos::View<NNScalar*, DeviceType> t_nn_1d;
  typedef Kokkos::View<NNScalar**, DeviceType> t_nn_2d;
  typedef Kokkos::View<NNScalar***, DeviceType> t_nn_3d;
  typedef Kokkos::View<NNScalar****, DeviceType> t_nn_4d;

  // ---------- Per-chunk intermediate arrays ----------

  // Neighbor list (short, filtered by cutoff)
  t_int_1d d_ncount;           // [chunk_size] neighbor count per atom
  t_int_2d d_nearest;          // [chunk_size, maxneigh] neighbor indices
  t_geom_2d d_rnorms;          // [chunk_size, maxneigh] distances
  t_geom_3d3 d_rhats;          // [chunk_size, maxneigh, 3] unit vectors
  t_int_1d d_mu_i;             // [chunk_size] element type of center atom
  t_int_2d d_mu_j;             // [chunk_size, maxneigh] element type of neighbor

  // Radial basis g_k(r) and derivatives
  t_geom_3d d_radial_basis;    // [chunk_size, maxneigh, nradbase]
  t_geom_3d d_dradial_basis;   // [chunk_size, maxneigh, nradbase] (for forces)

  // MLP radial output and derivative intermediate
  t_nn_4d d_R_nl;              // [chunk_size, maxneigh, nradmax, lmax+1]
  t_nn_4d d_dR_nl;             // [chunk_size, maxneigh, nradmax, lmax+1]
  t_nn_3d d_h2;                // [chunk_size, maxneigh, mlp_hidden] (MLP last hidden output)
  t_nn_3d d_dh2;               // [chunk_size, maxneigh, mlp_hidden] (MLP layer 1 derivative)

  // Single particle basis A
  t_nn_3d d_A;                 // [chunk_size, (lmax+1)^2, nradmax]

  // FC outputs
  t_nn_3d d_A1;                // [chunk_size, n_funcs_A1, n_out_fc]
  t_nn_3d d_AA;                // [chunk_size, n_funcs_AA, n_out_fc]
  t_nn_3d d_AA1;               // [chunk_size, n_funcs_AA1, n_out_fc]
  t_nn_3d d_AAA;               // [chunk_size, n_funcs_AAA, n_out_fc]
  t_nn_3d d_AA2;               // [chunk_size, n_funcs_AA2, n_out_fc]
  t_nn_3d d_AAAA;              // [chunk_size, n_funcs_AAAA, n_out_fc]

  // Density and energy
  t_nn_2d d_rho;               // [chunk_size, rho_n_out]
  Kokkos::View<KKA*, DeviceType> d_e_atom;    // [chunk_size], fp64 (KKA) as the energy sum

  // Per-bond force (for Stage 2)
  t_geom_3d3 d_f_ij;           // [chunk_size, maxneigh, 3]

  // Adjoint views for backward pass (Stage 2: Forces)
  t_nn_2d d_rho_adj;            // [chunk_size, rho_n_out]
  t_nn_3d d_AAAA_adj;           // [chunk_size, fc_n_out, n_funcs_AAAA]
  t_nn_3d d_AA2_adj;            // [chunk_size, fc_n_out, n_funcs_AA]
  t_nn_3d d_AAA_adj;            // [chunk_size, fc_n_out, n_funcs_AAA]
  t_nn_3d d_AA1_adj;            // [chunk_size, fc_n_out, n_funcs_AA]
  t_nn_3d d_AA_adj;             // [chunk_size, fc_n_out, n_funcs_AA]
  t_nn_3d d_A1_adj;             // [chunk_size, fc_n_out, n_funcs_A1]
  t_nn_3d d_A_adj;              // [chunk_size, nradmax, nlm] ("weights")

  // ---------- Constant device arrays (weights) ----------

  // Chemical embedding
  t_nn_2d d_chem_embed;        // [n_elements, embedding_size]

  // A indicator transform
  t_nn_2d d_A_lin_W;           // [embedding_size, nradmax]
  NNScalar A_lin_norm;
  GeomScalar inv_avg_n_neigh;

  // Precomputed z_tr[mu,n] = A_lin_norm * inv_avg_n_neigh * Σ_e A_lin_W[e,n] * chem_embed[mu,e]
  t_nn_2d d_z_tr;              // [n_elements, nradmax]

  // MLP radial weights (generalized N-layer)
  t_nn_3d d_mlp_rad_W;          // [n_layers, max_in, max_out] (padded)
  t_nn_1d d_mlp_rad_norms;      // [n_layers]
  t_int_1d d_mlp_rad_dims;      // [n_layers+1] — dims[0]=input, dims[i+1]=output of layer i
  int mlp_rad_n_layers;
  int mlp_rad_max_dim;           // max(all layer dims) for scratch sizing
  int mlp_rad_last_hidden_dim;
  NNScalar mlp_rad_output_norm;

  // FC weights
  t_nn_3d d_fc_A1_wl, d_fc_A1_wr;
  t_nn_3d d_fc_AA1_wl, d_fc_AA1_wr;
  t_nn_3d d_fc_AA2_wl, d_fc_AA2_wr;
  t_int_1d d_fc_A1_wtl, d_fc_A1_wtr;
  t_int_1d d_fc_AA1_wtl, d_fc_AA1_wtr;
  t_int_1d d_fc_AA2_wtl, d_fc_AA2_wtr;
  t_int_1d d_fc_A1_ct, d_fc_A1_cf;
  t_int_1d d_fc_AA1_ct, d_fc_AA1_cf;
  t_int_1d d_fc_AA2_ct, d_fc_AA2_cf;
  t_nn_1d d_fc_A1_nof, d_fc_AA1_nof, d_fc_AA2_nof;
  NNScalar fc_A1_nl, fc_A1_nr, fc_AA1_nl, fc_AA1_nr, fc_AA2_nl, fc_AA2_nr;
  int n_funcs_A1, n_funcs_AA, n_funcs_AA1, n_funcs_AAA, n_funcs_AA2, n_funcs_AAAA;
  int fc_n_out;  // 64 for all FC layers

  // Forward FC gather index: for each target lm, list of (src, tile) from right part
  // CSR format: entries [rg_start[lm], rg_start[lm+1]) index into rg_src/rg_tile
  t_int_1d d_fc_A1_rg_start, d_fc_A1_rg_src, d_fc_A1_rg_tile;
  t_int_1d d_fc_AA1_rg_start, d_fc_AA1_rg_src, d_fc_AA1_rg_tile;
  t_int_1d d_fc_AA2_rg_start, d_fc_AA2_rg_src, d_fc_AA2_rg_tile;

  // Reverse FC gather index: for each source lm, list of (tgt, tile) for right part backward
  t_int_1d d_fc_A1_rev_start, d_fc_A1_rev_src, d_fc_A1_rev_tile;
  t_int_1d d_fc_AA1_rev_start, d_fc_AA1_rev_src, d_fc_AA1_rev_tile;
  t_int_1d d_fc_AA2_rev_start, d_fc_AA2_rev_src, d_fc_AA2_rev_tile;

  // CG product metadata
  t_int_1d d_prod_AA_li, d_prod_AA_ri, d_prod_AA_si;
  t_nn_1d d_prod_AA_cg;
  t_int_1d d_prod_AAA_li, d_prod_AAA_ri, d_prod_AAA_si;
  t_nn_1d d_prod_AAA_cg;
  t_int_1d d_prod_AAAA_li, d_prod_AAAA_ri, d_prod_AAAA_si;
  t_nn_1d d_prod_AAAA_cg;
  int n_cg_AA, n_cg_AAA, n_cg_AAAA;

  // CG product gather indices (forward): for each output func, list of contributing CG terms
  // CSR: gs[n_out+1] start offsets, gli/gri/gcg[n_terms] sorted by output
  t_int_1d d_prod_AA_gs, d_prod_AA_gli, d_prod_AA_gri;  t_nn_1d d_prod_AA_gcg;
  t_int_1d d_prod_AAA_gs, d_prod_AAA_gli, d_prod_AAA_gri;  t_nn_1d d_prod_AAA_gcg;
  t_int_1d d_prod_AAAA_gs, d_prod_AAAA_gli, d_prod_AAAA_gri;  t_nn_1d d_prod_AAAA_gcg;
  // Reverse self-product gather: for each adjoint func f, entries where li==f or ri==f
  // Stores (oa_index, fwd_index, cg_coeff) — fwd_index is the "other" index (ri if li==f, li if ri==f)
  t_int_1d d_rprod_AA_gs, d_rprod_AA_goa, d_rprod_AA_gfwd;  t_nn_1d d_rprod_AA_gcg;
  t_int_1d d_rprod_AAAA_gs, d_rprod_AAAA_goa, d_rprod_AAAA_gfwd;  t_nn_1d d_rprod_AAAA_gcg;
  // Reverse asymmetric-product gather for AAA (AA1⊗A1): separate left/right CSRs
  t_int_1d d_rprod_AAA_left_gs, d_rprod_AAA_left_goa, d_rprod_AAA_left_gfwd;  t_nn_1d d_rprod_AAA_left_gcg;
  t_int_1d d_rprod_AAA_right_gs, d_rprod_AAA_right_goa, d_rprod_AAA_right_gfwd;  t_nn_1d d_rprod_AAA_right_gcg;

  // FunctionReduceN weights
  t_nn_4d d_reduce_A_W;        // [n_elements, rho_n_out, n_in, w_shape]
  t_nn_4d d_reduce_AA_W;
  t_nn_4d d_reduce_AAA_W;
  t_nn_4d d_reduce_AAAA_W;
  NNScalar reduce_A_norm, reduce_AA_norm, reduce_AAA_norm, reduce_AAAA_norm;
  t_int_1d d_reduce_A_ci, d_reduce_AA_ci, d_reduce_AAA_ci, d_reduce_AAAA_ci;
  int rho_n_out;

  // Energy MLP (generalized N-layer)
  t_nn_3d d_energy_W;           // [n_layers, max_in, max_out] (padded)
  t_nn_1d d_energy_norms;       // [n_layers]
  t_int_1d d_energy_dims;       // [n_layers+1]
  int energy_n_layers;
  int energy_max_dim;
  int energy_activation;        // 0=silu, 1=tanh (device-side mirror)

  // Per-element shifts
  t_nn_1d d_shifts;            // [n_elements]

  // ConstantScaleShiftTarget output scale (broadcast scalar)
  NNScalar output_scale;

  // Per-element-pair cutoffs
  t_geom_2d d_bond_cutoff;     // [n_elements, n_elements]

  // Spherical harmonics precomputed coefficients
  t_geom_1d d_idx_sph, alm, blm, cl, dl;
  int idx_sph_max;

  // ---------- Helper functions ----------
  void grow(int natom, int maxneigh);
  bool realloc_all = true;    // set by coeff(): grow() resizes every buffer to the new model
  void copy_weights_to_device();
  void precompute_harmonics();
  void allocate();
  void compute_mlp_radial();

 public:    // (public: nvcc rejects extended lambdas in non-public member functions)
  // compute stages, moved verbatim from compute(); virtual so that grace/1l/fused/kk can replace them
  virtual void stage_bond_fwd(int team_size, int vector_length);   // neighbours, radial basis + MLP, A
  virtual void stage_dense_fwd();                                   // FC A1 .. product AAAA
  virtual void stage_reduce();                                      // rho
  virtual void stage_zero_adj();
  virtual void stage_dense_bwd();                                   // reverse reduce .. reverse FC A1
  virtual void stage_bond_bwd(int team_size, int vector_length);   // per-bond derivative + forces, virial
  // per-bond buffers (radial basis, R_nl, cuBLAS MLP scratch) used only by the base bond stages
  virtual bool need_bond_buffers() const { return true; }
 protected:



  // silu activation: x * sigmoid(x)
  KOKKOS_INLINE_FUNCTION
  static NNScalar silu(NNScalar x) {
    return x / (NNScalar(1.0) + Kokkos::exp(-x));
  }

  KOKKOS_INLINE_FUNCTION
  static NNScalar tanh_act(NNScalar x) {
    return Kokkos::tanh(x);
  }

  // Energy-only mode (skips backward pass / forces)

  // ---------- UQ / extrapolation grade (forward only) ----------
  // FP64 device storage regardless of NNScalar (numerical safety: 1e6 untrained
  // precision, wide sigma^2 dynamic range).
  typedef Kokkos::View<double*, DeviceType>    t_uq_1d;
  // UQ GMM artifacts are read in the team-parallel ComputeUQ kernel with the
  // last (contiguous) index reduced across team lanes; LayoutRight makes those
  // lane reads coalesced (rp_matrix over rp_dim, centroids/inv_cov over the
  // feature dims). Layout is transparent to the logical-index copy_* loaders.
  typedef Kokkos::View<double**, Kokkos::LayoutRight, DeviceType>   t_uq_2d;
  typedef Kokkos::View<double***, Kokkos::LayoutRight, DeviceType>  t_uq_3d;
  typedef Kokkos::View<double****, Kokkos::LayoutRight, DeviceType> t_uq_4d;
  t_uq_3d d_uq_centroids;          // [E, Kmax, D]
  t_uq_4d d_uq_inv_cov;            // [E, Kmax, D, D]
  t_int_1d d_uq_n_clusters;        // [E]
  t_uq_2d d_uq_interp_thresholds;  // [E, Kmax]
  t_uq_2d d_uq_rp_matrix;          // [D_basis, rp_dim] projection R
  // per-atom UQ outputs (device, indexed by local atom; sized nmax_uq)
  t_uq_1d d_gamma, d_sigma, d_gmm_cluster;

  // fp32 UQ projection (uq_sgemm, CUDA fp32 engines; also the hand-written kernels): the fp32 copy of R
  bool uq_sgemm = false;
  Kokkos::View<float*, DeviceType> d_uq_rp_matrix_f;
#ifdef KOKKOS_ENABLE_CUDA
  // ---- cuBLAS DGEMM UQ projection scratch (CUDA only; built lazily on 1st UQ call) ----
  // Bmat = dense gathered invariant basis, col-major [D_basis x subN]; proj = R·Bmat
  // col-major [rp_dim x subN]; n2 = ||B||^2 per sub-block atom. The gather map replays
  // the exact (view,n,collect) walk of ComputeUQ so R-row order matches uq_rp_matrix.
  Kokkos::View<double*, DeviceType> d_uq_Bmat;    // [D_basis * subN]
  Kokkos::View<double*, DeviceType> d_uq_proj;    // [rp_dim * subN]
  // fp32 projection GEMM for the fp32 / mixed styles (uq_sgemm), as grace/2l/kk
  Kokkos::View<float*, DeviceType> d_uq_Bmat_f, d_uq_proj_f;
  Kokkos::View<double*, DeviceType> d_uq_n2;      // [subN]
  t_int_1d d_uq_map_view;          // [D_basis] source view id 0..3 (A/AA/AAA/AAAA)
  t_int_1d d_uq_map_n;             // [D_basis] middle index n
  t_int_1d d_uq_map_col;           // [D_basis] resolved collect column (= reduce_*_ci(c))
  bool uq_map_built = false;
  int uq_subN = 0;                 // Bmat sub-chunk width (caps device memory)
  int uq_bmat_s0 = 0;              // current sub-block base (set before each Bmat/Finish launch)
#endif

  bool has_uq = false;
  int uq_Kmax = 0, uq_D = 0;          // uq_D = full GMM feature dim (rp_dim + n_density)
  int uq_rp_dim = 0;                  // projection output width (R cols; e.g. 128, read from artifact, not enforced)
  int uq_n_density = 0;               // density channels appended after proj (uq_D - rp_dim)
  int uq_normalize = 0;               // 1 -> L2-normalize basis before projection (v6)
  double uq_density_scale = 1.0;      // density_scale
  int uq_d_basis = 0;                 // concatenated invariant (L1) basis width = rows of R
  // host per-atom arrays exposed via extract_peratom (indexed by local atom)
  bool uq_on = false;            // per-run: UQ requested (RunInput::uq) and available    // fix pair flags and per-local-atom outputs (host; a pointer: the functor
                               // copies of this class share it, only the original deletes it)

#ifdef KOKKOS_ENABLE_CUDA
  // ---- cuBLAS true-fp32 radial MLP (CUDA device + NNScalar=float only) ----
  // Repack the padded LayoutLeft d_mlp_rad_W into tight row-major [n_in x n_out]
  // weights and run the per-bond MLP as batched true-fp32 SGEMMs. The fp64 style
  // and non-CUDA backends keep TagComputeMLPRadial.
  typedef Kokkos::View<NNScalar**, Kokkos::LayoutRight, DeviceType> t_nn_2d_r;
  t_nn_2d_r d_mlp_Wg[GRACE1L_MAX_MLP_LAYERS];
  t_nn_2d_r d_mlp_X, d_mlp_dX, d_mlp_h0, d_mlp_h1, d_mlp_dh0, d_mlp_dh1, d_mlp_R;
  int mlp_M = 0;
  int mlp_scatter_deriv = 0;
  int mlp_act_layer = 0;
  int mlp_act_n = 0;
  NNScalar mlp_act_norm = 1;

  // ---- cuBLAS true-fp32 block-sparse FC forward (mirror of 2l Opt-2L-17) ----
  // Each FC output is out(ii,k,func) [=|+=] norm*nof(func) * Σ_n W(k,n,tile(func))
  // * in(ii,n,src(func)): a contraction over n with a per-function TILE-selected
  // weight. Reformulated as gather in-columns grouped by tile -> one true-fp32
  // SGEMM per tile (W_t[n_out x n_in] @ G[n_in x cs*|funcs_t|]) -> scatter. The
  // scatter keeps the hand kernel's (ii,k)-parallel, function-order accumulation
  // (no atomics) so only the Σ_n order changes vs the hand kernel. Active only for
  // the CUDA device instantiation with NNScalar=float (fp32 & Mixed styles); the
  // fp64 device style and all host backends keep the FC hand kernels (handle stays
  // null). FCCublasOp is public because it appears in the signature of the public
  // fc_forward_cublas (nvcc's extended-lambda registration needs the type visible).
 public:
  struct FCCublasOp {
    int n_out = 0, n_in_l = 0, n_in_r = 0, N_l = 0, N_r = 0, ntiles = 0;
    NNScalar nl = 1, nr = 1;
    t_nn_1d Wl_packed, Wr_packed;   // [ntiles * n_out * n_in] col-major per tile
    t_nn_1d nof;                    // [N_l] per-output-function norm factor
    t_int_1d L_src, L_pos;          // L_src[p]=func(lm) tile-sorted; L_pos[lm]=p
    t_int_1d R_src, R_pos, R_tgt;   // R_src[p]=cf tile-sorted; R_pos[idx]=p; R_tgt[idx]=ct
    std::vector<int> L_pstart, L_count, R_pstart, R_count;  // host, [ntiles]
    bool active = false;
  };
  void build_fc_cublas_op(FCCublasOp &op, const GRACE1LModel::FCWeights &fc);
  // run one forward FC (left assign + right accumulate) via gather -> per-tile
  // true-fp32 SGEMM -> scatter, into out (float only). Public for the same reason
  // as the struct (contains extended device lambdas).
  void fc_forward_cublas(FCCublasOp &op, const t_nn_3d &in_left,
      const t_nn_3d &in_right, const t_nn_3d &out);
  // hard gate: verify a tiny SGEMM (transpose/leading-dim convention + true-fp32
  // path) reproduces a host reference before use.
  void fc_cublas_selftest();
  void mlp_sgemm(const NNScalar* Wg, const NNScalar* In, NNScalar* C,
      int n_out, int M, int n_in, int ldW, int ldIn, int ldC, NNScalar alpha);
  void mlp_cublas_selftest();
  // cuBLAS DGEMM UQ projection: build gather map (once), then per sub-block
  // gather Bmat -> cublasDgemm proj = R·Bmat -> GMM finish. Runs on the current
  // chunk (chunk_size/chunk_offset). Falls back to nothing if cublas_handle null.
  void compute_uq_cublas();
 protected:
  cublasHandle_t cublas_handle = nullptr;
  bool fc_fused = false;    // a derived engine does the FC layers itself (no cuBLAS needed)
  bool fc_ops_on() const { return cublas_handle != nullptr || fc_fused; }
  // owner of cublas_handle, shared by every copy of the engine (Kokkos copies it as the kernel functor; a caller may
  // copy it too): the last copy destroys the handle
  std::shared_ptr<std::remove_pointer_t<cublasHandle_t>> cublas_owner;
  FCCublasOp fc_op_AA1, fc_op_AA2;
  t_nn_1d d_fc_G, d_fc_C;         // shared gather / GEMM-output scratch (float)
  int fc_cublas_max_nin = 0, fc_cublas_max_nout = 0, fc_cublas_max_N = 0;
#endif

  // Host-side model data
  std::shared_ptr<GRACE1LModel> grace_model;
  std::vector<int> h_type2model;  // LAMMPS type -> model element index (for init_one)
};

}    // namespace kk
}    // namespace grace

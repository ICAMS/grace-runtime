// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// GRACE-3L Kokkos engine (contributing author Yury Lysogorskiy, ICAMS). Independent of LAMMPS: the consumer supplies
// positions, types, a full neighbour list, the output views and the exchange of the two layers' messages with ghost
// atoms through RunInput3 (layer 0: eq1_norm, layer 1: eq2_norm), and the Kokkos view types through Traits
// (grace/kk/traits.h). Implementation: grace/kk/grace_3l_kk_impl.h (setup, buffers, exchange, run) with the kernel
// parts grace_3l_kk_{radial,spbf,cpl,dense}.h, included by one translation unit per consumer.
#pragma once

#include "grace/coupling.h"
#include "grace/error.h"
#include "grace/model_3l.h"
#include "grace/type_map.h"
#include "grace/system.h"    // spline_n_3l
#include "grace/kk/support.h"
#include "grace/kk/kk_mfma.h"    // wave_sync, the MFMA primitives of the SPBF kernels

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

// GEMM paths (radial MLP: CUDA only; cp_l projections, ReduceN, FC GEMM: CUDA and HIP); defines GRACE_KK_GEMM
#include "grace/kk/blas.h"

namespace grace {
namespace kk {

struct alignas(16) G3F4 { float x, y, z, w; };
struct alignas(8) G3I2 { int i; float v; };   // packed (index, value) of the CG tables
// a radial table row c (cubic in t within the interval): the value, and its derivative in t
KOKKOS_INLINE_FUNCTION float g3l_cubic(const G3F4 &c, float t) { return c.x + t * (c.y + t * (c.z + t * c.w)); }
KOKKOS_INLINE_FUNCTION float g3l_cubic_dt(const G3F4 &c, float t) { return c.y + t * (2.0f * c.z + 3.0f * t * c.w); }
// A layer's cubic radial tables (d_splR, Grace3LKK::spl_tab): per species pair p = sp_i n_sp + sp_j and interval k one
// row of ld outputs o = n (lmax + 1) + l, each the G3F4 of the cubic in t
struct G3LTab {
  const float *c;
  int N, NT, off, ld;
  // the row of pair p at s = (r / rcut) N as the index of its first G3F4, t in the interval (r = rcut: the last
  // interval, t = 1)
  KOKKOS_FORCEINLINE_FUNCTION long row_s(long p, float s, float &t) const
  {
    int k = (int) s;
    if (k > N - 1) k = N - 1;
    t = s - k;
    return (p * NT + off + k) * ld;
  }
  // the same from distance r and cutoff rc (s = r / rc N in float), and ds/dr = N / rc
  KOKKOS_FORCEINLINE_FUNCTION long row(long p, float r, float rc, float &t) const { return row_s(p, r / rc * N, t); }
  KOKKOS_INLINE_FUNCTION float dsdr(float rc) const { return (float) N / rc; }
};

// Warp primitives of the GPU fast paths (32-lane warps). Those paths are selected on Kokkos::Cuda and Kokkos::HIP
// only; reached anywhere else (SYCL device code, a host execution space) they abort instead of returning partial
// results. HIP: the kernels' 32-lane "warp" is half of a 64-lane wavefront (CDNA: gfx90a, gfx942) or one 32-lane
// wavefront (RDNA: gfx11, gfx12); shuffles with width 32 stay inside it either way, and the warp barrier is a
// wavefront-scope fence (all lanes of a wavefront execute together).
KOKKOS_INLINE_FUNCTION int g3l_shfl(int v, int src)
{
#if defined(__CUDA_ARCH__)
  return __shfl_sync(0xffffffffu, v, src);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __shfl(v, src, 32);
#else
  (void) src;
  Kokkos::abort("GRACE-3L/KK: CUDA warp primitive outside a CUDA device kernel");
  return v;
#endif
}

KOKKOS_INLINE_FUNCTION float g3l_shfl_xor(float v, int mask)
{
#if defined(__CUDA_ARCH__)
  return __shfl_xor_sync(0xffffffffu, v, mask);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __shfl_xor(v, mask, 32);
#else
  (void) mask;
  Kokkos::abort("GRACE-3L/KK: CUDA warp primitive outside a CUDA device kernel");
  return v;
#endif
}

KOKKOS_INLINE_FUNCTION double g3l_shfl_xor(double v, int mask)
{
#if defined(__CUDA_ARCH__)
  return __shfl_xor_sync(0xffffffffu, v, mask);
#elif defined(__HIP_DEVICE_COMPILE__)
  return __shfl_xor(v, mask, 32);
#else
  (void) mask;
  Kokkos::abort("GRACE-3L/KK: CUDA warp primitive outside a CUDA device kernel");
  return v;
#endif
}

KOKKOS_INLINE_FUNCTION void g3l_syncwarp()
{
#if defined(__CUDA_ARCH__)
  __syncwarp();
#elif defined(__HIP_DEVICE_COMPILE__)
  wave_sync();
#else
  Kokkos::abort("GRACE-3L/KK: CUDA warp primitive outside a CUDA device kernel");
#endif
}

// ============================================================================
// Grace3LKK: Kokkos engine for GRACE-3L
//
// Compile-time capacity constants. These MUST agree with the exporter's
// authoritative CAPS_3L table (the tensorpotential kokkos.npz export). Named GRACE3L_<CAPS-key>
// so the mapping to that table is one-to-one.
// ============================================================================
static constexpr int GRACE3L_MAX_NRADMAX       = 64;     // SPBF.n_rad_max (A1/A2/A3)
static constexpr int GRACE3L_LMAX              = 4;      // SPBF._lmax / Lmax
static constexpr int GRACE3L_MAX_NRADBASIS     = 12;     // SPBF.n_rad_basis (Chebyshev)
static constexpr int GRACE3L_MAX_MLP_DIM       = 320;    // radial-MLP hidden / output width
static constexpr int GRACE3L_MAX_MLP_LAYERS    = 4;      // radial MLP depth
static constexpr int GRACE3L_MAX_SPBF_NFUNC    = 960;    // A3 CG-couple nfunc
static constexpr int GRACE3L_MAX_SPBF_NLMIND   = 64;     // A3 n_lm_indicator
static constexpr int GRACE3L_MAX_PROD_RANK     = 80;     // A*_2 cp_l rank
static constexpr int GRACE3L_MAX_PROD_NFUNC    = 1920;   // A2_3 product nfunc
static constexpr int GRACE3L_MAX_PROD_NIN      = 160;    // max(n_left,n_right) across products
static constexpr int GRACE3L_MAX_PROD_NGROUPS  = 16;     // n_groups_left/right (L2/L3 products)
static constexpr int GRACE3L_MAX_PROD_NCG      = 10400;  // A2_3 cp_l coupling terms
static constexpr int GRACE3L_MAX_REDUCE_NOUT   = 160;    // A*_2_red / A*_red n_out
static constexpr int GRACE3L_MAX_REDUCE_NIN    = 160;    // largest reduce collector input width
static constexpr int GRACE3L_MAX_REDUCE_NFUNCS = 64;     // largest reduce coupling_meta_data
static constexpr int GRACE3L_MAX_REDUCE_WSHAPE = 336;    // A2_3_red reducing-weight w_shape
static constexpr int GRACE3L_MAX_FC_NOUT       = 160;    // FC n_out / feature width
static constexpr int GRACE3L_MAX_FC_TILE       = 16;     // FC w_right tile dim (A2/A3 FCs)
static constexpr int GRACE3L_MAX_FC_NFUNCS     = 64;     // FC n_funcs_left/right accumulator cap
static constexpr int GRACE3L_MAX_RHO_NOUT      = 40;     // rho*_norm / rho* density width
static constexpr int GRACE3L_MAX_ENERGY_MLP_DIM = 80;    // readout MLP hidden
static constexpr int GRACE3L_MAX_ATOM_TYPES    = 112;    // n_elements / elem-dependent reduce dim

// Inventory-size ceilings for the fixed device-side arrays below. These bound
// the *count* of instructions (not their inner dims — those use GRACE3L_MAX_*
// above). GRACE-3L-OMAT has 3 SPBF / 9 products / 16 reduces / 6 FCs /
// 2 eqnorms / 3 rmsnorms and <=4 collectors per reduce; the values carry slack.
static constexpr int GRACE3L_N_SPBF          = 3;   // A1, A2, A3 (architectural)
static constexpr int GRACE3L_MAX_PRODS       = 12;  // cp_l products (9 used)
static constexpr int GRACE3L_MAX_REDUCES     = 20;  // FunctionReduceN (16 used)
static constexpr int GRACE3L_MAX_FCS         = 8;   // FCRight2Left (6 used)
static constexpr int GRACE3L_MAX_EQNORMS     = 4;   // EquivariantRMSNorm (2 used)
static constexpr int GRACE3L_MAX_RMSNORMS    = 4;   // InvariantLayerRMSNorm (3 used)
static constexpr int GRACE3L_MAX_REDUCE_INSTR = 6;  // collectors per reduce (4 used)
static constexpr int GRACE3L_SPBF_ELL_MAX     = 6;   // max CG row length for the fast SPBF backward
static constexpr int GRACE3L_MAX_REDUCE_ROUNDS = 64; // max BLAS fwd rounds (=max connections sharing one output f; 34 observed)
static constexpr int GRACE3L_UQ_MAX_RP_DIM = 136;    // bounds the FULL UQ feature dim D = rp_dim + n_density (z/f/delta stack arrays); 136 = 128 rp_dim + up to 4 density channels + slack; a larger model fails the D > MAX guard at load

template<class DeviceType, typename NNScalarT, typename GeomScalarT, class Traits>
class Grace3LKK {
 public:
  // Kernel tags
  struct TagPackForwardComm{};
  struct TagUnpackForwardComm{};
  struct TagPackReverseComm{};
  struct TagUnpackReverseComm{};
  struct TagComputeNeigh{};
  struct TagComputeRadialBasis{};
  struct TagComputeMLPRadial{};      // generic radial MLP (per bond, spbf-indexed) -> d_R1_nl
  struct TagComputeA1{};             // Layer-1 scalar-indicator SPBF basis -> d_A1
  struct TagComputeA2{};             // Layer-2 equivariant-indicator SPBF basis -> d_A2
  struct TagComputeA3{};             // Layer-3 equivariant-indicator SPBF basis -> d_A3
  struct TagComputeSPBFFused{};      // fused equiv-SPBF (A2/A3) — one team per (ii,n); prod slice + CG couple in team shared
  struct TagCPLProject{};            // cp_l Pass A: per-lm U/V projection to rank axis
  struct TagCPLCouple{};             // cp_l Pass B: bilinear CG on rank axis + segment-sum
  struct TagCPLFused{};              // fused fwd cp_l (team/(a,r): project+couple in shared, out via shared atomics)
  struct TagCPLBwdFused{};           // fused bwd cp_l (team/(a,r): recompute-project + bwd-couple in shared)
  struct TagReduceN{};                // generic FunctionReduceN (multi-instr, elem-dep)
  struct TagFCRight2Left{};           // generic FCRight2Left (right->left scatter)
  struct TagEqNorm{};                 // generic EquivariantRMSNorm (eq1_norm, eq2_norm)
  struct TagRMSNorm{};                // generic InvariantLayerRMSNorm (rho*_norm)
  struct TagComputeMLPEnergy{};       // 3-density readout (rho1/2/3_norm) -> per-atom energy
  struct TagComputeUQ_rho1{};         // basis-RP: rho1 (L1) block -> raw proj z (Phase 1; d_A1.. current) — forward only
  struct TagComputeUQ_rho2{};         // basis-RP: rho2 (L2) block += into z (Phase 2; d_A2_red.. current) — forward only
  struct TagComputeUQ{};              // basis-RP: rho3 (L3) block += z, normalize + density + GMM (gamma/sigma/gmm_cluster) in Phase 3 — forward only

  // ---- backward (reverse-mode adjoint) tags ----
  struct TagReadoutBwd{};             // readout backward -> d_rho{1,2,3}_norm_adj (all equal)
  struct TagRMSNormBwd{};             // generic InvariantLayerRMSNorm backward (adj_out -> adj_in)
  struct TagReduceNBwd{};             // generic FunctionReduceN backward (per-instruction, accumulate)
  struct TagFCBwdLeft{};              // generic FCRight2Left backward, left path -> adj_left
  struct TagFCBwdRight{};             // generic FCRight2Left backward, right path -> adj_right (atomic)
  struct TagCPLBwdCouple{};           // generic cp_l backward Pass B (couple) -> adj_lproj/adj_rproj
  struct TagCPLBwdProject{};          // generic cp_l backward Pass A (project) -> adj_left/adj_right
  struct TagComputeAdjProd{};         // adj_prod = inv_avg*A_adj@cg_W^T -> d_adj_prod (output-parallel, shared by SPBF-bwd + ForceEquiv)
  struct TagA3SPBFBwd{};              // L3 equiv-SPBF backward: d_A3_adj -> d_eq2_norm_adj (message adj)
  struct TagEqNormBwd{};              // generic EquivariantRMSNorm VJP (adj_out -> adj_in)
  struct TagA2SPBFBwd{};              // L2 equiv-SPBF backward: d_A2_adj -> d_grad_I_global (eq1_norm message adj)
  struct TagSPBFBwdFused{};           // team+shared equiv-SPBF backward (A2/A3) — adj_prod row staged in shared once
  // ---- radial r-derivative + per-bond force (geometry backward) tags ----
  struct TagComputeMLPRadialDeriv{}; // radial MLP forward + dR/dr -> d_R1_nl, d_DR1_nl (per layer)
#ifdef KOKKOS_ENABLE_CUDA
  struct TagMLPAssembleVal{};        // build cuBLAS input X[M x n_in0] (value path), zero padding rows
  struct TagMLPAssembleDeriv{};      // build X[M x n_in0] + dX[M x nradbase] (deriv path)
  struct TagMLPActVal{};             // silu activation between GEMMs (value only)
  struct TagMLPActDeriv{};           // silu (value) + silu'(s)*ds (deriv) between GEMMs
  struct TagMLPScatterR{};           // scatter LayoutRight R/DR[M x d3] -> LayoutLeft d_R1_nl/d_DR1_nl
  struct TagMLPLayer0Val{};          // radial MLP layer 0 from the Chebyshev basis + per-element embedding terms, silu
  struct TagMLPLayer0Deriv{};        // same + the r-derivative (silu' * ds)
#endif
  struct TagForceL1{};               // L1 scalar-SPBF geometry backward -> d_f_ij
  struct TagForceEquiv{};            // L2/L3 equiv-SPBF geometry backward -> d_f_ij
  struct TagForceEquivFused{};       // team-per-(ii,n) ForceEquiv — adj_prod row staged in shared, one bond per thread

  typedef DeviceType device_type;
  using EV = typename Traits::ev_type;        // energy / virial reduction type (LAMMPS: EV_FLOAT)
  using KKF = typename Traits::kk_float;      // position precision (LAMMPS: KK_FLOAT)
  using KKA = typename Traits::kk_acc_float;  // force / energy accumulation precision (LAMMPS: KK_ACC_FLOAT)
  typedef EV value_type;
  using model_type = GRACE3LModel;
  using traits_type = Traits;

  using GeomScalar = GeomScalarT;
  using NNScalar = NNScalarT;
  static constexpr bool fused = false;   // Grace3LFusedKK: true

  Grace3LKK();
  ~Grace3LKK();
  Grace3LKK(const Grace3LKK &) = default;    // Kokkos copies the engine as the kernel functor

  // ---- engine API (the LAMMPS glue and the Python module call these) ----
  // model: loaded GRACE-3L model; type2mu[t] (t = 1..ntypes): model element of atom type t, -1 unmapped.
  // The model's reduce inputs are grouped by output once (GRACE3LModel::reduce_inputs_grouped; a function-preserving
  // permutation of producer columns, see group_reduce_inputs in grace_3l_kk_impl.h), so engines may share a model.
  void set_model(std::shared_ptr<GRACE3LModel> model, const std::vector<int> &type2mu);
  void setup();                           // scalars, harmonics, device weights, radial tables, BLAS (LAMMPS: init_style)
  // another species set on a set-up engine: type tables (atom type -> model element, pair cutoffs) and the radial
  // tables of the new species pairs are rebuilt; weights stay
  void set_types(const std::vector<int> &type2mu);
  // the type tables of the current species set (views shared, not copied: the radial spline tables of its species
  // pairs too, with the shape they were built with): a caller keeps them and hands them back with use_type_tables() to
  // switch to that species set again without rebuilding (set_types builds new ones), also after a setup() of the same
  // model with other table settings (spline_n, spline_n_layer, spline_mb)
  struct TypeTables {
    std::vector<int> type2model;
    typename Traits::t_int_1d map;
    Kokkos::DualView<typename Traits::kk_float **, DeviceType> cutsq;
    int spline_on = 0, n_sp = 0;
    Kokkos::View<int *, DeviceType> sp_of_mu;
    Kokkos::View<float *, DeviceType> splR;
    int spl_N[3] = {0, 0, 0}, spl_off[3] = {0, 0, 0}, spl_Ntot = 0, spl_ld = 0;    // the shape splR was built with
    std::vector<int> spl_mus;
  };
  TypeTables type_tables() const;
  // the per-atom buffers (chunk work buffers, the per-image rows, kept state) shrunk to one atom; weights and type
  // tables stay, the next run() grows the buffers again (callers between independent jobs: free the device memory)
  virtual void release_buffers();
  void use_type_tables(const TypeTables &t);
  double cutoff(int mu_i, int mu_j) const;    // pair cutoff of model elements
  // Exchange of the message rows between the atoms and their ghost images (LAMMPS: comm->forward_comm /
  // reverse_comm of the pair style, which calls pack_forward ... below with the layer). Order in run():
  // forward(0), forward(1), reverse(1), reverse(0) (energy only: the two forwards). Without an exchange (no ghosts,
  // or ghosts that are not images of local atoms) all steps are skipped.
  struct Exchange3 {
    virtual ~Exchange3() = default;
    virtual void forward(int layer) = 0;    // message rows of layer `layer` of the owners -> their ghost images
    virtual void reverse(int layer) = 0;    // their adjoint rows of the ghost images summed into the owners
  };
  struct RunInput3 {
    typename Traits::t_kkfloat_1d_3_lr_randomread x;    // [nall] positions, local atoms then ghosts
    typename Traits::t_kkacc_1d_3 f;
    typename Traits::t_int_1d_randomread type, ilist, numneigh;    // ilist: mapped types only (grace/type_map.h)
    typename Traits::t_neighbors_2d neighbors;
    int inum = 0, nlocal = 0, nall = 0, ntypes = 0;
    int eflag_global = 0, eflag_atom = 0, vflag_global = 0, vflag_atom = 0, cvflag_atom = 0;
    bool energy_only = false, uq = false;
    // two-group coupling (grace/coupling.h): couple_eq 0 off; mask [nall] (local atoms then ghosts) with couple_eq
    int couple_eq = 0, couple_bit1 = 0, couple_bit2 = 0;
    double couple_lambda = 0.0;
    typename Traits::t_int_1d_randomread mask;
    typename Traits::t_kkacc_1d eatom;
    typename Traits::t_kkacc_1d_6 vatom;
    typename Traits::t_kkacc_1d_9 cvatom;
    Exchange3 *ex = nullptr;
  };
  using RunInput = RunInput3;    // as the 1L / 2L engines (consumers generic over the engine)
  struct RunOutput {
    double eng_vdwl = 0.0;          // energy of the inum atoms (summed per chunk, as LAMMPS eng_vdwl)
    double virial[6] = {0, 0, 0, 0, 0, 0};
    double dedl = 0.0;              // coupling: dE/dlambda of the inum atoms' bonds (with the backward pass)
  };
  // forces added to f, per-atom energy / virial added to eatom / vatom / cvatom; global energy and virial returned;
  // with uq (and a usable UQ head) the per-atom gamma / sigma / cluster in uq_gamma() ... [nlocal]
  RunOutput run(const RunInput3 &in);
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

  // ---- message rows for an Exchange3: comm_width(layer) doubles per atom (layer 0: eq1_norm, 1: eq2_norm; valid
  // from set_model() on) ----
  int comm_width(int layer) const { return layer == 0 ? I_n_funcs * I_n_out : I2_n_funcs * I2_n_out; }
  // device buffers (LAMMPS pack_forward_comm_kokkos ...); return the number of doubles packed
  int pack_forward(int layer, int n, const typename Traits::t_int_1d &sendlist, const typename Traits::t_double_1d &buf);
  void unpack_forward(int layer, int n, int first, const typename Traits::t_double_1d &buf);
  int pack_reverse(int layer, int n, int first, const typename Traits::t_double_1d &buf);
  void unpack_reverse(int layer, int n, const typename Traits::t_int_1d &sendlist,
                      const typename Traits::t_double_1d &buf);
  // host buffers (LAMMPS legacy pair comm, MPI without GPU support) on host copies of the rows: copy_rows_to_host
  // before packing, copy_rows_from_host after unpacking; forward: the message rows, reverse: their adjoint rows
  void copy_rows_to_host(int layer, bool reverse);
  void copy_rows_from_host(int layer, bool reverse);
  int pack_forward_host(int layer, int n, const int *list, double *buf);
  void unpack_forward_host(int layer, int n, int first, const double *buf);
  int pack_reverse_host(int layer, int n, int first, double *buf);
  void unpack_reverse_host(int layer, int n, const int *list, const double *buf);

  KOKKOS_INLINE_FUNCTION
  void operator() (TagPackForwardComm, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagUnpackForwardComm, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagPackReverseComm, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagUnpackReverseComm, const int&) const;

  // Kernel operators (neighbor + Chebyshev radial + Y_lm precompute)
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeNeigh, const typename Kokkos::TeamPolicy<DeviceType, TagComputeNeigh>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeRadialBasis, const typename Kokkos::TeamPolicy<DeviceType, TagComputeRadialBasis>::member_type& team) const;

  // Kernel operators (L1 scalar-indicator SPBF basis A1)
  // Generic radial MLP: serves A1 and A2, spbf-indexed by
  // the member radial_mlp_spbf set before launch.
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeMLPRadial, const typename Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadial>::member_type& team) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeA1, const int) const;

  // Kernel operator (L2 equivariant-indicator SPBF basis A2)
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeA2, const typename Kokkos::TeamPolicy<DeviceType, TagComputeA2>::member_type& team) const;

  // Kernel operator (L3 equivariant-indicator SPBF basis A3)
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeA3, const typename Kokkos::TeamPolicy<DeviceType, TagComputeA3>::member_type& team) const;

  // Kernel operator (fused equiv-SPBF for A2/A3 — team per (ii,n), prod slice + CG couple in shared)
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeSPBFFused, const typename Kokkos::TeamPolicy<DeviceType, TagComputeSPBFFused>::member_type& team) const;

  // Kernel operators (generic cp_l GeneralProductFunction product)
  KOKKOS_INLINE_FUNCTION
  void operator() (TagCPLProject, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagCPLCouple, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagCPLFused,
      const typename Kokkos::TeamPolicy<DeviceType, TagCPLFused>::member_type& team) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagCPLBwdFused,
      const typename Kokkos::TeamPolicy<DeviceType, TagCPLBwdFused>::member_type& team) const;

  // Kernel operators (generic FunctionReduceN + FCRight2Left)
  KOKKOS_INLINE_FUNCTION
  void operator() (TagReduceN, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagFCRight2Left, const int&) const;

  // Kernel operators (generic EquivariantRMSNorm + InvariantLayerRMSNorm)
  KOKKOS_INLINE_FUNCTION
  void operator() (TagEqNorm, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagRMSNorm, const int&) const;

  // Kernel operator (3-density readout -> per-atom energy)
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeMLPEnergy, const typename Kokkos::TeamPolicy<DeviceType, TagComputeMLPEnergy>::member_type& team) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeUQ_rho1, const typename Kokkos::TeamPolicy<DeviceType, TagComputeUQ_rho1>::member_type& team) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeUQ_rho2, const typename Kokkos::TeamPolicy<DeviceType, TagComputeUQ_rho2>::member_type& team) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeUQ, const typename Kokkos::TeamPolicy<DeviceType, TagComputeUQ>::member_type& team) const;

  // Kernel operators (readout + L3 backward / generic bwd routines)
  KOKKOS_INLINE_FUNCTION
  void operator() (TagReadoutBwd, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagRMSNormBwd, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagReduceNBwd, const int, const int) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagFCBwdLeft, const int, const int, const int) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagFCBwdRight, const int, const int, const int) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagCPLBwdCouple, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagCPLBwdProject, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeAdjProd, const int, const int, const int) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagA3SPBFBwd, const int, const int) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagEqNormBwd, const int&) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagA2SPBFBwd, const int, const int) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagSPBFBwdFused, const typename Kokkos::TeamPolicy<DeviceType, TagSPBFBwdFused>::member_type& team) const;

  // ---- force kernels ----
  KOKKOS_INLINE_FUNCTION
  void operator() (TagComputeMLPRadialDeriv, const typename Kokkos::TeamPolicy<DeviceType, TagComputeMLPRadialDeriv>::member_type& team) const;
#ifdef KOKKOS_ENABLE_CUDA
  KOKKOS_INLINE_FUNCTION
  void operator() (TagMLPAssembleVal, const int) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagMLPAssembleDeriv, const int) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagMLPActVal, const int64_t) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagMLPActDeriv, const int64_t) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagMLPScatterR, const int64_t) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagMLPLayer0Val, const int64_t) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagMLPLayer0Deriv, const int64_t) const;
#endif
  KOKKOS_INLINE_FUNCTION
  void operator() (TagForceL1, const typename Kokkos::TeamPolicy<DeviceType, TagForceL1>::member_type& team) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagForceEquiv, const int, const int) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (TagForceEquivFused, const typename Kokkos::TeamPolicy<DeviceType, TagForceEquivFused>::member_type& team) const;
  // Shared per-bond geometry-force chain (spec §4.1): given the per-lm weight
  // array w[nlm_y] (= dE/da_nl for this (ii,jj,n) up to the R*Y*env factor),
  // accumulates the Cartesian force f = Σ_{l,m} w*(Y*DRenv*rhat + DY*Renv/r)
  // into (f0,f1,f2). Renv = R*env, DRenv = dR/dr*env + R*denv/dr; DY is the
  // tangential (rhat-orthogonal) gradient of the real SH. Reused by L1/L2/L3.
  KOKKOS_INLINE_FUNCTION
  void bond_geom_force(const int ii, const int jj, const int n,
      const NNScalar* w, GeomScalar& f0, GeomScalar& f1, GeomScalar& f2) const;
  // Same force with the channel sum done by the caller: a[lm] = sum_n w[n,lm] d(R env)/dr [n,l(lm)],
  // b[lm] = sum_n w[n,lm] (R env)[n,l(lm)];  f += a Y rhat + b DY / r  (one P_lm recurrence per bond).
  KOKKOS_INLINE_FUNCTION
  void bond_geom_force_ab(const int ii, const int jj, const GeomScalar* a, const GeomScalar* b,
      GeomScalar& f0, GeomScalar& f1, GeomScalar& f2) const;

 protected:
  int inum = 0, maxneigh = 0, chunk_size = 0, chunk_offset = 0;
  int chunk_lim = 1;    // atoms per chunk of the current run() (chunk_cap())
  // chunksize auto: the chunk chosen for up to auto_inum atoms and auto_maxneigh neighbours (10 % headroom, so a
  // slowly growing system does not re-decide every step)
  int chunk_auto = 0, auto_inum = 0, auto_maxneigh = 0;
  bool store_auto_ok = true;    // chunksize auto: the forward store fits the budget next to the chunk's work buffers
  int chunk_cap() const { const int c = chunksize < 0 ? chunk_auto : chunksize; return c > 0 ? c : inum; }
  void choose_chunk();
  // chunksize auto: bytes per atom kept for all atoms when there is more than one chunk (the store_fwd tensors)
  double kept_bytes_per_atom() const;
  int host_flag;
  int eflag = 0, vflag = 0;
  int eflag_global = 0, eflag_atom = 0, vflag_global = 0, vflag_atom = 0, cvflag_atom = 0, nlocal = 0, nall = 0;
  double eng_vdwl = 0.0, virial[6] = {0, 0, 0, 0, 0, 0};
  bool setup_done = false;    // setup() after the last set_model(): run() refuses otherwise
  void set_type_map(const std::vector<int> &type2mu);    // h_type2model, d_map, cutsq

  // Architecture params
  int nelements, lmax, nradbase, embedding_size;
  int radial_basis_p;
  GeomScalar rcut;

  // Layer-specific radial dimensions
  int L1_nradmax, L2_nradmax, L3_nradmax;

  // I feature dims (drive comm buffer sizing): set by set_model() from eq1_norm.
  int I_n_out = 0, I_n_funcs = 0;
  // Round-2 (L3) indicator dims. The Layer-3 indicator is eq2_norm,
  // which is PARITY-DOUBLED vs eq1_norm: I2_n_out = eq2_norm.M (=50) is twice
  // I_n_out (=25), so round 2 needs its own buffer d_eq2_norm_global (below) —
  // d_I_global [nall,64,25] cannot hold a [nall,64,50] payload.
  int I2_n_out = 0, I2_n_funcs = 0;
  // Forward-comm stage selector: 1 = round 1 (eq1_norm -> d_I_global,
  // I_n_funcs*I_n_out/atom); 2 = round 2 (eq2_norm -> d_eq2_norm_global,
  // I2_n_funcs*I2_n_out/atom). Set on host immediately before each
  // comm->forward_comm; captured by value into the pack/unpack tag-functors.
  int comm_stage = 1;


  // Neighbour list and atom data of the current run() (LAMMPS: the pair style's views)
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
  tdual_fparams k_cutsq;
  typedef Kokkos::View<KKF**, DeviceType> t_fparams;
  t_fparams d_cutsq;
  typename Traits::t_int_1d d_map;


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
  typedef Kokkos::View<NNScalar***, Kokkos::LayoutRight, DeviceType> t_nn_3d_r; // row-major 3D
  // the arrays with a row per atom of nall (local + ghost): 64-bit offsets (grace::kk::View64)
  typedef grace::kk::View64<t_nn_3d_r> t_nn_3d_rg;
  typedef grace::kk::View64<t_nn_3d> t_nn_3d_g;
  typedef grace::kk::View64<typename t_nn_3d_r::host_mirror_type> t_nn_3d_rg_host;
  typedef Kokkos::View<NNScalar***, Kokkos::LayoutStride, DeviceType, Kokkos::MemoryTraits<Kokkos::Unmanaged>> t_nn_3d_s;
  typedef Kokkos::View<NNScalar****, DeviceType> t_nn_4d;
  typedef Kokkos::View<NNScalar****, Kokkos::LayoutRight, DeviceType> t_nn_4d_r; // row-major 4D

  // ---------- Per-chunk geometry arrays (neighbor + Chebyshev
  // radial + Y_lm precompute). GeomScalar (fp32 or fp64); bond vectors, envelope and Chebyshev basis fp64. ----------
  t_int_1d d_ncount;              // [chunk_size]
  t_int_2d d_nearest;             // [chunk_size, maxneigh] local/ghost neighbor index
  // r_ij, |r| and rhat in fp64 for every GeomScalar: the kernels that use them
  // cast to GeomScalar on read, the virial takes del = -rhat r in fp64
  using BondScalar = double;
  Kokkos::View<BondScalar**, DeviceType> d_rnorms;       // [chunk_size, maxneigh] bond length r (coupling: r')
  Kokkos::View<BondScalar**[3], DeviceType> d_cpl;       // coupling only: true r, dr'/dr, dr'/dlambda
  Kokkos::View<BondScalar**[3], DeviceType> d_rhats;     // [chunk_size, maxneigh, 3] unit vector (x_j - x_i)/r
  t_int_1d d_mu_i;                // [chunk_size] central-atom model species index
  t_int_2d d_mu_j;                // [chunk_size, maxneigh] neighbor model species index
  t_geom_3d d_radial_basis;       // [chunk_size, maxneigh, nradbase] RAW Chebyshev T_1..T_nradbase (NO envelope)
  t_geom_2d d_env;                // [chunk_size, maxneigh] polynomial cutoff envelope (separate scalar)
  // Per-bond real spherical harmonics, precomputed once per chunk from
  // d_rhats and reused by every downstream SPBF layer. Declared LayoutRight
  // so the innermost (lm) dim is contiguous for readers that sweep it.
  Kokkos::View<GeomScalar***, Kokkos::LayoutRight, DeviceType> d_Y_bond;  // [chunk_size, maxneigh, (lmax+1)^2]
  // Tangential gradient of each real SH, dY_lm/d(rhat) projected orthogonal to rhat (the DY of bond_geom_force):
  // the force of a bond weight w_lm on (R env)(r) Y_lm is w_lm (d(R env)/dr Y_lm rhat + (R env) DY_lm / r).
  Kokkos::View<GeomScalar****, Kokkos::LayoutRight, DeviceType> d_dY_bond; // [chunk_size, maxneigh, (lmax+1)^2, 3]

  // ---------- Layer-1 NN intermediates (scalar-indicator SPBF A1).
  // NNScalar (fp32 in the Mixed default); geometry feeding these kernels
  // (d_radial_basis, d_Y_bond, d_env) is cast down from GeomScalar. ----------
  t_nn_4d_r d_R1_nl;              // [chunk_size, maxneigh, L1_nradmax, lmax+1] (pre-l_tile-gather); row-major so
                                  // the radial-MLP output GEMM writes it directly (row = bond)
  t_nn_3d d_A1;                   // [chunk_size, L1_nradmax, (lmax+1)^2]
  t_nn_3d d_A1_2;                 // [chunk_size, rank, nfunc] — first cp_l product
  t_nn_3d d_A1_2_red;             // [chunk_size, n_out, n_funcs] — FunctionReduceN(A1_2)
  t_nn_3d d_A1_2a;                // [chunk_size, n_out, n_lm] — FCRight2Left(A1_2_red, A1)
  // ---- Remaining L1 chain ----
  t_nn_3d d_A1_3;                 // [chunk_size, rank, nfunc] — cp_l(A1_2a, A1)
  t_nn_3d d_A1_3_red;             // [chunk_size, n_out, n_funcs] — FunctionReduceN(A1_3)
  t_nn_3d d_A1_2b;                // [chunk_size, n_out, n_lm] — FCRight2Left(A1_2_red, A1)
  t_nn_3d d_A1_4;                 // [chunk_size, rank, nfunc] — cp_l(A1_2b, A1_2b)
  t_nn_3d d_A1_4_red;             // [chunk_size, n_out, n_funcs] — FunctionReduceN(A1_4)
  t_nn_3d d_eq1;                  // [chunk_size, 64, 25] — multi-instr elem-dep ReduceN([A1,A1_2_red,A1_3_red,A1_4_red])
  t_nn_3d d_eq1_norm;             // [chunk_size, 64, 25] — EquivariantRMSNorm(eq1); THE Layer-2 indicator
  t_nn_3d d_rho1;                 // [chunk_size, 32, 1] — multi-instr elem-dep only_invar ReduceN (same 4 inputs as eq1)
  t_nn_3d d_rho1_norm;            // [chunk_size, 32, 1] — InvariantLayerRMSNorm(rho1, only_nonlin)

  // ---------- Layer-2 NN intermediates (equiv-indicator SPBF A2) ----------
  t_nn_3d d_A2;                   // [chunk_size, A2_n_rad_max=64, A2_nfunc=375] — L2 equiv-ind SPBF basis
  // Which SPBF layer the generic radial-MLP kernel (TagComputeMLPRadial) uses.
  // Set on host before each launch (0 = A1, idx_spbf_A2 = A2); captured by value
  // when *this is copied into the parallel_for functor.
  int radial_mlp_spbf = 0;

  // ---- Remaining L2 chain. Mirrors the L1 chain exactly EXCEPT
  // every downstream product/FC/reduce here is built on A2_red (a reduced
  // A2), not on the raw SPBF output A2 itself (L1 used raw A1 directly). ----
  t_nn_3d d_A2_red;               // [chunk_size, n_out, n_funcs] — FunctionReduceN({A2})
  t_nn_3d d_A2_2;                 // [chunk_size, rank, nfunc] — cp_l(A2_red, A2_red)
  t_nn_3d d_A2_2_red;             // [chunk_size, n_out, n_funcs] — FunctionReduceN(A2_2)
  t_nn_3d d_A2_2a;                // [chunk_size, n_out, n_lm] — FCRight2Left(A2_2_red, A2_red)
  t_nn_3d d_A2_3;                 // [chunk_size, rank, nfunc] — cp_l(A2_2a, A2_red)
  t_nn_3d d_A2_3_red;             // [chunk_size, n_out, n_funcs] — FunctionReduceN(A2_3)
  t_nn_3d d_A2_2b;                // [chunk_size, n_out, n_lm] — FCRight2Left(A2_2_red, A2_red)
  t_nn_3d d_A2_4;                 // [chunk_size, rank, nfunc] — cp_l(A2_2b, A2_2b)
  t_nn_3d d_A2_4_red;             // [chunk_size, n_out, n_funcs] — FunctionReduceN(A2_4)
  t_nn_3d d_eq2;                  // [chunk_size, n_out, M] — multi-instr elem-dep ReduceN([A2_red,A2_2_red,A2_3_red,A2_4_red])
  t_nn_3d d_eq2_norm;             // [chunk_size, n_out, M] — EquivariantRMSNorm(eq2); THE Layer-3 indicator
  t_nn_3d d_rho2;                 // [chunk_size, n_out, n_lm] — multi-instr elem-dep ReduceN (same 4 inputs as eq2)
  t_nn_3d d_rho2_norm;            // [chunk_size, n_out, n_lm] — InvariantLayerRMSNorm(rho2, FULL branch)

  // ---------- Layer-3 NN intermediates (equiv-indicator SPBF A3;
  // indicator = eq2_norm). TERMINAL layer: produces rho3/rho3_norm ONLY — there
  // is NO eq3/eq3_norm. The A3_red..rho3 chain is structurally identical to L2's
  // A2_red..rho2 chain (built on A3_red, a reduce of the raw SPBF output A3).
  // A3_3/A3_4 are Lmax=0 (out width ~10), so their dims come from the resolved
  // per-instruction metadata, NOT L2's 50-wide shapes. ----------
  t_nn_3d d_A3;                   // [chunk_size, A3_n_rad_max=64, A3_nfunc=750] — L3 equiv-ind SPBF basis
  t_nn_3d d_A3_red;              // [chunk_size, n_out, n_funcs] — FunctionReduceN({A3})
  t_nn_3d d_A3_2;               // [chunk_size, rank, nfunc] — cp_l(A3_red, A3_red)
  t_nn_3d d_A3_2_red;           // [chunk_size, n_out, n_funcs] — FunctionReduceN(A3_2)
  t_nn_3d d_A3_2a;              // [chunk_size, n_out, n_lm] — FCRight2Left(A3_2_red, A3_red)
  t_nn_3d d_A3_3;               // [chunk_size, rank, nfunc] — cp_l(A3_2a, A3_red) (Lmax=0)
  t_nn_3d d_A3_3_red;           // [chunk_size, n_out, n_funcs] — FunctionReduceN(A3_3)
  t_nn_3d d_A3_2b;              // [chunk_size, n_out, n_lm] — FCRight2Left(A3_2_red, A3_red)
  t_nn_3d d_A3_4;               // [chunk_size, rank, nfunc] — cp_l(A3_2b, A3_2b) (Lmax=0)
  t_nn_3d d_A3_4_red;           // [chunk_size, n_out, n_funcs] — FunctionReduceN(A3_4)
  t_nn_3d d_rho3;               // [chunk_size, 32, 1] — multi-instr elem-dep only_invar ReduceN({A3_red,A3_2_red,A3_3_red,A3_4_red})
  t_nn_3d d_rho3_norm;          // [chunk_size, 32, 1] — InvariantLayerRMSNorm(rho3, FULL branch)

  // ---------- Backward adjoint buffers (dE/dX), NNScalar. Mostly chunk-local,
  // each mirroring its forward tensor's shape. Exceptions: the [nall] readout
  // seeds d_rho{1,2}_norm_adj and the message adjoint d_eq2_norm_adj (scattered
  // to owned+ghost); the shared scratch d_adj_prod and d_cpl_{l,r}proj_adj. ----
  t_nn_3d_g d_rho1_norm_adj, d_rho2_norm_adj;    // readout bwd (all 3 equal; rho1 / rho2 [nall], see below)
  t_nn_3d d_rho3_norm_adj;
  // d_rho1_norm_adj/d_rho2_norm_adj are PROMOTED to natom ([nall],
  // allocated in grow_global) so TagReadoutBwd (run over ALL P3 chunks) can seed
  // every owned atom's rho1/rho2 readout adjoint at (ii+chunk_offset), surviving
  // until P5/P4. The shared TagRMSNormBwd reads adj_out chunk-local, so P5/P4
  // gather the natom seed into these chunk-scratch _local buffers per chunk
  // (mirrors d_eq{1,2}_norm_adj_local). d_rho3_norm_adj stays chunk-scratch.
  t_nn_3d d_rho1_norm_adj_local, d_rho2_norm_adj_local;
  t_nn_3d d_rho1_adj, d_rho2_adj, d_rho3_adj;                 // invariant-RMSNorm bwd
  t_nn_3d d_A3_red_adj, d_A3_2_adj, d_A3_2_red_adj, d_A3_2a_adj;
  t_nn_3d d_A3_3_adj, d_A3_3_red_adj, d_A3_2b_adj, d_A3_4_adj, d_A3_4_red_adj;
  t_nn_3d d_A3_adj;             // [chunk_size, A3_n_rad_max, A3_nfunc]
  // shared adj_prod buffer [chunk_size, MAX_NRADMAX, nlm_y*MAX_NLMIND].
  // adj_prod[ii,n,p] = inv_avg*Σ_f A_adj(ii,n,f)*cg_W(p,f) (p = ly*n_lm_ind+lr),
  // computed ONCE per layer by TagComputeAdjProd (output-parallel — no per-thread
  // local array) and read by BOTH the SPBF-bwd (A2/A3) AND ForceEquiv consumers,
  // which previously each recomputed it into a register-spilling 1600-float local.
  t_nn_3d d_adj_prod;
  t_nn_3d_rg d_eq2_norm_adj;      // [nall, I2_n_funcs, I2_n_out] — message adjoint (owned PARTIAL, pre-reverse-comm)
  t_nn_3d_s d_cpl_lproj_adj, d_cpl_rproj_adj;  // cp_l bwd rank-projection scratch (same layout as d_cpl_lproj/rproj)
  // ---- Layer-2 backward adjoint buffers (mirror the L3 d_A3_* set) ----
  t_nn_3d d_A2_red_adj, d_A2_2_adj, d_A2_2_red_adj, d_A2_2a_adj;
  t_nn_3d d_A2_3_adj, d_A2_3_red_adj, d_A2_2b_adj, d_A2_4_adj, d_A2_4_red_adj;
  t_nn_3d d_A2_adj;             // [chunk_size, A2_n_rad_max, A2_nfunc]
  t_nn_3d d_eq2_adj;            // [chunk_size, eq2.n_out, eq2.M] — EquivariantRMSNorm bwd input adj
  // Chunk-local gather of the reverse-comm-completed d_eq2_norm_adj (owned rows,
  // gathered by d_ilist so the generic equiv bwd stays chunk-local).
  t_nn_3d d_eq2_norm_adj_local; // [chunk_size, I2_n_funcs, I2_n_out]
  // d_eq1_norm_adj (round-1 message adjoint, owned partial) REUSES d_grad_I_global
  // [nall, I_n_funcs, I_n_out] — the round-1 reverse buffer, reverse-comm'd in PHASE 4.5.

  // ---- Layer-1 backward adjoint buffers. UNLIKE L2/L3, the L1 chain
  // consumes the RAW SPBF basis A1 directly (there is no A1_red), so d_A1_adj is
  // the DIRECT fan-out target (accumulates from eq1[0]+rho1[0] + cp_l A1_2 (both)
  // + fc A1_2a/2b (right) + cp_l A1_3 (right)). ----
  t_nn_3d d_A1_2_adj, d_A1_2_red_adj, d_A1_2a_adj, d_A1_3_adj, d_A1_3_red_adj;
  t_nn_3d d_A1_2b_adj, d_A1_4_adj, d_A1_4_red_adj;
  t_nn_3d d_A1_adj;             // [chunk_size, L1_nradmax, (lmax+1)^2]
  t_nn_3d d_eq1_adj;            // d_rho1_adj already declared with the rho{1,2,3}_adj set above
  // Chunk-local gather of the reverse-comm-completed round-1 message adjoint
  // (d_grad_I_global owned rows).
  t_nn_3d d_eq1_norm_adj_local; // [chunk_size, I_n_funcs, I_n_out]

  // ---- per-bond forces + radial r-derivative (geometry backward) ----
  t_geom_3d3 d_f_ij;            // [chunk_size, maxneigh, 3] per-bond force (F_i += f_ij, F_j -= f_ij)
  t_nn_4d_r d_DR1_nl;           // [chunk_size, maxneigh, L1_nradmax, lmax+1] dR/dr (shared scratch, per layer)
  t_geom_2d d_denv;             // [chunk_size, maxneigh] denv/dr (layer-independent; TagComputeRadialBasis)
  // per-call state for the equivariant force kernel (TagForceEquiv)
  int force_spbf_idx = 0;       // idx_spbf_A2 or idx_spbf_A3
  int force_ind_stage = 1;      // 1 = L2 (indicator d_I_global), 2 = L3 (indicator d_eq2_norm_global)
  int force_n_lm_ind = 0;       // indicator lm width for the current layer
  t_nn_3d force_A_adj;          // d_A2_adj or d_A3_adj (the basis adjoint being backpropped)

#ifdef GRACE_KK_GEMM
  // ---- BLAS handle (every GEMM path); true-fp32 radial-MLP scratch + launch state ----
  blas::handle_t blas_handle = nullptr;
  // owner of blas_handle, shared by every copy of the engine (Kokkos copies it as the kernel functor; a caller may
  // copy it too): the last copy destroys the handle. No handle (CUDA: Python wheels without cuBLAS,
  // GRACE_RUNTIME_NO_CUBLAS; HIP: GRACE_RUNTIME_NO_ROCBLAS and GRACE_RUNTIME_NO_KKGEMM): the hand-written kernels.
  std::shared_ptr<std::remove_pointer_t<blas::handle_t>> blas_owner;
  typedef Kokkos::View<NNScalar**, Kokkos::LayoutRight, DeviceType> t_nn_2d_r; // row-major [M x dim]
  // Per-(spbf,layer) weights repacked TIGHTLY row-major [n_in x n_out] at load
  // time. The native mlp_W view is LayoutLeft (column-major) + padded, so a fixed
  // layer's (k,j) block is NOT contiguous -> cannot feed cuBLAS directly. These
  // packed copies give lda=n_out (col-major W^T leading dim). One-time, ~1 MB.
  t_nn_2d_r d_mlp_Wg[GRACE3L_N_SPBF][GRACE3L_MAX_MLP_LAYERS];
  t_nn_2d_r d_mlp_X;    // [M x n_in0=138] value input
  t_nn_2d_r d_mlp_dX;   // [M x nradbase=10] deriv input (only Chebyshev cols are r-dependent)
  t_nn_2d_r d_mlp_h0;   // [M x 256]       layer-0 value activation (raw in-place -> silu)
  t_nn_2d_r d_mlp_dh0;  // [M x 256]       layer-0 deriv activation (raw in-place -> silu'*ds)
  t_nn_2d_r d_mlp_h1;   // [M x 128]       layer-1 value activation
  t_nn_2d_r d_mlp_dh1;  // [M x 128]       layer-1 deriv activation
  t_nn_2d_r d_mlp_R;    // [M x n_out(last)=320] value output GEMM (LayoutRight); scattered
                        // into the LayoutLeft d_R1_nl by MLPScatterR
  t_nn_2d_r d_mlp_dR;   // [M x n_out(last)=320] deriv output GEMM; scattered into d_DR1_nl
  int mlp_M = 0;                 // current bonds = chunk_size * maxneigh
  int mlp_scatter_which = 0;     // MLPScatterR target: 0 = R->d_R1_nl, 1 = DR->d_DR1_nl
  int mlp_act_layer = 0;         // layer index for the activation kernel
  int mlp_act_n = 0;             // n_out width for the activation kernel
  NNScalar mlp_act_norm = 1;     // norm for the activation kernel
  int mlp_assemble_idx = 0;      // spbf idx for the assembly kernel
  // Layer 0 of the radial MLP split by input block: input = [cheb(nradbase), emb(mu_i), emb(mu_j)], so the
  // embedding part of the pre-activation is a per-element constant (linear in n_elements):
  //   s_j = sum_k T_k W0c(k, j) + E(mu_i, j) + E(mu_j, d1 + j),  norm0 and the bias folded in.
  t_nn_2d_r d_mlp_W0c[GRACE3L_N_SPBF];  // [nradbase x d1]
  t_nn_2d_r d_mlp_E[GRACE3L_N_SPBF];    // [n_elements x 2 d1]
  int mlp_split[GRACE3L_N_SPBF] = {0, 0, 0};
  t_nn_2d_r d_mlp_W0c_cur, d_mlp_E_cur;  // the current layer's tables (launch state for TagMLPLayer0*)
  // device pointer-array buffer for blas::sgemm_batched (ReduceN). One packed
  // buffer holds [A ptrs | B ptrs | C ptrs], each block sized reduceN_max_conn, filled
  // on host + deep_copied per call (base pointers change only on grow, but rebuilding
  // is cheap vs the GEMMs). uintptr_t storage reinterpreted as float** for the batched GEMM.
  Kokkos::View<uintptr_t*, DeviceType> d_redptr;
  Kokkos::View<uintptr_t*, Kokkos::HostSpace> h_redptr;   // explicit HostSpace (portable;
                                                          // ::HostMirror alias trips some nvcc)
  // per-reduce pointer arrays of the batched GEMMs, uploaded again only when a buffer moved (the synchronous upload
  // per call was a host fence per reduce); variant 0 forward, 1 backward, 2 / 5 grouped row-major forward /
  // backward, 3 / 4 grouped forward / backward
  struct RedPtrs { Kokkos::View<uintptr_t *, DeviceType> d; uintptr_t key[6] = {0, 0, 0, 0, 0, 0}; };
  RedPtrs red_ptrs[GRACE3L_MAX_REDUCES][6];
  const uintptr_t *red_upload(int idx, int var, int n, const uintptr_t key[6]);
  int reduceN_max_conn = 0;
#endif

  // per-call backward launch state (mirrors the forward per-call state members)
  t_nn_3d rmsnorm_adj_in, rmsnorm_adj_out;                 // TagRMSNormBwd
  t_nn_3d eqnorm_adj_out, eqnorm_adj_in;                   // TagEqNormBwd
  t_nn_3d reduceN_adj_out, reduceN_adj_in;                 // TagReduceNBwd
  int reduceN_bwd_instr = 0;                               // current reduce instruction index
  t_nn_3d fc_adj_out, fc_adj_left, fc_adj_right;           // TagFCBwd{Left,Right}
  t_nn_3d cpl_adj_out, cpl_adj_left, cpl_adj_right;        // TagCPLBwd{Couple,Project}
  int cpl_bwd_nmax = 0;                                    // max(n_left,n_right) stride for TagCPLBwdProject
  int cpl_bwd_projected = 0;                               // lproj/rproj supplied by BLAS

  // ---------- Global arrays used by the comm subsystem (size = nall) ----------
  t_nn_3d_rg d_I_global;         // [nall, I_n_funcs, I_n_out] — I features (round-1 indicator eq1_norm) for forward comm
  t_nn_3d_rg d_grad_I_global;    // [nall, I_n_funcs, I_n_out] — grad_I for reverse comm
  // Round-2 (L3) indicator: eq2_norm forward-comm'd to ghosts. Sized
  // [nall, I2_n_funcs, I2_n_out] — parity-doubled lm axis, distinct from d_I_global.
  t_nn_3d_rg d_eq2_norm_global;  // [nall, I2_n_funcs, I2_n_out] — round-2 indicator (eq2_norm)

  // ---- natom promotions of d_rho1_norm/d_rho2_norm so the P3
  // readout can read atoms from EARLIER chunks after P1/P2's chunk-scratch
  // d_rho{1,2}_norm has been overwritten by later chunks (multi-chunk forward
  // correctness). UNLIKE d_I_global/d_eq2_norm_global above, these are PURELY
  // LOCAL (no MPI comm — the 3L readout only ever needs OWNED-atom densities),
  // so they are filled by a dedicated per-chunk scatter (ScatterRho1Norm/
  // ScatterRho2Norm) indexed by the dense list-order ii+chunk_offset, NOT
  // d_ilist(ii+chunk_offset). Dims mirror d_rho1_norm/d_rho2_norm's trailing
  // dims exactly; sized to nall in grow_global() (safe superset of inum).
  t_nn_3d_g d_rho1_norm_full;    // [nall, 32, 1] — natom copy of d_rho1_norm
  t_nn_3d_g d_rho2_norm_full;    // [nall, n_out, n_lm] — natom copy of d_rho2_norm

  // Host mirrors for MPI communication
  t_nn_3d_rg_host h_I_global;
  t_nn_3d_rg_host h_grad_I_global;
  t_nn_3d_rg_host h_eq2_norm_global;
  t_nn_3d_rg_host h_eq2_norm_adj;   // reverse-comm host fallback

  // ==========================================================================
  // Constant device weight storage (populated by copy_weights_to_device).
  //
  // These are POD-of-Kokkos::View structs held in FIXED C-arrays so the whole
  // set is safe to capture inside a `*this` tag-functor (as GRACE-2L does):
  // Views/scalars/C-arrays-thereof are device-copyable; std::vector would NOT
  // be, so it is deliberately confined to the host GRACE3LModel. Downstream
  // kernels index these arrays positionally (order == the npz *_names lists,
  // also mirrored in grace_model->{spbf,prod,reduce,fc,eqnorm,rmsnorm}); use
  // grace_model to map an instruction name to its array index at launch time.
  //
  // NORM CONVENTION: raw weight + separate `norm` scalar (kernel applies once),
  // EXCEPT SPBF z_tr/z_proj which fold the norm into a per-element precompute.
  // ==========================================================================

  // --- SPBF (A1 scalar, A2/A3 equivariant) ---
 public:   // device weight structs: used in signatures of the public lambda-holding launchers
  struct DeviceSPBF {
    int equivariant = 0;
    int n_rad_max = 0, n_rad_basis = 0, spbf_lmax = 0, Lmax = 0, p = 0;
    int nfunc = 0, n_lm_ind = 0;        // equivariant only
    int mlp_n_layers = 0, mlp_max_dim = 0;
    GeomScalar rcut = 0, inv_avg_n_neigh = 1;
    t_int_1d l_tile;                    // [nlm]
    // radial MLP: W padded [n_layers, max_in, max_out]; bias padded
    // [n_layers, max_out] (hidden rows filled, output row zero); norms/dims
    t_nn_3d mlp_W;
    t_nn_2d mlp_b;
    t_nn_1d mlp_norms;                  // [n_layers]
    t_int_1d mlp_dims;                  // [n_layers+1]
    // scalar mode (A1): per-element species modulation (norm folded once)
    t_nn_2d z_tr;                       // [n_elements, n_rad_max]
    // equivariant mode (A2/A3): per-element L=0 chem injection (norm folded once)
    t_nn_2d z_proj;                     // [n_elements, n_rad_max]
    t_nn_1d chem_l0_mask;              // [n_lm_ind]
    t_nn_2d cg_W;                      // [(lmax+1)^2 * n_lm_ind, nfunc]
    // sparse forms of cg_W. The CG-coupling matrix is ~99% zeros
    // (mean ~3-6 nnz per row/column), so the dense couple loops streamed
    // mostly zeros. Ascending index order within each column/row preserves
    // the dense loops' accumulation order exactly.
    t_int_1d cg_col_ptr;               // [nfunc+1]  CSC (SPBFFused couple)
    t_int_1d cg_row_idx;               // [nnz]
    t_nn_1d cg_col_val;                // [nnz]
    t_int_1d cg_row_ptr;               // [P+1]      CSR (ComputeAdjProd)
    t_int_1d cg_col_idx;               // [nnz]
    t_nn_1d cg_row_val;                // [nnz]
    // the CSR rows as ELL [slot][row], padded with (col 0, val 0): independent loads in the fast backward
    int ell_w = 0;
    t_int_1d cg_ell_col;               // [ell_w * P]
    t_nn_1d cg_ell_val;                // [ell_w * P]
    // the same ELL rows and the CSC entries as (index, fp32 value bits) pairs: one 8-byte load per entry
    t_int_1d cg_ell2;                  // [ell_w * P][2]
    t_int_1d cg_csc2;                  // [nnz][2]
  };
  DeviceSPBF d_spbf[GRACE3L_N_SPBF];
  int n_spbf = 0;

  // --- cp_l products ---
  struct DeviceCPL {
    int rank = 0, nfunc = 0, n_left = 0, n_right = 0;
    int n_groups_left = 0, n_groups_right = 0, n_cg = 0;
    NNScalar norm_u = 1, norm_v = 1;
    t_nn_3d U;                          // [n_groups_left, rank, n_left]
    t_nn_3d V;                          // [n_groups_right, rank, n_right]
    t_int_1d group_left, group_right;   // [n_lm_left], [n_lm_right]
    t_int_1d left_ind, right_ind, m_sum_ind;   // [n_cg]
    // the three couple indices packed into ONE int32 per entry
    // (m_sum_ind<<16 | right_ind<<8 | left_ind) so the fused couple loops
    // stream 8 B/entry (packed+cg) instead of 16 B. Bounds checked at load.
    t_int_1d cg_packed;                 // [n_cg]
    t_nn_1d cg;                         // [n_cg]
    t_int_1d out_l;                     // [nfunc]
    // BLAS CPLBwdProject scratch. Per-lm-slab (w) group-gathered +
    // transposed U/V, tightly packed so each w-slab is a col-major [rank x n]
    // matrix (ldb=rank, w-stride = n*rank) for blas::sgemm_strided_batched.
    // Uw(w,n,r) = U(group_left(w), r, n); Vw(w,n,r) = V(group_right(w), r, n).
    t_nn_3d_r Uw;                       // [n_lm_left,  n_left,  rank]
    t_nn_3d_r Vw;                       // [n_lm_right, n_right, rank]
    // [Uw | Vw] per lm (n_lm_left == n_lm_right, n_left == n_right): a self-product (same input on both sides)
    // projects both sides in one GEMM; norms folded in when norm_u != norm_v (lr_alpha = 1 then)
    int lr_ok = 0;
    NNScalar lr_alpha = 1;
    t_nn_3d_r Wlr;                      // [n_lm, n_left, 2 rank]
    // Couple terms in gather form (one write per element, no atomics):
    //   forward by output f:    (left index, right index, cg)  foff[nfunc+1]
    //   VJP by left index w:    (output, right index, cg)      loff[n_lm_left+1]
    //   VJP by right index w:   (output, left index, cg)       roff[n_lm_right+1]
    t_int_1d foff, fa, fb, loff, la, lb, roff, ra, rb;
    t_int_1d fcol;                      // [nfunc] output column of forward row it (rows in file order: L1 locality)
    // couple by coupling path (grace_3l_cpl_path.h): path_n paths; per side s (0: left blocks outer, forward and
    // left adjoint; 1: right blocks outer, right adjoint) the tables ob / pr / pt, cg per compile-time candidate
    int path_n = 0, path_nob[2] = {0, 0};
    t_int_1d path_ob[2], path_pr[2], path_pt[2];
    t_nn_1d path_cg;
    t_nn_1d fcg, lcg, rcg;
  };
  DeviceCPL d_prod[GRACE3L_MAX_PRODS];
  int n_prods = 0;

  // ---- cp_l generic-product scratch + per-call launch state ----
  // Rank-axis projections written by TagCPLProject, read by TagCPLCouple. Sized
  // once to the max rank / max n_lm over ALL products so the same two views
  // serve every compute_cp_l call (A1_2, A1_3, ... A3_2). Per-(atom,rank)
  // ownership in Pass B makes the segment-sum serial -> no atomics needed.
  // Projections of product p live in its slot of d_cpl_store as [chunk, 2 rank, n_lm] (lproj = ranks 0..rank-1,
  // rproj = rank..2 rank-1, column-major slabs with ld = slot capacity), kept from the forward for the backward of
  // the same chunk and evaluation (cpl_stamp); the backward adjoints share one such scratch (d_cpl_adjbuf).
  // cpl_bind(p) points the views below at product p.
  t_nn_3d_s d_cpl_lproj, d_cpl_rproj;
  t_nn_1d d_cpl_store, d_cpl_adjbuf;
  long cpl_slot_off[GRACE3L_MAX_PRODS] = {0};
  long cpl_stamp[GRACE3L_MAX_PRODS] = {0};
  long cpl_gen = 0;
  int cpl_ld = 0;
  // atoms per rank row the cp_l path kernels map threads over: the chunk rounded up to 64 within the row stride cpl_ld,
  // so that on a 64-row-aligned stride (HIP, grow()) every wavefront reads 64 consecutive, aligned floats; the chunk
  // where the stride is the chunk
  int cpl_row_width() const { return std::min(cpl_ld, (chunk_size + 63) / 64 * 64); }
  int cpl_scratch_rank = 0, cpl_scratch_nlm = 0;
  void cpl_bind(const DeviceCPL &p);
  long cpl_stamp_now() const { return cpl_gen * 1048576L + chunk_offset + 1; }
  // Captured by the TagCPL* functors for the current compute_cp_l call.
  DeviceCPL cpl_p;
  t_nn_3d cpl_in_left, cpl_in_right, cpl_out;
  int cpl_nlm = 0;                     // max(n_lm_left, n_lm_right) for current product

  // --- fused equiv-SPBF (A2/A3) launch state (captured by TagComputeSPBFFused) ---
  int spbf_fused_idx = -1;             // idx_spbf for the current fused equiv-SPBF launch
  t_nn_3d_rg spbf_fused_indicator;       // per-neighbor indicator source: d_I_global (A2) or d_eq2_norm_global (A3)
  t_nn_3d spbf_fused_out;              // output basis: d_A2 or d_A3

  // --- fused equiv-SPBF backward launch state (captured by TagSPBFBwdFused) ---
  int spbf_bwd_idx = -1;               // idx_spbf_A2 or idx_spbf_A3
  t_nn_3d_rg spbf_bwd_target;            // message-adjoint scatter target: d_grad_I_global (A2) or d_eq2_norm_adj (A3)
  enum { SPBF_BWD_TILE = 32 };         // neighbours staged per shared a_nl tile

  // --- FunctionReduceN ---
  struct DeviceReduce {
    int n_out = 0, n_funcs = 0, n_instr = 0;
    int only_invar = 0, elem_dep = 0;
    int n_in[GRACE3L_MAX_REDUCE_INSTR];
    int w_shape[GRACE3L_MAX_REDUCE_INSTR];
    int n_conn[GRACE3L_MAX_REDUCE_INSTR];
    NNScalar norm[GRACE3L_MAX_REDUCE_INSTR];
    // W stored uniformly 4D: [n_types, n_out, n_in, w_shape] with
    // n_types = elem_dep ? n_elements : 1  (kernel uses e = elem_dep?mu_i:0).
    t_nn_4d W[GRACE3L_MAX_REDUCE_INSTR];
    t_int_1d collect_ind[GRACE3L_MAX_REDUCE_INSTR];   // [n_conn]
    t_int_1d w_l_tile[GRACE3L_MAX_REDUCE_INSTR];      // [n_conn]
    t_int_1d total_sum_ind[GRACE3L_MAX_REDUCE_INSTR]; // [n_conn]
    t_nn_1d norm_map;                   // [n_funcs] (or [1] for only_invar)
    int has_norm_map = 0;
    // BLAS batched-GEMM scratch for elem-indep, non-only_invar reduces
    // (n_instr==1). Wsc[c] = norm*norm_map(tsi(c)) * W(0,:,:,wlt(c)), stored as a
    // col-major [n_out x n_in] block per connection (c-stride n_out*n_in, ld n_out).
    // Serves BOTH forward (rounds; alpha=1,OP_N/OP_T) and backward (single batch;
    // alpha=1,OP_N/OP_N,beta=1) — the norm folding is identical for both.
    t_nn_1d Wsc;                        // [n_conn * n_out * n_in]  (instr 0 only)
    // Blocked hand path (reduces without the BLAS form): norm-folded weights with n_out fastest,
    // Wb[wb_off[ins] + ((e * w_shape + t) * n_in + n) * n_out + k], and the connections of all
    // instructions grouped by output f (CSR: roff[f] .. roff[f+1] -> (rins, rci, rtile)).
    int blk_ok = 0;
    int ci_unique[GRACE3L_MAX_REDUCE_INSTR] = {};   // collect_ind of the instruction has no repeats
    long wb_off[GRACE3L_MAX_REDUCE_INSTR];
    t_nn_1d Wb;
    t_int_1d roff, rins, rci, rtile;
  };
  DeviceReduce d_reduce[GRACE3L_MAX_REDUCES];
  int n_reduces = 0;
  // host-side per-reduce metadata for the BLAS batched path (index-parallel
  // to d_reduce). Holds the host copies of the connection indices + the forward round
  // schedule used to build the batched-GEMM device pointer arrays each call.
  struct ReduceHost {
    bool gemm_form_ok = false;         // elem_dep==0 && only_invar==0 && n_instr==1
    int n_conn = 0, n_out = 0, n_in = 0;
    std::vector<int> ci, tsi;          // [n_conn] collect_ind / total_sum_ind (host)
    std::vector<int> fwd_order;        // [n_conn] connections grouped by round
    int nrounds = 0;
    int round_ptr[GRACE3L_MAX_REDUCE_ROUNDS + 1] = {0};
    // Grouped form (the producer's output columns are ordered so the connections of every output f are the
    // contiguous columns gstart[f] .. gstart[f] + gn[f] - 1; see group_reduce_inputs): out(:,:,f) = in(:, :, block f)
    // as one [cs x gn n_in] matrix times Bg block f [gn n_in x n_out]; outputs batched by equal gn.
    bool grouped = false, has_empty = false;
    std::vector<int> gstart, gn;
    std::vector<long> boff;
    std::vector<std::vector<int>> gsets;   // outputs f with the same gn (> 0)
  };
  ReduceHost h_reduce[GRACE3L_MAX_REDUCES];

  // --- FCRight2Left ---
  struct DeviceFC {
    int n_out = 0, left_coefs = 0;
    int n_funcs_left = 0, n_funcs_right = 0;
    int n_in_left = 0, n_in_right = 0, w_shape_left = 0, w_shape_right = 0;
    NNScalar norm_left = 1, norm_right = 1;
    t_nn_3d w_left;                     // [n_out, n_in_left, w_shape_left] (empty if !left_coefs)
    t_nn_3d w_right;                    // [n_out, n_in_right, w_shape_right]
    t_int_1d w_tile_left, w_tile_right; // [n_funcs_left], [n_funcs_right]
    t_int_1d collect_to, collect_from;  // [n_funcs_left], [n_funcs_right]
    t_nn_1d norm_out_factor;            // [n_funcs_left]
    // GEMM form when left_coefs = 0 and every function w has exactly one connection with src = dst = w:
    //   out(:,:,w) = nof[w] left(:,:,w) + right(:,:,w) Bw[w],  Bw[w](n, k) = norm_right nof[w] w_right[k, n, tile(w)]
    int gemm_ok = 0;
    t_nn_3d_r Bw;                       // [n_funcs, n_out, n_in_right]: col-major [n_in x n_out] per w
  };
  DeviceFC d_fc[GRACE3L_MAX_FCS];
  int n_fcs = 0;
 protected:

  // ---- FunctionReduceN generic-kernel scratch + per-call launch state.
  // Mirrors the cp_l pattern (DeviceCPL/cpl_p above): caller supplies the
  // per-instruction input views by name-matching grace_model->reduce[i].instr[j]
  // .name at the call site; the device kernel itself never does string lookup. ----
  DeviceReduce reduceN_r;
  t_nn_3d reduceN_inputs[GRACE3L_MAX_REDUCE_INSTR];
  t_nn_3d reduceN_out;

  // ---- FCRight2Left per-call launch state ----
  DeviceFC fc_p;
  t_nn_3d fc_in_left, fc_in_right, fc_out;

  // --- EquivariantRMSNorm ---
  struct DeviceEqNorm {
    int center_l0 = 0, n_groups = 0, n_out = 0, M = 0;
    NNScalar eps = 0;
    t_nn_2d affine_weight;              // [n_groups, n_out]
    t_nn_1d degree_weights;            // [M]
    t_int_1d expand_index;             // [M]
    t_nn_1d l0_mask;                   // [M]
  };
  DeviceEqNorm d_eqnorm[GRACE3L_MAX_EQNORMS];
  int n_eqnorms = 0;

  // --- InvariantLayerRMSNorm ---
  struct DeviceRMSNorm {
    int type = 0, n_out = 0, scale_len = 0;
    t_nn_1d scale;                     // [scale_len]
  };
  DeviceRMSNorm d_rmsnorm[GRACE3L_MAX_RMSNORMS];
  int n_rmsnorms = 0;

  // ---- EquivariantRMSNorm per-call launch state (same pattern as
  // reduceN_r/fc_p above: caller stashes metadata + in/out views, kernel is
  // pure tag-functor code with no string lookups). ----
  DeviceEqNorm eqnorm_p;
  t_nn_3d eqnorm_in, eqnorm_out;

  // ---- InvariantLayerRMSNorm per-call launch state ----
  DeviceRMSNorm rmsnorm_p;
  t_nn_3d rmsnorm_in, rmsnorm_out;

  // --- Chemical embedding / shifts / readout ---
  t_nn_2d d_chem_embed;                 // [n_elements, embedding_size]
  t_nn_1d d_shifts;                     // [n_elements]
  NNScalar output_scale = 1;

  // Energy readout MLP (no bias): W padded [n_layers, max_in, max_out]
  t_nn_3d d_energy_W;
  t_nn_1d d_energy_norms;               // [n_layers]
  t_int_1d d_energy_dims;               // [n_layers+1]
  int energy_n_layers = 0;
  int energy_max_dim = 0;
  int energy_activation = 0;            // 0=silu, 1=tanh

  // Per-atom energy. Deliberately KKA (fp64 by default),
  // NOT NNScalar/GeomScalar: the MLP math runs in NNScalar (matching TF's fp32
  // readout), but the accumulated per-atom result — and the eng_vdwl reduction
  // over it — must stay fp64 regardless of which Grace3LKK<> NNScalar/
  // GeomScalar instantiation (Mixed or FP32) is active. Chunk-local (size
  // natom, like GRACE-2L's d_e_atom); indexed by ii exactly like rho3_norm.
  Kokkos::View<KKA*, DeviceType> d_e_atom;

  // Per-element-pair cutoffs
  t_geom_2d d_bond_cutoff;              // [n_elements, n_elements]

  // Spherical harmonics precomputed coefficients (verbatim from GRACE-2L)
  t_geom_1d d_idx_sph, alm, blm, cl, dl;
  int idx_sph_max = 0;
  t_int_1d d_l_from_lm;                 // [(lmax+1)^2] -> l

  // ---------- Helper functions ----------
  void grow(int natom, int maxneigh);
  bool realloc_all = true;    // set by set_model(): grow() and grow_global() resize every buffer to the new model
  // Allocate the global (nall-sized) inter-layer comm arrays. Sized
  // to I_n_funcs/I_n_out (set in set_model()); mirrors GRACE-2L's grow_global.
  void grow_global(int nall);
  // Generic cp_l GeneralProductFunction product: out[a,r,f] from
  // left/right [a,n,lm] for product p over the current chunk. Reused for every
  // A*_* cp_l product; caller supplies the per-product weights + in/out views.
  void compute_cp_l(const DeviceCPL &p, const t_nn_3d &in_left,
                    const t_nn_3d &in_right, const t_nn_3d &out);
  // fused equiv-SPBF (A2/A3) launcher — one team per (ii,n); the prod
  // slice + dense CG couple live in team shared (replaces the register-spilling
  // TagComputeA2/TagComputeA3). idx picks the SPBF weights; indicator is the
  // per-neighbor message source (d_I_global for A2, d_eq2_norm_global for A3);
  // out is the basis output (d_A2 or d_A3).
  void compute_spbf_equiv(int idx, const t_nn_3d_rg &indicator, const t_nn_3d &out);
  // Generic FunctionReduceN: out[a,k,f] from per-instruction gathered
  // inputs for reduce r over the current chunk. Handles multi-instruction sums,
  // elem_dep W indexing (W stored uniformly 4D [n_types,n_out,n_in,w_shape],
  // n_types=1 if !elem_dep, so the kernel always indexes e=elem_dep?mu_i:0 with
  // no branch on the W read), only_invar collapse (every connection accumulates
  // into f=0, ignoring total_sum_ind) vs the general scatter-by-total_sum_ind,
  // and an optional norm_map applied once at the end. Reused for every reduce
  // (A*_*_red, eq*, rho*, ...); caller supplies inputs[j] matching instr[j].name.
  void compute_reduceN(const DeviceReduce &r, const t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR],
                       const t_nn_3d &out);
  // Generic FCRight2Left: out[a,k,lm] = left_t[lm,a,k] with right's
  // per-connection contribution scattered in via collect_from/collect_to.
  // Handles left_coefs=True (left projected via w_left/w_tile_left, like the
  // right branch) and left_coefs=False (left already at this FC's n_out, just
  // transposed) generically; always applies norm_out_factor (the loader
  // requires it unconditionally, matching the reference's `if W.has(...)` being
  // always-true for this exporter). Reused for every FC (A*_2a/2b, ...);
  // caller supplies the left/right input views.
  // compute_fc: public (extended device lambdas), declared next to compute_Y_bond_chunk.
  // Generic EquivariantRMSNorm: x=[atoms,n_feat=n_out,n_lm=M].
  // center_l0 (per-(atom,lm) cross-feature mean, masked to l==0 slots) ->
  // ONE degree-balanced RMS scalar per atom (mean over feat of sum over lm of
  // dw[lm]*x^2) -> per-(feat,lm) affine scale = affine_weight[expand_index[lm], feat].
  // Reused for eq1_norm/eq2_norm.
  void compute_equiv_rms_norm(const DeviceEqNorm &e, const t_nn_3d &in, const t_nn_3d &out);
  // Generic InvariantLayerRMSNorm: x=[atoms,n_out,n_lm]. Branches
  // on scale_len vs n_out exactly like the reference: full (scale_len==n_out) ->
  // every channel scaled by rsqrt(mean_over_channel(x^2)+eps); only_nonlin
  // (scale_len==n_out-1) -> channel 0 passthrough, channels 1..n_out-1 scaled
  // by rsqrt(mean_over_the_nonlin_channels(x^2)+eps). eps=1e-10 fixed (not
  // loaded from the npz — matches the reference's hardcoded F32(1e-10)). Reused
  // for rho1_norm/rho2_norm/rho3_norm.
  void compute_invariant_rms_norm(const DeviceRMSNorm &r, const t_nn_3d &in, const t_nn_3d &out);

  // ---- backward (reverse-mode adjoint) routines ----
  // Readout backward: seeds d_rho{1,2,3}_norm_adj = dE/d(rho_k_norm) (all equal,
  // since the 3 densities enter the readout symmetrically). ch0=output_scale
  // (linear skip), ch1..31 from the MLP backward; every lm>0 slot = 0.
  void readout_bwd();
  // Generic InvariantLayerRMSNorm VJP (mirrors compute_invariant_rms_norm): given
  // the forward input `in`, forward output `out` and adj_out, writes adj_in
  // (assignment; the rho densities have a single consumer). Handles both branches
  // (full / only_nonlin) data-driven off scale_len, eps=1e-10.
  void compute_invariant_rms_norm_bwd(const DeviceRMSNorm &r, const t_nn_3d &in,
      const t_nn_3d &out, const t_nn_3d &adj_out, const t_nn_3d &adj_in);
  // Generic FunctionReduceN VJP (mirrors compute_reduceN): scatters adj_out into
  // each per-instruction input adjoint via the transpose contraction, applying
  // norm + norm_map. ACCUMULATES (+=) into adj_inputs[j] (inputs fan out to many
  // reduces/products) — caller must zero the adjoint buffers first.
  void compute_reduceN_bwd(const DeviceReduce &r, const t_nn_3d &adj_out,
      const t_nn_3d adj_inputs[GRACE3L_MAX_REDUCE_INSTR]);
  // adj_in = VJP of a single-input reduce, assigned (adj_in has no other writer): one beta = 0 GEMM per output
  // block when the grouped blocks cover every column of adj_in once, else zero + compute_reduceN_bwd
  void compute_reduceN_bwd_set(const DeviceReduce &r, const t_nn_3d &adj_out, const t_nn_3d &adj_in);
  // Generic FCRight2Left VJP (mirrors compute_fc): ACCUMULATES (+=) into adj_left
  // (identity/left_coefs path) and adj_right (scatter via collect_from/to, atomic).
  // compute_fc_bwd: public, see compute_fc.
  // Generic cp_l VJP (product rule; spec §4.6). Recomputes the forward Pass-A
  // projections (shared scratch is overwritten per call), backprops the CG couple
  // to the rank-projection adjoints, then the U/V projection to adj_left/adj_right.
  // ACCUMULATES (+=). Self-products pass the same buffer for adj_left==adj_right;
  // both contributions accumulate correctly (serial per (a,n,w) thread).
  void compute_cp_l_bwd(const DeviceCPL &p, const t_nn_3d &in_left,
      const t_nn_3d &in_right, const t_nn_3d &adj_out,
      const t_nn_3d &adj_left, const t_nn_3d &adj_right);
  // L3 equivariant-SPBF backward: scatters d_A3_adj into d_eq2_norm_adj (the L3
  // message adjoint) over every (owned atom i, radial n, neighbor j). j may be
  // owned OR ghost -> d_eq2_norm_adj is nall-sized, atomic-add scatter. The
  // species-injection (z_proj/chem_l0_mask) term is constant and contributes
  // nothing to the indicator adjoint. NOT reverse-comm'd here.
  void compute_A3_spbf_bwd();
  // fill d_adj_prod for the given layer (spbf_idx = idx_spbf_A2 or _A3,
  // A_adj = d_A2_adj or d_A3_adj) BEFORE its SPBF-bwd + ForceEquiv consumers.
  void compute_adj_prod(int spbf_idx, const t_nn_3d &A_adj);
  // Generic EquivariantRMSNorm VJP (mirrors compute_equiv_rms_norm).
  // Given the forward input `in` and adj_out, ASSIGNS adj_in (single consumer:
  // eq2 feeds only eq2_norm). Recomputes center_l0 mean + degree-balanced rms
  // from `in` (the forward caches neither). Reused for eq1_norm in the L1 backward.
  void compute_equiv_rms_norm_bwd(const DeviceEqNorm &e, const t_nn_3d &in,
      const t_nn_3d &adj_out, const t_nn_3d &adj_in);
  // L2 equivariant-SPBF backward: scatters d_A2_adj into d_grad_I_global
  // (the round-1 message adjoint d_eq1_norm_adj, owned partial) over (owned atom i,
  // radial n, neighbor j; j owned OR ghost -> atomic scatter). The z_proj/
  // chem_l0_mask species-injection term is constant -> no indicator adj. NOT
  // reverse-comm'd here. Requires d_R1_nl to hold A2's radial (caller
  // recomputes it — A3's forward overwrote the shared d_R1_nl scratch).
  void compute_A2_spbf_bwd();
  // shared launcher for the fused team+shared equiv-SPBF backward
  // (TagSPBFBwdFused). idx = idx_spbf_A2 or idx_spbf_A3; adj_target = the
  // nall-sized message-adjoint buffer the kernel atomic-add scatters into
  // (d_grad_I_global for A2, d_eq2_norm_adj for A3).
  void compute_spbf_bwd_fused(int idx, const t_nn_3d_rg &adj_target);
  // radial MLP forward + its r-derivative for the layer selected by
  // radial_mlp_spbf. Writes d_R1_nl (R, matching the forward) and d_DR1_nl (dR/dr).
  // Bit-identical R to TagComputeMLPRadial.
  void compute_mlp_radial_deriv();
  // value-only radial MLP forward for layer `radial_mlp_spbf`, writing
  // d_R1_nl. Dispatches to cuBLAS SGEMM (CUDA) or the hand kernel (fallback).
  // Replaces the inline TagComputeMLPRadial launches in l{1,2}_forward_chunk.
  void compute_mlp_radial();
#ifdef GRACE_KK_GEMM
  // helper: C[M x n_out] = alpha * In[M x n_in] * Wg (true-fp32 SGEMM,
  // CUBLAS_PEDANTIC_MATH). Row-major operands; ld* are the physical row strides
  // (allocated widths): ldW=n_out for the tight Wg, ldIn=In's alloc width,
  // ldC=C's alloc width (e.g. L1_nradmax*(lmax+1) when C aliases d_R1_nl).
  void mlp_sgemm(const NNScalar* Wg, const NNScalar* In, NNScalar* C,
      int n_out, int M, int n_in, int ldW, int ldIn, int ldC, NNScalar alpha);
  // one-time layer-0 transpose/leading-dim self-test vs a host reference
  // (aborts on mismatch). Called from setup() before the BLAS path is used.
  void mlp_blas_selftest();
  // helper: batched projection SGEMM for TagCPLBwdProject. For each lm-slab
  // w in [0,batch): C(:,:,w)[M x n] += alpha * A(:,:,w)[M x rank] * Bw[w][rank x n].
  // A/C are LayoutLeft slabs (col-major, ld = extent(0), w-stride = extent(0)*extent(1));
  // Bw is the tightly-packed Uw/Vw (col-major [rank x n], ldb=rank, w-stride=n*rank).
  // beta=1 (accumulates into the shared adj buffer). True-fp32 (PEDANTIC handle).
  void cpl_project_sgemm_batched(const NNScalar* A, long strideA, int ldA,
      const NNScalar* Bw, int n, int rank, NNScalar* C, long strideC, int ldC,
      int M, int batch, NNScalar alpha);
  // BLAS batched-GEMM ReduceN forward/backward (elem-indep, non-only_invar).
  // Return true if handled via BLAS; false -> caller runs the hand-kernel fallback.
  bool compute_reduceN_blas(const DeviceReduce &r, int idx,
      const t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR], const t_nn_3d &out);
  bool compute_reduceN_bwd_blas(const DeviceReduce &r, int idx,
      const t_nn_3d &adj_out, const t_nn_3d &adj_in);
  // fp32 only: BLAS U/V rank-projection for cp_l forward. Writes the global
  // lproj/rproj scratch (d_cpl_lproj/d_cpl_rproj) via two strided-batched GEMMs (reusing
  // the Uw/Vw weights of the backward projection), replacing the shared-memory projection in
  // TagCPLFused; the couple then reads lproj/rproj back (cpl_couple_gather). fp64 and builds without
  // a BLAS handle keep the fused kernel (no SGEMM benefit, and un-fusing reintroduces the round-trip
  // the fused kernel avoids).
  void compute_cp_l_project_blas(const DeviceCPL &p,
      const t_nn_3d &in_left, const t_nn_3d &in_right);
#endif
  // per-bond geometry-force kernels. Each ACCUMULATES (atomic +=) into
  // d_f_ij; caller zeros d_f_ij once before invoking all three layers.
  void compute_force_L1();
  void compute_force_equiv(int spbf_idx, const t_nn_3d &A_adj, int ind_stage, int n_lm_ind);
  void copy_weights_to_device();
  void precompute_harmonics();

 public:
  // Public because it contains an extended __host__ __device__ lambda; nvcc
  // forbids those inside private/protected members.
  // Fills d_Y_bond from d_rhats for the current chunk (uses member
  // chunk_size, maxneigh). Must be (re)called whenever d_rhats for the
  // current chunk changes before any kernel reads d_Y_bond.
  void compute_Y_bond_chunk();
  // forward-chunk recompute helpers (the L1/L2 per-chunk forward
  // bodies extracted from the P1/P2 loops, WITHOUT the global eq-scatter) and
  // the per-chunk force scatter.
  // P1 calls l1_forward_chunk() then the eq1_norm scatter; P5 (L1 backward)
  // re-runs l1_forward_chunk() to repopulate the chunk-scratch forward
  // intermediates. Likewise l2_forward_chunk() for P2/P4. scatter_forces_chunk()
  // scatters the current chunk's d_f_ij into f[] + virial. Public for the same
  // reason as compute_Y_bond_chunk() (they contain extended device lambdas).
  void l1_forward_chunk();
  void l2_forward_chunk();
  void scatter_forces_chunk();
  // cp_l couple and its VJP on the rank axis in gather form (thread per (atom, rank)), from d_cpl_lproj/rproj.
  void cpl_couple_gather(const DeviceCPL &p, const t_nn_3d &out);
  void cpl_couple_path(const DeviceCPL &p, const t_nn_3d &out);
  void cpl_vjp_path(const DeviceCPL &p, const t_nn_3d &adj_out);
  bool cpl_path = true;                 // couple by coupling path where the model allows (else table kernels)
  void compute_fc(const DeviceFC &f, const t_nn_3d &in_left, const t_nn_3d &in_right,
                  const t_nn_3d &out);
  void compute_fc_bwd(const DeviceFC &f, const t_nn_3d &adj_out,
      const t_nn_3d &adj_left, const t_nn_3d &adj_right);
  void cpl_couple_vjp_gather(const DeviceCPL &p, const t_nn_3d &adj_out);
  template<int TB, int SIDE> void cpl_vjp_t(const DeviceCPL &p, const t_nn_3d &adj_out);
  // Equivariant SPBF (A2/A3) for lmax 4 and 25 / 50 indicator components: forward with bonds staged in
  // blocks and a register-tiled outer product; backward (message adjoint + geometry force) in one kernel.
  bool spbf_fast_ok(int idx) const;
  int spbf_ltile_std[GRACE3L_N_SPBF] = {0, 0, 0};   // l_tile(y) == l(y)
  void spbf_transpose(const t_nn_3d &A, int NR, int nfunc, bool to_rowmajor);
  // A2 / A3 straight from d_spbf_T into their grouped reduce (no transposes): fast SPBF, single-input grouped
  // BLAS reduce with n_in = n_rad_max whose blocks cover every SPBF function.
  bool spbf_rm_ok(int si, int ridx) const;
  bool eqnorm_team_ok(int M) const;
  // EquivariantRMSNorm forward (bwd = false: Y = norm(X)) / VJP (bwd = true: DX from G = dE/dY), team per 8 atoms
  // (fields of DeviceEqNorm passed one by one: an extended lambda's parent may not take a protected type)
  void eqnorm_team(int center_l0, int n_out, int M, float eps, const t_nn_2d &aw, const t_nn_1d &dw,
                   const t_int_1d &ex, const t_nn_1d &mk, const t_nn_3d &X, const t_nn_3d &Y, const t_nn_3d &G,
                   const t_nn_3d &DX, bool bwd);
  void force_L1_fast();
  template<int W> void force_L1_fast_t();
  // Radial MLP as cubic-Hermite tables in x = r / rcut_ij, one block per ordered species pair of the run
  // (the MLP input holds both embeddings, so R depends on (mu_i, mu_j)). Built at setup() when the tables fit
  // spline_mb (MB; < 0: auto, a quarter of the device memory per rank within what is free, at least 2048); otherwise
  // the cuBLAS MLP path stays.
  int spline_n = 1024;                   // intervals of the layer-3 tables (layers 1-2: spline_n / 2)
  int spline_n_layer[3] = {0, 0, 0};     // per-layer intervals, 0: as spline_n (grace::spline_n_3l)
  double spline_mb = 2048.0;
  int spline_on = 0, n_sp = 0, spl_ld = 0;
  Kokkos::View<int *, DeviceType> d_sp_of_mu;   // model species -> species index of this run (-1: absent)
  Kokkos::View<float *, DeviceType> d_splR;     // [((p * spl_Ntot + spl_off[si] + k) * spl_ld + o][4], p = sp_i n_sp + sp_j
  G3LTab spl_tab(int si) const { return G3LTab{d_splR.data(), spl_N[si], spl_Ntot, spl_off[si], spl_ld}; }
  void build_radial_splines();
  Kokkos::View<NNScalar *, DeviceType> d_Bg[GRACE3L_MAX_REDUCES];   // grouped reduce weights
#ifdef GRACE_KK_GEMM
  bool compute_reduceN_grouped(int idx, const t_nn_3d &in, const t_nn_3d &out, bool bwd, bool assign = false);
  void compute_reduceN_grouped_rm(int idx, const t_nn_3d &out, bool bwd);
#endif
  // Multi-chunk: the L1 / L2 forward tensors the backward reads (product inputs, norm inputs / outputs) kept per
  // chunk, so the backward recomputes only the geometry. store_fwd: -1 auto (if it fits in 70 % of free memory),
  // 0 off, 1 on.
  // FC folded into the cp_l projection it feeds (GPU BLAS fp32 path): A*_2a = fc(A*_2_red, R) only enters
  // A*_3 = cp_l(A*_2a, R) and A*_2b only A*_4 = cp_l(A*_2b, A*_2b), and fc is linear,
  //   proj(A*_2a) = A*_2_red UL^T + R UR^T,  UL[w] = nof[w] P[w],  UR[w] = Bw[w] P[w]   (P[w] = norm_u U[w];
  //   Bw[w] = norm_right nof[w] w_right[tile(w)])
  // (both projection sides of the self-product A*_4 in one [U | V] block). A*_2a / A*_2b and their adjoints are
  // never formed; the store keeps A*_2_red instead. fc_fold: 0 off (default), 1 on (GRACE3L_FC_FOLD=1 too); GPU BLAS
  // fp32 path only (cuBLAS / rocBLAS / own GEMM; otherwise off, with a log line).
  int fc_fold = 0, fold_on[3] = {0, 0, 0};
  // fuse_radial: the radial evaluated from the cubic tables inside the kernels that use it (ComputeA1, ForceL1F, the
  // fast SPBF forward / backward of layers 2 and 3): no d_R1_nl / d_DR1_nl round trip and no table kernel where those
  // kernels run; only with the tables (spline_on) and the CUDA / HIP fp32 fast kernels
  int fuse_radial = 0;
  bool l1_radial_fused() const;
  void compute_A1_tables();
  t_nn_3d_r d_fold_UL[GRACE3L_MAX_PRODS], d_fold_UR[GRACE3L_MAX_PRODS];   // [n_lm, n_in, rank or 2 rank]
  void build_fc_fold();
  void cpl_fold_project(const DeviceCPL &p, const t_nn_3d &Lred, const t_nn_3d &R, const t_nn_3d &right);
  void compute_cp_l_fold(const DeviceCPL &p, const t_nn_3d &Lred, const t_nn_3d &R, const t_nn_3d &right,
                         const t_nn_3d &out);
  void compute_cp_l_fold_bwd(const DeviceCPL &p, const t_nn_3d &Lred, const t_nn_3d &R, const t_nn_3d &right,
                             const t_nn_3d &adj_out, const t_nn_3d &adj_Lred, const t_nn_3d &adj_R,
                             const t_nn_3d &adj_right);
  int store_fwd = -1, fwd_stored = 0;
  t_nn_3d st_none;                      // store slot unused with fc_fold (A*_2b)
  std::vector<t_nn_3d> st_fwd[12];
  std::vector<int> spl_mus;              // species of the last radial tables (spline_mb auto keeps them on a re-setup)
  int spl_N[3] = {0, 0, 0}, spl_off[3] = {0, 0, 0}, spl_Ntot = 0;    // tables: intervals and node offset per layer
  t_nn_3d st_base[12];
  t_nn_3d *st_ptr(int i);
  void setup_fwd_store();
  void select_fwd_store(int layer, int c);
  void restore_fwd_store();
  void geometry_chunk();
  void radial_from_splines(bool deriv);
  t_nn_1d d_pool_fwd, d_pool_adj;   // backing store of the pooled transient tensors (grow)
  void reduceN_blocked(const DeviceReduce &r, const t_nn_3d inputs[GRACE3L_MAX_REDUCE_INSTR], const t_nn_3d &out);
  void reduceN_bwd_blocked(const DeviceReduce &r, int ins, const t_nn_3d &adj_out, const t_nn_3d &adj_in);
  Kokkos::View<float **, Kokkos::LayoutRight, DeviceType> d_spbf_T;   // [chunk][f * NR + n] row-major A / dE/dA
  // MFMA SPBF backward: (Y, dYx, dYy, dYz) of every (bond, y) as one record [chunk][maxneigh][(lmax + 1)^2 * 4]
  Kokkos::View<GeomScalar ***, Kokkos::LayoutRight, DeviceType> d_spbf_ydy;
  // MFMA SPBF: the CG table as sliced ELL, rows (backward M build) and columns (forward), built on first use: slices of
  // 64 rows / columns of similar length (sorted by length), entries [slice offset + e * 64 + lane] as (index, fp32 bits)
  struct SpbfSell {
    t_int_1d perm, len, ent;        // [slots] row / column (-1: none), [slots] length, entries [n][2]
    std::vector<int> slc;           // [slices][2] (offset, width), passed to the kernels by value
    const void *src = nullptr;      // cg_row_ptr / cg_col_ptr the table was built from
  };
  SpbfSell d_spbf_rsell[GRACE3L_N_SPBF], d_spbf_csell[GRACE3L_N_SPBF];
  void spbf_mfma_tables(int idx);
  // rm: A stays in d_spbf_T (forward) / dE/dA is already there (backward); out / A_adj unused
  void spbf_fwd_fast(int idx, const t_nn_3d_rg &indicator, const t_nn_3d &out, bool rm = false);
  template<int TR> void spbf_fwd_fast_t(int idx, const t_nn_3d_rg &indicator, const t_nn_3d &out, bool rm);
  void spbf_bwd_fast(int idx, const t_nn_3d &A_adj, const t_nn_3d_rg &indicator, const t_nn_3d_rg &target,
                     bool rm = false);
  template<int H> void spbf_bwd_fast_t(int idx, const t_nn_3d &A_adj, const t_nn_3d_rg &indicator,
                                       const t_nn_3d_rg &target, bool rm);
  template<int H, int MB> void spbf_bwd_fast2_t(int idx, const t_nn_3d &A_adj, const t_nn_3d_rg &indicator,
                                               const t_nn_3d_rg &target, bool rm);
  // CDNA matrix-core variants of the two above (grace_3l_kk_spbf_mfma.h; default on gfx90a, opt-in on gfx94x,
  // environment GRACE3L_SPBF_MFMA=0 / 1)
  bool spbf_use_mfma(int idx) const;
  template<int NLI> void spbf_fwd_mfma_t(int idx, const t_nn_3d_rg &indicator, const t_nn_3d &out, bool rm);
  template<int NLI> void spbf_bwd_mfma_t(int idx, const t_nn_3d &A_adj, const t_nn_3d_rg &indicator,
                                         const t_nn_3d_rg &target, bool rm);
 protected:



  // Activation functions (used by the NN kernels)
  KOKKOS_INLINE_FUNCTION
  static NNScalar silu(NNScalar x) {
    return x / (NNScalar(1.0) + Kokkos::exp(-x));
  }

  KOKKOS_INLINE_FUNCTION
  static NNScalar tanh_act(NNScalar x) {
    return Kokkos::tanh(x);
  }


  // Comm helper views for Kokkos-native communication
  typename Traits::t_int_1d d_sendlist;
  typename Traits::t_double_1d v_buf;
  int comm_first;  // for unpack: starting ghost index
  int comm_nfo = 1; // values per atom of the current comm stage (device pack / unpack)

  // Host-side model data
  std::shared_ptr<GRACE3LModel> grace_model;
  std::vector<int> h_type2model;

  // reduce/FC array indices resolved by name-match at setup()
  // time (index-parallel with d_reduce/d_fc — see copy_weights_to_device,
  // which fills d_reduce[i]/d_fc[i] in the same order as grace_model->reduce
  // /fc). Set to -1 until resolved; setup() errors out if not found.
  int idx_reduce_A1_2_red = -1;
  int idx_fc_A1_2a = -1;

  // remaining L1 prod/reduce/FC/eqnorm/rmsnorm indices, resolved by
  // NAME-match at setup() time (same convention as idx_reduce_A1_2_red).
  int idx_prod_A1_3 = -1;
  int idx_prod_A1_4 = -1;
  int idx_reduce_A1_3_red = -1;
  int idx_fc_A1_2b = -1;
  int idx_reduce_A1_4_red = -1;
  int idx_reduce_eq1 = -1;
  int idx_reduce_rho1 = -1;
  int idx_eqnorm_eq1_norm = -1;    // resolved in set_model() (comm_width() before setup())
  int idx_rmsnorm_rho1_norm = -1;
  // SPBF layer index of "A2" (equivariant), resolved by name in
  // setup() (spbf order is [A1,A2,A3]; index 1, but resolved defensively).
  int idx_spbf_A2 = -1;
  // eq1/rho1 both collect the same 4 named L1 tensors {A1, A1_2_red,
  // A1_3_red, A1_4_red}; role[j] tells compute() which of those 4 views to
  // plug into inputs[j] for instruction j, resolved by matching
  // grace_model->reduce[idx]->instr[j].name (0=A1,1=A1_2_red,2=A1_3_red,
  // 3=A1_4_red) — NOT assumed positional, per the loaded instruction_names.
  int eq1_input_role[GRACE3L_MAX_REDUCE_INSTR];
  int rho1_input_role[GRACE3L_MAX_REDUCE_INSTR];

  // remaining L2 prod/reduce/FC/eqnorm/rmsnorm indices, resolved by
  // NAME-match at setup() time (same convention as the L1 idx_*
  // members above). idx_reduce_A2_red is the extra reduce L2 has that L1
  // does not (L1's products consume raw A1; L2's consume A2_red).
  int idx_reduce_A2_red = -1;
  int idx_prod_A2_2 = -1;
  int idx_reduce_A2_2_red = -1;
  int idx_fc_A2_2a = -1;
  int idx_prod_A2_3 = -1;
  int idx_reduce_A2_3_red = -1;
  int idx_fc_A2_2b = -1;
  int idx_prod_A2_4 = -1;
  int idx_reduce_A2_4_red = -1;
  int idx_reduce_eq2 = -1;
  int idx_reduce_rho2 = -1;
  int idx_eqnorm_eq2_norm = -1;    // resolved in set_model() (comm_width() before setup())
  int idx_rmsnorm_rho2_norm = -1;
  // eq2/rho2 both collect the same 4 named L2 tensors {A2_red, A2_2_red,
  // A2_3_red, A2_4_red}; role[j] tells compute() which of those 4 views to
  // plug into inputs[j] for instruction j, resolved by matching
  // grace_model->reduce[idx]->instr[j].name (0=A2_red,1=A2_2_red,2=A2_3_red,
  // 3=A2_4_red) — NOT assumed positional, per the loaded instruction_names.
  int eq2_input_role[GRACE3L_MAX_REDUCE_INSTR];
  int rho2_input_role[GRACE3L_MAX_REDUCE_INSTR];

  // L3 SPBF + remaining L3 prod/reduce/FC/rmsnorm indices, resolved
  // by NAME-match at setup() (same convention as the L2 idx_*
  // members). L3 is TERMINAL: it has A3_red (like A2_red) and the same
  // cp_l/reduce/fc chain, but produces ONLY rho3/rho3_norm — there is NO eq3.
  int idx_spbf_A3 = -1;
  int idx_reduce_A3_red = -1;
  int idx_prod_A3_2 = -1;
  int idx_reduce_A3_2_red = -1;
  int idx_fc_A3_2a = -1;
  int idx_prod_A3_3 = -1;
  int idx_reduce_A3_3_red = -1;
  int idx_fc_A3_2b = -1;
  int idx_prod_A3_4 = -1;
  int idx_reduce_A3_4_red = -1;
  int idx_reduce_rho3 = -1;
  int idx_rmsnorm_rho3_norm = -1;
  // rho3 collects the same 4 named L3 tensors {A3_red, A3_2_red, A3_3_red,
  // A3_4_red}; role[j] maps instruction j to one of those 4 views, resolved by
  // matching grace_model->reduce[idx_reduce_rho3].instr[j].name (NOT positional).
  // Only rho3 needs a role array here — there is no eq3.
  int rho3_input_role[GRACE3L_MAX_REDUCE_INSTR];

  // ---------- UQ / extrapolation grade (forward only) ----------
  // FP64 device storage regardless of NNScalar (numerical safety). The GMM feature
  // is the basis-RP projection of the three per-layer scalar reduces rho1/rho2/rho3
  // (R-row order [rho1, rho2, rho3]; density channels [full, rho1, rho2, rho3]).
  typedef Kokkos::View<double*, DeviceType>    t_uq_1d;
  // LayoutRight so the last (contiguous) index — split/reduced across team lanes in
  // the team-parallel ComputeUQ kernels — gives coalesced lane reads (rp_matrix over
  // rp_dim, centroids/inv_cov over the feature dims, d_uq_z over rp_dim). Transparent
  // to the logical-index copy_* loaders.
  typedef Kokkos::View<double**, Kokkos::LayoutRight, DeviceType>   t_uq_2d;
  typedef Kokkos::View<double***, Kokkos::LayoutRight, DeviceType>  t_uq_3d;
  typedef Kokkos::View<double****, Kokkos::LayoutRight, DeviceType> t_uq_4d;
  t_uq_3d d_uq_centroids;          // [E, Kmax, D]
  t_uq_4d d_uq_inv_cov;            // [E, Kmax, D, D]
  t_int_1d d_uq_n_clusters;        // [E]
  t_uq_2d d_uq_interp_thresholds;  // [E, Kmax]
  t_uq_2d d_uq_rp_matrix;          // [D_basis, rp_dim] projection R
  t_uq_2d d_uq_z;                  // [nlocal, rp_dim] per-atom RAW proj accumulated across rho1 (P1) + rho2 (P2), finished in P3
  t_uq_1d d_uq_n2_rho1;            // [nlocal] ||B^(rho1)||^2 from Phase 1 (block density carry)
  t_uq_1d d_uq_n2_rho2;            // [nlocal] ||B^(rho2)||^2 from Phase 2 (block density carry)
  t_uq_1d d_gamma, d_sigma, d_gmm_cluster;

  bool has_uq = false;
  int uq_Kmax = 0, uq_D = 0;          // uq_D = full GMM feature dim (rp_dim + n_density)
  int uq_rp_dim = 0;                  // projection output width (R cols; e.g. 128)
  int uq_n_density = 0;               // density channels appended after proj (uq_D - rp_dim)
  int uq_normalize = 0;               // 1 -> L2-normalize basis before projection (v6)
  double uq_density_scale = 1.0;      // density_scale
  int uq_d_basis = 0;                 // total basis width (rows of R)
  int uq_d_basis_rho1 = 0;            // rho1 block width (R rows [0, rho1))
  int uq_d_basis_rho2 = 0;            // rho2 block width (R rows [rho1, rho1+rho2))
  int uq_d_basis_rho3 = 0;            // rho3 block width (R rows [rho1+rho2, d_basis))
};

}    // namespace kk
}    // namespace grace

// The GRACE-3L engine a consumer (LAMMPS) instantiates explicitly, naming only the device and the traits: this
// declaration in its headers, GRACE_KK_INSTANTIATE_3L (grace/kk/grace_3l_kk_impl.h) in one translation unit. Only the
// base: the members of Grace3LFusedKK are inline (a member defined out of line must be added here). At global scope,
// as a statement: GRACE_KK_EXTERN_3L(Device, Traits);
#define GRACE_KK_EXTERN_3L(Device, Traits) extern template class grace::kk::Grace3LKK<Device, float, float, Traits>

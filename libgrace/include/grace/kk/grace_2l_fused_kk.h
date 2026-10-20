// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// GRACE-2L fused Kokkos engine (from the LAMMPS style grace/2l/fused/kk, lammps-ace feature/grace-runtime a6ba5104dd;
// contributing author Yury Lysogorskiy, ICAMS): the base engine with fused bond, dense and reduction stages; bond
// geometry in fp64, the network and the radial tables in fp32, energies / forces / virials accumulated in fp64 (KKA).
#pragma once

#include "grace/kk/grace_2l_kk.h"
#include "grace/kk/fused_dense.h"

#include <array>
#include <vector>

namespace grace {
namespace kk {

template<class DeviceType, class Traits>
class Grace2LFusedKK : public Grace2LKK<DeviceType, float, float, Traits> {
 public:
  using Base = Grace2LKK<DeviceType, float, float, Traits>;
  using F = float;
  template<class T> using V1 = Kokkos::View<T*, DeviceType>;
  template<class T> using V2 = Kokkos::View<T**, Kokkos::LayoutRight, DeviceType>;
  template<class T> using V3 = Kokkos::View<T***, Kokkos::LayoutRight, DeviceType>;
  template<class T> using V4 = Kokkos::View<T****, Kokkos::LayoutRight, DeviceType>;

  // checks the model against the compiled kernels, builds the dense metadata and the radial tables
  void setup() override;
  typename Base::RunOutput run(const typename Base::RunInput &in) override;
  int store_l1 = 1;                    // keep the L1 forward state per chunk (no recompute in the L1 backward)
  bool l1_kept = true;                 // chunksize auto: store_l1 fits the GPU memory budget next to the chunk
  int no_shape = 0;                    // 'noshape': force the generic CG-product kernels
  int spline_n = 1024;                 // cubic-Hermite intervals on x in [0, 1] ('spline' keyword)

 protected:
  std::vector<typename Base::t_nn_3d> st_A, st_A1, st_AA1, st_AA2;
  std::vector<typename Base::t_nn_2d> st_rho;
  int shape_id = 0;                    // compiled CG shape matching the model: 0 generic, 1 medium, 2 large
  int ng_AA = 0, ng_AAA = 0, ng_AAAA = 0, ng_BB = 0, ng_BBB = 0, ng_BBBB = 0;

  // Radial tables: per interval k the cubic coefficients in t = N x - k, outputs o = n * (lmax+1) + l.
  V3<F> d_splR;                        // [N, L1_nradmax*(lmax+1), 4]
  V3<F> d_splR1;                       // [N, L2_nradmax*(lmax+1), 4]


 public:
  using FCMeta = GraceFusedKK::FCMetaT<DeviceType>;
  FCMeta fcm_A1, fcm_AA1, fcm_AA2, fcm_B1, fcm_BB1, fcm_BB2;
  using ProdMeta = GraceFusedKK::ProdMetaT<DeviceType>;
  // ReduceN metadata: up to 8 weight sets (reducers) over up to 4 input tensors; W[type][tile][n_in][n_out pad 4]
  struct RedSpec {
    const GRACE2LModel::ReduceWeights *rw;
    int x;          // input tensor index
    int fo;         // output function for every connection (-1: total_sum_ind)
    int n_out;
  };
  struct RedMeta {
    int nws = 0, nx = 0, nop = 0, nfo = 0, nfo1 = 0, gtot = 0;
    bool elem = false;
    int ws_x[8] = {0}, ws_nin[8] = {0}, nfi[4] = {0}, gbase[4] = {0};
    V4<F> W[8];
    V1<int> toff, tws, tfi, ttile, roff, rfo, rtile, rws;
  };
  RedMeta rm_I1, rm_B, rm_I2;
  ProdMeta pm_AA, pm_AAA, pm_AAAA, pm_BB, pm_BBB, pm_BBBB;
  using t_nn3 = typename Base::t_nn_3d;

 protected:

  V1<int> d_yi_gperm, d_yi_goff;       // YI pairs sorted by (I, l); group offsets
  grace::kk::View64<V2<F>> d_IT, d_gIT;    // message I and dE/dI, [nall][I * NR2 + n] (neighbour-contiguous)
  V2<double> d_vir;                    // [inum, 6] per-atom global-virial partials (bond kernels)

 public:    // methods holding extended device lambdas must be public (nvcc)
  void build_spline_tables();
  void build_type_metas();    // per-type packed reducer weights (I1 / rho, I2)
  void retype() override { build_type_metas(); }
  std::shared_ptr<void> save_derived_types() const override
  {
    return std::make_shared<std::array<RedMeta, 2>>(std::array<RedMeta, 2>{rm_I1, rm_I2});
  }
  void use_derived_types(const std::shared_ptr<void> &d) override
  {
    const auto &t = *std::static_pointer_cast<std::array<RedMeta, 2>>(d);
    rm_I1 = t[0];
    rm_I2 = t[1];
  }
  void fused_layer1_A();
  template<bool CPL> void fused_layer1_A_c();     // CPL: two-group coupling
  int l2_channel_block(bool bwd) const;
  void fused_layer2_fwd();
  template<int NIM, bool CPL> void fused_layer2_fwd_t();
  void fused_layer2_bwd();
  template<int NIM, bool CPL> void fused_layer2_bwd_t();
  void fused_layer1_bwd();
  template<bool CPL> void fused_layer1_bwd_c();
  void stage_L1_bond() override;
  void stage_L1_bond_recompute() override;
  void stage_L2_bond_fwd() override;
  void stage_L2_bond_bwd() override;
  void stage_L1_bond_bwd() override;
  void begin_L1_chunk(bool backward) override;
  double kept_bytes_per_atom() const override    // store_l1: A, A1, AA1, AA2, rho of every chunk
  {
    if (!store_l1 || this->d_A.extent(0) == 0) return 0.0;
    using M = grace::kk::MemKK;
    return (M::memory_usage(this->d_A) + M::memory_usage(this->d_A1) + M::memory_usage(this->d_AA1) +
            M::memory_usage(this->d_AA2) + M::memory_usage(this->d_rho)) / (double) this->d_A.extent(0);
  }
  bool need_L1_recompute() override;
  bool l1_store_on() const { return store_l1 && (this->chunksize >= 0 || l1_kept); }
  void use_kept(bool on) override
  {
    l1_kept = on;
    if (!on) { st_A.clear(); st_A1.clear(); st_AA1.clear(); st_AA2.clear(); st_rho.clear(); }
  }
  void build_fc_meta(FCMeta &, const GRACE2LModel::FCWeights &, const std::vector<int> &used);
  template<int NO> void fc_fwd(const FCMeta &, const t_nn3 &, const t_nn3 &, const t_nn3 &);
  template<int NO> void fc_bwd(const FCMeta &, const t_nn3 &, const t_nn3 &, const t_nn3 &);
  template<int NO> void fc_fwd4(const FCMeta &, const t_nn3 &, const t_nn3 &, const t_nn3 &);
#ifdef KOKKOS_ENABLE_CUDA
  void fc_forward_cublas(typename Base::FCCublasOp &, const t_nn3 &, const t_nn3 &, const t_nn3 &) override;
#endif
  void stage_rev_fc_AA2() override;
  void stage_rev_fc_AA1() override;
  void stage_rev_fc_A1() override;
  void stage_rev_fc_BB2() override;
  void stage_rev_fc_BB1() override;
  void stage_rev_fc_B1() override;
  void build_prod_meta(ProdMeta &, const GRACE2LModel::CGProduct &, bool self);
  void prod_fwd(const ProdMeta &, const t_nn3 &, const t_nn3 &, const t_nn3 &);
  void prod_rev(const ProdMeta &, const t_nn3 &, const t_nn3 &, const t_nn3 &, const t_nn3 &, const t_nn3 &);
  void stage_prod_AA() override;
  void stage_prod_AAA() override;
  void stage_prod_AAAA() override;
  void stage_prod_BB() override;
  void stage_prod_BBB() override;
  void stage_prod_BBBB() override;
  void stage_rev_prod_AA() override;
  void stage_rev_prod_AAA() override;
  void stage_rev_prod_AAAA() override;
  void stage_rev_prod_BB() override;
  void stage_rev_prod_BBB() override;
  void stage_rev_prod_BBBB() override;
  void build_red_meta(RedMeta &, const std::vector<RedSpec> &, int nfo, int nfo1, bool elem, const std::vector<int> &nfi_x);
  template<int NQ, int MODE, int NO1, int NO2>
  void red_fwd(const RedMeta &, const t_nn3 &, const t_nn3 &, const t_nn3 &, const t_nn3 &, const t_nn3 &,
               const typename Base::t_nn_2d &);
  template<int NQ, int NIN, int MODE, int NO1, int NO2, bool ASSIGN>
  void red_bwd(const RedMeta &, const t_nn3 &, const typename Base::t_nn_2d &, const t_nn3 &, const t_nn3 &,
               const t_nn3 &, const t_nn3 &);
  void stage_reduce_rho() override;
  void stage_rev_reduce_rho() override;
  void stage_zero_L1_adj() override;
  void stage_zero_L2_adj() override;
  bool need_bond_buffers() const override { return false; }    // every bond stage is fused
  void stage_reduce_I2() override;
  void stage_rev_reduce_I2() override;
  void stage_reduce_B() override;
  void stage_rev_reduce_B() override;
  void stage_reduce_I1() override;
  void stage_rev_reduce_I1() override;
};

}    // namespace kk
}    // namespace grace

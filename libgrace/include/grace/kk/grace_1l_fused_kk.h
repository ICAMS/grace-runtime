// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// GRACE-1L fused Kokkos engine (from the LAMMPS style grace/1l/fused/kk, lammps-ace feature/grace-runtime 07b8fb50d4;
// contributing author Yury Lysogorskiy, ICAMS): the base engine with the fused bond and dense stages; bond geometry
// in fp64, the network and the radial tables in fp32, energies / forces / virials accumulated in fp64 (KKA).
#pragma once

#include "grace/kk/grace_1l_kk.h"
#include "grace/kk/fused_dense.h"

#include <array>
#include <vector>

namespace grace {
namespace kk {

template<class DeviceType, class Traits>
class Grace1LFusedKK : public Grace1LKK<DeviceType, float, float, Traits> {
 public:
  using Base = Grace1LKK<DeviceType, float, float, Traits>;
  using F = float;
  template<class T> using V1 = Kokkos::View<T *, DeviceType>;
  template<class T> using V2 = Kokkos::View<T **, Kokkos::LayoutRight, DeviceType>;
  template<class T> using V3 = Kokkos::View<T ***, Kokkos::LayoutRight, DeviceType>;   // float4 rows along the last index
  template<class T> using V4 = Kokkos::View<T ****, Kokkos::LayoutRight, DeviceType>;
  using t_nn3 = typename Base::t_nn_3d;
  using t_nn2 = typename Base::t_nn_2d;

  // checks the model against the compiled kernels, builds the dense metadata and the radial tables
  void setup() override;
  typename Base::RunOutput run(const typename Base::RunInput &in) override;
  int spline_n = 1024;                  // cubic-Hermite intervals of the radial tables ('spline' keyword)
  int no_shape = 0;                     // 'noshape': generic CG products

  // stage overrides (public: nvcc extended lambdas)
  void stage_bond_fwd(int, int) override;
  void stage_dense_fwd() override;
  void stage_reduce() override;
  void stage_zero_adj() override;
  void stage_dense_bwd() override;
  void stage_bond_bwd(int, int) override;
  bool need_bond_buffers() const override { return false; }

  void fused_A();
  void fused_bond_bwd();
  template<bool CPL> void fused_A_c();          // CPL: two-group coupling
  template<bool CPL> void fused_bond_bwd_c();

  using FCMeta = GraceFusedKK::FCMetaT<DeviceType>;
  using ProdMeta = GraceFusedKK::ProdMetaT<DeviceType>;
  struct RedMeta {                      // rho: four sources (A, AA, AAA, AAAA), element-dependent
    int nx = 0, gtot = 0;
    int nin[4] = {0}, nfi[4] = {0}, gbase[4] = {0};
    V4<F> W[4];                         // [type][tile][n_in][n_out pad 4]
    V1<int> toff, tx, tfi, ttile, roff, rx, rtile;
  };

  void build_fc_meta(FCMeta &, const typename GRACE1LModel::FCWeights &, const std::vector<int> &used);
  void build_prod_meta(ProdMeta &, const typename GRACE1LModel::CGProduct &, bool self);
  void build_red_meta();
  void build_spline_tables();
  void retype() override { build_red_meta(); }    // per-type packed rho weights
  std::shared_ptr<void> save_derived_types() const override { return std::make_shared<RedMeta>(rm); }
  void use_derived_types(const std::shared_ptr<void> &d) override { rm = *std::static_pointer_cast<RedMeta>(d); }
  template<int NO> void fc_fwd(const FCMeta &, const t_nn3 &, const t_nn3 &, const t_nn3 &);
  template<int NO, int NIR> void fc_bwd(const FCMeta &, const t_nn3 &, const t_nn3 &, const t_nn3 &);
  template<int NO> void fc_fwd4(const FCMeta &, const t_nn3 &, const t_nn3 &, const t_nn3 &);
  void prod_fwd(const ProdMeta &, const t_nn3 &, const t_nn3 &, const t_nn3 &);
  void prod_rev(const ProdMeta &, const t_nn3 &, const t_nn3 &, const t_nn3 &, const t_nn3 &, const t_nn3 &,
                bool acc_left);
  template<int NQ> void red_fwd();
  template<int NQ, int NIN> void red_bwd();

 protected:
  int shape_id = 0;                     // 1: generated GRACE-1L large shape
  int ng_AAA = 0;
  V3<F> d_splR;                         // [N][nradmax * (lmax+1)][4] cubic coefficients in t
  V1<int> d_lfrom;                      // l of lm
  V2<double> d_vir;                     // [inum][6]
  FCMeta fcm_A1, fcm_AA1, fcm_AA2;
  ProdMeta pm_AA, pm_AAA, pm_AAAA;
  RedMeta rm;
};

}    // namespace kk
}    // namespace grace

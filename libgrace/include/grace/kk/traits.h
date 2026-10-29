// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// View types of the Kokkos engines for consumers without LAMMPS (the Python module). The LAMMPS glue defines its own
// traits from ArrayTypes (GraceLammpsTraits), so the engines there see exactly the LAMMPS types.
#pragma once

#include "grace/kk/support.h"    // View64

#include <Kokkos_Core.hpp>

namespace grace {
namespace kk {

// energy / virial reduction value, as LAMMPS EV_FLOAT (same members and join)
template<class A> struct EVT {
  A evdwl, ecoul, v[6];
  KOKKOS_INLINE_FUNCTION EVT()
  {
    evdwl = 0;
    ecoul = 0;
    for (int i = 0; i < 6; ++i) v[i] = 0;
  }
  KOKKOS_INLINE_FUNCTION void operator+=(const EVT &rhs)
  {
    evdwl += rhs.evdwl;
    ecoul += rhs.ecoul;
    for (int i = 0; i < 6; ++i) v[i] += rhs.v[i];
  }
};

// KKF: position precision, KKA: force / energy accumulation precision (LAMMPS KK_FLOAT / KK_ACC_FLOAT)
template<class DeviceType, class KKFloat = double, class AccFloat = double> struct DefaultTraits {
  using kk_float = KKFloat;
  using kk_acc_float = AccFloat;
  using ev_type = EVT<AccFloat>;
  using kk_device = Kokkos::Device<typename DeviceType::execution_space, typename DeviceType::memory_space>;
  using layout = typename DeviceType::array_layout;
  using layout_f = layout;
  using layout_vatom = layout;
  using layout_cvatom = layout;
  using t_kkfloat_1d_3_lr_randomread =
      Kokkos::View<const KKFloat *[3], Kokkos::LayoutRight, DeviceType, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
  using t_kkacc_1d_3 = Kokkos::View<AccFloat *[3], layout, DeviceType>;
  using t_kkacc_1d = Kokkos::View<AccFloat *, layout, DeviceType>;
  using t_kkacc_1d_6 = Kokkos::View<AccFloat *[6], layout, DeviceType>;
  using t_kkacc_1d_9 = Kokkos::View<AccFloat *[9], layout, DeviceType>;
  using t_int_1d = Kokkos::View<int *, DeviceType>;
  using t_double_1d = Kokkos::View<double *, Kokkos::LayoutRight, DeviceType>;    // exchange buffers
  using t_int_1d_randomread = Kokkos::View<const int *, DeviceType, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
  // [nlocal][maxneigh]: more than 2^31 entries from 22M atoms x 100 neighbours, so 64-bit offsets (View64)
  using t_neighbors_2d = View64<Kokkos::View<int **, layout, DeviceType>>;
  static constexpr int neighmask = 0x1FFFFFFF;    // LAMMPS NEIGHMASK (special-bond bits above it)
};

}    // namespace kk
}    // namespace grace

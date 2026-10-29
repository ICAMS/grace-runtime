// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// GRACE-3L fused Kokkos engine: the fp32 engine Grace3LKK<float, float> (fp64 only in r_ij, |r|, rhat, the cutoff
// envelope and the Chebyshev basis) with its fused stages switched on. The stages live in the base engine behind flags,
// so this type adds no kernel instantiation of its own:
//   fc_fold      the FC A*_2a / A*_2b folded into the cp_l projections they feed;
//   fuse_radial  the radial evaluated from the cubic tables inside the kernels that use it;
//   spline_mb    auto (< 0): tables in up to a quarter of the device memory, so many species get them too.
#pragma once

#include "grace/kk/grace_3l_kk.h"

namespace grace {
namespace kk {

template<class DeviceType, class Traits>
class Grace3LFusedKK : public Grace3LKK<DeviceType, float, float, Traits> {
 public:
  static constexpr bool fused = true;
  Grace3LFusedKK()
  {
    this->fc_fold = 1;
    this->fuse_radial = 1;
    this->spline_mb = -1.0;    // auto: the radial tables in up to a quarter of the device memory
  }
};

}    // namespace kk
}    // namespace grace

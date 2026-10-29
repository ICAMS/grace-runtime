// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The GRACE-3L Kokkos engine of the GPU module (kk_core.cpp declares it extern; Grace3LFusedKK derives from it with
// the fused stages on): its own translation unit, so the launch compiler builds it next to the 1L / 2L engines.
#include "grace/kk/csr_images.h"
#include "grace/kk/grace_3l_kk_impl.h"
#include "grace/kk/traits.h"

GRACE_KK_INSTANTIATE_3L(Kokkos::DefaultExecutionSpace, grace::kk::DefaultTraits<Kokkos::DefaultExecutionSpace>);

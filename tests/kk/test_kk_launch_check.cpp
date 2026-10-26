// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// grace::kk::check_kernel_launches(): on a GPU backend it throws grace::Error after a kernel launch that failed (here
// an invalid launch configuration, as nothing else checks it in a release build of Kokkos) or after another failed
// runtime call, and the error is consumed (the next check passes); on host backends it never throws.
#include "grace/kk/support.h"
#include "../test_util.h"

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <string>

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
__global__ void grace_test_noop() {}
#endif

namespace {
using Device = Kokkos::DefaultExecutionSpace::device_type;
bool throws()
{
  try {
    grace::kk::check_kernel_launches<Device>("test");
  } catch (const grace::Error &e) {
    std::printf("  %s\n", e.what());
    return true;
  }
  return false;
}
}    // namespace

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  CHECK(!throws());
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  grace_test_noop<<<1, 4096>>>();    // more threads per block than any GPU allows: the launch fails
  CHECK(throws());
  CHECK(!throws());
  void *p = nullptr;
#if defined(KOKKOS_ENABLE_CUDA)
  CHECK(cudaMalloc(&p, (size_t) 1 << 52) != cudaSuccess);    // 4 PB
#else
  CHECK(hipMalloc(&p, (size_t) 1 << 52) != hipSuccess);
#endif
  CHECK(throws());
  CHECK(!throws());
  grace_test_noop<<<1, 32>>>();    // a valid launch leaves nothing behind
  Kokkos::fence();
  CHECK(!throws());
#endif
  std::printf("check_kernel_launches: ok\n");
  return 0;
}

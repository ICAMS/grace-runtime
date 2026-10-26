// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// grace::kk::device_key(): on a GPU backend a non-empty "uuid-<32 hex>" or "pci-<bus id>" that is the same on every
// call, and the call leaves no error behind; on host backends empty. Prints the key, so runs on two GPUs of a node
// (--kokkos-device-id=0 / 1) can be compared by hand.
#include "grace/kk/support.h"
#include "../test_util.h"

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <string>

int main(int argc, char **argv)
{
  Kokkos::ScopeGuard guard(argc, argv);
  const std::string key = grace::kk::device_key();
  std::printf("device_key: \"%s\"\n", key.c_str());
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  CHECK(key.rfind("uuid-", 0) == 0 || key.rfind("pci-", 0) == 0);
  CHECK(key.rfind("uuid-", 0) != 0 || key.size() == 5 + 32);
  CHECK(grace::kk::device_key() == key);
#if defined(KOKKOS_ENABLE_CUDA)
  CHECK(cudaGetLastError() == cudaSuccess);
#else
  CHECK(hipGetLastError() == hipSuccess);
#endif
#else
  CHECK(key.empty());
#endif
  std::printf("device_key: ok\n");
  return 0;
}

// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// GRACE_HD: a function compiled for the host and, in CUDA / HIP translation units, for the device.
#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GRACE_HD __host__ __device__ inline
#else
#define GRACE_HD inline
#endif

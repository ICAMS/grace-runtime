// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Cell-list neighbour search for grace_runtime (the "native" NeighborList backend).
//
// Bins in fractional coordinates; atoms are put in bin order (the spatial order the 2L driver wants) and every
// neighbour within the cutoff, including periodic images and self-images, is listed per atom. Bond vectors in the
// caller's (unwrapped) frame are x[jo] - x[io] + shift, shift = S cell with integer S.
#pragma once

#include <cstdint>
#include <vector>

namespace grace_runtime {

struct NeighborResult {
  std::vector<int64_t> perm;      // new index -> original index
  std::vector<int64_t> offset;    // CSR rows in the new order, n + 1
  std::vector<int32_t> j;         // neighbour, new index
  std::vector<int64_t> io;        // central atom, original index (one per bond)
  std::vector<int32_t> jo;        // neighbour, original index
  std::vector<double> shift;      // S cell, 3 per bond
};

// pos [n,3], cell [3,3] (rows are the cell vectors, full rank), pbc[3]; throws std::invalid_argument on bad input
NeighborResult neighbor_search(const double *pos, int64_t n, const double *cell, const bool *pbc, double cutoff,
                               int num_threads);

}    // namespace grace_runtime

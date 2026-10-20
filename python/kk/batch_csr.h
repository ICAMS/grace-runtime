// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Host side of the GPU batch (kk_core.cpp Batch): the CPU cell list (python/src/neighbors.cpp) per structure, OpenMP
// over structures, laid out as grace/kk/csr_images.h CsrImagesKK holds it: local atoms of all structures (structure by
// structure, each in its list's spatial order), then one ghost image per (owner, lattice translation != 0) of every
// structure; neighbour rows point at the local atom or its image. Host compiler only (no Kokkos).
#pragma once

#include "neighbors.h"

#include <cstdint>
#include <vector>

namespace grace_runtime {

struct BatchCsr {
  int n = 0, nall = 0, maxneigh = 0;
  std::vector<int32_t> perm;         // [n] pass order -> input atom index
  std::vector<int32_t> sys;          // [n] structure of each atom, counted from the pass's first structure
  std::vector<int32_t> type;         // [nall]
  std::vector<double> x;             // [nall][3]
  std::vector<int32_t> numneigh;     // [n]
  std::vector<int32_t> neighbors;    // [n][maxneigh]
  std::vector<int32_t> owner;        // [nall - n] local atom of each image
  std::vector<double> tcart;         // [nall - n][3] translation of each image
};

// the neighbour lists and images of every structure, before the layout of an engine pass
struct BatchSearch {
  struct Part {
    NeighborResult nl;
    std::vector<int32_t> slot;     // [bonds] -1: the local atom j, else the image index
    std::vector<int32_t> img_j;    // [images] local owner
    std::vector<double> img_t;     // [images][3] translation
    int maxneigh = 0;
  };
  std::vector<int64_t> start;      // [nsys + 1] first input atom of each structure
  std::vector<Part> parts;         // [nsys]
  int64_t images(int s) const { return (int64_t) parts[s].img_j.size(); }
};

// nsys structures concatenated in the input: natoms [nsys], pos [N][3], cell [nsys][3][3] (rows, full rank: zero
// vectors padded by the caller), pbc [nsys][3]; bonds shorter than cutoff. Throws std::invalid_argument (bad input, a
// lattice translation beyond +-511 cells as the device list).
BatchSearch batch_search(int nsys, const int32_t *natoms, const double *pos, const double *cell, const bool *pbc,
                         double cutoff, int num_threads);

// structures [s0, s1) of a search laid out for one engine pass; types [N], pos [N][3] as given to batch_search
BatchCsr batch_layout(const BatchSearch &bs, const int32_t *types, const double *pos, int s0, int s1, int num_threads);

}    // namespace grace_runtime

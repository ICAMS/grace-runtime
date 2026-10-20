// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// UQ (uqv6 gamma_only) of the CPU fused drivers: layout of the basis B that feeds the random projection.
//
// B of a central atom = concat over the energy-path invariant reduces, sorted by reduce name (tensorpotential
// uq/factories.py trace_energy_path_invariant_reduces: 1L [rho], 2L [I2, rho]), of concat over the reduce's sources
// (A, AA, AAA, AAAA / B, BB, BBB, BBBB) of X_s[n][ci_s[c]] flattened n-outer, c-inner (tpmodel.py
// extract_basis_functions): the raw l=0 inputs of the reduce, not its output. Row b of uq_rp_matrix multiplies B[b].
#ifndef GRACE_UQ_H
#define GRACE_UQ_H

#include <cstdint>
#include <string>
#include <vector>

namespace grace {

struct UqSource {
  int buf = 0;                // 0..3: A, AA, AAA, AAAA (B, BB, BBB, BBBB) of the block
  int nin = 0;                // channels n
  std::vector<int> ci;        // collected l=0 functions, in the reduce's collect_ind order (c = position)
  int64_t off = 0;            // first index in B
};

struct UqBlock {
  std::string name;           // reduce name
  std::vector<UqSource> src;
  int64_t off = 0, width = 0;
};

// a block from its four source reduces (A..AAAA or B..BBBB); off = first index of the block in B
template<class RW> UqBlock make_uq_block(const std::string &name, const RW *const (&rs)[4], int64_t off)
{
  UqBlock b;
  b.name = name;
  b.off = off;
  for (int s = 0; s < 4; s++) {
    UqSource u;
    u.buf = s;
    u.nin = rs[s]->n_in;
    u.ci = rs[s]->collect_ind;
    u.off = off;
    off += (int64_t) u.nin * (int64_t) u.ci.size();
    b.src.push_back(std::move(u));
  }
  b.width = off - b.off;
  return b;
}

// the checks of validate_uq() shared by 1L and 2L (beyond the loader's): basis width vs R, density channels
// = 1 + blocks, elements, and the source order of each block against its npz instruction names
void check_uq_layout(const char *who, const std::vector<UqBlock> &L, int d_basis, int n_density, int uq_n_elements,
                     int n_elements, const std::vector<std::vector<std::string>> &instruction_names);

}    // namespace grace

#endif

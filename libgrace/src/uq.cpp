// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
#include "grace/uq.h"

#include "grace/error.h"

namespace grace {

void check_uq_layout(const char *who, const std::vector<UqBlock> &L, int d_basis, int n_density, int uq_n_elements,
                     int n_elements, const std::vector<std::vector<std::string>> &instruction_names)
{
  const std::string w(who);
  int64_t total = 0;
  for (auto &b : L) total += b.width;
  if (total != d_basis)
    throw Error(w + " UQ: uq_rp_matrix has " + std::to_string(d_basis) + " rows, the model's invariant basis has " +
                std::to_string(total) + "; re-export the UQ artifacts for this model");
  if (n_density != 1 + (int) L.size())
    throw Error(w + " UQ: " + std::to_string(n_density) + " density channels, expected " +
                std::to_string(1 + L.size()) + " (1 + basis blocks)");
  if (uq_n_elements != n_elements)
    throw Error(w + " UQ: uq_n_elements " + std::to_string(uq_n_elements) + " != model elements " +
                std::to_string(n_elements));
  static const std::vector<std::string> l1{"A", "AA", "AAA", "AAAA"}, l2{"B", "BB", "BBB", "BBBB"};
  for (size_t k = 0; k < L.size() && k < instruction_names.size(); k++) {
    const auto &want = L[k].name == "I2" ? l2 : l1;
    if (!instruction_names[k].empty() && instruction_names[k] != want)
      throw Error(w + " UQ: reduce " + L[k].name + " sources are not in the order the UQ basis assumes");
  }
}

}    // namespace grace

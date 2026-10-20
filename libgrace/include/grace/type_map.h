// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Atom type -> model element maps (type2mu) shared by the CPU drivers and the Kokkos engines.
// type2mu[t], t = 1..ntypes: model element of atom type t, -1 unmapped (pair_style hybrid NULL: no bonds to or from
// that type).
#pragma once
#include "grace/error.h"

#include <string>
#include <vector>

namespace grace {

// cutoff of a bond between model elements mu_i and mu_j (GRACE1LModel / GRACE2LModel / GRACE3LModel); 0 when either is
// unmapped (-1), so the bond filters (r^2 < cutsq) drop it
template<class Model> double pair_cutoff(const Model &m, int mu_i, int mu_j)
{
  if (mu_i < 0 || mu_j < 0) return 0.0;
  return m.has_bond_specific_cutoff ? m.bond_cutoff_map[mu_i * m.n_elements + mu_j] : m.rcut;
}

// throws Error unless type2mu has ntypes + 1 entries, each -1 or a model element
inline void check_type2mu(const std::vector<int> &type2mu, int ntypes, int n_elements, const char *who)
{
  if ((int) type2mu.size() != ntypes + 1) throw Error(std::string(who) + ": type2mu must have ntypes + 1 entries");
  for (int t = 1; t <= ntypes; t++)
    if (type2mu[t] < -1 || type2mu[t] >= n_elements)
      throw Error(std::string(who) + ": atom type " + std::to_string(t) + " maps to model element " +
                  std::to_string(type2mu[t]) + " (the model has " + std::to_string(n_elements) + ")");
}

}    // namespace grace

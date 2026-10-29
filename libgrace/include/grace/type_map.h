// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Atom type -> model element maps (type2mu) shared by the CPU drivers and the Kokkos engines.
// type2mu[t], t = 1..ntypes: model element of atom type t, -1 unmapped (pair_style hybrid NULL: no bonds to or from
// that type).
// The central atoms (ilist) must all be mapped: the Kokkos engines read the tables of a central atom's element (energy
// shift, weights, radial tables, UQ head) without a check, and the CPU drivers throw at compute(). The caller ensures
// it: LAMMPS lists only mapped types (pair_style hybrid gives a sub-style a skip list without its NULL types, and a
// plain pair_style with NULL stops at init), the Python modules refuse species the model lacks and types outside
// 1 .. len(species). Unmapped neighbours are fine: their pair cutoff is 0, so the bond filters drop them.
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

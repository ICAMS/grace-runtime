// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Dispersion correction added to a model's energy, forces, virial and per-atom energies. Methods: DFT-D3 with
// Becke-Johnson (rational) damping, two-body terms C6 and C8 and optionally the Axilrod-Teller-Muto three-body term
// (grace/d3.h). Later methods (D4) add a Kind and their fields here.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace grace {

// CODATA 2018
inline constexpr double kBohr = 0.529177210903;       // Angstrom
inline constexpr double kHartree = 27.211386245988;    // eV

struct DispersionParams {
  enum Kind : int { D3BJ = 1 };
  int kind = D3BJ;
  double s6 = 1.0, s8 = 0.0, a1 = 0.0, a2 = 0.0;    // D3(BJ) damping, a2 in Angstrom
  // pair terms: full weight below cutoff_on, switched to zero at cutoff (cutoff_on == cutoff: hard cutoff);
  // coordination numbers: hard cutoff cn_cutoff. Angstrom.
  double cutoff = 15.0, cutoff_on = 12.0, cn_cutoff = 10.0;
  // three-body (ATM) term, s9 = 0: off; each of a triangle's three distances switched as the pair terms between
  // atm_cutoff_on and atm_cutoff (Angstrom)
  double s9 = 0.0, atm_cutoff = 10.0, atm_cutoff_on = 9.0;
};

inline void validate(const DispersionParams &p)
{
  if (p.kind != DispersionParams::D3BJ)
    throw std::invalid_argument("dispersion: unknown method kind " + std::to_string(p.kind));
  for (double v : {p.s6, p.s8, p.a1, p.a2, p.cutoff, p.cutoff_on, p.cn_cutoff, p.s9, p.atm_cutoff, p.atm_cutoff_on})
    if (!std::isfinite(v)) throw std::invalid_argument("dispersion: parameters must be finite");
  if (p.s9 != 0.0 && !(p.atm_cutoff_on > 0.0 && p.atm_cutoff_on <= p.atm_cutoff))
    throw std::invalid_argument("dispersion: need 0 < atm_cutoff_on <= atm_cutoff");
  if (!(p.cutoff > 0.0) || !(p.cn_cutoff > 0.0)) throw std::invalid_argument("dispersion: cutoffs must be positive");
  if (!(p.cutoff_on > 0.0 && p.cutoff_on <= p.cutoff))
    throw std::invalid_argument("dispersion: need 0 < cutoff_on <= cutoff");
  if (p.a1 < 0.0 || p.a2 < 0.0 || (p.a1 == 0.0 && p.a2 == 0.0))
    throw std::invalid_argument("dispersion: D3(BJ) needs a1, a2 >= 0, not both zero (the damping radius)");
}

// published D3(BJ) parameters (a2 in Bohr, as published)
struct D3BJPreset {
  const char *name;
  double s6, a1, s8, a2_bohr;
  const char *doi;
};
inline constexpr D3BJPreset kD3BJPresets[] = {
    {"pbe", 1.0, 0.4289, 0.7875, 4.4407, "10.1002/jcc.21759"},
    {"r2scan", 1.0, 0.49484001, 0.78981345, 5.73083694, "10.1063/5.0041008"},
};

// the published D3(BJ) parameters of a functional (lower-case name); throws std::invalid_argument for an unknown name
inline const D3BJPreset &d3bj_preset(const std::string &name)
{
  std::string names;
  for (const auto &q : kD3BJPresets) {
    if (name == q.name) return q;
    names += (names.empty() ? "" : ", ") + std::string(q.name);
  }
  throw std::invalid_argument("unknown D3(BJ) functional '" + name + "' (presets: " + names + ")");
}

}    // namespace grace

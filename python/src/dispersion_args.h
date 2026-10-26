// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Dispersion parameters of the Python modules (grace_runtime._core and _kk_<arch>) from the dict of
// grace_runtime.D3.params(): kind, s6, s8, a1, a2 [A], cutoff, cutoff_on, cn_cutoff [A]; optional (default: the
// three-body term off) s9, atm_cutoff, atm_cutoff_on [A].
#pragma once

#include "grace/dispersion.h"

#include <nanobind/nanobind.h>

#include <string>

inline grace::DispersionParams dispersion_params(const nanobind::dict &d)
{
  grace::DispersionParams p;
  auto get = [&](const char *k, double &v) {
    if (!d.contains(k)) throw nanobind::value_error((std::string("dispersion parameters: missing ") + k).c_str());
    v = nanobind::cast<double>(d[k]);
  };
  p.kind = d.contains("kind") ? nanobind::cast<int>(d["kind"]) : (int) grace::DispersionParams::D3BJ;
  get("s6", p.s6);
  get("s8", p.s8);
  get("a1", p.a1);
  get("a2", p.a2);
  get("cutoff", p.cutoff);
  get("cutoff_on", p.cutoff_on);
  get("cn_cutoff", p.cn_cutoff);
  auto opt = [&](const char *k, double &v) {
    if (d.contains(k)) v = nanobind::cast<double>(d[k]);
  };
  opt("s9", p.s9);
  opt("atm_cutoff", p.atm_cutoff);
  opt("atm_cutoff_on", p.atm_cutoff_on);
  return p;
}

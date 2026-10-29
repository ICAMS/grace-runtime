// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Two-group lambda coupling (alchemical decoupling; contributed by Pablo Pena Cano). A bond between an atom of group 1
// and an atom of group 2 is stretched along its own direction, r -> r'(r, lambda); the model sees r' rhat_ij. The
// forces are the gradient of the stretched energy: the radial part of a bond force is multiplied by dr'/dr, the angular
// part keeps the true 1/r (the r'/r of the Jacobian cancels the model's 1/r'), the virial uses the true r_ij. A bond
// with r' >= rc is dropped (the cutoff envelope and its derivative vanish at rc). dE/dlambda is the sum over the
// stretched bonds of dE/dr' dr'/dlambda. Host and device (CUDA / HIP) code.
#pragma once

#include "grace/hd.h"

#include <cmath>

namespace grace {

// shift equations (rc: the pair cutoff): 0 off, 1 r + l rc, 2 r + l max(0, rc - r), 3 sqrt(r^2 + (l rc)^2),
// 4 sqrt(r^2 + l^2 max(0, rc^2 - r^2))
GRACE_HD bool coupling_cross(int mi, int mj, int bit1, int bit2)
{
  return ((mi & bit1) && (mj & bit2)) || ((mi & bit2) && (mj & bit1));
}

// r' and its derivatives dr'/dr, dr'/dlambda of a cross-group bond (r < rc)
GRACE_HD void coupling_shift(int eq, double r, double rc, double lam, double &rp, double &drp_dr, double &drp_dl)
{
  switch (eq) {
    case 1:
      rp = r + lam * rc; drp_dr = 1.0; drp_dl = rc;
      break;
    case 2:
      rp = r + lam * (rc - r); drp_dr = 1.0 - lam; drp_dl = rc - r;
      break;
    case 3: {
      const double a = lam * rc;
      rp = sqrt(r * r + a * a); drp_dr = r / rp; drp_dl = a * rc / rp;
      break;
    }
    default: {    // 4
      const double d = rc * rc - r * r;
      rp = sqrt(r * r + lam * lam * d); drp_dr = r * (1.0 - lam * lam) / rp; drp_dl = lam * d / rp;
      break;
    }
  }
}

// host side (CPU drivers): per-atom group mask of local and ghost atoms
struct Coupling {
  int eq = 0;                 // 0: off
  double lambda = 0.0;
  const int *mask = nullptr;
  int bit1 = 0, bit2 = 0;
};

}    // namespace grace

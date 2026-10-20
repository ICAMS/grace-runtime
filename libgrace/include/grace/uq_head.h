// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from lammps-ace src/ML-PACE/grace_uq.h (GraceUQHead, feature/grace-runtime a73b8b17ee): the uqv6 UQ head of
// a model (GRACE-3L). Filled by the loader (load_uq_head in src/npz3.h).
#ifndef GRACE_UQ_HEAD_H
#define GRACE_UQ_HEAD_H

#include <string>
#include <vector>

namespace grace {

// Schema v6 (basis-RP, normalize + density): the GMM feature phi is built from the invariant (l=0) energy-path
// basis B as proj = (B / ||B||) R (L2-normalized) with n_density log-norm density channels appended:
// phi = concat(proj, dens). R is stored verbatim; gamma is the only UQ signal (no force-error model).
struct UqHead {
  bool has_uq = false;
  int uq_schema_version = 0;
  int uq_n_elements = 0;
  int uq_max_clusters = 0;              // Kmax
  int uq_feature_dim = 0;               // D = rp_dim + n_density (full GMM feature dim)
  int uq_rp_dim = 0;                    // random-projection output dim (R cols, read from the artifact)
  int uq_n_density = 0;                 // density channels appended after proj (D - rp_dim)
  int uq_normalize = 0;                 // 1 -> L2-normalize basis before projection (v6)
  double uq_density_scale = 1.0;        // density_scale (=1.0 for v6)
  int uq_d_basis = 0;                   // concatenated invariant basis width (rows of R)
  std::vector<double> uq_centroids;            // [E*Kmax*D]
  std::vector<double> uq_inv_cov;              // [E*Kmax*D*D] (pre-inverted precision)
  std::vector<int> uq_n_clusters;              // [E]
  std::vector<double> uq_interp_thresholds;    // [E*Kmax]
  std::vector<double> uq_rp_matrix;            // [D_basis*rp_dim] row-major projection R

};

}    // namespace grace
#endif

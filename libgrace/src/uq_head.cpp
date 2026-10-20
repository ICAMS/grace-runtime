// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// Moved from lammps-ace src/ML-PACE/grace_uq.cpp (GraceUQHead::load_uq).
#include "grace/uq_head.h"
#include "npz3.h"

#include <cmath>
#include <string>

namespace grace {

using namespace npz;

void load_uq_head(UqHead &h, const cnpy::npz_t &npz, int n_elements, int max_density, const std::string &who)
{
  // Exclusive to uqv6 (schema 6: L2-normalize + log-norm density channels); older v3/v4 artifacts are rejected
  // here so there is no silent miscompute.
  h = UqHead();    // a model loaded again into the same object must not keep the previous UQ head
  const std::string pre = who + " UQ: ";
  auto fail = [&](const std::string &msg) { throw Error(pre + msg); };
  h.uq_schema_version = npz_get_int_scalar(npz, "uq_schema_version", 0);
  if (h.uq_schema_version == 0) return;
  const int supported = 6;
  if (h.uq_schema_version != supported)
    fail("schema v" + std::to_string(h.uq_schema_version) + " unsupported by this GRACE pair style build (expects v" +
         std::to_string(supported) + ", uqv6 basis-RP normalize+density). Re-export the UQ artifacts at schema v6 "
         "with a current grace_utils, or update the binary.");
  h.has_uq = true;
  h.uq_n_elements = npz_get_int_scalar(npz, "uq_n_elements", n_elements);
  h.uq_max_clusters = npz_get_int_scalar(npz, "uq_max_clusters", 0);
  h.uq_feature_dim = npz_get_int_scalar(npz, "uq_feature_dim", 0);
  h.uq_rp_dim = npz_get_int_scalar(npz, "uq_rp_dim", 0);
  // v6 feature-build flags (confirmatory: must match the v6 profile below)
  h.uq_normalize = npz_get_int_scalar(npz, "uq_rp_normalize", 0);
  const int add_density = npz_get_int_scalar(npz, "uq_rp_add_density_channel", 0);
  h.uq_density_scale = npz_get_double_scalar(npz, "uq_rp_density_scale", 1.0);
  const int transform = npz_get_int_scalar(npz, "uq_feature_transform", 0);
  auto require_double = [&](const char *key) {
    auto v = npz_get_double(npz, key);
    if (v.empty()) fail(std::string("required array '") + key + "' missing from the .npz file");
    return v;
  };
  h.uq_centroids = require_double("uq_centroids");
  h.uq_inv_cov = require_double("uq_inv_cov");
  h.uq_n_clusters = npz_get_int(npz, "uq_n_clusters");
  if (h.uq_n_clusters.empty()) fail("required array 'uq_n_clusters' missing from the .npz file");
  h.uq_interp_thresholds = require_double("uq_interp_thresholds");
  h.uq_rp_matrix = require_double("uq_rp_matrix");    // [D_basis, rp_dim] row-major

  // the evaluators implement normalize + density with the identity transform only (no asinh)
  if (h.uq_normalize != 1 || add_density != 1 || transform != 0)
    fail("schema v6 expects normalize=1, add_density_channel=1, feature_transform=0 (identity); got normalize=" +
         std::to_string(h.uq_normalize) + ", add_density=" + std::to_string(add_density) +
         ", transform=" + std::to_string(transform) + ". Artifact mislabeled or unsupported variant.");
  if (h.uq_rp_dim <= 0 || h.uq_feature_dim < h.uq_rp_dim)
    fail("uq_feature_dim (" + std::to_string(h.uq_feature_dim) + ") must be >= uq_rp_dim (" +
         std::to_string(h.uq_rp_dim) + ")");
  h.uq_n_density = h.uq_feature_dim - h.uq_rp_dim;    // = 1 + number of invariant blocks
  // the kernels write exactly max_density density channels; more would be read uninitialised by the GMM
  if (max_density > 0 && h.uq_n_density > max_density)
    fail("uq_n_density (" + std::to_string(h.uq_n_density) + ") > " + std::to_string(max_density) +
         ", the number of density channels this model's UQ kernels write. Artifact has an unexpected "
         "invariant-block count.");
  // D_basis from R; the flat-size check is blind to a transposed [rp_dim, D_basis] R (same element count), so
  // the stored column count is checked too
  if (h.uq_rp_matrix.size() % (size_t) h.uq_rp_dim != 0)
    fail("uq_rp_matrix size (" + std::to_string(h.uq_rp_matrix.size()) + ") not divisible by rp_dim " +
         std::to_string(h.uq_rp_dim));
  h.uq_d_basis = (int) (h.uq_rp_matrix.size() / (size_t) h.uq_rp_dim);
  {
    auto it = npz.find("uq_rp_matrix");
    if (it != npz.end() && it->second.shape.size() == 2 && (int) it->second.shape[1] != h.uq_rp_dim)
      fail("uq_rp_matrix has shape [" + std::to_string(it->second.shape[0]) + ", " +
           std::to_string(it->second.shape[1]) + "]; expected column count == uq_rp_dim (" +
           std::to_string(h.uq_rp_dim) + "). R may be transposed or mislabeled.");
  }
  // the evaluators index the dense arrays flat with no bounds check: a mismatched export must fail here
  const size_t E = h.uq_n_elements, K = h.uq_max_clusters, D = h.uq_feature_dim;
  auto chk = [&](const char *name, size_t got, size_t want) {
    if (got != want)
      fail(std::string(name) + " has " + std::to_string(got) + " elements, expected " + std::to_string(want));
  };
  chk("uq_centroids", h.uq_centroids.size(), E * K * D);
  chk("uq_inv_cov", h.uq_inv_cov.size(), E * K * D * D);
  chk("uq_n_clusters", h.uq_n_clusters.size(), E);
  chk("uq_interp_thresholds", h.uq_interp_thresholds.size(), E * K);
  // every model element needs real GMM clusters (an incomplete export is rejected rather than given a sentinel
  // gamma), and no more than Kmax of them (the GMM loops index centroids / inv_cov / thresholds by cluster)
  for (size_t e = 0; e < h.uq_n_clusters.size(); e++) {
    if (h.uq_n_clusters[e] <= 0)
      fail("element index " + std::to_string(e) + " has no GMM clusters; uqv6 artifacts must cover all "
           "elements. Re-export the UQ artifacts with a current grace_utils.");
    if (h.uq_n_clusters[e] > h.uq_max_clusters)
      fail("element index " + std::to_string(e) + " has " + std::to_string(h.uq_n_clusters[e]) +
           " GMM clusters, more than uq_max_clusters = " + std::to_string(h.uq_max_clusters));
  }
  // a partially populated / mis-inverted export (e.g. a singular covariance) can leave NaN / Inf in the GMM
  // arrays, which would silently poison sigma at run time
  auto finite = [&](const char *name, const std::vector<double> &v) {
    for (double x : v)
      if (!std::isfinite(x)) fail(std::string(name) + " contains a non-finite value (NaN/Inf); re-export the UQ "
                                  "artifacts.");
  };
  finite("uq_centroids", h.uq_centroids);
  finite("uq_inv_cov", h.uq_inv_cov);
  finite("uq_interp_thresholds", h.uq_interp_thresholds);
  finite("uq_rp_matrix", h.uq_rp_matrix);
}

}    // namespace grace

// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// load_uq_head: the uqv6 UQ head of a model from kokkos.npz.
#include "grace/uq_head.h"
#include "npz3.h"

#include <string>

namespace grace {

using namespace npz;

static void load_uq_head_arrays(UqHead &h, const cnpy::npz_t &npz, int n_elements, int max_density,
                                const std::string &who)
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
  // the GMM arrays are indexed by the model's element index; the sizes below multiply, so each is bounded first
  h.uq_n_elements = npz_get_int_scalar(npz, "uq_n_elements", n_elements);
  require_eq("uq_n_elements", h.uq_n_elements, n_elements, "the model's n_elements");
  h.uq_max_clusters = npz_get_int_in(npz, "uq_max_clusters", 0, 1, kMaxDim);
  h.uq_rp_dim = npz_get_int_in(npz, "uq_rp_dim", 0, 1, kMaxDim);
  h.uq_feature_dim = npz_get_int_in(npz, "uq_feature_dim", 0, h.uq_rp_dim, kMaxDim);
  // v6 feature-build flags (confirmatory: must match the v6 profile below)
  h.uq_normalize = npz_get_int_scalar(npz, "uq_rp_normalize", 0);
  const int add_density = npz_get_int_scalar(npz, "uq_rp_add_density_channel", 0);
  h.uq_density_scale = npz_get_double_scalar(npz, "uq_rp_density_scale", 1.0);
  const int transform = npz_get_int_scalar(npz, "uq_feature_transform", 0);
  h.uq_centroids = npz_require_double(npz, "uq_centroids");
  h.uq_inv_cov = npz_require_double(npz, "uq_inv_cov");
  h.uq_n_clusters = npz_require_int(npz, "uq_n_clusters");
  h.uq_interp_thresholds = npz_require_double(npz, "uq_interp_thresholds");
  h.uq_rp_matrix = npz_require_double(npz, "uq_rp_matrix");    // [D_basis, rp_dim] row-major

  // the evaluators implement normalize + density with the identity transform only (no asinh)
  if (h.uq_normalize != 1 || add_density != 1 || transform != 0)
    fail("schema v6 expects normalize=1, add_density_channel=1, feature_transform=0 (identity); got normalize=" +
         std::to_string(h.uq_normalize) + ", add_density=" + std::to_string(add_density) +
         ", transform=" + std::to_string(transform) + ". Artifact mislabeled or unsupported variant.");
  h.uq_n_density = h.uq_feature_dim - h.uq_rp_dim;    // = 1 + number of invariant blocks
  // the kernels write exactly max_density density channels; more would be read uninitialised by the GMM
  if (max_density > 0 && h.uq_n_density > max_density)
    fail("uq_n_density (" + std::to_string(h.uq_n_density) + ") > " + std::to_string(max_density) +
         ", the number of density channels this model's UQ kernels write. Artifact has an unexpected "
         "invariant-block count.");
  // R is [D_basis, rp_dim]: the column count guards against a transposed R (the same element count)
  const auto rs = npz_shape_rank(npz, "uq_rp_matrix", 2);
  require_eq("uq_rp_matrix", rs[1], h.uq_rp_dim, "uq_rp_dim", "columns");
  h.uq_d_basis = (int) rs[0];
  // the evaluators index the dense arrays flat with no bounds check: a mismatched export must fail here
  const size_t E = h.uq_n_elements, K = h.uq_max_clusters, D = h.uq_feature_dim;
  require_size("uq_centroids", h.uq_centroids, mul_sat(mul_sat(E, K), D));
  require_size("uq_inv_cov", h.uq_inv_cov, mul_sat(mul_sat(mul_sat(E, K), D), D));
  require_size("uq_n_clusters", h.uq_n_clusters, E);
  require_size("uq_interp_thresholds", h.uq_interp_thresholds, mul_sat(E, K));
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
  // (NaN / Inf in the GMM arrays, e.g. from a singular covariance, are rejected by npz_get_double)
}

// the UQ head; every error names the UQ artifacts ("<who> UQ: ..."), so only they need a re-export
void load_uq_head(UqHead &h, const cnpy::npz_t &npz, int n_elements, int max_density, const std::string &who)
{
  const std::string pre = who + " UQ: ";
  try {
    load_uq_head_arrays(h, npz, n_elements, max_density, who);
  } catch (const Error &e) {
    const std::string m = e.what();
    if (m.rfind(pre, 0) == 0) throw;
    throw Error(pre + m);
  }
}

}    // namespace grace

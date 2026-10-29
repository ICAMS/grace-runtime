// SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
// The kind of a model file (GRACE-1L / 2L / 3L kokkos.npz, PACE), for the consumers that pick the engine from the file:
// the Python modules and the LAMMPS styles grace/runtime and pace/runtime.
#ifndef GRACE_MODEL_KIND_H
#define GRACE_MODEL_KIND_H

#include <string>

namespace grace {

enum class ModelKind { Grace1L, Grace2L, Grace3L, Pace };

// A GRACE kokkos.npz names its kind in grace_model_kind ("GRACE-1L", "GRACE-2L", "GRACE-3L"; tensorpotential exports
// with grace_format_version >= 1, checked by the loaders). Older exports have neither; for them the npz member names
// decide (ZIP central directory, ZIP64 too): mlp_rad_W0.npy -> 1L, B0_lin_transform_W.npy -> 2L, prod_names.npy -> 3L,
// pace_format_version.npy -> PACE. A file ending in .yaml / .yml is PACE (pacemaker's B-basis, read by
// PaceModel::load). Throws grace::Error when the file cannot be opened or is none of these.
ModelKind model_kind(const std::string &path);

const char *model_kind_name(ModelKind);    // "GRACE-1L", "GRACE-2L", "GRACE-3L", "PACE"
const char *model_kind_tag(ModelKind);     // "1l", "2l", "3l", "pace" (the Python module names)

}    // namespace grace

#endif

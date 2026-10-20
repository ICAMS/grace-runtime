# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Test model helpers without a pytest dependency (CI calls them from plain envs)."""


def strip_uq(src, dst):
    """Copy of a kokkos.npz without its uq_* arrays: a model exported without a UQ head."""
    import numpy as np
    with np.load(src) as z:
        np.savez(dst, **{k: z[k] for k in z.files if not k.startswith("uq_")})
    return str(dst)

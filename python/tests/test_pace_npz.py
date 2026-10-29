# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""The internal PACE npz: the arrays of libgrace's converter (_core.pace_arrays) are flat, and the loader checks them."""
import os

import numpy as np
import pytest

from pace_tools import FX, arrays


@pytest.mark.parametrize("name", ["carbon_zbl", "pdh_zbl", "mnbtw_4el", "ho_nd5"])
def test_arrays_are_flat(name):
    z = arrays(os.path.join(FX, f"{name}.yaml"))
    assert int(z["pace_format_version"][0]) == 1
    for k, v in z.items():    # np.savez / np.load(allow_pickle=False) keep them: no object arrays
        assert v.dtype in (np.int32, np.float64) or v.dtype.kind == "S", (k, v.dtype)
    nel = len(z["element_names"])
    assert z["bond_defined"].shape == (nel, nel)
    assert z["bond_defined"].sum() == sum(1 for k in z if k.endswith("_crad"))


def test_loader_rejects_missing_radparameters(tmp_path):
    from grace_runtime import Model
    z = arrays(os.path.join(FX, "al_fs_plain.yaml"))    # ChebPow
    z["b0_0_radparameters"] = np.zeros(0)
    np.savez(tmp_path / "bad.npz", **z)
    with pytest.raises(ValueError, match="radparameters"):
        Model(str(tmp_path / "bad.npz"))

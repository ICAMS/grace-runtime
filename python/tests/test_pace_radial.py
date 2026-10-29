# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""libgrace's PACE radial functions R_nl(r), dR_nl/dr and the core repulsion (fp64) against pyace
(tests/fixtures/pace/ref_pace.npz, radial/...; pyace's cubic splines of step 5e-5 A)."""
import os

import numpy as np
import pytest

from grace_runtime import _core
from pace_tools import FX, MODELS, model_path, write_npz

MODELS = MODELS + ["al_cheblinear"]
TOL = {"fr": 1e-10, "cr": 1e-10, "dfr": 1e-8, "dcr": 1e-8}    # relative to max(1, max |ref|)


@pytest.fixture(scope="module")
def ref():
    return np.load(os.path.join(FX, "ref_pace.npz"))


@pytest.mark.parametrize("name", MODELS)
def test_radial_matches_pyace(name, ref, tmp_path):
    src = model_path(name, tmp_path)
    npz = write_npz(src, tmp_path / "m.npz")
    pairs = sorted({k.split("/")[2] for k in ref.files if k.startswith(f"radial/{name}/")})
    assert pairs
    defined = np.load(npz)["bond_defined"]
    for pair in pairs:
        i, j = map(int, pair.split("_"))
        g = lambda k: ref[f"radial/{name}/{pair}/{k}"]
        if not defined[i, j]:    # a pair without a block: pyace's functions are zero
            assert not np.abs(g("fr")).any() and not np.abs(g("cr")).any()
            continue
        r = g("r")
        keep = slice(None)
        if name == "al_cheblinear":    # not zero at rcut: pyace's last spline interval (pinned to 0) is off
            keep = r < float(np.load(npz)[f"b{i}_{j}_rcut"][0]) - 1e-4
        out = _core._pace_radial(npz, i, j, r[keep])
        for k in ("fr", "dfr", "cr", "dcr"):
            want = g(k)[keep]
            if k in ("fr", "dfr"):    # pyace's arrays have the model's largest nradmax / lmax: compare the pair's part
                want = want[:, :out[k].shape[1], :out[k].shape[2]]
            err = np.abs(out[k] - want).max() / max(1.0, np.abs(want).max())
            assert err < TOL[k], (name, pair, k, err)

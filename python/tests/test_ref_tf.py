# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Registry models against independent references: tensorpotential's TPCalculator on their -fp64 SavedModels, on the
fixture cells (tests/fixtures/ref_tf/registry_tf_fp64.npz). Covers the
GRACE-2L-SMAX-OMAT-large architecture, which the LAMMPS references (conftest TEST_MODELS) do not; the GPU side is
python/tests/test_cuda_parity.py::test_against_tf_fp64. Thresholds: fp32 network against fp64 (test_vs_tp.py)."""
import os

import numpy as np
import pytest
from ase.io import read

from grace_runtime import GRACERuntimeCalculator

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "fixtures")
REF = os.path.join(FX, "ref_tf", "registry_tf_fp64.npz")
MODELS = {"GRACE-2L-SMAX-OMAT-large": "GRACE_TEST_M2L"}
CELLS = ["W54", "WMo54", "W2tri"]
TOL = {"energy": 1e-6, "forces": 2e-5, "energies": 1e-5, "stress": 1e-5}    # eV/atom, eV/A, eV, relative
TOL_SMALL = 2e-6       # eV/atom below 10 atoms (fp32 floor of the total energy, as test_vs_tp.py)


@pytest.fixture(scope="module")
def ref():
    return np.load(REF)


@pytest.fixture(scope="module", params=list(MODELS))
def model(request):
    from grace_runtime import registry
    return request.param, os.environ.get(MODELS[request.param]) or registry.resolve(request.param)


@pytest.mark.parametrize("cell", CELLS)
def test_matches_tf_fp64(model, ref, cell):
    name, path = model
    at = read(f"{FX}/{cell}.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})
    at.calc = GRACERuntimeCalculator(path, num_threads=2)
    k = f"{name}/{cell}"
    n = len(at)
    d = {"energy": abs(at.get_potential_energy() - ref[f"{k}/energy"]) / n,
         "forces": np.abs(at.get_forces() - ref[f"{k}/forces"]).max(),
         "energies": np.abs(at.get_potential_energies() - ref[f"{k}/energies"]).max(),
         "stress": np.abs(at.get_stress() - ref[f"{k}/stress"]).max() / np.abs(ref[f"{k}/stress"]).max()}
    print(k, {q: f"{v:.2e}" for q, v in d.items()})
    for q, v in d.items():
        assert v < (TOL_SMALL if q == "energy" and n < 10 else TOL[q]), (q, v)

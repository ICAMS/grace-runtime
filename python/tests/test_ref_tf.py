# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Registry models against independent references: tensorpotential's TPCalculator on their -fp64 SavedModels (the
legacy GRACE-1L/2L models: their fp64 SavedModels), on the fixture cells (tests/fixtures/ref_tf/registry_tf_fp64.npz).
Covers the GRACE-2L-SMAX-OMAT-large architecture, which the LAMMPS references (conftest TEST_MODELS) do not, and the
legacy layout (tanh radial MLPs, RMS layer norm of the energy-MLP input, 1L AAA = AA1 x A, 2L without RMSNorms); the
GPU side is python/tests/test_gpu_parity.py::test_against_tf_fp64. Thresholds: fp32 network against fp64
(test_vs_tp.py)."""
import os

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from grace_runtime import GRACERuntimeCalculator

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "fixtures")
REF = os.path.join(FX, "ref_tf", "registry_tf_fp64.npz")
MODELS = ["GRACE-2L-SMAX-OMAT-large", "GRACE-1L-OAM", "GRACE-1L-OMAT", "GRACE-2L-OAM", "GRACE-2L-OMAT"]
ENV = {"GRACE-2L-SMAX-OMAT-large": "GRACE_TEST_M2L"}    # name: the environment variable of another file
CELLS = ["W54", "WMo54", "W2tri"]
TOL = {"energy": 1e-6, "forces": 2e-5, "energies": 1e-5, "stress": 1e-5}    # eV/atom, eV/A, eV, relative
TOL_SMALL = 2e-6       # eV/atom below 10 atoms (fp32 floor of the total energy, as test_vs_tp.py)


@pytest.fixture(scope="module")
def ref():
    return np.load(REF)


@pytest.fixture(scope="module", params=MODELS)
def model(request):
    from grace_runtime import registry
    name = request.param
    try:
        return name, os.environ.get(ENV.get(name, "")) or registry.resolve(name)
    except ValueError as e:    # no export in the registry yet (a user registry in the cache may have one)
        if name not in registry.load_entries().get("unsupported", {}):
            raise
        pytest.skip(str(e))


def check_against_ref(path, ref, name, cell):
    """the model file at path on the fixture cell against the reference arrays ref[f"{name}/{cell}/..."]"""
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


@pytest.mark.parametrize("cell", CELLS)
def test_matches_tf_fp64(model, ref, cell):
    name, path = model
    check_against_ref(path, ref, name, cell)


DIMERS = os.path.join(FX, "ref_tf", "dimers_tf_fp64.npz")
DIMER_MODELS = ["GRACE-2L-SMAX-OMAT-medium", "GRACE-2L-OMAT-medium-ft-E"]


@pytest.mark.parametrize("name", DIMER_MODELS)
def test_dimers(name):
    """GRACE-2L on W2, C2, O2 stretched toward the cutoff (4 - 5.9 A; TPCalculator on the -fp64 SavedModels,
    tests/fixtures/ref_tf/dimers_tf_fp64.npz): the RMSNorm inputs are small there and its epsilon (1e-10, as
    tensorpotential's) matters; with 1e-8 the energy was 0.9 eV off at 5.9 A. Measured (CPU): dE <= 1.4e-6 eV,
    dF <= 2.9e-5 eV/A."""
    from grace_runtime import registry
    ref = np.load(DIMERS)
    calc = GRACERuntimeCalculator(registry.resolve(name), num_threads=2)
    for el in ref["elements"]:
        for r in ref["distances"]:
            at = Atoms(f"{el}2", positions=[[0, 0, 0], [r, 0, 0]])
            at.calc = calc
            k = f"{name}/{el}2/{r}"
            dE = abs(at.get_potential_energy() - ref[f"{k}/energy"])
            dF = np.abs(at.get_forces() - ref[f"{k}/forces"]).max()
            assert dE <= 1e-5 and dF <= 1e-4, (k, dE, dF)

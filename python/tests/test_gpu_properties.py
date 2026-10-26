# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime.elastic_batch / eos_batch on the GPU against the CPU: the stress method keeps the fp32 kernels'
round-off small (measured: at most 0.02 GPa, eps 0.0025 - 0.02), so C to 0.05 GPa, clamped and with internal
relaxation; EOS bulk modulus to 0.1 %."""
import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from grace_runtime import elastic_batch, eos_batch, relax_batch

from gpu_tools import GPU, NO_GPU, gpu_ok

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M2 = CACHE / "GRACE-2L-OMAT-medium-ft-E" / "kokkos.npz"

pytestmark = pytest.mark.skipif(not gpu_ok() or not M2.exists(), reason=f"{NO_GPU} / registry model missing")


@pytest.fixture(scope="module")
def cells():
    w = bulk("W", "bcc", a=3.17, cubic=True)
    mo = bulk("Mo", "bcc", a=3.15, cubic=True).repeat(2)
    mo.symbols[0] = "Nb"
    del mo[5]
    return relax_batch(str(M2), [w, mo], fmax=1e-4, cell_filter="frechet", device="cpu")


@pytest.mark.parametrize("relax_atoms", [False, True])
def test_elastic_gpu_as_cpu(cells, relax_atoms):
    ats = [r["atoms"] for r in cells]
    g = elastic_batch(str(M2), ats, relax_atoms=relax_atoms, device=GPU)
    c = elastic_batch(str(M2), ats, relax_atoms=relax_atoms, device="cpu")
    for a, b in zip(g, c):
        assert np.abs(a["C"] - b["C"]).max() <= 0.05
        assert abs(a["BH"] - b["BH"]) <= 0.05 and abs(a["GH"] - b["GH"]) <= 0.05


def test_eos_gpu_as_cpu(cells):
    ats = [r["atoms"] for r in cells]
    g = eos_batch(str(M2), ats, device=GPU)
    c = eos_batch(str(M2), ats, device="cpu")
    for a, b in zip(g, c):
        assert abs(a["equilibrium_bulk_modulus"] / b["equilibrium_bulk_modulus"] - 1) <= 1e-3
        assert abs(a["equilibrium_volume"] / b["equilibrium_volume"] - 1) <= 1e-5

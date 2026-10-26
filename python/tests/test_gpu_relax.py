# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime.relax_batch on the GPU (one Batch.compute per step, neighbour lists on the GPU) against the CPU
path: the same minima (energy 1e-5 eV/atom, positions and cells 2e-3 A; fp32 kernels, other summation order, so the
trajectories differ by round-off), positions only and with the Frechet cell filter (also with D3); a new element in a
lazy input rebuilds the GPU batch; molecules (cells with zero vectors)."""
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from grace_runtime import Model, relax_batch, relax_batch_iter

from gpu_tools import GPU, NO_GPU, gpu_ok

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M2 = CACHE / "GRACE-2L-OMAT-medium-ft-E" / "kokkos.npz"

pytestmark = pytest.mark.skipif(not gpu_ok() or not M2.exists(), reason=f"{NO_GPU} / registry model missing")


@pytest.fixture(scope="module")
def model():
    return Model(str(M2))


def cells():
    w = bulk("W", "bcc", a=3.2, cubic=True).repeat(3)
    w.rattle(0.05, seed=1)
    wmo = bulk("W", "bcc", a=3.1, cubic=True).repeat((2, 2, 3))
    wmo.symbols[::3] = "Mo"
    del wmo[5]
    wmo.rattle(0.04, seed=2)
    nb = bulk("Nb", "bcc", a=3.25).repeat((2, 3, 2))
    nb.set_cell(nb.cell.array @ [[1, 0.03, 0], [0.02, 1, 0.01], [0, 0, 1.02]], scale_atoms=True)
    nb.rattle(0.03, seed=3)
    return [w, wmo, nb]


def same_minimum(g, c, n):
    assert g["converged"] and c["converged"]
    assert abs(g["energy"] - c["energy"]) / n <= 1e-5
    assert np.abs(g["atoms"].positions - c["atoms"].positions).max() <= 2e-3
    assert np.abs(g["atoms"].cell.array - c["atoms"].cell.array).max() <= 2e-3


@pytest.mark.parametrize("cell_filter, dispersion", [(None, None), ("frechet", None), ("frechet", "d3-pbe")])
def test_gpu_relaxes_as_cpu(model, cell_filter, dispersion):
    ats = cells()
    gpu = relax_batch(model, ats, fmax=0.01, cell_filter=cell_filter, device=GPU, dispersion=dispersion)
    cpu = relax_batch(model, ats, fmax=0.01, cell_filter=cell_filter, device="cpu", dispersion=dispersion)
    for g, c, at in zip(gpu, cpu, ats):
        same_minimum(g, c, len(at))


def test_gpu_lazy_input_new_element(model):
    """W first, Mo and Nb later in a generator: the species set grows, the GPU batch is set up again; results as with
    the list (all species known at the start)"""
    ats = cells()
    ref = relax_batch(model, ats, fmax=0.01, device=GPU)
    got = dict(relax_batch_iter(model, (a for a in ats), fmax=0.01, device=GPU, max_atoms=60))
    for k, at in enumerate(ats):
        same_minimum(got[k], ref[k], len(at))


def test_gpu_molecule(model):
    mol = Atoms("W3", positions=[[0, 0, 0], [2.6, 0, 0], [1.2, 2.1, 0.1]])
    g = relax_batch(model, [mol], fmax=0.01, device=GPU)[0]
    c = relax_batch(model, [mol], fmax=0.01, device="cpu")[0]
    same_minimum(g, c, 3)
    assert np.all(g["stress"] == 0)


def test_gpu_fix_symmetry(model):
    """fix_symmetry on the GPU: the minimum of the CPU path, the space group kept"""
    import spglib
    w = bulk("W", "bcc", a=3.2, cubic=True).repeat((1, 1, 2))
    w.symbols[0] = "Mo"
    w.set_cell(w.cell.array @ np.diag([1.0, 1.0, 1.03]), scale_atoms=True)
    w.positions[2, 2] += 0.08

    def number(at):
        return spglib.get_symmetry_dataset((at.cell.array, at.get_scaled_positions(), at.numbers), symprec=1e-3).number

    kw = {"fmax": 0.01, "cell_filter": "frechet", "fix_symmetry": True, "symprec": 1e-3}
    g = relax_batch(model, [w], device=GPU, **kw)[0]
    c = relax_batch(model, [w], device="cpu", **kw)[0]
    same_minimum(g, c, len(w))
    assert number(g["atoms"]) == number(c["atoms"])

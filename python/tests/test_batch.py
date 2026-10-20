# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime.compute_batch: per-structure results of a batch against single-structure runs
of GRACERuntimeCalculator(device="cuda") (energy 2e-6 eV/atom, forces 2e-5 eV/A + 1e-6 max|F|, stress 1e-6 eV/A^3:
fp32 with another summation order); chunking and the species union do not change them; device="cpu" gives the same
result format and the CPU calculator's values."""
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.cluster import Icosahedron
from ase.io import read

from grace_runtime import GRACERuntimeCalculator, _cuda, compute_batch

ROOT = Path(__file__).resolve().parents[2]
FX = ROOT / "tests" / "fixtures"
CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M2 = CACHE / "GRACE-2L-OMAT-medium-ft-E" / "kokkos.npz"    # 2L with a UQ head
M1 = CACHE / "GRACE-1L-OMAT-large-base" / "kokkos.npz"


def _cuda_ok():
    try:
        _cuda.module_for(0)
        return True
    except (ImportError, ValueError):
        return False


cuda = pytest.mark.skipif(not _cuda_ok(), reason="no CUDA module for this device")
pytestmark = pytest.mark.skipif(not (M1.exists() and M2.exists()), reason="registry models missing")


def gpu_batch(*args, **kw):
    return compute_batch(*args, device="cuda", **kw)


def lmp(name):
    return read(FX / f"{name}.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})


def structures():
    shifted = lmp("WMo54")
    shifted.positions += 1.3 * shifted.cell.array.sum(axis=0) + np.random.default_rng(1).normal(0, 0.02, (54, 3))
    slab = bulk("W", "bcc", a=3.165, cubic=True).repeat((3, 3, 4))
    slab.center(vacuum=8.0, axis=2)
    slab.pbc = [True, True, False]
    slab.rattle(0.03, seed=2)
    cluster = Icosahedron("W", 3, latticeconstant=3.9)
    cluster.rattle(0.03, seed=3)
    cluster.symbols[::3] = "Mo"
    nb = bulk("Nb", "bcc", a=3.30, cubic=True).repeat(2)    # a third element: species union {Mo, Nb, W}
    nb.rattle(0.03, seed=4)
    iso = Atoms("Mo", positions=[[1.0, 2.0, 3.0]], cell=[10, 10, 10], pbc=True)
    prim = bulk("W", "bcc", a=3.165)    # 1 atom, small triclinic cell: many images
    return [lmp("W54"), lmp("WMo54"), Atoms(cell=[8, 8, 8], pbc=True), lmp("W2tri"), shifted, slab, nb, iso, prim,
            cluster]


def single(at, calc, props=("energy", "forces", "stress")):
    at = at.copy()
    at.calc = calc
    r = {"energy": at.get_potential_energy()}
    if "forces" in props:
        r["forces"] = at.get_forces()
        r["stress"] = at.get_stress() if at.cell.rank == 3 and len(at) else np.zeros(6)
    return r


def close(b, s, n, e=2e-6, f=2e-5, st=1e-6):
    assert abs(b["energy"] - s["energy"]) / max(n, 1) <= e, (b["energy"], s["energy"])
    if n:
        assert b["forces"].shape == (n, 3)
        assert np.abs(b["forces"] - s["forces"]).max() <= f + 1e-6 * np.abs(s["forces"]).max()
        assert np.abs(b["stress"] - s["stress"]).max() <= st


@cuda
@pytest.mark.parametrize("model", [M1, M2], ids=["1L", "2L"])
def test_batch_matches_single_structure_runs(model):
    ats = structures()
    res = gpu_batch(str(model), ats)
    assert len(res) == len(ats)
    calc = GRACERuntimeCalculator(str(model), device="cuda")
    for at, r in zip(ats, res):
        if len(at) == 0:
            assert r["energy"] == 0.0 and r["forces"].shape == (0, 3)
            continue
        close(r, single(at, calc), len(at))


@cuda
def test_chunking_does_not_change_results():
    ats = structures()
    a = gpu_batch(str(M2), ats)
    b = gpu_batch(str(M2), ats, max_atoms=60)    # several chunks, the 216-atom slab alone
    for at, x, y in zip(ats, a, b):
        close(x, y, len(at), e=1e-6, f=1e-5, st=1e-7)


@cuda
def test_cpu_device_matches_cuda():
    ats = structures()
    g = gpu_batch(str(M2), ats)
    c = compute_batch(str(M2), ats, device="cpu", num_threads=8)
    for at, x, y in zip(ats, g, c):
        close(x, y, len(at), e=1e-5, f=5e-5)


@cuda
def test_uq_and_energies_as_calculator():
    ats = [lmp("WMo54"), lmp("W2tri")]
    res = gpu_batch(str(M2), ats, uq=True, energies=True)
    calc = GRACERuntimeCalculator(str(M2), device="cuda", uq=True)
    for at, r in zip(ats, res):
        a = at.copy()
        a.calc = calc
        e = a.get_potential_energies()
        assert np.abs(r["energies"] - e).max() <= 1e-5
        for k in ("gamma", "atomic_sigma"):
            ref = calc.results[k]
            assert np.abs(r[k] - ref).max() <= 1e-5 * max(1.0, np.abs(ref).max()), k
        assert np.array_equal(r["gmm_cluster"], calc.results["gmm_cluster"])
        assert abs(r["total_sigma"] - calc.results["total_sigma"]) <= 1e-5 * abs(calc.results["total_sigma"])


@cuda
def test_energy_only():
    ats = [lmp("WMo54"), lmp("W2tri")]
    full = gpu_batch(str(M1), ats)
    eo = gpu_batch(str(M1), ats, energy_only=True)
    for at, x, y in zip(ats, full, eo):
        assert "forces" not in y and "stress" not in y
        assert abs(x["energy"] - y["energy"]) / len(at) <= 2e-6


@cuda
def test_empty_list_and_errors():
    assert gpu_batch(str(M2), []) == []
    assert gpu_batch(str(M2), [Atoms(cell=[5, 5, 5], pbc=True)])[0]["energy"] == 0.0
    with pytest.raises(ValueError, match="not in the model"):
        gpu_batch(str(M2), [lmp("WMo54"), Atoms("Og", positions=[[0, 0, 0]], cell=[9, 9, 9], pbc=True)])
    with pytest.raises(ValueError, match="unknown device"):
        compute_batch(str(M2), [lmp("WMo54")], device="gpu")


@cuda
def test_batch_cache_bounded():
    """GPU Batches are cached per (model, species union) but only the most recent few: no device memory growth over
    calls with ever new element sets"""
    from grace_runtime import batch
    for el in ("Nb", "Ta", "V", "Cr"):
        at = bulk(el, "bcc", a=3.1, cubic=True).repeat(2)
        gpu_batch(str(M1), [at, lmp("W54")])
    assert len(batch._KK_BATCHES) <= 2

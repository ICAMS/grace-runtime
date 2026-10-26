# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime._kk_<arch>.Batch: many structures in one engine run (concatenated CSR with per-structure
ghost images, per-system energy / virial sums) against System on each structure alone: energy 2e-6 eV/atom, forces
2e-5 eV/A + 1e-6 max|F| (fp32, other summation order), virial 2e-5 eV/atom, UQ 1e-5 relative. Neighbour lists built
on the GPU from the host arrays (against the CPU calculator: test_batch.py::test_cpu_device_matches_gpu)."""
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.cluster import Icosahedron
from ase.io import read

from grace_runtime import _gpu
from grace_runtime.neighbors import _padded_cell

from gpu_tools import GPU, NO_GPU

ROOT = Path(__file__).resolve().parents[2]
FX = ROOT / "tests" / "fixtures"
CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
MODELS = {"1l": CACHE / "GRACE-1L-OMAT-large-base" / "kokkos.npz", "2l": CACHE / "GRACE-2L-OMAT-medium-ft-E" / "kokkos.npz",
          "3l": CACHE / "GRACE-3L-OMAT-large" / "kokkos.npz"}
SPECIES = ["Mo", "W"]


def _module():
    try:
        return _gpu.ensure_init(GPU) if GPU else None
    except (ImportError, ValueError):
        return None


pytestmark = pytest.mark.skipif(_module() is None or not all(p.exists() for p in MODELS.values()),
                                reason=f"{NO_GPU} / registry models missing")


@pytest.fixture(scope="module")
def mod():
    return _module()


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
    iso = Atoms("W", positions=[[1.0, 2.0, 3.0]], cell=[10, 10, 10], pbc=True)
    empty = Atoms(cell=[8, 8, 8], pbc=True)
    return [lmp("W54"), lmp("WMo54"), empty, lmp("W2tri"), shifted, slab, iso, cluster]


def arrays(at, rc):
    types = np.array([SPECIES.index(s) + 1 for s in at.get_chemical_symbols()], np.int32)
    cell = np.ascontiguousarray(_padded_cell(at, rc + 0.5)[0], dtype=np.float64)
    return types, np.ascontiguousarray(at.positions, dtype=np.float64).reshape(-1, 3), cell, [bool(p) for p in at.pbc]


def concat(ats, rc):
    a = [arrays(at, rc) for at in ats]
    natoms = np.array([len(x[0]) for x in a], np.int32)
    types = np.ascontiguousarray(np.concatenate([x[0] for x in a]), dtype=np.int32)
    pos = np.ascontiguousarray(np.concatenate([x[1] for x in a]).reshape(-1, 3))
    cells = np.ascontiguousarray(np.stack([x[2] for x in a]))
    pbc = np.array([x[3] for x in a], dtype=bool)
    return natoms, types, pos, cells, pbc


def check(res, ats, singles, uq=False):
    n0 = 0
    for k, (at, s) in enumerate(zip(ats, singles)):
        n = len(at)
        sl = slice(n0, n0 + n)
        assert abs(res["energy"][k] - s["energy"]) / max(n, 1) <= 2e-6, (k, res["energy"][k], s["energy"])
        if n:
            f = res["forces"][sl]
            assert np.abs(f - s["forces"]).max() <= 2e-5 + 1e-6 * np.abs(s["forces"]).max(), k
            assert np.abs(res["virial"][k] - s["virial"]).max() / n <= 2e-5, k
        if uq and n:
            for key in ("gamma", "atomic_sigma"):
                assert np.abs(res[key][sl] - s[key]).max() <= 1e-5 * max(1.0, np.abs(s[key]).max()), (k, key)
            assert np.array_equal(res["gmm_cluster"][sl], s["gmm_cluster"]), k
        n0 += n


@pytest.mark.parametrize("layer", ["1l", "2l", "3l"])
def test_batch_matches_single(mod, layer):
    m = mod.Model(str(MODELS[layer]))
    b = mod.Batch(m, SPECIES)
    ats = structures()
    rc = b.cutoff
    singles = []
    for at in ats:
        s = mod.System(m, SPECIES)
        t, p, c, pb = arrays(at, rc)
        singles.append(s.compute(t, p, c, pb) if len(at) else {"energy": 0.0})
    res = b.compute(*concat(ats, rc))
    assert res["energy"].shape == (len(ats),) and res["virial"].shape == (len(ats), 6)
    assert res["forces"].shape == (sum(len(a) for a in ats), 3)
    check(res, ats, singles)


def test_batch_uq_and_energies(mod):
    m = mod.Model(str(MODELS["2l"]))
    b = mod.Batch(m, SPECIES)
    ats = [lmp("WMo54"), lmp("W2tri")]
    singles = []
    for at in ats:
        t, p, c, pb = arrays(at, b.cutoff)
        singles.append(mod.System(m, SPECIES).compute(t, p, c, pb, energies=True, uq=True))
    res = b.compute(*concat(ats, b.cutoff), energies=True, uq=True)
    check(res, ats, singles, uq=True)
    assert np.abs(res["energies"] - np.concatenate([s["energies"] for s in singles])).max() <= 1e-5


def test_batch_energy_only(mod):
    m = mod.Model(str(MODELS["2l"]))
    b = mod.Batch(m, SPECIES)
    ats = [lmp("WMo54"), lmp("W2tri")]
    full = b.compute(*concat(ats, b.cutoff))
    eo = b.compute(*concat(ats, b.cutoff), energy_only=True)
    assert "forces" not in eo and "virial" not in eo
    assert np.abs(eo["energy"] - full["energy"]).max() / 54 <= 2e-6


def test_batch_errors_leave_batch_usable(mod):
    m = mod.Model(str(MODELS["2l"]))
    b = mod.Batch(m, SPECIES)
    ats = [lmp("WMo54"), lmp("W2tri")]
    ref = b.compute(*concat(ats, b.cutoff))
    natoms, types, pos, cells, pbc = concat(ats, b.cutoff)
    bad = types.copy()
    bad[5] = 3
    with pytest.raises(ValueError, match="types"):
        b.compute(natoms, bad, pos, cells, pbc)
    far = pos.copy()
    far[0] += 600 * cells[0][0]
    with pytest.raises(ValueError, match="511"):
        b.compute(natoms, types, far, cells, pbc)
    nan = pos.copy()
    nan[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        b.compute(natoms, types, nan, cells, pbc)
    with pytest.raises(ValueError, match="natoms"):
        b.compute(natoms + 1, types, pos, cells, pbc)
    again = b.compute(natoms, types, pos, cells, pbc)
    assert np.abs(again["energy"] - ref["energy"]).max() <= 1e-4
    assert np.abs(again["forces"] - ref["forces"]).max() <= 2e-5


def test_batch_keep_buffers_device_neighbors(mod):
    """keep_buffers (MD / relaxation loops): the device inputs and outputs are reused across calls of other sizes,
    results as a fresh batch; release_buffers in between does not change them"""
    m = mod.Model(str(MODELS["2l"]))
    b = mod.Batch(m, SPECIES)
    b.keep_buffers = True
    big, small = concat(structures(), b.cutoff), concat([lmp("W2tri"), lmp("WMo54")], b.cutoff)
    ref_big, ref_small = mod.Batch(m, SPECIES).compute(*big), mod.Batch(m, SPECIES).compute(*small)
    for args, ref in ((big, ref_big), (small, ref_small), (big, ref_big), (small, ref_small)):
        got = b.compute(*args)
        assert np.abs(got["energy"] - ref["energy"]).max() / 54 <= 1e-6
        assert np.abs(got["forces"] - ref["forces"]).max() <= 1e-5
    b.release_buffers()
    got = b.compute(*small, energies=True)
    assert np.abs(got["forces"] - ref_small["forces"]).max() <= 1e-5 and got["energies"].shape == (56,)


def test_batch_unknown_element(mod):
    m = mod.Model(str(MODELS["2l"]))
    with pytest.raises(ValueError, match="not in the model"):
        mod.Batch(m, ["W", "Xx"])


def test_batch_passes_bounded_by_atoms_and_images(mod):
    """device memory scales with atoms + ghost images: max_nall splits a call into several engine passes (structures
    whole), per-structure results as in one pass"""
    m = mod.Model(str(MODELS["2l"]))
    b = mod.Batch(m, SPECIES)
    ats = structures()
    one = b.compute(*concat(ats, b.cutoff))
    assert b.passes == 1
    many = b.compute(*concat(ats, b.cutoff), max_nall=600)
    assert b.passes > 2
    assert np.abs(many["energy"] - one["energy"]).max() / 54 <= 1e-6
    assert np.abs(many["forces"] - one["forces"]).max() <= 1e-5
    assert np.abs(many["virial"] - one["virial"]).max() <= 1e-4
    # one-atom cells (hundreds of images each): grouped by the images estimated from the cells before the
    # neighbour lists are built, then split by the real counts
    small = [bulk("W", "bcc", a=3.165 + 0.002 * k) for k in range(40)]
    one = b.compute(*concat(small, b.cutoff))
    many = b.compute(*concat(small, b.cutoff), max_nall=1500)
    assert b.passes > 4
    assert np.abs(many["energy"] - one["energy"]).max() <= 1e-6
    assert np.abs(many["forces"]).max() <= 1e-5 and np.abs(many["virial"] - one["virial"]).max() <= 1e-5


def test_batch_concurrent_threads(mod):
    """one Batch used from several Python threads (GIL released in compute): each call its own correct result"""
    from concurrent.futures import ThreadPoolExecutor
    m = mod.Model(str(MODELS["2l"]))
    b = mod.Batch(m, SPECIES)
    sets = [[lmp("WMo54"), lmp("W2tri")], [lmp("W54")], structures()[3:6], [lmp("W2tri"), lmp("WMo54"), lmp("W54")]]
    args = [concat(s, b.cutoff) for s in sets]
    ref = [b.compute(*a) for a in args]
    with ThreadPoolExecutor(4) as ex:
        for _ in range(5):
            got = list(ex.map(lambda a: b.compute(*a), args))
            for g, r in zip(got, ref):
                assert np.abs(g["energy"] - r["energy"]).max() <= 1e-4
                assert np.abs(g["forces"] - r["forces"]).max() <= 2e-5


def test_batch_3l_many_small_cells_fit(mod):
    """GRACE-3L keeps ~64 KB per atom + image on the GPU (layer arrays, fp64 exchange rows) beyond the chunk: 1000 W54
    cells (~430k atoms + images in one default pass) need ~30 GB. The batch shrinks its passes after an allocation
    failure instead of failing (on the same engine, its buffers reallocated); energies and forces as each cell alone"""
    from grace_runtime import compute_batch
    from ase.build import bulk
    base = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    ats = []
    for k in range(1000):
        a = base.copy()
        a.rattle(0.03, seed=k)
        ats.append(a)
    r = compute_batch(str(MODELS["3l"]), ats, device=GPU)
    one = compute_batch(str(MODELS["3l"]), [ats[0], ats[999]], device=GPU)
    for k, o in ((0, one[0]), (999, one[1])):
        assert abs(r[k]["energy"] - o["energy"]) <= 1e-6 * 54
        assert np.abs(r[k]["forces"] - o["forces"]).max() <= 2e-5
    import gc
    from grace_runtime import batch, calculator
    with calculator._KK_LOCK:    # the cached 3L Batch / engine hold GBs of device memory: free them for the next tests
        batch._KK_BATCHES.clear()
        for k in [k for k in calculator._KK_MODELS if "GRACE-3L" in k]:
            calculator._KK_MODELS.pop(k)
    gc.collect()

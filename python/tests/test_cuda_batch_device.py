# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Batch.compute_device: device inputs and outputs (torch tensors via DLPack), neighbour lists and images
built on the GPU, against Batch.compute (host lists) on the batch structures of test_batch (periodic, shifted,
triclinic 1-2 atom cells, slab, cluster, empty cell, one atom): energies 1e-6 eV/atom, forces 2e-5 eV/A, virials
1e-5 eV (fp32 kernels, another atom order inside a pass); split passes (small max_nall), energy_only, per-atom
energies, UQ; inputs on the CPU or of the wrong dtype are refused. Skipped without torch or a GPU module."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ase.data import atomic_numbers, chemical_symbols    # noqa: E402

from grace_runtime import _cuda    # noqa: E402
from grace_runtime.neighbors import _padded_cell_arrays    # noqa: E402

from test_batch import M1, M2, structures    # noqa: E402

M3 = M1.parent.parent / "GRACE-3L-OMAT-large" / "kokkos.npz"


def _ok():
    try:
        _cuda.module_for(0)
        return torch.cuda.is_available()
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _ok() or not M3.exists(), reason="no GPU module / torch GPU / models")


def arrays(structs, cutoff):
    species = sorted({chemical_symbols[z] for a in structs for z in a.numbers})
    lut = {atomic_numbers[s]: t + 1 for t, s in enumerate(species)}
    natoms = np.array([len(a) for a in structs], np.int32)
    types = np.array([lut[z] for a in structs for z in a.numbers], np.int32)
    pos = np.concatenate([a.positions for a in structs]).reshape(-1, 3).astype(np.float64)
    cells = np.stack([_padded_cell_arrays(a.cell.array, a.positions, cutoff)[0] for a in structs])
    pbc = np.array([[bool(p) for p in a.pbc] for a in structs])
    return species, natoms, types, pos, cells, pbc


def run_device(b, natoms, types, pos, cells, pbc, energies=False, uq=False, energy_only=False, max_nall=1 << 21):
    dev = torch.device("cuda", 0)
    n, s = len(types), len(natoms)
    out = {"energy": torch.zeros(s, dtype=torch.float64, device=dev)}
    kw = {}
    if not energy_only:
        out["forces"] = torch.zeros((n, 3), dtype=torch.float64, device=dev)
        out["virial"] = torch.zeros((s, 6), dtype=torch.float64, device=dev)
        kw.update(forces=out["forces"], virial=out["virial"])
    if energies:
        out["energies"] = kw["energies"] = torch.zeros(n, dtype=torch.float64, device=dev)
    if uq:
        for k in ("gamma", "atomic_sigma", "gmm_cluster"):
            out[k] = kw[k] = torch.zeros(n, dtype=torch.float64, device=dev)
    torch.cuda.synchronize()
    b.compute_device(natoms, torch.as_tensor(types, device=dev), torch.as_tensor(pos, device=dev),
                     np.ascontiguousarray(cells), np.ascontiguousarray(pbc), out["energy"], energy_only=energy_only,
                     max_nall=max_nall, **kw)
    return {k: v.cpu().numpy() for k, v in out.items()}


def batch_of(path, species):
    kk = _cuda.ensure_init("cuda")
    return kk.Batch(kk.Model(str(path)), species, chunksize=-1)


def compare(d, h, natoms):
    nat = np.maximum(natoms, 1)
    assert np.abs(d["energy"] - h["energy"]).max() / 1 <= 1e-6 * nat.max()
    # + 5e-6 eV: a one-atom cell with ~100 images sums fp32 bond terms in another order (1.9e-6 seen on an L40S)
    assert (np.abs(d["energy"] - h["energy"]) <= 1e-6 * nat + 5e-6).all()
    if "forces" in h:
        assert np.abs(d["forces"] - h["forces"]).max(initial=0) <= 2e-5
        assert np.abs(d["virial"] - h["virial"]).max() <= 1e-5 * max(1.0, np.abs(h["virial"]).max())


@pytest.mark.parametrize("path", [M1, M2, M3], ids=["1L", "2L", "3L"])
def test_device_equals_host(path):
    structs = structures()
    kk = _cuda.ensure_init("cuda")
    species = sorted({chemical_symbols[z] for a in structs for z in a.numbers})
    b = batch_of(path, species)
    species, natoms, types, pos, cells, pbc = arrays(structs, b.cutoff)
    h = b.compute(natoms, types, pos, cells, pbc, energies=True)
    d = run_device(b, natoms, types, pos, cells, pbc, energies=True)
    compare(d, h, natoms)
    assert np.abs(d["energies"] - h["energies"]).max() <= 2e-6
    assert b.passes == 1 and b.last_host_seconds > 0
    # split into passes of whole structures (the largest structure alone exceeds it)
    ds = run_device(b, natoms, types, pos, cells, pbc, max_nall=400)
    assert b.passes > 1
    compare(ds, h, natoms)
    e = run_device(b, natoms, types, pos, cells, pbc, energy_only=True)
    assert set(e) == {"energy"}
    assert np.abs(e["energy"] - h["energy"]).max() <= 1e-6 * natoms.max()
    del kk


def test_device_uq():
    structs = structures()
    species = sorted({chemical_symbols[z] for a in structs for z in a.numbers})
    b = batch_of(M2, species)
    species, natoms, types, pos, cells, pbc = arrays(structs, b.cutoff)
    h = b.compute(natoms, types, pos, cells, pbc, uq=True)
    d = run_device(b, natoms, types, pos, cells, pbc, uq=True)
    for k in ("gamma", "atomic_sigma"):
        assert np.abs(d[k] - h[k]).max() <= 1e-4 * max(1.0, np.abs(h[k]).max()), k
    assert (d["gmm_cluster"] == h["gmm_cluster"]).mean() > 0.99


def test_device_inputs_checked():
    structs = structures()[:2]
    species = sorted({chemical_symbols[z] for a in structs for z in a.numbers})
    b = batch_of(M1, species)
    species, natoms, types, pos, cells, pbc = arrays(structs, b.cutoff)
    dev = torch.device("cuda", 0)
    e = torch.zeros(len(natoms), dtype=torch.float64, device=dev)
    with pytest.raises((ValueError, TypeError)):    # positions on the host
        b.compute_device(natoms, torch.as_tensor(types, device=dev), torch.as_tensor(pos), cells, pbc, e)
    with pytest.raises((ValueError, TypeError)):    # float32 positions
        b.compute_device(natoms, torch.as_tensor(types, device=dev), torch.as_tensor(pos, device=dev).float(), cells,
                         pbc, e)
    with pytest.raises((ValueError, TypeError)):    # an output on the host: no silent copy
        b.compute_device(natoms, torch.as_tensor(types, device=dev), torch.as_tensor(pos, device=dev), cells, pbc,
                         torch.zeros(len(natoms), dtype=torch.float64))
    bad = types.copy()
    bad[0] = len(species) + 1
    with pytest.raises(ValueError, match="types"):
        b.compute_device(natoms, torch.as_tensor(bad, device=dev), torch.as_tensor(pos, device=dev), cells, pbc, e)


def test_host_arguments_on_the_device_refused():
    """natoms / cells / pbc are read on the host: CUDA arrays there raise instead of crashing the process"""
    b = batch_of(M1, ["W"])
    dev = torch.device("cuda", 0)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.6, 1.6, 1.6]], dtype=torch.float64, device=dev)
    types = torch.ones(2, dtype=torch.int32, device=dev)
    natoms, cells, pbc = np.array([2], np.int32), np.eye(3)[None] * 3.2, np.ones((1, 3), bool)
    e = torch.zeros(1, dtype=torch.float64, device=dev)
    with pytest.raises((TypeError, ValueError)):
        b.compute_device(natoms, types, pos, torch.as_tensor(cells, device=dev), pbc, e)
    with pytest.raises((TypeError, ValueError)):
        b.compute_device(torch.as_tensor(natoms, device=dev), types, pos, cells, pbc, e)
    with pytest.raises((TypeError, ValueError)):
        b.compute_device(natoms, types, pos, cells, torch.as_tensor(pbc, device=dev), e)
    b.compute_device(natoms, types, pos, cells, pbc, e)    # the host arrays: fine
    assert e.item() < 0

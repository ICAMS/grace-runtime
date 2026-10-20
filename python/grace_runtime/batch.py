# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""compute_batch: many structures per fused-kernel pass on the GPU (one CSR with each structure's ghost images,
per-structure sums on the device), or a loop over the CPU calculator with the same result format."""
import os
from collections import OrderedDict

import numpy as np
from ase.calculators.calculator import all_changes
from ase.data import atomic_numbers, chemical_symbols

from . import registry
from ._core import Model
from .calculator import _KK_CALCS, _KK_LOCK, _KK_MODELS, GRACERuntimeCalculator, _chunk_arg, _select_device, default_threads
from .neighbors import _padded_cell_arrays

_KK_BATCHES = OrderedDict()    # (model path, species) -> GPU Batch, the most recent _KK_BATCHES_KEEP (device memory)
_KK_BATCHES_KEEP = 2


def compute_batch(model, structures, device: str = "auto", uq: bool = False, energies: bool = False,
                  energy_only: bool = False, max_atoms: int = 131072, max_nall: int = 1 << 21,
                  num_threads: int | None = None, chunksize="auto") -> list:
    """Energy, forces and stress of many ASE structures at once.

    model: path to a kokkos.npz, a registry name or a grace_runtime.Model.
    device: "auto" (default: the GPU when usable, as GRACERuntimeCalculator), "cuda" / "cuda:N" (fused GPU kernels,
        fp32) or "cpu" (a loop over GRACERuntimeCalculator).
    uq: the UQ head (uqv6 "gamma_only") as GRACERuntimeCalculator(uq=True).
    energies: per-atom energies. energy_only: no forces / stress (no backward pass).
    max_atoms: atoms per call into the GPU module (host memory of the neighbour lists; structures are not split).
    max_nall: atoms + periodic images per engine pass (GPU memory; small cells have many images per atom).
    num_threads: host threads of the neighbour search (cuda) or of the CPU calculator.
    chunksize: atoms per GPU engine pass, as GRACERuntimeCalculator ("auto" or a number).

    Returns one dict per structure: "energy" (eV), "forces" [n, 3] (eV/A), "stress" (ASE Voigt, eV/A^3, zeros for
    cells of rank < 3), with energies: "energies" [n], with uq: "gamma", "atomic_sigma", "gmm_cluster" [n] and
    "total_sigma".
    """
    structures = list(structures)
    nth = int(num_threads) if num_threads else default_threads()
    m = model if isinstance(model, Model) else Model(registry.resolve(os.fspath(model)))
    device, kkm, _ = _select_device(device, m)
    if kkm is None:
        return _cpu(m, structures, uq, energies, energy_only, nth)
    kk = kkm[0]
    if uq:
        if not m.has_uq:
            raise ValueError("the model has no UQ head (kokkos.npz exported without UQ artifacts)")
        m.validate_uq()
    z = np.unique(np.concatenate([np.asarray(a.numbers) for a in structures])) if structures else np.zeros(0, int)
    if len(z) == 0:
        return [_split(None, a, 0, 0, uq, energies, energy_only) for a in structures]
    species = sorted(chemical_symbols[k] for k in z)
    b = _gpu_batch(kk, m, species, _chunk_arg(chunksize))
    results = []
    for lo, hi in _chunks([len(a) for a in structures], max_atoms):
        chunk = structures[lo:hi]
        out = _gpu_compute(b, species, [len(a) for a in chunk], np.concatenate([a.numbers for a in chunk]),
                           np.concatenate([a.positions for a in chunk]).reshape(-1, 3),
                           np.stack([a.cell.array for a in chunk]), [[bool(p) for p in a.pbc] for a in chunk],
                           uq=uq, energies=energies, energy_only=energy_only, num_threads=nth, max_nall=max_nall)
        n0 = 0
        for k, a in enumerate(chunk):
            results.append(_split(out, a, k, n0, uq, energies, energy_only))
            n0 += len(a)
    return results


def _gpu_batch(kk, m, species, chunk, keep=False):
    """the cached GPU Batch of a Model for a species set (sorted symbols) and a chunksize argument of the module;
    keep: one that keeps its device buffers between calls (MD loops, the TorchSim model)"""
    key = (m.path, tuple(species), chunk, keep)
    with _KK_LOCK:
        if m.path not in _KK_MODELS:
            _KK_MODELS[m.path] = kk.Model(m.path)
        if key not in _KK_BATCHES:
            _KK_BATCHES[key] = kk.Batch(_KK_MODELS[m.path], species, chunksize=chunk)    # ValueError: element the model lacks
            _KK_BATCHES[key].keep_buffers = keep
            while len(_KK_BATCHES) > _KK_BATCHES_KEEP:
                _KK_BATCHES.popitem(last=False)    # device memory goes with the last reference
        _KK_BATCHES.move_to_end(key)
        return _KK_BATCHES[key]


def _gpu_compute(b, species, natoms, numbers, pos, cells, pbc, uq=False, energies=False, energy_only=False,
                 num_threads=1, max_nall=1 << 21):
    """One call into the GPU module for structures given as arrays: natoms [S], numbers [N] and pos [N, 3] of all
    atoms (structure after structure), cells [S, 3, 3] (rows: cell vectors; zero vectors are padded), pbc [S, 3].
    Returns the module's output (energy [S], forces [N, 3], virial [S, 6] xx yy zz xy xz yz, ...)."""
    natoms = np.asarray(natoms, dtype=np.int32)
    pos = np.ascontiguousarray(pos, dtype=np.float64).reshape(-1, 3)
    lut = np.zeros(max(atomic_numbers[s] for s in species) + 1, np.int32)
    for t, s in enumerate(species):
        lut[atomic_numbers[s]] = t + 1
    types = np.ascontiguousarray(lut[np.asarray(numbers)], dtype=np.int32)
    ends = np.cumsum(natoms)
    cells = np.ascontiguousarray([_padded_cell_arrays(c, pos[e - n:e], b.cutoff)[0]
                                  for c, n, e in zip(np.asarray(cells, dtype=np.float64), natoms, ends)],
                                 dtype=np.float64).reshape(-1, 3, 3)
    pbc = np.ascontiguousarray(pbc, dtype=bool).reshape(-1, 3)
    return b.compute(natoms, types, pos, cells, pbc, energies=energies, uq=uq, energy_only=energy_only,
                     num_threads=num_threads, max_nall=max_nall)


def _chunks(sizes, max_atoms):
    """consecutive [lo, hi) ranges of at most max_atoms atoms (a larger structure alone)"""
    lo, acc = 0, 0
    for i, n in enumerate(sizes):
        if acc and acc + n > max_atoms:
            yield lo, i
            lo, acc = i, 0
        acc += n
    if lo < len(sizes):
        yield lo, len(sizes)


def _split(out, at, k, n0, uq, energies, energy_only):
    """structure k of a batch result (out None: no atoms in the whole call)"""
    n = len(at)
    sl = slice(n0, n0 + n)
    r = {"energy": float(out["energy"][k]) if out is not None else 0.0}
    if not energy_only:
        r["forces"] = out["forces"][sl].copy() if out is not None else np.zeros((n, 3))
        if at.cell.rank == 3 and out is not None:
            w = out["virial"][k]    # xx yy zz xy xz yz
            r["stress"] = -np.array([w[0], w[1], w[2], w[5], w[4], w[3]]) / at.get_volume()
        else:
            r["stress"] = np.zeros(6)
    if energies:
        r["energies"] = out["energies"][sl].copy() if out is not None else np.zeros(n)
    if uq:
        for key in ("gamma", "atomic_sigma", "gmm_cluster"):
            r[key] = out[key][sl].copy() if out is not None else np.zeros(n)
        r["total_sigma"] = float(r["atomic_sigma"].sum())
    return r


def _cpu(m, structures, uq, energies, energy_only, nth):
    calc = GRACERuntimeCalculator(m, num_threads=nth, uq=uq, device="cpu")
    props = ["energy"] + ([] if energy_only else ["forces", "stress"]) + (["energies"] if energies else [])
    results = []
    for at in structures:
        if len(at) == 0:    # the calculator needs atoms; same zeros as the GPU path
            results.append(_split(None, at, 0, 0, uq, energies, energy_only))
            continue
        calc.calculate(at.copy(), props, all_changes)    # one evaluation for all requested properties
        res = calc.results
        r = {"energy": float(res["energy"])}
        if not energy_only:
            r["forces"] = res["forces"].copy()
            r["stress"] = res["stress"].copy() if at.cell.rank == 3 else np.zeros(6)
        if energies:
            r["energies"] = res["energies"].copy()
        if uq:
            for key in ("gamma", "atomic_sigma", "gmm_cluster"):
                r[key] = res[key].copy()
            r["total_sigma"] = float(res["total_sigma"])
        results.append(r)
    return results


def free_gpu_memory():
    """Return the per-atom GPU buffers of the live calculators (GRACERuntimeCalculator.free_gpu_memory()) and drop the
    GPU models and batches cached by compute_batch and the calculators (weights, type tables, engines): their device
    memory goes back unless a live GRACERuntimeCalculator still uses the model. The next GPU call loads it again."""
    import gc
    with _KK_LOCK:
        calcs = list(_KK_CALCS)
        _KK_BATCHES.clear()
        _KK_MODELS.clear()
    for c in calcs:    # outside the lock: a compute running in another thread holds its System until it returns
        c.free_gpu_memory()
    gc.collect()

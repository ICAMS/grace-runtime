# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""compute_batch: many structures per fused-kernel pass on the GPU (one CSR with each structure's ghost images,
per-structure sums on the device), or a loop over the CPU kernels with the same result format."""
from collections import OrderedDict

import numpy as np
from ase.data import atomic_numbers, chemical_symbols

from ._core import Model
from .calculator import (_KK_CALCS, _KK_LOCK, _KK_MODELS, _chunk_arg, _CpuEngine, _results, _select_device,
                         _type_lut, default_threads)
from .dispersion import resolve as _resolve_dispersion
from .neighbors import _padded_cell_arrays

# (path, spline, species, chunk, dispersion, keep) -> GPU Batch, the most recent _KK_BATCHES_KEEP (device memory)
_KK_BATCHES = OrderedDict()
_KK_BATCHES_KEEP = 2


def compute_batch(model, structures, device: str = "auto", uq: bool = False, energies: bool = False,
                  energy_only: bool = False, max_atoms: int = 131072, max_nall: int = 1 << 21,
                  num_threads: int | None = None, chunksize="auto", dispersion=None,
                  dispersion_parts: bool = False) -> list:
    """Energy, forces and stress of many ASE structures at once.

    model: path to a kokkos.npz, a registry name or a grace_runtime.Model.
    device: "auto" (default: the GPU when usable, as GRACERuntimeCalculator), "cuda" / "cuda:N", "rocm" / "rocm:N",
        "gpu" / "gpu:N" (fused GPU kernels, fp32) or "cpu" (the CPU kernels, one structure after another).
    uq: the UQ head (uqv6 "gamma_only") as GRACERuntimeCalculator(uq=True).
    energies: per-atom energies. energy_only: no forces / stress (no backward pass).
    max_atoms: atoms per call into the GPU module (GPU memory of the neighbour lists; structures are not split).
    max_nall: atoms + periodic images per engine pass (GPU memory; small cells have many images per atom).
    num_threads: threads of the CPU kernels (the GPU builds its neighbour lists itself).
    chunksize: atoms per GPU engine pass, as GRACERuntimeCalculator ("auto" or a number).
    dispersion: a dispersion correction added to the model, as GRACERuntimeCalculator (a grace_runtime.D3 or "d3-pbe",
        "d3-r2scan"); dispersion_parts: also its own forces, stress and per-atom energies.

    Returns one dict per structure: "energy" (eV), "forces" [n, 3] (eV/A), "stress" (ASE Voigt, eV/A^3, zeros for
    cells of rank < 3), with energies: "energies" [n], with uq: "gamma", "atomic_sigma", "gmm_cluster" [n] and
    "total_sigma"; with dispersion "dispersion_energy" (included in "energy"), with dispersion_parts also
    "dispersion_energies" and (unless energy_only) "dispersion_forces", "dispersion_stress" (included in the totals).
    compute_batch_iter streams the same results.
    """
    return [r for _, r in compute_batch_iter(model, list(structures), device=device, uq=uq, energies=energies,
                                             energy_only=energy_only, max_atoms=max_atoms, max_nall=max_nall,
                                             num_threads=num_threads, chunksize=chunksize,
                                             dispersion=dispersion, dispersion_parts=dispersion_parts)]


def compute_batch_iter(model, structures, device: str = "auto", uq: bool = False, energies: bool = False,
                       energy_only: bool = False, max_atoms: int = 131072, max_nall: int = 1 << 21,
                       num_threads: int | None = None, chunksize="auto", dispersion=None,
                       dispersion_parts: bool = False):
    """compute_batch as a stream: yields (input index, result) in input order, reading `structures` (any iterable)
    one call into the GPU module at a time (at most max_atoms atoms, or one larger structure), so memory holds one
    portion of the input and its results. Arguments and results as compute_batch. On the GPU the species set grows
    with the elements read so far (all of them at once for a list or tuple); a new element sets up a new GPU batch.
    """
    disp = _resolve_dispersion(dispersion)
    parts = bool(dispersion_parts) and disp is not None
    m = model if isinstance(model, Model) else Model(model)
    device, kkm, _ = _select_device(device, m)
    if uq:
        if not m.has_uq:
            raise ValueError("the model has no UQ head (kokkos.npz exported without UQ artifacts)")
        m.validate_uq()
    if kkm is None:
        nth = int(num_threads) if num_threads else default_threads()
        yield from enumerate(_cpu_iter(m, structures, uq, energies, energy_only, nth, disp, parts))
        return
    kk = kkm[0]
    species = _species_of(structures) if isinstance(structures, (list, tuple)) else []
    chunk_arg = _chunk_arg(chunksize)
    i0 = 0
    for chunk in _portions(structures, max_atoms):
        species = sorted(set(species) | set(_species_of(chunk)))
        if sum(len(a) for a in chunk) == 0:
            out = None
        else:
            b = _gpu_batch(kk, m, species, chunk_arg, dispersion=disp)
            out = _gpu_compute(b, species, [len(a) for a in chunk], np.concatenate([a.numbers for a in chunk]),
                               np.concatenate([a.positions for a in chunk]).reshape(-1, 3),
                               np.stack([a.cell.array for a in chunk]), [[bool(p) for p in a.pbc] for a in chunk],
                               uq=uq, energies=energies, energy_only=energy_only, max_nall=max_nall,
                               dispersion_parts=parts)
        n0 = 0
        for k, a in enumerate(chunk):
            yield i0 + k, _split(out, a, k, n0, uq, energies, energy_only, disp is not None, parts)
            n0 += len(a)
        i0 += len(chunk)


def _species_of(structures) -> list:
    """sorted element symbols of the structures"""
    z = {int(n) for a in structures for n in np.unique(a.numbers)}
    return sorted(chemical_symbols[k] for k in z)


def _portions(structures, max_atoms):
    """consecutive lists of structures read lazily, at most max_atoms atoms each (a larger structure alone)"""
    chunk, acc = [], 0
    for a in structures:
        if chunk and acc + len(a) > max_atoms:
            yield chunk
            chunk, acc = [], 0
        chunk.append(a)
        acc += len(a)
    if chunk:
        yield chunk


def _gpu_batch(kk, m, species, chunk, keep=False, dispersion=None):
    """the cached GPU Batch of a Model (its spline) for a species set (sorted symbols) and a chunksize argument of the
    module; keep: one that keeps its device buffers between calls (MD loops, the TorchSim model); dispersion: a D3 set
    on the batch (part of the cache key, so batches with and without it never mix)"""
    dkey = None if dispersion is None else tuple(sorted(dispersion.params().items()))
    key = (m.path, m.spline, tuple(species), chunk, dkey, keep)
    with _KK_LOCK:
        if m.path not in _KK_MODELS:
            _KK_MODELS[m.path] = kk.Model(m.path)
        if key not in _KK_BATCHES:
            # ValueError: element the model lacks
            b = kk.Batch(_KK_MODELS[m.path], species, spline=m.spline, chunksize=chunk)
            if dispersion is not None:
                b.set_dispersion(dispersion.params(), [atomic_numbers[s] for s in species])
            b.keep_buffers = keep
            _KK_BATCHES[key] = b
            while len(_KK_BATCHES) > _KK_BATCHES_KEEP:
                _KK_BATCHES.popitem(last=False)    # device memory goes with the last reference
        _KK_BATCHES.move_to_end(key)
        return _KK_BATCHES[key]


def _gpu_compute(b, species, natoms, numbers, pos, cells, pbc, uq=False, energies=False, energy_only=False,
                 max_nall=1 << 21, dispersion_parts=False):
    """One call into the GPU module for structures given as arrays: natoms [S], numbers [N] and pos [N, 3] of all
    atoms (structure after structure), cells [S, 3, 3] (rows: cell vectors; zero vectors are padded), pbc [S, 3].
    Returns the module's output (energy [S], forces [N, 3], virial [S, 6] xx yy zz xy xz yz, ...)."""
    natoms = np.asarray(natoms, dtype=np.int32)
    pos = np.ascontiguousarray(pos, dtype=np.float64).reshape(-1, 3)
    types = np.ascontiguousarray(_type_lut(species)[np.asarray(numbers)], dtype=np.int32)
    cells = np.array(cells, dtype=np.float64).reshape(-1, 3, 3)
    starts = np.concatenate([[0], np.cumsum(natoms)])
    for k in np.flatnonzero(~np.any(cells != 0.0, axis=2).all(axis=1)):    # zero vectors: padded (open directions)
        cells[k] = _padded_cell_arrays(cells[k], pos[starts[k]:starts[k + 1]], b.cutoff)[0]
    pbc = np.ascontiguousarray(pbc, dtype=bool).reshape(-1, 3)
    return b.compute(natoms, types, pos, cells, pbc, energies=energies, uq=uq, energy_only=energy_only,
                     max_nall=max_nall, dispersion_parts=dispersion_parts)


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


def _split(out, at, k, n0, uq, energies, energy_only, disp=False, parts=False):
    """structure k of a batch result (out None: no atoms in the whole call, zeros) as _results gives it; disp: with
    the dispersion energy, parts: also its per-atom energies, forces and stress"""
    n = len(at)
    sl = slice(n0, n0 + n)
    if out is None:
        out = {"energy": np.zeros(k + 1), "forces": np.zeros((n0 + n, 3)), "virial": np.zeros((k + 1, 6)),
               "dispersion_energy": np.zeros(k + 1), "dispersion_forces": np.zeros((n0 + n, 3)),
               "dispersion_virial": np.zeros((k + 1, 6))}
        out.update({key: np.zeros(n0 + n) for key in ("energies", "gamma", "atomic_sigma", "gmm_cluster",
                                                      "dispersion_energies")})
    o = {"energy": out["energy"][k]}
    if not energy_only:
        o["forces"], o["virial"] = out["forces"][sl].copy(), out["virial"][k]
    if disp:
        o["dispersion_energy"] = out["dispersion_energy"][k]
    if parts:
        o["dispersion_energies"] = out["dispersion_energies"][sl].copy()
        if not energy_only:
            o["dispersion_forces"] = out["dispersion_forces"][sl].copy()
            o["dispersion_virial"] = out["dispersion_virial"][k]
    if energies:
        o["energies"] = out["energies"][sl].copy()
    if uq:
        o.update({key: out[key][sl].copy() for key in ("gamma", "atomic_sigma", "gmm_cluster")})
        o["total_sigma"] = o["atomic_sigma"].sum()
    return _results(at, o)


def _cpu_iter(m, structures, uq, energies, energy_only, nth, disp=None, parts=False):
    eng = _CpuEngine(m, num_threads=nth, dispersion=disp)
    for at in structures:
        if len(at) == 0:    # no kernel call; same zeros as the GPU path
            yield _split(None, at, 0, 0, uq, energies, energy_only, disp is not None, parts)
            continue
        r = eng.compute(at, forces=not energy_only, energies=energies, uq=uq)
        if not parts:    # the engine always gives the correction's own parts
            for key in ("dispersion_energies", "dispersion_forces", "dispersion_stress"):
                r.pop(key, None)
        yield r


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

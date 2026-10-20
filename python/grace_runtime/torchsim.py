# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""TorchSim model of grace-runtime: ``GRACERuntimeModel`` implements ``torch_sim.models.interface.ModelInterface``, so
``torch_sim.integrate`` / ``optimize`` / ``static`` and the autobatchers run on the fused GRACE kernels (CPU or GPU).

Needs ``torch`` and ``torch-sim-atomistic`` (>= 0.6, Python >= 3.12): ``pip install grace-runtime[torchsim]``.

A state on the model's GPU stays there: types and positions go to the GPU module as device tensors (DLPack), the
neighbour lists and periodic images are built on the GPU, energies / forces / virials come back as device tensors.
Other states (CPU model, a state on another device, cells with zero vectors) take the host path: positions, cells
and atomic numbers copied to the host, neighbour lists built there, one batch of the GPU module (or the CPU calculator,
system after system). Outputs are tensors on the model's device in its dtype. The GPU buffers stay allocated between calls (an MD loop;
``grace_runtime.free_gpu_memory()`` returns them). The GPU kernels compute in fp32 (sums in fp64), as
``GRACERuntimeCalculator``. Forces come from the kernels, not from autograd: the outputs carry no gradient, and states
with ``positions.requires_grad`` are refused.
"""
import os

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.stress import voigt_6_to_full_3x3_stress
from torch_sim.models.interface import ModelInterface

from . import registry
from ._core import Model
from .batch import _chunks, _cpu, _gpu_batch, _gpu_compute
from .calculator import _chunk_arg, _select_device, default_threads

_UQ_ATOM = ("gamma", "atomic_sigma", "gmm_cluster")


class GRACERuntimeModel(ModelInterface):
    """A GRACE model (kokkos.npz path, registry name or ``grace_runtime.Model``) as a TorchSim model.

    device: torch device or string; None: "cuda" when torch sees a GPU and grace-runtime can use it (as
        ``device="auto"`` of ``GRACERuntimeCalculator``), else "cpu". "cuda:N" runs the kernels on GPU N.
    dtype: dtype of the outputs (torch.float64 default; inputs of any float dtype are accepted).
    compute_forces / compute_stress: return "forces" [n_atoms, 3] / "stress" [n_systems, 3, 3] (eV/A^3, zero for
        cells with zero vectors); without both only energies are computed (no backward pass).
    energies: also per-atom energies "energies" [n_atoms].
    uq: the UQ head (uqv6 "gamma_only"): "gamma", "atomic_sigma", "gmm_cluster" [n_atoms] and "total_sigma"
        [n_systems].
    chunksize, max_nall, max_atoms: as ``compute_batch`` (GPU memory per engine pass; atoms per call into the GPU
        module, host path only). num_threads: host threads (host neighbour search, the CPU calculator).
    device_path: states on the model's GPU stay there (neighbour lists built on the GPU, inputs and outputs as
        device tensors); False, or cells with zero vectors (molecules), or a state on another device: the host path
        (copies to the host, host neighbour lists).

    Autobatching: ``memory_scales_with`` is "n_atoms"; ``max_memory_scaler(state)`` estimates the atoms per call that
    fit the free GPU memory, for ``BinningAutoBatcher`` / ``InFlightAutoBatcher(max_memory_scaler=...)`` (TorchSim's
    own estimate probes for "CUDA out of memory", which this model does not raise: it splits large batches itself).
    """

    def __init__(self, model, device=None, dtype=torch.float64, compute_forces: bool = True,
                 compute_stress: bool = True, energies: bool = False, uq: bool = False, chunksize="auto",
                 max_nall: int = 1 << 21, max_atoms: int = 131072, num_threads: int | None = None,
                 device_path: bool = True):
        super().__init__()
        m = model if isinstance(model, Model) else Model(registry.resolve(os.fspath(model)))
        if device is None:
            want = "auto" if torch.cuda.is_available() else "cpu"
        else:
            dev = torch.device(device)
            if dev.type not in ("cpu", "cuda"):
                raise ValueError(f"device must be cpu or cuda, not {dev}")
            want = "cpu" if dev.type == "cpu" else f"cuda:{torch.cuda.current_device() if dev.index is None else dev.index}"
        resolved, kkm, self.backend_info = _select_device(want, m)    # as GRACERuntimeCalculator.backend_info
        if kkm is not None:
            idx = int(resolved.split(":")[1]) if ":" in resolved else kkm[0].device()
            self._device = torch.device("cuda", idx)
        else:
            self._device = torch.device("cpu")
        if uq:
            if not m.has_uq:
                raise ValueError("the model has no UQ head (kokkos.npz exported without UQ artifacts)")
            m.validate_uq()
        self._model, self._kk = m, (kkm[0] if kkm is not None else None)
        self._dtype = dtype
        self._compute_forces, self._compute_stress = bool(compute_forces), bool(compute_stress)
        self._energies, self._uq = bool(energies), bool(uq)
        self._chunk = _chunk_arg(chunksize)
        self._max_nall, self._max_atoms = int(max_nall), int(max_atoms)
        self._nth = int(num_threads) if num_threads else default_threads()
        self._memory_scales_with = "n_atoms"
        # GPU states on the model's device: neighbour lists on the GPU, no host copies (False: the host path)
        self._device_path = bool(device_path)
        self._lut, self._lut_key = None, None

    @property
    def model(self) -> Model:
        """the grace_runtime.Model"""
        return self._model

    def forward(self, state, **kwargs) -> dict[str, torch.Tensor]:
        if state.positions.requires_grad:
            raise RuntimeError("GRACERuntimeModel computes forces in its own kernels and has no autograd graph; "
                               "pass positions without requires_grad")
        cells_t = state.row_vector_cell.detach()
        nsys = cells_t.shape[0]
        eonly = not (self._compute_forces or self._compute_stress)
        if (self._kk is not None and self._device_path and state.positions.device == self._device
                and len(state.positions) > 0):
            cells = cells_t.to("cpu", torch.float64).numpy()
            if all(np.any(c[k]) for c in cells for k in range(3)):    # zero cell vectors: the host path pads them
                return self._gpu_device(state, cells_t, cells, eonly)
        natoms = torch.bincount(state.system_idx, minlength=nsys).cpu().numpy().astype(np.int32)
        numbers = state.atomic_numbers.detach().cpu().numpy()
        pos = state.positions.detach().to("cpu", torch.float64).numpy()
        cells = cells_t.to("cpu", torch.float64).numpy()
        pbc = np.broadcast_to(np.asarray(state.pbc.detach().cpu().numpy(), dtype=bool).reshape(-1), (nsys, 3))
        if self._kk is None:
            out = self._cpu(natoms, numbers, pos, cells, pbc, eonly)
        elif len(numbers) == 0:
            out = {"energy": np.zeros(nsys), "forces": np.zeros((0, 3)), "virial": np.zeros((nsys, 6))}
        else:
            out = self._gpu(natoms, numbers, pos, cells, pbc, eonly)
        return self._outputs(out, natoms, cells, eonly)

    def max_memory_scaler(self, state, fraction: float = 0.7) -> float:
        """Atoms per call whose GPU memory fits `fraction` of the free device memory, measured on one call with
        `state` (systems like the ones to run, repeated to at least 2048 atoms), at most max_atoms. Larger batches
        still run (the engine splits them into passes): the estimate sets the batch size, not a hard limit. CPU:
        max_atoms. Memory held by torch's caching allocator is not free for this model (torch.cuda.empty_cache())."""
        import torch_sim as ts
        if self._kk is None:
            return float(self._max_atoms)
        probe = ts.concatenate_states([state] * max(1, -(-2048 // state.n_atoms)))
        self.forward(probe)    # model, engine and tables set up outside the measured call
        from .batch import _KK_BATCHES
        for key, b in list(_KK_BATCHES.items()):    # kept buffers back: the measured call allocates a full step
            if key[0] == self._model.path and key[-1]:
                b.release_buffers()
        self._kk._debug_track_alloc(True)
        try:
            self.forward(probe)
            peak = self._kk._debug_peak_alloc()
        finally:
            self._kk._debug_track_alloc(False)
        per_atom = max(peak, 1) / probe.n_atoms
        return float(max(1, min(self._max_atoms, int(fraction * self._kk.memory_info()[0] / per_atom))))

    def _gpu_device(self, state, cells_t, cells, eonly):
        """one call of Batch.compute_device: positions, types and outputs stay on the GPU (neighbour lists built there)"""
        dev = self._device
        nsys = len(cells)
        natoms = torch.bincount(state.system_idx, minlength=nsys).cpu().numpy().astype(np.int32)
        z = state.atomic_numbers.detach()
        zs = torch.unique(z).cpu().numpy()
        species = sorted(chemical_symbols[k] for k in zs)
        b = _gpu_batch(self._kk, self._model, species, self._chunk, keep=True)
        key = tuple(species)
        if self._lut_key != key:    # atomic number -> type 1..len(species), on the device
            lut = torch.zeros(int(max(atomic_numbers[e] for e in species)) + 1, dtype=torch.int32, device=dev)
            for t, e in enumerate(species):
                lut[atomic_numbers[e]] = t + 1
            self._lut, self._lut_key = lut, key
        types = self._lut[z].contiguous()
        pos = state.positions.detach().to(torch.float64).contiguous()
        pbc = np.ascontiguousarray(np.broadcast_to(
            np.asarray(state.pbc.detach().cpu().numpy(), dtype=bool).reshape(-1), (nsys, 3)))
        n = len(pos)
        f64 = dict(dtype=torch.float64, device=dev)
        out = {"energy": torch.empty(nsys, **f64)}
        kw = {}
        if not eonly:
            out["forces"] = kw["forces"] = torch.empty((n, 3), **f64)
            out["virial"] = kw["virial"] = torch.empty((nsys, 6), **f64)
        if self._energies:
            out["energies"] = kw["energies"] = torch.empty(n, **f64)
        if self._uq:
            for k in _UQ_ATOM:
                out[k] = kw[k] = torch.empty(n, **f64)
        torch.cuda.synchronize(dev)    # positions / types written by torch before the module reads them
        b.compute_device(natoms, types, pos, np.ascontiguousarray(cells), pbc, out["energy"], energy_only=eonly,
                         max_nall=self._max_nall, **kw)
        r = {"energy": out["energy"].to(self._dtype)}
        if not eonly:
            if self._compute_forces:
                r["forces"] = out["forces"].to(self._dtype)
            if self._compute_stress:
                w = out["virial"]
                full = torch.stack([w[:, [0, 3, 4]], w[:, [3, 1, 5]], w[:, [4, 5, 2]]], dim=1)
                vol = torch.linalg.det(cells_t.to(torch.float64)).abs()
                r["stress"] = (-full / vol[:, None, None]).to(self._dtype)
        if self._energies:
            r["energies"] = out["energies"].to(self._dtype)
        if self._uq:
            for k in _UQ_ATOM:
                r[k] = out[k].to(torch.int64 if k == "gmm_cluster" else self._dtype)
            r["total_sigma"] = torch.zeros(nsys, **f64).index_add_(0, state.system_idx, out["atomic_sigma"]).to(
                self._dtype)
        return r

    def _gpu(self, natoms, numbers, pos, cells, pbc, eonly):
        """calls into the GPU module, at most max_atoms atoms each (a larger system alone), outputs concatenated"""
        species = sorted(chemical_symbols[k] for k in np.unique(numbers))
        b = _gpu_batch(self._kk, self._model, species, self._chunk, keep=True)    # buffers kept across steps
        starts = np.concatenate([[0], np.cumsum(natoms)])
        parts = []
        for lo, hi in _chunks(natoms, self._max_atoms):
            a0, a1 = starts[lo], starts[hi]
            parts.append(_gpu_compute(b, species, natoms[lo:hi], numbers[a0:a1], pos[a0:a1], cells[lo:hi],
                                      pbc[lo:hi], uq=self._uq, energies=self._energies, energy_only=eonly,
                                      num_threads=self._nth, max_nall=self._max_nall))
        if len(parts) == 1:
            return dict(parts[0])
        return {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in parts[0]}

    def _cpu(self, natoms, numbers, pos, cells, pbc, eonly):
        """the calculator loop of compute_batch(device="cpu") -> arrays as the GPU module returns them"""
        ends = np.cumsum(natoms)
        structures = [Atoms(numbers=numbers[e - n:e], positions=pos[e - n:e], cell=c, pbc=p)
                      for n, e, c, p in zip(natoms, ends, cells, pbc)]
        res = _cpu(self._model, structures, self._uq, self._energies, eonly, self._nth)
        out = {"energy": np.array([r["energy"] for r in res])}
        if not eonly:
            out["forces"] = np.concatenate([r["forces"] for r in res]).reshape(-1, 3)
            out["stress33"] = np.stack([voigt_6_to_full_3x3_stress(r["stress"]) for r in res])
        if self._energies:
            out["energies"] = np.concatenate([r["energies"] for r in res])
        if self._uq:
            for key in _UQ_ATOM:
                out[key] = np.concatenate([r[key] for r in res])
        return out

    def _outputs(self, out, natoms, cells, eonly):
        def t(a, dtype=None):
            return torch.as_tensor(np.asarray(a), device=self._device, dtype=dtype or self._dtype)

        r = {"energy": t(out["energy"])}
        if not eonly:
            if self._compute_forces:
                r["forces"] = t(out["forces"])
            if self._compute_stress:
                r["stress"] = t(out["stress33"] if "stress33" in out else _stress(out["virial"], cells))
        if self._energies:
            r["energies"] = t(out["energies"])
        if self._uq:
            for key in _UQ_ATOM:
                r[key] = t(out[key], torch.int64 if key == "gmm_cluster" else None)
            sig = np.asarray(out["atomic_sigma"], dtype=np.float64)
            r["total_sigma"] = t(np.bincount(np.repeat(np.arange(len(natoms)), natoms), weights=sig,
                                             minlength=len(natoms)))
        return r


def _stress(virial, cells):
    """-virial / volume as [S, 3, 3] (virial rows xx yy zz xy xz yz); zero for cells with zero vectors"""
    w = np.asarray(virial, dtype=np.float64).reshape(-1, 6)
    full = np.stack([w[:, [0, 3, 4]], w[:, [3, 1, 5]], w[:, [4, 5, 2]]], axis=1)
    rank3 = np.array([all(np.any(c[k]) for k in range(3)) for c in cells])
    vol = np.where(rank3, np.abs(np.linalg.det(cells)) if len(cells) else 1.0, 1.0)
    return np.where(rank3[:, None, None], -full / vol[:, None, None], 0.0)

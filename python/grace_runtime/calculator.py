# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""ASE calculator over grace_runtime._core (CPU fused GRACE-1L/2L/3L kernels of libgrace) or, on a GPU (device "cuda",
"rocm", "gpu" or "auto"), the GPU module _kk_<arch> (fused Kokkos kernels, neighbour list on the GPU)."""
import os
import threading
import weakref
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers, chemical_symbols

from . import _core, _gpu
from ._core import Model
from .dispersion import resolve as _resolve_dispersion
from .neighbors import NeighborList, _padded_cell, check_skin

_KK_MODELS = {}    # path -> GPU model (weights loaded once per process)
_KK_LOCK = threading.Lock()    # guards _KK_MODELS and grace_runtime.batch's Batch cache
_KK_CALCS = weakref.WeakSet()    # live GPU calculators (grace_runtime.free_gpu_memory() returns their buffers)


def physical_cores(cpus, topology_root="/sys/devices/system/cpu"):
    """Distinct (package, core) pairs among the given logical CPUs; None when the topology is unreadable."""
    root = Path(topology_root)
    try:
        return len({((root / f"cpu{c}/topology/physical_package_id").read_text().strip(),
                     (root / f"cpu{c}/topology/core_id").read_text().strip()) for c in cpus})
    except OSError:
        return None


def _chunk_arg(chunksize) -> int:
    """chunksize argument -> the GPU module's: -1 auto, else atoms per pass (0: one pass)"""
    if chunksize == "auto":
        return -1
    if isinstance(chunksize, (int, np.integer)) and not isinstance(chunksize, bool) and chunksize >= 0:
        return int(chunksize)
    raise ValueError(f"chunksize must be 'auto' or an integer >= 0 (0: one pass), not {chunksize!r}")


def _cpu_info(fallback=None) -> dict:
    return {"device": "cpu", "platform": "cpu", "arch": None, "cc": None, "native": None, "fallback": fallback}


def _cpu_fallback(why: str):
    """device="auto" cannot use the GPU that is there: the CPU, announced by _gpu.notice (stderr banner and
    RuntimeWarning); GRACE_RUNTIME_STRICT=1: a RuntimeError instead"""
    if _gpu.strict():
        raise RuntimeError(f"GRACE_RUNTIME_STRICT=1: device='auto' would run on the CPU: {why}")
    _gpu.notice(f"device='auto': {why}; running on the CPU, much slower than the GPU", stacklevel=5)
    return "cpu", None, _cpu_info(why)


def _select_device(device: str, model):
    """(resolved device, (GPU module, its model) or None, backend info) for a CPU Model. device="auto" falls back to the
    CPU (announced by _gpu.notice; an error with GRACE_RUNTIME_STRICT=1) when a GPU is there but has no module, the
    module predates the GRACE-3L engine, the fused kernels refuse the model or the GPU fails to start / set up the model
    (e.g. its memory held by another process; not remembered); an explicit GPU device ("cuda", "rocm", "gpu") raises.
    Backend info: {"device", "platform", "arch", "cc", "native", "fallback"} (fallback: None, "ptx" or the reason of a
    CPU fallback)."""
    path = model.path
    auto = device == "auto"
    if auto:
        device, why = _gpu.auto_device()
        if why:
            return _cpu_fallback(why)
        if device == "cpu" and _gpu.strict() and _gpu.env_device() != "cpu":    # strict: a native GPU or an error
            raise RuntimeError("GRACE_RUNTIME_STRICT=1: device='auto' found no usable GPU; pass device='cpu' to run "
                               "on the CPU")
    if device == "cpu":
        return "cpu", None, _cpu_info()
    try:
        kk = _gpu.ensure_init(device)    # ValueError for a bad device string, ImportError without a module
        with _KK_LOCK:
            if path not in _KK_MODELS:
                try:
                    _KK_MODELS[path] = kk.Model(path)
                except ValueError as e:    # a GPU module built before the GRACE-3L engine
                    if not auto or "GRACE-3L" not in str(e):
                        raise
                    return _cpu_fallback(str(e))
            kk_model = _KK_MODELS[path]
        if auto:
            why = _gpu.unsupported(kk, kk_model, path, model.spline)
            if why:
                return _cpu_fallback(why)
    except RuntimeError as e:
        if not auto or _gpu.strict():
            raise
        return _cpu_fallback(f"the GPU cannot be used ({e})")
    info = dict(_gpu.info())
    info["fallback"] = None if info["native"] else "ptx"
    if device.startswith(("gpu", "hip")):    # aliases: the device they resolved to
        device = info["device"]
    return device, (kk, kk_model), info


def _type_lut(species) -> np.ndarray:
    """atomic number -> type 1 .. len(species) of the kernels (species: sorted symbols)"""
    lut = np.zeros(max((atomic_numbers[s] for s in species), default=0) + 1, np.int32)
    for t, s in enumerate(species):
        lut[atomic_numbers[s]] = t + 1
    return lut


def _virial_to_stress(w, vol):
    """ASE Voigt stress (eV/A^3) from the kernels' virial (xx yy zz xy xz yz, LAMMPS sum) and the cell volume"""
    return -np.asarray(w)[..., [0, 1, 2, 5, 4, 3]] / np.asarray(vol)[..., None]


def _results(at, out, perm=None):
    """one structure's kernel output as ASE results in the caller's atom order (perm: the order of the CPU neighbour
    list; None: already the caller's); forces and stress absent after an energy-only evaluation. The dispersion's own
    share (dispersion_energy, _energies, _forces, _virial -> _stress), if present, is in the caller's order."""
    n = len(at)

    def caller_order(v, shape):
        if perm is None:
            return np.asarray(v).reshape(shape)
        r = np.empty(shape)
        r[perm] = v
        return r
    res = {"energy": float(out["energy"])}
    if "forces" in out:
        res["forces"] = caller_order(out["forces"], (n, 3))
    if "virial" in out:    # asymmetry is round-off
        res["stress"] = _virial_to_stress(out["virial"], at.get_volume()) if at.cell.rank == 3 else np.zeros(6)
    for k in ("energies", "gamma", "atomic_sigma", "gmm_cluster"):
        if k in out:
            res[k] = caller_order(out[k], (n,))
    if "total_sigma" in out:
        res["total_sigma"] = float(out["total_sigma"])
    if "dispersion_energy" in out:
        res["dispersion_energy"] = float(out["dispersion_energy"])
    for k, shape in (("dispersion_energies", (n,)), ("dispersion_forces", (n, 3))):
        if k in out:
            res[k] = np.asarray(out[k], dtype=np.float64).reshape(shape)
    if "dispersion_virial" in out:
        res["dispersion_stress"] = (_virial_to_stress(out["dispersion_virial"], at.get_volume()) if at.cell.rank == 3
                                    else np.zeros(6))
    return res


class _CpuEngine:
    """The CPU evaluation of one independently changing structure (an ase.Atoms read as data): species tables and a
    neighbour list (rebuilt when an atom moved more than skin / 2 or the cell, pbc or atoms change) for the fused CPU
    kernels of a Model. GRACERuntimeCalculator on the CPU, compute_batch, relax_batch and the TorchSim model use it.
    dispersion: a resolved correction (grace_runtime.D3) added to the totals and returned on its own as well."""

    def __init__(self, model, skin: float = 0.5, num_threads: int = 0, dispersion=None):
        self.model, self.skin, self.num_threads, self.dispersion = model, float(skin), num_threads, dispersion
        self.nl = self._z = self._species = self._lut = self._types = None    # set for the first species set

    def compute(self, at, forces: bool = True, energies: bool = False, uq: bool = False) -> dict:
        """energy; forces and stress unless forces=False (no backward pass); per-atom energies; with uq gamma,
        atomic_sigma, gmm_cluster and total_sigma (the Model must have a usable UQ head)"""
        z = np.unique(at.numbers)
        if self._z is None or not np.array_equal(z, self._z):
            species = sorted(chemical_symbols[k] for k in z)
            rc = float(self.model.cutoff_matrix(species).max()) if species else 0.0    # raises for unknown elements
            self.nl = NeighborList(rc, self.skin, num_threads=self.num_threads)
            self._species, self._lut, self._z = species, _type_lut(species), z
        nl = self.nl
        if nl.update(at):
            self._types = self._lut[at.numbers][nl.perm]
        out = self.model.compute(self._species, self._types, nl.offset, nl.j, nl.rij(at, self.num_threads),
                                 energies=energies, num_threads=self.num_threads, uq=uq, energy_only=not forces)
        res = _results(at, out, nl.perm)
        if self.dispersion is not None:
            res = self._add_dispersion(at, res, forces)
        return res

    def _add_dispersion(self, at, res, forces):
        """the correction (caller's order) added to the model's results and stored on its own"""
        cell = _padded_cell(at, self.dispersion.reach + 1.0)[0]
        d = _core.dispersion_compute(self.dispersion.params(), np.ascontiguousarray(at.numbers, dtype=np.int32),
                                     np.ascontiguousarray(at.positions, dtype=np.float64),
                                     np.ascontiguousarray(cell, dtype=np.float64), [bool(p) for p in at.pbc],
                                     energy_only=not forces, num_threads=self.num_threads)
        part = _results(at, {"energy": 0.0, **{"dispersion_" + k: v for k, v in d.items()}})
        res["energy"] += part["dispersion_energy"]
        if "energies" in res:
            res["energies"] = res["energies"] + part["dispersion_energies"]
        if forces:
            res["forces"] = res["forces"] + part["dispersion_forces"]
            res["stress"] = res["stress"] + part["dispersion_stress"]
        part.pop("energy")
        res.update(part)
        return res


def default_threads() -> int:
    """OMP_NUM_THREADS, else the physical cores among the CPUs this process may use."""
    env = os.environ.get("OMP_NUM_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    try:
        cpus = os.sched_getaffinity(0)
    except AttributeError:
        cpus = set(range(os.cpu_count() or 1))
    return max(1, physical_cores(cpus) or len(cpus))


class GRACERuntimeCalculator(Calculator):
    """GRACE-1L/2L/3L on CPU or GPU from a kokkos.npz.

    model: path to a kokkos.npz, a registry name or a grace_runtime.Model (one Model can serve several calculators
    and threads).
    skin: Verlet skin of the neighbour list (A); the list is rebuilt when an atom moved more than skin / 2.
    num_threads: OpenMP threads per call; default OMP_NUM_THREADS or the physical cores of the affinity mask.
    neighbor_backend: "native" (default and only value: the OpenMP cell list of grace_runtime._core).
    uq: evaluate the model's uncertainty head (uqv6 "gamma_only", as TPCalculator.enable_uq("gamma_only")): results
        gamma, atomic_sigma, gmm_cluster (per atom, float64) and total_sigma. Off by default; see enable_uq().
    device: "auto" (default), "cpu", "cuda" / "cuda:N" (NVIDIA), "rocm" / "rocm:N" (AMD; "hip" / "hip:N" too) or
        "gpu" / "gpu:N" (whichever GPU is usable: NVIDIA, then AMD): the GPU kernels (as LAMMPS
        grace/{1l,2l,3l}/fused/kk: the network in fp32, bond vectors and lengths in fp64 (GRACE-3L also the cutoff
        envelope), sums in fp64) with the neighbour list on the GPU; one GPU per process; num_threads and
        neighbor_backend do not apply. "auto" (as TensorFlow): the GPU (NVIDIA, then AMD) when a module for it is
        installed and the fused kernels take the model, else the CPU (a RuntimeWarning says why when a GPU is there);
        the environment variable GRACE_RUNTIME_DEVICE sets the choice of "auto". An explicit GPU device raises instead
        of falling back.
    chunksize: atoms per GPU engine pass ("auto", default: chosen by the engine from the free GPU memory, as LAMMPS
        chunksize auto - all atoms in one pass while they fit; after an allocation failure half of it, down to 1024; a
        number: fixed; 0: one pass). Larger is faster (2L: up to ~6 % for 100k atoms), smaller needs less GPU memory.
        CPU: no effect.
    energy_only: when ASE asks for no forces / stress (get_potential_energy, get_potential_energies, UQ keys), skip the
        backward pass (as LAMMPS eflag_only); forces asked later are computed then. Default True.
    dispersion: a dispersion correction added to the model (default None: none): a grace_runtime.D3 object or a short
        string ("d3-pbe", "d3-r2scan"); see grace_runtime.dispersion. energy, forces, stress and energies are then the
        sums of model and correction; the correction alone is in dispersion_energy, dispersion_forces,
        dispersion_stress and dispersion_energies (calc.get_property(...); stored with every evaluation of the
        totals, the forces and stress with one that has forces). UQ results are the model's.
    """

    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress",
                              "gamma", "atomic_sigma", "gmm_cluster", "total_sigma",
                              "dispersion_energy", "dispersion_forces", "dispersion_stress", "dispersion_energies"]

    def __init__(self, model, skin: float = 0.5, num_threads: int | None = None, neighbor_backend: str = "native",
                 uq: bool = False, device: str = "auto", energy_only: bool = True, chunksize="auto", dispersion=None,
                 **kwargs):
        if neighbor_backend != "native":
            raise ValueError(f"neighbour list backend {neighbor_backend!r}: only 'native' "
                             "(the default; matscipy was removed)")
        super().__init__(**kwargs)
        self.neighbor_backend = neighbor_backend
        self.energy_only = bool(energy_only)
        self.dispersion = _resolve_dispersion(dispersion)
        self._chunk = _chunk_arg(chunksize)
        self._kk = self._kk_system = None
        self.backend = "cpu"    # where calculations run: "cpu", "cuda:N (smXY)" or "rocm:N (gfxXYZ)"
        skin = check_skin(skin)
        self.model = model if isinstance(model, Model) else Model(model)
        # backend_info: {"device", "platform", "arch", "cc", "native", "fallback"} - check "fallback" is None before
        # timing
        self.device, kk, self.backend_info = _select_device(device, self.model)
        if kk is not None:
            self._kk, self._kk_model = kk
            i = self.backend_info
            self.backend = (f"{i['platform']}:{self._kk.device()} ({self._kk.arch()})" if i["native"] else
                            f"cuda:{self._kk.device()} ({self._kk.arch()} PTX on sm{i['cc'].replace('.', '')}, JIT)")
        self.skin = skin
        self.num_threads = int(num_threads) if num_threads else default_threads()
        self._z = None    # species set of the GPU System
        self._cpu = None    # the CPU engine (CPU path)
        self.uq_enabled = False
        if uq:
            self.enable_uq()

    @property
    def chunksize(self) -> int:
        """atoms per GPU engine pass in use (auto: may have shrunk to fit the GPU memory)"""
        if self._kk_system is not None:
            return self._kk_system.chunksize
        return 0 if self._chunk < 0 else self._chunk

    def free_gpu_memory(self):
        """Return this calculator's per-atom GPU buffers (neighbour list, outputs, engine work buffers) now; weights and
        tables stay and the next calculation grows them again. CPU: no effect."""
        if self._kk_system is not None:
            self._kk_system.release_buffers()

    def enable_uq(self, mode: str | None = "gamma_only"):
        """Evaluate the UQ head from the next calculation on (only mode "gamma_only"); ValueError without a usable head."""
        if mode not in (None, "gamma_only"):
            raise ValueError(f"UQ mode {mode!r} is not supported by grace-runtime; only 'gamma_only'")
        if not self.model.has_uq:
            raise ValueError("the model has no UQ head (kokkos.npz exported without UQ artifacts)")
        self.model.validate_uq()
        self.uq_enabled = True
        self.reset()

    def disable_uq(self):
        self.uq_enabled = False
        self.reset()

    def _gpu_setup(self, atoms):
        """the GPU System of the atoms' species set, new when it changes; the GPU module checks the elements and keeps
        its own neighbour list; may raise (engine self-test, device memory)"""
        z = np.unique(atoms.numbers)
        if self._z is None or not np.array_equal(z, self._z):
            species = sorted(chemical_symbols[k] for k in z)
            system = self._kk.System(self._kk_model, species, skin=self.skin, spline=self.model.spline,
                                     chunksize=self._chunk)
            if self.dispersion is not None:
                system.set_dispersion(self.dispersion.params(), [atomic_numbers[s] for s in species])
            # the calculator changes only after everything is built
            self._z, self._lut = z, _type_lut(species)
            self._kk_system, self._kk_cellkey = system, None
            with _KK_LOCK:
                _KK_CALCS.add(self)

    def _calculate_gpu(self, at, properties, eonly):
        """GPU path: positions, cell and pbc to the device module; it keeps the neighbour list and its skin."""
        key = (at.cell.array.tobytes(), np.asarray(at.pbc).tobytes())
        if key != self._kk_cellkey:    # a full-rank cell for the search (zero vectors padded), fixed until cell / pbc change
            self._kk_cell = np.ascontiguousarray(_padded_cell(at, self._kk_system.cutoff + self.skin)[0], dtype=np.float64)
            self._kk_cellkey = key
        types = np.ascontiguousarray(self._lut[at.numbers], dtype=np.int32)
        return self._kk_system.compute(types, np.ascontiguousarray(at.positions, dtype=np.float64), self._kk_cell,
                                       [bool(p) for p in at.pbc], energies="energies" in properties, uq=self.uq_enabled,
                                       energy_only=eonly)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if system_changes:    # results of other atoms must not survive a call that does not overwrite them all
            self.results = {}
        at = self.atoms
        eonly = self.energy_only and not {"forces", "stress", "dispersion_forces", "dispersion_stress"} & set(properties)
        if self._kk is not None:
            self._gpu_setup(at)
            res = _results(at, self._calculate_gpu(at, properties, eonly))
        else:
            if self._cpu is None:
                self._cpu = _CpuEngine(self.model, self.skin, self.num_threads, self.dispersion)
            self._cpu.num_threads = self.num_threads    # the kernels' threads as set now (the neighbour list keeps its own)
            res = self._cpu.compute(at, forces=not eonly, energies="energies" in properties, uq=self.uq_enabled)
        res["free_energy"] = res["energy"]
        self.results.update(res)

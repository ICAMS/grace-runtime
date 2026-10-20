# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Verlet neighbour list of grace_runtime.

Backend ``"native"`` (default): the OpenMP cell list of ``grace_runtime._core``; backend ``"matscipy"``: matscipy
``ijS`` at ``cutoff + skin`` (optional dependency). The list is kept until an atom has moved more than ``skin / 2`` since the
last build (or the cell, pbc or atomic numbers change). Atoms are put in spatial-bin order: the 2L driver distributes atoms over threads
with a static schedule and relies on neighbours of one thread's atoms being close in index.

Cells with zero vectors (molecules, ``cell.rank < 3``) get padded vectors orthogonal to the others, long enough that no
image is within range; positions are shifted into that box for matscipy only, bond vectors use the true positions.
"""
import numpy as np
from ._core import bond_vectors, neighbor_search

BACKENDS = ("matscipy", "native")


def _padded_cell(atoms, reach):
    """Cell for the neighbour search and the shift that puts the atoms inside it."""
    return _padded_cell_arrays(atoms.cell.array, atoms.positions, reach)


def _padded_cell_arrays(cell, pos, reach):
    """_padded_cell for a cell (rows: vectors) and positions [n, 3]."""
    cell = np.array(cell, dtype=np.float64)
    if all(np.any(cell[k]) for k in range(3)):
        return cell, np.zeros(3)
    zero = [k for k in range(3) if not np.any(cell[k])]
    given = np.array([cell[k] for k in range(3) if k not in zero]).reshape(-1, 3)
    # orthonormal basis of the complement of span(given): the last right-singular vectors
    dirs = np.linalg.svd(given)[2][len(given):] if len(given) else np.eye(3)
    shift = np.zeros(3)
    for k, d in zip(zero, dirs):
        proj = pos @ d
        cell[k] = d * (np.ptp(proj) + 2.0 * reach + 1.0) if len(pos) else d * (2.0 * reach + 1.0)
        if len(pos):
            shift += d * (reach + 0.5 - proj.min())
    return cell, shift


def matscipy_usable():
    """matscipy is installed and >= 1.3 (1.2 misses bonds with mixed pbc and atoms outside the cell)."""
    try:
        import matscipy
    except ImportError:
        return False
    try:
        major, minor = (int(x) for x in matscipy.__version__.split(".")[:2])
    except ValueError:      # a development version string: trust it
        return True
    return (major, minor) >= (1, 3)


class NeighborList:
    def __init__(self, cutoff: float, skin: float = 0.5, backend: str = "native", num_threads: int = 0):
        if backend not in BACKENDS:
            raise ValueError(f"unknown neighbour list backend {backend!r}; use one of {BACKENDS}")
        if backend == "matscipy":
            try:
                import matscipy.neighbours  # noqa: F401
            except ImportError as e:
                raise ImportError("neighbour list backend 'matscipy' needs matscipy (pip install matscipy)") from e
            if not matscipy_usable():
                raise ImportError("neighbour list backend 'matscipy' needs matscipy >= 1.3 (1.2 misses bonds with "
                                  "mixed pbc and atoms outside the cell); use backend='native'")
        self.cutoff, self.skin = float(cutoff), float(skin)
        self.backend, self.num_threads = backend, int(num_threads)
        self.nbuild = 0
        self._x0 = None
        self._key = None

    @staticmethod
    def _system_key(atoms):
        return (atoms.cell.array.tobytes(), np.asarray(atoms.pbc).tobytes(), atoms.numbers.tobytes())

    def update(self, atoms) -> bool:
        """Rebuild when needed; True when the list was rebuilt."""
        key = self._system_key(atoms)
        if self._x0 is not None and key == self._key and len(atoms) == len(self._x0):
            if len(atoms) == 0 or np.linalg.norm(atoms.positions - self._x0, axis=1).max() <= 0.5 * self.skin:
                return False
        self._build(atoms)
        self._key = key
        self._x0 = atoms.positions.copy()
        self.nbuild += 1
        return True

    def _build(self, atoms):
        n = len(atoms)
        reach = self.cutoff + self.skin
        cell, shift = _padded_cell(atoms, reach)
        self._cell = cell
        pos = atoms.positions + shift
        if self.backend == "native":
            self.perm, self.offset, self.j, self._io, self._jo, self._shift = neighbor_search(
                np.ascontiguousarray(pos, dtype=np.float64), np.ascontiguousarray(cell, dtype=np.float64),
                [bool(p) for p in atoms.pbc], reach, num_threads=self.num_threads)
            self.inv = np.empty(n, np.int64)
            self.inv[self.perm] = np.arange(n)
            return
        from matscipy.neighbours import neighbour_list    # optional dependency, matscipy backend only

        # spatial bins of edge ~ cutoff / 2 in scaled coordinates (wrapped where periodic)
        frac = np.linalg.solve(cell.T, pos.T).T if n else np.zeros((0, 3))
        frac = np.where(atoms.pbc, np.mod(frac, 1.0), np.clip(frac, 0.0, 1.0 - 1e-12))
        nbin = np.maximum(1, (np.linalg.norm(cell, axis=1) / (0.5 * self.cutoff)).astype(np.int64))
        b = np.minimum((frac * nbin).astype(np.int64), nbin - 1)
        key = (b[:, 0] * nbin[1] + b[:, 1]) * nbin[2] + b[:, 2]
        self.perm = np.argsort(key, kind="stable").astype(np.int64)
        self.inv = np.empty(n, np.int64)
        self.inv[self.perm] = np.arange(n)
        if n:
            i, j, S = neighbour_list("ijS", positions=pos, cell=cell, pbc=atoms.pbc, numbers=atoms.numbers.astype(np.int32),
                                     cutoff=reach)
        else:
            i = j = np.zeros(0, np.int64)
            S = np.zeros((0, 3), np.int64)
        i, j = self.inv[i], self.inv[j]
        o = np.argsort(i, kind="stable")    # CSR rows; the kernels do not need j sorted within a row
        self._i = np.ascontiguousarray(i[o], dtype=np.int64)
        self.j = np.ascontiguousarray(j[o], dtype=np.int32)
        S = S[o]
        self.offset = np.searchsorted(self._i, np.arange(n + 1)).astype(np.int64)
        self._io, self._jo = self.perm[self._i], self.perm[self.j].astype(np.int32)   # original atom indices
        # image shifts S @ cell (cell changes force a rebuild, so they stay valid); broadcast, as matmul with K=3 is slow
        self._shift = np.ascontiguousarray(S[:, 0:1] * cell[0] + S[:, 1:2] * cell[1] + S[:, 2:3] * cell[2])

    def rij(self, atoms, num_threads: int = 0):
        """Bond vectors x_j - x_i + S cell for the current positions (new atom order), [nbonds, 3]."""
        return bond_vectors(np.ascontiguousarray(atoms.positions, dtype=np.float64), self._io, self._jo, self._shift,
                            num_threads=num_threads)

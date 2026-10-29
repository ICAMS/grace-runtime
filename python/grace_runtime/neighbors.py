# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Verlet neighbour list of grace_runtime.

The OpenMP cell list of ``grace_runtime._core`` at ``cutoff + skin``. The list is kept until an atom has moved more
than ``skin / 2`` since the last build (or the cell, pbc or atomic numbers change). Atoms are put in spatial-bin order:
the 2L driver distributes atoms over threads with a static schedule and relies on neighbours of one thread's atoms
being close in index.

Zero cell vectors (molecules, ``cell.rank < 3``) and the vectors of open directions get padded vectors orthogonal to the
periodic ones, long enough that no image is within range; positions are shifted into that box for the search only, bond
vectors use the true positions.

Per bond the list keeps the neighbour ``j`` (new order) and ``image``, its row of ``shifts`` (the distinct S cell): 8
bytes per bond.
"""
import numpy as np
from ._core import bond_vectors, neighbor_search


def _padded_cell(atoms, reach):
    """Cell for the neighbour search and the shift that puts the atoms inside it."""
    return _padded_cell_arrays(atoms.cell.array, atoms.positions, reach, atoms.pbc)


def _padded_cell_arrays(cell, pos, reach, pbc=None):
    """_padded_cell for a cell (rows: vectors) and positions [n, 3]. Zero vectors and, with pbc, the vectors of the
    open directions are replaced by directions normal to the periodic vectors: the bonds do not depend on an open
    vector, and one nearly in the periodic plane made the bins of the periodic directions about as thin as its normal
    component (a search over ~1e10 bins for c = (5, 5, 1e-4))."""
    cell = np.array(cell, dtype=np.float64)
    pad = [k for k in range(3) if not np.any(cell[k]) or (pbc is not None and not pbc[k])]
    if not pad:
        return cell, np.zeros(3)
    given = np.array([cell[k] for k in range(3) if k not in pad]).reshape(-1, 3)
    # orthonormal basis of the complement of span(given): the last right-singular vectors
    dirs = np.linalg.svd(given)[2][len(given):] if len(given) else np.eye(3)
    shift = np.zeros(3)
    for k, d in zip(pad, dirs):
        proj = pos @ d
        cell[k] = d * (np.ptp(proj) + 2.0 * reach + 1.0) if len(pos) else d * (2.0 * reach + 1.0)
        if len(pos):
            shift += d * (reach + 0.5 - proj.min())
    return cell, shift


def check_skin(skin) -> float:
    """the Verlet skin as a float: finite and >= 0 (a negative one drops bonds inside the cutoff, silently)"""
    s = float(skin)
    if not (np.isfinite(s) and s >= 0.0):
        raise ValueError(f"skin must be a finite number >= 0, got {skin!r}")
    return s


class NeighborList:
    def __init__(self, cutoff: float, skin: float = 0.5, num_threads: int = 0):
        self.cutoff, self.skin = float(cutoff), check_skin(skin)
        if not (np.isfinite(self.cutoff) and self.cutoff > 0.0):
            raise ValueError(f"cutoff must be a finite number > 0, got {cutoff!r}")
        self.num_threads = int(num_threads)
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
            if len(atoms) == 0 or ((atoms.positions - self._x0) ** 2).sum(axis=1).max() <= (0.5 * self.skin) ** 2:
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
        pos = atoms.positions + shift
        self.perm, self.offset, self.j, self.image, self.shifts = neighbor_search(
            np.ascontiguousarray(pos, dtype=np.float64), np.ascontiguousarray(cell, dtype=np.float64),
            [bool(p) for p in atoms.pbc], reach, num_threads=self.num_threads)
        self.inv = np.empty(n, np.int64)
        self.inv[self.perm] = np.arange(n)

    def rij(self, atoms, num_threads: int = 0):
        """Bond vectors x_j - x_i + S cell for the current positions (new atom order), [nbonds, 3]. The engines form
        them themselves: Model.compute(..., positions=atoms.positions[perm], image=image, shifts=shifts)."""
        i = np.repeat(np.arange(len(self.perm)), np.diff(self.offset))
        return bond_vectors(np.ascontiguousarray(atoms.positions[self.perm], dtype=np.float64), i, self.j,
                            self.shifts[self.image], num_threads=num_threads)

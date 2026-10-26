# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Verlet neighbour list of grace_runtime.

The OpenMP cell list of ``grace_runtime._core`` at ``cutoff + skin``. The list is kept until an atom has moved more
than ``skin / 2`` since the last build (or the cell, pbc or atomic numbers change). Atoms are put in spatial-bin order:
the 2L driver distributes atoms over threads with a static schedule and relies on neighbours of one thread's atoms
being close in index.

Cells with zero vectors (molecules, ``cell.rank < 3``) get padded vectors orthogonal to the others, long enough that no
image is within range; positions are shifted into that box for the search only, bond vectors use the true positions.
"""
import numpy as np
from ._core import bond_vectors, neighbor_search


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
        pos = atoms.positions + shift
        self.perm, self.offset, self.j, self._io, self._jo, self._shift = neighbor_search(
            np.ascontiguousarray(pos, dtype=np.float64), np.ascontiguousarray(cell, dtype=np.float64),
            [bool(p) for p in atoms.pbc], reach, num_threads=self.num_threads)
        self.inv = np.empty(n, np.int64)
        self.inv[self.perm] = np.arange(n)

    def rij(self, atoms, num_threads: int = 0):
        """Bond vectors x_j - x_i + S cell for the current positions (new atom order), [nbonds, 3]."""
        return bond_vectors(np.ascontiguousarray(atoms.positions, dtype=np.float64), self._io, self._jo, self._shift,
                            num_threads=num_threads)

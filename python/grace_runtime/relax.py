# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""relax_batch: many structures relaxed together, without torch. FIRE as ase.optimize.FIRE (one optimizer state per
structure), optionally with the cell as ase.filters.FrechetCellFilter and the space group kept as
ase.constraints.FixSymmetry; the forces of all structures in flight come from one GPU batch per step (or the CPU
kernels, structure after structure); finished structures leave the batch and waiting ones enter it."""
import warnings
from typing import NamedTuple

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixSymmetry
from ase.stress import voigt_6_to_full_3x3_stress

from ._core import Model
from .batch import _gpu_batch, _gpu_compute, _species_of
from .calculator import _chunk_arg, _CpuEngine, _select_device, _virial_to_stress, default_threads
from .dispersion import resolve as _resolve_dispersion

# [13/13] Pade coefficients and the 1-norm bound of scipy.linalg.expm_frechet (Al-Mohy, Higham 2009)
_B13 = (64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800., 129060195264000.,
        10559470521600., 670442572800., 33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)
_THETA13 = 4.74


def expm_frechet(A, E=None):
    """expm(A) and, with E, its Frechet derivative in the direction E, for stacks of 3x3 matrices [..., 3, 3]:
    the [13/13] Pade approximant with scaling and squaring of scipy.linalg.expm_frechet, batched. Returns (expm(A),
    derivative or None)."""
    A = np.asarray(A, dtype=np.float64)
    shape = A.shape
    A = A.reshape(-1, 3, 3)
    with np.errstate(divide="ignore"):
        s = np.ceil(np.log2(np.abs(A).sum(axis=1).max(axis=1) / _THETA13))
    s = np.maximum(s, 0).astype(np.int64)
    scale = (2.0 ** -s)[:, None, None]
    A = A * scale
    b = _B13
    eye = np.eye(3)
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4
    W1 = b[13] * A6 + b[11] * A4 + b[9] * A2
    W2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * eye
    Z1 = b[12] * A6 + b[10] * A4 + b[8] * A2
    Z2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * eye
    W = A6 @ W1 + W2
    U = A @ W
    V = A6 @ Z1 + Z2
    R = np.linalg.solve(V - U, U + V)
    L = None
    if E is not None:
        E = np.broadcast_to(np.asarray(E, dtype=np.float64), shape).reshape(-1, 3, 3) * scale
        M2 = A @ E + E @ A
        M4 = A2 @ M2 + M2 @ A2
        M6 = A4 @ M2 + M4 @ A2
        Lw = A6 @ (b[13] * M6 + b[11] * M4 + b[9] * M2) + M6 @ W1 + (b[7] * M6 + b[5] * M4 + b[3] * M2)
        Lu = A @ Lw + E @ W
        Lv = A6 @ (b[12] * M6 + b[10] * M4 + b[8] * M2) + M6 @ Z1 + (b[6] * M6 + b[4] * M4 + b[2] * M2)
        L = np.linalg.solve(V - U, Lu + Lv + (Lu - Lv) @ R)
    for k in range(int(s.max()) if len(s) else 0):    # squaring, each matrix its own number of times
        sel = s > k
        if L is not None:
            L[sel] = R[sel] @ L[sel] + L[sel] @ R[sel]
        R[sel] = R[sel] @ R[sel]
    return R.reshape(shape), (None if L is None else L.reshape(shape))


def _logm(X):
    """principal logarithm of a stack of real 3x3 matrices with positive eigenvalues [S, 3, 3], as scipy.linalg.logm:
    Denman-Beavers square roots until |X - I|_1 <= 0.25, then log X = 2 artanh(Z), Z = (X + I)^-1 (X - I), as its
    series (|Z| <= 1/7: 14 terms below 1e-16)"""
    X = np.array(X, dtype=np.float64)
    eye = np.eye(3)
    roots = np.zeros(len(X))
    for _ in range(64):
        big = np.abs(X - eye).sum(axis=1).max(axis=1) > 0.25
        if not big.any():
            break
        Y, Z = X[big], np.broadcast_to(eye, X[big].shape).copy()
        for _ in range(64):    # Y -> sqrt(X), Z -> its inverse
            Y, Z, Y0 = 0.5 * (Y + np.linalg.inv(Z)), 0.5 * (Z + np.linalg.inv(Y)), Y
            if np.abs(Y - Y0).max() <= 1e-14 * np.abs(Y).max():
                break
        X[big] = Y
        roots[big] += 1
    Z = np.linalg.solve(X + eye, X - eye)
    Z2, term, out = Z @ Z, Z, Z.copy()
    for j in range(1, 14):
        term = term @ Z2
        out += term / (2 * j + 1)
    return 2.0 * (2.0 ** roots)[:, None, None] * out


_VOIGT33 = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])    # Voigt index of each entry of a symmetric 3x3 tensor


class _Symmetry(NamedTuple):
    """the symmetry of a structure as one operation per rotation acting on the orbits of the atoms under the pure
    translations (see _symmetry_of)"""
    rot: np.ndarray    # [p, 3, 3] rotations in fractional coordinates (p <= 48)
    omap: np.ndarray    # [p, k] orbit o goes to orbit omap[p, o]
    orbits: np.ndarray    # [n] the orbit of each atom (0 .. k-1)
    adjust_positions: bool
    adjust_cell: bool


def _symmetry_of(i, at, fix_symmetry, symprec):
    """the _Symmetry of a structure, from an ase FixSymmetry of the input or from spglib (fix_symmetry); None without.
    The group of a supercell holds every lattice translation of its primitive cell (g grows with n): the average over
    it is the average over each orbit of the atoms under the pure translations, then over one operation per rotation
    acting on the orbits."""
    for c in at.constraints:
        if isinstance(c, FixSymmetry):
            rot = np.asarray(c.rotations, np.float64)
            amap = np.asarray(c.symm_map, np.int64)
            if amap.shape != (len(rot), len(at)):    # atoms added or removed after the constraint was made
                raise ValueError(f"structure {i}: its FixSymmetry was made for {amap.shape[-1]} atoms, not {len(at)}")
            pure = _pure_translations(rot)
            orbits = np.unique(amap[pure].min(axis=0), return_inverse=True)[1].reshape(-1)
            reps, rep_of = _one_per_rotation(i, rot, len(pure))
            # the reduction needs a group (a constraint edited by hand may not be one): every operation is its
            # rotation's first one after a pure translation, map = map_first[map_translation]
            first = amap[reps][rep_of]
            t_of = np.full(len(at), -1)
            t_of[amap[pure, 0]] = np.arange(len(pure))    # the translation that takes atom 0 to an atom
            t = t_of[np.argsort(first, axis=1)[np.arange(len(rot)), amap[:, 0]]]
            if (t < 0).any() or (np.take_along_axis(first, amap[pure][t], axis=1) != amap).any():
                raise ValueError(f"structure {i}: the symmetry operations are not a space group")
            return _on_orbits(i, rot[reps], amap[reps], orbits, len(pure), c.do_adjust_positions, c.do_adjust_cell)
    if not fix_symmetry:
        return None
    import spglib
    frac = at.get_scaled_positions() % 1.0 % 1.0    # in [0, 1) (-1e-17 % 1.0 is 1.0)
    ds = spglib.get_symmetry_dataset((at.cell.array, frac, at.numbers), symprec=symprec)
    if ds is None:
        raise ValueError(f"structure {i}: spglib found no symmetry ({spglib.get_error_message()})")
    rot, trans = np.asarray(ds.rotations, np.float64), np.asarray(ds.translations, np.float64)
    pure = _pure_translations(rot)
    # the orbits: the atoms of spglib's primitive cell; each atom is checked to be its orbit's first one shifted by a
    # pure translation
    orbits = np.unique(ds.mapping_to_primitive, return_inverse=True)[1].reshape(-1)
    _nearest(i, at, trans[pure], frac - frac[np.unique(orbits, return_index=True)[1]][orbits], symprec)
    reps, _ = _one_per_rotation(i, rot, len(pure))
    return _on_orbits(i, rot[reps], _atom_map(i, at, frac, rot[reps], trans[reps], symprec), orbits, len(pure), True,
                      True)


def _pure_translations(rot):
    """indices of the operations without rotation"""
    return np.flatnonzero((np.rint(rot) == np.eye(3)).all(axis=(1, 2)))


def _one_per_rotation(i, rot, ntrans):
    """the first operation of each rotation (the group: these times the ntrans pure translations) and, for each
    operation, the index of its rotation among them"""
    r = np.rint(rot).astype(np.int64).reshape(len(rot), 9)
    if np.abs(r).max(initial=0) >= 64:
        raise ValueError(f"structure {i}: a rotation in fractional coordinates has an entry beyond 63")
    key = (r + 64) @ (128 ** np.arange(9))    # each rotation as one integer (a 1-d unique: ~20x faster than rows)
    _, first, rep_of = np.unique(key, return_index=True, return_inverse=True)
    if len(first) * ntrans != len(rot):
        raise ValueError(f"structure {i}: the symmetry operations are not a space group")
    return first, rep_of.reshape(-1)


def _on_orbits(i, rot, amap, orbits, ntrans, adjust_positions, adjust_cell):
    """the _Symmetry of the operations rot [p, 3, 3] with the atom maps amap [p, n]: each orbit (labels 0 .. k-1) has
    one atom per pure translation and goes to one orbit as a whole"""
    first = np.unique(orbits, return_index=True)[1]    # an atom of each orbit
    omap = orbits[amap[:, first]]
    if (np.bincount(orbits) != ntrans).any() or (orbits[amap] != omap[:, orbits]).any():
        raise ValueError(f"structure {i}: the symmetry operations do not map the atoms onto each other")
    return _Symmetry(rot, omap.astype(np.int32), orbits.astype(np.int32), bool(adjust_positions), bool(adjust_cell))


def _nearest(i, at, points, x, symprec):
    """the fractional points [m, 3] nearest to the fractional points x [..., 3] (periodic); an error beyond 10 symprec"""
    from scipy.spatial import cKDTree
    d, j = cKDTree(points % 1.0 % 1.0, boxsize=1.0).query(x.reshape(-1, 3) % 1.0 % 1.0)
    if d.max(initial=0.0) * np.linalg.norm(at.cell.array, axis=1).max() > 10 * symprec:
        raise ValueError(f"structure {i}: the symmetry operations do not map the atoms onto each other")
    return j.reshape(x.shape[:-1])


def _atom_map(i, at, frac, rot, trans, symprec):
    """map [g, n] of the atoms under each operation (nearest image of R x + t)"""
    amap = _nearest(i, at, frac, np.einsum("gij,nj->gni", rot, frac) + trans[:, None], symprec)
    if (np.sort(amap, axis=1) != np.arange(len(at))).any():
        raise ValueError(f"structure {i}: the symmetry operations do not map the atoms onto each other")
    return amap


def _sum_rows(idx, x, n):
    """the rows of x [m, d] summed into n rows by idx [m] (one bincount)"""
    d = x.shape[1]
    return np.bincount((idx[:, None] * d + np.arange(d)).reshape(-1), x.reshape(-1), n * d).reshape(n, d)


def _frechet_cell_forces(logD, grad):
    """sum_ij d expm(logD)_ij / d logD_mu,nu * grad_ij for stacks [S, 3, 3] (ase CellUtility.get_forces_frechet):
    <L(A, E_mu,nu), grad> = <E_mu,nu, L(A^T, grad)>, one Frechet derivative instead of nine"""
    return expm_frechet(np.transpose(logD, (0, 2, 1)), grad)[1]


class _GPUForces:
    """energy, forces and stress of all structures in flight from one Batch.compute (neighbour lists on the GPU)"""

    def __init__(self, kk, m, species, chunk, max_nall, disp):
        self.kk, self.m, self.chunk, self.max_nall, self.disp = kk, m, chunk, max_nall, disp
        self.species, self.b = [], None
        self.add_species(species)

    def add_species(self, symbols):
        """grow the species set; a new element sets up a new Batch"""
        new = sorted(set(self.species) | set(symbols))
        if new and (new != self.species or self.b is None):
            self.species = new
            self.b = _gpu_batch(self.kk, self.m, new, self.chunk, keep=True, dispersion=self.disp)

    def __call__(self, pool, pos, cells):
        out = _gpu_compute(self.b, self.species, pool.natoms, pool.numbers, pos, cells, pool.pbc,
                           max_nall=self.max_nall)
        stress = np.zeros((len(cells), 6))
        r3 = pool.rank3
        stress[r3] = _virial_to_stress(out["virial"][r3], np.abs(np.linalg.det(cells[r3])))
        return np.asarray(out["energy"]), np.asarray(out["forces"]).reshape(-1, 3), stress


class _CPUForces:
    """the CPU engine on each structure in flight (one per structure: its neighbour list keeps its skin)"""

    def __init__(self, m, nth, disp):
        self.m, self.nth, self.disp, self.engines = m, nth, disp, {}

    def add_species(self, symbols):
        pass

    def __call__(self, pool, pos, cells):
        S = len(pool.idx)
        energy, stress = np.zeros(S), np.zeros((S, 6))
        forces = np.zeros_like(pos)
        live = set()
        for k in range(S):
            i = int(pool.idx[k])
            live.add(i)
            if i not in self.engines:
                self.engines[i] = (_CpuEngine(self.m, num_threads=self.nth, dispersion=self.disp),
                                   pool.frames[k].copy())
            eng, at = self.engines[i]
            a0, a1 = pool.starts[k], pool.starts[k + 1]
            at.set_cell(cells[k], apply_constraint=False)    # a FixSymmetry of the input: the cell as it is
            at.positions[:] = pos[a0:a1]
            r = eng.compute(at)
            energy[k] = r["energy"]
            forces[a0:a1] = r["forces"]
            stress[k] = r["stress"]
        for i in set(self.engines) - live:
            del self.engines[i]
        return energy, forces, stress


class _Pool:
    """the structures in flight: optimizer state in flat arrays (atoms contiguous per structure)"""

    ATOM = ("X", "V", "numbers", "fixed")    # per atom
    SYS = ("idx", "natoms", "cell0", "G", "C", "CV", "pbc", "dt", "a", "ecf", "nst", "nsteps", "err")    # per structure

    def __init__(self, filt):
        self.filt = filt
        self.frames, self.sym = [], []
        for name, v in self._rows(-1, Atoms(), np.zeros(0, bool), 1.0, {"dt": 0.0, "a": 0.0}).items():
            setattr(self, name, v[:0])
        self._index()

    @staticmethod
    def _rows(i, at, fixed, ecf, fire):
        """the arrays of one new structure: its atoms, and one row of each per-structure field"""
        n = len(at)
        return {"X": at.positions.astype(np.float64), "V": np.zeros((n, 3)), "numbers": at.numbers.astype(np.int64),
                "fixed": fixed, "idx": np.array([i], np.int64), "natoms": np.array([n], np.int32),
                "cell0": at.cell.array[None].astype(np.float64), "G": np.eye(3)[None], "C": np.zeros((1, 3, 3)),
                "CV": np.zeros((1, 3, 3)), "pbc": np.array([at.pbc], bool), "dt": np.array([float(fire["dt"])]),
                "a": np.array([float(fire["a"])]), "ecf": np.array([float(ecf)]), "nst": np.zeros(1, np.int64),
                "nsteps": np.zeros(1, np.int64), "err": np.array([None], object)}

    def _index(self):
        self.starts = np.concatenate([[0], np.cumsum(self.natoms)]).astype(np.int64)
        self.sid = np.repeat(np.arange(len(self.natoms)), self.natoms)
        self.rank3 = np.any(self.cell0 != 0.0, axis=2).all(axis=1)
        # symmetry: one operation per rotation of all symmetric structures acting on their orbits, as flat arrays
        # (operation, orbit) -> image orbit; the atoms of the symmetric structures and their orbits
        ks = [k for k, s in enumerate(self.sym) if s is not None]
        self.sym_n = np.zeros(len(self.sym))    # operations per structure (0: no symmetry)
        self.sym_n[ks] = [len(self.sym[k].rot) for k in ks]
        self.has_sym = self.sym_n > 0
        self.sym_pos = np.array([s is not None and s.adjust_positions for s in self.sym], bool)
        self.sym_cell = np.array([s is not None and s.adjust_cell for s in self.sym], bool)
        if not ks:
            return
        sym = [self.sym[k] for k in ks]
        o0 = np.cumsum([0] + [s.omap.shape[1] for s in sym])    # first orbit of each
        g0 = np.cumsum([0] + [len(s.rot) for s in sym])
        self.sym_atoms = np.concatenate([np.arange(self.starts[k], self.starts[k + 1]) for k in ks])
        self.sym_orbit = np.concatenate([s.orbits + o for s, o in zip(sym, o0)])
        self.sym_norb = int(o0[-1])
        self.sym_orbit_n = np.bincount(self.sym_orbit, minlength=self.sym_norb)[:, None]    # atoms per orbit
        self.sym_R = np.concatenate([s.rot for s in sym])
        self.sym_osid = np.concatenate([np.full(len(s.rot), k) for k, s in zip(ks, sym)])
        self.sym_op = np.concatenate([np.repeat(np.arange(s.omap.shape[0]), s.omap.shape[1]) + g
                                      for s, g in zip(sym, g0)])
        self.sym_src = np.concatenate([np.tile(np.arange(s.omap.shape[1]), s.omap.shape[0]) + o
                                       for s, o in zip(sym, o0)])
        self.sym_dst = np.concatenate([s.omap.reshape(-1) + o for s, o in zip(sym, o0)])

    def sym_rank1(self, vec, lat):
        """vectors [N, 3] of the symmetric structures averaged over their operations (ase symmetrize_rank1); lat
        [S, 3, 3]: the cells (rows)"""
        if not self.has_sym.any():
            return vec
        a, o, k = self.sym_atoms, self.sym_orbit, self.sym_norb
        sid = self.sid[a]
        inv = np.linalg.inv(np.where(self.has_sym[:, None, None], lat, np.eye(3)))    # others: any (molecules: 0)
        fs = np.einsum("ni,nij->nj", vec[a], inv[sid])
        fo = _sum_rows(o, fs, k) / self.sym_orbit_n    # the average over each orbit
        acc = np.zeros_like(fo)
        for p in range(0, len(self.sym_op), 1 << 20):    # blocks of (operation, orbit) pairs: bounded memory
            b = slice(p, p + (1 << 20))
            acc += _sum_rows(self.sym_dst[b], np.einsum("pij,pj->pi", self.sym_R[self.sym_op[b]], fo[self.sym_src[b]]),
                             k)
        out = vec.copy()
        out[a] = np.einsum("ni,nij->nj", acc[o] / self.sym_n[sid, None], lat[sid])
        return out

    def sym_rank2(self, T, lat):
        """tensors [S, 3, 3] of the symmetric structures averaged over their operations (ase symmetrize_rank2)"""
        if not self.has_sym.any():
            return T
        lat = np.where(self.has_sym[:, None, None], lat, np.eye(3))
        scaled = lat @ T @ np.transpose(lat, (0, 2, 1))
        R = self.sym_R
        t = np.transpose(R, (0, 2, 1)) @ scaled[self.sym_osid] @ R
        acc = _sum_rows(self.sym_osid, t.reshape(-1, 9), len(T))
        inv = np.linalg.inv(lat)
        out = inv @ (acc.reshape(-1, 3, 3) / np.maximum(self.sym_n, 1.0)[:, None, None]) @ np.transpose(inv, (0, 2, 1))
        return np.where(self.has_sym[:, None, None], out, T)

    def add(self, entries, fire):
        """entries: (input index, Atoms, fixed mask, exp_cell_factor, symmetry or None)"""
        if not entries:
            return
        rows = [self._rows(i, at, f, e, fire) for i, at, f, e, _ in entries]
        for name in self.ATOM + self.SYS:
            setattr(self, name, np.concatenate([getattr(self, name)] + [r[name] for r in rows]))
        self.frames += [at.copy() for _, at, _, _, _ in entries]    # the input as read (the caller may reuse it)
        self.sym += [s for _, _, _, _, s in entries]
        self._index()

    def keep(self, keep):
        """keep the structures where keep [S] is True"""
        atom = keep[self.sid]
        for name in self.ATOM:
            setattr(self, name, getattr(self, name)[atom])
        for name in self.SYS:
            setattr(self, name, getattr(self, name)[keep])
        self.frames = [f for f, k in zip(self.frames, keep) if k]
        self.sym = [s for s, k in zip(self.sym, keep) if k]
        self._index()

    def geometry(self):
        """Cartesian positions and cells: x = X G, cell = cell0 G (ase set_positions_unitcellfilter)"""
        if not self.filt:
            return self.X, self.cell0
        return np.einsum("ni,nij->nj", self.X, self.G[self.sid]), self.cell0 @ self.G

    def snapshot(self, k, pos, cells, energy, forces, stress):
        """structure k as a copy of its input Atoms: current positions and cell, a SinglePointCalculator"""
        a0, a1 = self.starts[k], self.starts[k + 1]
        at = self.frames[k].copy()
        at.set_cell(cells[k], apply_constraint=False)    # a FixSymmetry of the input: the cell as it is
        at.positions[:] = pos[a0:a1]
        at.calc = SinglePointCalculator(at, energy=float(energy[k]), forces=forces[a0:a1].copy(),
                                        stress=stress[k].copy())
        return at


def _check_input(i, at, filt, fix_symmetry):
    """the FixAtoms mask of a structure, after the checks"""
    if filt and len(at) and at.cell.rank < 3:
        raise ValueError(f"structure {i}: the cell filter needs a cell of rank 3")
    fixed = _fixed_mask(at)
    if len(at) and (fix_symmetry or any(isinstance(c, FixSymmetry) for c in at.constraints)):
        if at.cell.rank < 3 or not all(at.pbc):
            raise ValueError(f"structure {i}: symmetry needs a periodic cell of rank 3")
        if fixed.any():
            raise ValueError(f"structure {i}: FixAtoms and symmetry together are not supported")
    return fixed


def _fixed_mask(at):
    fixed = np.zeros(len(at), bool)
    for c in at.constraints:
        if isinstance(c, FixAtoms):
            fixed[np.asarray(c.index)] = True
        elif not isinstance(c, FixSymmetry):
            raise ValueError(f"relax_batch takes FixAtoms and FixSymmetry constraints only, not {type(c).__name__}")
    return fixed


def relax_batch_iter(model, structures, fmax: float = 0.05, max_steps: int = 500, cell_filter=None,
                     scalar_pressure: float = 0.0, hydrostatic_strain: bool = False, constant_volume: bool = False,
                     mask=None, exp_cell_factor=None, dt: float = 0.1, maxstep: float = 0.2, dtmax: float = 1.0,
                     Nmin: int = 5, finc: float = 1.1, fdec: float = 0.5, astart: float = 0.1, fa: float = 0.99,
                     a: float = 0.1, device: str = "auto", max_atoms: int = 131072, max_nall: int = 1 << 21,
                     num_threads: int | None = None, chunksize="auto", callback=None, fix_symmetry: bool = False,
                     symprec: float = 0.01, dispersion=None):
    """Relax many structures together; yields (input index, result) as each one finishes (completion order).

    model: path to a kokkos.npz, a registry name or a grace_runtime.Model.
    structures: ASE Atoms, any iterable (read lazily: a structure is read when there is room in the batch).
    fmax: convergence as ase.optimize.Optimizer: the largest force row (eV/A) below fmax; with the cell filter the
        rows of ase.filters.FrechetCellFilter (atom forces in the frame of the initial cell, and the cell rows).
    max_steps: FIRE steps per structure (as Optimizer.run(fmax, steps)); a structure is checked before each step, so
        n_steps steps take n_steps + 1 force evaluations.
    cell_filter: None (positions only) or "frechet" (positions and cell, as ase.filters.FrechetCellFilter).
    scalar_pressure (eV/A^3), hydrostatic_strain, constant_volume, mask (6 Voigt or 3x3 booleans), exp_cell_factor
        (default: the number of atoms of each structure): as FrechetCellFilter; only with cell_filter="frechet".
    dt, maxstep, dtmax, Nmin, finc, fdec, astart, fa, a: as ase.optimize.FIRE (maxstep limits the norm of the whole
        step of one structure, atoms and cell together).
    device, max_nall, num_threads, chunksize: as compute_batch. max_atoms: atoms in flight (one GPU call per step).
    dispersion: a dispersion correction added to the model, as compute_batch (a grace_runtime.D3 or "d3-pbe",
        "d3-r2scan"); the relaxation follows the sums, and the results are the sums.
    callback(step, indices, atoms): after each force evaluation, for the structures still running (input indices;
        Atoms with the current positions, cell and a SinglePointCalculator; built only for a callback). It may return
        input indices to stop now (result converged=False, stopped=True).
    fix_symmetry: keep the space group of every structure as ase.constraints.FixSymmetry (forces, stress, position
        and cell steps averaged over the symmetry operations; no refinement of the input), the operations found by
        spglib with symprec (A).

    Constraints of the input Atoms are kept: FixAtoms (their atoms do not move; their forces count as zero) and
    FixSymmetry (that structure keeps its symmetry, with the constraint's operations, as in ASE); other constraints,
    and FixAtoms with symmetry, are a ValueError. Inputs are not changed.

    Result per structure: "atoms" (a copy: relaxed positions and cell, a SinglePointCalculator with energy, forces,
    stress), "energy" (eV), "forces" [n, 3] (eV/A, model forces, with dispersion added), "stress" (ASE Voigt, eV/A^3,
    zeros for cells of rank < 3), "fmax" (the convergence figure at the end), "n_steps", "converged", "stopped",
    "error" (None, or why the cell-step limit of ase FixSymmetry stopped the structure; the others go on).
    """
    filt = {None: False, "frechet": True}.get(cell_filter)
    if filt is None:
        raise ValueError(f"cell_filter must be None or 'frechet', not {cell_filter!r}")
    if not filt and (scalar_pressure or hydrostatic_strain or constant_volume or mask is not None
                     or exp_cell_factor is not None):
        raise ValueError("scalar_pressure, hydrostatic_strain, constant_volume, mask and exp_cell_factor need "
                         "cell_filter='frechet'")
    mask3 = None
    if mask is not None:
        mask3 = np.asarray(mask, dtype=np.float64)
        if mask3.shape == (6,):
            mask3 = voigt_6_to_full_3x3_stress(mask3)
        elif mask3.shape != (3, 3):
            raise ValueError("mask must have shape (6,) or (3, 3)")
        if np.all(mask3 == 1.0):
            mask3 = None
    disp = _resolve_dispersion(dispersion)
    fire = {"dt": dt, "maxstep": maxstep, "dtmax": dtmax, "Nmin": Nmin, "finc": finc, "fdec": fdec, "astart": astart,
            "fa": fa, "a": a}
    nth = int(num_threads) if num_threads else default_threads()
    m = model if isinstance(model, Model) else Model(model)
    device, kkm, _ = _select_device(device, m)
    species, found = [], {}    # found: the symmetry of the list's structures, until they enter
    if isinstance(structures, (list, tuple)):    # the input checks before the first step (a stream: on entry)
        species = _species_of(structures)
        for i, at in enumerate(structures):
            _check_input(i, at, filt, fix_symmetry)
            if len(at) and (sym := _symmetry_of(i, at, fix_symmetry, symprec)) is not None:
                found[i] = sym
    forces_of = (_CPUForces(m, nth, disp) if kkm is None
                 else _GPUForces(kkm[0], m, species, _chunk_arg(chunksize), int(max_nall), disp))
    pool = _Pool(filt)
    source = enumerate(structures)
    waiting = None
    exhausted = False
    step = 0
    eye = np.eye(3)

    def finish(k, energy, forces, stress, fmax_k, pos, cells, converged, stopped=False):
        at = pool.snapshot(k, pos, cells, energy, forces, stress)
        return int(pool.idx[k]), {"atoms": at, "energy": float(energy[k]), "forces": at.calc.results["forces"].copy(),
                                  "stress": at.calc.results["stress"].copy(), "fmax": float(fmax_k),
                                  "n_steps": int(pool.nsteps[k]), "converged": bool(converged),
                                  "stopped": bool(stopped), "error": pool.err[k]}

    while True:
        # fill the batch up to max_atoms atoms (at least one structure)
        entries, inflight = [], int(pool.natoms.sum())
        while not exhausted:
            if waiting is None:
                try:
                    waiting = next(source)
                except StopIteration:
                    exhausted = True
                    break
            i, at = waiting
            n = len(at)
            if n == 0:    # nothing to relax
                waiting = None
                at = at.copy()
                at.calc = SinglePointCalculator(at, energy=0.0, forces=np.zeros((0, 3)), stress=np.zeros(6))
                yield i, {"atoms": at, "energy": 0.0, "forces": np.zeros((0, 3)), "stress": np.zeros(6), "fmax": 0.0,
                          "n_steps": 0, "converged": True, "stopped": False, "error": None}
                continue
            if inflight and inflight + n > max_atoms:
                break
            fixed = _check_input(i, at, filt, fix_symmetry)
            sym = found.pop(i) if i in found else _symmetry_of(i, at, fix_symmetry, symprec)
            entries.append((i, at, fixed, float(n) if exp_cell_factor is None else float(exp_cell_factor), sym))
            inflight += n
            waiting = None
        if entries:    # a new element sets up a new GPU batch
            forces_of.add_species(_species_of([at for _, at, _, _, _ in entries]))
            pool.add(entries, fire)
        S = len(pool.idx)
        if S == 0:
            return
        # forces of all structures in flight
        pos, cells = pool.geometry()
        energy, forces, stress = forces_of(pool, pos, cells)
        # the forces and stress the optimizer sees: FixAtoms, symmetry (results: the model's)
        fu = np.where(pool.fixed[:, None], 0.0, pool.sym_rank1(forces, cells))
        sid = pool.sid
        FC = None
        if filt:
            D = np.transpose(pool.G, (0, 2, 1))    # deform_grad of the current cell
            vol = np.abs(np.linalg.det(cells))
            virial = -vol[:, None, None] * (pool.sym_rank2(stress[:, _VOIGT33], cells) + scalar_pressure * eye)
            if hydrostatic_strain:
                virial = (np.trace(virial, axis1=1, axis2=2) / 3.0)[:, None, None] * eye
            if mask3 is not None:
                virial = virial * mask3
            ucf = virial @ np.linalg.inv(np.transpose(D, (0, 2, 1)))
            logD = pool.C / pool.ecf[:, None, None]
            FC = _frechet_cell_forces(logD, ucf)
            if constant_volume:
                FC = FC - (np.trace(FC, axis1=1, axis2=2) / 3.0)[:, None, None] * eye
            FC = FC / pool.ecf[:, None, None]
            fu = np.einsum("ni,nij->nj", fu, D[sid])
        fmax_s = np.maximum.reduceat(np.linalg.norm(fu, axis=1), pool.starts[:-1])    # no empty structures
        if FC is not None:
            fmax_s = np.maximum(fmax_s, np.linalg.norm(FC, axis=2).max(axis=1))
        failed = np.not_equal(pool.err, None)    # its last step undone (_symmetrize_step)
        conv = (fmax_s < fmax) & ~failed
        done = conv | (pool.nsteps >= max_steps) | failed
        stopped = np.zeros(S, bool)
        if callback is not None:
            run = np.flatnonzero(~done)
            if len(run):
                ats = [pool.snapshot(k, pos, cells, energy, forces, stress) for k in run]
                stop = callback(step, [int(pool.idx[k]) for k in run], ats)
                stop = [] if stop is None else [int(i) for i in stop]    # any iterable of indices, numpy too
                if stop:
                    stopped = np.isin(pool.idx, stop) & ~done
                    done = done | stopped
        for k in np.flatnonzero(done):
            yield finish(k, energy, forces, stress, fmax_s[k], pos, cells, conv[k], stopped[k])
        if done.any():
            fu = fu[~done[sid]]
            if FC is not None:
                FC = FC[~done]
            pool.keep(~done)
        step += 1
        if done.all():
            continue
        if pool.has_sym.any():
            pos0, cells0 = pool.geometry()
            X0, G0, C0 = pool.X, pool.G, pool.C
        _fire_step(pool, fu, FC, fire)
        if filt:
            L = pool.C / pool.ecf[:, None, None]
            F, _ = expm_frechet(L)
            dev = np.transpose(F, (0, 2, 1)) - eye
            pool.G = eye + (dev if mask3 is None else dev * mask3)
            if mask3 is not None:    # ASE takes the log of the masked deformation again (get_positions_frechet)
                pool.C = _logm(np.transpose(pool.G, (0, 2, 1))) * pool.ecf[:, None, None]
        if pool.has_sym.any():
            _symmetrize_step(pool, X0, G0, C0, pos0, cells0)


def _symmetrize_step(pool, X0, G0, C0, pos0, cells0):
    """the step of the symmetric structures as ase FixSymmetry makes it: with the cell filter, the new cell's
    deformation relative to the current cell symmetrised (adjust_cell), the atoms scaled with the cell, then their step
    to the filter's new positions symmetrised (adjust_positions), and the filter's variables taken from the result
    again (FrechetCellFilter.get_positions); without, the position step symmetrised. A cell step beyond the limit of
    ase FixSymmetry.adjust_cell (|dF| > 0.25, where ase raises) is undone (positions and cell: X0, G0, C0 from before
    the FIRE step; its FIRE velocities are not, the structure does not go on) and the structure marked to stop: the
    next force evaluation, at the same geometry, finishes it; above 0.15 a warning, as ase"""
    sym, sid, eye = pool.has_sym, pool.sid, np.eye(3)
    if not pool.filt:
        step = pool.X - X0
        X = X0 + pool.sym_rank1(step, pool.cell0)
        pool.X = np.where(pool.sym_pos[sid, None], X, pool.X)
        return
    pos_req, cell_req = pool.geometry()
    inv0 = np.linalg.inv(cells0)
    dF = np.transpose(inv0 @ cell_req, (0, 2, 1)) - eye
    big = np.where(pool.sym_cell, np.abs(dF).max(axis=(1, 2)), 0.0)    # the limits of ase FixSymmetry.adjust_cell
    for k in np.flatnonzero(big > 0.15):
        if big[k] > 0.25:
            pool.err[k] = ("FixSymmetry adjust_cell does not work properly with large deformation gradient step "
                           f"{big[k]} > 0.25")
            warnings.warn(f"structure {int(pool.idx[k])}: {pool.err[k]}: it stops here (converged False), the others "
                          f"go on")
        else:
            warnings.warn(f"structure {int(pool.idx[k])}: FixSymmetry adjust_cell may be ill behaved with large "
                          f"deformation gradient step {big[k]}")
    dF = np.where(pool.sym_cell[:, None, None], pool.sym_rank2(dF, cells0), dF)
    cell = cells0 @ np.transpose(dF + eye, (0, 2, 1))
    scaled = np.einsum("ni,nij->nj", pos0, (inv0 @ cell)[sid])
    step = pos_req - scaled
    pos = scaled + np.where(pool.sym_pos[sid, None], pool.sym_rank1(step, cell), step)
    G = np.linalg.solve(pool.cell0, cell)
    X = np.einsum("ni,nij->nj", pos, np.linalg.inv(G)[sid])
    pool.C = pool.C.copy()
    pool.C[sym] = _logm(np.transpose(G[sym], (0, 2, 1))) * pool.ecf[sym, None, None]
    pool.G = np.where(sym[:, None, None], G, pool.G)
    pool.X = np.where(sym[sid, None], X, pool.X)
    bad = big > 0.25
    if bad.any():    # the step undone: the structure ends where its forces were evaluated
        pool.X = np.where(bad[sid, None], X0, pool.X)
        pool.G = np.where(bad[:, None, None], G0, pool.G)
        pool.C = np.where(bad[:, None, None], C0, pool.C)
        pool.nsteps = pool.nsteps - bad


def _fire_step(pool, fu, FC, fire):
    """one ase.optimize.FIRE step of every structure in the pool: forces fu [N, 3] (and cell forces FC [S, 3, 3])"""
    sid, S = pool.sid, len(pool.idx)
    V, CV = pool.V, pool.CV
    old = pool.nsteps > 0    # the first step of a structure skips the velocity mixing
    if old.any():
        vf = np.bincount(sid, (fu * V).sum(1), S)
        f2 = np.bincount(sid, (fu * fu).sum(1), S)
        v2 = np.bincount(sid, (V * V).sum(1), S)
        if FC is not None:
            vf += (FC * CV).sum(axis=(1, 2))
            f2 += (FC * FC).sum(axis=(1, 2))
            v2 += (CV * CV).sum(axis=(1, 2))
        up = old & (vf > 0.0)
        mix = np.where(up, pool.a * np.sqrt(v2) / np.sqrt(np.where(up, f2, 1.0)), 0.0)
        keepv = np.where(up, 1.0 - pool.a, 0.0)
        # first steps keep their (zero) velocities; downhill: mixed; uphill: zeroed
        keepv = np.where(old, keepv, 1.0)
        V = keepv[sid, None] * V + mix[sid, None] * fu
        if FC is not None:
            CV = keepv[:, None, None] * CV + mix[:, None, None] * FC
        inc = up & (pool.nst > fire["Nmin"])
        down = old & ~up
        pool.dt = np.where(inc, np.minimum(pool.dt * fire["finc"], fire["dtmax"]),
                           np.where(down, pool.dt * fire["fdec"], pool.dt))
        pool.a = np.where(inc, pool.a * fire["fa"], np.where(down, fire["astart"], pool.a))
        pool.nst = np.where(up, pool.nst + 1, np.where(down, 0, pool.nst))
    dt = pool.dt
    V = V + dt[sid, None] * fu
    dr = dt[sid, None] * V
    n2 = np.bincount(sid, (dr * dr).sum(1), S)
    if FC is not None:
        CV = CV + dt[:, None, None] * FC
        dC = dt[:, None, None] * CV
        n2 += (dC * dC).sum(axis=(1, 2))
    norm = np.sqrt(n2)
    scale = np.where(norm > fire["maxstep"], fire["maxstep"] / np.where(norm > 0, norm, 1.0), 1.0)
    pool.X = pool.X + scale[sid, None] * dr
    pool.V = V
    if FC is not None:
        pool.C = pool.C + scale[:, None, None] * dC
        pool.CV = CV
    pool.nsteps = pool.nsteps + 1


def relax_batch(model, structures, **kwargs) -> list:
    """relax_batch_iter with the results as a list in input order (see relax_batch_iter for the arguments)."""
    out = {}
    for i, r in relax_batch_iter(model, structures, **kwargs):
        out[i] = r
    return [out[i] for i in range(len(out))]

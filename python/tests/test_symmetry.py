# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""relax_batch with symmetry on the CPU: an ase FixSymmetry on the input against ase.optimize.FIRE with the same
constraint (positions only and with the Frechet cell filter: the same path to round-off, the same step count);
fix_symmetry=True (operations from spglib) as the constraint; the input's operations still map the result."""
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.spacegroup.symmetrize import symmetrize_rank1, symmetrize_rank2

from grace_runtime import GRACERuntimeCalculator, Model, elastic_batch, relax_batch
from grace_runtime.relax import _atom_map, _Pool, _symmetry_of

import spglib

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M1 = CACHE / "GRACE-1L-OMAT-large-base" / "kokkos.npz"

pytestmark = pytest.mark.skipif(not M1.exists(), reason="registry model missing")


@pytest.fixture(scope="module")
def model():
    return Model(str(M1))


def cells():
    """symmetric cells away from their minimum: tetragonal W3Mo (strained, one site displaced along the 4-fold axis)
    and a sheared hcp Ti supercell (monoclinic); each with an ase FixSymmetry (which refines it)"""
    w = bulk("W", "bcc", a=3.2, cubic=True).repeat((1, 1, 2))
    w.symbols[0] = "Mo"
    w.set_cell(w.cell.array @ np.diag([1.0, 1.0, 1.03]), scale_atoms=True)
    w.positions[2, 2] += 0.08
    ti = bulk("Ti", "hcp", a=2.95, c=4.68).repeat((2, 2, 1))
    ti.set_cell(ti.cell.array @ [[1, 0, 0], [0, 1, 0], [0.04, 0, 1.0]], scale_atoms=True)
    out = []
    for at in (w, ti):
        at.set_constraint(FixSymmetry(at, symprec=1e-3))
        out.append(at)
    return out


def keeps_symmetry(at, relaxed):
    """the input's operations (its FixSymmetry) still map the relaxed atoms onto each other (the relaxed structure
    may gain operations, not lose them)"""
    c = at.constraints[0]
    _atom_map(0, relaxed, relaxed.get_scaled_positions(), np.asarray(c.rotations, float), c.translations, 1e-5)
    return True


@pytest.mark.parametrize("filt", [None, "frechet"])
def test_fix_symmetry_matches_ase(model, filt):
    ats = cells()
    res = relax_batch(model, ats, fmax=1e-6, max_steps=25, cell_filter=filt, device="cpu")
    for at, r in zip(ats, res):
        a = at.copy()
        a.calc = GRACERuntimeCalculator(model, device="cpu")
        opt = FIRE(FrechetCellFilter(a) if filt else a, logfile=None)
        opt.run(fmax=1e-6, steps=25)
        assert r["n_steps"] == opt.nsteps == 25
        assert np.abs(r["atoms"].positions - a.positions).max() <= 1e-6, filt
        assert np.abs(r["atoms"].cell.array - a.cell.array).max() <= 1e-6, filt
        assert keeps_symmetry(at, r["atoms"])


def test_fix_symmetry_as_constraint(model):
    """fix_symmetry=True finds the operations with spglib: the same relaxation as the inputs' FixSymmetry, to
    convergence (the input is not refined: round-off), keeping the symmetry"""
    ats = cells()
    plain = [at.copy() for at in ats]
    for a in plain:
        a.set_constraint()
    got = relax_batch(model, plain, fmax=1e-3, cell_filter="frechet", fix_symmetry=True, symprec=1e-3, device="cpu")
    ref = relax_batch(model, ats, fmax=1e-3, cell_filter="frechet", device="cpu")
    for g, r, at in zip(got, ref, ats):
        assert g["converged"] and g["n_steps"] == r["n_steps"]
        assert np.abs(g["atoms"].positions - r["atoms"].positions).max() <= 1e-6
        assert np.abs(g["atoms"].cell.array - r["atoms"].cell.array).max() <= 1e-6
        assert keeps_symmetry(at, g["atoms"])


def test_atom_map():
    ti = bulk("Ti", "hcp", a=2.95, c=4.68).repeat((3, 3, 2))
    frac = ti.get_scaled_positions()
    sym = spglib.get_symmetry((ti.cell.array, frac, ti.numbers), symprec=1e-3)
    rot, tr = np.array(sym["rotations"], float), sym["translations"]
    amap = _atom_map(0, ti, frac, rot, tr, 1e-3)
    assert amap.shape == (len(rot), len(ti))
    for g in (1, len(rot) // 2, len(rot) - 1):
        d = (frac @ rot[g].T + tr[g]) - frac[amap[g]]
        assert np.abs(d - np.round(d)).max() <= 1e-8


@pytest.mark.parametrize("name", ["bcc", "nacl", "hcp-sheared"])
def test_supercell_one_operation_per_rotation(name):
    """a supercell's group holds every lattice translation of the primitive cell: it is kept as one operation per
    rotation and the orbits under the translations, and averages forces and stress as the whole group (ase
    symmetrize_rank1 / _rank2), from the FixSymmetry and from spglib (on the positions the FixSymmetry refined)"""
    if name == "bcc":
        at = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    elif name == "nacl":
        at = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 3, 1))
    else:
        at = bulk("Ti", "hcp", a=2.95, c=4.68).repeat((3, 3, 2))
        at.set_cell(at.cell.array @ [[1, 0, 0], [0, 1, 0], [0.04, 0, 1.0]], scale_atoms=True)
    c = FixSymmetry(at, symprec=1e-3)
    at.set_constraint(c)
    plain = at.copy()
    plain.set_constraint()
    n, g = len(at), len(c.rotations)
    rng = np.random.default_rng(0)
    f, s = rng.normal(size=(n, 3)), rng.normal(size=(3, 3))
    lat, inv = at.cell.array, np.linalg.inv(at.cell.array)
    for x, fix in ((at, False), (plain, True)):
        sym = _symmetry_of(0, x, fix, 1e-3)
        assert len(sym.rot) <= 48 and len(sym.rot) * np.bincount(sym.orbits)[0] == g
        pool = _Pool(None)
        pool.add([(0, x, np.zeros(n, bool), float(n), sym)], {"dt": 0.1, "a": 0.1})
        got = pool.sym_rank1(f, lat[None])
        assert np.abs(got - symmetrize_rank1(lat, inv, f, c.rotations, c.translations, c.symm_map)).max() <= 1e-12
        got = pool.sym_rank2((s + s.T)[None], lat[None])[0]
        assert np.abs(got - symmetrize_rank2(lat, inv, s + s.T, c.rotations)).max() <= 1e-12


def test_symmetry_errors(model):
    ats = cells()
    mol = ats[0].copy()
    mol.set_constraint()
    mol.pbc = [True, True, False]
    with pytest.raises(ValueError, match="symmetry needs a periodic cell"):
        relax_batch(model, [mol], fix_symmetry=True, device="cpu")
    both = ats[0].copy()
    both.set_constraint([FixAtoms([0]), FixSymmetry(both, symprec=1e-3)])
    with pytest.raises(ValueError, match="FixAtoms and symmetry"):
        relax_batch(model, [both], device="cpu")


def test_symmetry_mixed_pool_and_results(model):
    """a symmetric crystal next to a molecule and an empty structure in one pool; a large cell change of a
    FixSymmetry input (the result atoms keep the constraint; their cell is set as it is)"""
    big = bulk("W", "bcc", a=3.6, cubic=True)
    big.set_constraint(FixSymmetry(big, symprec=1e-3))
    mol = Atoms("W2", positions=[[0, 0, 0], [2.6, 0, 0]])
    res = relax_batch(model, [big, mol, Atoms()], fmax=1e-3, cell_filter=None, device="cpu")
    assert all(r["converged"] for r in res)
    r = relax_batch(model, [big], fmax=1e-3, cell_filter="frechet", device="cpu")[0]
    assert r["converged"] and abs(r["atoms"].cell[0, 0] / 3.6 - 1) > 0.1    # |dF| > 0.1 from the input cell
    assert np.allclose(r["atoms"].cell.array, np.diag(np.diag(r["atoms"].cell.array)))
    assert isinstance(r["atoms"].constraints[0], FixSymmetry)


def test_elastic_ignores_input_symmetry(model):
    """a FixSymmetry of an elastic_batch input is not carried into the strained cells"""
    w = relax_batch(model, [bulk("W", "bcc", a=3.17, cubic=True)], fmax=1e-4, cell_filter="frechet",
                    device="cpu")[0]["atoms"]
    sym = w.copy()
    sym.set_constraint(FixSymmetry(sym, symprec=1e-3))    # refines sym
    plain = sym.copy()
    plain.set_constraint()
    a, b = (elastic_batch(model, [x], relax_atoms=False, device="cpu")[0]["C"] for x in (plain, sym))
    assert np.abs(a - b).max() <= 1e-6


def test_symmetry_checked_before_the_first_step(model):
    """a list: a stale FixSymmetry (atoms added after it was made) is an error before any step"""
    good, stale = cells()
    stale = stale.copy()
    stale += Atoms("Ti", positions=[[0.1, 0.2, 0.3]])
    calls = []
    with pytest.raises(ValueError, match="structure 1: its FixSymmetry was made for 8 atoms, not 9"):
        relax_batch(model, [good, stale], max_atoms=1, device="cpu", callback=lambda *a: calls.append(a))
    assert calls == []


def test_symmetry_not_a_group():
    """a FixSymmetry whose operations are not a space group (one dropped; two images swapped in an operation that is
    not the first of its rotation) is refused"""
    at = bulk("W", "bcc", a=3.165, cubic=True).repeat(2)
    for broken in ("dropped", "swapped"):
        c = FixSymmetry(at, symprec=1e-3)
        if broken == "dropped":
            c.rotations, c.translations, c.symm_map = c.rotations[:-1], c.translations[:-1], c.symm_map[:-1]
        else:
            g = int(np.flatnonzero((np.asarray(c.rotations) == c.rotations[1]).all(axis=(1, 2)))[-1])
            m = np.array(c.symm_map)
            m[g, [0, 1]] = m[g, [1, 0]]
            c.symm_map = m
        x = at.copy()
        x.set_constraint(c)
        with pytest.raises(ValueError, match="not a space group"):
            _symmetry_of(0, x, False, 1e-3)


@pytest.mark.parametrize("ecf", [0.7, 0.4])
def test_large_cell_step_as_ase(model, ecf):
    """the limits of ase FixSymmetry.adjust_cell on the cell step |dF| (a small exp_cell_factor makes the first steps
    large), against ase FIRE with FrechetCellFilter: a warning above 0.15; above 0.25 ase raises, relax_batch stops
    that structure where ase stopped (same cell and step count; converged False, "error") and finishes the others"""
    at = bulk("W", "bcc", a=3.6, cubic=True)
    at.set_constraint(FixSymmetry(at, symprec=1e-3))
    good = bulk("W", "bcc", a=3.17, cubic=True)    # near its minimum: small steps
    good.set_constraint(FixSymmetry(good, symprec=1e-3))
    a = at.copy()
    a.calc = GRACERuntimeCalculator(model, device="cpu")
    opt = FIRE(FrechetCellFilter(a, exp_cell_factor=ecf), logfile=None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            opt.run(fmax=1e-3, steps=30)
            raised = False
        except RuntimeError as e:
            assert "adjust_cell does not work properly" in str(e)
            raised = True
        warned_ase = sum("may be ill behaved" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r, g = relax_batch(model, [at, good], fmax=1e-3, max_steps=30, cell_filter="frechet", exp_cell_factor=ecf,
                           device="cpu")
        warned = sum(str(x.message).startswith("structure 0:") and "may be ill behaved" in str(x.message) for x in w)
    assert raised == (ecf < 0.5) and warned == warned_ase, (raised, warned, warned_ase)
    assert g["error"] is None
    if raised:
        assert not r["converged"] and "adjust_cell does not work properly" in r["error"]
        assert r["n_steps"] == opt.nsteps
        assert np.abs(r["atoms"].cell.array - a.cell.array).max() <= 1e-6
    else:
        assert r["error"] is None

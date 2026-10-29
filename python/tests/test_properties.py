# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime.eos_batch / elastic_batch on the CPU: the fits on synthetic data (a linear-elastic stress-strain
relation, a Birch-Murnaghan E(V)), and with a model against the same quantities done by hand with
GRACERuntimeCalculator and ASE."""
import os
from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.eos import EquationOfState, birchmurnaghan

from grace_runtime import GRACERuntimeCalculator, Model, elastic_batch, eos_batch, relax_batch
from grace_runtime.properties import _elastic_from_stresses, _fit_eos, _strained

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M1 = CACHE / "GRACE-1L-OMAT-large-base" / "kokkos.npz"

needs_model = pytest.mark.skipif(not M1.exists(), reason="registry model missing")


@pytest.fixture(scope="module")
def model():
    return Model(str(M1))


@pytest.fixture(scope="module")
def w(model):
    """bcc W (cubic cell, 2 atoms) at the model's lattice constant"""
    at = bulk("W", "bcc", a=3.17, cubic=True)
    return relax_batch(model, [at], fmax=1e-4, cell_filter="frechet", device="cpu")[0]["atoms"]


def test_elastic_fit_recovers_linear_elastic():
    """stress = C eps + a quadratic term: the slope over symmetric strains is C to round-off (any 6x6 C); an
    isotropic C gives B = lambda + 2 mu / 3, G = mu for Voigt, Reuss and Hill"""
    rng = np.random.default_rng(0)
    A = rng.normal(0, 30, (6, 6))
    C = A @ A.T + 200 * np.eye(6)
    q = rng.normal(0, 500, (6, 6))
    strains = np.linspace(-0.005, 0.005, 5)
    stresses = np.array([[C[:, j] * e + q[j] * e * e for e in strains] for j in range(6)])
    r = _elastic_from_stresses(strains, stresses)
    assert np.abs(r["C"] - C).max() <= 1e-9 * np.abs(C).max() and r["asymmetry"] <= 1e-9 * np.abs(C).max()
    assert np.allclose(r["S"], np.linalg.inv(C)) and np.allclose(np.sort(r["C_eigval"]), np.linalg.eigvalsh(C))
    lam, mu = 120.0, 80.0
    iso = np.zeros((6, 6))
    iso[:3, :3] = lam
    iso[np.arange(6), np.arange(6)] += [2 * mu] * 3 + [mu] * 3
    r = _elastic_from_stresses(strains, np.array([[iso[:, j] * e for e in strains] for j in range(6)]))
    for t in "VRH":
        assert np.isclose(r["B" + t], lam + 2 * mu / 3) and np.isclose(r["G" + t], mu)
        assert np.isclose(r["E" + t], mu * (3 * lam + 2 * mu) / (lam + mu))
        assert np.isclose(r["nu" + t], lam / (2 * (lam + mu)))
    assert abs(r["AVR"]) <= 1e-12


def test_strained_cell():
    at = bulk("Mo", "bcc", a=3.15).repeat((1, 2, 1))
    s = _strained(at, 3, 0.01)    # engineering shear yz
    eps = np.zeros((3, 3))
    eps[1, 2] = eps[2, 1] = 0.005
    assert np.allclose(s.cell.array, at.cell.array @ (np.eye(3) + eps))
    assert np.allclose(s.positions, at.positions @ (np.eye(3) + eps))


def test_eos_fits_on_birch_murnaghan_curve():
    """amstools' polynomial (order 5) and the ase.eos fit on an analytic Birch-Murnaghan E(V)"""
    E0, B0, BP, V0 = -8.0, 1.2, 4.3, 16.0    # eV, eV/A^3, -, A^3
    v = np.linspace(0.9 * V0, 1.1 * V0, 11)
    e = birchmurnaghan(v, E0, B0, BP, V0)
    for fit, tol in (("poly", 2e-3), ("birchmurnaghan", 1e-6)):
        r = _fit_eos(v, e, fit, 5)
        assert abs(r["equilibrium_volume"] / V0 - 1) <= tol / 10, fit
        assert abs(r["equilibrium_energy"] - E0) <= 1e-6
        assert abs(r["equilibrium_bulk_modulus"] / (B0 / units.GPa) - 1) <= tol, fit
        assert abs(r["equilibrium_b_prime"] / BP - 1) <= 20 * tol, fit
        assert r["energy_rms"] <= 1e-5
    r = _fit_eos(v[:4], e[:4], "poly", 3)    # compressed side only: no minimum
    assert r["equilibrium_volume"] is None and r["energy_rms"] is not None


@needs_model
def test_eos_matches_ase_by_hand(model, w):
    """relax="none": the energies and pressures of the scaled cells of GRACERuntimeCalculator, the ASE fit of the same
    points; the polynomial (another model of E(V)) within 5 % of it"""
    res = eos_batch(model, [w, w.repeat((1, 1, 2))], fit="birchmurnaghan", device="cpu")
    calc = GRACERuntimeCalculator(model, device="cpu")
    v, e, p = [], [], []
    for s in np.linspace(0.9, 1.1, 11):
        a = w.copy()
        a.set_cell(w.cell.array * s ** (1 / 3), scale_atoms=True)
        a.calc = calc
        v.append(a.get_volume())
        e.append(a.get_potential_energy())
        p.append(-a.get_stress()[:3].mean())
    r = res[0]
    assert np.allclose(r["volume"], v, rtol=1e-12) and np.allclose(r["energy"], e, rtol=0, atol=1e-6)
    assert np.allclose(r["pressure"], p, rtol=0, atol=1e-7)
    v0, e0, b0 = EquationOfState(v, e, eos="birchmurnaghan").fit(warn=False)
    assert abs(r["equilibrium_volume"] - v0) <= 1e-6 * v0
    assert abs(r["equilibrium_bulk_modulus"] - b0 / units.GPa) <= 1e-4 * b0 / units.GPa
    big = res[1]    # the doubled cell: the same per-atom curve
    assert abs(big["equilibrium_volume"] / 2 - r["equilibrium_volume"]) <= 1e-4 * v0
    assert abs(big["equilibrium_bulk_modulus"] - r["equilibrium_bulk_modulus"]) <= 1e-3 * r["equilibrium_bulk_modulus"]
    poly = eos_batch(model, [w], device="cpu")[0]
    assert abs(poly["equilibrium_bulk_modulus"] / r["equilibrium_bulk_modulus"] - 1) <= 0.05
    assert "n_steps" not in poly


@needs_model
def test_eos_relax_keeps_volumes(model):
    """relax="atoms+shape" keeps the volume of each point (constant_volume; the volumes are those of the relaxed
    cells) and converges; "atoms" keeps the cells"""
    tri = bulk("Mo", "bcc", a=3.15).repeat((2, 1, 1))
    tri.set_cell(tri.cell.array @ [[1, 0.03, 0], [0.02, 1, 0.01], [0, 0, 1.02]], scale_atoms=True)
    tri.rattle(0.02, seed=1)
    vols = tri.get_volume() * np.linspace(0.95, 1.05, 6)
    r = {relax: eos_batch(model, [tri], num_of_point=6, volume_range=0.05, relax=relax, fmax=0.01, device="cpu")[0]
         for relax in ("atoms", "atoms+shape")}
    for x in r.values():
        assert np.allclose(x["volume"], vols, rtol=1e-9) and x["converged"].all() and (x["n_steps"] > 0).all()
    assert (r["atoms+shape"]["energy"] <= r["atoms"]["energy"] + 1e-6).all()    # the shape relaxation lowers it further
    with pytest.warns(UserWarning, match=r"structures \[0\]: relaxations not converged"):
        x = eos_batch(model, [tri], num_of_point=6, volume_range=0.05, relax="atoms", fmax=1e-6, max_steps=2,
                      device="cpu")[0]
    assert not x["converged"].any() and (x["n_steps"] == 2).all()


@needs_model
def test_eos_errors_and_no_minimum(model, w):
    with pytest.raises(ValueError, match="relax must be"):
        eos_batch(model, [w], relax="cell")
    with pytest.raises(ValueError, match="fit must be"):
        eos_batch(model, [w], fit="sj")
    with pytest.raises(ValueError, match="num_of_point must be >= 6"):
        eos_batch(model, [w], num_of_point=5)
    with pytest.raises(ValueError, match="same number of atoms"):
        eos_batch(model, [w, w.repeat(2)], volume_range=(30.0, 34.0))
    with pytest.raises(ValueError, match="eos_batch sets cell_filter itself"):
        eos_batch(model, [w], relax="atoms+shape", cell_filter="frechet")
    mol = w.copy()
    mol.cell[2] = 0.0
    with pytest.raises(ValueError, match="structure 1 needs a cell of rank 3"):
        eos_batch(model, [w, mol])
    v0 = w.get_volume()
    with pytest.warns(UserWarning, match="structure 0: no energy minimum"):
        r = eos_batch(model, [w], num_of_point=6, volume_range=(0.8 * v0, 0.7 * v0), device="cpu")[0]
    assert r["equilibrium_bulk_modulus"] is None and np.all(np.diff(r["volume"]) > 0)    # the pair in any order


@needs_model
def test_elastic_matches_hand_fit(model, w):
    """clamped ions: C from the stresses of GRACERuntimeCalculator on the same strained cells, fitted by hand; bcc W
    is cubic (C11 = C22 = C33, C12 = C13 = C23, C44 = C55 = C66, other entries 0) to the figure of the fit"""
    r = elastic_batch(model, [w], relax_atoms=False, device="cpu")[0]
    calc = GRACERuntimeCalculator(model, device="cpu")
    strains = np.linspace(-0.005, 0.005, 5)
    C = np.zeros((6, 6))
    for j in range(6):
        sig = []
        for e in strains:
            a = _strained(w, j, e)
            a.calc = calc
            sig.append(a.get_stress() / units.GPa)
        C[:, j] = np.polyfit(strains, np.array(sig), 1)[0]
    assert np.abs(r["C"] - (C + C.T) / 2).max() <= 1e-6 and r["stresses"].shape == (6, 5, 6)
    c = r["C"]
    assert r["asymmetry"] <= 0.5
    for a, b in ((0, 1), (0, 2), (3, 4), (3, 5)):
        assert abs(c[a, a] - c[b, b]) <= 0.5
    assert abs(c[0, 1] - c[0, 2]) <= 0.5 and abs(c[0, 1] - c[1, 2]) <= 0.5
    assert np.abs(c[:3, 3:]).max() <= 0.5 and abs(c[3, 4]) + abs(c[3, 5]) + abs(c[4, 5]) <= 0.5
    assert 300 < c[0, 0] < 700 and 100 < c[3, 3] < 250 and 200 < r["BH"] < 400    # W: 523, 160, 310 GPa (exp.)
    assert "n_steps" not in r


@needs_model
def test_dispersion_reaches_eos_and_elastic(model, w):
    """dispersion= goes through to compute_batch (eos, relax "none") and relax_batch (elastic, relax_atoms): energies,
    pressures and stresses of GRACERuntimeCalculator(dispersion=...) on the same cells"""
    calc = GRACERuntimeCalculator(model, device="cpu", dispersion="d3-pbe")
    r = eos_batch(model, [w], fit="birchmurnaghan", device="cpu", dispersion="d3-pbe")[0]
    for v, e, p in zip(r["volume"], r["energy"], r["pressure"]):
        a = w.copy()
        a.set_cell(w.cell.array * (v / w.get_volume()) ** (1 / 3), scale_atoms=True)
        a.calc = calc
        assert abs(a.get_potential_energy() - e) <= 1e-6 and abs(-a.get_stress()[:3].mean() - p) <= 1e-7
    el = elastic_batch(model, [w], num_of_point=3, fmax=1e-3, device="cpu", dispersion="d3-pbe")[0]
    for j in range(6):
        for k, e in enumerate(el["strains"]):
            a = _strained(w, j, e)
            a.calc = calc
            assert np.abs(el["stresses"][j, k] - a.get_stress() / units.GPa).max() <= 1e-4


@needs_model
def test_elastic_internal_relaxation(model):
    """a cell with internal degrees of freedom (substitution and vacancy): relaxing the atoms at each strain lowers C
    (clamped - relaxed is positive semi-definite) and every strained cell converges"""
    at = bulk("W", "bcc", a=3.17, cubic=True).repeat(2)
    at.symbols[0] = "Mo"
    del at[5]
    at = relax_batch(model, [at], fmax=1e-4, cell_filter="frechet", device="cpu")[0]["atoms"]
    clamped = elastic_batch(model, [at], relax_atoms=False, device="cpu")[0]
    relaxed = elastic_batch(model, [at], device="cpu")[0]
    assert relaxed["converged"].all() and relaxed["n_steps"].shape == (6, 5)
    assert relaxed["n_steps"][:, 2].max() <= 1    # the unstrained input is relaxed already
    d = np.linalg.eigvalsh(clamped["C"] - relaxed["C"])
    assert d.min() >= -0.5 and d.max() >= 1.0


@needs_model
def test_elastic_errors(model, w):
    with pytest.raises(ValueError, match="num_of_point"):
        elastic_batch(model, [w], num_of_point=1)
    mol = w.copy()
    mol.cell[0] = 0.0
    with pytest.raises(ValueError, match="structure 0 needs a cell of rank 3"):
        elastic_batch(model, [mol])
    with pytest.raises(ValueError, match="elastic_batch sets energy_only itself"):
        elastic_batch(model, [w], relax_atoms=False, energy_only=True)


def test_elastic_singular_keeps_keys():
    """a zero-stiffness direction (singular C): the Reuss and Hill values are None, not missing"""
    C = np.diag([300.0, 300.0, 300.0, 100.0, 100.0, 0.0])
    strains = np.linspace(-0.005, 0.005, 5)
    r = _elastic_from_stresses(strains, np.array([[C[:, j] * e for e in strains] for j in range(6)]))
    assert r["BH"] is None and r["S"] is None and np.isclose(r["BV"], 100.0)


def test_eos_and_elastic_range_validation():
    at = bulk("W", "bcc", a=3.17, cubic=True)
    for bad in (0.0, -0.1, 1.0, 1.5, float("nan"), float("inf"), (30.0,), (30.0, 32.0, 34.0),
                (30.0, 30.0), (0.0, 30.0), (-5.0, 30.0), (30.0, float("nan")), (30.0, float("inf"))):
        with pytest.raises(ValueError, match="volume_range"):
            eos_batch(None, [at], volume_range=bad)
    for bad in (0.0, -0.005, 1.0, 2.0, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="eps_range"):
            elastic_batch(None, [at], eps_range=bad)

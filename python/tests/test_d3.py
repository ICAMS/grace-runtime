# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Dispersion correction DFT-D3(BJ) on the CPU: GRACERuntimeCalculator(dispersion=...) minus the model alone equals
simple-dftd3 (tests/fixtures/d3_ref.npz; two-body and three-body cases), the dispersion_* results equal it directly,
compute_batch(dispersion=...) equals the calculator, energy-only evaluation, the argument forms and their errors."""
import numpy as np
import pytest
from ase.calculators.calculator import all_changes

from grace_runtime import D3, GRACERuntimeCalculator, _core, compute_batch

from d3_ref import cases

CASES = cases()
# elements of the GRACE test model: the heavy-element cluster (Cs, Ba, Au, U, Lr) only through _core directly
MODEL_CASES = [c for c in CASES if c["name"] != "heavy_cluster"]


def _close(a, b, atol):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=atol)


def _check_part(results, c, tol_e=1e-9, tol_f=1e-9):
    n = len(c["atoms"])
    _close(results["dispersion_energy"], c["energy"], tol_e * max(1.0, abs(c["energy"])))
    _close(results["dispersion_energies"], c["energies"], tol_e * max(1.0, abs(c["energy"])) / n * 10)
    if "dispersion_forces" in results:
        _close(results["dispersion_forces"], c["forces"], tol_f)
        if c["atoms"].pbc.all():
            _close(results["dispersion_stress"], c["stress"], tol_f)


@pytest.mark.parametrize("k", range(len(CASES)))
def test_core_against_simple_dftd3(k):
    c = CASES[k]
    at = c["atoms"]
    from grace_runtime.neighbors import _padded_cell
    d = _core.dispersion_compute(c["d3"].params(), at.numbers.astype(np.int32), at.positions,
                                 _padded_cell(at, c["d3"].reach + 1.0)[0], [bool(p) for p in at.pbc])
    _close(d["energy"], c["energy"], 1e-11 * abs(c["energy"]))
    _close(d["forces"], c["forces"], 1e-11)
    _close(d["energies"].sum(), d["energy"], 1e-12 * abs(c["energy"]))


@pytest.mark.parametrize("k", range(len(MODEL_CASES)))
def test_calculator_adds_dispersion(models, k):
    c = MODEL_CASES[k]
    at = c["atoms"]
    props = ["energy", "forces", "stress", "energies"]
    base = GRACERuntimeCalculator(models["1l"], device="cpu")
    base.calculate(at.copy(), props, all_changes)
    calc = GRACERuntimeCalculator(models["1l"], device="cpu", dispersion=c["d3"])
    calc.calculate(at.copy(), props, all_changes)
    r, b = calc.results, base.results
    _check_part(r, c)
    _close(r["energy"] - b["energy"], c["energy"], 1e-9 * max(1.0, abs(c["energy"])))
    _close(r["free_energy"], r["energy"], 0)
    _close(r["forces"] - b["forces"], c["forces"], 1e-9)
    _close(r["energies"] - b["energies"], c["energies"], 1e-9)
    _close(r["stress"] - b["stress"], r["dispersion_stress"], 1e-12)
    for key in ("dispersion_energy", "dispersion_forces", "dispersion_stress", "dispersion_energies"):
        assert key not in b


def test_no_dispersion_unchanged(models):
    at = MODEL_CASES[6]["atoms"]
    a = GRACERuntimeCalculator(models["1l"], device="cpu")
    b = GRACERuntimeCalculator(models["1l"], device="cpu", dispersion=None)
    for calc in (a, b):
        calc.calculate(at.copy(), ["energy", "forces", "stress"], all_changes)
    for key in ("energy", "forces", "stress"):    # round-off only: the model's OpenMP sums are not bitwise repeatable
        np.testing.assert_allclose(a.results[key], b.results[key], rtol=1e-13, atol=1e-13)
    assert not [k for k in b.results if k.startswith("dispersion")]


def test_energy_only_then_forces(models):
    c = MODEL_CASES[0]
    at = c["atoms"].copy()
    at.calc = GRACERuntimeCalculator(models["1l"], device="cpu", dispersion=c["d3"], energy_only=True)
    e = at.get_potential_energy()
    assert "forces" not in at.calc.results and "dispersion_forces" not in at.calc.results
    _close(at.calc.get_property("dispersion_energy", at), c["energy"], 1e-9 * abs(c["energy"]))
    f = at.calc.get_property("dispersion_forces", at)    # computes forces now
    _close(f, c["forces"], 1e-9)
    _close(at.get_potential_energy(), e, 1e-12 * abs(e))
    assert "forces" in at.calc.results


def test_energies_then_forces(models):
    """per-atom energies of an earlier call get the correction once: a later forces-only call keeps them"""
    c = MODEL_CASES[0]
    at = c["atoms"].copy()
    at.calc = GRACERuntimeCalculator(models["1l"], device="cpu", dispersion=c["d3"], energy_only=True)
    e1 = at.get_potential_energies().copy()
    at.get_forces()
    _close(at.get_potential_energies(), e1, 1e-12)


def test_compute_batch_cpu(models):
    cs = MODEL_CASES[:4]
    res = compute_batch(models["1l"], [c["atoms"] for c in cs], device="cpu", dispersion=cs[0]["d3"],
                        dispersion_parts=True, energies=True)
    calc = GRACERuntimeCalculator(models["1l"], device="cpu", dispersion=cs[0]["d3"])
    for c, r in zip(cs, res):
        calc.calculate(c["atoms"].copy(), ["energy", "forces", "stress", "energies"], all_changes)
        for key in ("energy", "forces", "stress", "energies", "dispersion_energy", "dispersion_forces",
                    "dispersion_stress", "dispersion_energies"):
            _close(r[key], calc.results[key], 1e-12 * max(1.0, abs(calc.results["energy"])))
    plain = compute_batch(models["1l"], [cs[0]["atoms"]], device="cpu", dispersion="d3-pbe")
    assert "dispersion_energy" in plain[0] and "dispersion_forces" not in plain[0]


def test_argument_forms(models):
    assert GRACERuntimeCalculator(models["1l"], device="cpu", dispersion="d3-PBE").dispersion.functional == "pbe"
    d = D3("r2scan")
    assert (d.a1, d.s8, d.a2, d.s6) == (0.49484001, 0.78981345, 5.73083694, 1.0)
    assert D3("pbe").params()["a2"] == pytest.approx(4.4407 * 0.529177210903, rel=1e-15)
    assert D3("pbe").params()["cutoff_on"] == 12.0
    with pytest.raises(ValueError, match="d3-pbe"):
        GRACERuntimeCalculator(models["1l"], device="cpu", dispersion="pbe")
    with pytest.raises(ValueError, match="unknown dispersion"):
        GRACERuntimeCalculator(models["1l"], device="cpu", dispersion="d4-r2scan-x")
    with pytest.raises(ValueError, match="unknown D3"):
        D3("b3lyp-unknown")
    with pytest.raises(TypeError):
        GRACERuntimeCalculator(models["1l"], device="cpu", dispersion=3)
    with pytest.raises(ValueError, match="not both"):
        D3("pbe", a1=0.4)
    with pytest.raises(ValueError, match="all of a1, a2, s8"):
        D3(a1=0.4, a2=4.0)
    with pytest.raises(ValueError, match="switch_width"):
        D3("pbe", cutoff=10.0, switch_width=10.0)
    # the three-body term: off unless asked ("d3-pbe" keeps its meaning), "-atm" turns it on
    assert not D3("pbe").atm and D3("pbe").params()["s9"] == 0.0
    a = GRACERuntimeCalculator(models["1l"], device="cpu", dispersion="d3-r2scan-ATM").dispersion
    assert a.atm and a.functional == "r2scan" and a.params()["s9"] == 1.0 and a.reach == 15.0
    assert D3("pbe", atm=True, atm_cutoff=20.0, atm_switch_width=3.0).params()["atm_cutoff_on"] == 17.0
    assert D3("pbe", atm=True, atm_cutoff=20.0).reach == 20.0
    with pytest.raises(ValueError, match="unknown dispersion"):
        GRACERuntimeCalculator(models["1l"], device="cpu", dispersion="d3-pbe-abc")
    with pytest.raises(ValueError, match="atm_switch_width"):
        D3("pbe", atm=True, atm_cutoff=8.0, atm_switch_width=8.0)
    with pytest.raises(ValueError, match="atm_switch_width"):
        D3("pbe", atm=True, atm_cutoff=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        D3("pbe", cutoff=float("inf"))
    for a1, a2 in ((0.0, 0.0), (-0.1, 4.0), (0.4, float("nan"))):
        with pytest.raises(ValueError, match="damping radius"):
            D3(a1=a1, a2=a2, s8=0.8)


def test_three_body_keys_optional():
    # the modules' parameter dict without s9, atm_cutoff, atm_cutoff_on: the three-body term off
    p = D3("pbe").params()
    q = {k: v for k, v in p.items() if k not in ("s9", "atm_cutoff", "atm_cutoff_on")}
    args = (np.array([6, 6, 8], np.int32), np.array([[0.0, 0, 0], [0, 0, 1.4], [1.1, 0, 0.3]]), np.eye(3) * 40.0,
            [False, False, False])
    assert _core.dispersion_compute(q, *args)["energy"] == _core.dispersion_compute(p, *args)["energy"]


def test_element_without_reference_data():
    p = D3("pbe").params()
    with pytest.raises(ValueError, match="atomic number 104"):
        _core.dispersion_compute(p, np.array([6, 104], np.int32), np.array([[0.0, 0, 0], [0, 0, 2.0]]),
                                 np.eye(3) * 40.0, [False, False, False])

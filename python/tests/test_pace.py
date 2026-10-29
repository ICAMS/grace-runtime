# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""PACE models on the CPU engine against pyace fp64 (tests/fixtures/pace/ref_pace.npz, efs/...): with the radial
functions evaluated directly (Model(spline=0)) to the accuracy of pyace's splines, with the default radial tables within
the gate of test_ref_tf.py (the engine is fp64; in fp32 its products miss that gate by orders of magnitude)."""
import os

import numpy as np
import pytest
from ase import Atoms

from grace_runtime import GRACERuntimeCalculator, Model
from pace_tools import FX, MODELS, cell, cells, model_path, write_npz
from test_ref_tf import TOL as GATE, TOL_SMALL as GATE_SMALL

REF = np.load(os.path.join(FX, "ref_pace.npz"))
CASES = [(c.info["model"], c.info["name"].split("/")[1]) for c in cells()]


@pytest.fixture(scope="session")
def model_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("pace")
    return {name: model_path(name, d) for name in MODELS + ["al_cheblinear"]}


def evaluate(path, at, spline=0):
    at.calc = GRACERuntimeCalculator(Model(path, spline=spline), device="cpu", num_threads=2)
    return {"energy": at.get_potential_energy(), "forces": at.get_forces(), "energies": at.get_potential_energies(),
            "stress": at.get_stress() if at.pbc.all() else np.zeros(6)}


def ref(model, name):
    return {k: REF[f"efs/{model}/{name}/{k}"] for k in ("energy", "forces", "energies", "stress")}


@pytest.mark.parametrize("model,name", CASES)
def test_matches_pyace(model, name, model_files):
    at = cell(f"{model}/{name}")
    n = len(at)
    got, want = evaluate(model_files[model], at), ref(model, name)
    scale = lambda v: max(1.0, np.abs(v).max())
    assert abs(got["energy"] - want["energy"]) / n < 1e-10 * scale(want["energy"] / n)
    assert np.abs(got["energies"] - want["energies"]).max() < 1e-9 * scale(want["energies"])
    # pyace's spline derivatives near the C1 kink of the cos cutoff converge to first order only (ethanol_nd1: 1.4e-8
    # between the reference's DeltaSplineBins 1e-4 and 5e-5); test_forces_are_gradient checks
    # the forces against the engine's own energy
    assert np.abs(got["forces"] - want["forces"]).max() < 3e-8 * scale(want["forces"])
    assert np.abs(got["stress"] - want["stress"]).max() < 1e-9 * scale(want["stress"])


@pytest.mark.parametrize("model,name", CASES)
def test_default_tables_within_gate(model, name, model_files):
    """the default radial tables (1024 intervals per pair) against pyace, within the fp32 gate"""
    at = cell(f"{model}/{name}")
    n = len(at)
    got, want = evaluate(model_files[model], at, spline=1024), ref(model, name)
    d = {"energy": abs(got["energy"] - want["energy"]) / n,
         "forces": np.abs(got["forces"] - want["forces"]).max(),
         "energies": np.abs(got["energies"] - want["energies"]).max(),
         "stress": (np.abs(got["stress"] - want["stress"]).max() / np.abs(want["stress"]).max()
                    if np.abs(want["stress"]).max() > 0 else 0.0)}
    for q, v in d.items():
        assert v < (GATE_SMALL if q == "energy" and n < 10 else GATE[q]), (q, v)


@pytest.mark.parametrize("model", ["carbon_zbl", "pdh_zbl", "au_chebpow_density", "ho_nd5", "mnbtw_4el",
                                   "ethanol_nd1"])
@pytest.mark.parametrize("spline", [0, 1024])
def test_forces_are_gradient(model, model_files, spline):
    at = cell(f"{model}/bulk")
    m = Model(model_files[model], spline=spline)
    at.calc = GRACERuntimeCalculator(m, device="cpu", num_threads=2)
    f = at.get_forces()
    h = 1e-4    # smaller steps see the round-off of the energy (~1e-11 eV, as pyace): 1e-6 eV/A at h = 1e-6
    for a, d in ((0, 0), (len(at) // 2, 1), (len(at) - 1, 2)):
        e = []
        for s in (h, -h):
            b = at.copy()
            b.positions[a, d] += s
            b.calc = GRACERuntimeCalculator(m, device="cpu", num_threads=2)
            e.append(b.get_potential_energy())
        assert -(e[0] - e[1]) / (2 * h) == pytest.approx(f[a, d], abs=1e-6)


def test_isolated_atom(model_files):
    """Review focus 1: no neighbour (rho = 0): finite energy as pyace, zero forces"""
    for model in ("al_fs_plain", "carbon_zbl", "ho_nd5"):
        at = cell(f"{model}/lone")
        got = evaluate(model_files[model], at)
        assert np.isfinite(got["energy"]) and got["energy"] == pytest.approx(float(ref(model, "lone")["energy"]),
                                                                             abs=1e-6)
        assert not np.abs(got["forces"]).any()


def test_subset_and_unknown_element(model_files):
    """Review focus 3: a structure with part of the model's elements; an element the model lacks"""
    at = cell("pdh_zbl/bulk")
    pd = at[[s == "Pd" for s in at.get_chemical_symbols()]]
    got = evaluate(model_files["pdh_zbl"], pd)
    assert np.isfinite(got["energy"])
    with pytest.raises(ValueError, match="Cu"):
        evaluate(model_files["pdh_zbl"], Atoms("Cu2", positions=[[0, 0, 0], [2, 0, 0]], cell=[9, 9, 9], pbc=True))


def test_coincident_atoms_are_an_error(model_files):
    """two atoms at the same position (a bond of length 0) are not a physical structure, as for the GRACE engines"""
    at = cell("carbon_zbl/bulk")
    at.positions[1] = at.positions[0]
    with pytest.raises(ValueError, match="same position"):
        evaluate(model_files["carbon_zbl"], at)


def test_self_images(model_files):
    """Review focus 4: a 2-atom cell that sees its images equals its 2x2x2 supercell per atom"""
    at = cell("carbon_zbl/pair2")
    a = evaluate(model_files["carbon_zbl"], at.copy())
    b = evaluate(model_files["carbon_zbl"], at * (2, 2, 2))
    assert b["energy"] / 16 == pytest.approx(a["energy"] / 2, abs=1e-10)
    np.testing.assert_allclose(b["stress"], a["stress"], atol=1e-10)


def test_energy_only(model_files):
    at = cell("carbon_zbl/bulk")
    m = Model(model_files["carbon_zbl"])
    at.calc = GRACERuntimeCalculator(m, device="cpu", num_threads=2, energy_only=True)
    e = at.get_potential_energy()
    assert "forces" not in at.calc.results
    at.calc = GRACERuntimeCalculator(m, device="cpu", num_threads=2)
    at.get_forces()
    assert at.calc.results["energy"] == pytest.approx(e, abs=1e-9)


def test_yaml_and_npz_same(model_files, tmp_path):
    at = cell("mnbtw_4el/bulk")
    npz = write_npz(model_files["mnbtw_4el"], tmp_path / "m.npz")
    r = []
    for p in (model_files["mnbtw_4el"], npz):
        b = at.copy()
        b.calc = GRACERuntimeCalculator(p, device="cpu", num_threads=2)
        r.append((b.get_potential_energy(), b.get_forces()))
    assert r[0][0] == r[1][0]
    np.testing.assert_array_equal(r[0][1], r[1][1])


def test_yaml_converted_on_the_fly_without_cache(tmp_path, monkeypatch):
    """a .yaml is converted on the fly on load: Model.path stays the .yaml path and GRACE_CACHE is untouched"""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    monkeypatch.setenv("GRACE_CACHE", str(blocker))
    src = os.path.join(FX, "al_fs_plain.yaml")
    m = Model(src)
    assert m.path == src
    at = cell("al_fs_plain/bulk")
    assert evaluate(m.path, at.copy())["energy"] == pytest.approx(float(ref("al_fs_plain", "bulk")["energy"]), abs=1e-6)


def test_one_and_four_threads_agree(model_files):
    """energy, forces, stress and per-atom energies on 1 and 4 threads: the same up to the force reduction order"""
    at = cell("carbon_zbl/bulk")
    r = []
    for nth in (1, 4):
        a = at.copy()
        a.calc = GRACERuntimeCalculator(Model(model_files["carbon_zbl"]), device="cpu", num_threads=nth)
        r.append((a.get_potential_energy(), a.get_potential_energies(), a.get_forces(), a.get_stress()))
    for x, y in zip(*r):
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-11 * max(1.0, np.abs(y).max()))

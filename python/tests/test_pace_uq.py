# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""The extrapolation grade gamma of PACE models (an active set from Model(yaml, asi=...) or in the npz)
against pyace references (PyACECalculator.set_active_set), with the
deterministic active set of pace_tools.active_set_file: linear and non-linear grade, every fixture model and cell."""
import os

import numpy as np
import pytest

from grace_runtime import GRACERuntimeCalculator, Model
from pace_tools import FX, active_set_file, basis_sizes, cell, write_npz

REF = np.load(os.path.join(FX, "ref_pace_uq.npz"))
CASES = sorted({tuple(k.split("/")[1:]) for k in REF.files})    # (model, cell, grade)


def asi_for(model, grade, d):
    nfunc, nd = basis_sizes(os.path.join(FX, f"{model}.yaml"))
    return active_set_file(os.path.join(d, f"{model}_{grade}.asi"), nfunc, nd, int(grade))


@pytest.mark.parametrize("model,name,grade", CASES)
def test_gamma_matches_pyace(model, name, grade, tmp_path):
    at = cell(f"{model}/{name}")
    m = Model(os.path.join(FX, f"{model}.yaml"), asi=asi_for(model, grade, tmp_path))
    at.calc = GRACERuntimeCalculator(m, device="cpu", num_threads=2, uq=True, energy_only=True)
    at.get_potential_energy()
    got, want = at.calc.results["gamma"], REF[f"gamma/{model}/{name}/{grade}"]
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9 * max(1.0, np.abs(want).max()))


def test_gamma_from_converted_npz(tmp_path):
    model, name, grade = CASES[0]
    npz = write_npz(os.path.join(FX, f"{model}.yaml"), tmp_path / "m.npz", asi=asi_for(model, grade, tmp_path))
    at = cell(f"{model}/{name}")
    at.calc = GRACERuntimeCalculator(Model(npz), device="cpu", uq=True, energy_only=True)
    at.get_potential_energy()
    np.testing.assert_allclose(at.calc.results["gamma"], REF[f"gamma/{model}/{name}/{grade}"], rtol=1e-6, atol=1e-9)
    assert "atomic_sigma" not in at.calc.results


def test_uq_without_active_set_is_refused():
    with pytest.raises(ValueError, match="active set"):
        GRACERuntimeCalculator(Model(os.path.join(FX, "carbon_zbl.yaml")), device="cpu", uq=True)


def test_active_set_of_wrong_size_is_refused(tmp_path):
    p = str(tmp_path / "bad.asi")
    with open(p, "wb") as f:
        np.savez(f, C=np.eye(3))
    with pytest.raises(ValueError, match="size"):
        Model(os.path.join(FX, "carbon_zbl.yaml"), asi=p)


def test_asi_with_npz_is_refused(tmp_path):
    npz = write_npz(os.path.join(FX, "carbon_zbl.yaml"), tmp_path / "m.npz")
    with pytest.raises(ValueError, match="asi"):
        Model(npz, asi=str(tmp_path / "x.asi"))


@pytest.mark.parametrize("nth", [1, 4])
def test_gamma_with_forces_on_one_and_four_threads(nth, tmp_path):
    """gamma alongside energy, forces and stress (the full path; the tests above request the energy only)"""
    model, name, grade = next(c for c in CASES if c[2] == "1")
    at = cell(f"{model}/{name}")
    at.calc = GRACERuntimeCalculator(Model(os.path.join(FX, f"{model}.yaml"), asi=asi_for(model, grade, tmp_path)),
                                     device="cpu", num_threads=nth, uq=True)
    at.get_forces(), at.get_stress()
    want = REF[f"gamma/{model}/{name}/{grade}"]
    np.testing.assert_allclose(at.calc.results["gamma"], want, rtol=1e-6, atol=1e-9 * max(1.0, np.abs(want).max()))

# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""The extrapolation grade of PACE models on the GPU against pyace (references of test_pace_uq.py) and the CPU
engine: every fixture model and cell, linear and non-linear grade, the calculator and the batch path."""
import os

import numpy as np
import pytest

from grace_runtime import GRACERuntimeCalculator, Model, compute_batch
from gpu_tools import GPU, NO_GPU, gpu_ok
from pace_tools import FX, cell
from test_pace_uq import CASES, REF, asi_for

pytestmark = pytest.mark.skipif(not gpu_ok(), reason=NO_GPU)


def gamma(model, at, device):
    at.calc = GRACERuntimeCalculator(model, device=device, uq=True, energy_only=True)
    at.get_potential_energy()
    return at.calc.results["gamma"]


@pytest.mark.parametrize("model,name,grade", CASES)
def test_gpu_gamma_matches_pyace_and_cpu(model, name, grade, tmp_path):
    m = Model(os.path.join(FX, f"{model}.yaml"), asi=asi_for(model, grade, tmp_path))
    g, c = gamma(m, cell(f"{model}/{name}"), GPU), gamma(m, cell(f"{model}/{name}"), "cpu")
    want = REF[f"gamma/{model}/{name}/{grade}"]
    np.testing.assert_allclose(g, want, rtol=1e-6, atol=1e-9 * max(1.0, np.abs(want).max()))
    np.testing.assert_allclose(g, c, rtol=1e-10, atol=1e-12 * max(1.0, np.abs(c).max()))


def test_gpu_gamma_batch(tmp_path):
    model, _, grade = CASES[0]
    m = Model(os.path.join(FX, f"{model}.yaml"), asi=asi_for(model, grade, tmp_path))
    ats = [cell(k) for k in sorted({f"{mo}/{na}" for mo, na, _ in CASES if mo == model})]
    res = compute_batch(m, ats, device=GPU, uq=True)
    for at, r in zip(ats, res):
        np.testing.assert_allclose(r["gamma"], gamma(m, at.copy(), "cpu"), rtol=1e-10, atol=1e-12)
        assert "atomic_sigma" not in r


def test_gpu_gamma_with_forces(tmp_path):
    """gamma alongside energy, forces and stress on the GPU (the tests above request the energy only)"""
    model, name, grade = next(c for c in CASES if c[2] == "1")
    m = Model(os.path.join(FX, f"{model}.yaml"), asi=asi_for(model, grade, tmp_path))
    at = cell(f"{model}/{name}")
    at.calc = GRACERuntimeCalculator(m, device=GPU, uq=True)
    at.get_forces(), at.get_stress()
    np.testing.assert_allclose(at.calc.results["gamma"], gamma(m, cell(f"{model}/{name}"), "cpu"), rtol=1e-10, atol=1e-12)

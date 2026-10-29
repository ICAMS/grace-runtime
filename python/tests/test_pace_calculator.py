# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""PACE models in GRACERuntimeCalculator and the batch functions on the CPU: no UQ, threads, dispersion."""
import os

import numpy as np
import pytest

from grace_runtime import GRACERuntimeCalculator, Model, compute_batch, relax_batch
from pace_tools import FX, cell

CARBON = os.path.join(FX, "carbon_zbl.yaml")


def results(calc, at):
    at = at.copy()
    at.calc = calc
    return at.get_potential_energy(), at.get_forces(), at.get_stress()


def test_uq_raises():
    with pytest.raises(ValueError, match="PACE"):
        GRACERuntimeCalculator(CARBON, device="cpu", uq=True)
    calc = GRACERuntimeCalculator(CARBON, device="cpu")
    with pytest.raises(ValueError, match="PACE"):
        calc.enable_uq()


def test_threads_identical():
    """Review focus 5: the number of threads does not change the results (chunksize has no effect on the CPU)"""
    at = cell("carbon_zbl/bulk")
    m = Model(CARBON)
    ref = results(GRACERuntimeCalculator(m, device="cpu", num_threads=1), at)
    for nth in (2, 3, 7):
        got = results(GRACERuntimeCalculator(m, device="cpu", num_threads=nth, chunksize=7), at)
        assert got[0] == pytest.approx(ref[0], rel=1e-12)
        np.testing.assert_allclose(got[1], ref[1], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got[2], ref[2], rtol=1e-12, atol=1e-14)


def test_compute_batch():
    cells = [cell("carbon_zbl/bulk"), cell("carbon_zbl/pair2")]
    out = compute_batch(CARBON, cells, device="cpu")
    for at, r in zip(cells, out):
        e, f, s = results(GRACERuntimeCalculator(CARBON, device="cpu"), at)
        assert r["energy"] == pytest.approx(e, abs=1e-9)
        np.testing.assert_allclose(r["forces"], f, atol=1e-9)


def test_relax_batch_cpu():
    at = cell("carbon_zbl/bulk")
    at.rattle(0.05, seed=3)
    out = relax_batch(CARBON, [at], device="cpu", fmax=0.05, max_steps=300)
    assert out[0]["converged"], out[0]


def test_dispersion_added():
    at = cell("carbon_zbl/bulk")
    m = Model(CARBON)
    e0 = results(GRACERuntimeCalculator(m, device="cpu"), at)[0]
    b = at.copy()
    b.calc = GRACERuntimeCalculator(m, device="cpu", dispersion="d3-pbe")
    e = b.get_potential_energy()
    assert b.calc.results["dispersion_energy"] < 0
    assert e == pytest.approx(e0 + b.calc.results["dispersion_energy"], abs=1e-9)


def test_compute_batch_uq_raises():
    with pytest.raises(ValueError, match="PACE"):
        compute_batch(CARBON, [cell("carbon_zbl/bulk")], device="cpu", uq=True)


def test_torchsim_uq_raises():
    pytest.importorskip("torch")
    pytest.importorskip("torch_sim")
    from grace_runtime.torchsim import GRACERuntimeModel
    with pytest.raises(ValueError, match="PACE"):
        GRACERuntimeModel(CARBON, device="cpu", uq=True)

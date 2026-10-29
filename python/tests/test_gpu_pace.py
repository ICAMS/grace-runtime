# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""PACE models on the GPU engine against the CPU engine, both fp64 with the same radial tables, on the fixtures of
test_pace.py: per-atom energies, forces, stress; chunks, species subsets, the batch path, UQ refused. Different
summation orders only: 1e-10 eV/atom, forces and stress 1e-9 relative to max(1, the largest value)."""
import os

import numpy as np
import pytest
from ase import Atoms

from grace_runtime import GRACERuntimeCalculator, Model, compute_batch
from gpu_tools import GPU, NO_GPU, gpu_ok
from pace_tools import MODELS, cell, cells, model_path

pytestmark = pytest.mark.skipif(not gpu_ok(), reason=NO_GPU)
CASES = [(c.info["model"], c.info["name"].split("/")[1]) for c in cells()]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("pace")
    return {name: Model(model_path(name, d)) for name in MODELS + ["al_cheblinear"]}


def evaluate(model, at, device, **kw):
    at.calc = GRACERuntimeCalculator(model, device=device, **kw)
    return {"energies": at.get_potential_energies(), "energy": at.get_potential_energy(), "forces": at.get_forces(),
            "stress": at.get_stress() if at.pbc.all() else np.zeros(6), "backend": at.calc.backend}


def close(got, want):
    n = max(len(want["forces"]), 1)
    scale = lambda v: max(1.0, float(np.abs(v).max()) if np.size(v) else 1.0)
    assert abs(got["energy"] - want["energy"]) / n < 1e-10 * scale(want["energy"] / n)
    np.testing.assert_allclose(got["energies"], want["energies"], rtol=0, atol=1e-9 * scale(want["energies"]))
    np.testing.assert_allclose(got["forces"], want["forces"], rtol=0, atol=1e-9 * scale(want["forces"]))
    np.testing.assert_allclose(got["stress"], want["stress"], rtol=0, atol=1e-9 * scale(want["stress"]))


@pytest.mark.parametrize("model,name", CASES)
def test_gpu_matches_cpu(models, model, name):
    at = cell(f"{model}/{name}")
    got = evaluate(models[model], at.copy(), GPU)
    assert got["backend"].startswith(("cuda", "rocm"))
    close(got, evaluate(models[model], at.copy(), "cpu"))


@pytest.mark.parametrize("model", ["pdh_zbl", "ni_lmax6", "ethanol_nd1"])
def test_gpu_matches_cpu_direct(tmp_path, model):
    """spline=0: the radial functions evaluated directly on both"""
    m = Model(model_path(model, tmp_path), spline=0)
    at = cell(f"{model}/bulk")
    close(evaluate(m, at.copy(), GPU), evaluate(m, at.copy(), "cpu"))


def test_gpu_energy_only(models):
    """energy_only on the GPU: the energy of the full path, no forces computed"""
    at = cell("carbon_zbl/bulk")
    at.calc = GRACERuntimeCalculator(models["carbon_zbl"], device=GPU, energy_only=True)
    e = at.get_potential_energy()
    assert "forces" not in at.calc.results
    assert e == pytest.approx(evaluate(models["carbon_zbl"], at.copy(), "cpu")["energy"], abs=1e-9)


def test_chunks_identical(models):
    """atoms in chunks of 7: the same results (forces summed over the chunks)"""
    at = cell("mnbtw_4el/bulk") * (2, 1, 1)
    close(evaluate(models["mnbtw_4el"], at.copy(), GPU, chunksize=7),
          evaluate(models["mnbtw_4el"], at.copy(), GPU))


def test_species_subset_and_unknown(models):
    at = cell("pdh_zbl/bulk")
    pd = at[[a.index for a in at if a.symbol == "Pd"]]
    close(evaluate(models["pdh_zbl"], pd.copy(), GPU), evaluate(models["pdh_zbl"], pd.copy(), "cpu"))
    with pytest.raises(ValueError, match="Cu"):
        evaluate(models["pdh_zbl"], Atoms("Cu2", positions=[[0, 0, 0], [2, 0, 0]], cell=[9, 9, 9], pbc=True), GPU)


def test_batch_matches_cpu(models):
    m = models["pdh_zbl"]
    structs = [cell(f"pdh_zbl/{n}") for n in ("bulk", "pair2", "cluster")] + [cell("pdh_zbl/bulk") * (1, 1, 2)]
    for s in structs:
        s.rattle(0.02, seed=len(s))
    gpu = compute_batch(m, structs, device=GPU, energies=True)
    cpu = compute_batch(m, structs, device="cpu", energies=True)
    for g, c in zip(gpu, cpu):
        close({k: g[k] for k in ("energy", "energies", "forces", "stress")},
              {k: c[k] for k in ("energy", "energies", "forces", "stress")})


def test_uq_raises(models):
    with pytest.raises(ValueError, match="PACE"):
        GRACERuntimeCalculator(models["carbon_zbl"], device=GPU, uq=True)


def test_yaml_path_on_gpu(tmp_path, monkeypatch):
    """a .yaml loads directly on the GPU without writing to GRACE_CACHE"""
    cache = tmp_path / "cache"
    monkeypatch.setenv("GRACE_CACHE", str(cache))
    path = model_path("al_fs_plain", tmp_path)
    at = cell("al_fs_plain/bulk")
    close(evaluate(path, at.copy(), GPU), evaluate(path, at.copy(), "cpu"))
    assert not cache.exists()


def test_zbl_equidistant_neighbours(models):
    """the middle atom of a symmetric C3 chain has two neighbours at 0.85 A, inside the ZBL switch (0.8 .. 0.9 A): both
    devices give the switch's force to the same (the first) neighbour"""
    at = Atoms("C3", positions=[[5, 5, 5], [5.85, 5, 5], [6.7, 5, 5]], cell=[12, 12, 12], pbc=True)
    close(evaluate(models["carbon_zbl"], at.copy(), GPU), evaluate(models["carbon_zbl"], at.copy(), "cpu"))


def test_coincident_atoms(models):
    """two atoms at one position (a bond of length 0) are not a physical structure: an error on both devices"""
    at = cell("carbon_zbl/bulk")
    at.positions[1] = at.positions[0]
    for device in (GPU, "cpu"):
        with pytest.raises(ValueError, match="same position"):
            evaluate(models["carbon_zbl"], at.copy(), device)

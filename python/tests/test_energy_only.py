# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Energy-only evaluation on CPU (ComputeFlags::energy_only, as LAMMPS eflag_only): the calculator skips the backward
pass when ASE asks for no forces / stress; the energy is that of the full evaluation (1 thread: same operations)."""
import os

import numpy as np
import pytest
from ase.io import read

from grace_runtime import GRACERuntimeCalculator, Model
from grace_runtime.neighbors import NeighborList

CACHE = os.environ.get("GRACE_CACHE", os.path.expanduser("~/.cache/grace-runtime-test"))
FTE = f"{CACHE}/GRACE-2L-OMAT-medium-ft-E/kokkos.npz"    # 2L with a UQ head


def wmo54(fixtures):
    at = read(f"{fixtures}/WMo54.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})
    at.rattle(0.03, seed=7)
    return at


@pytest.mark.parametrize("arch", ["1l", "2l", "3l"])
def test_energy_without_forces(models, fixtures, arch):
    at = wmo54(fixtures)
    full = at.copy()
    full.calc = GRACERuntimeCalculator(models[arch], num_threads=1, energy_only=False)
    e_full, f_full = full.get_potential_energy(), full.get_forces()
    at.calc = GRACERuntimeCalculator(models[arch], num_threads=1, energy_only=True)
    e = at.get_potential_energy()
    assert "forces" not in at.calc.results
    assert abs(e - e_full) / len(at) <= 1e-9
    assert np.abs(at.get_forces() - f_full).max() <= 1e-9    # computed on demand
    assert abs(at.get_potential_energy() - e_full) / len(at) <= 1e-9


@pytest.mark.parametrize("arch", ["1l", "2l", "3l"])
def test_default_caches_energy_forces_and_stress(models, fixtures, arch, monkeypatch):
    at = wmo54(fixtures)
    at.calc = GRACERuntimeCalculator(models[arch], num_threads=1)
    energy = at.get_potential_energy()
    assert "forces" in at.calc.results
    assert "stress" in at.calc.results
    forces, stress = at.calc.results["forces"].copy(), at.calc.results["stress"].copy()

    def unexpected_calculation(*args, **kwargs):
        pytest.fail("unchanged atoms should reuse the energy evaluation's forces and stress")

    monkeypatch.setattr(at.calc, "calculate", unexpected_calculation)
    assert np.array_equal(at.get_forces(), forces)
    assert np.array_equal(at.get_stress(), stress)
    assert at.get_potential_energy() == energy


def test_energies_property_energy_only(models, fixtures):
    at = wmo54(fixtures)
    at.calc = GRACERuntimeCalculator(models["2l"], num_threads=1, energy_only=True)
    ei = at.get_potential_energies()
    assert "forces" not in at.calc.results
    assert abs(ei.sum() - at.get_potential_energy()) <= 1e-6


def test_model_compute_energy_only(models, fixtures):
    at = wmo54(fixtures)
    m = Model(models["1l"])
    rc = float(m.cutoff_matrix(["Mo", "W"]).max())
    nl = NeighborList(rc, 0.0, num_threads=1)
    nl.update(at)
    types = np.ascontiguousarray(np.where(at.numbers[nl.perm] == 42, 1, 2), dtype=np.int32)
    args = (["Mo", "W"], types, nl.offset, nl.j, nl.rij(at, 1))
    full = m.compute(*args, num_threads=1)
    eo = m.compute(*args, num_threads=1, energy_only=True)
    assert "forces" not in eo and "virial" not in eo
    assert abs(eo["energy"] - full["energy"]) <= 1e-9 * len(at)


def test_uq_with_energy_only(fixtures):
    if not os.path.exists(FTE):
        pytest.skip("UQ model absent")
    at = wmo54(fixtures)
    ref = at.copy()
    ref.calc = GRACERuntimeCalculator(FTE, num_threads=1, uq=True, energy_only=False)
    ref.get_potential_energy()
    at.calc = GRACERuntimeCalculator(FTE, num_threads=1, uq=True, energy_only=True)
    at.get_potential_energy()
    assert "forces" not in at.calc.results
    for k in ("gamma", "atomic_sigma", "gmm_cluster"):
        assert np.abs(at.calc.results[k] - ref.calc.results[k]).max() <= 1e-9 * max(1.0, np.abs(ref.calc.results[k]).max())


def test_direct_calculate_does_not_keep_other_structures_forces(models, fixtures):
    """calculate() on new atoms with energy only must drop the previous structure's forces / stress"""
    from ase.calculators.calculator import all_changes
    a1 = wmo54(fixtures)
    a2 = a1.copy()
    a2.positions[0] += 0.2
    calc = GRACERuntimeCalculator(models["1l"], num_threads=1, energy_only=True)
    calc.calculate(a1, ["forces"], all_changes)
    calc.calculate(a2, ["energy"], all_changes)
    ref = a2.copy()
    ref.calc = GRACERuntimeCalculator(models["1l"], num_threads=1)
    assert np.abs(calc.get_forces() - ref.get_forces()).max() <= 1e-9

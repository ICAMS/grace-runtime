# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime.phonopy_forces_batch / phonons_batch: the forces and force constants are those of phonopy driven by
GRACERuntimeCalculator one supercell at a time; masses, the supercell from min_length, the errors."""
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from grace_runtime import GRACERuntimeCalculator, Model, phonons_batch, phonopy_forces_batch, relax_batch
from grace_runtime.phonons import _supercell_matrix


@pytest.fixture(scope="module")
def model(models):
    return Model(models["1l"])


@pytest.fixture(scope="module")
def crystals(model):
    """fcc Al (1 atom) and diamond Si (2 atoms), relaxed"""
    pytest.importorskip("phonopy", reason="phonopy not installed")
    cells = [bulk("Al", "fcc", a=4.04), bulk("Si", "diamond", a=5.43)]
    return [r["atoms"] for r in relax_batch(model, cells, fmax=1e-4, cell_filter="frechet", device="cpu")]


def _phonopy(at, n, **displacement_options):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    ph = Phonopy(PhonopyAtoms(numbers=at.numbers, cell=at.cell.array, scaled_positions=at.get_scaled_positions()),
                 supercell_matrix=[n, n, n])
    ph.generate_displacements(distance=0.01, **displacement_options)
    return ph


def _forces_one_at_a_time(model, ph):
    """the forces of the displaced supercells, each minus those of the undisplaced one"""
    calc = GRACERuntimeCalculator(model, device="cpu")

    def forces(s):
        return calc.get_forces(Atoms(numbers=s.numbers, cell=s.cell, scaled_positions=s.scaled_positions, pbc=True))
    residual = forces(ph.supercell)
    return np.array([forces(s) - residual for s in ph.supercells_with_displacements])


def test_phonons_batch_matches_phonopy_one_supercell_at_a_time(model, crystals):
    phonons = phonons_batch(model, crystals, supercell=2, device="cpu")
    for at, ph in zip(crystals, phonons):
        ref = _phonopy(at, 2)
        ref.forces = _forces_one_at_a_time(model, ref)
        ref.produce_force_constants()
        ref.symmetrize_force_constants()
        np.testing.assert_allclose(ph.force_constants, ref.force_constants, rtol=0, atol=1e-10)
        assert np.abs(ph.force_constants.sum(axis=1)).max() < 1e-8    # the acoustic sum rule
    al = phonons[0]
    al.run_mesh([8, 8, 8], is_gamma_center=True)
    assert al.mesh.frequencies.min() > -0.05    # fcc Al is stable


def test_phonopy_forces_batch_takes_any_displacements(model, crystals):
    phonons = [_phonopy(crystals[0], 2), _phonopy(crystals[1], 2, is_plusminus=True)]
    assert phonopy_forces_batch(model, phonons, device="cpu") == phonons
    for ph in phonons:
        np.testing.assert_allclose(ph.forces, _forces_one_at_a_time(model, ph), rtol=0, atol=1e-12)


def test_masses(model, crystals):
    heavy = crystals[0].copy()
    heavy.set_masses(2 * heavy.get_masses())    # e.g. an isotope: frequencies scale as 1 / sqrt(mass)
    light, double = phonons_batch(model, [crystals[0], heavy], supercell=2, device="cpu")
    q = [0.5, 0.0, 0.5]
    assert light.primitive.masses[0] != double.primitive.masses[0]
    np.testing.assert_allclose(double.get_frequencies(q), light.get_frequencies(q) / np.sqrt(2), rtol=1e-3)


def test_supercell_from_min_length():
    cube = Atoms("Al", cell=np.eye(3) * 4.0, pbc=True)
    assert _supercell_matrix(cube, None, 10.0) == [3, 3, 3]    # 3 x 4 >= 10
    assert _supercell_matrix(cube, None, 12.0) == [3, 3, 3]    # exactly 12
    assert _supercell_matrix(bulk("Al", "fcc", a=4.0), None, 10.0) == [5, 5, 5]    # plane spacing a / sqrt(3)
    assert _supercell_matrix(cube, 2, 10.0) == [2, 2, 2]
    with pytest.raises(ValueError, match="min_length"):
        _supercell_matrix(cube, None, 0.0)


def test_errors(model, crystals):
    with pytest.raises(ValueError, match="energy_only"):
        phonons_batch(model, crystals, supercell=2, energy_only=True, device="cpu")
    in_a_box = crystals[0].copy()
    in_a_box.pbc = [True, True, False]
    with pytest.raises(ValueError, match="periodic"):
        phonons_batch(model, [in_a_box], supercell=2, device="cpu")
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    at = crystals[0]
    no_displacements = Phonopy(PhonopyAtoms(numbers=at.numbers, cell=at.cell.array,
                                            scaled_positions=at.get_scaled_positions()))
    with pytest.raises(ValueError, match="no displacements"):
        phonopy_forces_batch(model, [no_displacements])


def test_without_phonopy(model, monkeypatch):
    monkeypatch.setitem(sys.modules, "phonopy", None)    # import phonopy now raises ImportError
    with pytest.raises(ImportError, match="pip install"):
        phonons_batch(model, [bulk("Al", "fcc", a=4.04)], supercell=2, device="cpu")

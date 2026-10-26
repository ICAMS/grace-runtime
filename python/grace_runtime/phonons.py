# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""phonopy_forces_batch, phonons_batch: phonons by finite displacements with phonopy (an optional dependency), the
displaced supercells of many structures in one compute_batch. The results are phonopy's own objects, ready for its
analysis (mesh, DOS, thermal properties, band structures)."""
import numpy as np
from ase import Atoms

from .batch import compute_batch_iter
from .properties import _full_rank


def phonopy_forces_batch(model, phonons, device: str = "auto", **kwargs) -> list:
    """Forces of the displaced supercells of phonopy.Phonopy objects, all in one compute_batch, each minus the forces
    of its undisplaced supercell (phonopy's --fz: the residual forces of an imperfect relaxation drop out).

    model: as eos_batch. phonons: Phonopy objects with displacements generated (any scheme: generate_displacements
        with its options, random displacements, a dataset set by hand).
    device and other keyword arguments: to compute_batch.

    Sets `.forces` of each Phonopy and returns the list; then e.g. produce_force_constants of phonopy.
    """
    if "energy_only" in kwargs:
        raise ValueError("phonopy_forces_batch sets energy_only itself")
    phonons = list(phonons)
    cells = []
    for k, ph in enumerate(phonons):
        if not ph.supercells_with_displacements:
            raise ValueError(f"phonopy_forces_batch: phonon {k} has no displacements (generate_displacements first)")
        cells += [Atoms(numbers=s.numbers, cell=s.cell, scaled_positions=s.scaled_positions, pbc=True)
                  for s in [ph.supercell] + list(ph.supercells_with_displacements)]    # the undisplaced one first
    results = compute_batch_iter(model, cells, device=device, **kwargs)    # in input order
    for ph in phonons:
        residual = next(results)[1]["forces"]
        ph.forces = np.array([next(results)[1]["forces"] - residual for _ in ph.supercells_with_displacements])
    return phonons


def _supercell_matrix(at, supercell, min_length):
    """the supercell of phonopy for a structure: n for an int (n x n x n), as given otherwise (3 repeats or a 3x3
    matrix); None: the smallest diagonal one whose lattice-plane spacings are all >= min_length (A)"""
    if supercell is None:
        if min_length <= 0:
            raise ValueError(f"min_length must be > 0, not {min_length}")
        spacings = 1 / np.linalg.norm(at.cell.reciprocal(), axis=1)
        return np.ceil(min_length / spacings - 1e-9).astype(int).tolist()    # an exact multiple of a spacing stays
    return [supercell] * 3 if np.ndim(supercell) == 0 else supercell


def phonons_batch(model, structures, supercell=None, min_length: float = 10.0, distance: float = 0.01,
                  device: str = "auto", **kwargs) -> list:
    """Phonons of each structure by finite displacements with phonopy (pip install phonopy): the displaced supercells
    of all structures go through one compute_batch (phonopy_forces_batch).

    model: as eos_batch. structures: periodic ASE Atoms with relaxed cells (e.g. relax_batch(cell_filter="frechet");
        the residual forces drop out, see phonopy_forces_batch); masses set on them are used.
    supercell: n (n x n x n), 3 repeats or a 3x3 matrix, for every structure; None (default): for each structure the
        smallest diagonal supercell of its cell with all lattice-plane spacings >= min_length (A).
    distance: phonopy's displacement amplitude (A).
    device and other keyword arguments: to compute_batch.

    Returns one phonopy.Phonopy per structure with its forces and symmetrised force constants (the acoustic sum rule;
    produce_force_constants() again gives them unsymmetrised): then e.g. run_mesh, run_total_dos,
    run_thermal_properties, auto_band_structure of phonopy.
    """
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
    except ImportError as e:
        raise ImportError("phonons_batch needs phonopy: pip install phonopy (or grace-runtime[phonons])") from e
    structures = list(structures)
    _full_rank(structures, "phonons_batch")
    for k, at in enumerate(structures):
        if not at.pbc.all():
            raise ValueError(f"phonons_batch: structure {k} must be periodic in all three directions")
    phonons = []
    for at in structures:
        masses = at.get_masses() if at.has("masses") else None    # set by hand (isotopes); else phonopy's own
        ph = Phonopy(PhonopyAtoms(numbers=at.numbers, masses=masses, cell=at.cell.array,
                                  scaled_positions=at.get_scaled_positions()),
                     supercell_matrix=_supercell_matrix(at, supercell, min_length))
        ph.generate_displacements(distance=distance)
        phonons.append(ph)
    phonopy_forces_batch(model, phonons, device=device, **kwargs)
    for ph in phonons:
        ph.produce_force_constants()
        ph.symmetrize_force_constants()
    return phonons

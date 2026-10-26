# Equation of state, elastic constants and phonons

[Documentation map](../README.md#guides) · [Relaxation](relax.md)

`eos_batch`, `elastic_batch` and `phonons_batch` compute the energy-volume curve, the elastic constants and the
phonons of many structures at once. All scaled, strained or displaced cells of all inputs go through one
`compute_batch` (atoms fixed) or one `relax_batch` (atoms relaxed), so on a GPU they share its batches. The
parameters and result keys of `eos_batch` and `elastic_batch` follow amstools' `MurnaghanCalculator` and
`ElasticMatrixCalculator`; `phonons_batch` returns phonopy's own objects.

```python
from ase.build import bulk
from grace_runtime import Model, elastic_batch, eos_batch, relax_batch

model = Model("GRACE-2L-OMAT-medium-ft-E")
crystals = [bulk("Al", "fcc", a=4.04, cubic=True), bulk("Ti", "hcp", a=2.95, c=4.68)]
relaxed = [r["atoms"] for r in relax_batch(model, crystals, fmax=1e-3, cell_filter="frechet")]

for r in eos_batch(model, relaxed):
    print(r["equilibrium_volume"], r["equilibrium_bulk_modulus"], r["equilibrium_b_prime"])
for r in elastic_batch(model, relaxed):
    print(r["C"].round(1), r["BH"], r["GH"])
```

The example uses `device="auto"` by default. Pass `device="cpu"`, `"cuda"` or `"rocm"` to each batch call to
require a backend. Reusing `model` avoids loading its weights for every property calculation. Check the relaxation
results' `"converged"` fields before using them as reference structures; do not treat a step limit as convergence.

## eos_batch

`num_of_point` volumes (default 11) from V0 (1 - `volume_range`) to V0 (1 + `volume_range`) (default 0.1), or from
`volume_range[0]` to `volume_range[1]` in Å³ for a pair; each cell is scaled isotropically with its atoms.

- `relax`: `"none"` (default: one `compute_batch`), `"atoms"` (positions at fixed cell) or `"atoms+shape"`
  (positions and cell shape at fixed volume: the Frechet cell filter with `constant_volume=True`); `fmax` (default
  0.005 eV/Å) and `max_steps` (500) for the relaxation, other keyword arguments (such as `dispersion="d3-pbe"`) to
  `compute_batch` or `relax_batch`.
- `fit`: `"poly"` (default; amstools: a polynomial of `fit_order` 5, the lowest minimum within the volumes,
  B = V E'', B' = -(V E''' / E'' + 1)) or an `ase.eos` type: `"murnaghan"`, `"birch"`, `"birchmurnaghan"`,
  `"pouriertarantola"`, `"vinet"`.

Result per structure, points in volume order: `"volume"` (Å³), `"energy"` (eV), `"pressure"` (eV/Å³),
`"equilibrium_volume"`, `"equilibrium_energy"`, `"equilibrium_bulk_modulus"` (GPa), `"equilibrium_b_prime"`,
`"energy_rms"` (eV, of the fit); with a relaxation also `"n_steps"` and `"converged"` per point. Without a minimum
within the volumes the equilibrium values are `None`, with a warning.

B' needs a range of several percent: with ±2 % the fp32 round-off of the GPU energies moved B' by up to 20 % in our
checks (B0 by less than 0.5 %); the default ±10 % agreed with fp64 references to 0.3 %.

## elastic_batch

Six Voigt strains (xx, yy, zz and the engineering shears yz, xz, xy), each at `num_of_point` values (default 5) in
[-`eps_range`, `eps_range`] (default 0.005); the cell becomes `cell @ (I + eps)` with its atoms. Column j of C is the
slope of a least-squares line through the stresses against strain j. This is the stress method: it needs no energy
differences, which lose their precision at small strains with fp32 kernels (on the GPU the energy method was off by
tens of percent for soft metals such as Li or K at strains of 0.25-0.5 %; the stress method stayed within 0.02 GPa of
fp64). The full 6x6 matrix is fitted for any crystal system.

- Inputs should be relaxed (for example with `relax_batch(cell_filter="frechet")`); a residual stress shifts the
  stresses but not their slope.
- `relax_atoms=True` (default) relaxes the atoms in each strained cell (internal relaxation, `fmax` default 1e-3 eV/Å,
  `max_steps` 500); `False` keeps them clamped. Small strains need a tight `fmax`: the error of C goes as the residual
  force over the strain.
- Other keyword arguments go to `relax_batch` (or `compute_batch` with `relax_atoms=False`). A `FixSymmetry` of an
  input is not carried into its strained cells. `fix_symmetry=True` keeps the symmetry of each strained cell during
  the relaxation; its `symprec` must then be well below the atom displacements of the smallest strain (0.008 Å for a
  3.2 Å cell at the defaults), or the unstrained symmetry is found and the internal relaxation is averaged away.

Result per structure: `"C"` (6x6, GPa, symmetrised), `"asymmetry"` (max |C - Cᵀ| / 2 before symmetrising, GPa; large
values point to unconverged relaxations or too large strains), `"S"` (the compliance, 1/GPa), the Voigt, Reuss and
Hill moduli `"BV"`, `"GV"`, `"EV"`, `"nuV"`, `"BR"`, `"GR"`, `"ER"`, `"nuR"`, `"BH"`, `"GH"`, `"EH"`, `"nuH"` (GPa),
`"AVR"` (100 (GV - GR) / (GV + GR)), `"C_eigval"`, `"strains"` and `"stresses"` [6, num_of_point, 6] (GPa), and with
`relax_atoms` `"n_steps"` and `"converged"` [6, num_of_point].

The strain truncation error is O(eps²): about 2 % for C' = (C11 - C12) / 2 of Li at eps 0.005, more at larger
strains.

## phonons_batch, phonopy_forces_batch

Finite displacements with [phonopy](https://phonopy.github.io/phonopy/) (`pip install phonopy`, or the extra
`grace-runtime[phonons]`): phonopy makes the displaced supercells of each structure, the supercells of all structures
go through one `compute_batch`, and phonopy builds the force constants. The result is one `phonopy.Phonopy` per
structure, ready for phonopy's own analysis:

```python
from ase.build import bulk
from grace_runtime import Model, phonons_batch, relax_batch

model = Model("GRACE-2L-OMAT-medium-ft-E")
crystals = [bulk("Al", "fcc", a=4.04), bulk("Si", "diamond", a=5.43)]
relaxed = [r["atoms"] for r in relax_batch(model, crystals, fmax=1e-4, cell_filter="frechet")]

for ph in phonons_batch(model, relaxed):
    ph.run_mesh([20, 20, 20], is_gamma_center=True)
    ph.run_total_dos()
    ph.run_thermal_properties(temperatures=[300])
    print(ph.force_constants.shape)
```

Automatic band paths (`ph.auto_band_structure()`) additionally need `seekpath`
(`python -m pip install seekpath`); it is not installed by the `phonons` extra. Consult
[phonopy's API](https://phonopy.github.io/phonopy/phonopy-module.html) for DOS and thermal-property results.

- `supercell`: n (n x n x n), 3 repeats or a 3x3 matrix, the same for every structure. Default `None`: for each
  structure the smallest diagonal supercell of its cell whose lattice-plane spacings are all at least `min_length`
  (default 10 Å); for the primitive fcc cell of Al that is 5 x 5 x 5 (125 atoms).
- `distance`: phonopy's displacement amplitude (default 0.01 Å).
- The force constants are symmetrised (`symmetrize_force_constants`: the acoustic sum rule), so the acoustic branches
  go to zero at Γ; `ph.produce_force_constants()` gives them unsymmetrised.
- The forces of the undisplaced supercell are computed in the same batch and subtracted from those of each displaced
  one (phonopy's `--fz`), so the residual forces of an imperfect relaxation drop out; relax the cell all the same
  (`relax_batch(cell_filter="frechet")`), a residual stress is not corrected.
- Masses set on the input structures (`atoms.set_masses`, e.g. isotopes) are used; otherwise phonopy's own.
- The structures must be periodic in all three directions. `device` and other keyword arguments go to
  `compute_batch`.

For other displacement schemes or phonopy options (`symprec`, `primitive_matrix`, plus-minus or random
displacements), build the Phonopy objects yourself and let `phonopy_forces_batch` compute the forces of all their
displaced supercells in one batch (the residual forces subtracted as above):

```python
from ase.build import bulk
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from grace_runtime import Model, phonopy_forces_batch

model = Model("GRACE-2L-OMAT-medium-ft-E")
atoms = bulk("Si", "diamond", a=5.43)
unitcell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
                       scaled_positions=atoms.get_scaled_positions())
phonons = [Phonopy(unitcell, supercell_matrix=[2, 2, 2])]
for ph in phonons:
    ph.generate_displacements(distance=0.01, is_plusminus=True)
phonopy_forces_batch(model, phonons)    # sets ph.forces
for ph in phonons:
    ph.produce_force_constants()
```

An example for several crystals: `python/examples/phonons/phonons_batch.py`.

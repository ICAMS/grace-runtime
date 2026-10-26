# Batched relaxation

[Documentation map](../README.md#guides) · [Usage](usage.md)

`relax_batch` relaxes many structures together using the core runtime dependencies (no torch) and runs on the CPU and on
NVIDIA and AMD GPUs. The optimizer is FIRE with the semantics of `ase.optimize.FIRE`: each structure has its own
optimizer state, and the convergence test and step count are those of `Optimizer.run(fmax, steps)`. The cell can be
relaxed with the semantics of `ase.filters.FrechetCellFilter`. CPU and GPU execution use these same update rules;
round-off can change their trajectories (see [precision](#precision)).

```python
from ase.build import bulk
from grace_runtime import Model, relax_batch

model = Model("GRACE-2L-OMAT-medium-ft-E")
structures = [bulk("W", "bcc", a=a, cubic=True) for a in (3.1, 3.2)]
results = relax_batch(model, structures, fmax=0.02, cell_filter="frechet", device="cpu", num_threads=1)

for r in results:
    print(r["energy"], r["n_steps"], r["converged"])
    relaxed = r["atoms"]    # a copy with the relaxed positions and cell, energy / forces / stress attached
```

For a file, use `ase.io.read("candidates.extxyz", index=":")` in place of the generated structures. On a GPU,
select `device="cuda"` or `"rocm"`. Check `r["converged"]` before using a structure as a relaxed reference.

On a GPU, the forces of all structures in flight come from one call per step; the neighbour lists are built on the
GPU. A structure leaves the batch when it has converged or reached `max_steps`, and a waiting structure takes its
place. `max_atoms` sets the number of atoms relaxed together (default 131072). On the CPU the structures are
evaluated one after another, each with its own neighbour list.

## Results

`relax_batch` returns one dictionary per structure, in input order:

| key | |
|---|---|
| `"atoms"` | copy of the input: relaxed positions and cell, a `SinglePointCalculator` with energy, forces, stress |
| `"energy"` | eV (without the `pV` term of `scalar_pressure`) |
| `"forces"` | eV/Å, model forces (`FixAtoms` not applied; `atoms.get_forces()` applies it) |
| `"stress"` | ASE Voigt order, eV/Å³; zeros for cells with zero vectors |
| `"fmax"` | the convergence figure at the end (eV/Å; with the cell filter, over the atom and cell rows) |
| `"n_steps"`, `"converged"` | FIRE steps taken (`n_steps + 1` force evaluations), and whether `fmax` was reached |
| `"stopped"` | `True` when the callback stopped the structure |
| `"error"` | `None`, or why the cell-step limit of `FixSymmetry` stopped the structure (see `fix_symmetry`); the others go on |

Inputs are not changed.

## Streaming: `relax_batch_iter`

`relax_batch_iter` takes the same arguments and yields `(index, result)` as each structure finishes. It reads its
input lazily, when there is room in the batch, so a generator of millions of structures works with bounded memory, and
finished structures can be written immediately:

```python
from ase.io import iread, write
from grace_runtime import Model, relax_batch_iter

model = Model("/path/to/kokkos.npz")
with open("relaxed.extxyz", "w") as out:
    for i, r in relax_batch_iter(model, iread("candidates.extxyz", index=":"), fmax=0.02):
        r["atoms"].info["relax_index"] = i
        write(out, r["atoms"], format="extxyz")
```

The results come in completion order. After an interruption, rerun only the structures whose index is not in the
output file. `compute_batch_iter` is the same streaming form for single-point calculations.

## Options

- `fmax` (default 0.05 eV/Å) and `max_steps` (default 500): as `Optimizer.run(fmax, steps)`.
- `cell_filter="frechet"`: positions and cell, as `ase.filters.FrechetCellFilter`. The filter options
  `scalar_pressure` (eV/Å³; 1 GPa = `ase.units.GPa`), `hydrostatic_strain`, `constant_volume`, `mask` (6 Voigt or
  3x3 booleans) and `exp_cell_factor` (default: the number of atoms of each structure) take the same names and
  defaults. With the filter, `fmax` applies to the filter's rows: the atom forces in the frame of the initial cell, and
  the cell rows. The filter needs cells of rank 3: relax molecules without it.
- FIRE parameters, with the names and defaults of `ase.optimize.FIRE`: `dt=0.1`, `maxstep=0.2`, `dtmax=1.0`, `Nmin=5`,
  `finc=1.1`, `fdec=0.5`, `astart=0.1`, `fa=0.99`, `a=0.1`. `maxstep` limits the norm of the whole step of a
  structure, atoms and cell together.
- Constraints: `FixAtoms` on the input `Atoms` is kept: its atoms do not move, and their forces count as zero.
  `FixSymmetry` is kept too (see below). Other constraint types, and `FixAtoms` together with symmetry, raise a
  `ValueError`.
- `fix_symmetry=True` (with `symprec`, default 0.01 Å): every structure keeps its space group, as with
  `ase.constraints.FixSymmetry`: forces, stress and the steps of positions and cell are averaged over the symmetry
  operations. The operations come from spglib, as in `FixSymmetry`; unlike `FixSymmetry`, the
  input is not refined to the exact symmetry first (`ase.spacegroup.symmetrize.refine_symmetry` does that). A
  structure with its own `FixSymmetry` uses that constraint's operations, so it follows ASE's path. With the cell
  filter, the limits of `FixSymmetry.adjust_cell` on a cell step (largest entry of the deformation-gradient change):
  above 0.15 a warning; above 0.25, where ASE raises, that structure stops before the step (`converged` `False`,
  `"error"` says why) and the others go on. FIRE's `maxstep` (0.2) keeps a cell step below 0.25 when
  `exp_cell_factor` >= 1 (the default is the number of atoms); a one-atom cell can pass the 0.15 warning.
- `callback(step, indices, atoms)`: called after each force evaluation with the input indices and current `Atoms`
  of the structures still running. It may return input indices to stop now; for example, write a trajectory, or stop
  structures whose energy has run away.
- `device`, `max_nall`, `num_threads`, `chunksize`: as `compute_batch`.
- `dispersion`: a dispersion correction added to the model (`"d3-pbe"`, see [dispersion](dispersion.md)); the
  relaxation follows the sums, and the results are the sums.

## Command line

```bash
grace-runtime relax GRACE-2L-OMAT-medium-ft-E candidates.extxyz -o relaxed.extxyz --fmax 0.02 --cell
```

This example expects an existing ASE-readable `candidates.extxyz`; `grace-runtime relax --help` lists all options.
The output is overwritten. Use a different output path if results from an earlier job must be kept.

Each relaxed structure is written as it finishes, with `relax_index` (input frame), `relax_n_steps`,
`relax_converged` and `relax_fmax` in its `info`. The exit status is 1 when a structure did not converge.
`--pressure` (eV/Å³) needs `--cell`.

## Precision

The GPU kernels compute in fp32 (sums in fp64), as elsewhere in grace-runtime. CPU and GPU trajectories and step
counts can differ through round-off. Check final forces, stress and convergence on the backend you use; identical
minima and step counts are not guaranteed.

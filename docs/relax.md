# Batched relaxation

[Documentation map](index.md#guides) · [Usage](usage.md)

`relax_batch` optimizes structures with FIRE on CPU or GPU, without PyTorch. Add `cell_filter="frechet"`
to relax the cell as well as atomic positions:

```python
from ase.build import bulk
from grace_runtime import Model, relax_batch

model = Model("GRACE-2L-OMAT-medium-ft-E")
structures = [bulk("W", "bcc", a=a, cubic=True) for a in (3.1, 3.2)]
results = relax_batch(model, structures, fmax=0.02, cell_filter="frechet")

for r in results:
    print(r["energy"], r["n_steps"], r["converged"])
    relaxed = r["atoms"]
```

Inputs are unchanged. Always check `converged`; reaching `max_steps` does not imply convergence.

## Results

Results are dictionaries in input order. `atoms` is a relaxed copy with energy, forces and stress attached.
Check `converged`, `stopped` and `error` before using it. See the [result reference](api.md#relaxation-results)
for keys, units and shapes. Model forces are unconstrained; `result["atoms"].get_forces()` applies constraints.

## Streaming: `relax_batch_iter`

This complete script requires an ASE-readable `candidates.extxyz` file and writes `relaxed.extxyz`:

```python
from ase.io import iread, write
from grace_runtime import Model, relax_batch_iter

model = Model("GRACE-2L-OMAT-medium-ft-E")
with open("relaxed.extxyz", "w") as out:
    for i, r in relax_batch_iter(model, iread("candidates.extxyz", index=":"), fmax=0.02):
        r["atoms"].info["relax_index"] = i
        write(out, r["atoms"], format="extxyz")
```

The iterator reads lazily and yields `(input_index, result)` in completion order.
On GPU, `max_atoms` controls the batch size; converged structures are replaced by waiting ones.

## Options

Use `fmax` and `max_steps` to control convergence, and `cell_filter="frechet"` to relax a rank-3 cell.
`scalar_pressure` is in eV/Å³ (`ase.units.GPa` converts GPa). Omit the cell filter for molecules.
`fix_symmetry=True` preserves the input space group. A `callback(step, indices, atoms)` can observe steps
or return input indices to stop. Add `dispersion` as in the [dispersion guide](dispersion.md).

FIRE and cell-filter parameters follow ASE. See the [API reference](api.md#relaxation) for signatures and the full option list.
`FixAtoms` and `FixSymmetry` are supported; other constraints, or combining those two, raise `ValueError`.
With symmetry, the largest deformation-gradient change above 0.15 warns; above 0.25 it stops that structure with an error.
`device`, `num_threads` and GPU memory options work as in [batch evaluation](advanced.md#options-and-performance).

## Command line

```bash
grace-runtime relax GRACE-2L-OMAT-medium-ft-E candidates.extxyz -o relaxed.extxyz --fmax 0.02 --cell
```

The output file is overwritten. Structures are written as they finish, with `relax_index`, `relax_n_steps`,
`relax_converged` and `relax_fmax` metadata. Exit status 1 means a structure did not converge.
Use `grace-runtime relax --help` for options; `--pressure` needs `--cell`.

## Precision

CPU and GPU round-off can change optimization trajectories and step counts.
Check final forces, stress and convergence on the backend you use.

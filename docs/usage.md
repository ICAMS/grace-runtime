# ASE and batch workflows

[Manual](index.md) · [API reference](api.md)

Start with the [first calculation](index.md#first-calculation). APIs accept a registry name, a local model file
or a loaded `Model`. Energy, forces and stress are computed together and cached by default.
Set `energy_only=True` for energy screening; a later force or stress request then needs another evaluation.
See the [API reference](api.md#results) for result keys, units and shapes.

## Share a model

Keep a calculator attached during optimization or MD to reuse its model and neighbor list.
Separate calculators can share one loaded model:

```python
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator, Model

model = Model("GRACE-2L-OMAT-medium-ft-E")
for a in (3.1, 3.2):
    atoms = bulk("W", "bcc", a=a, cubic=True)
    atoms.calc = GRACERuntimeCalculator(model)
    print(atoms.get_potential_energy())
```

## Batch evaluation

Structures may have different sizes, compositions, cells and periodicity. Results follow input order.
GPU execution batches structures; CPU execution evaluates them sequentially.

```python
from ase.build import bulk
from grace_runtime import Model, compute_batch

model = Model("GRACE-2L-OMAT-medium-ft-E")
structures = [bulk("W", "bcc", a=a, cubic=True) for a in (3.1, 3.2)]
results = compute_batch(model, structures)
for result in results:
    print(result["energy"], result["forces"].shape)
```

## Streaming

For large inputs, `compute_batch_iter` reads bounded batches and yields `(input_index, result)` in input order.
This complete script requires an ASE-readable `structures.extxyz` file:

```python
from ase.io import iread
from grace_runtime import Model, compute_batch_iter

model = Model("GRACE-2L-OMAT-medium-ft-E")
for i, result in compute_batch_iter(model, iread("structures.extxyz", index=":"), energy_only=True):
    print(i, result["energy"])
```

For streaming relaxation, use [`relax_batch_iter`](relax.md#streaming-relax_batch_iter), which yields in completion order.

## ASE optimization

```python
from ase.build import bulk
from ase.optimize import BFGS
from grace_runtime import GRACERuntimeCalculator

atoms = bulk("W", "bcc", a=3.165, cubic=True)
atoms.positions[0, 0] += 0.05
atoms.calc = GRACERuntimeCalculator("GRACE-2L-OMAT-medium-ft-E")
BFGS(atoms, trajectory="relax.traj").run(fmax=0.05)
```

For many structures, use [batched relaxation](relax.md).

## Molecular dynamics

This complete example runs ten steps with a 1 fs timestep:

```python
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from grace_runtime import GRACERuntimeCalculator

atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat(2)
atoms.calc = GRACERuntimeCalculator("GRACE-2L-OMAT-medium-ft-E")
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
VelocityVerlet(atoms, timestep=units.fs).run(10)
print(atoms.get_potential_energy())
```

For TorchSim integrators and optimizers, use the [TorchSim adapter](torchsim.md).

## Uncertainty quantification

UQ is optional and requires a GRACE model with a supported schema-6 uncertainty head (or a PACE model with an
active set; see [PACE extrapolation grade](pace.md#extrapolation-grade)):

```python
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator

atoms = bulk("W", "bcc", a=3.165, cubic=True)
calc = GRACERuntimeCalculator("GRACE-2L-OMAT-medium-ft-E", uq=True)
atoms.calc = calc
atoms.get_potential_energy()
print(calc.results["gamma"])        # per atom
print(calc.results["total_sigma"])  # scalar
```

`enable_uq()` and `disable_uq()` clear cached results; calculate again before reading them.
UQ reports diagnostics and does not bias forces. See [uncertainty outputs](api.md#results).

## Dispersion correction

The [dispersion guide](dispersion.md#turning-it-on) has a complete DFT-D3 example, parameter choices and limits.
Do not add dispersion when the model's training data already includes it.

## Workflow tips

- Log `calc.backend`; require `device="cuda"` or `"rocm"` when CPU fallback is unacceptable.
- Download models before [offline jobs](models.md#offline-use).
- Match CPU threads to the job allocation with `OMP_NUM_THREADS` if affinity is not constrained.
- Check relaxation `converged` and `error` fields and property-fit warnings before using results.

See [performance and diagnostics](advanced.md) for memory, threading and diagnostics.

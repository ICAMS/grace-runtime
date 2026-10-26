# ASE, batches and UQ

[Documentation map](../README.md#guides) · [Installation](installation.md) · [Models](models.md)

## Calculator

Start with the self-contained [first calculation](../README.md#first-calculation). To evaluate your own structure,
replace `bulk(...)` with `ase.io.read("structure.extxyz")` and keep a calculator attached to it while it changes.
The calculator reuses its model and neighbor list, as in standard ASE optimization and MD.

Results use ASE conventions:

| Request | Shape | Unit |
| --- | --- | --- |
| `atoms.get_potential_energy()` | scalar | eV, total energy |
| `atoms.get_forces()` | `(natoms, 3)` | eV/Å |
| `atoms.get_stress()` | `(6,)`, xx yy zz yz xz xy | eV/Å³; requires a rank-3 cell |
| `atoms.get_potential_energies()` | `(natoms,)` | eV |

With the default `energy_only=True`, requesting energy skips forces and stress; asking for forces afterward runs
another evaluation. Use `energy_only=False` when both are needed, or request forces first. Repeated requests for
unchanged atoms use ASE's cached results.

### Share a model

Every calculator and batch function accepts a registry name, a local `kokkos.npz` path or a loaded `Model`. Reuse a
`Model` across calls to avoid reading its weights repeatedly. Give independently changing structures separate
calculators:

```python
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator, Model

model = Model("GRACE-2L-OMAT-medium-ft-E")
structures = [bulk("W", "bcc", a=a, cubic=True) for a in (3.1, 3.2)]
for atoms in structures:
    atoms.calc = GRACERuntimeCalculator(model, device="cpu", num_threads=1)
    print(atoms.get_potential_energy())
```

For evaluating many independent structures, prefer the batch API below. For optimizing one structure, use an ASE
optimizer; for optimizing many structures, see [batched relaxation](relax.md).

## Uncertainty quantification

UQ is off by default and requires a usable schema-6 head in the model. Enabling it otherwise raises `ValueError`.
The supported mode is `"gamma_only"`; it also returns atomic sigma, cluster assignments and total sigma.

```python
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator

atoms = bulk("W", "bcc", a=3.165, cubic=True)
calc = GRACERuntimeCalculator("GRACE-2L-OMAT-medium-ft-E", device="cpu", num_threads=1, uq=True)
atoms.calc = calc
atoms.get_potential_energy()
print(calc.results["gamma"])         # (natoms,)
print(calc.results["atomic_sigma"])  # (natoms,)
print(calc.results["gmm_cluster"])   # (natoms,)
print(calc.results["total_sigma"])  # scalar
```

`calc.enable_uq()` and `calc.disable_uq()` toggle it later. Both clear cached results: run a new evaluation before
reading `calc.results`. UQ values are diagnostics; this interface does not add a UQ bias to the forces.

## Dispersion correction

DFT-D3(BJ) is optional and off by default. This example adds it to a registry foundation model using PBE parameters:

```python
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator

atoms = bulk("W", "bcc", a=3.165, cubic=True)
calc = GRACERuntimeCalculator(
    "GRACE-2L-OMAT-medium-ft-E", device="cpu", num_threads=1, dispersion="d3-pbe",
)
atoms.calc = calc
print(atoms.get_potential_energy())       # model + D3, eV
print(calc.results["dispersion_energy"])  # D3 contribution alone, already included in the total
```

The same `dispersion` argument works for batch evaluation, relaxation, property functions and TorchSim.
UQ outputs remain those of the model. Choose parameters for the training functional, and do not add D3 when the
training data already includes dispersion. See [dispersion](dispersion.md) for cutoffs, component outputs and limits.

## Batch evaluation

`compute_batch` returns one result dictionary per input structure, in input order. Structures can have different
sizes, compositions, cells and periodicity. See the [file-free example](../README.md#many-structures).

| Result key | Default | Meaning |
| --- | --- | --- |
| `"energy"` | included | total energy in eV |
| `"forces"` | included | `(natoms, 3)`, eV/Å |
| `"stress"` | included | `(6,)`, ASE Voigt order, eV/Å³; zero for cells of rank < 3 |
| `"energies"` | `energies=True` | `(natoms,)`, atomic energies in eV |
| `"gamma"`, `"atomic_sigma"`, `"gmm_cluster"`, `"total_sigma"` | `uq=True` | same outputs as the calculator |

`energy_only=True` omits `"forces"` and `"stress"` and skips the backward pass. CPU batches evaluate structures
one after another; GPU batches group structures into engine passes. `compute_batch` materializes its input and
all results: use [compute_batch_iter](advanced.md#batches-of-many-structures-streaming) for bounded-memory processing.

## Short molecular dynamics example

This ten-step CPU example creates its own structure and writes no files. Increase the system size and step count
for your application, and select `device="cuda"` or `"rocm"` when a GPU backend is installed.

```python
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from grace_runtime import GRACERuntimeCalculator

atoms = bulk("W", "bcc", a=3.165, cubic=True).repeat(2)
atoms.calc = GRACERuntimeCalculator("GRACE-2L-OMAT-medium-ft-E", device="cpu", num_threads=1)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
VelocityVerlet(atoms, timestep=units.fs).run(10)
print(atoms.get_potential_energy())
```

ASE times use `ase.units` (the step above is 1 fs). TorchSim uses its own interface; see the
[TorchSim example](advanced.md#torchsim-model-outputs-and-memory).

## Scripts and agents

- Choose `device="cpu"`, `"cuda"` or `"rocm"` when the backend is a requirement. `"auto"` permits CPU fallback;
  log `calc.backend` and `calc.backend_info` for calculator runs. `GRACE_RUNTIME_STRICT=1` also refuses PTX fallback.
- Load a `Model` once; keep a calculator for repeated evaluations of changing atoms. Batch independent structures;
  stream large files. Request only the outputs needed.
- Set `num_threads` or `OMP_NUM_THREADS` to the CPU allocation. The default is the physical cores in the process
  affinity mask, not the scheduler request unless affinity is constrained. Small cells often need fewer threads.
- Download models before an offline job. See [offline use](models.md#offline-use).
- For relaxation, inspect each result's `"converged"` and `"error"` fields; reaching `max_steps` is not convergence.
  `grace-runtime relax` returns status 1 if a structure does not converge and overwrites its output file.
- For EOS and elastic calculations, inspect convergence information and warnings. A missing EOS minimum yields
  `None` equilibrium values; a singular elastic matrix yields `None` Reuss/Hill moduli. See [properties](properties.md).

Performance options, memory ownership, numerical precision and backend diagnostics are in [advanced usage](advanced.md).

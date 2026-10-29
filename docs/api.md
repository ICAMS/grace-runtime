# Python API reference

[Manual](index.md) · [Usage examples](usage.md)

Import these objects from `grace_runtime`, except the optional TorchSim adapter.
Signatures, defaults and parameter descriptions below are read from source at documentation-build time.
These blocks show call signatures; runnable examples are in the guides.
The documentation build does not import the runtime, compile kernels or download models.

## Model

<!-- api: _core.Model -->

`Model` holds loaded weights and can be shared across calculators and batch calls.
`elements`, `arch`, `path`, `spline`, `store_max_gb` and `has_uq` expose model metadata;
`validate_uq()` checks that the model has a supported UQ head or PACE active set;
`cutoff_matrix(species)` returns pair cutoffs in Å for the given element symbols.
The native `compute` method is the low-level CPU CSR interface; normal workflows use the calculator or batch APIs.

## ASE calculator

<!-- api: calculator.GRACERuntimeCalculator -->

ASE requests: `get_potential_energy()`, `get_forces()`, `get_stress()` and `get_potential_energies()`.
Request atomic energies explicitly; the default calculation caches total energy, forces and stress.
`backend` names the actual device; `backend_info` gives its platform, architecture and fallback information.

### enable_uq

<!-- api: calculator.GRACERuntimeCalculator.enable_uq -->

### disable_uq

<!-- api: calculator.GRACERuntimeCalculator.disable_uq -->

### free_gpu_memory

<!-- api: calculator.GRACERuntimeCalculator.free_gpu_memory -->

## Batch evaluation

### compute_batch

<!-- api: batch.compute_batch -->

### compute_batch_iter

<!-- api: batch.compute_batch_iter -->

### free_gpu_memory

The package-level function releases buffers for all calculators and batches, including cached GPU models:

<!-- api: batch.free_gpu_memory -->

## Relaxation

### relax_batch

<!-- api: relax.relax_batch -->

### relax_batch_iter

<!-- api: relax.relax_batch_iter -->

## Properties

### eos_batch

<!-- api: properties.eos_batch -->

### elastic_batch

<!-- api: properties.elastic_batch -->

### phonons_batch

<!-- api: phonons.phonons_batch -->

### phonopy_forces_batch

<!-- api: phonons.phonopy_forces_batch -->

## Dispersion

<!-- api: dispersion.D3 -->

## TorchSim adapter

Requires the `torchsim` extra and Python >= 3.12:

```python
from grace_runtime.torchsim import GRACERuntimeModel
```

<!-- api: torchsim.GRACERuntimeModel -->

### max_memory_scaler

<!-- api: torchsim.GRACERuntimeModel.max_memory_scaler -->

## Results

Let `N` be the atoms in one structure, `S` the number of systems, and `A` the total atoms across a TorchSim state.
ASE stress uses Voigt order xx, yy, zz, yz, xz, xy. Stress is zero for cells without a full-rank cell.

### Calculator and batch results

A calculator exposes cached values through `calc.results`; a batch returns one dictionary per structure.

| Key | Shape | Units / availability |
| --- | --- | --- |
| `energy` | scalar | eV |
| `forces` | `(N, 3)` | eV/Å; omitted for energy-only evaluation |
| `stress` | `(6,)` | eV/Å³; omitted for energy-only evaluation |
| `energies` | `(N,)` | eV; request atomic energies explicitly |
| `gamma` | `(N,)` | Dimensionless extrapolation grade (latent-space distance divided by threshold for GRACE `uqv6`, or active-set grade for PACE); with `uq=True` |
| `atomic_sigma` | `(N,)` | Dimensionless latent-space Mahalanobis distance; GRACE `uqv6` models with `uq=True` |
| `gmm_cluster` | `(N,)` | Cluster index; GRACE `uqv6` models with `uq=True` |
| `total_sigma` | scalar | Sum of `atomic_sigma`; GRACE `uqv6` models with `uq=True` |
| `dispersion_energy` | scalar | eV; included when dispersion is enabled |
| `dispersion_forces`, `dispersion_stress`, `dispersion_energies` | As the corresponding total | Correction alone; calculator `get_property(...)` or batch `dispersion_parts=True` |

`free_energy` is the calculator's ASE alias for `energy`. Dispersion is already included in the totals.
UQ values are latent-space diagnostics, not energy or force error bars. They do not include dispersion uncertainty.

### Relaxation results

| Key | Shape / meaning |
| --- | --- |
| `atoms` | Relaxed ASE `Atoms` copy, with energy, forces and stress attached |
| `energy`, `forces`, `stress` | Same units and shapes as above; energy excludes external pressure's `pV` term |
| `fmax` | Final force/filter convergence measure in eV/Å |
| `n_steps`, `converged` | FIRE step count and convergence flag |
| `stopped`, `error` | Callback stop flag and symmetry cell-step error, or `None` |

Model forces are unconstrained; `result["atoms"].get_forces()` applies the retained constraints.

### EOS and elastic results

| API | Keys and units |
| --- | --- |
| EOS samples | `volume` (Å³), `energy` (eV), `pressure` (eV/Å³), each `(num_of_point,)` |
| EOS fit | `equilibrium_volume` (Å³), `equilibrium_energy` (eV), `equilibrium_bulk_modulus` (GPa), dimensionless `equilibrium_b_prime`, `energy_rms` (eV) |
| Elastic matrix | `C` `(6, 6)` (GPa), `S` `(6, 6)` (1/GPa), `asymmetry` (GPa), `C_eigval` `(6,)` (GPa) |
| Elastic samples | `strains` `(num_of_point,)`, `stresses` `(6, num_of_point, 6)` (GPa) |
| Isotropic moduli | `BV`, `GV`, `EV`, `BR`, `GR`, `ER`, `BH`, `GH`, `EH` (GPa) |
| Ratios | `nuV`, `nuR`, `nuH` are dimensionless; `AVR` is percent |
| Relaxed samples | `n_steps`, `converged`: `(num_of_point,)` for EOS, `(6, num_of_point)` for elastic |

Missing EOS minima and singular elastic matrices yield `None` for the affected fit/modulus values.
`phonons_batch` and `phonopy_forces_batch` return lists of phonopy objects, not result dictionaries.

### TorchSim results

| Key | Shape | Units / availability |
| --- | --- | --- |
| `energy` | `(S,)` | eV |
| `forces` | `(A, 3)` | eV/Å; with `compute_forces=True` |
| `stress` | `(S, 3, 3)` | eV/Å³; with `compute_stress=True` |
| `energies` | `(A,)` | eV; with `energies=True` |
| `gamma`, `atomic_sigma`, `gmm_cluster` | `(A,)` | With `uq=True` (PACE models return `gamma` only) |
| `total_sigma` | `(S,)` | GRACE `uqv6` models with `uq=True` |
| `dispersion_energy` | `(S,)` | eV; when dispersion is enabled |
| `dispersion_forces`, `dispersion_stress`, `dispersion_energies` | As the corresponding total | Correction alone, with `dispersion_parts=True`; force/stress parts are included when either forces or stress are computed |

## Command-line interface

| Command | Purpose |
| --- | --- |
| `grace-runtime info` | Print the active CPU module, visible GPU, installed GPU modules and fallback status (returns 1 if a visible GPU cannot run natively) |
| `grace-runtime models list` | List foundation models in the registry and whether each is cached |
| `grace-runtime models download [--force] NAME...` | Download one or more registry models into `$GRACE_CACHE` (`~/.cache/grace`) |
| `grace-runtime relax MODEL INPUT -o OUTPUT [--fmax F] [--max-steps N] [--cell] [--pressure P] [--device DEV] [--max-atoms M]` | Relax all frames of `INPUT` in batches and stream relaxed structures to `OUTPUT` (`extxyz`); returns 1 if any structure did not converge |

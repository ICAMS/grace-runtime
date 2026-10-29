# grace-runtime

Run GRACE-1L, GRACE-2L and GRACE-3L interatomic potentials on CPU or GPU without TensorFlow, PyTorch or JAX.
CPU execution is standalone; GPU execution uses Kokkos for NVIDIA CUDA and AMD HIP.
The kernels also run in LAMMPS, with MPI support.

Start with [installation](installation.md), then try a calculation below.
The optional TorchSim adapter requires PyTorch and TorchSim.

## First calculation

```python
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator

atoms = bulk("W", "bcc", a=3.165, cubic=True)
atoms.calc = GRACERuntimeCalculator("GRACE-2L-OMAT-medium-ft-E")
print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())            # eV/Å
print(atoms.calc.backend)
```

The runtime selects a usable GPU or CPU automatically and caches energy, forces and stress together.
Named models download on first use; local `kokkos.npz` files work too.

## Many structures

```python
from ase.build import bulk
from grace_runtime import Model, compute_batch

model = Model("GRACE-2L-OMAT-medium-ft-E")
structures = [bulk("W", "bcc", a=a, cubic=True) for a in (3.1, 3.2)]
results = compute_batch(model, structures)
print([r["energy"] for r in results])
```

## Guides

| Task | Guide |
| --- | --- |
| Install CPU or GPU packages | [Installation](installation.md) |
| Choose GPU hardware or build a backend | [GPU backends](backends.md) |
| Download models and run offline | [Models](models.md) |
| Use ASE, batches and UQ | [ASE and batch workflows](usage.md) |
| Relax positions and cells | [Relaxation](relax.md) |
| Calculate EOS, elastic constants and phonons | [Properties](properties.md) |
| Add DFT-D3 | [Dispersion](dispersion.md) |
| Run PACE (ACE) models from pacemaker `.yaml` | [PACE models](pace.md) |
| Use TorchSim integrators and optimizers | [TorchSim](torchsim.md) |
| Build LAMMPS and run GRACE or PACE in it | [LAMMPS](lammps.md) |
| Manage memory, threads and fallbacks | [Performance and diagnostics](advanced.md) |
| See speed and the largest systems per GPU | [Benchmarks](benchmarks.md) |
| Look up parameters, defaults, CLI and results | [API reference](api.md) |
| Build and test | [Development](development.md) |

## LAMMPS

The [LAMMPS fork](https://github.com/yury-lysogorskiy/lammps/tree/grace-runtime) supplies pair styles
`grace/runtime` (CPU) and `grace/runtime/kk` (GPU) for GRACE-1L / 2L / 3L models (`kokkos.npz`), and
`pace/runtime` / `pace/runtime/kk` for [PACE models](pace.md). Installing the Python package does not install
LAMMPS: [LAMMPS](lammps.md) shows how to build it (CPU, CUDA, HIP) and run a first input.

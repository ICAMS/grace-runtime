# grace-runtime

Run GRACE-1L, GRACE-2L and GRACE-3L interatomic potentials with an ASE calculator, batch functions or TorchSim.
CPU and GPU inference needs no TensorFlow. The same kernels run in LAMMPS.

## Install

**The first release is in preparation. The packages currently on PyPI are placeholders with no runtime code.**
Install from a source checkout for now. We recommend [micromamba](docs/installation.md#1-create-an-environment)
(`mamba` works too), as in the GRACEmaker installation guide.

From the repository root:

```bash
micromamba create -n grace-runtime -c conda-forge python=3.12 pip cxx-compiler
micromamba activate grace-runtime
python -m pip install . --config-settings=cmake.define.GRACE_RUNTIME_ISAS=native
python -c "import grace_runtime; print(grace_runtime.__version__, grace_runtime.cpu_module)"
```

This installs the CPU package, optimized for the machine building it, and prints its version and loaded CPU module.
No GPU toolkit is needed for this step. `grace-runtime info` provides additional GPU diagnostics.
For NVIDIA or AMD GPU inference, also build the matching [GPU package](docs/installation.md#gpu-backends).
Ready-made wheel installation commands are listed [separately](docs/installation.md#release-wheels).

## First calculation

This example creates its own structure and runs on the CPU:

```python
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator

atoms = bulk("W", "bcc", a=3.165, cubic=True)
calc = GRACERuntimeCalculator(
    "GRACE-2L-OMAT-medium-ft-E", device="cpu", num_threads=1, energy_only=False,
)
atoms.calc = calc

print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())            # (natoms, 3), eV/Å; reuses the calculation above
print(calc.backend)                 # actual execution backend
```

The first use of a registry name downloads the model into `~/.cache/grace` (or `$GRACE_CACHE`). Later runs reuse it.
A local `kokkos.npz` path works in place of the name; all structure elements must be supported by the model.
`energy_only=False` computes energy and forces together. For energy screening alone, keep the default `True`.

Choose the device explicitly when execution on a particular backend matters:

| `device` | Behavior |
| --- | --- |
| `"cpu"` | CPU execution. `num_threads` controls CPU threads. |
| `"auto"` (default) | Usable NVIDIA GPU, then AMD GPU, otherwise CPU. Reports a fallback when a GPU is present but unusable. |
| `"cuda"` / `"cuda:N"` | Require NVIDIA GPU execution; raise if unavailable. |
| `"rocm"` / `"rocm:N"` | Require AMD GPU execution; needs the source-built ROCm backend for now. |
| `"gpu"` / `"gpu:N"` | Require a usable GPU, NVIDIA first, then AMD; raise if none is usable. |

The runtime currently uses one GPU per process. For strict native GPU execution and performance checks, see
[backend diagnostics](docs/advanced.md#where-does-it-run-fallbacks-strict-mode-benchmarks).

## Many structures

Load a `Model` once and reuse it across calculations. This small example needs no input files:

```python
from ase.build import bulk
from grace_runtime import Model, compute_batch

model = Model("GRACE-2L-OMAT-medium-ft-E")
structures = [bulk("W", "bcc", a=a, cubic=True) for a in (3.1, 3.165, 3.2)]
results = compute_batch(model, structures, device="auto", energy_only=True, num_threads=1)
print([r["energy"] for r in results])
```

`compute_batch` returns results in input order. Omit `energy_only=True` to also compute forces and stress.
GPU execution batches structures together; CPU execution evaluates them one after another.
For large files, use [streaming](docs/advanced.md#batches-of-many-structures-streaming).

## Dispersion correction

Optional DFT-D3 with Becke-Johnson damping is off by default. Pass `dispersion="d3-pbe"` to the calculator, batch,
relaxation, property or TorchSim API for the registry foundation models (PBE/PBE+U without dispersion). For other
training functionals, parameters and cutoffs, see [dispersion](docs/dispersion.md); do not add it to a model whose
training data already includes dispersion. The optional Axilrod-Teller-Muto three-body term is off by default;
`dispersion="d3-pbe-atm"` includes it. See the guide for its additional cost and GPU reproducibility limits.

## Guides

| Task | API or command | Guide |
| --- | --- | --- |
| Install, build for your CPU/GPU, add optional packages | `python -m pip` | [Installation](docs/installation.md) |
| ASE calculations, UQ, short MD, automation | `GRACERuntimeCalculator` | [Usage](docs/usage.md) |
| List, download and reuse models offline | `grace-runtime models` | [Models and cache](docs/models.md) |
| Add optional DFT-D3(BJ) dispersion | `dispersion="d3-pbe"` | [Dispersion](docs/dispersion.md) |
| Relax positions and cells | `relax_batch`, `relax_batch_iter`, `grace-runtime relax` | [Relaxation](docs/relax.md) |
| Equation of state, elastic constants, phonons | `eos_batch`, `elastic_batch`, `phonons_batch` | [Properties](docs/properties.md) |
| Tune memory/threads, stream data, use TorchSim | `compute_batch_iter`, `GRACERuntimeModel` | [Advanced usage](docs/advanced.md) |
| Build the library, run tests, build release wheels | CMake, pytest, wheel scripts | [Development](docs/development.md) |

An interactive tour is in the [ASE notebook](python/examples/ase/grace_runtime_ase.ipynb).

## LAMMPS

The [LAMMPS fork](https://github.com/yury-lysogorskiy/lammps/tree/grace-runtime) supplies pair styles
`grace/{1l,2l,3l}/fused` (CPU) and `grace/{1l,2l,3l}/fused/kk` (GPU). Follow that repository's build instructions;
installing the Python package does not install LAMMPS.

## Licence

Academic Software Licence (ASL, [LICENSE.md](LICENSE.md)): academic non-commercial use. For commercial use contact
ACEworks GmbH, contact@aceworks.works. Bundled third-party code keeps its own licence
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Changes: [CHANGELOG.md](CHANGELOG.md).

## Reference

Please cite:

* Y. Lysogorskiy, A. Bochkarev, R. Drautz, Graph atomic cluster expansion for foundational machine learning interatomic
  potentials. npj Computational Materials 12(1), 114 (2026), https://www.nature.com/articles/s41524-026-01979-1
* For UQ (extrapolation grade `gamma`): Y. Lysogorskiy, A. Bochkarev, R. Drautz, A latent-space extrapolation grade
  built into graph atomic cluster expansion foundation potentials. arXiv:2609.40060 (2026),
  https://arxiv.org/abs/2609.40060
* A. Bochkarev, Y. Lysogorskiy, R. Drautz, Graph Atomic Cluster Expansion for Semilocal Interactions beyond Equivariant
  Message Passing. Phys. Rev. X 14, 021036 (2024), https://doi.org/10.1103/PhysRevX.14.021036

```bibtex
@article{lysogorskiy2026graph,
  title={Graph atomic cluster expansion for foundational machine learning interatomic potentials},
  author={Lysogorskiy, Yury and Bochkarev, Anton and Drautz, Ralf},
  journal={npj Computational Materials}, volume={12}, number={1}, pages={114}, year={2026},
  doi={10.1038/s41524-026-01979-1}
}
@article{lysogorskiy2026calm,
  title={A latent-space extrapolation grade built into graph atomic cluster expansion foundation potentials},
  author={Lysogorskiy, Yury and Bochkarev, Anton and Drautz, Ralf},
  journal={arXiv preprint arXiv:2609.40060}, year={2026}
}
@article{PhysRevX.14.021036,
  title={Graph Atomic Cluster Expansion for Semilocal Interactions beyond Equivariant Message Passing},
  author={Bochkarev, Anton and Lysogorskiy, Yury and Drautz, Ralf},
  journal={Phys. Rev. X}, volume={14}, pages={021036}, year={2024}, doi={10.1103/PhysRevX.14.021036}
}
```

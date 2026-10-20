# grace-runtime

Fast inference of GRACE interatomic potentials (GRACE-1L, GRACE-2L, GRACE-3L) on CPU and NVIDIA GPU, behind an ASE
calculator and a TorchSim model. No TensorFlow needed. The same kernels run in LAMMPS (pair styles
`grace/{1l,2l,3l}/fused` and `grace/{1l,2l,3l}/fused/kk`).

Licence: Academic Software Licence (ASL, [LICENSE.md](LICENSE.md)): academic non-commercial use; for commercial use
contact ACEworks GmbH, contact@aceworks.works. Bundled third-party code keeps its own licence
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Install

```bash
pip install grace-runtime                          # CPU
pip install grace-runtime grace-runtime-cuda12     # CPU and NVIDIA GPU
```

### Requirements

| | |
| --- | --- |
| System | Linux x86-64, CPython >= 3.10 |
| CPU | x86-64-v3 (AVX2, FMA; Intel Haswell, AMD Zen and newer). Each wheel also has an AVX-512 build, which is loaded on CPUs that have AVX-512 |
| NVIDIA GPU | driver for CUDA 12 (>= 525; B200 / B300 and RTX 50xx: >= 570); no CUDA toolkit. Native modules for V100, RTX 20xx / T4, A100 / A30, RTX 30xx / A-series, L40S / RTX 40xx, H100 / H200, B200 / B300, RTX 50xx ([python-cuda/README.md](python-cuda/README.md)) |
| AMD GPU (ROCm) | not available yet; a ROCm backend is in preparation |

`grace_runtime.cpu_module` tells which CPU build is loaded; `GRACE_RUNTIME_CPU=v3|v4` forces one.

### Optional dependencies

`pip install grace-runtime` installs only what the calculator needs (`numpy`, `ase`, `tqdm`). Extras add more; name
them in brackets, several separated by commas (`pip install "grace-runtime[matscipy,torchsim]"`):

| Extra | Installs | Needed for |
| --- | --- | --- |
| `matscipy` | `matscipy>=1.3` | `neighbor_backend="matscipy"` (the default neighbour list is built in) |
| `torchsim` | `torch>=2`, `torch-sim-atomistic>=0.6` (CPython >= 3.12; nothing on 3.10 / 3.11) | `grace_runtime.torchsim` |
| `test` | `pytest`, `matscipy>=1.3` | the Python tests (`pytest python/tests`) |

The GPU package has one extra, `grace-runtime-cuda12[cublas]`: `nvidia-cublas-cu12`, used for the UQ projections
when installed.

## Build from source

### CPU

Run these commands from the repository root with your target Python environment activated. You need Python >= 3.10,
a C++17 compiler with OpenMP, and Python development headers. Pip installs the Python build dependencies in an
isolated build environment.

```bash
python -m pip install .
```

This builds and installs the main package, including the CPU backend and ASE calculator. By default it builds both
x86-64-v3 and x86-64-v4 modules. To optimize only for the CPU of the build machine:

```bash
python -m pip install . --config-settings=cmake.define.GRACE_RUNTIME_ISAS=native
```

A native build may not run on another CPU. To create a wheel for later installation instead:

```bash
python -m pip wheel --no-deps . --wheel-dir dist/cpu-wheelhouse
python -m pip install --no-index --find-links=dist/cpu-wheelhouse grace-runtime
```

The last command needs the runtime dependencies (`numpy`, `ase`, and `tqdm`) already installed, because `--no-index` prevents
pip from fetching them. For development, use `python -m pip install -e .`.

### NVIDIA GPU (CUDA)

The CUDA backend is the separate package `grace-runtime-cuda12`, built by `scripts/build_cuda_wheels.sh` (needs
`micromamba` and network access; CUDA 12.9 and the compilers come from conda-forge, no system CUDA toolkit). For one
GPU architecture, e.g. an RTX 3060 (`sm86`):

```bash
export PYTHONS="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export GRACE_KK_ARCHS='sm86' GRACE_KOKKOS_CACHE="$PWD/.kokkos-wheel" NPROC=8
bash scripts/build_cuda_wheels.sh "$PWD/dist/cuda-wheelhouse"
python -m pip install .                                                       # the CPU package first
python -m pip install --no-index --find-links="$PWD/dist/cuda-wheelhouse" grace-runtime-cuda12
```

Other architectures and details: [docs/development.md](docs/development.md#a-cuda-wheel-for-other-gpu-architectures).

### AMD GPU (ROCm)

Not available yet; it will come with a later release.

## Use

### ASE calculator

```python
from ase.io import read
from grace_runtime import GRACERuntimeCalculator

atoms = read("structure.extxyz")
calc = GRACERuntimeCalculator("GRACE-2L-OMAT-medium-ft-E", device="auto")
atoms.calc = calc

energy = atoms.get_potential_energy()     # eV
forces = atoms.get_forces()               # (natoms, 3), eV/Å
stress = atoms.get_stress()               # (6,), eV/Å³; full-rank cell required
atomic_energies = atoms.get_potential_energies()  # (natoms,), eV
print(calc.backend)                      # actual CPU or CUDA backend
```

A registry name downloads the model once into the cache. To use a local export, pass
`GRACERuntimeCalculator("/path/to/kokkos.npz")`. All elements in the structure must be supported by the model.

`device="auto"` uses the GPU when a compatible backend and model are available, otherwise the CPU. Use
`device="cuda"` (or `"cuda:0"`) to require GPU execution and raise an error if unavailable; use `device="cpu"` to
force CPU execution. GRACE-1L, GRACE-2L and GRACE-3L support CUDA (GPU modules built with the GRACE-3L engine;
an older module runs GRACE-3L on the CPU with a warning).

### Uncertainty quantification (UQ)

UQ is off by default. The model must contain a usable schema-6 UQ head; otherwise enabling UQ raises `ValueError`.
Only the `"gamma_only"` mode is supported.

```python
calc.enable_uq()                         # also: calc.enable_uq("gamma_only")
atoms.get_potential_energy()             # evaluate with UQ enabled

gamma = calc.results["gamma"]            # per atom
atomic_sigma = calc.results["atomic_sigma"]
gmm_cluster = calc.results["gmm_cluster"]
total_sigma = calc.results["total_sigma"]

calc.disable_uq()
atoms.get_potential_energy()             # evaluate again without UQ
```

You can also enable UQ at construction: `GRACERuntimeCalculator(model, uq=True)`. Toggling UQ clears the cached
calculator results; read UQ results after a new evaluation.

### Many structures at once

`compute_batch` accepts ASE structures and returns one dictionary per structure in input order. On CUDA it groups
structures into fused engine passes; on CPU it loops over the calculator with the same result format. Structures
may have different sizes, cells, periodicity, and compositions.

```python
from ase.io import read
from grace_runtime import compute_batch

structures = read("structures.extxyz", index=":")
results = compute_batch(
    "GRACE-2L-OMAT-medium-ft-E",  # or a local kokkos.npz path
    structures,
    device="cuda",
    num_threads=8,              # host threads for batch neighbor searches
    chunksize="auto",
)

for atoms, result in zip(structures, results):
    energy = result["energy"]  # eV
    forces = result["forces"]  # (len(atoms), 3), eV/Å
    stress = result["stress"]  # ASE Voigt order, eV/Å³
```

Batch stress is zero for cells of rank < 3. `energies=True` adds per-atom `"energies"`; `uq=True` adds `"gamma"`,
`"atomic_sigma"`, `"gmm_cluster"`, and `"total_sigma"`. For energy screening only, use `energy_only=True` to skip
forces and stress (those keys are then absent).

For datasets that do not fit in memory, see [streaming](docs/advanced.md#batches-of-many-structures-streaming).

### TorchSim (extra)

`grace_runtime.torchsim.GRACERuntimeModel` is a [TorchSim](https://github.com/TorchSim/torch-sim) model
(`ModelInterface`): batched MD, relaxations and static calculations of `torch_sim` run on the fused kernels. Needs
Python >= 3.12 and the `torchsim` extra (`pip install "grace-runtime[torchsim]"`).

```python
import numpy as np    # import numpy before torch if torch fails to load an old system libstdc++
import torch
import torch_sim as ts
from ase.build import bulk
from grace_runtime.torchsim import GRACERuntimeModel

model = GRACERuntimeModel("GRACE-2L-OMAT-medium-ft-E", device="cuda")    # dtype=torch.float64 by default
cells = [bulk("W", "bcc", a=3.165, cubic=True).repeat(4) for _ in range(64)]
state = ts.io.atoms_to_state(cells, torch.device("cuda"), torch.float64)

batcher = ts.BinningAutoBatcher(model, memory_scales_with=model.memory_scales_with,
                                max_memory_scaler=model.max_memory_scaler(state))
final = ts.integrate(state, model, integrator=ts.Integrator.nvt_langevin, n_steps=1000, temperature=300.0,
                     timestep=0.001, autobatcher=batcher)
relaxed = ts.optimize(state, model, optimizer=ts.Optimizer.fire,
                      convergence_fn=ts.generate_force_convergence_fn(force_tol=0.05))
```

Outputs and memory: [docs/advanced.md](docs/advanced.md#torchsim-model-outputs-and-memory).

Options (neighbour-list skin, threads, GPU memory), several calculators in one process, fallbacks and precision:
[docs/advanced.md](docs/advanced.md).

## Supported models

| Architecture | Foundation models (`grace-runtime models list`) |
| --- | --- |
| GRACE-1L | GRACE-1L-OMAT-medium-{base,ft-E,ft-AM}, GRACE-1L-OMAT-large-{base,ft-E,ft-AM}, GRACE-1L-SMAX-OMAT-large, GRACE-1L-SMAX-large |
| GRACE-2L | GRACE-2L-OMAT-medium-{base,ft-E,ft-AM}, GRACE-2L-OMAT-large-{base,ft-E,ft-AM}, GRACE-2L-SMAX-OMAT-medium, GRACE-2L-SMAX-OMAT-large, GRACE-2L-SMAX-medium, GRACE-2L-SMAX-large |
| GRACE-3L | GRACE-3L-OMAT-large, GRACE-3L-OMAT-large-ft-AM |

OMAT: trained on OMat24 (`-base`), fine-tuned with more weight on energies (`-ft-E`) or on sAlex + MPtrj (`-ft-AM`);
6 Å cutoff. SMAX and SMAX-OMAT: trained on SMAX (and OMat24), element-dependent cutoffs between 5.0 and 7.5 Å. The
older names (e.g. `GRACE-2L-OAM-L`, `GRACE-2L-OMAT-M`) are aliases. All of them have a UQ head. Your own GRACE-1L / 2L / 3L models run
too: export them with tensorpotential's `grace_utils -p model.yaml -c <checkpoint> export_kokkos -o model.npz` and
pass the file. Legacy models (GRACE-FS, the first OAM / OMAT releases) have no such export; use tensorpotential's
`TPCalculator` for them. The model cache and downloads: [docs/models.md](docs/models.md).

## Utilities

```bash
grace-runtime info                                  # where calculations run: CPU build, GPU, driver, GPU module
grace-runtime models list                           # the foundation models, aliases, cached or not
grace-runtime models download GRACE-2L-OMAT-medium-ft-E
```

## Troubleshooting

- `ImportError: grace-runtime needs an x86-64-v3 CPU (AVX2, FMA, BMI2)`: the CPU is too old for the wheels.
- A GPU is present but the calculation runs on the CPU, with a warning: no GPU module for it, or the model is outside
  the GPU kernels. `grace-runtime info` and `calc.backend_info["fallback"]` tell why. `device="cuda"` raises instead
  of falling back; [docs/advanced.md](docs/advanced.md#where-does-it-run-fallbacks-strict-mode-benchmarks) lists the
  ways out.
- `ValueError: one CUDA device per process`: use one process per GPU (`CUDA_VISIBLE_DEVICES`).

## LAMMPS

LAMMPS with these pair styles: https://github.com/yury-lysogorskiy/lammps, branch `grace-runtime`. Its ML-PACE package
compiles libgrace from a release of this repository (downloaded at configure time) or from a local checkout
(`-DLOCAL_GRACE_RUNTIME=<path>`).

## Development

Build and tests, test models, CI: [docs/development.md](docs/development.md). Changes: [CHANGELOG.md](CHANGELOG.md).

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

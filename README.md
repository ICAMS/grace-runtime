# grace-runtime

Fast, efficient execution of GRACE-1L, GRACE-2L and GRACE-3L interatomic potentials without TensorFlow, PyTorch or JAX.
CPU execution is standalone; GPU execution uses Kokkos for NVIDIA CUDA and AMD HIP.
Use ASE-compatible calculators, batch processing, an optional TorchSim adapter or the shared kernels in LAMMPS with MPI.
PACE (ACE) potentials fitted with pacemaker run from their `.yaml` ([PACE models](docs/pace.md)).

**[Read the manual](docs/index.md)** · [Installation](docs/installation.md) · [API reference](docs/api.md)

## Install

Install one package from PyPI ([source builds](docs/installation.md#cpu-from-source) for other targets):

| Backend | Install |
| --- | --- |
| CPU | `pip install grace-runtime` |
| NVIDIA | `pip install "grace-runtime-cuda12[cublas]"` |
| AMD | `pip install grace-runtime-rocm6` |

GPU packages install the matching CPU package automatically.
See the manual for [GPU requirements](docs/backends.md) and [environment setup](docs/installation.md#python-environment).

## First calculation

```python
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator

atoms = bulk("W", "bcc", a=3.165, cubic=True)
atoms.calc = GRACERuntimeCalculator("GRACE-2L-OMAT-medium-ft-E")
print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())            # eV/Å; reuses the calculation above
print(atoms.calc.backend)
```

Device and CPU-thread selection are automatic. Energy, forces and stress are computed together and cached.
Named models download on first use; local `kokkos.npz` files work too.

## Guides

The [manual](docs/index.md#guides) covers [batching and MD](docs/usage.md), [relaxation](docs/relax.md),
[properties](docs/properties.md), [dispersion](docs/dispersion.md), [PACE models](docs/pace.md), [TorchSim](docs/torchsim.md) and
[development](docs/development.md). An interactive tour is in the [ASE notebook](python/examples/ase/grace_runtime_ase.ipynb).

## LAMMPS

The [LAMMPS fork](https://github.com/yury-lysogorskiy/lammps/tree/grace-runtime) supplies pair styles
`grace/runtime` (CPU) and `grace/runtime/kk` (GPU), which take the GRACE-1L / 2L / 3L model from the `kokkos.npz` given to
`pair_coeff`, and `pace/runtime(/kk)` for PACE models. Installing the Python package does not install LAMMPS:
[build and run it](docs/lammps.md).

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

# PACE models

[Documentation map](index.md#guides) · [Usage](usage.md)

grace-runtime also evaluates PACE (linear and nonlinear ACE) potentials fitted with pacemaker, on single- and
multi-core CPUs (SIMD + OpenMP) and, with a GPU module, on NVIDIA and AMD GPUs. It reads the B-basis `.yaml` pacemaker
writes (`output_potential.yaml`, the interim and ladder-step files) directly; python-ace is not needed.

```python
from ase.io import read
from grace_runtime import GRACERuntimeCalculator

atoms = read("structure.extxyz")
atoms.calc = GRACERuntimeCalculator("output_potential.yaml")
# Single-core CPU:
# atoms.calc = GRACERuntimeCalculator("output_potential.yaml", device="cpu", num_threads=1)
# Multi-core CPU (default: OMP_NUM_THREADS or all physical cores):
# atoms.calc = GRACERuntimeCalculator("output_potential.yaml", device="cpu", num_threads=8)
print(atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
```

The `.yaml` is converted to its C-tilde basis on the fly when the model is loaded. LAMMPS
`pair_style pace/runtime` and `pace/runtime/kk` read the `.yaml` directly too, with the same `pair_coeff` as
`pair_style pace`:

```text
pair_coeff * * output_potential.yaml C
```

`Model`, `compute_batch`, `relax_batch` (`grace-runtime relax`), `eos_batch`, `elastic_batch`, `phonons_batch` and
`GRACERuntimeModel` (TorchSim) take PACE `.yaml` files the same way, and `dispersion=` adds DFT-D3 to them.

## Supported features

Every feature pacemaker writes:

- radial bases `ChebExpCos`, `ChebPow`, `ChebLinear`, `SBessel`, per element pair, with the `cos` cutoff;
- inner cutoff `density` (the default of older files), `distance` and `zbl`, with exponential or ZBL core repulsion;
- embeddings `FinnisSinclair` and `FinnisSinclairShiftedScaled`, up to 64 densities, the density cutoff
  `rho_core_cut` / `drho_core_cut`;
- up to 128 elements, basis functions up to rank 6 and lmax 9, up to 64 radial functions per pair, `lint` couplings,
  `skip` flags;
- pair blocks that take their radial basis or core repulsion from the reverse pair block, as in LAMMPS.

A file with anything else (another radial basis or embedding, larger limits, the `.yace` format) raises `ValueError`
naming the key.

## Extrapolation grade

With the model's active set (the `.asi` pacemaker writes, `pace_activeset`), `uq=True` gives the extrapolation grade
`gamma` per atom, as python-ace's `PyACECalculator` with `set_active_set` and LAMMPS `pair_style pace/extrapolation`
(linear or non-linear grade, from the shape of the active set).

Evaluating `gamma` costs about the square of the number of basis functions per atom (the active-set matrix-vector
product), and is skipped completely when UQ is off. In MD or long trajectories, evaluate it only every `N` steps.

In Python, switch it on and off on the calculator with `enable_uq()` and `disable_uq()` (or pass `uq=True` to
`compute_batch` on selected frames):

```python
from grace_runtime import GRACERuntimeCalculator, Model

model = Model("output_potential.yaml", asi="output_potential.asi")
calc = GRACERuntimeCalculator(model)  # uq=False by default
atoms.calc = calc

# Switch on only when needed (for example every N steps):
calc.enable_uq()
atoms.get_potential_energy()
gamma = calc.results["gamma"]
calc.disable_uq()
```

In LAMMPS, `pair_style pace/runtime` and `pace/runtime/kk` take the `.asi` after the `.yaml`, as
`pair_style pace/extrapolation` does. `fix pair` with `gamma 1` raises the internal `gamma_flag` only on multiples of
its step interval `N` (and resets it to `0` after the step, so intermediate steps pay no UQ cost):

```text
pair_style pace/runtime
pair_coeff * * output_potential.yaml output_potential.asi C

# Raise gamma_flag and evaluate gamma only every 100 steps:
fix g all pair 100 pace/runtime gamma 1
compute gmax all reduce max f_g
thermo 100
thermo_style custom step temp pe c_gmax

# Turn off completely if no longer needed:
# unfix g
```

PACE models have no `atomic_sigma` / `gmm_cluster`. The grade and its active sets: Y. Lysogorskiy, A. Bochkarev,
M. Mrovec, R. Drautz, Phys. Rev. Materials 7, 043801 (2023).

## Device, precision, accuracy

- PACE models run on single- and multi-core CPUs (`num_threads` / `OMP_NUM_THREADS`, with MPI in LAMMPS) and, with a
  GPU module (`grace-runtime-cuda12`, `grace-runtime-rocm6`), on the GPU.
- The evaluation is in double precision. Single precision does not reach the accuracy below: large terms cancel in
  the densities.
- The radial functions come from cubic tables per element pair: 1024 intervals above a quarter of the pair cutoff,
  512 between a tenth and a quarter (`Model(path, spline=N)`: N and N / 2; `spline=0` evaluates them directly);
  pairs closer than a tenth of their cutoff are evaluated directly. Against python-ace (with fine splines), the tests require energies within 1e-6 eV/atom, forces
  within 2e-5 eV/Å and stress within 1e-5 relative; with `spline=0`, 1e-10 eV/atom and 3e-8 relative.
- `chunksize` has no effect on the CPU. Per thread the engine needs a force array of the structure (24 bytes per
  atom) and buffers for the neighbourhoods of 8 atoms.
- Energy-only evaluations (`energy_only=True` in Python, and LAMMPS `eflag_only` calls such as Monte Carlo trial moves
  in `fix atom/swap`) run only the forward basis, product and embedding passes, skipping the adjoint (`dE/dA`) and
  neighbour force/virial passes on both CPU (`pace/runtime`) and GPU (`pace/runtime/kk`).

## Performance

One core of an Intel Xeon Gold 6148 (AVX-512), energy, forces and stress of cells of 3500-4100 atoms, against the same
models in LAMMPS `pair_style pace` (recursive evaluator):

| Model | lmax | Rank | grace-runtime (us/atom/step) | `pair_style pace` | Speed-up |
| --- | --- | --- | --- | --- | --- |
| Carbon, diamond | 3 | 6 | 29 | 101 | 3.4x |
| Water | 3 | 6 | 60 | 200 | 3.4x |
| Al-Li | 3 | 6 | 37 | 119 | 3.2x |
| Mo-Nb-Ta-W | 3 | 5 | 24 | 64 | 2.7x |
| Si | 4 | 6 | 68 | 202 | 3.0x |
| Pt-Rh | 6 | 4 | 180 | 704 | 3.9x |

Models with lmax up to 4 run faster than the GRACE-1L-OMAT-medium foundation model on the same cells.

LAMMPS loop time on one GPU, in us/atom/step, for `pair_style pace/runtime/kk` and for the Kokkos style `pace/kk` of
the LAMMPS branch that ships it, both in double precision. On the MI250X, one GPU means one GCD. Each cell shows carbon
(32768 atoms) / water (27648 atoms; 34560 on the MI250X):

| GPU | `pace/runtime/kk` | `pace/kk` |
| --- | --- | --- |
| NVIDIA A100 | 0.43 / 0.71 | 0.78 / 1.64 |
| NVIDIA L40S | 0.58 / 1.05 | 0.97 / 1.89 |
| NVIDIA RTX 3090 | 1.20 / 2.10 | 1.57 / 3.22 |
| AMD MI250X (one GCD) | 0.78 / 1.30 | 1.66 / 3.87 |

`pace/runtime/kk` is faster on all four GPUs, by 1.3x to 3.0x (least on the RTX 3090, most on the MI250X).

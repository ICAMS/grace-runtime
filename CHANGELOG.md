# Changelog

Notable changes of grace-runtime, grace-runtime-cuda12 and grace-runtime-rocm6 (they share one version). Each
release has a section "Changed defaults" when a default behaves differently.

## Unreleased

First release 0.1.0 in preparation.

## [0.0.1.dev1](https://github.com/ICAMS/grace-runtime/releases/tag/0.0.1.dev1) (2026-10-26)

Pre-release on TestPyPI only: the second rehearsal of the release pipeline, now with the GPU packages
`grace-runtime-cuda12` (NVIDIA, CUDA 12) and `grace-runtime-rocm6` (AMD, ROCm 6). The GPU packages pin the same
version of `grace-runtime`: their interface to the CPU package changed in this release.

### Added

- **AMD GPUs** (HIP / ROCm 6): the fused GRACE-1L / 2L / 3L engines on MI210 / MI250 / MI250X (gfx90a) and
  MI300A / MI300X (gfx942), in the Python package (`device="rocm"`, `"rocm:N"`) and in the LAMMPS styles
  `grace/{1l,2l,3l}/fused/kk`. Package `grace-runtime-rocm6`.
- **Batched relaxation**: `relax_batch` and `relax_batch_iter` relax many structures in one GPU batch (FIRE as in
  ASE; optional cell relaxation as `ase.filters.FrechetCellFilter`, pressure, fixed atoms, `fix_symmetry`).
  Structures that converge leave the batch and waiting ones take their place. Command line:
  `grace-runtime relax MODEL IN -o OUT`. `compute_batch_iter` evaluates a stream of structures.
- **Properties in batches**: `eos_batch` (equation of state), `elastic_batch` (elastic constants, stress method),
  `phonons_batch` and `phonopy_forces_batch` (phonopy).
- **Dispersion correction DFT-D3(BJ)**, optionally with the Axilrod-Teller-Muto three-body term, on the CPU, CUDA and
  HIP: argument `dispersion=` of the calculator, `compute_batch`, the relaxation and property functions and the
  TorchSim model, as a short form (`"d3-pbe"`, `"d3-r2scan"`, `"d3-pbe-atm"`) or `D3(...)`. Off unless given.
- `Model` takes a registry name or a `pathlib.Path`; one loaded `Model` can be passed to all functions.
- CPU kernels for Apple Silicon (NEON, Accelerate); source builds on macOS with Homebrew or conda OpenMP.
- Example notebook with ASE: energies, relaxations, EOS, elastic constants, UQ, MD.

### Changed

- Faster GPU kernels: CUDA bond kernels for GRACE-1L / 2L (A100 2L 4-16 % faster), GRACE-2L / 3L kernels on CUDA,
  matrix-core kernels on AMD GPUs.
- The GPU calculator builds no CPU neighbour list.
- GPU per-atom energies: the last step (scale and shift) runs in fp64, as on the CPU; GRACE-1L sums one of its
  intermediate terms in fp64. GPU results change at the level of fp32 rounding (GRACE-1L on strongly compressed
  cells: up to about 1e-4 eV per atom).
- The engine self-tests compare with the fp32 rounding bound: models whose first-layer outputs cancel, which the GPU
  refused before, now run there.
- A `Model` loaded with a `spline` other than 1024 uses it on the GPU too (before: always 1024).
- New errors for inputs that were accepted before: a negative, NaN or infinite `skin`; `GRACE_RUNTIME_STRICT=1` with
  `device="auto"` on a machine without a usable GPU; a malformed `kokkos.npz` (each array is checked against the
  model's own sizes, with a message that names the array).

### Changed defaults

- GPU `compute_batch` and the TorchSim model build their neighbour lists on the GPU (before: on the host);
  `num_threads` applies to their CPU paths only.
- CUDA: GRACE-1L / 2L use the GPU bond kernels (no switch).
- AMD gfx90a (MI250X): the matrix-core kernels are on by default (`GRACE_FUSED_MFMA=0` turns them off); on gfx94x they
  are opt-in. The GRACE-3L FC fold (`fc_fold = 1`) takes effect on AMD GPUs too.
- macOS source builds: `GRACE_RUNTIME_ISAS=native`; GRACE-3L CPU matrix products through Accelerate.

### Removed

- The matscipy neighbour list backend (`neighbor_backend="matscipy"`) and the `matscipy` extra; the built-in list is
  the only one.

### Fixed

- A failed GPU kernel launch stops the run instead of returning wrong results.
- `free_gpu_memory()` frees every per-atom GPU buffer, also those of the GRACE-3L matrix-core kernels.
- A forced older GPU module (PTX) needs a driver for CUDA 12.9 or newer; this is checked.
- GRACE-1L on Turing GPUs (RTX 20xx, T4): kernel launch bounds.
- Runs with no central atoms (empty MPI ranks): neighbour count and UQ.
- GPU bond compaction with atoms of unmapped elements.
- `GRACE_RUNTIME_ROCM_ARCH` works when the architecture cannot be read from the PCI bus.
- Build warnings with GCC 16.

## [0.0.1.dev0](https://github.com/ICAMS/grace-runtime/releases/tag/0.0.1.dev0) (2026-10-20)

Pre-release on TestPyPI only, to test the release pipeline: the CPU package `grace-runtime` (manylinux2014 x86_64
wheels for CPython 3.10, 3.11 and >= 3.12, with x86-64-v3 and x86-64-v4 builds).

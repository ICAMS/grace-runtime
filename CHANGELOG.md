# Changelog

Notable changes of grace-runtime, grace-runtime-cuda12 and grace-runtime-rocm6 (they share one version). Each
release has a section "Changed defaults" when a default behaves differently.

## [0.1.0](https://github.com/ICAMS/grace-runtime/releases/tag/0.1.0) (2026-10-29)

First release on PyPI: `grace-runtime` (CPU), `grace-runtime-cuda12` (NVIDIA) and `grace-runtime-rocm6` (AMD). It
contains everything of the 0.0.1.dev1 rehearsal below, and the changes listed here.

### Added

- **PACE models** (pacemaker's ACE potentials): energies, forces, stress and the extrapolation grade gamma on the CPU,
  NVIDIA and AMD GPUs. `Model` reads the B-basis `.yaml` (and the active set `.asi`) directly. In LAMMPS the style
  `pace/runtime(/kk)` runs them; on GPUs it is 1.3x to 3.0x faster than `pace/kk`, both in double precision.
- **macOS arm64 wheels** (Apple Silicon, macOS 11 and later).
- **LAMMPS**: one style `grace/runtime(/kk)` for GRACE-1L, 2L and 3L models. It reads the model kind from the file.
  A keyword that does not fit the model is an error; a keyword of the other backend (CPU or Kokkos) gives a warning.
  A repeated `pair_style` rebuilds the engine from the loaded model.
- grace-runtime reads the model kind and format version that newer tensorpotential exports write into `kokkos.npz`.
  Older exports still load; a file from a newer exporter stops with a clear message.
- `Model(store_max_gb=)`, `Model.store_max_gb` and `backend_info["store"]`: the memory budget of the forward store,
  with a `RuntimeWarning` when the store does not fit and the backward pass recomputes it.
- `Model.compute(positions=, image=, shifts=)` and `NeighborList.image` / `.shifts`: the CPU engines form the bond
  vectors from the positions and a shift index per bond (PACE, 32768 carbon atoms, 20 threads: 16 % faster per step).
- The documentation site with an API reference.

### Changed

- NVIDIA without the `[cublas]` extra: GRACE-3L takes 0-15 % longer than with cuBLAS (0.0.1.dev1: up to 16x), with
  the same results within fp32 rounding. The extra stays optional.
- The neighbour list keeps 8 bytes per bond instead of 40.
- `energy_only=True` calls on the CPU skip the derivative work and its buffers (GRACE-1L/2L/3L and D3), and GRACE-3L
  UQ projects all atoms of a block in one matrix product.
- A GPU calculation that runs out of memory retries with smaller chunks, also when a kernel launch fails.
- More input checks: atomic numbers, property ranges (`eos_batch`, `elastic_batch`), nearly flat cells, two atoms at
  the same position (an error).

### Changed defaults

- `GRACERuntimeCalculator(energy_only=False)`: the first request computes energy, forces and stress together, and
  later requests on the same atoms reuse them. Pass `energy_only=True` for energy-only screening.
- The forward store (GRACE-2L layer 1, GRACE-3L layers 1-2) follows the available memory per process, in Python and in
  LAMMPS; above it the backward pass recomputes the layers. Before: GRACE-2L a fixed 8 GB, GRACE-3L no limit.
- The automatic memory budgets (forward store, GRACE-3L radial tables) follow a cgroup memory limit (Slurm `--mem`,
  containers) when there is one, instead of the memory of the node.
- NVIDIA without cuBLAS: the built-in Kokkos GEMMs run (`blas()` reports `kkgemm`); `GRACE_RUNTIME_NO_KKGEMM=1` brings
  back the earlier kernels. Without the `[cublas]` extra a system cuBLAS 12 is used when found; one older than 12.0
  (12.8 on compute capability 10 and above) is skipped with a warning.
- The automatic GPU chunk size goes down to 512 atoms when memory is short (before: 1024).
- C++ builds of libgrace and its tests with CMake: `GRACE_ARCH_FLAGS` defaults to `-march=native`. The Python
  package, the wheels and LAMMPS are not affected.

### Removed

- LAMMPS styles `grace/{1l,2l,3l}/fused(/kk)`: use `grace/runtime(/kk)`.
  The TensorFlow styles `grace/1layer/chunk`, `grace/2layer/chunk` and `grace/2layer/parallel`: use `grace/runtime`
  with a `kokkos.npz` export. `pair_style grace` (TensorFlow) and `grace/fs(/kk)` stay.
- LAMMPS `pace/kk/fp64` (`/device`, `/host`): `pace/kk` runs in double precision again. On GPUs with slow double
  precision it is about 2x slower than its earlier mixed-precision default, which failed the accuracy tests.

### Fixed

- Systems above about 33 million atoms on large GPUs (64-bit offsets of the neighbour and bond lists).
- A GPU batch is usable again after an allocation failure.
- GRACE-3L on the GPU after a re-setup with other radial-table settings.
- Open cell directions that are not orthogonal to the periodic ones.
- AMD RDNA (32-lane) GPUs in the GRACE-2L backward kernel.
- `free_gpu_memory()` also releases calculators that only a reference cycle keeps alive, and their GPU models.
- GRACE-3L on the CPU ignored `num_threads` in some loops when `OMP_NUM_THREADS` was larger.
- `grace-runtime info` with no visible CUDA device.
- The rocm6 wheels claimed manylinux2014 but need glibc 2.28: they are tagged `manylinux_2_28` now.
- LAMMPS: `comm_style tiled` with the Kokkos GRACE-2L style gave wrong forces and GRACE-3L stopped; `fix pair` UQ on
  empty CPU ranks; `pace/kk` and `pace/extrapolation/kk` stopped on the GPU for multi-element files without pair
  blocks.

### Known issues

- LAMMPS `fix atom/swap` with Kokkos pair styles (also `lj/cut/kk`) makes other swap decisions than on the CPU.
- LAMMPS `pair_style grace` (TensorFlow) with a two-layer model gives wrong results on more than one MPI rank: use
  `grace/runtime`.

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

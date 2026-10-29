# LAMMPS

[Documentation map](index.md#guides) · [Models](models.md) · [PACE models](pace.md)

GRACE and PACE models run in LAMMPS through four pair styles of the
[LAMMPS fork](https://github.com/yury-lysogorskiy/lammps/tree/grace-runtime):

| Style | Runs on | Models |
|---|---|---|
| `grace/runtime` | CPU (MPI + OpenMP) | GRACE-1L / 2L / 3L (`kokkos.npz`) |
| `grace/runtime/kk` | GPU (Kokkos CUDA or HIP) | GRACE-1L / 2L / 3L (`kokkos.npz`) |
| `pace/runtime` | CPU (MPI + OpenMP) | PACE (pacemaker `.yaml`) |
| `pace/runtime/kk` | GPU (Kokkos CUDA or HIP) | PACE (pacemaker `.yaml`) |

These styles use the same kernels as the Python package, but the Python package does not install LAMMPS. Build
LAMMPS as described below.

## 1. Get the source

You need git, CMake 3.22 or newer, a C++ compiler with OpenMP and MPI (without MPI, leave out `-D BUILD_MPI=yes`
below). Use GCC 12 or newer: older GCC vectorizes the CPU kernels poorly (one thread, bcc W: with GCC 11 GRACE-3L runs
about 37 % slower, with GCC 9 GRACE-1L about 4.6x and GRACE-3L about 6.6x slower). Clone only the latest commit of the
`grace-runtime` branch:

```bash
git clone --depth 1 -b grace-runtime https://github.com/yury-lysogorskiy/lammps.git
cd lammps
```

## 2. Build

The pair styles are part of the `ML-PACE` package. When CMake configures the build, it downloads grace-runtime
0.1.0 from GitHub, so the configure step needs network access. It also downloads the ML-PACE library. Pick one of the
three recipes; run each from the `lammps` folder.

### CPU

```bash
cmake -S cmake -B build-cpu -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="-march=native" \
  -D BUILD_MPI=yes -D BUILD_OMP=yes -D PKG_ML-PACE=yes
cmake --build build-cpu -j 8
```

`-march=native` compiles for the CPU of the build machine, so that the styles use its SIMD instructions. Run the
binary on the same kind of CPU. `BUILD_OMP=yes` enables the OpenMP threads that the styles use inside each MPI rank;
the `OPENMP` package is not needed.

### NVIDIA GPU (Kokkos CUDA)

You need the CUDA toolkit 12.2 or newer, 12.8 for Blackwell GPUs (`nvcc` on `PATH`).

```bash
cmake -S cmake -B build-cuda -C cmake/presets/kokkos-cuda.cmake -D CMAKE_BUILD_TYPE=Release \
  -D Kokkos_ARCH_AMPERE86=yes -D BUILD_MPI=yes -D PKG_ML-PACE=yes
cmake --build build-cuda -j 8
```

The preset enables Kokkos with CUDA and uses the compiler wrapper `nvcc_wrapper`. `Kokkos_ARCH_...` must match
your GPU. Find its compute capability with `nvidia-smi --query-gpu=name,compute_cap --format=csv` and pick the flag:

| Compute capability | Examples | Flag |
|---|---|---|
| 7.0 | V100 | `Kokkos_ARCH_VOLTA70` |
| 7.5 | T4, RTX 20xx | `Kokkos_ARCH_TURING75` |
| 8.0 | A100 | `Kokkos_ARCH_AMPERE80` |
| 8.6 | RTX 30xx, A40, RTX A6000 | `Kokkos_ARCH_AMPERE86` |
| 8.9 | L40S, RTX 40xx | `Kokkos_ARCH_ADA89` |
| 9.0 | H100, H200 | `Kokkos_ARCH_HOPPER90` |
| 10.0 | B200 | `Kokkos_ARCH_BLACKWELL100` |
| 12.0 | RTX 50xx, RTX PRO 6000 Blackwell | `Kokkos_ARCH_BLACKWELL120` |

A binary runs only on GPUs of the architecture it was built for.

### AMD GPU (Kokkos HIP)

You need ROCm 6.2 or newer (`hipcc` on `PATH`). For MI210 / MI250 / MI250X:

```bash
cmake -S cmake -B build-hip -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER=hipcc \
  -D PKG_KOKKOS=yes -D Kokkos_ENABLE_HIP=yes -D Kokkos_ARCH_AMD_GFX90A=yes \
  -D GPU_TARGETS=gfx90a -D CMAKE_CXX_FLAGS="-munsafe-fp-atomics" -D BUILD_MPI=yes -D PKG_ML-PACE=yes
cmake --build build-hip -j 8
```

`GPU_TARGETS` names the same target as `Kokkos_ARCH_...` for ROCm's own CMake files, which otherwise warn and choose
a default target when no GPU is visible (for example on a login node).

On HPE Cray systems (`PrgEnv-amd` with `cray-mpich`), also add `-D GRACE_BLAS=OFF`: the Cray compiler wrappers find
a BLAS at configure time that `hipcc` does not link, and the link then fails with `undefined symbol: cblas_sgemm`. The
setting affects only the CPU styles, which then use their built-in matrix product.

Other AMD GPUs ([GPU backends](backends.md#amd-rocm)):

| GPU | Flags |
|---|---|
| MI300X, MI325X | `-D Kokkos_ARCH_AMD_GFX942=yes -D GPU_TARGETS=gfx942` |
| MI300A | `-D Kokkos_ARCH_AMD_GFX942_APU=yes -D GPU_TARGETS=gfx942` |

### Without network access

Download grace-runtime 0.1.0 on a machine with network access, copy it over and pass its path:

```bash
git clone --depth 1 -b 0.1.0 https://github.com/ICAMS/grace-runtime.git
cmake -S cmake -B build-cpu ... -D LOCAL_GRACE_RUNTIME=$PWD/grace-runtime
```

The ML-PACE library is downloaded too. Download the archive named by `PACELIB_URL` in
`cmake/Modules/Packages/ML-PACE.cmake` and copy it into the build folder as `libpace.tar.gz` (for example
`build-cpu/libpace.tar.gz`) before you configure.

## 3. Get a model

**GRACE foundation model.** [Install](installation.md) the Python package and download the model by name
([Models](models.md) lists them):

```bash
pip install grace-runtime
grace-runtime models download GRACE-2L-OMAT-medium-ft-E
```

The file is `~/.cache/grace/GRACE-2L-OMAT-medium-ft-E/kokkos.npz` (or `$GRACE_CACHE/...` when `GRACE_CACHE` is
set).

**Your own GRACE model.** Export it to `model.npz` as in [Models](models.md#supported-models) and use that path.

**PACE model.** Use the `.yaml` pacemaker writes (`output_potential.yaml`) with `pace/runtime`; see
[PACE models](pace.md).

## 4. Run

Save this input as `in.lmp`. It is a bcc W/Mo alloy of 1024 atoms; `pair_coeff` maps LAMMPS atom types 1 and 2 to W
and Mo, and the model file is given on the command line as the variable `model`:

```text
units        metal
atom_style   atomic
lattice      bcc 3.16
region       box block 0 8 0 8 0 8
create_box   2 box
create_atoms 1 box
set          type 1 type/fraction 2 0.5 4928459
mass         1 183.84
mass         2 95.95

pair_style   grace/runtime
pair_coeff   * * ${model} W Mo

velocity     all create 300 4928459
fix          1 all nvt temp 300 300 0.1
thermo       20
run          100
```

CPU, 4 MPI ranks with 4 OpenMP threads each:

```bash
export OMP_NUM_THREADS=4
mpirun -np 4 build-cpu/lmp -in in.lmp -var model ~/.cache/grace/GRACE-2L-OMAT-medium-ft-E/kokkos.npz
```

One GPU (`build-hip/lmp` on AMD GPUs):

```bash
build-cuda/lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in in.lmp \
  -var model ~/.cache/grace/GRACE-2L-OMAT-medium-ft-E/kokkos.npz
```

`-sf kk` turns `grace/runtime` into `grace/runtime/kk`, so the same input runs on both. The GPU style needs
`newton on` and `neigh half`. For several GPUs, run one MPI rank per GPU.

For a PACE model, write `pair_style pace/runtime` and pass the `.yaml` as the model.

### What you should see

The style prints which model kind it loaded. On the CPU (`grace/runtime`):

```text
[grace/runtime] loading the GRACE-2L model .../GRACE-2L-OMAT-medium-ft-E/kokkos.npz
[GRACE-2L] lmax 4 | L1 42 ch, A1/AA/AAA/AAAA 25/141/107/378 | I 4x32 | L2 32 / 64 ch, ... | spline N 1024
   Step          Temp          E_pair         E_mol          TotEng         Press
         0   300           -12248.756      0             -12209.086      40140.971
```

On the GPU (`grace/runtime/kk`), also the element mapping and the chunk size chosen from the free GPU memory:

```text
[grace/runtime/kk] loading the GRACE-2L model .../GRACE-2L-OMAT-medium-ft-E/kokkos.npz
[grace/runtime/kk] LAMMPS type 1 -> W (model index 83)
[grace/runtime/kk] LAMMPS type 2 -> Mo (model index 43)
[GRACE-2L/KK] chunksize auto: 1126 atoms per chunk (...)
   Step          Temp          E_pair         E_mol          TotEng         Press
         0   300           -12248.756      0             -12209.086      40140.93
```

The energy at step 0 is the same on the CPU and the GPU. Both evaluate the network in fp32, in a different order, so
the last digits and later steps of the trajectory can differ slightly.

Wrong input stops LAMMPS with an error that names the problem: a PACE `.yaml` in `grace/runtime`, a GRACE model in
`pace/runtime`, a keyword that does not fit the model, or units other than `metal`.

## More

The fork's documentation pages describe every keyword, several GPUs, uncertainty (`fix pair ... gamma`) and the
precision:

- [`pair_style grace/runtime`](https://github.com/yury-lysogorskiy/lammps/blob/grace-runtime/doc/src/pair_grace_runtime.rst)
- [`pair_style pace/runtime`](https://github.com/yury-lysogorskiy/lammps/blob/grace-runtime/doc/src/pair_pace_runtime.rst)
- [PACE models](pace.md): PACE features, the extrapolation grade and GPU timings.

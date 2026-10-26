# Installation

[Documentation map](../README.md#guides) · [First calculation](../README.md#first-calculation)

The first release is in preparation. The `0.0.0.dev0` packages on PyPI are placeholders and contain no runtime code.
Use a source checkout or a wheel supplied with a release; installing those placeholders does not install GRACE.

## 1. Create an environment

We recommend [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
following the [GRACEmaker installation guide](https://gracemaker.readthedocs.io/en/latest/gracemaker/install/).
Install it if needed, open a shell initialized for micromamba, then create and activate an environment:

```bash
micromamba create -n grace-runtime -c conda-forge python=3.12 pip cxx-compiler
micromamba activate grace-runtime
```

If you already use `mamba`, use the same `create` and `activate` commands with `mamba` instead of `micromamba`.
Python 3.12 is recommended so the optional TorchSim adapter works too; the runtime supports Python >= 3.10.
`cxx-compiler` supplies the C++ compiler for the current source build. The environment's Python includes its
development headers, so no system Python development package is needed.

If activation reports that the shell is not initialized, follow the micromamba installation guide. For the current
Bash session only, you can use `eval "$(micromamba shell hook --shell bash)"` before activating the environment.

## 2. Install the CPU package

### CPU from source

If you do not have a checkout yet:

```bash
git clone https://github.com/ICAMS/grace-runtime.git
cd grace-runtime
```

With the environment above active, run from the repository root. Pip downloads the Python build and runtime
dependencies; no TensorFlow or GPU toolkit is needed:

```bash
python -m pip install . --config-settings=cmake.define.GRACE_RUNTIME_ISAS=native
```

The native module is optimized for the build CPU, including its SIMD instructions, and may not run on a different
CPU. To build the default dual-ISA Linux x86-64 package instead, use `python -m pip install .`: it builds AVX2/FMA
(`x86-64-v3`) and AVX-512 (`x86-64-v4`) modules and selects one at import. That build needs a compiler that recognizes
both architecture names. Other CPU architectures and macOS build one native module.

Pip installs the build dependencies in an isolated environment. No TensorFlow or GPU toolkit is needed for the
CPU package. Runtime dependencies are `numpy`, `ase`, `tqdm` and `spglib>=2.5`. An existing environment also works
if it has Python >= 3.10 and a C++17 compiler with OpenMP.

If compilation uses too much memory, set `CMAKE_BUILD_PARALLEL_LEVEL=4` before installing. To build a wheel for
another environment on a compatible machine:

```bash
python -m pip wheel --no-deps . --wheel-dir dist/cpu-wheelhouse \
    --config-settings=cmake.define.GRACE_RUNTIME_ISAS=native
```

Install the resulting `.whl` file with `python -m pip install /path/to/package.whl`; pip fetches its runtime
dependencies if necessary. For an offline installation, supply those dependency wheels too. For editable installs
and tests, see [development](development.md).

### macOS

On macOS, also install the OpenMP runtime in the active environment before building:

```bash
micromamba install -c conda-forge llvm-openmp
python -m pip install .
```

The build discovers that runtime; `OpenMP_ROOT` selects another installation. A wheel built this way needs the
OpenMP runtime on the target Mac and a compatible CPU. For TorchSim, use conda-forge PyTorch with the same environment's
`libomp`: mixing the `libomp` shipped with PyPI PyTorch and another OpenMP runtime can abort with `OMP: Error #15`.

### Release wheels

These commands are for runtime wheel releases, **once available**. They do not work with the current PyPI
placeholders. `--only-binary=:all:` prevents pip from silently trying to build a source distribution:

```bash
python -m pip install --only-binary=:all: grace-runtime                     # CPU
python -m pip install --only-binary=:all: grace-runtime grace-runtime-cuda12 # NVIDIA GPU
```

Create an environment with Python and pip first (the `cxx-compiler` package is not needed for prebuilt wheels).
Install the GPU package and CPU package at the same version. Supported wheel targets:

| Package | Platform and runtime |
| --- | --- |
| CPU | Linux x86-64, CPython >= 3.10, AVX2/FMA (`x86-64-v3`) minimum; AVX-512 chosen automatically when available. Build from source for other targets. |
| CUDA 12 | Linux x86-64, compatible NVIDIA driver; no system CUDA toolkit required. GPU families and driver requirements are in the [CUDA guide](../python-cuda/README.md). |
| ROCm 6 | Source build for now; Linux x86-64 with system ROCm 6.2+ and a supported AMD GPU. See the [ROCm guide](../python-rocm6/README.md). |

`grace_runtime.cpu_module` reports the loaded CPU module. `GRACE_RUNTIME_CPU=v3|v4|native` forces an installed
module; it cannot add instructions the CPU lacks.

## 3. Check the installation

```bash
python -c "import grace_runtime; print(grace_runtime.__version__, grace_runtime.cpu_module)"
grace-runtime info
```

The first command prints the installed version and CPU module. The second diagnoses GPU availability and installed
backends; a GPU needs its additional package below. Then run the [first calculation](../README.md#first-calculation).
`grace-runtime info` returns 1 when a visible GPU cannot run natively; this does not mean CPU installation failed.

To return to the environment in a new shell, run `micromamba activate grace-runtime`.

## GPU backends

Installing the CPU package alone does not install a GPU backend. Build a GPU wheel from the **same source revision**
as your CPU package. Installing the matching wheel adds GPU support to the existing calculator and batch APIs.

### NVIDIA GPU (CUDA)

The wheel script needs Linux x86-64, `micromamba` on `PATH` and network access. It obtains CUDA 12.9 and compilers
from conda-forge; you do not need a system CUDA toolkit. From the repository root with your Python environment active,
build only your target architecture first, for example an RTX 3060 (`sm86`):

```bash
export PYTHONS="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export GRACE_KK_ARCHS=sm86 NPROC=4
bash scripts/build_cuda_wheels.sh "$PWD/dist/cuda-wheelhouse"
python -m pip install --no-index --find-links="$PWD/dist/cuda-wheelhouse" grace-runtime-cuda12
grace-runtime info
```

Install the CPU package first, as above: `--no-index` cannot fetch it. Use a fresh output directory when rebuilding
the same version, to avoid choosing an older wheel. Other GPU targets and full release-wheel builds are described
in [development](development.md#a-cuda-wheel-for-other-gpu-architectures).

### AMD GPU (ROCm)

The Python ROCm backend is not released yet; build it with your system ROCm 6.2+ and `g++`, as described in the
[ROCm package guide](../python-rocm6/README.md#source-build-for-another-amd-gpu). Select `device="rocm"` (or
`"rocm:N"`; `"hip"` is an alias) to require it. Without a usable GPU backend, `device="auto"` uses the CPU.

The Kokkos engines also run in [LAMMPS](../README.md#lammps) with HIP. MI250X (gfx90a) has been tested; gfx942
(MI300), gfx1100 (RX 7900) and gfx1201 (RX 9070) have been compiled but not run. Engine build options are in
[development](development.md#gpu-engine-tests-and-tuning).

## Optional dependencies

Add extras to the source install, for example `python -m pip install ".[phonons]"`. For release wheels, the same
syntax is `python -m pip install "grace-runtime[phonons]"`.

| Extra | Adds | Purpose |
| --- | --- | --- |
| `phonons` | `phonopy>=2.20` | `phonons_batch`, `phonopy_forces_batch` |
| `torchsim` | `torch>=2`, `torch-sim-atomistic>=0.6`, on Python >= 3.12 | TorchSim adapter; on Python 3.10/3.11 this extra installs neither package. |
| `test` | `pytest` | Python test suite; GPU and other optional tests need their own dependencies. |

The CUDA wheel's `cublas` extra adds `nvidia-cublas-cu12` for UQ projections. It is optional; do not use the current
PyPI placeholder to obtain it. `matscipy` is not needed: the neighbor list is built in.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Installation succeeds but no calculator or CLI exists | Check that a runtime package was installed, rather than the PyPI placeholder. Install from source for now. |
| `No matching distribution found` with `--only-binary` | Check release availability, Python, platform and CPU requirements. Use the source build above when no wheel exists. |
| `x86-64-v3` CPU error, or compiler does not recognize `x86-64-v3` | Build the CPU package with `GRACE_RUNTIME_ISAS=native` on the target machine. |
| GPU present but calculation uses CPU | Run `grace-runtime info`, inspect `calc.backend_info["fallback"]`, and use `device="cuda"` or `"rocm"` to require GPU execution. |
| `one GPU per process` | Select one GPU before starting Python and use one process per GPU. See [multiple calculators](advanced.md#several-calculators-in-one-process). |

`grace-runtime info` diagnoses installed modules and visible devices; it does not evaluate a model. Follow it with
the [first calculation](../README.md#first-calculation), selecting your intended device.

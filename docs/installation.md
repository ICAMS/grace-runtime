# Installation

[Manual](index.md) · [GPU requirements](backends.md)

## Python environment

We recommend [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
(`mamba` works too):

```bash
micromamba create -n grace-runtime -c conda-forge python=3.12 pip
micromamba activate grace-runtime
```

## Choose a package

| Backend | Install | Wheel requirements |
| --- | --- | --- |
| CPU | `pip install grace-runtime` | Linux x86-64 (AVX2/FMA), also [WSL2 on Windows](#windows); macOS arm64 (Apple M1 and newer) |
| NVIDIA CUDA 12 | `pip install "grace-runtime-cuda12[cublas]"` | Linux x86-64, compatible NVIDIA driver |
| AMD ROCm 6 | `pip install grace-runtime-rocm6` | Linux x86-64, system ROCm 6.2+ |

Either GPU package installs its matching CPU package automatically. No compiler is needed for wheels. On macOS with
PyTorch, see [macOS](#macos) (one OpenMP runtime per process).
The `[cublas]` extra adds NVIDIA cuBLAS (`nvidia-cublas-cu12`, several hundred MB): without it GRACE-3L takes up
to about 15 % longer on the GPUs we measured, and UQ slightly longer ([per GPU](backends.md#nvidia-cuda)). Without it,
another cuBLAS 12 is used if one is found (PyTorch's, or a CUDA toolkit's), else the package's own GPU kernels, with
the same results within fp32 round-off.
See [GPU backends](backends.md) for supported cards and runtime requirements; build from source for other targets.

## Verify installation

```bash
grace-runtime info
```

Then run the [first calculation](index.md#first-calculation). If a visible GPU cannot run natively,
`grace-runtime info` returns 1; see [backend diagnostics](advanced.md#backend-diagnostics).

## Windows

Native Windows is not supported yet. Use [WSL2](https://learn.microsoft.com/windows/wsl/install) (Windows 10 or 11)
with an Ubuntu distribution, and install the Linux packages inside it:

```bash
pip install grace-runtime                    # CPU
pip install "grace-runtime-cuda12[cublas]"   # NVIDIA GPU
grace-runtime info
```

NVIDIA GPUs work through the Windows NVIDIA driver ([CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)):
install the driver on Windows only, not inside WSL. `grace-runtime info` shows whether the GPU is visible.

## Build from source

### CPU from source

If needed, install a compiler with C++17 and OpenMP support. Use GCC 12 or newer: older GCC builds much slower CPU
kernels (with GCC 11 GRACE-3L about 37 % slower, with GCC 9 up to 6.6x slower).

```bash
micromamba install -c conda-forge cxx-compiler
```

On macOS, install [OpenMP](#macos) too. Then clone and install:

```bash
git clone https://github.com/ICAMS/grace-runtime.git
cd grace-runtime
pip install .
```

### macOS

Before building, install OpenMP in the active environment:

```bash
micromamba install -c conda-forge llvm-openmp
```

The macOS wheel ships its OpenMP runtime and shares it with conda-forge packages in the same process (NumPy, PyTorch).
PyTorch from PyPI ships its own, and the two abort with `OMP: Error #15`: for TorchSim on macOS, use conda-forge
PyTorch, which shares the environment's OpenMP runtime.

### GPU source builds

Install the CPU package first, then build a backend from the same source revision:
[NVIDIA CUDA](backends.md#build-the-nvidia-backend) or [AMD ROCm](backends.md#build-the-amd-backend).
For editable installs and library tests, see [development](development.md).

## Optional dependencies

| Feature | Install |
| --- | --- |
| Phonons | `pip install "grace-runtime[phonons]"` |
| TorchSim adapter (Python >= 3.12; on a CUDA 12 driver torch from the CUDA 12 index, see [TorchSim](torchsim.md)) | `pip install "grace-runtime[torchsim]"` |
| Tests | `pip install "grace-runtime[test]"` |
| cuBLAS for the NVIDIA package (GRACE-3L up to ~15 % faster, UQ slightly faster) | `pip install "grace-runtime-cuda12[cublas]"` |

For source installs, use the same extra with `.`, for example `pip install ".[phonons]"`.

# GPU backends

[Installation](installation.md) · [Device selection](advanced.md#device-selection)

GPU inference needs a backend package matching the CPU package version. GPU wheels target Linux x86-64;
ARM hosts such as GB200/GB300 are not supported by these wheels. Use one process per GPU (or AMD GCD).
The runtime selects a usable GPU automatically; an explicit `device="cuda"` or `"rocm"` requires it.

## NVIDIA CUDA

The CUDA 12 backend needs an NVIDIA driver >= 525; Blackwell GPUs need >= 570.
No system CUDA toolkit is needed for the wheels.

| GPU family | Included module |
| --- | --- |
| V100, RTX 20xx, T4, T1000 | `sm70` |
| A100, A30, RTX 30xx/40xx, A-series, L40S | `sm80` |
| H100/H200 | `sm90` |
| B200/B300 | `sm100` |
| RTX 50xx | `sm120` |

Other architecture families need a native build or the optional [PTX fallback](advanced.md#backend-diagnostics).
The `[cublas]` extra (`grace-runtime-cuda12[cublas]`) loads cuBLAS for the dense GEMMs. The GPU modules take the first
cuBLAS 12 they find: an installed `nvidia-cublas-cu12` (the extra, or one that another package such as PyTorch
installed), else a system one (a CUDA 12 toolkit on the loader path), else the built-in Kokkos GEMMs, with the same
results within fp32 round-off. It must be 12.0 or newer, 12.8 or newer on Blackwell GPUs; an older one is skipped with
a warning. `GRACE_RUNTIME_NO_CUBLAS=1` forces the built-in GEMMs; `grace-runtime info` names the cuBLAS in use. On the
GPUs below, without cuBLAS GRACE-3L energies and forces take 0-14 % longer and 1L / 2L UQ 2-11 % longer; 1L / 2L
energies and forces run at the same speed. GRACE-3L-OMAT-large energies and forces on bcc W/Mo, time without cuBLAS
divided by the time with it:

| GPU | 1024 atoms | 8192 atoms |
| --- | --- | --- |
| RTX 3060 | 1.13 | 1.12 |
| RTX 5060 Ti | 1.04 | 1.00 |
| A100 | 1.10 | 1.06 |
| L40S | 1.14 | 1.08 |

### Build the NVIDIA backend

From the repository root, with the [CPU package installed from source](installation.md#cpu-from-source)
and the target Python environment active, build for your GPU. The script requires Linux x86-64 and micromamba;
it downloads the CUDA toolkit and compilers. For example, RTX 3060 (`sm86`):

```bash
export PYTHONS="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export GRACE_KK_ARCHS=sm86 NPROC=4
bash scripts/build_cuda_wheels.sh "$PWD/dist/cuda-wheelhouse"
pip install --no-index --find-links="$PWD/dist/cuda-wheelhouse" grace-runtime-cuda12
grace-runtime info
```

| GPU generation | Build targets |
| --- | --- |
| Volta / Turing | `sm70`, `sm75` |
| Ampere / Ada | `sm80`, `sm86`, `sm89` |
| Hopper | `sm90` |
| Blackwell | `sm100`, `sm120` |

The default multi-family build uses `GRACE_KK_ARCHS='sm70;sm80;sm90;sm100;sm120'`.
Use a fresh output directory for each rebuild; lower `NPROC` if compilation runs out of memory.

## AMD ROCm

The ROCm 6 backend needs system ROCm 6.2+ and a supported GPU. Make the HIP runtime visible to the dynamic loader;
load your system's ROCm module if needed. The wheel bundles no ROCm runtime. rocBLAS is optional.

| GPU | Module | Validation |
| --- | --- | --- |
| MI210 / MI250 / MI250X | `gfx90a` | Tested on MI250X |
| MI300A / MI300X / MI325X | `gfx942` | Built; execution not yet tested |

AMD modules run only on their target architecture. Without a usable module, `device="auto"` can fall back to CPU;
`device="rocm"` raises instead.

### Build the AMD backend

Install the [CPU package from the same source revision](installation.md#cpu-from-source) first.
From the repository root, with the target Python environment active, ROCm and `g++` available,
build for your GPU; for example, RX 7900 (`gfx1100`):

```bash
export ROCM_PATH=/opt/rocm GRACE_KK_ARCHS=gfx1100 NPROC=4
export PYTHONS="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
bash scripts/build_rocm_wheels.sh "$PWD/dist/rocm-wheelhouse"
pip install --no-index --find-links="$PWD/dist/rocm-wheelhouse" grace-runtime-rocm6
grace-runtime info
```

Other targets include `gfx906`, `gfx908`, `gfx90a`, `gfx940`, `gfx942`, `gfx1030`, `gfx1103` and `gfx1201`.
Use a fresh output directory when rebuilding the same version.

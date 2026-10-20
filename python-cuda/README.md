# grace-runtime-cuda12

NVIDIA GPU backend (CUDA 12) of [grace-runtime](../README.md): the fused GRACE-1L / 2L / 3L Kokkos kernels (as
LAMMPS `grace/{1l,2l,3l}/fused/kk`) for `GRACERuntimeCalculator` and `grace_runtime.compute_batch`.

    pip install grace-runtime grace-runtime-cuda12            # cuBLAS (UQ projections) used when installed
    pip install "grace-runtime-cuda12[cublas]"               # with nvidia-cublas-cu12

`device="auto"` (the default) then runs on the GPU. Modules: sm70 (V100; RTX 20xx / T4 / T1000 run it), sm80 (A100,
A30; RTX 30xx / A-series and L40S / RTX 40xx run it), sm90 (H100 / H200), sm100 (B200; B300 runs it), sm120 (RTX 50xx);
other GPUs fall back to the CPU (`device="auto"`) or raise (`device="cuda"`), or, with `GRACE_RUNTIME_ALLOW_PTX=1`,
run an older module's PTX compiled by the driver (see [Where does it run?](../docs/advanced.md#where-does-it-run-fallbacks-strict-mode-benchmarks)). Needs an NVIDIA driver for CUDA 12
(>= 525; B200 / B300 and RTX 50xx: >= 570), no CUDA toolkit. Linux x86-64 only (not GB200 / GB300 hosts, which are
aarch64).

With TensorFlow on the same GPU (e.g. TPCalculator and GRACERuntimeCalculator in one process) set
`TF_FORCE_GPU_ALLOW_GROWTH=true`, else TensorFlow reserves the whole GPU memory. One CUDA device per process; on a
cluster run one process per GPU (`srun --gpus-per-task=1`). `chunksize="auto"` (the default) fits large structures
to the GPU memory by itself (one pass while it fits, then smaller chunks).

## Source builds and usage

See the [main README](../README.md#nvidia-gpu-cuda) for a source build, e.g. an RTX 3060 (`sm86`) wheel, and
[docs/development.md](../docs/development.md#a-cuda-wheel-for-other-gpu-architectures) for other architectures. The main README also covers
[calculator setup and UQ](../README.md#use), [batch processing](../README.md#many-structures-at-once),
and [memory tuning and multiple GPUs](../docs/advanced.md).

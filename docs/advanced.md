# Performance, memory and diagnostics

[Manual](index.md) · [GPU requirements](backends.md) · [API reference](api.md)

## Device selection

Accepted by `GRACERuntimeCalculator`, `GRACERuntimeModel` (TorchSim), and all `*_batch` functions (`compute_batch`,
`relax_batch`, `eos_batch`, `elastic_batch`, `phonons_batch`):

| `device` | Behavior |
| --- | --- |
| `"auto"` (default) | Usable NVIDIA GPU, then AMD GPU, otherwise CPU |
| `"cpu"` | CPU execution |
| `"cuda"` / `"cuda:N"` | Require NVIDIA GPU execution |
| `"rocm"` / `"rocm:N"` | Require AMD GPU execution |
| `"gpu"` / `"gpu:N"` | Require a usable GPU, NVIDIA first |

`N` is the visible device index. `"hip"` / `"hip:N"` aliases `"rocm"` / `"rocm:N"`.
`GRACE_RUNTIME_DEVICE=cpu`, `cuda:0` or `rocm:0` overrides calls using `"auto"`.

## Options and performance

Defaults choose CPU threads and GPU chunk sizes automatically. The [API reference](api.md) lists full signatures
and defaults; useful tuning options and the functions/classes that accept them are:

| Option | Accepted by | Effect |
| --- | --- | --- |
| `skin` | `GRACERuntimeCalculator`, `relax_batch` (and `eos_batch`, `elastic_batch`) | Neighbor-list Verlet skin in Å (default `0.5`); must be finite and nonnegative. Larger values reduce rebuilds during MD/relaxation but include more candidate pairs. |
| `num_threads` | `GRACERuntimeCalculator`, `GRACERuntimeModel`, all `*_batch` functions | OpenMP CPU threads per call; defaults to `OMP_NUM_THREADS` or physical cores in the process affinity mask. Ignored on GPU. |
| `chunksize` | `GRACERuntimeCalculator`, `GRACERuntimeModel`, all `*_batch` functions | GPU atoms per engine pass: `"auto"` (default) fits buffers to free GPU memory, a positive integer fixes the pass size, `0` forces a single pass. Ignored on CPU. |
| `energy_only` | `GRACERuntimeCalculator`, `GRACERuntimeModel`, `compute_batch`, `eos_batch` | Skip the backward force/virial pass when only energies (or UQ `gamma`) are requested. |
| `max_atoms` | All `*_batch` functions (`compute_batch`, `relax_batch`, `eos_batch`, `elastic_batch`, `phonons_batch`) | Maximum total atoms grouped into one GPU batch call; a single structure larger than `max_atoms` runs alone. |
| `max_nall` | All `*_batch` functions, `GRACERuntimeModel` | Maximum local + periodic-image atoms per GPU pass (default `2097152`); lower it if small unit cells with many periodic images exhaust GPU memory. |
| `spline` | `Model(path, spline=...)` | Radial cubic-spline intervals on CPU and GPU (default `1024`; for PACE models `0` evaluates radial functions directly). |
| `store_max_gb` | `Model(path, store_max_gb=...)` | CPU, GRACE-2L / 3L only: host memory limit in GB for storing per-atom forward tensors; above it the backward pass recomputes them (same results, ~10% slower). Default `None` (half of available host memory, within the memory limit of the job, e.g. Slurm `--mem`). `calc.backend_info["store"]` reports whether tensors were kept or recomputed. |

CPU and GPU inference use fp32 network arithmetic with fp64 geometry and sums.
Expect small differences from fully fp64 evaluations and between repeated parallel runs.

## Several calculators in one process

Use a separate calculator for each changing structure; they can share a loaded `Model`.
GPU weights are shared, while calculators keep their own neighbor lists and outputs.
Buffers stay allocated between calls: `calc.free_gpu_memory()` releases one calculator's buffers;
`grace_runtime.free_gpu_memory()` releases all calculators' buffers and cached GPU models.

Only one GPU device and module can be used per process. For multiple GPUs, run one process per GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python your_script.py
# AMD:
ROCR_VISIBLE_DEVICES=1 python your_script.py
```

The selected GPU is device 0 inside that process. Python-thread calls are safe, but GPU calls run in turn.
For concurrent CPU calculators, divide the cores with `num_threads` to avoid oversubscription.
If sharing a GPU with TensorFlow or JAX, limit their reservation before starting Python with
`TF_FORCE_GPU_ALLOW_GROWTH=true` or `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Backend diagnostics

Run `grace-runtime info` to inspect installed backends. It returns 1 if a visible GPU cannot run natively.
For an actual calculation, inspect `calc.backend` and `calc.backend_info`; diagnostics alone do not validate a model.

- `device="auto"` can fall back to CPU; a one-time `stderr` banner and a `RuntimeWarning` explain an unusable GPU (`GRACE_RUNTIME_QUIET=1` silences the banner).
- `device="cuda"` or `"rocm"` raises when the required backend is unavailable.
- `GRACE_RUNTIME_STRICT=1` rejects CPU and PTX fallbacks. Request `device="cpu"` explicitly for CPU execution.
- `GRACE_RUNTIME_ALLOW_PTX=1` lets an unsupported NVIDIA family try an older module's PTX. It needs a CUDA 12.9
  compatible driver and may compile slowly on first use. Prefer a [native GPU build](backends.md#build-the-nvidia-backend).
- `grace_runtime.cpu_module` reports the loaded CPU extension (`"x86_64_v4"`, `"x86_64_v3"` or `"native"`).

For GPU timings, enable strict mode, check `calc.backend_info["native"]`, and warm up before measuring.

## Environment variables

| Variable | Effect |
| --- | --- |
| `GRACE_CACHE` | Model cache directory for registry downloads (default `~/.cache/grace`) |
| `GRACE_RUNTIME_DEVICE` | Override `device="auto"` (for example `cpu`, `cuda:0`, `rocm:0`) |
| `GRACE_RUNTIME_STRICT=1` | Error instead of falling back to CPU or PTX |
| `GRACE_RUNTIME_ALLOW_PTX=1` | Allow PTX JIT fallback on NVIDIA GPU families without a native module |
| `GRACE_RUNTIME_QUIET=1` | Silence the `stderr` banner when `device="auto"` falls back |
| `GRACE_RUNTIME_CPU` | Force the CPU ISA module (`v3`, `v4` or `native`) |
| `OMP_NUM_THREADS` | Default CPU thread count when `num_threads` is not given |
| `CUDA_VISIBLE_DEVICES` | Select the visible NVIDIA GPU for the process |
| `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES` | Select the visible AMD GPU (or GCD) for the process |

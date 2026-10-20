# Advanced usage

## Options and performance

| Option | Use |
| --- | --- |
| `skin=0.5` | Calculator Verlet neighbor-list skin in Å; larger values reduce rebuilds but include more neighbors. |
| `num_threads=8` | CPU calculator threads, or host neighbor-search threads in `compute_batch`. Defaults to `OMP_NUM_THREADS`, otherwise physical cores in the process affinity mask. Ignored by the CUDA calculator. |
| `neighbor_backend="matscipy"` | Alternative CPU calculator neighbor list; needs the `matscipy` extra. Default is `"native"`; ignored on CUDA. |
| `chunksize="auto"` | GPU engine memory management. A positive integer fixes atoms per engine pass; `0` requests one pass. Smaller chunks use less GPU memory. No effect on CPU. |
| `max_atoms=32768` | Batch-only limit on total atoms grouped into a GPU module call (default 131072). A larger individual structure is processed alone. |
| `max_nall=1048576` | Batch-only limit on atoms plus periodic images per engine pass (default 2097152). Small periodic cells may have many images. |

For the calculator, `energy_only=True` is the default: energy-only requests skip the backward pass, and a later force
request computes it then. If you always need both energy and forces, set `energy_only=False` to compute forces with
the first energy request. For `compute_batch`, the default is `energy_only=False`.

A `Model("/path/to/kokkos.npz")` can be shared between calculators; attach a separate calculator to each independently
changing structure. Standard ASE workflows, such as geometry optimization, work with the calculator:

```python
from ase.optimize import BFGS

atoms.calc = GRACERuntimeCalculator("GRACE-2L-OMAT-medium-ft-E", device="cuda")
BFGS(atoms, trajectory="relax.traj").run(fmax=0.05)
```

A CUDA calculator keeps its GPU buffers (sized for the largest structure so far) until deleted, also after an
interrupted call; `calc.free_gpu_memory()` returns them now, `grace_runtime.free_gpu_memory()` those of all calculators
and the cached models.

`GRACE_RUNTIME_DEVICE=cpu` or `cuda:0` overrides the device choice for calls using `"auto"`. CPU and GPU kernels use one precision scheme: bond vectors and lengths in fp64, the network
and the radial tables in fp32, energies, forces and stresses summed in fp64 (the CPU also evaluates the energy readout in
fp64). Results differ from 64-bit evaluations at the fp32 level (about 1e-6 eV/atom, a few 1e-6 eV/Å).

## Several calculators in one process

Any number of calculators, of the same or of different models, on the CPU and on one GPU, can live in one process:

- **Kokkos and the GPU.** The first GPU calculator starts Kokkos on its device; all later ones use it. Kokkos runs on
  one CUDA device per process, so a calculator on another GPU raises `ValueError: one CUDA device per process`. For
  several GPUs, use one process per GPU: select the GPU before starting Python (`CUDA_VISIBLE_DEVICES=1 python
  your_script.py`) and use `device="cuda:0"` inside.
- **GPU memory.** The weights and tables of a model file are loaded once and shared by all its calculators. Each model
  has its own GPU engine, whose work buffers grow to the largest structure seen and stay; each calculator also has its
  own neighbour list and outputs. With `chunksize="auto"` an engine sizes its chunk from the GPU memory free at its first
  call, so a model started later may get smaller chunks (slower, never an error: on an allocation failure the chunk is
  halved). Set `chunksize` to a number for predictable sizes; `calc.free_gpu_memory()` returns a calculator's buffers,
  `grace_runtime.free_gpu_memory()` those of all calculators and the cached models.
- **Other GPU frameworks in the process** reserve most of the GPU memory by default: set `TF_FORCE_GPU_ALLOW_GROWTH=true`
  (TensorFlow, e.g. the TensorFlow GRACE calculator) or `XLA_PYTHON_CLIENT_PREALLOCATE=false` (JAX) before starting
  Python.
- **Threads.** Calls release the GIL, so calculators may be used from several Python threads. A calculator runs one
  call at a time; GPU calls of all calculators and batches take turns on the GPU (safe, but no parallel speed-up).
  Several CPU calculators running at the same time each use `num_threads` (default: all physical cores), so
  split the cores between them, e.g. `num_threads=8` each for two calculators on 16 cores.

`python/tests/test_cuda_several_calculators.py` checks these cases.

## Where does it run? Fallbacks, strict mode, benchmarks

`grace-runtime info` prints what `device="auto"` / `"cuda"` will use: the CPU module, the GPU, its driver, the
installed GPU modules and whether the GPU runs natively (exit status 1 when it would not). In Python,
`calc.backend` (e.g. `"cuda:0 (sm80)"`) and `calc.backend_info` (`{"device", "arch", "cc", "native", "fallback"}`,
also on the TorchSim model) tell the same per calculator.

Fallbacks are announced by a banner on stderr (once per process and message; not affected by warning filters;
`GRACE_RUNTIME_QUIET=1` silences it) and a `RuntimeWarning`:

- **CPU fallback.** `device="auto"` runs on the CPU when a GPU is there but cannot be used (no module for it, the
  model is outside the fused kernels, the GPU fails to start). `calc.backend_info["fallback"]` holds the reason.
- **PTX fallback, opt-in.** The wheels have modules for the architecture families sm70, sm80, sm90, sm100 and sm120;
  a GPU of another family (a future one) has no native module. With `GRACE_RUNTIME_ALLOW_PTX=1` it runs the PTX of
  the newest older module instead: the CUDA driver compiles it at the first call (can take minutes; cached in
  `~/.nv/ComputeCache`, `CUDA_CACHE_PATH`; compiled again where the cache is not kept, e.g. in fresh containers). The
  results are the same, the kernels are tuned for the older architecture (a few % slower in our test). It needs a
  driver for CUDA >= 12.9. Without the variable such a GPU runs on the CPU (`"auto"`) or raises (`"cuda"`); the
  message names both ways out, PTX and building a native module (`GRACE_KK_ARCHS=sm<XY>`, see [Build from source](../README.md#build-from-source)).
- **Strict mode.** `GRACE_RUNTIME_STRICT=1` makes every fallback an error: a native GPU module or nothing.

Benchmark checklist: set `GRACE_RUNTIME_STRICT=1`, check `grace-runtime info` (or `calc.backend_info["native"]`), and
time after a warm-up call (the first call sets up the model and, on the PTX path, compiles the GPU code).

## Batches of many structures: streaming

For large datasets, stream the input rather than loading every structure into memory. Reuse a `Model` to avoid
loading the model again for each call:

```python
from itertools import islice
from ase.io import iread
from grace_runtime import Model, compute_batch

model = Model("/path/to/kokkos.npz")
frames = iter(iread("structures.extxyz", index=":"))

while batch := list(islice(frames, 256)):
    results = compute_batch(model, batch, device="cuda", energy_only=True)
    for result in results:
        print(result["energy"])  # save or process each batch here
```

`compute_batch` materializes its input and returns all results for that call; passing a generator directly does not
make the call stream. The explicit loop above bounds the number of structures held in host memory.

## TorchSim model: outputs and memory

Outputs: `"energy"` [n_systems], `"forces"` [n_atoms, 3], `"stress"` [n_systems, 3, 3] (eV/Å³, zero for cells with
zero vectors); `energies=True` adds per-atom `"energies"`, `uq=True` adds `"gamma"`, `"atomic_sigma"`,
`"gmm_cluster"` and `"total_sigma"`. Each step copies positions and cells to the host and builds the neighbour lists
there; the GPU buffers stay allocated between steps (`grace_runtime.free_gpu_memory()` returns them). Forces come from
the kernels, not from autograd (states with `positions.requires_grad` are refused). `max_memory_scaler(state)` sizes
TorchSim's autobatchers from one measured call (TorchSim's own estimate waits for "CUDA out of memory", which this
model does not raise: it splits large batches into passes itself).

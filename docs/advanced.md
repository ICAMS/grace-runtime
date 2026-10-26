# Advanced usage

[Documentation map](../README.md#guides) · [Usage](usage.md) · [Installation](installation.md)

## Options and performance

| Option | Use |
| --- | --- |
| `skin=0.5` | Calculator Verlet neighbor-list skin in Å, finite and >= 0; larger values reduce rebuilds but include more neighbors. |
| `num_threads=8` | CPU calculator threads (also of `compute_batch` and `relax_batch` on the CPU). Defaults to `OMP_NUM_THREADS`, otherwise physical cores in the process affinity mask. Ignored on the GPU. |
| `chunksize="auto"` | GPU engine memory management. A positive integer fixes atoms per engine pass; `0` requests one pass. Smaller chunks use less GPU memory. A chunk whose largest work buffer would exceed 2^31 values is an error (`"auto"` stays below). No effect on CPU. |
| `max_atoms=32768` | Batch-only limit on total atoms grouped into a GPU module call (default 131072). A larger individual structure is processed alone. |
| `max_nall=1048576` | Batch-only limit on atoms plus periodic images per engine pass (default 2097152). Small periodic cells may have many images. |

For the calculator, `energy_only=True` is the default: energy-only requests skip the backward pass, and a later force
request computes it then. If you always need both energy and forces, set `energy_only=False` to compute forces with
the first energy request. For `compute_batch`, the default is `energy_only=False`.

A loaded `Model` (constructed from a registry name or a `kokkos.npz` path) can be shared between calculators and
the batch functions; attach a separate calculator to each independently changing structure. Standard ASE workflows, such as geometry
optimization, work with the calculator:

```python
from ase.build import bulk
from ase.optimize import BFGS
from grace_runtime import GRACERuntimeCalculator

atoms = bulk("W", "bcc", a=3.165, cubic=True)
atoms.positions[0, 0] += 0.05
atoms.calc = GRACERuntimeCalculator("GRACE-2L-OMAT-medium-ft-E", device="cpu", num_threads=1)
BFGS(atoms, trajectory="relax.traj").run(fmax=0.05)
```

A CUDA calculator keeps its GPU buffers (sized for the largest structure so far) until deleted, also after an
interrupted call; `calc.free_gpu_memory()` returns them now, `grace_runtime.free_gpu_memory()` those of all calculators
and the cached models.

`GRACE_RUNTIME_DEVICE=cpu`, `cuda:0` or `rocm:0` overrides the device choice for calls using `"auto"`.
CPU and GPU kernels use one precision scheme: bond vectors and lengths in fp64, the network
and the radial tables in fp32, energies, forces and stresses summed in fp64, the per-atom energy readout
(`(MLP + linear) * scale + shift`) in fp64. Results differ from 64-bit evaluations at the fp32 level: total energies by about 1e-6 eV/atom
(a few 1e-6 eV/atom in cells of a few atoms), forces by a few 1e-6 eV/Å, per-atom energies by a few 1e-6 eV on the CPU
and up to about 1e-5 eV on the GPU.

## Several calculators in one process

Any number of calculators, of the same or of different models, on the CPU and on one GPU, can live in one process:

- **Kokkos and the GPU.** The first GPU calculator starts Kokkos on its device; all later ones use it. The current runtime keeps
  one GPU module and device per process, so a calculator on another GPU raises `ValueError: one GPU per process`
  (for either vendor). For several GPUs, use one process per GPU: select the GPU before starting Python
  (`CUDA_VISIBLE_DEVICES=1 python your_script.py`) and use `device="cuda:0"` inside. For AMD, use
  `ROCR_VISIBLE_DEVICES=1` and `device="rocm:0"`.
- **GPU memory.** The weights and tables of a model file are loaded once and shared by all its calculators. Each model
  has its own GPU engine, whose work buffers grow to the largest structure seen and stay; each calculator also has its
  own neighbour list and outputs. With `chunksize="auto"` an engine sizes its chunk from the GPU memory free at its first
  call, so a model started later may get smaller chunks. Auto chunking retries allocation with smaller chunks,
  but cannot guarantee that every structure fits. Set `chunksize` to a number for predictable sizes;
  `calc.free_gpu_memory()` returns a calculator's buffers,
  `grace_runtime.free_gpu_memory()` those of all calculators and the cached models.
- **Other GPU frameworks in the process** reserve most of the GPU memory by default: set `TF_FORCE_GPU_ALLOW_GROWTH=true`
  (TensorFlow, e.g. the TensorFlow GRACE calculator) or `XLA_PYTHON_CLIENT_PREALLOCATE=false` (JAX) before starting
  Python.
- **Threads.** Calls release the GIL, so calculators may be used from several Python threads. A calculator runs one
  call at a time; GPU calls of all calculators and batches take turns on the GPU (safe, but no parallel speed-up).
  Several CPU calculators running at the same time each use `num_threads` (default: all physical cores), so
  split the cores between them, e.g. `num_threads=8` each for two calculators on 16 cores. With more than one CPU
  thread, and on the GPU, sums are added in a varying order, so repeated runs can differ in the last bits (fp32
  rounding, about 1e-7 eV/A in the forces).

`python/tests/test_gpu_several_calculators.py` checks these cases.

## Where does it run? Fallbacks, strict mode, benchmarks

`grace-runtime info` diagnoses the CPU module and the first visible GPU, its driver/runtime and installed modules.
It returns 0 on a CPU-only machine, or with a native GPU module; 1 means a visible GPU cannot run natively.
It does not initialize a model and does not prove a particular model or requested device will run. In Python,
`calc.backend` (e.g. `"cuda:0 (sm80)"`) and `calc.backend_info` (`{"device", "platform", "arch", "cc", "native", "fallback"}`,
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
  message names both ways out, PTX and building a native module (`GRACE_KK_ARCHS=sm<XY>`, see [GPU builds](installation.md#gpu-backends)).
- **Strict mode.** `GRACE_RUNTIME_STRICT=1` makes every fallback an error: a native GPU module or nothing. That
  includes `device="auto"` on a machine without a usable GPU; `device="cpu"` (or `GRACE_RUNTIME_DEVICE=cpu`) is the
  explicit way to run on the CPU.

For native GPU benchmarks, set `GRACE_RUNTIME_STRICT=1` (with `device="auto"` it then requires a usable GPU
and its native module). Check `calc.backend_info["native"]` and time after a warm-up call (the first call sets up the model;
on the PTX path it also compiles GPU code).

## Batches of many structures: streaming

For large datasets, stream the input rather than loading every structure into memory. `compute_batch_iter` takes
any iterable, reads one GPU call (`max_atoms` atoms) at a time and yields `(index, result)` in input order; reuse a
`Model` to avoid loading the model again:

```python
from ase.io import iread
from grace_runtime import Model, compute_batch_iter

model = Model("/path/to/kokkos.npz")
for i, result in compute_batch_iter(model, iread("structures.extxyz", index=":"), energy_only=True):
    print(i, result["energy"])  # save or process each result here
```

`compute_batch` materializes its input and returns all results at once. On the GPU, the species set of a stream
grows with the elements read so far; a new element sets up a new GPU batch (once per new element).
`relax_batch_iter` streams relaxations the same way ([batched relaxation](relax.md)).

## TorchSim model: outputs and memory

Requires Python >= 3.12 and the `torchsim` extra ([installation](installation.md#optional-dependencies)). This small
single-point example creates its own structures; it requires the CUDA backend and a usable NVIDIA GPU:

```python
import numpy as np  # load numpy first if torch otherwise finds an older system libstdc++
import torch
import torch_sim as ts
from ase.build import bulk
from grace_runtime.torchsim import GRACERuntimeModel

model = GRACERuntimeModel("GRACE-2L-OMAT-medium-ft-E", device="cuda")
structures = [bulk("W", "bcc", a=a, cubic=True) for a in (3.1, 3.2)]
state = ts.io.atoms_to_state(structures, torch.device("cuda"), torch.float64)
result = model(state)
print(result["energy"])
```

For MD, optimization and larger batches, see the [TorchSim examples](../python/examples/torchsim/). A TorchSim
model is a `ModelInterface`; use TorchSim's integrators, optimizers and autobatchers with it.

Outputs: `"energy"` [n_systems], `"forces"` [n_atoms, 3], `"stress"` [n_systems, 3, 3] (eV/Å³, zero for cells with
zero vectors); `energies=True` adds per-atom `"energies"`, `uq=True` adds `"gamma"`, `"atomic_sigma"`,
`"gmm_cluster"` and `"total_sigma"`. States on the model's GPU stay there; other states (and cells with zero vectors)
are copied to the host and back each step. With a GPU model the neighbour lists are built on the GPU in both cases;
the GPU buffers stay allocated between steps (`grace_runtime.free_gpu_memory()` returns them). Forces come from the
kernels, not from autograd (states with `positions.requires_grad` are refused). `max_memory_scaler(state)` sizes
TorchSim's autobatchers from one measured call (TorchSim's own estimate waits for "CUDA out of memory", which this
model does not raise: it splits large batches into passes itself).

# TorchSim adapter

[Installation](installation.md#optional-dependencies) · [API reference](api.md#torchsim-adapter)

Install the `torchsim` extra with Python >= 3.12 (on macOS, PyTorch from conda-forge instead: see
[macOS](installation.md#macos)). This complete example uses the default device selection:

```python
import numpy  # load before torch
import torch
import torch_sim as ts
from ase.build import bulk
from grace_runtime.torchsim import GRACERuntimeModel

model = GRACERuntimeModel("GRACE-2L-OMAT-medium-ft-E")
structures = [bulk("W", "bcc", a=a, cubic=True) for a in (3.1, 3.2)]
state = ts.io.atoms_to_state(structures, model.device, torch.float64)
result = model(state)
print(result["energy"])
```

The adapter implements TorchSim's `ModelInterface`, so its integrators, optimizers and autobatchers can use
GRACE kernels. GPU execution needs the matching [backend](backends.md); torch uses `"cuda"` for AMD GPUs too.
Use a ROCm build of torch for AMD.

Recent torch releases on PyPI are CUDA 13 builds and need a driver for CUDA 13. With `grace-runtime-cuda12` on a
CUDA 12 driver, install torch from the PyTorch CUDA 12 index, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu129`; otherwise torch runs on the CPU only and the
adapter does not use the GPU. Check with `python -c "import torch; print(torch.cuda.is_available())"`.

Outputs are energy per system, forces per atom and full 3×3 stress per system. See the
[API reference](api.md#results) for units, shapes and optional outputs.

States on the model's GPU stay there. Other states and cells with zero vectors use host copies.
Forces come from runtime kernels; autograd and `positions.requires_grad` are unsupported.
Use `model.max_memory_scaler(state)` to size TorchSim autobatches; release retained GPU buffers with
`grace_runtime.free_gpu_memory()`.

See the [TorchSim examples](https://github.com/ICAMS/grace-runtime/tree/master/python/examples/torchsim/)
for MD, optimization and autobatching.

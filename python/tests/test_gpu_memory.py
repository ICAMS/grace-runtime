# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Device and host memory the GPU backend keeps between calls (GRACE-3L: ~64 KB per atom + image beyond the chunk):
compute_batch returns its per-atom buffers when it returns (the model's weights and tables stay cached);
grace_runtime.free_gpu_memory() drops the cached models and batches; the 3L engine allocates the host copies of its
per-image rows only for the host comm path (not for the Python module). Each case in its own process."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from gpu_tools import NO_GPU, gpu_ok

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M3 = CACHE / "GRACE-3L-OMAT-large" / "kokkos.npz"
M12 = {"1l": CACHE / "GRACE-1L-OMAT-large-base" / "kokkos.npz", "2l": CACHE / "GRACE-2L-SMAX-OMAT-medium" / "kokkos.npz"}



pytestmark = pytest.mark.skipif(not gpu_ok() or not M3.exists(), reason=f"{NO_GPU} / 3L model")

PRELUDE = """
import gc, numpy as np
from ase.build import bulk
import grace_runtime
from grace_runtime import GRACERuntimeCalculator, compute_batch, _gpu
GPU = __import__("os").environ["GRACE_TEST_GPU"]
kk = _gpu.ensure_init(GPU)
GB = 2**30
def free():
    gc.collect()
    return kk.memory_info()[0] / GB
def rss():
    return int(open("/proc/self/statm").read().split()[1]) * 4096 / GB
cells = []
base = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
for k in range(300):
    a = base.copy()
    a.rattle(0.03, seed=k)
    cells.append(a)
"""

CASES = {
    # after compute_batch: only the weights / tables of the model stay (well under 1.5 GB for GRACE-3L-OMAT-large);
    # free_gpu_memory() returns them too
    "batch_returns_buffers": """
f0 = free()
r = compute_batch("{M3}", cells, device=GPU, energy_only=True)
f1 = free()
print("FREE", f0, f1)
assert f0 - f1 < 1.5, ("kept after compute_batch (GB)", f0 - f1)
grace_runtime.free_gpu_memory()
f2 = free()
print("FREE", f2)
assert f0 - f2 < 0.3, ("kept after free_gpu_memory (GB)", f0 - f2)
""",
    # calc.free_gpu_memory() / grace_runtime.free_gpu_memory() with the calculator alive: its System's and engine's
    # per-atom buffers go back (weights and tables stay); it computes again with the same result. "Kept": the live
    # allocations, by label, that are larger than before the 16k-atom call (the free memory of the card also moves
    # with the CUDA libraries' own pools)
    "calculator_frees_buffers": """
kk._debug_track_alloc(True)
def live():
    d = {{}}
    for name, b in kk._debug_live_alloc():
        d[name] = d.get(name, 0) + b
    return d
def kept(a, b):
    return {{k: (v - a.get(k, 0)) / 2**20 for k, v in b.items() if v - a.get(k, 0) > 2**16}}    # MB, > 64 KB
c = GRACERuntimeCalculator("{M3}", device=GPU)
w = base.copy()
w.calc = c
w.get_forces()    # model, engine and tables loaded
f0, l0 = free(), live()
at = bulk("W", "bcc", a=3.165, cubic=True).repeat(20)
at.rattle(0.03, seed=1)
at.calc = c
fr1, e1 = at.get_forces(), at.get_potential_energy()
f1 = free()
c.free_gpu_memory()
f2, l2 = free(), live()
c.reset()
fr2, e2 = at.get_forces(), at.get_potential_energy()
grace_runtime.free_gpu_memory()
f3, l3 = free(), live()
print("FREE", f0, f1, f2, f3)
assert f0 - f1 > 0.5, ("16k atoms took (GB)", f0 - f1)
assert not kept(l0, l2) and not kept(l0, l3), ("kept after free_gpu_memory (MB)", kept(l0, l2), kept(l0, l3))
assert abs(e2 - e1) < 1e-6 * len(at) and np.abs(fr2 - fr1).max() < 1e-5, (e2 - e1, np.abs(fr2 - fr1).max())
""",
    # a 16k-atom 3L System: host memory grows by the neighbour list and outputs, not by ~38 KB per atom + image
    "no_host_rows_3l": """
c = GRACERuntimeCalculator("{M3}", device=GPU)
small = bulk("W", "bcc", a=3.165, cubic=True).repeat(2)
small.calc = c
small.get_forces()    # model, engine and tables loaded
r0 = rss()
at = bulk("W", "bcc", a=3.165, cubic=True).repeat(20)
at.calc = c
at.get_forces()
r1 = rss()
print("RSS", r0, r1)
assert r1 - r0 < 0.4, ("host GB", r1 - r0)
""",
    # the owner <-> image exchange buffer of GRACE-3L (fp64 rows of up to 3200 values) is bounded by its block of 8192
    # images, not by the number of images (16000 atoms: ~11000 images)
    "exchange_buffer_bounded": """
kk._debug_track_alloc(True)
at = bulk("W", "bcc", a=3.165, cubic=True).repeat(20)
at.calc = GRACERuntimeCalculator("{M3}", device=GPU)
at.get_forces()
xb = sum(b for name, b in kk._debug_live_alloc() if name == "csr:xbuf3")
print("XBUF3 MB", xb / 2**20)
assert 0 < xb <= 8192 * 3200 * 8, ("csr:xbuf3 bytes", xb)
""",
}


@pytest.mark.parametrize("case", sorted(CASES) + ["calculator_frees_buffers-1l", "calculator_frees_buffers-2l"])
def test_memory(case):
    case, _, layer = case.partition("-")
    m = M12[layer] if layer else M3
    if not m.exists():
        pytest.skip(f"{m.parent.name} not in the test cache")
    code = PRELUDE + textwrap.dedent(CASES[case]).format(M3=m)
    env = {k: v for k, v in os.environ.items() if k != "GRACE_RUNTIME_DEVICE"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]

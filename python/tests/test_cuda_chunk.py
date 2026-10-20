# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""chunksize (atoms per engine pass) of the GPU path: "auto" (the default) lets the engine choose it from the free GPU
memory (one pass while it fits) and, on a device allocation failure, halves it down to 1024 (the failed engine and its
memory dropped); a structure that does not fit even then
is a clear RuntimeError, not "model not supported". Small-card cases run only on such cards (dandolo RTX 4070 SUPER
12 GB, belgica RTX 2070 8 GB)."""
import ctypes
import os
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from grace_runtime import GRACERuntimeCalculator, _cuda

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M2M = CACHE / "GRACE-2L-SMAX-OMAT-medium" / "kokkos.npz"
M2L = CACHE / "GRACE-2L-OMAT-large-base" / "kokkos.npz"
HEA = "Al Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Hf Ta W Re Os".split()


def _cuda_ok():
    try:
        _cuda.module_for(0)
        return True
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _cuda_ok() or not (M2M.exists() and M2L.exists()), reason="no GPU module / models")


def total_gb():
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuInit(0)
    dev, mem = ctypes.c_int(), ctypes.c_size_t()
    cu.cuDeviceGet(ctypes.byref(dev), 0)
    cu.cuDeviceTotalMem_v2(ctypes.byref(mem), dev)
    return mem.value / 2**30


def hea128k():
    a = bulk("W", "bcc", a=3.165, cubic=True).repeat((40, 40, 40))
    a.set_chemical_symbols(np.random.default_rng(20261008).permutation(np.resize(HEA, len(a))))
    return a


def test_chunksize_parameter():
    w = bulk("W", "bcc", a=3.165, cubic=True).repeat(4)
    fixed = GRACERuntimeCalculator(str(M2M), device="cuda", chunksize=2048)
    auto = GRACERuntimeCalculator(str(M2M), device="cuda")
    e = []
    for c in (fixed, auto):
        a = w.copy()
        a.calc = c
        e.append(a.get_potential_energy())
    assert fixed.chunksize == 2048 and auto.chunksize == 0    # auto: one pass while it fits
    assert abs(e[0] - e[1]) / len(w) <= 2e-6


SNIPPET = """
import sys, numpy as np
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator
HEA = "Al Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Hf Ta W Re Os".split()
a = bulk("W", "bcc", a=3.165, cubic=True).repeat((40, 40, 40))
a.set_chemical_symbols(np.random.default_rng(20261008).permutation(np.resize(HEA, len(a))))
a.calc = GRACERuntimeCalculator(sys.argv[1], device="cuda")
f = a.get_forces()
print("OK", np.isfinite(f).all(), a.calc.chunksize)
"""


@pytest.mark.skipif(not 10 < (total_gb() if _cuda_ok() else 0) <= 16, reason="for 10-16 GB cards")
@pytest.mark.parametrize("model", [M2M, M2L], ids=["2L-medium", "2L-large"])
def test_auto_chunk_fits_a_large_system_on_a_small_card(model):
    """one model per process (a Model keeps its engine and buffers for the process)"""
    import subprocess, sys
    import grace_runtime
    grace_runtime.free_gpu_memory()    # the GPU memory this pytest process holds from earlier tests is not the child's
    r = subprocess.run([sys.executable, "-c", SNIPPET, str(model)], capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-1500:]
    ok, finite, chunk = r.stdout.split()[-3:]
    assert ok == "OK" and finite == "True" and 1024 <= int(chunk) <= 65536


@pytest.mark.skipif(not 0 < (total_gb() if _cuda_ok() else 0) <= 10, reason="for cards up to 10 GB")
def test_does_not_fit_is_a_clear_error():
    at = hea128k()
    at.calc = GRACERuntimeCalculator(str(M2L), device="cuda")
    with pytest.raises(RuntimeError, match="does not fit"):
        at.get_forces()


def test_3l_chunks_match_one_pass():
    """GRACE-3L in chunks of 32 atoms (the layered exchange per chunk, the forward store) == one pass"""
    m3 = CACHE / "GRACE-3L-OMAT-large" / "kokkos.npz"
    if not m3.exists():
        pytest.skip("GRACE-3L-OMAT-large not in the test cache")
    w = bulk("W", "bcc", a=3.165, cubic=True).repeat(4)
    w.rattle(0.05, seed=2)
    r = []
    for c in (GRACERuntimeCalculator(str(m3), device="cuda", chunksize=32), GRACERuntimeCalculator(str(m3), device="cuda")):
        a = w.copy()
        a.calc = c
        r.append((a.get_potential_energy(), a.get_forces()))
    assert abs(r[0][0] - r[1][0]) / len(w) <= 2e-6
    assert np.abs(r[0][1] - r[1][1]).max() <= 2e-5


AUTO_3L = """
import sys, numpy as np
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator, compute_batch
from grace_runtime.calculator import _KK_MODELS
m3 = sys.argv[1]
base = bulk("W", "bcc", a=3.165, cubic=True)
if sys.argv[2] == "system":
    a = base.repeat(20)    # 16000 atoms: more than one chunk of GRACE-3L on a 24 GB card
    a.rattle(0.03, seed=1)
    a.calc = GRACERuntimeCalculator(m3, device="cuda")
    a.get_forces()
    a.rattle(0.01, seed=2)
    a.get_forces()
else:
    cells = []
    for k in range(int(sys.argv[2][5:])):
        c = base.repeat(3)
        c.rattle(0.03, seed=k)
        cells.append(c)
    compute_batch(m3, cells, device="cuda", energy_only=True)
    compute_batch(m3, cells, device="cuda", energy_only=True)
print("SETUPS", list(_KK_MODELS.values())[0].engine_setups)
"""


@pytest.mark.parametrize("case", ["system", "batch300", "batch1000"])
def test_3l_auto_chunk_from_free_memory(case):
    """chunksize="auto": the engine picks the chunk from the free GPU memory, so a GRACE-3L structure (16k atoms) or a
    batch (300 cells) sets the engine up once - no out-of-memory retries with smaller chunks (several s each); a batch
    whose atoms + images per pass do not fit (1000 cells: ~64 KB per image row) retries with fewer per pass on the same
    engine"""
    import subprocess, sys
    m3 = CACHE / "GRACE-3L-OMAT-large" / "kokkos.npz"
    if not m3.exists():
        pytest.skip("GRACE-3L-OMAT-large not in the test cache")
    r = subprocess.run([sys.executable, "-c", AUTO_3L, str(m3), case], capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.split()[-1] == "1", r.stdout[-500:]

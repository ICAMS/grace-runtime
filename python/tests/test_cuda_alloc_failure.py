# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""A device allocation failure part-way through a growth (injected with the module's _debug_fail_alloc: the nth
allocation whose label matches throws "failed to allocate") must not leave a half-grown engine or System behind: the
same calculator, and other calculators sharing the engine, give the result of a fresh calculator afterwards. Each case
runs in its own process (a broken engine may lose the CUDA context)."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from grace_runtime import _cuda

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M2 = CACHE / "GRACE-2L-SMAX-OMAT-medium" / "kokkos.npz"
M1 = CACHE / "GRACE-1L-OMAT-large-base" / "kokkos.npz"
M3 = CACHE / "GRACE-3L-OMAT-large" / "kokkos.npz"


def _cuda_ok():
    try:
        _cuda.module_for(0)
        return True
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _cuda_ok() or not (M1.exists() and M2.exists() and M3.exists()), reason="no GPU module / models")

PRELUDE = """
import numpy as np
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator, _cuda
kk = _cuda.ensure_init("cuda")
small = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
big = bulk("W", "bcc", a=3.165, cubic=True).repeat((10, 10, 10))
big.rattle(0.02, seed=1)
def ef(calc, at):
    a = at.copy()
    a.calc = calc
    return a.get_potential_energy(), a.get_forces()
def check(calc, at, ref_calc):
    e, f = ef(calc, at)
    e0, f0 = ef(ref_calc, at)
    de, df = abs(e - e0) / len(at), np.abs(f - f0).max()
    assert de <= 1e-6 and df <= 1e-5, (de, df)
def inject_and_fail(calc, at, label, nth):
    kk._debug_fail_alloc(label, nth)
    try:
        ef(calc, at)
        raise SystemExit("no error from the injected allocation failure")
    except RuntimeError as e:
        assert "failed to allocate" in str(e), e
        print("injected:", str(e)[:300])
    finally:
        kk._debug_fail_alloc("", 0)
"""

CASES = {
    # fixed chunksize: the failed engine stays with the calculator
    "engine_2l_fixed": """
m = "{M2}"
c = GRACERuntimeCalculator(m, device="cuda", chunksize=4096)
ef(c, small)
inject_and_fail(c, big, "g2l:A1", 1)
check(c, big, GRACERuntimeCalculator(m, device="cuda", chunksize=2048))
""",
    "engine_1l_fixed": """
m = "{M1}"
c = GRACERuntimeCalculator(m, device="cuda", chunksize=4096)
ef(c, small)
inject_and_fail(c, big, "g1l:A1", 1)
check(c, big, GRACERuntimeCalculator(m, device="cuda", chunksize=2048))
""",
    "engine_3l_fixed": """
m = "{M3}"
c = GRACERuntimeCalculator(m, device="cuda", chunksize=4096)
ef(c, small)
inject_and_fail(c, big, "g3l:A1", 1)
check(c, big, GRACERuntimeCalculator(m, device="cuda", chunksize=2048))
""",
    # a second calculator holding the same shared engine
    "engine_shared": """
m = "{M2}"
a = GRACERuntimeCalculator(m, device="cuda", chunksize=4096)
b = GRACERuntimeCalculator(m, device="cuda", chunksize=4096)
ef(a, small); ef(b, small)
inject_and_fail(a, big, "g2l:A1", 1)
check(b, big, GRACERuntimeCalculator(m, device="cuda", chunksize=2048))
""",
    # auto chunksize below 2048 atoms: nothing smaller to try, the error goes out; the next call must work
    "engine_auto": """
m = "{M2}"
c = GRACERuntimeCalculator(m, device="cuda")
ef(c, small)
inject_and_fail(c, big, "g2l:A1", 1)
check(c, big, GRACERuntimeCalculator(m, device="cuda", chunksize=2048))
""",
    # device="auto": the GPU engine cannot be set up (GPU memory held by another process): the CPU with a warning, not
    # remembered (a later calculator takes the GPU)
    "auto_setup_oom": """
import warnings
m = "{M2}"
kk._debug_fail_alloc("g2l:", 1)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    c = GRACERuntimeCalculator(m)
kk._debug_fail_alloc("", 0)
assert c.backend == "cpu", c.backend
assert any("GPU memory" in str(x.message) for x in w), [str(x.message) for x in w]
assert GRACERuntimeCalculator(m).backend.startswith("cuda"), "the failure was remembered"
""",
    # device="cuda": the same is an error that says what happened
    "cuda_setup_oom": """
m = "{M2}"
kk._debug_fail_alloc("g2l:", 1)
try:
    ef(GRACERuntimeCalculator(m, device="cuda"), small)
    raise SystemExit("no error")
except RuntimeError as e:
    assert "engine" in str(e) and "GPU memory" in str(e) and "0 atoms" not in str(e), e
finally:
    kk._debug_fail_alloc("", 0)
""",
    # the System's own buffers (positions, list)
    "system_buffers": """
m = "{M2}"
c = GRACERuntimeCalculator(m, device="cuda", chunksize=4096)
ef(c, small)
inject_and_fail(c, big, "kk:pos_build", 1)
check(c, big, GRACERuntimeCalculator(m, device="cuda", chunksize=2048))
""",
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_usable_after_allocation_failure(case):
    code = PRELUDE + textwrap.dedent(CASES[case]).format(M1=M1, M2=M2, M3=M3)
    env = {k: v for k, v in os.environ.items() if k != "GRACE_RUNTIME_DEVICE"}    # conftest pins the CPU
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]

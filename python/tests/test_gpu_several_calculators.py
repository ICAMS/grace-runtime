# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Several GPU calculators in one process (README "Several calculators in one process"): one Kokkos for all of them,
one kk.Model (weights, tables) per model file shared by its calculators, calculators of different models interleaved
or called from two threads give what each gives alone, a second GPU is refused. GPU sums use atomics, so
results are compared at the run-to-run level (energy 1e-6 eV/atom, forces 1e-5 eV/A). Each case in its own process
(fresh Kokkos)."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from gpu_tools import NO_GPU, gpu_count, gpu_ok

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M1 = CACHE / "GRACE-1L-OMAT-large-base" / "kokkos.npz"
M2 = CACHE / "GRACE-2L-SMAX-OMAT-medium" / "kokkos.npz"



pytestmark = pytest.mark.skipif(not gpu_ok() or not (M1.exists() and M2.exists()), reason=f"{NO_GPU} / models")

PRELUDE = """
import threading, numpy as np
from ase.build import bulk
from grace_runtime import GRACERuntimeCalculator
GPU = __import__("os").environ["GRACE_TEST_GPU"]
PLATFORM = GPU.split(":")[0]

def structures():
    w = bulk("W", "bcc", a=3.165, cubic=True).repeat(4)
    w.rattle(0.05, seed=1)
    wmo = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    wmo.symbols[::2] = "Mo"
    wmo.rattle(0.05, seed=2)
    return [w, wmo]

def run(calc, at):
    at = at.copy()
    at.calc = calc
    return at.get_potential_energy(), at.get_forces().copy()

def same(a, b, n, what):
    de, df = abs(a[0] - b[0]), np.abs(a[1] - b[1]).max()
    assert de < 1e-6 * n and df < 1e-5, (what, de, df)
"""

CASES = {
    # one kk.Model per model file: the second calculator of a model shares the first one's weights and engine
    "same_model_shares_weights": """
c1 = GRACERuntimeCalculator("{M2}", device=GPU)
c2 = GRACERuntimeCalculator("{M2}", device=GPU)
c3 = GRACERuntimeCalculator("{M1}", device=GPU)
assert c1._kk is c2._kk is c3._kk    # one GPU module, one Kokkos
assert c1._kk_model is c2._kk_model
assert c3._kk_model is not c1._kk_model
""",
    # 1L and 2L calculators (and a CPU one) used in turn give what each gives alone
    "different_models_interleaved": """
s = structures()
alone = {{}}
for name, path in (("1l", "{M1}"), ("2l", "{M2}")):
    c = GRACERuntimeCalculator(path, device=GPU)
    alone[name] = [run(c, at) for at in s]
c1, c2 = GRACERuntimeCalculator("{M1}", device=GPU), GRACERuntimeCalculator("{M2}", device=GPU)
cpu = GRACERuntimeCalculator("{M1}", device="cpu", num_threads=4)
for rep in range(3):
    for k, at in enumerate(s):
        same(run(c1, at), alone["1l"][k], len(at), ("1l", rep, k))
        same(run(c2, at), alone["2l"][k], len(at), ("2l", rep, k))
        e_cpu = run(cpu, at)[0]
        assert abs(e_cpu - alone["1l"][k][0]) < 1e-5 * len(at), ("cpu vs gpu 1l", e_cpu, alone["1l"][k][0])
""",
    # two Python threads, each with its own calculator (same model, then different models), at the same time
    "two_threads": """
s = structures()
for paths in (("{M2}", "{M2}"), ("{M1}", "{M2}")):
    calcs = [GRACERuntimeCalculator(p, device=GPU) for p in paths]
    ref = [[run(c, at) for at in s] for c in calcs]
    errors = []
    def work(i):
        try:
            for rep in range(5):
                for k, at in enumerate(s):
                    same(run(calcs[i], at), ref[i][k], len(at), (paths[i], rep, k))
        except BaseException as e:
            errors.append(e)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors
""",
    # one CUDA device per process (the runtime's design): a calculator on another GPU is refused, the first one keeps working
    "second_device_refused": """
c0 = GRACERuntimeCalculator("{M1}", device=PLATFORM + ":0")
if {NGPU} > 1:
    try:
        GRACERuntimeCalculator("{M1}", device=PLATFORM + ":1")
        raise AssertionError(":1 accepted after :0")
    except ValueError as e:
        assert "one GPU per process" in str(e), e
else:
    print("SKIP-PART one visible GPU")
run(c0, structures()[0])
""",
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_several_calculators(case):
    code = PRELUDE + textwrap.dedent(CASES[case]).format(M1=M1, M2=M2, NGPU=gpu_count())
    env = {k: v for k, v in os.environ.items() if k != "GRACE_RUNTIME_DEVICE"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]

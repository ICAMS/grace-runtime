# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""OpenMP runtimes in one process: grace-runtime's libgomp (vendored by auditwheel under a mangled name)
next to scikit-learn's libgomp, an MKL numpy (libiomp5) and TensorFlow (TPCalculator). Every scenario runs in a fresh
subprocess: the energy and forces must equal those of a process with grace-runtime alone, after the other library has
used its threads, and threadpoolctl must list the runtimes. Scenarios whose library is not installed are skipped, so
running this file in different environments (plain, MKL numpy, TF) covers the combinations."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_PRE = r"""
import json, sys, numpy as np
from ase.build import bulk
def grace(model):
    import grace_runtime
    a = bulk("W", "bcc", 3.16, cubic=True).repeat(4)
    a.rattle(0.05, seed=3)
    a.calc = grace_runtime.GRACERuntimeCalculator(model, num_threads=4)
    return {"e": a.get_potential_energy(), "f": a.get_forces().tolist()}
"""
_INFO = r"""
from threadpoolctl import threadpool_info
out["runtimes"] = sorted({(i["internal_api"], i.get("prefix", "")) for i in threadpool_info()})
print(json.dumps(out, default=list))
"""


def _run(body, model, extra_env=None):
    code = _PRE + body + _INFO
    env = dict(os.environ, **(extra_env or {}))
    r = subprocess.run([sys.executable, "-c", code, model], capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def alone(models):
    pytest.importorskip("threadpoolctl")
    return _run("out = grace(sys.argv[1])", models["2l"])


def _same(a, b):
    assert abs(a["e"] - b["e"]) <= 1e-9 * abs(b["e"])
    assert np.abs(np.array(a["f"]) - np.array(b["f"])).max() <= 1e-9


def test_alone_lists_grace_runtime_openmp(alone):
    assert any(api == "openmp" for api, _ in alone["runtimes"])


def test_with_scikit_learn(models, alone):
    pytest.importorskip("sklearn")
    body = r"""
from sklearn.cluster import KMeans
x = np.random.default_rng(0).normal(size=(20000, 8))
KMeans(n_clusters=8, n_init=2, random_state=0).fit(x)          # sklearn's OpenMP threads
out = grace(sys.argv[1])
KMeans(n_clusters=8, n_init=2, random_state=0).fit(x)          # and again after ours
out.update(grace(sys.argv[1]))
"""
    out = _run(body, models["2l"])
    _same(out, alone)
    assert sum(api == "openmp" for api, _ in out["runtimes"]) >= 1


def test_with_mkl_numpy(models, alone):
    from threadpoolctl import threadpool_info
    if not any(i["internal_api"] == "mkl" for i in threadpool_info()):
        pytest.skip("numpy is not linked to MKL in this environment")
    body = r"""
a = np.random.default_rng(1).normal(size=(2000, 2000))
s = float(np.linalg.eigvalsh(a @ a.T)[-1])                      # MKL threads (libiomp5)
out = grace(sys.argv[1])
s2 = float(np.linalg.eigvalsh(a @ a.T)[-1])
assert s == s2
"""
    out = _run(body, models["2l"])
    _same(out, alone)
    apis = {api for api, _ in out["runtimes"]}
    assert {"mkl", "openmp"} <= apis


def test_with_tensorflow_tpcalculator(models, alone):
    pytest.importorskip("tensorpotential")
    cache = os.path.expanduser(os.environ.get("GRACE_TF_CACHE", os.environ.get("GRACE_CACHE", "~/.cache/grace")))
    saved = os.path.join(cache, "GRACE-1L-OMAT-large-base")
    if not os.path.isfile(os.path.join(saved, "saved_model.pb")):
        pytest.skip(f"no SavedModel {saved}")
    body = r"""
from tensorpotential.calculator.asecalculator import TPCalculator
b = bulk("W", "bcc", 3.16, cubic=True).repeat(2)
b.calc = TPCalculator(sys.argv[2])
e_tf = b.get_potential_energy()                                 # TensorFlow's thread pools
out = grace(sys.argv[1])
b.rattle(0.01, seed=1)
out["tf_after"] = float(b.get_potential_energy())               # TF still works after grace-runtime's OpenMP
"""
    code = _PRE + body + _INFO
    r = subprocess.run([sys.executable, "-c", code, models["2l"], saved], capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    _same(out, alone)
    assert np.isfinite(out["tf_after"])

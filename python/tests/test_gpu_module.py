# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""GPU module selection and Kokkos lifetime. Skipped without a GPU or a matching module."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from grace_runtime import _cuda, _gpu, _rocm

from gpu_tools import GPU, NO_GPU, NO_MODULE_ARCH, NO_MODULE_ENV, PLATFORM, gpu_ok, other_gpu

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M2 = CACHE / "GRACE-2L-SMAX-OMAT-medium" / "kokkos.npz"



pytestmark = pytest.mark.skipif(not gpu_ok(), reason=NO_GPU)


def run_py(code, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=e,
                          timeout=600)


def test_module_matches_device():
    mod = _gpu.ensure_init(GPU)
    if PLATFORM == "cuda":
        major, minor = _cuda.compute_capability(0)
        assert mod.arch() == _cuda.arch_for(major, minor, _cuda.available())    # sm86 runs sm80 of the wheels
    else:
        assert mod.arch() == _rocm.probe(0)["arch"]    # AMD: exactly the device's architecture
    assert getattr(mod, "platform", lambda: "cuda")() == PLATFORM
    assert mod.initialized() and mod.device() == 0
    assert _gpu.ensure_init(PLATFORM + ":0") is mod    # second call: no-op
    assert _gpu.ensure_init("gpu") is mod    # "gpu": the GPU in use


def test_one_device_per_process():
    _gpu.ensure_init(GPU)
    with pytest.raises(ValueError, match="one GPU per process"):
        _gpu.ensure_init(other_gpu(1))
    other = {"cuda": "rocm", "rocm": "cuda"}[PLATFORM]    # nor a GPU of the other vendor
    with pytest.raises(ValueError, match="one GPU per process"):
        _gpu.ensure_init(other)


def test_bad_device_string():
    with pytest.raises(ValueError, match="unknown device"):
        _gpu.parse_device("gpu0")


def test_missing_arch_is_importerror_naming_installed():
    r = run_py("""
        import os
        from grace_runtime import _gpu
        try:
            _gpu.ensure_init(os.environ["GRACE_TEST_GPU"])
        except ImportError as e:
            print("IMPORTERROR", e)
        """, env=NO_MODULE_ENV)
    assert "IMPORTERROR" in r.stdout and NO_MODULE_ARCH in r.stdout and "installed:" in r.stdout, r.stdout + r.stderr


@pytest.mark.skipif(not M2.exists(), reason="registry model GRACE-2L-SMAX-OMAT-medium not in the test cache")
def test_lifecycle_clean_exit():
    """several systems, one deleted, one alive at exit: finalize releases it before Kokkos goes; no GPU error"""
    r = run_py(f"""
        import numpy as np
        import os
        from grace_runtime import _gpu
        mod = _gpu.ensure_init(os.environ["GRACE_TEST_GPU"])
        m = mod.Model({str(M2)!r})
        cell = np.eye(3) * 6.33
        pos = np.array([[0, 0, 0], [1.6, 1.6, 1.6]], float)
        sys_ = [mod.System(m, ["W"]) for _ in range(3)]
        for s in sys_:
            s.compute(np.array([1, 1], np.int32), pos, cell, [True, True, True])
        del sys_[1]
        print("OK", len(sys_))
        """)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "OK 2" in r.stdout
    low = r.stderr.lower()
    assert "error" not in low and "kokkos" not in low, r.stderr[-2000:]

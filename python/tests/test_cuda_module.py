# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""GPU module selection and Kokkos lifetime. Skipped without a CUDA device or a matching module."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from grace_runtime import _cuda

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M2 = CACHE / "GRACE-2L-SMAX-OMAT-medium" / "kokkos.npz"


def _cuda_ok():
    try:
        _cuda.module_for(0)
        return True
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device or no grace_runtime._kk_<arch> for it")


def run_py(code, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=e,
                          timeout=600)


def test_module_matches_device():
    major, minor = _cuda.compute_capability(0)
    mod = _cuda.ensure_init("cuda")
    assert mod.arch() == _cuda.arch_for(major, minor, _cuda.available())    # sm86 runs sm80 of the wheels
    assert mod.initialized() and mod.device() == 0
    assert _cuda.ensure_init("cuda:0") is mod    # second call: no-op


def test_one_device_per_process():
    _cuda.ensure_init("cuda")
    with pytest.raises(ValueError, match="one CUDA device per process"):
        _cuda.ensure_init("cuda:1")


def test_bad_device_string():
    with pytest.raises(ValueError, match="unknown device"):
        _cuda.parse_device("gpu0")


def test_missing_arch_is_importerror_naming_installed():
    r = run_py("""
        from grace_runtime import _cuda
        try:
            _cuda.ensure_init("cuda")
        except ImportError as e:
            print("IMPORTERROR", e)
        """, env={"GRACE_RUNTIME_CUDA_ARCH": "sm00"})
    assert "IMPORTERROR" in r.stdout and "sm00" in r.stdout and "installed:" in r.stdout, r.stdout + r.stderr


@pytest.mark.skipif(not M2.exists(), reason="registry model GRACE-2L-SMAX-OMAT-medium not in the test cache")
def test_lifecycle_clean_exit():
    """several systems, one deleted, one alive at exit: finalize releases it before Kokkos goes; no CUDA error"""
    r = run_py(f"""
        import numpy as np
        from grace_runtime import _cuda
        mod = _cuda.ensure_init("cuda")
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

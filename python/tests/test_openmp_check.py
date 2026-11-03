# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""macOS: a second OpenMP runtime in the process (PyTorch from PyPI) gives a RuntimeError at our first call into the
extension instead of the abort "OMP: Error #15"; the libraries loaded are faked here."""
import sys

import numpy as np
import pytest

from grace_runtime import _core

TWO = ["/env/lib/libomp.dylib", "/env/lib/python3.12/site-packages/torch/lib/libomp.dylib"]
macos = pytest.mark.skipif(sys.platform != "darwin", reason="the OpenMP check runs on macOS only")


@pytest.fixture
def unchecked(monkeypatch):
    _core._check_openmp.cache_clear()
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    yield monkeypatch
    _core._check_openmp.cache_clear()


def test_two_runtimes_raise(unchecked):
    unchecked.setattr(_core, "_openmp_libraries", lambda: TWO)
    for _ in range(2):    # a failure is checked again
        with pytest.raises(RuntimeError, match="two OpenMP runtimes .*torch/lib/libomp.dylib.*conda-forge"):
            _core._check_openmp()


@pytest.mark.parametrize("libs, env", [(TWO[:1], None), ([], None), (TWO, "True"), (TWO, "1"), (TWO, "y")])
def test_one_runtime_or_duplicates_allowed(unchecked, libs, env):
    unchecked.setattr(_core, "_openmp_libraries", lambda: libs)
    if env:
        unchecked.setenv("KMP_DUPLICATE_LIB_OK", env)    # libomp does not abort then (scikit-learn sets it)
    _core._check_openmp()
    unchecked.setattr(_core, "_openmp_libraries", lambda: TWO)
    unchecked.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    _core._check_openmp()    # checked once: a runtime loaded after ours has started is out of reach


@pytest.mark.parametrize("value, true", [("TRUE", True), ("t", True), ("tr", True), ("on", True), ("o", False),
                                         ("1", True), (".TRUE.", True), (".t.", True), ("y", True), ("enable", True),
                                         ("enabl", False), ("", False), ("0", False), ("false", False), ("no", False)])
def test_kmp_true_reads_values_as_libomp(value, true):
    assert _core._kmp_true(value) is true


@macos
def test_every_call_into_the_extension_checks(unchecked, models):
    unchecked.setattr(_core, "_openmp_libraries", lambda: TWO)
    with pytest.raises(RuntimeError, match="two OpenMP runtimes"):
        _core.neighbor_search(np.zeros((1, 3)), np.eye(3) * 5, np.ones(3, dtype=bool), 3.0)
    unchecked.setattr(_core, "_openmp_libraries", lambda: TWO[:1])
    model = _core.Model(models["3l"])    # GRACE-3L: cutoff_matrix starts OpenMP (the network tables)
    _core._check_openmp.cache_clear()
    unchecked.setattr(_core, "_openmp_libraries", lambda: TWO)
    with pytest.raises(RuntimeError, match="two OpenMP runtimes"):
        model.cutoff_matrix(["W"])
    for k, v in [*vars(_core._ext).items(), *vars(_core._ext.Model).items()]:
        if callable(v) and not k.startswith("__") and not isinstance(v, type):
            assert getattr(_core.Model if k in vars(_core._ext.Model) else _core, k).__wrapped__ is v, k


@macos
def test_the_loaded_runtime_is_found():
    assert len(_core._openmp_libraries()) == 1

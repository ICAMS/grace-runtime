# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""GPU backend discovery, no GPU needed: plug-ins of the entry point group grace_runtime.backends with a
matching ABI, the architecture chosen for a compute capability (same major, highest minor not above the device's),
clear ImportErrors (no plug-in, ABI mismatch, no module for the device) and device="auto" falling back to the CPU."""
import os
import subprocess
import sys
import textwrap

import pytest

from grace_runtime import _cuda


@pytest.mark.parametrize("cc, want", [((8, 0), "sm80"), ((8, 6), "sm80"), ((8, 7), "sm80"), ((8, 9), "sm89"),
                                      ((9, 0), "sm90"), ((7, 5), None), ((10, 0), None), ((12, 0), None)])
def test_arch_for(cc, want):
    assert _cuda.arch_for(*cc, ["sm80", "sm89", "sm90"]) == want


def test_arch_for_never_picks_a_newer_minor():
    assert _cuda.arch_for(8, 0, ["sm89"]) is None
    assert _cuda.arch_for(8, 6, ["sm86", "sm80"]) == "sm86"


def fake_plugin(tmp_path, name, abi, archs=("sm80",)):
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(textwrap.dedent(f"""
        ABI = {abi}
        __version__ = "9.9.9"
        def archs():
            return {list(archs)!r}
    """))
    dist = tmp_path / f"{name}-9.9.9.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 9.9.9\n")
    (dist / "entry_points.txt").write_text(f"[grace_runtime.backends]\n{name} = {name}\n")
    return tmp_path


def test_abi_mismatch_is_a_clear_import_error(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(fake_plugin(tmp_path, "gn_fake_backend", abi=_cuda.ABI + 1)))
    with pytest.raises(ImportError, match=r"9\.9\.9 has GPU backend ABI 2.*needs ABI 1"):
        _cuda.plugins()
    sys.modules.pop("gn_fake_backend", None)


def test_matching_plugin_is_found(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(fake_plugin(tmp_path, "gn_fake_ok", abi=_cuda.ABI, archs=("sm75",))))
    assert any(p.__name__ == "gn_fake_ok" for p in _cuda.plugins())
    assert _cuda.available()["sm75"] == "gn_fake_ok._kk_sm75"
    sys.modules.pop("gn_fake_ok", None)


def test_no_backend_installed(monkeypatch):
    monkeypatch.setattr(_cuda, "plugins", lambda: [])
    monkeypatch.setattr(_cuda, "installed_archs", lambda: [])
    monkeypatch.setattr(_cuda, "compute_capability", lambda index: (8, 0))
    monkeypatch.setitem(_cuda._state, "module", None)
    monkeypatch.delenv("GRACE_RUNTIME_DEVICE", raising=False)
    with pytest.raises(ImportError, match="grace-runtime-cuda12"):
        _cuda.module_for(0)
    device, why = _cuda.auto_device()
    assert device == "cpu" and "grace-runtime-cuda12" in why


def test_no_module_for_the_device(monkeypatch):
    monkeypatch.setattr(_cuda, "available", lambda: {"sm80": "x._kk_sm80", "sm89": "x._kk_sm89"})
    monkeypatch.setattr(_cuda, "compute_capability", lambda index: (12, 0))
    monkeypatch.setitem(_cuda._state, "module", None)
    monkeypatch.delenv("GRACE_RUNTIME_DEVICE", raising=False)
    with pytest.raises(ImportError, match=r"sm120.*sm80, sm89"):
        _cuda.module_for(0)
    device, why = _cuda.auto_device()
    assert device == "cpu" and "sm120" in why


def test_no_driver_is_silent(monkeypatch):
    def none(index):
        raise _cuda.NoGPU("no NVIDIA driver (libcuda.so.1)")
    monkeypatch.setattr(_cuda, "compute_capability", none)
    monkeypatch.setitem(_cuda._state, "module", None)
    monkeypatch.delenv("GRACE_RUNTIME_DEVICE", raising=False)
    assert _cuda.auto_device() == ("cpu", None)


class FakeLibcuda:
    """libcuda.so.1 of a driver that supports CUDA `version` (e.g. 11080: CUDA 11.8) with one sm80 device"""
    def __init__(self, version):
        self.version = version

    def cuInit(self, flags):
        return 0

    def cuDriverGetVersion(self, out):
        out._obj.value = self.version
        return 0

    def cuDeviceGetCount(self, out):
        out._obj.value = 1
        return 0

    def cuDeviceGet(self, out, index):
        out._obj.value = 0
        return 0

    def cuDeviceGetAttribute(self, out, attr, dev):
        out._obj.value = {75: 8, 76: 0}[attr]
        return 0


def test_driver_older_than_cuda12_is_a_clear_import_error(monkeypatch):
    monkeypatch.setattr(_cuda.ctypes, "CDLL", lambda name, *a, **k: FakeLibcuda(11080))
    monkeypatch.setitem(_cuda._state, "module", None)
    monkeypatch.delenv("GRACE_RUNTIME_DEVICE", raising=False)
    with pytest.raises(ImportError, match=r"CUDA 11\.8.*CUDA >= 12\.0.*525") as e:
        _cuda.compute_capability(0)
    assert not isinstance(e.value, _cuda.NoGPU)    # a GPU is there: auto warns
    device, why = _cuda.auto_device()
    assert device == "cpu" and "11.8" in why


def test_cuda12_driver_passes(monkeypatch):
    monkeypatch.setattr(_cuda.ctypes, "CDLL", lambda name, *a, **k: FakeLibcuda(12010))
    assert _cuda.compute_capability(0) == (8, 0)


# ---- no module of the device's family: opt-in PTX fallback, strict mode, visible notices ----

@pytest.mark.parametrize("cc, want", [((10, 0), "sm90"), ((13, 0), "sm120"), ((9, 0), "sm89"), ((8, 0), "sm70"),
                                     ((7, 0), None)])
def test_ptx_arch_for_takes_the_newest_older_family(cc, want):
    assert _cuda.ptx_arch_for(*cc, ["sm70", "sm80", "sm89", "sm90", "sm120"]) == want


@pytest.fixture
def sm100_without_module(monkeypatch):
    """an sm100 device, modules sm80 and sm90 only, a driver for CUDA 12.9"""
    monkeypatch.setattr(_cuda, "available", lambda: {"sm80": "x._kk_sm80", "sm90": "x._kk_sm90"})
    monkeypatch.setattr(_cuda, "compute_capability", lambda index: (10, 0))
    monkeypatch.setitem(_cuda._state, "module", None)
    monkeypatch.setitem(_cuda._state, "driver", 12090)
    for k in ("GRACE_RUNTIME_DEVICE", "GRACE_RUNTIME_ALLOW_PTX", "GRACE_RUNTIME_STRICT", "GRACE_RUNTIME_CUDA_ARCH"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_without_opt_in_no_ptx_and_the_ways_out_are_named(sm100_without_module):
    with pytest.raises(ImportError, match=r"sm100.*GRACE_RUNTIME_ALLOW_PTX=1.*sm90.*GRACE_KK_ARCHS=sm100"):
        _cuda.select(0)
    device, why = _cuda.auto_device()
    assert device == "cpu" and "GRACE_RUNTIME_ALLOW_PTX" in why


def test_opt_in_takes_the_older_ptx(sm100_without_module):
    sm100_without_module.setenv("GRACE_RUNTIME_ALLOW_PTX", "1")
    assert _cuda.select(0) == ("sm90", True)
    sm100_without_module.setattr(_cuda.importlib, "import_module", lambda name: object())    # the fake modules
    assert _cuda.auto_device() == ("cuda", None)


def test_ptx_needs_a_driver_for_the_toolkit_of_the_modules(sm100_without_module):
    sm100_without_module.setenv("GRACE_RUNTIME_ALLOW_PTX", "1")
    sm100_without_module.setitem(_cuda._state, "driver", 12040)
    with pytest.raises(ImportError, match=r"CUDA >= 12\.9.*supports 12\.4.*update the driver"):
        _cuda.select(0)


def test_strict_mode_refuses_ptx_and_does_not_suggest_it(sm100_without_module):
    sm100_without_module.setenv("GRACE_RUNTIME_ALLOW_PTX", "1")
    sm100_without_module.setenv("GRACE_RUNTIME_STRICT", "1")
    with pytest.raises(ImportError) as e:
        _cuda.select(0)
    assert "ALLOW_PTX" not in str(e.value) and "GRACE_KK_ARCHS=sm100" in str(e.value)


def test_native_module_is_not_ptx(monkeypatch):
    monkeypatch.setattr(_cuda, "available", lambda: {"sm80": "x._kk_sm80", "sm90": "x._kk_sm90"})
    monkeypatch.setattr(_cuda, "compute_capability", lambda index: (8, 9))
    monkeypatch.delenv("GRACE_RUNTIME_CUDA_ARCH", raising=False)
    assert _cuda.select(0) == ("sm80", False)


def test_notice_reaches_stderr_despite_warning_filters():
    """the banner is printed even when RuntimeWarnings are ignored; GRACE_RUNTIME_QUIET=1 silences only the banner"""
    code = ("import warnings; warnings.simplefilter('ignore')\n"
            "from grace_runtime import _cuda\n_cuda.notice('test fallback XYZ')\n_cuda.notice('test fallback XYZ')\n")
    env = {k: v for k, v in os.environ.items() if k != "GRACE_RUNTIME_QUIET"}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stderr.count("grace-runtime WARNING: test fallback XYZ") == 1, r.stderr
    r = subprocess.run([sys.executable, "-c", code], env=dict(env, GRACE_RUNTIME_QUIET="1"), capture_output=True,
                       text=True, timeout=120)
    assert "XYZ" not in r.stderr, r.stderr

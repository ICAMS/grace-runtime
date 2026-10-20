# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""device="auto" (the default, as TensorFlow places work on a GPU when there is one): the GPU when a module for it is
installed and the fused kernels take the model, else the CPU — silently without a GPU, with a RuntimeWarning naming
the reason when a GPU is there but cannot be used. GRACE_RUNTIME_DEVICE overrides the choice; an explicit "cuda"
still raises."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from grace_runtime import GRACERuntimeCalculator, _cuda, compute_batch

FX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M2 = CACHE / "GRACE-2L-OMAT-medium-ft-E" / "kokkos.npz"
UNSUPPORTED = Path.home() / ".cache" / "grace" / "GRACE-2L-OMAT-large-mx" / "kokkos.npz"    # fused: ReduceN widths


def _cuda_ok():
    try:
        _cuda.module_for(0)
        return True
    except (ImportError, ValueError):
        return False


pytestmark = [pytest.mark.skipif(not _cuda_ok(), reason="no CUDA module for this device"),
              pytest.mark.skipif(not M2.exists(), reason="registry model missing")]


def wmo54():
    return read(FX / "WMo54.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})


def test_auto_is_default_and_takes_the_gpu(monkeypatch):
    monkeypatch.delenv("GRACE_RUNTIME_DEVICE", raising=False)
    calc = GRACERuntimeCalculator(str(M2))
    assert calc.backend.startswith("cuda:"), calc.backend
    at = wmo54()
    at.calc = calc
    ref = wmo54()
    ref.calc = GRACERuntimeCalculator(str(M2), device="cuda")
    assert abs(at.get_potential_energy() - ref.get_potential_energy()) <= 1e-6 * len(at)
    a = compute_batch(str(M2), [wmo54()])[0]
    b = compute_batch(str(M2), [wmo54()], device="cuda")[0]
    assert abs(a["energy"] - b["energy"]) <= 1e-6 * 54


def test_environment_overrides_auto(monkeypatch):
    monkeypatch.setenv("GRACE_RUNTIME_DEVICE", "cpu")
    assert GRACERuntimeCalculator(str(M2)).backend == "cpu"


@pytest.mark.skipif(not UNSUPPORTED.exists(), reason="unsupported test model absent")
def test_auto_falls_back_for_a_model_outside_the_fused_kernels(monkeypatch):
    monkeypatch.delenv("GRACE_RUNTIME_DEVICE", raising=False)
    with pytest.warns(RuntimeWarning, match="fused GPU kernels"):
        calc = GRACERuntimeCalculator(str(UNSUPPORTED))
    assert calc.backend == "cpu"
    with pytest.raises(ValueError, match="not supported on device='cuda'"):
        GRACERuntimeCalculator(str(UNSUPPORTED), device="cuda").get_potential_energy(wmo54())
    with pytest.warns(RuntimeWarning, match="fused GPU kernels"):
        r = compute_batch(str(UNSUPPORTED), [wmo54()])
    assert "energy" in r[0]


def test_auto_falls_back_without_a_module_for_the_gpu():
    """a GPU of an architecture without a module (simulated): CPU with a warning, in a fresh process"""
    code = ("import warnings, sys; warnings.simplefilter('error', RuntimeWarning)\n"
            "from grace_runtime import GRACERuntimeCalculator\n"
            "try:\n    GRACERuntimeCalculator(sys.argv[1])\nexcept RuntimeWarning as w:\n    print('WARN', w)\n")
    env = dict(os.environ, GRACE_RUNTIME_CUDA_ARCH="sm00")
    env.pop("GRACE_RUNTIME_DEVICE", None)
    out = subprocess.run([sys.executable, "-c", code, str(M2)], env=env, capture_output=True, text=True, timeout=300)
    assert "WARN" in out.stdout and "sm00" in out.stdout, out.stdout + out.stderr


def test_auto_runs_3l_on_the_gpu(monkeypatch, models):
    """GRACE-3L: auto takes the GPU without a warning, as an explicit "cuda"; energies as the CPU (fp32)"""
    import warnings
    monkeypatch.delenv("GRACE_RUNTIME_DEVICE", raising=False)
    M3 = models["3l"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        calc = GRACERuntimeCalculator(str(M3))
        r = compute_batch(str(M3), [wmo54()])
    assert calc.backend.startswith("cuda:"), calc.backend
    at = wmo54()
    at.calc = calc
    ref = wmo54()
    ref.calc = GRACERuntimeCalculator(str(M3), device="cpu")
    e_ref = ref.get_potential_energy()
    assert abs(at.get_potential_energy() - e_ref) <= 2e-6 * len(at)
    assert abs(r[0]["energy"] - e_ref) <= 2e-6 * 54
    assert GRACERuntimeCalculator(str(M3), device="cuda").backend.startswith("cuda:")


def _fresh(code, *args):
    """run code in a new process without GRACE_RUNTIME_DEVICE; its stdout"""
    env = dict(os.environ)
    env.pop("GRACE_RUNTIME_DEVICE", None)
    r = subprocess.run([sys.executable, "-c", code, *map(str, args)], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return r.stdout


def test_compute_batch_on_the_cpu_stays_there():
    """compute_batch(device="cpu") must not start the GPU backend"""
    out = _fresh("import sys\n"
                 "from ase.build import bulk\n"
                 "from grace_runtime import compute_batch, _cuda\n"
                 "compute_batch(sys.argv[1], [bulk('W', 'bcc', a=3.165, cubic=True)], device='cpu')\n"
                 "print('KK', _cuda._state['module'] is not None)\n", M2)
    assert "KK False" in out, out


def _run(code, **env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("GRACE_RUNTIME_DEVICE", "GRACE_RUNTIME_QUIET",
                                                            "GRACE_RUNTIME_STRICT", "GRACE_RUNTIME_ALLOW_PTX")}
    env.update(env_extra)
    return subprocess.run([sys.executable, "-c", code, str(M2)], env=env, capture_output=True, text=True, timeout=600)


def test_backend_info_of_a_native_module(monkeypatch):
    monkeypatch.delenv("GRACE_RUNTIME_DEVICE", raising=False)
    i = GRACERuntimeCalculator(str(M2)).backend_info
    assert i["native"] is True and i["fallback"] is None and i["device"].startswith("cuda:"), i


def test_cpu_fallback_is_announced_on_stderr_and_in_backend_info():
    code = ("import warnings, sys; warnings.simplefilter('ignore')\n"
            "from grace_runtime import GRACERuntimeCalculator\n"
            "c = GRACERuntimeCalculator(sys.argv[1]); print('INFO', c.backend, c.backend_info['fallback'])\n")
    r = _run(code, GRACE_RUNTIME_CUDA_ARCH="sm00")
    assert "INFO cpu" in r.stdout and "sm00" in r.stdout, r.stdout + r.stderr
    assert "grace-runtime WARNING: device='auto'" in r.stderr and "running on the CPU" in r.stderr, r.stderr


def test_strict_mode_makes_the_cpu_fallback_an_error():
    code = "import sys\nfrom grace_runtime import GRACERuntimeCalculator\nGRACERuntimeCalculator(sys.argv[1])\n"
    r = _run(code, GRACE_RUNTIME_CUDA_ARCH="sm00", GRACE_RUNTIME_STRICT="1")
    assert r.returncode != 0 and "GRACE_RUNTIME_STRICT=1" in r.stderr, r.stdout + r.stderr


def _older_family_module():
    """an installed module of an older major than this GPU (its PTX runs here through the driver's JIT), or None"""
    major, _ = _cuda.compute_capability(0)
    older = [a for a in _cuda.available() if int(a[2:-1]) < major]
    return max(older, key=lambda a: int(a[2:])) if older else None


@pytest.mark.skipif(_cuda_ok() and _older_family_module() is None, reason="no module of an older GPU family installed")
def test_ptx_of_an_older_module_gives_the_same_results(tmp_path):
    """the PTX path (forced here with GRACE_RUNTIME_CUDA_ARCH; real use: GRACE_RUNTIME_ALLOW_PTX=1 on a GPU without a
    module of its family): announced, visible in backend / backend_info, same energy and forces as the native module;
    refused in strict mode"""
    code = ("import sys, json, numpy as np\nfrom ase.build import bulk\nfrom grace_runtime import GRACERuntimeCalculator\n"
            "at = bulk('W', 'bcc', a=3.165, cubic=True).repeat(3); at.rattle(0.05, seed=3)\n"
            "c = GRACERuntimeCalculator(sys.argv[1], device='cuda'); at.calc = c\n"
            "print(json.dumps({'backend': c.backend, 'info': c.backend_info, 'e': at.get_potential_energy(), "
            "'f': at.get_forces().tolist()}))\n")
    import json
    native = _run(code)
    assert native.returncode == 0, native.stderr
    a = json.loads(native.stdout.splitlines()[-1])
    older = _older_family_module()
    cache = str(tmp_path / "jit-cache")
    ptx = _run(code, GRACE_RUNTIME_CUDA_ARCH=older, CUDA_CACHE_PATH=cache)
    assert ptx.returncode == 0, ptx.stderr
    b = json.loads(ptx.stdout.splitlines()[-1])
    assert "PTX" in b["backend"] and b["info"]["native"] is False and b["info"]["fallback"] == "ptx", b
    assert f"running the {older} module's PTX" in ptx.stderr and "grace-runtime WARNING" in ptx.stderr, ptx.stderr
    assert abs(a["e"] - b["e"]) <= 1e-6 * 54 and np.abs(np.array(a["f"]) - np.array(b["f"])).max() <= 1e-5
    strict = _run(code, GRACE_RUNTIME_CUDA_ARCH=older, GRACE_RUNTIME_STRICT="1", CUDA_CACHE_PATH=cache)
    assert strict.returncode != 0 and "GRACE_RUNTIME_STRICT=1" in strict.stderr, strict.stderr

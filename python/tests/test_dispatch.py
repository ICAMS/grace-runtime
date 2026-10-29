# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""ISA dispatch: _core_x86_64_v3 / _core_x86_64_v4 / _core_aarch64 (/ _core_native) chosen at import;
GRACE_RUNTIME_CPU forces one."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from grace_runtime import _dispatch

V4 = {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}
V3 = {"avx", "avx2", "bmi1", "bmi2", "fma", "f16c", "movbe", "lzcnt"}   # lzcnt shows as "abm" in /proc/cpuinfo


def test_select_prefers_v4_on_avx512():
    assert _dispatch.select(V3 | V4 | {"abm"}, None, {"x86_64_v3", "x86_64_v4"}) == "x86_64_v4"


def test_select_v3_without_avx512():
    assert _dispatch.select(V3 | {"abm"}, None, {"x86_64_v3", "x86_64_v4"}) == "x86_64_v3"


def test_select_native_build_wins():
    assert _dispatch.select(V3 | {"abm"}, None, {"native"}) == "native"


def test_forced_v4_without_avx512_names_missing_flags():
    with pytest.raises(ImportError, match="avx512f"):
        _dispatch.select(V3 | {"abm"}, "v4", {"x86_64_v3", "x86_64_v4"})


def test_forced_bad_value():
    with pytest.raises(ImportError, match="GRACE_RUNTIME_CPU"):
        _dispatch.select(V3 | V4 | {"abm"}, "v5", {"x86_64_v3", "x86_64_v4"})


def test_no_usable_module():
    with pytest.raises(ImportError, match="x86-64-v3"):
        _dispatch.select({"sse2"}, None, {"x86_64_v3", "x86_64_v4"})


A64 = {"fp", "asimd", "atomics", "asimdrdm", "crc32"}    # Linux "Features" of an ARMv8.2-A CPU (Neoverse N1)


def test_cpu_flags_reads_arm_features(tmp_path):
    p = tmp_path / "cpuinfo"
    p.write_text("processor\t: 0\nBogoMIPS\t: 50.00\nFeatures\t: fp asimd atomics asimdrdm\nCPU part\t: 0xd0c\n")
    assert _dispatch.cpu_flags(str(p)) == {"fp", "asimd", "atomics", "asimdrdm"}


def test_select_aarch64():
    assert _dispatch.select(A64, None, {"aarch64"}) == "aarch64"
    assert _dispatch.select(set(), None, {"aarch64"}) == "aarch64"      # macOS: no /proc/cpuinfo
    assert _dispatch.select(A64, "aarch64", {"aarch64"}) == "aarch64"


def test_aarch64_on_an_armv8_0_cpu():
    with pytest.raises(ImportError, match="ARMv8.2-A.*lacks atomics.*--no-binary"):
        _dispatch.select({"fp", "asimd", "crc32"}, None, {"aarch64"})    # Cortex-A72 (Raspberry Pi 4)


def test_no_extension_at_all():
    with pytest.raises(ImportError, match="no compiled extension"):
        _dispatch.select(V3 | V4 | {"abm"}, None, set())


def test_loaded_module_is_reported():
    import grace_runtime
    assert grace_runtime.cpu_module in _dispatch.available()
    assert grace_runtime.cpu_module == _dispatch.select(_dispatch.cpu_flags(), os.environ.get("GRACE_RUNTIME_CPU"),
                                                       _dispatch.available())


_RUN = r"""
import json, sys, numpy as np, grace_runtime
from ase.build import bulk
a = bulk("W", "bcc", 3.16, cubic=True).repeat(3)
a.rattle(0.1, seed=7)
a.symbols[::5] = "Mo"
out = {"module": grace_runtime.cpu_module}
for arch, path in json.loads(sys.argv[1]).items():
    a.calc = grace_runtime.GRACERuntimeCalculator(path, num_threads=1)
    out[arch] = {"e": a.get_potential_energy(), "f": a.get_forces().tolist()}
print(json.dumps(out))
"""


def _run(cpu, models):
    env = dict(os.environ, GRACE_RUNTIME_CPU=cpu)
    r = subprocess.run([sys.executable, "-c", _RUN, json.dumps(models)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(not V4 <= _dispatch.cpu_flags(), reason="needs AVX-512 for the v4 module")
@pytest.mark.skipif(not {"x86_64_v3", "x86_64_v4"} <= _dispatch.available(),
                    reason="needs both ISA modules (a dual-ISA install)")
def test_v3_and_v4_agree(models):
    v3, v4 = _run("v3", models), _run("v4", models)
    assert (v3["module"], v4["module"]) == ("x86_64_v3", "x86_64_v4")
    n = 54
    for arch in models:
        assert abs(v3[arch]["e"] - v4[arch]["e"]) / n < 1e-6
        assert np.abs(np.array(v3[arch]["f"]) - np.array(v4[arch]["f"])).max() < 2e-5

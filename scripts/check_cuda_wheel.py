# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""check_cuda_wheel.py <wheel> [archs]: a grace-runtime-cuda12 wheel holds one module per architecture (default sm70,
sm80, sm90, sm100, sm120) for its CPython tag, the plug-in package with ABI and the grace_runtime.backends entry point; no CUDA driver
or toolkit library except the redistributable cudart; every module exports only PyInit_* (Kokkos and libstdc++ hidden);
it is manylinux_2_17 x86_64 but for libcuda.so.1, which the driver provides (wheel_policy.py). Prints the size. Exit
status 1 on a failed check."""
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import wheel_policy

wheel = Path(sys.argv[1])
archs = (sys.argv[2] if len(sys.argv) > 2 else "sm70,sm80,sm90,sm100,sm120").split(",")
bad = []
with zipfile.ZipFile(wheel) as z:
    names = z.namelist()
    tag = re.search(r"-(cp\d+)-(cp\d+|abi3)-", wheel.name)
    for a in archs:
        mods = [n for n in names if re.fullmatch(rf"grace_runtime_cuda12/_kk_{a}\..*\.so", n)]
        if len(mods) != 1:
            bad.append(f"module _kk_{a}: {mods}")
        elif tag and tag.group(2) == "abi3" and ".abi3." not in mods[0]:
            bad.append(f"_kk_{a} is not an abi3 module: {mods[0]}")
    if "grace_runtime_cuda12/__init__.py" not in names:
        bad.append("grace_runtime_cuda12/__init__.py missing")
    else:
        init = z.read("grace_runtime_cuda12/__init__.py").decode()
        if not re.search(r"^ABI = \d+", init, re.M):
            bad.append("no ABI in grace_runtime_cuda12/__init__.py")
    ep = [n for n in names if n.endswith(".dist-info/entry_points.txt")]
    if not ep or "grace_runtime.backends" not in z.read(ep[0]).decode():
        bad.append("no grace_runtime.backends entry point")
    libs = [n for n in names if re.search(r"\.so(\.\d+)*$", n) and "/_kk_" not in n]
    for n in libs:
        base = n.rsplit("/", 1)[-1]
        if re.match(r"lib(cuda|cublas|cublasLt|nvrtc|cufft|cusparse)[.-]", base):
            bad.append(f"bundles {base} (driver / toolkit library)")
    with tempfile.TemporaryDirectory() as d:
        elfs = {n: Path(z.extract(n, d)) for n in names if re.search(r"\.so(\.\d+)*$", n)}
        for n, p in elfs.items():
            if "/_kk_" in n:
                out = subprocess.run(["nm", "-D", "--defined-only", str(p)], capture_output=True, text=True).stdout
                extra = [l.split()[-1] for l in out.splitlines() if l.split() and not l.split()[-1].startswith("PyInit_")]
                if extra:
                    bad.append(f"{n} exports {len(extra)} symbols besides PyInit_*: {extra[:5]}")
        needs = {n: wheel_policy.needed(p) for n, p in elfs.items()}
        bad += wheel_policy.check(wheel, "manylinux_2_17_x86_64", needs, ["libcuda.so.1"])
print(f"{wheel.name}: {wheel.stat().st_size / 2**20:.1f} MB, bundled libs: "
      f"{', '.join(n.rsplit('/', 1)[-1] for n in libs) or 'none'}")
for b in bad:
    print("FAIL", b)
sys.exit(1 if bad else 0)

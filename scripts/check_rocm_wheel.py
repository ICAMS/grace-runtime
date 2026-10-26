# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""check_rocm_wheel.py <wheel> [archs] [platform tag]: a grace-runtime-rocm6 wheel holds one module per architecture
(default gfx90a,gfx942) for its CPython tag, the plug-in package with ABI, PLATFORM = "rocm" and ROCM_MAJOR, and the
grace_runtime.backends entry point; it bundles no ROCm library (the HIP runtime comes from the system ROCm; rocBLAS,
hipBLAS, rocSOLVER and their Tensile kernels never: rocBLAS is loaded at run time) and no other library than libgomp;
the modules need libamdhip64.so.6 and no BLAS library (DT_NEEDED), carry no absolute RUNPATH and export only
PyInit_*; its glibc symbol versions (auditwheel) allow the platform tag (default manylinux_2_28_x86_64; '-': no platform
checks). Prints the size. Exit status 1
on a failed check."""
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

wheel = Path(sys.argv[1])
archs = (sys.argv[2] if len(sys.argv) > 2 else "gfx90a,gfx942").split(",")
plat = sys.argv[3] if len(sys.argv) > 3 else "manylinux_2_28_x86_64"
PKG = "grace_runtime_rocm6"
bad = []


def elf(path, what):
    return subprocess.run(["readelf", "-W", what, str(path)], capture_output=True, text=True).stdout


with zipfile.ZipFile(wheel) as z:
    names = z.namelist()
    tag = re.search(r"-(cp\d+)-(cp\d+|abi3)-", wheel.name)
    for a in archs:
        mods = [n for n in names if re.fullmatch(rf"{PKG}/_kk_{a}\..*\.so", n)]
        if len(mods) != 1:
            bad.append(f"module _kk_{a}: {mods}")
        elif tag and tag.group(2) == "abi3" and ".abi3." not in mods[0]:
            bad.append(f"_kk_{a} is not an abi3 module: {mods[0]}")
    if f"{PKG}/__init__.py" not in names:
        bad.append(f"{PKG}/__init__.py missing")
    else:
        init = z.read(f"{PKG}/__init__.py").decode()
        for pat in (r"^ABI = \d+", r'^PLATFORM = "rocm"', r"^ROCM_MAJOR = 6\b"):
            if not re.search(pat, init, re.M):
                bad.append(f"{PKG}/__init__.py: no {pat[1:]}")
    ep = [n for n in names if n.endswith(".dist-info/entry_points.txt")]
    if not ep or not re.search(rf"\[grace_runtime\.backends\]\s*rocm6\s*=\s*{PKG}\b", z.read(ep[0]).decode()):
        bad.append("no grace_runtime.backends entry point rocm6")
    # rocBLAS' Tensile kernels and other GPU code objects (rocblas/library/*.co, *.hsaco, TensileLibrary*.dat)
    blobs = [n for n in names if re.search(r"(\.co|\.hsaco|\.dat)$|Tensile|/rocblas/|/library/", n)]
    if blobs:
        bad.append(f"bundles GPU libraries' data files: {blobs[:5]}")
    libs = [n for n in names if re.search(r"\.so(\.\d+)*$", n) and "/_kk_" not in n]
    for n in libs:
        base = n.rsplit("/", 1)[-1]
        if not re.match(r"libgomp[.-]", base):
            bad.append(f"bundles {base} (only libgomp may be: the ROCm libraries come from the system)")
    with tempfile.TemporaryDirectory() as d:
        for n in names:
            if not re.search(r"\.so(\.\d+)*$", n):
                continue
            p = Path(z.extract(n, d))
            dyn = elf(p, "-d")
            needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^]]+)\]", dyn)
            blas = [x for x in needed if re.match(r"lib(rocblas|hipblas|hipblaslt|rocsolver|cublas)", x)]
            if blas:
                bad.append(f"{n} needs {blas} (BLAS must be loaded at run time, not linked)")
            if "/_kk_" in n and "libamdhip64.so.6" not in needed:
                bad.append(f"{n} does not need libamdhip64.so.6: {needed}")
            for rp in re.findall(r"\((?:RPATH|RUNPATH)\)\s+Library r(?:un)?path: \[([^]]+)\]", dyn):
                absolute = [x for x in rp.split(":") if x and not x.startswith("$ORIGIN")]
                if absolute:
                    bad.append(f"{n}: absolute RUNPATH {absolute} (build machine paths)")
            if "/_kk_" in n:
                out = subprocess.run(["nm", "-D", "--defined-only", str(p)], capture_output=True, text=True).stdout
                extra = [l.split()[-1] for l in out.splitlines() if l.split() and not l.split()[-1].startswith("PyInit_")]
                if extra:
                    bad.append(f"{n} exports {len(extra)} symbols besides PyInit_*: {extra[:5]}")

def glibc(tag):
    """(major, minor) of a manylinux tag (manylinux2014 = 2.17), None for others"""
    m = re.match(r"manylinux_(\d+)_(\d+)_", tag)
    return (int(m.group(1)), int(m.group(2))) if m else (2, 17) if tag.startswith("manylinux2014") else None


if plat.startswith("manylinux"):    # '-': no platform checks (a wheel for this machine)
    show = subprocess.run([sys.executable, "-m", "auditwheel", "show", str(wheel)], capture_output=True, text=True)
    # the excluded HIP runtime keeps the verdict at linux_x86_64; the glibc symbol versions give the constraint
    m = re.search(r'constrains the platform tag to\s*"([^"]+)"', show.stdout) or \
        re.search(r'consistent with the following platform tag:\s*"([^"]+)"', show.stdout)
    if not m or not glibc(m.group(1)) or glibc(m.group(1)) > glibc(plat):
        bad.append(f"auditwheel: not {plat} compatible:\n" + show.stdout[-800:] + show.stderr[-400:])
    if plat not in wheel.name:
        bad.append(f"platform tag of {wheel.name} is not {plat}")
print(f"{wheel.name}: {wheel.stat().st_size / 2**20:.1f} MB, bundled libs: "
      f"{', '.join(n.rsplit('/', 1)[-1] for n in libs) or 'none'}")
for b in bad:
    print("FAIL", b)
sys.exit(1 if bad else 0)

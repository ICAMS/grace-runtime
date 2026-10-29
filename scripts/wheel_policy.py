# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""The manylinux check of the GPU wheels (check_cuda_wheel.py, check_rocm_wheel.py). auditwheel calls such a wheel
linux_x86_64, because its modules need the GPU runtime the build leaves out on purpose (libcuda.so.1,
libamdhip64.so.6). So a wheel passes when auditwheel's tag fits the platform tag, or when the tag is linux_x86_64 for
missing libraries alone (its glibc symbol versions fit, no blacklisted symbol, no other policy failure) and every
library its ELF files need is in the policy's whitelist, vendored in the wheel, or one of the left-out GPU runtime
libraries."""
import importlib.resources
import json
import re
import subprocess
import sys

LOADER = "ld-linux-x86-64.so.2"    # the dynamic loader: needed by some objects, never in a whitelist


def glibc(tag):
    """(major, minor) of a manylinux tag (manylinux2014 = 2.17), None for others"""
    m = re.match(r"manylinux_(\d+)_(\d+)_", tag or "")
    return (int(m.group(1)), int(m.group(2))) if m else (2, 17) if (tag or "").startswith("manylinux2014") else None


def needed(path):
    """the DT_NEEDED libraries of an ELF file (CalledProcessError when readelf cannot read it)"""
    out = subprocess.run(["readelf", "-W", "-d", str(path)], capture_output=True, text=True, check=True).stdout
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^]]+)\]", out)


def check(wheel, plat, needs, left_out):
    """failures (list of str) of the manylinux check of wheel for plat (manylinux_<x>_<y>_x86_64). needs: {name in the
    wheel: its DT_NEEDED} of its ELF files (modules and vendored libraries); left_out: the GPU runtime libraries the
    build excludes"""
    g = glibc(plat)
    if g is None:
        return [f"unknown platform tag {plat}"]
    canon = f"manylinux_{g[0]}_{g[1]}"
    policies = json.loads(importlib.resources.files("auditwheel.policy").joinpath("manylinux-policy.json").read_text())
    policy = next((p for p in policies if canon == p["name"] or canon in p["aliases"]), None)
    if policy is None:
        return [f"auditwheel has no policy {canon}"]
    bad = []
    allowed = set(policy["lib_whitelist"]) | {n.rsplit("/", 1)[-1] for n in needs} | set(left_out) | {LOADER}
    for n, libs in needs.items():
        foreign = [x for x in libs if x not in allowed]
        if foreign:
            bad.append(f"{n} needs {foreign}: not in the {plat} policy, not vendored")
    show = subprocess.run([sys.executable, "-m", "auditwheel", "show", "--json", str(wheel)], capture_output=True,
                          text=True)
    try:
        r = json.loads(show.stdout)
    except ValueError:
        return bad + [f"auditwheel show --json failed: {show.stdout[-400:]}{show.stderr[-400:]}"]

    def fits(tag):
        return glibc(tag) is not None and glibc(tag) <= g

    if fits(r["overall_tag"]):
        return bad
    other = [k for k in ("pyfpe", "ucs2", "unsupported_isa") if r.get(k)]
    blacklisted = r["policy_upgrades"].get(f"{canon}_x86_64", {}).get("blacklisted_symbols")
    if r["overall_tag"] != "linux_x86_64" or not fits(r["sym_tag"]) or other or blacklisted:
        bad.append(f"auditwheel: not {plat}: tag {r['overall_tag']}, symbol versions {r['sym_tag']}, "
                   f"failed {other}, blacklisted symbols {blacklisted}")
    return bad

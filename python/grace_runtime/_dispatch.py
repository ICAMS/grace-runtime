# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Which ISA build of the extension to load: ``_core_x86_64_v4`` (AVX-512) or ``_core_x86_64_v3`` (AVX2/FMA), or
``_core_native`` when the package was built from source with ``-DGRACE_RUNTIME_ISAS=native``. ``GRACE_RUNTIME_CPU=v3|v4|native``
forces one. Only one is loaded per process."""
import importlib.util
import os

_V3 = ("avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe")       # and lzcnt, listed as "abm" in /proc/cpuinfo
_V4 = _V3 + ("avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl")
_REQUIRES = {"x86_64_v3": _V3, "x86_64_v4": _V4, "native": ()}
_FORCE = {"v3": "x86_64_v3", "v4": "x86_64_v4", "native": "native"}


def cpu_flags(path="/proc/cpuinfo"):
    """CPU flags as the kernel reports them (it hides those the OS does not enable); empty if unknown."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def available(package=__package__):
    """ISA builds importable as ``<package>._core_<isa>`` (find_spec: also right for editable installs, where the
    extensions and the Python files live in different directories)."""
    return {isa for isa in _REQUIRES if importlib.util.find_spec(f"{package}._core_{isa}") is not None}


def _missing(isa, flags):
    miss = [f for f in _REQUIRES[isa] if f not in flags]
    if isa != "native" and not ({"abm", "lzcnt"} & flags):
        miss.append("lzcnt")
    return miss


def select(flags, forced, avail):
    if forced:
        isa = _FORCE.get(forced.strip().lower())
        if isa is None:
            raise ImportError(f"GRACE_RUNTIME_CPU={forced!r}: expected v3, v4 or native")
        if isa not in avail:
            raise ImportError(f"GRACE_RUNTIME_CPU={forced}: this installation has no {isa} module "
                              f"(it has: {', '.join(sorted(avail)) or 'none'})")
        miss = _missing(isa, flags) if flags else []
        if miss:
            raise ImportError(f"GRACE_RUNTIME_CPU={forced}: this CPU lacks {', '.join(miss)}")
        return isa
    if not avail:
        raise ImportError("grace-runtime: no compiled extension (_core_x86_64_v3 / _v4 / _native) found; "
                          "the installation is broken")
    if "native" in avail:
        return "native"
    if not flags and "x86_64_v3" in avail:     # flags unknown: the most compatible build
        return "x86_64_v3"
    for isa in ("x86_64_v4", "x86_64_v3"):
        if isa in avail and not _missing(isa, flags):
            return isa
    raise ImportError("grace-runtime needs an x86-64-v3 CPU (AVX2, FMA, BMI2); this CPU lacks "
                      + ", ".join(_missing("x86_64_v3", flags)))

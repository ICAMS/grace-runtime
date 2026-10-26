# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""NVIDIA GPU backend loader. The GPU modules (_kk_sm<major><minor>, one per architecture) come from a backend plug-in,
the distribution grace-runtime-cuda12 (entry point group grace_runtime.backends, integer ABI equal to ABI here, no
PLATFORM attribute or PLATFORM = "cuda"), or, in development trees, from grace_runtime itself; the module is chosen
from the device's compute capability (CUDA driver API through ctypes, so no CUDA Python package is needed). The
process-wide GPU state (one GPU per process), device="auto" / "gpu" and the notices are vendor-neutral
(grace_runtime._gpu; re-exported here).

A GPU without a module of its own architecture family (same major compute capability) runs on the CPU (device="auto")
or is an error (device="cuda"). With GRACE_RUNTIME_ALLOW_PTX=1 it runs the PTX of the newest older module instead, which
the driver compiles at the first call (JIT; cached in ~/.nv/ComputeCache): same results, kernels tuned for the older
architecture. GRACE_RUNTIME_STRICT=1 makes every fallback (PTX, CPU) an error, for benchmarks. Fallbacks are announced
by notice(): a banner on stderr once per process (GRACE_RUNTIME_QUIET=1 silences it) and a RuntimeWarning."""
import ctypes
import importlib
import os
import re
from pathlib import Path

from . import _gpu
from ._gpu import ABI, NoGPU, _flag, _state, notice, strict, unsupported    # noqa: F401 (re-exported)

PTX_DRIVER = 12090    # the wheels' modules are built with CUDA 12.9 (scripts/build_cuda_wheels.sh): JIT of their PTX
#                       needs a driver for CUDA >= 12.9


def allow_ptx() -> bool:
    """GRACE_RUNTIME_ALLOW_PTX=1 (and not strict): a GPU without a module of its family runs an older module's PTX"""
    return _flag("GRACE_RUNTIME_ALLOW_PTX") and not strict()


def parse_device(device: str) -> int:
    """"cuda" -> 0 (first visible device), "cuda:N" -> N."""
    m = re.fullmatch(r"cuda(?::(\d+))?", device.strip())
    if not m:
        raise ValueError(f"unknown device {device!r}; use 'cpu', 'cuda' or 'cuda:N'")
    return int(m.group(1) or 0)


def compute_capability(index: int) -> tuple:
    """(major, minor) of CUDA device `index` (after CUDA_VISIBLE_DEVICES), through libcuda.so.1."""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        raise NoGPU(f"no NVIDIA driver (libcuda.so.1): {e}") from None
    if cu.cuInit(0) != 0:
        raise NoGPU("cuInit failed: no usable CUDA device")
    v = ctypes.c_int()
    cu.cuDriverGetVersion(ctypes.byref(v))
    _state["driver"] = v.value
    if v.value < 12000:    # CUDA 12 minor-version compatibility starts at driver 525
        raise ImportError(f"the NVIDIA driver supports CUDA {v.value // 1000}.{v.value % 1000 // 10}; "
                          f"grace-runtime-cuda12 needs CUDA >= 12.0 (driver >= 525): update the driver or use "
                          f"device='cpu'")
    n = ctypes.c_int()
    cu.cuDeviceGetCount(ctypes.byref(n))
    if not 0 <= index < n.value:
        raise ValueError(f"cuda:{index} does not exist ({n.value} visible CUDA device(s))")
    dev, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), index)
    cu.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    cu.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)    # ..._MINOR
    return major.value, minor.value


def _preload_cublas():
    """libcublas of the nvidia-cublas-cu12 wheel (when installed) into the process, so that the modules' run-time
    dlopen("libcublas.so.12") finds it (UQ projections); without it they use hand-written kernels"""
    off = os.environ.get("GRACE_RUNTIME_NO_CUBLAS", "")
    if off and off != "0":
        return
    try:
        import nvidia.cublas
    except ImportError:
        return
    for d in nvidia.cublas.__path__:
        for name in ("libcublasLt.so.12", "libcublas.so.12"):
            p = Path(d) / "lib" / name
            if p.exists():
                try:
                    ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _archs_in(directory):
    return sorted({m.group(1) for p in Path(directory).iterdir() if (m := re.match(r"_kk_(sm\d+)\.", p.name))})


def installed_archs():
    """GPU modules in grace_runtime itself (development builds)"""
    return _archs_in(Path(__file__).parent)


def plugins() -> list:
    """the NVIDIA backend packages of the entry point group grace_runtime.backends (_gpu.plugins); an ABI other than
    ours is an ImportError naming both versions"""
    return _gpu.plugins("cuda")


def available() -> dict:
    """{arch: module name}: the plug-ins' modules, then grace_runtime's own (development builds)"""
    found = {}
    for pkg in plugins():
        for arch in pkg.archs():
            found.setdefault(arch, f"{pkg.__name__}._kk_{arch}")
    for arch in installed_archs():
        found.setdefault(arch, f"grace_runtime._kk_{arch}")
    return found


def install_hint() -> str:
    """the command that installs the NVIDIA GPU backend matching this grace-runtime"""
    from . import __version__
    return f"pip install 'grace-runtime-cuda12{'==' + __version__ if 'unknown' not in __version__ else ''}'"


def arch_for(major: int, minor: int, archs) -> str | None:
    """the module for compute capability major.minor: same major, the highest minor not above the device's (SASS of
    sm_XY runs on sm_XZ, Z >= Y); None when there is none"""
    best = None
    for a in archs:
        m = re.fullmatch(r"sm(\d+)(\d)", a)
        if m and int(m.group(1)) == major and int(m.group(2)) <= minor:
            if best is None or int(m.group(2)) > int(best[-1]):
                best = a
    return best


def ptx_arch_for(major: int, minor: int, archs) -> str | None:
    """the module whose PTX the driver can compile for compute capability major.minor when no module of its family is
    there: the newest one of an older major (the PTX of sm_XY runs on any newer device); None when there is none"""
    older = [(int(m.group(1)), int(m.group(2)), a) for a in archs
             if (m := re.fullmatch(r"sm(\d+)(\d)", a)) and int(m.group(1)) < major]
    return max(older)[2] if older else None


def _ptx_driver(index: int, dev: str, arch: str):
    """ImportError unless the driver can compile the modules' PTX (else Kokkos aborts the process at its start)"""
    d = _state["driver"] or 0
    if d < PTX_DRIVER:
        raise ImportError(f"cuda:{index} ({dev}) has no native grace-runtime module, and running the {arch} module's PTX "
                          f"needs a driver for CUDA >= {PTX_DRIVER // 1000}.{PTX_DRIVER % 1000 // 10}; this one "
                          f"supports {d // 1000}.{d % 1000 // 10}: update the driver")


def select(index: int) -> tuple:
    """(arch, ptx) of the GPU module for CUDA device `index`: the native module of its family, or, with
    GRACE_RUNTIME_ALLOW_PTX=1, the PTX of an older one (ptx True). ImportError: no backend installed, ABI mismatch, no
    module for the device's architecture (naming the installed ones and the ways out), a driver too old for the PTX.
    GRACE_RUNTIME_CUDA_ARCH forces an architecture (tests; an older family counts as PTX)."""
    major, minor = compute_capability(index)
    avail = available()
    if not avail:
        raise ImportError("no GPU module installed: pip install grace-runtime-cuda12 (NVIDIA GPUs, CUDA 12); "
                          "use device='cpu' meanwhile")
    dev = f"sm{major}{minor}"
    forced = os.environ.get("GRACE_RUNTIME_CUDA_ARCH")
    want = forced or arch_for(major, minor, avail)
    if want in avail:
        m = re.fullmatch(r"sm(\d+)\d", want)
        if m and int(m.group(1)) < major:    # a forced older family runs as PTX
            _ptx_driver(index, dev, want)
            return want, True
        return want, False
    ptx = None if forced else ptx_arch_for(major, minor, avail)
    if ptx and allow_ptx():
        _ptx_driver(index, dev, ptx)
        return ptx, True
    ways = [f"build a native module: GRACE_KK_ARCHS={dev} bash scripts/build_cuda_wheels.sh <dir> in the grace-runtime "
            f"source tree (README: Build and install a Kokkos CUDA wheel)"]
    if ptx and not strict():
        ways.insert(0, f"set GRACE_RUNTIME_ALLOW_PTX=1 to run the {ptx} module's PTX, compiled by the driver at the "
                       f"first call (can take minutes, cached afterwards; a few % slower)")
    raise ImportError(f"grace-runtime has no GPU module for {want or dev} (cuda:{index}); installed: "
                      f"{', '.join(sorted(avail))}. Ways out: " + "; or ".join(ways) + "; or use device='cpu'.")


def module_for(index: int):
    """The GPU module for CUDA device `index` (select())."""
    arch, _ = select(index)
    _preload_cublas()
    return importlib.import_module(available()[arch])


def start(index: int):
    """(the GPU module for CUDA device `index`, its state for _gpu: arch, ptx, cc); announces the PTX fallback. Kokkos is
    started by _gpu.ensure_init."""
    arch, ptx = select(index)
    if ptx and strict():
        raise ImportError(f"GRACE_RUNTIME_STRICT=1: {arch} on cuda:{index} would run as PTX (no native module)")
    major, minor = compute_capability(index)
    _preload_cublas()
    mod = importlib.import_module(available()[arch])
    if ptx:
        cache = os.environ.get("CUDA_CACHE_PATH") or "~/.nv/ComputeCache"
        notice(f"cuda:{index} is sm{major}{minor}, which has no native grace-runtime module: running the {arch} "
               f"module's PTX, compiled by the CUDA driver at the first call (can take minutes; cached in {cache}, "
               f"compiled again where that cache is not kept, e.g. in fresh containers). Same results; kernels are "
               f"tuned for {arch}, expect a few % lower speed. For benchmarks build a native module: "
               f"GRACE_KK_ARCHS=sm{major}{minor} bash scripts/build_cuda_wheels.sh <dir>.", stacklevel=5)
    return mod, {"arch": arch, "ptx": ptx, "cc": f"{major}.{minor}"}


def ensure_init(device: str = "cuda"):
    """Kokkos on CUDA `device` (first call), the GPU module; another device later is a ValueError (_gpu.ensure_init)."""
    return _gpu.ensure_init(f"cuda:{parse_device(device)}")


def info() -> dict | None:
    """the GPU module in use: {"device": "cuda:N", "platform": "cuda", "arch": "sm80", "cc": "8.6", "native": bool};
    None before the first GPU model (_gpu.info)"""
    return _gpu.info()


def probe():
    """the NVIDIA part of device="auto": ("cuda", None) when a module for cuda:0 is installed, ("cpu", None) without an
    NVIDIA GPU, ("cpu", why) when one is there but cannot be used"""
    try:
        module_for(0)
    except NoGPU:
        return "cpu", None    # no driver / device: the CPU, silently
    except ValueError:
        return "cpu", None
    except ImportError as e:
        return "cpu", str(e)    # a GPU, but no module for it / no plug-in / ABI mismatch
    return "cuda", None


def auto_device():
    """device="auto" among NVIDIA GPUs only: ("cuda[:N]" | "cpu", reason). GRACE_RUNTIME_DEVICE when set; else the GPU
    in use, or cuda:0 when a module for it is installed; reason (a GPU is there but cannot be used) is None when there
    is simply no GPU. The calculators use _gpu.auto_device (NVIDIA, then AMD)."""
    env = os.environ.get("GRACE_RUNTIME_DEVICE", "").strip()
    if env and env != "auto":
        return env, None
    if _state["module"] is not None:
        return _gpu.in_use(), None
    return probe()

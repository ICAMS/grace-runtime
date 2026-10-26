# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""AMD GPU backend loader (ROCm / HIP). The GPU modules (_kk_gfx<target>, one per GPU architecture, e.g. _kk_gfx90a for
MI210/MI250X, _kk_gfx942 for MI300A/X) come from a backend plug-in (entry point group grace_runtime.backends,
PLATFORM = "rocm", integer ABI equal to grace_runtime._gpu.ABI, ROCM_MAJOR = the soname major of libamdhip64 its
modules link), or, in development trees, from grace_runtime itself. The process-wide GPU state, device="auto" / "gpu"
and the notices are vendor-neutral (grace_runtime._gpu).

The device is probed without loading a module: /dev/kfd and the KFD topology in sysfs tell whether an AMD GPU is there
at all (no library loaded on CPU nodes); the HIP runtime (libamdhip64.so.<major>, through ctypes: the same library
instance the modules and torch use) gives the visible devices, and each device's PCI address finds its KFD node, whose
gfx_target_version is the GPU architecture. AMD code objects are exact-ISA: a module runs only on its own target (no
PTX-like fallback), so the module is chosen by exact match of the architecture (gfx90a module for a gfx90a device,
whatever its xnack / sramecc mode). GRACE_RUNTIME_ROCM_ARCH forces one (tests)."""
import ctypes
import importlib
import os
import re
import sys
from pathlib import Path

from . import _gpu
from ._gpu import NoGPU, notice, strict    # noqa: F401 (re-exported)

# PyPI name of the ROCm plug-in per HIP runtime major (as grace-runtime-cuda12): grace-runtime-rocm6 (import package
# grace_runtime_rocm6, entry point rocm6), later grace-runtime-rocm7. The loader finds plug-ins by entry point.
PLUGIN = "grace-runtime-rocm{major}"
ROCM_MAJORS = (6, 7)    # HIP runtime sonames tried for development modules (a plug-in names its ROCM_MAJOR)
UNTESTED = {"gfx942"}   # modules built but not yet run on such a GPU: grace-runtime info says so
KFD_DEV = "/dev/kfd"
KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"
_hip = {}    # soname -> (CDLL, path) of the loaded HIP runtime


def parse_device(device: str) -> int:
    """"rocm" / "hip" -> 0 (first visible device), "rocm:N" / "hip:N" -> N."""
    m = re.fullmatch(r"(?:rocm|hip)(?::(\d+))?", device.strip())
    if not m:
        raise ValueError(f"unknown device {device!r}; use 'cpu', 'rocm' or 'rocm:N'")
    return int(m.group(1) or 0)


def gfx_name(target_version: int) -> str:
    """KFD gfx_target_version (major * 10000 + minor * 100 + stepping) -> "gfx<major><minor hex><stepping hex>": 90010 ->
    gfx90a, 90402 -> gfx942, 110001 -> gfx1101"""
    major, minor, step = target_version // 10000, target_version // 100 % 100, target_version % 100
    return f"gfx{major}{minor:x}{step:x}"


def kfd_nodes(root: str | None = None) -> list:
    """the GPU nodes of the KFD topology: [{"arch": "gfx90a", "domain": int, "location_id": int}] (CPU nodes, with
    gfx_target_version 0, left out); [] when unreadable"""
    out = []
    try:
        nodes = sorted(Path(root or KFD_NODES).iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0)
    except OSError:
        return out
    for n in nodes:
        try:
            props = dict(line.split()[:2] for line in (n / "properties").read_text().splitlines() if line.strip())
        except (OSError, ValueError):
            continue
        v = int(props.get("gfx_target_version", 0))
        if v:
            out.append({"arch": gfx_name(v), "domain": int(props.get("domain", 0)),
                        "location_id": int(props.get("location_id", 0))})
    return out


def kfd_present() -> bool:
    """an AMD GPU driver with at least one GPU (no library loaded): /dev/kfd and a KFD GPU node"""
    return os.path.exists(KFD_DEV) and bool(kfd_nodes())


def _override_arch():
    """HSA_OVERRIDE_GFX_VERSION (e.g. 11.0.0, a consumer-GPU workaround): the target the HIP runtime reports instead"""
    v = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "").strip()
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", v)
    return gfx_name(int(m.group(1)) * 10000 + int(m.group(2)) * 100 + int(m.group(3))) if m else None


def rocm_majors() -> tuple:
    """HIP runtime majors to try: the plug-ins' ROCM_MAJOR, else all supported"""
    try:
        majors = tuple(sorted({int(getattr(p, "ROCM_MAJOR", 0)) for p in plugins()} - {0}, reverse=True))
    except ImportError:
        majors = ()
    return majors or tuple(sorted(ROCM_MAJORS, reverse=True))


def hip_runtime():
    """(CDLL, path, major) of the HIP runtime libamdhip64.so.<major> (the copy already in the process when torch or
    a module loaded one); ImportError when none loads"""
    errors = []
    for major in rocm_majors():
        name = f"libamdhip64.so.{major}"
        if name not in _hip:
            lib = None
            # the dynamic loader's search, then ROCm installations it does not know (no ld.so.conf entry, no module):
            # loaded by path, the library also satisfies the modules' NEEDED entry
            for cand in [name] + [os.path.join(d, "lib", name) for d in (os.environ.get("ROCM_PATH"),
                                                                       os.environ.get("HIP_PATH"), "/opt/rocm") if d]:
                try:
                    lib = ctypes.CDLL(cand)
                    break
                except OSError as e:
                    errors.append(str(e))
            if lib is None:
                continue
            _hip[name] = (lib, _lib_path(name))
        lib, path = _hip[name]
        return lib, path, major
    want = " or ".join(f"libamdhip64.so.{m}" for m in rocm_majors())
    raise ImportError(f"an AMD GPU is there, but no ROCm runtime ({want}) loads: install ROCm "
                      f"{' or '.join(map(str, rocm_majors()))} (or load its module, e.g. module load rocm), or use "
                      f"device='cpu' ({'; '.join(errors)})")


def _lib_path(name) -> str:
    """file of a loaded library (from /proc/self/maps), else its soname"""
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.split(None, 5)[-1].strip()
                if os.path.basename(path).startswith(name):    # libamdhip64.so.6 -> libamdhip64.so.6.3.60304
                    return path
    except OSError:
        pass
    return name


def probe(index: int = 0) -> dict:
    """the AMD GPU `index` (after ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES, which HIP also
    reads): {"arch", "name", "total_mem", "count", "runtime", "driver", "lib", "major"}. NoGPU without an AMD GPU (no
    /dev/kfd, no KFD GPU node, none visible); ImportError without a HIP runtime; ValueError for a missing index."""
    if not kfd_present():
        raise NoGPU("no AMD GPU (/dev/kfd, KFD topology)")
    hip, path, major = hip_runtime()
    n = ctypes.c_int()
    if hip.hipGetDeviceCount(ctypes.byref(n)) != 0 or n.value == 0:
        raise NoGPU("no visible AMD GPU (HIP: none; ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES "
                    "may hide them)")
    if not 0 <= index < n.value:
        raise ValueError(f"rocm:{index} does not exist ({n.value} visible AMD GPU(s))")
    rt, drv = ctypes.c_int(), ctypes.c_int()
    hip.hipRuntimeGetVersion(ctypes.byref(rt))
    hip.hipDriverGetVersion(ctypes.byref(drv))
    name = ctypes.create_string_buffer(256)
    hip.hipDeviceGetName(name, 256, index)
    mem = ctypes.c_size_t()
    hip.hipDeviceTotalMem(ctypes.byref(mem), index)
    bus = ctypes.create_string_buffer(64)
    arch = None
    nodes = kfd_nodes()
    if hip.hipDeviceGetPCIBusId(bus, 64, index) == 0:
        m = re.match(r"([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-7])", bus.value.decode())
        if m:
            dom, b, d, f = (int(x, 16) for x in m.groups())
            loc = (b << 8) | (d << 3) | f
            arch = next((x["arch"] for x in nodes if x["domain"] == dom and x["location_id"] == loc), None)
    if arch is None and len({x["arch"] for x in nodes}) == 1:    # one GPU type on the node
        arch = nodes[0]["arch"]
    if arch is None:    # the way out the error below names (select() checks that its module is installed)
        arch = os.environ.get("GRACE_RUNTIME_ROCM_ARCH", "").strip() or None
    if arch is None:
        raise ImportError(f"rocm:{index}: GPU architecture unknown (PCI {bus.value.decode() or '?'} not in the KFD "
                          f"topology); set GRACE_RUNTIME_ROCM_ARCH")
    arch = _override_arch() or arch
    return {"arch": arch, "name": name.value.decode(errors="replace"), "total_mem": mem.value, "count": n.value,
            "runtime": rt.value, "driver": drv.value, "lib": path, "major": major}


def version_str(v: int) -> str:
    """HIP version number (major * 10000000 + minor * 100000 + patch) -> "6.3.42134" """
    return f"{v // 10000000}.{v // 100000 % 100}.{v % 100000}"


def _archs_in(directory):
    return sorted({m.group(1) for p in Path(directory).iterdir() if (m := re.match(r"_kk_(gfx[0-9a-f]+)\.", p.name))})


def installed_archs():
    """AMD GPU modules in grace_runtime itself (development builds)"""
    return _archs_in(Path(__file__).parent)


def plugins() -> list:
    """the AMD backend packages of the entry point group grace_runtime.backends (_gpu.plugins)"""
    return _gpu.plugins("rocm")


def available() -> dict:
    """{arch: module name}: the plug-ins' modules, then grace_runtime's own (development builds)"""
    found = {}
    for pkg in plugins():
        for arch in pkg.archs():
            found.setdefault(arch, f"{pkg.__name__}._kk_{arch}")
    for arch in installed_archs():
        found.setdefault(arch, f"grace_runtime._kk_{arch}")
    return found


def install_hint(major: int = 6) -> str:
    """the command that installs the AMD GPU backend for ROCm `major` matching this grace-runtime"""
    from . import __version__
    pin = f"=={__version__}" if "unknown" not in __version__ else ""
    return f"pip install '{PLUGIN.format(major=major)}{pin}'"


def _check_torch(major: int):
    """torch built for another ROCm major in this process: two HIP runtimes, which does not work"""
    torch = sys.modules.get("torch")
    hip = getattr(getattr(torch, "version", None), "hip", None) if torch is not None else None
    if hip:
        tmajor = int(str(hip).split(".")[0])
        if tmajor != major:
            raise ImportError(f"torch in this process is built for ROCm {tmajor} (torch.version.hip {hip}), the "
                              f"grace-runtime AMD GPU modules for ROCm {major}: install the grace-runtime ROCm "
                              f"{tmajor} backend or a torch for ROCm {major}")


def select(index: int) -> str:
    """the arch of the GPU module for AMD device `index`: the module of exactly its architecture. ImportError: no
    runtime, no backend installed, ABI mismatch, no module for the device's architecture (naming the installed ones and
    the ways out). GRACE_RUNTIME_ROCM_ARCH forces an architecture (tests)."""
    p = probe(index)
    avail = available()
    if not avail:
        raise ImportError(f"no AMD GPU module installed for rocm:{index} ({p['arch']}): {install_hint(p['major'])} "
                          f"(AMD GPUs, ROCm {p['major']}); use device='cpu' meanwhile")
    forced = os.environ.get("GRACE_RUNTIME_ROCM_ARCH")
    want = forced or p["arch"]
    if want in avail:
        return want
    raise ImportError(f"grace-runtime has no GPU module for {want} (rocm:{index}, {p['name']}); installed: "
                      f"{', '.join(sorted(avail))}. AMD GPU code runs only on its own architecture: build a module "
                      f"for {want} from the grace-runtime sources (GRACE_KK_ARCHS={want}; README, AMD GPUs), or use "
                      f"device='cpu'.")


def module_for(index: int):
    """The GPU module for AMD device `index` (select())."""
    return start(index)[0]


def start(index: int):
    """(the GPU module for AMD device `index`, its state for _gpu: arch, ptx, cc); ImportError as select(), or when torch
    in this process uses another ROCm major. Kokkos is started by _gpu.ensure_init."""
    arch = select(index)
    p = probe(index)
    _check_torch(p["major"])
    return importlib.import_module(available()[arch]), {"arch": arch, "ptx": False, "cc": p["arch"]}


def ensure_init(device: str = "rocm"):
    """Kokkos on AMD `device` (first call), the GPU module; another device later is a ValueError (_gpu.ensure_init)."""
    return _gpu.ensure_init(f"rocm:{parse_device(device)}")

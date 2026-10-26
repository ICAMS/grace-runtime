# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""GPU device selection, vendor-neutral. Device strings: "cuda" / "cuda:N" (NVIDIA, grace_runtime._cuda), "rocm" /
"rocm:N" (AMD, grace_runtime._rocm; "hip" / "hip:N" are aliases), "gpu" / "gpu:N" (the first vendor with a usable GPU:
NVIDIA, then AMD; an error, never the CPU, when there is none), "auto" (as "gpu", but the CPU when no GPU is usable:
silently without a GPU, announced when one is there but cannot be used) and "cpu".

Owns the process-wide GPU state: one GPU module on one device per process (the runtime's design: its engines use Kokkos'
default execution space instance; Kokkos itself can drive several GPUs through explicit instances), initialised on the
first GPU model and finalised at interpreter exit. Fallbacks are announced by notice(): a banner on stderr once per
process (GRACE_RUNTIME_QUIET=1 silences it) and a RuntimeWarning. GRACE_RUNTIME_STRICT=1 makes every fallback an
error; device="auto" without a usable GPU is then an error too (device="cpu" runs on the CPU)."""
import atexit
import os
import re
import sys
import warnings

ABI = 5    # the C++ interface of the GPU modules (kk_api.h); a plug-in must carry the same
# the GPU module in use: platform "cuda" | "rocm", device index, module arch, ptx (CUDA only), cc (CUDA: "8.6"; ROCm:
# the gfx target), driver (CUDA driver version, set by _cuda.compute_capability)
_state = {"module": None, "platform": None, "device": None, "arch": None, "ptx": False, "cc": None, "driver": None}
_noticed = set()
PLATFORMS = ("cuda", "rocm")


def _flag(name) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


def strict() -> bool:
    """GRACE_RUNTIME_STRICT=1: a native GPU module or an error, never a PTX or CPU fallback"""
    return _flag("GRACE_RUNTIME_STRICT")


def notice(msg: str, stacklevel: int = 3):
    """A fallback the user must not miss: a banner on stderr once per process and message, not subject to warning
    filters (GRACE_RUNTIME_QUIET=1 silences it), then a RuntimeWarning (for warning filters and pytest.warns)."""
    if msg not in _noticed and not _flag("GRACE_RUNTIME_QUIET"):
        _noticed.add(msg)
        bar = "!" * 100
        print(f"{bar}\ngrace-runtime WARNING: {msg}\n{bar}", file=sys.stderr, flush=True)
    warnings.warn(msg, RuntimeWarning, stacklevel=stacklevel)


class NoGPU(ImportError):
    """no GPU of a vendor (no driver / runtime presence, no visible device): device="auto" goes on silently"""


def parse_device(device: str) -> tuple:
    """"cuda[:N]" -> ("cuda", N), "rocm[:N]" / "hip[:N]" -> ("rocm", N), "gpu[:N]" -> ("gpu", N); N defaults to 0"""
    m = re.fullmatch(r"(cuda|rocm|hip|gpu)(?::(\d+))?", device.strip())
    if not m:
        raise ValueError(f"unknown device {device!r}; use 'auto', 'cpu', 'gpu[:N]', 'cuda[:N]' (NVIDIA) or "
                         f"'rocm[:N]' (AMD)")
    platform = "rocm" if m.group(1) == "hip" else m.group(1)
    return platform, int(m.group(2) or 0)


def _vendor(platform):
    from . import _cuda, _rocm
    return {"cuda": _cuda, "rocm": _rocm}[platform]


def plugins(platform: str) -> list:
    """the backend packages of the entry point group grace_runtime.backends for `platform` (a package's PLATFORM
    attribute; without one it is "cuda", as the grace-runtime-cuda12 releases); an ABI other than ours is an
    ImportError naming both versions"""
    from importlib.metadata import entry_points
    from . import __version__
    out = []
    for ep in entry_points(group="grace_runtime.backends"):
        pkg = ep.load()
        if getattr(pkg, "PLATFORM", "cuda") != platform:
            continue
        abi = getattr(pkg, "ABI", None)
        if abi != ABI:
            name = ep.dist.name if ep.dist is not None else ep.value
            raise ImportError(f"{name} {getattr(pkg, '__version__', '?')} has GPU backend ABI {abi}, grace-runtime "
                              f"{__version__} needs ABI {ABI}: install matching versions, e.g. pip install "
                              f"'{name}=={__version__}'")
        out.append(pkg)
    return out


def _probe(platform: str, index: int):
    """(device string, None) when `platform` has a usable GPU `index` with a module, (None, None) without a GPU of that
    vendor, (None, why) when one is there but cannot be used"""
    try:
        _vendor(platform).module_for(index)
    except NoGPU:
        return None, None
    except ValueError as e:    # no such index: silent for the first device (as before), a reason for "gpu:N"
        return None, (str(e) if index else None)
    except ImportError as e:    # no plug-in / no module for the arch / no runtime / ABI mismatch
        return None, str(e)
    return f"{platform}:{index}" if index else platform, None


def in_use() -> str | None:
    """the GPU in use, "cuda:0" / "rocm:1", or None before the first GPU model"""
    return None if _state["module"] is None else f"{_state['platform']}:{_state['device']}"


def resolve_gpu(index: int = 0) -> str:
    """device="gpu[:N]": the GPU in use, else "cuda[:N]" when NVIDIA GPU N is usable, else "rocm[:N]"; ImportError
    (naming the reasons) when neither is"""
    if _state["module"] is not None:
        return f"{_state['platform']}:{index}"    # ensure_init refuses another index
    why = []
    for platform in PLATFORMS:
        dev, w = _probe(platform, index)
        if dev:
            return dev
        if w:
            why.append(w)
    raise ImportError(f"device='gpu{':' + str(index) if index else ''}': no usable GPU"
                      + (": " + "; ".join(why) if why else " (no NVIDIA or AMD GPU found)")
                      + ". Use device='auto' to fall back to the CPU, or device='cpu'.")


def env_device() -> str:
    """GRACE_RUNTIME_DEVICE ("" when unset): the device="auto" choice made by the environment"""
    return os.environ.get("GRACE_RUNTIME_DEVICE", "").strip()


def auto_device():
    """device="auto": (device, reason). GRACE_RUNTIME_DEVICE when set; else the GPU in use, else the first usable GPU
    (NVIDIA, then AMD), else "cpu"; reason (a GPU is there but cannot be used) is None when there is simply no GPU."""
    env = env_device()
    if env and env != "auto":
        return env, None
    if _state["module"] is not None:
        return in_use(), None
    why = []
    for platform in PLATFORMS:
        dev, w = _probe(platform, 0)
        if dev:
            return dev, None
        if w:
            why.append(w)
    return "cpu", ("; ".join(why) or None)


def ensure_init(device: str):
    """the GPU module for `device` ("cuda[:N]", "rocm[:N]", "hip[:N]", "gpu[:N]"), Kokkos started on it at the first
    call; another device later is a ValueError (one GPU per process). ImportError: no module / plug-in / runtime."""
    platform, index = parse_device(device)
    if platform == "gpu":
        platform, index = parse_device(resolve_gpu(index))
    if _state["module"] is not None:
        if (platform, index) != (_state["platform"], _state["device"]):
            raise ValueError(f"one GPU per process: already using {in_use()}")
        return _state["module"]
    mod, info = _vendor(platform).start(index)
    if not mod.initialized():
        mod.initialize(index)
        atexit.register(mod.finalize)
    _state.update(module=mod, platform=platform, device=index, **info)
    return mod


def info() -> dict | None:
    """the GPU module in use: {"device": "rocm:0", "platform": "rocm", "arch": "gfx90a", "cc": ..., "native": bool};
    None before the first GPU model. cc: CUDA compute capability ("8.6") or the AMD gfx target of the device."""
    if _state["module"] is None:
        return None
    return {"device": in_use(), "platform": _state["platform"], "arch": _state["arch"], "cc": _state["cc"],
            "native": not _state["ptx"]}


_unsupported = {}    # model path -> why the fused GPU kernels refuse it


def unsupported(mod, kk_model, path, spline):
    """None when the fused GPU kernels take the model, else the reason; probed once per path (the engine of the first
    probe's spline is set up then, so it costs nothing later)"""
    if path not in _unsupported:
        try:
            mod.System(kk_model, [kk_model.element_names[0]], spline=spline)
            _unsupported[path] = None
        except ValueError as e:
            _unsupported[path] = str(e)
    return _unsupported[path]


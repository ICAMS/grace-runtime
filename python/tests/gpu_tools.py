# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""The GPU of the GPU tests (test_gpu_*.py and the GPU cases of other files): GRACE_TEST_GPU ("cuda", "rocm", "cuda:1",
...) when set, else the first usable GPU with a module (NVIDIA, then AMD), else None (the tests skip)."""
import os

from grace_runtime import _gpu


def _first_gpu():
    env = os.environ.get("GRACE_TEST_GPU", "").strip()
    if env:
        return env
    for platform in _gpu.PLATFORMS:
        dev, _ = _gpu._probe(platform, 0)
        if dev:
            return dev
    return None


GPU = _first_gpu()    # device string of the GPU tests: "cuda" / "rocm" (or GRACE_TEST_GPU), None without a GPU
PLATFORM = _gpu.parse_device(GPU)[0] if GPU else None    # "cuda" | "rocm" | None
if GPU:    # child processes of the tests: GPU = os.environ["GRACE_TEST_GPU"]
    os.environ["GRACE_TEST_GPU"] = GPU
NO_GPU = "no GPU with a grace-runtime module (GRACE_TEST_GPU selects one)"
# environment that makes the installed modules miss the device's architecture (simulated GPU without a module)
NO_MODULE_ENV = {"rocm": {"GRACE_RUNTIME_ROCM_ARCH": "gfx000"}}.get(PLATFORM, {"GRACE_RUNTIME_CUDA_ARCH": "sm00"})
NO_MODULE_ARCH = next(iter(NO_MODULE_ENV.values()))


def gpu_ok() -> bool:
    return GPU is not None


def gpu_count() -> int:
    """visible GPUs of the test platform"""
    if PLATFORM == "rocm":
        from grace_runtime import _rocm
        return _rocm.probe(0)["count"]
    import ctypes
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuInit(0)
    n = ctypes.c_int()
    cu.cuDeviceGetCount(ctypes.byref(n))
    return n.value


def gpu_total_gb() -> float:
    """memory of the test GPU (GiB), 0 without one"""
    if GPU is None:
        return 0.0
    index = _gpu.parse_device(GPU)[1]
    if PLATFORM == "rocm":
        from grace_runtime import _rocm
        return _rocm.probe(index)["total_mem"] / 2**30
    import ctypes
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuInit(0)
    dev, b = ctypes.c_int(), ctypes.c_size_t()
    cu.cuDeviceGet(ctypes.byref(dev), index)
    cu.cuDeviceTotalMem_v2(ctypes.byref(b), dev)
    return b.value / 2**30


def other_gpu(index: int = 1) -> str:
    """device string of another GPU of the test platform, e.g. "rocm:1\""""
    return f"{PLATFORM}:{index}"

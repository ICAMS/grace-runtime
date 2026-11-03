# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime._core: the extension built for this CPU (see ``_dispatch``), re-exported under one name."""
import functools
import importlib
import os
import sys

from . import _dispatch, registry

cpu_module = _dispatch.select(_dispatch.cpu_flags(), os.environ.get("GRACE_RUNTIME_CPU"), _dispatch.available())
_ext = importlib.import_module(f"._core_{cpu_module}", __package__)
globals().update({k: v for k, v in vars(_ext).items() if not k.startswith("__")})


def _openmp_libraries() -> list:
    """the OpenMP runtime files loaded in this process (macOS)"""
    import ctypes
    dyld = ctypes.CDLL(None)
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    names = (os.fsdecode(dyld._dyld_get_image_name(i) or b"") for i in range(dyld._dyld_image_count()))
    return sorted({os.path.realpath(n) for n in names if os.path.basename(n) in ("libomp.dylib", "libiomp5.dylib")})


def _kmp_true(value: str) -> bool:
    """libomp's __kmp_str_match_true: whether it reads `value` as true (KMP_DUPLICATE_LIB_OK)"""
    value = value.lower()

    def match(word, n):    # __kmp_str_match: equal up to the shorter end, and at least n characters (n = 0: all)
        i = len(os.path.commonprefix([word, value]))
        if i < min(len(word), len(value)):
            return False
        return i >= n if n else i == len(word) == len(value)
    return any(match(w, n) for w, n in (("true", 1), ("on", 2), ("1", 1), (".true.", 2), (".t.", 2), ("yes", 1),
                                        ("enable", 0)))


@functools.cache    # once it passed; a failure is checked again
def _check_openmp():
    """macOS, before our first OpenMP call: two OpenMP runtimes in one process (PyTorch from PyPI ships its own) abort it
    with "OMP: Error #15" when the second starts, unless KMP_DUPLICATE_LIB_OK is set (scikit-learn sets it); raise
    instead. A runtime loaded after ours has started aborts in its own library (e.g. import torch), out of our reach."""
    libs = _openmp_libraries()
    if len(libs) > 1 and not _kmp_true(os.environ.get("KMP_DUPLICATE_LIB_OK", "")):
        raise RuntimeError(f"two OpenMP runtimes in this process ({', '.join(libs)}): the process would abort with "
                           "'OMP: Error #15'. PyTorch from PyPI ships its own on macOS: install PyTorch from conda-forge "
                           "instead, which shares one runtime with grace-runtime; a source build in a conda environment: "
                           "build it with that environment activated (see the macOS section of the installation docs)")


def _after_openmp_check(fn):
    @functools.wraps(fn)
    def call(*args, **kwargs):
        _check_openmp()
        return fn(*args, **kwargs)
    return call


class Model(_ext.Model):
    """A loaded GRACE kokkos.npz or PACE .yaml model on the host, shared across calculators and batch calls.

    path: a model file (GRACE kokkos.npz or PACE pacemaker B-basis .yaml) or a registry name (downloaded once into the
        cache); a path with a directory, .npz or .yaml / .yml is never looked up in the registry. A PACE .yaml is
        converted to its C-tilde basis on the fly when loaded.
    spline: intervals of the radial tables (default 1024) on CPU and GPU; for PACE models also 0 (direct radial
        evaluation).
    store_max_gb: CPU limit in GB for keeping per-atom forward tensors of GRACE-2L / 3L (~64 KB/atom for
        GRACE-2L-OMAT-medium); above it the backward pass recomputes them (same results, ~10% slower, warns once).
        Default None: half the available host memory, within the memory limit of the job (e.g. Slurm --mem).
    asi: active-set file (pacemaker .asi) of a PACE .yaml, enabling the extrapolation grade gamma (uq=True).
    """

    def __init__(self, path, *args, asi=None, **kwargs):
        path = os.fspath(path)
        if asi is not None:
            asi = os.fspath(asi)
            if not path.endswith((".yaml", ".yml")):
                raise ValueError("asi= goes with a PACE .yaml")
        elif os.sep not in path and not path.endswith((".npz", ".yaml", ".yml")):
            path = registry.resolve(path)
        super().__init__(path, *args, **({} if asi is None else {"asi": asi}), **kwargs)


if sys.platform == "darwin":    # any call into the extension may start our OpenMP runtime: check before the first
    for _k, _v in vars(_ext).items():
        if callable(_v) and not _k.startswith("__") and not isinstance(_v, type):
            globals()[_k] = _after_openmp_check(_v)
    for _k, _v in vars(_ext.Model).items():
        if callable(_v) and not _k.startswith("__"):
            setattr(Model, _k, _after_openmp_check(_v))
    Model.__init__ = _after_openmp_check(Model.__init__)
    del _k, _v

# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime._core: the extension built for this CPU (see ``_dispatch``), re-exported under one name."""
import importlib
import os

from . import _dispatch, registry

cpu_module = _dispatch.select(_dispatch.cpu_flags(), os.environ.get("GRACE_RUNTIME_CPU"), _dispatch.available())
_ext = importlib.import_module(f"._core_{cpu_module}", __package__)
globals().update({k: v for k, v in vars(_ext).items() if not k.startswith("__")})


class Model(_ext.Model):
    """A loaded GRACE kokkos.npz (host). path: a file or a registry name (downloaded once into the cache); a path
    (with a directory or .npz) is never looked up in the registry. spline: intervals of the radial tables (default
    1024), on the CPU and the GPU. One Model can be shared by calculators and the batch functions, so the file is read
    once."""

    def __init__(self, path, *args, **kwargs):
        path = os.fspath(path)
        if os.sep not in path and not path.endswith(".npz"):
            path = registry.resolve(path)
        super().__init__(path, *args, **kwargs)

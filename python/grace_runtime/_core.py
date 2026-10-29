# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime._core: the extension built for this CPU (see ``_dispatch``), re-exported under one name."""
import importlib
import os

from . import _dispatch, registry

cpu_module = _dispatch.select(_dispatch.cpu_flags(), os.environ.get("GRACE_RUNTIME_CPU"), _dispatch.available())
_ext = importlib.import_module(f"._core_{cpu_module}", __package__)
globals().update({k: v for k, v in vars(_ext).items() if not k.startswith("__")})


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

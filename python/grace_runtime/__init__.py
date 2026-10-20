# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Native CPU inference of GRACE interatomic potentials (libgrace) with an ASE calculator."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("grace-runtime")
except PackageNotFoundError:    # source tree without an installed distribution
    __version__ = "0+unknown"

from ._core import Model, cpu_module
from .calculator import GRACERuntimeCalculator
from .batch import compute_batch, free_gpu_memory
from ._cuda import ABI as abi    # GPU backend interface version; grace-runtime-cuda12 carries the same

__all__ = ["Model", "GRACERuntimeCalculator", "compute_batch", "free_gpu_memory", "cpu_module"]

# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Fast CPU and GPU inference of GRACE and PACE interatomic potentials with ASE, batching and TorchSim."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("grace-runtime")
except PackageNotFoundError:    # source tree without an installed distribution
    __version__ = "0+unknown"

from ._core import Model, cpu_module
from .calculator import GRACERuntimeCalculator
from .batch import compute_batch, compute_batch_iter, free_gpu_memory
from .relax import relax_batch, relax_batch_iter
from .properties import elastic_batch, eos_batch
from .phonons import phonons_batch, phonopy_forces_batch
from .dispersion import D3
from ._cuda import ABI as abi    # GPU backend interface version; grace-runtime-cuda12 carries the same

__all__ = ["Model", "GRACERuntimeCalculator", "compute_batch", "compute_batch_iter", "relax_batch", "relax_batch_iter",
           "eos_batch", "elastic_batch", "phonons_batch", "phonopy_forces_batch", "free_gpu_memory", "cpu_module", "D3"]

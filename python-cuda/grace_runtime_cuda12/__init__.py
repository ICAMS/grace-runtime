# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""NVIDIA GPU backend of grace-runtime (CUDA 12): one module per GPU architecture (_kk_sm80, _kk_sm89, _kk_sm90), each
with its own static Kokkos; loaded by grace_runtime._cuda through the entry point group grace_runtime.backends."""
import re
from pathlib import Path

ABI = 5    # = grace_runtime._cuda.ABI (the C++ interface kk_api.h of the modules)

try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("grace-runtime-cuda12")
except PackageNotFoundError:    # source tree
    __version__ = "0+unknown"


def archs() -> list:
    """GPU architectures with a module in this package, e.g. ["sm80", "sm89", "sm90"]"""
    here = Path(__file__).parent
    return sorted({m.group(1) for p in here.iterdir() if (m := re.match(r"_kk_(sm\d+)\.", p.name))})

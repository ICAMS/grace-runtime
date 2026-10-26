# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""AMD GPU backend of grace-runtime (ROCm 6): one module per GPU architecture (_kk_gfx90a, _kk_gfx942), each with its
own static Kokkos, linked against the HIP runtime of the system ROCm 6 (libamdhip64.so.6, not bundled); loaded by
grace_runtime._rocm through the entry point group grace_runtime.backends."""
import re
from pathlib import Path

ABI = 5             # = grace_runtime._gpu.ABI (the C++ interface kk_api.h of the modules)
PLATFORM = "rocm"   # grace_runtime._gpu.plugins: an AMD backend (a plug-in without PLATFORM is CUDA)
ROCM_MAJOR = 6      # soname major of the HIP runtime the modules link (libamdhip64.so.6)

try:
    from importlib.metadata import PackageNotFoundError, version
    __version__ = version("grace-runtime-rocm6")
except PackageNotFoundError:    # source tree
    __version__ = "0+unknown"


def archs() -> list:
    """GPU architectures with a module in this package, e.g. ["gfx90a", "gfx942"]"""
    here = Path(__file__).parent
    return sorted({m.group(1) for p in here.iterdir() if (m := re.match(r"_kk_(gfx[0-9a-f]+)\.", p.name))})

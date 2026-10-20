# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime._core: the extension built for this CPU (see ``_dispatch``), re-exported under one name."""
import importlib
import os

from . import _dispatch

cpu_module = _dispatch.select(_dispatch.cpu_flags(), os.environ.get("GRACE_RUNTIME_CPU"), _dispatch.available())
_ext = importlib.import_module(f"._core_{cpu_module}", __package__)
globals().update({k: v for k, v in vars(_ext).items() if not k.startswith("__")})

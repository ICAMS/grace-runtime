# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""download_test_models.py NAME...: registry models into $GRACE_CACHE from the source tree, without building or
installing grace_runtime (CI ctest)."""
import importlib.util
import json
import os
import sys
import types

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "grace_runtime")
pkg = types.ModuleType("gn_src")    # registry.py imports __version__ from its package
pkg.__path__ = [here]
pkg.__version__ = "source tree (CI)"
sys.modules["gn_src"] = pkg
spec = importlib.util.spec_from_file_location("gn_src.registry", os.path.join(here, "registry.py"))
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)
with open(os.path.join(here, "models.json")) as f:
    entries = json.load(f)
for name in sys.argv[1:]:
    print(registry.resolve(name, entries=entries))

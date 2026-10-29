# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
import os
import re

import pytest

from model_tools import strip_uq

HERE = os.path.dirname(os.path.abspath(__file__))
# device="auto" (the default) would take a GPU when there is one: the CPU tests stay on the CPU, the GPU tests
# pass device="cuda" (test_auto_device.py clears this)
os.environ.setdefault("GRACE_RUNTIME_DEVICE", "cpu")


TEST_MODELS = {"2l": ("GRACE_TEST_M2", "GRACE-2L-SMAX-OMAT-medium"), "1l": ("GRACE_TEST_M1", "GRACE-1L-OMAT-large-base"),
               "3l": ("GRACE_TEST_M3", "GRACE-3L-OMAT-large")}


@pytest.fixture(scope="session")
def models():
    """The fixture references (tests/fixtures/ref) are LAMMPS runs of these registry models; $GRACE_TEST_M1/M2 override,
    else they are resolved (downloaded if needed) in the GRACE cache."""
    from grace_runtime import registry
    return {k: os.environ.get(env) or registry.resolve(name) for k, (env, name) in TEST_MODELS.items()}


@pytest.fixture(scope="session")
def fixtures():
    return os.path.normpath(os.path.join(HERE, "..", "..", "tests", "fixtures"))


@pytest.fixture(scope="session")
def model_2l_nouq(models, tmp_path_factory):
    return strip_uq(models["2l"], tmp_path_factory.mktemp("nouq") / "kokkos.npz")


@pytest.fixture(scope="module", autouse=True)
def _free_gpu_after_module():
    """GPU engines stay cached until free_gpu_memory(): without this, the engines of all earlier test files add up in
    one pytest process and later GPU tests (and their child processes) run out of memory on 8 GB cards."""
    yield
    from grace_runtime import free_gpu_memory
    free_gpu_memory()



# Release gates set GRACE_TEST_ALLOW_SKIP to a regular expression: a skip whose reason does not match it fails the run
# (listed at the end), so a missing model, GPU module or package cannot pass a gate as a skip.
ALLOW_SKIP = os.environ.get("GRACE_TEST_ALLOW_SKIP")
_not_allowed = []


def _gate(rep):
    if ALLOW_SKIP is None or not rep.skipped or hasattr(rep, "wasxfail"):
        return
    reason = rep.longrepr[2] if isinstance(rep.longrepr, tuple) else str(rep.longrepr)
    if not re.search(ALLOW_SKIP, reason):
        _not_allowed.append(f"{rep.nodeid}: {reason}")


def pytest_runtest_logreport(report):
    _gate(report)


def pytest_collectreport(report):    # module-level skips (importorskip)
    _gate(report)


def pytest_terminal_summary(terminalreporter):
    if _not_allowed:
        terminalreporter.section(f"skips not allowed by GRACE_TEST_ALLOW_SKIP={ALLOW_SKIP!r}")
        for line in _not_allowed:
            terminalreporter.line(line)


def pytest_sessionfinish(session):
    if _not_allowed and session.exitstatus == 0:
        session.exitstatus = 1

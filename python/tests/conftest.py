# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
import os

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

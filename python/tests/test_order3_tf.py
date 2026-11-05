# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""GRACE-1L/2L models with max_order 3 (no AAAA / BBBB stage: the loader adds a dead one) against tensorpotential's
TPCalculator on their SavedModels (fp64), on the fixture cells (tests/fixtures/ref_tf/order3_tf.npz; random weights,
tests/fixtures/{legacy,grace}_{1l,2l}_order3.npz). legacy_*: the layout of the GRACE-OFF models. Comparison and
thresholds of test_ref_tf.py."""
import os

import numpy as np
import pytest

from test_ref_tf import CELLS, FX, check_against_ref

REF = os.path.join(FX, "ref_tf", "order3_tf.npz")
MODELS = ["legacy_1l_order3", "grace_1l_order3", "legacy_2l_order3", "grace_2l_order3"]


@pytest.fixture(scope="module")
def ref():
    return np.load(REF)


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("name", MODELS)
def test_matches_tf_fp64(ref, name, cell):
    check_against_ref(f"{FX}/{name}.npz", ref, name, cell)
    assert np.abs(ref[f"{name}/{cell}/forces"]).max() > 1e-2    # random weights, but not a vanishing model

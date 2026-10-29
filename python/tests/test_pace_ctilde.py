# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""The B -> C-tilde conversion of libgrace (_core.pace_arrays) against pyace (tests/fixtures/pace/ref_pace.npz).
The Clebsch-Gordan coefficients and the m-lists alone: tests/test_pace_yaml.cpp."""
import os

import numpy as np
import pytest

from pace_tools import FX, MODELS, arrays, model_path

MODELS = MODELS + ["al_cheblinear"]


@pytest.fixture(scope="module")
def ref():
    return np.load(os.path.join(FX, "ref_pace.npz"))


def rows(rank, mus, ns, ls, ms, ct):
    """{(rank, mus, ns, ls, ms): ctildes} in canonical form: the factors of each product sorted by (mu, n, l, m), equal
    products summed, zero rows (|c| < 1e-14) left out. pyace orders the factors of some products differently (when
    round-off of its Clebsch-Gordan coefficients lets it merge combinations); the products are the same."""
    out = {}
    for k in range(len(rank)):
        r = int(rank[k])
        f = sorted(zip(mus[k][:r], ns[k][:r], ls[k][:r], ms[k][:r]))
        key = (r, tuple(int(x[0]) for x in f), tuple(int(x[1]) for x in f), tuple(int(x[2]) for x in f),
               tuple(int(x[3]) for x in f))
        out[key] = out.get(key, 0.0) + np.asarray(ct[k])
    return {k: v for k, v in out.items() if np.abs(v).max() >= 1e-14}


@pytest.mark.parametrize("name", MODELS)
def test_ctilde_matches_pyace(name, ref, tmp_path):
    z = arrays(model_path(name, tmp_path))
    els = [e.decode() for e in z["element_names"]]
    assert list(ref[f"ctilde/{name}/elements"]) == els
    for mu in range(len(els)):
        g = lambda k: ref[f"ctilde/{name}/{mu}/{k}"]
        want = rows(g("rank"), g("mus"), g("ns"), g("ls"), g("ms"), g("ctildes"))
        c = lambda k: z[f"e{mu}_{k}"]
        got = rows(c("rank"), c("mus"), c("ns"), c("ls"), c("ms"), c("ctildes"))
        assert set(got) == set(want), (sorted(set(got) - set(want))[:3], sorted(set(want) - set(got))[:3])
        for key, w in want.items():
            np.testing.assert_allclose(got[key], w, rtol=1e-12, atol=1e-14, err_msg=str(key))


def test_rows_sorted_and_padded(tmp_path):
    z = arrays(model_path("carbon_zbl", tmp_path))
    rank, mus, ns, ls, ms = (z[f"e0_{k}"] for k in ("rank", "mus", "ns", "ls", "ms"))
    keys = [(int(rank[k]), *mus[k], *ns[k], *ls[k], *ms[k]) for k in range(len(rank))]
    assert keys == sorted(keys)
    for k in range(len(rank)):
        r = int(rank[k])
        assert not ns[k][r:].any() and not ls[k][r:].any() and not ms[k][r:].any()

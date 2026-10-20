# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""UQ (uqv6 gamma_only) through Model.compute and GRACERuntimeCalculator: API, errors, invariances.
Values against TPCalculator: test_vs_tp.py."""
import os

import numpy as np
import pytest
from ase.build import bulk

from grace_runtime import GRACERuntimeCalculator, Model

CACHE = os.path.expanduser(os.environ.get("GRACE_CACHE", "~/.cache/grace"))
UQ_MODELS = {"1l": f"{CACHE}/GRACE-1L-OMAT-large-base/kokkos.npz", "2l": f"{CACHE}/GRACE-2L-OMAT-medium-ft-E/kokkos.npz",
             "3l": f"{CACHE}/GRACE-3L-OMAT-large/kokkos.npz"}
KEYS = ("gamma", "atomic_sigma", "gmm_cluster", "total_sigma")


@pytest.fixture(scope="module")
def uqm():
    missing = [p for p in UQ_MODELS.values() if not os.path.exists(p)]
    if missing:
        pytest.skip(f"UQ models absent: {missing}")
    return {k: Model(v) for k, v in UQ_MODELS.items()}


def cell():
    at = bulk("W", "bcc", a=3.165, cubic=True).repeat((3, 3, 3))
    at.symbols[::5] = "Mo"
    at.rattle(0.05, seed=3)
    return at


@pytest.mark.parametrize("arch", ["1l", "2l", "3l"])
def test_uq_results(uqm, arch):
    at = cell()
    at.calc = GRACERuntimeCalculator(uqm[arch], num_threads=1, uq=True)    # 1 thread: bit-comparable energy
    e = at.get_potential_energy()
    r = at.calc.results
    n = len(at)
    for k in ("gamma", "atomic_sigma", "gmm_cluster"):
        assert r[k].shape == (n,) and r[k].dtype == np.float64
    assert np.all(r["gamma"] > 0) and np.all(np.isfinite(r["gamma"]))
    assert np.all(r["gmm_cluster"] == np.round(r["gmm_cluster"])) and r["gmm_cluster"].min() >= 0
    assert r["total_sigma"] == pytest.approx(r["atomic_sigma"].sum(), rel=1e-12)
    assert at.calc.uq_enabled
    plain = cell()
    plain.calc = GRACERuntimeCalculator(uqm[arch], num_threads=1)
    assert plain.get_potential_energy() == e                      # UQ does not touch the energy path
    assert not any(k in plain.calc.results for k in KEYS)


@pytest.mark.parametrize("arch", ["1l", "2l", "3l"])
def test_uq_thread_count_and_permutation(uqm, arch):
    at = cell()
    res = []
    for nth in (1, 4):
        a = at.copy()
        a.calc = GRACERuntimeCalculator(uqm[arch], num_threads=nth, uq=True)
        a.get_potential_energy()
        res.append(a.calc.results["gamma"])
    assert np.array_equal(res[0], res[1])                         # one thread per atom, fixed order
    p = np.random.default_rng(0).permutation(len(at))
    b = at[p]
    b.calc = GRACERuntimeCalculator(uqm[arch], num_threads=2, uq=True)
    b.get_potential_energy()
    # another neighbour order: fp32 round-off of the basis only (measured 1.2e-6 relative), same clusters
    assert np.allclose(b.calc.results["gamma"], res[0][p], rtol=1e-5, atol=0)


def test_enable_disable(uqm):
    at = cell()
    calc = GRACERuntimeCalculator(uqm["2l"], num_threads=2)
    at.calc = calc
    at.get_potential_energy()
    assert "gamma" not in calc.results
    calc.enable_uq()
    at.get_potential_energy()                                     # results reset by enable_uq: recomputed
    assert "gamma" in calc.results
    calc.disable_uq()
    at.get_potential_energy()
    assert "gamma" not in calc.results
    with pytest.raises(ValueError, match="gamma_only"):
        calc.enable_uq(mode="full")


def test_uq_without_head(model_2l_nouq):
    m = Model(model_2l_nouq)                                      # exported without UQ arrays
    assert not m.has_uq
    with pytest.raises(ValueError, match="UQ"):
        GRACERuntimeCalculator(m, uq=True)
    calc = GRACERuntimeCalculator(m)
    with pytest.raises(ValueError, match="UQ"):
        calc.enable_uq()


def test_model_compute_uq(uqm):
    m = uqm["1l"]
    assert m.has_uq
    out = m.compute(["W"], np.ones(1, np.int32), np.zeros(2, np.int64), np.zeros(0, np.int32), np.zeros((0, 3)),
                    uq=True)                                      # isolated atom: no bonds
    assert out["gamma"].shape == (1,) and np.isfinite(out["gamma"][0]) and out["gamma"][0] > 1
    assert out["total_sigma"] == out["atomic_sigma"][0]
    assert "gamma" not in m.compute(["W"], np.ones(1, np.int32), np.zeros(2, np.int64), np.zeros(0, np.int32),
                                    np.zeros((0, 3)))


def test_more_gmm_clusters_than_max_is_rejected(uqm, tmp_path):
    """uq_n_clusters[e] > uq_max_clusters would index the GMM arrays out of bounds (lammps-ace 3cd0775d99)."""
    src = UQ_MODELS["1l"]
    with np.load(src) as z:
        d = {k: z[k] for k in z.files}
    d["uq_n_clusters"] = d["uq_n_clusters"].copy()
    d["uq_n_clusters"][0] = int(np.ravel(d["uq_max_clusters"])[0]) + 1
    np.savez(tmp_path / "bad.npz", **d)
    m = Model(str(tmp_path / "bad.npz"))      # energy weights load; the UQ head is checked by enable_uq
    with pytest.raises(Exception, match="uq_max_clusters"):
        GRACERuntimeCalculator(m, uq=True)


def test_3l_uq_with_energy_only_and_without_head(uqm, tmp_path):
    """GRACE-3L: energy_only gives the gamma of the full call; a 3L model stripped of its UQ head: ValueError"""
    from model_tools import strip_uq
    at = cell()
    full = at.copy()
    full.calc = GRACERuntimeCalculator(uqm["3l"], num_threads=1, uq=True, energy_only=False)
    full.get_potential_energy()
    eo = at.copy()
    eo.calc = GRACERuntimeCalculator(uqm["3l"], num_threads=1, uq=True, energy_only=True)
    eo.get_potential_energy()
    assert "forces" not in eo.calc.results
    assert np.array_equal(eo.calc.results["gamma"], full.calc.results["gamma"])
    nouq = strip_uq(UQ_MODELS["3l"], tmp_path / "kokkos.npz")
    with pytest.raises(ValueError, match="UQ"):
        GRACERuntimeCalculator(str(nouq), uq=True)

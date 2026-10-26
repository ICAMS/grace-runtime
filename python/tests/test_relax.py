# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime.relax_batch on the CPU against ase.optimize.FIRE (+ ase.filters.FrechetCellFilter) with
GRACERuntimeCalculator on each structure alone: same forces, so the same trajectory to round-off (positions and cells
1e-8 A after 25 steps without, 1e-6 A after 20 steps with the cell filter; the same step count to convergence);
in-flight batching, lazy input, constraints, callback, CLI."""
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixAtoms, FixCom
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

from grace_runtime import GRACERuntimeCalculator, Model, relax_batch, relax_batch_iter
from grace_runtime.relax import _frechet_cell_forces, _logm, expm_frechet

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M1 = CACHE / "GRACE-1L-OMAT-large-base" / "kokkos.npz"

needs_model = pytest.mark.skipif(not M1.exists(), reason="registry model missing")


@pytest.fixture(scope="module")
def model():
    return Model(str(M1))


def cells():
    w = bulk("W", "bcc", a=3.2, cubic=True).repeat(2)
    w.rattle(0.05, seed=1)
    wmo = bulk("W", "bcc", a=3.1, cubic=True).repeat((2, 2, 3))
    wmo.symbols[::3] = "Mo"
    del wmo[5]
    wmo.rattle(0.04, seed=2)
    tri = bulk("Mo", "bcc", a=3.15).repeat((2, 3, 2))
    tri.set_cell(tri.cell.array @ [[1, 0.03, 0], [0.02, 1, 0.01], [0, 0, 1.02]], scale_atoms=True)
    tri.rattle(0.03, seed=3)
    return [w, wmo, tri]


def ase_relax(at, model, fmax, steps, filt=None, dispersion=None, **fkw):
    a = at.copy()
    a.calc = GRACERuntimeCalculator(model, device="cpu", dispersion=dispersion)
    opt = FIRE(FrechetCellFilter(a, **fkw) if filt else a, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return a, opt.nsteps


def test_expm_frechet_matches_scipy():
    from scipy.linalg import expm, expm_frechet as sp
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (20, 3, 3)) * np.array([1e-3, 0.1, 1.0, 5.0, 20.0]).repeat(4)[:, None, None]
    E = rng.normal(0, 1, (20, 3, 3))
    R, L = expm_frechet(A, E)
    for k in range(20):
        r, l = sp(A[k], E[k])
        assert np.allclose(R[k], expm(A[k]), rtol=1e-12, atol=1e-14 * np.abs(r).max())
        assert np.allclose(L[k], l, rtol=1e-10, atol=1e-12 * np.abs(l).max())
    R0, L0 = expm_frechet(np.zeros((2, 3, 3)), E[:2])
    assert np.allclose(R0, np.eye(3), rtol=0, atol=1e-15) and np.allclose(L0, E[:2], rtol=1e-14, atol=0)


def test_frechet_cell_forces_and_logm_match_scipy():
    """the cell forces (one Frechet derivative by the adjoint) as the nine directional derivatives of
    ase CellUtility; the batched logarithm (masked deformations) as scipy.linalg.logm, a defective shear too"""
    from scipy.linalg import expm_frechet as sp, logm
    rng = np.random.default_rng(1)
    A, G = 0.1 * rng.normal(0, 1, (8, 3, 3)), rng.normal(0, 1, (8, 3, 3))
    ref = [[(sp(A[s], d.reshape(3, 3), compute_expm=False) * G[s]).sum() for d in np.eye(9)] for s in range(8)]
    assert np.abs(_frechet_cell_forces(A, G) - np.reshape(ref, (8, 3, 3))).max() <= 1e-13
    X = np.concatenate([np.eye(3) + 0.05 * rng.normal(0, 1, (8, 3, 3)),
                        np.eye(3) + rng.normal(0, 1, (8, 3, 3)) * [0.3, 0.2, 0.1],
                        [[[1, 0.3, 0], [0, 1, 0], [0, 0, 1]], [[1.5, 0.9, 0], [0, 1.5, 0], [0, 0, 0.7]]]])
    assert np.abs(_logm(X) - [np.real(logm(x)) for x in X]).max() <= 1e-13


@needs_model
@pytest.mark.parametrize("steps", [25, 1000])
def test_positions_match_ase_fire(model, steps):
    ats = cells()
    fmax = 1e-6 if steps == 25 else 0.05
    res = relax_batch(model, ats, fmax=fmax, max_steps=steps, device="cpu")
    for at, r in zip(ats, res):
        ref, nref = ase_relax(at, model, fmax, steps)
        assert r["n_steps"] == nref and r["converged"] == (steps > 25)
        assert np.abs(r["atoms"].positions - ref.positions).max() <= 1e-8
        assert abs(r["energy"] - ref.get_potential_energy()) <= 1e-8 * len(at)
        assert np.array_equal(r["atoms"].cell.array, at.cell.array)


@needs_model
@pytest.mark.parametrize("opts", [{}, {"scalar_pressure": 0.01}, {"hydrostatic_strain": True},
                                  {"constant_volume": True}, {"mask": [1, 1, 0, 0, 0, 1]},
                                  {"exp_cell_factor": 4.0}, {"dispersion": "d3-pbe"}])
def test_frechet_filter_matches_ase(model, opts):
    ats = cells()
    res = relax_batch(model, ats, fmax=1e-6, max_steps=20, cell_filter="frechet", device="cpu", **opts)
    for at, r in zip(ats, res):
        ref, nref = ase_relax(at, model, 1e-6, 20, filt=True, **opts)
        # expm / logm of scipy vs the batched Pade approximant: cells differ by ~1e-16, which moves the fp32 forces
        # by their round-off; 20 FIRE steps carry that to ~1e-8 A
        assert r["n_steps"] == nref == 20
        assert np.abs(r["atoms"].cell.array - ref.cell.array).max() <= 1e-6, opts
        assert np.abs(r["atoms"].positions - ref.positions).max() <= 1e-6, opts
    if opts.get("dispersion"):    # the result is the sums: the calculator's energy at the result atoms, D3 included
        calc = GRACERuntimeCalculator(model, device="cpu", dispersion=opts["dispersion"])
        for at, r in zip(ats, res):
            assert abs(r["energy"] - calc.get_potential_energy(r["atoms"].copy())) <= 1e-8 * len(at)
            assert calc.results["dispersion_energy"] < -0.1 * len(at)
    if opts.get("constant_volume"):
        for at, r in zip(ats, res):
            assert abs(r["atoms"].get_volume() / at.get_volume() - 1) <= 1e-3


@needs_model
def test_frechet_converges_as_ase(model):
    ats = cells()[:2]
    res = relax_batch(model, ats, fmax=0.02, cell_filter="frechet", device="cpu")
    for at, r in zip(ats, res):
        ref, nref = ase_relax(at, model, 0.02, 500, filt=True)
        assert r["converged"] and r["n_steps"] == nref and r["fmax"] < 0.02
        assert abs(r["energy"] - ref.get_potential_energy()) <= 1e-6 * len(at)    # fp32 round-off of the path


@needs_model
def test_in_flight_batching_does_not_change_results(model):
    ats = cells()
    full = relax_batch(model, ats, fmax=0.05, device="cpu")
    one = relax_batch(model, ats, fmax=0.05, device="cpu", max_atoms=1)    # one structure in flight at a time
    for a, b in zip(full, one):
        assert a["n_steps"] == b["n_steps"]    # the CPU kernels' OpenMP sums are not bitwise reproducible
        assert np.abs(a["atoms"].positions - b["atoms"].positions).max() <= 1e-8


@needs_model
def test_iterator_lazy_input_and_order(model):
    ats = cells() + [Atoms(cell=[5, 5, 5], pbc=True)]
    read = []

    def gen():
        for k, a in enumerate(ats):
            read.append(k)
            yield a

    it = relax_batch_iter(model, gen(), fmax=0.05, device="cpu", max_atoms=20)
    first = next(it)
    assert len(read) < len(ats)    # not all read before the first result
    got = dict([first] + list(it))
    assert sorted(got) == list(range(len(ats)))
    assert got[3]["converged"] and got[3]["n_steps"] == 0 and got[3]["energy"] == 0.0
    ref = relax_batch(model, ats[:3], fmax=0.05, device="cpu")
    for k in range(3):
        assert np.abs(got[k]["atoms"].positions - ref[k]["atoms"].positions).max() <= 1e-8


@needs_model
def test_fix_atoms(model):
    at = cells()[1]
    at.set_constraint(FixAtoms(indices=[0, 3, 7]))
    r = relax_batch(model, [at], fmax=0.05, device="cpu")[0]
    ref, nref = ase_relax(at, model, 0.05, 500)
    assert np.array_equal(r["atoms"].positions[[0, 3, 7]], at.positions[[0, 3, 7]])
    assert r["n_steps"] == nref and np.abs(r["atoms"].positions - ref.positions).max() <= 1e-8
    assert r["atoms"].constraints and np.abs(r["atoms"].get_forces()[[0, 3, 7]]).max() == 0.0
    bad = cells()[0]
    bad.set_constraint(FixCom())
    with pytest.raises(ValueError, match="FixAtoms"):
        relax_batch(model, [bad], device="cpu")


@needs_model
def test_inputs_unchanged_and_result_atoms(model):
    ats = cells()
    before = [a.positions.copy() for a in ats]
    res = relax_batch(model, ats, fmax=0.05, cell_filter="frechet", device="cpu")
    for a, p, r in zip(ats, before, res):
        assert np.array_equal(a.positions, p) and a.calc is None
        assert r["atoms"] is not a and r["atoms"].get_potential_energy() == r["energy"]
        assert r["forces"].shape == (len(a), 3) and r["stress"].shape == (6,)
        assert not r["stopped"]


@needs_model
def test_callback_sees_steps_and_stops(model):
    ats = cells()
    seen = []

    def cb(step, indices, atoms):
        seen.append((step, list(indices)))
        assert len(indices) == len(atoms) and all(a.calc is not None for a in atoms)
        return np.array([1, 7]) if step == 3 else None    # numpy indices; 7 is not an input

    res = relax_batch(model, ats, fmax=1e-6, max_steps=10, device="cpu", callback=cb)
    assert seen[0] == (0, [0, 1, 2]) and [s for s, _ in seen] == list(range(len(seen)))
    assert res[1]["stopped"] and not res[1]["converged"] and res[1]["n_steps"] == 3
    assert not res[0]["stopped"] and res[0]["n_steps"] == 10


def test_bad_arguments():
    with pytest.raises(ValueError, match="cell_filter"):
        next(relax_batch_iter("x", [], cell_filter="unit"))
    with pytest.raises(ValueError, match="need"):
        next(relax_batch_iter("x", [], scalar_pressure=0.1))


@needs_model
def test_filter_needs_full_rank_cell(model):
    mol = Atoms("W2", positions=[[0, 0, 0], [2.5, 0, 0]])
    calls = []
    with pytest.raises(ValueError, match="structure 2: the cell filter needs a cell of rank 3"):
        # a list is checked before the first step, not when the molecule enters the batch
        relax_batch(model, cells()[:2] + [mol], cell_filter="frechet", device="cpu", max_atoms=1,
                    callback=lambda *a: calls.append(a))
    assert calls == []
    r = relax_batch(model, [mol], fmax=0.05, device="cpu")[0]    # positions only: molecules relax
    assert r["converged"] and np.all(r["stress"] == 0)


@needs_model
def test_cli_relax(tmp_path):
    from ase.io import read, write
    from grace_runtime import cli
    src, dst = tmp_path / "in.extxyz", tmp_path / "out.extxyz"
    write(src, cells()[:2])
    assert cli.main(["relax", str(M1), str(src), "-o", str(dst), "--device", "cpu", "--cell", "--fmax", "0.05"]) == 0
    got = read(dst, ":")
    assert sorted(a.info["relax_index"] for a in got) == [0, 1]
    assert all(a.info["relax_converged"] and a.info["relax_fmax"] < 0.05 for a in got)
    assert all(a.get_forces().shape == (len(a), 3) and a.get_stress().shape == (6,) for a in got)
    assert cli.main(["relax", str(M1), str(src), "-o", str(dst), "--pressure", "0.1"]) == 2
    assert cli.main(["relax", str(M1), str(src), "-o", str(src), "--device", "cpu"]) == 2    # would empty the input
    assert len(read(src, ":")) == 2
    assert cli.main(["relax", str(M1), str(src), "-o", str(dst), "--device", "cpu", "--max-steps", "2"]) == 1

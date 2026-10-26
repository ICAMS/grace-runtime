# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Dispersion correction DFT-D3(BJ) on the GPU (fp32 pair terms, fp64 sums): the dispersion_* results of the GPU
calculator against simple-dftd3 (tests/fixtures/d3_ref.npz) and the CPU, the totals equal model + dispersion, atoms moved
without a neighbour-list rebuild, compute_batch, Batch.compute (device neighbour lists and the CPU cell list) and the
TorchSim model equal the calculator."""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import all_changes

from grace_runtime import GRACERuntimeCalculator, _gpu, compute_batch

from d3_ref import cases
from gpu_tools import GPU, NO_GPU

CASES = [c for c in cases() if c["name"] != "heavy_cluster"]    # elements of the test model


def _module():
    try:
        return _gpu.ensure_init(GPU) if GPU else None
    except (ImportError, ValueError):
        return None


pytestmark = pytest.mark.skipif(_module() is None, reason=NO_GPU)
PROPS = ["energy", "forces", "stress", "energies"]


def _tol(c):
    """fp32 pair terms: energy 1e-6 relative, forces 5e-6 of max(|F|, 0.05 eV/A)"""
    return 1e-6 * max(1.0, abs(c["energy"])), 5e-6 * max(np.abs(c["forces"]).max(), 0.05)


def _run(model, at, device, d3=None):
    calc = GRACERuntimeCalculator(model, device=device, dispersion=d3)
    calc.calculate(at.copy(), PROPS, all_changes)
    return calc


@pytest.mark.parametrize("k", range(len(CASES)))
def test_gpu_dispersion(models, k):
    c = CASES[k]
    te, tf = _tol(c)
    g = _run(models["1l"], c["atoms"], GPU, c["d3"]).results
    base = _run(models["1l"], c["atoms"], GPU).results
    cpu = _run(models["1l"], c["atoms"], "cpu", c["d3"]).results
    np.testing.assert_allclose(g["dispersion_energy"], c["energy"], rtol=0, atol=te)
    np.testing.assert_allclose(g["dispersion_energies"], c["energies"], rtol=0, atol=te)
    np.testing.assert_allclose(g["dispersion_forces"], c["forces"], rtol=0, atol=tf)
    np.testing.assert_allclose(g["dispersion_forces"], cpu["dispersion_forces"], rtol=0, atol=tf)
    np.testing.assert_allclose(g["dispersion_stress"], cpu["dispersion_stress"], rtol=0,
                               atol=1e-6 * max(np.abs(cpu["dispersion_stress"]).max(), 1e-4))
    # the totals are model + dispersion (two GPU evaluations of the model: atomics may change the summation order)
    fmax = np.abs(base["forces"]).max()
    for key, atol in (("energy", 1e-6 * max(1.0, abs(base["energy"]))), ("forces", 2e-5 + 1e-6 * fmax),
                      ("stress", 1e-6), ("energies", 2e-5)):
        np.testing.assert_allclose(g[key] - g["dispersion_" + key], base[key], rtol=0, atol=atol)


def test_gpu_dispersion_moved_without_rebuild(models):
    c = CASES[0]    # water, 81 atoms
    at = c["atoms"].copy()
    calc = GRACERuntimeCalculator(models["1l"], device=GPU, dispersion=c["d3"], skin=1.0)
    cpu = GRACERuntimeCalculator(models["1l"], device="cpu", dispersion=c["d3"])
    rng = np.random.default_rng(3)
    for step in range(3):
        if step:
            at.positions += rng.uniform(-0.08, 0.08, size=at.positions.shape)    # < skin / 2 per atom in total
        calc.calculate(at.copy(), PROPS, ["positions"] if step else all_changes)
        cpu.calculate(at.copy(), PROPS, all_changes)
        te, tf = _tol({"energy": cpu.results["dispersion_energy"], "forces": cpu.results["dispersion_forces"]})
        np.testing.assert_allclose(calc.results["dispersion_energy"], cpu.results["dispersion_energy"], atol=te, rtol=0)
        np.testing.assert_allclose(calc.results["dispersion_forces"], cpu.results["dispersion_forces"], atol=tf, rtol=0)
    assert calc._kk_system.rebuilds == 1


def test_gpu_batch_host_path(models):
    cs = CASES[:6]
    d3 = cs[0]["d3"]
    res = compute_batch(models["1l"], [c["atoms"] for c in cs], device=GPU, dispersion=d3, dispersion_parts=True,
                        energies=True)
    for c, r in zip(cs, res):
        g = _run(models["1l"], c["atoms"], GPU, d3).results
        te, tf = _tol({"energy": g["dispersion_energy"], "forces": g["dispersion_forces"]})
        np.testing.assert_allclose(r["dispersion_energy"], g["dispersion_energy"], atol=te, rtol=0)
        np.testing.assert_allclose(r["dispersion_forces"], g["dispersion_forces"], atol=tf, rtol=0)
        np.testing.assert_allclose(r["dispersion_energies"], g["dispersion_energies"], atol=te, rtol=0)
        np.testing.assert_allclose(r["dispersion_stress"], g["dispersion_stress"], rtol=0,
                                   atol=1e-6 * max(np.abs(g["dispersion_stress"]).max(), 1e-4))
        np.testing.assert_allclose(r["energy"], g["energy"], atol=2e-6 * len(c["atoms"]), rtol=0)
    # a batch without dispersion is not the cached one with it
    plain = compute_batch(models["1l"], [cs[0]["atoms"]], device=GPU)
    assert "dispersion_energy" not in plain[0]
    np.testing.assert_allclose(plain[0]["energy"], res[0]["energy"] - res[0]["dispersion_energy"],
                               atol=2e-6 * len(cs[0]["atoms"]), rtol=0)


def test_gpu_dispersion_no_atoms(models):
    """a structure without atoms: a zero share (calculator: no species at all; batch: next to a non-empty one)"""
    empty = Atoms(cell=np.eye(3) * 5.0, pbc=True)
    r = _run(models["1l"], empty, GPU, "d3-pbe").results
    assert r["energy"] == 0.0 and r["dispersion_energy"] == 0.0
    assert r["dispersion_forces"].shape == (0, 3) and not r["dispersion_stress"].any()
    res = compute_batch(models["1l"], [empty, CASES[0]["atoms"]], device=GPU, dispersion="d3-pbe",
                        dispersion_parts=True)
    assert res[0]["dispersion_energy"] == 0.0 and not res[0]["dispersion_stress"].any()
    assert res[1]["dispersion_energy"] < 0.0


def test_gpu_batch_module_host_arrays(models):
    """Batch.compute from host arrays with dispersion: the calculator's share per structure, a zero share for a
    structure without atoms"""
    from grace_runtime import Model
    from grace_runtime.batch import _gpu_batch, _species_of, _type_lut
    from grace_runtime.calculator import _chunk_arg
    d3 = CASES[0]["d3"]
    seen, ats = set(), [Atoms(cell=np.eye(3) * 5.0, pbc=True)]
    for c in CASES:
        if c["atoms"].pbc.all() and c["name"] not in seen:
            seen.add(c["name"])
            ats.append(c["atoms"])
    species = _species_of(ats)
    b = _gpu_batch(_module(), Model(models["1l"]), species, _chunk_arg("auto"), dispersion=d3)
    natoms = np.array([len(a) for a in ats], np.int32)
    types = np.ascontiguousarray(_type_lut(species)[np.concatenate([a.numbers for a in ats])], dtype=np.int32)
    pos = np.ascontiguousarray(np.concatenate([a.positions for a in ats]).reshape(-1, 3))
    cells = np.ascontiguousarray(np.stack([a.cell.array for a in ats]))
    pbc = np.ones((len(ats), 3), dtype=bool)
    out = b.compute(natoms, types, pos, cells, pbc, energies=True, dispersion_parts=True)
    assert out["dispersion_energy"][0] == 0.0 and not out["dispersion_virial"][0].any()
    n0 = 0
    for k, at in enumerate(ats[1:], start=1):
        n0 += len(ats[k - 1])
        sl = slice(n0, n0 + len(at))
        g = _run(models["1l"], at, GPU, d3).results
        te, tf = _tol({"energy": g["dispersion_energy"], "forces": g["dispersion_forces"]})
        np.testing.assert_allclose(out["dispersion_energy"][k], g["dispersion_energy"], atol=te, rtol=0)
        np.testing.assert_allclose(out["dispersion_forces"][sl], g["dispersion_forces"], atol=tf, rtol=0)
        np.testing.assert_allclose(out["dispersion_energies"][sl], g["dispersion_energies"], atol=te, rtol=0)
        np.testing.assert_allclose(out["energy"][k], g["energy"], atol=2e-6 * len(at), rtol=0)


def test_torchsim_device_path(models):
    torch = pytest.importorskip("torch")
    ts = pytest.importorskip("torch_sim")
    from grace_runtime.torchsim import GRACERuntimeModel
    if not torch.cuda.is_available():
        pytest.skip("torch sees no GPU")
    cs = [c for c in CASES if c["atoms"].pbc.all()][:3]
    d3 = cs[0]["d3"]
    m = GRACERuntimeModel(models["1l"], device="cuda", dispersion=d3, dispersion_parts=True)
    state = ts.io.atoms_to_state([c["atoms"] for c in cs], device=torch.device("cuda"), dtype=torch.float64)
    out = m(state)
    n0 = 0
    for k, c in enumerate(cs):
        g = _run(models["1l"], c["atoms"], GPU, d3).results
        n = len(c["atoms"])
        te, tf = _tol({"energy": g["dispersion_energy"], "forces": g["dispersion_forces"]})
        np.testing.assert_allclose(out["dispersion_energy"][k].item(), g["dispersion_energy"], atol=te, rtol=0)
        np.testing.assert_allclose(out["dispersion_forces"][n0:n0 + n].cpu().numpy(), g["dispersion_forces"],
                                   atol=tf, rtol=0)
        np.testing.assert_allclose(out["energy"][k].item(), g["energy"], atol=2e-6 * n, rtol=0)
        n0 += n

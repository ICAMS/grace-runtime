# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime.torchsim.GRACERuntimeModel: the TorchSim model contract (validate_model_outputs), energy /
forces / stress of batched states against GRACERuntimeCalculator on the same device (CPU: the same code path up to
3L fp32 noise; GPU: tolerances of test_batch), extras (per-atom energies, UQ), dtype / device of the outputs, non-periodic and
one-atom systems, a species set changing between calls; refused autograd. Skipped without torch_sim."""
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ts = pytest.importorskip("torch_sim")

from ase import Atoms    # noqa: E402
from ase.build import bulk    # noqa: E402
from ase.cluster import Icosahedron    # noqa: E402
from ase.io import read    # noqa: E402
from ase.stress import voigt_6_to_full_3x3_stress    # noqa: E402
from torch_sim.models.interface import validate_model_outputs    # noqa: E402

from grace_runtime import GRACERuntimeCalculator, _cuda    # noqa: E402
from grace_runtime.torchsim import GRACERuntimeModel    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FX = ROOT / "tests" / "fixtures"
CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M1 = CACHE / "GRACE-1L-OMAT-large-base" / "kokkos.npz"
M2 = CACHE / "GRACE-2L-OMAT-medium-ft-E" / "kokkos.npz"    # 2L with a UQ head
M3 = CACHE / "GRACE-3L-OMAT-large" / "kokkos.npz"
MODELS = {"1L": M1, "2L": M2, "3L": M3}


def _cuda_ok():
    try:
        _cuda.module_for(0)
        return torch.cuda.is_available()
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not all(m.exists() for m in MODELS.values()), reason="registry models missing")
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not _cuda_ok(), reason="no CUDA module / torch GPU"))]
# (energy per atom, forces, stress): CPU = the calculator's own code (3L: fp32 parts summed by threads in varying
# order, ~2e-7 eV/A); GPU = fp32 with another summation order
TOL = {"cpu": (1e-8, 2e-6, 1e-8), "cuda": (2e-6, 3e-5, 1e-6)}


def lmp(name):
    return read(FX / f"{name}.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})


def periodic():
    nb = bulk("Nb", "bcc", a=3.30, cubic=True).repeat(2)
    nb.rattle(0.03, seed=4)
    iso = Atoms("Mo", positions=[[1.0, 2.0, 3.0]], cell=[10, 10, 10], pbc=True)
    prim = bulk("W", "bcc", a=3.165)    # 1 atom, small triclinic cell: many images
    return [lmp("W54"), lmp("WMo54"), nb, iso, prim, lmp("W2tri")]


def cluster():
    c = Icosahedron("W", 3, latticeconstant=3.9)
    c.rattle(0.03, seed=3)
    c.symbols[::3] = "Mo"
    c.pbc = False
    return c


def reference(path, structures, device, **kw):
    calc = GRACERuntimeCalculator(str(path), device=device, **kw)
    out = []
    for at in structures:
        at = at.copy()
        at.calc = calc
        r = {"energy": at.get_potential_energy(), "forces": at.get_forces()}
        r["stress"] = voigt_6_to_full_3x3_stress(at.get_stress()) if at.cell.rank == 3 else np.zeros((3, 3))
        if kw.get("uq"):
            for key in ("gamma", "atomic_sigma", "gmm_cluster", "total_sigma"):
                r[key] = np.array(calc.results[key])
        out.append(r)
    return out


def compare(res, ref, structures, device):
    te, tf, ts_ = TOL[device]
    n0 = 0
    for k, (at, r) in enumerate(zip(structures, ref)):
        n = len(at)
        assert abs(float(res["energy"][k]) - r["energy"]) <= te * max(n, 1), k
        assert np.abs(res["forces"][n0:n0 + n].cpu().numpy() - r["forces"]).max(initial=0) <= tf, k
        assert np.abs(res["stress"][k].cpu().numpy() - r["stress"]).max() <= ts_, k
        n0 += n


@pytest.mark.parametrize("device", DEVICES)
def test_validate_model_outputs(device):
    model = GRACERuntimeModel(str(M1), device=device)
    validate_model_outputs(model, torch.device(device), torch.float64)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layers", sorted(MODELS))
def test_periodic_batch_matches_calculator(device, layers):
    structures = periodic()
    model = GRACERuntimeModel(str(MODELS[layers]), device=device)
    state = ts.io.atoms_to_state(structures, torch.device(device), torch.float64)
    res = model(state)
    assert res["energy"].shape == (len(structures),) and res["stress"].shape == (len(structures), 3, 3)
    assert res["energy"].device.type == device and res["energy"].dtype == torch.float64
    compare(res, reference(MODELS[layers], structures, device), structures, device)


@pytest.mark.parametrize("device", DEVICES)
def test_nonperiodic_and_slab(device):
    model = GRACERuntimeModel(str(M2), device=device)
    mol = [cluster()]
    res = model(ts.io.atoms_to_state(mol, torch.device(device), torch.float64))
    compare(res, reference(M2, mol, device), mol, device)
    assert not res["stress"].any()    # zero cell: no stress
    slab = bulk("W", "bcc", a=3.165, cubic=True).repeat((3, 3, 4))
    slab.center(vacuum=8.0, axis=2)
    slab.pbc = [True, True, False]
    slab.rattle(0.03, seed=2)
    res = model(ts.io.atoms_to_state([slab], torch.device(device), torch.float64))
    compare(res, reference(M2, [slab], device), [slab], device)


@pytest.mark.parametrize("device", DEVICES)
def test_species_change_and_extras(device):
    model = GRACERuntimeModel(str(M2), device=device, energies=True, uq=True)
    for structures in (periodic()[:2], periodic()[2:4]):    # {Mo, W}, then {Mo, Nb}
        res = model(ts.io.atoms_to_state(structures, torch.device(device), torch.float64))
        ref = reference(M2, structures, device, uq=True)
        compare(res, ref, structures, device)
        n = sum(len(a) for a in structures)
        assert res["energies"].shape == (n,) and res["gmm_cluster"].dtype == torch.int64
        for k, r in enumerate(ref):
            assert abs(float(res["total_sigma"][k]) - float(r["total_sigma"])) <= 1e-5 * max(1.0, abs(r["total_sigma"]))
        e_sum = torch.zeros(len(structures), dtype=torch.float64, device=res["energies"].device)
        e_sum.index_add_(0, torch.repeat_interleave(torch.arange(len(structures), device=e_sum.device),
                                                     torch.tensor([len(a) for a in structures], device=e_sum.device)),
                         res["energies"])
        assert torch.allclose(e_sum, res["energy"], atol=1e-6 * n)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_and_energy_only(device):
    structures = periodic()[:3]
    state = ts.io.atoms_to_state(structures, torch.device(device), torch.float32)
    res = GRACERuntimeModel(str(M1), device=device, dtype=torch.float32)(state)
    assert all(v.dtype == torch.float32 for v in res.values())
    e = GRACERuntimeModel(str(M1), device=device, compute_forces=False, compute_stress=False)(state)
    assert set(e) == {"energy"}
    assert torch.allclose(e["energy"].float(), res["energy"], rtol=1e-6)


def test_autograd_refused():
    model = GRACERuntimeModel(str(M1), device="cpu")
    state = ts.io.atoms_to_state([periodic()[0]], torch.device("cpu"), torch.float64)
    state.positions.requires_grad_(True)
    with pytest.raises(RuntimeError, match="autograd"):
        model(state)


@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA module / torch GPU")
def test_cpu_state_on_cuda_model():
    structures = periodic()[:2]
    model = GRACERuntimeModel(str(M1), device="cuda:0")
    assert model.device == torch.device("cuda", 0)
    res = model(ts.io.atoms_to_state(structures, torch.device("cpu"), torch.float64))
    assert res["forces"].device == torch.device("cuda", 0)
    compare(res, reference(M1, structures, "cuda"), structures, "cuda")


@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA module / torch GPU")
def test_split_calls_and_memory_scaler():
    structures = periodic()
    state = ts.io.atoms_to_state(structures, torch.device("cuda"), torch.float64)
    one = GRACERuntimeModel(str(M3), device="cuda")(state)
    split = GRACERuntimeModel(str(M3), device="cuda", max_atoms=60)(state)    # several calls into the module
    for key in one:
        assert torch.allclose(one[key], split[key], atol=1e-6), key
    from grace_runtime.batch import _KK_BATCHES
    assert all(b.last_host_seconds > 0 for b in _KK_BATCHES.values())    # host neighbour search + layouts timed
    model = GRACERuntimeModel(str(M3), device="cuda")
    model(state)    # the model's Batch keeps its buffers: the probe must still measure a call's full allocation
    scaler = model.max_memory_scaler(state)
    # 3L keeps ~64 KB per atom + image (small cells: several images per atom): on a 24 GB card thousands to a few
    # tens of thousands of atoms per call, not the max_atoms cap; 1L is lighter
    assert 2048 <= scaler <= 60000, scaler
    assert GRACERuntimeModel(str(M1), device="cuda").max_memory_scaler(state) > scaler
    batcher = ts.autobatching.BinningAutoBatcher(model, memory_scales_with=model.memory_scales_with,
                                                 max_memory_scaler=scaler)
    batcher.load_states(state)
    batches = list(batcher)
    assert len(batches) == 1 and sorted(batches[0][1]) == list(range(len(structures)))    # ~130 atoms: one batch


def md_cells(k=8):
    cells = []
    for i in range(k):
        a = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
        a.symbols[i::5] = "Mo"
        a.rattle(0.05, seed=10 + i)
        cells.append(a)
    return cells


@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA module / torch GPU")
@pytest.mark.parametrize("layers", sorted(MODELS))
def test_nve_energy_conserved(layers):
    from torch_sim.integrators import nve_init, nve_step
    from torch_sim.quantities import calc_kinetic_energy
    from torch_sim.units import MetalUnits
    model = GRACERuntimeModel(str(MODELS[layers]), device="cuda")
    state = ts.io.atoms_to_state(md_cells(), torch.device("cuda"), torch.float64)
    state.rng = 7
    md = nve_init(state, model, kT=torch.tensor(600 * MetalUnits.temperature, dtype=torch.float64, device="cuda"))

    def etot(s):
        return s.energy + calc_kinetic_energy(masses=s.masses, momenta=s.momenta, system_idx=s.system_idx)

    e0 = etot(md)
    dt = torch.tensor(0.001 * MetalUnits.time, dtype=torch.float64, device="cuda")    # 1 fs
    for _ in range(200):
        md = nve_step(md, model, dt=dt)
    drift = ((etot(md) - e0).abs() / 54).max().item()
    print(f"{layers} NVE 200 x 1 fs, 8 x 54 atoms at 600 K: max |dE_tot| {drift:.2e} eV/atom")
    assert drift < 5e-5    # measured 4-6e-6 (RTX 3090)


@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA module / torch GPU")
def test_nvt_langevin_with_autobatcher():
    model = GRACERuntimeModel(str(M2), device="cuda")
    state = ts.io.atoms_to_state(md_cells(16), torch.device("cuda"), torch.float64)
    scaler = model.max_memory_scaler(state)
    batcher = ts.autobatching.BinningAutoBatcher(model, memory_scales_with="n_atoms", max_memory_scaler=scaler)
    final = ts.integrate(state, model, integrator=ts.Integrator.nvt_langevin, n_steps=100, temperature=600.0,
                         timestep=0.001, autobatcher=batcher)
    assert final.n_systems == 16 and torch.isfinite(final.positions).all()
    out = model(final)
    assert torch.isfinite(out["energy"]).all() and (out["energy"] / 54 > -20).all()


@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA module / torch GPU")
def test_fire_with_inflight_autobatcher():
    model = GRACERuntimeModel(str(M1), device="cuda")
    state = ts.io.atoms_to_state(md_cells(8), torch.device("cuda"), torch.float64)
    batcher = ts.autobatching.InFlightAutoBatcher(model, memory_scales_with="n_atoms",
                                                  max_memory_scaler=4 * 54)    # 4 systems in flight, 8 in all
    final = ts.optimize(state, model, optimizer=ts.Optimizer.fire, max_steps=1000, autobatcher=batcher,
                        convergence_fn=ts.generate_force_convergence_fn(force_tol=0.05))
    fmax = model(final)["forces"].norm(dim=1).max().item()
    print(f"FIRE 8 x 54 atoms: final max |F| {fmax:.3f} eV/A")
    assert final.n_systems == 8 and fmax < 0.05


@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA module / torch GPU")
def test_buffers_kept_between_steps():
    """the model's Batch keeps its device buffers (an MD loop; compute_batch's own Batch returns them)"""
    import grace_runtime
    from grace_runtime import compute_batch
    from grace_runtime.batch import _KK_BATCHES
    structures = periodic()[:2]
    model = GRACERuntimeModel(str(M2), device="cuda")
    state = ts.io.atoms_to_state(structures, torch.device("cuda"), torch.float64)
    model(state)
    compute_batch(str(M2), structures, device="cuda")
    keep = {k[-1]: b.keep_buffers for k, b in _KK_BATCHES.items() if k[0] == model.model.path}
    assert keep == {True: True, False: False}
    kk = _cuda.ensure_init("cuda")

    def peak(f):    # device bytes allocated during a repeated call
        f()
        kk._debug_track_alloc(True)
        try:
            f()
            return kk._debug_peak_alloc()
        finally:
            kk._debug_track_alloc(False)

    kept = peak(lambda: model(state))
    released = peak(lambda: compute_batch(str(M2), structures, device="cuda"))
    print(f"device bytes allocated by a repeated call: model {kept}, compute_batch {released}")
    assert released > 0 and kept < 0.01 * released
    grace_runtime.free_gpu_memory()


@pytest.mark.skipif(not _cuda_ok(), reason="no CUDA module / torch GPU")
@pytest.mark.parametrize("layers", sorted(MODELS))
def test_device_and_host_paths_agree(layers):
    """a cuda state on a cuda model takes the device path (neighbour lists on the GPU); device_path=False the host
    path: same results within fp32 noise; zero cell vectors (a cluster) fall back to the host path"""
    structures = periodic()
    state = ts.io.atoms_to_state(structures, torch.device("cuda"), torch.float64)
    dev = GRACERuntimeModel(str(MODELS[layers]), device="cuda", energies=True)
    host = GRACERuntimeModel(str(MODELS[layers]), device="cuda", energies=True, device_path=False)
    rd, rh = dev(state), host(state)
    n = torch.tensor([len(a) for a in structures], dtype=torch.float64, device="cuda")
    assert ((rd["energy"] - rh["energy"]).abs() <= 1e-6 * n).all()
    assert (rd["forces"] - rh["forces"]).abs().max() <= 3e-5
    assert (rd["stress"] - rh["stress"]).abs().max() <= 1e-6
    assert (rd["energies"] - rh["energies"]).abs().max() <= 2e-6
    mol = [cluster()]
    compare(dev(ts.io.atoms_to_state(mol, torch.device("cuda"), torch.float64)), reference(MODELS[layers], mol, "cuda"),
            mol, "cuda")

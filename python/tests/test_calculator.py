# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""GRACERuntimeCalculator: LAMMPS references, small cells, pbc variants, errors, system changes, finite differences,
MD with neighbour-list rebuilds."""
import numpy as np
import pytest

from grace_runtime.neighbors import NeighborList
from ase import Atoms, units
from ase.build import bulk, molecule
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from grace_runtime import GRACERuntimeCalculator, Model

NKTV2P = 1.6021765e6    # eV/A^3 -> bar


@pytest.fixture(scope="module")
def mdl(models):
    return {k: Model(v) for k, v in models.items()}


def fixture_atoms(fixtures, cell):
    return read(f"{fixtures}/{cell}.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})


def read_ref(stem):
    a = np.loadtxt(stem + ".dump", skiprows=9)
    a = a[np.argsort(a[:, 0])]
    th = [float(x) for x in open(stem + ".thermo").read().split()]
    return a[:, 1:4], a[:, 4], th[1], np.array(th[2:8])


@pytest.mark.parametrize("cell", ["W54", "WMo54", "W2tri"])
@pytest.mark.parametrize("arch", ["2l", "1l", "3l"])
def test_matches_lammps_refs(mdl, fixtures, cell, arch):
    at = fixture_atoms(fixtures, cell)
    at.calc = GRACERuntimeCalculator(mdl[arch], num_threads=2)
    f_ref, e_ref, etot, p = read_ref(f"{fixtures}/ref/{cell}_{arch}")
    n = len(at)
    assert abs(at.get_potential_energy() - etot) / n < 1e-6
    assert np.abs(at.get_forces() - f_ref).max() < 2e-5
    assert np.abs(at.get_potential_energies() - e_ref).max() < 1e-5
    s_ref = -np.array([p[0], p[1], p[2], p[5], p[4], p[3]]) / NKTV2P     # LAMMPS xx yy zz xy xz yz -> Voigt
    assert np.abs(at.get_stress() - s_ref).max() / np.abs(s_ref).max() < 1e-5


def test_small_cells(mdl):
    prim = bulk("W", "bcc", a=3.165)
    sup = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    for m in mdl.values():
        prim.calc = GRACERuntimeCalculator(m, num_threads=1)
        sup.calc = GRACERuntimeCalculator(m, num_threads=1)
        assert abs(prim.get_potential_energy() - sup.get_potential_energy() / len(sup)) < 1e-6
        assert np.abs(prim.get_forces()).max() < 1e-6
        assert np.allclose(prim.get_stress(), sup.get_stress(), atol=1e-7)


def test_pbc_variants(mdl):
    mol = molecule("CH3CH2OH")
    mol.calc = GRACERuntimeCalculator(mdl["2l"], num_threads=1)
    e = mol.get_potential_energy()
    assert np.isfinite(e)
    assert np.abs(mol.get_forces().sum(axis=0)).max() < 1e-4
    assert np.all(mol.get_stress() == 0.0)
    slab = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 3))
    slab.center(vacuum=8.0, axis=2)              # vacuum 16 A > cutoff + skin: no z-images in range
    slab.rattle(0.03, seed=3)
    a, b = slab.copy(), slab.copy()
    a.pbc, b.pbc = (True, True, False), (True, True, True)
    a.calc = GRACERuntimeCalculator(mdl["2l"], num_threads=1)
    b.calc = GRACERuntimeCalculator(mdl["2l"], num_threads=1)
    # same bonds; the open direction is binned over its occupied range: another order
    assert np.abs(a.get_forces() - b.get_forces()).max() < 1e-5
    assert abs(a.get_potential_energy() - b.get_potential_energy()) / len(a) < 1e-7


def test_unknown_element(mdl):
    at = bulk("Og", "fcc", a=5.0)
    at.calc = GRACERuntimeCalculator(mdl["2l"])
    with pytest.raises(ValueError, match="Og"):
        at.get_potential_energy()


def test_system_change(mdl, fixtures):
    calc = GRACERuntimeCalculator(mdl["2l"], num_threads=1)
    for at in [fixture_atoms(fixtures, "W54"), fixture_atoms(fixtures, "WMo54"), fixture_atoms(fixtures, "W54")]:
        at.rattle(0.01, seed=4)
        at.calc = calc
        f = at.get_forces()
        fresh = at.copy()
        fresh.calc = GRACERuntimeCalculator(mdl["2l"], num_threads=1)
        assert np.array_equal(f, fresh.get_forces())


def test_failed_gpu_system_is_not_kept(mdl):
    """A GPU System that cannot be built (an engine self-test, device memory) raises its own error on every
    calculation; the calculator keeps no species setup without it (before: AttributeError _kk_cellkey on the next)."""
    class FailingModule:
        def System(self, *args, **kwargs):
            raise RuntimeError("engine self-test failed")
    calc = GRACERuntimeCalculator(mdl["2l"], num_threads=1, device="cpu")
    calc._kk, calc._kk_model = FailingModule(), None
    at = bulk("W", "bcc", a=3.165, cubic=True)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="self-test"):
            calc.get_potential_energy(at)


@pytest.mark.parametrize("arch", ["2l", "1l", "3l"])
def test_finite_differences(mdl, fixtures, arch):
    at = fixture_atoms(fixtures, "WMo54")
    at.calc = GRACERuntimeCalculator(mdl[arch], num_threads=2)
    f = at.get_forces()
    h = 1e-2
    for a, d in [(0, 0), (7, 1), (31, 2)]:
        x = at.positions.copy()
        at.positions[a, d] = x[a, d] + h
        ep = at.get_potential_energy()
        at.positions[a, d] = x[a, d] - h
        em = at.get_potential_energy()
        at.positions[:] = x
        assert abs(-(ep - em) / (2 * h) - f[a, d]) < 1e-2
    s = at.get_stress()
    cell0 = at.cell.array.copy()
    vol = at.get_volume()
    for (i, j), v in [((0, 0), 0), ((0, 1), 5)]:
        de = 1e-3
        es = []
        for sgn in (1, -1):
            eps = np.eye(3)
            eps[i, j] += sgn * de / (2 if i != j else 1)
            if i != j:
                eps[j, i] += sgn * de / 2
            at.set_cell(cell0 @ eps, scale_atoms=True)
            es.append(at.get_potential_energy())
        at.set_cell(cell0, scale_atoms=True)
        fd = (es[0] - es[1]) / (2 * de) / vol
        assert abs(fd - s[v]) <= 2e-3 * np.abs(s).max()


def test_md_with_rebuilds(mdl, fixtures):
    at = fixture_atoms(fixtures, "W54")
    MaxwellBoltzmannDistribution(at, temperature_K=2000, rng=np.random.default_rng(5))
    calc = GRACERuntimeCalculator(mdl["2l"], skin=0.05, num_threads=2)
    ref = GRACERuntimeCalculator(mdl["2l"], skin=0.0, num_threads=2)
    at.calc = calc
    dyn = VelocityVerlet(at, 2 * units.fs)
    worst = [0.0]

    def check():
        b = at.copy()
        b.calc = ref
        worst[0] = max(worst[0], np.abs(at.get_forces() - b.get_forces()).max())

    dyn.attach(check, interval=1)
    dyn.run(300)
    assert calc._cpu.nl.nbuild >= 3
    assert worst[0] < 1e-5


def _fake_topology(root, cores):
    """cores[cpu] = (package, core) written as sysfs topology files."""
    for cpu, (pkg, core) in enumerate(cores):
        d = root / f"cpu{cpu}" / "topology"
        d.mkdir(parents=True)
        (d / "physical_package_id").write_text(f"{pkg}\n")
        (d / "core_id").write_text(f"{core}\n")


def test_physical_cores_counts_distinct_cores(tmp_path):
    from grace_runtime.calculator import physical_cores
    # 2 packages x 2 cores, siblings numbered after all cores (cpu4-7 = siblings of cpu0-3)
    _fake_topology(tmp_path, [(0, 0), (0, 1), (1, 0), (1, 1)] * 2)
    assert physical_cores({0, 1}, tmp_path) == 2          # taskset -c 0-1: two cores of socket 0
    assert physical_cores({0, 4}, tmp_path) == 1          # one core and its sibling
    assert physical_cores(set(range(8)), tmp_path) == 4
    assert physical_cores({0, 1}, tmp_path / "missing") is None


def test_unknown_neighbor_backend(mdl):
    with pytest.raises(ValueError, match="backend"):
        GRACERuntimeCalculator(mdl["2l"], neighbor_backend="ase")
    with pytest.raises(ValueError, match="removed"):
        GRACERuntimeCalculator(mdl["2l"], neighbor_backend="matscipy")


def test_3l_species_subset_and_unknown(mdl, fixtures):
    """one element of a multi-element model; an element outside the model is a ValueError (Review Focus 1)"""
    at = fixture_atoms(fixtures, "W54")
    at.calc = GRACERuntimeCalculator(mdl["3l"], num_threads=2)
    assert np.isfinite(at.get_potential_energy())
    bad = at.copy()
    bad.numbers[0] = 118    # Og
    bad.calc = GRACERuntimeCalculator(mdl["3l"], num_threads=2)
    with pytest.raises(ValueError, match="not in the model"):
        bad.get_potential_energy()


def test_3l_threads_and_isolated_atom(mdl):
    """thread counts change the summation order only; an atom without neighbours: finite energy, zero force"""
    from ase import Atoms
    a = Atoms("W", positions=[[0, 0, 0]], cell=[20, 20, 20], pbc=False)
    a.calc = GRACERuntimeCalculator(mdl["3l"], num_threads=1)
    assert np.isfinite(a.get_potential_energy()) and np.abs(a.get_forces()).max() == 0.0
    w = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    w.rattle(0.05, seed=1)
    r = []
    for nth in (1, 4, 2):
        b = w.copy()
        b.calc = GRACERuntimeCalculator(mdl["3l"], num_threads=nth)
        r.append((b.get_potential_energy(), b.get_forces()))
    for e, f in r[1:]:
        assert abs(e - r[0][0]) / len(w) <= 1e-9 and np.abs(f - r[0][1]).max() <= 1e-6


def test_3l_two_calculators_two_threads(mdl):
    """two calculators on one Model from two Python threads, on different species sets (two drivers sharing the
    GRACE3LModel run at the same time), several rounds: each gives its single-threaded result exactly (Review Focus 3)"""
    import threading
    w = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    w.rattle(0.05, seed=1)
    wmo = w.copy()
    wmo.symbols[::3] = "Mo"
    cells = [w, wmo]
    def ef(k):
        b = cells[k].copy()
        b.calc = GRACERuntimeCalculator(mdl["3l"], num_threads=1)    # 1 thread: a fixed summation order
        return b.get_potential_energy(), b.get_forces()
    ref = [ef(0), ef(1)]
    bad = []
    def run(k):
        for _ in range(4):
            e, f = ef(k)
            if e != ref[k][0] or not np.array_equal(f, ref[k][1]):
                bad.append((k, e - ref[k][0], np.abs(f - ref[k][1]).max()))
    ts = [threading.Thread(target=run, args=(k,)) for k in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not bad, bad


def test_driver_cache_is_bounded(models):
    """one driver per species set is kept for reuse, at most 4 (GRACE-3L radial tables: ~15 MiB per species pair);
    an evicted set is rebuilt on its next use with the same result"""
    m = Model(models["2l"])
    calc = GRACERuntimeCalculator(m, num_threads=2)
    e = {}
    for el in ["W", "Mo", "Nb", "Ta", "V", "Cr"]:
        at = bulk(el, "bcc", a=3.1, cubic=True)
        at.calc = calc
        e[el] = at.get_potential_energy()
    assert m.cached_drivers == 4, m.cached_drivers
    at = bulk("W", "bcc", a=3.1, cubic=True)
    at.calc = calc
    assert at.get_potential_energy() == e["W"]


def test_skin_validation(models):
    """a negative skin dropped bonds inside the cutoff (W pair 5.5 A apart, cutoff 6 A: zero forces); now an error.
    Valid skins keep every bond inside the cutoff: the same energy and forces."""
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="skin"):
            GRACERuntimeCalculator(models["1l"], skin=bad)
        with pytest.raises(ValueError, match="skin"):
            NeighborList(6.0, skin=bad)
    out = []
    for skin in (0.0, 0.5, 2.0):
        at = Atoms("W2", positions=[(0, 0, 0), (5.5, 0, 0)], cell=[30, 30, 30], pbc=False)
        at.calc = GRACERuntimeCalculator(models["1l"], skin=skin, num_threads=1)
        out.append((at.get_potential_energy(), at.get_forces()))
    assert np.abs(out[0][1]).max() > 1e-4    # the pair interacts
    for e, f in out[1:]:
        assert abs(e - out[0][0]) < 1e-9 and np.allclose(f, out[0][1], atol=1e-9)


@pytest.mark.parametrize("arch", ["2l", "3l"])
def test_store_limit(models, monkeypatch, arch):
    """CPU 2L / 3L: the reverse pass keeps per-atom forward tensors up to Model(store_max_gb) (default None: half the
    available memory) and recomputes them above it, with the same results; backend_info["store"] tells which, and a
    RuntimeWarning says it once per model"""
    import warnings
    from grace_runtime import calculator
    monkeypatch.setattr(calculator, "_STORE_WARNED", set())
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="store_max_gb"):
            Model(models[arch], store_max_gb=bad)
    out = {}
    for limit in (None, 0.0):
        m = Model(models[arch], store_max_gb=limit)
        assert m.store_max_gb == limit
        at = bulk("W", "bcc", a=3.1, cubic=True) * (2, 2, 2)
        at.rattle(0.05, seed=1)
        calc = GRACERuntimeCalculator(m, device="cpu", num_threads=1, energy_only=False)
        at.calc = calc
        assert calc.backend_info["store"] is None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out[limit] = at.get_potential_energy(), at.get_forces()
            at.positions[0, 0] += 0.01    # a second call: no second warning
            at.get_potential_energy()
        st = calc.backend_info["store"]
        assert st["kept"] == (limit is None) and st["need_gb"] > 0, st
        assert st["limit_gb"] >= st["need_gb"] if limit is None else st["limit_gb"] == 0.0, st
        assert len([x for x in w if "forward store" in str(x.message)]) == (0 if limit is None else 1)
    assert out[None][0] == out[0.0][0] and np.array_equal(out[None][1], out[0.0][1])


def test_strict_mode_without_a_gpu(models):
    """GRACE_RUNTIME_STRICT=1: device="auto" without a usable GPU is an error (a native GPU or nothing); an explicit
    device="cpu" or GRACE_RUNTIME_DEVICE=cpu still runs on the CPU."""
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if k != "GRACE_RUNTIME_DEVICE"}
    env.update(GRACE_RUNTIME_STRICT="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    code = "import sys\nfrom grace_runtime import GRACERuntimeCalculator\nGRACERuntimeCalculator(sys.argv[1]{})\n"
    r = subprocess.run([sys.executable, "-c", code.format(""), models["1l"]], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "found no usable GPU" in r.stderr, r.stderr[-2000:]
    for extra, e in ((", device='cpu'", env), ("", dict(env, GRACE_RUNTIME_DEVICE="cpu"))):    # the explicit CPU
        r = subprocess.run([sys.executable, "-c", code.format(extra), models["1l"]], capture_output=True, text=True,
                           env=e)
        assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.parametrize("bad", [0, -1, -5, 119, 200])
def test_invalid_atomic_numbers_rejected(mdl, bad):
    """Atomic numbers < 1 or >= len(chemical_symbols) raise a clear ValueError instead of wrapping around (e.g. -1 -> Og)."""
    from grace_runtime import compute_batch
    at = bulk("W", "bcc", a=3.165, cubic=True)
    at.numbers[0] = bad
    at.calc = GRACERuntimeCalculator(mdl["1l"], num_threads=1)
    with pytest.raises(ValueError, match="atomic number"):
        at.get_potential_energy()
    with pytest.raises(ValueError, match="atomic number"):
        compute_batch(mdl["1l"], [at], device="cpu", num_threads=1)


def test_mutating_num_threads_updates_neighbor_list(mdl):
    """Mutating calc.num_threads updates both the CPU engine and its NeighborList."""
    at = bulk("W", "bcc", a=3.165, cubic=True).repeat(2)
    calc = GRACERuntimeCalculator(mdl["1l"], num_threads=1)
    at.calc = calc
    at.get_potential_energy()
    assert calc._cpu.num_threads == 1 and calc._cpu.nl.num_threads == 1
    calc.num_threads = 3
    at.positions[0, 0] += 0.5
    at.get_potential_energy()
    assert calc._cpu.num_threads == 3 and calc._cpu.nl.num_threads == 3

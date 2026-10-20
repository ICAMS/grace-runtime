# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""GRACERuntimeCalculator(device="cuda") against the CPU calculator: periodic, small triclinic,
shifted (atoms outside the cell), slab (mixed pbc) and open (cluster) structures, growth and species changes on one
calculator, NVE energy conservation, UQ keys. Both evaluate fp32 fused kernels with different summation orders:
energy 1e-5 eV/atom, forces 5e-5 eV/A + 1e-6 max|F| (fp32 relative), stress 1e-6 eV/A^3."""
import os
from pathlib import Path

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.cluster import Icosahedron
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from grace_runtime import GRACERuntimeCalculator, _cuda

ROOT = Path(__file__).resolve().parents[2]
FX = ROOT / "tests" / "fixtures"
CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
M2 = CACHE / "GRACE-2L-OMAT-medium-ft-E" / "kokkos.npz"    # 2L with a UQ head
M1 = CACHE / "GRACE-1L-OMAT-large-base" / "kokkos.npz"
M3 = CACHE / "GRACE-3L-OMAT-large" / "kokkos.npz"


def _cuda_ok():
    try:
        _cuda.module_for(0)
        return True
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _cuda_ok() or not (M1.exists() and M2.exists()),
                                reason="no CUDA module for this device or registry models missing")


def wmo54():
    return read(FX / "WMo54.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})


def structures():
    w54 = read(FX / "W54.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})
    w2tri = read(FX / "W2tri.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})
    shifted = wmo54()
    shifted.positions += 1.3 * shifted.cell.array.sum(axis=0) + np.random.default_rng(1).normal(0, 0.02, (54, 3))
    slab = bulk("W", "bcc", a=3.165, cubic=True).repeat((3, 3, 4))
    slab.center(vacuum=8.0, axis=2)
    slab.pbc = [True, True, False]
    slab.rattle(0.03, seed=2)
    cluster = Icosahedron("W", 3, latticeconstant=3.9)    # nearest neighbours 2.76 A, as in bcc W
    cluster.rattle(0.03, seed=3)
    cluster.symbols[::3] = "Mo"
    return {"W54": w54, "WMo54": wmo54(), "W2tri": w2tri, "WMo54-shifted": shifted, "W-slab": slab,
            "WMo-cluster": cluster}


def gpu(model, **kw):
    calc = GRACERuntimeCalculator(str(model), device="cuda", **kw)
    assert calc.backend.startswith("cuda:"), calc.backend
    return calc


def evaluate(at, calc):
    at = at.copy()
    at.calc = calc
    return at.get_potential_energy(), at.get_forces(), at.get_stress() if at.cell.rank == 3 else np.zeros(6)


@pytest.mark.parametrize("model", [M1, M2, M3], ids=["1L", "2L", "3L"])
@pytest.mark.parametrize("name", ["W54", "WMo54", "W2tri", "WMo54-shifted", "W-slab", "WMo-cluster"])
def test_parity_with_cpu(model, name):
    at = structures()[name]
    e_g, f_g, s_g = evaluate(at, gpu(model))
    e_c, f_c, s_c = evaluate(at, GRACERuntimeCalculator(str(model), num_threads=8))
    n = len(at)
    assert abs(e_g - e_c) / n <= 1e-5, (e_g, e_c)
    assert np.abs(f_g - f_c).max() <= 5e-5 + 1e-6 * np.abs(f_c).max()
    assert np.abs(s_g - s_c).max() <= 1e-6


def test_growth_and_species_change_on_one_calculator():
    calc = gpu(M2)
    cpu = GRACERuntimeCalculator(str(M2), num_threads=8)
    w = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    w.rattle(0.03, seed=4)
    big = wmo54().repeat(2)    # 54 -> 432 atoms and a new element
    for at in (w, wmo54(), big, w):
        e_g, f_g, _ = evaluate(at, calc)
        e_c, f_c, _ = evaluate(at, cpu)
        assert abs(e_g - e_c) / len(at) <= 1e-5
        assert np.abs(f_g - f_c).max() <= 5e-5


def test_uq_keys_match_cpu():
    at = wmo54()
    g = at.copy()
    g.calc = gpu(M2, uq=True)
    g.get_potential_energy()
    c = at.copy()
    c.calc = GRACERuntimeCalculator(str(M2), uq=True, num_threads=8)
    c.get_potential_energy()
    for k in ("gamma", "atomic_sigma"):
        a, b = g.calc.results[k], c.calc.results[k]
        assert np.abs(a - b).max() / np.abs(b).max() <= 1e-5, k
    assert np.array_equal(g.calc.results["gmm_cluster"], c.calc.results["gmm_cluster"])
    assert abs(g.calc.results["total_sigma"] - c.calc.results["total_sigma"]) <= 1e-5 * abs(c.calc.results["total_sigma"])


def test_energies_property():
    at = wmo54()
    at.calc = gpu(M2)
    e = at.get_potential_energies()
    assert abs(e.sum() - at.get_potential_energy()) <= 1e-4


def nve_drift(calc, steps=200):
    at = wmo54().repeat((2, 1, 1))
    MaxwellBoltzmannDistribution(at, temperature_K=1500, rng=np.random.default_rng(5))
    at.calc = calc
    md = VelocityVerlet(at, timestep=1.0 * units.fs)
    e = []
    md.attach(lambda: e.append(at.get_total_energy()), interval=10)
    md.run(steps)
    e = np.array(e) / len(at)
    return np.abs(e - e[0]).max()


def test_nve_conservation_as_cpu():
    d_g = nve_drift(gpu(M2, skin=0.5))
    d_c = nve_drift(GRACERuntimeCalculator(str(M2), num_threads=8, skin=0.5))
    print(f"NVE max |dE|/atom: cuda {d_g:.2e}  cpu {d_c:.2e}")
    assert d_g <= max(2 * d_c, 1e-4)


def test_bad_device():
    with pytest.raises(ValueError, match="unknown device"):
        GRACERuntimeCalculator(str(M2), device="gpu")


@pytest.mark.parametrize("model", [M1, M2], ids=["1L", "2L"])
def test_energy_only_as_full(model):
    at = wmo54()
    at.rattle(0.03, seed=8)
    full = at.copy()
    full.calc = gpu(model, energy_only=False)
    e_full, f_full = full.get_potential_energy(), full.get_forces()
    at.calc = gpu(model)
    e = at.get_potential_energy()
    assert "forces" not in at.calc.results
    assert abs(e - e_full) / len(at) <= 2e-6
    assert np.abs(at.get_forces() - f_full).max() <= 2e-5


@pytest.mark.parametrize("model", [M1, M2, M3], ids=["1l", "2l", "3l"])
def test_bond_vectors_do_not_depend_on_the_coordinate_origin(model):
    """bond vectors are differences of positions taken in double: a perfect diamond 14^3 (35 A box) has the zero
    forces of the CPU calculator (fp32 network rounding only, ~2e-6 eV/A), and the same cell translated by 500 A
    gives the same forces (fp32 positions made them grow with |x|: 3e-5 eV/A at 35 A)"""
    at = bulk("C").repeat((14, 14, 14))
    calc = GRACERuntimeCalculator(str(model), device="cuda")
    a = at.copy()
    a.calc = calc
    f0 = a.get_forces()
    b = at.copy()
    b.translate([500.0, 500.0, 500.0])
    b.calc = calc
    f1 = b.get_forces()
    assert np.abs(f0).max() <= 5e-6, np.abs(f0).max()
    assert np.abs(f1).max() <= 5e-6, np.abs(f1).max()

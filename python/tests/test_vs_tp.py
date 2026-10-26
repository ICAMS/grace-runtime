# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Parity with tensorpotential's TPCalculator on the -fp64 SavedModels (GRACE-3L: its fp32 SavedModel,
there is no -fp64 one, so both sides are fp32 and the thresholds are wider).

Skipped without tensorpotential / the cached models. Run on CPU: CUDA_VISIBLE_DEVICES= pytest -s.
Energy <= 1e-6 eV/atom (2e-6 below 10 atoms: fp32 floor), forces <= 2e-5 eV/A, stress <= 2e-5 relative to
max(|stress|, 1e-3 eV/A^3), per-atom energies <= 1e-5 eV. Measured worst (CPU, 2026-10-20): stress 6.8e-6 (1L, W1),
per-atom energy 3.1e-6 eV (1L, W54).
"""
import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.spacegroup import crystal

tp = pytest.importorskip("tensorpotential.calculator.asecalculator")
from grace_runtime import GRACERuntimeCalculator, Model  # noqa: E402

CACHE = os.path.expanduser(os.environ.get("GRACE_CACHE", "~/.cache/grace"))
MODELS = [  # (name, kokkos.npz of the fp32 model, fp64 SavedModel)
    ("1L-OMAT-large-base", f"{CACHE}/GRACE-1L-OMAT-large-base/kokkos.npz", f"{CACHE}/GRACE-1L-OMAT-large-base-fp64"),
    ("2L-OMAT-medium-ft-E", f"{CACHE}/GRACE-2L-OMAT-medium-ft-E/kokkos.npz", f"{CACHE}/GRACE-2L-OMAT-medium-ft-E-fp64"),
    ("2L-SMAX-OMAT-medium", f"{CACHE}/GRACE-2L-SMAX-OMAT-medium/kokkos.npz", f"{CACHE}/GRACE-2L-SMAX-OMAT-medium-fp64"),
    ("3L-OMAT-large", f"{CACHE}/GRACE-3L-OMAT-large/kokkos.npz", f"{CACHE}/GRACE-3L-OMAT-large"),    # fp32 reference
]
TOL = {"3L-OMAT-large": (2e-6, 2e-6, 5e-5)}    # (dE/atom >= 10 atoms, below, dF); default (1e-6, 2e-6, 2e-5); 3L measured
# dE/atom <= 2.2e-7 (C64; 9.6e-7 for the 1-atom W1), dF <= 1.0e-5 against the fp32 SavedModel
TOL_S, TOL_E_ATOM = 2e-5, 1e-5    # stress (relative, see the module doc), per-atom energy (eV); all models
HEA25 = ["Al", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd",
         "Ag", "Hf", "Ta", "W", "Re", "Os", "Ir"]


def structures():
    rng = np.random.default_rng(7)
    out = {}
    w = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    w.rattle(0.05, seed=1)
    out["W54"] = w
    sto = crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                  cellpar=[3.905] * 3 + [90] * 3).repeat(2)
    sto.rattle(0.03, seed=2)
    out["SrTiO3-40"] = sto
    hea = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    hea.set_chemical_symbols(rng.permutation(np.resize(HEA25, len(hea))))
    hea.rattle(0.03, seed=3)
    out["HEA25-54"] = hea
    c = bulk("C", "diamond", a=3.567, cubic=True).repeat(2)
    c.rattle(0.03, seed=4)
    out["C64"] = c
    tri = bulk("W", "bcc", a=3.165).repeat((2, 1, 1))
    tri.rattle(0.03, seed=5)
    out["W2tri"] = tri
    out["W1"] = bulk("W", "bcc", a=3.165)
    out["H2O"] = molecule("H2O")
    slab = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 3))
    slab.center(vacuum=8.0, axis=2)
    slab.pbc = (True, True, False)
    slab.rattle(0.03, seed=6)
    out["Al-slab"] = slab
    return out


@pytest.mark.parametrize("name,npz,saved", MODELS, ids=[m[0] for m in MODELS])
def test_parity(name, npz, saved):
    if not (os.path.exists(npz) and os.path.isdir(saved)):
        pytest.skip(f"{name}: cached model missing")
    ref = tp.TPCalculator(saved, extra_properties=["atomic_energy"], enable_uq_if_available=False)
    nat = GRACERuntimeCalculator(Model(npz), num_threads=4)
    worst = {"dE": 0.0, "dF": 0.0, "dS": 0.0, "de": 0.0}
    for sname, at in structures().items():
        a, b = at.copy(), at.copy()
        a.calc, b.calc = ref, nat
        n = len(at)
        dE = abs(a.get_potential_energy() - b.get_potential_energy()) / n
        dF = np.abs(a.get_forces() - b.get_forces()).max()
        sa, sb = a.get_stress(), b.get_stress()
        dS = np.abs(sa - sb).max() / max(np.abs(sa).max(), 1e-3)
        de = np.abs(np.asarray(ref.results["atomic_energy"]).reshape(-1)[:n] - b.get_potential_energies()).max()
        print(f"{name:22s} {sname:10s} dE/n {dE:.2e}  dF {dF:.2e}  dS rel {dS:.2e}  de_i {de:.2e}")
        for k, v in (("dE", dE), ("dF", dF), ("dS", dS), ("de", de)):
            worst[k] = max(worst[k], v)
        # fp32 kernels: per-atom energy errors do not average out in cells of a few atoms (2-atom triclinic W, ~130
        # self-image bonds per atom: the LAMMPS fused style itself is 1.0e-6 eV/atom off the fp64 model there)
        tE, tE_small, tF = TOL.get(name, (1e-6, 2e-6, 2e-5))
        assert dE <= (tE if n >= 10 else tE_small), (sname, dE)
        assert dF <= tF, (sname, dF)
        assert dS <= TOL_S, (sname, dS)
        assert de <= TOL_E_ATOM, (sname, de)
    print(f"{name:22s} worst: " + "  ".join(f"{k} {v:.2e}" for k, v in worst.items()))


UQ_MODELS = MODELS[:2] + MODELS[3:]    # the SMAX-OMAT-medium kokkos.npz carries no UQ arrays
UQ_TOL = {"3L-OMAT-large": 3e-4}    # fp32 reference: measured rel dgamma <= 9.7e-5 (SrTiO3-40), dtotal <= 5e-6


@pytest.mark.parametrize("name,npz,saved", UQ_MODELS, ids=[m[0] for m in UQ_MODELS])
def test_uq_parity(name, npz, saved):
    """gamma_only: gamma, atomic_sigma, gmm_cluster, total_sigma against TPCalculator.enable_uq("gamma_only") (fp64)."""
    if not (os.path.exists(npz) and os.path.isdir(saved)):
        pytest.skip(f"{name}: cached model missing")
    ref = tp.TPCalculator(saved)
    ref.enable_uq(mode="gamma_only")
    nat = GRACERuntimeCalculator(Model(npz), num_threads=4, uq=True)
    worst = {"dgamma": 0.0, "dsigma": 0.0, "dtotal": 0.0}
    for sname, at in structures().items():
        a, b = at.copy(), at.copy()
        a.calc, b.calc = ref, nat
        n = len(at)
        a.get_potential_energy()
        b.get_potential_energy()
        ra, rb = ref.results, nat.results
        ga, sa = np.asarray(ra["gamma"]).reshape(-1)[:n], np.asarray(ra["atomic_sigma"]).reshape(-1)[:n]
        ca = np.asarray(ra["gmm_cluster"]).reshape(-1)[:n]
        same = ca == rb["gmm_cluster"]
        dg = np.max(np.abs(ga - rb["gamma"])[same] / ga[same]) if same.any() else 0.0
        ds = np.max(np.abs(sa - rb["atomic_sigma"])[same] / sa[same]) if same.any() else 0.0
        dt = abs(float(np.sum(ra["total_sigma"])) - rb["total_sigma"]) / float(np.sum(ra["total_sigma"]))
        print(f"{name:22s} {sname:10s} gamma max {ga.max():8.3f}  rel dgamma {dg:.2e}  dsigma {ds:.2e}  "
              f"dtotal {dt:.2e}  cluster mismatches {int((~same).sum())}/{n}")
        for k, v in (("dgamma", dg), ("dsigma", ds), ("dtotal", dt)):
            worst[k] = max(worst[k], v)
        assert same.all(), (sname, np.nonzero(~same))
        t = UQ_TOL.get(name, 1e-5)    # 1L/2L: measured <= 1.3e-6 (fp32 basis vs fp64)
        assert dg <= t and ds <= t and dt <= t, (sname, dg, ds, dt)
    print(f"{name:22s} UQ worst: " + "  ".join(f"{k} {v:.2e}" for k, v in worst.items()))

# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Cells of the parity gates (test_vs_tp.py, test_compressed.py and their stored TensorFlow references).

Compressed cells test the short-bond part of the radial functions, the steep inner region that the equilibrium cells
never reach. Each base cell is scaled isotropically to a volume fraction of 0.8, 0.65 and 0.5 (the nearest-neighbour
distance by 0.93, 0.87 and 0.79), then rattled; dimers in vacuum go further in."""
import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.spacegroup import crystal

VFRAC = (0.8, 0.65, 0.5)
HEA25 = ["Al", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd",
         "Ag", "Hf", "Ta", "W", "Re", "Os", "Ir"]


def _bases():
    rng = np.random.default_rng(11)
    w = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    wmo = w.copy()
    wmo.set_chemical_symbols(rng.permutation(["W", "Mo"] * 27))
    hea = w.copy()
    hea.set_chemical_symbols(rng.permutation(np.resize(HEA25, len(hea))))
    c = bulk("C", "diamond", a=3.567, cubic=True).repeat(2)
    sto = crystal(["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                  cellpar=[3.905] * 3 + [90] * 3).repeat(2)
    return {"W54": w, "WMo54": wmo, "HEA25-54": hea, "C64": c, "SrTiO3-40": sto}


def compressed_structures():
    """{name: Atoms}: <base>-v<percent of the volume> and dimers <pair>-<distance in A>"""
    out = {}
    for k, (name, base) in enumerate(_bases().items()):
        for f in VFRAC:
            at = base.copy()
            at.set_cell(at.cell * f ** (1 / 3), scale_atoms=True)
            at.rattle(0.03, seed=100 + 10 * k + int(100 * f))
            out[f"{name}-v{int(round(100 * f))}"] = at
    for sym, d in (("W2", 1.6), ("W2", 2.0), ("WMo", 1.8), ("C2", 1.0), ("OH", 0.7)):
        out[f"{sym}-{d}"] = Atoms(sym, positions=[(0, 0, 0), (d, 0, 0)], cell=[20, 20, 20], pbc=False)
    return out


def equilibrium_structures():
    """{name: Atoms}: the equilibrium cells of test_vs_tp.py"""
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

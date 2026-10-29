# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Model.compute with positions and a table of shifts (the engines form r_ij = x[j] - x[i] + shifts[image]) gives the
results of the caller's r_ij (NeighborList.rij) bit for bit, for the GRACE drivers and PACE. One thread: with more,
the per-thread energy sums depend on the dynamic schedule (round-off from run to run, either way)."""
import os

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read

from grace_runtime import Model
from grace_runtime.neighbors import NeighborList
from pace_tools import FX


def wmo54(fixtures):
    at = read(f"{fixtures}/WMo54.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})
    at.rattle(0.03, seed=7)
    return at


def both(m, at, species):
    nl = NeighborList(float(m.cutoff_matrix(species).max()), 0.5, num_threads=1)
    nl.update(at)
    assert np.any(nl.shifts[nl.image] != 0.0)    # bonds across the cell
    lut = {s: k + 1 for k, s in enumerate(species)}
    types = np.array([lut[s] for s in np.array(at.get_chemical_symbols())[nl.perm]], dtype=np.int32)
    args = (species, types, nl.offset, nl.j)
    kw = dict(energies=True, num_threads=1)
    ref = m.compute(*args, nl.rij(at, 1), **kw)
    out = m.compute(*args, positions=at.positions[nl.perm], image=nl.image, shifts=nl.shifts, **kw)
    return ref, out, (args, nl, at)


def assert_same(ref, out):
    assert sorted(ref) == sorted(out)
    for k in ref:
        np.testing.assert_array_equal(np.asarray(out[k]), np.asarray(ref[k]), err_msg=k)


@pytest.mark.parametrize("arch", ["1l", "2l", "3l"])
def test_grace_positions_shift_same_as_rij(models, fixtures, arch):
    assert_same(*both(Model(models[arch]), wmo54(fixtures), ["Mo", "W"])[:2])


def test_pace_positions_shift_same_as_rij():
    at = bulk("Pd", "fcc", a=3.9, cubic=True) * (3, 3, 3)
    at.symbols[::5] = "H"
    at.rattle(0.05, seed=3)
    assert_same(*both(Model(os.path.join(FX, "pdh_zbl.yaml")), at, ["H", "Pd"])[:2])


def test_bad_bond_source(models, fixtures):
    m = Model(models["1l"])
    _, _, (args, nl, at) = both(m, wmo54(fixtures), ["Mo", "W"])
    x, rij, im, sh = at.positions[nl.perm], nl.rij(at, 1), nl.image, nl.shifts
    msg = "rij or positions, image and shifts"
    with pytest.raises(ValueError, match=msg):
        m.compute(*args, rij, positions=x, image=im, shifts=sh)    # both
    with pytest.raises(ValueError, match=msg):
        m.compute(*args)                                          # neither
    with pytest.raises(ValueError, match=msg):
        m.compute(*args, positions=x, shifts=sh)                  # no image
    with pytest.raises(ValueError, match="image"):
        m.compute(*args, positions=x, image=im[:-1], shifts=sh)   # an image per bond
    bad = im.copy(); bad[0] = len(sh)
    with pytest.raises(ValueError, match="image"):
        m.compute(*args, positions=x, image=bad, shifts=sh)       # a row of shifts
    with pytest.raises(ValueError, match="positions"):
        m.compute(*args, positions=x[:-1], image=im, shifts=sh)   # a position per atom

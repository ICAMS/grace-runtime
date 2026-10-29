# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""NeighborList: bonds equal ASE's neighbour list, skin rebuild criterion, spatial order, non-periodic cells."""
import numpy as np
from ase import Atoms
import pytest
from ase.build import bulk, fcc111, graphene, molecule
from ase.neighborlist import neighbor_list

from grace_runtime.neighbors import NeighborList


def bonds_original(nl, at, rc):
    """(i, j) in original indices and distances, for bonds shorter than rc."""
    r = np.linalg.norm(nl.rij(at), axis=1)
    i_new = np.repeat(np.arange(len(at)), np.diff(nl.offset))
    keep = r < rc
    return nl.perm[i_new][keep], nl.perm[nl.j][keep], r[keep]


def assert_same_bonds(nl, at, rc, ref=None):
    """ref: the Atoms ASE's list is made of (default at)"""
    i, j, d = bonds_original(nl, at, rc)
    ri, rj, rd = neighbor_list("ijd", at if ref is None else ref, rc)
    a = np.lexsort((np.round(d, 6), j, i))
    b = np.lexsort((np.round(rd, 6), rj, ri))
    assert np.array_equal(i[a], ri[b]) and np.array_equal(j[a], rj[b])
    assert np.allclose(d[a], rd[b], atol=1e-10)


def test_bonds_match_ase_periodic():
    at = bulk("W", "bcc", a=3.165, cubic=True).repeat((3, 3, 3))
    at.rattle(0.05, seed=1)
    nl = NeighborList(5.0, skin=0.5)
    nl.update(at)
    assert_same_bonds(nl, at, 5.0)


def test_small_triclinic_cell():
    at = bulk("W", "bcc", a=3.165)        # one atom: every bond is a self-image
    nl = NeighborList(5.0, skin=0.0)
    nl.update(at)
    assert len(at) == 1 and len(nl.j) > 20 and (nl.j == 0).all()
    assert_same_bonds(nl, at, 5.0)


def test_molecule_without_cell():
    at = molecule("CH3CH2OH")             # zero cell, pbc False
    assert at.cell.rank == 0
    nl = NeighborList(4.0, skin=0.3)
    nl.update(at)
    assert_same_bonds(nl, at, 4.0)


def test_slab_mixed_pbc():
    at = bulk("Al", "fcc", a=4.05, cubic=True).repeat((2, 2, 3))
    at.center(vacuum=8.0, axis=2)
    at.pbc = (True, True, False)
    nl = NeighborList(5.0, skin=0.4)
    nl.update(at)
    assert_same_bonds(nl, at, 5.0)


@pytest.mark.parametrize("make", [
    lambda: graphene(size=(3, 3, 1)),                     # hexagonal layer, zero third vector
    lambda: fcc111("Al", size=(2, 2, 3)),                 # 60 degree in-plane cell, zero third vector
    lambda: Atoms("W4", positions=[[0, 0, 0], [1.4, 1.5, 0.2], [2.9, 3.0, -0.1], [4.3, 4.4, 0.3]],
                  cell=[[5.8, 6.0, 0.4], [0, 0, 0], [0, 0, 0]], pbc=(True, False, False)),   # oblique wire
], ids=["graphene", "fcc111", "oblique_wire"])
def test_non_orthogonal_low_rank_cells(make):
    at = make()
    assert at.cell.rank < 3
    nl = NeighborList(4.5, skin=0.3)
    nl.update(at)
    assert_same_bonds(nl, at, 4.5)


@pytest.mark.parametrize("cz", [1e-2, 1e-4, 1e-6])
def test_open_vector_near_the_periodic_plane(cz):
    """an open cell vector nearly in the plane of the periodic ones: the bins of the periodic directions were
    about cz thick, and the search visited ~1e10 of them (no result after minutes for cz 1e-4); the bonds do not
    depend on the open vector: the reference is ASE's list with it normal to the periodic ones (ASE's list takes
    minutes on the near-flat cell)"""
    at = bulk("W", "bcc", a=3.165, cubic=True).repeat((2, 2, 1))
    at.rattle(0.05, seed=4)
    at.set_cell([at.cell[0], at.cell[1], [5.0, 5.0, cz]], scale_atoms=False)
    at.pbc = (True, True, False)
    nl = NeighborList(4.5, skin=0.3)
    nl.update(at)
    ref = at.copy()
    ref.set_cell([at.cell[0], at.cell[1], [0.0, 0.0, 20.0]], scale_atoms=False)
    assert_same_bonds(nl, at, 4.5, ref)


def test_skin_rebuild():
    at = bulk("W", "bcc", a=3.165, cubic=True).repeat(2)
    nl = NeighborList(5.0, skin=0.5)
    assert nl.update(at)
    at.positions[0] += [0.2, 0, 0]
    assert not nl.update(at)              # 0.2 <= skin/2
    assert_same_bonds(nl, at, 5.0)        # the kept list still holds every bond within the cutoff
    at.positions[0] += [0.1, 0, 0]
    assert nl.update(at)                  # 0.3 > skin/2
    at.set_cell(at.cell * 1.01, scale_atoms=True)
    assert nl.update(at)                  # cell changed
    at.numbers[1] = 42
    assert nl.update(at)                  # composition changed


def test_spatial_order_is_a_permutation():
    at = bulk("W", "bcc", a=3.165, cubic=True).repeat(4)
    at.rattle(0.1, seed=2)
    at = at[np.random.default_rng(0).permutation(len(at))]
    nl = NeighborList(5.0)
    nl.update(at)
    assert sorted(nl.perm.tolist()) == list(range(len(at)))
    assert (nl.inv[nl.perm] == np.arange(len(at))).all()
    # neighbours in the new order are close in index: a bin holds nearby atoms
    i_new = np.repeat(np.arange(len(at)), np.diff(nl.offset))
    assert np.median(np.abs(i_new - nl.j)) < len(at) / 4


def test_empty_and_isolated():
    at = Atoms("W", positions=[[0, 0, 0]], cell=[20, 20, 20], pbc=True)
    nl = NeighborList(5.0)
    nl.update(at)
    assert len(nl.j) == 0 and nl.offset.tolist() == [0, 0]


@pytest.mark.parametrize("seed", range(6))
def test_random_cells(seed):
    """Random triclinic cells, mixed pbc, atoms outside the cell, cutoffs larger than the cell."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    cell = np.diag(rng.uniform(2.5, 9.0, 3)) + rng.uniform(-2.0, 2.0, (3, 3)) * (1 - np.eye(3))
    pbc = rng.integers(0, 2, 3).astype(bool) if seed else np.array([True, True, True])
    frac = rng.uniform(-0.3, 1.3, (n, 3))
    at = Atoms(numbers=rng.integers(1, 80, n), scaled_positions=frac, cell=cell, pbc=pbc)
    rc = float(rng.uniform(2.0, 7.0))
    nl = NeighborList(rc, skin=0.2)
    nl.update(at)
    assert_same_bonds(nl, at, rc)


@pytest.mark.parametrize("cell", [
    [[5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 0.0]],
])
def test_low_rank_cell_rejected(cell):
    """Collinear periodic cell vectors are rejected by the neighbour search, not searched with a huge inverse"""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    at = Atoms("W2", positions=pos, cell=cell, pbc=(True, True, False))
    with pytest.raises(ValueError, match="full-rank cell"):
        NeighborList(4.5, skin=0.3).update(at)


@pytest.mark.parametrize("cell", [
    [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-0.5, -1.0, -1.5]],    # open vector collinear with the periodic one
    [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 5.0, 1e-13]],    # open vector in the periodic plane: det 1.4e-14 |a||b||c|
])
def test_degenerate_open_vector_ignored(cell):
    """Open cell vectors (and zero ones) are replaced by padded ones: a cell degenerate only through them is
    searched, with ASE's bonds of the cell whose open vectors are normal to the periodic ones"""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    at = Atoms("W2", positions=pos, cell=cell, pbc=(True, True, False))
    nl = NeighborList(4.5, skin=0.3)
    nl.update(at)
    per = at.pbc & np.any(at.cell.array != 0.0, axis=1)
    ref = at.copy()
    ref.pbc = per
    ref.cell[~per] = 20.0 * np.linalg.svd(at.cell.array[per])[2][per.sum():]
    assert_same_bonds(nl, at, 4.5, ref)


def test_shifts_are_a_table():
    """per bond only j and an image index into a table of the distinct S cell (8 bytes per bond, not 36 with the
    central atom, the neighbour's original index and a shift vector)"""
    at = bulk("W", "bcc", a=3.165, cubic=True).repeat(4)
    at.rattle(0.05, seed=4)
    nl = NeighborList(5.0, skin=0.5)
    nl.update(at)
    assert nl.image.dtype == np.int32 and nl.image.shape == nl.j.shape
    assert nl.image.min() == 0 and nl.image.max() == len(nl.shifts) - 1 and len(nl.shifts) <= 27
    assert nl.shifts.shape[1] == 3 and len(np.unique(nl.shifts, axis=0)) == len(nl.shifts)
    assert any(not s.any() for s in nl.shifts)    # the bonds inside the cell


def test_unwrapped_positions():
    """atoms up to 1000 cells away from the cell (an unwrapped trajectory): the bonds of the wrapped atoms"""
    at = bulk("W", "bcc", a=3.165, cubic=True).repeat(3)
    at.rattle(0.05, seed=5)
    far = at.copy()
    far.positions += np.random.default_rng(5).integers(-1000, 1001, (len(at), 3)) @ at.cell.array
    nl = NeighborList(5.0, skin=0.5)
    nl.update(far)
    i, j, d = bonds_original(nl, far, 5.0)
    ri, rj, rd = neighbor_list("ijd", at, 5.0)
    a, b = np.lexsort((np.round(d, 6), j, i)), np.lexsort((np.round(rd, 6), rj, ri))
    assert np.array_equal(i[a], ri[b]) and np.array_equal(j[a], rj[b])
    assert np.allclose(d[a], rd[b], atol=1e-8)    # positions ~1e4 A: round-off of x_j - x_i

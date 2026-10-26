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


def assert_same_bonds(nl, at, rc):
    i, j, d = bonds_original(nl, at, rc)
    ri, rj, rd = neighbor_list("ijd", at, rc)
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


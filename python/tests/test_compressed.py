# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Parity gate on compressed cells (gate_cells.py: 80 / 65 / 50 % of the volume, dimers in vacuum) and the
equilibrium cells of test_vs_tp.py, against stored TensorFlow references (tests/fixtures/ref_tf/compressed_tf.npz,
tensorpotential's TPCalculator on CPU): the fp64 SavedModels (GRACE-3L: its fp32 one, there is no fp64 one)
and, for 1L / 2L, the fp32 SavedModels as the yardstick of an fp32 evaluation.

Symmetric relative errors 2|a - b| / (|a| + |b| + scale) of the total energy, the forces (the cell's forces as one
vector) and the stress, the scale (1 eV, 1 eV/A, 0.01 eV/A^3) making them absolute where a quantity is ~0; per-atom
energies in eV (they cross zero under compression). Bound per cell and quantity: the larger of a
floor and 3 x the error of TensorFlow's fp32 SavedModel on that cell; GRACE-3L: the floor x 3. CPU always, the GPU when
there is one."""
import os
from pathlib import Path

import numpy as np
import pytest

from gate_cells import compressed_structures, equilibrium_structures
from gpu_tools import GPU, NO_GPU, gpu_ok
from grace_runtime import GRACERuntimeCalculator, Model

CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
REF = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "ref_tf" / "compressed_tf.npz"
MODELS = ["GRACE-1L-OMAT-large-base", "GRACE-2L-OMAT-medium-ft-E", "GRACE-2L-SMAX-OMAT-medium",
          "GRACE-2L-SMAX-OMAT-large", "GRACE-3L-OMAT-large"]
FLOOR = {"E": 5e-6, "F": 2e-5, "S": 2e-5, "Ei": 2e-5}    # relative (Ei: eV)
SCALE = {"E": 1.0, "F": 1.0, "S": 1e-2}    # eV, eV/A, eV/A^3 in the denominator: absolute errors where a quantity is ~0
FACTOR = 3.0
DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not gpu_ok(), reason=NO_GPU))]


def smre(a, b, scale):
    """2 |a - b| / (|a| + |b| + scale) of the vectors a, b (the scale keeps it finite where both are ~0: W1's forces)"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return 2 * np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + scale)


def errors(res, ref):
    """{quantity: error} of one result against one reference (dicts energy / forces / stress / energies)"""
    return {**{q: smre(res[k], ref[k], SCALE[q]) for q, k in (("E", "energy"), ("F", "forces"), ("S", "stress"))},
            "Ei": float(np.abs(res["energies"] - ref["energies"]).max())}


def bounds(ref, ref32):
    """{quantity: bound}: max(floor, FACTOR x the TF fp32 error), the floor x FACTOR without a TF fp32 reference"""
    if ref32 is None:
        return {q: FACTOR * f for q, f in FLOOR.items()}
    e32 = errors(ref32, ref)
    return {q: max(FLOOR[q], FACTOR * e32[q]) for q in FLOOR}


def _ref(npz, src, model, cell):
    k = f"{src}/{model}/{cell}"
    if f"{k}/energy" not in npz:
        return None
    return {q: npz[f"{k}/{q}"] for q in ("energy", "forces", "stress", "energies")}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("model", MODELS)
def test_compressed_parity(model, device):
    path = CACHE / model / "kokkos.npz"
    if not path.exists():
        pytest.skip(f"{model} not in the test cache")
    npz = np.load(REF)
    calc = GRACERuntimeCalculator(Model(str(path)), device=GPU if device == "gpu" else "cpu", num_threads=4)
    over = []
    for name, at in {**equilibrium_structures(), **compressed_structures()}.items():
        ref = _ref(npz, "tf64", model, name)
        a = at.copy()
        a.calc = calc
        res = {"energy": a.get_potential_energy(), "forces": a.get_forces(),
               "stress": a.get_stress() if a.pbc.all() else np.zeros(6), "energies": a.get_potential_energies()}
        err, bnd = errors(res, ref), bounds(ref, _ref(npz, "tf32", model, name))
        print(f"{model} {device} {name:14s} " + "  ".join(f"{q} {err[q]:.1e} (<= {bnd[q]:.1e})" for q in FLOOR))
        over += [f"{name} {q} {err[q]:.2e} > {bnd[q]:.2e}" for q in FLOOR if err[q] > bnd[q]]
    assert not over, over

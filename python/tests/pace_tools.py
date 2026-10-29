# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Helpers of the PACE tests, shared with the generator of their references."""
import functools
import glob
import os

import numpy as np
import yaml
from ase.io import read

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "fixtures", "pace")
MODELS = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(FX, "*.yaml")))    # the fixture yaml

# synthetic: no pacemaker fit uses ChebLinear; it has no envelope (not zero at rcut), so the radial coefficients of
# al_fs_plain are scaled to keep the energies moderate (they need not be physical)
CHEBLINEAR_BASE, CHEBLINEAR_SCALE = "al_fs_plain", 0.1


def cheblinear_variant(src, dst):
    """write to dst the model src with every radial basis replaced by ChebLinear and crad scaled"""
    with open(src) as f:
        d = yaml.safe_load(f)
    for sb in d["species"]:
        if "radbase" in sb:
            sb["radbase"] = "ChebLinear"
            sb["radcoefficients"] = (np.array(sb["radcoefficients"]) * CHEBLINEAR_SCALE).tolist()
    with open(dst, "w") as f:
        yaml.safe_dump(d, f, default_flow_style=None, sort_keys=False)
    return dst


def model_path(name, tmp_dir):
    """the yaml of a fixture model; al_cheblinear: cheblinear_variant written into tmp_dir"""
    if name == "al_cheblinear":
        return str(cheblinear_variant(os.path.join(FX, f"{CHEBLINEAR_BASE}.yaml"), os.path.join(tmp_dir, f"{name}.yaml")))
    return os.path.join(FX, f"{name}.yaml")


def arrays(yaml_path, asi=None):
    """the PACE npz arrays of a B-basis yaml (asi: its active set, .asi), by the converter of libgrace"""
    from grace_runtime._core import pace_arrays
    raw = pace_arrays(os.fspath(yaml_path), "" if asi is None else os.fspath(asi))
    return {k: np.frombuffer(b, dtype=np.dtype(d)).reshape(shape) for k, (d, shape, b) in raw.items()}


def write_npz(yaml_path, npz_path, asi=None):
    """write the PACE npz of a B-basis yaml to npz_path; its path"""
    npz_path = os.fspath(npz_path)
    with open(npz_path, "wb") as f:    # a file object: np.savez would append .npz to other names
        np.savez(f, **arrays(yaml_path, asi))
    return npz_path


@functools.cache
def cells():
    """the fixture cells (info: name "<model>/<cell>", model)"""
    return read(os.path.join(FX, "cells.extxyz"), ":")


def cell(name):
    """a copy of the fixture cell <model>/<cell>"""
    return next(c.copy() for c in cells() if c.info["name"] == name)


def basis_sizes(path):
    """({element: number of B functions}, {element: ndensity}) of a B-basis yaml, counted with PyYAML (independent
    of the converter): the sizes of its active sets"""
    with open(path) as f:
        d = yaml.safe_load(f)
    nd = {b["speciesblock"].strip(): int(b["ndensityi"]) for b in d["species"] if len(b["speciesblock"].split()) == 1}
    nfunc = dict.fromkeys(nd, 0)
    for b in d["species"]:
        for f in b.get("nbody") or []:
            if not f.get("skip", False):
                nfunc[f["type"].split()[0]] += 1
    return nfunc, nd


def active_set_file(path, nfunc, ndensity, grade):
    """write to path an active set (.asi: one square matrix per element name) for the extrapolation-grade tests:
    gamma is defined for any matrix, so a deterministic dense one (no fit, no random stream) of the size of the
    linear (grade 0: nfunc) or non-linear (grade 1: nfunc * ndensity) grade checks the projections and their order.
    nfunc / ndensity: {element: value}"""
    mats = {}
    for e in nfunc:
        n = nfunc[e] * (ndensity[e] if grade else 1)
        j, k = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        mats[e] = np.cos(0.3 + 0.71 * j + 1.37 * k + 0.013 * j * k) / np.sqrt(n)
    with open(path, "wb") as f:    # a file object: np.savez would append .npz to the name
        np.savez(f, **mats)
    return path

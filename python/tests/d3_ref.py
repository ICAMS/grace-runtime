# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""The DFT-D3(BJ) references of tests/fixtures/d3_ref.npz (simple-dftd3; two-body and three-body cases) as ASE
structures and D3 options."""
from pathlib import Path

import numpy as np
from ase import Atoms

from grace_runtime import D3

FILE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "d3_ref.npz"


def cases():
    """list of dicts: name, atoms, d3 (grace_runtime.D3 of the case), energy, energies, forces, stress (ASE Voigt)"""
    d = np.load(FILE)
    out = []
    for k in range(int(d["ncase"])):
        p = d[f"{k}/params"]
        rc, width, rcn = d[f"{k}/cutoffs"]
        s9, rc3, w3 = d[f"{k}/atm"]
        at = Atoms(numbers=d[f"{k}/numbers"], positions=d[f"{k}/positions"], cell=d[f"{k}/cell"], pbc=d[f"{k}/pbc"])
        out.append({"name": str(d[f"{k}/name"]), "functional": str(d[f"{k}/functional"]), "atoms": at,
                    "d3": D3(a1=p[0], a2=p[1], s6=p[2], s8=p[3], cutoff=rc, switch_width=width, cn_cutoff=rcn,
                             atm=bool(s9), atm_cutoff=rc3, atm_switch_width=w3),
                    "energy": float(d[f"{k}/energy"]), "energies": d[f"{k}/energies"], "forces": d[f"{k}/forces"],
                    "stress": d[f"{k}/stress"]})
    return out

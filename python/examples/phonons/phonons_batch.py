# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Phonons of several crystals with phonons_batch (no torch): the cells relaxed with relax_batch, the displaced
supercells of all crystals (phonopy, finite displacements) evaluated in one batch, then frequencies at the zone centre
and boundary and the vibrational free energy by phonopy (needs `pip install phonopy`).

  python phonons_batch.py --model GRACE-2L-OMAT-medium-base
"""
import argparse
import json
import time

import numpy as np
from ase.build import bulk

from grace_runtime import Model, phonons_batch, relax_batch

CRYSTALS = {    # name: (unit cell, q-point on the zone boundary)
    "Al fcc": (bulk("Al", "fcc", a=4.04), [0.5, 0.0, 0.5]),
    "Cu fcc": (bulk("Cu", "fcc", a=3.61), [0.5, 0.0, 0.5]),
    "W bcc": (bulk("W", "bcc", a=3.17), [0.5, -0.5, 0.5]),
    "Si diamond": (bulk("Si", "diamond", a=5.43), [0.5, 0.0, 0.5]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="GRACE-2L-OMAT-medium-base")
    p.add_argument("--device", default="auto")
    p.add_argument("--min-length", type=float, default=10.0, help="supercell lattice-plane spacing at least (A)")
    p.add_argument("--distance", type=float, default=0.01, help="displacement (A)")
    a = p.parse_args()
    t0 = time.perf_counter()
    model = Model(a.model)
    cells = [c for c, _ in CRYSTALS.values()]
    relaxed = [r["atoms"] for r in relax_batch(model, cells, fmax=1e-4, cell_filter="frechet", device=a.device)]
    phonons = phonons_batch(model, relaxed, min_length=a.min_length, distance=a.distance, device=a.device)
    out = {"model": a.model, "supercells": sum(len(ph.supercells_with_displacements) for ph in phonons),
           "atoms": sum(len(ph.supercells_with_displacements) * len(ph.supercell) for ph in phonons)}
    for (name, (_, q)), ph in zip(CRYSTALS.items(), phonons):
        ph.run_mesh([16, 16, 16])
        ph.run_thermal_properties(temperatures=[300])
        out[name] = {"supercell": np.diag(ph.supercell_matrix).tolist(),
                     "freq_gamma_THz": np.round(ph.get_frequencies([0, 0, 0]), 3).tolist(),
                     "freq_boundary_THz": np.round(ph.get_frequencies(q), 3).tolist(),
                     "F_vib_300K_kJ_mol": round(float(ph.get_thermal_properties_dict()["free_energy"][0]), 3)}
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

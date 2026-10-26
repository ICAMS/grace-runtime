# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""High-throughput relaxation with relax_batch (no torch): solute - vacancy binding in bcc W.

For every solute X (default: 30 elements) one 128-atom W cell with X, one with X and a vacancy in each of the first
`--shells` neighbour shells of X, plus the pure and the single-vacancy cells: all relaxed together (FIRE, positions,
fixed cell); finished cells leave the batch and waiting ones enter it. Binding energy
E_b(X, shell) = E(X) + E(vac) - E(X + vac) - E(bulk) (positive: attraction). The same calculation with TorchSim:
../torchsim/solute_vacancy_relax.py.

  python solute_vacancy_relax.py --model GRACE-2L-OMAT-medium-base --out relax_2l
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.build import bulk

from grace_runtime import relax_batch

SOLUTES = ["Re", "Os", "Ir", "Pt", "Au", "Ta", "Hf", "Nb", "Mo", "V", "Ti", "Zr", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
           "Al", "Si", "Y", "Sc", "Ru", "Rh", "Pd", "Ag", "Zn", "Ga", "Sn", "Be"]
A_W = 3.172    # bcc W lattice constant (A); the cells are not relaxed in volume (dilute limit)


def structures(solutes, shells):
    base = bulk("W", "bcc", a=A_W, cubic=True).repeat(4)
    d = base.get_distances(0, range(len(base)), mic=True)
    radii = np.unique(np.round(d[1:], 3))[:shells]
    site = {k + 1: int(np.flatnonzero(np.isclose(d, r, atol=1e-3))[0]) for k, r in enumerate(radii)}
    out = {("bulk", 0): base.copy()}
    v = base.copy()
    del v[0]
    out[("vac", 0)] = v
    for x in solutes:
        a = base.copy()
        a.symbols[0] = x
        out[(x, 0)] = a
        for sh, j in site.items():
            b = a.copy()
            del b[j]
            out[(x, sh)] = b
    return out, radii


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="GRACE-2L-OMAT-medium-base")
    p.add_argument("--out", default="relax_out")
    p.add_argument("--shells", type=int, default=5)
    p.add_argument("--fmax", type=float, default=0.02)
    p.add_argument("--device", default="auto")
    p.add_argument("--solutes", nargs="+", default=SOLUTES)
    a = p.parse_args()
    cells, radii = structures(a.solutes, a.shells)
    keys = list(cells)
    t0 = time.perf_counter()
    res = relax_batch(a.model, [cells[k] for k in keys], fmax=a.fmax, max_steps=2000, device=a.device)
    wall = time.perf_counter() - t0
    E = {k: r["energy"] for k, r in zip(keys, res)}
    eb = {x: [E[(x, 0)] + E[("vac", 0)] - E[(x, s)] - E[("bulk", 0)] for s in range(1, a.shells + 1)]
          for x in a.solutes}
    result = {"model": a.model, "structures": len(keys), "atoms": sum(len(c) for c in cells.values()),
              "wall_s": round(wall, 1), "fmax": a.fmax, "converged": sum(r["converged"] for r in res),
              "fire_steps": sum(r["n_steps"] for r in res), "shell_radii_A": radii.tolist(),
              "E_vac_formation_eV": E[("vac", 0)] - E[("bulk", 0)] * 127 / 128, "binding_eV": eb}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "binding_eV"}))
    print("X    " + " ".join(f"{s + 1}NN".rjust(7) for s in range(a.shells)))
    for x, v in eb.items():
        print(f"{x:4s} " + " ".join(f"{e:7.3f}" for e in v))


if __name__ == "__main__":
    main()

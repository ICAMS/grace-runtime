# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Temperature scan in one TorchSim batch: thermal expansion and (homogeneous) melting.

`--temps` copies of one crystal (default fcc Cu, 4x4x4 cubic = 256 atoms), each at its own temperature, run NPT
(isotropic Langevin barostat, 0 GPa) side by side. Per cell: volume per atom and mean squared displacement over the
second half of the run. The volume jump and the onset of diffusion mark melting; a perfect crystal without a surface
superheats, so the jump is above the melting point (by ~20 % for metals); a two-phase run gives the melting point.

  python temperature_scan.py --model GRACE-2L-OMAT-medium-base --out tscan [--element Cu --a 3.63 --tmax 2400]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch_sim as ts
from ase.build import bulk
from torch_sim.integrators import npt_langevin_isotropic_init, npt_langevin_isotropic_step
from torch_sim.units import MetalUnits


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="GRACE-2L-OMAT-medium-base")
    p.add_argument("--out", default="tscan_out")
    p.add_argument("--element", default="Cu")
    p.add_argument("--crystal", default="fcc")
    p.add_argument("--a", type=float, default=3.63)
    p.add_argument("--rep", type=int, default=4)
    p.add_argument("--temps", type=int, default=48)
    p.add_argument("--tmin", type=float, default=300.0)
    p.add_argument("--tmax", type=float, default=2400.0)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--dt_fs", type=float, default=2.0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from _models import load
    model, dtype, label = load(a.model)
    dev = torch.device("cuda")
    cell = bulk(a.element, a.crystal, a=a.a, cubic=True).repeat(a.rep)
    nat = len(cell)
    state = ts.io.atoms_to_state([cell.copy() for _ in range(a.temps)], dev, dtype)
    temps = torch.linspace(a.tmin, a.tmax, a.temps, dtype=dtype, device=dev)
    kT = temps * MetalUnits.temperature
    dt = torch.tensor(a.dt_fs * 1e-3 * MetalUnits.time, dtype=dtype, device=dev)
    p0 = torch.tensor(0.0, dtype=dtype, device=dev)
    state.rng = 11
    md = npt_langevin_isotropic_init(state, model, kT=kT, dt=dt)
    half = max(50, a.steps // 100 * 50)    # a sample step (every 50)
    vols, msd = [], []
    x0 = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        md = npt_langevin_isotropic_step(md, model, dt=dt, kT=kT, external_pressure=p0)
        if step % 50 == 0:
            vols.append((torch.det(md.cell) / nat).abs().cpu().numpy())
            if step == half:
                x0, disp = md.positions.clone(), torch.zeros_like(md.positions)
            elif x0 is not None:
                # displacement since the last sample by the nearest image (positions may be wrapped into the cell),
                # accumulated: atoms move far less than half a cell in 50 steps
                h = md.row_vector_cell[md.system_idx]    # [N, 3, 3] rows = cell vectors
                d = md.positions - x0
                f = torch.linalg.solve(h.transpose(1, 2), d.unsqueeze(-1)).squeeze(-1)
                disp += torch.einsum("nk,nkd->nd", f - torch.round(f), h)
                x0 = md.positions.clone()
                # relative to the cell's centre of mass: the Langevin thermostat does not conserve momentum, so the
                # whole crystal drifts (1-3 A in 10 ps) and would read as diffusion
                com = torch.zeros((a.temps, 3), dtype=dtype, device=dev).index_add_(0, md.system_idx, disp) / nat
                d2 = ((disp - com[md.system_idx]) ** 2).sum(dim=1)
                msd.append(torch.zeros(a.temps, dtype=dtype, device=dev).index_add_(0, md.system_idx, d2).cpu()
                           .numpy() / nat)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    vols = np.array(vols)
    v = vols[len(vols) // 2:].mean(axis=0)
    m = np.array(msd)[-1] if msd else np.zeros(a.temps)
    T = temps.cpu().numpy()
    jump = int(np.argmax(np.diff(v))) if a.temps > 1 else 0
    res = {"model": label, "element": a.element, "atoms_per_cell": nat, "cells": a.temps, "steps": a.steps,
           "dt_fs": a.dt_fs, "wall_s": round(wall, 1),
           "us_per_atom_step": round(wall / a.steps / (a.temps * nat) * 1e6, 3),
           "largest_volume_step_between_K": [float(T[jump]), float(T[jump + 1])],
           "T_K": T.tolist(), "volume_A3_per_atom": v.tolist(), "msd_A2_second_half": m.tolist()}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    plot(res, out)
    print(json.dumps({k: res[k] for k in ("model", "element", "wall_s", "us_per_atom_step",
                                          "largest_volume_step_between_K")}))


def plot(r, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T, v, m = np.array(r["T_K"]), np.array(r["volume_A3_per_atom"]), np.array(r["msd_A2_second_half"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].plot(T, v, "o-", ms=3)
    ax[0].set(xlabel="T (K)", ylabel="volume per atom (A^3)", title=f"{r['element']}, {r['model']}: NPT, 0 GPa")
    ax[1].semilogy(T, np.maximum(m, 1e-3), "o-", ms=3)
    ax[1].set(xlabel="T (K)", ylabel="MSD over the second half (A^2)",
              title=f"{r['cells']} cells x {r['atoms_per_cell']} atoms, {r['steps']} steps in {r['wall_s']} s")
    fig.savefig(out / "tscan.png", dpi=130)


if __name__ == "__main__":
    main()

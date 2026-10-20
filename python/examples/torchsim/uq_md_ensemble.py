# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Uncertainty-guided sampling in a TorchSim MD ensemble.

Many NVT Langevin runs in one batch: bcc high-entropy alloy cells on a grid of temperature (T_min .. T_max) and
volume strain (compressed .. expanded lattice); every `--every` steps the GRACE UQ head (uq=True) gives the
extrapolation grade gamma of every atom. Frames whose max gamma exceeds `--gamma` are written to an extxyz file as
candidates for DFT (per-atom gamma in the arrays); the max gamma of each run is mapped over (T, strain). A run
whose largest force exceeds `--fstop` has blown up (the model far outside its data): its frame is kept as a candidate
and the run leaves the batch, the others go on (TorchSim states can be indexed by system).

  python uq_md_ensemble.py --model GRACE-2L-OMAT-medium-ft-E --out uq_out [--nT 16 --nS 8 --steps 4000]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np    # before torch (see the README)
import torch
import torch_sim as ts
from ase.build import bulk
from torch_sim.integrators import nvt_langevin_init, nvt_langevin_step
from torch_sim.units import MetalUnits

from grace_runtime.torchsim import GRACERuntimeModel

ELEMENTS = ["Mo", "Nb", "Ta", "W", "V"]


def cells(strains, rep, seed):
    rng = np.random.default_rng(seed)
    out = []
    for eps in strains:
        a = bulk("W", "bcc", a=3.22 * (1 + eps), cubic=True).repeat(rep)
        a.symbols = rng.choice(ELEMENTS, len(a))    # random solid solution
        a.rattle(0.02, seed=int(rng.integers(1 << 30)))
        out.append(a)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="GRACE-2L-OMAT-medium-ft-E")
    p.add_argument("--out", default="uq_out")
    p.add_argument("--nT", type=int, default=16, help="temperatures")
    p.add_argument("--nS", type=int, default=8, help="volume strains")
    p.add_argument("--smin", type=float, default=-0.15, help="lattice strain of the most compressed cells")
    p.add_argument("--smax", type=float, default=0.15)
    p.add_argument("--rep", type=int, default=4, help="bcc cubic repeats per cell (4: 128 atoms)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--tmin", type=float, default=300.0)
    p.add_argument("--tmax", type=float, default=5000.0)
    p.add_argument("--gamma", type=float, default=1.0, help="max gamma above which a frame is a DFT candidate")
    p.add_argument("--fstop", type=float, default=50.0, help="a run whose largest force exceeds it (eV/A) has blown up")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    model = GRACERuntimeModel(a.model, device=dev, uq=True)
    a.cells = a.nT * a.nS
    tgrid, sgrid = np.meshgrid(np.linspace(a.tmin, a.tmax, a.nT), np.linspace(a.smin, a.smax, a.nS), indexing="ij")
    strains = sgrid.ravel()
    atoms = cells(strains, a.rep, 1)
    state = ts.io.atoms_to_state(atoms, dev, torch.float64)
    temps = torch.tensor(tgrid.ravel(), dtype=torch.float64, device=dev)
    kT = temps * MetalUnits.temperature
    dt = torch.tensor(0.002 * MetalUnits.time, dtype=torch.float64, device=dev)    # 2 fs
    state.rng = 7
    md = nvt_langevin_init(state, model, kT=kT)
    nat = len(atoms[0])
    alive = np.arange(a.cells)    # batch index -> run (runs that blow up leave the batch)
    t_rec, gmax, selected, unstable = [], [], [], {}
    t0 = time.perf_counter()

    def keep(at_s, s_run, step, g_atoms, why):
        at = ts.io.state_to_atoms(md[at_s])[0]
        at.arrays["gamma"] = g_atoms
        at.info.update(T_K=float(temps[s_run]), strain=float(strains[s_run]), time_ps=step * 2e-3,
                       max_gamma=float(g_atoms.max()), run=int(s_run), reason=why)
        selected.append(at)

    for step in range(1, a.steps + 1):
        md = nvt_langevin_step(md, model, dt=dt, kT=kT)
        # blow-up guard every step from the integrator's forces: a run whose largest force exceeds --fstop leaves
        fmax = torch.zeros(len(alive), dtype=md.forces.dtype, device=dev).scatter_reduce(
            0, md.system_idx, md.forces.norm(dim=1), reduce="amax")
        bad = torch.nonzero(fmax > a.fstop).flatten().tolist()
        check = step % a.every == 0
        if not (bad or check):
            continue
        r = model(md)    # the UQ outputs of this frame (the integrator keeps energy / forces / stress only)
        g = r["gamma"].view(len(alive), nat)
        gm = g.max(dim=1).values.cpu().numpy()
        if check:
            row = np.full(a.cells, np.nan)
            row[alive] = gm
            t_rec.append(step * 2e-3)    # ps
            gmax.append(row)
            for s in np.flatnonzero(gm > a.gamma):
                if s not in bad:
                    keep(int(s), alive[s], step, g[s].cpu().numpy(), "gamma")
        if bad:
            for s in bad:
                keep(s, alive[s], step, g[s].cpu().numpy(), "unstable")
                unstable[int(alive[s])] = {"T_K": float(temps[alive[s]]), "strain": float(strains[alive[s]]),
                                           "time_ps": step * 2e-3, "max_force": float(fmax[s]),
                                           "max_gamma": float(gm[s])}
            stay = [s for s in range(len(alive)) if s not in bad]
            if not stay:
                break
            md = md[stay]
            kT = kT[torch.as_tensor(stay, device=dev)]
            alive = alive[stay]
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gmax = np.array(gmax)    # [frames, runs], NaN after a run left
    if selected:
        from ase.io import write
        write(out / "dft_candidates.extxyz", selected)
    summary = {"model": a.model, "cells": a.cells, "atoms_per_cell": nat, "steps": a.steps, "dt_fs": 2.0,
               "wall_s": round(wall, 1), "us_per_atom_step": round(wall / a.steps / (a.cells * nat) * 1e6, 3),
               "candidates": len(selected), "gamma_threshold": a.gamma, "unstable_runs": unstable}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez(out / "gamma_max.npz", t_ps=np.array(t_rec), gamma_max=gmax, T_K=temps.cpu().numpy(), strain=strains)
    plot(out, np.array(t_rec), gmax, temps.cpu().numpy(), strains, a.nT, a.nS, a.gamma, len(selected), unstable)
    print(json.dumps(summary))


def plot(out, t, gmax, temps, strains, nT, nS, thr, ncand, unstable):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    peak = np.nanmax(gmax, axis=0).reshape(nT, nS)
    T = temps.reshape(nT, nS)[:, 0]
    S = strains.reshape(nT, nS)[0] * 100
    im = ax[0].pcolormesh(S, T, peak, shading="nearest", cmap="viridis")
    cs = ax[0].contour(S, T, peak, levels=[thr], colors="w", linewidths=1.5)
    ax[0].clabel(cs, fmt=f"gamma = {thr:g}")
    fig.colorbar(im, ax=ax[0], label="max gamma over the run")
    for u in unstable.values():
        ax[0].plot(u["strain"] * 100, u["T_K"], "rx", ms=10, mew=2)
    if unstable:
        ax[0].plot([], [], "rx", ms=8, mew=2, label=f"{len(unstable)} runs blew up (|F| > 50 eV/A)")
        ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set(xlabel="lattice strain (%)", ylabel="T (K)",
              title=f"{nT * nS} MD runs in one batch: where the model extrapolates")
    cmap = plt.get_cmap("plasma")
    for s in range(gmax.shape[1]):
        ax[1].plot(t, gmax[:, s], color=cmap(abs(strains[s]) / max(np.abs(strains).max(), 1e-9)), lw=0.5, alpha=0.6)
    ax[1].axhline(thr, color="k", ls="--", lw=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, np.abs(strains).max() * 100))
    fig.colorbar(sm, ax=ax[1], label="|lattice strain| (%)")
    ax[1].set(xlabel="time (ps)", ylabel="max gamma in the cell", yscale="log",
              title=f"gamma(t) per run; {ncand} frames -> DFT candidates")
    fig.savefig(out / "gamma_map.png", dpi=130)


if __name__ == "__main__":
    main()

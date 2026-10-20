# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Elastic constants of many elemental crystals in TorchSim: one batched cell relaxation (FIRE with the Frechet cell
filter) of all crystals together, then the elastic tensor of each from batched strained cells
(torch_sim.elastic.calculate_elastic_tensor). Run per model, then compare:

  python elastic_batch.py --model GRACE-2L-OMAT-medium-base --out elastic_2l
  python elastic_batch.py --model GRACE-1L-OMAT-medium-base --out elastic_1l
  python elastic_batch.py --compare elastic_2l elastic_1l --out elastic_compare
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch_sim as ts
from ase.build import bulk
from torch_sim.elastic import calculate_elastic_moduli, calculate_elastic_tensor, get_bravais_type

# element, crystal, a (A), c/a for hcp: starting points only (the cells are relaxed)
CRYSTALS = [("W", "bcc", 3.17), ("Mo", "bcc", 3.15), ("Nb", "bcc", 3.31), ("Ta", "bcc", 3.31), ("V", "bcc", 3.00),
            ("Cr", "bcc", 2.88), ("Fe", "bcc", 2.83), ("Al", "fcc", 4.04), ("Cu", "fcc", 3.63), ("Ni", "fcc", 3.52),
            ("Ag", "fcc", 4.15), ("Au", "fcc", 4.16), ("Pt", "fcc", 3.97), ("Pd", "fcc", 3.95), ("Ir", "fcc", 3.88),
            ("Rh", "fcc", 3.84), ("Ti", "hcp", 2.94, 1.58), ("Zr", "hcp", 3.23, 1.60), ("Hf", "hcp", 3.20, 1.58),
            ("Mg", "hcp", 3.19, 1.62), ("Co", "hcp", 2.50, 1.62), ("Re", "hcp", 2.77, 1.61), ("Ru", "hcp", 2.71, 1.58)]
GPA = 160.21766    # eV/A^3 -> GPa


def crystal(c):
    if c[1] == "hcp":
        return bulk(c[0], "hcp", a=c[2], c=c[2] * c[3])
    return bulk(c[0], c[1], a=c[2], cubic=True)


def run(spec):
    from _models import load
    model, dtype, label = load(spec)
    dev = torch.device("cuda")
    state = ts.io.atoms_to_state([crystal(c) for c in CRYSTALS], dev, dtype)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    relaxed = ts.optimize(state, model, optimizer=ts.Optimizer.fire, max_steps=3000,
                          init_kwargs={"cell_filter": ts.CellFilter.frechet},
                          convergence_fn=ts.generate_force_convergence_fn(force_tol=1e-3, include_cell_forces=True))
    t_relax = time.perf_counter() - t0
    out = {}
    for k, c in enumerate(CRYSTALS):
        s = relaxed[k]
        bt = get_bravais_type(s)
        C = calculate_elastic_tensor(s, model, bravais_type=bt) * GPA
        K, G, nu, _ = calculate_elastic_moduli(C)
        cm = C.double().cpu().numpy()
        out[c[0]] = {"crystal": c[1], "volume_A3_per_atom": float(torch.det(s.cell[0]).abs() / s.n_atoms),
                     "C11": cm[0, 0], "C12": cm[0, 1], "C33": cm[2, 2], "C13": cm[0, 2], "C44": cm[3, 3],
                     "K_VRH": float(K), "G_VRH": float(G), "poisson": float(nu)}
    torch.cuda.synchronize()
    return {"model": label, "relax_s": round(t_relax, 1), "total_s": round(time.perf_counter() - t0, 1),
            "elements": out}


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["C11", "C12", "C44", "K_VRH", "G_VRH"]
    if len(results) == 1:
        r = results[0]
        el = list(r["elements"])
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        w = 0.16
        for i, k in enumerate(keys):
            ax.bar(np.arange(len(el)) + (i - 2) * w, [r["elements"][e][k] for e in el], w, label=k)
        ax.set(xticks=range(len(el)), xticklabels=el, ylabel="GPa",
               title=f"{r['model']}: {len(el)} crystals, relax + elastic tensors in {r['total_s']} s")
        ax.legend(ncol=5)
    else:
        a, b = results
        el = [e for e in a["elements"] if e in b["elements"]]
        fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4), constrained_layout=True)
        for i, k in enumerate(keys):
            x = np.array([a["elements"][e][k] for e in el])
            y = np.array([b["elements"][e][k] for e in el])
            ax[i].scatter(x, y, s=14)
            for e, xx, yy in zip(el, x, y):
                ax[i].annotate(e, (xx, yy), fontsize=7)
            lim = [0, max(x.max(), y.max()) * 1.05]
            ax[i].plot(lim, lim, "k--", lw=1)
            ax[i].set(xlabel=f"{a['model']} (GPa)", ylabel=f"{b['model']} (GPa)", xlim=lim, ylim=lim, aspect="equal",
                      title=f"{k}: MAE {np.abs(x - y).mean():.1f} GPa")
    fig.savefig(out / "elastic.png", dpi=130)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="GRACE-2L-OMAT-medium-base")
    p.add_argument("--out", default="elastic_out")
    p.add_argument("--compare", nargs=2, metavar="DIR")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.compare:
        res = [json.loads((Path(d) / "result.json").read_text()) for d in a.compare]
    else:
        res = [run(a.model)]
        (out / "result.json").write_text(json.dumps(res[0], indent=1))
        print(json.dumps({k: v for k, v in res[0].items() if k != "elements"}))
    plot(res, out)


if __name__ == "__main__":
    main()

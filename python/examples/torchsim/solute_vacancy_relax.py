# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""High-throughput relaxation in TorchSim: solute - vacancy binding in bcc W.

For every solute X (default: 30 elements) one 128-atom W cell with X, one with X and a vacancy in each of the first
`--shells` neighbour shells of X, plus the pure and the single-vacancy cells: all relaxed together (FIRE, positions,
fixed cell) with TorchSim's in-flight autobatcher, which replaces converged cells by waiting ones. Binding energy
E_b(X, shell) = E(X) + E(vac) - E(X + vac) - E(bulk) (positive: attraction). Run it once per model to compare them:

  python solute_vacancy_relax.py --model GRACE-2L-OMAT-medium-base --out relax_2l
  python solute_vacancy_relax.py --model GRACE-1L-OMAT-medium-base --out relax_1l
  python solute_vacancy_relax.py --compare relax_2l relax_1l --out relax_compare
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch_sim as ts
from ase.build import bulk

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


def relax(model_spec, solutes, shells, fmax):
    from _models import load
    model, dtype, label = load(model_spec)
    cells, radii = structures(solutes, shells)
    keys = list(cells)
    state = ts.io.atoms_to_state([cells[k] for k in keys], torch.device("cuda"), dtype)
    scaler = model.max_memory_scaler(state) if hasattr(model, "max_memory_scaler") else 20000.0
    batcher = ts.autobatching.InFlightAutoBatcher(model, memory_scales_with="n_atoms", max_memory_scaler=scaler)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    final = ts.optimize(state, model, optimizer=ts.Optimizer.fire, max_steps=2000, autobatcher=batcher,
                        convergence_fn=ts.generate_force_convergence_fn(force_tol=fmax))
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    e = model(final)["energy"].double().cpu().numpy()
    E = dict(zip(keys, e))
    eb = {x: [float(E[(x, 0)] + E[("vac", 0)] - E[(x, s)] - E[("bulk", 0)]) for s in range(1, shells + 1)]
          for x in solutes}
    return {"model": label, "structures": len(keys), "atoms": int(state.n_atoms), "wall_s": round(wall, 1),
            "fmax": fmax, "shell_radii_A": radii.tolist(), "E_vac_formation_eV": float(
                E[("vac", 0)] - E[("bulk", 0)] * 127 / 128), "binding_eV": eb}


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, ax = plt.subplots(1, n + (n == 2), figsize=(5.5 * (n + (n == 2)), 6), constrained_layout=True)
    ax = np.atleast_1d(ax)
    vmax = max(np.abs(np.array(list(r["binding_eV"].values()))).max() for r in results)
    for k, r in enumerate(results):
        sol = list(r["binding_eV"])
        m = np.array([r["binding_eV"][x] for x in sol])
        im = ax[k].imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax[k].set(yticks=range(len(sol)), yticklabels=sol, xticks=range(m.shape[1]),
                  xticklabels=[f"{i + 1}NN" for i in range(m.shape[1])],
                  title=f"{r['model']}: X-vacancy binding in W (eV)\n{r['structures']} cells relaxed in {r['wall_s']} s")
        fig.colorbar(im, ax=ax[k])
    if n == 2:
        a, b = (np.array([r["binding_eV"][x] for x in results[0]["binding_eV"]]).ravel() for r in results)
        ax[2].scatter(a, b, s=12)
        lim = [min(a.min(), b.min()) - 0.05, max(a.max(), b.max()) + 0.05]
        ax[2].plot(lim, lim, "k--", lw=1)
        ax[2].set(xlabel=f"{results[0]['model']} (eV)", ylabel=f"{results[1]['model']} (eV)", xlim=lim, ylim=lim,
                  title=f"binding energies: MAE {np.abs(a - b).mean() * 1000:.0f} meV", aspect="equal")
    fig.savefig(out / "binding.png", dpi=130)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="GRACE-2L-OMAT-medium-base")
    p.add_argument("--out", default="relax_out")
    p.add_argument("--shells", type=int, default=5)
    p.add_argument("--fmax", type=float, default=0.02)
    p.add_argument("--compare", nargs=2, metavar="DIR", help="plot two finished runs side by side")
    p.add_argument("--solutes", nargs="+", default=SOLUTES)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.compare:
        res = [json.loads((Path(d) / "result.json").read_text()) for d in a.compare]
    else:
        res = [relax(a.model, a.solutes, a.shells, a.fmax)]
        (out / "result.json").write_text(json.dumps(res[0], indent=1))
        print(json.dumps({k: v for k, v in res[0].items() if k != "binding_eV"}))
    plot(res, out)


if __name__ == "__main__":
    main()

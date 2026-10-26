# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""eos_batch, elastic_batch: equation of state and elastic constants of many structures at once. All scaled or strained
cells of all inputs go through one compute_batch (fixed atoms) or one relax_batch (relaxed atoms). Parameters and
result keys follow amstools MurnaghanCalculator / ElasticMatrixCalculator; the elastic constants come from the stress
(slope of stress against strain), not from the energy: the stress keeps its precision with fp32 kernels at small
strains, where energy differences do not."""
import warnings

import numpy as np
from ase import units
from ase.constraints import FixSymmetry
from ase.stress import voigt_6_to_full_3x3_strain

from .batch import compute_batch
from .relax import relax_batch

_EOS_TYPES = ("murnaghan", "birch", "birchmurnaghan", "pouriertarantola", "vinet")    # ase.eos: (E0, B0, B', V0)
_RELAX_EOS = {"none": None, "atoms": {}, "atoms+shape": {"cell_filter": "frechet", "constant_volume": True}}


_RESERVED = ("cell_filter", "scalar_pressure", "hydrostatic_strain", "constant_volume", "mask", "exp_cell_factor",
             "energy_only")    # set by eos_batch / elastic_batch themselves


def _evaluate(name, model, structures, relax, fmax, max_steps, device, kwargs):
    """energy [N] (eV), stress [N, 6] (ASE Voigt, eV/A^3), volume [N] (A^3, after the relaxation), FIRE steps [N] and
    converged [N] of the structures: compute_batch (relax None) or relax_batch with these options; a warning names
    the structures with unconverged points"""
    bad = sorted(set(kwargs) & set(_RESERVED))
    if bad:
        raise ValueError(f"{name} sets {', '.join(bad)} itself")
    if relax is None:
        res = compute_batch(model, structures, device=device, **kwargs)
        vol = [at.get_volume() for at in structures]
    else:
        res = relax_batch(model, structures, fmax=fmax, max_steps=max_steps, device=device, **relax, **kwargs)
        vol = [r["atoms"].get_volume() for r in res]
    return (np.array([r["energy"] for r in res]), np.array([r["stress"] for r in res]).reshape(-1, 6), np.array(vol),
            np.array([r.get("n_steps", 0) for r in res]), np.array([r.get("converged", True) for r in res]))


def _warn_unconverged(name, conv):
    """conv [S, ...]: converged relaxations of each structure's points"""
    bad = [k for k, c in enumerate(conv) if not np.all(c)]
    if bad:
        warnings.warn(f"{name}: structures {bad}: relaxations not converged within max_steps (see 'converged')",
                      stacklevel=3)


def _full_rank(structures, name):
    for k, at in enumerate(structures):
        if at.cell.rank < 3:
            raise ValueError(f"{name}: structure {k} needs a cell of rank 3")


def eos_batch(model, structures, num_of_point: int = 11, volume_range=0.1, relax: str = "none", fit: str = "poly",
              fit_order: int = 5, fmax: float = 0.005, max_steps: int = 500, device: str = "auto", **kwargs) -> list:
    """Energy-volume curve and its fit for each structure (as amstools MurnaghanCalculator).

    model: path to a kokkos.npz, a registry name or a grace_runtime.Model. structures: ASE Atoms with rank-3 cells.
    num_of_point: volumes per structure, from V0 (1 - volume_range) to V0 (1 + volume_range), or from volume_range[0]
        to volume_range[1] (A^3) for a pair (structures with the same number of atoms); cells scaled isotropically
        with the atoms.
    relax: "none" (one compute_batch), "atoms" (positions at fixed cell) or "atoms+shape" (positions and cell shape at
        fixed volume, the Frechet cell filter with constant_volume); fmax (eV/A) and max_steps for the relaxation.
    fit: "poly" (amstools: polynomial of fit_order; the lowest minimum within the volumes; B = V E'', B' = -(V E''' /
        E'' + 1)) or an ase.eos type: "murnaghan", "birch", "birchmurnaghan", "pouriertarantola", "vinet".
    device and other keyword arguments (e.g. dispersion="d3-pbe"): to compute_batch (relax "none") or relax_batch.

    Returns one dict per structure, the points in volume order: "volume" (A^3), "energy" (eV), "pressure" (eV/A^3),
    "equilibrium_volume", "equilibrium_energy", "equilibrium_bulk_modulus" (GPa), "equilibrium_b_prime", "energy_rms"
    (eV, of the fit); relaxed: "n_steps", "converged" per point. No minimum within the volumes: the equilibrium values
    are None (a warning names the structure); unconverged relaxations are kept and warned about.
    """
    if relax not in _RELAX_EOS:
        raise ValueError(f"relax must be one of {', '.join(_RELAX_EOS)}, not {relax!r}")
    if fit != "poly" and fit not in _EOS_TYPES:
        raise ValueError(f"fit must be 'poly' or one of {', '.join(_EOS_TYPES)}, not {fit!r}")
    need = fit_order + 1 if fit == "poly" else 5
    if num_of_point < need:
        raise ValueError(f"num_of_point must be >= {need} for fit={fit!r}" + (f", fit_order={fit_order}"
                                                                                 if fit == "poly" else ""))
    structures = list(structures)
    _full_rank(structures, "eos_batch")
    if np.ndim(volume_range) == 0:
        lo, hi = 1 - volume_range, 1 + volume_range
    else:
        lo, hi = sorted(volume_range)
        if len({len(at) for at in structures}) > 1:
            raise ValueError("eos_batch: a volume pair needs structures with the same number of atoms (or a fraction)")
    scaled = []
    for at in structures:
        v0 = at.get_volume()
        for s in np.linspace(lo, hi, num_of_point) / (1.0 if np.ndim(volume_range) == 0 else v0):
            a = at.copy()
            a.set_cell(at.cell.array * s ** (1.0 / 3.0), scale_atoms=True)
            scaled.append(a)
    e, stress, vol, steps, conv = (x.reshape(len(structures), num_of_point, *x.shape[1:]) for x in
                                   _evaluate("eos_batch", model, scaled, _RELAX_EOS[relax], fmax, max_steps, device,
                                             kwargs))
    if relax != "none":
        _warn_unconverged("eos_batch", conv)
    out = []
    for k in range(len(structures)):
        r = {"volume": vol[k], "energy": e[k], "pressure": -stress[k, :, :3].mean(axis=1)}
        r.update(_fit_eos(vol[k], e[k], fit, fit_order))
        if r["energy_rms"] is None:
            warnings.warn(f"eos_batch: structure {k}: the {fit} fit did not converge", stacklevel=2)
        elif r["equilibrium_volume"] is None:
            warnings.warn(f"eos_batch: structure {k}: no energy minimum within the volumes", stacklevel=2)
        if relax != "none":
            r.update(n_steps=steps[k], converged=conv[k])
        out.append(r)
    return out


def _fit_eos(v, e, fit, fit_order):
    none = {"equilibrium_volume": None, "equilibrium_energy": None, "equilibrium_bulk_modulus": None,
            "equilibrium_b_prime": None}
    if fit == "poly":
        # amstools' polynomial, fitted in the basis of the volumes mapped to [-1, 1] (well conditioned)
        p = np.polynomial.Polynomial.fit(v, e, fit_order)
        rms = float(np.sqrt(np.mean((e - p(v)) ** 2)))
        roots = p.deriv(1).roots()
        x = np.real(roots[np.abs(np.imag(roots)) < 1e-10])
        x = x[(x >= v.min()) & (x <= v.max()) & (p.deriv(2)(x) > 0)]    # minima within the volumes
        if not len(x):
            return {**none, "energy_rms": rms}
        v0 = float(x[np.argmin(p(x))])
        a2, a3 = p.deriv(2)(v0), p.deriv(3)(v0)
        return {"equilibrium_volume": v0, "equilibrium_energy": float(p(v0)),
                "equilibrium_bulk_modulus": float(v0 * a2 / units.GPa),
                "equilibrium_b_prime": float(-(v0 * a3 / a2 + 1)), "energy_rms": rms}
    from ase.eos import EquationOfState
    eos = EquationOfState(v, e, eos=fit)
    try:
        v0, e0, b0 = eos.fit(warn=False)
    except (RuntimeError, ValueError):    # the fit did not converge
        return {**none, "energy_rms": None}
    rms = float(np.sqrt(np.mean((e - eos.func(v, *eos.eos_parameters)) ** 2)))
    if not v.min() <= v0 <= v.max():
        return {**none, "energy_rms": rms}
    return {"equilibrium_volume": float(v0), "equilibrium_energy": float(e0),
            "equilibrium_bulk_modulus": float(b0 / units.GPa), "equilibrium_b_prime": float(eos.eos_parameters[2]),
            "energy_rms": rms}


def _strained(at, j, e):
    """at with Voigt strain j of size e (engineering shear for j >= 3): cell @ (I + eps), atoms scaled; a FixSymmetry
    of the input dropped (it belongs to the unstrained cell and would symmetrise the strain away)"""
    s = at.copy()
    s.set_constraint([c for c in s.constraints if not isinstance(c, FixSymmetry)])
    s.set_cell(at.cell.array @ voigt_6_to_full_3x3_strain(e * np.eye(6)[j]), scale_atoms=True)
    return s


def elastic_batch(model, structures, num_of_point: int = 5, eps_range: float = 0.005, relax_atoms: bool = True,
                  fmax: float = 1e-3, max_steps: int = 500, device: str = "auto", **kwargs) -> list:
    """Elastic constants of each structure from stress against strain (the full 6x6 matrix, any crystal system).

    model: as eos_batch. structures: ASE Atoms with relaxed rank-3 cells (e.g. relax_batch(cell_filter="frechet")
    first; a residual stress shifts the stresses, not their slope).
    num_of_point: strains per Voigt direction in [-eps_range, eps_range] (the unstrained cell computed once); column j
        of C is the slope of a linear least-squares fit of the stress (ASE Voigt) against strain j.
    relax_atoms: the atoms relaxed at each strained cell (relax_batch, fmax eV/A, max_steps; internal relaxation), or
        clamped (one compute_batch). Small strains need a tight fmax: the error of C goes as residual force / strain.
    device and other keyword arguments (e.g. dispersion="d3-pbe"): to relax_batch (relax_atoms) or compute_batch.

    Returns one dict per structure: "C" (6x6, GPa, symmetrised), "asymmetry" (GPa, max |C - C^T| / 2 before that: large
    for unconverged relaxations or too large strains), "S" (inverse of C, 1/GPa), Voigt / Reuss / Hill moduli (GPa)
    "BV", "GV", "EV", "nuV", "BR", "GR", "ER", "nuR", "BH", "GH", "EH", "nuH", "AVR" (100 (GV - GR) / (GV + GR)),
    "C_eigval"; "strains" [num_of_point], "stresses" [6, num_of_point, 6] (GPa); relax_atoms: "n_steps", "converged"
    [6, num_of_point] (unconverged relaxations are kept and warned about). A singular C: the Reuss and Hill values None.
    """
    if num_of_point < 2:
        raise ValueError("num_of_point must be >= 2")
    structures = list(structures)
    _full_rank(structures, "elastic_batch")
    strains = np.linspace(-eps_range, eps_range, num_of_point)
    nz = np.ones(num_of_point, bool)
    if num_of_point % 2:    # the middle strain: exactly 0, the input computed once for all six directions
        strains[num_of_point // 2] = 0.0
        nz[num_of_point // 2] = False
    cells = []
    for at in structures:
        cells += ([at.copy()] if not nz.all() else []) + [_strained(at, j, e) for j in range(6) for e in strains[nz]]
    S, per = len(structures), 6 * nz.sum() + (not nz.all())
    grid = []    # stress, FIRE steps, converged of each structure on the [6, num_of_point] strain grid
    _, stress, _, steps, conv = _evaluate("elastic_batch", model, cells, {} if relax_atoms else None, fmax, max_steps,
                                          device, kwargs)
    for x in (stress, steps, conv):
        x = x.reshape(S, per, *x.shape[1:])
        g = np.empty((S, 6, num_of_point) + x.shape[2:], x.dtype)
        g[:, :, nz] = x[:, per - 6 * nz.sum():].reshape(S, 6, nz.sum(), *x.shape[2:])
        if not nz.all():
            g[:, :, ~nz] = x[:, :1, None]
        grid.append(g)
    stress, steps, conv = grid
    if relax_atoms:
        _warn_unconverged("elastic_batch", conv)
    out = []
    for k in range(S):
        r = _elastic_from_stresses(strains, stress[k] / units.GPa)
        r.update(strains=strains, stresses=stress[k] / units.GPa)
        if relax_atoms:
            r.update(n_steps=steps[k], converged=conv[k])
        out.append(r)
    return out


def _elastic_from_stresses(strains, stresses):
    """C (column j: slope of stresses[j] against the strains), its symmetric part and the polycrystal moduli"""
    C = np.stack([np.polyfit(strains, stresses[j], 1)[0] for j in range(6)], axis=1)
    asym = float(np.abs(C - C.T).max() / 2)
    C = (C + C.T) / 2
    r = {"C": C, "asymmetry": asym}
    BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
    GV = (C[0, 0] + C[1, 1] + C[2, 2] - (C[0, 1] + C[0, 2] + C[1, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
    r.update(BV=BV, GV=GV, **_e_nu(BV, GV, "V"))
    r.update(dict.fromkeys(("S", "BR", "GR", "ER", "nuR", "BH", "GH", "EH", "nuH", "AVR")))    # singular C: None
    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        S = None
    if S is not None:
        BR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
        GR = 15 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
                   + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
        BH, GH = (BV + BR) / 2, (GV + GR) / 2
        r.update(S=S, BR=BR, GR=GR, **_e_nu(BR, GR, "R"), BH=BH, GH=GH, **_e_nu(BH, GH, "H"),
                 AVR=100.0 * (GV - GR) / (GV + GR))
    r["C_eigval"] = np.linalg.eigvalsh(C)
    return r


def _e_nu(B, G, tag):
    """Young's modulus and Poisson's ratio of an isotropic solid with bulk modulus B and shear modulus G"""
    return {f"E{tag}": 9 * B * G / (3 * B + G), f"nu{tag}": (1.5 * B - G) / (3 * B + G)}

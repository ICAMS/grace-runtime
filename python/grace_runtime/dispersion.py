# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Dispersion corrections added to a GRACE model (argument ``dispersion=`` of the calculator, ``compute_batch`` and
the TorchSim model).

``D3``: DFT-D3 with Becke-Johnson (rational) damping, two-body terms C6 and C8 (S. Grimme et al., J. Chem. Phys. 132,
154104 (2010); J. Comput. Chem. 32, 1456 (2011)), as simple-dftd3; the Axilrod-Teller-Muto three-body term on request
(``atm=True``). Reference data of tad-dftd3 / tad-mctc (Apache-2.0). Short forms: ``"d3-<functional>"``, e.g.
``"d3-pbe"`` (two-body), and ``"d3-<functional>-atm"`` (with the three-body term).

Use the functional of the model's training data, and only for data without dispersion: all GRACE foundation models of
the registry are PBE / PBE+U without dispersion, so ``"d3-pbe"``. Do not add it to models trained on data that
already contains dispersion.
"""
import math

from . import _core

_KIND_D3BJ = 1


class D3:
    """DFT-D3(BJ) dispersion: two-body, optionally with the three-body (ATM) term.

    functional, a1, a2, s8, s6: ``"pbe"`` or ``"r2scan"`` (published damping parameters); or None with ``a1``, ``a2``
        (Bohr, as published) and ``s8`` (``s6`` defaults to 1).
    cutoff, switch_width: pair cutoff (Angstrom). The pair terms are switched off smoothly over the last
        ``switch_width`` Angstrom below it (``switch_width=0``: hard cutoff).
    cn_cutoff: cutoff of the coordination numbers (Angstrom).
    atm, atm_cutoff, atm_switch_width: add the Axilrod-Teller-Muto three-body term (s9 = 1, the C6 of the two-body
        term, zero damping with the D3 van der Waals radii, as simple-dftd3). Off by default: the published D3(BJ)
        parameters were fitted without it. ``atm_cutoff`` is the largest side of a triangle (Angstrom); each side is
        switched off smoothly over the last ``atm_switch_width`` Angstrom below it.

    The defaults 15 / 3 / 10 Angstrom miss ~1 meV/atom of the dispersion energy in graphite and ~8 meV/atom in copper
    against simple-dftd3 at its 32 / 21 Angstrom (measured; differences between similar structures largely cancel);
    raise ``cutoff`` for more (docs/dispersion.md).
    """

    kind = _KIND_D3BJ

    def __init__(self, functional: str | None = None, *, a1: float | None = None, a2: float | None = None,
                 s8: float | None = None, s6: float | None = None, cutoff: float = 15.0, switch_width: float = 3.0,
                 cn_cutoff: float = 10.0, atm: bool = False, atm_cutoff: float = 10.0, atm_switch_width: float = 1.0):
        custom = (a1, a2, s8)
        if functional is not None:
            if any(v is not None for v in custom) or s6 is not None:
                raise ValueError("D3: give either a functional or a1, a2, s8 (s6), not both")
            name = str(functional).lower()
            p = _core.d3bj_preset(name)    # raises ValueError for an unknown name
            self.functional = name
            self.s6, self.a1, self.s8, self.a2 = p["s6"], p["a1"], p["s8"], p["a2_bohr"]
        else:
            if any(v is None for v in custom):
                raise ValueError(f"D3: give a functional ({', '.join(_core.d3bj_presets())}) or all of a1, a2, s8")
            self.functional = None
            self.s6 = 1.0 if s6 is None else float(s6)
            self.a1, self.a2, self.s8 = float(a1), float(a2), float(s8)
        self.cutoff, self.switch_width, self.cn_cutoff = float(cutoff), float(switch_width), float(cn_cutoff)
        if not (0.0 < self.cutoff < math.inf and 0.0 < self.cn_cutoff < math.inf):
            raise ValueError("D3: cutoff and cn_cutoff must be positive and finite")
        if not (0.0 <= self.switch_width < self.cutoff):
            raise ValueError("D3: need 0 <= switch_width < cutoff")
        self.atm, self.atm_cutoff, self.atm_switch_width = bool(atm), float(atm_cutoff), float(atm_switch_width)
        if not (0.0 <= self.atm_switch_width < self.atm_cutoff < math.inf):
            raise ValueError("D3: need 0 <= atm_switch_width < atm_cutoff, finite")
        if not (self.a1 >= 0.0 and self.a2 >= 0.0 and self.a1 + self.a2 > 0.0) or not math.isfinite(
                self.a1 + self.a2 + self.s6 + self.s8):
            raise ValueError("D3: need finite parameters, a1, a2 >= 0 and not both zero (the damping radius)")

    @property
    def reach(self) -> float:
        """largest distance (Angstrom) at which atoms interact through the correction"""
        return max(self.cutoff, self.cn_cutoff, self.atm_cutoff if self.atm else 0.0)

    def params(self) -> dict:
        """parameters for the C++ modules (a2 in Angstrom)"""
        return {"kind": self.kind, "s6": self.s6, "s8": self.s8, "a1": self.a1, "a2": self.a2 * _core.BOHR,
                "cutoff": self.cutoff, "cutoff_on": self.cutoff - self.switch_width, "cn_cutoff": self.cn_cutoff,
                "s9": 1.0 if self.atm else 0.0, "atm_cutoff": self.atm_cutoff,
                "atm_cutoff_on": self.atm_cutoff - self.atm_switch_width}

    def __repr__(self):
        which = repr(self.functional) if self.functional else f"a1={self.a1}, a2={self.a2}, s8={self.s8}, s6={self.s6}"
        atm = f", atm=True, atm_cutoff={self.atm_cutoff}, atm_switch_width={self.atm_switch_width}" if self.atm else ""
        return f"D3({which}, cutoff={self.cutoff}, switch_width={self.switch_width}, cn_cutoff={self.cn_cutoff}{atm})"


def resolve(dispersion):
    """None, a D3 object or a short string ("d3-pbe", "d3-r2scan", "d3-pbe-atm", ...) -> None or the object"""
    if dispersion is None or isinstance(dispersion, D3):
        return dispersion
    if isinstance(dispersion, str):
        method, _, functional = dispersion.lower().partition("-")
        if method == "d3" and functional:
            name, _, term = functional.partition("-")
            if term in ("", "atm"):
                return D3(name, atm=term == "atm")
        presets = _core.d3bj_presets()
        if dispersion.lower() in presets:
            raise ValueError(f"dispersion={dispersion!r}: name the method too, e.g. 'd3-{dispersion.lower()}'")
        raise ValueError(f"unknown dispersion {dispersion!r}; use " + ", ".join(f"'d3-{p}'" for p in presets) +
                         " (with '-atm' for the three-body term) or a grace_runtime.D3 object")
    raise TypeError(f"dispersion must be None, a string such as 'd3-pbe' or a grace_runtime.D3, not "
                    f"{type(dispersion).__name__}")

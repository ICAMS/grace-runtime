# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""The example notebook runs: its code cells in order, in one namespace (no Jupyter needed), plots off screen; without
phonopy the cells tagged "phonopy" are left out."""
import importlib.util
import json
import os

import pytest

NOTEBOOK = os.path.join(os.path.dirname(__file__), "..", "examples", "ase", "grace_runtime_ase.ipynb")


def test_ase_notebook(models, tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib", reason="example notebook: no matplotlib")
    pytest.importorskip("ase.filters", reason="example notebook: ASE < 3.23")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have_phonopy = importlib.util.find_spec("phonopy") is not None
    monkeypatch.chdir(tmp_path)
    with open(NOTEBOOK) as f:
        cells = json.load(f)["cells"]
    namespace = {}
    try:
        for i, cell in enumerate(cells):
            source = "".join(cell["source"])
            if cell["cell_type"] != "code" or ("phonopy" in cell["metadata"].get("tags", []) and not have_phonopy):
                continue
            # the test model (the cache or $GRACE_TEST_M2) for the registry name
            source = source.replace('"GRACE-2L-SMAX-OMAT-medium"', repr(models["2l"]))
            exec(compile(source, f"{NOTEBOOK} cell {i}", "exec"), namespace)
    finally:
        plt.close("all")
    assert all(r["converged"] for r in namespace["relax_results"].values())
    assert -0.2 < namespace["e_f"] < 0    # Al3Li L12 is stable against fcc Al + bcc Li
    assert all(r["converged"] for r in namespace["vac_results"].values())
    assert 0.3 < namespace["e_vac_f"] < 1.2    # vacancy formation energy of Al, eV
    gamma_max, factors = namespace["gamma_max"], namespace["volume_factors"]
    assert 0.7 < factors[gamma_max.argmin()] < 1.4    # the model is most at home near the equilibrium volume
    if have_phonopy:    # fcc Al: no imaginary modes, the highest frequency near 9.7 THz
        freqs = namespace["phonons"]["Al", "fcc"].mesh.frequencies
        assert freqs.min() > namespace["imaginary_tol"] and 7 < freqs.max() < 12

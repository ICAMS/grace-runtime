# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Reading and validating pacemaker B-basis yaml (the converter of libgrace, through _core.pace_arrays) on
tests/fixtures/pace and edited copies (written by PyYAML: block style, another style than the fixtures')."""
import os
import re

import numpy as np
import pytest
import yaml

from pace_tools import FX, MODELS, active_set_file, arrays, basis_sizes


def raw(name):
    with open(os.path.join(FX, f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def edited(tmp_path, name, fn=None):
    """a copy of fixture `name` changed by fn(dict), written by PyYAML (block style)"""
    d = raw(name)
    if fn:
        fn(d)
    p = tmp_path / f"{name}.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(d, f)
    return p


def elements(z):
    return [e.decode() for e in z["element_names"]]


def with_asi(path, tmp_path):
    """the arrays with an active set of the sizes PyYAML counts: the converter must count the same functions"""
    nfunc, nd = basis_sizes(path)
    return arrays(path, active_set_file(str(tmp_path / "t.asi"), nfunc, nd, 0)), nfunc


@pytest.mark.parametrize("name", MODELS)
def test_all_fixtures_load(name, tmp_path):
    d = raw(name)
    z, nfunc = with_asi(os.path.join(FX, f"{name}.yaml"), tmp_path)
    els = elements(z)
    singles = sorted(b["speciesblock"].strip() for b in d["species"] if len(b["speciesblock"].split()) == 1)
    assert els == singles    # ML-PACE: element names sorted
    ref = np.load(os.path.join(FX, "ref_pace.npz"))
    assert els == list(ref[f"ctilde/{name}/elements"])    # pyace's element order
    for mu, el in enumerate(els):
        assert z[f"e{mu}_uq_nfunc"][0] == nfunc[el]
        mus, rank = z[f"e{mu}_mus"], z[f"e{mu}_rank"]
        used = mus[np.arange(mus.shape[1])[None, :] < rank[:, None]]    # the factors (not the padding)
        assert all(z["bond_defined"][mu, m] == 1 for m in np.unique(used))    # every pair a product uses has a bond


@pytest.mark.parametrize("name", MODELS)
def test_block_style_same_arrays(name, tmp_path):
    """the PyYAML block style (sequences at their key's indentation, no flow collections) gives the arrays of the
    fixture (the yaml-cpp style of pacemaker)"""
    a, b = arrays(os.path.join(FX, f"{name}.yaml")), arrays(edited(tmp_path, name))
    assert sorted(a) == sorted(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_pairs_without_block_do_not_interact():
    z = arrays(os.path.join(FX, "mnbtw_4el.yaml"))    # unaries only: no pair blocks
    np.testing.assert_array_equal(z["bond_defined"], np.eye(4, dtype=np.int32))


def test_missing_inner_cutoff_is_density():
    assert "inner_cutoff_type" not in str(raw("ethanol_nd1"))
    assert arrays(os.path.join(FX, "ethanol_nd1.yaml"))["inner_cutoff_type"][0] == 0


def test_defaults_of_optional_keys(tmp_path):
    def drop(d):
        for b in d["species"]:
            for k in ("core-repulsion", "rho_core_cut", "drho_core_cut", "r_in", "delta_in"):
                b.pop(k, None)
    z = arrays(edited(tmp_path, "al_fs_plain", drop))
    assert (z["e0_rho_core_cut"][0], z["e0_drho_core_cut"][0]) == (100000.0, 250.0)
    assert [z[f"b0_0_{k}"][0] for k in ("prehc", "lambdahc", "r_in", "delta_in")] == [0.0] * 4


def test_pair_block_copied_from_its_dual(tmp_path):
    def drop_radial(d):    # radial basis only in "H Pd", not in "Pd H"
        for b in d["species"]:
            if b["speciesblock"].split() == ["Pd", "H"]:
                for k in ("nradmaxi", "lmaxi", "nradbaseij", "radbase", "radparameters", "radcoefficients",
                          "rcutij", "dcutij"):
                    b.pop(k, None)
    z = arrays(edited(tmp_path, "pdh_zbl", drop_radial))
    h, pd = elements(z).index("H"), elements(z).index("Pd")
    np.testing.assert_array_equal(z[f"b{pd}_{h}_crad"], z[f"b{h}_{pd}_crad"])


def _set(block_filter, key, value):
    def fn(d):
        for b in d["species"]:
            if block_filter(b):
                b[key] = value
    return fn


ONE = lambda b: len(b["speciesblock"].split()) == 1
PDH = lambda b: b["speciesblock"].split() == ["Pd", "H"]
@pytest.mark.parametrize("name,fn,match", [
    ("al_fs_plain", _set(ONE, "radbase", "TEST_SBessel"), "radbase"),
    ("al_fs_plain", _set(ONE, "npoti", "Foo"), "npoti"),
    ("al_fs_plain", _set(ONE, "NameOfCutoffFunctionij", "exp"), "NameOfCutoffFunctionij"),
    ("al_fs_plain", _set(ONE, "inner_cutoff_type", "foo"), "inner_cutoff_type"),
    ("pdh_zbl", _set(lambda b: b["speciesblock"].split() == ["H", "Pd"], "inner_cutoff_type", "distance"),
     "inner_cutoff_type"),
    ("al_fs_plain", lambda d: d["species"][0].pop("nbody"), "nbody"),
    ("al_fs_plain", lambda d: d["species"][0]["nbody"][0].update(c=[1.0]), "coefficients"),
    ("al_fs_plain", _set(ONE, "radparameters", []), "radparameters"),
    ("al_fs_plain", _set(ONE, "radcoefficients", [[[1.0]]]), "radcoefficients"),
    ("al_fs_plain", _set(ONE, "lmaxi", 12), "radcoefficients"),
    ("pdh_zbl", _set(PDH, "rcutij", 5.5), "rcutij"),
    ("pdh_zbl", _set(PDH, "core-repulsion", [2.0, 1.0]), "core-repulsion"),
    ("al_fs_plain", lambda d: next(f for f in d["species"][0]["nbody"] if len(f["nl"]) == 2).update(nl=[1, 2]),
     "equal ls"),
    ("al_fs_plain", lambda d: d.pop("species"), "species"),
])
def test_unsupported_values_raise(tmp_path, name, fn, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        arrays(edited(tmp_path, name, fn))


def test_syntax_error_names_the_line(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("species:\n  - speciesblock: Al\n    parameters: [1, 2\n")
    with pytest.raises(ValueError, match=re.escape(f"{p}:3: unclosed flow collection")):
        arrays(p)


def test_skip_dropped(tmp_path):
    p = edited(tmp_path, "al_fs_plain", lambda d: d["species"][0]["nbody"][3].update(skip=True))
    z, nfunc = with_asi(str(p), tmp_path)
    assert z["e0_uq_nfunc"][0] == nfunc["Al"] == basis_sizes(os.path.join(FX, "al_fs_plain.yaml"))[0]["Al"] - 1


def _strip_pair(d):    # "Pd H" keeps its functions only: radial basis, core repulsion, inner cutoff come from "H Pd"
    for b in d["species"]:
        if PDH(b):
            for k in [k for k in b if k not in ("speciesblock", "nbody")]:
                del b[k]


def test_pair_block_from_its_dual(tmp_path):
    """as ML-PACE (ace_yaml_input.cpp): a pair block without a radial basis or core repulsion takes them, with the
    inner cutoff, from its dual block"""
    want = arrays(os.path.join(FX, "pdh_zbl.yaml"))
    got = arrays(edited(tmp_path, "pdh_zbl", _strip_pair))
    assert got["inner_cutoff_type"][0] == want["inner_cutoff_type"][0] == 2    # zbl
    pd, h = elements(want).index("Pd"), elements(want).index("H")
    for k in ("r_in", "delta_in", "prehc", "lambdahc", "rcut", "nradmax", "radbase"):
        assert got[f"b{pd}_{h}_{k}"][0] == want[f"b{pd}_{h}_{k}"][0], k


def test_function_on_pair_without_radial_basis(tmp_path):
    def fn(d):
        mo = d["species"][0]
        d["species"].append({"speciesblock": "Mo Nb", "inner_cutoff_type": mo["inner_cutoff_type"], "nbody": [
            {"type": "Mo Nb", "nr": [1], "nl": [0], "c": list(mo["nbody"][0]["c"])}]})    # a copy: no yaml alias
    with pytest.raises(ValueError, match="no radial basis"):
        arrays(edited(tmp_path, "mnbtw_4el", fn))

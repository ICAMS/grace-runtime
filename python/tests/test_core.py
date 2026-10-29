# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""grace_runtime._core.Model on the CSR fixtures against the LAMMPS styles (1 rank x 1 thread)."""
import threading

import numpy as np
import pytest

from grace_runtime import _core

SPECIES = ["W", "Mo"]


def load_fixture(path):
    d = np.load(path)
    types = np.where(d["numbers"] == 74, 1, 2).astype(np.int32)
    return types, d["offset"].astype(np.int64), d["j"].astype(np.int32), d["rij"], d["cell"]


def read_ref(stem, n):
    a = np.loadtxt(stem + ".dump", skiprows=9)
    a = a[np.argsort(a[:, 0])]
    th = open(stem + ".thermo").read().split()
    return a[:, 1:4], a[:, 4], float(th[1]), np.array([float(x) for x in th[2:8]])


@pytest.mark.parametrize("cell", ["W54", "WMo54", "W2tri"])
@pytest.mark.parametrize("arch", ["2l", "1l"])
def test_compute_matches_lammps(models, fixtures, cell, arch):
    m = _core.Model(models[arch])
    assert m.arch == arch
    types, off, j, rij, h = load_fixture(f"{fixtures}/{cell}.npz")
    out = m.compute(SPECIES, types, off, j, rij, energies=True, num_threads=1)
    f_ref, e_ref, etot_ref, p_ref = read_ref(f"{fixtures}/ref/{cell}_{arch}", len(types))
    n = len(types)
    assert abs(out["energy"] - etot_ref) / n < 1e-6
    assert np.abs(out["forces"] - f_ref).max() < 2e-5
    assert np.abs(out["energies"] - e_ref).max() < 1e-5
    p = out["virial"] / abs(np.linalg.det(h)) * 1.6021765e6    # bar, LAMMPS order xx yy zz xy xz yz
    assert np.abs(p - p_ref).max() / max(1.0, np.abs(p_ref).max()) < 1e-5


def test_cutoff_matrix(models):
    m = _core.Model(models["2l"])
    c = m.cutoff_matrix(SPECIES)
    assert c.shape == (2, 2) and np.all(c > 3.0) and np.allclose(c, c.T)


def test_unknown_element(models):
    m = _core.Model(models["2l"])
    with pytest.raises(ValueError, match="Xx"):
        m.cutoff_matrix(["W", "Xx"])


def test_bad_inputs(models, fixtures):
    m = _core.Model(models["2l"])
    types, off, j, rij, _ = load_fixture(f"{fixtures}/W54.npz")
    with pytest.raises(ValueError):
        m.compute(SPECIES, types, off[:-1], j, rij)          # offset too short
    bad = j.copy(); bad[0] = len(types)
    with pytest.raises(ValueError):
        m.compute(SPECIES, types, off, bad, rij)             # neighbour index out of range
    for t in (0, 3):
        bt = types.copy(); bt[0] = t
        with pytest.raises(ValueError, match="types must be"):
            m.compute(SPECIES, bt, off, j, rij)              # type outside the species list


def test_not_a_model(tmp_path):
    import zipfile

    p = tmp_path / "x.npz"
    np.savez(p, a=np.zeros(3))
    with pytest.raises(ValueError):
        _core.Model(str(p))
    s = tmp_path / "short.npz"
    with zipfile.ZipFile(s, "w") as zf:
        zf.writestr("x", b"123")
    with pytest.raises(ValueError, match="not a GRACE"):
        _core.Model(str(s))
    raw = bytearray(s.read_bytes())
    cd_pos = raw.index(b"PK\x01\x02")
    raw[cd_pos + 28 : cd_pos + 30] = (1000).to_bytes(2, "little")    # nl past end of central directory
    t = tmp_path / "trunc.npz"
    t.write_bytes(raw)
    with pytest.raises(ValueError, match="not a GRACE"):
        _core.Model(str(t))


def test_integer_stored_float_is_read_by_value(models, fixtures, tmp_path):
    """A float field stored as an integer array (here output_scale = 1 as int64) is converted, not reinterpreted (the
    int64 bit pattern of 1 read as a float64 scaled the energy to ~5e-324 and gave zero forces)."""
    with np.load(models["1l"]) as z:
        arrays = {k: z[k] for k in z.files}
    assert float(arrays.get("output_scale", 1.0)) == 1.0
    arrays["output_scale"] = np.array(1, dtype=np.int64)
    np.savez(tmp_path / "kokkos.npz", **arrays)
    types, off, j, rij, _ = load_fixture(f"{fixtures}/W54.npz")
    ref = _core.Model(models["1l"]).compute(SPECIES, types, off, j, rij, num_threads=1)
    out = _core.Model(str(tmp_path / "kokkos.npz")).compute(SPECIES, types, off, j, rij, num_threads=1)
    assert out["energy"] == ref["energy"]
    assert np.array_equal(out["forces"], ref["forces"]) and np.abs(ref["forces"]).max() > 0.1


@pytest.mark.parametrize("bad, match", [
    ({"n_elements": np.array(2.0)}, "n_elements.*<f8.*integer"),               # index / size field as a float
    ({"n_elements": np.array(2, dtype=">i8")}, "byte order"),                  # big-endian
    ({"n_elements": np.array(2**40)}, "n_elements.*range of int"),
    ({"chem_embedding": np.array([["a", "b"]], dtype="S1")}, "chem_embedding.*\\|S1.*float"),
    ({"chem_embedding": np.asfortranarray(np.ones((3, 4)))}, "chem_embedding.*Fortran order"),
    ({"chem_embedding": np.ones(3, dtype=np.complex128)}, "chem_embedding.*<c16"),
])
def test_bad_dtype_rejected(tmp_path, bad, match):
    """Arrays the loader would misread are an error naming the array and its dtype."""
    p = tmp_path / "kokkos.npz"
    np.savez(p, mlp_rad_W0=np.ones((2, 2)), **bad)    # mlp_rad_W0: detected as a GRACE-1L export
    with pytest.raises(ValueError, match=match):
        _core.Model(str(p))


def test_compressed_npz_rejected(models, tmp_path):
    p = tmp_path / "kokkos.npz"
    np.savez_compressed(p, mlp_rad_W0=np.ones((2, 2)), n_elements=np.array(2))
    with pytest.raises(ValueError, match="compressed"):
        _core.Model(str(p))


def test_threads_share_model(models, fixtures):
    """Two species sets used concurrently through one Model (stateless towards callers, Review Focus 5)."""
    m = _core.Model(models["2l"])
    types, off, j, rij, _ = load_fixture(f"{fixtures}/WMo54.npz")
    ref = m.compute(SPECIES, types, off, j, rij, num_threads=1)["forces"]
    ref_sw = m.compute(["Mo", "W"], 3 - types, off, j, rij, num_threads=1)["forces"]
    assert np.abs(ref - ref_sw).max() < 1e-6                  # relabelled types, same physics
    errs = []

    def work(sp, ty, fr):
        for _ in range(4):
            f = m.compute(sp, ty, off, j, rij, num_threads=2)["forces"]
            if np.abs(f - fr).max() > 1e-6:
                errs.append(np.abs(f - fr).max())

    ts = [threading.Thread(target=work, args=a) for a in [(SPECIES, types, ref), (["Mo", "W"], 3 - types, ref_sw)] * 2]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs


def test_bond_vectors():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(7, 3)) * 5
    i = rng.integers(0, 7, 40).astype(np.int64)
    j = rng.integers(0, 7, 40).astype(np.int32)
    shift = rng.normal(size=(40, 3))
    for nth in (1, 3):
        r = _core.bond_vectors(x, i, j, shift, num_threads=nth)
        assert np.array_equal(r, x[j] - x[i] + shift)
    assert _core.bond_vectors(x, i[:0], j[:0], shift[:0]).shape == (0, 3)
    with pytest.raises(ValueError):
        _core.bond_vectors(x, i, np.where(j == 0, 7, j).astype(np.int32), shift)
    with pytest.raises(ValueError):
        _core.bond_vectors(x, i[:-1], j, shift)


class _CudaTagged:
    """A host array exported through DLPack with its device set to CUDA: tests that the CPU bindings reject device
    memory by the tag, without a GPU and without dereferencing a real device pointer."""

    def __init__(self, a):
        import ctypes as C

        class Device(C.Structure):
            _fields_ = [("device_type", C.c_int), ("device_id", C.c_int)]

        class DType(C.Structure):
            _fields_ = [("code", C.c_uint8), ("bits", C.c_uint8), ("lanes", C.c_uint16)]

        class Tensor(C.Structure):
            _fields_ = [("data", C.c_void_p), ("device", Device), ("ndim", C.c_int), ("dtype", DType),
                        ("shape", C.POINTER(C.c_int64)), ("strides", C.POINTER(C.c_int64)),
                        ("byte_offset", C.c_uint64)]

        self.a = np.ascontiguousarray(a)
        self.capsule = self.a.__dlpack__()
        get = C.pythonapi.PyCapsule_GetPointer
        get.argtypes = [C.py_object, C.c_char_p]
        get.restype = C.c_void_p
        C.cast(get(self.capsule, b"dltensor"), C.POINTER(Tensor)).contents.device.device_type = 2    # kDLCUDA

    def __dlpack_device__(self):
        return (2, 0)

    def __dlpack__(self, *args, **kwargs):
        return self.capsule


def test_device_arrays_rejected(models, fixtures):
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    i, j, shift = np.array([0], dtype=np.int64), np.array([1], dtype=np.int32), np.zeros((1, 3))
    for k in range(4):
        args = [x, i, j, shift]
        args[k] = _CudaTagged(args[k])
        with pytest.raises(TypeError):
            _core.bond_vectors(*args, num_threads=1)
    with pytest.raises(TypeError):
        _core.neighbor_search(_CudaTagged(x), np.eye(3) * 10, [True] * 3, 3.0, 1)
    with pytest.raises(TypeError):
        _core.neighbor_search(x, _CudaTagged(np.eye(3) * 10), [True] * 3, 3.0, 1)
    m = _core.Model(models["1l"])
    types, off, jj, rij, _ = load_fixture(f"{fixtures}/W54.npz")
    for k in range(4):
        args = [types, off, jj, rij]
        args[k] = _CudaTagged(args[k])
        with pytest.raises(TypeError):
            m.compute(SPECIES, *args, num_threads=1)
    # the same arrays from host memory are accepted
    assert np.array_equal(_core.bond_vectors(x, i, j, shift, num_threads=1), [[3.0, 3.0, 3.0]])

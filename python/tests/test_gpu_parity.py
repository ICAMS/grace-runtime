# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""GPU backend (grace_runtime._kk_<arch>) against the LAMMPS fused kk styles on the fixture cells:
tests/fixtures/ref_gpu/<cell>_<1l|2l>_fused.{dump,thermo} from LAMMPS runs of the fused kk styles. fp32 parity: energy
2e-6 eV/atom, per-atom energy 1e-5 eV, forces 2e-5 eV/A, pressure 1e-5 relative."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grace_runtime import _gpu

from gpu_tools import GPU, NO_GPU, PLATFORM, gpu_ok

ROOT = Path(__file__).resolve().parents[2]
FX = ROOT / "tests" / "fixtures"
CACHE = Path(os.environ.get("GRACE_CACHE", Path.home() / ".cache" / "grace-runtime-test"))
MODELS = {"1l": CACHE / "GRACE-1L-OMAT-large-base" / "kokkos.npz", "2l": CACHE / "GRACE-2L-SMAX-OMAT-medium" / "kokkos.npz",
          "3l": CACHE / "GRACE-3L-OMAT-large" / "kokkos.npz"}
NKTV2P = 1.6021765e6    # eV/A^3 -> bar



pytestmark = pytest.mark.skipif(not gpu_ok() or not all(p.exists() for p in MODELS.values()),
                                reason=f"{NO_GPU} / registry models missing")


def load_cell(name):
    d = np.load(FX / f"{name}.npz")
    types = np.where(d["numbers"] == 74, 1, 2).astype(np.int32)
    return types, d["positions"], d["cell"]


def load_ref(name, layer, n):
    stem = FX / "ref_gpu" / f"{name}_{layer}_{'kk' if layer == '3l' else 'fused'}"    # 3L: frozen refs of Grace3LKK<float, double>
    t = np.loadtxt(f"{stem}.thermo")
    d = np.loadtxt(f"{stem}.dump", skiprows=9)
    d = d[np.argsort(d[:, 0])]
    return {"etot": t[1], "p": t[2:8], "f": d[:, 1:4], "pe": d[:, 4]}


@pytest.fixture(scope="module")
def mod():
    return _gpu.ensure_init(GPU)


@pytest.mark.parametrize("layer", ["1l", "2l", "3l"])
@pytest.mark.parametrize("cell", ["W54", "WMo54", "W2tri"])
def test_against_lammps_fused_kk(mod, layer, cell):
    types, pos, h = load_cell(cell)
    n = len(types)
    m = mod.Model(str(MODELS[layer]))
    s = mod.System(m, ["W", "Mo"])
    r = s.compute(types, np.ascontiguousarray(pos), np.ascontiguousarray(h), [True, True, True], energies=True)
    ref = load_ref(cell, layer, n)
    vol = abs(np.linalg.det(h))
    p = np.asarray(r["virial"]) / vol * NKTV2P
    assert abs(r["energy"] - ref["etot"]) / n <= 2e-6
    # per-atom energies: 2L 2e-5 (its blocked forward reducers sum in another fp32 order; 1.14e-5 eV on WMo54)
    assert np.abs(r["energies"] - ref["pe"]).max() <= (2e-5 if layer == "2l" else 1e-5)
    assert np.abs(r["forces"] - ref["f"]).max() <= 2e-5
    assert np.abs(p - ref["p"]).max() / max(1.0, np.abs(ref["p"]).max()) <= 1e-5


TF_REF = FX / "ref_tf" / "registry_tf_fp64.npz"    # TPCalculator, -fp64 SavedModels; CPU side: test_ref_tf.py


@pytest.mark.parametrize("model", ["GRACE-2L-SMAX-OMAT-large"])    # its own 2L shape on the GPU (g2l_shape_smaxl.h)
@pytest.mark.parametrize("cell", ["W54", "WMo54", "W2tri"])
def test_against_tf_fp64(mod, model, cell):
    """independent references: TPCalculator on the -fp64 SavedModel"""
    path = Path(os.environ.get("GRACE_TEST_M2L") or CACHE / model / "kokkos.npz")
    if not path.exists():
        pytest.skip(f"registry model {model} not in the test cache")
    types, pos, h = load_cell(cell)
    n = len(types)
    r = mod.System(mod.Model(str(path)), ["W", "Mo"]).compute(types, np.ascontiguousarray(pos), np.ascontiguousarray(h),
                                                              [True, True, True], energies=True)
    ref = np.load(TF_REF)
    k = f"{model}/{cell}"
    v = np.asarray(r["virial"]) / abs(np.linalg.det(h))     # xx yy zz xy xz yz = -stress
    stress = -v[[0, 1, 2, 5, 4, 3]]                        # ASE Voigt
    # the GPU runs the fp32 network with other kernels and sum orders than the CPU: per-atom energies within ~1e-5 eV of
    # the CPU, small-cell totals within ~5e-6 eV/atom (measured: W2tri, the registry test models)
    tol_e = (5e-6 if n < 10 else 1e-6) * n
    assert abs(r["energy"] - ref[f"{k}/energy"]) <= tol_e
    assert np.abs(r["forces"] - ref[f"{k}/forces"]).max() <= 2e-5
    assert np.abs(r["energies"] - ref[f"{k}/energies"]).max() <= 1e-5
    assert np.abs(stress - ref[f"{k}/stress"]).max() / np.abs(ref[f"{k}/stress"]).max() <= 1e-5


def test_permuted_input_same_result(mod):
    """results come back in the caller's atom order whatever the list's spatial order"""
    types, pos, h = load_cell("WMo54")
    m = mod.Model(str(MODELS["2l"]))
    a = mod.System(m, ["W", "Mo"]).compute(types, np.ascontiguousarray(pos), h, [True] * 3)
    perm = np.random.default_rng(3).permutation(len(types))
    b = mod.System(m, ["W", "Mo"]).compute(types[perm], np.ascontiguousarray(pos[perm]), h, [True] * 3)
    assert abs(a["energy"] - b["energy"]) <= 1e-4
    assert np.abs(a["forces"][perm] - b["forces"]).max() <= 2e-5


def test_skin_reuses_list_and_rebuilds(mod):
    types, pos, h = load_cell("W54")
    m = mod.Model(str(MODELS["2l"]))
    s = mod.System(m, ["W", "Mo"], skin=0.5)
    s.compute(types, np.ascontiguousarray(pos), h, [True] * 3)
    s.compute(types, np.ascontiguousarray(pos + 0.1), h, [True] * 3)    # within skin / 2: same list
    assert s.rebuilds == 1
    s.compute(types, np.ascontiguousarray(pos + 0.3), h, [True] * 3)    # beyond skin / 2: rebuilt
    assert s.rebuilds == 2


def test_uq_against_cpu(mod):
    pytest.importorskip("grace_runtime._core")
    from grace_runtime import GRACERuntimeCalculator
    from ase.io import read
    m2 = CACHE / "GRACE-2L-OMAT-medium-ft-E" / "kokkos.npz"
    if not m2.exists():
        pytest.skip("GRACE-2L-OMAT-medium-ft-E (UQ head) not in the test cache")
    types, pos, h = load_cell("WMo54")
    s = mod.System(mod.Model(str(m2)), ["W", "Mo"])
    g = s.compute(types, np.ascontiguousarray(pos), h, [True] * 3, uq=True)
    at = read(FX / "WMo54.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})
    at.calc = GRACERuntimeCalculator(str(m2), uq=True, num_threads=4)
    at.get_potential_energy()
    gc = at.calc.results["gamma"]
    assert np.abs(g["gamma"] - gc).max() / np.abs(gc).max() <= 1e-5


def test_uq_against_cpu_3l(mod):
    """GRACE-3L UQ (gamma) on the GPU as the CPU driver's, also energy_only"""
    pytest.importorskip("grace_runtime._core")
    from grace_runtime import GRACERuntimeCalculator
    from ase.io import read
    types, pos, h = load_cell("WMo54")
    s = mod.System(mod.Model(str(MODELS["3l"])), ["W", "Mo"])
    g = s.compute(types, np.ascontiguousarray(pos), h, [True] * 3, uq=True)
    ge = s.compute(types, np.ascontiguousarray(pos), h, [True] * 3, uq=True, energy_only=True)
    at = read(FX / "WMo54.data", format="lammps-data", atom_style="atomic", Z_of_type={1: 74, 2: 42})
    at.calc = GRACERuntimeCalculator(str(MODELS["3l"]), uq=True, num_threads=4, device="cpu")
    at.get_potential_energy()
    gc = at.calc.results["gamma"]
    assert np.abs(g["gamma"] - gc).max() / np.abs(gc).max() <= 1e-5
    assert np.abs(ge["gamma"] - g["gamma"]).max() / np.abs(gc).max() <= 1e-6


def test_setup_refusal_names_cpu(mod):
    """when the fused engine refuses a model / setting at setup, the error names device='cpu' and TPCalculator
    (registry models all pass; a spline table below the minimum takes the same path)"""
    m = mod.Model(str(MODELS["2l"]))
    with pytest.raises(ValueError, match="device='cpu'.*TPCalculator"):
        mod.System(m, ["W"], spline=8)


def test_unknown_element(mod):
    m = mod.Model(str(MODELS["2l"]))
    with pytest.raises(ValueError, match="Xx is not in the model"):
        mod.System(m, ["W", "Xx"])


def test_failed_build_leaves_system_usable(mod):
    """a call that throws inside the neighbour list build after the list was already overwritten (an atom 600 cells
    away: lattice translation beyond +-511, with other species) must not leave the System believing it still holds the
    previous structure's list: the next call on that structure gives the fresh result"""
    types, pos, h = load_cell("WMo54")
    m = mod.Model(str(MODELS["2l"]))
    pos = np.ascontiguousarray(pos)
    fresh = mod.System(m, ["W", "Mo"]).compute(types, pos, h, [True] * 3)
    s = mod.System(m, ["W", "Mo"])
    s.compute(types, pos, h, [True] * 3)
    bad = pos.copy()
    bad[0] += 600 * np.asarray(h)[0]
    bad[1:] = bad[1:][::-1]
    with pytest.raises(ValueError, match="511"):
        s.compute(np.ascontiguousarray((3 - types)[::-1]), bad, h, [True] * 3)
    again = s.compute(types, pos, h, [True] * 3)
    assert abs(again["energy"] - fresh["energy"]) <= 1e-4
    assert np.abs(again["forces"] - fresh["forces"]).max() <= 2e-5


@pytest.mark.parametrize("layer", ["1l", "2l"])
def test_energy_only(mod, layer):
    """RunInput::energy_only (no backward pass, as LAMMPS eflag_only): the energy and per-atom energies of the full
    evaluation within fp32 round-off, no forces / virial keys"""
    types, pos, h = load_cell("WMo54")
    m = mod.Model(str(MODELS[layer]))
    s = mod.System(m, ["W", "Mo"])
    pos = np.ascontiguousarray(pos)
    full = s.compute(types, pos, h, [True] * 3, energies=True)
    eo = s.compute(types, pos, h, [True] * 3, energies=True, energy_only=True)
    assert "forces" not in eo and "virial" not in eo
    assert abs(eo["energy"] - full["energy"]) / len(types) <= 2e-6
    assert np.abs(eo["energies"] - full["energies"]).max() <= 1e-5
    again = s.compute(types, pos, h, [True] * 3)    # a full call after an energy-only one
    assert np.abs(again["forces"] - full["forces"]).max() <= 2e-5


SMAX = {"1l": CACHE / "GRACE-1L-SMAX-OMAT-large" / "kokkos.npz", "2l": CACHE / "GRACE-2L-SMAX-OMAT-medium" / "kokkos.npz"}


@pytest.mark.parametrize("layer", ["1l", "2l"])
def test_species_sets_share_one_engine(mod, layer):
    """the species-independent engine part (weights, spline tables, cuBLAS) is set up once per Model and chunk size;
    Systems and Batches of other species sets only add their type tables; results equal those of a fresh Model per species set
    (pair-specific cutoffs of SMAX models: cutoff of the set's own pairs)"""
    from grace_runtime import Model as CpuModel
    types, pos, h = load_cell("WMo54")
    pos = np.ascontiguousarray(pos)
    m = mod.Model(str(SMAX[layer]))
    cases = [(["W", "Mo"], types), (["Mo", "W"], np.ascontiguousarray(3 - types)), (["W"], np.ones_like(types)),
             (["Mo"], np.ones_like(types))]
    systems = [mod.System(m, sp, chunksize=16384) for sp, _ in cases]    # one engine per (spline, chunksize)
    b = mod.Batch(m, ["Nb", "W"], chunksize=16384)
    assert m.engine_setups == 1
    cpu = CpuModel(str(SMAX[layer]))
    for s, (sp, _) in zip(systems, cases):
        assert abs(s.cutoff - float(cpu.cutoff_matrix(sorted(sp)).max())) < 1e-12, sp
    fresh = [mod.System(mod.Model(str(SMAX[layer])), sp).compute(t, pos, h, [True] * 3) for sp, t in cases]
    for _ in range(2):    # alternate the species sets on the shared engine
        for s, (sp, t), r in zip(systems, cases, fresh):
            out = s.compute(t, pos, h, [True] * 3)
            assert abs(out["energy"] - r["energy"]) / 54 <= 2e-6, sp
            assert np.abs(out["forces"] - r["forces"]).max() <= 2e-5, sp
        nb = b.compute(np.array([54], np.int32), np.ascontiguousarray(types), pos, h[None].copy(),
                       np.ones((1, 3), bool))
        assert np.isfinite(nb["energy"]).all()


def test_shared_engine_concurrent_systems(mod):
    """Systems of different species sets on one Model from several threads: each call its own correct result"""
    from concurrent.futures import ThreadPoolExecutor
    types, pos, h = load_cell("WMo54")
    pos = np.ascontiguousarray(pos)
    m = mod.Model(str(SMAX["2l"]))
    cases = [(["W", "Mo"], types), (["W"], np.ones_like(types)), (["Mo"], np.ones_like(types))]
    systems = [mod.System(m, sp) for sp, _ in cases]
    ref = [s.compute(t, pos, h, [True] * 3) for s, (_, t) in zip(systems, cases)]
    with ThreadPoolExecutor(3) as ex:
        for _ in range(5):
            got = list(ex.map(lambda k: systems[k].compute(cases[k][1], pos, h, [True] * 3), range(3)))
            for g, r in zip(got, ref):
                assert abs(g["energy"] - r["energy"]) <= 1e-4
                assert np.abs(g["forces"] - r["forces"]).max() <= 2e-5


FTE = CACHE / "GRACE-2L-OMAT-medium-ft-E" / "kokkos.npz"
UQ_SNIPPET = """
import sys, numpy as np
import os
from grace_runtime import _gpu
mod = _gpu.ensure_init(os.environ["GRACE_TEST_GPU"])
d = np.load(sys.argv[2])
types = np.where(d["numbers"] == 74, 1, 2).astype(np.int32)
out = mod.System(mod.Model(sys.argv[1]), ["W", "Mo"]).compute(types, np.ascontiguousarray(d["positions"]), d["cell"],
                                                            [True] * 3, uq=True)
np.save(sys.argv[3], out["gamma"])
print("BLAS", mod.blas())
"""


def run_uq(tmp_path, tag, env):
    import subprocess, sys
    out = tmp_path / f"{tag}.npy"
    r = subprocess.run([sys.executable, "-c", UQ_SNIPPET, str(FTE), str(FX / "WMo54.npz"), str(out)],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return np.load(out), f"BLAS {BLAS}" in r.stdout


BLAS = {"cuda": "cublas", "rocm": "rocblas"}.get(PLATFORM)
NO_BLAS = {"cuda": "GRACE_RUNTIME_NO_CUBLAS", "rocm": "GRACE_RUNTIME_NO_ROCBLAS"}.get(PLATFORM)


def _need_blas():
    """NVIDIA: cuBLAS of the nvidia-cublas-cu12 package; AMD: rocBLAS of the system ROCm (librocblas.so.5 / .so.4)"""
    if PLATFORM == "cuda":
        pytest.importorskip("nvidia.cublas")
        return
    import ctypes
    for name in ("librocblas.so.5", "librocblas.so.4"):
        try:
            ctypes.CDLL(name)
            return
        except OSError:
            pass
    pytest.skip("no rocBLAS")


def test_blas_loaded_at_run_time(mod):
    """cuBLAS / rocBLAS is loaded at run time (nvidia-cublas-cu12 preloaded by grace_runtime._cuda, before any system
    cuBLAS; the system rocBLAS), not a link dependency"""
    _need_blas()
    assert mod.blas() == BLAS
    if PLATFORM == "cuda":
        from grace_runtime import _cuda
        lib = _cuda.cublas_library()
        assert lib["source"] == "wheel" and "/nvidia/cublas/lib/" in lib["path"], lib


def test_uq_without_blas_matches(tmp_path):
    """GRACE_RUNTIME_NO_CUBLAS / NO_ROCBLAS: the hand-written UQ kernels give the gamma of the BLAS path (fresh
    processes)"""
    _need_blas()
    env = dict(os.environ)
    env.pop(NO_BLAS, None)
    g1, c1 = run_uq(tmp_path, "blas", env)
    g0, c0 = run_uq(tmp_path, "plain", dict(env, **{NO_BLAS: "1"}))
    assert c1 and not c0
    assert np.abs(g1 - g0).max() <= 1e-5 * np.abs(g1).max()


UQ_COST = """
import sys, time, numpy as np
from ase.build import bulk
import os
from grace_runtime import _gpu
kk = _gpu.ensure_init(os.environ["GRACE_TEST_GPU"])
at = bulk("W", "bcc", a=3.165, cubic=True).repeat((14, 14, 13))
at.symbols[::3] = "Mo"
ty = np.where(at.numbers == 74, 1, 2).astype(np.int32)
pos, h = np.ascontiguousarray(at.positions), np.ascontiguousarray(at.cell.array)
s = kk.System(kk.Model(sys.argv[1]), ["W", "Mo"])
t = {False: [], True: []}
for uq in (False, True, False, True):
    s.compute(ty, pos, h, [True] * 3, uq=uq)
    t0 = time.perf_counter()
    for _ in range(20):
        s.compute(ty, pos, h, [True] * 3, uq=uq)
    t[uq].append(time.perf_counter() - t0)
print(min(t[True]) / min(t[False]))
"""


@pytest.mark.skipif(os.environ.get("GRACE_PERF_TESTS") != "1", reason="timing test: GRACE_PERF_TESTS=1 on an idle card")
@pytest.mark.parametrize("model", ["GRACE-2L-OMAT-medium-ft-E", "GRACE-1L-OMAT-large-base"])
def test_uq_without_cublas_is_cheap(model):
    """the hand-written UQ kernels (fp32, as the cuBLAS path) cost at most 80 % on top of energy + forces, also on
    cards with slow fp64 (W5k, device call)"""
    env = dict(os.environ, **{NO_BLAS: "1"})
    r = subprocess.run([sys.executable, "-c", UQ_COST, str(CACHE / model / "kokkos.npz")], capture_output=True,
                       text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    ratio = float(r.stdout.split()[-1])
    assert ratio <= 1.8, f"UQ costs {100 * (ratio - 1):.0f} % without cuBLAS"


E3L_TIME = """
import sys, time, numpy as np
from ase.build import bulk
import os
from grace_runtime import _gpu
kk = _gpu.ensure_init(os.environ["GRACE_TEST_GPU"])
at = bulk("W", "bcc", a=3.165, cubic=True).repeat((8, 8, 8))
at.symbols[::3] = "Mo"
ty = np.where(at.numbers == 74, 1, 2).astype(np.int32)
pos, h = np.ascontiguousarray(at.positions), np.ascontiguousarray(at.cell.array)
s = kk.System(kk.Model(sys.argv[1]), ["W", "Mo"])
t = []
for _ in range(3):
    s.compute(ty, pos, h, [True] * 3)
    t0 = time.perf_counter()
    for _ in range(10):
        s.compute(ty, pos, h, [True] * 3)
    t.append(time.perf_counter() - t0)
print(kk.blas() or "-", min(t))
"""


@pytest.mark.skipif(os.environ.get("GRACE_PERF_TESTS") != "1", reason="timing test: GRACE_PERF_TESTS=1 on an idle card")
def test_3l_without_blas_is_not_much_slower():
    """GRACE-3L energy + forces without cuBLAS / rocBLAS (the built-in Kokkos GEMMs) take at most twice the time with
    it (L40S, RTX 2070: 1.4-1.5x; the hand-written fallback before the Kokkos GEMMs on CUDA: 13-16x), W/Mo 1024 atoms"""
    _need_blas()
    env = {k: v for k, v in os.environ.items() if k != NO_BLAS}
    out = {}
    for tag, e in (("blas", env), ("none", dict(env, **{NO_BLAS: "1"}))):
        r = subprocess.run([sys.executable, "-c", E3L_TIME, str(MODELS["3l"])], capture_output=True, text=True, env=e,
                           timeout=900)
        assert r.returncode == 0, r.stderr[-2000:]
        out[tag] = r.stdout.split()[-2:]
    assert out["blas"][0] == BLAS and out["none"][0] == "kkgemm", out
    ratio = float(out["none"][1]) / float(out["blas"][1])
    assert ratio <= 2.0, f"3L without {BLAS}: {ratio:.1f}x the time with it"


@pytest.mark.parametrize("cell", ["W54", "WMo54"])
def test_3l_without_cublas_against_lammps(cell):
    """the hand-written kernels of the 3L engine (grace-runtime-cuda12 without the [cublas] extra; AMD without rocBLAS):
    a fresh process with GRACE_RUNTIME_NO_CUBLAS=1 / NO_ROCBLAS=1 matches the frozen LAMMPS references of Grace3LKK<float, double> (which used cuBLAS)"""
    import subprocess
    import sys
    code = (
        "import sys, numpy as np\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import os\n"
        "from test_gpu_parity import load_cell, load_ref, MODELS, NKTV2P\n"
        "from grace_runtime import _gpu\n"
        "mod = _gpu.ensure_init(os.environ['GRACE_TEST_GPU'])\n"
        "assert mod.blas() not in ('cublas', 'rocblas'), 'BLAS still in use'\n"
        "types, pos, h = load_cell(sys.argv[2])\n"
        "r = mod.System(mod.Model(str(MODELS['3l'])), ['W', 'Mo']).compute(types, np.ascontiguousarray(pos), "
        "np.ascontiguousarray(h), [True] * 3, energies=True)\n"
        "ref = load_ref(sys.argv[2], '3l', len(types))\n"
        "print('DE', abs(r['energy'] - ref['etot']) / len(types), 'DF', np.abs(r['forces'] - ref['f']).max())\n")
    env = {k: v for k, v in os.environ.items() if k != "GRACE_RUNTIME_DEVICE"}
    env[NO_BLAS] = "1"
    out = subprocess.run([sys.executable, "-c", code, str(Path(__file__).parent), cell], env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    de, df = (float(x) for x in out.stdout.split()[1::2][:2])
    assert de <= 2e-6 and df <= 2e-5, out.stdout


def test_3l_spline_n_refused(mod):
    """spline_n below 16 is refused at setup (it built an empty table set: illegal address, a sticky GPU error)"""
    m = mod.Model(str(MODELS["3l"]))
    with pytest.raises(ValueError):
        mod.System(m, ["W"], spline=0)
    types, pos, h = load_cell("W54")
    r = mod.System(m, ["W", "Mo"]).compute(types, np.ascontiguousarray(pos), np.ascontiguousarray(h), [True] * 3)
    assert np.isfinite(r["energy"])

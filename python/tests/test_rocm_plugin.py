# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""AMD GPU backend discovery, no GPU needed (a fake KFD topology, a fake HIP runtime, fake plug-ins): the GPU
architecture from the KFD topology, exact-match module selection, clear ImportErrors (no runtime, no plug-in, no module
for the architecture, ABI mismatch, torch for another ROCm major), plug-ins grouped by PLATFORM, the device strings
("rocm", "hip", "gpu") and the order of device="auto" (NVIDIA, then AMD, then the CPU), one GPU per process across
vendors, and the AMD section of grace-runtime info."""
import ctypes
import sys
import textwrap
import types

import pytest

from grace_runtime import _cuda, _gpu, _rocm, cli


# ---- KFD topology and architecture names ----

@pytest.mark.parametrize("version, arch", [(90010, "gfx90a"), (90402, "gfx942"), (90008, "gfx908"),
                                           (110000, "gfx1100"), (110001, "gfx1101"), (120001, "gfx1201"),
                                           (90500, "gfx950")])
def test_gfx_name(version, arch):
    assert _rocm.gfx_name(version) == arch


def kfd_tree(tmp_path, gpus=((90010, 0, 0xc100),)):
    """/sys/class/kfd/kfd/topology/nodes with a CPU node and the given GPU nodes (target version, domain, location id);
    a /dev/kfd stand-in"""
    root = tmp_path / "nodes"
    nodes = [(0, 0, 0)] + list(gpus)
    for i, (v, dom, loc) in enumerate(nodes):
        (root / str(i)).mkdir(parents=True)
        (root / str(i) / "properties").write_text(f"cpu_cores_count 8\ngfx_target_version {v}\ndomain {dom}\n"
                                                  f"location_id {loc}\n")
    dev = tmp_path / "kfd"
    dev.write_text("")
    return root, dev


@pytest.fixture
def no_env(monkeypatch):
    for k in ("GRACE_RUNTIME_DEVICE", "GRACE_RUNTIME_ROCM_ARCH", "GRACE_RUNTIME_STRICT", "HSA_OVERRIDE_GFX_VERSION"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setitem(_gpu._state, "module", None)
    return monkeypatch


def test_kfd_presence(tmp_path, no_env):
    root, dev = kfd_tree(tmp_path, gpus=((90010, 0, 0xc100), (90010, 0, 0xc600)))
    no_env.setattr(_rocm, "KFD_NODES", str(root))
    no_env.setattr(_rocm, "KFD_DEV", str(dev))
    assert [n["arch"] for n in _rocm.kfd_nodes()] == ["gfx90a", "gfx90a"]    # the CPU node is left out
    assert _rocm.kfd_present()
    no_env.setattr(_rocm, "KFD_DEV", str(tmp_path / "missing"))
    assert not _rocm.kfd_present()    # no driver device
    cpu_only, dev = kfd_tree(tmp_path / "cpu", gpus=())
    no_env.setattr(_rocm, "KFD_NODES", str(cpu_only))
    no_env.setattr(_rocm, "KFD_DEV", str(dev))
    assert not _rocm.kfd_present()    # driver loaded, no GPU


class FakeHip:
    """libamdhip64.so.6 of ROCm 6.3 with `n` visible devices: an MI250X GCD at PCI 0000:c1:00.0 (+ one at c6:00.0)"""
    BUS = ["0000:c1:00.0", "0000:c6:00.0"]

    def __init__(self, n=1):
        self.n = n

    def hipGetDeviceCount(self, out):
        out._obj.value = self.n
        return 0

    def hipRuntimeGetVersion(self, out):
        out._obj.value = 60342134
        return 0

    def hipDriverGetVersion(self, out):
        out._obj.value = 60342134
        return 0

    def hipDeviceGetName(self, buf, size, index):
        buf.value = b"AMD Instinct MI250X"
        return 0

    def hipDeviceTotalMem(self, out, index):
        out._obj.value = 68702699520
        return 0

    def hipDeviceGetPCIBusId(self, buf, size, index):
        buf.value = self.BUS[index].encode()
        return 0


@pytest.fixture
def amd_node(tmp_path, no_env):
    """a node with two GPUs of different architectures in the KFD topology (gfx90a at c1:00.0, gfx942 at c6:00.0), a
    HIP runtime that sees one or both, no NVIDIA driver"""
    root, dev = kfd_tree(tmp_path, gpus=((90010, 0, 0xc100), (90402, 0, 0xc600)))
    no_env.setattr(_rocm, "KFD_NODES", str(root))
    no_env.setattr(_rocm, "KFD_DEV", str(dev))
    no_env.setattr(_rocm, "_hip", {})
    hip = FakeHip(2)
    real = ctypes.CDLL

    def cdll(name, *a, **k):
        if name == "libamdhip64.so.6":
            return hip
        if name in ("libamdhip64.so.7", "libcuda.so.1"):
            raise OSError(f"{name}: cannot open shared object file")
        return real(name, *a, **k)
    no_env.setattr(ctypes, "CDLL", cdll)
    no_env.setattr(_rocm, "plugins", lambda: [])
    return no_env


def test_probe_finds_the_architecture_by_pci_address(amd_node):
    p = _rocm.probe(0)
    assert p["arch"] == "gfx90a" and p["count"] == 2 and p["major"] == 6 and p["name"] == "AMD Instinct MI250X"
    assert _rocm.version_str(p["runtime"]) == "6.3.42134"
    assert _rocm.probe(1)["arch"] == "gfx942"
    with pytest.raises(ValueError, match=r"rocm:2 does not exist \(2 visible"):
        _rocm.probe(2)


def test_hsa_override_is_what_hip_reports(amd_node):
    amd_node.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    assert _rocm.probe(0)["arch"] == "gfx1100"


def test_no_runtime_is_a_clear_error_and_auto_warns(amd_node):
    amd_node.setattr(ctypes, "CDLL", lambda name, *a, **k: (_ for _ in ()).throw(OSError(f"{name}: not found")))
    with pytest.raises(ImportError, match=r"no ROCm runtime \(libamdhip64\.so\.7 or libamdhip64\.so\.6\)"):
        _rocm.probe(0)
    device, why = _gpu.auto_device()
    assert device == "cpu" and "no ROCm runtime" in why


def test_no_visible_device_is_silent(amd_node):
    amd_node.setattr(FakeHip, "hipGetDeviceCount", lambda self, out: setattr(out._obj, "value", 0) or 0)
    with pytest.raises(_gpu.NoGPU):
        _rocm.probe(0)
    assert _gpu.auto_device() == ("cpu", None)


def test_no_amd_gpu_is_silent(tmp_path, no_env):
    no_env.setattr(_rocm, "KFD_DEV", str(tmp_path / "missing"))
    no_env.setattr(_cuda, "compute_capability", lambda index: (_ for _ in ()).throw(_gpu.NoGPU("no NVIDIA driver")))
    assert _gpu.auto_device() == ("cpu", None)
    with pytest.raises(ImportError, match="no usable GPU"):
        _gpu.resolve_gpu(0)


# ---- module selection: exact match, the ways out ----

def test_exact_match(amd_node):
    amd_node.setattr(_rocm, "available", lambda: {"gfx90a": "x._kk_gfx90a", "gfx942": "x._kk_gfx942"})
    assert _rocm.select(0) == "gfx90a" and _rocm.select(1) == "gfx942"


def test_no_module_of_the_architecture(amd_node):
    """gfx942 code does not run on gfx90a (no PTX-like fallback): the installed ones and the way out are named"""
    amd_node.setattr(_rocm, "available", lambda: {"gfx942": "x._kk_gfx942", "gfx1100": "x._kk_gfx1100"})
    with pytest.raises(ImportError, match=r"no GPU module for gfx90a \(rocm:0.*installed: gfx1100, gfx942.*"
                                          r"GRACE_KK_ARCHS=gfx90a"):
        _rocm.select(0)
    device, why = _gpu.auto_device()
    assert device == "cpu" and "gfx90a" in why


def test_forced_architecture(amd_node):
    amd_node.setattr(_rocm, "available", lambda: {"gfx90a": "x._kk_gfx90a"})
    amd_node.setenv("GRACE_RUNTIME_ROCM_ARCH", "gfx000")
    with pytest.raises(ImportError, match="gfx000"):
        _rocm.select(0)


def test_forced_architecture_when_the_pci_address_is_not_in_the_topology(amd_node):
    """HIP's PCI address of the device matches no KFD node and the node has two GPU types: the architecture is unknown,
    and the error names GRACE_RUNTIME_ROCM_ARCH, which then selects the module (it was read only after the error)"""
    amd_node.setattr(FakeHip, "BUS", ["0000:ff:00.0", "0000:c6:00.0"])
    amd_node.setattr(_rocm, "available", lambda: {"gfx90a": "x._kk_gfx90a"})
    with pytest.raises(ImportError, match="not in the KFD topology.*GRACE_RUNTIME_ROCM_ARCH"):
        _rocm.select(0)
    amd_node.setenv("GRACE_RUNTIME_ROCM_ARCH", "gfx90a")
    assert _rocm.select(0) == "gfx90a" and _rocm.probe(0)["arch"] == "gfx90a"
    amd_node.setenv("GRACE_RUNTIME_ROCM_ARCH", "gfx942")    # forced, but no module of it
    with pytest.raises(ImportError, match="no GPU module for gfx942"):
        _rocm.select(0)


def test_no_plugin_names_the_install_command(amd_node):
    amd_node.setattr(_rocm, "installed_archs", lambda: [])
    from grace_runtime import __version__
    pin = f"=={__version__}" if "unknown" not in __version__ else ""
    with pytest.raises(ImportError, match=f"pip install 'grace-runtime-rocm6{pin}'"):
        _rocm.module_for(0)
    device, why = _gpu.auto_device()
    assert device == "cpu" and "grace-runtime-rocm6" in why


def test_torch_for_another_rocm_major_is_refused(monkeypatch):
    fake = types.SimpleNamespace(version=types.SimpleNamespace(hip="7.0.51831"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(ImportError, match="torch in this process is built for ROCm 7"):
        _rocm._check_torch(6)
    _rocm._check_torch(7)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(version=types.SimpleNamespace(hip=None)))
    _rocm._check_torch(6)    # a CUDA / CPU torch


# ---- plug-ins: PLATFORM, ABI ----

def fake_plugin(tmp_path, name, abi, platform=None, archs=("gfx90a",)):
    pkg = tmp_path / name
    pkg.mkdir()
    plat = f"PLATFORM = {platform!r}\nROCM_MAJOR = 6\n" if platform else ""
    (pkg / "__init__.py").write_text(textwrap.dedent(f"""
        ABI = {abi}
        __version__ = "9.9.9"
        def archs():
            return {list(archs)!r}
    """) + plat)
    dist = tmp_path / f"{name}-9.9.9.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 9.9.9\n")
    (dist / "entry_points.txt").write_text(f"[grace_runtime.backends]\n{name} = {name}\n")
    return tmp_path


def test_plugins_grouped_by_platform(tmp_path, monkeypatch):
    """a ROCm plug-in is invisible to the CUDA loader and the other way round; one without PLATFORM is CUDA (the
    released grace-runtime-cuda12)"""
    monkeypatch.syspath_prepend(str(fake_plugin(tmp_path, "gn_fake_rocm", abi=_gpu.ABI, platform="rocm")))
    monkeypatch.syspath_prepend(str(fake_plugin(tmp_path, "gn_fake_cuda", abi=_gpu.ABI, archs=("sm80",))))
    assert "gn_fake_rocm" in [p.__name__ for p in _rocm.plugins()]
    assert "gn_fake_rocm" not in [p.__name__ for p in _cuda.plugins()]
    assert "gn_fake_cuda" in [p.__name__ for p in _cuda.plugins()]
    assert "gn_fake_cuda" not in [p.__name__ for p in _rocm.plugins()]
    assert _rocm.available()["gfx90a"] == "gn_fake_rocm._kk_gfx90a" and "gfx90a" not in _cuda.available()
    assert _rocm.rocm_majors() == (6,)
    for m in ("gn_fake_rocm", "gn_fake_cuda"):
        sys.modules.pop(m, None)


def test_rocm_plugin_abi_mismatch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(fake_plugin(tmp_path, "gn_fake_rocm_old", abi=_gpu.ABI + 1, platform="rocm")))
    with pytest.raises(ImportError, match=rf"gn_fake_rocm_old 9\.9\.9 has GPU backend ABI {_gpu.ABI + 1}"
                                           r".*pip install 'gn_fake_rocm_old=="):
        _rocm.plugins()
    _cuda.plugins()    # the CUDA loader does not care
    sys.modules.pop("gn_fake_rocm_old", None)


# ---- device strings, auto order, one GPU per process ----

@pytest.mark.parametrize("s, want", [("rocm", ("rocm", 0)), ("rocm:3", ("rocm", 3)), ("hip", ("rocm", 0)),
                                     ("hip:1", ("rocm", 1)), ("cuda:2", ("cuda", 2)), ("gpu", ("gpu", 0)),
                                     ("gpu:1", ("gpu", 1))])
def test_parse_device(s, want):
    assert _gpu.parse_device(s) == want


@pytest.mark.parametrize("s", ["gpu0", "rocm:", "amd", "cuda:x", "cpu"])
def test_bad_device(s):
    with pytest.raises(ValueError, match="unknown device"):
        _gpu.parse_device(s)


def vendors(monkeypatch, cuda, rocm):
    """fake vendor loaders: None (no GPU), "ok" (module there) or an error text (GPU there, not usable)"""
    def module_for(state, name):
        def f(index):
            if state is None:
                raise _gpu.NoGPU(f"no {name} GPU")
            if state != "ok":
                raise ImportError(state)
            return object()
        return f
    monkeypatch.setattr(_cuda, "module_for", module_for(cuda, "NVIDIA"))
    monkeypatch.setattr(_rocm, "module_for", module_for(rocm, "AMD"))


@pytest.mark.parametrize("cuda, rocm, want", [
    ("ok", "ok", ("cuda", None)),    # NVIDIA first: unchanged for CUDA users
    (None, "ok", ("rocm", None)),
    ("no module for sm120", "ok", ("rocm", None)),
    (None, None, ("cpu", None)),
    ("no module for sm120", None, ("cpu", "no module for sm120")),
    (None, "no module for gfx1101", ("cpu", "no module for gfx1101")),
    ("no module for sm120", "no ROCm runtime", ("cpu", "no module for sm120; no ROCm runtime")),
])
def test_auto_order(no_env, cuda, rocm, want):
    vendors(no_env, cuda, rocm)
    assert _gpu.auto_device() == want


def test_gpu_device(no_env):
    vendors(no_env, None, "ok")
    assert _gpu.resolve_gpu(0) == "rocm" and _gpu.resolve_gpu(1) == "rocm:1"
    vendors(no_env, "ok", "ok")
    assert _gpu.resolve_gpu(0) == "cuda"
    vendors(no_env, "no module for sm120", None)
    with pytest.raises(ImportError, match=r"device='gpu': no usable GPU: no module for sm120.*device='auto'"):
        _gpu.resolve_gpu(0)


class FakeModule:
    def __init__(self):
        self.inits = []

    def initialized(self):
        return bool(self.inits)

    def initialize(self, index):
        self.inits.append(index)

    def finalize(self):
        pass


def test_ensure_init_and_one_gpu_per_process(no_env):
    mod = FakeModule()
    no_env.setattr(_rocm, "start", lambda index: (mod, {"arch": "gfx90a", "ptx": False, "cc": "gfx90a"}))
    no_env.setattr("atexit.register", lambda f: f)
    for k in ("platform", "device", "arch", "ptx", "cc"):
        no_env.setitem(_gpu._state, k, _gpu._state[k])
    assert _gpu.ensure_init("hip") is mod and mod.inits == [0]
    assert _gpu.info() == {"device": "rocm:0", "platform": "rocm", "arch": "gfx90a", "cc": "gfx90a", "native": True}
    assert _gpu.ensure_init("rocm:0") is mod and _gpu.ensure_init("gpu") is mod and mod.inits == [0]
    assert _gpu.auto_device() == ("rocm:0", None)
    for other in ("rocm:1", "cuda", "gpu:1"):
        with pytest.raises(ValueError, match="one GPU per process: already using rocm:0"):
            _gpu.ensure_init(other)
    with pytest.raises(ValueError, match="one GPU per process: already using rocm:0"):
        _cuda.ensure_init("cuda")


# ---- grace-runtime info ----

def test_info_amd_without_module_prints_the_install_command(amd_node, capsys):
    amd_node.setattr(_rocm, "installed_archs", lambda: [])
    assert cli.main(["info"]) == 1
    out = capsys.readouterr().out
    assert "GPU rocm:0: gfx90a (AMD Instinct MI250X), 68.70 GB, 2 visible; HIP runtime 6.3.42134" in out, out
    assert "ROCR_VISIBLE_DEVICES=" in out and "install: pip install 'grace-runtime-rocm6" in out, out


def test_info_amd_native(amd_node, capsys):
    amd_node.setattr(_rocm, "available", lambda: {"gfx90a": "x._kk_gfx90a", "gfx942": "x._kk_gfx942"})
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "GPU modules installed: gfx90a, gfx942 (gfx942: built, untested)" in out and "native module gfx90a" in out, out
    assert "rocBLAS:" in out

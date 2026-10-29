# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""``grace-runtime models list | download NAME...``, ``grace-runtime info``, ``grace-runtime relax MODEL IN -o OUT``"""
import argparse
import os
import sys

from . import registry


def main(argv=None):
    ap = argparse.ArgumentParser(prog="grace-runtime")
    sub = ap.add_subparsers(dest="cmd", required=True)
    models = sub.add_parser("models", help="foundation models of the registry").add_subparsers(dest="action",
                                                                                               required=True)
    models.add_parser("list", help="list the models grace-runtime can run")
    dl = models.add_parser("download", help="download models into the GRACE cache")
    dl.add_argument("--force", action="store_true", help="download again even if a cached file is accepted")
    dl.add_argument("names", nargs="+")
    sub.add_parser("info", help="where calculations will run: CPU module, GPU (NVIDIA or AMD), driver / runtime, GPU "
                                "module (native or PTX), fallback settings, JIT cache, install command")
    rl = sub.add_parser("relax", help="relax all structures of a file together (FIRE as ASE, optionally with the "
                                      "cell); each one is written as it finishes")
    rl.add_argument("model", help="registry name, kokkos.npz or PACE .yaml")
    rl.add_argument("input", help="structure file, all frames (any format ASE reads)")
    rl.add_argument("-o", "--output", required=True, help="extxyz file of the relaxed structures (overwritten); info: "
                                                          "relax_index (input frame), relax_n_steps, relax_converged, "
                                                          "relax_fmax")
    rl.add_argument("--fmax", type=float, default=0.05, help="largest force (eV/A), default 0.05")
    rl.add_argument("--max-steps", type=int, default=500, help="FIRE steps per structure, default 500")
    rl.add_argument("--cell", action="store_true", help="relax the cell too (as ase.filters.FrechetCellFilter)")
    rl.add_argument("--pressure", type=float, default=0.0, help="scalar pressure with --cell (eV/A^3)")
    rl.add_argument("--device", default="auto", help="auto (default), cpu, gpu, cuda[:N], rocm[:N]")
    rl.add_argument("--max-atoms", type=int, default=131072, help="atoms relaxed together, default 131072")
    a = ap.parse_args(argv)
    if a.cmd == "info":
        return info()
    if a.cmd == "relax":
        return relax(a)
    cache = registry.default_cache_dir()
    if a.action == "list":
        for m in registry.list_models():
            path = os.path.join(cache, m["name"], "kokkos.npz")
            state = "cached" if os.path.isfile(path) else "not cached"
            aliases = f"  (aliases: {', '.join(m['aliases'])})" if m.get("aliases") else ""
            print(f"{m['name']:32s} {m['arch']}  uq={'yes' if m.get('uq') else 'no '}  {state}{aliases}")
        print(f"cache: {cache}")
        return 0
    rc = 0
    for name in a.names:
        try:
            print(registry.resolve(name, cache_dir=cache, force=a.force))
        except (KeyError, ValueError, RuntimeError, OSError) as e:
            print(f"{name}: {e}", file=sys.stderr)
            rc = 1
    return rc


def relax(a):
    """relax_batch_iter over the frames of a.input; exit status 1 when a structure did not converge"""
    import time
    from ase.io import iread, write
    from .relax import relax_batch_iter
    if a.pressure and not a.cell:
        print("--pressure needs --cell", file=sys.stderr)
        return 2
    if os.path.exists(a.output) and os.path.samefile(a.input, a.output):    # written while the input is read
        print("the output must not be the input file", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    n = converged = 0
    with open(a.output, "w") as fh:
        for i, r in relax_batch_iter(a.model, iread(a.input, index=":"), fmax=a.fmax, max_steps=a.max_steps,
                                     cell_filter="frechet" if a.cell else None, scalar_pressure=a.pressure,
                                     device=a.device, max_atoms=a.max_atoms):
            at = r["atoms"]
            at.info.update(relax_index=i, relax_n_steps=r["n_steps"], relax_converged=r["converged"],
                           relax_fmax=r["fmax"])
            write(fh, at, format="extxyz")
            fh.flush()
            n += 1
            converged += r["converged"]
    print(f"relaxed {n} structures ({converged} converged) in {time.perf_counter() - t0:.1f} s: {a.output}",
          file=sys.stderr)
    return 0 if converged == n else 1


def info():
    """print the backend choice of device="auto" / "gpu" without starting Kokkos: the NVIDIA GPU (cuda:0), then the AMD
    GPU (rocm:0); exit status 1 when a GPU is there but would not run natively (NVIDIA: no module of its family: CPU,
    or PTX with GRACE_RUNTIME_ALLOW_PTX=1; AMD: no runtime or no module of its architecture)"""
    import grace_runtime
    from . import _cuda, _rocm
    print(f"grace-runtime {grace_runtime.__version__}, CPU module {grace_runtime.cpu_module}")
    flags = {k: os.environ[k] for k in ("GRACE_RUNTIME_DEVICE", "GRACE_RUNTIME_ALLOW_PTX", "GRACE_RUNTIME_STRICT",
                                        "GRACE_RUNTIME_QUIET", "GRACE_RUNTIME_CUDA_ARCH", "GRACE_RUNTIME_NO_CUBLAS",
                                        "CUDA_VISIBLE_DEVICES", "GRACE_RUNTIME_ROCM_ARCH", "GRACE_RUNTIME_NO_ROCBLAS",
                                        "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_XNACK",
                                        "HSA_OVERRIDE_GFX_VERSION")
             if k in os.environ}
    print("settings:", ", ".join(f"{k}={v}" for k, v in flags.items()) or "defaults")
    if sys.platform != "linux":
        print("GPU: none (GPU modules are Linux-only); device='auto' runs on the CPU")
        return 0
    try:
        _cuda.compute_capability(0)
    except (_cuda.NoGPU, ValueError) as e:
        if _rocm.kfd_present():
            return _info_rocm()
        print(f"GPU: none usable ({e}); device='auto' runs on the CPU")
        return 0
    except ImportError:
        pass
    return _info_cuda()


def _info_cuda():
    from . import _cuda
    try:
        major, minor = _cuda.compute_capability(0)
    except ImportError as e:    # a driver older than CUDA 12
        print(f"GPU: none usable ({e}); device='auto' runs on the CPU")
        return 1
    d = _cuda._state["driver"]
    print(f"GPU cuda:0: compute capability {major}.{minor} (sm{major}{minor}); driver for CUDA {d // 1000}.{d % 1000 // 10}")
    try:
        avail = _cuda.available()
    except ImportError as e:
        print(f"GPU modules: {e}")
        return 1
    print(f"GPU modules installed: {', '.join(sorted(avail)) or 'none'}")
    print(f"cuBLAS: {_cublas(major)}")
    try:
        arch, ptx = _cuda.select(0)
    except ImportError as e:
        print(f"NOT NATIVE: {e}")
        if not avail:
            print(f"install: {_cuda.install_hint()}")
        return 1
    if ptx:
        cache = os.environ.get("CUDA_CACHE_PATH") or os.path.expanduser("~/.nv/ComputeCache")
        ok = os.access(cache if os.path.isdir(cache) else os.path.dirname(cache.rstrip("/")) or ".", os.W_OK)
        print(f"NOT NATIVE: runs the {arch} module's PTX, compiled by the driver (JIT cache {cache}, "
              f"{'writable' if ok else 'NOT writable: compiled again in every process'}); a few % slower")
        return 1
    print(f"native module {arch}")
    return 0


def _cublas(cc_major: int) -> str:
    """the cuBLAS the NVIDIA GPU modules would use (_cuda.cublas_library), or why they run the built-in GEMMs"""
    from . import _cuda
    c = _cuda.cublas_library()
    out = c["broken"] + "; " if "broken" in c else ""
    if "version" not in c:
        return out + f"none: built-in Kokkos GEMMs ({c['why']})"
    v, f = c["version"], _cuda.cublas_floor(cc_major)
    out += f"{v[0]}.{v[1]}.{v[2]} ({c['source']}: {c['path']})"
    return out + (f", older than {f[0]}.{f[1]} for this GPU: built-in Kokkos GEMMs" if v[:2] < f else "")


def _rocblas() -> str:
    """rocBLAS as the AMD GPU modules load it: the soname of their ROCm (6: librocblas.so.4, 7: .so.5)"""
    import ctypes
    from . import _rocm
    if os.environ.get("GRACE_RUNTIME_NO_ROCBLAS", "") not in ("", "0"):
        return "off (GRACE_RUNTIME_NO_ROCBLAS)"
    for major in _rocm.rocm_majors():
        name = f"librocblas.so.{major - 2}"
        try:
            ctypes.CDLL(name)
            return f"{name} found (ROCm {major})"
        except OSError:
            pass
    return "not found: hand-written GEMMs (UQ somewhat slower)"


def _info_rocm():
    from . import _rocm
    try:
        p = _rocm.probe(0)
    except _rocm.NoGPU as e:
        print(f"GPU: none usable ({e}); device='auto' runs on the CPU")
        return 0
    except (ImportError, ValueError) as e:    # no HIP runtime
        print(f"GPU: AMD GPU, not usable: {e}")
        return 1
    vis = " ".join(f"{k}={os.environ.get(k, '-')}" for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                                                              "CUDA_VISIBLE_DEVICES"))
    print(f"GPU rocm:0: {p['arch']} ({p['name']}), {p['total_mem'] / 1e9:.2f} GB, {p['count']} visible; HIP runtime "
          f"{_rocm.version_str(p['runtime'])} ({p['lib']})")
    print(f"visible: {vis}")
    try:
        avail = _rocm.available()
        dists = sorted({_plugin_dist(pkg) for pkg in _rocm.plugins()})
    except ImportError as e:
        print(f"GPU modules: {e}")
        return 1
    untested = [a for a in sorted(avail) if a in _rocm.UNTESTED]
    print(f"GPU modules installed{' (' + ', '.join(dists) + ')' if dists else ''}: "
          f"{', '.join(sorted(avail)) or 'none'}" + (f" ({', '.join(untested)}: built, untested)" if untested else ""))
    print(f"rocBLAS: {_rocblas()}")
    try:
        arch = _rocm.select(0)
    except ImportError as e:
        print(f"NOT NATIVE: {e}")
        if not avail:
            print(f"install: {_rocm.install_hint(p['major'])}")
        return 1
    print(f"native module {arch}" + (" (built, untested on this GPU type)" if arch in _rocm.UNTESTED else ""))
    return 0


def _plugin_dist(pkg) -> str:
    """'<distribution> <version>' of a backend package"""
    from importlib.metadata import packages_distributions
    names = packages_distributions().get(pkg.__name__.split(".")[0], [pkg.__name__])
    return f"{names[0]} {getattr(pkg, '__version__', '?')}"


if __name__ == "__main__":
    sys.exit(main())

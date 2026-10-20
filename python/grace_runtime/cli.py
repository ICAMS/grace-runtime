# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""``grace-runtime models list | download NAME...``, ``grace-runtime info``"""
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
    sub.add_parser("info", help="where calculations will run: CPU module, GPU, driver, GPU module (native or PTX), "
                                "fallback settings, JIT cache")
    a = ap.parse_args(argv)
    if a.cmd == "info":
        return info()
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


def info():
    """print the backend choice of device="auto" / "cuda" without starting Kokkos; exit status 1 when a GPU is there but
    would not run natively (no module of its family: CPU, or PTX with GRACE_RUNTIME_ALLOW_PTX=1)"""
    import grace_runtime
    from . import _cuda
    print(f"grace-runtime {grace_runtime.__version__}, CPU module {grace_runtime.cpu_module}")
    flags = {k: os.environ[k] for k in ("GRACE_RUNTIME_DEVICE", "GRACE_RUNTIME_ALLOW_PTX", "GRACE_RUNTIME_STRICT",
                                        "GRACE_RUNTIME_QUIET", "GRACE_RUNTIME_CUDA_ARCH", "CUDA_VISIBLE_DEVICES")
             if k in os.environ}
    print("settings:", ", ".join(f"{k}={v}" for k, v in flags.items()) or "defaults")
    try:
        major, minor = _cuda.compute_capability(0)
    except ImportError as e:    # no driver / device, or a driver older than CUDA 12
        print(f"GPU: none usable ({e}); device='auto' runs on the CPU")
        return 0 if isinstance(e, _cuda.NoGPU) else 1
    d = _cuda._state["driver"]
    print(f"GPU cuda:0: compute capability {major}.{minor} (sm{major}{minor}); driver for CUDA {d // 1000}.{d % 1000 // 10}")
    try:
        avail = _cuda.available()
    except ImportError as e:
        print(f"GPU modules: {e}")
        return 1
    print(f"GPU modules installed: {', '.join(sorted(avail)) or 'none'}")
    try:
        arch, ptx = _cuda.select(0)
    except ImportError as e:
        print(f"NOT NATIVE: {e}")
        return 1
    if ptx:
        cache = os.environ.get("CUDA_CACHE_PATH") or os.path.expanduser("~/.nv/ComputeCache")
        ok = os.access(cache if os.path.isdir(cache) else os.path.dirname(cache.rstrip("/")) or ".", os.W_OK)
        print(f"NOT NATIVE: runs the {arch} module's PTX, compiled by the driver (JIT cache {cache}, "
              f"{'writable' if ok else 'NOT writable: compiled again in every process'}); a few % slower")
        return 1
    print(f"native module {arch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

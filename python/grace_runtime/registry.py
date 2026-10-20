# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Foundation model names -> ``kokkos.npz`` in the GRACE cache shared with tensorpotential.

``models.json`` is generated from tensorpotential's ``foundation_models.py`` (``ci/gen_grace_runtime_models.py`` there).
Files live in ``$GRACE_CACHE`` or ``~/.cache/grace``, at ``<name>/kokkos.npz``; variants sharing one file
(``X`` and ``X-fp64``) are one entry with aliases, cached once under the entry name. Next to the file,
``.grace_origin.kokkos.json`` records where it came from; tensorpotential writes the same record. Rules:
a record with the entry's URL or sha256 -> use the file; a record grace-runtime wrote with another URL -> download again,
one tensorpotential wrote -> use it with a warning (no re-download ping-pong between releases); no record -> adopt the
file if its sha256 matches, else download again; a downloaded file whose sha256 differs from the registry's -> error.
User entries with a ``kokkos_url`` in ``<cache>/models_registry.yaml`` are read too (no sha256: a warning).
"""
import hashlib
import json
import os
import sys
import urllib.request
import warnings
from datetime import datetime, timezone
from importlib import resources

from tqdm.auto import tqdm

ORIGIN_RECORD = ".grace_origin.kokkos.json"
USER_REGISTRY = "models_registry.yaml"
_CHUNK = 1 << 20


def default_cache_dir():
    return os.environ.get("GRACE_CACHE") or os.path.expanduser("~/.cache/grace")


def load_entries():
    """The bundled registry: {"models": [{name, aliases, arch, url, sha256, uq}], "unsupported": {name: reason}}."""
    return json.loads(resources.files(__package__).joinpath("models.json").read_text())


def _user_entries(cache):
    path = os.path.join(cache, USER_REGISTRY)
    if not os.path.isfile(path):
        return []
    try:
        import yaml
    except ImportError:
        return []
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return [{"name": k, "aliases": [], "arch": None, "url": v["kokkos_url"], "sha256": None}
            for k, v in data.items() if isinstance(v, dict) and v.get("kokkos_url")]


def _lookup(name, entries, cache):
    models = entries.get("models", [])
    for m in models + _user_entries(cache):
        if name == m["name"] or name in m.get("aliases", []):
            return m
    reason = entries.get("unsupported", {}).get(name)
    if reason:
        raise ValueError(f"{name}: {reason}")
    known = sorted(n for m in models for n in [m["name"], *m.get("aliases", [])])
    raise KeyError(f"unknown GRACE model {name!r}; known: {', '.join(known)}")


def list_models(entries=None):
    return list((entries or load_entries())["models"])


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(_CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def _read_record(directory):
    try:
        with open(os.path.join(directory, ORIGIN_RECORD)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_record(directory, m, sha):
    from . import __version__
    url = m["url"]
    rec = {"model_name": m["name"], "payload": "kokkos", "url": url,
           "tag": url.split("/resolve/", 1)[1].split("/", 1)[0] if "/resolve/" in url else None,
           "sha256": sha, "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "writer": f"grace-runtime {__version__}"}
    try:
        with open(os.path.join(directory, ORIGIN_RECORD), "w") as f:
            json.dump(rec, f, indent=2)
    except OSError:
        pass


def _download(m, directory, target, quiet):
    os.makedirs(directory, exist_ok=True)
    part = f"{target}.part-{os.getpid()}"
    if not quiet:
        print(f"[grace-runtime] downloading {m['name']} from {m['url']}", file=sys.stderr)
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(m["url"], timeout=60) as r, open(part, "wb") as f:
            try:
                total = int(r.headers.get("Content-Length", ""))
            except (TypeError, ValueError):
                total = None
            if total is not None and total <= 0:
                total = None
            with tqdm(total=total, desc=f"{m['name']}/kokkos.npz", unit="B", unit_scale=True,
                      unit_divisor=1024, file=sys.stderr, disable=quiet) as progress:
                for b in iter(lambda: r.read(_CHUNK), b""):
                    h.update(b)
                    f.write(b)
                    progress.update(len(b))
        sha = h.hexdigest()
        if m.get("sha256") and sha != m["sha256"]:
            raise RuntimeError(f"{m['name']}: the model file changed on the hub (sha256 {sha}, registry "
                               f"{m['sha256']}); update grace-runtime")
        os.replace(part, target)    # atomic: another process never sees a partial kokkos.npz
    finally:
        if os.path.exists(part):
            os.remove(part)
    _write_record(directory, m, sha)


def resolve(name, *, cache_dir=None, entries=None, force=False, quiet=False):
    """Path of the ``kokkos.npz`` for a registry name (downloaded if needed), or ``name`` itself if it is a file."""
    if os.path.isfile(name):
        return name
    cache = cache_dir or default_cache_dir()
    m = _lookup(name, entries or load_entries(), cache)
    directory = os.path.join(cache, m["name"])
    target = os.path.join(directory, "kokkos.npz")
    if m.get("sha256") is None:
        warnings.warn(f"{m['name']}: user registry entry with no sha256; the file is not verified", UserWarning)
    if os.path.isfile(target) and not force:
        rec = _read_record(directory)
        if rec is not None and rec.get("url") == m["url"]:
            return target
        if rec is not None and m.get("sha256") and rec.get("sha256") == m["sha256"]:
            return target    # the same file under another release tag
        if rec is not None and not str(rec.get("writer", "")).startswith("grace-runtime"):
            # tensorpotential's file from its own registry snapshot: use it rather than replace it, else two packages
            # of different releases re-download in turn
            warnings.warn(f"{m['name']}: using the kokkos.npz {rec.get('writer')} downloaded from {rec.get('url')}; "
                          f"this grace-runtime lists {m['url']} (grace-runtime models download --force "
                          f"{m['name']} replaces it)", UserWarning)
            return target
        if rec is None and m.get("sha256"):
            sha = _sha256(target)
            if sha == m["sha256"]:
                _write_record(directory, m, sha)
                return target
        if rec is None and m.get("sha256") is None:
            return target
    _download(m, directory, target, quiet)
    return target

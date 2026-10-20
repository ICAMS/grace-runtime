# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Model registry: names -> kokkos.npz in the shared GRACE cache, origin record, sha256 rules (spec component 3)."""
import hashlib
import http.server
import io
import json
import os
import threading

import pytest

from grace_runtime import registry

PAYLOAD = b"fake kokkos.npz payload " * 100
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _Handler(http.server.BaseHTTPRequestHandler):
    hits = []
    files = {}

    def do_GET(self):
        _Handler.hits.append(self.path)
        body = _Handler.files.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    _Handler.hits, _Handler.files = [], {"/m/A-kokkos.npz": PAYLOAD}
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _entries(base, sha=SHA, url="/m/A-kokkos.npz"):
    return {"schema": 1, "models": [{"name": "A", "aliases": ["A-fp64", "A-short"], "arch": "2L", "url": base + url,
                                     "sha256": sha, "uq": True}],
            "unsupported": {"B-3L": "3L models are not supported by grace-runtime 0.1",
                            "C-FS": "legacy model without a kokkos.npz export"}}


def _record(cache, name="A"):
    with open(os.path.join(cache, name, registry.ORIGIN_RECORD)) as f:
        return json.load(f)


def test_download_writes_file_and_record(server, tmp_path):
    p = registry.resolve("A", cache_dir=str(tmp_path), entries=_entries(server))
    assert p == str(tmp_path / "A" / "kokkos.npz")
    assert open(p, "rb").read() == PAYLOAD
    rec = _record(tmp_path)
    assert (rec["url"], rec["sha256"], rec["payload"], rec["model_name"]) == (server + "/m/A-kokkos.npz", SHA, "kokkos", "A")
    assert rec["writer"].startswith("grace-runtime")
    assert not [f for f in os.listdir(tmp_path / "A") if ".part" in f]


@pytest.mark.parametrize("length", [str(len(PAYLOAD)), None, "invalid", "0"])
def test_download_progress(tmp_path, monkeypatch, capsys, length):
    response = io.BytesIO(PAYLOAD)
    response.headers = {} if length is None else {"Content-Length": length}
    monkeypatch.setattr(registry.urllib.request, "urlopen", lambda *a, **k: response)
    monkeypatch.setattr(registry, "_CHUNK", 100)
    path = registry.resolve("A", cache_dir=str(tmp_path), entries=_entries("https://example.org"))
    assert open(path, "rb").read() == PAYLOAD
    output = capsys.readouterr().err
    assert "A/kokkos.npz" in output
    assert "B" in output
    if length == str(len(PAYLOAD)):
        assert "100%" in output
    else:
        assert "100%" not in output
    # A cache hit must not show another progress bar or make another request.
    registry.resolve("A", cache_dir=str(tmp_path), entries=_entries("https://example.org"))
    assert capsys.readouterr().err == ""


def test_quiet_download_has_no_output(server, tmp_path, capsys):
    path = registry.resolve("A", cache_dir=str(tmp_path), entries=_entries(server), quiet=True)
    assert open(path, "rb").read() == PAYLOAD
    output = capsys.readouterr()
    assert output.out == output.err == ""


def test_aliases_share_one_file(server, tmp_path):
    e = _entries(server)
    paths = {registry.resolve(n, cache_dir=str(tmp_path), entries=e) for n in ("A", "A-fp64", "A-short")}
    assert paths == {str(tmp_path / "A" / "kokkos.npz")}
    assert len(_Handler.hits) == 1


def test_cached_with_matching_record_needs_no_network(server, tmp_path):
    e = _entries(server)
    registry.resolve("A", cache_dir=str(tmp_path), entries=e)
    _Handler.files.clear()
    assert registry.resolve("A", cache_dir=str(tmp_path), entries=e) == str(tmp_path / "A" / "kokkos.npz")
    assert len(_Handler.hits) == 1


def test_file_without_record_is_adopted_when_sha_matches(server, tmp_path):
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "kokkos.npz").write_bytes(PAYLOAD)
    registry.resolve("A", cache_dir=str(tmp_path), entries=_entries(server))
    assert _Handler.hits == []
    assert _record(tmp_path)["sha256"] == SHA


def test_file_without_record_and_other_sha_is_replaced(server, tmp_path):
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "kokkos.npz").write_bytes(b"old export")
    p = registry.resolve("A", cache_dir=str(tmp_path), entries=_entries(server))
    assert open(p, "rb").read() == PAYLOAD and len(_Handler.hits) == 1


PAYLOAD2 = b"newer kokkos.npz payload " * 100
SHA2 = hashlib.sha256(PAYLOAD2).hexdigest()


def test_own_record_with_other_url_is_redownloaded(server, tmp_path):
    registry.resolve("A", cache_dir=str(tmp_path), entries=_entries(server))
    _Handler.files["/m2/A-kokkos.npz"] = PAYLOAD2
    registry.resolve("A", cache_dir=str(tmp_path), entries=_entries(server, sha=SHA2, url="/m2/A-kokkos.npz"))
    assert _Handler.hits == ["/m/A-kokkos.npz", "/m2/A-kokkos.npz"]
    assert _record(tmp_path)["url"].endswith("/m2/A-kokkos.npz")
    assert (tmp_path / "A" / "kokkos.npz").read_bytes() == PAYLOAD2


def test_record_with_other_url_but_same_sha_is_kept(server, tmp_path):
    """The same file under another release tag: no download."""
    registry.resolve("A", cache_dir=str(tmp_path), entries=_entries(server))
    _Handler.files["/m2/A-kokkos.npz"] = PAYLOAD
    registry.resolve("A", cache_dir=str(tmp_path), entries=_entries(server, url="/m2/A-kokkos.npz"))
    assert _Handler.hits == ["/m/A-kokkos.npz"]


def _tp_record(tmp_path, url, sha):
    (tmp_path / "A").mkdir(exist_ok=True)
    (tmp_path / "A" / "kokkos.npz").write_bytes(b"tensorpotential's file")
    (tmp_path / "A" / registry.ORIGIN_RECORD).write_text(
        json.dumps({"url": url, "sha256": sha, "writer": "tensorpotential 0.6.1"}))


def test_file_from_tensorpotential_with_other_url_is_used(server, tmp_path):
    """Two packages of different releases must not re-download in turn: tensorpotential's file is used, with a
    warning, and only files grace-runtime wrote itself are refreshed."""
    _tp_record(tmp_path, "https://hub/resolve/kk-other/kokkos/A-kokkos.npz", "2" * 64)
    with pytest.warns(UserWarning, match="tensorpotential"):
        p = registry.resolve("A", cache_dir=str(tmp_path), entries=_entries(server))
    assert open(p, "rb").read() == b"tensorpotential's file" and _Handler.hits == []


def test_force_downloads_again(server, tmp_path):
    _tp_record(tmp_path, "https://hub/resolve/kk-other/kokkos/A-kokkos.npz", "2" * 64)
    registry.resolve("A", cache_dir=str(tmp_path), entries=_entries(server), force=True)
    assert (tmp_path / "A" / "kokkos.npz").read_bytes() == PAYLOAD and _record(tmp_path)["sha256"] == SHA


def test_sha_mismatch_after_download_raises_and_leaves_nothing(server, tmp_path):
    with pytest.raises(RuntimeError, match="changed on the hub"):
        registry.resolve("A", cache_dir=str(tmp_path), entries=_entries(server, sha="0" * 64))
    assert not os.path.exists(tmp_path / "A" / "kokkos.npz")
    assert not [f for f in os.listdir(tmp_path / "A") if ".part" in f]


def test_unknown_name_lists_known(server, tmp_path):
    with pytest.raises(KeyError, match="A-fp64"):
        registry.resolve("Z", cache_dir=str(tmp_path), entries=_entries(server))


def test_unsupported_name_gives_reason(server, tmp_path):
    with pytest.raises(ValueError, match="3L models are not supported"):
        registry.resolve("B-3L", cache_dir=str(tmp_path), entries=_entries(server))


def test_existing_path_passes_through(tmp_path):
    f = tmp_path / "my.npz"
    f.write_bytes(b"x")
    assert registry.resolve(str(f), cache_dir=str(tmp_path / "c"), entries={"models": []}) == str(f)


def test_user_registry_entry_with_kokkos_url(server, tmp_path):
    yaml = pytest.importorskip("yaml")
    with open(tmp_path / "models_registry.yaml", "w") as f:
        yaml.safe_dump({"MyModel": {"kokkos_url": server + "/m/A-kokkos.npz", "url": "ignored"}}, f)
    with pytest.warns(UserWarning, match="no sha256"):
        p = registry.resolve("MyModel", cache_dir=str(tmp_path), entries=_entries(server))
    assert open(p, "rb").read() == PAYLOAD


def test_bundled_registry_is_well_formed():
    e = registry.load_entries()
    names = [n for m in e["models"] for n in [m["name"], *m.get("aliases", [])]]
    assert len(names) == len(set(names)) > 0
    for m in e["models"]:
        assert m["arch"] in ("1L", "2L", "3L")
        assert m["url"].startswith("https://") and len(m["sha256"]) == 64
    assert not set(names) & set(e.get("unsupported", {}))


def test_calculator_resolves_registry_names(monkeypatch, models):
    import grace_runtime.calculator as gc
    seen = []
    monkeypatch.setattr(gc.registry, "resolve", lambda name, **k: seen.append(name) or models["1l"])
    gc.GRACERuntimeCalculator("GRACE-1L-OMAT-large-base")
    assert seen == ["GRACE-1L-OMAT-large-base"]


def test_cli_list_and_download(server, tmp_path, monkeypatch, capsys):
    from grace_runtime import cli
    monkeypatch.setattr(registry, "load_entries", lambda: _entries(server))
    monkeypatch.setenv("GRACE_CACHE", str(tmp_path))
    assert cli.main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "A" in out and "A-fp64" in out and "2L" in out and "not cached" in out
    assert cli.main(["models", "download", "A-short"]) == 0
    assert (tmp_path / "A" / "kokkos.npz").read_bytes() == PAYLOAD
    cli.main(["models", "list"])
    assert "cached" in capsys.readouterr().out.replace("not cached", "")
    assert cli.main(["models", "download", "Z"]) == 1
    (tmp_path / "A" / "kokkos.npz").write_bytes(b"damaged")
    assert cli.main(["models", "download", "--force", "A"]) == 0
    assert (tmp_path / "A" / "kokkos.npz").read_bytes() == PAYLOAD


def test_3l_models_are_listed():
    names = {m["name"]: m for m in registry.list_models()}
    for n in ("GRACE-3L-OMAT-large", "GRACE-3L-OMAT-large-ft-AM"):
        assert names[n]["arch"] == "3L" and names[n]["url"].endswith("kokkos.npz")

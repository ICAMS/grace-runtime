# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""The API reference reads source without running package code or loading optional dependencies."""
import ast
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "docs/hooks/api_reference.py"


@pytest.fixture
def reference():
    spec = importlib.util.spec_from_file_location("api_reference", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_api_entries_render_without_site_packages():
    """Even Model and TorchSim metadata can be read without importing any inference package."""
    code = r'''
import ast
import re
import runpy
import sys
from types import SimpleNamespace
hook = runpy.run_path(sys.argv[1])
root = hook["ROOT"]
source = (root / "docs/api.md").read_text()
page = SimpleNamespace(file=SimpleNamespace(src_uri="api.md"))
rendered = hook["on_page_markdown"](source, page)
assert "<!-- api:" not in rendered
assert "Model(path, spline=1024, store_max_gb=None, asi=None)" in rendered
signatures = re.findall(r"```text\n(.*?)\n```", rendered, re.S)
for signature in signatures:
    ast.parse("def " + signature + ":\n    pass")
signature = next(s for s in signatures if s.startswith("GRACERuntimeCalculator("))
args = ast.parse("def " + signature + ":\n    pass").body[0].args
defaults = dict(zip([a.arg for a in args.args[-len(args.defaults):]], args.defaults))
assert ast.literal_eval(defaults["energy_only"]) is False
assert "GRACERuntimeModel(" in rendered
assert "**model:**" in rendered
assert not any(name in sys.modules for name in ("grace_runtime", "numpy", "torch", "torch_sim", "mkdocs"))
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(HOOK)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_source_changes_update_signatures_and_docstrings(reference, tmp_path, monkeypatch):
    source = tmp_path / "python/grace_runtime"
    source.mkdir(parents=True)
    path = source / "example.py"
    monkeypatch.setattr(reference, "ROOT", tmp_path)
    for value in (7, 19):
        path.write_text(f'''raise RuntimeError("do not import")
def example(limit: int = {value}):
    """limit: source value {value}

    The native code: stays as prose.
    """
''')
        rendered = reference.render_api("example.example")
        signature = re.search(r"```text\n(.*?)\n```", rendered, re.S)[1]
        args = ast.parse("def " + signature + ":\n    pass").body[0].args
        assert ast.literal_eval(args.defaults[0]) == value
        assert f"source value {value}" in rendered
        assert "**limit:**" in rendered
        assert "**The native code:**" not in rendered


def test_unknown_api_entry_fails_instead_of_silently_disappearing(reference):
    with pytest.raises(ValueError, match="Unknown API entry"):
        reference.render_api("batch.no_such_function")


@pytest.mark.parametrize("separator", [" ", "\n           "])
def test_native_model_signature_handles_wrapped_optional_defaults(reference, tmp_path, monkeypatch, separator):
    source = tmp_path / "python/src"
    source.mkdir(parents=True)
    (source / "core.cpp").write_text(f'''nb::class_<Model>(m, "Model")
      .def(nb::init<const std::string &, int, std::optional<double>>(), "path"_a,{separator}"spline"_a = 2048, "store_max_gb"_a = nb::none())
      .def_ro("arch", &Model::arch_);
''')
    monkeypatch.setattr(reference, "ROOT", tmp_path)
    assert reference._model_signature() == "Model(path, spline=2048, store_max_gb=None, asi=None)"


def test_other_pages_are_not_rewritten(reference):
    source = "<!-- api: batch.no_such_function -->"
    page = SimpleNamespace(file=SimpleNamespace(src_uri="usage.md"))
    assert reference.on_page_markdown(source, page) == source

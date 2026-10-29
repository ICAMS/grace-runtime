# SPDX-License-Identifier: LicenseRef-ASL   (Academic Software Licence, see LICENSE.md)
"""Render API signatures and docstrings from source, without importing grace_runtime."""
import ast
import copy
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MARKER = re.compile(r"<!-- api: ([\w.]+) -->")


def _signature(node, name):
    """Python signature, with self removed from constructors and methods."""
    if isinstance(node, ast.ClassDef):
        node = next(child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "__init__")
    args = copy.deepcopy(node.args)
    positional = args.posonlyargs or args.args
    if positional and positional[0].arg == "self":
        positional.pop(0)
    parameters = ast.unparse(args)
    if len(name) + len(parameters) > 95:
        parameters = "\n    " + parameters.replace(", ", ",\n    ") + "\n"
    result = f"{name}({parameters})"
    if node.returns:
        result += " -> " + ast.unparse(node.returns)
    return result


def _model_signature():
    """The Model wrapper forwards its arguments to this native constructor."""
    bindings = (ROOT / "python/src/core.cpp").read_text()
    model = bindings.split('nb::class_<Model>(m, "Model")', 1)[1]
    constructor = re.search(r'\.def\(nb::init<.*?>\(\),\s*(.*?)\)\s*(?=\.def)', model, re.S)
    if constructor is None:
        raise ValueError("Cannot read the native Model constructor")
    args = re.findall(r'"(\w+)"_a(?:\s*=\s*(nb::none\(\)|[^,)]+))?', constructor[1])
    if not args:
        raise ValueError("The native Model constructor has no named arguments")
    parameters = [name + ("=" + value.strip().replace("nb::none()", "None") if value else "")
                  for name, value in args]
    if not any(name == "asi" for name, _ in args):
        parameters.append("asi=None")
    return "Model(" + ", ".join(parameters) + ")"


def render_api(symbol):
    """Read one class, function or method; never execute the source module."""
    module, *names = symbol.split(".")
    path = ROOT / "python/grace_runtime" / f"{module}.py"
    node = ast.parse(path.read_text(), filename=str(path))
    for name in names:
        matches = [child for child in node.body if isinstance(child, (ast.ClassDef, ast.FunctionDef)) and child.name == name]
        if len(matches) != 1:
            raise ValueError(f"Unknown API entry: {symbol}")
        node = matches[0]
    if not names:
        raise ValueError(f"API entry needs a symbol: {symbol}")
    signature = _model_signature() if symbol == "_core.Model" else _signature(node, names[-1])
    doc = ast.get_docstring(node) or ""
    # Existing docstrings use name: description paragraphs rather than a parser-specific format.
    doc = re.sub(r"^([A-Za-z_]\w*(?:\s*[,/]\s*[A-Za-z_]\w*)*): (.+)$", r"\n**\1:** \2", doc, flags=re.M)
    source = path.relative_to(ROOT).as_posix()
    return f"```text\n{signature}\n```\n\n{doc}\n\n[Source](https://github.com/ICAMS/grace-runtime/blob/master/{source}#L{node.lineno})\n"


def on_page_markdown(markdown, page, **kwargs):
    if page.file.src_uri != "api.md":
        return markdown
    return MARKER.sub(lambda match: render_api(match[1]), markdown)

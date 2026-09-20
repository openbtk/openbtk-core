"""The documentation site is code: every claim on it that can be executed or
resolved is, on every change.

* each ``python`` block in a guide runs (in order, in one namespace, as a real
  module so anything it defines behaves as it would for a reader);
* each ``yaml`` block that is a pipeline config validates against the registries;
* each ``openbtk ...`` command in a ``bash`` block parses with the real CLI parser
  (a renamed flag or subcommand fails here, not for a reader);
* each ``::: dotted.path`` API directive resolves to a real object;
* every page under ``mkdocs/`` is in the navigation, and every nav entry exists.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import re
import shlex
import sys
import types
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "mkdocs"
_GUIDES = sorted((_DOCS / "guides").glob("*.md"))

# Guides that need an optional extra to execute their code.
_REQUIRES: dict[str, tuple[str, ...]] = {
    "ehr.md": ("fhir.resources", "pyarrow", "hl7apy"),
}


def _blocks(path: Path, lang: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(rf"```{lang}\n(.*?)```", text, re.DOTALL)


def test_there_are_guides() -> None:
    assert {p.name for p in _GUIDES} >= {
        "clinical-text.md",
        "deidentification.md",
        "ehr.md",
        "evaluation.md",
        "cli.md",
    }


@pytest.mark.parametrize("guide", _GUIDES, ids=lambda p: p.name)
def test_every_python_block_runs(guide: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for module in _REQUIRES.get(guide.name, ()):
        pytest.importorskip(module)
    module = types.ModuleType(f"guide_{guide.stem.replace('-', '_')}")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for i, block in enumerate(_blocks(guide, "python")):
        code = compile(block, f"{guide.name}[python block {i}]", "exec")
        with contextlib.redirect_stdout(io.StringIO()):  # library logs
            exec(code, module.__dict__)


@pytest.mark.parametrize("guide", _GUIDES, ids=lambda p: p.name)
def test_every_pipeline_yaml_block_validates(guide: Path) -> None:
    from openbtk.cli._components import load_components
    from openbtk.core.config import PipelineConfig

    with contextlib.redirect_stdout(io.StringIO()):
        load_components()
        for block in _blocks(guide, "yaml"):
            data = yaml.safe_load(block)
            if not isinstance(data, dict) or "steps" not in data:
                continue
            issues = PipelineConfig.model_validate(data).validate_registry()
            assert [i.message for i in issues if i.severity == "error"] == []


@pytest.mark.parametrize("guide", _GUIDES, ids=lambda p: p.name)
def test_every_cli_example_parses(guide: Path) -> None:
    from openbtk.cli.main import build_parser

    parser = build_parser()
    seen = 0
    for block in _blocks(guide, "bash"):
        for line in block.splitlines():
            command = line.split("#", 1)[0].strip()
            if not command.startswith("openbtk "):
                continue
            seen += 1
            try:
                parser.parse_args(shlex.split(command)[1:])
            except SystemExit as e:  # argparse exits on a bad command line
                pytest.fail(
                    f"{guide.name}: `{command}` is not a valid command ({e.code})"
                )
    if guide.name == "cli.md":
        assert seen >= 10


def test_cli_guide_documents_every_subcommand() -> None:
    from openbtk.cli.main import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    text = (_DOCS / "guides" / "cli.md").read_text(encoding="utf-8")
    for name in sub.choices:
        assert f"openbtk {name}" in text, f"cli.md does not document `openbtk {name}`"


_DIRECTIVE = re.compile(r"^::: ([\w.]+)\s*$", re.MULTILINE)
_API_PAGES = sorted((_DOCS / "api").glob("*.md"))


def _resolve(dotted: str) -> object:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                obj: object = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(dotted)


@pytest.mark.parametrize("page", _API_PAGES, ids=lambda p: p.name)
def test_every_api_directive_resolves(page: Path) -> None:
    if page.name == "integrations.md":
        pytest.importorskip("langchain_core")
    directives = _DIRECTIVE.findall(page.read_text(encoding="utf-8"))
    assert directives, f"{page.name} documents nothing"
    for dotted in directives:
        assert _resolve(dotted) is not None, dotted


def test_navigation_and_files_agree() -> None:
    config = yaml.safe_load((_ROOT / "mkdocs.yml").read_text(encoding="utf-8"))

    def entries(node: object) -> list[str]:
        if isinstance(node, str):
            return [node]
        if isinstance(node, list):
            return [e for item in node for e in entries(item)]
        if isinstance(node, dict):
            return [e for value in node.values() for e in entries(value)]
        return []

    in_nav = set(entries(config["nav"]))
    on_disk = {
        str(p.relative_to(_DOCS)).replace("\\", "/") for p in _DOCS.rglob("*.md")
    }
    assert in_nav == on_disk, {
        "only in nav": in_nav - on_disk,
        "not in nav": on_disk - in_nav,
    }


def test_tutorial_links_point_at_real_notebooks() -> None:
    text = (_DOCS / "tutorials.md").read_text(encoding="utf-8")
    linked = re.findall(r"/notebooks/([\w-]+\.ipynb)\)", text)
    assert len(linked) == 8, linked
    for name in linked:
        assert (_ROOT / "notebooks" / name).is_file(), name

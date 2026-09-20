"""Extract OpenBTK's public API surface as plain data, for the freeze.

**What is public.** Exactly what the API reference documents: every
``::: dotted.path`` directive under ``mkdocs/api/``, plus the registry keys and
the command-line surface. The documentation is the single definition, so
"documented" and "frozen" can never disagree, and adding a page to the API
reference is what puts something under the stability promise.

The snapshot records, for each documented object:

* a **function**: its signature;
* a **class**: its constructor signature and every public member (methods with
  signatures, properties, class-level constants), including members inherited
  from other OpenBTK classes but not from the standard library or pydantic;
* a **pydantic model**: its field names and whether each is required (not the
  annotations, whose ``repr`` differs between Python versions);
* an **enum**: its member names and values.

Signatures are rendered by :func:`_signature` from the *source* annotations
(every module uses ``from __future__ import annotations``), with object defaults
reduced to a stable name, so the snapshot is identical on Python 3.11-3.13 and
on every OS. Run ``python tests/api/surface.py --update`` to regenerate it after
a *deliberate* change.
"""

from __future__ import annotations

import contextlib
import enum
import importlib
import inspect
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_API_PAGES = _ROOT / "mkdocs" / "api"
SNAPSHOT = Path(__file__).with_name("api_surface.json")
REGISTRY_SNAPSHOT = Path(__file__).with_name("registry_keys.json")

_DIRECTIVE = re.compile(r"^::: ([\w.]+)\s*$", re.MULTILINE)
_LITERALS = (str, int, float, bool, type(None))


def _default(value: object) -> str:
    if isinstance(value, _LITERALS):
        return repr(value)
    if isinstance(value, (tuple, list)) and all(
        isinstance(v, _LITERALS) for v in value
    ):
        return repr(value)
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    name = getattr(value, "__qualname__", None) or type(value).__name__
    return f"<{name}>"


def _signature(obj: Any) -> str:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(...)"
    parts: list[str] = []
    seen_kw_marker = False
    for p in sig.parameters.values():
        if p.name in {"self", "cls"}:
            continue
        text = p.name
        if p.kind is p.VAR_POSITIONAL:
            text = "*" + text
            seen_kw_marker = True
        elif p.kind is p.VAR_KEYWORD:
            text = "**" + text
        elif p.kind is p.KEYWORD_ONLY and not seen_kw_marker:
            parts.append("*")
            seen_kw_marker = True
        if p.annotation is not p.empty:
            ann = p.annotation if isinstance(p.annotation, str) else repr(p.annotation)
            text += f": {ann}"
        if p.default is not p.empty:
            text += f" = {_default(p.default)}"
        parts.append(text)
    ret = ""
    if sig.return_annotation is not sig.empty:
        r = sig.return_annotation
        ret = f" -> {r if isinstance(r, str) else repr(r)}"
    return f"({', '.join(parts)}){ret}"


def _is_openbtk(cls: type) -> bool:
    return cls.__module__.split(".")[0] == "openbtk"


def _pydantic_fields(cls: type) -> dict[str, bool] | None:
    fields = getattr(cls, "model_fields", None)
    if not isinstance(fields, dict):
        return None
    return {name: bool(f.is_required()) for name, f in sorted(fields.items())}


def _members(cls: type) -> dict[str, str]:
    """Public members declared on ``cls`` or on an OpenBTK base of it."""
    out: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        if not _is_openbtk(klass):
            continue
        for name, raw in vars(klass).items():
            if name.startswith("_") or name in {"model_config"}:
                continue
            if isinstance(raw, property):
                out[name] = "property"
            elif isinstance(raw, (staticmethod, classmethod)):
                out[name] = ("static" if isinstance(raw, staticmethod) else "class") + (
                    _signature(raw.__func__)
                )
            elif inspect.isfunction(raw):
                out[name] = _signature(raw)
            elif isinstance(raw, _LITERALS):
                out[name] = f"= {raw!r}"
    return dict(sorted(out.items()))


def describe(obj: object) -> dict[str, Any]:
    if inspect.isclass(obj):
        if issubclass(obj, enum.Enum):
            return {"kind": "enum", "members": {m.name: m.value for m in obj}}
        entry: dict[str, Any] = {"kind": "class"}
        fields = _pydantic_fields(obj)
        if fields is not None:
            # A pydantic model's __init__ signature carries real annotation
            # objects whose repr differs across versions; its fields are the API.
            entry["fields"] = fields
        else:
            entry["init"] = _signature(obj)
        entry["members"] = _members(obj)
        return entry
    if callable(obj):
        return {"kind": "function", "signature": _signature(obj)}
    return {"kind": "other", "repr": type(obj).__name__}


def resolve(dotted: str) -> object:
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


def documented_targets() -> list[str]:
    targets: list[str] = []
    for page in sorted(_API_PAGES.glob("*.md")):
        targets += _DIRECTIVE.findall(page.read_text(encoding="utf-8"))
    return targets


def collect_api() -> dict[str, dict[str, Any]]:
    """The API snapshot. A target whose optional dependency is not installed
    (the LangChain adapter without ``langchain-core``) is omitted, not faked."""
    out: dict[str, dict[str, Any]] = {}
    for dotted in documented_targets():
        try:
            out[dotted] = describe(resolve(dotted))
        except ImportError:
            continue
        except (
            Exception
        ) as e:  # missing optional dependency raises MissingDependencyError
            if type(e).__name__ != "MissingDependencyError":
                raise
    return dict(sorted(out.items()))


def collect_cli() -> dict[str, Any]:
    """Sub-commands and their flags: the command line is public API too."""
    from openbtk.cli.main import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    commands: dict[str, Any] = {}
    for name, p in sorted(sub.choices.items()):
        commands[name] = sorted(
            "/".join(a.option_strings) if a.option_strings else a.dest
            for a in p._actions
            if a.dest != "help"
        )
    return commands


def collect_registry() -> dict[str, list[str]]:
    with contextlib.redirect_stdout(io.StringIO()):
        from openbtk.cli._components import load_components
        from openbtk.core.registry import get_registry, list_categories

        load_components()
        return {c: get_registry(c).list_keys() for c in list_categories()}


def collect_all() -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        import openbtk
        from openbtk.core.provenance import RunManifest
        from openbtk.eval.manifest import EvalManifest

    return {
        "top_level": sorted(getattr(openbtk, "__all__", [])),
        "schema_versions": {
            "RunManifest": RunManifest.model_fields["manifest_version"].default,
            "EvalManifest": EvalManifest.model_fields["manifest_version"].default,
        },
        "api": collect_api(),
        "cli": collect_cli(),
    }


def _dump(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    if "--registry" in argv:
        # Printed for a fresh-interpreter check: the registries of a long-lived
        # test process also hold every test double other modules registered.
        sys.stdout.write(_dump(collect_registry()))
        return 0
    if "--update" not in argv:
        sys.stderr.write("usage: python tests/api/surface.py --update\n")
        return 2
    SNAPSHOT.write_text(_dump(collect_all()), encoding="utf-8")
    REGISTRY_SNAPSHOT.write_text(_dump(collect_registry()), encoding="utf-8")
    sys.stdout.write(f"wrote {SNAPSHOT.name} and {REGISTRY_SNAPSHOT.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

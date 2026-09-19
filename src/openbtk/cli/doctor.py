"""``openbtk doctor``: what is installed, what is missing, and what to do.

Everything here is a *presence* check. Extras are probed with
``importlib.util.find_spec`` (nothing heavy is imported, so ``doctor`` stays
fast and cannot itself fail on a broken optional dependency). Credentials are
reported as set / not set **by name only** -- a value is never read into the
report, printed or logged.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

# extra -> the import names whose presence means the extra is usable. Kept in
# step with pyproject.toml by a test (tests/unit/cli/test_doctor.py).
EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "text": (
        "scispacy",
        "medspacy",
        "presidio_analyzer",
        "presidio_anonymizer",
        "transformers",
        "spacy",
        "pandas",
    ),
    "ehr": ("fhir.resources", "pyarrow", "pandas", "hl7apy"),
    "retrieval": ("faiss", "chromadb", "qdrant_client"),
    "llms": ("openai", "anthropic", "torch"),
    "langchain": ("langchain_core",),
    "langgraph": ("langgraph",),
}

CORE_DEPENDENCIES = (
    "pydantic",
    "numpy",
    "structlog",
    "pyyaml",
    "httpx",
    "typing-extensions",
)

# Conventional names, checked by name only. OpenBTK reads none of them
# implicitly: a config opts in with ${NAME}, or the provider's own SDK does.
CREDENTIAL_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
    "UMLS_API_KEY",
)

SPACY_MODEL = "en_core_web_sm"


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # find_spec("a.b") imports "a"; a broken parent counts as missing.
        return False


def _dist_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def collect() -> dict[str, Any]:
    """Gather the diagnostic report as plain, JSON-serialisable data."""
    extras: dict[str, Any] = {}
    for extra, modules in EXTRA_MODULES.items():
        missing = [m for m in modules if not _has_module(m)]
        extras[extra] = {"installed": not missing, "missing": missing}

    spacy_ready = _has_module("spacy") and _has_module(SPACY_MODEL)
    hints: list[str] = []
    for extra, info in extras.items():
        if not info["installed"]:
            names = ", ".join(info["missing"])
            hints.append(f"pip install 'openbtk[{extra}]'  (missing: {names})")
    if extras["text"]["installed"] and not spacy_ready:
        hints.append(
            f"python -m spacy download {SPACY_MODEL}  (needed by the NER recognizer)"
        )

    return {
        "openbtk": _dist_version("openbtk") or "unknown (not installed as a package)",
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "executable": sys.executable,
        "core_dependencies": {name: _dist_version(name) for name in CORE_DEPENDENCIES},
        "extras": extras,
        "models": {f"spacy:{SPACY_MODEL}": spacy_ready},
        "credentials": {
            name: bool(os.environ.get(name)) for name in CREDENTIAL_ENV_VARS
        },
        "vocabularies": {
            "bundled_icd10cm_subset": _has_module("openbtk.terminology.bundled"),
            "umls": "needs your own licence and an API key (never bundled)",
        },
        "hints": hints,
    }


def format_text(report: dict[str, Any]) -> str:
    """A human-readable rendering of :func:`collect`'s report."""
    lines = [
        f"openbtk {report['openbtk']}  |  Python {report['python']}  "
        f"|  {report['platform']}",
        "",
        "Core dependencies",
    ]
    for name, ver in report["core_dependencies"].items():
        lines.append(
            f"  {'ok     ' if ver else 'MISSING'}  {name} {ver or ''}".rstrip()
        )
    lines += ["", "Optional extras"]
    for extra, info in report["extras"].items():
        state = "ok     " if info["installed"] else "missing"
        detail = "" if info["installed"] else f"  (needs: {', '.join(info['missing'])})"
        lines.append(f"  {state}  {extra}{detail}")
    lines += ["", "Models"]
    for name, ready in report["models"].items():
        lines.append(f"  {'ok     ' if ready else 'missing'}  {name}")
    lines += ["", "Credentials (set / not set -- values are never shown)"]
    for name, present in report["credentials"].items():
        lines.append(f"  {'set    ' if present else 'not set'}  {name}")
    lines += ["", "Vocabularies"]
    vocab = report["vocabularies"]
    bundled = "ok     " if vocab["bundled_icd10cm_subset"] else "missing"
    lines.append(f"  {bundled}  bundled ICD-10-CM subset")
    lines.append(f"  n/a      UMLS: {vocab['umls']}")
    if report["hints"]:
        lines += ["", "To fix"]
        lines += [f"  {h}" for h in report["hints"]]
    return "\n".join(lines) + "\n"


def unmet(report: dict[str, Any], required: list[str]) -> list[str]:
    """The names in ``required`` that are not installed (unknown names too)."""
    return [
        name
        for name in required
        if name not in report["extras"] or not report["extras"][name]["installed"]
    ]

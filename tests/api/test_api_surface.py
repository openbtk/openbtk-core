"""The v1.0 stability promise, enforced (M11 task 11.1, NFR-13).

Three things are frozen and compared against checked-in snapshots:

* the **registry keys** -- "keys are permanent" (CLAUDE.md rule 6);
* the **public API surface** -- everything the API reference documents
  (signatures, pydantic fields, enum members, public methods);
* the **command line** and the manifest **schema versions**.

A failure here is not a bug report, it is a question: *is this change allowed by
the stability policy* (``mkdocs/stability.md``)? Adding something is fine but must
be deliberate (regenerate the snapshot); changing or removing something documented
needs a deprecation cycle first. To accept a deliberate, policy-compliant change::

    python tests/api/surface.py --update

and say why in the commit message and the CHANGELOG.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any

from . import surface

if TYPE_CHECKING:
    from pathlib import Path

_HOW_TO_FIX = (
    "\n\nIf this change is deliberate and allowed by mkdocs/stability.md, regenerate "
    "the snapshot with `python tests/api/surface.py --update` and record the change "
    "in CHANGELOG.md. A removed or renamed public name/key needs a deprecation "
    "cycle first (keep the old one working, warn, remove in a later minor)."
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _diff(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in sorted(old.keys() - new.keys()):
        lines.append(f"REMOVED  {key}")
    for key in sorted(new.keys() - old.keys()):
        lines.append(f"ADDED    {key}")
    for key in sorted(old.keys() & new.keys()):
        if old[key] != new[key]:
            lines.append(f"CHANGED  {key}")
            for sub in sorted(set(old[key]) | set(new[key])):
                before, after = old[key].get(sub), new[key].get(sub)
                if before != after:
                    lines.append(f"           .{sub}: {before!r} -> {after!r}")
    return lines


def test_registry_keys_are_frozen() -> None:
    """Run in a fresh interpreter: a long-lived test process's registries also
    hold every test double another module registered."""
    proc = subprocess.run(
        [sys.executable, surface.__file__, "--registry"],
        capture_output=True,
        text=True,
        check=True,
    )
    current = json.loads(proc.stdout)
    frozen = _load(surface.REGISTRY_SNAPSHOT)
    problems: list[str] = []
    for category in sorted(frozen.keys() | current.keys()):
        old, new = set(frozen.get(category, [])), set(current.get(category, []))
        problems += [f"REMOVED  {k}" for k in sorted(old - new)]
        problems += [f"ADDED    {k}" for k in sorted(new - old)]
    assert not problems, "Registry keys changed:\n" + "\n".join(problems) + _HOW_TO_FIX


def test_the_public_api_surface_is_frozen() -> None:
    frozen = _load(surface.SNAPSHOT)
    current = surface.collect_all()
    # Documented objects that need an optional dependency (the LangChain adapter)
    # are absent when it is not installed; that is not a removal.
    optional = "openbtk.integrations.langchain"
    expected = {
        k: v
        for k, v in frozen["api"].items()
        if k in current["api"] or not k.startswith(optional)
    }
    problems = _diff(expected, current["api"])
    for section in ("top_level", "cli", "schema_versions"):
        if frozen[section] != current[section]:
            problems.append(
                f"CHANGED  {section}: {frozen[section]!r} -> {current[section]!r}"
            )
    assert not problems, "Public API changed:\n" + "\n".join(problems) + _HOW_TO_FIX


def test_the_snapshot_covers_a_meaningful_surface() -> None:
    """Guards against the freeze quietly becoming vacuous."""
    frozen = _load(surface.SNAPSHOT)
    assert len(frozen["api"]) >= 100
    assert {"list", "validate", "run", "deid", "replay", "doctor"} <= set(frozen["cli"])
    kinds = {v["kind"] for v in frozen["api"].values()}
    assert kinds >= {"class", "function"}


def test_manifest_schema_versions_are_semver_strings() -> None:
    """ADR-0005: the manifest schema is versioned independently of the package
    and additive-only within a major."""
    for name, version in _load(surface.SNAPSHOT)["schema_versions"].items():
        major, minor, patch = version.split(".")
        assert (major.isdigit(), minor.isdigit(), patch.isdigit()) == (True,) * 3, name

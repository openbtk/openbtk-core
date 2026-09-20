"""The GitHub Actions workflows are code that only runs after you push, so a mistake in
one is found by a red X on a pull request. These checks catch the two mistakes that have
actually happened: a workflow that is not valid YAML, and an action referenced by a
mutable tag instead of a commit (security review finding S-6)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

_WORKFLOWS = sorted(
    (Path(__file__).resolve().parents[2] / ".github/workflows").glob("*.yml")
)
_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_there_are_workflows() -> None:
    assert {p.name for p in _WORKFLOWS} >= {"ci.yml", "release.yml", "docs.yml"}


@pytest.mark.parametrize("path", _WORKFLOWS, ids=lambda p: p.name)
def test_each_workflow_is_valid_yaml_with_jobs(path: Path) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict) and document.get("jobs"), path.name


@pytest.mark.parametrize("path", _WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_commit(path: Path) -> None:
    unpinned = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(?:-\s*)?uses:\s*(\S+)", line)
        if not match:
            continue
        reference = match.group(1)
        if reference.startswith("./"):
            continue  # a local action
        _, _, version = reference.partition("@")
        if not _SHA.match(version):
            unpinned.append(reference)
    assert not unpinned, f"{path.name}: pin these to a commit SHA: {unpinned}"


def test_the_release_workflow_still_gates_on_manual_approval() -> None:
    """PyPI versions are permanent: the publish job must stay behind the protected
    ``pypi`` environment and must not run unless verification passed."""
    release = yaml.safe_load(
        (_WORKFLOWS[0].parent / "release.yml").read_text(encoding="utf-8")
    )
    publish = release["jobs"]["pypi"]
    assert publish["environment"]["name"] == "pypi"
    assert publish["needs"] == "verify"

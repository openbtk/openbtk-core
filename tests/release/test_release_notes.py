"""The v0.5.0 release must state the n2c2 gap up front -- in the CHANGELOG, in
the GitHub release notes, and at the top of the README that PyPI renders --
before any of the good news. If the wording is ever softened away, this fails.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _read(*parts: str) -> str:
    return _ROOT.joinpath(*parts).read_text(encoding="utf-8")


def test_the_changelog_release_opens_with_the_gap() -> None:
    text = _read("CHANGELOG.md")
    release = text[text.index("## [0.5.0]") :]
    first_section = release[: release.index("### What is new since 0.1.1")]
    assert "has not been run" in first_section
    assert "synthetic" in first_section
    assert "does not detect names" in first_section.replace("**", "")


def test_the_github_release_notes_open_with_the_gap() -> None:
    notes = _read(".github", "release-notes", "v0.5.0.md")
    head = notes[: notes.index("## What is new")]
    assert "has not been run" in head
    assert "synthetic" in head


def test_the_readme_states_the_gap_before_the_feature_list() -> None:
    readme = _read("README.md")
    assert readme.index("has not been run") < readme.index("## What's built")


def test_the_release_workflow_publishes_those_notes() -> None:
    workflow = _read(".github", "workflows", "release.yml")
    assert "body_path" in workflow
    assert ".github/release-notes/" in workflow

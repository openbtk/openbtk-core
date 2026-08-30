"""Installation and import integrity.

These tests exist because the v1 codebase was never importable: three package
names coexisted (the directory ``src/``, imports of ``openbtk.*``, and
``opentbtk.*`` -- a typo appearing 79 times), and ``pyproject.toml`` declared a
wheel package that was absent from disk. Nothing caught it because the only
verification ever run was ``py_compile``, which resolves no imports.

Every test here would have failed on the v1 tree. See ADR-0003.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every package that must import with ZERO optional dependencies installed.
PUBLIC_MODULES = [
    "openbtk",
    "openbtk.core",
    "openbtk.deid",
    "openbtk.deid.recognizers",
    "openbtk.data",
    "openbtk.data.clinical_text",
    "openbtk.data.ehr",
    "openbtk.terminology",
    "openbtk.embeddings",
    "openbtk.llms",
    "openbtk.retrieval",
    "openbtk.guardrails",
    "openbtk.pipelines",
    "openbtk.eval",
    "openbtk.integrations",
    "openbtk.cli",
]


def test_package_imports() -> None:
    """The package imports at all. This is not a trivial assertion here."""
    import openbtk

    assert openbtk is not None


def test_version_is_a_string() -> None:
    import openbtk

    assert isinstance(openbtk.__version__, str)
    assert openbtk.__version__


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_public_module_imports_with_zero_extras(module_name: str) -> None:
    """Importing any public module must not require an optional dependency.

    A heavy import at module scope (``import torch``, ``import spacy``) breaks
    the zero-extras install. Heavy imports belong inside methods, behind
    ``openbtk.core._lazy.require()``.
    """
    module = importlib.import_module(module_name)
    assert module.__doc__, f"{module_name} is missing a module docstring"


def test_package_is_installed_not_path_relative() -> None:
    """The imported package comes from an installation, not the CWD.

    Run from outside the repository (as CI does), a path-relative import cannot
    satisfy this -- which is the entire point of the src/ layout.
    """
    import openbtk

    location = Path(openbtk.__file__).resolve()
    assert location.name == "__init__.py"
    assert location.parent.name == "openbtk"


def test_imports_from_a_different_working_directory(tmp_path: Path) -> None:
    """Import succeeds with the CWD outside the repository."""
    result = subprocess.run(
        [sys.executable, "-c", "import openbtk; print(openbtk.__version__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_py_typed_marker_is_present() -> None:
    """PEP 561 marker, so downstream mypy sees our annotations."""
    import openbtk

    assert (Path(openbtk.__file__).parent / "py.typed").is_file()


def test_no_forbidden_package_names() -> None:
    """No ``opentbtk`` typo, no ``src.`` imports. The v1 killer.

    A pre-commit hook and a CI step enforce this too; this test makes it fail
    in the suite as well, so it cannot be bypassed with --no-verify.
    """
    self_path = Path(__file__).resolve()
    offenders: list[str] = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(
            part in {".git", ".venv", "build", "dist", "__pycache__"}
            for part in path.parts
        ):
            continue
        if path.resolve() == self_path:
            continue  # this file names the forbidden strings in order to detect them
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if "opentbtk" in line or "from src." in line or "import src." in line:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
                )
    assert not offenders, "Forbidden package names found:\n" + "\n".join(offenders)


def test_core_dependency_budget() -> None:
    """Core declares at most six dependencies. A seventh requires an ADR.

    Guards the promise that reading a clinical note does not pull in an ML
    framework. See docs/09_CODING_STANDARDS.md section 2 and ADR-0001.
    """
    import tomllib

    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    deps = pyproject["project"]["dependencies"]
    assert len(deps) <= 6, f"Core dependency budget exceeded: {deps}"

    banned = {
        "langchain",
        "langchain-core",
        "langgraph",
        "torch",
        "transformers",
        "spacy",
    }
    names = {d.split(">")[0].split("=")[0].split("[")[0].strip().lower() for d in deps}
    assert not (names & banned), f"Heavy dependency in core: {names & banned}"

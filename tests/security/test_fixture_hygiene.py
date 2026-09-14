"""Release blocker (docs/06_SECURITY_COMPLIANCE.md section 3.2, T2): scan
committed test source for realistic-looking identifiers.

There is no tests/fixtures/ directory yet -- no modality has been built past
M1, so no real data fixture (Synthea, Faker-generated notes, etc.) exists to
scan. What DOES exist today, and is exactly as real a T2 risk, is test
*source* containing a hand-typed realistic-looking identifier. This scanner
covers that surface now and is the natural place to extend to
tests/fixtures/ (glob widened to data files, not just .py) once M3+ adds one.

Suppression convention: a line that legitimately needs a realistic-looking
identifier for adversarial testing (e.g. proving an error path does NOT leak
one) is marked with a trailing ``# phi-fixture-ok: <reason>`` comment, the
same shape as ruff's own ``# noqa`` -- deliberately a distinct token so it is
never confused with a lint suppression. The whole of tests/security/ is
exempt outright: every file in it exists specifically to hold realistic-
looking PHI-shaped strings for adversarial testing, and that directory's own
tests are what verify none of it leaks anywhere real.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TESTS_DIR = _REPO_ROOT / "tests"
_EXEMPT_DIRS = {_TESTS_DIR / "security"}
_SUPPRESSION_MARKER = "phi-fixture-ok"

# Deliberately narrower than a general PII scanner: docs/06_SECURITY_COMPLIANCE.md
# section 3.2 names three concrete shapes ("valid SSN checksums, real-looking
# MRNs, plausible DOB clusters"). SSN and MRN shapes are checked here; DOB
# "clustering" requires a corpus of many records to detect a cluster against,
# which does not exist at M1 (no real fixture corpus yet) -- tracked as a
# known gap, not silently skipped.
_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn_shaped": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "mrn_shaped": re.compile(r"\bMRN-?\d{4,}\b", re.IGNORECASE),
    "email_shaped": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}\b"),
}


def _iter_scannable_files() -> list[Path]:
    files = []
    for path in _TESTS_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if any(exempt in path.parents for exempt in _EXEMPT_DIRS):
            continue
        files.append(path)
    return files


def _find_hits(path: Path) -> list[str]:
    hits = []
    try:
        display_path = path.relative_to(_REPO_ROOT)
    except ValueError:
        # Only reachable from this file's own meta-tests, which deliberately
        # scan a pytest tmp_path outside the repository.
        display_path = path
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if _SUPPRESSION_MARKER in line:
            continue
        for label, pattern in _PATTERNS.items():
            if pattern.search(line):
                hits.append(f"{display_path}:{lineno}: {label}: {line.strip()}")
    return hits


def test_no_unsuppressed_realistic_identifiers_in_test_source() -> None:
    all_hits: list[str] = []
    for path in _iter_scannable_files():
        all_hits.extend(_find_hits(path))
    assert not all_hits, (
        "Realistic-looking identifier(s) found in test source with no "
        f"'# {_SUPPRESSION_MARKER}: <reason>' suppression:\n" + "\n".join(all_hits)
    )


def test_scanner_actually_catches_a_real_violation(tmp_path: Path) -> None:
    """Meta-test: proves the scan can fail, so a passing suite above means
    something -- not that the patterns silently never match anything."""
    planted = tmp_path / "test_planted_violation.py"
    planted.write_text('SSN = "123-45-6789"\n')
    hits = _find_hits(planted)
    assert hits, "scanner failed to catch a deliberately planted SSN-shaped string"


def test_suppression_marker_actually_suppresses(tmp_path: Path) -> None:
    """The other half of the meta-test: a properly marked line must NOT be
    flagged, so the mechanism doesn't just accidentally match everything or
    force every real hit into the exempt list."""
    suppressed = tmp_path / "test_suppressed.py"
    line = f'SSN = "123-45-6789"  # {_SUPPRESSION_MARKER}: adversarial\n'
    suppressed.write_text(line)
    assert _find_hits(suppressed) == []

"""Memory benchmark (docs/07_TEST_CHARTER.md section 3.7, NFR-01,
docs/02_PRD.md section 5.1): "Stream 10M notes in < 4 GB RSS." Nightly
only -- ``@pytest.mark.benchmark``, skipped by default (see
tests/conftest.py's ``pytest_collection_modifyitems``).

Real end to end: writes ``note_count`` synthetic notes to a single JSON
Lines file (never ``note_count`` separate files -- that alone would make
this a filesystem-metadata benchmark, not a memory one), then runs the
real four-stage pipeline (``loader.clinical_text.jsonl`` ->
``preprocessor.general.deidentify`` -> ``preprocessor.clinical_text.section_segment``
-> ``chunker.clinical_text.section_aware``) through the real
``openbtk.pipelines`` executor, sampling this PROCESS's own peak resident
set size across the run.

``OPENBTK_BENCHMARK_NOTE_COUNT`` overrides the note count for fast local
iteration on the benchmark's own mechanics (generation, streaming,
measurement, cleanup) -- the checked-in default is the real, literal NFR-01
target, not a diluted one; the override exists so a developer is not
forced to wait for a 10M-note run just to find a typo in this file.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
from typing import TYPE_CHECKING

import pytest

from openbtk.data import clinical_text as _clinical_text  # noqa: F401
from openbtk.pipelines import Pipeline, Step

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.benchmark

_DEFAULT_NOTE_COUNT = 10_000_000
_RSS_TARGET_BYTES = 4 * 1024**3  # 4 GB, NFR-01's literal number


def _note_count() -> int:
    override = os.environ.get("OPENBTK_BENCHMARK_NOTE_COUNT")
    return int(override) if override else _DEFAULT_NOTE_COUNT


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    )


def _peak_rss_bytes() -> int:
    """This process's peak resident-set size, in bytes.

    Deliberately stdlib/ctypes-only, not a new ``psutil`` dependency, for
    one platform-conditional measurement used by exactly one test:

      * Linux/macOS: ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` -- true
        peak RSS. Linux reports KB, macOS reports bytes (both platforms'
        own documented, long-standing behaviour).
      * Windows: ``GetProcessMemoryInfo``'s ``PeakWorkingSetSize`` -- the
        closest Windows analogue to Unix peak RSS.

    Written as three separate ``sys.platform == "literal"`` branches
    (never ``platform.system()``, and never combined with ``in``) because
    mypy specifically special-cases exactly that shape: it type-checks
    only the branch matching its configured/host platform and skips the
    others entirely, so a Linux stdlib call and a Windows-only
    ``ctypes.windll`` call can coexist here with no ``type: ignore``
    anywhere -- one would always be "unused" on whichever single platform
    mypy actually runs on in CI.

    Raises:
        RuntimeError: On a platform none of the three branches cover.
    """
    if sys.platform == "win32":
        # GetCurrentProcess's real return type is a pointer-sized pseudo
        # handle. Left undeclared, ctypes assumes the default restype
        # (c_int, 32-bit) and TRUNCATES it on 64-bit Windows -- confirmed by
        # direct reproduction: GetProcessMemoryInfo then fails outright
        # because it receives a corrupted handle, not because the API
        # itself is misused. Declaring restype/argtypes explicitly is the
        # actual fix, not a defensive nicety.
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            raise RuntimeError("GetProcessMemoryInfo failed.")
        return int(counters.PeakWorkingSetSize)
    # The elif here (not "if", which ruff's RET505 would otherwise suggest)
    # is load-bearing: keeping every branch part of one if/elif/.../else
    # chain is what makes mypy's platform-narrowing skip each non-matching
    # branch entirely, rather than flagging one as dead code after
    # another's return -- verified directly against --platform
    # win32/linux/darwin, not assumed.
    elif sys.platform == "linux":  # noqa: RET505
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    elif sys.platform == "darwin":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    else:
        raise RuntimeError(
            f"Peak RSS measurement not implemented for platform {sys.platform!r}."
        )


def _write_synthetic_notes_jsonl(path: Path, count: int) -> None:
    """Fast, dependency-free synthetic note generation -- deliberately not
    ``fixtures.labelled_phi_corpus`` (Faker-per-document is far too slow at
    this scale, and this benchmark is about VOLUME, not PHI accuracy).
    Real section headers so the real ``SectionSegmenter``/
    ``SectionAwareChunker`` exercise their normal path, not the
    no-sections degenerate case, at scale."""
    with path.open("w", encoding="utf-8") as f:
        for i in range(count):
            record = {
                "record_id": f"n{i}",
                "text": (
                    "Chief Complaint:\n"
                    f"Patient {i} presents with routine symptoms.\n"
                    "Plan:\n"
                    "Follow up as needed.\n"
                ),
            }
            f.write(json.dumps(record))
            f.write("\n")


def test_streams_notes_under_the_rss_target(tmp_path: Path) -> None:
    count = _note_count()
    notes_path = tmp_path / "notes.jsonl"
    _write_synthetic_notes_jsonl(notes_path, count)

    pipeline = (
        Pipeline("memory-benchmark")
        .add(Step("load", "loader.clinical_text.jsonl", path=str(notes_path)))
        .add(Step("deid", "preprocessor.general.deidentify", mode="redact"))
        .add(Step("segment", "preprocessor.clinical_text.section_segment"))
        .add(Step("chunk", "chunker.clinical_text.section_aware", max_tokens=100))
    )
    manifest = pipeline.run()

    peak_rss = _peak_rss_bytes()
    print(  # noqa: T201 -- benchmark result, not library code; nightly-only, human-read output
        f"\n[memory benchmark] {count:,} notes, peak RSS = "
        f"{peak_rss / 1024**3:.3f} GB (target < {_RSS_TARGET_BYTES / 1024**3:.0f} GB), "
        f"python={sys.version_info.major}.{sys.version_info.minor}",
        file=sys.stderr,
    )

    assert manifest.status == "success", manifest.error
    by_id = {s.step_id: s for s in manifest.steps}
    assert by_id["load"].records_out == count
    assert peak_rss < _RSS_TARGET_BYTES

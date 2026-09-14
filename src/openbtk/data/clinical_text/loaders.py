"""Clinical text loaders (docs/05_DATA_MODALITY_SPEC.md section 1.2).

All three P0 loaders are streaming (ADR-0004): each yields one
``ClinicalTextRecord`` at a time, never materialising the whole source.
``CDALoader`` (lxml, P1) is not built -- not on this milestone's exit
criteria.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openbtk.core._lazy import require
from openbtk.core.base import BaseLoader
from openbtk.core.errors import LoaderError
from openbtk.core.logging import get_logger
from openbtk.core.registry import LOADER_REGISTRY
from openbtk.data.clinical_text.schemas import ClinicalTextRecord

if TYPE_CHECKING:
    from collections.abc import Iterator

log = get_logger(__name__)


@LOADER_REGISTRY.register("loader.clinical_text.plain_text")
class PlainTextLoader(BaseLoader[str, ClinicalTextRecord]):
    """Load every ``*.txt`` file in a directory as one record each.

    Example:
        >>> import contextlib
        >>> import io
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "note1.txt").write_text("Chief Complaint: pain.")
        ...     loader = PlainTextLoader()
        ...     with contextlib.redirect_stdout(io.StringIO()):  # logging noise
        ...         records = list(loader.load(d))
        >>> records[0].record_id
        'note1'
        >>> records[0].source
        'plain_text'
    """

    def __init__(
        self, *, note_type: str | None = None, encoding: str = "utf-8"
    ) -> None:
        self._note_type = note_type
        self._encoding = encoding

    def load(self, source: str) -> Iterator[ClinicalTextRecord]:
        """Yield one record per ``*.txt`` file in the directory ``source``,
        in sorted filename order (deterministic, not filesystem-order-dependent).

        Raises:
            LoaderError: If ``source`` is not a directory, or a file cannot
                be read or decoded.
        """
        directory = Path(source)
        if not directory.is_dir():
            raise LoaderError(
                f"Not a directory: {source}",
                context={"modality": "clinical_text", "stage": "load"},
            )
        log.info("loader.start", modality="clinical_text", source=str(directory))
        for path in sorted(directory.glob("*.txt")):
            try:
                text = path.read_text(encoding=self._encoding)
            except (OSError, UnicodeDecodeError) as e:
                raise LoaderError(
                    f"Failed to read {path.name}.",
                    context={
                        "modality": "clinical_text",
                        "stage": "load",
                        "filename": path.name,
                    },
                ) from e
            yield ClinicalTextRecord(
                record_id=path.stem,
                source="plain_text",
                text=text,
                note_type=self._note_type,
            )


@LOADER_REGISTRY.register("loader.clinical_text.jsonl")
class JSONLLoader(BaseLoader[str, ClinicalTextRecord]):
    """Load records from a JSON Lines file, one ``ClinicalTextRecord`` per
    line.

    Each line must be a JSON object with at least ``record_id`` and
    ``text``; any other ``ClinicalTextRecord`` field may be present. A
    line with no ``source`` key defaults to ``"jsonl"``.

    Example:
        >>> import contextlib
        >>> import io
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.NamedTemporaryFile(
        ...     mode="w", suffix=".jsonl", delete=False
        ... ) as f:
        ...     _ = f.write('{"record_id": "n1", "text": "chest pain"}\\n')
        ...     path = f.name
        >>> with contextlib.redirect_stdout(io.StringIO()):  # logging noise
        ...     records = list(JSONLLoader().load(path))
        >>> records[0].text
        'chest pain'
        >>> Path(path).unlink()
    """

    def load(self, source: str) -> Iterator[ClinicalTextRecord]:
        """Raises:
        LoaderError: If ``source`` cannot be opened, or a line is not
            valid JSON or does not satisfy ``ClinicalTextRecord``.
        """
        path = Path(source)
        log.info("loader.start", modality="clinical_text", source=str(path))
        try:
            handle = path.open(encoding="utf-8")
        except OSError as e:
            raise LoaderError(
                f"Could not open {source}.",
                context={"modality": "clinical_text", "stage": "load"},
            ) from e
        try:
            for lineno, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw: dict[str, Any] = json.loads(stripped)
                except json.JSONDecodeError as e:
                    raise LoaderError(
                        f"Invalid JSON on line {lineno} of {path.name}.",
                        context={
                            "modality": "clinical_text",
                            "stage": "load",
                            "filename": path.name,
                            "line": lineno,
                        },
                    ) from e
                raw.setdefault("source", "jsonl")
                try:
                    yield ClinicalTextRecord.model_validate(raw)
                except Exception as e:
                    raise LoaderError(
                        f"Line {lineno} of {path.name} is not a valid "
                        "ClinicalTextRecord.",
                        context={
                            "modality": "clinical_text",
                            "stage": "load",
                            "filename": path.name,
                            "line": lineno,
                        },
                    ) from e
        finally:
            handle.close()


@LOADER_REGISTRY.register("loader.clinical_text.mimic_notes")
class MIMICNotesLoader(BaseLoader[str, ClinicalTextRecord]):
    """Load a MIMIC-III/IV ``NOTEEVENTS``-shaped CSV, chunked via pandas so
    peak memory stays ``O(chunk_size)``, not ``O(corpus)`` (ADR-0004).

    Expects (a subset of) MIMIC's real column names: ``ROW_ID``, ``TEXT``,
    and optionally ``SUBJECT_ID``, ``HADM_ID``, ``CATEGORY``, ``CHARTDATE``.
    MIMIC-IV is itself a de-identified, IRB-released research dataset --
    ``SUBJECT_ID`` is already a study-assigned pseudonymous identifier, not
    real PHI, so it is passed through as ``patient_ref`` unchanged.
    """

    def __init__(self, *, chunk_size: int = 1000) -> None:
        self._chunk_size = chunk_size

    def load(self, source: str) -> Iterator[ClinicalTextRecord]:
        """Raises:
        LoaderError: If ``source`` cannot be read, or lacks the required
            ``ROW_ID``/``TEXT`` columns.
        """
        pandas = require("pandas", extra="text")
        log.info("loader.start", modality="clinical_text", source=source)
        try:
            reader = pandas.read_csv(source, chunksize=self._chunk_size)
        except FileNotFoundError as e:
            raise LoaderError(
                f"Could not read {source}.",
                context={"modality": "clinical_text", "stage": "load"},
            ) from e
        # pandas' chunked TextFileReader holds an open file handle;
        # generator scope alone does not guarantee it closes (a caller who
        # never exhausts load() would otherwise leak it until GC, which
        # surfaces as a real "Exception ignored" warning -- caught by
        # actually running these tests, not assumed away).
        try:
            for chunk in reader:
                yield from self._records_from_chunk(chunk, source)
        finally:
            reader.close()

    def _records_from_chunk(
        self, chunk: Any, source: str
    ) -> Iterator[ClinicalTextRecord]:
        missing = {"ROW_ID", "TEXT"} - set(chunk.columns)
        if missing:
            raise LoaderError(
                f"{source} is missing required column(s): {sorted(missing)}.",
                context={"modality": "clinical_text", "stage": "load"},
            )
        for row in chunk.itertuples(index=False):
            row_dict = row._asdict()
            yield ClinicalTextRecord(
                record_id=str(row_dict["ROW_ID"]),
                source="mimic-iv-note",
                text=str(row_dict["TEXT"]),
                note_type=self._optional_str(row_dict, "CATEGORY"),
                patient_ref=self._optional_str(row_dict, "SUBJECT_ID"),
                encounter_ref=self._optional_str(row_dict, "HADM_ID"),
            )

    @staticmethod
    def _optional_str(row: dict[str, Any], key: str) -> str | None:
        value = row.get(key)
        if value is None:
            return None
        # pandas represents a missing cell as float('nan'), not None.
        if isinstance(value, float) and math.isnan(value):
            return None
        return str(value)

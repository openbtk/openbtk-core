"""
Clinical text loaders — plain text, MIMIC-IV notes, HL7 CDA.

All loaders wrap existing I/O libraries; no custom parsing is implemented
for formats that have adequate library support.
"""
from __future__ import annotations

import csv
import uuid
from pathlib import Path
from typing import Iterable

import structlog

from opentbtk.core.base import BaseLoader
from opentbtk.core.errors import LoaderError
from opentbtk.core.registry import LOADER_REGISTRY
from .schemas import ClinicalTextRecord

log = structlog.get_logger(__name__)


@LOADER_REGISTRY.register("loader.clinical_text.plain_text")
class PlainTextLoader(BaseLoader[str, ClinicalTextRecord]):
    """Load clinical text records from plain .txt files or directories.

    Wraps standard Python file I/O. For a directory, loads all .txt files.

    Args:
        note_type: Optional note type label applied to all loaded records.
        encoding: File encoding (default: utf-8).
    """

    def __init__(
        self,
        note_type: str | None = None,
        encoding: str = "utf-8",
    ) -> None:
        self._note_type = note_type
        self._encoding = encoding

    def load(self, source: str) -> Iterable[ClinicalTextRecord]:
        p = Path(source)
        if not p.exists():
            raise LoaderError(
                f"Source not found: {source}",
                context={"modality": "clinical_text", "stage": "load", "source": source},
            )
        paths = list(p.glob("*.txt")) if p.is_dir() else [p]
        log.info("loader.start", modality="clinical_text", loader="plain_text", n_files=len(paths))
        for path in paths:
            try:
                text = path.read_text(encoding=self._encoding)
            except OSError as e:
                raise LoaderError(
                    f"Cannot read file: {path}",
                    context={"source": str(path)},
                ) from e
            yield ClinicalTextRecord(
                record_id=str(uuid.uuid4()),
                source="plain_text",
                note_type=self._note_type,
                raw_text=text,
                metadata={"file_name": path.name},
            )
        log.info("loader.complete", modality="clinical_text", loader="plain_text")


@LOADER_REGISTRY.register("loader.clinical_text.mimic_notes")
class MIMICNotesLoader(BaseLoader[str, ClinicalTextRecord]):
    """Load clinical notes from MIMIC-III/IV NOTEEVENTS CSV format.

    Wraps Python csv stdlib. Expects columns: ROW_ID, SUBJECT_ID, HADM_ID,
    CHARTDATE, CATEGORY, TEXT (MIMIC-III) or equivalent MIMIC-IV columns.
    Subject and admission IDs are hashed — never stored raw.

    Args:
        id_column: Column name for note row ID (default: 'note_id').
        text_column: Column name for note text (default: 'text').
        category_column: Column name for note category (default: 'note_type').
        hash_salt: Salt for hashing patient/admission IDs. Use a fixed
            project-level salt for reproducible pseudonymous IDs.
    """

    _MIMIC3_DEFAULTS = {
        "id_column": "ROW_ID",
        "text_column": "TEXT",
        "category_column": "CATEGORY",
    }

    def __init__(
        self,
        id_column: str = "note_id",
        text_column: str = "text",
        category_column: str = "note_type",
        hash_salt: str = "opentbtk-default",
    ) -> None:
        self._id_col = id_column
        self._text_col = text_column
        self._cat_col = category_column
        self._hash_salt = hash_salt

    def load(self, source: str) -> Iterable[ClinicalTextRecord]:
        import hashlib

        path = Path(source)
        if not path.exists():
            raise LoaderError(
                f"MIMIC notes CSV not found: {source}",
                context={"modality": "clinical_text", "loader": "mimic_notes", "source": source},
            )
        log.info("loader.start", modality="clinical_text", loader="mimic_notes", path=source)
        try:
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    text = row.get(self._text_col, "").strip()
                    if not text:
                        continue
                    raw_id = row.get(self._id_col, str(i))
                    hashed_id = hashlib.sha256(
                        f"{self._hash_salt}:{raw_id}".encode()
                    ).hexdigest()[:16]
                    yield ClinicalTextRecord(
                        record_id=hashed_id,
                        source="mimic",
                        note_type=row.get(self._cat_col),
                        raw_text=text,
                        metadata={
                            "chart_date": row.get("CHARTDATE") or row.get("chartdate"),
                        },
                    )
        except OSError as e:
            raise LoaderError(
                f"Failed to read MIMIC notes: {source}",
                context={"source": source},
            ) from e
        log.info("loader.complete", modality="clinical_text", loader="mimic_notes")


@LOADER_REGISTRY.register("loader.clinical_text.cda")
class CDAClinicalLoader(BaseLoader[str, ClinicalTextRecord]):
    """Load clinical documents from HL7 CDA/CCD XML files.

    Wraps Python's stdlib xml.etree for basic CDA parsing. Extracts
    narrative text blocks from ClinicalDocument/component sections.
    For production use, consider the `hl7apy` library for richer parsing.

    Args:
        extract_sections: If True, attempt to extract section titles alongside
            text. Requires section/title elements to be present in the CDA.
    """

    def __init__(self, extract_sections: bool = True) -> None:
        self._extract_sections = extract_sections

    def load(self, source: str) -> Iterable[ClinicalTextRecord]:
        import xml.etree.ElementTree as ET  # stdlib — safe for CDA, no net access

        path = Path(source)
        paths = list(path.glob("*.xml")) if path.is_dir() else [path]
        log.info("loader.start", modality="clinical_text", loader="cda", n_files=len(paths))

        _CDA_NS = "urn:hl7-org:v3"

        for p in paths:
            try:
                tree = ET.parse(str(p))
                root = tree.getroot()
            except ET.ParseError as e:
                raise LoaderError(
                    f"Invalid CDA XML: {p}",
                    context={"source": str(p)},
                ) from e

            sections: dict[str, str] = {}
            full_parts: list[str] = []

            for section in root.iter(f"{{{_CDA_NS}}}section"):
                title_el = section.find(f"{{{_CDA_NS}}}title")
                title = title_el.text.strip() if title_el is not None and title_el.text else "Unknown"
                text_el = section.find(f"{{{_CDA_NS}}}text")
                if text_el is not None:
                    text = "".join(text_el.itertext()).strip()
                    if text:
                        sections[title] = text
                        full_parts.append(f"[{title}]\n{text}")

            full_text = "\n\n".join(full_parts)
            if not full_text:
                log.warning("loader.cda_empty", file=str(p))
                continue

            yield ClinicalTextRecord(
                record_id=str(uuid.uuid4()),
                source="cda",
                note_type="CDA Document",
                raw_text=full_text,
                sections=sections if self._extract_sections else None,
                metadata={"file_name": p.name},
            )
        log.info("loader.complete", modality="clinical_text", loader="cda")

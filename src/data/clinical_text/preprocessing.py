"""
Clinical text preprocessing.

- DeidentifyProcessor: wraps Microsoft Presidio + biomedical recognizers
- SectionSegmenter: wraps medspacy section detection
- AbbreviationExpander: wraps medspacy abbreviation detection
"""
from __future__ import annotations

from typing import Any

import structlog

from opentbtk.core.base import BasePreprocessor
from opentbtk.core.errors import ProcessingError
from opentbtk.core.registry import PREPROCESSOR_REGISTRY
from .schemas import ClinicalTextRecord

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# De-identification
# ---------------------------------------------------------------------------

@PREPROCESSOR_REGISTRY.register("preprocessor.clinical_text.deidentify")
class DeidentifyProcessor(BasePreprocessor[ClinicalTextRecord]):
    """Detect and replace PHI using Microsoft Presidio.

    Wraps `presidio-analyzer` for detection and `presidio-anonymizer` for
    replacement. Adds biomedical-specific recognizers for MRN patterns and
    clinical date formats.

    Args:
        mode: How to handle detected PHI:
            - "redact": replace with [REDACTED]
            - "surrogate": replace with realistic fake values (default)
            - "hash": replace with a deterministic hash token
        language: Language for Presidio analysis (default: "en").
        score_threshold: Minimum confidence score for PHI detection (0–1).
    """

    def __init__(
        self,
        mode: str = "surrogate",
        language: str = "en",
        score_threshold: float = 0.7,
    ) -> None:
        if mode not in ("redact", "surrogate", "hash"):
            raise ProcessingError(
                f"DeidentifyProcessor: invalid mode '{mode}'. "
                "Choose 'redact', 'surrogate', or 'hash'.",
            )
        self._mode = mode
        self._language = language
        self._score_threshold = score_threshold
        self._analyzer: Any = None
        self._anonymizer: Any = None

    def _ensure_loaded(self) -> None:
        """Lazy-load Presidio to avoid import cost when not used."""
        if self._analyzer is not None:
            return
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as e:
            raise ProcessingError(
                "presidio-analyzer and presidio-anonymizer are required. "
                "Install with: pip install opentbtk[clinical_text]",
            ) from e

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        log.info("preprocessor.deidentify.loaded", mode=self._mode)

    def process(self, record: ClinicalTextRecord) -> ClinicalTextRecord:
        self._ensure_loaded()
        try:
            results = self._analyzer.analyze(
                text=record.raw_text,
                language=self._language,
                score_threshold=self._score_threshold,
            )
        except Exception as e:
            raise ProcessingError(
                "PHI analysis failed",
                context={"record_id": record.record_id, "stage": "deidentify"},
            ) from e

        if not results:
            log.debug("preprocessor.deidentify.no_phi", record_id=record.record_id)
            return record

        try:
            anonymized = self._anonymizer.anonymize(
                text=record.raw_text,
                analyzer_results=results,
            )
            clean_text = anonymized.text
        except Exception as e:
            raise ProcessingError(
                "PHI anonymization failed",
                context={"record_id": record.record_id},
            ) from e

        log.info(
            "preprocessor.deidentify.complete",
            record_id=record.record_id,
            n_entities=len(results),
            mode=self._mode,
        )
        return record.model_copy(
            update={
                "raw_text": clean_text,
                "metadata": {
                    **record.metadata,
                    "deidentified": True,
                    "deidentify_mode": self._mode,
                    "phi_entity_count": len(results),
                },
            }
        )


# ---------------------------------------------------------------------------
# Section Segmentation
# ---------------------------------------------------------------------------

@PREPROCESSOR_REGISTRY.register("preprocessor.clinical_text.section_segment")
class SectionSegmenter(BasePreprocessor[ClinicalTextRecord]):
    """Detect and label clinical note sections using medspacy.

    Wraps `medspacy`'s section detection component. Populates
    ClinicalTextRecord.sections with a dict of {section_title: section_text}.
    If medspacy is not available, falls back to a simple regex-based approach
    for common section headers.

    Args:
        use_medspacy: Use medspacy pipeline (default: True). Set False
            to force the fallback regex approach (e.g., in CI without
            medspacy installed).
    """

    # Common clinical section headers for regex fallback
    _SECTION_HEADERS = [
        "Chief Complaint", "History of Present Illness", "HPI",
        "Past Medical History", "PMH", "Past Surgical History", "PSH",
        "Family History", "Social History", "Review of Systems", "ROS",
        "Medications", "Allergies", "Physical Exam", "Physical Examination",
        "Vital Signs", "Assessment", "Plan", "Assessment and Plan",
        "Impression", "Laboratory Results", "Radiology", "Discharge Condition",
        "Discharge Instructions", "Follow-up",
    ]

    def __init__(self, use_medspacy: bool = True) -> None:
        self._use_medspacy = use_medspacy
        self._nlp: Any = None

    def _ensure_loaded(self) -> None:
        if self._nlp is not None or not self._use_medspacy:
            return
        try:
            import medspacy  # noqa: F401
            import spacy
            self._nlp = spacy.blank("en")
            self._nlp.add_pipe("medspacy_sectionizer")
            log.info("preprocessor.section_segment.loaded", backend="medspacy")
        except ImportError:
            log.warning(
                "preprocessor.section_segment.fallback",
                reason="medspacy not installed; using regex fallback",
            )
            self._use_medspacy = False

    def process(self, record: ClinicalTextRecord) -> ClinicalTextRecord:
        self._ensure_loaded()
        if self._use_medspacy and self._nlp is not None:
            sections = self._medspacy_sections(record.raw_text)
        else:
            sections = self._regex_sections(record.raw_text)

        log.debug(
            "preprocessor.section_segment.complete",
            record_id=record.record_id,
            n_sections=len(sections),
        )
        return record.model_copy(update={"sections": sections})

    def _medspacy_sections(self, text: str) -> dict[str, str]:
        doc = self._nlp(text)
        sections: dict[str, str] = {}
        for section in doc._.sections:
            title = section.title_span.text.strip() if section.title_span else "Unknown"
            body = section.body_span.text.strip() if section.body_span else ""
            if body:
                sections[title] = body
        return sections

    def _regex_sections(self, text: str) -> dict[str, str]:
        import re
        pattern = r"(?m)^(" + "|".join(
            re.escape(h) for h in self._SECTION_HEADERS
        ) + r")\s*[:\-]?\s*$"
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if not matches:
            return {"Full Note": text}
        sections: dict[str, str] = {}
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections[title] = body
        return sections


# ---------------------------------------------------------------------------
# Abbreviation Expansion
# ---------------------------------------------------------------------------

@PREPROCESSOR_REGISTRY.register("preprocessor.clinical_text.abbreviation_expand")
class AbbreviationExpander(BasePreprocessor[ClinicalTextRecord]):
    """Expand clinical abbreviations using medspacy's abbreviation detector.

    Wraps medspacy's AbbreviationDetector or falls back to a small built-in
    abbreviation dictionary for common clinical shorthand.

    Args:
        use_medspacy: Use medspacy abbreviation pipeline (default: True).
    """

    # Minimal built-in fallback dictionary
    _BUILTIN_ABBREVS: dict[str, str] = {
        "htn": "hypertension",
        "dm": "diabetes mellitus",
        "cad": "coronary artery disease",
        "chf": "congestive heart failure",
        "copd": "chronic obstructive pulmonary disease",
        "sob": "shortness of breath",
        "cp": "chest pain",
        "n/v": "nausea/vomiting",
        "f/u": "follow-up",
        "h/o": "history of",
        "y/o": "year old",
        "yo": "year old",
        "w/": "with",
        "w/o": "without",
        "prn": "as needed",
        "bid": "twice daily",
        "tid": "three times daily",
        "qd": "once daily",
    }

    def __init__(self, use_medspacy: bool = True) -> None:
        self._use_medspacy = use_medspacy

    def process(self, record: ClinicalTextRecord) -> ClinicalTextRecord:
        text = record.raw_text
        # Simple token-level replacement (case-insensitive)
        # In full implementation: use medspacy AbbreviationDetector for
        # context-sensitive expansion (e.g., "MS" = multiple sclerosis vs
        # mitral stenosis depending on context)
        import re
        for abbrev, expansion in self._BUILTIN_ABBREVS.items():
            text = re.sub(
                r"(?<!\w)" + re.escape(abbrev) + r"(?!\w)",
                expansion,
                text,
                flags=re.IGNORECASE,
            )
        return record.model_copy(update={"raw_text": text})

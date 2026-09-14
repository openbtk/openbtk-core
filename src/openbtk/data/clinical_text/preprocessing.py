"""``SectionSegmenter`` (docs/05_DATA_MODALITY_SPEC.md section 1.2): splits
a ``ClinicalTextRecord``'s text into labelled sections, populating
``sections``.

Two backends:

  - ``"rule"`` (default): a curated vocabulary of common clinical section
    headers (chief complaint, history of present illness, assessment and
    plan, ...), matched case-insensitively against a line ending in a
    colon. Zero dependencies, always available -- safe to import and
    construct with zero extras installed.
  - ``"medspacy"``: medspacy's own sectionizer (a spaCy pipeline
    underneath), lazily loaded on first ``process()`` call, never in
    ``__init__`` (CLAUDE.md rule 11) -- real, trained section detection,
    at the cost of the ``text`` extra and a downloaded spaCy model. NOT
    the default, for the same zero-extras-safety reason
    ``openbtk.deid.recognizers.ner.NERRecognizer`` isn't either.

**Real, disclosed limitation of the rule backend**: its vocabulary is a
fixed, curated list -- nothing in the project's docs specifies one, so
this is a from-scratch, reasonable-but-incomplete choice, not a claim of
completeness. An institution's non-standard headers will not be
recognized; the medspacy backend or a larger vocabulary are both real
options for closing that gap, not silently assumed away.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, ClassVar, Literal

from openbtk.core._lazy import require
from openbtk.core.base import BasePreprocessor
from openbtk.core.errors import ProcessingError
from openbtk.core.registry import PREPROCESSOR_REGISTRY
from openbtk.core.schemas import TextSpan
from openbtk.data.clinical_text.schemas import ClinicalTextRecord

if TYPE_CHECKING:
    from spacy.language import Language

_RULE_CONFIDENCE = 1.0
"""A deterministic keyword match either happened or didn't -- this is not
a claim about RECALL (the vocabulary is incomplete, by design, see module
docstring), only that a match found IS the thing it claims to be."""

_MEDSPACY_CONFIDENCE = 0.85
"""A documented default, not a calibrated value: medspacy's sectionizer
does not expose a native per-section confidence score either."""

_KNOWN_SECTIONS: dict[str, str] = {
    "chief complaint": "chief_complaint",
    "history of present illness": "history_of_present_illness",
    "past medical history": "past_medical_history",
    "past surgical history": "past_surgical_history",
    "medications": "medications",
    "allergies": "allergies",
    "social history": "social_history",
    "family history": "family_history",
    "review of systems": "review_of_systems",
    "physical exam": "physical_exam",
    "physical examination": "physical_exam",
    "assessment and plan": "assessment_and_plan",
    "assessment": "assessment",
    "plan": "plan",
    "discharge diagnosis": "discharge_diagnosis",
    "discharge diagnoses": "discharge_diagnosis",
    "discharge medications": "discharge_medications",
    "discharge instructions": "discharge_instructions",
    "hospital course": "hospital_course",
    "follow-up": "follow_up",
    "followup": "follow_up",
}
"""Header phrase (lowercase, no trailing colon) -> snake_case section
label. Labels deliberately match medspacy's own naming convention where
they overlap, so a caller comparing output across backends sees the same
vocabulary for the sections both happen to find."""

_HEADER_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z /\-]{1,45}:[ \t]*$")


def _rule_based_sections(text: str) -> dict[str, TextSpan]:
    # (label, this header line's own start offset, this section's body start)
    headers: list[tuple[str, int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if _HEADER_LINE_RE.match(stripped):
            candidate = stripped[:-1].strip().lower()  # drop trailing colon
            label = _KNOWN_SECTIONS.get(candidate)
            if label is not None:
                headers.append((label, offset, offset + len(line)))
        offset += len(line)

    sections: dict[str, TextSpan] = {}
    for i, (label, _header_start, body_start) in enumerate(headers):
        # A section's body ends where the NEXT header LINE begins -- not
        # where the next section's body begins, which would swallow that
        # next header's own text into this section by mistake.
        body_end = headers[i + 1][1] if i + 1 < len(headers) else len(text)
        sections[label] = TextSpan(
            start=body_start, end=body_end, label=label, confidence=_RULE_CONFIDENCE
        )
    return sections


_medspacy_pipeline_cache: Language | None = None


def _get_medspacy_pipeline() -> Language:
    """Lazily build and cache a spaCy + medspaCy sectionizer pipeline. Not
    called from ``__init__`` -- CLAUDE.md rule 11: constructors do no I/O."""
    global _medspacy_pipeline_cache
    if _medspacy_pipeline_cache is None:
        spacy = require("spacy", extra="text")
        require("medspacy", extra="text")  # registers spaCy's factory as a side effect
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise ProcessingError(
                "spaCy model 'en_core_web_sm' is required but not "
                "downloaded. Install it with: "
                "python -m spacy download en_core_web_sm",
                context={"model": "en_core_web_sm"},
            ) from e
        nlp.add_pipe("medspacy_sectionizer")
        _medspacy_pipeline_cache = nlp
    return _medspacy_pipeline_cache


def _medspacy_sections(text: str) -> dict[str, TextSpan]:
    nlp = _get_medspacy_pipeline()
    # medspacy's sectionizer fires spaCy's own internal "[W036] The
    # component 'matcher' does not have any patterns defined" UserWarning
    # on every call, using its default rule set -- confirmed by direct
    # reproduction, third-party library noise unrelated to anything in
    # this module. openbtk's own pytest config promotes every warning to
    # an error (pyproject.toml's filterwarnings), so left unhandled this
    # would fail every real call. Narrowly scoped to this one call rather
    # than weakening that policy project-wide.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*does not have any patterns defined.*"
        )
        doc = nlp(text)
    sections: dict[str, TextSpan] = {}
    for section in doc._.sections:
        if section.category is None or section.body_span is None:
            continue
        # medspacy's Section.body_span is a (start_token, end_token) TOKEN
        # index pair, NOT a spaCy Span with .start_char/.end_char --
        # confirmed by direct inspection (its own repr looks Span-shaped,
        # which is what made this easy to get wrong without actually
        # running it). Slicing the Doc converts it to a real Span, which
        # DOES expose character offsets.
        body_start_tok, body_end_tok = section.body_span
        if body_start_tok == body_end_tok:
            continue  # empty body -- nothing to report a span for
        body = doc[body_start_tok:body_end_tok]
        sections[section.category] = TextSpan(
            start=body.start_char,
            end=body.end_char,
            label=section.category,
            confidence=_MEDSPACY_CONFIDENCE,
        )
    return sections


@PREPROCESSOR_REGISTRY.register("preprocessor.clinical_text.section_segment")
class SectionSegmenter(BasePreprocessor[ClinicalTextRecord]):
    """Populate ``record.sections`` from ``record.text``. See module
    docstring for the two backends.

    Example:
        >>> record = ClinicalTextRecord(
        ...     record_id="n1",
        ...     source="synthea",
        ...     text="Chief Complaint:\\nchest pain\\nPlan:\\nadmit",
        ... )
        >>> result = SectionSegmenter().process(record)
        >>> sorted(result.sections)
        ['chief_complaint', 'plan']
    """

    requires_model_download: ClassVar[bool] = False
    """Only true for the (non-default) "medspacy" backend -- see
    _get_medspacy_pipeline. No single class-level flag can express
    "sometimes"; callers who need to know should check `self._backend`."""

    def __init__(self, *, backend: Literal["rule", "medspacy"] = "rule") -> None:
        self._backend = backend

    def process(self, record: ClinicalTextRecord) -> ClinicalTextRecord:
        """Raises:
        ProcessingError: If the "medspacy" backend's model is not
            downloaded. The "rule" backend never raises.
        """
        if self._backend == "rule":
            sections = _rule_based_sections(record.text)
        else:
            sections = _medspacy_sections(record.text)
        return record.model_copy(update={"sections": sections})

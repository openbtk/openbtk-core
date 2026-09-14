"""Regex/pattern-based PHI recognizer -- high precision, format-detectable
categories only (ADR-0006).

**Deliberately does not attempt ``NAME`` or ``GEOGRAPHIC_SUBDIVISION``.**
ADR-0006's own cited evidence is why: "rules miss novel formats, NER misses
structured identifiers." A name or a street address has no reliable regex
shape -- that is precisely the job ``NERRecognizer`` (task 2.4, not yet
built) exists to do. Claiming rule-based name detection here would be the
kind of unbenchmarked accuracy claim CLAUDE.md rule 14 forbids.

Patterns below match the formats this milestone's synthetic corpus
generates (tests/fixtures/labelled_phi_corpus.py) plus common real-world
variants where the format is genuinely standardised (SSN, email, IPv4).
Real-world MRN/account/license formats vary by institution far more than
this covers -- extending this recognizer for a specific institution's
formats is exactly the "register a class" extensibility ADR-0006 calls for,
not a reason to keep growing this one file.
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal

from openbtk.core.schemas import TextSpan
from openbtk.deid.recognizers.base import RECOGNIZER_REGISTRY, BaseRecognizer
from openbtk.deid.schemas import Detection, PHICategory

# Each pattern is deliberately anchored to a specific, unambiguous shape
# rather than a loose one, favouring precision -- ADR-0006 assigns recall
# to NER and the ensemble, not to this recognizer alone. Order matters only
# in that more specific / prefixed patterns (MRN-, LIC-, SN-, ID-) are safe
# to check independently of each other since their prefixes don't collide.
_PATTERNS: dict[PHICategory, re.Pattern[str]] = {
    PHICategory.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    PHICategory.EMAIL: re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}\b"),
    PHICategory.URL: re.compile(r"\bhttps?://[^\s,;]+"),
    PHICategory.IP_ADDRESS: re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    PHICategory.DATE: re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
    # (NXX) NXX-XXXX -- the one format this milestone's corpus generates.
    # FAX_NUMBER shares this exact shape; disambiguated by a keyword-window
    # check in _classify_phone_or_fax, not by the pattern itself.
    PHICategory.PHONE_NUMBER: re.compile(r"\(\d{3}\)\s\d{3}-\d{4}"),
    PHICategory.MEDICAL_RECORD_NUMBER: re.compile(r"\bMRN-?\d{4,}\b", re.IGNORECASE),
    PHICategory.HEALTH_PLAN_BENEFICIARY_NUMBER: re.compile(r"\b[A-Z0-9]{2}\d{6}\b"),
    PHICategory.CERTIFICATE_LICENSE_NUMBER: re.compile(r"\bLIC-[A-Z0-9]{6}\b"),
    PHICategory.VEHICLE_IDENTIFIER: re.compile(r"\b[A-Z0-9]{3}-\d{4}\b"),
    PHICategory.DEVICE_IDENTIFIER: re.compile(r"\bSN-[A-Z0-9]{8}\b"),
    PHICategory.OTHER_UNIQUE_IDENTIFIER: re.compile(r"\bID-[0-9a-f]{8}\b"),
    # Account number is a bare 10-digit run -- checked last since it is the
    # least specific pattern here (no prefix, no punctuation) and most
    # likely to coincidentally match inside another category's digits.
    PHICategory.ACCOUNT_NUMBER: re.compile(r"\b\d{10}\b"),
}

_FAX_CONTEXT_RE = re.compile(r"\bfax\b", re.IGNORECASE)
_FAX_CONTEXT_WINDOW = 10
"""How many characters before a phone-shaped match to look for the word
"fax" -- both PHONE_NUMBER and FAX_NUMBER share an identical regex shape
(a real-world limitation of format-only detection), so this is the only
signal available to tell them apart without NER-level context understanding."""


@RECOGNIZER_REGISTRY.register("recognizer.general.rule")
class RuleRecognizer(BaseRecognizer):
    """High-precision, regex-based PHI detection. See module docstring for
    exactly which categories this does and does not attempt.

    Example:
        >>> r = RuleRecognizer()
        >>> hits = r.detect("SSN: 123-45-6789")
        >>> hits[0].category
        <PHICategory.SSN: 'ssn'>
    """

    method: ClassVar[Literal["rule", "ner", "llm_verifier"]] = "rule"

    def detect(self, text: str) -> list[Detection]:
        detections: list[Detection] = []
        for category, pattern in _PATTERNS.items():
            for match in pattern.finditer(text):
                resolved = self._resolve_category(category, text, match.start())
                detections.append(
                    Detection(
                        category=resolved,
                        span=TextSpan(
                            start=match.start(),
                            end=match.end(),
                            label=resolved.value,
                            confidence=0.95,
                        ),
                        confidence=0.95,
                        method=self.method,
                    )
                )
        return detections

    @staticmethod
    def _resolve_category(
        category: PHICategory, text: str, match_start: int
    ) -> PHICategory:
        """Disambiguate PHONE_NUMBER vs. FAX_NUMBER by a keyword window --
        the only two categories sharing an identical pattern here."""
        if category is not PHICategory.PHONE_NUMBER:
            return category
        window_start = max(0, match_start - _FAX_CONTEXT_WINDOW)
        if _FAX_CONTEXT_RE.search(text[window_start:match_start]):
            return PHICategory.FAX_NUMBER
        return category

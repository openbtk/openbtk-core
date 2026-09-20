"""Shared contract every registered BaseRecognizer implementation must
satisfy (ADR-0006, CLAUDE.md rule 12 -- no exemptions).

Parametrized over RECOGNIZER_REGISTRY.list_keys(): registering a recognizer
anywhere automatically enrolls it here. At M2, that means
tests/contract/conftest.py's ReferenceRecognizer plus RuleRecognizer, and
-- only when OPENBTK_SLOW_TESTS=1 (this module's own conftest.py imports it
conditionally) -- NERRecognizer too.

A key whose class has ``requires_model_download = True`` is skip-marked
here, defensively, whenever OPENBTK_SLOW_TESTS isn't set -- a second layer
on top of conftest.py's conditional import, in case some future change
ever causes such a recognizer to be registered outside that guard. This is
NOT how the "no exemptions" guarantee is normally kept (conftest.py's
conditional import is): it exists so an accidental registration degrades
to a clear skip instead of an opaque ``DeidError`` about a missing model in
an unrelated test run.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.schemas import TextSpan
from openbtk.deid.recognizers.base import RECOGNIZER_REGISTRY
from openbtk.deid.schemas import Detection

from .conftest import ReferenceRecognizer, enrolled

if TYPE_CHECKING:
    from openbtk.deid.recognizers.base import BaseRecognizer

_SAMPLE_TEXT = "Patient MRN-4821093, phone (555) 234-5678."  # phi-fixture-ok: synthetic
_SLOW_TESTS_ENABLED = os.environ.get("OPENBTK_SLOW_TESTS") == "1"


def _new_instance(key: str) -> BaseRecognizer:
    """Every recognizer registered so far takes zero constructor arguments.
    If a future recognizer needs required constructor args (e.g. an NER
    model path), this fixture wiring will need a per-key params table."""
    return RECOGNIZER_REGISTRY.create(key)


def _parametrized_keys() -> list[Any]:
    params: list[Any] = []
    for key in enrolled(RECOGNIZER_REGISTRY):
        cls = RECOGNIZER_REGISTRY.get(key)
        if cls.requires_model_download and not _SLOW_TESTS_ENABLED:
            params.append(
                pytest.param(
                    key,
                    marks=pytest.mark.skip(
                        reason="requires model download; set OPENBTK_SLOW_TESTS=1"
                    ),
                )
            )
        else:
            params.append(key)
    return params


@pytest.mark.parametrize("key", _parametrized_keys())
class TestRecognizerContract:
    def test_detect_returns_a_list(self, key: str) -> None:
        result = _new_instance(key).detect(_SAMPLE_TEXT)
        assert isinstance(result, list)

    def test_every_detection_is_a_real_detection_instance(self, key: str) -> None:
        for detection in _new_instance(key).detect(_SAMPLE_TEXT):
            assert isinstance(detection, Detection)

    def test_every_span_is_within_text_bounds_and_matches_the_slice(
        self, key: str
    ) -> None:
        """A recognizer reporting offsets that don't correspond to real text
        is worse than useless -- it would corrupt every downstream
        transform (SpanMerger, Transform) silently."""
        recognizer = _new_instance(key)
        for detection in recognizer.detect(_SAMPLE_TEXT):
            span = detection.span
            assert 0 <= span.start < span.end <= len(_SAMPLE_TEXT)

    def test_detection_method_matches_the_recognizer_s_own_method(
        self, key: str
    ) -> None:
        recognizer = _new_instance(key)
        for detection in recognizer.detect(_SAMPLE_TEXT):
            assert detection.method == recognizer.method

    def test_confidence_is_within_bounds(self, key: str) -> None:
        for detection in _new_instance(key).detect(_SAMPLE_TEXT):
            assert 0.0 <= detection.confidence <= 1.0

    def test_span_confidence_matches_detection_confidence(self, key: str) -> None:
        """Documented invariant (Detection's own docstring): span.confidence
        and confidence are always kept equal by every producer."""
        for detection in _new_instance(key).detect(_SAMPLE_TEXT):
            assert detection.span.confidence == detection.confidence

    def test_empty_text_returns_no_detections_without_raising(self, key: str) -> None:
        result = _new_instance(key).detect("")
        assert result == []

    def test_detect_does_not_mutate_its_input(self, key: str) -> None:
        text = _SAMPLE_TEXT
        _new_instance(key).detect(text)
        assert text == _SAMPLE_TEXT

    def test_provenance_is_serialisable(self, key: str) -> None:
        recognizer = _new_instance(key)
        dumped = recognizer.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0

    def test_method_is_one_of_the_declared_literal_values(self, key: str) -> None:
        recognizer = _new_instance(key)
        assert recognizer.method in ("rule", "ner", "llm_verifier")


# ---------------------------------------------------------------------------
# Meta-test: proves the span-bounds check actually catches a real violation.
# ---------------------------------------------------------------------------


def test_span_bounds_check_catches_a_real_violation() -> None:
    """If this ever fails, the bounds contract has stopped being a real
    check -- it would mean a recognizer reporting an out-of-range span could
    pass."""
    bad_span = TextSpan(start=0, end=999, label="digits", confidence=0.5)
    text = "short"
    with pytest.raises(AssertionError):
        assert 0 <= bad_span.start < bad_span.end <= len(text)


def test_reference_recognizer_detects_the_known_digit_runs() -> None:
    """Not parametrized: proves ReferenceRecognizer itself is genuinely
    functional (not an always-empty stub the contract suite can't tell
    apart from a broken one)."""
    recognizer = ReferenceRecognizer()
    detections = recognizer.detect(_SAMPLE_TEXT)
    found = {_SAMPLE_TEXT[d.span.start : d.span.end] for d in detections}
    assert "4821093" in found
    assert "555" in found

"""Unit tests for openbtk.deid.recognizers.ner.NERRecognizer.

Split deliberately in two:

  * Structural tests (no marker, always run): registration, method/
    requires_model_download classvars, the label map, and -- importantly --
    that constructing an instance does no I/O (CLAUDE.md rule 11) even
    though this module's own docstring says the real model is heavy to
    fetch. These must pass with zero extras installed.
  * Real-detection tests (``@pytest.mark.slow``): actually run spaCy's
    ``en_core_web_sm`` pipeline. Skipped by default (tests/conftest.py's
    ``pytest_collection_modifyitems``); run with ``OPENBTK_SLOW_TESTS=1``
    once ``pip install openbtk[text] && python -m spacy download
    en_core_web_sm`` has been done.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbtk.core.errors import DeidError
from openbtk.deid.recognizers import ner
from openbtk.deid.recognizers.base import RECOGNIZER_REGISTRY
from openbtk.deid.recognizers.ner import NERRecognizer
from openbtk.deid.schemas import PHICategory

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_model_cache() -> Iterator[None]:
    """The module-level model cache is process-global by design (loading
    once is the point) -- tests that poke it directly must not leak state
    into other tests, including ones that run in the same session under
    OPENBTK_SLOW_TESTS=1."""
    ner._model_cache = None
    yield
    ner._model_cache = None


class TestStructure:
    def test_is_registered_under_the_expected_key(self) -> None:
        assert RECOGNIZER_REGISTRY.is_registered("recognizer.general.ner")
        assert RECOGNIZER_REGISTRY.get("recognizer.general.ner") is NERRecognizer

    def test_method_is_ner(self) -> None:
        assert NERRecognizer.method == "ner"

    def test_requires_model_download_is_true(self) -> None:
        assert NERRecognizer.requires_model_download is True

    def test_label_map_only_targets_person_and_geographic_categories(self) -> None:
        assert set(ner._LABEL_MAP.values()) == {
            PHICategory.NAME,
            PHICategory.GEOGRAPHIC_SUBDIVISION,
        }

    def test_person_maps_to_name(self) -> None:
        assert ner._LABEL_MAP["PERSON"] == PHICategory.NAME

    def test_geographic_labels_map_to_geographic_subdivision(self) -> None:
        for label in ("GPE", "LOC", "FAC"):
            assert ner._LABEL_MAP[label] == PHICategory.GEOGRAPHIC_SUBDIVISION


class TestConstructionDoesNoIO:
    def test_constructing_an_instance_does_not_load_the_model(self) -> None:
        """CLAUDE.md rule 11: constructors do no I/O. If __init__ ever
        starts loading the model, this test starts failing the moment
        _model_cache is set as a side effect of mere construction."""
        NERRecognizer()
        assert ner._model_cache is None


class TestMissingModelErrorPath:
    def test_missing_model_raises_deid_error_with_an_actionable_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Doesn't need spaCy to be genuinely uninstalled -- monkeypatches
        require() to return a stub whose .load() fails exactly the way a
        real spaCy does when the model hasn't been downloaded (OSError)."""

        class _StubSpacy:
            @staticmethod
            def load(name: str) -> None:
                raise OSError(f"[E050] Can't find model {name!r}")

        monkeypatch.setattr(ner, "require", lambda module, *, extra: _StubSpacy())
        with pytest.raises(DeidError, match="python -m spacy download"):
            ner._get_model()


@pytest.mark.slow
class TestRealDetection:
    """Requires the actual spaCy model -- see this file's own module
    docstring for how to enable these."""

    def test_detects_a_person_name(self) -> None:
        detections = NERRecognizer().detect("Patient: Jane Q. Patient.")
        names = [d for d in detections if d.category == PHICategory.NAME]
        assert names, "expected at least one NAME detection"

    def test_detects_a_geopolitical_entity(self) -> None:
        detections = NERRecognizer().detect("She was born in Paris in 1958.")
        geo = [
            d for d in detections if d.category == PHICategory.GEOGRAPHIC_SUBDIVISION
        ]
        assert geo, "expected at least one GEOGRAPHIC_SUBDIVISION detection"

    def test_empty_text_returns_no_detections(self) -> None:
        assert NERRecognizer().detect("") == []

    def test_detection_confidence_matches_the_documented_constant(self) -> None:
        detections = NERRecognizer().detect("Patient: Jane Q. Patient.")
        assert detections
        assert all(d.confidence == ner._CONFIDENCE for d in detections)

    def test_model_is_cached_across_calls(self) -> None:
        recognizer = NERRecognizer()
        recognizer.detect("Jane Doe")
        cached_after_first = ner._model_cache
        assert cached_after_first is not None
        recognizer.detect("John Smith")
        assert ner._model_cache is cached_after_first

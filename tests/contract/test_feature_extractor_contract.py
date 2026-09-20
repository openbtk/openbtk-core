"""Shared contract every registered BaseFeatureExtractor implementation must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.registry import FEATURE_EXTRACTOR_REGISTRY

from .conftest import FixtureChunk, enrolled

if TYPE_CHECKING:
    from openbtk.core.base import BaseFeatureExtractor


def _new_instance(key: str) -> BaseFeatureExtractor[Any]:
    return FEATURE_EXTRACTOR_REGISTRY.create(key)


@pytest.mark.parametrize("key", enrolled(FEATURE_EXTRACTOR_REGISTRY))
class TestFeatureExtractorContract:
    def test_extract_returns_dict_of_floats(self, key: str) -> None:
        extractor = _new_instance(key)
        chunk = FixtureChunk(chunk_id="c1", record_id="r1", text="hello")
        features = extractor.extract(chunk)
        assert isinstance(features, dict)
        assert all(isinstance(k, str) for k in features)
        assert all(isinstance(v, float) for v in features.values())

    def test_extract_is_deterministic(self, key: str) -> None:
        extractor = _new_instance(key)
        chunk = FixtureChunk(chunk_id="c1", record_id="r1", text="hello")
        assert extractor.extract(chunk) == extractor.extract(chunk)

    def test_provenance_is_serialisable(self, key: str) -> None:
        extractor = _new_instance(key)
        dumped = extractor.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0

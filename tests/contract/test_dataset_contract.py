"""Shared contract every registered BaseDatasetAdapter must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbtk.core.errors import DatasetError
from openbtk.core.registry import DATASET_REGISTRY

if TYPE_CHECKING:
    from openbtk.core.base import BaseDatasetAdapter


def _new_instance(key: str) -> BaseDatasetAdapter:
    return DATASET_REGISTRY.create(key)


@pytest.mark.parametrize("key", DATASET_REGISTRY.list_keys())
class TestDatasetAdapterContract:
    def test_name_is_a_nonempty_string(self, key: str) -> None:
        ds = _new_instance(key)
        assert isinstance(ds.name, str) and ds.name

    def test_license_is_declared(self, key: str) -> None:
        """Every dataset adapter must declare a licence -- an SPDX
        identifier or an access-request URL for restricted data
        (docs/04_API_DESIGN.md section 3). Never left as the base
        placeholder in a real implementation."""
        ds = _new_instance(key)
        assert isinstance(ds.license, str) and ds.license

    def test_requires_credentials_is_a_bool(self, key: str) -> None:
        ds = _new_instance(key)
        assert isinstance(ds.requires_credentials, bool)

    def test_load_returns_something(self, key: str) -> None:
        """An open dataset's zero-argument load() yields something. A
        credentialed one must instead REFUSE, with a DatasetError that names
        where to register, when given no source -- it must never download,
        fall back to a substitute, or return an empty stand-in
        (docs/05_DATA_MODALITY_SPEC.md section 1.5: "Credentialed adapters
        never auto-download")."""
        ds = _new_instance(key)
        if ds.requires_credentials:
            with pytest.raises(DatasetError, match="Register at"):
                ds.load()
            return
        assert ds.load() is not None

    def test_provenance_is_serialisable(self, key: str) -> None:
        ds = _new_instance(key)
        dumped = ds.provenance().model_dump_json()
        assert isinstance(dumped, str) and len(dumped) > 0

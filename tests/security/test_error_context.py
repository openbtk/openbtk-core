"""Release blocker: no realistic-looking PHI may ever reach an exception's
message or .context.

This is the enforcement counterpart to the contract errors.py already
documents (module docstring, section 8): ".context holds identifiers only
... enforced by tests/security/test_error_context.py, not just documented
here." Drives every fabricated identifier in conftest.py's
REALISTIC_PHI_STRINGS through the real error-raising paths that exist at M1:
registry lookups, config validation, and ReferenceLoader's failure path
(tests/contract/test_loader_contract.py already covers ReferenceLoader's own
bad-source case with one fixed string; this file adds the fuzz across every
adversarial string, and covers registry/config paths that file does not
touch at all).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import BaseModel, ConfigDict

from openbtk.core.base import BaseLoader
from openbtk.core.config import PipelineConfig
from openbtk.core.errors import ConfigError, LoaderError, OpenBTKError, RegistryError
from openbtk.core.registry import LOADER_REGISTRY

# Self-contained, deliberately not imported from tests/contract/conftest.py:
# pytest loads that file as a conftest plugin under its own import identity,
# so a second, explicit `import tests.contract.conftest` would create a
# distinct module object whose top-level `@REGISTRY.register(...)`
# decorators run again -- a real duplicate-registration RegistryError,
# discovered by actually running this file alongside the contract suite
# rather than in isolation. Same self-registration pattern already used by
# tests/unit/core/test_config.py for the same reason.
_PROBE_LOADER_KEY = "loader.general.error_context_probe"


class _ProbeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str


if not LOADER_REGISTRY.is_registered(_PROBE_LOADER_KEY):

    class _ProbeLoader(BaseLoader[Iterator[str], _ProbeRecord]):
        """Raises LoaderError for a non-iterable source -- same bad-source
        shape as tests/contract/conftest.py's ReferenceLoader, reproduced
        locally rather than imported."""

        def load(self, source: Iterator[str]) -> Iterator[_ProbeRecord]:
            try:
                iterator = iter(source)
            except TypeError as e:
                raise LoaderError(
                    "Source is not iterable.",
                    context={"modality": "general", "stage": "load"},
                ) from e
            for i, line in enumerate(iterator):
                yield _ProbeRecord(record_id=f"{i}:{line}")

    LOADER_REGISTRY.register(_PROBE_LOADER_KEY)(_ProbeLoader)


def _assert_no_leak(exc: OpenBTKError, secret: str) -> None:
    assert secret not in str(exc), (
        f"PHI-shaped value leaked into exception message: {exc}"
    )
    assert secret not in repr(exc), (
        f"PHI-shaped value leaked into exception repr: {exc!r}"
    )
    assert secret not in str(exc.context), (
        f"PHI-shaped value leaked into exception context: {exc.context}"
    )


class TestConfigValidationFailuresDoNotLeak:
    def test_bad_yaml_containing_phi_shaped_content_does_not_leak(
        self, tmp_path: object, realistic_phi_string: str
    ) -> None:
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        yaml_file = tmp_path / "bad.yaml"
        # Malformed YAML (unclosed flow sequence) whose payload happens to
        # be a realistic PHI-shaped string -- confirms the parser's own
        # error does not echo the offending content back in a way that
        # would show up in a log downstream.
        yaml_file.write_text(f"name: [{realistic_phi_string}\n")
        with pytest.raises(ConfigError) as exc_info:
            PipelineConfig.from_yaml(yaml_file)
        _assert_no_leak(exc_info.value, realistic_phi_string)

    def test_create_from_config_error_context_never_holds_param_values(
        self, realistic_phi_string: str
    ) -> None:
        """create_from_config's own docstring/comment says .context holds
        config_keys only, never values -- this is the behavioural proof."""
        with pytest.raises(RegistryError) as exc_info:
            LOADER_REGISTRY.create_from_config(
                {"params": {"patient_name": realistic_phi_string}}
            )
        _assert_no_leak(exc_info.value, realistic_phi_string)
        assert exc_info.value.context == {
            "registry": "loader",
            "config_keys": ["params"],
        }


class TestLoaderFailuresDoNotLeak:
    def test_reference_loader_bad_source_does_not_leak_phi_shaped_repr(
        self, realistic_phi_string: str
    ) -> None:
        """Same technique as test_loader_contract.py's
        test_reference_loader_error_context_contains_no_phi, fuzzed across
        every adversarial string in conftest.py instead of one fixed one."""

        class _PhiShapedSource:
            def __repr__(self) -> str:
                return realistic_phi_string

        loader = LOADER_REGISTRY.create(_PROBE_LOADER_KEY)
        with pytest.raises(LoaderError) as exc_info:
            list(loader.load(_PhiShapedSource()))
        _assert_no_leak(exc_info.value, realistic_phi_string)

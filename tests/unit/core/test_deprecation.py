"""The deprecation mechanism (M11 11.3): the decorator, ``warn_deprecated``, and
the full registry-key rename cycle exercised end to end on a private registry."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from openbtk.core.base import BaseLoader
from openbtk.core.deprecation import (
    OpenBTKDeprecationWarning,
    deprecated,
    warn_deprecated,
)
from openbtk.core.registry import Registry


class TestWarnDeprecated:
    def test_message_names_the_release_the_removal_and_the_replacement(self) -> None:
        with pytest.warns(OpenBTKDeprecationWarning) as caught:
            warn_deprecated("thing()", since="1.2.0", removal="2.0.0", use="other()")
        text = str(caught[0].message)
        assert "thing()" in text and "1.2.0" in text and "2.0.0" in text
        assert "use other() instead" in text

    def test_the_replacement_is_optional(self) -> None:
        with pytest.warns(OpenBTKDeprecationWarning) as caught:
            warn_deprecated("thing()", since="1.2.0", removal="2.0.0")
        assert "use" not in str(caught[0].message)

    @pytest.mark.parametrize("bad", ["1.2", "v1.2.0", "one", "1.2.x", ""])
    def test_versions_must_be_semver_shaped(self, bad: str) -> None:
        with pytest.raises(ValueError, match="must look like"):
            warn_deprecated("x", since=bad, removal="2.0.0")
        with pytest.raises(ValueError, match="must look like"):
            warn_deprecated("x", since="1.0.0", removal=bad)

    def test_it_is_a_deprecation_warning_subclass(self) -> None:
        assert issubclass(OpenBTKDeprecationWarning, DeprecationWarning)

    def test_this_suite_turns_it_into_an_error_so_nothing_internal_may_use_it(
        self,
    ) -> None:
        """pyproject's filterwarnings escalates the category. If an OpenBTK code
        path ever called a deprecated API, its test would fail rather than pass
        with a hidden warning."""
        with pytest.raises(OpenBTKDeprecationWarning):
            warn_deprecated("x", since="1.0.0", removal="2.0.0")


class TestDecorator:
    def test_a_deprecated_function_still_works_and_warns(self) -> None:
        @deprecated(since="1.2.0", removal="2.0.0", use="new_fn()")
        def old_fn(x: int, *, y: int = 1) -> int:
            """Add."""
            return x + y

        with pytest.warns(OpenBTKDeprecationWarning, match="new_fn"):
            assert old_fn(2, y=3) == 5

    def test_the_docstring_and_marker_carry_the_notice(self) -> None:
        @deprecated(since="1.2.0", removal="2.0.0")
        def old_fn() -> None:
            """Original docs."""

        assert old_fn.__doc__ is not None
        assert "Original docs." in old_fn.__doc__
        assert "Deprecated since 1.2.0" in old_fn.__doc__
        assert "2.0.0" in old_fn.__deprecated__  # type: ignore[attr-defined]

    def test_name_and_signature_are_preserved(self) -> None:
        @deprecated(since="1.2.0", removal="2.0.0")
        def old_fn(a: int, b: str = "x") -> None: ...

        assert old_fn.__name__ == "old_fn"
        assert list(old_fn.__wrapped__.__code__.co_varnames[:2]) == ["a", "b"]  # type: ignore[attr-defined]

    def test_a_deprecated_method(self) -> None:
        class Thing:
            @deprecated(since="1.2.0", removal="2.0.0", use="Thing.new()")
            def old(self) -> str:
                return "ok"

        with pytest.warns(OpenBTKDeprecationWarning, match=r"Thing\.new"):
            assert Thing().old() == "ok"

    def test_a_deprecated_class_warns_on_construction_and_still_works(self) -> None:
        @deprecated(since="1.2.0", removal="2.0.0", use="NewThing")
        class OldThing:
            def __init__(self, value: int) -> None:
                self.value = value

        with pytest.warns(OpenBTKDeprecationWarning, match="OldThing.*NewThing"):
            assert OldThing(7).value == 7
        assert "Deprecated since 1.2.0" in (OldThing.__doc__ or "")

    def test_it_validates_versions_when_applied(self) -> None:
        with pytest.raises(ValueError, match="must look like"):
            deprecated(since="soon", removal="2.0.0")


class _Loader(BaseLoader[str, Any]):
    def load(self, source: str) -> Any:
        return iter(())


def _registry() -> Registry[Any]:
    registry: Registry[Any] = Registry("loader", BaseLoader)  # type: ignore[type-abstract]
    registry.register("loader.general.new_name")(_Loader)
    return registry


class TestRegistryKeyRenameCycle:
    """The complete rename cycle a real key rename would follow."""

    def test_the_old_key_keeps_resolving_but_warns_naming_the_new_key(self) -> None:
        registry = _registry()
        registry.register_alias(
            "loader.general.old_name",
            "loader.general.new_name",
            since="1.2.0",
            removal="2.0.0",
        )
        with pytest.warns(OpenBTKDeprecationWarning, match="new_name") as caught:
            assert registry.get("loader.general.old_name") is _Loader
        assert "old_name" in str(caught[0].message) and "2.0.0" in str(
            caught[0].message
        )

    def test_create_through_the_old_key_works_and_warns(self) -> None:
        registry = _registry()
        registry.register_alias(
            "loader.general.old_name",
            "loader.general.new_name",
            since="1.2.0",
            removal="2.0.0",
        )
        with pytest.warns(OpenBTKDeprecationWarning):
            assert isinstance(registry.create("loader.general.old_name"), _Loader)

    def test_the_new_key_is_silent(self) -> None:
        registry = _registry()
        registry.register_alias(
            "loader.general.old_name",
            "loader.general.new_name",
            since="1.2.0",
            removal="2.0.0",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would raise
            assert registry.get("loader.general.new_name") is _Loader

    def test_listing_and_membership_do_not_warn_and_include_the_alias(self) -> None:
        registry = _registry()
        registry.register_alias(
            "loader.general.old_name",
            "loader.general.new_name",
            since="1.2.0",
            removal="2.0.0",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            keys = registry.list_keys()
            assert registry.is_registered("loader.general.old_name")
        assert {"loader.general.old_name", "loader.general.new_name"} <= set(keys)

    def test_an_alias_without_since_is_a_plain_alias_and_stays_silent(self) -> None:
        registry = _registry()
        registry.register_alias("loader.general.other", "loader.general.new_name")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert registry.get("loader.general.other") is _Loader

    def test_since_and_removal_must_come_together(self) -> None:
        registry = _registry()
        with pytest.raises(ValueError, match="both since= and removal="):
            registry.register_alias(
                "loader.general.old_name", "loader.general.new_name", since="1.2.0"
            )
        with pytest.raises(ValueError, match="both since= and removal="):
            registry.register_alias(
                "loader.general.old_name", "loader.general.new_name", removal="2.0.0"
            )

    def test_an_old_config_naming_the_old_key_still_validates(self) -> None:
        """A user's config predates the rename; the alias is what keeps it valid."""
        registry = _registry()
        registry.register_alias(
            "loader.general.old_name",
            "loader.general.new_name",
            since="1.2.0",
            removal="2.0.0",
        )
        with pytest.warns(OpenBTKDeprecationWarning):
            assert registry.describe("loader.general.old_name").class_name == "_Loader"

"""Unit tests for openbtk.core.registry beyond what the contract suite covers.

The contract suite exercises registered components' *behaviour*; this file
exercises the registry's own machinery -- key grammar validation, duplicate
and wrong-base-type rejection, aliasing, create_from_config()'s error paths,
describe(), and get_registry(). Originally verified as throwaway adversarial
probes during development; converted to permanent tests here so the same
guarantees are enforced by CI going forward, not just checked once by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, ConfigDict

from openbtk.core.base import BaseChunker, BaseLoader
from openbtk.core.errors import RegistryError
from openbtk.core.registry import Registry, get_registry

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Rec(BaseModel):
    model_config = ConfigDict(frozen=True)
    text: str


@pytest.fixture
def loader_registry() -> Registry[Any]:
    """A throwaway Registry instance per test -- never touches the real
    global LOADER_REGISTRY, so tests here cannot pollute or be polluted by
    the contract suite's reference registrations.

    Typed Registry[Any], not Registry[Any]: BaseLoader is itself
    generic, so a bare reference needs [Any, Any] to satisfy mypy, and the
    fixture's actual job -- being "a registry" for grammar/aliasing/lookup
    tests -- gains nothing from full parametrisation. Registry("loader",
    BaseLoader) deliberately passes an abstract class for issubclass checks
    only, the same pattern (and the same mypy suppression) already
    established in core/registry.py's own global registrations.
    """
    return Registry("loader", BaseLoader)


class _ValidLoader(BaseLoader[str, _Rec]):
    def load(self, source: str) -> Iterator[_Rec]:
        yield _Rec(text=source)


class _ValidChunker(BaseChunker[_Rec, _Rec]):
    def chunk(self, record: _Rec) -> Iterator[_Rec]:
        yield record


class TestKeyGrammar:
    @pytest.mark.parametrize(
        "bad_key",
        [
            "chunker.general.x",  # wrong category for this registry
            "loader.not_a_real_scope.x",  # invalid scope
            "loader.general.BadName",  # uppercase in name
            "loader.general.123start",  # name must start with a letter
            "loader.general",  # too few segments
            "loader.general.x.y",  # too many segments
        ],
    )
    def test_rejects_every_grammar_violation(
        self, loader_registry: Registry[Any], bad_key: str
    ) -> None:
        with pytest.raises(RegistryError):
            loader_registry.register(bad_key)(_ValidLoader)

    def test_accepts_a_well_formed_key(self, loader_registry: Registry[Any]) -> None:
        loader_registry.register("loader.general.well_formed")(_ValidLoader)
        assert "loader.general.well_formed" in loader_registry


class TestRegistration:
    def test_duplicate_key_raises(self, loader_registry: Registry[Any]) -> None:
        loader_registry.register("loader.general.dup")(_ValidLoader)
        with pytest.raises(RegistryError, match="already registered"):
            loader_registry.register("loader.general.dup")(_ValidLoader)

    def test_wrong_base_type_raises(self, loader_registry: Registry[Any]) -> None:
        with pytest.raises(RegistryError, match="does not subclass"):
            loader_registry.register("loader.general.wrong_type")(_ValidChunker)

    def test_register_stamps_registry_key_on_the_class(
        self, loader_registry: Registry[Any]
    ) -> None:
        loader_registry.register("loader.general.stamped")(_ValidLoader)
        assert _ValidLoader.registry_key == "loader.general.stamped"

    def test_registry_constructor_rejects_unknown_category(self) -> None:
        with pytest.raises(RegistryError, match="Unknown registry category"):
            Registry("not_a_real_category", BaseLoader)  # type: ignore[type-abstract]


class TestAliasing:
    def test_alias_resolves_to_the_same_class(
        self, loader_registry: Registry[Any]
    ) -> None:
        loader_registry.register("loader.general.original")(_ValidLoader)
        loader_registry.register_alias(
            "loader.general.old_name", "loader.general.original"
        )
        assert loader_registry.get("loader.general.old_name") is _ValidLoader

    def test_alias_to_nonexistent_key_raises(
        self, loader_registry: Registry[Any]
    ) -> None:
        with pytest.raises(RegistryError, match="not registered"):
            loader_registry.register_alias(
                "loader.general.alias", "loader.general.nope"
            )

    def test_alias_must_also_satisfy_the_grammar(
        self, loader_registry: Registry[Any]
    ) -> None:
        loader_registry.register("loader.general.original")(_ValidLoader)
        with pytest.raises(RegistryError):
            loader_registry.register_alias(
                "chunker.general.wrong_cat", "loader.general.original"
            )

    def test_list_keys_includes_aliases_sorted(
        self, loader_registry: Registry[Any]
    ) -> None:
        loader_registry.register("loader.general.zzz")(_ValidLoader)
        loader_registry.register_alias("loader.general.aaa", "loader.general.zzz")
        assert loader_registry.list_keys() == [
            "loader.general.aaa",
            "loader.general.zzz",
        ]


class TestLookup:
    def test_get_unknown_key_raises_with_available_keys_listed(
        self, loader_registry: Registry[Any]
    ) -> None:
        loader_registry.register("loader.general.known")(_ValidLoader)
        with pytest.raises(RegistryError, match=r"loader\.general\.known"):
            loader_registry.get("loader.general.unknown")

    def test_is_registered_true_and_false(self, loader_registry: Registry[Any]) -> None:
        loader_registry.register("loader.general.exists")(_ValidLoader)
        assert loader_registry.is_registered("loader.general.exists") is True
        assert loader_registry.is_registered("loader.general.nope") is False

    def test_contains_operator(self, loader_registry: Registry[Any]) -> None:
        loader_registry.register("loader.general.exists")(_ValidLoader)
        assert "loader.general.exists" in loader_registry
        assert "loader.general.nope" not in loader_registry


class TestCreate:
    def test_create_instantiates_with_kwargs(
        self, loader_registry: Registry[Any]
    ) -> None:
        loader_registry.register("loader.general.x")(_ValidLoader)
        instance = loader_registry.create("loader.general.x")
        assert isinstance(instance, _ValidLoader)

    def test_create_from_config_happy_path(
        self, loader_registry: Registry[Any]
    ) -> None:
        loader_registry.register("loader.general.x")(_ValidLoader)
        instance = loader_registry.create_from_config({"type": "loader.general.x"})
        assert isinstance(instance, _ValidLoader)

    def test_create_from_config_missing_type_key_raises(
        self, loader_registry: Registry[Any]
    ) -> None:
        with pytest.raises(RegistryError, match="'type' key"):
            loader_registry.create_from_config({"not_type": "x"})

    def test_create_from_config_error_context_has_keys_not_values(
        self, loader_registry: Registry[Any]
    ) -> None:
        """A missing-'type' error must name which keys WERE present, never
        echo their values -- params may legitimately carry operational
        config that shouldn't be duplicated into an error."""
        with pytest.raises(RegistryError) as exc_info:
            loader_registry.create_from_config({"not_type": "sensitive-value-xyz"})
        assert "not_type" in exc_info.value.context["config_keys"]
        assert "sensitive-value-xyz" not in str(exc_info.value.context)


class TestDescribe:
    def test_describe_returns_real_introspection(
        self, loader_registry: Registry[Any]
    ) -> None:
        loader_registry.register("loader.general.x")(_ValidLoader)
        info = loader_registry.describe("loader.general.x")
        assert info.class_name == "_ValidLoader"
        assert info.registry_key == "loader.general.x"
        assert "self" not in info.signature

    def test_describe_unknown_key_raises(self, loader_registry: Registry[Any]) -> None:
        with pytest.raises(RegistryError):
            loader_registry.describe("loader.general.nope")


class TestGetRegistry:
    def test_get_registry_returns_the_global_instance(self) -> None:
        from openbtk.core.registry import LOADER_REGISTRY

        assert get_registry("loader") is LOADER_REGISTRY

    @pytest.mark.parametrize("category", ["finetuner", "recognizer", "totally_made_up"])
    def test_get_registry_rejects_reserved_and_unknown_categories(
        self, category: str
    ) -> None:
        """ "finetuner" is reserved in the key grammar (docs/04_API_DESIGN.md
        section 1) but has no backing base class yet -- it must be rejected
        exactly like any other unknown category until BaseFineTuner exists.
        "recognizer" IS backed by a real class (BaseRecognizer,
        openbtk.deid.recognizers.base) but deliberately lives in its own
        Registry instance, not one of core's 12 -- get_registry() must
        reject it exactly the same way, since core never imports from
        openbtk.deid (layering)."""
        with pytest.raises(RegistryError):
            get_registry(category)


def test_repr_shows_category_and_keys(loader_registry: Registry[Any]) -> None:
    loader_registry.register("loader.general.x")(_ValidLoader)
    r = repr(loader_registry)
    assert "loader" in r
    assert "loader.general.x" in r

"""Unit tests for openbtk.core.registry.Registry."""
from __future__ import annotations

import pytest

from openbtk.core.registry import Registry
from openbtk.core.errors import RegistryError


class _DummyBase:
    """Stand-in base class for registry tests."""


class _DummyImpl(_DummyBase):
    def __init__(self, value: int = 0) -> None:
        self.value = value


def test_register_and_get() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    reg.register("dummy.impl")(_DummyImpl)

    cls = reg.get("dummy.impl")
    assert cls is _DummyImpl


def test_register_duplicate_key_raises() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    reg.register("dummy.impl")(_DummyImpl)

    with pytest.raises(RegistryError, match="already registered"):
        reg.register("dummy.impl")(_DummyImpl)


def test_get_unknown_key_raises() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    with pytest.raises(RegistryError, match="not found"):
        reg.get("does.not.exist")


def test_create_instantiates_with_kwargs() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    reg.register("dummy.impl")(_DummyImpl)

    instance = reg.create("dummy.impl", value=42)
    assert isinstance(instance, _DummyImpl)
    assert instance.value == 42


def test_create_unknown_key_raises() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    with pytest.raises(RegistryError):
        reg.create("nonexistent.key")


def test_create_from_config_valid() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    reg.register("dummy.impl")(_DummyImpl)

    instance = reg.create_from_config({"type": "dummy.impl", "params": {"value": 7}})
    assert isinstance(instance, _DummyImpl)
    assert instance.value == 7


def test_create_from_config_missing_type_raises() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    with pytest.raises(RegistryError, match="'type' key"):
        reg.create_from_config({"params": {}})


def test_create_from_config_no_params_defaults_empty() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    reg.register("dummy.impl")(_DummyImpl)

    instance = reg.create_from_config({"type": "dummy.impl"})
    assert instance.value == 0  # default


def test_list_keys_sorted() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    reg.register("dummy.zebra")(_DummyImpl)
    reg.register("dummy.alpha")(_DummyImpl)

    assert reg.list_keys() == ["dummy.alpha", "dummy.zebra"]


def test_is_registered_and_contains() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    reg.register("dummy.impl")(_DummyImpl)

    assert reg.is_registered("dummy.impl") is True
    assert reg.is_registered("dummy.missing") is False
    assert "dummy.impl" in reg
    assert "dummy.missing" not in reg


def test_register_alias() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    reg.register("dummy.original")(_DummyImpl)
    reg.register_alias("dummy.new_name", "dummy.original")

    assert reg.get("dummy.new_name") is reg.get("dummy.original")


def test_register_alias_missing_source_raises() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    with pytest.raises(RegistryError, match="not registered"):
        reg.register_alias("dummy.new_name", "dummy.does_not_exist")


def test_repr_includes_name_and_keys() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")
    reg.register("dummy.impl")(_DummyImpl)

    text = repr(reg)
    assert "dummy" in text
    assert "dummy.impl" in text


def test_decorator_returns_class_unmodified() -> None:
    reg: Registry[_DummyBase] = Registry("dummy")

    @reg.register("dummy.impl")
    class MyImpl(_DummyBase):
        pass

    assert MyImpl.__name__ == "MyImpl"
    assert reg.get("dummy.impl") is MyImpl

"""OpenBTK component registry.

The ``Registry[T]`` class is what makes a pipeline config-driven rather than
code-driven: a registered class is addressed by a permanent string key, and
config references that key rather than importing the class directly.

Usage -- registration (inside a component module):

    @LOADER_REGISTRY.register("loader.clinical_text.plain_text")
    class PlainTextLoader(BaseLoader[str, ClinicalTextRecord]):
        ...

Usage -- instantiation (pipeline or user code):

    loader = LOADER_REGISTRY.create(
        "loader.clinical_text.plain_text", encoding="utf-8"
    )

Registry keys are public API (docs/04_API_DESIGN.md section 7): renaming one
requires ``register_alias()`` plus a deprecation period of at least one minor
version, never a silent rename.
"""

from __future__ import annotations

import inspect
import re
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from openbtk.core.base import (
    BaseChunker,
    BaseDatasetAdapter,
    BaseEmbeddingProvider,
    BaseFeatureExtractor,
    BaseGuardrail,
    BaseLLMProvider,
    BaseLoader,
    BasePreprocessor,
    BaseReranker,
    BaseSegmenter,
    BaseTerminologyService,
    BaseVectorStore,
    Component,
)
from openbtk.core.errors import RegistryError
from openbtk.core.logging import get_logger

log = get_logger(__name__)

# Bound to Component, not Any -- every registrable class is a Component.
# Imported eagerly (not TYPE_CHECKING) because TypeVar's `bound` argument is
# evaluated immediately, unlike a deferred annotation. This does not create a
# cycle: core.base never imports core.registry (verified when base.py was
# written -- it deliberately avoids importing the top-level openbtk package
# for exactly this reason).
T = TypeVar("T", bound=Component)

# ---------------------------------------------------------------------------
# Key grammar (docs/04_API_DESIGN.md section 1)
# ---------------------------------------------------------------------------

# NOTE: "finetuner" is a reserved category name in the grammar (present in
# both docs/03_ARCHITECTURE.md section 4.2 and docs/04_API_DESIGN.md section
# 1) but has no corresponding BaseFineTuner class -- docs/02_PRD.md's FR-C-03
# (the actual "these base classes must exist" requirement) does not list
# fine-tuning among the required v1 base classes. Add "finetuner" here, and a
# FINETUNER_REGISTRY below, only once a real BaseFineTuner exists to validate
# against -- an empty registry with no base type would be pure speculation.
_VALID_CATEGORIES = frozenset(
    {
        "loader",
        "preprocessor",
        "chunker",
        "segmenter",
        "feature_extractor",
        "embedding",
        "llm",
        "vectorstore",
        "reranker",
        "guardrail",
        "dataset",
        "terminology",
    }
)
_VALID_SCOPES = frozenset(
    {
        "clinical_text",
        "ehr",
        "imaging",
        "biosignals",
        "genomics",
        "video",
        "audio",
        "general",
    }
)
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_key(key: str, *, expected_category: str) -> None:
    """Validate a key against the grammar, and against this registry's own
    category -- catching, for example, a "chunker.*" key mistakenly
    registered on the embedding registry, which the bare grammar alone
    would not catch.

    Raises:
        RegistryError: If the key is malformed or the category does not
            match ``expected_category``.
    """
    parts = key.split(".")
    if len(parts) != 3:
        raise RegistryError(
            f"Invalid registry key {key!r}: expected exactly three "
            "dot-separated segments <category>.<scope>.<name>.",
            context={"key": key},
        )
    category, scope, name = parts
    if category != expected_category:
        raise RegistryError(
            f"Invalid registry key {key!r}: category {category!r} does not "
            f"match this registry's category {expected_category!r}.",
            context={"key": key, "expected_category": expected_category},
        )
    if scope not in _VALID_SCOPES:
        raise RegistryError(
            f"Invalid registry key {key!r}: scope {scope!r} is not one of "
            f"{sorted(_VALID_SCOPES)}.",
            context={"key": key, "scope": scope},
        )
    if not _NAME_RE.match(name):
        raise RegistryError(
            f"Invalid registry key {key!r}: name {name!r} must match "
            "^[a-z][a-z0-9_]*$.",
            context={"key": key, "name": name},
        )


class ComponentInfo(BaseModel):
    """Introspective summary of one registered component.

    Deliberately does not attempt a full JSON Schema of constructor
    parameters: OpenBTK components take plain keyword arguments, not a
    single Pydantic config object, so a fabricated "schema" for an arbitrary
    ``__init__`` signature would often be misleading (a default value that
    is not itself a JSON type, for instance). ``signature`` gives the same
    information in an honest, always-obtainable form.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    registry_key: str = Field(..., description="The key or alias looked up.")
    class_name: str = Field(..., description="The concrete class name.")
    module: str = Field(..., description="Fully qualified module path.")
    docstring: str | None = Field(None, description="The class's docstring, if any.")
    signature: str = Field(
        ..., description="Human-readable constructor signature, excluding 'self'."
    )


class Registry(Generic[T]):
    """A namespaced, type-checked registry of component classes.

    Each registry is bound to exactly one category (e.g. ``"loader"``) and
    one base type (e.g. ``BaseLoader``). Registering a class that is not a
    subclass of that base type, or whose key does not belong to this
    category, raises immediately -- at import time, not when the component
    is eventually used.
    """

    def __init__(self, category: str, base_type: type[T]) -> None:
        if category not in _VALID_CATEGORIES:
            raise RegistryError(
                f"Unknown registry category {category!r}. "
                f"Available: {sorted(_VALID_CATEGORIES)}",
                context={"category": category},
            )
        self._category = category
        self._base_type = base_type
        self._items: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Return a decorator that registers a class under ``key``.

        Args:
            key: A key following ``<category>.<scope>.<name>``, matching
                this registry's own category. Keys are permanent once
                released -- treat as public API.

        Raises:
            RegistryError: If the key is malformed, already registered, or
                the decorated class does not subclass this registry's base
                type.
        """
        _validate_key(key, expected_category=self._category)

        def decorator(cls: type[T]) -> type[T]:
            if not (isinstance(cls, type) and issubclass(cls, self._base_type)):
                raise RegistryError(
                    f"Registry {self._category!r}: {cls!r} does not "
                    f"subclass {self._base_type.__name__}.",
                    context={"registry": self._category, "key": key},
                )
            if key in self._items:
                raise RegistryError(
                    f"Registry {self._category!r}: key {key!r} already "
                    f"registered by {self._items[key].__qualname__}.",
                    context={"registry": self._category, "key": key},
                )
            self._items[key] = cls
            cls.registry_key = key
            log.debug(
                "registry.register",
                registry=self._category,
                key=key,
                cls=cls.__qualname__,
            )
            return cls

        return decorator

    def register_alias(self, alias: str, existing_key: str) -> None:
        """Register ``alias`` as a second name for an already-registered key.

        The primary use is a deprecation transition: a key is renamed by
        registering the new key normally, then aliasing the old key to it,
        so existing configs keep working for at least one minor version.

        Raises:
            RegistryError: If ``alias`` is malformed, or ``existing_key`` is
                not already registered.
        """
        _validate_key(alias, expected_category=self._category)
        if existing_key not in self._items:
            raise RegistryError(
                f"Registry {self._category!r}: cannot alias {alias!r} to "
                f"{existing_key!r} -- {existing_key!r} is not registered.",
                context={"registry": self._category, "existing_key": existing_key},
            )
        self._items[alias] = self._items[existing_key]
        log.debug(
            "registry.alias",
            registry=self._category,
            alias=alias,
            existing_key=existing_key,
        )

    def get(self, key: str) -> type[T]:
        """Return the registered class for ``key``.

        Raises:
            RegistryError: If ``key`` is not registered.
        """
        if key not in self._items:
            raise RegistryError(
                f"Registry {self._category!r}: key {key!r} not found. "
                f"Available: {self.list_keys()}",
                context={"registry": self._category, "key": key},
            )
        return self._items[key]

    def create(self, key: str, **kwargs: Any) -> T:
        """Instantiate the registered class for ``key`` with ``kwargs``.

        Raises:
            RegistryError: If ``key`` is not registered.
            Exception: Whatever the class's constructor raises.
        """
        cls = self.get(key)
        log.debug("registry.create", registry=self._category, key=key)
        return cls(**kwargs)

    def create_from_config(self, config: dict[str, Any]) -> T:
        """Instantiate from a ``{"type": ..., "params": {...}}`` dict.

        Args:
            config: Must contain a ``"type"`` key (the registry key); an
                optional ``"params"`` dict supplies constructor kwargs.

        Raises:
            RegistryError: If ``config`` has no ``"type"`` key, or the type
                is not registered.

        Example:
            >>> from openbtk.core.registry import GUARDRAIL_REGISTRY
            >>> GUARDRAIL_REGISTRY.create_from_config(
            ...     {"not_type": "nope"}
            ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            openbtk.core.errors.RegistryError: ...must have a 'type' key
        """
        if "type" not in config:
            raise RegistryError(
                f"Registry {self._category!r}: config dict must have a 'type' key.",
                # Only the keys, never the values -- params may legitimately
                # hold operational config; there is no reason for an error
                # about a MISSING key to echo the rest of it back.
                context={"registry": self._category, "config_keys": sorted(config)},
            )
        key = config["type"]
        params = config.get("params", {})
        return self.create(key, **params)

    def list_keys(self) -> list[str]:
        """Return every registered key and alias, sorted."""
        return sorted(self._items)

    def is_registered(self, key: str) -> bool:
        """Return whether ``key`` (or an alias of it) is registered."""
        return key in self._items

    def describe(self, key: str) -> ComponentInfo:
        """Return an introspective summary of the class registered at ``key``.

        Raises:
            RegistryError: If ``key`` is not registered.
        """
        cls = self.get(key)
        return ComponentInfo(
            registry_key=key,
            class_name=cls.__name__,
            module=cls.__module__,
            docstring=inspect.getdoc(cls),
            signature=str(inspect.signature(cls)),
        )

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __repr__(self) -> str:
        return f"Registry(category={self._category!r}, keys={self.list_keys()})"


# ---------------------------------------------------------------------------
# Global registry instances -- one per category. Import these; never create
# a new Registry outside this module.
# ---------------------------------------------------------------------------

# Each construction below deliberately passes an ABSTRACT base class as
# `base_type` -- it is used only for issubclass() checks in Registry.register,
# never instantiated. mypy's type-abstract check assumes any `type[T]` value
# will be called; that assumption does not hold for this use, so every line
# below carries a scoped ignore rather than a blanket one. Verified this is
# the correct suppression point (not on Registry.__init__ itself) empirically.
LOADER_REGISTRY: Registry[BaseLoader[Any, Any]] = Registry(
    "loader",
    BaseLoader,  # type: ignore[type-abstract]
)
PREPROCESSOR_REGISTRY: Registry[BasePreprocessor[Any]] = Registry(
    "preprocessor",
    BasePreprocessor,  # type: ignore[type-abstract]
)
CHUNKER_REGISTRY: Registry[BaseChunker[Any, Any]] = Registry(
    "chunker",
    BaseChunker,  # type: ignore[type-abstract]
)
SEGMENTER_REGISTRY: Registry[BaseSegmenter[Any, Any]] = Registry(
    "segmenter",
    BaseSegmenter,  # type: ignore[type-abstract]
)
FEATURE_EXTRACTOR_REGISTRY: Registry[BaseFeatureExtractor[Any]] = Registry(
    "feature_extractor",
    BaseFeatureExtractor,  # type: ignore[type-abstract]
)
EMBEDDING_REGISTRY: Registry[BaseEmbeddingProvider] = Registry(
    "embedding",
    BaseEmbeddingProvider,  # type: ignore[type-abstract]
)
LLM_REGISTRY: Registry[BaseLLMProvider] = Registry(
    "llm",
    BaseLLMProvider,  # type: ignore[type-abstract]
)
VECTORSTORE_REGISTRY: Registry[BaseVectorStore] = Registry(
    "vectorstore",
    BaseVectorStore,  # type: ignore[type-abstract]
)
RERANKER_REGISTRY: Registry[BaseReranker] = Registry(
    "reranker",
    BaseReranker,  # type: ignore[type-abstract]
)
GUARDRAIL_REGISTRY: Registry[BaseGuardrail] = Registry(
    "guardrail",
    BaseGuardrail,  # type: ignore[type-abstract]
)
DATASET_REGISTRY: Registry[BaseDatasetAdapter] = Registry(
    "dataset",
    BaseDatasetAdapter,  # type: ignore[type-abstract]
)
TERMINOLOGY_REGISTRY: Registry[BaseTerminologyService] = Registry(
    "terminology",
    BaseTerminologyService,  # type: ignore[type-abstract]
)

_ALL_REGISTRIES: dict[str, Registry[Any]] = {
    "loader": LOADER_REGISTRY,
    "preprocessor": PREPROCESSOR_REGISTRY,
    "chunker": CHUNKER_REGISTRY,
    "segmenter": SEGMENTER_REGISTRY,
    "feature_extractor": FEATURE_EXTRACTOR_REGISTRY,
    "embedding": EMBEDDING_REGISTRY,
    "llm": LLM_REGISTRY,
    "vectorstore": VECTORSTORE_REGISTRY,
    "reranker": RERANKER_REGISTRY,
    "guardrail": GUARDRAIL_REGISTRY,
    "dataset": DATASET_REGISTRY,
    "terminology": TERMINOLOGY_REGISTRY,
}


def get_registry(category: str) -> Registry[Any]:
    """Return the global registry for ``category``.

    Raises:
        RegistryError: If ``category`` is not a known registry -- including
            ``"finetuner"``, which is reserved in the key grammar but has no
            backing base class yet.
    """
    if category not in _ALL_REGISTRIES:
        raise RegistryError(
            f"Unknown registry category {category!r}. "
            f"Available: {sorted(_ALL_REGISTRIES)}",
            context={"category": category},
        )
    return _ALL_REGISTRIES[category]

"""
openbtk provider and component registry.

The Registry[T] class enables config-driven instantiation of any framework
component. Global registry instances are created here for each category.
Components register themselves via the @REGISTRY.register("key") decorator.
"""
from __future__ import annotations

from typing import Any, Callable, Generic, Type, TypeVar

import structlog

from openbtk.core.errors import RegistryError

log = structlog.get_logger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """A namespaced, type-safe registry of component classes.

    Usage — registration (in component module):
        @LOADER_REGISTRY.register("clinical_text.mimic_notes")
        class MIMICNotesLoader(BaseLoader):
            ...

    Usage — instantiation (in pipeline or user code):
        loader = LOADER_REGISTRY.create(
            "clinical_text.mimic_notes", path="/data/notes.csv"
        )
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._items: dict[str, Type[T]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator that registers a class under the given key.

        Args:
            key: Dot-separated registry key, e.g. "clinical_text.section_aware".
                Keys are permanent once set — treat as public API.

        Returns:
            The unmodified class (transparent decorator).

        Raises:
            RegistryError: If the key is already registered.
        """
        def decorator(cls: Type[T]) -> Type[T]:
            if key in self._items:
                raise RegistryError(
                    f"Registry '{self._name}': key '{key}' already registered "
                    f"by {self._items[key].__qualname__}",
                    context={"registry": self._name, "key": key},
                )
            self._items[key] = cls
            log.debug(
                "registry.register",
                registry=self._name,
                key=key,
                cls=cls.__qualname__,
            )
            return cls
        return decorator

    def register_alias(self, alias: str, existing_key: str) -> None:
        """Register an alias for an existing key (for deprecation transitions).

        Args:
            alias: New key that should point to the same class.
            existing_key: The original registered key.

        Raises:
            RegistryError: If existing_key is not registered.
        """
        if existing_key not in self._items:
            raise RegistryError(
                f"Registry '{self._name}': cannot alias '{alias}' → "
                f"'{existing_key}' because '{existing_key}' is not registered.",
                context={"registry": self._name, "existing_key": existing_key},
            )
        self._items[alias] = self._items[existing_key]
        log.debug(
            "registry.alias",
            registry=self._name,
            alias=alias,
            existing_key=existing_key,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, key: str) -> Type[T]:
        """Return the registered class for the given key.

        Args:
            key: The registration key.

        Returns:
            The registered class (not an instance).

        Raises:
            RegistryError: If key is not found.
        """
        if key not in self._items:
            raise RegistryError(
                f"Registry '{self._name}': key '{key}' not found. "
                f"Available keys: {self.list_keys()}",
                context={"registry": self._name, "key": key},
            )
        return self._items[key]

    def create(self, key: str, **kwargs: Any) -> T:
        """Instantiate the registered class with the given kwargs.

        Args:
            key: The registration key.
            **kwargs: Constructor keyword arguments for the registered class.

        Returns:
            A new instance of the registered class.

        Raises:
            RegistryError: If key is not found.
            Any exception raised by the class constructor.
        """
        cls = self.get(key)
        log.debug("registry.create", registry=self._name, key=key)
        return cls(**kwargs)

    def create_from_config(self, config: dict[str, Any]) -> T:
        """Instantiate from a config dict with a 'type' key.

        Args:
            config: Dict with required 'type' key (registry key) and
                optional 'params' dict of constructor kwargs.

        Returns:
            A new instance of the registered class.

        Example:
            loader = LOADER_REGISTRY.create_from_config({
                "type": "clinical_text.mimic_notes",
                "params": {"path": "/data/notes.csv"}
            })
        """
        if "type" not in config:
            raise RegistryError(
                f"Registry '{self._name}': config dict must have a 'type' key.",
                context={"registry": self._name, "config": config},
            )
        key = config["type"]
        params = config.get("params", {})
        return self.create(key, **params)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_keys(self) -> list[str]:
        """Return sorted list of all registered keys."""
        return sorted(self._items)

    def is_registered(self, key: str) -> bool:
        """Return True if key is registered."""
        return key in self._items

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, keys={self.list_keys()})"


# ---------------------------------------------------------------------------
# Global registry instances — one per component category
# ---------------------------------------------------------------------------
# Import these in component modules; never create new registries outside core.

from openbtk.core.base import (  # noqa: E402 — after Registry definition
    BaseChunker,
    BaseDatasetAdapter,
    BaseEmbeddingProvider,
    BaseFeatureExtractor,
    BaseFineTuner,
    BaseGuardrail,
    BaseLoader,
    BaseLLMProvider,
    BasePreprocessor,
    BaseSegmenter,
    BaseVectorStore,
)

LOADER_REGISTRY: Registry[BaseLoader] = Registry("loader")
PREPROCESSOR_REGISTRY: Registry[BasePreprocessor] = Registry("preprocessor")
CHUNKER_REGISTRY: Registry[BaseChunker] = Registry("chunker")
SEGMENTER_REGISTRY: Registry[BaseSegmenter] = Registry("segmenter")
FEATURE_EXTRACTOR_REGISTRY: Registry[BaseFeatureExtractor] = Registry("feature_extractor")
EMBEDDING_REGISTRY: Registry[BaseEmbeddingProvider] = Registry("embedding")
LLM_REGISTRY: Registry[BaseLLMProvider] = Registry("llm")
VECTORSTORE_REGISTRY: Registry[BaseVectorStore] = Registry("vectorstore")
GUARDRAIL_REGISTRY: Registry[BaseGuardrail] = Registry("guardrail")
DATASET_REGISTRY: Registry[BaseDatasetAdapter] = Registry("dataset")
FINETUNER_REGISTRY: Registry[BaseFineTuner] = Registry("finetuner")

# Convenience mapping for config-driven lookup by category string
_ALL_REGISTRIES: dict[str, Registry] = {
    "loader": LOADER_REGISTRY,
    "preprocessor": PREPROCESSOR_REGISTRY,
    "chunker": CHUNKER_REGISTRY,
    "segmenter": SEGMENTER_REGISTRY,
    "feature_extractor": FEATURE_EXTRACTOR_REGISTRY,
    "embedding": EMBEDDING_REGISTRY,
    "llm": LLM_REGISTRY,
    "vectorstore": VECTORSTORE_REGISTRY,
    "guardrail": GUARDRAIL_REGISTRY,
    "dataset": DATASET_REGISTRY,
    "finetuner": FINETUNER_REGISTRY,
}


def get_registry(category: str) -> Registry:
    """Return the registry for the given category name.

    Args:
        category: One of the registry category names (e.g., "loader",
            "embedding", "llm").

    Raises:
        RegistryError: If category is not a known registry.
    """
    if category not in _ALL_REGISTRIES:
        raise RegistryError(
            f"Unknown registry category '{category}'. "
            f"Available: {sorted(_ALL_REGISTRIES)}",
            context={"category": category},
        )
    return _ALL_REGISTRIES[category]

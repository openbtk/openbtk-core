"""OpenBTK abstract base classes -- every extension point, in one file.

Read this file to understand every way OpenBTK can be extended. Third-party
modalities, providers and guardrails all subclass something defined here and
register it with a category registry (openbtk.core.registry).

Design decisions worth knowing before extending these (docs/04_API_DESIGN.md
section 3, ADR-0007):

  * **ABCs, not Protocols.** A registry instantiates classes from string keys
    in config; a missing method must fail loudly at construction, not with a
    deep AttributeError mid-pipeline after PHI has already been loaded.
  * **Iterator returns are load-bearing, not decorative** (ADR-0004).
    ``BaseLoader.load()`` and ``BaseChunker.chunk()`` return ``Iterator``, not
    ``list`` -- a single record can produce thousands of chunks, and memory
    must stay O(batch), not O(corpus). Returning a list satisfies mypy and
    breaks the memory guarantee; the contract suite (tests/contract/) checks
    this behaviourally.
  * **``BaseGuardrail.check()`` never raises on a failed check.** A guardrail
    that raises makes composition impossible. ``GuardrailViolation`` is
    raised by the pipeline layer, after inspecting a ``GuardrailResult`` --
    never by the guardrail itself.
  * **``sends_data_offsite`` is enforced, not advisory.** Under
    ``policy.allow_offsite_providers: false`` (the pipeline default),
    constructing a provider with this set to ``True`` raises ``PolicyError``.
    That enforcement lives in ``Registry.create``/``create_from_config``
    (openbtk.core.registry), not here; this module only declares the
    contract. A component with no ``sends_data_offsite`` attribute at all
    (anything outside ``BaseEmbeddingProvider``/``BaseLLMProvider``) is
    unaffected -- the check is ``getattr(cls, "sends_data_offsite", False)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel

from openbtk.core.provenance import ComponentProvenance, ModelIdentity

if TYPE_CHECKING:
    # Annotation-only in this file -- base.py defines no Pydantic model
    # fields (unlike core/provenance.py, where an identical-looking import
    # had to stay eager because Pydantic resolves field types at
    # class-definition time). Verified: no runtime numpy/schema usage exists
    # here beyond type annotations, so `from __future__ import annotations`
    # makes deferring these safe.
    from collections.abc import Iterable, Iterator

    import numpy as np
    from numpy.typing import NDArray

    from openbtk.core.schemas import (
        CodeSystem,
        Concept,
        GuardrailResult,
        LLMResponse,
        Message,
        SearchResult,
    )

InputT = TypeVar("InputT")
RecordT = TypeVar("RecordT", bound=BaseModel)
ChunkT = TypeVar("ChunkT", bound=BaseModel)


def _openbtk_version() -> str:
    """Resolve the installed openbtk version for provenance records.

    Deliberately independent of ``openbtk.__version__`` rather than importing
    it: ``openbtk/__init__.py`` will (once core/registry.py lands) import from
    ``openbtk.core.registry``, which imports this module to validate
    registered types against their base classes. Importing the top-level
    ``openbtk`` package from here would close that into a cycle. Duplicating
    this small, resilient lookup is cheaper than untangling that later.
    """
    try:
        return version("openbtk")
    except PackageNotFoundError:
        return "0.0.0.dev0+unknown"


class Component(ABC):  # noqa: B024 -- intentionally abstract-in-spirit only;
    # every concrete role subclass below adds real @abstractmethods. See
    # docs/04_API_DESIGN.md section 3, which defines Component this way.
    """Root of every OpenBTK component. Supplies identity and provenance.

    Subclasses are never instantiated directly through this class -- always
    through one of the role base classes below, via a registry.
    """

    registry_key: ClassVar[str] = ""
    """Set by the registry's ``register()`` decorator at class-definition
    time. Empty string is the "not yet registered" sentinel -- a class used
    only in tests without going through the registry will report this
    rather than raising, since not every use of a component needs a key.
    """

    def provenance(self) -> ComponentProvenance:
        """Identity and configuration of this instance, for a run manifest.

        The default here is deliberately conservative: an empty ``config``
        dict, no ``model_identity``. It does NOT attempt to introspect
        ``self.__dict__`` for constructor arguments -- a generic
        introspection default risks capturing something that is not
        JSON-serialisable (a loaded model object, an open file handle) or,
        worse, something that should never reach a manifest at all. PHI
        safety is the project's central claim; a clever default that
        occasionally leaks is a worse failure mode than a plain default that
        never does.

        Override this in any component whose configuration or wrapped model
        is worth recording -- which is the expected, normal case for
        anything beyond the simplest components.

        Returns:
            A ``ComponentProvenance`` with this component's registry key,
            class name and package version populated, empty config, and no
            model identity.
        """
        return ComponentProvenance(
            registry_key=self.registry_key,
            class_name=type(self).__name__,
            package_version=_openbtk_version(),
            config={},
            model_identity=None,
        )


class BaseLoader(Component, Generic[InputT, RecordT]):
    """Load raw data from a source into typed domain records.

    Thin wrappers around format-specific libraries (pydicom, wfdb,
    fhir.resources, ...). Loaders stream: they yield records one at a time so
    a corpus larger than memory can still be processed.
    """

    @abstractmethod
    def load(self, source: InputT) -> Iterator[RecordT]:
        """Yield records lazily from ``source``.

        MUST yield the first record without first consuming the whole
        source -- this is what makes the memory guarantee real rather than
        aspirational, and the contract suite checks it behaviourally (first
        record arrives fast even from a source too large to read fully).

        Args:
            source: Source identifier -- file path, directory, URL, or
                connection string, depending on the implementation.

        Yields:
            Typed domain records, in a deterministic order.

        Raises:
            LoaderError: On any I/O or format-parsing failure.
        """

    def load_all(self, source: InputT) -> list[RecordT]:
        """Materialise every record from ``source`` into a list.

        Warning:
            Memory hazard -- O(corpus), not O(batch). Never used in
            OpenBTK's own code, docs, or examples (ADR-0004; enforced by a
            grep gate in CI). Provided only for interactive/notebook use on
            small sources.
        """
        return list(self.load(source))


class BasePreprocessor(Component, Generic[RecordT]):
    """Transform a loaded record into a cleaner or more enriched one.

    Preprocessors handle de-identification, normalisation, enrichment and
    quality checks. They return the same record type and must be stateless
    per call -- the same input always produces the same output.
    """

    @abstractmethod
    def process(self, record: RecordT) -> RecordT:
        """Return a processed record of the same type.

        Args:
            record: A loaded (or previously processed) domain record.

        Returns:
            A record of the same type, cleaned or enriched.

        Raises:
            ProcessingError: On any preprocessing failure.
        """

    def process_stream(self, records: Iterable[RecordT]) -> Iterator[RecordT]:
        """Lazily process a stream of records.

        Default implementation applies :meth:`process` one record at a time.
        Override for batched efficiency (e.g. a model that processes in
        batches of N more cheaply than N separate calls).
        """
        return (self.process(r) for r in records)


class BaseChunker(Component, Generic[RecordT, ChunkT]):
    """Split a record into smaller, retrievable chunks.

    Returns an ``Iterator``, not a ``list`` (ADR-0004) -- a single large
    record can produce thousands of chunks, and the executor must be able to
    stream them through embedding and indexing without holding them all in
    memory at once.
    """

    @abstractmethod
    def chunk(self, record: RecordT) -> Iterator[ChunkT]:
        """Yield ordered chunks lazily.

        Args:
            record: A preprocessed domain record.

        Yields:
            Chunks in document order. Yielding nothing is valid (e.g. a
            record below the minimum chunkable length).

        Raises:
            ProcessingError: On chunking failure.
        """


class BaseSegmenter(Component, Generic[RecordT, ChunkT]):
    """Non-text equivalent of :class:`BaseChunker`: patches, windows, clips.

    Semantically identical contract to ``BaseChunker`` -- a separate name is
    used for imaging/signal/video modalities where "segment" reads more
    naturally than "chunk."
    """

    @abstractmethod
    def segment(self, record: RecordT) -> Iterator[ChunkT]:
        """Yield ordered segments lazily. See :meth:`BaseChunker.chunk`."""


class BaseFeatureExtractor(Component, Generic[ChunkT]):
    """Extract named scalar features from a chunk (biosignals, audio)."""

    @abstractmethod
    def extract(self, chunk: ChunkT) -> dict[str, float]:
        """Extract scalar features from a chunk.

        Args:
            chunk: A chunk or segment of a domain record.

        Returns:
            Feature name to value.

        Raises:
            ProcessingError: On extraction failure.
        """


class BaseEmbeddingProvider(Component):
    """Provide text (or modality-specific) embeddings.

    Implementations wrap PubMedBERT, BioBERT, OpenAI, BiomedCLIP and similar.
    All return float32 arrays so downstream vector stores have one dtype to
    plan around.
    """

    sends_data_offsite: ClassVar[bool]
    """Whether embedding with this provider sends data to a third party.
    Enforced by the pipeline layer under a local-only policy (FR-V-07).
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a list of texts.

        Args:
            texts: Strings to embed. Must be non-empty.

        Returns:
            Array of shape ``(len(texts), self.dimension)``, dtype float32.

        Raises:
            ProviderError: On model inference or API failure.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the vectors this provider produces."""

    @property
    def batch_size(self) -> int:
        """Default batch size for streaming embedding. Override to tune."""
        return 32

    def model_identity(self) -> ModelIdentity:
        """The specific model this provider wraps.

        Not abstract, so a lightweight test double can be embedded in the
        contract suite without declaring one -- but calling this without an
        override is a programming error for any real provider, hence the
        explicit failure rather than a fabricated placeholder identity.

        Raises:
            NotImplementedError: Unless overridden.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override model_identity() to report "
            "the specific model it wraps."
        )

    def embed_one(self, text: str) -> NDArray[np.float32]:
        """Embed a single text. Returns a 1D array of shape (dimension,)."""
        # numpy's __getitem__ stubs return Any for a general index -- too
        # polymorphic to narrow statically. embed()'s own contract guarantees
        # shape (1, dimension) here, so row 0 genuinely is NDArray[np.float32];
        # this cast documents that assumption rather than suppressing the
        # check silently.
        return cast("NDArray[np.float32]", self.embed([text])[0])


class BaseLLMProvider(Component):
    """Generate text from a language model.

    Implementations wrap OpenAI, Anthropic, local HuggingFace/vLLM, and
    biomedical-tuned model presets (Meditron, MedGemma, OpenBioLLM).
    """

    sends_data_offsite: ClassVar[bool]
    """Whether calling this provider sends data to a third party. Enforced
    by the pipeline layer under a local-only policy (FR-V-07)."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion for a single prompt string.

        Args:
            prompt: The input prompt.
            **kwargs: Provider-specific generation parameters (temperature,
                max_tokens, stop sequences, ...).

        Raises:
            ProviderError: On API or model failure.
        """

    @abstractmethod
    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream a completion token-by-token (or chunk-by-chunk).

        Raises:
            ProviderError: On API or model failure.
        """

    @abstractmethod
    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        """Generate a response to a multi-turn conversation.

        Args:
            messages: The conversation so far, oldest first.
            **kwargs: Provider-specific generation parameters.

        Raises:
            ProviderError: On API or model failure.
        """

    def structured(
        self, messages: list[Message], schema: type[BaseModel], **kwargs: Any
    ) -> BaseModel:
        """Generate output validated against a Pydantic schema.

        Default implementation asks the underlying model for JSON via
        :meth:`chat` and validates the response text against ``schema``.
        This is a naive fallback, not the intended steady state: providers
        with native structured-output support (OpenAI's ``response_format``,
        Anthropic's tool-use-based extraction) should override this for
        reliability, since asking a model to return JSON in a plain prompt
        is the least reliable way to get it.

        Raises:
            ProviderError: On API or model failure.
            pydantic.ValidationError: If the response text does not validate
                against ``schema``.
        """
        response = self.chat(messages, **kwargs)
        return schema.model_validate_json(response.text)

    def model_identity(self) -> ModelIdentity:
        """The specific model this provider wraps.

        See :meth:`BaseEmbeddingProvider.model_identity` for why this is not
        abstract but still requires an override to use meaningfully.

        Raises:
            NotImplementedError: Unless overridden.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override model_identity() to report "
            "the specific model it wraps."
        )


class BaseVectorStore(Component):
    """Store and retrieve embedding vectors.

    Implementations wrap FAISS, Chroma, Qdrant and similar.
    """

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: NDArray[np.float32],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Insert or update vectors with associated metadata.

        Args:
            ids: Unique string identifiers, one per vector.
            vectors: Array of shape ``(len(ids), dimension)``.
            metadata: One metadata dict per vector.

        Raises:
            RetrievalError: On storage failure.
        """

    @abstractmethod
    def query(
        self,
        vector: NDArray[np.float32],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query for nearest neighbours.

        Args:
            vector: Query vector of shape ``(dimension,)``.
            top_k: Number of results to return.
            filter: Optional metadata filter, store-specific in semantics.

        Returns:
            Up to ``top_k`` results, ordered by decreasing relevance.

        Raises:
            RetrievalError: On query failure.
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete vectors by id.

        Raises:
            RetrievalError: On deletion failure.
        """

    def persist(self, path: str) -> None:
        """Persist this store to disk.

        Not every store supports persistence (an in-memory test double,
        for instance) -- the default raises rather than silently no-op'ing,
        so a caller who expects persistence to have happened finds out
        immediately.

        Raises:
            NotImplementedError: Unless overridden.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support persist().")

    @classmethod
    def load(cls, path: str) -> BaseVectorStore:
        """Load a previously persisted store from disk.

        Raises:
            NotImplementedError: Unless overridden.
        """
        raise NotImplementedError(f"{cls.__name__} does not support load().")


class BaseGuardrail(Component):
    """Check a payload for a policy violation.

    Used for PHI detection, clinical hallucination checks, terminology
    validity, and signal-quality thresholds.
    """

    @abstractmethod
    def check(self, payload: Any) -> GuardrailResult:
        """Check a payload and return a structured result.

        Never raises for a failed check -- always returns a result with
        ``passed=False``. The pipeline layer, not this method, decides
        whether a ``BLOCK`` severity should raise ``GuardrailViolation``.

        Args:
            payload: The item to check -- raw text, generated text, a
                record, or any domain object, depending on the guardrail.

        Returns:
            A ``GuardrailResult`` describing the outcome.
        """


class BaseReranker(Component):
    """Reorder search results by a criterion other than the original score.

    Implementations include concept-overlap reranking (shared UMLS CUIs
    between query and result) and cross-encoder reranking.
    """

    @abstractmethod
    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """Reorder (and optionally truncate) a list of search results.

        Args:
            query: The original query string.
            results: Results to reorder, in their original order.
            top_k: If given, return at most this many results.

        Returns:
            Results reordered by this reranker's criterion.
        """


class BaseDatasetAdapter(Component):
    """Load a known biomedical dataset by name.

    Handles credentialed access, local caching, and synthetic data
    generation (Synthea and similar). Never auto-downloads restricted data.
    """

    @abstractmethod
    def load(self, **kwargs: Any) -> Any:
        """Load the dataset.

        Args:
            **kwargs: Dataset-specific parameters (split, subset, path,
                credentials, filters, ...).

        Returns:
            Dataset-appropriate return type -- typically a list of records,
            a DataFrame, or a HuggingFace ``Dataset``.

        Raises:
            DatasetError: On access failure (missing credentials, bad path,
                network error).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""

    @property
    @abstractmethod
    def license(self) -> str:
        """SPDX identifier, or an access-request URL for restricted data."""

    @property
    def requires_credentials(self) -> bool:
        """Whether this dataset requires credentialed access."""
        return False


class BaseTerminologyService(Component):
    """Resolve, validate and map codes across clinical terminology systems.

    Implementations back onto UMLS, a locally supplied vocabulary release,
    or a small bundled permissively-licensed subset -- never a bundled
    restricted vocabulary.
    """

    @abstractmethod
    def resolve(self, code: str, system: CodeSystem) -> Concept | None:
        """Resolve a code to its concept, if known.

        Raises:
            TerminologyError: If the backend is unavailable or unlicensed.
        """

    @abstractmethod
    def validate(self, code: str, system: CodeSystem) -> bool:
        """Return whether ``code`` exists in ``system``.

        Raises:
            TerminologyError: If the backend is unavailable or unlicensed.
        """

    @abstractmethod
    def map(
        self, code: str, from_system: CodeSystem, to_system: CodeSystem
    ) -> list[Concept]:
        """Map a code from one terminology system to another, if a
        crosswalk exists.

        Returns:
            Zero or more equivalent concepts in ``to_system``.

        Raises:
            TerminologyError: If the backend is unavailable or unlicensed.
        """

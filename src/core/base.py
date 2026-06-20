"""
openbtk abstract base classes.

Every component in every modality implements one of these base classes.
The orchestration layer works with base class interfaces only — never
with concrete implementations directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar

import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    pass

# Generic type variables
InputT = TypeVar("InputT")          # raw input source type (str path, bytes, etc.)
RecordT = TypeVar("RecordT", bound=BaseModel)   # domain record (ClinicalTextRecord, etc.)
ChunkT = TypeVar("ChunkT", bound=BaseModel)     # chunk/segment (ClinicalTextChunk, etc.)
OutputT = TypeVar("OutputT")


# ---------------------------------------------------------------------------
# Data pipeline base classes
# ---------------------------------------------------------------------------

class BaseLoader(ABC, Generic[InputT, RecordT]):
    """Load raw data from a source into typed domain records.

    Each modality provides one or more Loader implementations. Loaders are
    thin wrappers around format-specific libraries (pydicom, wfdb, fhirclient,
    etc.) and yield records one at a time to support streaming large datasets.
    """

    @abstractmethod
    def load(self, source: InputT) -> Iterable[RecordT]:
        """Load records from source.

        Args:
            source: Source identifier — file path, directory, URL, or database
                connection string depending on implementation.

        Yields:
            Typed domain records one at a time.

        Raises:
            LoaderError: On any I/O or format parsing failure.
        """

    def load_all(self, source: InputT) -> list[RecordT]:
        """Convenience: load all records into memory as a list."""
        return list(self.load(source))


class BasePreprocessor(ABC, Generic[RecordT]):
    """Transform a raw loaded record into a clean, normalized record.

    Preprocessors handle: de-identification, normalization, quality checks,
    format conversion. They return the same record type (possibly enriched)
    and must be stateless per call.
    """

    @abstractmethod
    def process(self, record: RecordT) -> RecordT:
        """Process and return a cleaned/normalized record.

        Args:
            record: A loaded domain record.

        Returns:
            A processed record of the same type, enriched or cleaned.

        Raises:
            ProcessingError: On any preprocessing failure.
        """

    def process_batch(self, records: Iterable[RecordT]) -> list[RecordT]:
        """Process a batch of records. Override for batched efficiency."""
        return [self.process(r) for r in records]


class BaseChunker(ABC, Generic[RecordT, ChunkT]):
    """Split a record into smaller chunks/segments for embedding or retrieval.

    Text modalities produce text chunks; imaging produces patches; biosignals
    produce windows/beats; video produces clips. The chunk type carries
    provenance back to the parent record.
    """

    @abstractmethod
    def chunk(self, record: RecordT) -> list[ChunkT]:
        """Split a record into ordered chunks.

        Args:
            record: A preprocessed domain record.

        Returns:
            An ordered list of chunks. Empty list is valid (e.g., record
            below minimum length).

        Raises:
            ProcessingError: On chunking failure.
        """


class BaseSegmenter(ABC, Generic[RecordT, ChunkT]):
    """Alias base for non-text modalities (imaging patches, signal windows).

    Semantically equivalent to BaseChunker; use this name for imaging/signal
    modalities for clarity. Both share the same interface.
    """

    @abstractmethod
    def segment(self, record: RecordT) -> list[ChunkT]:
        """Segment a record into pieces.

        Args:
            record: A preprocessed domain record.

        Returns:
            An ordered list of segments.

        Raises:
            ProcessingError: On segmentation failure.
        """


class BaseFeatureExtractor(ABC, Generic[ChunkT]):
    """Extract numerical features from a chunk (primarily for biosignals/audio)."""

    @abstractmethod
    def extract(self, chunk: ChunkT) -> dict[str, float]:
        """Extract scalar features from a chunk.

        Args:
            chunk: A chunk/segment of a domain record.

        Returns:
            Dictionary of feature_name → value.

        Raises:
            ProcessingError: On extraction failure.
        """


# ---------------------------------------------------------------------------
# Provider base classes (LLM, embedding, vector store)
# ---------------------------------------------------------------------------

class BaseEmbeddingProvider(ABC):
    """Provide text (or modality-specific) embeddings.

    Implementations wrap: PubMedBERT, BioBERT, ClinicalBERT, OpenAI,
    Cohere, BiomedCLIP, gene2vec, etc. All return float32 numpy arrays.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a 2D float32 array.

        Args:
            texts: List of strings to embed. Must be non-empty.

        Returns:
            Array of shape (len(texts), self.dimension), dtype float32.

        Raises:
            EmbeddingError: On model inference failure or API error.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of embedding vectors produced by this provider."""

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns 1D array of shape (dimension,)."""
        return self.embed([text])[0]


class BaseLLMProvider(ABC):
    """Generate text from a language model.

    Implementations wrap: OpenAI, Anthropic, Google, Azure, Bedrock,
    local HuggingFace/vLLM, and biomedical-tuned models (Meditron, BioGPT).
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text completion for the given prompt.

        Args:
            prompt: The input prompt string.
            **kwargs: Provider-specific generation parameters
                (temperature, max_tokens, stop sequences, etc.).

        Returns:
            Generated text string.

        Raises:
            ProviderError: On API/model failure.
        """

    @abstractmethod
    def stream(self, prompt: str, **kwargs: Any) -> Iterable[str]:
        """Stream generated text token-by-token.

        Args:
            prompt: The input prompt string.
            **kwargs: Provider-specific parameters.

        Yields:
            Token/chunk strings as they are generated.

        Raises:
            ProviderError: On API/model failure.
        """

    def generate_with_system(
        self, system: str, user: str, **kwargs: Any
    ) -> str:
        """Generate with a system prompt and user message (chat models).

        Default implementation concatenates; override for chat-native APIs.
        """
        return self.generate(f"System: {system}\n\nUser: {user}", **kwargs)


class BaseVectorStore(ABC):
    """Store and retrieve embedding vectors.

    Implementations wrap: FAISS, Chroma, Pinecone, Weaviate, Milvus.
    """

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Insert or update vectors with metadata.

        Args:
            ids: Unique string IDs, one per vector.
            vectors: Array of shape (len(ids), dimension).
            metadata: List of metadata dicts, one per vector.

        Raises:
            RetrievalError: On storage failure.
        """

    @abstractmethod
    def query(
        self, vector: np.ndarray, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Query for nearest neighbors.

        Args:
            vector: Query vector of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            List of dicts, each with keys: id, score, metadata.

        Raises:
            RetrievalError: On query failure.
        """

    def upsert_one(
        self, id: str, vector: np.ndarray, metadata: dict[str, Any]
    ) -> None:
        """Convenience: upsert a single vector."""
        self.upsert([id], vector.reshape(1, -1), [metadata])


# ---------------------------------------------------------------------------
# Guardrail base class
# ---------------------------------------------------------------------------

class BaseGuardrail(ABC):
    """Check inputs or outputs for policy violations.

    Used for: PHI detection, clinical hallucination checks, code validity,
    signal quality thresholds, de-identification completeness.
    """

    @abstractmethod
    def check(self, payload: Any) -> "GuardrailResult":
        """Check a payload and return a structured result.

        Args:
            payload: The item to check — raw text, generated text, a record,
                or any domain object depending on the guardrail type.

        Returns:
            GuardrailResult with passed/severity/message/details.
            Never raises for a failed check — always returns a result.
            Only raises GuardrailViolation if severity == BLOCK and
            the implementation is configured to raise on block.
        """


# ---------------------------------------------------------------------------
# Dataset adapter base class
# ---------------------------------------------------------------------------

class BaseDatasetAdapter(ABC):
    """Load a known biomedical dataset by name.

    Implementations handle credentialed access, local caching, format
    conversion, and synthetic data generation (Synthea, etc.).
    """

    @abstractmethod
    def load(self, **kwargs: Any) -> Any:
        """Load the dataset.

        Args:
            **kwargs: Dataset-specific parameters (split, subset, path,
                credentials, filters, etc.)

        Returns:
            Dataset in an appropriate format — typically a list of records,
            a pandas DataFrame, or a HuggingFace Dataset.

        Raises:
            DatasetError: On access failure (missing credentials, bad path,
                network error).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""

    @property
    def requires_credentials(self) -> bool:
        """Whether this dataset requires credentialed access (e.g., PhysioNet)."""
        return False

    @property
    def license(self) -> str:
        """SPDX license identifier or access URL for restricted datasets."""
        return "unknown"


# ---------------------------------------------------------------------------
# Fine-tuning base class
# ---------------------------------------------------------------------------

class BaseFineTuner(ABC):
    """Adapter for fine-tuning a pretrained model on biomedical tasks.

    Wraps PEFT/LoRA configuration and HuggingFace Trainer setup.
    Modality-specific subclasses add task heads and data collators.
    """

    @abstractmethod
    def prepare(self, model_name_or_path: str, **kwargs: Any) -> Any:
        """Load and prepare a base model for fine-tuning.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            **kwargs: PEFT config, quantization config, etc.

        Returns:
            A PEFT-wrapped model ready for training.
        """

    @abstractmethod
    def train(
        self,
        model: Any,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run fine-tuning.

        Args:
            model: Prepared model from `prepare()`.
            train_dataset: Training data in model-expected format.
            eval_dataset: Optional evaluation data.
            **kwargs: TrainingArguments overrides.

        Returns:
            Trained model.
        """

    @abstractmethod
    def save(self, model: Any, output_path: str) -> None:
        """Save fine-tuned model (adapter weights) to disk."""

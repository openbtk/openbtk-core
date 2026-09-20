"""``CrossEncoderReranker``: rerank results with a cross-encoder (FR-R-04).

A vector store ranks by *embedding similarity*: the query and each chunk are encoded
separately and compared. A cross-encoder reads the query and one chunk **together**
and scores how well the chunk answers the query, which is slower (one model pass per
pair) and markedly better at ordering a short candidate list. The usual recipe, and
the one :class:`~openbtk.pipelines.rag.RAGPipeline` follows, is to fetch a wider pool
cheaply and let this reorder it down to ``top_k``.

The default model is NCBI's **MedCPT cross-encoder**, a BERT sequence-classification
model with one relevance logit, pinned to the commit it was verified at (the Hub id
and the commit were checked against the Hub, not recalled). Any other single-logit
sequence-classification model works too; name its Hub id and a pinned ``revision``.

**What it reads:** each result's chunk text from ``SearchResult.metadata[text_key]``
(by convention ``"text"``, the same key ``RAGPipeline`` prompts from). A result with
no text cannot be scored, so it is **kept, after every scored result, in its original
order**: a reranker reorders, it never silently drops. If no result has text the model
is not even loaded.

**What it returns:** copies of the results, best first. ``score`` becomes the
cross-encoder's raw logit (comparable *within this call*, not a probability and not
comparable with the original scores), and the original score is kept in
``metadata["retrieval_score"]``.

Models are downloaded and run **locally**; the query and the text never leave the
machine. Loading is lazy (first ``rerank``), so constructing one, validating a config
and checking provenance cost nothing. Needs ``pip install "openbtk[text,llms]"``
(``transformers`` and ``torch``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openbtk.core._lazy import require
from openbtk.core.base import BaseReranker
from openbtk.core.errors import ConfigError, MissingDependencyError, RetrievalError
from openbtk.core.logging import get_logger
from openbtk.core.provenance import ModelIdentity
from openbtk.core.registry import RERANKER_REGISTRY

if TYPE_CHECKING:
    from openbtk.core.provenance import ComponentProvenance
    from openbtk.core.schemas import SearchResult

log = get_logger(__name__)

DEFAULT_MODEL = "ncbi/MedCPT-Cross-Encoder"
DEFAULT_REVISION = (
    "71caf65d4927987813984f54c284405a13fcca49"  # pragma: allowlist secret
)
"""The Hub commit ``DEFAULT_MODEL`` was verified at (its ``main`` on 2026-09-20)."""

_DEFAULT_TEXT_KEY = "text"


@RERANKER_REGISTRY.register("reranker.general.cross_encoder")
class CrossEncoderReranker(BaseReranker):
    """Rerank search results with a local cross-encoder model.

    Args:
        model: A Hugging Face Hub model id (or local path) of a sequence-classification
            model with a single relevance logit. Defaults to MedCPT's cross-encoder.
        revision: The pinned commit of ``model``. Optional only for the default model,
            whose verified commit is built in; for any other model it is required, so a
            run never silently follows a moving branch.
        text_metadata_key: The ``SearchResult.metadata`` key holding each chunk's text.
        batch_size: Query-chunk pairs scored per model pass.
        max_length: Tokens kept per pair (longer pairs are truncated).
        device: A torch device string, ``"cpu"`` by default.

    Example:
        >>> reranker = CrossEncoderReranker()  # nothing is loaded yet
        >>> reranker.provenance().model_identity.name
        'ncbi/MedCPT-Cross-Encoder'
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        revision: str | None = None,
        text_metadata_key: str = _DEFAULT_TEXT_KEY,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        if revision is None:
            if model != DEFAULT_MODEL:
                raise ConfigError(
                    f"A revision is required for model {model!r}: pin a commit "
                    "so a run never follows a moving branch.",
                    context={"component": "reranker.general.cross_encoder"},
                )
            revision = DEFAULT_REVISION
        if batch_size < 1 or max_length < 2:
            raise ConfigError(
                "batch_size must be at least 1 and max_length at least 2.",
                context={"component": "reranker.general.cross_encoder"},
            )
        self._model_name = model
        self._revision = revision
        self._text_key = text_metadata_key
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = device
        self._loaded: tuple[Any, Any, Any] | None = None

    # ------------------------------------------------------------------ loading

    def _load(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            transformers = require("transformers", extra="text")
            torch = require("torch", extra="llms")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._model_name, revision=self._revision
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self._model_name, revision=self._revision
            )
            model.to(self._device)
            model.eval()
            self._loaded = (tokenizer, model, torch)
        return self._loaded

    def _scores(self, query: str, texts: list[str]) -> list[float]:
        tokenizer, model, torch = self._load()
        scores: list[float] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            encoded = tokenizer(
                [query] * len(batch),
                batch,
                truncation=True,
                padding=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self._device) for name, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
            if logits.shape[-1] != 1:
                raise RetrievalError(
                    "The cross-encoder must output one relevance logit per pair; "
                    f"model {self._model_name!r} outputs {int(logits.shape[-1])}.",
                    context={"component": "reranker.general.cross_encoder"},
                )
            scores.extend(float(x) for x in logits.reshape(len(batch)).tolist())
        return scores

    # ---------------------------------------------------------------- reranking

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        if not results:
            return []
        indexed: list[tuple[int, str]] = []
        for i, result in enumerate(results):
            text = result.metadata.get(self._text_key)
            if isinstance(text, str) and text.strip():
                indexed.append((i, text))

        scored: list[SearchResult] = []
        if indexed:
            try:
                values = self._scores(query, [text for _, text in indexed])
            except (RetrievalError, MissingDependencyError):
                raise  # already say what is wrong, and what to install
            except Exception as e:  # torch / transformers raise their own hierarchy
                raise RetrievalError(
                    "Cross-encoder scoring failed.",
                    context={
                        "component": "reranker.general.cross_encoder",
                        "candidates": len(indexed),
                    },
                ) from e
            order = sorted(
                range(len(indexed)), key=lambda j: (-values[j], indexed[j][0])
            )
            for j in order:
                original = results[indexed[j][0]]
                scored.append(
                    original.model_copy(
                        update={
                            "score": values[j],
                            "metadata": {
                                **original.metadata,
                                "retrieval_score": original.score,
                            },
                        }
                    )
                )
        unscored_ids = set(range(len(results))) - {i for i, _ in indexed}
        if unscored_ids:
            log.warning(
                "reranker.cross_encoder.unscored",
                count=len(unscored_ids),
                scored=len(indexed),
            )
        ranked = scored + [results[i] for i in sorted(unscored_ids)]
        return ranked if top_k is None else ranked[:top_k]

    def provenance(self) -> ComponentProvenance:
        return (
            super()
            .provenance()
            .model_copy(
                update={
                    "config": {
                        "model": self._model_name,
                        "text_metadata_key": self._text_key,
                        "batch_size": self._batch_size,
                        "max_length": self._max_length,
                        "device": self._device,
                    },
                    "model_identity": ModelIdentity(
                        name=self._model_name,
                        revision=self._revision,
                        source="huggingface",
                    ),
                }
            )
        )

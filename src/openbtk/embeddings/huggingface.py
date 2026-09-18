"""A local HuggingFace ``transformers`` embedding provider.

Wraps, does not reinvent (docs/09_CODING_STANDARDS.md section 7): one
generic encoder-embedding implementation, parameterised by model/revision/
pooling strategy, serves every current biomedical BERT-family embedding
model (PubMedBERT/BiomedBERT, BioBERT, ClinicalBERT, SapBERT, MedCPT --
see :mod:`openbtk.embeddings.presets`) -- none of them need a bespoke
provider class, the same reasoning task 5.3 already applied to biomedical
LLMs.

``dimension`` is a *required* constructor argument, not derived by loading
the model: loading a model is expensive and this class's own contract
tests (``sends_data_offsite``/``dimension``/``model_identity``/
``provenance``) must stay possible without a real model load or network
access (mirroring how ``HuggingFaceLocalProvider`` requires an explicit
``revision`` rather than resolving one). :meth:`embed` still verifies the
declared dimension against what the model actually produces on first real
use, so a wrong value is caught immediately rather than silently
propagating mismatched vectors into a vector store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from openbtk.core._lazy import require
from openbtk.core.base import BaseEmbeddingProvider
from openbtk.core.errors import ProviderError
from openbtk.core.logging import get_logger
from openbtk.core.provenance import ComponentProvenance, ModelIdentity
from openbtk.core.registry import EMBEDDING_REGISTRY

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = get_logger(__name__)

Pooling = Literal["cls", "mean"]


@EMBEDDING_REGISTRY.register("embedding.general.huggingface")
class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """Embed text with a local HuggingFace ``transformers`` encoder.

    Args:
        model: A HuggingFace Hub model id or local path.
        revision: A pinned commit SHA (never a floating branch name --
            same FR-P-05 reasoning as ``HuggingFaceLocalProvider``).
        dimension: The model's real output width -- see this module's own
            docstring for why this is required rather than derived.
        pooling: How to reduce token-level hidden states to one vector.
            ``"mean"`` (masked mean over the attention mask) is the
            general-purpose default for a plain encoder checkpoint not
            fine-tuned as a sentence embedder. ``"cls"`` (the first
            token's hidden state) is the convention documented by
            SapBERT and MedCPT's own model cards specifically -- see
            :mod:`openbtk.embeddings.presets`, which sets this correctly
            per preset rather than leaving every caller to know it.
        device: Passed to the loaded model's ``.to(...)``.
        max_length: Tokenizer truncation length.

    No I/O happens in ``__init__`` (docs/09_CODING_STANDARDS.md rule 11):
    the tokenizer and model are downloaded/loaded lazily, on first real
    call.
    """

    sends_data_offsite: ClassVar[bool] = False

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        dimension: int,
        pooling: Pooling = "mean",
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        self._model_name = model
        self._revision = revision
        self._dimension_value = dimension
        self._pooling: Pooling = pooling
        self._device = device
        self._max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is None or self._model is None:
            require("torch", extra="llms")  # presence check, see huggingface.py
            # (the LLM one)'s own identical comment for why.
            transformers = require("transformers", extra="text")
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._model_name, revision=self._revision
            )
            model = transformers.AutoModel.from_pretrained(
                self._model_name, revision=self._revision
            )
            self._model = model.to(self._device)
        return self._tokenizer, self._model

    @property
    def dimension(self) -> int:
        return self._dimension_value

    def _pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        if self._pooling == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        tokenizer, model = self._load()
        torch = require("torch", extra="llms")
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)
        try:
            with torch.no_grad():
                output = model(**encoded)
        except Exception as e:
            raise ProviderError(f"Local embedding failed: {e}") from e
        pooled = self._pool(output.last_hidden_state, encoded["attention_mask"])
        vectors: NDArray[np.float32] = pooled.detach().cpu().numpy().astype(np.float32)
        if vectors.shape[1] != self._dimension_value:
            raise ProviderError(
                f"{self._model_name} produced {vectors.shape[1]}-dimensional "
                f"vectors, but this provider was constructed with "
                f"dimension={self._dimension_value}.",
                context={
                    "model": self._model_name,
                    "expected_dimension": self._dimension_value,
                    "actual_dimension": vectors.shape[1],
                },
            )
        return vectors

    def model_identity(self) -> ModelIdentity:
        return ModelIdentity(
            name=self._model_name, revision=self._revision, source="huggingface"
        )

    def provenance(self) -> ComponentProvenance:
        return (
            super()
            .provenance()
            .model_copy(
                update={
                    "config": {
                        "model": self._model_name,
                        "pooling": self._pooling,
                        "device": self._device,
                    },
                    "model_identity": self.model_identity(),
                }
            )
        )

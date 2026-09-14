"""Shared infrastructure for the contract test suite.

The contract suite (docs/07_TEST_CHARTER.md section 3.1) is designed to
parametrize over the LIVE registry (``@pytest.mark.parametrize("key",
LOADER_REGISTRY.list_keys())``), so that any component registered anywhere
-- now or in a future modality -- is automatically enrolled with no opt-out.

But at M1, no concrete modality component exists yet: ``LOADER_REGISTRY``
and every other category registry are empty until ``tests/contract`` itself
populates them. Without something registered, every parametrized contract
class would silently collect zero test cases and report as "passed" --
exactly the looks-done-but-isn't failure this whole project exists to avoid
(CLAUDE.md's "v1 Failure" section).

So this module registers one minimal, genuinely-correct REFERENCE
implementation per base class, under the ``general`` scope, before any
contract test file's ``@pytest.mark.parametrize(..., REGISTRY.list_keys())``
is evaluated. This guarantees:

  * The suite is never vacuous -- it always has at least one real
    implementation to run against, right now.
  * When a real modality component registers later (M3+), it is swept into
    the exact same parametrized suite automatically, with no extra wiring.

Reference implementations are deliberately simple but behaviorally
CORRECT -- they exist to prove the contract is satisfiable and to exercise
every check, not to be interesting.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
)
from openbtk.core.errors import LoaderError
from openbtk.core.registry import (
    CHUNKER_REGISTRY,
    DATASET_REGISTRY,
    EMBEDDING_REGISTRY,
    FEATURE_EXTRACTOR_REGISTRY,
    GUARDRAIL_REGISTRY,
    LLM_REGISTRY,
    LOADER_REGISTRY,
    PREPROCESSOR_REGISTRY,
    RERANKER_REGISTRY,
    SEGMENTER_REGISTRY,
    TERMINOLOGY_REGISTRY,
    VECTORSTORE_REGISTRY,
)
from openbtk.core.schemas import (
    CodeSystem,
    Concept,
    GuardrailResult,
    GuardrailSeverity,
    LLMResponse,
    Message,
    SearchResult,
    TextSpan,
)
from openbtk.deid.recognizers.base import RECOGNIZER_REGISTRY, BaseRecognizer
from openbtk.deid.schemas import Detection, PHICategory

if os.environ.get("OPENBTK_SLOW_TESTS") == "1":
    # openbtk.deid.recognizers.ner is never imported by
    # openbtk.deid.recognizers.__init__ (that module's own docstring: it
    # would force every caller of openbtk.deid to pay for spaCy and a
    # downloaded model). Left unimported, "recognizer.general.ner" simply
    # never exists during collection, so it never reaches the parametrize
    # below no matter which test files happen to import it later --
    # collection order is NOT something to rely on here (verified: tests/
    # contract is collected before tests/unit/deid, so a bare `import
    # openbtk.deid.recognizers.ner` anywhere under tests/unit/deid/ arrives
    # too late to be seen by this module's own parametrize evaluation).
    # Importing it here, gated on the same env var
    # tests/conftest.py's slow-test skip uses, is what makes "no
    # exemptions" (CLAUDE.md rule 12) actually true for a model-backed
    # recognizer once a developer opts in, rather than only true by
    # accident of directory naming.
    from openbtk.deid.recognizers import ner as _ner  # noqa: F401

# Unconditional (unlike the NER import above): importing this package costs
# nothing regardless of which loaders a caller ends up using -- none of its
# three loaders need their optional dependency merely to be *defined*, only
# to actually load (see that package's own __init__.py docstring). Ensures
# PlainTextLoader/JSONLLoader/MIMICNotesLoader are always registered before
# tests/contract/test_loader_contract.py's parametrize evaluates.
from openbtk.data import clinical_text as _clinical_text  # noqa: F401

# ---------------------------------------------------------------------------
# Shared fixture schemas -- deliberately NOT importing modality-specific
# SCHEMAS from any modality module; the contract layer tests base-class
# BEHAVIOUR, not any modality's specific schema. (The modality import right
# above is only for its registration side effect.)
# ---------------------------------------------------------------------------


class FixtureRecord(BaseModel):
    """A minimal record for exercising loader/preprocessor/chunker contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(..., min_length=1)
    text: str = Field(...)


class FixtureChunk(BaseModel):
    """A minimal chunk for exercising chunker/segmenter/feature-extractor contracts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., min_length=1)
    record_id: str = Field(..., min_length=1)
    text: str = Field(...)


def make_exploding_source(first_value: str) -> Iterator[str]:
    """An iterator that yields ``first_value`` once, then raises forever.

    The definitive way to test "does this actually stream" behaviourally,
    per docs/07_TEST_CHARTER.md section 3.1 ("the contract suite asserts
    laziness behaviourally, not structurally"): a loader that internally
    does ``list(source)`` (or otherwise consumes past the first item) hits
    the explosion immediately. A loader that correctly yields one-in,
    one-out never triggers it, because the test only ever asks for the
    first item.
    """

    def _gen() -> Iterator[str]:
        yield first_value
        raise RuntimeError(
            "source was consumed past its first item -- the loader under "
            "test is not streaming lazily (ADR-0004)."
        )

    return _gen()


# ---------------------------------------------------------------------------
# Reference implementations. One per base class, registered under
# "<category>.general.contract_reference".
# ---------------------------------------------------------------------------


@LOADER_REGISTRY.register("loader.general.contract_reference")
class ReferenceLoader(BaseLoader[Iterator[str], FixtureRecord]):
    """Yields one FixtureRecord per line in `source`, lazily.

    Raises LoaderError for a non-iterable source, giving the contract suite
    a genuine, reproducible failure path to test against (docs/07_TEST_CHARTER.md
    section 3.1's "raises_loader_error_on_bad_source" / "error_context_contains_no_phi"
    checks) -- an all-permissive reference implementation with no failure
    mode at all would leave those checks untestable.
    """

    def load(self, source: Iterator[str]) -> Iterator[FixtureRecord]:
        try:
            iterator = iter(source)
        except TypeError as e:
            raise LoaderError(
                "Source is not iterable.",
                context={"modality": "general", "stage": "load"},
            ) from e
        for i, line in enumerate(iterator):
            yield FixtureRecord(record_id=str(i), text=line)


@PREPROCESSOR_REGISTRY.register("preprocessor.general.contract_reference")
class ReferencePreprocessor(BasePreprocessor[FixtureRecord]):
    """Uppercases a record's text. Stateless, deterministic."""

    def process(self, record: FixtureRecord) -> FixtureRecord:
        return FixtureRecord(record_id=record.record_id, text=record.text.upper())


@CHUNKER_REGISTRY.register("chunker.general.contract_reference")
class ReferenceChunker(BaseChunker[FixtureRecord, FixtureChunk]):
    """Splits a record's text on whitespace into one chunk per word."""

    def chunk(self, record: FixtureRecord) -> Iterator[FixtureChunk]:
        for i, word in enumerate(record.text.split()):
            yield FixtureChunk(
                chunk_id=f"{record.record_id}-{i}",
                record_id=record.record_id,
                text=word,
            )


@SEGMENTER_REGISTRY.register("segmenter.general.contract_reference")
class ReferenceSegmenter(BaseSegmenter[FixtureRecord, FixtureChunk]):
    """Identical behaviour to ReferenceChunker -- segment() is chunk()'s
    non-text-modality twin, same contract."""

    def segment(self, record: FixtureRecord) -> Iterator[FixtureChunk]:
        for i, word in enumerate(record.text.split()):
            yield FixtureChunk(
                chunk_id=f"{record.record_id}-{i}",
                record_id=record.record_id,
                text=word,
            )


@FEATURE_EXTRACTOR_REGISTRY.register("feature_extractor.general.contract_reference")
class ReferenceFeatureExtractor(BaseFeatureExtractor[FixtureChunk]):
    """Extracts a single trivial numeric feature: chunk text length."""

    def extract(self, chunk: FixtureChunk) -> dict[str, float]:
        return {"length": float(len(chunk.text))}


@EMBEDDING_REGISTRY.register("embedding.general.contract_reference")
class ReferenceEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, dependency-free embedding: [len(text), 0.0, 0.0]."""

    sends_data_offsite = False

    @property
    def dimension(self) -> int:
        return 3

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        return np.array([[float(len(t)), 0.0, 0.0] for t in texts], dtype=np.float32)


@LLM_REGISTRY.register("llm.general.contract_reference")
class ReferenceLLMProvider(BaseLLMProvider):
    """Echoes its input. No real model, no network, no offsite data flow."""

    sends_data_offsite = False

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(text=f"echo: {prompt}")

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield f"echo: {prompt}"

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        last = messages[-1].content if messages else ""
        return LLMResponse(text=f"echo: {last}")


@VECTORSTORE_REGISTRY.register("vectorstore.general.contract_reference")
class ReferenceVectorStore(BaseVectorStore):
    """A trivial in-memory vector store. No persistence (uses the default
    NotImplementedError from BaseVectorStore -- deliberately not overridden,
    so the base class default itself gets exercised by the contract suite).
    """

    def __init__(self) -> None:
        self._vectors: dict[str, NDArray[np.float32]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        ids: list[str],
        vectors: NDArray[np.float32],
        metadata: list[dict[str, Any]],
    ) -> None:
        for i, vec, meta in zip(ids, vectors, metadata, strict=True):
            self._vectors[i] = vec
            self._metadata[i] = meta

    def query(
        self,
        vector: NDArray[np.float32],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        scored = [
            (i, float(-np.linalg.norm(v - vector))) for i, v in self._vectors.items()
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(id=i, score=score, metadata=self._metadata.get(i, {}))
            for i, score in scored[:top_k]
        ]

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._vectors.pop(i, None)
            self._metadata.pop(i, None)


@GUARDRAIL_REGISTRY.register("guardrail.general.contract_reference")
class ReferenceGuardrail(BaseGuardrail):
    """Fails (BLOCK) if the payload string is empty; passes otherwise.
    Never raises, per the base contract."""

    def check(self, payload: Any) -> GuardrailResult:
        text = str(payload)
        if not text:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.BLOCK,
                guardrail_key=self.registry_key,
                message="Payload is empty.",
            )
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message="OK.",
        )


@RERANKER_REGISTRY.register("reranker.general.contract_reference")
class ReferenceReranker(BaseReranker):
    """Reranks by reversing the input order -- trivial, but genuinely reorders."""

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        reordered = list(reversed(results))
        return reordered if top_k is None else reordered[:top_k]


@DATASET_REGISTRY.register("dataset.general.contract_reference")
class ReferenceDatasetAdapter(BaseDatasetAdapter):
    """Returns a small, synthetic, open-licence dataset."""

    def load(self, **kwargs: Any) -> Any:
        return [FixtureRecord(record_id="0", text="synthetic")]

    @property
    def name(self) -> str:
        return "contract-reference-dataset"

    @property
    def license(self) -> str:
        return "Apache-2.0"


@TERMINOLOGY_REGISTRY.register("terminology.general.contract_reference")
class ReferenceTerminologyService(BaseTerminologyService):
    """A hardcoded, two-entry terminology table."""

    _TABLE: ClassVar[dict[tuple[str, CodeSystem], str]] = {
        ("73211009", CodeSystem.SNOMED): "Diabetes mellitus",
    }

    def resolve(self, code: str, system: CodeSystem) -> Concept | None:
        display = self._TABLE.get((code, system))
        if display is None:
            return None
        return Concept(code=code, system=system, display=display)

    def validate(self, code: str, system: CodeSystem) -> bool:
        return (code, system) in self._TABLE

    def map(
        self, code: str, from_system: CodeSystem, to_system: CodeSystem
    ) -> list[Concept]:
        return []  # no crosswalk in this trivial reference table


@RECOGNIZER_REGISTRY.register("recognizer.general.contract_reference")
class ReferenceRecognizer(BaseRecognizer):
    """Detects any run of 3+ digits, tagged as a low-confidence generic
    identifier. Simple, but genuinely functional: the contract suite needs
    real spans (not an always-empty stub) to verify offsets, confidence
    bounds, and the `method` tag against."""

    method: ClassVar[Literal["rule", "ner", "llm_verifier"]] = "rule"
    _PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\d{3,}")

    def detect(self, text: str) -> list[Detection]:
        return [
            Detection(
                category=PHICategory.OTHER_UNIQUE_IDENTIFIER,
                span=TextSpan(
                    start=m.start(), end=m.end(), label="digits", confidence=0.5
                ),
                confidence=0.5,
                method=self.method,
            )
            for m in self._PATTERN.finditer(text)
        ]


@pytest.fixture
def fixture_record() -> FixtureRecord:
    return FixtureRecord(record_id="rec-1", text="hello world from a fixture")

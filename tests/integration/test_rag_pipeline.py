"""Integration test (task 5.8): a full RAG pipeline, end to end, through
REAL components at every stage -- not test doubles, except the LLM.

**Ingest** (load -> deid -> segment -> chunk) runs through the REAL
``openbtk.pipelines`` executor, the same components and capture-guardrail
pattern already established in
tests/integration/test_clinical_text_pipeline.py (M3 task 3.8) -- that
file's own docstring explains why a guardrail is the legitimate way to
observe chunks in-process, since ``Pipeline.run()`` returns only a
``RunManifest``.

**Indexing and retrieval** then run through :class:`openbtk.pipelines.rag.RAGPipeline`
against REAL, real embeddings (``HuggingFaceEmbeddingProvider`` with a
genuinely tiny, real test checkpoint --
``hf-internal-testing/tiny-random-bert``, the same one
tests/contract/test_embedding_contract.py uses for exactly this reason: a
real model, not a multi-GB one, for a test that needs one loaded), a REAL
``FAISSVectorStore``, and a REAL ``ConceptOverlapReranker``. Both need
their own optional dependency actually installed -- gated the same way
the rest of this project gates on that, not assumed.

The one deliberately-fake piece is the LLM: no real API credentials are
available in this environment, and what this test needs to prove is that
**real chunk-level provenance survives the entire round trip** (a
synthetic note's real ``record_id``/``chunk_id`` come back out the other
end in ``RAGAnswer.sources``) -- a fake, deterministic LLM answering
verifies that without needing a real API call, the same reasoning already
applied throughout tests/unit/llms and tests/unit/embeddings.
"""

from __future__ import annotations

import importlib.util
import re
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from openbtk.core.base import BaseGuardrail, BaseLLMProvider
from openbtk.core.registry import GUARDRAIL_REGISTRY
from openbtk.core.schemas import (
    GuardrailResult,
    GuardrailSeverity,
    LLMResponse,
    Message,
)

# Registers PlainTextLoader/DeidPreprocessor/SectionSegmenter/
# SectionAwareChunker -- this file is collected standalone (no
# tests/contract/conftest.py in this directory), so the import has to
# happen explicitly here, the same as test_clinical_text_pipeline.py does.
from openbtk.data import clinical_text as _clinical_text  # noqa: F401
from openbtk.pipelines import Pipeline, RAGPipeline, Step
from openbtk.retrieval.reranker import ConceptOverlapReranker

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from openbtk.data.clinical_text.schemas import ClinicalTextChunk

_HAS_TORCH_AND_TRANSFORMERS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)
_HAS_FAISS = importlib.util.find_spec("faiss") is not None

pytestmark = pytest.mark.skipif(
    not (_HAS_TORCH_AND_TRANSFORMERS and _HAS_FAISS),
    reason="requires the 'llms' extra (torch) and the 'retrieval' extra (faiss-cpu)",
)

if _HAS_TORCH_AND_TRANSFORMERS:
    from openbtk.embeddings.huggingface import HuggingFaceEmbeddingProvider
if _HAS_FAISS:
    from openbtk.retrieval.faiss import FAISSVectorStore

# Same real, tiny (126K-parameter), non-gated test checkpoint
# tests/contract/test_embedding_contract.py uses -- verified against the
# HuggingFace Hub API, not fabricated.
_TINY_BERT_MODEL = "hf-internal-testing/tiny-random-bert"
_TINY_BERT_SHA = "f171d7baecaf37b5da5a3616d8833b9969753535"  # pragma: allowlist secret
_TINY_BERT_DIMENSION = 32

_SSN_VALUE = "123-45-6789"  # phi-fixture-ok: synthetic, unassigned test value

_DIABETES_NOTE = (
    "Chief Complaint:\n"
    f"Patient SSN: {_SSN_VALUE}. Follow-up for type 2 diabetes mellitus. "
    "Blood glucose remains elevated.\n"
    "Plan:\n"
    "Continue metformin. Recheck A1c in three months.\n"
)
_FRACTURE_NOTE = (
    "Chief Complaint:\n"
    "Patient reports right wrist pain after a fall.\n"
    "Plan:\n"
    "X-ray shows a distal radius fracture. Cast applied, orthopedics follow-up.\n"
)


class _CaptureGuardrail(BaseGuardrail):
    """Records every chunk this pipeline run produces, in-process -- the
    same legitimate use of the guardrail attachment mechanism as
    test_clinical_text_pipeline.py's own, but keeping the whole chunk
    (chunk_id/record_id/text), not just its text, since this test needs
    those to actually index and later verify provenance."""

    captured: ClassVar[list[ClinicalTextChunk]] = []

    def check(self, payload: Any) -> GuardrailResult:
        type(self).captured.append(payload)
        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_key=self.registry_key,
            message="captured for test inspection",
        )

    @classmethod
    def reset(cls) -> None:
        cls.captured = []


GUARDRAIL_REGISTRY.register("guardrail.general.rag_integration_test_capture")(
    _CaptureGuardrail
)


class _FakeLLM(BaseLLMProvider):
    """Deterministic stand-in for a real LLM -- see this module's own
    docstring for why: no real credentials are available here, and a real
    call would test the LLM SDK (already covered in tests/unit/llms),
    not what this test exists to prove (real provenance surviving the
    round trip)."""

    sends_data_offsite = False

    def __init__(self, text: str = "answer") -> None:
        self._text = text
        self.last_prompt: str | None = None

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return self.chat([Message(role="user", content=prompt)], **kwargs)

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield self._text

    def chat(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        self.last_prompt = messages[0].content
        return LLMResponse(text=self._text)


def _lowercase_word_concepts(text: str) -> set[str]:
    """A trivial, dependency-free stand-in for a real UMLS linker --
    lowercased, punctuation-stripped word tokens as "concepts", real
    enough to genuinely discriminate between this test's two notes by
    real word overlap with the query, unlike a naive ``str.split()``
    (which a query's own trailing "?" would silently defeat)."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _write_notes(tmp_path: Path) -> Path:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "diabetes.txt").write_text(_DIABETES_NOTE, encoding="utf-8")
    (notes_dir / "fracture.txt").write_text(_FRACTURE_NOTE, encoding="utf-8")
    return notes_dir


def _ingest_real_chunks(notes_dir: Path) -> list[ClinicalTextChunk]:
    """Runs the REAL load -> deid -> segment -> chunk pipeline and
    returns the real chunks it produced -- the ingest half of this
    integration test."""
    _CaptureGuardrail.reset()
    pipeline = (
        Pipeline("rag-integration-ingest")
        .add(Step("load", "loader.clinical_text.plain_text", path=str(notes_dir)))
        .add(Step("deid", "preprocessor.general.deidentify", mode="redact"))
        .add(Step("segment", "preprocessor.clinical_text.section_segment"))
        .add(Step("chunk", "chunker.clinical_text.section_aware", max_tokens=100))
        .guard("guardrail.general.rag_integration_test_capture", at="after:chunk")
    )
    manifest = pipeline.run()
    assert manifest.status == "success", manifest.error
    return list(_CaptureGuardrail.captured)


def _index_chunks(
    chunks: list[ClinicalTextChunk], embedding: HuggingFaceEmbeddingProvider
) -> FAISSVectorStore:
    """Embeds and upserts every chunk into a real FAISS store, using
    RAGPipeline's own default metadata-key convention (text/record_id/
    chunk_id) -- the indexing half a real application would run once,
    offline, before any question is ever asked.

    Also computes and stores each chunk's own ``"cuis"`` metadata via the
    same ``_lowercase_word_concepts`` extractor the query side uses --
    ConceptOverlapReranker's own docstring is explicit that a result's
    concepts are read from metadata, never re-extracted at rerank time
    (real-world cost reasons), so skipping this step here would leave
    every result's overlap silently 0 and let tiny-random-bert's
    otherwise-meaningless (untrained) embedding distances decide ranking
    by pure chance -- caught by direct reproduction, not assumed away.
    """
    store = FAISSVectorStore(dimension=embedding.dimension)
    vectors = embedding.embed([c.text for c in chunks])
    metadata = [
        {
            "text": c.text,
            "record_id": c.record_id,
            "chunk_id": c.chunk_id,
            "cuis": sorted(_lowercase_word_concepts(c.text)),
        }
        for c in chunks
    ]
    store.upsert([c.chunk_id for c in chunks], vectors, metadata)
    return store


class TestFullRAGPipeline:
    def test_real_provenance_survives_ingest_through_to_the_answer(
        self, tmp_path: Path
    ) -> None:
        notes_dir = _write_notes(tmp_path)
        chunks = _ingest_real_chunks(notes_dir)
        assert len(chunks) >= 2  # at least one chunk per note

        embedding = HuggingFaceEmbeddingProvider(
            model=_TINY_BERT_MODEL,
            revision=_TINY_BERT_SHA,
            dimension=_TINY_BERT_DIMENSION,
        )
        store = _index_chunks(chunks, embedding)
        llm = _FakeLLM("The patient has type 2 diabetes mellitus.")
        # top_k=1 with a word-overlap reranker, not tiny-random-bert's own
        # (untrained, effectively random) embedding distances: "diabetes"
        # appears in exactly one real chunk here, so the reranker's
        # overlap count -- not embedding similarity -- deterministically
        # decides the single retrieved chunk, regardless of how the
        # random-weight model happens to embed either note.
        rag = RAGPipeline(
            embedding=embedding,
            vectorstore=store,
            llm=llm,
            reranker=ConceptOverlapReranker(extract_concepts=_lowercase_word_concepts),
            top_k=1,
        )

        diabetes_record_id = next(
            c.record_id for c in chunks if "diabetes" in c.text.lower()
        )

        answer = rag.ask("diabetes diagnosis")

        # The real point of this test: the SourceRef in the answer must
        # trace back to the exact chunk this run's real ingest step
        # actually produced for the diabetes note -- not a fabricated,
        # mismatched, or arbitrarily-chosen one.
        assert len(answer.sources) == 1
        assert answer.sources[0].record_id == diabetes_record_id
        assert answer.sources[0].chunk_id in {c.chunk_id for c in chunks}

        # The fake LLM's prompt must actually contain the retrieved
        # chunk's real (de-identified) text -- proving generation was
        # genuinely grounded, not that a source was attached without
        # being used.
        assert llm.last_prompt is not None
        assert "diabetes" in llm.last_prompt.lower()
        assert _SSN_VALUE not in llm.last_prompt  # redacted before it ever reached here

    def test_deidentification_actually_ran_before_indexing(
        self, tmp_path: Path
    ) -> None:
        """The real de-id step must have already redacted PHI-shaped
        content before any chunk ever reaches embedding/indexing --
        checked directly against the captured chunks' own text, not
        assumed from the pipeline reporting "success". This is what makes
        the RAG layer's own index (and, eventually, an LLM's context
        window) PHI-safe by construction, not by the RAG code's own
        discipline -- the real point of chunking downstream of a real
        de-id step, not a fake one."""
        notes_dir = _write_notes(tmp_path)
        chunks = _ingest_real_chunks(notes_dir)
        combined_text = " ".join(c.text for c in chunks)
        assert _SSN_VALUE not in combined_text  # PHI redacted before indexing
        assert "type 2 diabetes" in combined_text  # clinical content preserved
        assert "distal radius fracture" in combined_text

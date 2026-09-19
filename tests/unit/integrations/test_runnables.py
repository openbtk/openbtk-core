"""as_runnable / from_runnable against real langchain-core Runnables and real
OpenBTK components (a guardrail, a chunker, a preprocessor, the DeidEngine)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.runnables import Runnable, RunnableLambda

from openbtk.core.base import BasePreprocessor, BaseSegmenter
from openbtk.core.errors import ConfigError, ProcessingError
from openbtk.core.schemas import TextSpan
from openbtk.data.clinical_text.schemas import (
    ClinicalTextChunk,
    ClinicalTextRecord,
)
from openbtk.deid.engine import DeidEngine
from openbtk.deid.schemas import DeidMode
from openbtk.guardrails.phi_leakage import PHILeakageGuardrail
from openbtk.integrations.langchain import (
    RunnablePreprocessor,
    as_runnable,
    from_runnable,
)

from ._doubles import EchoLLM, OneHotEmbedding

_FAKE_SSN_TEXT = "SSN 123-45-6789"  # phi-fixture-ok: fictitious, obviously fake
_RECORD = ClinicalTextRecord(record_id="r1", source="synthetic", text="alpha beta")


class _WordChunker(BaseSegmenter[ClinicalTextRecord, ClinicalTextChunk]):
    """Real chunker: one chunk per word."""

    def segment(self, record: ClinicalTextRecord) -> Any:
        offset = 0
        for i, word in enumerate(record.text.split()):
            start = record.text.index(word, offset)
            offset = start + len(word)
            yield ClinicalTextChunk(
                chunk_id=f"{record.record_id}:{i}",
                record_id=record.record_id,
                text=word,
                span=TextSpan(start=start, end=offset, label="chunk", confidence=1.0),
                token_count=1,
            )


class _Upper(BasePreprocessor[ClinicalTextRecord]):
    def process(self, record: ClinicalTextRecord) -> ClinicalTextRecord:
        return record.model_copy(update={"text": record.text.upper()})


class TestAsRunnable:
    def test_result_is_a_langchain_runnable_named_after_the_component(self) -> None:
        r = as_runnable(PHILeakageGuardrail())
        assert isinstance(r, Runnable)
        assert r.get_name() == "PHILeakageGuardrail"

    def test_guardrail(self) -> None:
        r = as_runnable(PHILeakageGuardrail())
        assert r.invoke("Plan: rest.").passed is True
        assert r.invoke(_FAKE_SSN_TEXT).passed is False

    def test_llm_takes_and_returns_a_string(self) -> None:
        assert as_runnable(EchoLLM()).invoke("hi") == "HI"

    def test_embedding_provider_takes_a_list_of_texts(self) -> None:
        out = as_runnable(OneHotEmbedding()).invoke(["a", "b"])
        assert out == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]

    def test_preprocessor(self) -> None:
        assert as_runnable(_Upper()).invoke(_RECORD).text == "ALPHA BETA"

    def test_deid_engine_takes_text_and_patient_id(self) -> None:
        engine = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
        # phi-fixture-ok: a fictitious, obviously-fake phone-shaped literal
        result = as_runnable(engine).invoke(
            {"text": "Call (555) 010-2345.", "patient_id": "p-hash"}
        )
        assert "010-2345" not in result.text

    def test_deid_engine_rejects_a_malformed_input(self) -> None:
        engine = DeidEngine(mode=DeidMode.REDACT, recognizers=["rule"])
        with pytest.raises(ConfigError, match="patient_id"):
            as_runnable(engine).invoke({"text": "x"})

    def test_a_chunker_returns_a_list_of_chunks(self) -> None:
        # A real BaseChunker (the supported kind), delegating to the word segmenter.
        from openbtk.core.base import BaseChunker

        class _Words(BaseChunker[ClinicalTextRecord, ClinicalTextChunk]):
            def chunk(self, record: ClinicalTextRecord) -> Any:
                return _WordChunker().segment(record)

        chunks = as_runnable(_Words()).invoke(_RECORD)
        assert [c.text for c in chunks] == ["alpha", "beta"]

    def test_unsupported_components_are_refused_with_the_supported_list(self) -> None:
        with pytest.raises(ConfigError, match="Supported:"):
            as_runnable(_WordChunker())  # a segmenter is not a supported kind

    def test_composes_with_langchain_operators(self) -> None:
        chain = as_runnable(EchoLLM()) | RunnableLambda(lambda s: s + "!")
        assert chain.invoke("hi") == "HI!"
        assert chain.batch(["a", "b"]) == ["A!", "B!"]


class TestFromRunnable:
    def test_rewrites_text_and_drops_now_invalid_sections(self) -> None:
        rec = ClinicalTextRecord(
            record_id="r1",
            source="s",
            text="alpha beta",
            sections={
                "All": TextSpan(start=0, end=10, label="section", confidence=1.0)
            },
        )
        out = from_runnable(RunnableLambda(str.upper)).process(rec)
        assert out.text == "ALPHA BETA"
        assert out.sections is None
        assert rec.text == "alpha beta"  # the input record is untouched

    def test_an_unchanged_text_returns_the_record_as_is(self) -> None:
        rec = ClinicalTextRecord(
            record_id="r1",
            source="s",
            text="abc",
            sections={"All": TextSpan(start=0, end=3, label="section", confidence=1.0)},
        )
        out = from_runnable(RunnableLambda(lambda s: s)).process(rec)
        assert out is rec

    def test_deid_status_is_not_changed_by_a_rewrite(self) -> None:
        out = from_runnable(RunnableLambda(str.upper)).process(_RECORD)
        assert out.deid_status == _RECORD.deid_status

    def test_a_runnable_that_does_not_return_text_is_refused(self) -> None:
        bad: Runnable[str, Any] = RunnableLambda(lambda s: 42)
        with pytest.raises(ProcessingError, match="must return str") as exc:
            from_runnable(bad).process(_RECORD)
        assert exc.value.context == {"record_id": "r1"}

    def test_it_is_a_preprocessor_and_streams(self) -> None:
        pre = from_runnable(RunnableLambda(str.upper))
        assert isinstance(pre, RunnablePreprocessor)
        assert isinstance(pre, BasePreprocessor)
        assert [r.text for r in pre.process_stream([_RECORD, _RECORD])] == [
            "ALPHA BETA",
            "ALPHA BETA",
        ]

"""``CrossEncoderReranker`` (FR-R-04) against fake ``transformers``/``torch`` modules,
so nothing is downloaded. One real-model test at the bottom runs with
``OPENBTK_SLOW_TESTS=1``."""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any

import pytest

from openbtk.core.errors import ConfigError, MissingDependencyError, RetrievalError
from openbtk.core.registry import RERANKER_REGISTRY
from openbtk.core.schemas import SearchResult
from openbtk.retrieval import cross_encoder
from openbtk.retrieval.cross_encoder import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    CrossEncoderReranker,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Value:
    """Stands in for a tensor: remembers its payload and the device it was moved to."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.device: str | None = None

    def to(self, device: str) -> _Value:
        self.device = device
        return self


class _Logits:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = values
        self.shape = (len(values), len(values[0]) if values else 0)

    def reshape(self, n: int) -> _Logits:
        assert n == len(self._values)
        return self

    def tolist(self) -> list[float]:
        return [row[0] for row in self._values]


class _Fake:
    """A fake ``transformers`` + ``torch`` whose score is how many of the query's words
    a passage contains, so ordering is predictable."""

    def __init__(self, *, labels: int = 1) -> None:
        self.tokenizer_calls: list[dict[str, Any]] = []
        self.loads: list[tuple[str, dict[str, Any]]] = []
        self.moved_to: list[str] = []
        self.labels = labels
        fake = self

        class Tokenizer:
            @classmethod
            def from_pretrained(cls, name: str, **kwargs: Any) -> Any:
                fake.loads.append((name, kwargs))
                return cls()

            def __call__(
                self, queries: list[str], passages: list[str], **kw: Any
            ) -> Any:
                fake.tokenizer_calls.append(
                    {"queries": queries, "passages": passages, **kw}
                )
                return {"input_ids": _Value((queries, passages))}

        class Output:
            def __init__(self, logits: _Logits) -> None:
                self.logits = logits

        class Model:
            @classmethod
            def from_pretrained(cls, name: str, **kwargs: Any) -> Any:
                fake.loads.append((name, kwargs))
                return cls()

            def to(self, device: str) -> None:
                fake.moved_to.append(device)

            def eval(self) -> None:
                pass

            def __call__(self, **encoded: Any) -> Output:
                queries, passages = encoded["input_ids"].payload
                rows = []
                for q, p in zip(queries, passages, strict=True):
                    hits = float(sum(w in p.lower() for w in q.lower().split()))
                    rows.append([hits] + [0.0] * (fake.labels - 1))
                return Output(_Logits(rows))

        self.transformers = type(
            "T",
            (),
            {"AutoTokenizer": Tokenizer, "AutoModelForSequenceClassification": Model},
        )
        self.torch = type(
            "Torch", (), {"no_grad": staticmethod(contextlib.nullcontext)}
        )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Fake]:
    fake = _Fake()
    monkeypatch.setattr(
        cross_encoder,
        "require",
        lambda name, extra: {"transformers": fake.transformers, "torch": fake.torch}[
            name
        ],
    )
    yield fake


def _hits(*texts: str | None) -> list[SearchResult]:
    return [
        SearchResult(
            id=f"r{i}",
            score=round(0.9 - i * 0.1, 2),
            metadata={} if text is None else {"text": text},
        )
        for i, text in enumerate(texts)
    ]


class TestConstruction:
    def test_is_registered(self) -> None:
        assert "reranker.general.cross_encoder" in RERANKER_REGISTRY.list_keys()

    def test_constructing_loads_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(name: str, extra: str) -> None:
            raise AssertionError("constructing must not import or load anything")

        monkeypatch.setattr(cross_encoder, "require", explode)
        CrossEncoderReranker()

    def test_the_default_model_is_pinned_to_its_verified_commit(self) -> None:
        identity = CrossEncoderReranker().provenance().model_identity
        assert identity is not None
        assert (identity.name, identity.revision) == (DEFAULT_MODEL, DEFAULT_REVISION)
        assert identity.source == "huggingface"

    def test_another_model_must_be_pinned(self) -> None:
        with pytest.raises(ConfigError, match="revision is required"):
            CrossEncoderReranker(model="org/other-model")
        ok = CrossEncoderReranker(model="org/other-model", revision="abc1234")
        assert ok.provenance().model_identity.revision == "abc1234"  # type: ignore[union-attr]

    @pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"max_length": 1}])
    def test_bad_sizes_are_refused(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ConfigError):
            CrossEncoderReranker(**kwargs)

    def test_provenance_records_settings_and_is_serialisable(self) -> None:
        reranker = CrossEncoderReranker(batch_size=4, max_length=128, device="cpu")
        config = json.loads(reranker.provenance().model_dump_json())["config"]
        assert config["batch_size"] == 4 and config["max_length"] == 128


class TestReranking:
    def test_results_are_ordered_by_the_cross_encoder_not_the_original_score(
        self, fake: _Fake
    ) -> None:
        results = _hits("nothing relevant", "aspirin dose", "aspirin dose in adults")
        ranked = CrossEncoderReranker().rerank("aspirin dose adults", results)
        assert [r.id for r in ranked] == ["r2", "r1", "r0"]

    def test_score_becomes_the_logit_and_the_original_is_kept(
        self, fake: _Fake
    ) -> None:
        results = _hits("aspirin dose")
        (ranked,) = CrossEncoderReranker().rerank("aspirin dose", results)
        assert ranked.score == 2.0
        assert ranked.metadata["retrieval_score"] == 0.9
        assert ranked.metadata["text"] == "aspirin dose"

    def test_the_inputs_are_not_modified(self, fake: _Fake) -> None:
        results = _hits("aspirin dose")
        CrossEncoderReranker().rerank("aspirin dose", results)
        assert results[0].score == 0.9 and "retrieval_score" not in results[0].metadata

    def test_ties_keep_the_original_order(self, fake: _Fake) -> None:
        results = _hits("same words", "same words", "same words")
        ranked = CrossEncoderReranker().rerank("same words", results)
        assert [r.id for r in ranked] == ["r0", "r1", "r2"]

    def test_top_k_truncates_after_ranking(self, fake: _Fake) -> None:
        results = _hits("nothing", "aspirin dose", "aspirin")
        ranked = CrossEncoderReranker().rerank("aspirin dose", results, top_k=1)
        assert [r.id for r in ranked] == ["r1"]

    def test_empty_input(self, fake: _Fake) -> None:
        assert CrossEncoderReranker().rerank("q", []) == []
        assert fake.tokenizer_calls == []

    def test_a_custom_text_key(self, fake: _Fake) -> None:
        results = [SearchResult(id="a", score=1.0, metadata={"body": "aspirin dose"})]
        ranked = CrossEncoderReranker(text_metadata_key="body").rerank(
            "aspirin", results
        )
        assert ranked[0].score == 1.0


class TestResultsWithoutText:
    def test_they_are_kept_after_the_scored_ones_in_their_original_order(
        self, fake: _Fake
    ) -> None:
        results = _hits(None, "aspirin dose", "   ", "aspirin", None)
        ranked = CrossEncoderReranker().rerank("aspirin dose", results)
        assert [r.id for r in ranked] == ["r1", "r3", "r0", "r2", "r4"]

    def test_unscored_results_are_returned_unchanged(self, fake: _Fake) -> None:
        results = _hits(None, "aspirin")
        ranked = CrossEncoderReranker().rerank("aspirin", results)
        assert ranked[-1] == results[0]

    def test_a_non_string_text_is_not_scored(self, fake: _Fake) -> None:
        results = [SearchResult(id="a", score=1.0, metadata={"text": 5})]
        assert CrossEncoderReranker().rerank("q", results) == results

    def test_when_nothing_has_text_no_model_is_loaded(self, fake: _Fake) -> None:
        ranked = CrossEncoderReranker().rerank("q", _hits(None, None))
        assert [r.id for r in ranked] == ["r0", "r1"]
        assert fake.loads == []


class TestLoadingAndBatching:
    def test_the_model_is_loaded_once_at_the_pinned_revision(self, fake: _Fake) -> None:
        reranker = CrossEncoderReranker()
        reranker.rerank("aspirin", _hits("aspirin"))
        reranker.rerank("aspirin", _hits("aspirin"))
        names = [(n, kw["revision"]) for n, kw in fake.loads]
        assert (
            names == [(DEFAULT_MODEL, DEFAULT_REVISION)] * 2
        )  # tokenizer + model, once

    def test_the_model_is_moved_to_the_device(self, fake: _Fake) -> None:
        CrossEncoderReranker(device="cpu").rerank("a", _hits("a"))
        assert fake.moved_to == ["cpu"]

    def test_pairs_are_batched(self, fake: _Fake) -> None:
        results = _hits(*[f"aspirin {i}" for i in range(5)])
        CrossEncoderReranker(batch_size=2).rerank("aspirin", results)
        assert [len(c["passages"]) for c in fake.tokenizer_calls] == [2, 2, 1]
        assert all(set(c["queries"]) == {"aspirin"} for c in fake.tokenizer_calls)

    def test_pairs_are_truncated_and_padded(self, fake: _Fake) -> None:
        CrossEncoderReranker(max_length=64).rerank("a", _hits("a"))
        call = fake.tokenizer_calls[0]
        assert call["max_length"] == 64 and call["truncation"] and call["padding"]


class TestFailures:
    def test_a_model_with_several_outputs_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _Fake(labels=2)
        monkeypatch.setattr(
            cross_encoder,
            "require",
            lambda name, extra: {
                "transformers": fake.transformers,
                "torch": fake.torch,
            }[name],
        )
        with pytest.raises(RetrievalError, match="one relevance logit"):
            CrossEncoderReranker().rerank("a", _hits("a"))

    def test_a_missing_dependency_keeps_its_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(name: str, extra: str) -> None:
            raise MissingDependencyError(f"Install openbtk[{extra}] for {name}.")

        monkeypatch.setattr(cross_encoder, "require", missing)
        with pytest.raises(MissingDependencyError, match=r"openbtk\[text\]"):
            CrossEncoderReranker().rerank("a", _hits("a"))

    def test_other_failures_become_retrieval_errors_without_the_content(
        self, fake: _Fake, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(self: Any, *a: Any, **k: Any) -> None:
            raise RuntimeError("SECRET-QUERY leaked in a library message")

        monkeypatch.setattr(fake.transformers.AutoTokenizer, "__call__", boom)
        with pytest.raises(RetrievalError) as excinfo:
            CrossEncoderReranker().rerank("SECRET-QUERY", _hits("SECRET-PASSAGE"))
        assert "SECRET" not in str(excinfo.value)
        assert "SECRET" not in repr(excinfo.value.context)
        assert excinfo.value.__cause__ is not None  # chained for debugging


@pytest.mark.slow
def test_the_real_medcpt_cross_encoder_prefers_the_relevant_passage() -> None:
    """Runs the real model: needs ``transformers``, ``torch`` and a ~440 MB download."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    results = [
        SearchResult(
            id="off",
            score=0.9,
            metadata={"text": "The patient reported a mild headache after the flight."},
        ),
        SearchResult(
            id="on",
            score=0.1,
            metadata={
                "text": (
                    "Metformin is the first-line medication for "
                    "type 2 diabetes mellitus."
                )
            },
        ),
    ]
    ranked = CrossEncoderReranker().rerank("How is type 2 diabetes treated?", results)
    assert [r.id for r in ranked] == ["on", "off"]

"""ROUGE and BERTScore (FR-X-05). Summaries are made up; no real note is used."""

from __future__ import annotations

import contextlib
import importlib.util
import json
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pytest

from openbtk.core.errors import ConfigError, MissingDependencyError, ProcessingError
from openbtk.core.provenance import DataDigest
from openbtk.eval import summarisation
from openbtk.eval.model_card import ModelCard
from openbtk.eval.summarisation import (
    BertScorer,
    SummaryPair,
    evaluate_summaries,
    rouge_scores,
    summarisation_manifest,
)
from openbtk.retrieval.cross_encoder import CrossEncoderReranker

if TYPE_CHECKING:
    from collections.abc import Iterator


# ------------------------------------------------------------------------- ROUGE

requires_rouge = pytest.mark.skipif(
    importlib.util.find_spec("rouge_score") is None,
    reason="needs the 'eval' extra (rouge-score)",
)


@requires_rouge
class TestRouge:
    def test_identical_text_scores_one(self) -> None:
        scores = rouge_scores("the patient is stable", "the patient is stable")
        assert all(s.precision == s.recall == s.f1 == 1.0 for s in scores.values())

    def test_a_hand_computed_case(self) -> None:
        """Reference 5 words, candidate 4 of them: precision 4/4, recall 4/5."""
        scores = rouge_scores(
            "the patient is stable today", "the patient is stable", use_stemmer=False
        )
        r1 = scores["rouge1"]
        assert (r1.precision, r1.recall) == (1.0, 0.8)
        assert r1.f1 == pytest.approx(2 * 1.0 * 0.8 / 1.8)

    def test_rouge2_counts_word_pairs(self) -> None:
        # reference bigrams: (a b)(b c); candidate bigrams: (a b)(b d) -> 1 of 2 each
        scores = rouge_scores("a b c", "a b d", use_stemmer=False, types=["rouge2"])
        assert scores["rouge2"].precision == scores["rouge2"].recall == 0.5

    def test_rouge_l_uses_the_longest_common_subsequence(self) -> None:
        # LCS of "a x b y c" and "a b c" is 3 words
        scores = rouge_scores("a x b y c", "a b c", use_stemmer=False, types=["rougeL"])
        assert scores["rougeL"].precision == 1.0
        assert scores["rougeL"].recall == pytest.approx(3 / 5)

    def test_stemming_matches_inflected_forms_only_when_on(self) -> None:
        assert rouge_scores("walking", "walked", types=["rouge1"])["rouge1"].f1 == 1.0
        off = rouge_scores("walking", "walked", types=["rouge1"], use_stemmer=False)
        assert off["rouge1"].f1 == 0.0

    def test_matching_is_case_insensitive(self) -> None:
        assert rouge_scores("Stable", "stable")["rouge1"].f1 == 1.0

    def test_disjoint_text_scores_zero(self) -> None:
        assert rouge_scores("alpha beta", "gamma delta")["rouge1"].f1 == 0.0

    def test_an_empty_candidate_scores_zero(self) -> None:
        scores = rouge_scores("the patient", "")
        assert scores["rouge1"].precision == scores["rouge1"].recall == 0.0

    def test_no_variants_means_nothing_to_compute(self) -> None:
        assert rouge_scores("a", "a", types=[]) == {}

    def test_an_unknown_variant_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="Unknown ROUGE variant"):
            rouge_scores("a", "a", types=["rouge3"])

    def test_it_matches_the_library_called_directly(self) -> None:
        from rouge_score import rouge_scorer

        reference = "Chest pain resolved after nitroglycerin; troponin negative."
        candidate = "Troponin negative and the chest pain resolved."
        library = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        ).score(reference, candidate)
        ours = rouge_scores(reference, candidate)
        for name, score in library.items():
            assert ours[name].precision == pytest.approx(score.precision)
            assert ours[name].recall == pytest.approx(score.recall)
            assert ours[name].f1 == pytest.approx(score.fmeasure)

    def test_a_missing_library_says_what_to_install(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def missing(name: str, extra: str) -> None:
            raise MissingDependencyError(f"Install openbtk[{extra}] for {name}.")

        monkeypatch.setattr(summarisation, "require", missing)
        with pytest.raises(MissingDependencyError, match=r"openbtk\[eval\]"):
            rouge_scores("a", "a")


# ------------------------------------------------------ a numpy stand-in for torch


class _Tensor:
    """The few tensor operations BertScorer uses, on a numpy array."""

    def __init__(self, data: Any) -> None:
        self.data = np.asarray(data)

    def to(self, device: str) -> _Tensor:
        return self

    def bool(self) -> _Tensor:
        return _Tensor(self.data.astype(bool))

    def any(self) -> _Tensor:
        return _Tensor(self.data.any())

    def __bool__(self) -> bool:
        return bool(self.data)

    def __invert__(self) -> _Tensor:
        return _Tensor(~self.data)

    def __getitem__(self, key: Any) -> _Tensor:
        return _Tensor(self.data[key.data if isinstance(key, _Tensor) else key])

    def __matmul__(self, other: _Tensor) -> _Tensor:
        return _Tensor(self.data @ other.data)

    @property
    def T(self) -> _Tensor:  # noqa: N802 - mirrors torch
        return _Tensor(self.data.T)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def device(self) -> str:
        return "cpu"

    def __truediv__(self, other: _Tensor) -> _Tensor:
        return _Tensor(self.data / other.data)

    def clamp_min(self, low: float) -> _Tensor:
        return _Tensor(np.maximum(self.data, low))

    def max(self, dim: int) -> Any:
        class Result(NamedTuple):
            values: _Tensor

        return Result(_Tensor(self.data.max(axis=dim)))

    def mean(self) -> _Tensor:
        return _Tensor(self.data.mean())

    def __float__(self) -> float:
        return float(self.data)


class _Linalg:
    @staticmethod
    def norm(t: _Tensor, dim: int, keepdim: bool) -> _Tensor:
        return _Tensor(np.linalg.norm(t.data, axis=dim, keepdims=keepdim))


_TORCH = type(
    "Torch", (), {"linalg": _Linalg, "no_grad": staticmethod(contextlib.nullcontext)}
)

# A toy vocabulary: each word has a fixed 2-d direction, and layer L rotates nothing, so
# similarities are the cosines of the fixed vectors.
_VECTORS = {
    "[CLS]": (0.0, 1.0),
    "[SEP]": (0.6, 0.8),
    "stable": (1.0, 0.0),
    "fever": (0.0, 1.0),
    "pain": (0.6, 0.8),
    "resolved": (-1.0, 0.0),
}


class _FakeEncoding(dict):  # type: ignore[type-arg]
    pass


def _fake_transformers(layers: int = 6, seen: list[Any] | None = None) -> Any:
    class Tokenizer:
        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> Any:
            if seen is not None:
                seen.append((name, kwargs))
            return cls()

        def __call__(self, text: str, **kwargs: Any) -> Any:
            words = ["[CLS]", *text.split(), "[SEP]"] if text else ["[CLS]", "[SEP]"]
            special = [1] + [0] * (len(words) - 2) + [1]
            return _FakeEncoding(
                input_ids=_Tensor([words]),
                special_tokens_mask=_Tensor([special]),
            )

    class Config:
        num_hidden_layers = layers

    class Model:
        config = Config()

        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> Any:
            if seen is not None:
                seen.append((name, kwargs))
            return cls()

        def to(self, device: str) -> None:
            pass

        def eval(self) -> None:
            pass

        def __call__(self, input_ids: _Tensor, output_hidden_states: bool) -> Any:
            words = input_ids.data[0]
            vectors = np.array([_VECTORS[w] for w in words], dtype=float)
            states = [_Tensor(vectors[None, :, :]) for _ in range(layers + 1)]
            return type("Out", (), {"hidden_states": states})

    return type("T", (), {"AutoTokenizer": Tokenizer, "AutoModel": Model})


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    seen: list[Any] = []
    modules = {"transformers": _fake_transformers(seen=seen), "torch": _TORCH}
    monkeypatch.setattr(
        summarisation,
        "require",
        lambda name, extra: (
            modules[name] if name in modules else __import__(name, fromlist=["x"])
        ),
    )
    yield seen


def _scorer(**kwargs: Any) -> BertScorer:
    return BertScorer(model="org/enc", revision="abc1234", layer=3, **kwargs)


def _cos(a: str, b: str) -> float:
    va, vb = np.array(_VECTORS[a]), np.array(_VECTORS[b])
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


class TestBertScoreConstruction:
    def test_a_revision_is_required(self) -> None:
        with pytest.raises(ConfigError, match="pinned model revision"):
            BertScorer(model="org/enc", revision="", layer=3)

    @pytest.mark.parametrize("kwargs", [{"layer": 0}, {"layer": -1}, {"max_length": 2}])
    def test_bad_settings_are_refused(self, kwargs: dict[str, int]) -> None:
        settings = {"layer": 3, "max_length": 512, **kwargs}
        with pytest.raises(ConfigError):
            BertScorer(model="org/enc", revision="abc1234", **settings)

    def test_constructing_loads_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(name: str, extra: str) -> None:
            raise AssertionError("constructing must not import or load")

        monkeypatch.setattr(summarisation, "require", explode)
        _scorer()

    def test_the_model_revision_and_layer_are_recorded(self) -> None:
        provenance = _scorer().provenance()
        identity = provenance.model_identity
        assert identity is not None
        assert (identity.name, identity.revision) == ("org/enc", "abc1234")
        assert provenance.config["layer"] == 3

    def test_the_model_is_loaded_once_at_the_pinned_revision(
        self, fake: list[Any]
    ) -> None:
        scorer = _scorer()
        scorer.score("stable", "stable")
        scorer.score("fever", "pain")
        assert fake == [
            ("org/enc", {"revision": "abc1234"}),
            ("org/enc", {"revision": "abc1234"}),
        ]  # tokenizer and model, once each

    def test_a_layer_beyond_the_model_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        modules = {"transformers": _fake_transformers(layers=2), "torch": _TORCH}
        monkeypatch.setattr(summarisation, "require", lambda name, extra: modules[name])
        with pytest.raises(ConfigError, match="beyond the model"):
            _scorer().score("stable", "stable")


class TestBertScoreMath:
    def test_identical_text_scores_one(self, fake: list[Any]) -> None:
        s = _scorer().score("stable fever", "stable fever")
        assert s.precision == pytest.approx(1.0) and s.recall == pytest.approx(1.0)
        assert s.f1 == pytest.approx(1.0)

    def test_precision_and_recall_are_the_greedy_means_computed_by_hand(
        self, fake: list[Any]
    ) -> None:
        # candidate {stable, pain}; reference {stable, fever}. Each token can also match
        # the other text's [CLS]/[SEP], which carry zero weight but do take part.
        ref = ["[CLS]", "stable", "fever", "[SEP]"]
        hyp = ["[CLS]", "stable", "pain", "[SEP]"]
        p = np.mean([max(_cos(h, r) for r in ref) for h in ("stable", "pain")])
        r = np.mean([max(_cos(rt, h) for h in hyp) for rt in ("stable", "fever")])
        score = _scorer().score("stable fever", "stable pain")
        assert score.precision == pytest.approx(p)
        assert score.recall == pytest.approx(r)
        assert score.f1 == pytest.approx(2 * p * r / (p + r))

    def test_the_special_tokens_take_part_in_matching_but_not_in_the_average(
        self, fake: list[Any]
    ) -> None:
        """The reference implementation's behaviour, which this one reproduces: the
        reference token 'fever' is best matched by the candidate's [CLS] (same
        direction), so its recall is 1 even though the candidate never says 'fever'."""
        score = _scorer().score("fever", "stable")
        assert score.recall == pytest.approx(1.0)
        # ...while candidate "stable" best matches the reference [SEP]: cos = 0.6
        assert score.precision == pytest.approx(_cos("stable", "[SEP]"))
        assert score.precision == pytest.approx(0.6)

    def test_an_empty_text_scores_zero_on_everything(self, fake: list[Any]) -> None:
        for reference, candidate in (("", "stable"), ("stable", ""), ("", "")):
            s = _scorer().score(reference, candidate)
            assert (s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0)

    def test_failures_become_processing_errors_without_the_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Broken:
            @classmethod
            def from_pretrained(cls, name: str, **kwargs: Any) -> Any:
                raise RuntimeError("SECRET-TEXT in a library message")

        modules = {
            "transformers": type(
                "T", (), {"AutoTokenizer": Broken, "AutoModel": Broken}
            ),
            "torch": _TORCH,
        }
        monkeypatch.setattr(summarisation, "require", lambda name, extra: modules[name])
        with pytest.raises(ProcessingError) as excinfo:
            _scorer().score("SECRET-TEXT", "SECRET-TEXT")
        assert "SECRET" not in str(excinfo.value)
        assert "SECRET" not in repr(excinfo.value.context)


# ------------------------------------------------------------------- evaluation

_PAIRS = [
    SummaryPair(
        example_id="a",
        reference="the patient is stable",
        candidate="the patient is stable",
    ),
    SummaryPair(
        example_id="b", reference="alpha beta gamma delta", candidate="alpha beta"
    ),
]


@requires_rouge
class TestEvaluate:
    def test_the_report_is_a_macro_average_with_each_examples_own_scores(self) -> None:
        report = evaluate_summaries(_PAIRS, use_stemmer=False)
        assert report.n == 2
        # example b: precision 2/2, recall 2/4 for rouge1
        assert report.rouge["rouge1"].precision == pytest.approx((1.0 + 1.0) / 2)
        assert report.rouge["rouge1"].recall == pytest.approx((1.0 + 0.5) / 2)
        assert [e.example_id for e in report.examples] == ["a", "b"]
        assert report.examples[1].rouge["rouge1"].recall == 0.5

    def test_the_input_is_streamed(self) -> None:
        consumed = 0

        def pairs() -> Iterator[SummaryPair]:
            nonlocal consumed
            for pair in _PAIRS:
                consumed += 1
                yield pair

        assert evaluate_summaries(pairs()).n == 2 and consumed == 2

    def test_no_examples_is_an_empty_report_not_a_division_error(self) -> None:
        report = evaluate_summaries([])
        assert report.n == 0 and report.rouge == {} and report.bertscore is None

    def test_variants_can_be_chosen_and_skipped(self) -> None:
        assert set(evaluate_summaries(_PAIRS, rouge_types=["rouge1"]).rouge) == {
            "rouge1"
        }
        assert evaluate_summaries(_PAIRS, rouge_types=[]).rouge == {}

    def test_bertscore_is_added_when_a_scorer_is_given(self, fake: list[Any]) -> None:
        pairs = [
            SummaryPair(
                example_id="a", reference="stable fever", candidate="stable fever"
            )
        ]
        report = evaluate_summaries(pairs, bertscorer=_scorer())
        assert report.bertscore is not None
        assert report.bertscore.f1 == pytest.approx(1.0)
        assert report.examples[0].bertscore is not None

    def test_the_report_never_contains_the_text(self, fake: list[Any]) -> None:
        pairs = [
            SummaryPair(
                example_id="a",
                reference="SECRET-REF stable",
                candidate="SECRET-CAND stable",
            )
        ]
        # 'SECRET-...' is not in the toy vocabulary, so only score ROUGE here
        report = evaluate_summaries(pairs)
        assert "SECRET" not in report.model_dump_json()
        assert "SECRET" not in json.dumps(report.as_dict())

    def test_the_flat_view_names_every_score(self) -> None:
        flat = evaluate_summaries(_PAIRS).as_dict()
        assert flat["n"] == 2
        for name in ("rouge1", "rouge2", "rougeL"):
            for part in ("precision", "recall", "f1"):
                assert isinstance(flat[f"{name}_{part}"], float)


@requires_rouge
class TestManifest:
    def test_kind_scores_and_inputs(self) -> None:
        report = evaluate_summaries(_PAIRS)
        digest = DataDigest(uri="./pairs.jsonl", sha256="cd" * 32, record_count=2)
        manifest = summarisation_manifest(report, input_digests=[digest])
        assert manifest.kind == "summarisation"
        assert manifest.report["n"] == 2 and manifest.input_digests == [digest]
        assert manifest.component is None

    def test_the_bertscore_model_is_recorded_when_used(self) -> None:
        manifest = summarisation_manifest(
            evaluate_summaries(_PAIRS), bertscorer=_scorer()
        )
        assert manifest.component is not None
        assert manifest.component.model_identity is not None
        assert manifest.component.model_identity.revision == "abc1234"

    def test_a_score_reaches_a_model_card_only_through_the_manifest(self) -> None:
        manifest = summarisation_manifest(evaluate_summaries(_PAIRS))
        card = ModelCard.for_component(CrossEncoderReranker()).with_evaluation(
            manifest, "rouge1_f1", name="ROUGE-1 F1"
        )
        assert card.evaluations[0].eval_id == manifest.eval_id
        assert card.evaluations[0].value == manifest.report["rouge1_f1"]


# ------------------------------------------------------- the real reference numbers

# Recorded from the reference ``bert-score`` package (0.3.12), same model, commit
# and layer, one pair per batch so no padding is involved. To regenerate:
#   bert_score.score([candidate], [reference], model_type="distilbert-base-uncased",
#                    num_layers=5, batch_size=1)
_REFERENCE_PAIRS = [
    (
        "The patient was discharged home in stable condition on metformin.",
        "Patient discharged in stable condition, continue metformin.",
        (0.88984, 0.89956, 0.89468),
    ),
    (
        "Chest pain resolved after nitroglycerin; troponin negative.",
        "Troponin negative and the chest pain resolved.",
        (0.92271, 0.86448, 0.89265),
    ),
    (
        "No acute distress. Lungs clear to auscultation bilaterally.",
        "The patient reports severe abdominal pain with vomiting.",
        (0.79948, 0.71922, 0.75723),
    ),
    (
        "Follow up with cardiology in two weeks.",
        "Follow up with cardiology in two weeks.",
        (1.0, 1.0, 1.0),
    ),
    (
        "Type 2 diabetes, hypertension, and hyperlipidemia are well controlled.",
        "Well controlled diabetes.",
        (0.84062, 0.77433, 0.80612),
    ),
]


@pytest.mark.slow
def test_matches_the_reference_bert_score_package_on_a_real_model() -> None:
    """Needs ``transformers``, ``torch`` and a ~270 MB download."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    scorer = BertScorer(
        model="distilbert-base-uncased",
        revision="12040accade4e8a0f71eabdb258fecc2e7e948be",
        layer=5,
    )
    for reference, candidate, (p, r, f) in _REFERENCE_PAIRS:
        got = scorer.score(reference, candidate)
        assert got.precision == pytest.approx(p, abs=2e-5)
        assert got.recall == pytest.approx(r, abs=2e-5)
        assert got.f1 == pytest.approx(f, abs=2e-5)

"""Model cards (FR-P-06). Manifests are hand-built; no real evaluation is run."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from openbtk.core.errors import ConfigError
from openbtk.core.provenance import (
    ComponentProvenance,
    DataDigest,
    ModelIdentity,
    RunManifest,
    StepProvenance,
)
from openbtk.eval.manifest import EvalManifest
from openbtk.eval.model_card import ModelCard, cards_from_run
from openbtk.retrieval.cross_encoder import CrossEncoderReranker
from openbtk.retrieval.reranker import ConceptOverlapReranker

if TYPE_CHECKING:
    from pathlib import Path

_T = datetime(2026, 1, 1, tzinfo=UTC)


def _eval(report: dict[str, float | str | bool], **kwargs: object) -> EvalManifest:
    return EvalManifest(
        eval_id="eval-1",
        kind="retrieval",
        started_at=_T,
        ended_at=_T,
        report=report,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _card() -> ModelCard:
    return ModelCard.for_component(CrossEncoderReranker())


class TestForComponent:
    def test_identity_comes_from_the_components_own_provenance(self) -> None:
        card = _card()
        assert card.model.name == "ncbi/MedCPT-Cross-Encoder"
        assert len(card.model.revision) == 40
        assert card.component_key == "reranker.general.cross_encoder"
        assert card.openbtk_version

    def test_it_takes_provenance_directly_too(self) -> None:
        provenance = CrossEncoderReranker().provenance()
        assert ModelCard.for_component(provenance).model == provenance.model_identity

    def test_the_components_settings_are_recorded(self) -> None:
        assert _card().settings["batch_size"] == 16

    def test_a_component_that_wraps_no_model_has_no_card(self) -> None:
        with pytest.raises(ConfigError, match="wraps no model"):
            ModelCard.for_component(
                ConceptOverlapReranker(extract_concepts=lambda t: [])
            )

    def test_the_text_sections_are_only_what_the_caller_wrote(self) -> None:
        card = ModelCard.for_component(
            CrossEncoderReranker(),
            intended_use="Reranking passages.",
            limitations="Untested on non-English notes.",
        )
        assert card.intended_use == "Reranking passages."
        assert card.limitations == "Untested on non-English notes."
        assert card.training_data is None and card.ethical_considerations is None


class TestEvaluations:
    def test_a_metric_is_read_from_an_eval_manifest(self) -> None:
        card = _card().with_evaluation(_eval({"recall_at_5": 0.83}), "recall_at_5")
        (entry,) = card.evaluations
        assert (entry.metric, entry.value) == ("recall_at_5", 0.83)
        assert (entry.eval_id, entry.kind) == ("eval-1", "retrieval")

    def test_the_label_can_differ_from_the_report_key(self) -> None:
        card = _card().with_evaluation(_eval({"r5": 0.5}), "r5", name="Recall@5")
        assert card.evaluations[0].metric == "Recall@5"

    def test_the_inputs_the_number_came_from_are_recorded(self) -> None:
        manifest = _eval(
            {"f1": 0.9},
            input_digests=[
                DataDigest(uri="./labelled.jsonl", sha256="ab" * 32, record_count=10),
                DataDigest(uri="./dir", sha256=None, record_count=3),
            ],
        )
        entry = _card().with_evaluation(manifest, "f1").evaluations[0]
        assert entry.inputs == [
            f"./labelled.jsonl (sha256 {'ab' * 32})",
            "./dir (not hashed)",
        ]

    @pytest.mark.parametrize("report", [{"f1": "high"}, {"f1": True}, {"other": 1.0}])
    def test_a_number_that_the_report_does_not_hold_cannot_be_put_on_a_card(
        self, report: dict[str, float | str | bool]
    ) -> None:
        with pytest.raises(ConfigError, match="no numeric"):
            _card().with_evaluation(_eval(report), "f1")

    def test_evaluations_accumulate_and_the_card_is_not_mutated(self) -> None:
        base = _card()
        one = base.with_evaluation(_eval({"a": 1.0}), "a")
        two = one.with_evaluation(_eval({"b": 2}), "b")
        assert base.evaluations == [] and len(one.evaluations) == 1
        assert [e.value for e in two.evaluations] == [1.0, 2.0]

    def test_there_is_no_way_to_type_a_score_into_a_card(self) -> None:
        with pytest.raises(TypeError):
            _card().with_evaluation(_eval({"f1": 0.9}), "f1", value=0.99)  # type: ignore[call-arg]


class TestMarkdown:
    def test_sections_you_did_not_write_say_so(self) -> None:
        text = _card().to_markdown()
        for heading in (
            "Intended use",
            "Out-of-scope use",
            "Training data",
            "Limitations",
        ):
            assert f"## {heading}\n\nNot provided." in text

    def test_what_you_wrote_is_rendered(self) -> None:
        card = ModelCard.for_component(
            CrossEncoderReranker(),
            intended_use="Reranking passages.",
            contact="team@example.org",
        )
        text = card.to_markdown()
        assert "## Intended use\n\nReranking passages." in text
        assert "team@example.org" in text

    def test_the_identity_section_names_the_pinned_revision(self) -> None:
        text = _card().to_markdown()
        assert "`ncbi/MedCPT-Cross-Encoder`" in text
        assert f"`{_card().model.revision}`" in text
        assert "reranker.general.cross_encoder" in text

    def test_no_evaluation_is_stated_not_implied(self) -> None:
        text = _card().to_markdown()
        assert "No evaluation is attached" in text

    def test_an_attached_evaluation_is_a_table_row_with_its_source(self) -> None:
        text = (
            _card()
            .with_evaluation(_eval({"recall_at_5": 0.8312}), "recall_at_5")
            .to_markdown()
        )
        assert "| recall_at_5 | 0.8312 | `eval-1` (retrieval) |" in text

    def test_the_configuration_is_listed(self) -> None:
        assert "- `batch_size`: `16`" in _card().to_markdown()

    def test_it_is_deterministic_for_a_fixed_date(self) -> None:
        a = _card().model_copy(update={"generated_at": _T})
        b = _card().model_copy(update={"generated_at": _T})
        assert a.to_markdown() == b.to_markdown()
        assert "Generated 2026-01-01." in a.to_markdown()

    def test_write_saves_the_markdown_and_creates_directories(
        self, tmp_path: Path
    ) -> None:
        target = _card().write(tmp_path / "cards" / "MODEL_CARD.md")
        assert target.read_text(encoding="utf-8").startswith("# Model card:")

    def test_the_card_is_json_serialisable(self) -> None:
        dumped = _card().with_evaluation(_eval({"a": 1.0}), "a").model_dump_json()
        assert ModelCard.model_validate_json(dumped).evaluations[0].value == 1.0


def _step(step_id: str, provenance: ComponentProvenance) -> StepProvenance:
    return StepProvenance(
        step_id=step_id,
        component=provenance,
        records_in=1,
        records_out=1,
        status="success",
    )


class TestFromRun:
    def _run(self, *steps: StepProvenance) -> RunManifest:
        return RunManifest(
            run_id="r1",
            status="success",
            config={"name": "p", "steps": []},
            started_at=_T,
            steps=list(steps),
        )

    def test_one_card_per_distinct_model_used(self) -> None:
        rerank = CrossEncoderReranker().provenance()
        no_model = ConceptOverlapReranker(extract_concepts=lambda t: []).provenance()
        other = CrossEncoderReranker(model="org/other", revision="abc1234").provenance()
        manifest = self._run(
            _step("a", rerank),
            _step("b", no_model),
            _step("c", rerank),
            _step("d", other),
        )
        cards = cards_from_run(manifest)
        assert [c.model.name for c in cards] == [
            "ncbi/MedCPT-Cross-Encoder",
            "org/other",
        ]

    def test_shared_text_is_applied_to_every_card(self) -> None:
        manifest = self._run(_step("a", CrossEncoderReranker().provenance()))
        (card,) = cards_from_run(manifest, intended_use="Clinical QA support.")
        assert card.intended_use == "Clinical QA support."

    def test_a_run_with_no_models_has_no_cards(self) -> None:
        no_model = ConceptOverlapReranker(extract_concepts=lambda t: []).provenance()
        assert cards_from_run(self._run(_step("a", no_model))) == []


def test_a_bare_model_identity_can_be_carded_by_hand() -> None:
    card = ModelCard(
        model=ModelIdentity(
            name="org/finetune", revision="abc1234", source="huggingface"
        ),
        training_data="A synthetic corpus produced by the maintainers.",
    )
    assert "synthetic corpus" in card.to_markdown()

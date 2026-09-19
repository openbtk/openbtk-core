"""Clinical QA evaluation: readers for the MedQA/MedMCQA formats, answer
parsing, scoring and the provenance manifest. Every input is a hand-written
invented question -- not an item from either benchmark -- and every expected
value is worked out by hand."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from openbtk.core.base import BaseLLMProvider
from openbtk.core.errors import DatasetError
from openbtk.core.schemas import LLMResponse, TokenUsage
from openbtk.eval.qa import (
    LLMAnswerer,
    MCQItem,
    QAReport,
    evaluate_qa,
    format_prompt,
    parse_choice,
    qa_manifest,
    read_medmcqa_jsonl,
    read_medqa_jsonl,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _item(item_id: str, answer: str = "B", subject: str | None = None) -> MCQItem:
    return MCQItem(
        item_id=item_id,
        question=f"Invented question {item_id}?",
        options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        answer=answer,
        subject=subject,
    )


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


class TestMCQItem:
    def test_the_answer_must_be_an_option(self) -> None:
        with pytest.raises(ValidationError, match="not one of the option keys"):
            MCQItem(
                item_id="q", question="Q?", options={"A": "x", "B": "y"}, answer="C"
            )

    def test_needs_at_least_two_options(self) -> None:
        with pytest.raises(ValidationError):
            MCQItem(item_id="q", question="Q?", options={"A": "x"}, answer="A")


class TestReadMedQA:
    def _row(self, **over: Any) -> dict[str, Any]:
        row: dict[str, Any] = {
            "question": "Invented Q?",
            "answer": "beta",
            "options": {"A": "alpha", "B": "beta"},
            "meta_info": "step1",
            "answer_idx": "B",
        }
        row.update(over)
        return row

    def test_reads_the_documented_shape(self, tmp_path: Path) -> None:
        (item,) = read_medqa_jsonl(_write(tmp_path / "q.jsonl", [self._row()]))
        assert (item.answer, item.subject, item.item_id) == ("B", "step1", "q:1")
        assert item.options == {"A": "alpha", "B": "beta"}

    def test_falls_back_to_matching_the_answer_text(self, tmp_path: Path) -> None:
        row = self._row()
        del row["answer_idx"]
        (item,) = read_medqa_jsonl(_write(tmp_path / "q.jsonl", [row]))
        assert item.answer == "B"

    def test_an_answer_that_matches_no_option_is_refused_without_the_question(
        self, tmp_path: Path
    ) -> None:
        row = self._row(answer="zeta", question="SECRETQUESTION")
        del row["answer_idx"]
        with pytest.raises(DatasetError, match="line 1") as exc:
            list(read_medqa_jsonl(_write(tmp_path / "q.jsonl", [row])))
        assert "SECRETQUESTION" not in str(exc.value)

    def test_an_ambiguous_answer_text_is_refused(self, tmp_path: Path) -> None:
        row = self._row(options={"A": "same", "B": "same"}, answer="same")
        del row["answer_idx"]
        with pytest.raises(DatasetError, match="correct option"):
            list(read_medqa_jsonl(_write(tmp_path / "q.jsonl", [row])))

    def test_missing_options_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="options"):
            list(read_medqa_jsonl(_write(tmp_path / "q.jsonl", [{"question": "Q?"}])))

    def test_an_invalid_item_is_reported_by_line(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="not a valid item"):
            list(
                read_medqa_jsonl(_write(tmp_path / "q.jsonl", [self._row(question="")]))
            )

    def test_streams_lazily(self, tmp_path: Path) -> None:
        path = tmp_path / "q.jsonl"
        path.write_text(
            json.dumps(self._row()) + "\n" + "{not json\n", encoding="utf-8"
        )
        it = read_medqa_jsonl(path)
        assert next(it).answer == "B"
        with pytest.raises(DatasetError, match="line 2"):
            next(it)


class TestReadMedMCQA:
    def _row(self, **over: Any) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": "m1",
            "question": "Invented Q?",
            "opa": "alpha",
            "opb": "beta",
            "opc": "gamma",
            "opd": "delta",
            "cop": 2,
            "subject_name": "Anatomy",
        }
        row.update(over)
        return row

    def test_reads_the_documented_shape(self, tmp_path: Path) -> None:
        (item,) = read_medmcqa_jsonl(_write(tmp_path / "m.jsonl", [self._row()]))
        assert (item.item_id, item.answer, item.subject) == ("m1", "C", "Anatomy")
        assert item.options == {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}

    @pytest.mark.parametrize(
        ("cop", "expected"),
        [(0, "A"), (3, "D"), ("b", "B"), ("D", "D")],
    )
    def test_cop_forms(self, tmp_path: Path, cop: object, expected: str) -> None:
        (item,) = read_medmcqa_jsonl(_write(tmp_path / "m.jsonl", [self._row(cop=cop)]))
        assert item.answer == expected

    def test_a_hidden_answer_key_is_refused_not_scored_as_zero(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(DatasetError, match="hides its answers"):
            list(read_medmcqa_jsonl(_write(tmp_path / "m.jsonl", [self._row(cop=-1)])))

    @pytest.mark.parametrize("cop", [4, 5, "e", None, True, 2.5])
    def test_an_ambiguous_or_unknown_cop_is_refused(
        self, tmp_path: Path, cop: object
    ) -> None:
        with pytest.raises(DatasetError, match="ambiguous"):
            list(read_medmcqa_jsonl(_write(tmp_path / "m.jsonl", [self._row(cop=cop)])))

    def test_missing_option_fields_are_refused(self, tmp_path: Path) -> None:
        row = self._row()
        del row["opc"]
        with pytest.raises(DatasetError, match="opc"):
            list(read_medmcqa_jsonl(_write(tmp_path / "m.jsonl", [row])))

    def test_an_empty_question_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="not a valid item"):
            list(
                read_medmcqa_jsonl(
                    _write(tmp_path / "m.jsonl", [self._row(question="")])
                )
            )


class TestFileErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="Could not open"):
            list(read_medqa_jsonl(tmp_path / "nope.jsonl"))

    def test_blank_lines_are_skipped_and_a_non_object_line_is_refused(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "q.jsonl"
        path.write_text("\n[1, 2]\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="not a JSON object"):
            list(read_medqa_jsonl(path))


class TestParseChoice:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ("B", "B"),
            ("b", "B"),
            ("(C)", "C"),
            ("D.", "D"),
            ("B) beta", "B"),
            ("B. beta is right", "B"),
            ("The answer is C.", "C"),
            ("Answer: (a)", "A"),
            ("  \n B \n", "B"),
            ("I think the best answer is D because ...", "D"),
        ],
    )
    def test_recognised_forms(self, reply: str, expected: str) -> None:
        assert parse_choice(reply, "ABCD") == expected

    @pytest.mark.parametrize(
        "reply",
        [
            "A patient presents with fever.",  # the article, not option A
            "I cannot answer this.",
            "",
            "Both B and C could be right.",
            "E",  # not an offered option
            "The answer is E.",
            "The answer is a bit unclear.",  # lower-case 'a' is a word, not option A
        ],
    )
    def test_unparseable_replies_are_none(self, reply: str) -> None:
        assert parse_choice(reply, "ABCD") is None


class TestEvaluate:
    def test_counts_are_worked_out_by_hand(self) -> None:
        items = [
            _item("1", "B", "Anatomy"),  # replied B -> correct
            _item("2", "C", "Anatomy"),  # replied A -> wrong
            _item("3", "D", "Physiology"),  # rambles -> unanswered (wrong)
            _item("4", "A", None),  # replied "(a)" -> correct, no subject
        ]
        replies = iter(["B", "The answer is A", "hmm hard to say", "(a)"])
        report = evaluate_qa(lambda prompt: next(replies), items)
        assert (report.n, report.correct, report.unanswered) == (4, 2, 1)
        assert report.accuracy == 0.5
        assert report.per_subject["Anatomy"].n == 2
        assert report.per_subject["Anatomy"].correct == 1
        assert report.per_subject["Physiology"].accuracy == 0.0
        assert set(report.per_subject) == {"Anatomy", "Physiology"}

    def test_keep_results_records_ids_and_letters_only(self) -> None:
        report = evaluate_qa(
            lambda p: "B", [_item("1", "B"), _item("2", "C")], keep_results=True
        )
        assert [(r.item_id, r.predicted, r.correct) for r in report.results] == [
            ("1", "B", True),
            ("2", "B", False),
        ]
        assert "Invented question" not in report.model_dump_json()

    def test_results_are_off_by_default(self) -> None:
        assert evaluate_qa(lambda p: "B", [_item("1")]).results == []

    def test_items_stream_without_being_materialised(self) -> None:
        def stream() -> Iterator[MCQItem]:
            for i in range(1000):
                yield _item(str(i))

        assert evaluate_qa(lambda p: "B", stream()).n == 1000

    def test_the_prompt_lists_every_option_and_ends_asking_for_the_answer(self) -> None:
        seen: list[str] = []
        evaluate_qa(lambda p: seen.append(p) or "B", [_item("1")])
        prompt = seen[0]
        for line in ("A. alpha", "B. beta", "C. gamma", "D. delta"):
            assert line in prompt
        assert prompt.endswith("Answer:")
        assert prompt == format_prompt(_item("1"))

    def test_a_failing_answerer_propagates_rather_than_skewing_the_score(self) -> None:
        def boom(prompt: str) -> str:
            raise RuntimeError("provider down")

        with pytest.raises(RuntimeError, match="provider down"):
            evaluate_qa(boom, [_item("1")])

    def test_an_empty_run_has_a_defined_report(self) -> None:
        report = evaluate_qa(lambda p: "B", [])
        assert (report.n, report.accuracy, report.accuracy_ci95) == (0, 0.0, (0.0, 1.0))


class TestConfidenceInterval:
    def _report(self, n: int, correct: int) -> QAReport:
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        return QAReport(
            n=n, correct=correct, unanswered=0, started_at=now, ended_at=now
        )

    def test_matches_the_wilson_formula_at_a_known_point(self) -> None:
        # Wilson 95% for 50/100: (0.4038, 0.5962) -- standard tabulated value.
        lo, hi = self._report(100, 50).accuracy_ci95
        assert lo == pytest.approx(0.4038, abs=5e-4)
        assert hi == pytest.approx(0.5962, abs=5e-4)

    def test_is_bounded_and_narrows_with_n(self) -> None:
        lo_small, hi_small = self._report(20, 20).accuracy_ci95
        lo_big, hi_big = self._report(2000, 2000).accuracy_ci95
        assert hi_small == pytest.approx(1.0) and hi_big == pytest.approx(1.0)
        assert lo_big > lo_small


class _ScriptedLLM(BaseLLMProvider):
    sends_data_offsite = False

    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.kwargs.append(kwargs)
        return LLMResponse(
            text="B",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
        )

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        yield "B"

    def chat(self, messages: list[Any], **kwargs: Any) -> LLMResponse:
        return LLMResponse(text="B")


class TestLLMAnswerer:
    def test_drives_a_real_provider_and_sums_token_usage(self) -> None:
        provider = _ScriptedLLM()
        answerer = LLMAnswerer(provider, temperature=0)
        report = evaluate_qa(answerer, [_item("1", "B"), _item("2", "B")])
        assert report.correct == 2
        assert report.token_usage is not None
        assert report.token_usage.total_tokens == 22
        assert provider.kwargs == [{"temperature": 0}, {"temperature": 0}]

    def test_a_plain_function_has_no_token_usage(self) -> None:
        assert evaluate_qa(lambda p: "B", [_item("1")]).token_usage is None

    def test_a_reply_with_no_usage_leaves_usage_unset(self) -> None:
        class NoUsage(_ScriptedLLM):
            def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
                return LLMResponse(text="B")

        answerer = LLMAnswerer(NoUsage())
        assert evaluate_qa(answerer, [_item("1")]).token_usage is None


class TestManifest:
    def test_records_model_data_digest_and_counts_but_no_text(
        self, tmp_path: Path
    ) -> None:
        source = _write(
            tmp_path / "q.jsonl",
            [
                {
                    "question": "SECRETQUESTION",
                    "answer": "beta",
                    "options": {"A": "alpha", "B": "beta"},
                    "answer_idx": "B",
                }
            ],
        )
        provider = _ScriptedLLM()
        report = evaluate_qa(LLMAnswerer(provider), read_medqa_jsonl(source))
        manifest = qa_manifest(report, component=provider.provenance(), source=source)
        assert manifest.kind == "qa"
        assert manifest.component is not None
        assert manifest.component.class_name == "_ScriptedLLM"
        (digest,) = manifest.input_digests
        assert len(digest.sha256 or "") == 64 and digest.record_count == 1
        assert manifest.report["n"] == 1 and manifest.report["accuracy"] == 1.0
        dumped = manifest.model_dump_json()
        assert "SECRETQUESTION" not in dumped
        assert manifest.started_at == report.started_at

    def test_without_a_source_there_is_no_digest(self) -> None:
        manifest = qa_manifest(evaluate_qa(lambda p: "B", [_item("1")]))
        assert manifest.input_digests == [] and manifest.component is None

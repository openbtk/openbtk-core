"""Clinical multiple-choice QA evaluation, in the MedQA and MedMCQA formats
(FR-X-03).

This module scores a model on multiple-choice questions and reports accuracy.
It ships **no dataset and no accuracy figure**: MedQA and MedMCQA are public
but large, so the readers below take a local file the user already has and
never download anything, and no number for any model on either benchmark is
published by this project (CLAUDE.md rule 14: no claim without a harness -- and
running a real model is the user's to do).

**Formats, as verified against the datasets' Hub cards, not from memory.**

* MedQA (USMLE, ``GBaker/MedQA-USMLE-4-options``, CC-BY-4.0): JSON Lines,
  ``{"question", "answer", "options": {"A": ..., "B": ...}, "answer_idx"?}``.
  The key is ``answer_idx`` when present, otherwise the option whose text equals
  ``answer``.
* MedMCQA (``openlifescienceai/medmcqa``, Apache-2.0): JSON Lines with
  ``id, question, opa, opb, opc, opd, cop, subject_name``, where ``cop`` is the
  correct option as ``0``-``3`` (or ``a``-``d``). The public *test* split hides
  its answers (``cop = -1``); such a file is refused, not silently scored as 0.
  Some earlier MedMCQA releases number ``cop`` from 1; because ``1``-``3`` would
  be indistinguishable from the 0-indexed form, a value of ``4`` is refused as
  ambiguous rather than guessed at.

Reports hold counts and item ids only -- never question or answer text.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openbtk.core.errors import DatasetError
from openbtk.core.provenance import ComponentProvenance  # noqa: TC001
from openbtk.core.schemas import JsonValue, TokenUsage
from openbtk.eval.manifest import EvalManifest, build_manifest, file_digest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from openbtk.core.base import BaseLLMProvider

_LETTERS = "ABCDEFGHIJ"
_MEDMCQA_KEYS = ("opa", "opb", "opc", "opd")


class MCQItem(BaseModel):
    """One multiple-choice question.

    Example:
        >>> item = MCQItem(
        ...     item_id="q1",
        ...     question="Which vitamin deficiency causes scurvy?",
        ...     options={"A": "Vitamin A", "B": "Vitamin C"},
        ...     answer="B",
        ... )
        >>> item.options[item.answer]
        'Vitamin C'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    options: dict[str, str] = Field(..., min_length=2)
    answer: str = Field(..., description="The key of the correct option.")
    subject: str | None = None

    @model_validator(mode="after")
    def _answer_is_an_option(self) -> MCQItem:
        if self.answer not in self.options:
            raise ValueError(f"answer {self.answer!r} is not one of the option keys")
        return self


# ------------------------------------------------------------------ readers


def _lines(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        f = path.open(encoding="utf-8")
    except OSError as e:
        raise DatasetError(
            f"Could not open {path.name}.", context={"path": str(path)}
        ) from e
    with f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(
                    f"{path.name} line {n} is not valid JSON.",
                    context={"path": str(path), "line": n},
                ) from e
            if not isinstance(row, dict):
                raise DatasetError(
                    f"{path.name} line {n} is not a JSON object.",
                    context={"path": str(path), "line": n},
                )
            yield n, row


def read_medqa_jsonl(path: str | Path) -> Iterator[MCQItem]:
    """Stream MedQA-format questions from a JSON Lines file.

    Raises:
        DatasetError: On unreadable or malformed input, or a question whose
            correct option cannot be determined. The message names the line,
            never the question.
    """
    p = Path(path)
    for n, row in _lines(p):
        ctx = {"path": str(p), "line": n}
        options = row.get("options")
        if not isinstance(options, dict) or len(options) < 2:
            raise DatasetError(f"{p.name} line {n} has no options object.", context=ctx)
        options = {str(k): str(v) for k, v in options.items()}
        key = row.get("answer_idx")
        if key is None:
            matches = [k for k, v in options.items() if v == row.get("answer")]
            key = matches[0] if len(matches) == 1 else None
        if key is None or str(key) not in options:
            raise DatasetError(
                f"{p.name} line {n}: cannot determine the correct option.",
                context=ctx,
            )
        try:
            yield MCQItem(
                item_id=str(row.get("id", f"{p.stem}:{n}")),
                question=str(row.get("question", "")),
                options=options,
                answer=str(key),
                subject=row.get("meta_info"),
            )
        except ValueError as e:
            raise DatasetError(
                f"{p.name} line {n} is not a valid item.", context=ctx
            ) from e


def _medmcqa_answer(value: object, n: int, p: Path) -> str:
    ctx = {"path": str(p), "line": n}
    if isinstance(value, str) and value.strip().lower() in {"a", "b", "c", "d"}:
        return value.strip().upper()
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3:
        return "ABCD"[value]
    if value == -1:
        raise DatasetError(
            f"{p.name} line {n}: no answer key (cop = -1). MedMCQA's public test "
            "split hides its answers; use the validation split to score a model.",
            context=ctx,
        )
    raise DatasetError(
        f"{p.name} line {n}: unrecognised or ambiguous cop value; expected 0-3 or a-d.",
        context=ctx,
    )


def read_medmcqa_jsonl(path: str | Path) -> Iterator[MCQItem]:
    """Stream MedMCQA-format questions from a JSON Lines file.

    Raises:
        DatasetError: On unreadable or malformed input, a hidden answer key, or
            an ambiguous ``cop`` value (see the module docstring).
    """
    p = Path(path)
    for n, row in _lines(p):
        missing = [k for k in _MEDMCQA_KEYS if k not in row]
        if missing:
            raise DatasetError(
                f"{p.name} line {n} lacks option fields {missing}.",
                context={"path": str(p), "line": n},
            )
        answer = _medmcqa_answer(row.get("cop"), n, p)
        try:
            yield MCQItem(
                item_id=str(row.get("id", f"{p.stem}:{n}")),
                question=str(row.get("question", "")),
                options={"ABCD"[i]: str(row[k]) for i, k in enumerate(_MEDMCQA_KEYS)},
                answer=answer,
                subject=row.get("subject_name"),
            )
        except ValueError as e:
            raise DatasetError(
                f"{p.name} line {n} is not a valid item.",
                context={"path": str(p), "line": n},
            ) from e


# ------------------------------------------------------------------- prompt

_PROMPT = (
    "Answer the following multiple-choice question. Reply with only the letter "
    "of the single best option.\n\nQuestion: {question}\n\n{options}\n\nAnswer:"
)


def format_prompt(item: MCQItem) -> str:
    """The zero-shot prompt this module sends for ``item``.

    Example:
        >>> item = MCQItem(
        ...     item_id="q", question="Q?", options={"A": "x", "B": "y"}, answer="A"
        ... )
        >>> format_prompt(item).splitlines()[-4:]
        ['A. x', 'B. y', '', 'Answer:']
    """
    options = "\n".join(f"{k}. {v}" for k, v in item.options.items())
    return _PROMPT.format(question=item.question, options=options)


# "answer is B" / "Answer: B" (bare letter must be UPPER case, so "the answer is a
# bit unclear" is not read as option A) or "answer: (b)" (parenthesised, any case).
_ANSWER_IS = re.compile(r"(?i:\banswer)\s*(?:is|:)?\s*(?:\(([A-Ja-j])\)|([A-J])\b)")
# A reply that STARTS with the letter: "B", "(b)", "B.", "B) text", "b. text".
_LEADING = re.compile(r"^\W*\(?([A-J])\s*(?:[).:]\s|[).:]?$)", re.IGNORECASE)


def parse_choice(text: str, letters: Iterable[str]) -> str | None:
    """Extract the chosen option letter from a model's free-text reply.

    Recognises ``"B"``, ``"(B)"``, ``"B. text"``, ``"B) text"`` and
    ``"...the answer is B"`` / ``"Answer: B"``. It does *not* accept a bare
    letter buried in prose (``"A patient with ..."`` starts with the article
    "A"), so an evasive or rambling reply is counted as unanswered instead of
    being credited by accident. Returns ``None`` when no option can be
    identified, or when the letter is not one of ``letters``.

    Example:
        >>> parse_choice("The answer is C.", "ABCD")
        'C'
        >>> parse_choice("(b) Vitamin C", "ABCD")
        'B'
        >>> parse_choice("A patient presents with fever.", "ABCD") is None
        True
    """
    valid = {x.upper() for x in letters}
    stripped = text.strip()
    for pattern in (_ANSWER_IS, _LEADING):
        m = pattern.search(stripped)
        if m:
            letter = next(g for g in m.groups() if g).upper()
            return letter if letter in valid else None
    return None


# --------------------------------------------------------------- evaluation


class SubjectScore(BaseModel):
    """Counts for one subject (or one question source)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(..., ge=0)
    correct: int = Field(..., ge=0)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


class ItemResult(BaseModel):
    """One question's outcome: ids and letters only, never text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    predicted: str | None
    correct: bool


class QAReport(BaseModel):
    """Scores for one QA run. Counts and ids only.

    Unanswered questions (no parsable option) count as **wrong** and are also
    reported separately, so a model that rambles is visible rather than merely
    scoring low.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int = Field(..., ge=0)
    correct: int = Field(..., ge=0)
    unanswered: int = Field(..., ge=0)
    per_subject: dict[str, SubjectScore] = Field(default_factory=dict)
    results: list[ItemResult] = Field(default_factory=list)
    token_usage: TokenUsage | None = None
    started_at: datetime
    ended_at: datetime

    @property
    def accuracy(self) -> float:
        """Correct / n (0.0 for an empty run)."""
        return self.correct / self.n if self.n else 0.0

    @property
    def accuracy_ci95(self) -> tuple[float, float]:
        """95% Wilson score interval for ``accuracy``: how much a score of this
        size can move by chance alone. ``(0.0, 1.0)`` for an empty run."""
        if not self.n:
            return (0.0, 1.0)
        z, p = 1.959963984540054, self.accuracy
        denom = 1 + z * z / self.n
        centre = (p + z * z / (2 * self.n)) / denom
        half = z * math.sqrt(p * (1 - p) / self.n + z * z / (4 * self.n**2)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))

    def as_dict(self) -> dict[str, JsonValue]:
        """A JSON-serialisable summary (no per-item results), counts only."""
        lo, hi = self.accuracy_ci95
        return {
            "n": self.n,
            "correct": self.correct,
            "unanswered": self.unanswered,
            "accuracy": self.accuracy,
            "accuracy_ci95": [lo, hi],
            "per_subject": {
                k: {"n": v.n, "correct": v.correct, "accuracy": v.accuracy}
                for k, v in sorted(self.per_subject.items())
            },
        }


def evaluate_qa(
    answerer: Callable[[str], str],
    items: Iterable[MCQItem],
    *,
    keep_results: bool = False,
) -> QAReport:
    """Score ``answerer`` -- a function from a prompt to a reply -- on ``items``.

    ``items`` is consumed once and never materialised, so a full MedMCQA
    training split streams in constant memory. An exception from ``answerer``
    (a provider outage, say) propagates: a QA score silently computed over the
    questions that happened to succeed would be a wrong number.

    Args:
        answerer: ``prompt -> reply``. See :class:`LLMAnswerer`.
        items: Questions to ask.
        keep_results: Also record each question's id, chosen letter and
            correctness (ids and letters only).
    """
    started_at = datetime.now(UTC)
    n = correct = unanswered = 0
    subjects: Counter[str] = Counter()
    subject_correct: Counter[str] = Counter()
    results: list[ItemResult] = []
    for item in items:
        n += 1
        predicted = parse_choice(answerer(format_prompt(item)), item.options)
        hit = predicted == item.answer
        correct += hit
        unanswered += predicted is None
        if item.subject is not None:
            subjects[item.subject] += 1
            subject_correct[item.subject] += hit
        if keep_results:
            results.append(
                ItemResult(item_id=item.item_id, predicted=predicted, correct=hit)
            )
    usage = getattr(answerer, "usage", None)
    return QAReport(
        n=n,
        correct=correct,
        unanswered=unanswered,
        per_subject={
            s: SubjectScore(n=c, correct=subject_correct[s])
            for s, c in subjects.items()
        },
        results=results,
        token_usage=usage if isinstance(usage, TokenUsage) else None,
        started_at=started_at,
        ended_at=datetime.now(UTC),
    )


class LLMAnswerer:
    """Adapt a ``BaseLLMProvider`` to ``evaluate_qa``'s ``prompt -> reply``,
    accumulating its token usage across the run.

    Args:
        provider: The model under test.
        **generation: Passed to ``provider.generate`` on every call
            (provider-specific, e.g. ``temperature=0``).
    """

    def __init__(self, provider: BaseLLMProvider, **generation: Any) -> None:
        self.provider = provider
        self._generation = generation
        self.usage: TokenUsage | None = None

    def __call__(self, prompt: str) -> str:
        response = self.provider.generate(prompt, **self._generation)
        if response.usage is not None:
            self.usage = (
                response.usage if self.usage is None else self.usage + response.usage
            )
        return response.text


def qa_manifest(
    report: QAReport,
    *,
    component: ComponentProvenance | None = None,
    source: str | Path | None = None,
) -> EvalManifest:
    """The provenance record for a QA run (FR-X-06): which model, which file
    (by SHA-256), when, and the counts -- no question or answer text."""
    digests = [file_digest(source, record_count=report.n)] if source is not None else []
    return build_manifest(
        "qa",
        report.as_dict(),
        started_at=report.started_at,
        ended_at=report.ended_at,
        component=component,
        input_digests=digests,
    )

"""Adversarial-input timing (M11 security review, findings S-2, S-4 and S-5).

A regular expression that is quadratic in its input turns one crafted document
into a stalled worker: ~200 kB of ``a.a.a.`` kept the rule recognizer busy for
over a minute. These tests feed each text-facing pattern inputs built to trigger
super-linear backtracking and require it to finish inside a budget that is
generous for linear code (a fraction of a second here) and far below what the
quadratic versions took (tens of seconds to minutes at these sizes). They
assert an upper bound, not a benchmark, so they are not flaky on a slow runner.

The correctness tests beside them pin what the fixes must *not* change.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from openbtk.core.schemas import TextSpan
from openbtk.data.clinical_text.chunking import SectionAwareChunker, _split_sentences
from openbtk.data.clinical_text.schemas import ClinicalTextRecord
from openbtk.deid.recognizers.rule import RuleRecognizer
from openbtk.deid.schemas import PHICategory
from openbtk.eval.qa import parse_choice

if TYPE_CHECKING:
    from collections.abc import Callable

BUDGET_SECONDS = 3.0
SIZE = 100_000


def _within_budget(work: Callable[[], object]) -> None:
    started = time.perf_counter()
    work()
    elapsed = time.perf_counter() - started
    assert elapsed < BUDGET_SECONDS, f"took {elapsed:.1f}s (budget {BUDGET_SECONDS}s)"


def _repeat(unit: str, size: int = SIZE) -> str:
    return (unit * (size // len(unit) + 1))[:size]


# Each unit is a shape that stresses one of the recognizer's patterns.
_ADVERSARIAL_UNITS = [
    "a.",  # email local part: a word boundary at every dot
    "a-",  # ... and at every dash
    "a%",
    "1.",  # IPv4
    "1/",  # date
    "1",  # long digit run (account number, MRN)
    "A",  # long alnum run (beneficiary / vehicle id)
    "a@",  # many '@' with no valid domain
    "@a",
    "(",  # phone
    "http://a",  # URL
    "SN-",  # prefixed identifiers
    "MRN-",
    " ",
]


@pytest.mark.parametrize("unit", _ADVERSARIAL_UNITS)
def test_the_rule_recognizer_is_linear_on_adversarial_text(unit: str) -> None:
    text = _repeat(unit)
    _within_budget(lambda: RuleRecognizer().detect(text))


def test_a_very_long_dotted_local_part_is_still_matched_whole() -> None:
    """The fix anchors where a match may START; it must not cap how much of a
    long local part is matched, or the head of the address would survive
    redaction."""
    local = ".".join(["part"] * 200)  # 999 characters
    text = f"contact {local}@example.org now"
    hits = [d for d in RuleRecognizer().detect(text) if d.category is PHICategory.EMAIL]
    assert len(hits) == 1
    span = hits[0].span
    assert text[span.start : span.end] == f"{local}@example.org"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail jo.smith@example.org please", "jo.smith@example.org"),
        ("(a_b+tag@sub.example.co.uk)", "a_b+tag@sub.example.co.uk"),
        ("to:x@y.io, cc:z@w.io", "x@y.io"),
        ("x.y@z.com.", "x.y@z.com"),
    ],
)
def test_ordinary_email_shapes_are_still_detected(text: str, expected: str) -> None:
    hits = [
        text[d.span.start : d.span.end]
        for d in RuleRecognizer().detect(text)
        if d.category is PHICategory.EMAIL
    ]
    assert expected in hits


def test_parse_choice_is_linear_on_a_long_run_of_spaces() -> None:
    reply = "answer" + " " * SIZE + "!"
    _within_budget(lambda: parse_choice(reply, "ABCD"))


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("The answer is C.", "C"),
        ("Answer: B", "B"),
        ("answer   is   (d)", "D"),
        ("answer B", "B"),
        ("answer:(a)", "A"),
        ("the answer is a bit unclear", None),
    ],
)
def test_parse_choice_readings_are_unchanged(reply: str, expected: str | None) -> None:
    assert parse_choice(reply, "ABCD") == expected


def _sectioned(text: str) -> ClinicalTextRecord:
    return ClinicalTextRecord(
        record_id="r1",
        source="synthetic",
        text=text,
        sections={"body": TextSpan(start=0, end=len(text), label="s", confidence=1.0)},
    )


def _chunk_all(text: str) -> list[str]:
    chunker = SectionAwareChunker(max_tokens=10_000_000)
    return [c.text for c in chunker.chunk(_sectioned(text))]


@pytest.mark.parametrize(
    "text",
    [
        "a" * 40_000 + ".",  # one long word, then a sentence end
        "Dr. " * 10_000,  # a run of abbreviations: `start` never advances
        "x. " * 13_000,  # many real sentences
    ],
    ids=["long-word", "abbreviation-run", "many-sentences"],
)
def test_the_sentence_splitter_is_linear_on_adversarial_text(text: str) -> None:
    _within_budget(lambda: _chunk_all(text))


def test_abbreviations_still_do_not_end_a_sentence() -> None:
    """``Dr.`` and ``mg.`` keep the sentence going; a real full stop ends it."""
    text = "Seen by Dr. Lee today. Plan: 5 mg. daily."
    spans = _split_sentences(text)
    assert [text[a:b] for a, b in spans] == [
        "Seen by Dr. Lee today. ",
        "Plan: 5 mg. daily.",
    ]


# ------------------------------------------------------------ dose guardrail (FR-G-05)

_DOSE_TEXTS = {
    "drug-and-number-repeated": "examplamine 5 ",
    "drug-repeated": "examplamine ",
    "digits": "1",
    "digits-and-dots": "1.",
    "clause-breaks": "examplamine 5 mg. ",
    "frequency-words": "every 6 hours ",
    "spaces": " ",
    "ranges": "1-2 ",
}


@pytest.mark.parametrize("name", sorted(_DOSE_TEXTS))
def test_the_dose_guardrail_is_linear_on_adversarial_text(name: str) -> None:
    from openbtk.guardrails.dose import DoseLimit, DosePlausibilityGuardrail

    guardrail = DosePlausibilityGuardrail(
        limits=[DoseLimit(drug="examplamine", max_single=1, source="synthetic")]
    )
    text = _repeat(_DOSE_TEXTS[name])
    _within_budget(lambda: guardrail.check(text))


# ------------------------------------------------------------ HL7 v2 helpers (FR-E-03)


@pytest.mark.parametrize(
    "text",
    ["1" * SIZE, "1" * 4 + "+" * SIZE, "2024" + "0" * SIZE, ("^" * SIZE)],
    ids=["digits", "offset-run", "zeros", "separators"],
)
def test_the_hl7_field_helpers_are_linear_on_adversarial_text(text: str) -> None:
    from openbtk.data.ehr import hl7v2

    def work() -> None:
        hl7v2._timestamp(text, 0)
        hl7v2._date(text)
        hl7v2._coding(text)
        hl7v2._component(text, 3)
        hl7v2._patient_id(text)
        hl7v2._unescape(text)

    _within_budget(work)


@pytest.mark.parametrize(
    "text",
    ["MSH|" * (SIZE // 4), "\r" * SIZE, "MSH|\r" * (SIZE // 5), "\\" * SIZE],
    ids=["msh-run", "line-breaks", "msh-lines", "backslashes"],
)
def test_the_hl7_message_splitter_is_linear_on_adversarial_text(text: str) -> None:
    from openbtk.data.ehr import hl7v2

    _within_budget(lambda: list(hl7v2._split_messages(text)))

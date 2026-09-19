"""Unit tests for openbtk.guardrails.groundedness.GroundednessGuardrail."""

from __future__ import annotations

from openbtk.core.schemas import GuardrailSeverity
from openbtk.guardrails.groundedness import (
    GroundednessCheckInput,
    GroundednessGuardrail,
)


class TestUnrelatedPayload:
    def test_non_input_payload_passes_with_info(self) -> None:
        guardrail = GroundednessGuardrail()
        result = guardrail.check("plain string")
        assert result.passed is True
        assert result.severity == GuardrailSeverity.INFO

    def test_none_payload_passes(self) -> None:
        assert GroundednessGuardrail().check(None).passed is True


class TestGroundedClaims:
    def test_fully_supported_answer_passes(self) -> None:
        guardrail = GroundednessGuardrail()
        payload = GroundednessCheckInput(
            answer="The patient has type 2 diabetes.",
            context=["Assessment: type 2 diabetes mellitus, stable."],
        )
        result = guardrail.check(payload)
        assert result.passed is True

    def test_empty_answer_has_no_claims(self) -> None:
        guardrail = GroundednessGuardrail()
        result = guardrail.check(GroundednessCheckInput(answer="   "))
        assert result.passed is True
        assert "No claims" in result.message

    def test_non_substantive_claim_is_always_supported(self) -> None:
        """A claim with no significant words at all (every word is a
        stopword or too short) can't be meaningfully checked against
        context -- treated as supported rather than penalised."""
        guardrail = GroundednessGuardrail()
        payload = GroundednessCheckInput(answer="It is.", context=[])
        assert guardrail.check(payload).passed is True


class TestUnsupportedClaims:
    def test_unsupported_claim_blocks_with_a_span(self) -> None:
        guardrail = GroundednessGuardrail()
        payload = GroundednessCheckInput(
            answer="The patient has a fractured femur.",
            context=["Assessment: type 2 diabetes mellitus, stable."],
        )
        result = guardrail.check(payload)
        assert result.passed is False
        assert result.severity == GuardrailSeverity.BLOCK
        assert len(result.spans) == 1
        span = result.spans[0]
        assert (
            payload.answer[span.start : span.end]
            == "The patient has a fractured femur."
        )

    def test_substantive_claim_with_empty_context_is_unsupported(self) -> None:
        guardrail = GroundednessGuardrail()
        payload = GroundednessCheckInput(answer="The patient has pneumonia.")
        assert guardrail.check(payload).passed is False

    def test_multiple_sentences_only_flags_the_unsupported_one(self) -> None:
        guardrail = GroundednessGuardrail()
        payload = GroundednessCheckInput(
            answer=(
                "The patient has type 2 diabetes. "
                "The patient also has a fractured femur."
            ),
            context=["Assessment: type 2 diabetes mellitus, stable."],
        )
        result = guardrail.check(payload)
        assert result.passed is False
        assert len(result.spans) == 1
        assert "fractured femur" in payload.answer[result.spans[0].start :]

    def test_span_falls_back_to_whole_text_when_claim_not_found_verbatim(self) -> None:
        """Covers the defensive .find() == -1 branch directly, via a claim
        that a custom decomposer invents rather than lifts verbatim from
        the answer."""
        guardrail = GroundednessGuardrail(
            decompose_claims=lambda text: ["a claim not present in the answer"]
        )
        payload = GroundednessCheckInput(answer="Something else entirely.", context=[])
        result = guardrail.check(payload)
        assert result.passed is False
        assert result.spans[0].start == 0
        assert result.spans[0].end == len(payload.answer)


class TestCustomCallables:
    def test_custom_decompose_claims_is_used(self) -> None:
        guardrail = GroundednessGuardrail(
            decompose_claims=lambda text: [text],
            is_supported=lambda claim, context: True,
        )
        payload = GroundednessCheckInput(answer="anything at all", context=[])
        assert guardrail.check(payload).passed is True

    def test_custom_is_supported_overrides_the_default_heuristic(self) -> None:
        guardrail = GroundednessGuardrail(is_supported=lambda claim, context: False)
        payload = GroundednessCheckInput(
            answer="The patient has type 2 diabetes.",
            context=["Assessment: type 2 diabetes mellitus, stable."],
        )
        assert guardrail.check(payload).passed is False

    def test_custom_support_threshold(self) -> None:
        guardrail_strict = GroundednessGuardrail(support_threshold=1.0)
        guardrail_loose = GroundednessGuardrail(support_threshold=0.1)
        payload = GroundednessCheckInput(
            answer="The patient has diabetes and hypertension.",
            context=["The patient has diabetes."],
        )
        assert guardrail_strict.check(payload).passed is False
        assert guardrail_loose.check(payload).passed is True


class TestRegistration:
    def test_registered_under_the_expected_key(self) -> None:
        assert GroundednessGuardrail.registry_key == "guardrail.general.groundedness"

    def test_provenance_is_serialisable(self) -> None:
        assert isinstance(GroundednessGuardrail().provenance().model_dump_json(), str)

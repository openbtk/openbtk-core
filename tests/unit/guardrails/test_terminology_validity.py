"""Unit tests for openbtk.guardrails.terminology_validity."""

from __future__ import annotations

from openbtk.core.base import BaseTerminologyService
from openbtk.core.errors import TerminologyError
from openbtk.core.schemas import CodeSystem, Concept, GuardrailSeverity
from openbtk.data.ehr.schemas import (
    CodedEvent,
    Demographics,
    Measurement,
    PatientRecord,
)
from openbtk.guardrails.terminology_validity import TerminologyValidityGuardrail


def _patient(**overrides: object) -> PatientRecord:
    defaults: dict[str, object] = {
        "patient_id": "pt-1",
        "demographics": Demographics(),
        "source_system": "fhir-r4",
    }
    defaults.update(overrides)
    return PatientRecord(**defaults)  # type: ignore[arg-type]


class _FullVocabulary(BaseTerminologyService):
    """A complete vocabulary for its systems: absence IS a real answer, so a
    missing code is genuinely invalid (the default is_authoritative == True)."""

    def __init__(self, *valid: tuple[str, CodeSystem]) -> None:
        self._valid = set(valid)

    def resolve(self, code: str, system: CodeSystem) -> Concept | None:
        if (code, system) in self._valid:
            return Concept(code=code, system=system, display=code)
        return None

    def validate(self, code: str, system: CodeSystem) -> bool:
        return (code, system) in self._valid

    def map(
        self, code: str, from_system: CodeSystem, to_system: CodeSystem
    ) -> list[Concept]:
        return []


class _AlwaysErrorsBackend(BaseTerminologyService):
    def resolve(self, code: str, system: CodeSystem) -> Concept | None:
        raise TerminologyError("backend unavailable")

    def validate(self, code: str, system: CodeSystem) -> bool:
        raise TerminologyError("backend unavailable")

    def map(
        self, code: str, from_system: CodeSystem, to_system: CodeSystem
    ) -> list[Concept]:
        return []


class TestBareTuplePayload:
    def test_valid_code_passes(self) -> None:
        guardrail = TerminologyValidityGuardrail()
        result = guardrail.check(("E11.9", CodeSystem.ICD10CM))
        assert result.passed is True

    def test_a_code_missing_from_a_complete_vocabulary_blocks(self) -> None:
        guardrail = TerminologyValidityGuardrail(
            terminology=_FullVocabulary(("E11.9", CodeSystem.ICD10CM))
        )
        result = guardrail.check(("Z99.999", CodeSystem.ICD10CM))
        assert result.passed is False
        assert result.severity == GuardrailSeverity.BLOCK
        assert "ICD10CM:Z99.999" in result.details["invalid"]


class TestPartialVocabularyCannotAccuse:
    """Regression: the default backend is a small SUBSET. Before
    is_authoritative existed, any real code outside it (every SNOMED and LOINC
    code, and most ICD-10-CM ones) was reported as 'does not exist' and BLOCKED,
    so the default guardrail halted pipelines on valid data. A partial
    vocabulary can fail to confirm a code; it cannot call it invalid."""

    def test_an_unconfirmed_code_is_a_warning_not_a_block(self) -> None:
        result = TerminologyValidityGuardrail().check(("Z99.999", CodeSystem.ICD10CM))
        assert result.passed is False
        assert result.severity == GuardrailSeverity.WARNING
        assert result.details["unverifiable"] == ["ICD10CM:Z99.999"]
        assert "invalid" not in result.details
        assert "do not exist" not in result.message

    def test_a_real_snomed_code_is_never_called_invalid_by_the_bundled_subset(
        self,
    ) -> None:
        result = TerminologyValidityGuardrail().check(("73211009", CodeSystem.SNOMED))
        assert result.severity == GuardrailSeverity.WARNING
        assert result.details["unverifiable"] == ["SNOMED:73211009"]

    def test_a_confirmed_code_still_passes_alongside_unconfirmed_ones(self) -> None:
        result = TerminologyValidityGuardrail().check(
            [("E11.9", CodeSystem.ICD10CM), ("73211009", CodeSystem.SNOMED)]
        )
        assert result.severity == GuardrailSeverity.WARNING
        assert result.details["unverifiable"] == ["SNOMED:73211009"]

    def test_a_definitely_invalid_code_still_outranks_an_unconfirmed_one(self) -> None:
        guardrail = TerminologyValidityGuardrail(
            terminology=_FullVocabulary(("E11.9", CodeSystem.ICD10CM))
        )
        result = guardrail.check(
            [("Z99.999", CodeSystem.ICD10CM), ("E11.9", CodeSystem.ICD10CM)]
        )
        assert result.severity == GuardrailSeverity.BLOCK


class TestCodedEventPayload:
    def test_single_coded_event(self) -> None:
        guardrail = TerminologyValidityGuardrail()
        event = CodedEvent(code="E11.9", system=CodeSystem.ICD10CM)
        assert guardrail.check(event).passed is True

    def test_single_measurement(self) -> None:
        guardrail = TerminologyValidityGuardrail()
        m = Measurement(code="Z99.999", system=CodeSystem.ICD10CM)
        assert guardrail.check(m).passed is False


class TestListPayload:
    def test_list_of_coded_events(self) -> None:
        guardrail = TerminologyValidityGuardrail()
        events = [
            CodedEvent(code="E11.9", system=CodeSystem.ICD10CM),
            CodedEvent(code="I10", system=CodeSystem.ICD10CM),
        ]
        result = guardrail.check(events)
        assert result.passed is True
        assert "2 code(s) valid" in result.message

    def test_list_with_one_invalid(self) -> None:
        guardrail = TerminologyValidityGuardrail(
            terminology=_FullVocabulary(("E11.9", CodeSystem.ICD10CM))
        )
        events = [
            CodedEvent(code="E11.9", system=CodeSystem.ICD10CM),
            CodedEvent(code="bogus", system=CodeSystem.ICD10CM),
        ]
        result = guardrail.check(events)
        assert result.passed is False


class TestPatientRecordPayload:
    def test_all_valid_codes_across_every_field(self) -> None:
        guardrail = TerminologyValidityGuardrail()
        patient = _patient(
            conditions=[CodedEvent(code="E11.9", system=CodeSystem.ICD10CM)],
            medications=[CodedEvent(code="I10", system=CodeSystem.ICD10CM)],
            procedures=[CodedEvent(code="J18.9", system=CodeSystem.ICD10CM)],
            observations=[Measurement(code="K21.9", system=CodeSystem.ICD10CM)],
        )
        result = guardrail.check(patient)
        assert result.passed is True

    def test_an_invalid_code_anywhere_blocks(self) -> None:
        guardrail = TerminologyValidityGuardrail(
            terminology=_FullVocabulary(("E11.9", CodeSystem.ICD10CM))
        )
        patient = _patient(
            observations=[Measurement(code="not-real", system=CodeSystem.ICD10CM)]
        )
        result = guardrail.check(patient)
        assert result.passed is False


class TestUnrelatedPayload:
    def test_no_codes_found_passes_with_info(self) -> None:
        guardrail = TerminologyValidityGuardrail()
        result = guardrail.check("just a plain string")
        assert result.passed is True
        assert result.severity == GuardrailSeverity.INFO
        assert "No codes" in result.message

    def test_none_payload_passes(self) -> None:
        guardrail = TerminologyValidityGuardrail()
        assert guardrail.check(None).passed is True

    def test_list_of_unrelated_items_passes(self) -> None:
        guardrail = TerminologyValidityGuardrail()
        assert guardrail.check([1, 2, 3]).passed is True


class TestBackendErrors:
    def test_terminology_error_becomes_a_warning_not_a_block(self) -> None:
        guardrail = TerminologyValidityGuardrail(terminology=_AlwaysErrorsBackend())
        result = guardrail.check(("X", CodeSystem.SNOMED))
        assert result.passed is False
        assert result.severity == GuardrailSeverity.WARNING
        assert "SNOMED:X" in result.details["unverifiable"]

    def test_invalid_code_takes_priority_over_an_error_elsewhere(self) -> None:
        class _MixedBackend(BaseTerminologyService):
            def resolve(self, code: str, system: CodeSystem) -> Concept | None:
                return None

            def validate(self, code: str, system: CodeSystem) -> bool:
                if code == "errors":
                    raise TerminologyError("nope")
                return code == "valid"

            def map(
                self, code: str, from_system: CodeSystem, to_system: CodeSystem
            ) -> list[Concept]:
                return []

        guardrail = TerminologyValidityGuardrail(terminology=_MixedBackend())
        result = guardrail.check(
            [
                ("valid", CodeSystem.SNOMED),
                ("invalid", CodeSystem.SNOMED),
                ("errors", CodeSystem.SNOMED),
            ]
        )
        assert result.passed is False
        assert result.severity == GuardrailSeverity.BLOCK
        assert "SNOMED:invalid" in result.details["invalid"]
        assert "SNOMED:errors" in result.details["unverifiable"]


class TestRegistration:
    def test_registered_under_the_expected_key(self) -> None:
        assert (
            TerminologyValidityGuardrail.registry_key == "guardrail.general.terminology"
        )

    def test_provenance_is_serialisable(self) -> None:
        guardrail = TerminologyValidityGuardrail()
        assert isinstance(guardrail.provenance().model_dump_json(), str)

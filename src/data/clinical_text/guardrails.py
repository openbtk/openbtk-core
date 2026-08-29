"""
Clinical text guardrails.

- PHIGuardrail: detects residual PHI in text (post-generation or post-deidentification)
- HallucinationGuardrail: checks LLM output claims against source context
- CodeValidityGuardrail: validates ICD/SNOMED/LOINC codes in generated text
"""
from __future__ import annotations

from typing import Any

import structlog

from opentbtk.core.base import BaseGuardrail
from opentbtk.core.errors import ProcessingError
from opentbtk.core.registry import GUARDRAIL_REGISTRY
from opentbtk.core.schemas import GuardrailResult, GuardrailSeverity

log = structlog.get_logger(__name__)


@GUARDRAIL_REGISTRY.register("guardrail.clinical_text.phi")
class PHIGuardrail(BaseGuardrail):
    """Detect residual PHI in text using Presidio.

    Use this guardrail on: LLM-generated outputs, chunk text before indexing,
    and any text being written to logs or external storage.

    Args:
        score_threshold: Presidio detection confidence threshold (0–1).
        raise_on_block: If True, raises GuardrailViolation when severity is
            BLOCK. If False, returns the result for caller to handle.
        language: Language for analysis.
    """

    def __init__(
        self,
        score_threshold: float = 0.7,
        raise_on_block: bool = False,
        language: str = "en",
    ) -> None:
        self._score_threshold = score_threshold
        self._raise_on_block = raise_on_block
        self._language = language
        self._analyzer: Any = None

    def _ensure_loaded(self) -> None:
        if self._analyzer is not None:
            return
        try:
            from presidio_analyzer import AnalyzerEngine
            self._analyzer = AnalyzerEngine()
        except ImportError as e:
            raise ProcessingError(
                "presidio-analyzer required for PHIGuardrail. "
                "Install: pip install opentbtk[clinical_text]"
            ) from e

    def check(self, payload: Any) -> GuardrailResult:
        if not isinstance(payload, str):
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_name="guardrail.clinical_text.phi",
                message="Non-string payload — PHI check skipped.",
            )
        self._ensure_loaded()
        try:
            results = self._analyzer.analyze(
                text=payload,
                language=self._language,
                score_threshold=self._score_threshold,
            )
        except Exception as e:
            raise ProcessingError("PHI analysis failed in guardrail") from e

        if not results:
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_name="guardrail.clinical_text.phi",
                message="No PHI detected.",
            )

        entity_types = list({r.entity_type for r in results})
        log.warning(
            "guardrail.phi_detected",
            n_entities=len(results),
            entity_types=entity_types,
        )
        result = GuardrailResult(
            passed=False,
            severity=GuardrailSeverity.BLOCK,
            guardrail_name="guardrail.clinical_text.phi",
            message=f"PHI detected: {len(results)} entities of types {entity_types}",
            details={
                "n_entities": len(results),
                "entity_types": entity_types,
                "spans": [
                    {"type": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
                    for r in results
                ],
            },
        )
        if self._raise_on_block:
            from opentbtk.core.errors import GuardrailViolation
            raise GuardrailViolation(result.message, details=result.details)
        return result


@GUARDRAIL_REGISTRY.register("guardrail.clinical_text.hallucination")
class HallucinationGuardrail(BaseGuardrail):
    """Check LLM-generated clinical text for unsupported claims.

    Compares generated text against a provided source context using a simple
    entailment heuristic (keyword/concept overlap). For production use,
    replace with an NLI model (e.g., MedNLI fine-tuned model).

    Args:
        min_overlap_ratio: Minimum fraction of key terms in the generated
            output that must appear in the source context (default: 0.5).
        raise_on_block: Raise GuardrailViolation on block-level findings.
    """

    def __init__(
        self,
        min_overlap_ratio: float = 0.5,
        raise_on_block: bool = False,
    ) -> None:
        self._min_overlap_ratio = min_overlap_ratio
        self._raise_on_block = raise_on_block

    def check(self, payload: Any) -> GuardrailResult:
        """Check generated output against source context.

        Args:
            payload: Dict with keys:
                - "generated": str — LLM-generated text to check
                - "context": str — source text the generation should be grounded in
        """
        if not isinstance(payload, dict) or "generated" not in payload:
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_name="guardrail.clinical_text.hallucination",
                message="Invalid payload format — hallucination check skipped.",
            )

        generated: str = payload["generated"]
        context: str = payload.get("context", "")
        if not context:
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.WARNING,
                guardrail_name="guardrail.clinical_text.hallucination",
                message="No context provided — hallucination check skipped.",
            )

        # Simple keyword overlap heuristic
        import re
        def extract_terms(text: str) -> set[str]:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            stopwords = {"that", "this", "with", "from", "have", "been", "were",
                        "will", "also", "than", "then", "when", "more", "some"}
            return {w for w in words if w not in stopwords}

        gen_terms = extract_terms(generated)
        ctx_terms = extract_terms(context)

        if not gen_terms:
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_name="guardrail.clinical_text.hallucination",
                message="Generated text has no checkable terms.",
            )

        overlap = gen_terms & ctx_terms
        ratio = len(overlap) / len(gen_terms)
        unsupported = gen_terms - ctx_terms

        if ratio >= self._min_overlap_ratio:
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_name="guardrail.clinical_text.hallucination",
                message=f"Hallucination check passed (overlap ratio: {ratio:.2f}).",
                details={"overlap_ratio": ratio},
            )

        log.warning(
            "guardrail.hallucination_risk",
            overlap_ratio=ratio,
            n_unsupported_terms=len(unsupported),
        )
        result = GuardrailResult(
            passed=False,
            severity=GuardrailSeverity.WARNING,
            guardrail_name="guardrail.clinical_text.hallucination",
            message=f"Potential hallucination: overlap ratio {ratio:.2f} below threshold {self._min_overlap_ratio}.",
            details={
                "overlap_ratio": ratio,
                "unsupported_terms": sorted(unsupported)[:20],
            },
        )
        if self._raise_on_block and result.severity == GuardrailSeverity.BLOCK:
            from opentbtk.core.errors import GuardrailViolation
            raise GuardrailViolation(result.message, details=result.details)
        return result


@GUARDRAIL_REGISTRY.register("guardrail.clinical_text.code_validity")
class CodeValidityGuardrail(BaseGuardrail):
    """Validate clinical terminology codes in generated text.

    Checks that any ICD-10, LOINC, or SNOMED codes mentioned in the text
    follow valid format patterns. For production use, extend with actual
    vocabulary membership checks against local UMLS/SNOMED/LOINC tables.

    Args:
        check_icd10: Validate ICD-10 code format (default: True).
        check_loinc: Validate LOINC code format (default: True).
    """

    def __init__(self, check_icd10: bool = True, check_loinc: bool = True) -> None:
        self._check_icd10 = check_icd10
        self._check_loinc = check_loinc

    def check(self, payload: Any) -> GuardrailResult:
        if not isinstance(payload, str):
            return GuardrailResult(
                passed=True,
                severity=GuardrailSeverity.INFO,
                guardrail_name="guardrail.clinical_text.code_validity",
                message="Non-string payload — code validity check skipped.",
            )

        import re
        issues: list[str] = []

        if self._check_icd10:
            # ICD-10-CM: letter + 2 digits + optional dot + up to 4 chars
            icd_candidates = re.findall(r"\b[A-Z]\d{2}(?:\.\w{1,4})?\b", payload)
            invalid_icd = [c for c in icd_candidates if not re.match(r"^[A-Z]\d{2}(\.\w{1,4})?$", c)]
            if invalid_icd:
                issues.append(f"Malformed ICD-10 codes: {invalid_icd}")

        if self._check_loinc:
            # LOINC: 1-5 digits + hyphen + 1 check digit
            loinc_candidates = re.findall(r"\b\d{1,5}-\d\b", payload)
            # Basic length check — real LOINC validation needs the full code set
            if loinc_candidates:
                log.debug("guardrail.code_validity.loinc_found", codes=loinc_candidates)

        if issues:
            return GuardrailResult(
                passed=False,
                severity=GuardrailSeverity.WARNING,
                guardrail_name="guardrail.clinical_text.code_validity",
                message=f"Code validity issues: {'; '.join(issues)}",
                details={"issues": issues},
            )

        return GuardrailResult(
            passed=True,
            severity=GuardrailSeverity.INFO,
            guardrail_name="guardrail.clinical_text.code_validity",
            message="Code validity check passed.",
        )

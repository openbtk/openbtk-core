"""Shared schemas used across every modality and cross-cutting service.

These are the types that appear in more than one place: a guardrail's result,
a span of text, an entity linked to a terminology code, a retrieval hit, and
the provenance pointer that ties a retrieved chunk back to its source record.
Modality-specific schemas (``ClinicalTextRecord``, ``PatientRecord``, ...)
live in their own modules and import from here, never the reverse.

Every schema is frozen and rejects unknown fields (docs/09_CODING_STANDARDS.md
section 5): records are immutable, and a typo'd field is a construction error,
not a silently dropped value.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypeAliasType

# A JSON-serialisable value. Used for "details"/"metadata"-style fields that
# are deliberately open-ended (guardrail details, search result metadata)
# without falling back to a bare `Any`.
#
# This MUST be wrapped in TypeAliasType, not a bare `X | Y` assignment. A bare
# assignment builds a UnionType object that embeds a forward reference to its
# own name; resolving it re-evaluates the same object and recurses forever
# (RecursionError, discovered via this module's own doctest run). TypeAliasType
# is the Pydantic-documented fix and is what PEP 695's native `type` statement
# compiles to on 3.12+ -- this backport (via typing_extensions) is required
# because openbtk's floor is Python 3.11.
JsonValue = TypeAliasType(
    "JsonValue",
    "str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None",
)


class TextSpan(BaseModel):
    """A character range within some text, with a label and confidence.

    Used for de-identification detections, linked-entity mentions, and
    guardrail-flagged regions alike -- one span type serves all of them so
    downstream code has a single shape to reason about.

    Example:
        >>> span = TextSpan(start=10, end=14, label="DATE", confidence=0.97)
        >>> span.end - span.start
        4
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(..., ge=0, description="Start offset, inclusive.")
    end: int = Field(..., ge=0, description="End offset, exclusive.")
    label: str = Field(..., min_length=1, description="What this span represents.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detector confidence in [0, 1]."
    )


class GuardrailSeverity(StrEnum):
    """How seriously a guardrail result should be treated."""

    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"


class GuardrailResult(BaseModel):
    """The outcome of one ``BaseGuardrail.check()`` call.

    A guardrail never raises on a failed check (docs/04_API_DESIGN.md section
    3) -- it always returns one of these. Whether a ``BLOCK`` severity halts
    the pipeline is a policy decision made by the caller, not by the
    guardrail itself.

    Example:
        >>> result = GuardrailResult(
        ...     passed=False,
        ...     severity=GuardrailSeverity.BLOCK,
        ...     guardrail_key="guardrail.general.phi_leakage",
        ...     message="Detected a phone number in generated output.",
        ... )
        >>> result.passed
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool = Field(..., description="Whether the payload satisfied the check.")
    severity: GuardrailSeverity = Field(
        ..., description="Severity to apply if the check failed."
    )
    guardrail_key: str = Field(
        ..., description="Registry key of the guardrail that produced this result."
    )
    message: str = Field(..., min_length=1, description="Human-readable outcome.")
    spans: list[TextSpan] = Field(
        default_factory=list, description="Offending regions, if applicable."
    )
    details: dict[str, JsonValue] = Field(
        default_factory=dict, description="Structured, guardrail-specific detail."
    )


class LinkedEntity(BaseModel):
    """A clinical entity mention linked to one or more terminology codes.

    ``negated`` / ``uncertain`` / ``historical`` / ``subject`` carry the
    ConText-style assertions (docs/11_GLOSSARY.md Part 4) that determine
    whether a mention should be treated as a present, first-person finding --
    getting these wrong inverts clinical meaning, so they are first-class
    fields rather than free-text metadata.

    Example:
        >>> entity = LinkedEntity(
        ...     text="diabetes",
        ...     span=TextSpan(start=0, end=8, label="CONDITION", confidence=0.9),
        ...     cui="C0011849",
        ...     codes={"SNOMED": "73211009"},
        ... )
        >>> entity.subject
        'patient'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(..., min_length=1, description="The mention's surface text.")
    span: TextSpan = Field(..., description="Location of the mention in its source.")
    cui: str | None = Field(None, description="UMLS Concept Unique Identifier.")
    codes: dict[str, str] = Field(
        default_factory=dict,
        description='Code-system name to code, e.g. {"SNOMED": "73211009"}.',
    )
    negated: bool = Field(False, description="True if the mention is negated.")
    uncertain: bool = Field(False, description="True if the mention is hedged.")
    historical: bool = Field(False, description="True if the mention is historical.")
    subject: Literal["patient", "family", "other"] = Field(
        "patient", description="Who the mention is about."
    )


class SourceRef(BaseModel):
    """A pointer back from a retrieved item to its exact source location.

    This is what makes verifiable citations possible: a retrieved chunk
    carries a ``SourceRef`` so a ``GroundednessGuardrail`` -- or a human
    reviewer -- can check a generated claim against the precise offsets in
    the original record, not just "some document somewhere."

    Example:
        >>> ref = SourceRef(record_id="rec-1", chunk_id="chunk-3")
        >>> ref.char_start is None
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(..., min_length=1, description="Parent record identifier.")
    chunk_id: str | None = Field(None, description="Parent chunk identifier, if any.")
    char_start: int | None = Field(
        None, ge=0, description="Start offset in the source record, if known."
    )
    char_end: int | None = Field(
        None, ge=0, description="End offset in the source record, if known."
    )


class SearchResult(BaseModel):
    """One hit returned from a vector store or reranker query.

    Example:
        >>> hit = SearchResult(id="chunk-3", score=0.82, metadata={"section": "Plan"})
        >>> hit.source is None
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1, description="Identifier of the matched item.")
    score: float = Field(..., description="Similarity or relevance score.")
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict, description="Store-supplied metadata for this item."
    )
    source: SourceRef | None = Field(
        None, description="Provenance back to the originating record, if known."
    )


class CodeSystem(StrEnum):
    """A clinical terminology or unit-of-measure system.

    Named exactly per docs/11_GLOSSARY.md's "Naming Conventions Derived From
    This Glossary" section, which is authoritative for this enum. Note:
    docs/05_DATA_MODALITY_SPEC.md's CodedEvent.system comment lists a
    different set (SNOMED, ICD10, ICD10CM, RXNORM, LOINC, CPT -- both ICD10
    and ICD10CM, no UCUM). That is a documentation inconsistency to reconcile
    when the EHR module is built (M6), not a second source of truth -- the
    glossary's dedicated naming section wins here.
    """

    SNOMED = "SNOMED"
    LOINC = "LOINC"
    ICD10CM = "ICD10CM"
    RXNORM = "RXNORM"
    CPT = "CPT"
    UCUM = "UCUM"


class Concept(BaseModel):
    """A single resolved terminology concept.

    Example:
        >>> c = Concept(
        ...     code="73211009", system=CodeSystem.SNOMED, display="Diabetes mellitus"
        ... )
        >>> c.system.value
        'SNOMED'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(..., min_length=1, description="The code within its system.")
    system: CodeSystem = Field(
        ..., description="The terminology system this code belongs to."
    )
    display: str = Field(
        ..., min_length=1, description="Human-readable name for this code."
    )


class Message(BaseModel):
    """One turn in a conversation passed to an LLM provider.

    Example:
        >>> Message(role="user", content="Summarise this note.").role
        'user'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Who this message is from."
    )
    content: str = Field(..., description="The message text.")


class TokenUsage(BaseModel):
    """Token accounting for one or more LLM calls.

    Defined here, not in ``core.provenance`` (where ``RunManifest.token_usage``
    lives) -- ``core.provenance`` already imports from this module
    (``JsonValue``), so defining ``TokenUsage`` there and referencing it from
    ``LLMResponse`` here would be a real import cycle, not a hypothetical
    one. ``core.provenance`` re-exports this exact class for backward
    compatibility with the name it originally shipped under.

    Example:
        >>> TokenUsage().total_tokens
        0
        >>> a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        >>> b = TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        >>> (a + b).total_tokens
        20
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Combine two calls' usage into a running total -- real
        accounting for a pipeline that makes more than one LLM call in a
        run, not just a single-call struct with nothing to aggregate it."""
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class LLMResponse(BaseModel):
    """The result of one LLM generation call.

    Example:
        >>> LLMResponse(text="The patient has type 2 diabetes.").text
        'The patient has type 2 diabetes.'
        >>> resp = LLMResponse(text="...", usage=TokenUsage(total_tokens=42))
        >>> resp.usage.total_tokens
        42
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(..., description="The generated text.")
    usage: TokenUsage | None = Field(
        None, description="Token accounting for this call, if the provider reports it."
    )


class RAGAnswer(BaseModel):
    """The result of a full retrieve-(rerank)-generate RAG call
    (:class:`~openbtk.pipelines.rag.RAGPipeline`, task 5.8).

    ``sources`` is the whole point of carrying this as its own type rather
    than returning a bare ``LLMResponse``: every generated answer is
    traceable back to the exact retrieved chunks it was grounded in, in
    the order they were given to the model -- what makes a
    ``GroundednessGuardrail`` (or a human reviewer) able to check a claim
    against a precise source rather than "some document somewhere"
    (:class:`~openbtk.core.schemas.SourceRef`'s own docstring).

    Example:
        >>> answer = RAGAnswer(
        ...     text="The patient has type 2 diabetes.",
        ...     sources=[SourceRef(record_id="note-1", chunk_id="chunk-3")],
        ... )
        >>> answer.sources[0].record_id
        'note-1'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(..., description="The generated answer.")
    sources: list[SourceRef] = Field(
        default_factory=list,
        description="Retrieved chunks the answer was grounded in, in the "
        "order given to the model.",
    )
    usage: TokenUsage | None = Field(
        None, description="Token accounting for the generation call, if reported."
    )

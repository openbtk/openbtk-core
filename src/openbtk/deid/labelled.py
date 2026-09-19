"""Ground-truth-labelled documents for de-identification benchmarking.

Lives in ``openbtk.deid`` (not ``openbtk.eval`` or a modality package) so that
both a dataset adapter (``openbtk.data.clinical_text``, L2) and the benchmark
harness (``openbtk.eval``, L3) can depend on it without either importing the
other -- the same "a concept two layers need belongs in a module both already
depend on downward" reasoning as ``DeidStatus`` (docs/03_ARCHITECTURE.md
section 2).

Deliberately carries **no substring of the annotated text** in a span (only
offsets and a category): a benchmark result must never be able to leak
document content, and a real dataset's annotated PHI values are exactly the
thing not to copy into a report or log.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from openbtk.deid.schemas import PHICategory  # noqa: TC001 -- Pydantic field type


class LabelledSpan(BaseModel):
    """One annotated PHI span: category plus half-open ``[start, end)`` offsets.

    Example:
        >>> LabelledSpan(category=PHICategory.SSN, start=4, end=15).end
        15
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: PHICategory
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)


class LabelledDocument(BaseModel):
    """A document plus its ground-truth PHI annotations.

    ``unscored_spans`` counts annotations in the source dataset that have no
    Safe Harbor ``PHICategory`` equivalent (e.g. i2b2's PROFESSION or AGE
    tags) and were therefore left out of ``spans`` -- carried as a count, not
    dropped silently, so a benchmark report can state exactly how much of the
    dataset's annotation it did not score.

    Example:
        >>> doc = LabelledDocument(document_id="d1", text="x", spans=[])
        >>> doc.unscored_spans
        0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str = Field(..., min_length=1)
    text: str
    spans: list[LabelledSpan]
    unscored_spans: int = Field(0, ge=0)

"""Retrieval metrics: recall@k, MRR, nDCG@k (roadmap task 8.3).

Pure functions over a ranked list of ids plus graded relevance judgements,
and a streaming ``evaluate_retrieval`` that aggregates them over many
queries in O(1) memory (ADR-0004) -- queries are consumed once, only running
sums are kept.

Definitions, stated plainly because metric variants differ and a number is
meaningless without them:

* **recall@k** = |relevant ∩ top-k| / |relevant|. A document is relevant when
  its judged gain is > 0.
* **MRR** (per query: reciprocal rank) = 1 / rank of the first relevant
  result, 0 if none is retrieved. ``evaluate_retrieval`` averages over
  queries. Ranks are computed over the *whole* returned list, not truncated
  to ``k``.
* **nDCG@k** = DCG@k / IDCG@k with DCG = sum(gain_i / log2(i + 1)), i 1-based
  -- the *linear-gain* form (gain = the judged relevance value itself), not
  the exponential 2**rel - 1 variant. For binary judgements the two are
  identical. IDCG is the DCG of the ideal ordering of the judged documents.
* A ranking that repeats an id is de-duplicated, keeping the first
  occurrence: counting one document twice would inflate DCG.

A query with no relevant document has undefined recall/nDCG, so
``RetrievalQuery`` rejects it at construction rather than silently scoring it
0 or 1.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from openbtk.core.base import BaseEmbeddingProvider, BaseVectorStore
    from openbtk.core.schemas import SearchResult

DEFAULT_KS: tuple[int, ...] = (1, 5, 10)


def _dedupe(ranked_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in ranked_ids:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def recall_at_k(ranked_ids: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant documents found in the top ``k``.

    Example:
        >>> recall_at_k(["a", "b", "c"], {"a", "z"}, k=2)
        0.5
    """
    _check_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("recall is undefined with no relevant documents")
    top_k = set(_dedupe(ranked_ids)[:k])
    return len(top_k & relevant_set) / len(relevant_set)


def reciprocal_rank(ranked_ids: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant result, or 0.0 if none was retrieved.

    Example:
        >>> reciprocal_rank(["x", "a", "b"], {"a"})
        0.5
    """
    relevant_set = set(relevant)
    for rank, item in enumerate(_dedupe(ranked_ids), start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int
) -> float:
    """Normalised discounted cumulative gain at ``k`` (linear gain).

    Example:
        >>> ndcg_at_k(["a", "b"], {"a": 1.0, "b": 1.0}, k=2)
        1.0
        >>> round(ndcg_at_k(["x", "a"], {"a": 1.0}, k=2), 4)
        0.6309
    """
    _check_k(k)
    ideal_gains = sorted((g for g in relevance.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_gains, start=1))
    if idcg == 0.0:
        raise ValueError("nDCG is undefined with no positively-judged documents")
    dcg = sum(
        max(relevance.get(item, 0.0), 0.0) / math.log2(i + 1)
        for i, item in enumerate(_dedupe(ranked_ids)[:k], start=1)
    )
    return dcg / idcg


class RetrievalQuery(BaseModel):
    """One evaluation query with its graded relevance judgements.

    Example:
        >>> q = RetrievalQuery(query_id="q1", query="diabetes", relevance={"c3": 1.0})
        >>> q.relevant_ids
        ['c3']
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str = Field(..., min_length=1)
    query: str = Field(..., description="The query text handed to the retriever.")
    relevance: dict[str, float] = Field(
        ..., description="Document id -> judged gain. Gain > 0 means relevant."
    )

    @field_validator("relevance")
    @classmethod
    def _needs_a_relevant_document(cls, value: dict[str, float]) -> dict[str, float]:
        if not any(g > 0 for g in value.values()):
            raise ValueError(
                "a query needs at least one document with gain > 0; recall and "
                "nDCG are undefined otherwise"
            )
        return value

    @property
    def relevant_ids(self) -> list[str]:
        return sorted(doc_id for doc_id, gain in self.relevance.items() if gain > 0)


class RetrievalReport(BaseModel):
    """Macro-averaged metrics over every evaluated query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_queries: int = Field(..., ge=1)
    ks: list[int]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mrr: float


def evaluate_retrieval(
    retrieve: Callable[[str], Sequence[str]],
    queries: Iterable[RetrievalQuery],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> RetrievalReport:
    """Score ``retrieve`` (query text -> ranked document ids) over ``queries``.

    ``queries`` is consumed once (streaming); memory does not grow with the
    number of queries.

    Raises:
        ValueError: If ``ks`` is empty or contains a value < 1, or ``queries``
            is empty (an average over nothing is not a metric).

    Example:
        >>> qs = [RetrievalQuery(query_id="q", query="x", relevance={"a": 1.0})]
        >>> report = evaluate_retrieval(lambda text: ["a", "b"], qs, ks=(1,))
        >>> report.mrr, report.recall_at_k[1]
        (1.0, 1.0)
    """
    if not ks:
        raise ValueError("ks must not be empty")
    for k in ks:
        _check_k(k)
    recall_sums = dict.fromkeys(ks, 0.0)
    ndcg_sums = dict.fromkeys(ks, 0.0)
    mrr_sum = 0.0
    n = 0
    for q in queries:
        ranked = list(retrieve(q.query))
        relevant = q.relevant_ids
        for k in ks:
            recall_sums[k] += recall_at_k(ranked, relevant, k)
            ndcg_sums[k] += ndcg_at_k(ranked, q.relevance, k)
        mrr_sum += reciprocal_rank(ranked, relevant)
        n += 1
    if n == 0:
        raise ValueError("no queries to evaluate")
    return RetrievalReport(
        n_queries=n,
        ks=list(ks),
        recall_at_k={k: recall_sums[k] / n for k in ks},
        ndcg_at_k={k: ndcg_sums[k] / n for k in ks},
        mrr=mrr_sum / n,
    )


def retriever_from(
    embedding: BaseEmbeddingProvider,
    vectorstore: BaseVectorStore,
    *,
    top_k: int = 10,
    id_of: Callable[[SearchResult], str] | None = None,
) -> Callable[[str], list[str]]:
    """Build a ``query -> ranked ids`` callable from a real embedding
    provider and vector store, so the same metrics score whatever retrieval
    stack a user actually runs. ``id_of`` defaults to the hit's own id; pass
    e.g. ``lambda r: r.source.record_id`` to judge at record rather than
    chunk level.
    """
    pick = id_of if id_of is not None else (lambda r: r.id)

    def retrieve(query: str) -> list[str]:
        hits = vectorstore.query(embedding.embed_one(query), top_k=top_k)
        return [pick(h) for h in hits]

    return retrieve

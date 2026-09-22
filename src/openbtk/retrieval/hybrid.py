"""Hybrid dense + BM25 retrieval (FR-R-05).

A vector store finds chunks that *mean* something like the query. It is weak on exact
terms: a drug name, a lab code, an abbreviation, a rare word the embedding model never
learned. BM25 is the opposite: it ranks by the words a chunk shares with the query,
weighted by how rare they are, so "hydroxychloroquine" or "58410-2" matches exactly. Run
both and *fuse* the rankings and each covers the other's blind spot.

* :class:`BM25Index` is an in-memory Okapi BM25 index over chunk text.
* :func:`reciprocal_rank_fusion` merges any number of rankings.
* :class:`~openbtk.pipelines.rag.RAGPipeline` takes ``bm25=`` and does both, then fuses,
  then reranks.

**BM25 is implemented here rather than wrapped.** The popular ``rank_bm25`` package
(last released February 2022) uses an idf that is negative for a term in over half the
documents, then patches it with a floor of a fraction of the *average* idf, which is
itself negative on a small corpus. This index uses the idf Lucene uses,
``ln(1 + (N - df + 0.5) / (df + 0.5))``, which is never negative, is under 100 lines,
and is tested against hand-computed values.

**Limits, stated:**

* The index is **in memory** and is not persisted: rebuild it from your chunks when the
  process starts (it needs only the text and ids, in the same call you index the vector
  store with). It holds the term statistics *and each chunk's metadata*, so it is as
  sensitive as the vector store's own metadata.
* The default tokenizer lowercases and splits on anything that is not a letter or digit,
  so ``58410-2`` becomes ``58410`` and ``2``. That is fine for matching (both sides are
  tokenized alike) but is not a clinical tokenizer; pass ``tokenizer=`` to change it.
* No stemming and no stop-word list: "diabetic" does not match "diabetes", and common
  words are handled by the idf, not removed.
* Fusion is by *rank*, not score, because BM25 scores and cosine similarities are not on
  one scale. A hit at rank 1 in either list contributes ``1 / (k + 1)``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

from openbtk.core.errors import RetrievalError
from openbtk.core.schemas import SearchResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from openbtk.core.schemas import JsonValue

_TOKEN = re.compile(r"[^\W_]+")


def default_tokenizer(text: str) -> list[str]:
    """Lowercase alphanumeric runs (``"58410-2"`` -> ``["58410", "2"]``).

    Example:
        >>> default_tokenizer("Metformin 500 mg, BID; LOINC 58410-2")
        ['metformin', '500', 'mg', 'bid', 'loinc', '58410', '2']
    """
    return _TOKEN.findall(text.casefold())


class BM25Index:
    """An in-memory Okapi BM25 index. See the module docstring for its limits.

    Args:
        k1: Term-frequency saturation. Higher rewards repeated terms for longer.
        b: Length normalisation, ``0`` (none) to ``1`` (full).
        tokenizer: Splits text into terms; used for documents and queries alike.
        text_metadata_key: Where :meth:`search` results carry the indexed text, so they
            look like vector-store results (the key ``RAGPipeline`` prompts from).

    Example:
        >>> index = BM25Index()
        >>> index.upsert(
        ...     ["a", "b", "c"],
        ...     ["metformin for diabetes", "aspirin for pain", "insulin for diabetes"],
        ... )
        >>> ids = [hit.id for hit in index.search("metformin")]
        >>> ids
        ['a']
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = default_tokenizer,
        text_metadata_key: str = "text",
    ) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be non-negative and b between 0 and 1")
        self._k1 = k1
        self._b = b
        self._tokenize = tokenizer
        self._text_key = text_metadata_key
        self._ids: dict[str, int] = {}  # id -> slot
        self._slots: list[str | None] = []  # slot -> id (None once deleted)
        self._lengths: list[int] = []
        self._terms: list[tuple[str, ...]] = []  # the distinct terms of each slot
        self._metadata: list[dict[str, JsonValue]] = []
        self._postings: dict[str, dict[int, int]] = {}  # term -> {slot: tf}
        self._total_length = 0
        self._live = 0

    def __len__(self) -> int:
        return self._live

    # ------------------------------------------------------------------ writing

    def upsert(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadata: Sequence[dict[str, JsonValue]] | None = None,
    ) -> None:
        """Index ``texts`` under ``ids``, replacing any chunk that already has an id.

        Args:
            ids: One unique id per text (the ids you give the vector store).
            texts: The chunk texts.
            metadata: Optional metadata per chunk, returned with each hit. The text
                itself is added under ``text_metadata_key`` if it is not already there.

        Raises:
            RetrievalError: If the lists differ in length or ``ids`` repeats an id.
        """
        if len(ids) != len(texts) or (
            metadata is not None and len(metadata) != len(ids)
        ):
            raise RetrievalError(
                "ids, texts and metadata must be the same length.",
                context={"component": "retrieval.bm25"},
            )
        if len(set(ids)) != len(ids):
            raise RetrievalError(
                "ids must be unique within one upsert.",
                context={"component": "retrieval.bm25"},
            )
        self.delete([i for i in ids if i in self._ids])
        for n, (chunk_id, text) in enumerate(zip(ids, texts, strict=True)):
            terms = self._tokenize(text)
            slot = len(self._slots)
            self._slots.append(chunk_id)
            self._ids[chunk_id] = slot
            counts = Counter(terms)
            self._lengths.append(len(terms))
            self._terms.append(tuple(counts))
            meta = dict(metadata[n]) if metadata is not None else {}
            meta.setdefault(self._text_key, text)
            self._metadata.append(meta)
            self._total_length += len(terms)
            self._live += 1
            for term, count in counts.items():
                self._postings.setdefault(term, {})[slot] = count

    def delete(self, ids: Sequence[str]) -> None:
        """Remove chunks by id. An id that is not indexed is ignored."""
        for chunk_id in ids:
            slot = self._ids.pop(chunk_id, None)
            if slot is None:
                continue
            for term in self._terms[slot]:
                posting = self._postings.get(term)
                if posting is not None:
                    posting.pop(slot, None)
                    if not posting:
                        del self._postings[term]
            self._total_length -= self._lengths[slot]
            self._slots[slot] = None
            # Nothing about a deleted chunk is kept, its text and metadata included.
            self._terms[slot] = ()
            self._metadata[slot] = {}
            self._live -= 1

    # ------------------------------------------------------------------ reading

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """The ``top_k`` chunks that best match ``query``, best first.

        Only chunks sharing at least one term with the query are returned, so the result
        can be shorter than ``top_k`` (or empty). Ties break on id, so the order is
        deterministic.
        """
        if top_k < 1 or self._live == 0:
            return []
        average = self._total_length / self._live if self._live else 0.0
        scores: dict[int, float] = {}
        for term in set(self._tokenize(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (self._live - df + 0.5) / (df + 0.5))
            for slot, tf in posting.items():
                norm = (
                    1
                    - self._b
                    + (self._b * self._lengths[slot] / average if average else 0)
                )
                scores[slot] = scores.get(slot, 0.0) + idf * tf * (self._k1 + 1) / (
                    tf + self._k1 * norm
                )
        ranked = sorted(
            scores.items(), key=lambda kv: (-kv[1], self._slots[kv[0]] or "")
        )
        return [
            SearchResult(
                id=self._slots[slot] or "",
                score=score,
                metadata=dict(self._metadata[slot]),
            )
            for slot, score in ranked[:top_k]
        ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Merge several rankings into one by reciprocal rank (Cormack et al., 2009).

    Each ranking gives an id ``weight / (k + rank)`` (rank 1 is the best), the scores
    are summed per id, and ids are sorted by that sum. Only the *order* of each input
    matters, so rankings on different scales (cosine similarity, BM25) combine
    sensibly. A larger ``k`` flattens the advantage of a top rank.

    The returned results carry the fused score and the metadata of the **first ranking
    that contained the id**, so put the ranking whose metadata you prefer first.

    Args:
        rankings: Best-first lists of results.
        k: The smoothing constant; ``60`` is the value the original paper used.
        weights: One weight per ranking (default all ``1``).
        top_k: Keep at most this many.

    Example:
        >>> dense = [SearchResult(id="a", score=0.9), SearchResult(id="b", score=0.8)]
        >>> sparse = [SearchResult(id="b", score=12.0), SearchResult(id="c", score=3.0)]
        >>> ids = [hit.id for hit in reciprocal_rank_fusion([dense, sparse])]
        >>> ids
        ['b', 'a', 'c']
    """
    if k < 0:
        raise ValueError("k must not be negative")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("give one weight per ranking")
    fused: dict[str, float] = {}
    first: dict[str, SearchResult] = {}
    for n, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[n]
        seen: set[str] = set()
        for rank, result in enumerate(ranking, start=1):
            if result.id in seen:
                continue  # a repeated id in one ranking counts once, at its best rank
            seen.add(result.id)
            fused[result.id] = fused.get(result.id, 0.0) + weight / (k + rank)
            first.setdefault(result.id, result)
    ordered = sorted(fused, key=lambda i: (-fused[i], i))
    merged = [first[i].model_copy(update={"score": fused[i]}) for i in ordered]
    return merged if top_k is None else merged[:top_k]

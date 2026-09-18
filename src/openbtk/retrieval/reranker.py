"""``ConceptOverlapReranker`` -- reorders search results by shared UMLS
CUIs between the query and each result (FR-R-03, docs/02_PRD.md).

Concept *extraction* (raw text -> a set of UMLS CUIs) is deliberately
**not** built here: ``openbtk.terminology`` (UMLS/SNOMED/LOINC/RxNorm
resolution) is not yet implemented, and UMLS itself is a licensed,
restricted vocabulary this project's own terminology module docstring
already commits to never bundling. Building a real entity-linker would
mean either fabricating one against no real vocabulary (a correctness
risk far worse than the reranker itself) or blocking this task on
unrelated, unbuilt infrastructure. This reranker's own, real contribution
is the overlap-scoring and reordering logic; concept extraction is an
injected dependency (``extract_concepts: Callable[[str], Iterable[str]]``)
-- "wrap, don't reinvent" (docs/09_CODING_STANDARDS.md section 7) applied
to this project's own future terminology work, not just a third-party
library.

Each result's own CUIs are read from ``SearchResult.metadata`` (a list
under a configurable key, ``"cuis"`` by default) rather than re-extracted
per result at rerank time: a real indexing pipeline runs entity linking
once, when a chunk is embedded and stored, precisely so query time never
repeats that cost for every stored chunk on every single query -- only
the query text itself is run through ``extract_concepts`` here, once per
``rerank()`` call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openbtk.core.base import BaseReranker
from openbtk.core.logging import get_logger
from openbtk.core.registry import RERANKER_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from openbtk.core.provenance import ComponentProvenance
    from openbtk.core.schemas import SearchResult

log = get_logger(__name__)

_DEFAULT_CUIS_KEY = "cuis"


@RERANKER_REGISTRY.register("reranker.general.concept_overlap")
class ConceptOverlapReranker(BaseReranker):
    """Rerank results by the count of UMLS CUIs they share with the query.

    Args:
        extract_concepts: Maps a text string to the CUIs it mentions.
            Called once, on the query text, per :meth:`rerank` call. See
            this module's own docstring for why this is injected rather
            than built in.
        cuis_metadata_key: The ``SearchResult.metadata`` key each result's
            own pre-computed CUI list is read from. A result with no such
            key, or an empty list there, contributes zero overlap -- it is
            never excluded, just never promoted by this criterion.

    Sorts by ``(overlap_count, original_score)``, both descending: shared
    concepts are the primary criterion (the entire point of this
    reranker), the original retrieval score the tiebreaker among results
    that share the same number of concepts with the query -- including
    "none at all", so a result set with no concept metadata populated
    anywhere degrades gracefully to the original score order rather than
    an arbitrary one.

    No I/O happens in ``__init__`` (docs/09_CODING_STANDARDS.md rule 11):
    ``extract_concepts`` is only ever called from :meth:`rerank`.
    """

    def __init__(
        self,
        *,
        extract_concepts: Callable[[str], Iterable[str]],
        cuis_metadata_key: str = _DEFAULT_CUIS_KEY,
    ) -> None:
        self._extract_concepts = extract_concepts
        self._cuis_metadata_key = cuis_metadata_key

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        if not results:
            return []
        query_concepts = set(self._extract_concepts(query))

        def _overlap(result: SearchResult) -> int:
            raw = result.metadata.get(self._cuis_metadata_key)
            result_concepts = raw if isinstance(raw, list) else []
            return len(query_concepts.intersection(result_concepts))

        reranked = sorted(results, key=lambda r: (_overlap(r), r.score), reverse=True)
        return reranked if top_k is None else reranked[:top_k]

    def provenance(self) -> ComponentProvenance:
        return (
            super()
            .provenance()
            .model_copy(
                update={"config": {"cuis_metadata_key": self._cuis_metadata_key}}
            )
        )

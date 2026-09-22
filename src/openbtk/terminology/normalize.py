"""``ConceptNormalizer``: fuzzy matching of a free-text term to a concept (FR-M-06).

*"CBC"* means *complete blood count* means LOINC ``58410-2``; *"hypertenson"* is a typo
for hypertension. This maps such a term to the concepts of a vocabulary **you give it**,
tolerating case, punctuation, typos and the aliases you list.

**It ships no vocabulary of its own beyond the small ICD-10-CM subset that is already
bundled.** LOINC and SNOMED CT are licensed and OpenBTK never bundles them, so whether
"CBC" resolves is a property of the vocabulary you load
(:meth:`ConceptNormalizer.from_csv` takes a licensed extract with an ``aliases``
column), not of this code.

Matching is deliberately conservative, because a wrong clinical match is worse than
none:

* A candidate must **agree on the modifiers that change the meaning**: numbers ("type 1"
  vs "type 2"), laterality (left, right, bilateral), acuity (acute, chronic) and
  negation (no, not, without, non). Plain string similarity rates "type 1 diabetes" and
  "type 2 diabetes" as 93% alike; here they never match each other. The list is a
  safeguard, not a clinical ontology, and is not exhaustive.
* Nothing below ``threshold`` is returned, and :meth:`~ConceptNormalizer.resolve`
  returns ``None`` (not a guess) when the two best answers are different codes that
  score within ``ambiguity_margin`` of each other.
* It matches a **term**, not a sentence. "No history of diabetes" is not a term for
  diabetes and would (correctly) not match; run negation and entity extraction first.
* Scores are string similarity in ``[0, 1]``, not probabilities. Choose ``threshold`` by
  looking at your own data; no accuracy figure is claimed for any value.

Candidates come from a character-trigram index, read rarest trigram first within a fixed
budget (so a query costs a bounded amount of work, not one comparison per concept), then
are scored with ``difflib`` and token overlap. Only the standard library is used. The
budget means a query whose only shared trigrams are very common can miss a candidate;
in practice the rare trigrams (a number, an unusual word) are what identify a concept.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from openbtk.core.errors import TerminologyError
from openbtk.core.schemas import CodeSystem, Concept

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_MAX_QUERY_CHARS = 256
_MAX_CANDIDATES = 50
_POSTING_BUDGET = 4000

_LATERALITY = frozenset({"left", "right", "bilateral", "l", "r"})
_ACUITY = frozenset({"acute", "chronic", "subacute", "recurrent"})
_NEGATION = frozenset({"no", "not", "non", "without", "negative", "absent"})
_MODIFIERS = _LATERALITY | _ACUITY | _NEGATION


def _normal(text: str) -> str:
    """Case-folded, accent-stripped, punctuation-free, single-spaced."""
    folded = unicodedata.normalize("NFKD", text).casefold()
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", stripped).split())


def _trigrams(text: str) -> set[str]:
    padded = f"  {text} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def _critical(tokens: Sequence[str]) -> frozenset[str]:
    """The tokens whose difference means a different concept."""
    return frozenset(
        t for t in tokens if t in _MODIFIERS or any(c.isdigit() for c in t)
    )


class ConceptMatch(BaseModel):
    """A candidate concept for a term, with how it matched.

    ``method`` is ``exact`` (the term is the concept's display), ``alias`` (it is one of
    the aliases you listed) or ``fuzzy`` (a close, but not identical, string).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    concept: Concept
    score: float = Field(
        ..., ge=0, le=1, description="String similarity, not a probability."
    )
    matched_text: str = Field(..., description="The display or alias that matched.")
    method: Literal["exact", "alias", "fuzzy"]


class _Entry:
    __slots__ = ("concept", "is_alias", "normal", "text", "tokens")

    def __init__(self, concept: Concept, text: str, *, is_alias: bool) -> None:
        self.concept = concept
        self.text = text
        self.normal = _normal(text)
        self.tokens = self.normal.split()
        self.is_alias = is_alias


class ConceptNormalizer:
    """Map free-text terms to the concepts of a vocabulary you supply.

    Args:
        concepts: The vocabulary.
        aliases: Extra names per concept, keyed by ``code`` (a code in more than one
            system takes the aliases in each). This is how "CBC" reaches its code.
        threshold: The lowest score returned, in ``[0, 1]``. ``1.0`` allows only
            exact and alias matches.
        ambiguity_margin: :meth:`resolve` gives no answer when the two best distinct
            codes are this close.

    Example:
        >>> concepts = [
        ...     Concept(code="I10", system=CodeSystem.ICD10CM,
        ...             display="Essential (primary) hypertension"),
        ...     Concept(code="E11.9", system=CodeSystem.ICD10CM,
        ...             display="Type 2 diabetes mellitus without complications"),
        ... ]
        >>> normalizer = ConceptNormalizer(concepts, aliases={"I10": ["HTN"]})
        >>> normalizer.resolve("htn").concept.code
        'I10'
        >>> normalizer.resolve("Type 1 diabetes mellitus") is None
        True
    """

    def __init__(
        self,
        concepts: Iterable[Concept],
        *,
        aliases: Mapping[str, Sequence[str]] | None = None,
        threshold: float = 0.85,
        ambiguity_margin: float = 0.05,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not 0 <= ambiguity_margin <= 1:
            raise ValueError("ambiguity_margin must be between 0 and 1")
        self._threshold = threshold
        self._margin = ambiguity_margin
        self._entries: list[_Entry] = []
        self._by_normal: dict[str, list[int]] = defaultdict(list)
        self._by_trigram: dict[str, list[int]] = defaultdict(list)
        for concept in concepts:
            self._add(concept, concept.display, is_alias=False)
            for alias in (aliases or {}).get(concept.code, ()):
                self._add(concept, alias, is_alias=True)

    def _add(self, concept: Concept, text: str, *, is_alias: bool) -> None:
        entry = _Entry(concept, text, is_alias=is_alias)
        if not entry.normal:
            return
        index = len(self._entries)
        self._entries.append(entry)
        self._by_normal[entry.normal].append(index)
        for gram in _trigrams(entry.normal):
            self._by_trigram[gram].append(index)

    # ------------------------------------------------------------- constructors

    @classmethod
    def from_csv(
        cls, path: str, *, threshold: float = 0.85, ambiguity_margin: float = 0.05
    ) -> ConceptNormalizer:
        """Read ``code,system,display`` rows and an optional ``aliases`` column.

        Aliases are separated by ``|``. ``system`` must be a
        :class:`~openbtk.core.schemas.CodeSystem` value. Read now, not lazily: this is
        an explicit load, not a constructor.

        Raises:
            TerminologyError: If the file is missing, lacks a required column or has a
                row with an unknown system. The message names the line, never a value.
        """
        source = Path(path)
        try:
            handle = source.open(encoding="utf-8", newline="")
        except OSError as e:
            raise TerminologyError(
                f"Could not open the vocabulary file {source.name}.",
                context={"path": str(source)},
            ) from e
        concepts: list[Concept] = []
        aliases: dict[str, list[str]] = {}
        with handle:
            reader = csv.DictReader(handle)
            missing = {"code", "system", "display"} - set(reader.fieldnames or ())
            if missing:
                raise TerminologyError(
                    f"{source.name} lacks required column(s) {sorted(missing)}.",
                    context={"path": str(source)},
                )
            for number, row in enumerate(reader, start=2):
                try:
                    system = CodeSystem(row["system"].strip())
                    concept = Concept(
                        code=row["code"].strip(),
                        system=system,
                        display=row["display"].strip(),
                    )
                except ValueError as e:
                    raise TerminologyError(
                        f"{source.name} line {number} is not a valid concept.",
                        context={"path": str(source), "line": number},
                    ) from e
                concepts.append(concept)
                extra = [a.strip() for a in (row.get("aliases") or "").split("|")]
                aliases.setdefault(concept.code, []).extend(a for a in extra if a)
        return cls(
            concepts,
            aliases=aliases,
            threshold=threshold,
            ambiguity_margin=ambiguity_margin,
        )

    @classmethod
    def from_bundled_icd10cm(
        cls, *, threshold: float = 0.85, ambiguity_margin: float = 0.05
    ) -> ConceptNormalizer:
        """A normaliser over the small ICD-10-CM subset that ships with OpenBTK."""
        from openbtk.terminology.bundled import bundled_icd10cm_concepts

        return cls(
            bundled_icd10cm_concepts(),
            threshold=threshold,
            ambiguity_margin=ambiguity_margin,
        )

    # ------------------------------------------------------------------ matching

    def normalise(
        self, text: str, *, limit: int = 3, system: CodeSystem | None = None
    ) -> list[ConceptMatch]:
        """The best-scoring concepts for ``text``, best first, at or above threshold.

        One match per code (its best-scoring name). Ties break on code, so the order is
        deterministic. Returns ``[]`` for an empty term, one over
        256 characters (a sentence, not a term), or no candidate above the threshold.

        Args:
            text: The term.
            limit: The most matches to return.
            system: Only consider concepts of this code system.
        """
        if len(text) > _MAX_QUERY_CHARS or limit < 1:
            return []
        query = _normal(text)
        if not query:
            return []
        best: dict[tuple[str, CodeSystem], ConceptMatch] = {}
        for index in self._candidates(query):
            entry = self._entries[index]
            if system is not None and entry.concept.system is not system:
                continue
            score = self._score(query, entry)
            if score < self._threshold:
                continue
            key = (entry.concept.code, entry.concept.system)
            found = best.get(key)
            if found is None or score > found.score:
                method: Literal["exact", "alias", "fuzzy"] = (
                    "fuzzy" if score < 1.0 else ("alias" if entry.is_alias else "exact")
                )
                shown = (
                    round(score, 4)
                    if method != "fuzzy"
                    else min(round(score, 4), 0.9999)
                )
                best[key] = ConceptMatch(
                    concept=entry.concept,
                    score=shown,
                    matched_text=entry.text,
                    method=method,
                )
        ranked = sorted(
            best.values(),
            key=lambda m: (-m.score, m.concept.code, m.concept.system.value),
        )
        return ranked[:limit]

    def resolve(
        self, text: str, *, system: CodeSystem | None = None
    ) -> ConceptMatch | None:
        """The single best match, or ``None`` when there is none or it is not clear.

        ``None`` means: nothing reached the threshold, or the two best answers are
        different concepts scoring within ``ambiguity_margin`` of each other. In that
        case call :meth:`normalise` and decide with a person or more context.
        """
        matches = self.normalise(text, limit=2, system=system)
        if not matches:
            return None
        if len(matches) == 2 and matches[0].score - matches[1].score <= self._margin:
            return None
        return matches[0]

    # ---------------------------------------------------------------- internals

    def _candidates(self, query: str) -> list[int]:
        exact = self._by_normal.get(query)
        if exact:
            return list(exact)
        # Rarest trigrams first, within a fixed budget of postings: in a vocabulary of
        # near-identical names every common trigram lists thousands of concepts, and
        # counting them all would make a query cost the size of the vocabulary. The rare
        # trigrams are the ones that tell concepts apart (a number, an unusual word).
        grams = sorted(
            (g for g in _trigrams(query) if g in self._by_trigram),
            key=lambda g: (len(self._by_trigram[g]), g),
        )
        shared: Counter[int] = Counter()
        spent = 0
        for gram in grams:
            posting = self._by_trigram[gram]
            if shared and spent + len(posting) > _POSTING_BUDGET:
                break
            spent += len(posting)
            for index in posting:
                shared[index] += 1
        ranked = sorted(shared.items(), key=lambda kv: (-kv[1], kv[0]))
        return [index for index, _ in ranked[:_MAX_CANDIDATES]]

    @staticmethod
    def _score(query: str, entry: _Entry) -> float:
        if query == entry.normal:
            return 1.0
        tokens = query.split()
        if _critical(tokens) != _critical(entry.tokens):
            return 0.0
        ratio = SequenceMatcher(None, query, entry.normal).ratio()
        a, b = set(tokens), set(entry.tokens)
        dice = 2 * len(a & b) / (len(a) + len(b))
        # A whole-string near-match, or heavy token overlap; never a full 1.0, which is
        # reserved for identical normalised text so "exact" and "alias" mean it.
        return min(max(ratio, dice), 0.9999)

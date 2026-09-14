"""``SpanMerger``: combines detections from multiple recognizers into one
final list.

ADR-0006 names this "the accuracy-bearing component" -- this is where two
recognizers' individually-published F1 figures (rules 94.1%, NER 96.8%)
become one ensemble figure (98.4%), so its policy is documented precisely
and tested exhaustively rather than left as "seemed reasonable":

  * **Overlap** means two spans share at least one character
    (``TextSpan`` is a half-open ``[start, end)`` interval, matching its own
    docstring -- two spans that only touch at an endpoint do not overlap).
  * **Widest span wins.** Among a group of mutually, transitively
    overlapping detections, the one with the largest ``end - start`` is kept
    as-is (its own boundaries, not a synthesised union) -- recall-biased by
    construction: a wider kept span never leaves a PHI fragment sitting
    just outside the redacted region.
  * **Confidences combine by noisy-OR**: ``1 - prod(1 - p_i)`` over every
    detection in the group. Two independent, moderately-confident
    detections of the same thing should end up MORE confident than either
    alone, which a simple average would not capture.
  * **Ties break by recognizer priority.** If the widest span is shared by
    two detections of equal width, the one from the higher-priority
    recognizer (earlier in ``recognizer_priority``, default
    ``("rule", "ner", "llm_verifier")``) wins.
  * **Every output ``Detection.method`` is ``"ensemble"``**, even for a
    group of exactly one -- ``method`` records what produced the FINAL
    list, and that is always this class once ``merge()`` has run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openbtk.core.schemas import TextSpan
from openbtk.deid.schemas import Detection

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_DEFAULT_PRIORITY: tuple[str, ...] = ("rule", "ner", "llm_verifier")


class SpanMerger:
    """Resolve overlapping PHI detections into one final, non-overlapping list.

    Example:
        >>> from openbtk.deid.schemas import PHICategory
        >>> a = Detection(
        ...     category=PHICategory.SSN,
        ...     span=TextSpan(start=0, end=11, label="ssn", confidence=0.9),
        ...     confidence=0.9,
        ...     method="rule",
        ... )
        >>> merged = SpanMerger().merge([a])
        >>> merged[0].method
        'ensemble'
    """

    def __init__(
        self, *, recognizer_priority: Sequence[str] = _DEFAULT_PRIORITY
    ) -> None:
        self._priority = list(recognizer_priority)

    def merge(self, detections: Sequence[Detection]) -> list[Detection]:
        """Merge possibly-overlapping detections into a final, sorted,
        non-overlapping list.

        Args:
            detections: Detections from one or more recognizers, in any
                order, possibly overlapping.

        Returns:
            One ``Detection`` per group of mutually overlapping input
            detections, sorted by span start. Never overlaps within its
            own output.
        """
        if not detections:
            return []
        ordered = sorted(detections, key=lambda d: (d.span.start, d.span.end))
        clusters: list[list[Detection]] = [[ordered[0]]]
        cluster_max_end = ordered[0].span.end
        for detection in ordered[1:]:
            if detection.span.start < cluster_max_end:
                clusters[-1].append(detection)
                cluster_max_end = max(cluster_max_end, detection.span.end)
            else:
                clusters.append([detection])
                cluster_max_end = detection.span.end
        return [self._merge_cluster(cluster) for cluster in clusters]

    def _merge_cluster(self, cluster: list[Detection]) -> Detection:
        winner = max(
            cluster,
            key=lambda d: (d.span.end - d.span.start, -self._priority_rank(d.method)),
        )
        combined_confidence = self._noisy_or(d.confidence for d in cluster)
        return Detection(
            category=winner.category,
            span=TextSpan(
                start=winner.span.start,
                end=winner.span.end,
                label=winner.category.value,
                confidence=combined_confidence,
            ),
            confidence=combined_confidence,
            method="ensemble",
        )

    def _priority_rank(self, method: str) -> int:
        """Lower is higher-priority. A method not in the configured
        priority list ranks last -- present so an unrecognised method
        (e.g. a future recognizer type) never crashes tie-breaking, only
        loses every tie."""
        try:
            return self._priority.index(method)
        except ValueError:
            return len(self._priority)

    @staticmethod
    def _noisy_or(confidences: Iterable[float]) -> float:
        product = 1.0
        for confidence in confidences:
            product *= 1.0 - confidence
        return 1.0 - product

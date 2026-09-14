"""Exhaustive unit tests for SpanMerger -- ADR-0006 names this "the
accuracy-bearing component" and calls specifically for exhaustive unit
tests over hand-built overlap cases. Every case here is hand-built and
independently checkable by hand, not generated.
"""

from __future__ import annotations

from openbtk.core.schemas import TextSpan
from openbtk.deid.merger import SpanMerger
from openbtk.deid.schemas import Detection, PHICategory


def _detection(
    category: PHICategory,
    start: int,
    end: int,
    confidence: float,
    method: str = "rule",
) -> Detection:
    return Detection(
        category=category,
        span=TextSpan(
            start=start, end=end, label=category.value, confidence=confidence
        ),
        confidence=confidence,
        method=method,
    )


class TestEmptyAndSingleton:
    def test_empty_input_returns_empty_output(self) -> None:
        assert SpanMerger().merge([]) == []

    def test_single_detection_passes_through_with_method_relabelled(self) -> None:
        det = _detection(PHICategory.SSN, 0, 11, 0.9, method="rule")
        merged = SpanMerger().merge([det])
        assert len(merged) == 1
        assert merged[0].span.start == 0
        assert merged[0].span.end == 11
        assert merged[0].category == PHICategory.SSN
        assert merged[0].method == "ensemble"

    def test_single_detection_confidence_is_unchanged_by_noisy_or(self) -> None:
        """noisy-OR over a single value p is p itself: 1 - (1 - p) == p."""
        det = _detection(PHICategory.SSN, 0, 11, 0.73)
        merged = SpanMerger().merge([det])
        assert merged[0].confidence == 0.73


class TestNonOverlappingSpans:
    def test_two_disjoint_spans_both_survive_independently(self) -> None:
        a = _detection(PHICategory.SSN, 0, 11, 0.9)
        b = _detection(PHICategory.EMAIL, 20, 30, 0.8)
        merged = SpanMerger().merge([a, b])
        assert len(merged) == 2
        assert merged[0].span.start == 0
        assert merged[1].span.start == 20

    def test_touching_but_not_overlapping_spans_do_not_merge(self) -> None:
        """TextSpan is half-open [start, end) -- a span ending at 10 and one
        starting at 10 share no character and must not merge."""
        a = _detection(PHICategory.SSN, 0, 10, 0.9)
        b = _detection(PHICategory.EMAIL, 10, 20, 0.8)
        merged = SpanMerger().merge([a, b])
        assert len(merged) == 2

    def test_output_is_sorted_by_start_regardless_of_input_order(self) -> None:
        a = _detection(PHICategory.SSN, 20, 30, 0.9)
        b = _detection(PHICategory.EMAIL, 0, 10, 0.8)
        merged = SpanMerger().merge([a, b])
        assert [d.span.start for d in merged] == [0, 20]


class TestWidestSpanWins:
    def test_wider_span_s_category_and_boundaries_are_kept(self) -> None:
        narrow = _detection(PHICategory.OTHER_UNIQUE_IDENTIFIER, 2, 8, 0.6)
        wide = _detection(PHICategory.SSN, 0, 11, 0.6)
        merged = SpanMerger().merge([narrow, wide])
        assert len(merged) == 1
        assert merged[0].category == PHICategory.SSN
        assert (merged[0].span.start, merged[0].span.end) == (0, 11)

    def test_order_of_input_does_not_affect_which_span_wins(self) -> None:
        narrow = _detection(PHICategory.OTHER_UNIQUE_IDENTIFIER, 2, 8, 0.6)
        wide = _detection(PHICategory.SSN, 0, 11, 0.6)
        merged_a = SpanMerger().merge([narrow, wide])
        merged_b = SpanMerger().merge([wide, narrow])
        assert merged_a == merged_b

    def test_partial_overlap_still_resolves_to_the_wider_one(self) -> None:
        # [0, 12) width 12 vs [5, 10) width 5 -- they overlap on [5, 10).
        wide = _detection(PHICategory.SSN, 0, 12, 0.7)
        narrow = _detection(PHICategory.OTHER_UNIQUE_IDENTIFIER, 5, 10, 0.7)
        merged = SpanMerger().merge([wide, narrow])
        assert len(merged) == 1
        assert (merged[0].span.start, merged[0].span.end) == (0, 12)


class TestTieBreakByPriority:
    def test_equal_width_spans_broken_by_default_priority_rule_over_ner(self) -> None:
        rule_det = _detection(PHICategory.SSN, 0, 10, 0.6, method="rule")
        ner_det = _detection(
            PHICategory.OTHER_UNIQUE_IDENTIFIER, 0, 10, 0.6, method="ner"
        )
        merged = SpanMerger().merge([ner_det, rule_det])
        assert len(merged) == 1
        assert merged[0].category == PHICategory.SSN  # rule's category wins

    def test_equal_width_spans_broken_by_ner_over_llm_verifier(self) -> None:
        ner_det = _detection(PHICategory.NAME, 0, 10, 0.6, method="ner")
        llm_det = _detection(
            PHICategory.OTHER_UNIQUE_IDENTIFIER, 0, 10, 0.6, method="llm_verifier"
        )
        merged = SpanMerger().merge([llm_det, ner_det])
        assert merged[0].category == PHICategory.NAME

    def test_custom_priority_order_is_honoured(self) -> None:
        rule_det = _detection(PHICategory.SSN, 0, 10, 0.6, method="rule")
        ner_det = _detection(PHICategory.NAME, 0, 10, 0.6, method="ner")
        merger = SpanMerger(recognizer_priority=("ner", "rule", "llm_verifier"))
        merged = merger.merge([rule_det, ner_det])
        assert merged[0].category == PHICategory.NAME  # ner now outranks rule

    def test_unknown_method_loses_every_tie(self) -> None:
        """Defence against a crash, not a claim about correctness for a
        genuinely new recognizer type -- an unconfigured method must not
        raise, just lose ties."""
        known = _detection(PHICategory.SSN, 0, 10, 0.6, method="rule")
        unknown = _detection(PHICategory.NAME, 0, 10, 0.6, method="llm_verifier")
        merger = SpanMerger(recognizer_priority=("rule",))
        merged = merger.merge([unknown, known])
        assert merged[0].category == PHICategory.SSN


class TestTransitiveOverlap:
    def test_a_chain_of_three_overlapping_spans_merges_into_one(self) -> None:
        # A: [0,10) width 10, B: [5,15) width 10, C: [12,24) width 12.
        # A and C don't touch directly (12 >= 10), but A-B overlap (5<10)
        # and B-C overlap (12<15) chain all three into one cluster. C is
        # unambiguously widest (12 > 10), so ties between A and B never
        # come into it -- see TestTieBreakByPriority for that case instead.
        a = _detection(PHICategory.SSN, 0, 10, 0.5)
        b = _detection(PHICategory.OTHER_UNIQUE_IDENTIFIER, 5, 15, 0.5)
        c = _detection(PHICategory.ACCOUNT_NUMBER, 12, 24, 0.5)
        merged = SpanMerger().merge([a, b, c])
        assert len(merged) == 1
        assert (merged[0].span.start, merged[0].span.end) == (12, 24)

    def test_two_separate_clusters_stay_separate(self) -> None:
        a = _detection(PHICategory.SSN, 0, 10, 0.5)
        b = _detection(PHICategory.OTHER_UNIQUE_IDENTIFIER, 5, 15, 0.5)
        c = _detection(PHICategory.ACCOUNT_NUMBER, 100, 110, 0.5)
        merged = SpanMerger().merge([a, b, c])
        assert len(merged) == 2


class TestNoisyOrConfidenceCombination:
    def test_two_independent_moderate_confidences_combine_above_either_alone(
        self,
    ) -> None:
        a = _detection(PHICategory.SSN, 0, 10, 0.5)
        b = _detection(PHICategory.OTHER_UNIQUE_IDENTIFIER, 0, 10, 0.5)
        merged = SpanMerger().merge([a, b])
        # 1 - (1-0.5)(1-0.5) = 1 - 0.25 = 0.75
        assert merged[0].confidence == 0.75

    def test_three_way_combination_matches_hand_computed_value(self) -> None:
        a = _detection(PHICategory.SSN, 0, 10, 0.5)
        b = _detection(PHICategory.OTHER_UNIQUE_IDENTIFIER, 0, 10, 0.4)
        c = _detection(PHICategory.ACCOUNT_NUMBER, 0, 10, 0.2)
        merged = SpanMerger().merge([a, b, c])
        # 1 - (0.5)(0.6)(0.8) = 1 - 0.24 = 0.76
        assert abs(merged[0].confidence - 0.76) < 1e-9

    def test_a_zero_confidence_detection_does_not_zero_out_the_group(self) -> None:
        a = _detection(PHICategory.SSN, 0, 10, 0.0)
        b = _detection(PHICategory.OTHER_UNIQUE_IDENTIFIER, 0, 10, 0.9)
        merged = SpanMerger().merge([a, b])
        assert merged[0].confidence == 0.9

    def test_merged_span_confidence_matches_merged_detection_confidence(self) -> None:
        a = _detection(PHICategory.SSN, 0, 10, 0.5)
        b = _detection(PHICategory.OTHER_UNIQUE_IDENTIFIER, 0, 10, 0.5)
        merged = SpanMerger().merge([a, b])
        assert merged[0].span.confidence == merged[0].confidence

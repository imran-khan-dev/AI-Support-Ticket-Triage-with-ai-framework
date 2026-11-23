"""Deterministic unit tests for the common retrieval metrics in app/comparison.

All expected values are hand-calculated. Metrics are framework-independent
and use identical calculations for both the custom and framework editions.
"""

from __future__ import annotations

import pytest

from app.comparison.metrics import (
    average,
    dcg_at_k,
    idcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# judgments: {chunk_id: grade}, grade > 0 means relevant
JUDGMENTS = {"C1": 3, "C2": 2, "C3": 1}
RETRIEVED = ["C1", "C4", "C2", "C5", "C3"]


class TestPrecisionAtK:
    def test_top_three_hits_two_of_three(self):
        # top-3 = [C1, C4, C2], relevant = C1, C2 -> 2 hits, precision = 2/3
        assert precision_at_k(RETRIEVED, JUDGMENTS, 3) == pytest.approx(2 / 3)

    def test_top_five_hits_three_of_three(self):
        # top-5 = all five, relevant = C1, C2, C3 -> 3 hits, precision = 3/5
        assert precision_at_k(RETRIEVED, JUDGMENTS, 5) == pytest.approx(3 / 5)

    def test_k_zero(self):
        assert precision_at_k(RETRIEVED, JUDGMENTS, 0) == 0.0
        assert precision_at_k(RETRIEVED, JUDGMENTS, -1) == 0.0

    def test_no_relevant_items(self):
        assert precision_at_k(RETRIEVED, {}, 5) == 0.0

    def test_grade_zero_is_not_relevant(self):
        assert precision_at_k(["C1", "C2"], {"C1": 0, "C2": 0}, 2) == 0.0

    def test_fewer_results_than_k(self):
        # only 1 result returned but k=5 -> hits=1, precision = 1/5
        assert precision_at_k(["C1"], JUDGMENTS, 5) == pytest.approx(1 / 5)

    def test_duplicates_counted_per_position(self):
        # C1 at pos 1 and pos 2 -> 2 hits, precision = 2/2 = 1.0
        assert precision_at_k(["C1", "C1"], {"C1": 1}, 2) == 1.0

    def test_missing_chunk_ids_are_not_hits(self):
        assert precision_at_k(["C9", "C8"], {"C1": 1}, 2) == 0.0


class TestRecallAtK:
    def test_top_three_hits_two_of_three_relevant(self):
        # total_relevant = 3, hits in top-3 = 2 -> recall = 2/3
        assert recall_at_k(RETRIEVED, JUDGMENTS, 3) == pytest.approx(2 / 3)

    def test_top_five_recall_one(self):
        # all 3 relevant retrieved -> recall = 3/3 = 1.0
        assert recall_at_k(RETRIEVED, JUDGMENTS, 5) == 1.0

    def test_k_zero(self):
        assert recall_at_k(RETRIEVED, JUDGMENTS, 0) == 0.0

    def test_no_relevant_items(self):
        assert recall_at_k(["C1"], {}, 5) == 0.0

    def test_grade_zero_is_not_relevant(self):
        assert recall_at_k(["C1", "C2"], {"C1": 0, "C2": 0}, 2) == 0.0

    def test_missing_chunk_ids_are_not_hits(self):
        assert recall_at_k(["C9", "C8"], {"C1": 1}, 2) == 0.0


class TestReciprocalRank:
    def test_first_relevant_at_position_one(self):
        assert reciprocal_rank(RETRIEVED, JUDGMENTS) == 1.0

    def test_first_relevant_at_position_three(self):
        # ["C4", "C5", "C1"] -> C1 at pos 3 -> RR = 1/3
        assert reciprocal_rank(["C4", "C5", "C1"], JUDGMENTS) == pytest.approx(1 / 3)

    def test_no_relevant_item(self):
        assert reciprocal_rank(["C9", "C8"], JUDGMENTS) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank([], JUDGMENTS) == 0.0

    def test_empty_relevance(self):
        assert reciprocal_rank(["C1"], {}) == 0.0


class TestDCGAtK:
    def test_dcg_top_three(self):
        # pos1 C1 rel3: gain=7 / log2(2)=1 -> 7
        # pos2 C4 rel0: skip
        # pos3 C2 rel2: gain=3 / log2(4)=2 -> 1.5
        # DCG@3 = 8.5
        assert dcg_at_k(RETRIEVED, JUDGMENTS, 3) == pytest.approx(8.5)

    def test_dcg_k_zero(self):
        assert dcg_at_k(RETRIEVED, JUDGMENTS, 0) == 0.0

    def test_dcg_irrelevant_only(self):
        assert dcg_at_k(["C9", "C8"], {"C1": 1}, 2) == 0.0

    def test_dcg_grade_zero_skipped(self):
        assert dcg_at_k(["C1"], {"C1": 0}, 1) == 0.0


class TestIDCGAtK:
    def test_idcg_top_three(self):
        # grades sorted [3, 2, 1]
        # pos1: 7 / log2(2)=1 -> 7
        # pos2: 3 / log2(3)~1.5850 -> ~1.8927
        # pos3: 1 / log2(4)=2 -> 0.5
        # IDCG@3 ~ 9.3927
        assert idcg_at_k(JUDGMENTS, 3) == pytest.approx(9.3927, abs=1e-4)

    def test_idcg_empty(self):
        assert idcg_at_k({}, 5) == 0.0

    def test_idcg_k_zero(self):
        assert idcg_at_k(JUDGMENTS, 0) == 0.0

    def test_idcg_grade_zero_ignored(self):
        assert idcg_at_k({"C1": 0, "C2": 0}, 5) == 0.0


class TestNDCGAtK:
    def test_ndcg_perfect_order_is_one(self):
        # retrieved in ideal order -> NDCG = 1.0
        assert ndcg_at_k(["C1", "C2", "C3"], JUDGMENTS, 3) == 1.0

    def test_ndcg_non_ideal_order(self):
        # ["C3", "C2", "C1"] vs ideal ["C1", "C2", "C3"] -> 0 < NDCG < 1
        score = ndcg_at_k(["C3", "C2", "C1"], JUDGMENTS, 3)
        assert 0.0 < score < 1.0

    def test_ndcg_no_relevant(self):
        assert ndcg_at_k(["C1"], {}, 5) == 0.0

    def test_ndcg_k_zero(self):
        assert ndcg_at_k(["C1"], JUDGMENTS, 0) == 0.0

    def test_ndcg_partial_match(self):
        # retrieved = ["C1", "C4", "C2", "C5", "C3"], k=3
        # DCG@3 = 7 (C1@1) + 1.5 (C2@3) = 8.5
        # IDCG@3 ~ 9.3927 -> NDCG ~ 0.9049
        assert ndcg_at_k(RETRIEVED, JUDGMENTS, 3) == pytest.approx(0.9049, abs=1e-4)


class TestAverage:
    def test_average_of_values(self):
        assert average([1.0, 2.0, 3.0]) == 2.0

    def test_average_empty(self):
        assert average([]) == 0.0

"""Tests for metric computations with known inputs and outputs."""

import numpy as np
import pytest
from benchmark.metrics import buy_rate_lift, wr_diff, wr_diff_ci95, brier_score, log_loss, recall_at_k, ndcg_at_k


class TestBuyRateLift:
    def test_same_rate(self):
        assert buy_rate_lift(100, 500, 200, 1000) == pytest.approx(1.0)

    def test_double_rate(self):
        # Context: 200/500 = 0.4, Baseline: 200/1000 = 0.2 → lift = 2.0
        assert buy_rate_lift(200, 500, 200, 1000) == pytest.approx(2.0)

    def test_half_rate(self):
        assert buy_rate_lift(50, 500, 200, 1000) == pytest.approx(0.5)

    def test_zero_baseline(self):
        assert buy_rate_lift(10, 100, 0, 1000) == 1.0

    def test_zero_context(self):
        assert buy_rate_lift(0, 0, 100, 1000) == 1.0


class TestWrDiff:
    def test_positive_diff(self):
        # 120/200 = 0.6 with, (240-120)/(500-200) = 0.4 without → +0.2
        assert wr_diff(120, 200, 240, 500) == pytest.approx(0.2)

    def test_negative_diff(self):
        # 80/200 = 0.4 with, (260-80)/(500-200) = 0.6 without → -0.2
        assert wr_diff(80, 200, 260, 500) == pytest.approx(-0.2)

    def test_zero_diff(self):
        assert wr_diff(100, 200, 250, 500) == pytest.approx(0.0)

    def test_zero_games(self):
        assert wr_diff(0, 0, 100, 500) == 0.0

    def test_all_games(self):
        # games_with == total_games → no "without" games
        assert wr_diff(100, 500, 250, 500) == 0.0

    def test_ci95_returns_positive(self):
        diff, ci = wr_diff_ci95(120, 200, 240, 500)
        assert diff == pytest.approx(0.2)
        assert ci > 0


class TestBrierScore:
    def test_perfect(self):
        assert brier_score(np.array([1, 0, 1]), np.array([1.0, 0.0, 1.0])) == pytest.approx(0.0)

    def test_worst(self):
        assert brier_score(np.array([1, 0]), np.array([0.0, 1.0])) == pytest.approx(1.0)

    def test_coin_flip(self):
        assert brier_score(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.25)


class TestLogLoss:
    def test_perfect(self):
        assert log_loss(np.array([1, 0]), np.array([0.999, 0.001])) < 0.01

    def test_coin_flip(self):
        assert log_loss(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.6931, abs=0.01)


class TestRecallAtK:
    def test_all_hits(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
        assert recall_at_k(y_true, y_scores, k=2) == pytest.approx(1.0)

    def test_no_hits(self):
        y_true = np.array([0, 0, 1, 1, 0])
        y_scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
        assert recall_at_k(y_true, y_scores, k=2) == pytest.approx(0.0)

    def test_partial(self):
        y_true = np.array([1, 0, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
        assert recall_at_k(y_true, y_scores, k=3) == pytest.approx(1.0)


class TestNdcgAtK:
    def test_perfect_ranking(self):
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert ndcg_at_k(y_true, y_scores, k=4) == pytest.approx(1.0)

    def test_no_relevant(self):
        y_true = np.array([0, 0, 0])
        y_scores = np.array([0.9, 0.5, 0.1])
        assert ndcg_at_k(y_true, y_scores, k=3) == pytest.approx(0.0)

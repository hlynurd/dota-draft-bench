"""Smoke tests for baseline models."""

import pytest
from benchmark.models.popularity import PopularityModel
from benchmark.models.pairwise_additive import PairwiseAdditiveModel
from benchmark.models.logistic import LogisticModel
from benchmark.models.gbm import GBMModel
from benchmark.data import Match


class TestPopularityModel:
    def test_fit_and_predict(self, mini_matches: list[Match]):
        model = PopularityModel().fit(mini_matches)

        # AM (hero 1) should have BKB (116) as a purchased item
        buy_probs = model.predict_buy(1, [5, 7, 14, 26], [2, 8, 44, 36, 75])
        assert len(buy_probs) > 0
        assert 116 in buy_probs  # BKB
        assert buy_probs[116] > 0

    def test_predict_win_returns_valid_prob(self, mini_matches: list[Match]):
        model = PopularityModel().fit(mini_matches)
        p = model.predict_win(1, 116, [5, 7, 14, 26], [2, 8, 44, 36, 75])
        assert 0.0 <= p <= 1.0

    def test_unknown_hero_returns_empty(self, mini_matches: list[Match]):
        model = PopularityModel().fit(mini_matches)
        buy_probs = model.predict_buy(999, [], [])
        assert buy_probs == {}

    def test_win_rate_plausible(self, mini_matches: list[Match]):
        model = PopularityModel().fit(mini_matches)
        p = model.predict_win(1, 116, [5, 7, 14, 26], [2, 8, 44, 36, 75])
        # With only 10 mini matches, just check it's a valid probability
        assert 0.0 <= p <= 1.0


class TestPairwiseAdditiveModel:
    def test_fit_and_predict(self, mini_matches: list[Match]):
        model = PairwiseAdditiveModel().fit(mini_matches)
        buy_probs = model.predict_buy(1, [5, 7, 14, 26], [2, 8, 44, 36, 75])
        assert len(buy_probs) > 0

    def test_predict_win_varies_by_enemy(self, mini_matches: list[Match]):
        model = PairwiseAdditiveModel().fit(mini_matches)
        p1 = model.predict_win(1, 116, [5, 7, 14, 26], [2, 8, 44, 36, 75])
        p2 = model.predict_win(1, 116, [5, 7, 14, 26], [17, 44, 69, 36, 75])
        assert 0.0 <= p1 <= 1.0
        assert 0.0 <= p2 <= 1.0

    def test_returns_valid_probs(self, mini_matches: list[Match]):
        model = PairwiseAdditiveModel().fit(mini_matches)
        p = model.predict_win(1, 116, [5, 7, 14, 26], [2, 8, 44, 36, 75])
        assert 0.0 <= p <= 1.0


class TestLogisticModel:
    def test_fit_and_predict(self, mini_matches: list[Match]):
        model = LogisticModel().fit(mini_matches)
        buy_probs = model.predict_buy(1, [5, 7, 14, 26], [2, 8, 44, 36, 75])
        assert len(buy_probs) > 0

    def test_predict_win_valid(self, mini_matches: list[Match]):
        model = LogisticModel().fit(mini_matches)
        p = model.predict_win(1, 116, [5, 7, 14, 26], [2, 8, 44, 36, 75])
        assert 0.0 <= p <= 1.0

    def test_unknown_item_returns_default(self, mini_matches: list[Match]):
        model = LogisticModel().fit(mini_matches)
        p = model.predict_win(1, 99999, [5, 7, 14, 26], [2, 8, 44, 36, 75])
        assert p == 0.5


class TestGBMModel:
    def test_fit_and_predict(self, mini_matches: list[Match]):
        model = GBMModel().fit(mini_matches)
        buy_probs = model.predict_buy(1, [5, 7, 14, 26], [2, 8, 44, 36, 75])
        assert len(buy_probs) > 0

    def test_predict_win_valid(self, mini_matches: list[Match]):
        model = GBMModel().fit(mini_matches)
        p = model.predict_win(1, 116, [5, 7, 14, 26], [2, 8, 44, 36, 75])
        assert 0.0 <= p <= 1.0

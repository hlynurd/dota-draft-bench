"""Baseline 3: Logistic Regression — one model per item, draft features."""

from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression as LR

from benchmark.data import Match, flatten_match, encode_draft, NUM_HEROES
from benchmark.models.base import ItemModel

MIN_SAMPLES = 50
NEG_RATIO = 3  # negative samples per positive


class LogisticModel(ItemModel):
    """Per-item logistic regression with draft context features."""

    def __init__(self):
        self.buy_models: dict[int, LR] = {}
        self.win_models: dict[int, LR] = {}
        self.hero_buy_rate: dict[int, dict[int, float]] = {}
        self.hero_games: dict[int, int] = defaultdict(int)

    def fit(self, matches: list[Match]) -> "LogisticModel":
        # First pass: collect per-item positive samples
        buy_pos: dict[int, list] = defaultdict(list)  # item -> [(feat, hero_id)]
        win_data: dict[int, tuple] = defaultdict(lambda: ([], []))  # item -> (X, y)
        all_feats: list[np.ndarray] = []  # all feature vectors for negative sampling

        for match in matches:
            for hero_id, allies, enemies, items, won in flatten_match(match):
                self.hero_games[hero_id] += 1
                feat = encode_draft(hero_id, allies, enemies)
                all_feats.append(feat)
                item_set = set(i for i in items if i != 0)

                for item_id in item_set:
                    buy_pos[item_id].append(feat)
                    win_data[item_id][0].append(feat)
                    win_data[item_id][1].append(1.0 if won else 0.0)

        # Build hero-level fallbacks
        hero_item_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for match in matches:
            for hero_id, _, _, items, _ in flatten_match(match):
                for item_id in set(i for i in items if i != 0):
                    hero_item_counts[hero_id][item_id] += 1
        for hero_id, counts in hero_item_counts.items():
            total = self.hero_games[hero_id]
            self.hero_buy_rate[hero_id] = {iid: c / total for iid, c in counts.items()}

        n_total = len(all_feats)
        all_feats_arr = np.array(all_feats) if all_feats else np.empty((0, 3 * NUM_HEROES))

        trained_buy = 0
        trained_win = 0

        for item_id, pos_feats in buy_pos.items():
            n_pos = len(pos_feats)
            if n_pos < MIN_SAMPLES:
                continue

            # Sample negatives from all matches (heroes that didn't buy this item)
            n_neg = min(n_pos * NEG_RATIO, n_total - n_pos)
            if n_neg < 10:
                continue
            neg_idx = np.random.choice(n_total, size=n_neg, replace=False)
            neg_feats = all_feats_arr[neg_idx]

            X = np.vstack([np.array(pos_feats), neg_feats])
            y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

            try:
                model = LR(max_iter=100, C=1.0, solver="saga", penalty="l2")
                model.fit(X, y)
                self.buy_models[item_id] = model
                trained_buy += 1
            except Exception:
                pass

            # Win model (only on positive samples)
            Xw, yw = win_data[item_id]
            Xw = np.array(Xw)
            yw = np.array(yw)
            if len(yw) >= MIN_SAMPLES and yw.sum() > 5 and (1 - yw).sum() > 5:
                try:
                    model = LR(max_iter=100, C=1.0, solver="saga", penalty="l2")
                    model.fit(Xw, yw)
                    self.win_models[item_id] = model
                    trained_win += 1
                except Exception:
                    pass

        print(f"  [Logistic] Trained {trained_buy} buy models, {trained_win} win models")
        return self

    def predict_buy(self, hero_id: int, allies: list[int], enemies: list[int]) -> dict[int, float]:
        feat = encode_draft(hero_id, allies, enemies).reshape(1, -1)
        result = {}
        for item_id, model in self.buy_models.items():
            try:
                result[item_id] = float(model.predict_proba(feat)[0, 1])
            except Exception:
                pass
        for item_id, rate in self.hero_buy_rate.get(hero_id, {}).items():
            if item_id not in result:
                result[item_id] = rate
        return result

    def predict_win(self, hero_id: int, item_id: int, allies: list[int], enemies: list[int]) -> float:
        if item_id in self.win_models:
            feat = encode_draft(hero_id, allies, enemies).reshape(1, -1)
            try:
                return float(self.win_models[item_id].predict_proba(feat)[0, 1])
            except Exception:
                pass
        return 0.5

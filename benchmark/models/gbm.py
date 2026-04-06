"""Baseline 4: Gradient-Boosted Trees (LightGBM) — one model per item."""

from collections import defaultdict
import numpy as np
import lightgbm as lgb

from benchmark.data import Match, flatten_match, encode_draft, NUM_HEROES
from benchmark.models.base import ItemModel

MIN_SAMPLES = 50


class GBMModel(ItemModel):
    """Per-item LightGBM with draft context features."""

    def __init__(self):
        self.buy_models: dict[int, lgb.Booster] = {}
        self.win_models: dict[int, lgb.Booster] = {}
        self.hero_buy_rate: dict[int, dict[int, float]] = {}
        self.hero_games: dict[int, int] = defaultdict(int)

    def fit(self, matches: list[Match]) -> "GBMModel":
        buy_X: dict[int, list] = defaultdict(list)
        buy_y: dict[int, list] = defaultdict(list)
        win_X: dict[int, list] = defaultdict(list)
        win_y: dict[int, list] = defaultdict(list)

        all_items: set[int] = set()
        hero_item_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for match in matches:
            for hero_id, allies, enemies, items, won in flatten_match(match):
                self.hero_games[hero_id] += 1
                feat = encode_draft(hero_id, allies, enemies)
                item_set = set(i for i in items if i != 0)
                all_items.update(item_set)

                for item_id in item_set:
                    hero_item_counts[hero_id][item_id] += 1
                    buy_X[item_id].append(feat)
                    buy_y[item_id].append(1.0)
                    win_X[item_id].append(feat)
                    win_y[item_id].append(1.0 if won else 0.0)

                for item_id in all_items - item_set:
                    buy_X[item_id].append(feat)
                    buy_y[item_id].append(0.0)

        for hero_id, counts in hero_item_counts.items():
            total = self.hero_games[hero_id]
            self.hero_buy_rate[hero_id] = {iid: c / total for iid, c in counts.items()}

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "n_estimators": 50,
            "min_child_samples": 10,
        }

        for item_id in all_items:
            X = np.array(buy_X[item_id])
            y = np.array(buy_y[item_id])
            if len(y) >= MIN_SAMPLES and y.sum() > 5 and (1 - y).sum() > 5:
                try:
                    ds = lgb.Dataset(X, label=y, free_raw_data=False)
                    model = lgb.train(params, ds, num_boost_round=50, verbose_eval=False)
                    self.buy_models[item_id] = model
                except Exception:
                    pass

            Xw = np.array(win_X[item_id]) if win_X[item_id] else np.empty((0, 3 * NUM_HEROES))
            yw = np.array(win_y[item_id]) if win_y[item_id] else np.empty(0)
            if len(yw) >= MIN_SAMPLES and yw.sum() > 5 and (1 - yw).sum() > 5:
                try:
                    ds = lgb.Dataset(Xw, label=yw, free_raw_data=False)
                    model = lgb.train(params, ds, num_boost_round=50, verbose_eval=False)
                    self.win_models[item_id] = model
                except Exception:
                    pass

        print(f"  [GBM] Trained {len(self.buy_models)} buy models, {len(self.win_models)} win models")
        return self

    def predict_buy(self, hero_id: int, allies: list[int], enemies: list[int]) -> dict[int, float]:
        feat = encode_draft(hero_id, allies, enemies).reshape(1, -1)
        result = {}
        for item_id, model in self.buy_models.items():
            result[item_id] = float(model.predict(feat)[0])
        for item_id, rate in self.hero_buy_rate.get(hero_id, {}).items():
            if item_id not in result:
                result[item_id] = rate
        return result

    def predict_win(self, hero_id: int, item_id: int, allies: list[int], enemies: list[int]) -> float:
        if item_id in self.win_models:
            feat = encode_draft(hero_id, allies, enemies).reshape(1, -1)
            return float(np.clip(self.win_models[item_id].predict(feat)[0], 0.01, 0.99))
        return 0.5

"""Baseline 1: Popularity — most popular items per hero, no draft context."""

from collections import defaultdict
from benchmark.data import Match, flatten_match
from benchmark.models.base import ItemModel


class PopularityModel(ItemModel):
    """Predicts items based solely on hero-specific purchase frequency."""

    def __init__(self):
        # hero_id -> item_id -> {bought, won, total}
        self.stats: dict[int, dict[int, dict]] = defaultdict(lambda: defaultdict(lambda: {"bought": 0, "won": 0, "total": 0}))
        self.hero_games: dict[int, int] = defaultdict(int)
        self.hero_wins: dict[int, int] = defaultdict(int)

    def fit(self, matches: list[Match]) -> "PopularityModel":
        for match in matches:
            for hero_id, allies, enemies, items, won in flatten_match(match):
                self.hero_games[hero_id] += 1
                if won:
                    self.hero_wins[hero_id] += 1
                seen = set()
                for item_id in items:
                    if item_id == 0 or item_id in seen:
                        continue
                    seen.add(item_id)
                    self.stats[hero_id][item_id]["bought"] += 1
                    if won:
                        self.stats[hero_id][item_id]["won"] += 1
                    self.stats[hero_id][item_id]["total"] += 1
        return self

    def predict_buy(self, hero_id: int, allies: list[int], enemies: list[int]) -> dict[int, float]:
        total = self.hero_games.get(hero_id, 0)
        if total == 0:
            return {}
        return {
            item_id: s["bought"] / total
            for item_id, s in self.stats.get(hero_id, {}).items()
        }

    def predict_win(self, hero_id: int, item_id: int, allies: list[int], enemies: list[int]) -> float:
        s = self.stats.get(hero_id, {}).get(item_id)
        if not s or s["bought"] == 0:
            # Fall back to hero base win rate
            total = self.hero_games.get(hero_id, 0)
            wins = self.hero_wins.get(hero_id, 0)
            return wins / total if total > 0 else 0.5
        return s["won"] / s["bought"]

"""Baseline 2: Pairwise Additive — sum pairwise log-odds with shrinkage.

Port of the additive model from dota-items/scripts/predict-draft.ts.
For each (hero, item, context_hero, side) triple, compute pairwise win rate,
then combine across all 9 context heroes using additive log-odds.
"""

import math
from collections import defaultdict
from benchmark.data import Match, flatten_match
from benchmark.models.base import ItemModel

SHRINKAGE = 0.5  # lambda for combining pairwise excesses


def logit(p: float) -> float:
    p = max(1e-7, min(1 - 1e-7, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-max(-20, min(20, x))))


class PairwiseAdditiveModel(ItemModel):
    """Additive decomposition: P(win) ≈ sigmoid(base_logit + λ * Σ excess_k)."""

    def __init__(self, shrinkage: float = SHRINKAGE):
        self.shrinkage = shrinkage
        # (hero, item, context_hero, side) -> {games, wins}
        self.pairwise: dict[tuple, dict] = defaultdict(lambda: {"games": 0, "wins": 0})
        # (hero, item) -> {games, wins, total_hero_games}
        self.baselines: dict[tuple, dict] = defaultdict(lambda: {"games": 0, "wins": 0})
        self.hero_games: dict[int, int] = defaultdict(int)
        self.hero_wins: dict[int, int] = defaultdict(int)
        # (hero, item) -> {bought, won} for buy prediction
        self.buy_stats: dict[tuple, dict] = defaultdict(lambda: {"bought": 0})
        # (hero, item, context_hero, side) -> {bought, total} for buy rate
        self.buy_pairwise: dict[tuple, dict] = defaultdict(lambda: {"bought": 0, "total": 0})

    def fit(self, matches: list[Match]) -> "PairwiseAdditiveModel":
        for match in matches:
            for hero_id, allies, enemies, items, won in flatten_match(match):
                self.hero_games[hero_id] += 1
                if won:
                    self.hero_wins[hero_id] += 1

                item_set = set(i for i in items if i != 0)

                for item_id in item_set:
                    self.baselines[(hero_id, item_id)]["games"] += 1
                    if won:
                        self.baselines[(hero_id, item_id)]["wins"] += 1
                    self.buy_stats[(hero_id, item_id)]["bought"] += 1

                    # Pairwise: this item vs each context hero
                    for enemy in enemies:
                        key = (hero_id, item_id, enemy, "enemy")
                        self.pairwise[key]["games"] += 1
                        if won:
                            self.pairwise[key]["wins"] += 1
                        self.buy_pairwise[key]["bought"] += 1
                        self.buy_pairwise[key]["total"] += 1

                    for ally in allies:
                        key = (hero_id, item_id, ally, "ally")
                        self.pairwise[key]["games"] += 1
                        if won:
                            self.pairwise[key]["wins"] += 1
                        self.buy_pairwise[key]["bought"] += 1
                        self.buy_pairwise[key]["total"] += 1

                # Also track "didn't buy" for context heroes
                for enemy in enemies:
                    for item_id in item_set:
                        pass  # already counted above
                    # Count total appearances for buy rate denominator
                    for item_id in self.buy_stats:
                        if item_id[0] == hero_id:
                            key = (hero_id, item_id[1], enemy, "enemy")
                            # Only increment total if not already done
                            # Actually, we need total context appearances, not just item appearances
                            pass

        return self

    def predict_buy(self, hero_id: int, allies: list[int], enemies: list[int]) -> dict[int, float]:
        total = self.hero_games.get(hero_id, 0)
        if total == 0:
            return {}

        result = {}
        for (h, item_id), stats in self.buy_stats.items():
            if h != hero_id:
                continue
            base_rate = stats["bought"] / total
            # Could add pairwise adjustments here, but for v1 use baseline
            result[item_id] = base_rate
        return result

    def predict_win(self, hero_id: int, item_id: int, allies: list[int], enemies: list[int]) -> float:
        base = self.baselines.get((hero_id, item_id))
        if not base or base["games"] == 0:
            total = self.hero_games.get(hero_id, 0)
            wins = self.hero_wins.get(hero_id, 0)
            return wins / total if total > 0 else 0.5
        base_wr = base["wins"] / base["games"]
        base_logit = logit(base_wr)

        # Sum pairwise excesses
        excess = 0.0
        count = 0
        for enemy in enemies:
            pw = self.pairwise.get((hero_id, item_id, enemy, "enemy"))
            if pw and pw["games"] >= 5:
                pw_wr = pw["wins"] / pw["games"]
                excess += logit(pw_wr) - base_logit
                count += 1
        for ally in allies:
            pw = self.pairwise.get((hero_id, item_id, ally, "ally"))
            if pw and pw["games"] >= 5:
                pw_wr = pw["wins"] / pw["games"]
                excess += logit(pw_wr) - base_logit
                count += 1

        return sigmoid(base_logit + self.shrinkage * excess)

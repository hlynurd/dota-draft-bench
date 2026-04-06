"""Abstract base class for item recommendation models."""

from abc import ABC, abstractmethod
from benchmark.data import Match


class ItemModel(ABC):
    """Base class for all models.

    Models predict two things:
      1. P(hero buys item | draft context) — buy prediction
      2. P(team wins | hero bought item, draft context) — win prediction
    """

    @abstractmethod
    def fit(self, matches: list[Match]) -> "ItemModel":
        """Train on a list of matches."""
        ...

    @abstractmethod
    def predict_buy(self, hero_id: int, allies: list[int], enemies: list[int]) -> dict[int, float]:
        """Predict P(buy) for each item given a draft context.

        Returns: {item_id: probability}
        """
        ...

    @abstractmethod
    def predict_win(self, hero_id: int, item_id: int, allies: list[int], enemies: list[int]) -> float:
        """Predict P(win | hero bought item, draft context)."""
        ...

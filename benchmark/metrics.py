"""Core metrics for the Dota 2 item recommendation benchmark.

Two user-facing metrics:
  - buy_rate_lift: how much more likely is this item purchased in this context
  - wr_diff: win rate when hero has item minus win rate when hero doesn't

Plus standard ML metrics for evaluation:
  - brier_score, log_loss, recall_at_k, ndcg_at_k
"""

import numpy as np
from typing import Optional


# ─── User-facing metrics ─────────────────────────────────────────────────────

def buy_rate_lift(
    games_in_context: int,
    total_in_context: int,
    games_baseline: int,
    total_baseline: int,
) -> float:
    """How much more likely this item is bought in this context vs baseline.

    Returns a multiplier: 1.0 = same rate, 2.0 = bought 2x more often.
    """
    if total_in_context == 0 or total_baseline == 0 or games_baseline == 0:
        return 1.0
    context_rate = games_in_context / total_in_context
    baseline_rate = games_baseline / total_baseline
    return context_rate / baseline_rate


def wr_diff(
    wins_with: int,
    games_with: int,
    total_wins: int,
    total_games: int,
) -> float:
    """Win rate when item is bought minus win rate when it isn't.

    Args:
        wins_with: wins in matches where the hero bought this item
        games_with: total matches where the hero bought this item
        total_wins: total wins for this hero in this context
        total_games: total matches for this hero in this context
    """
    if games_with == 0 or total_games <= games_with:
        return 0.0
    wr_with = wins_with / games_with
    wr_without = (total_wins - wins_with) / (total_games - games_with)
    return wr_with - wr_without


def wr_diff_ci95(
    wins_with: int,
    games_with: int,
    total_wins: int,
    total_games: int,
) -> tuple[float, float]:
    """WR diff with approximate 95% confidence interval half-width."""
    diff = wr_diff(wins_with, games_with, total_wins, total_games)
    if games_with == 0 or total_games <= games_with:
        return diff, 0.0
    p = abs(diff)
    n = games_with  # conservative: use the smaller sample
    ci = 1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
    return diff, ci


# ─── Standard ML metrics ─────────────────────────────────────────────────────

def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    return float(np.mean((y_pred - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross-entropy loss."""
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 6) -> float:
    """Fraction of true items that appear in the top-k predictions.

    Args:
        y_true: binary array (1 = hero had this item)
        y_scores: predicted scores for each item
        k: number of top items to consider
    """
    if y_true.sum() == 0:
        return 0.0
    top_k = np.argsort(y_scores)[::-1][:k]
    hits = y_true[top_k].sum()
    return float(hits / min(y_true.sum(), k))


def ndcg_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int = 6) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    top_k = np.argsort(y_scores)[::-1][:k]
    dcg = sum(y_true[idx] / np.log2(rank + 2) for rank, idx in enumerate(top_k))
    # Ideal DCG: all true items at the top
    n_relevant = int(min(y_true.sum(), k))
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(n_relevant))
    return float(dcg / idcg) if idcg > 0 else 0.0

"""Evaluation harness — train all models, compute metrics, produce results table."""

import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

from benchmark.data import load_matches, temporal_split, flatten_match, NUM_HEROES
from benchmark.metrics import brier_score, log_loss, recall_at_k, ndcg_at_k
from benchmark.models.popularity import PopularityModel
from benchmark.models.pairwise_additive import PairwiseAdditiveModel
from benchmark.models.logistic import LogisticModel
from benchmark.models.gbm import GBMModel
from benchmark.models.neural import NeuralModel


def collect_items(matches) -> set[int]:
    """Get all item IDs seen in the dataset."""
    items = set()
    for m in matches:
        for _, _, _, item_list, _ in flatten_match(m):
            items.update(i for i in item_list if i != 0)
    return items


def evaluate_buy_prediction(model, test_matches, all_items: list[int]):
    """Evaluate item buy prediction: recall@6, ndcg@6."""
    recall_scores = []
    ndcg_scores = []

    for match in test_matches:
        for hero_id, allies, enemies, true_items, won in flatten_match(match):
            true_set = set(i for i in true_items if i != 0)
            if len(true_set) == 0:
                continue

            buy_probs = model.predict_buy(hero_id, allies, enemies)
            if not buy_probs:
                continue

            # Build arrays aligned to all_items
            y_true = np.array([1.0 if item in true_set else 0.0 for item in all_items])
            y_scores = np.array([buy_probs.get(item, 0.0) for item in all_items])

            if y_true.sum() == 0:
                continue

            recall_scores.append(recall_at_k(y_true, y_scores, k=6))
            ndcg_scores.append(ndcg_at_k(y_true, y_scores, k=6))

    return {
        "recall@6": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "ndcg@6": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "n_samples": len(recall_scores),
    }


def evaluate_win_prediction(model, test_matches, all_items: list[int]):
    """Evaluate win prediction: brier score, log-loss."""
    y_true_list = []
    y_pred_list = []

    for match in test_matches:
        for hero_id, allies, enemies, true_items, won in flatten_match(match):
            true_set = set(i for i in true_items if i != 0)
            for item_id in true_set:
                p = model.predict_win(hero_id, item_id, allies, enemies)
                y_true_list.append(1.0 if won else 0.0)
                y_pred_list.append(p)

    if not y_true_list:
        return {"brier": 1.0, "log_loss": 10.0, "n_samples": 0}

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    return {
        "brier": brier_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred),
        "n_samples": len(y_true_list),
        "mean_pred": float(y_pred.mean()),
        "actual_wr": float(y_true.mean()),
    }


def run_benchmark():
    print("Loading data...")
    matches = load_matches()
    print(f"Loaded {len(matches):,} matches")

    train, val, test = temporal_split(matches)
    print(f"Split: train={len(train):,} val={len(val):,} test={len(test):,}")

    all_items = sorted(collect_items(matches))
    print(f"Unique items: {len(all_items)}")

    models = {
        "Popularity": PopularityModel(),
        "Pairwise Additive": PairwiseAdditiveModel(),
        "Logistic Regression": LogisticModel(),
        "LightGBM": GBMModel(),
        "Neural (MLP)": NeuralModel(epochs=10),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        t0 = time.time()
        model.fit(train)
        train_time = time.time() - t0
        print(f"  Trained in {train_time:.1f}s")

        print(f"  Evaluating buy prediction...")
        buy_metrics = evaluate_buy_prediction(model, test, all_items)
        print(f"    Recall@6: {buy_metrics['recall@6']:.4f}")
        print(f"    NDCG@6:   {buy_metrics['ndcg@6']:.4f}")
        print(f"    Samples:  {buy_metrics['n_samples']:,}")

        print(f"  Evaluating win prediction...")
        win_metrics = evaluate_win_prediction(model, test, all_items)
        print(f"    Brier:    {win_metrics['brier']:.4f}")
        print(f"    Log-loss: {win_metrics['log_loss']:.4f}")
        print(f"    Mean pred:{win_metrics['mean_pred']:.4f}")
        print(f"    Actual WR:{win_metrics['actual_wr']:.4f}")
        print(f"    Samples:  {win_metrics['n_samples']:,}")

        results[name] = {
            "train_time": train_time,
            "buy": buy_metrics,
            "win": win_metrics,
        }

    # Print summary table
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Recall@6':>10} {'NDCG@6':>10} {'Brier':>10} {'LogLoss':>10}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<25} {r['buy']['recall@6']:>10.4f} {r['buy']['ndcg@6']:>10.4f} {r['win']['brier']:>10.4f} {r['win']['log_loss']:>10.4f}")

    return results


if __name__ == "__main__":
    run_benchmark()

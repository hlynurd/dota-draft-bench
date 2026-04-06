# Dota 2 Draft-Context Item Recommendation Benchmark

Hero-specific item recommendations given the full 10-hero draft context. Published dataset + benchmark suite + web app.

## Quick Start

```bash
# Install dependencies
pip install pandas pyarrow numpy scikit-learn lightgbm pytest

# Convert raw data to Parquet (requires ../dota-items/data/matches.ndjson)
python data/convert.py

# Run tests
PYTHONPATH=. pytest benchmark/tests/ -v

# Run benchmark
PYTHONPATH=. python benchmark/evaluate.py
```

## Data

Raw match data is collected by the [dota-items](https://github.com/hlynurd/dota-items) harvester from the Valve Steam API. Each match contains the full 10-hero lineup with per-hero end-game items and win/loss outcome.

**NDJSON format** (one line per match):
```json
{"m":8747967449,"w":1,"s":2106,"r":[[20,[232,22,23]],[110,[73,214]],...],"e":[[64,[36,102]],[22,[77,110]],...]}}
```

**Parquet schema** (after `convert.py`):
| Column | Type | Description |
|---|---|---|
| match_id | int64 | Unique match ID |
| radiant_win | bool | Did radiant win? |
| duration | int32 | Seconds |
| radiant_heroes | list[int16] | 5 hero IDs |
| dire_heroes | list[int16] | 5 hero IDs |
| radiant_items | list[list[int16]] | Per-player item lists |
| dire_items | list[list[int16]] | Per-player item lists |
| split | string | train/val/test (temporal 70/15/15) |

## Metrics

- **Buy Rate Lift**: How much more likely this item is purchased in this draft context vs baseline
- **WR Diff**: P(win | hero has item, context) - P(win | hero doesn't, context)
- **Recall@6**: Fraction of actual items in top-6 predictions
- **NDCG@6**: Ranking quality of predictions
- **Brier Score**: Calibration of win probability predictions
- **Log-Loss**: Cross-entropy of win predictions

## Baseline Models

| Model | Description |
|---|---|
| Popularity | Most popular items per hero, no draft context |
| Pairwise Additive | Sum pairwise log-odds across 9 context heroes with shrinkage |

More baselines (logistic regression, LightGBM, neural) coming soon.

## License

MIT

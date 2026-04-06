"""Convert dota-items NDJSON match log to Parquet with train/val/test splits."""

import json
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

NDJSON_PATH = Path(__file__).resolve().parent.parent.parent / "dota-items" / "data" / "matches.ndjson"
OUTPUT_PATH = Path(__file__).resolve().parent / "matches.parquet"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (remainder)


def load_ndjson(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def convert(rows: list[dict]) -> pa.Table:
    match_ids = []
    radiant_wins = []
    durations = []
    radiant_heroes = []
    dire_heroes = []
    radiant_items = []
    dire_items = []
    splits = []

    n = len(rows)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    for i, row in enumerate(rows):
        match_ids.append(row["m"])
        radiant_wins.append(bool(row["w"]))
        durations.append(row["s"])

        r_heroes = [p[0] for p in row["r"]]
        d_heroes = [p[0] for p in row["e"]]
        r_items = [p[1] for p in row["r"]]
        d_items = [p[1] for p in row["e"]]

        # Pad to exactly 5 players per side
        while len(r_heroes) < 5:
            r_heroes.append(0)
            r_items.append([])
        while len(d_heroes) < 5:
            d_heroes.append(0)
            d_items.append([])

        radiant_heroes.append(r_heroes[:5])
        dire_heroes.append(d_heroes[:5])
        radiant_items.append(r_items[:5])
        dire_items.append(d_items[:5])

        if i < train_end:
            splits.append("train")
        elif i < val_end:
            splits.append("val")
        else:
            splits.append("test")

    schema = pa.schema([
        ("match_id", pa.int64()),
        ("radiant_win", pa.bool_()),
        ("duration", pa.int32()),
        ("radiant_heroes", pa.list_(pa.int16())),
        ("dire_heroes", pa.list_(pa.int16())),
        ("radiant_items", pa.list_(pa.list_(pa.int16()))),
        ("dire_items", pa.list_(pa.list_(pa.int16()))),
        ("split", pa.string()),
    ])

    table = pa.table({
        "match_id": match_ids,
        "radiant_win": radiant_wins,
        "duration": durations,
        "radiant_heroes": radiant_heroes,
        "dire_heroes": dire_heroes,
        "radiant_items": radiant_items,
        "dire_items": dire_items,
        "split": splits,
    }, schema=schema)

    return table


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else NDJSON_PATH
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_PATH

    if not src.exists():
        print(f"Source not found: {src}")
        sys.exit(1)

    print(f"Loading {src}...")
    rows = load_ndjson(src)
    print(f"Loaded {len(rows):,} matches")

    table = convert(rows)
    pq.write_table(table, dst, compression="zstd")
    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"Wrote {dst} ({size_mb:.1f} MB, {len(rows):,} rows)")

    # Print split counts
    for split in ["train", "val", "test"]:
        count = sum(1 for s in table.column("split").to_pylist() if s == split)
        print(f"  {split}: {count:,}")


if __name__ == "__main__":
    main()

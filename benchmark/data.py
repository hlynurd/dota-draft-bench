"""Load and preprocess match data for the benchmark."""

import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

NUM_HEROES = 200  # max hero ID (sparse, not all used)

NDJSON_PATH = Path(__file__).resolve().parent.parent.parent / "dota-items" / "data" / "matches.ndjson"
PARQUET_PATH = Path(__file__).resolve().parent.parent / "data" / "matches.parquet"


@dataclass
class Match:
    match_id: int
    radiant_win: bool
    duration: int
    radiant_heroes: list[int]   # 5 hero IDs
    dire_heroes: list[int]      # 5 hero IDs
    radiant_items: list[list[int]]  # 5 lists of item IDs
    dire_items: list[list[int]]


def load_ndjson(path: Path = NDJSON_PATH) -> list[Match]:
    """Load matches from the NDJSON file produced by the dota-items harvester."""
    matches = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            r_heroes = [p[0] for p in row["r"]]
            d_heroes = [p[0] for p in row["e"]]
            r_items = [p[1] for p in row["r"]]
            d_items = [p[1] for p in row["e"]]
            if len(r_heroes) != 5 or len(d_heroes) != 5:
                continue
            matches.append(Match(
                match_id=row["m"],
                radiant_win=bool(row["w"]),
                duration=row["s"],
                radiant_heroes=r_heroes,
                dire_heroes=d_heroes,
                radiant_items=r_items,
                dire_items=d_items,
            ))
    return matches


def load_parquet(path: Path = PARQUET_PATH) -> list[Match]:
    """Load matches from Parquet."""
    if not HAS_PARQUET:
        raise ImportError("pyarrow required for Parquet loading")
    table = pq.read_table(path)
    matches = []
    for i in range(len(table)):
        matches.append(Match(
            match_id=table["match_id"][i].as_py(),
            radiant_win=table["radiant_win"][i].as_py(),
            duration=table["duration"][i].as_py(),
            radiant_heroes=table["radiant_heroes"][i].as_py(),
            dire_heroes=table["dire_heroes"][i].as_py(),
            radiant_items=table["radiant_items"][i].as_py(),
            dire_items=table["dire_items"][i].as_py(),
        ))
    return matches


def load_matches() -> list[Match]:
    """Load from Parquet if available, else NDJSON."""
    if HAS_PARQUET and PARQUET_PATH.exists():
        return load_parquet()
    return load_ndjson()


def temporal_split(matches: list[Match], train: float = 0.7, val: float = 0.15):
    """Split by position (matches are in temporal order from harvester)."""
    n = len(matches)
    t = int(n * train)
    v = int(n * (train + val))
    return matches[:t], matches[t:v], matches[v:]


def encode_draft(hero_id: int, allies: list[int], enemies: list[int]) -> np.ndarray:
    """Encode a draft context as a 3*NUM_HEROES binary vector.

    [hero_onehot | ally_vec | enemy_vec]
    """
    vec = np.zeros(3 * NUM_HEROES, dtype=np.float32)
    vec[hero_id] = 1.0
    for a in allies:
        vec[NUM_HEROES + a] = 1.0
    for e in enemies:
        vec[2 * NUM_HEROES + e] = 1.0
    return vec


def flatten_match(match: Match):
    """Yield (hero_id, allies, enemies, items, won) for each of the 10 players."""
    for side in ["radiant", "dire"]:
        if side == "radiant":
            heroes = match.radiant_heroes
            items_list = match.radiant_items
            opponents = match.dire_heroes
            won = match.radiant_win
        else:
            heroes = match.dire_heroes
            items_list = match.dire_items
            opponents = match.radiant_heroes
            won = not match.radiant_win

        for i, hero_id in enumerate(heroes):
            allies = [h for j, h in enumerate(heroes) if j != i]
            items = items_list[i] if i < len(items_list) else []
            yield hero_id, allies, opponents, items, won

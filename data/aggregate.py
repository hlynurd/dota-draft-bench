"""Aggregate raw NDJSON matches into hero-specific pairwise stats for the web app.

Reads matches.ndjson, computes:
  - Per (buyer_hero, item) baselines: games bought, wins
  - Per (buyer_hero, item, context_hero, side) pairwise: games, wins
  - Per hero: total games, wins

Writes web/public/draft-data.json in compact tuple format.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

NDJSON_PATH = Path(__file__).resolve().parent.parent.parent / "dota-items" / "data" / "matches.ndjson"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "web" / "public" / "draft-data.json"

MIN_GAMES = 10  # minimum games for inclusion in output


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else NDJSON_PATH
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_PATH

    print(f"Loading {src}...")
    matches = []
    with open(src) as f:
        for line in f:
            if line.strip():
                matches.append(json.loads(line))
    print(f"Loaded {len(matches):,} matches")

    # Accumulators
    # (buyer_hero, item) -> {bought, won}
    baselines = defaultdict(lambda: {"bought": 0, "won": 0})
    # (buyer_hero, item, context_hero, side) -> {games, wins}
    pairwise = defaultdict(lambda: {"games": 0, "wins": 0})
    # hero -> {games, wins}
    hero_totals = defaultdict(lambda: {"games": 0, "wins": 0})

    for row in matches:
        radiant_win = bool(row["w"])
        r_players = row["r"]  # [[hero, [items]], ...]
        d_players = row["e"]  # [[hero, [items]], ...]

        r_heroes = [p[0] for p in r_players]
        d_heroes = [p[0] for p in d_players]

        if len(r_heroes) != 5 or len(d_heroes) != 5:
            continue

        # Process each player
        for side_players, opponents, won in [
            (r_players, d_heroes, radiant_win),
            (d_players, r_heroes, not radiant_win),
        ]:
            side_heroes = [p[0] for p in side_players]
            for i, (hero_id, items) in enumerate(side_players):
                hero_totals[hero_id]["games"] += 1
                if won:
                    hero_totals[hero_id]["wins"] += 1

                item_set = set(it for it in items if it != 0)
                allies = [h for j, h in enumerate(side_heroes) if j != i]

                for item_id in item_set:
                    baselines[(hero_id, item_id)]["bought"] += 1
                    if won:
                        baselines[(hero_id, item_id)]["won"] += 1

                    # Pairwise vs enemies
                    for enemy in opponents:
                        key = (hero_id, item_id, enemy, 0)  # 0 = enemy
                        pairwise[key]["games"] += 1
                        if won:
                            pairwise[key]["wins"] += 1

                    # Pairwise with allies
                    for ally in allies:
                        key = (hero_id, item_id, ally, 1)  # 1 = ally
                        pairwise[key]["games"] += 1
                        if won:
                            pairwise[key]["wins"] += 1

    # Write compact JSON
    # p: [buyer_hero, item, context_hero, side(0=enemy,1=ally), games, wins]
    p_tuples = []
    for (bh, item, ch, side), stats in pairwise.items():
        if stats["games"] >= MIN_GAMES:
            p_tuples.append([bh, item, ch, side, stats["games"], stats["wins"]])

    # b: [buyer_hero, item, bought, won]
    b_tuples = []
    for (bh, item), stats in baselines.items():
        if stats["bought"] >= MIN_GAMES:
            b_tuples.append([bh, item, stats["bought"], stats["won"]])

    # h: [hero_id, games, wins]
    h_tuples = [[hid, s["games"], s["wins"]] for hid, s in hero_totals.items()]

    data = {"p": p_tuples, "b": b_tuples, "h": h_tuples, "ts": len(matches)}

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(data, f)

    size_kb = dst.stat().st_size / 1024
    print(f"Wrote {dst}")
    print(f"  Pairwise: {len(p_tuples):,} rows")
    print(f"  Baselines: {len(b_tuples):,} rows")
    print(f"  Heroes: {len(h_tuples)} heroes")
    print(f"  Size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()

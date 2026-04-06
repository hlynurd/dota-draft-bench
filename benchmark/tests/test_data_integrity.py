"""Data integrity tests — validates the NDJSON match data."""

import json
import pytest
from pathlib import Path

NDJSON_PATH = Path(__file__).resolve().parent.parent.parent.parent / "dota-items" / "data" / "matches.ndjson"


@pytest.fixture
def raw_matches():
    if not NDJSON_PATH.exists():
        pytest.skip(f"No data file at {NDJSON_PATH}")
    rows = []
    with open(NDJSON_PATH) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class TestNdjsonSchema:
    def test_has_matches(self, raw_matches):
        assert len(raw_matches) > 0

    def test_required_fields(self, raw_matches):
        for row in raw_matches[:100]:
            assert "m" in row  # match_id
            assert "w" in row  # win
            assert "s" in row  # duration
            assert "r" in row  # radiant
            assert "e" in row  # dire (enemy)

    def test_ten_players(self, raw_matches):
        for row in raw_matches[:100]:
            assert len(row["r"]) == 5, f"Match {row['m']}: {len(row['r'])} radiant players"
            assert len(row["e"]) == 5, f"Match {row['m']}: {len(row['e'])} dire players"

    def test_hero_ids_valid(self, raw_matches):
        for row in raw_matches[:100]:
            for hero, items in row["r"] + row["e"]:
                assert isinstance(hero, int)
                assert 1 <= hero <= 200

    def test_items_are_lists(self, raw_matches):
        for row in raw_matches[:100]:
            for hero, items in row["r"] + row["e"]:
                assert isinstance(items, list)
                assert all(isinstance(i, int) for i in items)

    def test_win_is_binary(self, raw_matches):
        for row in raw_matches[:100]:
            assert row["w"] in (0, 1)

    def test_duration_plausible(self, raw_matches):
        for row in raw_matches[:100]:
            assert 600 <= row["s"] <= 10000  # 10 min to ~2.7 hours

    def test_low_duplicate_rate(self, raw_matches):
        ids = [r["m"] for r in raw_matches]
        unique = len(set(ids))
        dup_rate = 1 - unique / len(ids)
        assert dup_rate < 0.05, f"Too many duplicates: {dup_rate:.1%}"


class TestHeroCoverage:
    def test_multiple_heroes(self, raw_matches):
        heroes = set()
        for row in raw_matches:
            for hero, _ in row["r"] + row["e"]:
                heroes.add(hero)
        assert len(heroes) >= 20, f"Only {len(heroes)} unique heroes"

    def test_win_rate_near_50(self, raw_matches):
        wins = sum(r["w"] for r in raw_matches)
        wr = wins / len(raw_matches)
        assert 0.35 < wr < 0.65, f"Win rate is {wr:.2%}, expected near 50%"

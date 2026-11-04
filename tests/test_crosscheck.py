"""Row-level agreement with Retrosheet's official parsed play-by-play (2023-2025)."""
import json

import pandas as pd

from conftest import PROCESSED, need


def test_state_matches_retrosheet_parsed_files():
    s = json.loads(need(PROCESSED / "crosscheck_retrosheet.json").read_text())
    for y, x in s["seasons"].items():
        assert x["games_row_count_equal"] == x["games_in_both"], y
        for f in ("inning", "half", "outs_pre", "outs_post", "bases_pre", "runner_ids_pre", "score", "runs",
                  "lineup_slot"):
            assert x["mismatch_rows"][f] == 0, (y, f)


def test_only_walkoff_end_state_differs():
    """Retrosheet blanks the runners after a game-ending play; every remaining
    base-state difference must be the last play of its game."""
    m = pd.read_parquet(need(PROCESSED / "crosscheck_mismatches.parquet"))
    last = m.groupby("gid")["k"].transform("max")
    b = m[m["bases_post"]]
    assert (b["k"] == last[b.index]).all()

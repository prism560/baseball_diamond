"""Section-0 validation helpers."""
import pandas as pd

from conftest import PROCESSED, need
from wp.retro_parse import check_pitches
from wp.validate import parse_linescore


def test_linescore_parse():
    assert parse_linescore("010(10)0000x") == [0, 1, 0, 10, 0, 0, 0, 0, None]


def test_pitch_rules():
    assert check_pitches("BBBB", "30", "BB") is None
    assert check_pitches("CSFFFS", "02", "K") is None          # fouls with 2 strikes keep the count
    assert check_pitches("CSL", "02", "K") is None             # foul bunt with 2 strikes = strikeout
    assert check_pitches("CSLX", "02", "OUT") == "pa_should_have_ended"
    assert check_pitches("BBB", "20", "BB") == "walk_without_ball_four"
    assert check_pitches("CS", "01", "K") == "strikeout_without_strike_three"
    assert check_pitches("BX", "00", "1B") == "count_field_mismatch"
    assert check_pitches("B*B1>X", "20", "1B") is None         # non-pitch markers ignored


def test_validation_rates_below_threshold():
    """Stop criterion agreed for B2: < 0.5% flagged games in every era."""
    import json
    s = json.loads(need(PROCESSED / "validation_summary.json").read_text())
    for row in s["by_era"]:
        assert row["rate"] < 0.005, row
    g = pd.read_parquet(PROCESSED / "games.parquet")
    assert not g["exclude_reason"].eq("score_mismatch_vs_gamelog").any()

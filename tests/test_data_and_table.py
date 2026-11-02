"""Integrity tests on the processed data and the B0 table (skip if not built)."""
import numpy as np
import pandas as pd

from conftest import MODELS, PROCESSED, need
from wp import b0_table as B0
from wp import states as S


def test_no_impossible_states_in_included_games():
    st = pd.read_parquet(need(PROCESSED / "states.parquet"))
    games = pd.read_parquet(PROCESSED / "games.parquet")
    st = st[st["game_id"].isin(games.loc[games["included"], "game_id"])]
    diff = st["home_score"] - st["vis_score"]
    assert st["outs"].between(0, 2).all()
    assert st["bases"].between(0, 7).all()
    bad = ~S.is_possible(st["inning"], st["half"], st["outs"], st["bases"], diff)
    assert int(bad.sum()) == 0, st[bad].head()
    # extra-inning tops start tied in every included game
    first_extra = st[(st["inning"] >= 10) & (st["half"] == 0)].groupby(["game_id", "inning"]).head(1)
    assert ((first_extra["home_score"] - first_extra["vis_score"]) == 0).all()


def test_impossible_cells_have_no_data():
    t = pd.read_parquet(need(MODELS / "wp_table_b0.parquet"))
    assert int(t.loc[~t["possible"], "n"].sum()) == 0


def test_monotonicity_report():
    """Report (not enforce) violations: WP decreasing as home lead grows."""
    t = pd.read_parquet(need(MODELS / "wp_table_b0.parquet"))
    v = B0.monotonic_violations(t)
    big = v[(v["prev_wp"] - v["wp"]) > 0.01]
    print(f"\nmonotonic violations: {len(v)} total, {len(big)} > 0.01, "
          f"{int((big['n'] >= 1000).sum())} > 0.01 with n>=1000")
    # well-populated cells must be monotone up to noise
    assert int(((v["prev_wp"] - v["wp"]) > 0.02)[v["n"] >= 1000].sum()) == 0
    mono = pd.read_parquet(need(MODELS / "wp_table_b0_monotone.parquet"))
    assert len(B0.monotonic_violations(mono, tol=1e-12)) == 0


def test_tied_states_stay_near_home_advantage():
    t = B0.WPTable.load(need(MODELS / "wp_table_b0.parquet"))
    tops = [t.wp(i, 0, 0, 0, 0) for i in range(1, 10)]
    print("\ntie, top of inning 1..9:", np.round(tops, 3))
    assert all(0.50 <= p <= 0.60 for p in tops)
    # bottom-of-inning ties: home bats last -> above 0.5, rising late but bounded
    bots = [t.wp(i, 1, 0, 0, 0) for i in range(1, 10)]
    print("tie, bottom of inning 1..9:", np.round(bots, 3))
    assert all(0.50 <= p <= 0.72 for p in bots)
    # extra innings under both rule eras
    for x in (0, 1):
        assert 0.45 <= t.wp(10, 0, 0, 0b010 if x else 0, 0, x) <= 0.60


def test_lookup_function():
    p = B0.lookup_wp(need(MODELS / "wp_table_b0.parquet"), 9, 1, 2, 0b111, -1)
    assert 0.0 < p < 1.0

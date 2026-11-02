"""Leakage tests for the Elo pre-game probability."""
import numpy as np
import pandas as pd

from conftest import PROCESSED, need
from wp import elo as E


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    teams = list("ABCDEFGH")
    rows = []
    for i in range(n):
        h, v = rng.choice(teams, 2, replace=False)
        hs, vs = rng.integers(0, 10, 2)
        if hs == vs:
            hs += 1
        rows.append({"game_id": f"G{i:05d}", "season": 2000 + i // 100,
                     "date": pd.Timestamp("2000-04-01") + pd.Timedelta(days=i), "number": "0",
                     "home": h, "visitor": v, "home_score": hs, "vis_score": vs, "forfeit": ""})
    return pd.DataFrame(rows)


def test_future_results_do_not_change_past_probabilities():
    g = synthetic()
    base = E.run_elo(g).set_index("game_id")["p_pregame"]
    cut = 250
    g2 = g.copy()
    # flip every result from game `cut` on
    g2.loc[cut:, ["home_score", "vis_score"]] = g2.loc[cut:, ["vis_score", "home_score"]].to_numpy()
    alt = E.run_elo(g2).set_index("game_id")["p_pregame"]
    ids = g["game_id"].iloc[: cut + 1]   # includes game `cut` itself: its own result is unused
    assert np.allclose(base[ids], alt[ids])
    assert not np.allclose(base[g["game_id"].iloc[cut + 1:]], alt[g["game_id"].iloc[cut + 1:]])


def test_shuffled_input_order_is_irrelevant():
    g = synthetic()
    a = E.run_elo(g).set_index("game_id")["p_pregame"]
    b = E.run_elo(g.sample(frac=1, random_state=1)).set_index("game_id")["p_pregame"]
    assert np.allclose(a, b.loc[a.index])


def test_real_gamelogs_truncation():
    """Elo on game logs truncated at 2023-01-01 gives identical pre-game
    probabilities for every earlier game as the full run."""
    gl = pd.read_parquet(need(PROCESSED / "gamelogs.parquet"))
    gl = gl[gl["season"] >= 2010]
    full = E.run_elo(gl).set_index("game_id")["p_pregame"]
    trunc = E.run_elo(gl[gl["date"] < "2023-01-01"]).set_index("game_id")["p_pregame"]
    assert np.allclose(full.loc[trunc.index], trunc)

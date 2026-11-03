"""B4 step 4: condition features use only pitches before the current PA."""
import numpy as np
import pandas as pd

from wp import condition as CD


def frame():
    return pd.DataFrame({
        "game_id": ["G1"] * 5 + ["G2"] * 2,
        "pitcher": ["p", "p", "q", "p", "p", "p", "p"],
        "pitches": ["BBX", "CSFB", "X", "BBBB", "CX", "SSS", "X"],
        "season": [2024] * 7,
        "date": pd.to_datetime(["2024-05-01"] * 5 + ["2024-05-02"] * 2)})


def test_counts_only_earlier_pitches_of_same_pitcher_and_game():
    F = CD.condition_features(frame())
    assert F["n_prev"].tolist() == [0, 3, 0, 7, 11, 0, 3]
    assert F["strikes_prev"].tolist() == [0, 1, 0, 4, 4, 0, 3]


def test_deleting_later_pitches_leaves_current_values():
    d = frame()
    F = CD.condition_features(d)
    t = d.copy()
    t.loc[4, "pitches"] = "SSSSSSSS"          # change a later PA of game G1
    t.loc[6, "pitches"] = "BBBB"              # and a later PA of game G2
    F2 = CD.condition_features(t)
    for i in (0, 1, 2, 3, 5):
        assert F.loc[i, "n_prev"] == F2.loc[i, "n_prev"] and F.loc[i, "strikes_prev"] == F2.loc[i, "strikes_prev"]
    # the baseline only uses earlier dates: game G1 rows are unaffected by G2
    assert np.allclose(F.loc[:4, "base"], F2.loc[:4, "base"])


def test_strong_prior_keeps_small_samples_near_zero():
    F = pd.DataFrame({"n_prev": [3.0, 90.0], "strikes_prev": [3.0, 90.0], "base": [0.64, 0.64]})
    dv = CD.cond_dev(F, 200.0)
    assert abs(dv[0]) < 0.01 < abs(dv[1])

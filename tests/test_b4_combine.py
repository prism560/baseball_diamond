"""Combination weights are fitted on seasons before the evaluation season only."""
import json

import numpy as np
import pandas as pd

from conftest import ROOT
from wp import run_b4_combine as C


def synthetic(seasons=(2022, 2023, 2024, 2025), n=3000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        p = rng.uniform(0.05, 0.95, n)
        rows.append(pd.DataFrame({"season": s, "game_id": [f"G{s}{i // 30}" for i in range(n)],
                                  "B3": p, "B2E": np.clip(p + rng.normal(0, .05, n), .01, .99),
                                  "Elo": rng.uniform(.4, .6, n), "progress": rng.uniform(0, 1, n),
                                  "home_win": (rng.uniform(size=n) < p).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def test_train_seasons_strictly_before():
    for S in C.EVAL:
        assert all(s < S for s in C.train_seasons(S, [2022, 2023, 2024, 2025]))


def test_future_seasons_do_not_change_fitted_weights():
    df = synthetic()
    _, info = C.apply(df, [2022, 2023, 2024, 2025], "pa")
    df2 = df.copy()
    m = df2["season"] == 2025
    df2.loc[m, "home_win"] = 1 - df2.loc[m, "home_win"]         # scramble the future
    _, info2 = C.apply(df2, [2022, 2023, 2024, 2025], "pa")
    for S in ("2023", "2024", "2025"):
        assert info[S]["avg_weight_on_B3"] == info2[S]["avg_weight_on_B3"]
        assert info[S]["b3e_coef"] == info2[S]["b3e_coef"]


def test_recorded_training_seasons_precede_evaluation():
    p = ROOT / "reports" / "b4_combine.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    for lvl in ("fit_pa", "fit_pitch"):
        for S, x in r[lvl].items():
            assert max(x["train_seasons"]) < int(S)

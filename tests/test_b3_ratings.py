"""B3 ratings: leakage, log5 identity, distributions."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import PROCESSED, ROOT, need
from wp import matchup as MU
from wp import ratings as RT
from wp import transitions as T


@pytest.fixture(scope="module")
def frame():
    pa = pd.read_parquet(need(PROCESSED / "pa.parquet"))
    games = pd.read_parquet(PROCESSED / "games.parquet")
    pa = pa[pa["season"].between(2020, 2024)]
    return RT.pa_frame(pa, games, RT.load_hands(ROOT / "data" / "raw", 2018, 2024))


def test_log5_identity_and_sums():
    rng = np.random.default_rng(0)
    q = rng.dirichlet(np.ones(T.NO), size=(2, 3, 8))
    assert np.abs(MU.combine(q, np.zeros(RT.NR)) - q).max() < 1e-12      # league vs league = league
    dev = rng.normal(0, 0.5, size=(2, 3, 8, RT.NR))
    m = MU.combine(q, dev)
    assert np.abs(m.sum(-1) - 1).max() < 1e-12
    assert np.allclose(m[..., T.O_IDX["INT"]], q[..., T.O_IDX["INT"]])   # INT stays league-level


def test_no_future_leakage(frame):
    """Deleting every PA from the game's date onward (other than the game itself)
    leaves that game's ratings unchanged."""
    d = frame
    g = d[d["season"] == 2024]["game_id"].drop_duplicates().iloc[1500]
    day = d.loc[d["game_id"] == g, "date"].iloc[0]
    trunc = d[(d["date"] < day) | (d["game_id"] == g)].reset_index(drop=True)
    F_full = RT.build_features(d)
    F_tr = RT.build_features(trunc)
    p = RT.RatingParams(1, 2, 6)
    a = RT.deviations(F_full, p)[(d["game_id"] == g).to_numpy()]
    b = RT.deviations(F_tr, p)[(trunc["game_id"] == g).to_numpy()]
    assert a.shape == b.shape and len(a) > 50
    assert np.abs(a - b).max() < 1e-9
    assert np.abs(F_full["L"][(d["game_id"] == g).to_numpy()] - F_tr["L"][(trunc["game_id"] == g).to_numpy()]).max() < 1e-12


def test_rates_are_distributions(frame):
    F = RT.build_features(frame.head(200000))
    r = RT.shrink(F["Cb"], F["Nb"], F["L"], 1.0)
    assert np.abs(r.sum(1) - 1).max() < 1e-12 and (r > 0).all()
    assert np.abs(F["L"].sum(1) - 1).max() < 1e-12

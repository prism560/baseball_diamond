"""B4 1.1 park-neutral ratings and 1.2 multinomial logistic matchup."""
import numpy as np
import pandas as pd
import pytest

from conftest import PROCESSED, ROOT, need
from wp import matchup as MU
from wp import matchup_v2 as M2
from wp import ratings as RT
from wp import transitions as T


@pytest.fixture(scope="module")
def frame():
    pa = pd.read_parquet(need(PROCESSED / "pa.parquet"))
    g = pd.read_parquet(PROCESSED / "games.parquet")
    pa = pa[pa["season"].between(2022, 2024)]
    return RT.pa_frame(pa, g, RT.load_hands(ROOT / "data" / "raw", 2020, 2024))


def test_park_neutral_equals_standard_when_all_parks_are_neutral(frame):
    d = frame.head(150000)
    C0, N0 = RT.asof_counts(d, ["batter"])
    C1, N1 = RT.asof_counts(d, ["batter"], np.ones((len(d), RT.NR)))
    assert np.abs(N0 - N1).max() == 0
    assert np.abs(C0 - C1).max() < 1e-6


def test_park_neutral_deflates_hitter_park_outcomes(frame):
    d = frame.head(150000)
    w = np.ones((len(d), RT.NR))
    w[:, RT.R_IDX["HR"]] = 0.5                       # every park doubles home runs
    C0, N0 = RT.asof_counts(d, ["batter"])
    C1, N1 = RT.asof_counts(d, ["batter"], w)
    hr = RT.R_IDX["HR"]
    assert (C1[:, hr] <= C0[:, hr] + 1e-6).all()     # never inflated (a batter with only HRs stays put)
    assert C1[:, hr].sum() < 0.8 * C0[:, hr].sum()
    assert np.abs(C1.sum(1) - N1).max() < 1e-3      # PA count preserved


def test_logit_log5_identity_and_reduction():
    rng = np.random.default_rng(0)
    q = rng.dirichlet(np.ones(T.NO), size=500)
    X = rng.normal(0, 0.3, (500, M2.NF, RT.NR))
    X[:, :, RT.REF] = 0
    m = M2.LogitMatchup.log5()
    assert np.abs(m.predict(q, X) - MU.combine(q, X.sum(1))).max() < 1e-12      # log5 special case
    assert np.abs(m.predict(q, np.zeros_like(X)) - q).max() < 1e-12             # all effects 0 -> league q
    assert np.abs(m.predict(q, X).sum(1) - 1).max() < 1e-12


def test_logit_fit_recovers_coefficients():
    rng = np.random.default_rng(1)
    n = 20000
    q = np.tile(np.r_[np.full(RT.NR, 0.95 / RT.NR), 0.05], (n, 1))[:, :T.NO]
    q = q / q.sum(1, keepdims=True)
    X = rng.normal(0, 0.4, (n, M2.NF, RT.NR))
    X[:, :, RT.REF] = 0
    true = M2.LogitMatchup.log5()
    true.beta[M2.NONREF, 0] = 0.6                     # batter effects shrunk
    p = true.predict(q, X)[:, MU.RC]
    p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(RT.NR, p=pi) for pi in p])
    fit = M2.fit(q, X, y, lam=1e-6)
    assert np.abs(fit.beta[M2.NONREF, 0] - 0.6).mean() < 0.1

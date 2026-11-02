"""Count chain / IPF / consistency with the DP."""
import numpy as np
import pandas as pd
import pytest

from conftest import PROCESSED, need
from wp import count_chain as CC
from wp import dp
from wp import transitions as T


@pytest.fixture(scope="module")
def setup():
    pa = pd.read_parquet(need(PROCESSED / "pa.parquet"))
    seas = [2022, 2023, 2024]
    tr = T.estimate(pa, seas)
    base = CC.estimate_base(pa, seas)
    return tr, base, CC.fit(base, tr.q)


def test_chain_probabilities_sum_to_one(setup):
    tr, base, ch = setup
    assert np.allclose(base.ev.sum(1), 1) and np.allclose(base.split.sum(1), 1)
    term, trans = CC._weights(base, ch.f)
    assert np.allclose(term.sum(-1) + trans.sum(-1), 1.0)
    assert np.allclose(ch.P.sum(-1), 1.0, atol=1e-12)


def test_ipf_matches_q(setup):
    tr, base, ch = setup
    assert np.abs(ch.P[..., 0, :] - tr.q).max() <= 1e-6


@pytest.mark.parametrize("use_N", [True, False])
def test_pitch_wp_at_0_0_equals_dp(setup, use_N):
    tr, base, ch = setup
    trn = tr if use_N else tr.without_N()
    res = dp.Engine(30, 1).solve(dp.kernels_from(trn))
    G = dp.after_outcome_values(res, trn)
    pw = CC.pitch_wp_table(ch, G)
    v = slice(res.pad, res.pad + 2 * res.R + 1)              # stored diff range (padding = decided)
    for k, g in G.items():
        W = np.einsum("xyO,xydO->xyd", trn.q[k[1]], g)
        assert np.abs(pw[k][..., 0] - W)[..., v].max() < 1e-9          # 0-0 pitch WP == first-pitch DP value
        if not use_N:                                                  # no pre-pitch events: == PA-start value
            assert np.abs(pw[k][..., 0] - res.V[k][..., 0, 0])[..., v].max() < 1e-9


def test_count_direction(setup):
    tr, base, ch = setup
    # batting team's outcome mix improves with balls and worsens with strikes
    P = ch.P[0, 0, 0]                                        # visitor batting, 0 out, empty
    good = [T.O_IDX[o] for o in ("BB", "1B", "2B", "3B", "HR")]
    c = {x: i for i, x in enumerate(CC.COUNTS)}
    assert P[c[(3, 0)], good].sum() > P[c[(0, 0)], good].sum() > P[c[(0, 2)], good].sum()

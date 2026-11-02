"""B3 engine: reduction to B2, symmetry, DP/count-chain consistency per matchup."""
import numpy as np
import pandas as pd
import pytest

from conftest import PROCESSED, need
from wp import b3_game as BG
from wp import count_chain as CC
from wp import dp
from wp import dp_b3 as D3
from wp import pitching_policy as PP
from wp import transitions as T
from wp.ratings import NR


@pytest.fixture(scope="module")
def base():
    pa = pd.read_parquet(need(PROCESSED / "pa.parquet"))
    g = pd.read_parquet(PROCESSED / "games.parquet")
    pa = pa[pa["game_id"].isin(g.loc[g["included"], "game_id"])]
    tr = T.estimate(pa, [2022, 2023, 2024])
    ann = PP.annotate(pa[pa["season"] == 2024], g)
    return tr, PP.Hazard.fit(ann, [2024]), pa


def team(rng=None, scale=0.0):
    r = (lambda *s: rng.normal(0, scale, s)) if rng is not None else (lambda *s: np.zeros(s))
    return D3.TeamInputs(r(9, 2, NR), r(NR), 1, r(3, 2, NR), np.full((3, 2), 0.5))


def test_reduction_to_b2(base):
    tr, H, _ = base
    gi = D3.GameInputs(team(), team(), tr.q, tr.N, tr.A, np.zeros(NR), H,
                       np.full((5, 5, 3), 1 / 3), np.zeros((3, NR)))
    r3 = D3.Engine(20).solve(gi)
    r2 = dp.Engine(20, 1).solve(dp.kernels_from(tr))
    for k in r2.V:
        assert np.abs(r3.V[k] - r2.V[k][:, :, :, 0, 0][..., None, None]).max() < 1e-9


def test_symmetry_identical_teams(base):
    """Identical teams, neutral park, no home effect and a score-independent pitching
    policy -> exactly 0.5.  (With the score-dependent hazard/tier mixture the game is
    not symmetric: the home side's opponents choose pitchers knowing the top-half
    result - the report measures that deviation.)"""
    tr, H, _ = base
    rng = np.random.default_rng(1)
    t = team(rng, 0.3)
    sym = T.Transitions(np.repeat(tr.N[:1], 2, 0), np.repeat(tr.q[:1], 2, 0), tr.A, tr.D)
    flat = np.full((5, 5, 3), 1 / 3)
    gi = D3.GameInputs(t, t, sym.q, sym.N, sym.A, np.zeros(NR), None, flat, np.zeros((3, NR)))
    r = D3.Engine(40).solve(gi)
    assert abs(r.wp(1, 0, 0, 0, 0, 0, 0) - 0.5) < 1e-9


def test_pitch_wp_at_0_0_equals_dp_for_a_matchup(base):
    """No pre-pitch events and no hazard: the DP value at a PA start equals the
    count-chain WP at 0-0 for that matchup (IPF per matchup, error <= 1e-6)."""
    tr, H, pa = base
    trn = tr.without_N()
    rng = np.random.default_rng(2)
    gi = D3.GameInputs(team(rng, 0.3), team(rng, 0.3), trn.q, trn.N, trn.A, np.zeros(NR), None,
                       np.full((5, 5, 3), 1 / 3), np.zeros((3, NR)))
    res = D3.Engine(20).solve(gi)
    cbase = CC.estimate_base(pa, [2023, 2024])
    for inn, half, o, b, d, slot in [(1, 0, 0, 0, 0, 0), (5, 1, 1, 3, -1, 4), (8, 0, 2, 7, 2, 8), (9, 1, 0, 2, -1, 2)]:
        stage = 0                                             # facing the starter on lap 1
        bat = gi.vis if half == 0 else gi.home
        fld = gi.home if half == 0 else gi.vis
        q = D3.q_for(gi, half, bat.bat_dev[slot, fld.starter_hand], fld.starter_dev)[o, b]
        ch = CC.fit(cbase, q[None])
        assert np.abs(ch.P[0, 0] - q).max() <= 1e-6
        c_oth = D3.composite(3, 0)
        G = BG.after_outcome_G(gi, res, inn, half, o, b, d, slot, stage, c_oth)
        pitch00 = ch.P[0, 0] @ G
        v = res.wp(inn, half, o, b, d, D3.composite(slot, stage), c_oth)
        assert abs(pitch00 - v) < 1e-9

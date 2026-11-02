"""Pitching-change policy: hazard sanity, tiers, mixture weights."""
import itertools

import numpy as np
import pandas as pd
import pytest

from conftest import PROCESSED, need
from wp import pitching_policy as PP


@pytest.fixture(scope="module")
def ann():
    pa = pd.read_parquet(need(PROCESSED / "pa.parquet"))
    g = pd.read_parquet(PROCESSED / "games.parquet")
    return PP.annotate(pa[pa["season"].between(2021, 2022)], g)


def test_hazard_is_a_probability_everywhere(ann):
    H = PP.Hazard.fit(ann, [2021])
    combos = np.array(list(itertools.product(range(1, 10), (0, 1), (1, 2, 3), range(-6, 7), (0, 1, 2))))
    p = H.prob(*combos.T)
    assert ((p > 0) & (p < 1)).all()
    # a starter is eventually removed: probability of surviving 9 innings x ~4.5 PAs is small
    late = H.prob(np.full(5, 9), np.zeros(5, int), np.full(5, 3), np.zeros(5, int), np.zeros(5, int))
    assert late.min() > 0.01


def test_tiers_and_mixture(ann):
    tiers = PP.bullpen_tiers(ann, [2022])
    assert set(tiers["tier"]) == {0, 1, 2}
    w = PP.tier_mixture(ann, tiers, [2022])
    assert np.abs(w.sum(-1) - 1).max() < 1e-12
    # high-leverage relievers are used more in close late games than early
    assert w[4, 3, 0] > w[0, 3, 0]


def test_tiers_use_only_past_appearances(ann):
    tiers = PP.bullpen_tiers(ann, [2022])
    first = tiers[tiers["date"] == tiers["date"].min()]
    # on the first game date of 2022 only previous-season appearances (weight 0.5 each) exist
    assert np.allclose((first["weight"] * 2) % 1, 0)

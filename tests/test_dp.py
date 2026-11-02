"""B2 DP engine tests (use the processed PA table; skip if not built)."""
import numpy as np
import pandas as pd
import pytest

from conftest import PROCESSED, need
from wp import dp
from wp import transitions as T


@pytest.fixture(scope="module")
def tr():
    pa = pd.read_parquet(need(PROCESSED / "pa.parquet"))
    return T.estimate(pa, [2022, 2023, 2024])


@pytest.fixture(scope="module")
def res(tr):
    return dp.Engine(30, 1, xrule=True).solve(dp.kernels_from(tr))


def test_distributions_sum_to_one(tr):
    for arr, axes in ((tr.N, (-3, -2, -1)), (tr.A, (-3, -2, -1)), (tr.D, (-3, -2, -1)), (tr.q, (-1,))):
        assert np.abs(arr.sum(axis=axes) - 1.0).max() < 1e-12
        assert arr[..., 3, 1:, :].sum() == 0 if arr.ndim >= 5 else True
    for h in (0, 1):
        K = dp.compose_kernel(tr.N[h], tr.q[h][None], tr.A)
        assert np.abs(K.sum(axis=(3, 4, 5, 6)) - 1.0).max() < 1e-12


def test_termination_rules(res, tr):
    d = np.arange(res.V[(9, 1)].shape[2]) - res.R - res.pad
    V9b = res.V[(9, 1)][..., 0, 0]
    assert np.all(V9b[:, :, d > 0] == 1.0)                  # bottom 9+, home ahead: game over
    assert np.all(res.V[(10, 1)][..., 0, 0][:, :, d > 0] == 1.0)
    # bottom of the 9th IS played when the visitor leads: values strictly inside (0, 1)
    v = V9b[0, 0, (d < 0) & (d >= -3)]
    assert np.all((v > 0) & (v < 1))
    # walk-off: a home run in the bottom of the 9th with the score tied -> 1
    G = dp.after_outcome_values(res, tr)
    assert G[(9, 1)][0, 0, res.R + res.pad, T.O_IDX["HR"]] == pytest.approx(1.0)
    # top of the 9th (or later) ends with the home team ahead -> no bottom half (value 1)
    for inn in (9, 10):
        e = dp._end_of_half(res, inn, 0, d)
        assert np.all(e[d > 0] == 1.0)
        assert np.all((e[(d <= 0) & (d >= -3)] > 0) & (e[(d <= 0) & (d >= -3)] < 1))


def test_extra_inning_fixed_point_matches_unrolled(tr):
    eng = dp.Engine(30, 1, xrule=True)
    res = eng.solve(dp.kernels_from(tr))
    K = dp.kernels_from(tr)
    d = eng.dvals
    # unroll 50 extra innings from an arbitrary continuation (0.5 for a tie)
    Eb = np.zeros((eng.Dp, 1, 1, 1)); Eb[d > 0] = 1.0; Eb[d == 0] = 0.5
    for _ in range(50):
        Vb = eng.solve_half(K(10, 1), Eb, 1, walkoff=True)
        Et = np.swapaxes(Vb[0, 0b010], 1, 2).copy(); Et[d > 0] = 1.0
        Vt = eng.solve_half(K(10, 0), Et, 0, walkoff=False)
        Eb = np.zeros((eng.Dp, 1, 1, 1)); Eb[d > 0] = 1.0
        Eb[d == 0] = np.swapaxes(Vt[0, 0b010], 1, 2)[d == 0]
    unrolled = float(Vt[0, 0b010, d == 0][0, 0, 0, 0])
    assert abs(unrolled - res.X[0, 0]) < 1e-9


def test_diff_range_convergence(tr):
    a = dp.Engine(30, 1).solve(dp.kernels_from(tr))
    b = dp.Engine(40, 1).solve(dp.kernels_from(tr))
    mx = max(np.abs(a.V[k][:, :, a.pad:a.pad + 61] - b.V[k][:, :, b.pad + 10:b.pad + 71]).max() for k in a.V)
    assert mx < 1e-6


def test_lineup_axis_equals_collapsed(tr, res):
    r9 = dp.Engine(30, 9).solve(dp.kernels_from(tr, S=9))
    for k in res.V:
        assert np.abs(r9.V[k] - res.V[k][..., :1, :1]).max() < 1e-12


def test_symmetry_identical_teams_give_half(tr):
    sym = T.Transitions(np.repeat(tr.N[:1], 2, 0), np.repeat(tr.q[:1], 2, 0), tr.A, tr.D)
    for xr in (True, False):
        # wide diff range so that the +-R truncation error is negligible
        r = dp.Engine(60, 1, xrule=xr).solve(dp.kernels_from(sym))
        assert abs(r.wp(1, 0, 0, 0, 0) - 0.5) < 1e-9
        assert abs(r.X[0, 0] - 0.5) < 1e-9

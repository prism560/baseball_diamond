"""State encoding round-trip tests."""
import itertools

import numpy as np

from wp import states as S


def test_roundtrip_all_canonical_states():
    n = 0
    for inn, half, outs, bases, diff in itertools.product(range(1, 11), (0, 1), range(3), range(8),
                                                          range(-10, 11)):
        for x in ((0, 1) if inn == 10 else (0,)):
            st = S.State(inn, half, outs, bases, diff, x)
            assert S.decode_state(S.encode_state(st)) == st
            n += 1
    assert n == 10 * 2 * 3 * 8 * 21 + 2 * 3 * 8 * 21


def test_keys_unique_and_in_range():
    rng = np.random.default_rng(0)
    inn = rng.integers(1, 11, 5000); h = rng.integers(0, 2, 5000); o = rng.integers(0, 3, 5000)
    b = rng.integers(0, 8, 5000); d = rng.integers(-10, 11, 5000); x = rng.integers(0, 2, 5000)
    k = S.encode(inn, h, o, b, d, x)
    assert k.min() >= 0 and k.max() < S.N_STATES
    i2, h2, o2, b2, d2, x2 = S.decode(k)
    assert (i2 == inn).all() and (h2 == h).all() and (o2 == o).all() and (b2 == b).all()
    assert (d2 == d).all() and (x2 == np.where(inn == 10, x, 0)).all()


def test_bucketing_and_clipping():
    # extra innings pooled into 10, diff clipped at +-10
    assert S.encode(14, 1, 2, 7, 25, 1) == S.encode(10, 1, 2, 7, 10, 1)
    assert S.decode_state(S.encode(12, 0, 0, 0, -13, 0)) == S.State(10, 0, 0, 0, -10, 0)
    # xrule ignored in regulation innings
    assert S.encode(5, 0, 1, 3, 2, 1) == S.encode(5, 0, 1, 3, 2, 0)


def test_progress():
    assert S.progress(1, 0, 0) == 0.0
    assert abs(S.progress(9, 1, 2) - 53 / 54) < 1e-12
    assert S.progress(11, 0, 0) <= 1.0

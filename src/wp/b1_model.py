"""B1: logistic blend of the B0 state table with Elo pre-game strength.

    logit(WP_B1) = a + b*logit(WP_B0) + c*logit(P_pre) + d*logit(P_pre)*progress

progress = 1 - outs_remaining/54 (0 at the first pitch, 1 at the end of
regulation; extra innings stay at 1), so the Elo term's weight is c at the
start and c+d at the end - d < 0 expresses fading pre-game strength.

Features must be out-of-sample: callers supply WP_B0 from a table fitted on
seasons strictly before the row's season (see run_pipeline).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-4


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def features(wp_b0, p_pre, progress) -> np.ndarray:
    lb0, lpre = logit(wp_b0), logit(p_pre)
    return np.column_stack([lb0, lpre, lpre * np.asarray(progress, float)])


@dataclass
class B1Model:
    a: float
    b: float
    c: float
    d: float

    @classmethod
    def fit(cls, wp_b0, p_pre, progress, y) -> "B1Model":
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(features(wp_b0, p_pre, progress), np.asarray(y))
        b, c, d = lr.coef_[0]
        return cls(float(lr.intercept_[0]), float(b), float(c), float(d))

    def predict(self, wp_b0, p_pre, progress):
        z = self.a + features(wp_b0, p_pre, progress) @ np.array([self.b, self.c, self.d])
        return 1.0 / (1.0 + np.exp(-z))

    def as_dict(self):
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}

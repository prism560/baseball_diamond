"""B4 1.2: multinomial logistic matchup model with the league q as offset.

For the 10 matchup categories (INT stays league-level, as in wp.matchup):

    z_o = log q_league(o|state) - log q_league(OUT|state)
          + gamma_o + sum_j beta_{o,j} x_{j,o}          (z_OUT = 0)

with per-category features x_j (log-odds deviations vs OUT, all as of the day
before the game): batter, pitcher, park, batter platoon split, season
environment.  The generalised log5 of B3 is the special case beta = 1, gamma = 0;
the L2 penalty pulls the coefficients toward that log5 solution, so with little
data the model falls back to B3.  Coefficients and the penalty strength are fitted
only on seasons before the evaluation season.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .ratings import NR, REF
from .transitions import NO, O_IDX
from .matchup import RC

FEATURES = ["batter", "pitcher", "park", "platoon", "env"]
NF = len(FEATURES)
INT = O_IDX["INT"]
NONREF = np.array([i for i in range(NR) if i != REF])


@dataclass
class LogitMatchup:
    beta: np.ndarray       # [NR, NF]  (row REF unused)
    gamma: np.ndarray      # [NR]
    lam: float

    @classmethod
    def log5(cls) -> "LogitMatchup":
        b = np.ones((NR, NF))
        b[REF] = 0
        return cls(b, np.zeros(NR), 0.0)

    def z(self, q_state10: np.ndarray, X: np.ndarray) -> np.ndarray:
        """q_state10 [n, NR] league probabilities of the 10 categories (any scale);
        X [n, NF, NR] features.  Returns logits [n, NR] (REF column = 0)."""
        with np.errstate(divide="ignore"):
            off = np.log(q_state10) - np.log(q_state10[:, [REF]])
        z = off + self.gamma[None] + np.einsum("nfo,of->no", X, self.beta)
        z[:, REF] = 0.0
        return z

    def predict(self, q_state: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Full 11-outcome distribution (INT kept at its league value)."""
        z = self.z(q_state[:, RC], X)
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        out = np.zeros((len(q_state), NO))
        out[:, RC] = p * (1 - q_state[:, INT])[:, None]
        out[:, INT] = q_state[:, INT]
        return out


def fit(q_state: np.ndarray, X: np.ndarray, y_r: np.ndarray, lam: float) -> LogitMatchup:
    """y_r: outcome index in the 10 categories (rows with INT must be removed)."""
    off = np.log(q_state[:, RC]) - np.log(q_state[:, RC][:, [REF]])
    n = len(y_r)
    Y = np.zeros((n, NR))
    Y[np.arange(n), y_r] = 1.0
    nb = len(NONREF) * NF

    def unpack(th):
        beta = np.zeros((NR, NF))
        beta[NONREF] = th[:nb].reshape(len(NONREF), NF)
        gamma = np.zeros(NR)
        gamma[NONREF] = th[nb:]
        return beta, gamma

    prior = np.concatenate([np.ones(nb), np.zeros(len(NONREF))])

    def f(th):
        beta, gamma = unpack(th)
        z = off + gamma[None] + np.einsum("nfo,of->no", X, beta)
        z[:, REF] = 0.0
        z -= z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        loss = -(z[np.arange(n), y_r] - lse).mean() + lam * ((th - prior) ** 2).sum()
        P = np.exp(z - lse[:, None])
        G = (P - Y) / n                                    # dL/dz  [n, NR]
        gb = np.einsum("no,nfo->of", G, X)[NONREF].ravel()
        gg = G.sum(0)[NONREF]
        return loss, np.concatenate([gb, gg]) + 2 * lam * (th - prior)

    res = minimize(f, prior.copy(), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    beta, gamma = unpack(res.x)
    return LogitMatchup(beta, gamma, lam)


def multiclass_logloss(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(o)), o], 1e-12, 1))

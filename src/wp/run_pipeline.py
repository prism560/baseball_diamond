"""End-to-end walk-forward evaluation of B0 and B1.

For each evaluation season S (default 2023, 2024, 2025):
  1. B0 pseudo-count k tuned: table fit on seasons < S-1, scored on S-1.
  2. B0 table fit on all included seasons < S  (extra-inning cells keyed by
     rule era, so for S >= 2023 the 10+ cells with xrule=1 use 2021..S-1 only).
  3. B1 fit on the last ``b1_seasons`` included seasons T < S, where each
     row's WP_B0 comes from a table fit on seasons < T (out-of-sample) and
     P_pre from the sequential Elo (pre-game, leak-free).
  4. Score B0 and B1 on S; block bootstrap by game.

Run:  PYTHONPATH=src python -m wp.run_pipeline
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import b0_table as B0
from . import elo as E
from . import evaluate as EV
from . import states as S
from .b1_model import B1Model

MODELS = ["B0", "B1", "Elo_only", "B0_mono", "B0_recent"]


def load(processed: Path):
    games = pd.read_parquet(processed / "games.parquet")
    gl = pd.read_parquet(processed / "gamelogs.parquet")
    st = pd.read_parquet(processed / "states.parquet")
    inc = games.loc[games["included"], ["game_id", "season", "home_win", "tiebreaker_base"]]
    st = st.merge(inc, on="game_id", how="inner")
    st["xrule"] = (st["tiebreaker_base"] == 2).astype(np.int8)
    st = S.add_state_columns(st)
    return games, gl, st


def table_before(counts: pd.DataFrame, season: int, k: float) -> pd.DataFrame:
    return B0.fit_table(counts[counts["season"] < season], k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--eval", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--b1-seasons", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--recent-window", type=int, default=30,
                    help="seasons used by the B0_recent sensitivity variant")
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    a = ap.parse_args()
    timings = {}
    t0 = time.time()
    games, gl, st = load(a.processed)
    timings["load_s"] = round(time.time() - t0, 1)
    a.reports.mkdir(exist_ok=True)
    (a.reports / "figures").mkdir(exist_ok=True)
    a.models.mkdir(exist_ok=True)
    out = {"n_states_included": int(len(st)), "n_games_included": int(games["included"].sum())}

    # ---- Elo ---------------------------------------------------------------
    t = time.time()
    first_eval = min(a.eval)
    tune_seasons = list(range(first_eval - 30, first_eval))  # strictly before every eval season
    elo_params, elo_grid = E.tune(gl, tune_seasons)
    elo = E.run_elo(gl, elo_params)
    timings["elo_tune_and_run_s"] = round(time.time() - t, 1)
    out["elo_params"] = elo_params.__dict__
    out["elo_tune_seasons"] = [tune_seasons[0], tune_seasons[-1]]
    out["elo_grid_best5"] = sorted(((v, str(k)) for k, v in elo_grid.items()))[:5]
    st = st.merge(elo[["game_id", "p_pregame"]], on="game_id", how="left")
    miss = int(st["p_pregame"].isna().sum())
    out["states_missing_elo"] = miss
    st = st[st["p_pregame"].notna()].reset_index(drop=True)

    # ---- B0 counts ---------------------------------------------------------
    t = time.time()
    counts = B0.season_counts(st)
    timings["b0_counts_s"] = round(time.time() - t, 1)
    seasons_avail = sorted(st["season"].unique())

    per_season, cal_tables, all_eval = {}, {}, []
    for Sx in a.eval:
        rec = {}
        t = time.time()
        prev = max(s for s in seasons_avail if s < Sx)
        val = st.loc[st["season"] == prev, ["key", "home_win"]]
        k, kgrid = B0.tune_k(counts[counts["season"] < prev], val)
        rec["b0_k"] = k
        rec["b0_k_grid"] = {str(kk): v for kk, v in kgrid.items()}
        tab = table_before(counts, Sx, k)
        tab.to_parquet(a.models / f"wp_table_b0_train_lt{Sx}.parquet", index=False)
        wpt = B0.WPTable(tab)
        rec["b0_train_seasons"] = [int(min(seasons_avail)), int(prev)]
        rec["b0_fit_s"] = round(time.time() - t, 1)

        # B1 training rows: last N seasons before Sx, each with out-of-sample B0
        t = time.time()
        b1_seasons = [s for s in seasons_avail if s < Sx][-a.b1_seasons:]
        parts = []
        for T in b1_seasons:
            wt = B0.WPTable(table_before(counts, T, k))
            d = st.loc[st["season"] == T, ["key", "p_pregame", "progress", "home_win"]].copy()
            d["wp_b0"] = wt.wp_key(d["key"].to_numpy())
            parts.append(d)
        tr = pd.concat(parts)
        b1 = B1Model.fit(tr["wp_b0"], tr["p_pregame"], tr["progress"], tr["home_win"])
        rec["b1_train_seasons"] = [int(x) for x in b1_seasons]
        rec["b1_coef"] = b1.as_dict()
        rec["b1_fit_s"] = round(time.time() - t, 1)

        ev = st[st["season"] == Sx].copy()
        ev["B0"] = wpt.wp_key(ev["key"].to_numpy())
        ev["B1"] = b1.predict(ev["B0"], ev["p_pregame"], ev["progress"])
        ev["Elo_only"] = ev["p_pregame"]
        # sensitivity variants (not the headline baselines)
        ev["B0_mono"] = B0.WPTable(B0.monotone_project(tab)).wp_key(ev["key"].to_numpy())
        recent = counts[(counts["season"] < Sx) & (counts["season"] >= Sx - a.recent_window)]
        ev["B0_recent"] = B0.WPTable(B0.fit_table(recent, k)).wp_key(ev["key"].to_numpy())
        rec["n_states"] = int(len(ev))
        rec["n_games"] = int(ev["game_id"].nunique())
        rec["xrule_cells_used"] = int(((ev["xrule"] == 1) & (ev["inning"] >= 10)).sum())

        # pre-game check: first state of each game (top 1st, 0 out, empty, 0-0)
        first = ev[(ev["seq"] == 0)]
        rec["pregame_check"] = {
            "n": int(len(first)),
            "mean_abs_diff_B1_vs_Elo": float(np.mean(np.abs(first["B1"] - first["p_pregame"]))),
            "max_abs_diff_B1_vs_Elo": float(np.max(np.abs(first["B1"] - first["p_pregame"]))),
            "corr": float(np.corrcoef(first["B1"], first["p_pregame"])[0, 1]),
            "mean_B1": float(first["B1"].mean()), "mean_Elo": float(first["p_pregame"].mean()),
            "B0_at_start": float(first["B0"].iloc[0]),
        }
        t = time.time()
        boot = EV.block_bootstrap(ev, MODELS, n_boot=a.n_boot, seed=Sx)
        rec["bootstrap_s"] = round(time.time() - t, 1)
        boot.insert(0, "season", Sx)
        rec["metrics"] = boot.to_dict(orient="records")
        cal_tables[Sx] = {m: EV.calibration(ev[m], ev["home_win"]) for m in ("B0", "B1")}
        per_season[Sx] = rec
        all_eval.append(ev[["game_id", "season", "inning", "diff", "home_win"] + MODELS])
        print(Sx, "k", k, "b1", rec["b1_coef"], flush=True)

    pooled = pd.concat(all_eval, ignore_index=True)
    boot = EV.block_bootstrap(pooled, MODELS, n_boot=a.n_boot, seed=1)
    boot.insert(0, "season", "pooled")
    out["pooled_metrics"] = boot.to_dict(orient="records")
    out["per_season"] = {str(k): v for k, v in per_season.items()}
    out["calibration"] = {str(s): {m: c.reset_index().rename(columns={"index": "bin"}).to_dict(orient="records")
                                   for m, c in d.items()} for s, d in cal_tables.items()}

    # calibration figure
    fig, axes = plt.subplots(1, len(a.eval), figsize=(4.2 * len(a.eval), 4.2), sharey=True)
    for ax, Sx in zip(np.atleast_1d(axes), a.eval):
        ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
        for m, col in (("B0", "#4C78A8"), ("B1", "#E45756")):
            c = cal_tables[Sx][m].dropna()
            ax.plot(c["mean_pred"], c["obs_rate"], marker="o", color=col, label=m)
        ax.set_title(f"Season {Sx}")
        ax.set_xlabel("predicted P(home win)")
        ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("observed home win rate")
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    fig.savefig(a.reports / "figures" / "calibration_b0_b1.png", dpi=120)

    # ---- final production artefacts (all seasons through the last one) -----
    t = time.time()
    last = max(seasons_avail)
    k_final = per_season[max(a.eval)]["b0_k"]
    final = B0.fit_table(counts, k_final)
    final.to_parquet(a.models / "wp_table_b0.parquet", index=False)
    B0.monotone_project(final).to_parquet(a.models / "wp_table_b0_monotone.parquet", index=False)
    timings["final_table_fit_s"] = round(time.time() - t, 1)
    viol = B0.monotonic_violations(final)
    viol_sig = viol[(viol["prev_wp"] - viol["wp"]) > 0.01]
    out["final_table"] = {
        "train_seasons": [int(min(seasons_avail)), int(last)], "k": k_final,
        "rows": int(len(final)), "possible_rows": int(final["possible"].sum()),
        "cells_with_data": int((final["n"] > 0).sum()),
        "cells_n_lt_30": int(((final["n"] > 0) & (final["n"] < 30) & final["possible"]).sum()),
        "possible_cells_without_data": int(((final["n"] == 0) & final["possible"]).sum()),
        "file_bytes": (a.models / "wp_table_b0.parquet").stat().st_size,
        "monotonic_violations": int(len(viol)),
        "monotonic_violations_by_window": {
            str(Sx): int(len(B0.monotonic_violations(pd.read_parquet(a.models / f"wp_table_b0_train_lt{Sx}.parquet"))))
            for Sx in a.eval},
        "monotonic_violations_gt_0.01": int(len(viol_sig)),
        "monotonic_violations_gt_0.01_with_n_ge_100": int((viol_sig["n"] >= 100).sum()),
    }
    # final B1 (seasons through `last`) with out-of-sample B0 features
    parts = []
    for T in seasons_avail[-a.b1_seasons:]:
        wt = B0.WPTable(table_before(counts, T, k_final))
        d = st.loc[st["season"] == T, ["key", "p_pregame", "progress", "home_win"]].copy()
        d["wp_b0"] = wt.wp_key(d["key"].to_numpy())
        parts.append(d)
    tr = pd.concat(parts)
    b1f = B1Model.fit(tr["wp_b0"], tr["p_pregame"], tr["progress"], tr["home_win"])
    out["final_b1"] = {"train_seasons": [int(x) for x in seasons_avail[-a.b1_seasons:]], **b1f.as_dict()}
    (a.models / "b1_coef.json").write_text(json.dumps(out["final_b1"], indent=1))
    out["elo_params_file"] = "models/elo_params.json"
    (a.models / "elo_params.json").write_text(json.dumps(elo_params.__dict__, indent=1))

    timings["total_s"] = round(time.time() - t0, 1)
    out["timings"] = timings
    (a.reports / "results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(timings))


if __name__ == "__main__":
    main()

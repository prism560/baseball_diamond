"""B2 walk-forward pipeline: exact DP + count chain, compared with B0 and B1.

For each evaluation season S (2023, 2024, 2025):
  1. window K in {3, 5, 10, 20}: for each K, transitions from the K included
     seasons before S-1 (2020 excluded), DP solved, log loss on S-1's PA-start
     states; the K with the lowest loss is used for S (only data < S used).
  2. transitions (q, A, N, direct D) from the K seasons before S; DP solved
     (diff range R chosen by a convergence check), count chain fitted by IPF.
  3. PA-start evaluation: B0, B1 (as in run_pipeline) and B2 variants on all
     PA-start states of S; pitch-level evaluation on every pitch of S.
Run:  PYTHONPATH=src python -m wp.run_b2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import b0_table as B0
from . import count_chain as CC
from . import dp
from . import elo as E
from . import evaluate as EV
from . import states as ST
from . import transitions as T
from .b1_model import B1Model
from .run_pipeline import load as load_states
from .run_pipeline import table_before

STRATA = dict(EV.STRATA)
STRATA["extra(inning>=10)"] = lambda d: d["inning"].to_numpy() >= 10


def window(seasons_avail, before: int, K: int):
    return [s for s in seasons_avail if s < before][-K:]


def b2_lookup(res: dp.DPResult, inning, half, outs, bases, diff, arr="V", G=None):
    inn = np.minimum(np.asarray(inning), 10)
    d = np.clip(np.asarray(diff), -res.R - res.pad, res.R + res.pad) + res.R + res.pad
    out = np.empty(len(inn))
    for (i, h), V in res.V.items():
        m = (inn == i) & (np.asarray(half) == h)
        if m.any():
            out[m] = V[np.asarray(outs)[m], np.asarray(bases)[m], d[m], 0, 0]
    return out


def solve_b2(tr: T.Transitions, R: int, use_N=True, direct=False) -> dp.DPResult:
    return dp.Engine(R, 1, xrule=True).solve(dp.kernels_from(tr, use_N=use_N, direct=direct))


def choose_R(tr: T.Transitions, start: int = 30, tol: float = 1e-6):
    hist = []
    R = start
    prev = solve_b2(tr, R)
    while True:
        nxt = solve_b2(tr, R + 10)
        mx = 0.0
        for k in prev.V:
            a = prev.V[k][:, :, prev.pad:prev.pad + 2 * R + 1, 0, 0]
            b = nxt.V[k][:, :, nxt.pad + 10:nxt.pad + 10 + 2 * R + 1, 0, 0]
            mx = max(mx, float(np.abs(a - b).max()))
        hist.append({"R": R, "vs_R": R + 10, "max_abs_change": mx})
        if mx < tol or R >= 100:
            return R, hist
        R += 10
        prev = nxt


def pitch_rows(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per pitch: state before the pitch (after N and any earlier
    in-PA runner events) and the count."""
    recs = []
    for gid, inn, half, o1, b1, d0, rN, pit, mid in zip(
            pa["game_id"].to_numpy(), pa["inning"].to_numpy(), pa["half"].to_numpy(),
            pa["outs1"].to_numpy(), pa["bases1"].to_numpy(), pa["diff0"].to_numpy(),
            pa["rN"].to_numpy(), pa["pitches"].to_numpy(), pa["mid"].to_numpy()):
        cnts = CC.counts_before_each_pitch(pit)
        if not cnts:
            continue
        d = int(d0) + (1 if half == 1 else -1) * int(rN)
        segs = []
        if mid:
            for s in mid.split("|"):
                n, o, b, dd = (int(x) for x in s.split(":"))
                segs.append((n, o, b, dd))
        o, b = int(o1), int(b1)
        si = 0
        for j, c in enumerate(cnts):
            while si < len(segs) and segs[si][0] <= j:
                o, b, d = segs[si][1], segs[si][2], segs[si][3]
                si += 1
            if o >= 3 or (half == 1 and inn >= 9 and d > 0):
                break
            recs.append((gid, inn, half, o, b, d, c))
    df = pd.DataFrame(recs, columns=["game_id", "inning", "half", "outs", "bases", "diff", "count"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--eval", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--K-grid", type=int, nargs="+", default=[3, 5, 10, 20])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    a = ap.parse_args()
    t_all = time.time()
    a.reports.mkdir(parents=True, exist_ok=True)
    a.models.mkdir(parents=True, exist_ok=True)
    timings = {}
    out: dict = {"eval": a.eval}

    # ---------- data ----------
    t = time.time()
    games, gl, st = load_states(a.processed)
    pa_path = a.processed / "pa.parquet"
    if pa_path.exists():
        pa = pd.read_parquet(pa_path)
    else:
        ev = pd.read_parquet(a.processed / "events.parquet")
        ev = ev[ev["game_id"].str[3:7].astype(int) >= 2000]
        pa = T.build_pa_table(ev)
        pa.to_parquet(pa_path, index=False)
    inc = set(games.loc[games["included"], "game_id"])
    pa = pa[pa["game_id"].isin(inc)].reset_index(drop=True)
    # recompute n_next on the filtered table (positions changed)
    g_, h_, i_ = pa["game_id"].to_numpy(), pa["half"].to_numpy(), pa["inning"].to_numpy()
    same = (g_[1:] == g_[:-1]) & (h_[1:] == h_[:-1]) & (i_[1:] == i_[:-1])
    pa["n_next"] = np.where(np.r_[same, False], np.arange(1, len(pa) + 1), -1)
    seasons_avail = sorted(s for s in pa["season"].unique() if s != 2020)
    timings["load_s"] = round(time.time() - t, 1)
    out["pa_rows"] = int(len(pa))
    out["share_pa_with_N_event"] = float(((pa["outs1"] != pa["outs0"]) | (pa["bases1"] != pa["bases0"])
                                          | (pa["rN"] > 0)).mean())
    out["share_pa_with_midpa_runner_event"] = float((pa["mid"] != "").mean())
    out["share_pa_INT"] = float((pa["o"] == T.O_IDX["INT"]).mean())

    # ---------- Elo / B0 counts (as in run_pipeline) ----------
    t = time.time()
    elo_params, _ = E.tune(gl, list(range(min(a.eval) - 30, min(a.eval))))
    elo = E.run_elo(gl, elo_params)
    st = st.merge(elo[["game_id", "p_pregame"]], on="game_id", how="left")
    st = st[st["p_pregame"].notna()].reset_index(drop=True)
    counts = B0.season_counts(st)
    st_seasons = sorted(st["season"].unique())
    timings["elo_b0_counts_s"] = round(time.time() - t, 1)

    # ---------- diff range convergence ----------
    t = time.time()
    tr_ref = T.estimate(pa, window(seasons_avail, max(a.eval), 5))
    R, rhist = choose_R(tr_ref)
    out["R"] = R
    out["R_convergence"] = rhist
    timings["choose_R_s"] = round(time.time() - t, 1)

    per = {}
    pa_eval_all, pitch_eval_all = [], []
    for S in a.eval:
        rec = {}
        prev = max(s for s in seasons_avail if s < S)
        # --- choose K on S-1 ---
        t = time.time()
        val = st[st["season"] == prev]
        kres = {}
        for K in a.K_grid:
            w = window(seasons_avail, prev, K)
            r_ = solve_b2(T.estimate(pa, w), R)
            p = b2_lookup(r_, val["inning"], val["half"], val["outs"], val["bases"], val["diff"])
            kres[K] = float(np.mean(EV.logloss_vec(p, val["home_win"])))
        Kbest = min(kres, key=kres.get)
        rec["K_grid_logloss_on_prev"] = {str(k): v for k, v in kres.items()}
        rec["K"] = Kbest
        rec["K_select_s"] = round(time.time() - t, 1)
        wS = window(seasons_avail, S, Kbest)
        rec["train_seasons"] = wS

        # --- transitions & DP ---
        t = time.time()
        tr = T.estimate(pa, wS)
        rec["transition_fit_s"] = round(time.time() - t, 2)
        res = solve_b2(tr, R)
        rec["dp_seconds"] = round(res.seconds, 3)
        rec["dp_peak_MB"] = round(res.peak_bytes / 1e6, 1)
        res_noN = solve_b2(tr, R, use_N=False)
        res_dir = solve_b2(tr, R, direct=True)
        t = time.time()
        res9 = dp.Engine(R, 9, xrule=True).solve(dp.kernels_from(tr, S=9))
        rec["dp_S9_seconds"] = round(res9.seconds, 2)
        rec["dp_S9_peak_MB"] = round(res9.peak_bytes / 1e6, 1)
        rec["dp_S9_max_abs_vs_S1"] = float(max(np.abs(res9.V[k] - res.V[k][..., :1, :1]).max() for k in res.V))
        del res9
        # --- count chain ---
        t = time.time()
        base = CC.estimate_base(pa, wS)
        chain = CC.fit(base, tr.q)
        G = dp.after_outcome_values(res, tr)
        pw = CC.pitch_wp_table(chain, G)
        rec["count_chain_s"] = round(time.time() - t, 2)
        rec["ipf_iterations"] = chain.iterations
        rec["ipf_max_abs_err"] = chain.max_err
        rec["count_chain_pa_used"] = base.n_pa
        rec["count_chain_pa_skipped_inconsistent"] = base.n_skipped
        Wtab = {k: np.einsum("xyO,xydO->xyd", tr.q[k[1]], g) for k, g in G.items()}
        rec["max_abs_pitch00_minus_W"] = float(max(np.abs(pw[k][..., 0] - Wtab[k]).max() for k in pw))

        # --- PA-start evaluation ---
        ev = st[st["season"] == S].copy()
        tab = table_before(counts, S, 30)   # B0 pseudo-count k = 30 (selected in every B0/B1 run)
        ev["B0"] = B0.WPTable(tab).wp_key(ev["key"].to_numpy())
        b1_seas = [s for s in st_seasons if s < S][-5:]
        parts = []
        for Tt in b1_seas:
            wt = B0.WPTable(table_before(counts, Tt, 30))
            d_ = st.loc[st["season"] == Tt, ["key", "p_pregame", "progress", "home_win"]].copy()
            d_["wp_b0"] = wt.wp_key(d_["key"].to_numpy())
            parts.append(d_)
        trb = pd.concat(parts)
        b1 = B1Model.fit(trb["wp_b0"], trb["p_pregame"], trb["progress"], trb["home_win"])
        ev["B1"] = b1.predict(ev["B0"], ev["p_pregame"], ev["progress"])
        args = (ev["inning"], ev["half"], ev["outs"], ev["bases"], ev["diff"])
        ev["B2"] = b2_lookup(res, *args)
        ev["B2_noN"] = b2_lookup(res_noN, *args)
        ev["B2_direct"] = b2_lookup(res_dir, *args)
        rec["n_states"], rec["n_games"] = int(len(ev)), int(ev["game_id"].nunique())
        t = time.time()
        bA = EV.block_bootstrap(ev, ["B0", "B2", "B1", "B2_noN", "B2_direct"], n_boot=a.n_boot, seed=S,
                                strata=STRATA)
        bB = EV.block_bootstrap(ev, ["B1", "B2"], n_boot=a.n_boot, seed=S, strata=STRATA)
        rec["pa_metrics_vs_B0"] = bA.to_dict(orient="records")
        rec["pa_metrics_vs_B1"] = bB.to_dict(orient="records")
        rec["pa_calibration"] = {m: EV.calibration(ev[m], ev["home_win"]).reset_index()
                                 .rename(columns={"index": "bin"}).to_dict(orient="records")
                                 for m in ("B0", "B1", "B2")}
        pa_eval_all.append(ev[["game_id", "season", "inning", "diff", "home_win", "B0", "B1", "B2",
                               "B2_noN", "B2_direct"]])

        # --- pitch-level evaluation ---
        pS = pa[pa["season"] == S]
        pr = pitch_rows(pS)
        pr = pr.merge(games[["game_id", "home_win"]], on="game_id")
        key = ST.encode(pr["inning"].to_numpy(), pr["half"].to_numpy(), pr["outs"].to_numpy(),
                        pr["bases"].to_numpy(), pr["diff"].to_numpy(), 1)
        pr["B0"] = B0.WPTable(tab).wp_key(key)
        pr["B2_nocount"] = b2_lookup(res, pr["inning"], pr["half"], pr["outs"], pr["bases"], pr["diff"])
        inn = np.minimum(pr["inning"].to_numpy(), 10)
        dI = np.clip(pr["diff"].to_numpy(), -res.R - res.pad, res.R + res.pad) + res.R + res.pad
        val_ = np.empty(len(pr))
        val_w = np.empty(len(pr))
        for (i, h), arr in pw.items():
            m = (inn == i) & (pr["half"].to_numpy() == h)
            val_[m] = arr[pr["outs"].to_numpy()[m], pr["bases"].to_numpy()[m], dI[m], pr["count"].to_numpy()[m]]
            val_w[m] = Wtab[(i, h)][pr["outs"].to_numpy()[m], pr["bases"].to_numpy()[m], dI[m]]
        pr["B2"] = val_
        pr["B2_firstpitch_nocount"] = val_w
        pr["season"] = S
        rec["n_pitches"] = int(len(pr))
        bP = EV.block_bootstrap(pr, ["B0", "B2", "B2_firstpitch_nocount"], n_boot=a.n_boot, seed=S + 7,
                                strata=STRATA)
        rec["pitch_metrics"] = bP.to_dict(orient="records")
        bC = EV.block_bootstrap(pr, ["B2_firstpitch_nocount", "B2"], n_boot=a.n_boot, seed=S + 11,
                                strata=STRATA)
        rec["pitch_count_gain"] = bC.to_dict(orient="records")
        rec["bootstrap_s"] = round(time.time() - t, 1)
        pitch_eval_all.append(pr[["game_id", "season", "inning", "diff", "home_win", "B0", "B2",
                                  "B2_firstpitch_nocount"]])

        # --- B0 vs B2 theory comparison (cells with n >= 1000 in the B0 table) ---
        cells = tab[(tab["n"] >= 1000) & tab["possible"]].copy()
        cells = cells[(cells["inning"] < 10) | (cells["xrule"] == 1)]
        a_ = (cells["inning"], cells["half"], cells["outs"], cells["bases"], cells["diff"])
        cells["B2"] = b2_lookup(res, *a_)
        cells["B2_noN"] = b2_lookup(res_noN, *a_)
        recent = B0.WPTable(B0.fit_table(counts[counts["season"].isin(wS)], 30))
        cells["B0_recent"] = recent.wp(cells["inning"], cells["half"], cells["outs"], cells["bases"],
                                       cells["diff"], cells["xrule"])
        cells["n_recent"] = recent.n(cells["inning"], cells["half"], cells["outs"], cells["bases"],
                                     cells["diff"], cells["xrule"])
        cells["delta"] = cells["B2"] - cells["wp"]
        se = np.sqrt(np.clip(cells["B0_recent"] * (1 - cells["B0_recent"]), 1e-4, None)
                     / np.maximum(cells["n_recent"], 1) * 5.0)   # x5: PA states within games are correlated
        big = cells["delta"].abs() > 0.02
        cls = np.where(~big, "small",
              np.where((cells["B2"] - cells["B0_recent"]).abs() < 0.5 * cells["delta"].abs(), "era_difference",
              np.where((cells["B2"] - cells["B2_noN"]).abs() > 0.5 * cells["delta"].abs(), "non_batting_events",
              np.where((cells["B2"] - cells["B0_recent"]).abs() < 2 * se, "sampling_noise",
                       "markov_or_structure"))))
        cells["class"] = cls
        rec["b0_compare"] = {
            "cells": int(len(cells)),
            "abs_delta_quantiles": {str(qq): float(cells["delta"].abs().quantile(qq))
                                    for qq in (0.5, 0.9, 0.99, 1.0)},
            "mean_delta": float(cells["delta"].mean()),
            "class_counts": pd.Series(cls).value_counts().to_dict(),
            "top": cells.reindex(cells["delta"].abs().sort_values(ascending=False).index)
            .head(12)[["inning", "half", "outs", "bases", "diff", "n", "wp", "B0_recent", "B2", "B2_noN",
                       "class"]].to_dict(orient="records"),
        }
        # --- decomposed vs direct, N effect on the table ---
        def tab_diff(r1, r2):
            ds = []
            for k in r1.V:
                a1 = r1.V[k][:, :, r1.pad:r1.pad + 2 * R + 1, 0, 0]
                a2 = r2.V[k][:, :, r2.pad:r2.pad + 2 * R + 1, 0, 0]
                dd = (a1 - a2)[:, :, R - 10:R + 11]            # |diff| <= 10
                ds.append(np.abs(dd).ravel())
            x = np.concatenate(ds)
            return {"max": float(x.max()), "mean": float(x.mean()), "p99": float(np.quantile(x, .99))}
        rec["direct_vs_decomposed_abs_diff(|d|<=10)"] = tab_diff(res, res_dir)
        rec["N_effect_abs_diff(|d|<=10)"] = tab_diff(res, res_noN)
        rec["X_extra_tie_value"] = float(res.X[0, 0])
        rec["start_wp"] = float(res.wp(1, 0, 0, 0, 0))
        per[S] = rec
        print(S, "K", Kbest, "R", R, flush=True)

        if S == max(a.eval):
            final = (res, tr, chain, G, pw, Wtab)

    # ---------- pooled ----------
    pooled = pd.concat(pa_eval_all, ignore_index=True)
    out["pooled_pa_vs_B0"] = EV.block_bootstrap(pooled, ["B0", "B2", "B1", "B2_noN", "B2_direct"],
                                                n_boot=a.n_boot, seed=1, strata=STRATA).to_dict(orient="records")
    out["pooled_pa_vs_B1"] = EV.block_bootstrap(pooled, ["B1", "B2"], n_boot=a.n_boot, seed=2,
                                                strata=STRATA).to_dict(orient="records")
    pp = pd.concat(pitch_eval_all, ignore_index=True)
    out["pooled_pitch"] = EV.block_bootstrap(pp, ["B0", "B2", "B2_firstpitch_nocount"], n_boot=a.n_boot,
                                             seed=3, strata=STRATA).to_dict(orient="records")
    out["pooled_pitch_count_gain"] = EV.block_bootstrap(pp, ["B2_firstpitch_nocount", "B2"], n_boot=a.n_boot,
                                                        seed=4, strata=STRATA).to_dict(orient="records")
    out["per_season"] = {str(k): v for k, v in per.items()}

    # ---------- artefacts (window of the last evaluation season) ----------
    res, tr, chain, G, pw, Wtab = final
    t = time.time()
    rows = []
    for (i, h), V in res.V.items():
        for o in range(3):
            for b in range(8):
                for di in range(2 * R + 1):
                    d = di - R
                    rows.append((i, h, o, b, d, V[o, b, di + res.pad, 0, 0], Wtab[(i, h)][o, b, di + res.pad]))
    tab_pa = pd.DataFrame(rows, columns=["inning", "half", "outs", "bases", "diff", "wp", "wp_first_pitch"])
    tab_pa["possible"] = ST.is_possible(tab_pa["inning"], tab_pa["half"], tab_pa["outs"], tab_pa["bases"],
                                        tab_pa["diff"])
    tab_pa.to_parquet(a.models / "wp_b2_pa.parquet", index=False)
    prow = []
    for (i, h), arr in pw.items():
        o, b, dd, c = np.meshgrid(range(3), range(8), range(2 * R + 1), range(CC.NC), indexing="ij")
        vals = arr[o, b, dd + res.pad, c]
        prow.append(pd.DataFrame({"inning": i, "half": h, "outs": o.ravel(), "bases": b.ravel(),
                                  "diff": (dd - R).ravel(), "balls": np.array([x[0] for x in CC.COUNTS])[c.ravel()],
                                  "strikes": np.array([x[1] for x in CC.COUNTS])[c.ravel()],
                                  "wp": vals.ravel()}))
    tab_p = pd.concat(prow, ignore_index=True)
    tab_p.to_parquet(a.models / "wp_b2_pitch.parquet", index=False)
    timings["write_tables_s"] = round(time.time() - t, 1)
    from .wp_b2 import PitchWP
    lk = PitchWP.load(a.models / "wp_b2_pitch.parquet")
    t = time.perf_counter()
    n = 20000
    rng = np.random.default_rng(0)
    for _ in range(n):
        lk.wp(int(rng.integers(1, 12)), int(rng.integers(0, 2)), 1, 3, 0, 1, 2)
    out["lookup_us_per_call"] = (time.perf_counter() - t) / n * 1e6
    out["table_rows"] = {"wp_b2_pa": int(len(tab_pa)), "wp_b2_pitch": int(len(tab_p))}
    out["table_bytes"] = {"wp_b2_pa": (a.models / "wp_b2_pa.parquet").stat().st_size,
                          "wp_b2_pitch": (a.models / "wp_b2_pitch.parquet").stat().st_size}
    out["final_window"] = per[max(a.eval)]["train_seasons"]
    timings["total_s"] = round(time.time() - t_all, 1)
    out["timings"] = timings
    (a.reports / "b2_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(timings))


if __name__ == "__main__":
    main()

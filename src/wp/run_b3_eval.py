"""B3 evaluation: B0, B1, B2, B2E and B3 on the games solved by run_b3_games.

B2E (new baseline):  logit WP = a + b logit(B2) + c logit(P_Elo) + d logit(P_Elo) * progress,
fitted like B1 on the five seasons before S with out-of-sample B2 values (B2 solved on
the fixed K=10 window before each training season).

Pitch level: every pitch thrown *before* any in-PA runner event (B3 values are
computed at the first-pitch state; the same pitches are used for every model).
  B3   = sum_o P(o | count; matchup q, IPF per PA) G(o)
  B2   = league count chain (as in run_b2), B2E = B2E formula on B2
  B0/B1 = state value at the pitch (count ignored)
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
from . import transitions as T
from .b1_model import B1Model
from .manifest import check_fresh, code_files, write_meta
from .run_pipeline import load as load_states
from .run_pipeline import table_before

EVAL = [2023, 2024, 2025]
LED = 0.025


def window(seasons, before, K=10):
    return [s for s in seasons if s < before and s != 2020][-K:]


def b2_values(res, inning, half, outs, bases, diff):
    inn = np.minimum(np.asarray(inning), 10)
    d = np.clip(np.asarray(diff), -res.R - res.pad, res.R + res.pad) + res.R + res.pad
    out = np.empty(len(inn))
    h = np.asarray(half)
    for (i, hh), V in res.V.items():
        m = (inn == i) & (h == hh)
        out[m] = V[np.asarray(outs)[m], np.asarray(bases)[m], d[m], 0, 0]
    return out


def strata(df):
    s = dict(EV.STRATA)
    s["extra(inning>=10)"] = lambda d: d["inning"].to_numpy() >= 10
    s["starter pitching"] = lambda d: d["starter_in"].to_numpy()
    s["bullpen pitching"] = lambda d: ~d["starter_in"].to_numpy()
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seasons", type=int, nargs="+", default=EVAL)
    ap.add_argument("--extra-seasons", type=int, nargs="*", default=[2022],
                    help="seasons with (partial) B3 runs that are dumped for combination training "
                         "but excluded from every metric")
    a = ap.parse_args()
    P = a.processed
    ins = [P / f for f in ("games.parquet", "gamelogs.parquet", "states.parquet", "pa.parquet")] + \
          [Path("reports/b3_check.json")]
    for f in ins:
        check_fresh(f)
    t0 = time.time()
    games, gl, st = load_states(P)
    pa = pd.read_parquet(P / "pa.parquet")
    inc = set(games.loc[games["included"], "game_id"])
    seasons_avail = sorted(int(s) for s in pa["season"].unique() if s != 2020)
    elo_params, _ = E.tune(gl, list(range(min(a.seasons) - 30, min(a.seasons))))
    elo = E.run_elo(gl, elo_params)
    st = st.merge(elo[["game_id", "p_pregame"]], on="game_id", how="left")
    st = st[st["p_pregame"].notna()].reset_index(drop=True)
    st["seq"] = st.groupby("game_id").cumcount()
    counts = B0.season_counts(st)
    st_seasons = sorted(st["season"].unique())
    runs_root = P / "b3_runs"
    # "pregame" holds the pre-game diagnostic run (one value per game), not an ablation
    abls = sorted(p.name for p in runs_root.iterdir() if p.name != "pregame") if runs_root.exists() else []
    out = {"ablations_found": abls, "per_season": {}}
    pa_all, pitch_all = [], []
    b2_cache = {}

    def b2_for(Tseason):
        if Tseason not in b2_cache:
            tr = T.estimate(pa[pa["game_id"].isin(inc)], window(seasons_avail, Tseason))
            b2_cache[Tseason] = (tr, dp.Engine(30, 1).solve(dp.kernels_from(tr)))
        return b2_cache[Tseason]

    for S in sorted(set(a.seasons) | set(a.extra_seasons)):
        rec = {}
        files = sorted((runs_root / "full" / str(S)).glob("*.npz"))
        gids = [f.stem for f in files]
        if not gids:
            continue
        # ---- PA-start frame (states table order == PA table order within a game) ----
        ev = st[st["game_id"].isin(gids)].copy()
        pS = pa[pa["game_id"].isin(gids)].copy()
        pS["seq"] = pS.groupby("game_id").cumcount()
        pS["starter_in"] = (pS["pitcher"].astype(str) == pS["starter"].astype(str)).to_numpy()
        n_st = ev.groupby("game_id").size()
        n_pa = pS.groupby("game_id").size()
        ok = n_st.index[n_st.reindex(n_pa.index).eq(n_pa).reindex(n_st.index).fillna(False)]
        rec["games"] = len(gids)
        rec["games_aligned"] = int(len(ok))
        ev = ev[ev["game_id"].isin(ok)]
        pS = pS[pS["game_id"].isin(ok)]
        ev = ev.merge(pS[["game_id", "seq", "starter_in", "outs1", "bases1", "rN", "o", "pitches", "mid"]],
                      on=["game_id", "seq"], how="inner")
        b3 = {}
        meta_t = []
        for ab in abls:
            vals = {}
            for gid in ok:
                f = runs_root / ab / str(S) / f"{gid}.npz"
                if f.exists():
                    z = np.load(f)
                    vals[gid] = z
                    if ab == "full":
                        meta_t.append((float(z["pregame_s"]), int(z["n_resolve"]), float(z["resolve_s"]),
                                       float(z["seconds"])))
            b3[ab] = vals
        full = b3["full"]
        ev["B3"] = np.concatenate([full[g]["wp"] for g in ev["game_id"].unique()])
        for ab in abls:
            if ab == "full":
                continue
            # ablations run on a game subset: NaN for games without that run
            n_by = ev.groupby("game_id", sort=False).size()
            ev[f"B3_{ab}"] = np.concatenate([b3[ab][g]["wp"] if g in b3[ab] else np.full(n, np.nan)
                                             for g, n in n_by.items()])
        mt = np.array(meta_t)
        rec["timing"] = {"pregame_s_mean": float(mt[:, 0].mean()), "pregame_s_max": float(mt[:, 0].max()),
                         "resolves_per_game": float(mt[:, 1].mean()), "resolve_s_mean": float(mt[:, 2].mean()),
                         "game_total_s_mean": float(mt[:, 3].mean())}
        # ---- B0 / B1 ----
        tab = table_before(counts, S, 30)
        ev["B0"] = B0.WPTable(tab).wp_key(ev["key"].to_numpy())
        b1_seas = [s for s in st_seasons if s < S][-5:]
        parts = []
        for Tt in b1_seas:
            wt = B0.WPTable(table_before(counts, Tt, 30))
            d_ = st.loc[st["season"] == Tt, ["key", "p_pregame", "progress", "home_win", "inning", "half",
                                               "outs", "bases", "diff"]].copy()
            d_["wp_b0"] = wt.wp_key(d_["key"].to_numpy())
            tr_t, res_t = b2_for(Tt)
            d_["wp_b2"] = b2_values(res_t, d_["inning"], d_["half"], d_["outs"], d_["bases"], d_["diff"])
            d_["season_"] = Tt
            parts.append(d_)
        trb = pd.concat(parts)
        b1 = B1Model.fit(trb["wp_b0"], trb["p_pregame"], trb["progress"], trb["home_win"])
        b2e = B1Model.fit(trb["wp_b2"], trb["p_pregame"], trb["progress"], trb["home_win"])
        rec["b2e_coef"] = b2e.as_dict()
        rec["b1_coef"] = b1.as_dict()
        ev["B1"] = b1.predict(ev["B0"], ev["p_pregame"], ev["progress"])
        trS, resS = b2_for(S)
        ev["B2"] = b2_values(resS, ev["inning"], ev["half"], ev["outs"], ev["bases"], ev["diff"])
        ev["B2E"] = b2e.predict(ev["B2"], ev["p_pregame"], ev["progress"])
        ev["Elo"] = ev["p_pregame"]
        rec["n_states"] = int(len(ev))
        pa_all.append(ev)

        # ---- pitch level ----
        t = time.time()
        q = np.concatenate([full[g]["q"] for g in ev["game_id"].unique()])
        G = np.concatenate([full[g]["G"] for g in ev["game_id"].unique()])
        has = ~np.isnan(q).any(1) & ~np.isnan(G).any(1)
        base = CC.estimate_base(pa[pa["game_id"].isin(inc)], window(seasons_avail, S))
        chain = CC.fit(base, q[has])
        P3 = np.full((len(ev), CC.NC), np.nan)
        P3[has] = np.einsum("nco,no->nc", chain.P, G[has])
        rec["ipf_max_abs_err_per_matchup"] = float(chain.max_err)
        # league (B2) pitch values
        G2 = dp.after_outcome_values(resS, trS)
        ch2 = CC.fit(base, trS.q)
        pw2 = CC.pitch_wp_table(ch2, G2)
        recs = []
        evr = ev.reset_index(drop=True)
        for i, rrow in enumerate(evr.itertuples(index=False)):
            if not has[i]:
                continue
            cnts = CC.counts_before_each_pitch(rrow.pitches)
            first_mid = min((int(x.split(":")[0]) for x in rrow.mid.split("|")), default=10 ** 6) if rrow.mid else 10 ** 6
            for j, c in enumerate(cnts):
                if j >= first_mid:
                    break
                recs.append((i, c))
        pr = pd.DataFrame(recs, columns=["i", "count"])
        src = evr.iloc[pr["i"]].reset_index(drop=True)
        pr = pd.concat([pr, src[["game_id", "season", "inning", "half", "home_win", "starter_in", "p_pregame",
                                  "outs1", "bases1", "rN", "diff"]]], axis=1)
        sgn = np.where(pr["half"] == 1, 1, -1)
        pr["d1"] = pr["diff"] + sgn * pr["rN"]
        pr["B3"] = P3[pr["i"], pr["count"]]
        inn = np.minimum(pr["inning"].to_numpy(), 10)
        dI = np.clip(pr["d1"].to_numpy(), -resS.R - resS.pad, resS.R + resS.pad) + resS.R + resS.pad
        v2 = np.empty(len(pr))
        for (ii, hh), arr in pw2.items():
            m = (inn == ii) & (pr["half"].to_numpy() == hh)
            v2[m] = arr[pr["outs1"].to_numpy()[m], pr["bases1"].to_numpy()[m], dI[m], pr["count"].to_numpy()[m]]
        pr["B2"] = v2
        from . import states as STt
        key = STt.encode(pr["inning"].to_numpy(), pr["half"].to_numpy(), pr["outs1"].to_numpy(),
                         pr["bases1"].to_numpy(), pr["d1"].to_numpy(), 1)
        prog = STt.progress(pr["inning"].to_numpy(), pr["half"].to_numpy(), pr["outs1"].to_numpy())
        pr["B0"] = B0.WPTable(tab).wp_key(key)
        pr["B1"] = b1.predict(pr["B0"], pr["p_pregame"], prog)
        pr["B2E"] = b2e.predict(pr["B2"], pr["p_pregame"], prog)
        pr["diff"] = pr["d1"]
        rec["n_pitches"] = int(len(pr))
        rec["pitch_build_s"] = round(time.time() - t, 1)
        pitch_all.append(pr)
        if S in a.seasons:
            out["per_season"][str(S)] = rec

    pa_df = pd.concat(pa_all, ignore_index=True)
    pt_df = pd.concat(pitch_all, ignore_index=True)
    # per-PA / per-pitch frames with every model column (inputs of wp.run_b4_combine)
    keep_pa = [c for c in pa_df.columns if c in ("game_id", "season", "seq", "inning", "half", "outs", "bases", "diff",
                                                   "home_win", "starter_in", "p_pregame", "progress", "Elo")
               or c.startswith(("B0", "B1", "B2", "B3"))]
    pa_df[keep_pa].to_parquet(P / "b3_eval_pa.parquet", index=False)
    keep_pt = [c for c in pt_df.columns if c not in ("i",)]
    pt_df[keep_pt].to_parquet(P / "b3_eval_pitch.parquet", index=False)
    out["extra_seasons_dumped_only"] = sorted(set(a.extra_seasons) - set(a.seasons))
    pa_df = pa_df[pa_df["season"].isin(a.seasons)].reset_index(drop=True)
    pt_df = pt_df[pt_df["season"].isin(a.seasons)].reset_index(drop=True)
    SD = strata(pa_df)
    nb = a.n_boot
    out["pa_vs_B1"] = EV.block_bootstrap(pa_df, ["B1", "B3", "B0", "B2"], n_boot=nb, seed=1, strata=SD).to_dict("records")
    out["pa_vs_B2E"] = EV.block_bootstrap(pa_df, ["B2E", "B3", "B2"], n_boot=nb, seed=2, strata=SD).to_dict("records")
    out["pitch_vs_B1"] = EV.block_bootstrap(pt_df, ["B1", "B3", "B0", "B2"], n_boot=nb, seed=3, strata=SD).to_dict("records")
    out["pitch_vs_B2E"] = EV.block_bootstrap(pt_df, ["B2E", "B3", "B2"], n_boot=nb, seed=4, strata=SD).to_dict("records")
    abl_cols = [c for c in pa_df.columns if c.startswith("B3_")]
    if abl_cols:
        sub = pa_df.dropna(subset=abl_cols)
        out["ablations"] = EV.block_bootstrap(sub, ["B3"] + abl_cols, n_boot=nb, seed=5, strata=SD).to_dict("records")
        out["ablation_games"] = int(sub["game_id"].nunique())
    # ---- pregame ----
    pre = pa_df[pa_df["seq"] == 0].copy()
    out["pregame"] = EV.block_bootstrap(pre, ["Elo", "B3", "B1", "B2E"], n_boot=nb, seed=6,
                                        strata={"all": EV.STRATA["all"]}).to_dict("records")
    out["pregame_calibration"] = {m: EV.calibration(pre[m], pre["home_win"]).reset_index().rename(
        columns={"index": "bin"}).to_dict("records") for m in ("Elo", "B3")}
    out["pregame_corr_B3_Elo"] = float(np.corrcoef(pre["B3"], pre["Elo"])[0, 1])
    out["pregame_sd"] = {m: float(pre[m].std()) for m in ("Elo", "B3")}
    odds = Path("data/external/odds")
    out["odds_comparison"] = "실시" if odds.exists() and any(odds.iterdir()) else "미실시 (data/external/odds/ 없음)"
    # ---- LED-size differences ----
    out["led"] = {}
    for name, df in (("pa_start", pa_df), ("pitch", pt_df)):
        dd = (df["B3"] - df["B2E"]).abs()
        out["led"][name] = {"share_gt_2.5pp_vs_B2E": float((dd > LED).mean()),
                            "share_gt_5pp_vs_B2E": float((dd > 2 * LED).mean()),
                            "share_gt_2.5pp_vs_B1": float(((df["B3"] - df["B1"]).abs() > LED).mean()),
                            "mean_abs_diff_vs_B2E": float(dd.mean()),
                            "p90_abs_diff_vs_B2E": float(dd.quantile(0.9))}
    out["calibration_pa"] = {m: EV.calibration(pa_df[m], pa_df["home_win"]).reset_index().rename(
        columns={"index": "bin"}).to_dict("records") for m in ("B1", "B2E", "B3")}
    out["n_games_total"] = int(pa_df["game_id"].nunique())
    out["seconds"] = round(time.time() - t0, 1)
    a.reports.mkdir(exist_ok=True)
    o = a.reports / "b3_results.json"
    o.write_text(json.dumps(out, indent=1, default=float))
    write_meta([o, P / "b3_eval_pa.parquet", P / "b3_eval_pitch.parquet"], ins, "run_b3_eval", code_files("run_b3_eval", "b3_game", "dp_b3", "ratings", "matchup",
                                                   "pitching_policy", "count_chain", "dp", "transitions",
                                                   "b0_table", "b1_model", "elo", "evaluate", "run_pipeline"))
    print(json.dumps({k: out[k] for k in ("n_games_total", "led", "seconds")}, indent=1))


if __name__ == "__main__":
    main()

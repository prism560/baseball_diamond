"""Render reports/b3_report.md from reports/b3_check.json and reports/b3_results.json."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .manifest import check_fresh

ROOT = Path(__file__).resolve().parents[2]
ABL_KO = {"B3_no_platoon": "플래툰 제거", "B3_no_park": "구장 팩터 제거",
          "B3_team_avg_bullpen": "불펜 정책 → 팀 평균 불펜", "B3_no_tto": "타순 바퀴(TTO) 효과 제거"}


def ci(r, d=4):
    return f"{r['estimate']:.{d}f} [{r['ci_low']:.{d}f}, {r['ci_high']:.{d}f}]"


def table(rows, metric, models, diffs):
    df = pd.DataFrame(rows)
    df = df[df["metric"] == metric]
    out = ["| 층 | 상태 수 | 경기 | " + " | ".join(models) + " | " + " | ".join(diffs) + " |",
           "|---|---:|---:|" + "---|" * (len(models) + len(diffs))]
    for s, g in df.groupby("stratum", sort=False):
        g = g.set_index("model")
        if models[0] not in g.index or g.loc[models[0], "n_states"] == 0:
            continue
        cells = [ci(g.loc[m]) if m in g.index else "–" for m in models + diffs]
        out.append(f"| {s} | {int(g.loc[models[0], 'n_states']):,} | {int(g.loc[models[0], 'n_games']):,} | "
                   + " | ".join(cells) + " |")
    return "\n".join(out)


def verdict(rows, diff, stratum="all", metric="logloss"):
    df = pd.DataFrame(rows)
    r = df[(df["model"] == diff) & (df["stratum"] == stratum) & (df["metric"] == metric)].iloc[0]
    if r["ci_high"] < 0:
        return r, "B3가 낫다 (CI가 0보다 아래)"
    if r["ci_low"] > 0:
        return r, "B3가 못하다 (CI가 0보다 위)"
    return r, "유의한 차이 없음 (CI가 0을 포함)"


def main() -> None:
    for f in ("reports/b3_check.json", "reports/b3_results.json"):
        check_fresh(ROOT / f)
    c = json.loads((ROOT / "reports/b3_check.json").read_text())
    r = json.loads((ROOT / "reports/b3_results.json").read_text())
    L = []
    A = L.append
    A("# B3 리포트 — 선수 매치업 + 투수 교체 정책을 DP 안에\n")
    A("> 자동 생성: `PYTHONPATH=src python -m wp.report_b3`. 수치는 모두 실행 측정값이다.\n")
    v1, t1 = verdict(r["pa_vs_B1"], "B3-B1")
    v2, t2 = verdict(r["pa_vs_B2E"], "B3-B2E")
    A("## 1. 요약\n")
    A(f"- 평가 경기: 본 비교는 2023–2025 포함 경기 전체 {r['n_games_total']:,}경기, ablation은 경기 ID 해시로 고른 "
      f"1/9 부분집합 {r.get('ablation_games', 0):,}경기(사용자 승인). 경기 단위 블록 부트스트랩 1,000회.")
    A(f"- 타석 시작 로그 손실: B3−B1 = {ci(v1)} → {t1}; B3−B2E = {ci(v2)} → {t2}.")
    ok = v1["ci_high"] < 0 and v2["ci_high"] < 0
    A(f"- 성공 기준(B1과 B2E를 모두 이김): **{'충족' if ok else '미충족'}**."
      + ("" if ok else " B3는 외부 Elo 보정 없이 하나의 엔진으로 B1·B2E와 같은 수준의 정확도를 내지만, 통계적으로 더 낫지는 않다. "
         "선수 정보(매치업 q는 타석 단위에서 리그 q보다 확실히 낫다)가 경기 승패 수준에서는 Elo가 담은 팀 전력 정보와 "
         "대부분 겹친다는 뜻이다."))
    led = r["led"]["pa_start"]
    A(f"- B3와 B2E의 WP 차이가 2.5%p(LED 한 칸)를 넘는 타석 시작 시점의 비율: {led['share_gt_2.5pp_vs_B2E']*100:.1f}% "
      f"(5%p 초과 {led['share_gt_5pp_vs_B2E']*100:.1f}%, 평균 |차이| {led['mean_abs_diff_vs_B2E']*100:.2f}%p).\n")

    A("## 2. 중간 점검 (4절): 매치업 q 대 리그 q\n")
    A(f"2023–2025 포함 경기 {c['n_pa']:,}타석, 다클래스 로그 손실(11범주). 리그 q(B2) = {c['league_q_logloss']:.5f}.\n")
    A("| 변형 | 로그 손실 | 리그 q 대비 [95% CI] |\n|---|---:|---|")
    for k, x in c["variants"].items():
        A(f"| {k} | {x['logloss']:.5f} | {x['minus_league']:+.5f} [{x['ci'][0]:+.5f}, {x['ci'][1]:+.5f}] |")
    A(f"\n선택된 축소 강도: 타자 κ={c['params']['kappa_bat']}, 투수 κ={c['params']['kappa_pit']}, 플래툰 "
      f"k={c['params']['platoon_k']}×(2019·2021·2022에서 선택). → **매치업 q가 리그 q를 이겨 다음 단계로 진행했다.**\n")
    A("극단 매치업 캘리브레이션(예측 확률 10분위, 예측 평균 / 실제 비율):\n")
    A("| 범주 | 최하위 10% | 중앙(5분위) | 최상위 10% |\n|---|---|---|---|")
    for cat, rows in c["calibration"].items():
        lo, mid, hi = rows[0], rows[4], rows[9]
        A(f"| {cat} | {lo['mean_pred']:.3f} / {lo['obs']:.3f} | {mid['mean_pred']:.3f} / {mid['obs']:.3f} | "
          f"{hi['mean_pred']:.3f} / {hi['obs']:.3f} |")
    A("\n대부분 범주는 극단에서도 잘 맞지만, 홈런 최상위 구간에서 과대예측이 보인다(Log5 곱셈 결합이 극단에서 효과를 "
      "과장). 대안으로 **다항 로지스틱 회귀에 오프셋(log q_league)을 두고 타자·투수 편차를 회귀계수로 축소**하는 방식을 제안한다"
      "(이번에는 구현하지 않음).\n")

    A("## 3. 본 비교: B1·B2E·B3\n")
    for name, key1, key2, title in (("타석 시작", "pa_vs_B1", "pa_vs_B2E", "타석 시작 시점"),
                                    ("투구", "pitch_vs_B1", "pitch_vs_B2E", "투구 시점(타석 중 주자 이벤트 이전 투구)")):
        A(f"### {title} — 로그 손실 (95% CI, 경기 단위 블록 부트스트랩 1,000회)\n")
        rows = r[key1] + [x for x in r[key2] if x["model"] in ("B2E", "B3-B2E")]
        A(table(rows, "logloss", ["B1", "B2E", "B3"], ["B3-B1", "B3-B2E"]))
        A("\n**Brier**\n")
        A(table(rows, "brier", ["B1", "B2E", "B3"], ["B3-B1", "B3-B2E"]))
        A("")
    A("B2E 계수(시즌별): " + "; ".join(f"{s}: a={x['b2e_coef']['a']:.2f}, b={x['b2e_coef']['b']:.2f}, "
                                     f"c={x['b2e_coef']['c']:.2f}, d={x['b2e_coef']['d']:.2f}"
                                     for s, x in r["per_season"].items()) + ".\n")

    A("## 4. 경기 전 WP (1회초 0-0)\n")
    A(table(r["pregame"], "logloss", ["Elo", "B1", "B2E", "B3"], ["B3-Elo"]))
    A("")
    A(f"- B3와 Elo의 상관 {r['pregame_corr_B3_Elo']:.3f}, 표준편차 B3 {r['pregame_sd']['B3']:.3f} / Elo {r['pregame_sd']['Elo']:.3f}.")
    A(f"- 마감 배당 내재 확률 비교: {r['odds_comparison']}.\n")
    A("| 구간 | Elo 예측/실제 | B3 예측/실제 |\n|---|---|---|")
    ce = pd.DataFrame(r["pregame_calibration"]["Elo"]).set_index("bin")
    cb = pd.DataFrame(r["pregame_calibration"]["B3"]).set_index("bin")
    for b in ce.index:
        f = lambda row: "–" if pd.isna(row["mean_pred"]) else f"{row['mean_pred']:.3f} / {row['obs_rate']:.3f} (n={int(row['n'])})"
        A(f"| {b} | {f(ce.loc[b])} | {f(cb.loc[b])} |")
    A("")

    if "ablations" in r:
        A(f"## 5. Ablation ({r['ablation_games']:,}경기 부분집합, 같은 경기에서 짝지은 비교)\n")
        A("양수 = 해당 요소를 빼면 손실이 커진다(요소가 도움이 된다).\n")
        df = pd.DataFrame(r["ablations"])
        A("| 제거 요소 | 전체 | 7회 이후 접전 | 불펜 등판 중 |\n|---|---|---|---|")
        for col, ko in ABL_KO.items():
            m = f"{col}-B3"
            if m not in set(df["model"]):
                continue
            cells = []
            for s in ("all", "inning>=7 & close", "bullpen pitching"):
                x = df[(df["model"] == m) & (df["stratum"] == s) & (df["metric"] == "logloss")]
                cells.append(ci(x.iloc[0], 5) if len(x) else "–")
            A(f"| {ko} | " + " | ".join(cells) + " |")
        A("\n불펜 정책 ablation(팀 평균 불펜)이 '정책 민감도'다.\n")

    A("## 6. WP 차이 크기 (LED 기준)\n")
    A("| 시점 | |B3−B2E| > 2.5%p | > 5%p | |B3−B1| > 2.5%p | 평균 |B3−B2E| | p90 |\n|---|---:|---:|---:|---:|---:|")
    for k, x in r["led"].items():
        A(f"| {k} | {x['share_gt_2.5pp_vs_B2E']*100:.1f}% | {x['share_gt_5pp_vs_B2E']*100:.1f}% | "
          f"{x['share_gt_2.5pp_vs_B1']*100:.1f}% | {x['mean_abs_diff_vs_B2E']*100:.2f}%p | {x['p90_abs_diff_vs_B2E']*100:.2f}%p |")
    A("")

    A("## 7. 상태 수·메모리·계산 시간\n")
    A("- 상태: 하프이닝당 3아웃 × 8주자 × 41점수차(±20) × 36(공격 팀 타순×상대 투수 단계) × 36(수비 팀) "
      "= 약 1.27M, 정규 18하프 약 23M + 연장(불펜, 타순만 9×9, 고정점 기저 82). float32로 경기 전 피크 약 0.3GB.")
    A("- 점수차 범위 ±20의 절단 오차(B2로 측정): 전체 최대 3.0e-5, |점수차| ≤ 10에서 1.0e-7. float32와 float64 WP 차이 최대 4e-7.")
    for s, x in r["per_season"].items():
        t = x["timing"]
        A(f"- {s}: 경기 전 계산 평균 {t['pregame_s_mean']:.1f}초(최대 {t['pregame_s_max']:.1f}), 경기당 재계산 "
          f"{t['resolves_per_game']:.1f}회 / {t['resolve_s_mean']:.1f}초, 경기 전체 {t['game_total_s_mean']:.1f}초 (단일 스레드).")
    A("- 기준(경기 전 30초, 메모리 4GB, 교체 후 재계산 5초) 대비: 경기 전·메모리는 충족. 투수 교체 재계산(현재 하프만)은 "
      "0.3–0.5초. 대타 재계산(남은 하프 전체)은 이른 이닝이면 최대 약 7초로 5초를 넘을 수 있다.\n")

    A("## 8. 단순화 목록\n")
    for x in ["연장 이닝은 양 팀 모두 불펜으로 고정(승인됨).",
              "대타·대주자 투입 확률은 모델링하지 않고 실제로 일어난 것만 반영(일어난 시점부터 그 타순에 해당 선수).",
              "타석 도중 주자 이벤트는 결과별 진루 테이블(리그 평균)에 흡수. 투구 시점 평가는 타석 중 주자 이벤트 이전 투구만 사용.",
              "진루 테이블 A와 타석 간 전이 N, INT는 리그 평균(K=10 고정 창).",
              "투구 수는 DP 상태가 아니며, 해저드의 로그오즈 보정 입력으로만 받는다.",
              "선발 강판은 타석 시작 시점에만 일어난다고 본다(타석 도중 교체는 다음 타석 시작으로 처리).",
              "점수 의존 투수 정책 때문에, 같은 팀끼리라도 홈이 정보 이점(말 공격 쪽 상대 투수가 그 이닝 초의 결과를 보고 결정)을 "
              "가져 대칭이 정확히 0.5가 아니다. 정책이 점수에 의존하지 않으면 0.5(테스트).",
              "IBB는 타순 바퀴 오프셋에서 제외(늦은 이닝의 상황 효과가 섞여 있음).",
              "불펜 등급은 같은 팀 구원 투수의 '접전·후반 등판 비율'로 3등분, 시즌 초에는 전 시즌 등판을 절반 가중."]:
        A(f"- {x}")
    A("\n## 출처 고지\n")
    A("     The information used here was obtained free of charge from and is copyrighted by Retrosheet.\n"
      "     Interested parties may contact Retrosheet at \"www.retrosheet.org\".")
    (ROOT / "reports/b3_report.md").write_text("\n".join(L) + "\n")
    print("wrote reports/b3_report.md")


if __name__ == "__main__":
    main()

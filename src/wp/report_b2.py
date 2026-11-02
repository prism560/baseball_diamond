"""Render reports/b2_report.md from b2_results.json and validation_summary.json."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NAME = {"B0": "B0", "B1": "B1", "B2": "B2", "B2_noN": "B2(N 제외)", "B2_direct": "B2(직접 전이)",
        "B2_firstpitch_nocount": "B2(카운트 무시)"}


def ci(r, d=4):
    return f"{r['estimate']:.{d}f} [{r['ci_low']:.{d}f}, {r['ci_high']:.{d}f}]"


def table(rows, metric, models, diffs):
    df = pd.DataFrame(rows)
    df = df[df["metric"] == metric]
    head = "| 층 | 상태 수 | 경기 수 | " + " | ".join(NAME.get(m, m) for m in models) + " | " + \
           " | ".join(diffs) + " |"
    out = [head, "|---|---:|---:|" + "---|" * (len(models) + len(diffs))]
    for s, g in df.groupby("stratum", sort=False):
        g = g.set_index("model")
        first = g.loc[models[0]]
        cells = [ci(g.loc[m]) for m in models] + [ci(g.loc[d]) if d in g.index else "–" for d in diffs]
        out.append(f"| {s} | {int(first['n_states']):,} | {int(first['n_games']):,} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def merged(rA, rB):
    """rows from bootstrap vs B0 plus the B2-B1 rows from the vs-B1 run."""
    return list(rA) + [r for r in rB if r["model"] == "B2-B1"]


def main() -> None:
    r = json.loads((ROOT / "reports/b2_results.json").read_text())
    v = json.loads((ROOT / "data/processed/validation_summary.json").read_text())
    L = []
    A = L.append
    A("# B2 리포트 — 정확 DP 엔진 + 현재 타석 카운트 체인\n")
    A("> 자동 생성: `PYTHONPATH=src python -m wp.report_b2`. 모든 수치는 실제 실행 측정값이다.\n")

    # ---------------- 0 ----------------
    A("## 0. 선행 작업: 파서 검증\n")
    A("검증 항목(`src/wp/validate.py`)과 게임로그 대조 필드:\n")
    A("| 항목 | 방법 |\n|---|---|")
    A("| 이닝별 득점 | 게임로그 라인스코어(필드 20·21)와 하프이닝별 대조 |")
    A("| 3아웃 | 마지막 하프이닝(끝내기·말 공격 생략) 외 모든 하프이닝이 정확히 3아웃 |")
    A("| 상태 연속성 | 하프이닝 종료 시 잔루 합 vs 게임로그 팀 잔루(필드 38·66), 타순 연속성(새 타석의 타자가 다음 타순), "
      "대주자(`sub` 포지션 12)가 실제로 루상 선수를 교체하는지, 파서 보정(미기록 진루 밀어내기·빈 루 진루) 횟수 |")
    A("| 박스 기록 | 팀 안타·홈런·볼넷(고의4구 포함)·고의4구·삼진·사구·도루·도루자 vs 게임로그 |")
    A("| 투구 시퀀스 | 볼넷=4볼, 삼진=3스트라이크, 2스트라이크 파울은 카운트 유지, 2스트라이크 번트 파울은 삼진, "
      "중간에 타석이 끝났어야 하는 시퀀스 없음, 인플레이 결과는 X/Y로 끝남, 카운트 필드와 일치 |")
    A(f"| Chadwick 교차검증 | {v['chadwick']['status']} |\n")
    A("### 핵심 발견: 최종 점수는 맞지만 중간 아웃이 틀린 경기\n")
    A("처음 검증에서 포함 경기 중 라인스코어 불일치 214경기, 3아웃 위반 351경기가 나왔다. 원인은 하나의 표기 규칙이었다. "
      "`BX3(35)(E1/TH1)`처럼 아웃을 만든 수비 그룹과 실책 그룹이 함께 있는 진루 표기를 \"마지막 그룹에 실책이 있으면 세이프\"로 "
      "해석한 것이다. 실제로는 실책으로 추가 진루를 시도하다 **결국 아웃**된 것이다. 규칙을 \"실책 없는 수비 그룹이 하나라도 "
      "있으면 아웃\"으로 고치자(`retro_parse._error_negates`)")
    A("- 이 표기를 포함한 **354개 플레이(354경기)**의 아웃 수가 바뀌었다(1900년대–2020년대 전 시대에 분포).")
    A("- B0·B1 때 '점수 불일치'로 제외했던 216경기가 **모두** 게임로그와 일치하게 되었다.")
    A("- 포함 경기의 라인스코어 불일치가 214경기에서 0경기로, 3아웃 위반이 351경기에서 2경기로 줄었다. 즉 이전 B0 데이터에는 "
      "최종 점수는 맞지만 경기 중간 아웃 상태가 틀린 경기가 수백 건 섞여 있었다. 리뷰에서 우려한 바로 그 유형이다. "
      "B0·B1 결과는 수정된 데이터로 다시 계산했다(`reports/b0_b1_report.md`).\n")
    A("### 수정 후 결과 (포함 경기 기준, 위반 경기 비율)\n")
    A("| 시대 | 경기 | 위반 경기 | 비율 |\n|---|---:|---:|---:|")
    for e in v["by_era"]:
        A(f"| {e['era']} | {e['games']:,} | {e['flagged']:,} | {e['rate']*100:.3f}% |")
    A("\n사유별 위반 경기 수(기존 사유가 없는 경기 중):\n")
    A("| 사유 | 경기 |\n|---|---:|")
    for k, n in v["validation_flags_among_previously_included"].items():
        A(f"| {k} | {n} |")
    A("\n- 중단 기준(어느 시대든 0.5% 초과)에 걸리지 않았고, 남은 위반은 특정 표기 패턴에 몰려 있지 않다. 타순 끊김에는 실제 "
      "타순 착오(`com` 기록 \"batted out of turn\")도 포함된다. 위반 경기는 모두 제외했다.")
    bx = v["box_count_mismatch_rate_2000plus"]
    A("- 박스 기록 불일치율(2000년 이후, 한 팀이라도 다르면 불일치): " + ", ".join(f"{k} {x*100:.3f}%" for k, x in bx.items()) + ".")
    A("- 잔루는 끝내기 하프이닝에 남은 주자까지 세야 게임로그와 맞는다(포함 시 불일치 4경기, 제외 시 약 6%).\n")
    A("### 시즌별 투구 시퀀스 완전 비율과 규칙 위반율\n")
    ps = pd.DataFrame(v.get("pitch_sequences_by_season", []))
    if len(ps):
        A("완전 = 타격 결과가 있는 타석 중 투구 시퀀스가 비어 있지 않고 `?`·`U`(미상)가 없는 비율. "
          "위반율 = 완전한 시퀀스 중 위 규칙 위반 비율.\n")
        A("| 시즌 | 타석 | 완전 비율 | 규칙 위반 | 위반율 |\n|---|---:|---:|---:|---:|")
        for _, x in ps.iterrows():
            if x["season"] >= 1950 and (x["season"] % 5 == 0 or x["season"] >= 1985):
                A(f"| {int(x['season'])} | {int(x['batting_pa']):,} | {x['complete_frac']*100:.1f}% | "
                  f"{int(x['pitch_rule_violations'])} | {x['violation_rate']*1e6:.1f} ppm |")
        A("\n(1908–1949는 0–0.5%로 생략, 1950–1987은 표의 5년 간격만 표시)\n")
    A("**카운트 체인 학습 시즌 기준: 완전 비율 99% 이상.** 1988–1998은 89–100%로 시즌 안에서도 일부 경기·팀에 누락이 몰려 "
      "있을 수 있어 제외했다. 2000년 이후는 모두 100%다. 실제로 선택된 학습 창은 모두 2000년 이후라 이 기준에 걸리는 "
      "시즌은 없었다.\n")

    # ---------------- 1 ----------------
    A("## 1. 데이터 범위와 학습 기간 K\n")
    A("K는 {3, 5, 10, 20} 중에서 고른다. 평가 시즌 S에 대해 S−1 직전 K시즌(2020 제외)으로 B2를 풀고, S−1 시즌의 "
      "타석 시작 로그 손실로 선택한다. 선택에 쓰는 데이터는 모두 S 이전이다.\n")
    A("| 평가 시즌 | K=3 | K=5 | K=10 | K=20 | 선택 K | 학습 시즌 |\n|---|---:|---:|---:|---:|---:|---|")
    for s, x in r["per_season"].items():
        g = x["K_grid_logloss_on_prev"]
        A(f"| {s} | " + " | ".join(f"{g.get(str(k), float('nan')):.5f}" for k in (3, 5, 10, 20)) +
          f" | **{x['K']}** | {int(x['train_seasons'][0])}–{int(x['train_seasons'][-1])} ({len(x['train_seasons'])}시즌) |")
    spread = max(max(x["K_grid_logloss_on_prev"].values()) - min(x["K_grid_logloss_on_prev"].values())
                 for x in r["per_season"].values())
    A(f"\n근거: 후보 간 손실 차이는 최대 {spread:.5f}로 작아서 K의 영향은 크지 않다. 선택된 K는 "
      + ", ".join(str(x["K"]) for x in r["per_season"].values())
      + "이고, 데드볼 시대는 어느 후보 창에도 들어가지 않는다(가장 긴 창도 2001년 이후). B0처럼 1908년부터 섞는 "
        "선택지는 후보에 없다.\n")

    # ---------------- 2 ----------------
    A("## 2. 타석 결과와 진루\n")
    A(f"- 결과 범주: K, BB, IBB, HBP, 1B, 2B, 3B, HR, OUT(필더스 초이스 포함), ROE(실책 출루, 포수 방해 포함), "
      f"**INT**(첫 투구 이후 주자 이벤트로 3아웃이 되거나 끝내기 득점이 나서 타석이 끝나지 않은 경우, 전체 타석의 "
      f"{r['share_pa_INT']*100:.2f}%). 정확한 DP를 위해 INT를 추가했다. INT는 타순을 넘기지 않는다.")
    A("- q(o | 공격 측, 아웃, 주자): 홈·원정 공격을 나눠 추정한다. 진루 테이블 A(아웃·주자·득점 | o, 아웃, 주자)에는 병살, "
      "희생플라이, 추가 진루가 경험적으로 들어 있다. 평활은 같은 결과·주자에서 아웃 수를 상대값으로 묶은 사전분포에 "
      "의사관측 5개를 쓴다.")
    A(f"- **타석 간 전이 N**: 첫 투구 전에 일어난 주자 이벤트(견제사, 보크, 투구 전 도루 등)로 정의했다. 투구 시퀀스로 "
      f"판정하며 전체 타석의 {r['share_pa_with_N_event']*100:.2f}%에서 일어난다. 투구 도중 일어난 주자 이벤트"
      f"({r['share_pa_with_midpa_runner_event']*100:.1f}%의 타석)는 결과 o에 조건부로 A에 흡수했다(지시된 단순화).")
    A("- N을 넣은 버전과 뺀 버전의 차이, 그리고 분해 버전과 직접 전이(타석 시작 → 다음 타석 시작) 대조군의 차이"
      "(|점수차| ≤ 10 칸):\n")
    A("| 평가 시즌 | N 효과 평균 | N 효과 최대 | 직접−분해 평균 | 직접−분해 p99 | 직접−분해 최대 |\n|---|---:|---:|---:|---:|---:|")
    for s, x in r["per_season"].items():
        n, d = x["N_effect_abs_diff(|d|<=10)"], x["direct_vs_decomposed_abs_diff(|d|<=10)"]
        A(f"| {s} | {n['mean']:.2e} | {n['max']:.2e} | {d['mean']:.2e} | {d['p99']:.2e} | {d['max']:.2e} |")
    nmax = max(x["N_effect_abs_diff(|d|<=10)"]["max"] for x in r["per_season"].values())
    dmean = [x["direct_vs_decomposed_abs_diff(|d|<=10)"]["mean"] for x in r["per_season"].values()]
    dmax = max(x["direct_vs_decomposed_abs_diff(|d|<=10)"]["max"] for x in r["per_season"].values())
    A(f"\nN의 영향은 WP 기준 최대 {nmax:.1e}로 작다. 첫 투구 전에 일어나는 주자 이벤트가 드물기 때문이다. 직접 전이와 "
      f"분해 버전의 차이는 칸 평균 {min(dmean):.1e}–{max(dmean):.1e}, 최대 {dmax:.1e}다. 타석 시작 평가 손실도 같은 수준이라"
      "(5.1절 변형 표) 분해(q·A)가 정보를 잃지 않는다. 차이는 \"결과가 같으면 진루도 같다\"는 분해 가정과 희소 상태의 "
      "평활 차이에서 온다.\n")

    # ---------------- 3 ----------------
    A("## 3. 정확 DP 엔진\n")
    A(f"- 점수차 범위: ±{r['R']}. 수렴 검사: " + "; ".join(
        f"±{h['R']} → ±{h['vs_R']} 최대 변화 {h['max_abs_change']:.2e}" for h in r["R_convergence"]) + " (기준 1e-6).")
    A("- 연장은 2루 주자 규칙으로 처리한다. 동점 연장 시작값 X는 `X = C + P·X` 선형방정식으로 정확히 푼다. 테스트에서 50이닝을 "
      "펼친 값과 1e-9 이내로 일치했다.")
    A("- 타순 축: S=9 엔진과 S=1(축 접음) 엔진의 결과 차이 = " +
      ", ".join(f"{s}: {x['dp_S9_max_abs_vs_S1']:.1e}" for s, x in r["per_season"].items()) + ".")
    A("\n| 평가 시즌 | DP S=1 (초) | 피크 메모리 | DP S=9 (초) | 피크 메모리 | 연장 동점 X | 1회초 시작 WP |\n|---|---:|---:|---:|---:|---:|---:|")
    for s, x in r["per_season"].items():
        A(f"| {s} | {x['dp_seconds']:.2f} | {x['dp_peak_MB']:.1f} MB | {x['dp_S9_seconds']:.2f} | {x['dp_S9_peak_MB']:.0f} MB | "
          f"{x['X_extra_tie_value']:.4f} | {x['start_wp']:.4f} |")
    A(f"\n산출물: `models/wp_b2_pa.parquet` {r['table_rows']['wp_b2_pa']:,}행({r['table_bytes']['wp_b2_pa']/1e3:.0f} KB, "
      f"타석 시작 WP와 첫 투구 시점 WP), 학습 창 {r['final_window'][0]}–{r['final_window'][-1]}.\n")

    # ---------------- 4 ----------------
    A("## 4. 카운트 체인과 정합 보정\n")
    A("- 12개 카운트. 투구 결과는 볼, 루킹 스트라이크, 헛스윙(파울팁 포함), 파울, 번트 파울, 몸에 맞는 공, 인플레이"
      "(1B/2B/3B/HR/OUT/ROE)로 나눈다. 드물게 IBB(첫 `I`가 나온 카운트에서 결정)와 INT도 종료 사건으로 둔다.")
    A("- `p(o | 카운트) = f_o(층) · m(o, 카운트) / Z`. m은 리그 카운트별 투구 결과율이고, f는 공격 측 × 아웃 × 주자"
      " 48개 층별로 IPF로 맞춘다.")
    A("\n| 평가 시즌 | IPF 반복 | 최대 오차 | 0-0 투구 WP − DP 첫 투구 WP | 체인 학습 타석 | 불일치 제외 |\n|---|---:|---:|---:|---:|---:|")
    for s, x in r["per_season"].items():
        A(f"| {s} | {x['ipf_iterations']} | {x['ipf_max_abs_err']:.1e} | {x['max_abs_pitch00_minus_W']:.1e} | "
          f"{x['count_chain_pa_used']:,} | {x['count_chain_pa_skipped_inconsistent']:,} |")
    A("\n- 정합의 직접 확인: 0-0 시점 투구 WP는 첫 투구 시점 DP 값 W(s)와 1e-9 이내로 같다(테스트). N을 뺀 버전에서는 "
      "타석 시작 DP 값 V(s)와도 같다. N을 넣으면 V(s) = Σ N(s'|s)·W(s')이므로, 투구 시점 WP는 N이 끝난 상태의 값이 된다.")
    A(f"- `models/wp_b2_pitch.parquet` {r['table_rows']['wp_b2_pitch']:,}행({r['table_bytes']['wp_b2_pitch']/1e6:.1f} MB), "
      f"조회 함수 `wp.wp_b2.PitchWP` 1회 {r['lookup_us_per_call']:.1f} µs.")
    A("- 단순화: 타석 도중 주자 이벤트는 투구 단위로 모델링하지 않고 결과별 진루 테이블 A에 흡수했다. 투구 시점 평가에서는 "
      "실제 주자 이벤트가 반영된 현재 상태로 조회한다.\n")

    # ---------------- 5 ----------------
    A("## 5. 검증\n")
    A("### 5.1 타석 시작 시점: B0·B1·B2 (로그 손실, 95% CI, 경기 단위 블록 부트스트랩 1,000회)\n")
    for s, x in r["per_season"].items():
        A(f"**{s}** — 상태 {x['n_states']:,} / {x['n_games']:,}경기, K={x['K']}\n")
        A(table(merged(x["pa_metrics_vs_B0"], x["pa_metrics_vs_B1"]), "logloss", ["B0", "B1", "B2"],
                ["B2-B0", "B2-B1"]))
        A("")
    A("**3시즌 합산 — 로그 손실**\n")
    A(table(merged(r["pooled_pa_vs_B0"], r["pooled_pa_vs_B1"]), "logloss", ["B0", "B1", "B2"], ["B2-B0", "B2-B1"]))
    A("\n**3시즌 합산 — Brier**\n")
    A(table(merged(r["pooled_pa_vs_B0"], r["pooled_pa_vs_B1"]), "brier", ["B0", "B1", "B2"], ["B2-B0", "B2-B1"]))
    A("\n**3시즌 합산 — B2 변형 (N 제외, 직접 전이), 로그 손실**\n")
    A(table(r["pooled_pa_vs_B0"], "logloss", ["B2", "B2_noN", "B2_direct"], ["B2_noN-B0", "B2_direct-B0"]))
    A("")
    pooled = pd.DataFrame(merged(r["pooled_pa_vs_B0"], r["pooled_pa_vs_B1"]))
    def g(m, s="all", met="logloss"):
        return pooled[(pooled.model == m) & (pooled.stratum == s) & (pooled.metric == met)].iloc[0]
    d0, d1, dx = g("B2-B0"), g("B2-B1"), g("B2-B0", "extra(inning>=10)")
    verdict0 = ("CI가 0을 포함한다. 즉 **B0 수준의 정확도를 재현한다**" if d0["ci_low"] <= 0 <= d0["ci_high"]
                else ("CI가 0보다 위라 B0보다 약간 못하다" if d0["ci_low"] > 0 else "CI가 0보다 아래라 B0보다 낫다"))
    A(f"해석: 전체 B2−B0 = {ci(d0)}로, {verdict0}. 층별로도 정규 이닝에서는 차이가 0 근처다. 연장(10회 이상)에서는 "
      f"B2−B0 = {ci(dx)}"
      + ("로 B2가 낫다. 2루 주자 시대의 연장 표본이 적어서, 상태 테이블(B0)보다 구조적으로 푸는 DP가 유리하다."
         if dx["ci_high"] < 0 else ".")
      + f" B2−B1 = {ci(d1)}로 B1보다 못한데, 전력 정보가 없는 리그 평균 엔진이므로 예상된 결과다. 실패가 아니다.\n")
    A("### 5.2 캘리브레이션 (타석 시작, 10구간)\n")
    for s, x in r["per_season"].items():
        cal = {m: pd.DataFrame(x["pa_calibration"][m]).set_index("bin") for m in ("B0", "B1", "B2")}
        A(f"**{s}**\n")
        A("| 구간 | B0 예측/실제 | B1 예측/실제 | B2 예측/실제 |\n|---|---|---|---|")
        for b in cal["B0"].index:
            cells = []
            for m in ("B0", "B1", "B2"):
                row = cal[m].loc[b]
                cells.append("–" if pd.isna(row["mean_pred"]) else f"{row['mean_pred']:.3f} / {row['obs_rate']:.3f}")
            A(f"| {b} | " + " | ".join(cells) + " |")
        A("")
    A("### 5.3 투구 시점 평가 (모든 투구 직전, 로그 손실)\n")
    A("B2는 카운트를 반영한 값이고, B2(카운트 무시)는 같은 엔진의 첫 투구 시점 값 W(s)다. 둘의 차이가 **카운트 정보의 "
      "순수 이득**이다. B0는 카운트를 무시한다.\n")
    for s, x in r["per_season"].items():
        A(f"**{s}** — 투구 {x['n_pitches']:,}개\n")
        rows = list(x["pitch_metrics"]) + [y for y in x["pitch_count_gain"] if y["model"] == "B2-B2_firstpitch_nocount"]
        A(table(rows, "logloss", ["B0", "B2_firstpitch_nocount", "B2"], ["B2-B0", "B2-B2_firstpitch_nocount"]))
        A("")
    A("**3시즌 합산**\n")
    rows = list(r["pooled_pitch"]) + [y for y in r["pooled_pitch_count_gain"] if y["model"] == "B2-B2_firstpitch_nocount"]
    A(table(rows, "logloss", ["B0", "B2_firstpitch_nocount", "B2"], ["B2-B0", "B2-B2_firstpitch_nocount"]))
    A("\n**Brier**\n")
    A(table(rows, "brier", ["B0", "B2_firstpitch_nocount", "B2"], ["B2-B0", "B2-B2_firstpitch_nocount"]))
    pp = pd.DataFrame(rows)
    cg = pp[(pp.model == "B2-B2_firstpitch_nocount") & (pp.stratum == "all") & (pp.metric == "logloss")].iloc[0]
    A(f"\n카운트 이득(전체) = {ci(cg)}. " + ("CI가 0보다 완전히 아래이므로 **카운트 정보는 실제로 이득을 준다**."
                                              if cg["ci_high"] < 0 else "CI가 0을 포함해 이득이 유의하지 않다.") + "\n")

    # ---------------- B0 comparison ----------------
    A("### 5.4 B0와의 이론 대조 (B0 표본 1,000개 이상 칸)\n")
    A("| 평가 시즌 | 칸 수 | 평균 B2−B0 | |차이| 중앙값 | p90 | p99 | 최대 | 분류 |\n|---|---:|---:|---:|---:|---:|---:|---|")
    for s, x in r["per_season"].items():
        c = x["b0_compare"]
        q = c["abs_delta_quantiles"]
        A(f"| {s} | {c['cells']:,} | {c['mean_delta']:+.4f} | {q['0.5']:.4f} | {q['0.9']:.4f} | {q['0.99']:.4f} | "
          f"{q['1.0']:.4f} | " + ", ".join(f"{k} {n}" for k, n in c["class_counts"].items()) + " |")
    A("\n분류 규칙(|B2−B0| > 0.02인 칸): 같은 학습 창의 B0(B0_recent)가 B2에 더 가까우면 **시대 차이**, N을 뺀 B2와의 차이가 "
      "크면 **비타석 이벤트**, B0_recent와의 차이가 표본오차(2σ, 경기 내 상관을 고려해 분산 ×5) 안이면 **표본 잡음**, "
      "그 밖은 **마르코프 가정·구조**(전력 차이가 있는 실제 경기에서 상태가 전력의 신호가 되는 선택 효과 등).\n")
    tot = {}
    for x in r["per_season"].values():
        for k, n in x["b0_compare"]["class_counts"].items():
            tot[k] = tot.get(k, 0) + n
    big = sum(n for k, n in tot.items() if k != "small")
    A(f"3시즌 합계로 |B2−B0| > 0.02인 칸 {big}개 중 표본 잡음 {tot.get('sampling_noise', 0)}개, 시대 차이 "
      f"{tot.get('era_difference', 0)}개, 비타석 이벤트 {tot.get('non_batting_events', 0)}개, 마르코프·구조 "
      f"{tot.get('markov_or_structure', 0)}개다. 큰 차이는 대부분 B0 칸의 표본 잡음이나, 1908년부터 섞은 B0와 최근 창으로 "
      "학습한 B2의 득점 환경 차이로 설명된다.\n")
    last = list(r["per_season"])[-1]
    top = pd.DataFrame(r["per_season"][last]["b0_compare"]["top"])
    A(f"**{last} 차이 상위 칸**\n")
    A("| 이닝 | 초/말 | 아웃 | 주자 | 점수차 | B0 n | B0(전체) | B0(같은 창) | B2 | 분류 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, t in top.iterrows():
        A(f"| {int(t['inning'])} | {'말' if t['half'] else '초'} | {int(t['outs'])} | {int(t['bases']):03b} | {int(t['diff'])} | "
          f"{int(t['n']):,} | {t['wp']:.3f} | {t['B0_recent']:.3f} | {t['B2']:.3f} | {t['class']} |")
    A("")

    # ---------------- timings ----------------
    A("## 6. 계산 시간과 메모리\n")
    A("| 단계 | 값 |\n|---|---:|")
    for k, x in r["timings"].items():
        A(f"| {k} | {x} s |")
    for s, x in r["per_season"].items():
        A(f"| {s}: K 선택(DP 4회) | {x['K_select_s']} s |")
        A(f"| {s}: 전이 추정 | {x['transition_fit_s']} s |")
        A(f"| {s}: 카운트 체인(기준율+IPF+투구 테이블) | {x['count_chain_s']} s |")
        A(f"| {s}: 부트스트랩(타석+투구) | {x['bootstrap_s']} s |")
    A(f"| 조회 1회 (`PitchWP.wp`) | {r['lookup_us_per_call']:.1f} µs |")
    A("")
    A("## 7. B3 인터페이스\n")
    A("`docs/B3_INTERFACE.md` 참조: 타순별 q(`compose_kernel`의 슬롯 축), 하프이닝별 커널 콜백(`Engine.solve(kernel_for)`, "
      "불펜·투수 품질), 매치업별 카운트 체인 재보정(`count_chain.fit`).\n")
    A("## 출처 고지\n")
    A("     The information used here was obtained free of charge from and is copyrighted by Retrosheet.\n"
      "     Interested parties may contact Retrosheet at \"www.retrosheet.org\".")
    (ROOT / "reports/b2_report.md").write_text("\n".join(L) + "\n")
    print("wrote reports/b2_report.md")


if __name__ == "__main__":
    main()

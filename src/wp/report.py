"""Render reports/b0_b1_report.md from the pipeline outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REASON_KO = {
    "tie_or_called_early": "무승부·조기 종료(콜드 등, 9이닝 미완료)",
    "season_2020": "2020 단축 시즌",
    "score_mismatch_vs_gamelog": "파싱 최종 점수 ≠ 게임로그 (검증 실패)",
    "suspended_or_completed_later": "서스펜디드(후일 재개 완료)",
    "scheduled_7_innings": "7이닝 더블헤더 (2021; 2020분은 시즌 제외에 포함)",
    "forfeit": "몰수 경기",
    "non_regular_gametype": "정규시즌 외(순위결정전 등)",
    "home_bats_first": "홈팀 선공 경기",
    "parse_error": "파서 오류",
    "not_in_gamelog": "게임로그에 없음",
    "validation:pitch_sequence_violation": "0절 검증: 투구 시퀀스 위반",
    "validation:batting_order_break": "0절 검증: 타순 연속성 끊김",
    "validation:three_out_violation": "0절 검증: 3아웃 위반",
    "validation:lob_mismatch": "0절 검증: 잔루 불일치",
    "validation:linescore_mismatch": "0절 검증: 이닝별 득점 불일치",
    "validation:parser_repair": "0절 검증: 파서 보정 발생",
}
MODEL_KO = {"B0": "B0", "B1": "B1", "Elo_only": "Elo만", "B0_mono": "B0(단조 투영)",
            "B0_recent": "B0(최근 30시즌만)"}


def fmt_ci(r, digits=4):
    return f"{r['estimate']:.{digits}f} [{r['ci_low']:.{digits}f}, {r['ci_high']:.{digits}f}]"


def metrics_table(rows: list[dict], metric: str, models=("B0", "B1", "Elo_only")) -> str:
    df = pd.DataFrame(rows)
    df = df[df["metric"] == metric]
    out = ["| 층 | 상태 수 | 경기 수 | " + " | ".join(MODEL_KO[m] for m in models) + " | B1−B0 (95% CI) | P(B1<B0) |",
           "|---|---:|---:|" + "---|" * len(models) + "---|---:|"]
    for s, g in df.groupby("stratum", sort=False):
        g = g.set_index("model")
        cells = [fmt_ci(g.loc[m]) for m in models]
        d = g.loc["B1-B0"]
        out.append(f"| {s} | {int(g.loc['B0','n_states']):,} | {int(g.loc['B0','n_games']):,} | "
                   + " | ".join(cells) + f" | {fmt_ci(d)} | {d['p_better']:.3f} |")
    return "\n".join(out)


def variant_table(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    df = df[df["metric"] == "logloss"]
    out = ["| 층 | B0(단조 투영)−B0 | B0(최근 30시즌)−B0 |", "|---|---|---|"]
    for s, g in df.groupby("stratum", sort=False):
        g = g.set_index("model")
        out.append(f"| {s} | {fmt_ci(g.loc['B0_mono-B0'], 5)} | {fmt_ci(g.loc['B0_recent-B0'], 5)} |")
    return "\n".join(out)


def calib_table(cal: dict, season: str) -> str:
    b0 = pd.DataFrame(cal[season]["B0"]).set_index("bin")
    b1 = pd.DataFrame(cal[season]["B1"]).set_index("bin")
    out = ["| 구간 | B0 n | B0 예측평균 | B0 실제비율 | B1 n | B1 예측평균 | B1 실제비율 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for b in b0.index:
        r0, r1 = b0.loc[b], b1.loc[b]
        f = lambda v, d=3: "–" if pd.isna(v) else f"{v:.{d}f}"
        out.append(f"| {b} | {f(r0['n'],0)} | {f(r0['mean_pred'])} | {f(r0['obs_rate'])} | "
                   f"{f(r1['n'],0)} | {f(r1['mean_pred'])} | {f(r1['obs_rate'])} |")
    return "\n".join(out)


def main() -> None:
    from .manifest import check_fresh
    for f in ("reports/results.json", "data/processed/build_summary.json",
              "data/processed/validation_summary.json", "data/raw/download_manifest.json"):
        check_fresh(ROOT / f)
    res = json.loads((ROOT / "reports/results.json").read_text())
    build = json.loads((ROOT / "data/processed/build_summary.json").read_text())
    vs = json.loads((ROOT / "data/processed/validation_summary.json").read_text())
    build["excluded_by_reason"] = vs["excluded_by_reason_after_validation"]
    build["games_included"] = vs["games_included_after_validation"]
    manifest = json.loads((ROOT / "data/raw/download_manifest.json").read_text())
    ft = res["final_table"]
    L = []
    A = L.append
    A("# B0·B1 승리확률 기준선 리포트\n")
    A("> 자동 생성: `PYTHONPATH=src python -m wp.report` (입력: `reports/results.json`). "
      "수치는 모두 실제 실행 측정값이다.\n")

    A("## 1. 요약\n")
    pooled = pd.DataFrame(res["pooled_metrics"])
    def g(model, stratum, metric="logloss"):
        return pooled[(pooled.model == model) & (pooled.stratum == stratum) & (pooled.metric == metric)].iloc[0]
    d_all, d_late = g("B1-B0", "all"), g("B1-B0", "inning>=7 & close")
    A(f"- 평가 시즌 {', '.join(res['per_season'])} (워크포워드), 평가 상태 {int(g('B0','all')['n_states']):,}개 / "
      f"{int(g('B0','all')['n_games']):,}경기.")
    sig = d_all["ci_high"] < 0
    A(f"- **B1 대 B0**: 3시즌 합산 로그 손실 차이 B1−B0 = {fmt_ci(d_all)} (합산 표본을 경기 단위로 1,000회 재표집한 "
      f"95% CI) → " + ("CI가 0보다 완전히 아래이므로 **B1이 B0보다 낫다**." if sig else "CI가 0을 포함해 차이가 유의하지 않다."))
    d_close, d_l7 = g("B1-B0", "close(|diff|<=2)"), g("B1-B0", "inning>=7")
    A(f"- 층별 B1−B0: 접전 {fmt_ci(d_close)}, 7회 이후 {fmt_ci(d_l7)}, 7회 이후 접전 {fmt_ci(d_late)}. "
      "개선은 경기 초·중반에 크고 후반으로 갈수록 작아진다 — 후반에는 점수·아웃 상태가 대부분의 정보를 갖기 때문이다"
      + (" (7회 이후 접전은 CI가 0을 포함)." if d_late["ci_high"] >= 0 else "."))
    coefs = [v["b1_coef"] for v in res["per_season"].values()]
    A(f"- B1 계수 `d`가 모든 평가 시즌에서 음수({', '.join(f'{c['d']:.2f}' for c in coefs)}) → 진행도가 커질수록 "
      f"Elo의 영향이 줄어든다. `c`≈{coefs[-1]['c']:.2f}>1은 Elo 확률이 실제보다 0.5 쪽으로 눌려 있어 B1이 경기 전 "
      "전력차를 조금 늘려 쓴다는 뜻이다.")
    A(f"- 학습 데이터를 1908년까지 최대로 쓴 B0가 최근 30시즌만 쓴 B0보다 약간 낫다(§6).\n")

    A("## 2. 데이터\n")
    A("- **출처**: Retrosheet 이벤트 파일·게임로그. retrosheet.org는 작업 환경에서 접속이 차단되어, Chadwick Baseball "
      "Bureau가 관리하는 GitHub 미러 `chadwickbureau/retrosheet`(master 브랜치)에서 받았다. 상세는 `docs/DATA_SOURCES.md`.")
    A("- **파싱**: Retrosheet가 제공하는 파싱된 CSV는 사이트 차단으로 확인하지 못했고, Chadwick(C) 빌드는 환경 정책상 "
      "외부 코드 컴파일이 막혀 사용하지 못했다. 그래서 이벤트 파일 문법(retrosheet.org/eventfile.htm)을 따라 "
      "순수 파이썬 파서(`src/wp/retro_parse.py`)를 작성했고, **모든 경기의 파싱 최종 점수를 게임로그와 대조**해 검증했다.")
    tot_mb = manifest["total_bytes"] / 1e6
    A(f"- 다운로드: 1901–2025 게임로그, 이벤트 파일 1908–2025 (1901–1907은 플레이 기록 없음, 게임로그만 Elo 워밍업에 사용). "
      f"원본 {tot_mb:,.0f} MB, 다운로드 {manifest['seconds']} 초.")
    A(f"- 이벤트 파일 경기 {build['games_in_event_files']:,}개, 타석 시작 상태 {build['pa_states_total']:,}개, "
      f"파싱+가공 {build['seconds']} 초 (4코어).")
    A(f"- 분석 포함 경기 {build['games_included']:,}개, 포함 상태 {res['n_states_included']:,}개.\n")
    A("### 제외 경기\n")
    A("| 사유 | 경기 수 |\n|---|---:|")
    for k, v in build["excluded_by_reason"].items():
        A(f"| {REASON_KO.get(k, k)} | {v:,} |")
    A(f"| **합계** | **{sum(build['excluded_by_reason'].values()):,}** |\n")
    A("사유는 하나만 붙인다(예: 2020년 7이닝 더블헤더는 '2020 단축 시즌'으로 집계). 0절 검증 사유(라인스코어·3아웃·잔루·"
      "타순 연속성·투구 시퀀스)는 `src/wp/validate.py`가 기존 사유가 없는 경기에만 붙인다. 이전 버전의 '파싱 점수 ≠ 게임로그' "
      "216경기는 `rX#(fielders)(E…)` 표기 해석을 고친 뒤 모두 일치하게 되어 사라졌다(상세는 `reports/b2_report.md` 0절).\n")

    A("## 3. 설계\n")
    A("**상태 키** (`src/wp/states.py`): 이닝(1–9, 10+), 초/말, 아웃(0–2), 주자(3비트), 점수차(홈−원정, ±10 절단), "
      "연장 규칙 시대 `xrule`(10+ 이닝에서만). 총 키 공간 11,088칸, 그중 가능한 상태 "
      f"{ft['possible_rows']:,}칸.\n")
    A("**연장 처리 — 규칙 시대별 분리 집계.** 이벤트 파일의 `info,tiebreaker,2`(연장 2루 주자) 필드가 2020–2025 모든 "
      "경기에 있고 2019년 이전에는 없음을 데이터로 확인했다. 10+ 이닝 칸만 `xrule`로 나누고 정규 이닝은 전 시대를 공유한다. "
      "따라서 2023 평가 모델의 연장 칸(xrule=1)은 2021–2022 시즌만으로 학습되며(2020 제외), 표본이 적은 만큼 "
      "아래의 축소가 같은 시대의 더 거친 칸(이닝·점수차)으로 끌어당긴다. 이유: 시대 분리는 2루 주자 규칙의 큰 효과를 "
      "정확히 반영하면서 정규 이닝의 방대한 과거 표본을 버리지 않는다.\n")
    A("**평활 — 계층적 베이지안 축소.** `wp_L = (승_L + k·wp_부모)/(n_L + k)`를 거친 단계부터 적용: "
      "(초/말) → (이닝, 초/말, 점수차 구간) → (…, 점수차) → (…, 아웃, 주자 수, 점수차) → 전체 상태. 표본이 많은 칸은 "
      "경험 비율을 유지하고, 적은 칸은 주자 상태 단순화·점수차 구간 확대 쪽으로 백오프된다. `k`는 평가 시즌 S에 대해 "
      "S−2 이전으로 학습하고 S−1에서 로그 손실로 선택(그리드 1–1000). 테이블에 칸별 `n`, `wins`, 원 비율 `p_raw`, "
      "부모 단계의 `n_L0..L2`, `wp_L0..L2`를 함께 저장했다.\n")
    A(f"**Elo** (`src/wp/elo.py`): 게임로그 1901–2025 전 정규시즌 경기를 날짜·더블헤더 순서대로 갱신(경기 전 확률을 먼저 기록한 뒤 "
      f"결과 반영), 홈 어드밴티지와 시즌 간 1500으로의 평균 회귀 포함, 연고 이전은 프랜차이즈로 연결. 하이퍼파라미터는 "
      f"{res['elo_tune_seasons'][0]}–{res['elo_tune_seasons'][1]} 시즌 로그 손실로 선택(모든 평가 시즌 이전): "
      f"K={res['elo_params']['k']}, HFA={res['elo_params']['hfa']}, 회귀={res['elo_params']['revert']}.\n")
    A("**B1** (`src/wp/b1_model.py`): `logit(WP_B1) = a + b·logit(WP_B0) + c·logit(P_pre) + d·logit(P_pre)·진행도`, "
      "진행도 = 1 − 남은 정규 아웃/54. 평가 시즌 S의 B1은 S 직전 5개 포함 시즌으로 학습하며, 학습 행의 WP_B0도 "
      "**그 행의 시즌 이전 데이터로만 만든 테이블**에서 가져온다(표본 내 B0 과적합이 계수 b를 부풀리는 것을 방지).\n")

    A("## 4. 시즌별 결과 (워크포워드)\n")
    for s, rec in res["per_season"].items():
        c = rec["b1_coef"]
        A(f"### 평가 시즌 {s}\n")
        A(f"- B0 학습 {rec['b0_train_seasons'][0]}–{rec['b0_train_seasons'][1]}, k={rec['b0_k']}; "
          f"B1 학습 시즌 {rec['b1_train_seasons']}; 평가 상태 {rec['n_states']:,} / {rec['n_games']:,}경기 "
          f"(연장 2루 주자 시대 상태 {rec['xrule_cells_used']:,}).")
        A(f"- B1 계수: a={c['a']:.3f}, b={c['b']:.3f}, c={c['c']:.3f}, d={c['d']:.3f} "
          f"(1회초 Elo 가중 c={c['c']:.2f} → 9회말 c+d={c['c']+c['d']:.2f}).")
        pc = rec["pregame_check"]
        A(f"- 경기 전 점검(1회초 시작 상태, {pc['n']}경기): |B1−Elo| 평균 {pc['mean_abs_diff_B1_vs_Elo']:.4f}, "
          f"최대 {pc['max_abs_diff_B1_vs_Elo']:.4f}, 상관 {pc['corr']:.5f}; 평균 B1 {pc['mean_B1']:.3f} vs Elo {pc['mean_Elo']:.3f} "
          f"(B0는 이 상태에서 {pc['B0_at_start']:.3f}로 상수). 평균 "
          f"{pc['mean_abs_diff_B1_vs_Elo']*100:.1f}%p 차이로 크게 어긋나지 않으며, "
          "차이는 주로 c>1에 따른 전력차 확대에서 온다.\n")
        A("**로그 손실** (추정 [95% CI], 경기 단위 블록 부트스트랩 1,000회)\n")
        A(metrics_table(rec["metrics"], "logloss"))
        A("\n**Brier score**\n")
        A(metrics_table(rec["metrics"], "brier"))
        A("")
    A("### 3시즌 합산\n")
    A("**로그 손실**\n")
    A(metrics_table(res["pooled_metrics"], "logloss"))
    A("\n**Brier score**\n")
    A(metrics_table(res["pooled_metrics"], "brier"))
    A("\n`Elo만`은 경기 전 확률을 경기 내내 고정한 참고값이다. P(B1<B0)는 부트스트랩 표본 중 B1 손실이 더 작은 비율.\n")

    A("## 5. 캘리브레이션\n")
    A("![calibration](figures/calibration_b0_b1.png)\n")
    for s in res["calibration"]:
        gaps = []
        for m in ("B0", "B1"):
            c = pd.DataFrame(res["calibration"][s][m]).dropna()
            c["gap"] = c["obs_rate"] - c["mean_pred"]
            w = c.loc[c["gap"].abs().idxmax()]
            gaps.append(f"{m} 최대 편차 {w['gap']:+.3f} ({w['bin']} 구간)")
        A(f"**{s}** (10개 등간격 구간, 예측 P(홈 승)) — " + ", ".join(gaps) + "\n")
        A(calib_table(res["calibration"], s))
        A("")

    A("## 6. 민감도 분석 (헤드라인 아님)\n")
    A("합산 로그 손실 차이 (양수 = B0보다 나쁨):\n")
    A(variant_table(res["pooled_metrics"]))
    A("\n- **최근 30시즌만** 쓴 B0는 전체(1908–) B0보다 일관되게 조금 나쁘다 → 사용 가능한 모든 시즌을 쓰는 선택을 유지한다. "
      "(시대 차이보다 표본 증가의 이득이 크다.)")
    A("- **단조 투영**(점수차에 대해 PAV)은 손실을 약간 늘린다. 단조성 위반은 대부분 표본이 적은 칸의 잡음이므로 헤드라인 "
      "B0는 투영하지 않고, 단조 버전을 `models/wp_table_b0_monotone.parquet`로 따로 제공한다.\n")

    A("## 7. 정합성 테스트와 B0 테이블\n")
    A(f"- 최종 테이블 `models/wp_table_b0.parquet`: 학습 {ft['train_seasons'][0]}–{ft['train_seasons'][1]}, k={ft['k']}, "
      f"{ft['rows']:,}행(가능 {ft['possible_rows']:,}), 파일 {ft['file_bytes']/1e3:,.0f} KB. 데이터가 있는 칸 "
      f"{ft['cells_with_data']:,}, 가능한데 데이터 없는 칸 {ft['possible_cells_without_data']:,}(모두 부모 단계 값으로 채움), "
      f"표본 1–29인 가능 칸 {ft['cells_n_lt_30']:,}.")
    A(f"- 단조성 위반(같은 이닝·초말·아웃·주자에서 홈 리드가 1 늘었는데 WP 감소): 전체 {ft['monotonic_violations']:,}칸, "
      f"그중 0.01 초과 {ft['monotonic_violations_gt_0.01']:,}칸, 0.01 초과이면서 n≥100 {ft['monotonic_violations_gt_0.01_with_n_ge_100']}칸. "
      f"평가용 테이블: " + ", ".join(f"{k}: {v:,}" for k, v in ft["monotonic_violations_by_window"].items()) + ".")
    A("- pytest (`tests/`): 상태 인코딩 왕복, 불가능 상태 부재(9회말 이후 홈 리드 상태 0건, 연장 초 시작 동점), "
      "단조성 보고, 동점 상태 WP 범위, Elo 누수(미래 결과 뒤집기·입력 순서 섞기·실제 게임로그 절단 비교), 파서 단위 테스트.\n")

    A("## 8. 처리 시간\n")
    A("| 단계 | 초 |\n|---|---:|")
    A(f"| 다운로드 (1901–2025) | {manifest['seconds']} |")
    A(f"| 이벤트 파싱 + 게임로그 결합 | {build['seconds']} |")
    for k, v in res["timings"].items():
        A(f"| 파이프라인: {k} | {v} |")
    A("")
    A("| 파일 | 크기 |\n|---|---:|")
    for f in ["data/processed/states.parquet", "data/processed/games.parquet", "data/processed/gamelogs.parquet",
              "models/wp_table_b0.parquet", "models/wp_table_b0_monotone.parquet"]:
        if (ROOT / f).exists():
            A(f"| `{f}` | {(ROOT / f).stat().st_size / 1e6:,.2f} MB |")
    A("")
    A("## 9. 다음 단계(B2)에서 재사용할 것\n")
    A("- `wp.states`: 상태 인코딩/디코딩, 진행도, 가능성 판정.")
    A("- `wp.evaluate`: 로그 손실·Brier·캘리브레이션·경기 블록 부트스트랩·층화 — 모델 열 이름만 넘기면 된다.")
    A("- `wp.run_pipeline`: 워크포워드 골격. B2 예측 열을 `MODELS`에 추가하면 같은 표가 나온다.")
    A("- `data/processed/states.parquet`, `games.parquet`: 타석 시작 상태와 경기 메타데이터.\n")
    A("## 출처 고지\n")
    A("     The information used here was obtained free of charge from and is copyrighted by Retrosheet.\n"
      "     Interested parties may contact Retrosheet at \"www.retrosheet.org\".")
    (ROOT / "reports/b0_b1_report.md").write_text("\n".join(L) + "\n")
    print("wrote reports/b0_b1_report.md")


if __name__ == "__main__":
    main()

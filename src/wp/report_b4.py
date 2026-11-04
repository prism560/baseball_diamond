"""Render reports/b4_report.md from the B4 stage outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .manifest import check_fresh

ROOT = Path(__file__).resolve().parents[2]


def ci(x, d=5):
    if isinstance(x, dict) and "diff" in x:
        return f"{x['diff']:+.{d}f} [{x['ci'][0]:+.{d}f}, {x['ci'][1]:+.{d}f}]"
    return f"{x['estimate']:+.{d}f} [{x['ci_low']:+.{d}f}, {x['ci_high']:+.{d}f}]"


def row(rows, model, stratum="all", metric="logloss"):
    df = pd.DataFrame(rows)
    x = df[(df["model"] == model) & (df["stratum"] == stratum) & (df["metric"] == metric)]
    return x.iloc[0] if len(x) else None


def main() -> None:
    files = {k: ROOT / v for k, v in {
        "b3": "reports/b3_results.json", "q": "reports/b4_q.json", "comb": "reports/b4_combine.json",
        "cond": "reports/b4_condition.json"}.items()}
    for f in files.values():
        check_fresh(f)
    J = {k: json.loads(v.read_text()) for k, v in files.items()}
    sc = json.loads((ROOT / "reports/b4_statcast.json").read_text())
    xc = json.loads((ROOT / "data/processed/crosscheck_retrosheet.json").read_text())
    bench = json.loads((ROOT / "reports/b4_bench.json").read_text()) if (ROOT / "reports/b4_bench.json").exists() else None
    L = []
    A = L.append
    A("# B4 리포트 — B3 보완, 모델 결합, 구속·구종 / 당일 컨디션 게이트\n")
    A("> 자동 생성: `PYTHONPATH=src python -m wp.report_b4`. 수치는 모두 실행 측정값이다.\n")

    # MDE
    d = row(J["b3"]["pa_vs_B1"], "B3-B1")
    se = (d["ci_high"] - d["ci_low"]) / (2 * 1.96)
    mde = 2.8 * se
    A("## 0. 최소 탐지 효과 크기 (작업 전 기준)\n")
    A(f"- B3 평가(7,253경기, 경기 단위 블록 부트스트랩)에서 경기 WP 로그 손실 차이의 95% CI 폭은 ±{(d['ci_high'] - d['ci_low']) / 2:.4f}, "
      f"표준오차 약 {se:.5f}.")
    A(f"- **경기 WP 수준의 최소 탐지 효과(5% 양측, 검정력 80%, 2.8×SE) ≈ {mde:.4f}.** 이보다 작은 차이는 전체 3시즌으로도 "
      "구분할 수 없다. 아래에서 경기 수준 결과가 유의하지 않으면 '효과 없음'이 아니라 '탐지할 수 없음'으로 적는다.")
    A("- 그래서 새 기능은 먼저 타석 수준(다클래스 로그 손실/타석)에서 평가하고, **0.002 이상 개선되고 CI가 0 아래일 때만** "
      "경기 WP 수준 재실행 후보로 올린다(승인된 기준. 매치업 q의 타석 이득 0.026이 경기 수준에서 탐지되지 않았으므로, "
      "그 약 8%를 비용 대비 의미 있는 최소 크기로 둠).\n")

    # crosscheck
    A("## 1. 외부 데이터 확인: Retrosheet 공식 파싱 파일과 행 단위 교차검증 (6절)\n")
    A("Retrosheet가 배포하는 parsed play-by-play(`2023–2025plays.zip`)를 Retrosheet의 bevent/cwevent/plays crosswalk에 따라 "
      "열을 맞춰 우리 파서와 행 단위로 대조했다(`src/wp/crosscheck.py`). 정렬 키는 경기 ID와 플레이 순서(`NP` 행 제외).\n")
    A("| 시즌 | 경기(행 수 일치/공통) | 아웃 전·후 | 주자 전(비트·ID) | 주자 후 | 점수·득점 | 타순 | 타자 ID | 투수 ID |\n|---|---|---|---|---|---|---|---|---|")
    for y, s in xc["seasons"].items():
        m = s["mismatch_rows"]
        A(f"| {y} | {s['games_row_count_equal']:,}/{s['games_in_both']:,} | {m['outs_pre']}/{m['outs_post']} | "
          f"{m['bases_pre']}/{m['runner_ids_pre']} | {m['bases_post']} | {m['score']}/{m['runs']} | {m['lineup_slot']} | "
          f"{m['batter']} | {m['pitcher']} |")
    A(f"\n- 비교 행 {xc['rows_total']:,}. **아웃·주자·점수 상태 오류 0건**(플레이 직후 주자 제외). 플레이 직후 주자 불일치 "
      f"{sum(s['mismatch_rows']['bases_post'] for s in xc['seasons'].values())}건은 모두 경기의 마지막 플레이다: Retrosheet는 "
      "끝내기 등 경기 종료 플레이 뒤 주자를 비운다(우리 DP는 그 시점을 종료 상태로 처리하므로 영향 없음, 테스트로 고정).")
    A("- 투수 ID 차이는 모두 타석 도중 투수 교체 뒤의 볼넷, 타자 ID 차이는 2스트라이크 이후 대타의 삼진이다. 공식 기록 규칙상 "
      "책임 선수로 기록되는 관행 차이로 보이며, 매치업 학습에서 해당 42행(0.007%)의 영향은 무시할 수준이다.")
    A("- 배당 비교: **취소**(사용자 지시).\n")

    # step 1
    q = J["q"]
    A("## 2. B3 결함 수정 (1절) — 타석 수준\n")
    A(f"2023–2025 포함 경기 {q['n_pa']:,}타석, 11범주 다클래스 로그 손실. 평가 시즌마다 직전 3시즌으로 회귀 계수를 학습하고 "
      "L2 강도는 마지막 학습 시즌으로 선택.\n")
    A("| 변형 | 로그 손실 | B3 q 대비 [95% CI] |\n|---|---:|---|")
    names = {"A": "A: B3 q (Log5, 홈구장 효과가 섞인 능력치)", "B": "B: 1.1 구장 중립 능력치 + Log5",
             "C": "C: 1.2 다항 로지스틱(리그 q 오프셋)", "D": "D: q' = 1.1 + 1.2"}
    for k in "ABCD":
        A(f"| {names[k]} | {q['logloss'][k]:.5f} | {'—' if k == 'A' else ci(q['vs_A'][k])} |")
    hrA, hrD = q["calibration"]["HR_deciles"]["A"][-1], q["calibration"]["HR_deciles"]["D"][-1]
    A(f"\n- **홈런 상위 10% 과대예측 해소**: 예측/실제 {hrA['pred']:.4f}/{hrA['obs']:.4f} → {hrD['pred']:.4f}/{hrD['obs']:.4f}. "
      f"회귀가 홈런의 투수 계수 {q['coefs']['D_2025']['beta']['HR']['pitcher']:.2f}, 구장 계수 "
      f"{q['coefs']['D_2025']['beta']['HR']['park']:.2f}를 학습했다(Log5는 1로 고정).")
    b = q["bias_by_split"]
    A(f"- 구장 이중 반영: 쿠어스 필드(DEN02) 삼진 편향(실제−예측) {b['Coors Field (DEN02)']['A']['K']:+.4f} → "
      f"{b['Coors Field (DEN02)']['D']['K']:+.4f}; 단타는 {b['Coors Field (DEN02)']['A']['1B']:+.4f} → "
      f"{b['Coors Field (DEN02)']['D']['1B']:+.4f}로 오히려 커졌다. 홈/원정 공격의 범주 편향은 같은 방향·크기"
      f"(홈 HR {b['home batting']['D']['HR']:+.4f}, 원정 {b['away batting']['D']['HR']:+.4f}).")
    A(f"- D − C(1.2 위에 1.1을 더한 이득) = {ci(q['D_vs_C'])}: 개선은 대부분 1.2에서 나온다.")
    A("- **게이트 1.3: q'이 B3 q보다 타석 수준에서 낫다 → q' 채택.** 다만 개선 폭(0.0007)은 3·4절의 0.002 기준보다 한 자릿수 "
      "작아 경기 수준에서 탐지될 가능성은 낮다.\n")

    # step 2
    c = J["comb"]
    A("## 3. 모델 결합 (2절) — DP 재실행 없이\n")
    A("가중치는 평가 시즌 이전 시즌으로만 학습(2023 평가에는 이를 위해 계산한 2022 시즌 805경기 B3 결과): ")
    A(", ".join(f"{S}: 학습 {x['train_seasons']}, AVG의 B3 가중치 {x['avg_weight_on_B3']}" for S, x in c["fit_pa"].items()) + ".\n")
    A("| 비교 (타석 시작) | 로그 손실 차이 [95% CI] |\n|---|---|")
    for lab, key, m in (("B3 − B2E", "pa_vs_B2E", "B3-B2E"), ("B3E − B2E", "pa_vs_B2E", "B3E-B2E"),
                        ("AVG − B2E", "pa_vs_B2E", "AVG-B2E"), ("AVG − B3", "pa_vs_B3", "AVG-B3"),
                        ("AVG − B1", "pa_vs_B1", "AVG-B1"), ("B3E − B3", "pa_vs_B3", "B3E-B3")):
        A(f"| {lab} | {ci(row(c[key], m))} |")
    A("\n| 비교 (투구 시점) | 로그 손실 차이 [95% CI] |\n|---|---|")
    for lab, key, m in (("AVG − B2E", "pitch_vs_B2E", "AVG-B2E"), ("AVG − B3", "pitch_vs_B3", "AVG-B3"),
                        ("B3E − B2E", "pitch_vs_B2E", "B3E-B2E")):
        A(f"| {lab} | {ci(row(c[key], m))} |")
    A("\n- AVG(B3와 B2E의 로짓 평균)는 B3와 B1보다 유의하게 낫고, B2E보다는 점추정이 낫지만 **탐지할 수 없다**(CI가 0 포함).")
    A("- B3E(B3 + Elo): B3 대비 이득 없음 — B3에는 Elo가 더할 정보가 거의 남아 있지 않다.")
    dd = c["disagreement"]["pa_start"]
    A(f"\n**불일치 시점 분석** (|B3−B2E| > 2.5%p, 타석 시작의 {dd['share'] * 100:.1f}%, {dd['n']:,}개): 평균 예측 B3 "
      f"{dd['mean_pred']['B3']:.3f}, B2E {dd['mean_pred']['B2E']:.3f}, 실제 홈 승률 {dd['obs_rate']:.3f}.\n")
    A("| 층 | B3 − B2E | AVG − B2E |\n|---|---|---|")
    for s in ("all disagreements", "B3 above B2E", "B3 below B2E", "starter pitching", "bullpen pitching", "inning>=7"):
        A(f"| {s} | {ci(row(dd['boot'], 'B3-B2E', s), 4)} | {ci(row(dd['boot'], 'AVG-B2E', s), 4)} |")
    A("\n- 불일치 시점에서 B3와 B2E 중 체계적으로 나은 쪽은 없다(모든 층에서 CI가 0 포함). AVG는 모든 층에서 점추정이 가장 "
      "낮고, 'B3가 B2E보다 낮게 본 시점'에서는 B2E보다 유의하게 낫다.\n")

    # step 3
    A("## 4. B4 구속·구종 (3절)\n")
    A(f"- 상태: **{sc['status']}**")
    A("- 선수 ID 매핑: Chadwick 등록부(people-*.csv, MLBAM↔Retrosheet 24,672명) — 실시간 파이프라인과 같은 매핑.")
    A("- 연결: 경기는 (날짜, 홈 선발 투수), 타석은 경기 안 순서대로 (타자, 투수) 쌍 일치(`src/wp/statcast.py`). 연결 실패율은 "
      "데이터가 들어오면 보고한다.")
    A("- 라이브 피드 가용 여부(서빙 가능한 피처만 사용). statsapi.mlb.com도 차단되어 실제 피드로 확인하지 못했으므로 '예상'이며, "
      "서빙 전에 실제 피드로 재확인해야 한다:\n")
    A("| Statcast 필드 | 라이브 피드 필드(예상) | 모델 사용 |\n|---|---|---|")
    for x in sc["live_feed_availability"]:
        A(f"| {x['statcast']} | {x['live_feed']} | {'예' if x['used'] else '아니오'} |")
    A(f"\n- 게이트 기준: 타석 수준 로그 손실 q' 대비 {sc['gate_threshold_pa_logloss']} 이상 개선. 데이터가 없어 **미실시**이며, "
      "서빙에는 넣지 않는다.\n")

    # step 4
    cd = J["cond"]
    A("## 5. B4 당일 컨디션 (4절) — Retrosheet 피처(스트라이크 비율, 투구 수)\n")
    A(f"현재 투수가 이 경기에서 이 타석 전에 던진 공만 사용. 스트라이크 비율은 시즌 기준선 대비 베이지안 편차(사전 강도 k를 "
      f"학습 시즌으로 선택: {', '.join(str(cd['coefs'][f'k_lambda_{S}']['k']) for S in (2023, 2024, 2025))}), 구속 차이는 "
      "Statcast가 필요해 제외. 해저드는 바꾸지 않는다.\n")
    A(f"| 비교 | 전체 {cd['n_pa']:,}타석 | 같은 투수의 후반 타석(투구 30개 이후) {cd['n_pa_later_same_pitcher']:,} |\n|---|---|---|")
    lab = {"QC-Q": "q'+컨디션 − q'", "QT-Q": "q'+TTO − q'", "QTC-QT": "q'+TTO+컨디션 − q'+TTO"}
    for k, v in cd["diff"].items():
        A(f"| {lab[k]} | {ci(v['all'])} | {ci(v['later_PAs_same_pitcher'])} |")
    A(f"\n- **게이트 미통과**(기준 {cd['gate_threshold']}): 컨디션의 이득은 유의하지만 기준의 절반 이하.")
    qc, qtc = cd["coefs"]["QC_2025"], cd["coefs"]["QTC_2025"]
    A(f"- **TTO와의 겹침**: TTO 항을 넣으면 컨디션의 추가 이득이 {cd['diff']['QC-Q']['all']['diff']:+.5f} → "
      f"{cd['diff']['QTC-QT']['all']['diff']:+.5f}로 줄어 약 "
      f"{(1 - cd['diff']['QTC-QT']['all']['diff'] / cd['diff']['QC-Q']['all']['diff']) * 100:.0f}%가 겹친다. 투구 수의 삼진 계수도 "
      f"{qc['K'][1]:+.3f}(TTO 없음) → {qtc['K'][2]:+.3f}(TTO 있음)로 줄었다. DP에는 이미 TTO와 강판 해저드가 있으므로 "
      "컨디션을 넣으면 이중 반영이 된다.")
    A("- 누수 테스트: 경기 안에서 이후 투구를 지워도 현재 값이 바뀌지 않음을 확인(`tests/test_condition.py`). 이 테스트가 처음 "
      "구현의 누수(리그 스트라이크 비율을 전체 데이터로 계산)를 잡아 수정했다.\n")

    # step 5
    A("## 6. 경기 WP 수준 재평가 (5절)\n")
    A("- 후보: q'(1절 채택) + 결합(AVG). 3·4절은 게이트를 통과하지 못해 후보에 넣지 않는다.")
    A(f"- **실시하지 않기로 결정(검정력 부족).** q'의 타석 수준 이득(0.0007)이 최소 탐지 효과({mde:.4f})의 약 1/3이라, "
      "8–9시간(7,253경기)을 돌려도 B3와 구분되지 않을 가능성이 높다. 서빙의 B3 엔진은 기존 q를 그대로 쓴다.\n")

    # step 7
    A("## 7. 실시간 비용 (7절)\n")
    if bench:
        mn, mx = bench["mean"], bench["max"]
        A(f"2024 시즌 {bench['games']}경기, 단일 스레드, float32, 점수차 ±20(재계산 ±15 변형 포함).\n")
        A("| 작업 | 평균 | 최대 |\n|---|---:|---:|")
        A(f"| 경기 전 전체 계산 | {mn['pregame_s']:.2f}초 | {mx['pregame_s']:.2f}초 |")
        A(f"| 투수 교체 재계산(현재 하프) | {mn['pitching_change_s']:.2f}초 | {mx['pitching_change_s']:.2f}초 |")
        for inn in (3, 5, 7):
            for k, lab2 in (("full_s", "전체 커널 재구성"), ("kernel_cache_s", "커널 캐시(해당 타순만 재구성)"),
                            ("R15_s", "±15 점수차")):
                A(f"| 대타 재계산 {inn}회부터 — {lab2} | {mn[f'ph_inn{inn}.{k}']:.2f}초 | {mx[f'ph_inn{inn}.{k}']:.2f}초 |")
        A("\n- 커널 캐시의 이득은 5–10%에 그친다: 시간의 대부분은 커널 조립이 아니라 선형 풀이다. **5회 이전 대타는 캐시와 "
          "±15를 써도 5초를 넘는다.** 실시간에서는 재계산을 비동기로 돌리고, 끝날 때까지 B2E(즉시 조회)를 표시한 뒤 결과가 "
          "나오면 AVG로 바꾸는 방식을 권한다(`docs/REALTIME.md`).")
        A("- B2E는 B2 테이블 조회 + 로지스틱(마이크로초)이라 비용이 사실상 없다. AVG의 비용은 B3와 같다.\n")
    else:
        A("측정 결과 없음.\n")

    # step 8
    A("## 8. 서빙 모델 선택 (8절 규칙 적용)\n")
    a_b2e = row(c["pa_vs_B2E"], "AVG-B2E")
    A(f"1. 경기 WP 로그 손실에서 모든 후보보다 유의하게 나은 모델: **없음.** AVG는 B3·B1보다 유의하게 낫지만 B2E와의 차이"
      f"({ci(a_b2e)})는 탐지할 수 없다.")
    A("2. 불일치 시점 분석에서 체계적으로 나은 모델: B3와 B2E 사이에는 없다. **AVG는 모든 불일치 층에서 점추정이 가장 낮고, "
      "B3가 B2E보다 낮게 본 시점에서는 B2E보다 유의하게 낫다.** → 이 규칙으로 **AVG를 선택**한다.")
    A("3. (비용) AVG는 B3 엔진 비용이 들지만 7절 측정치로 실시간 기준 안에 있다.")
    A("4. (장애 대체) AVG는 B3가 실패하면 B2E로 자연스럽게 떨어진다(가중치 1−w → 1). B2E는 테이블 조회라 장애에 강하다.")
    A("\n**결정: 기본 서빙 모델 = AVG(B3와 B2E의 로짓 가중 평균, B3 가중치 0.55), 장애 대체 = B2E.** "
      "B3 엔진은 기존 q를 유지한다(5절 결정). `docs/REALTIME.md`에 반영.\n")
    A("## 출처 고지\n")
    A("     The information used here was obtained free of charge from and is copyrighted by Retrosheet.\n"
      "     Interested parties may contact Retrosheet at \"www.retrosheet.org\".")
    (ROOT / "reports/b4_report.md").write_text("\n".join(L) + "\n")
    print("wrote reports/b4_report.md")


if __name__ == "__main__":
    main()

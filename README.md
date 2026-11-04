# baseballdiamond — MLB 실시간 승리확률

1구마다 "원정 24% / 홈 76%" 형태의 승리확률을 내는 시스템을 단계별로 만든다.

**현재 상태**: B0–B4 완료. 기본 서빙 모델은 AVG(B3와 B2E의 로짓 가중 평균)이고, 실시간 파이프라인은 아직 착수하지 않았다. 인수인계 문서는 [`docs/HANDOFF.md`](docs/HANDOFF.md)다.

저장소 루트의 `2023plays.zip`–`2025plays.zip`은 Retrosheet가 배포하는 공식 파싱 play-by-play 파일이다(사용자 업로드). `wp.crosscheck`(`make data/processed/crosscheck_retrosheet.json`)가 그 자리에서 읽어 우리 파서 결과와 행 단위로 교차검증한다.

- **B0** — 상태 테이블 WP: 이닝·초말·아웃·주자·점수차(·연장 규칙 시대)별 홈 승률. 계층적 베이지안 축소로 평활했다.
- **B1** — B0와 누수 없는 Elo 경기 전 승률을 결합한 로지스틱 모델. 경기 진행도에 따라 Elo 비중이 줄어든다.

- **B2** — 리그 평균 **정확 DP 엔진**(타석 단위 역방향 귀납, 연장 고정점, 타순 축)과 **현재 타석 볼카운트 체인**(IPF로 DP와 정합). 투구마다 WP를 조회할 수 있다.

- **B3** — 선수 능력치(그날 전날까지만 쓰는 누수 없는 추정), 매치업(다항 Log5), 선발 강판 해저드와 불펜 등급을 DP 안에 넣은 엔진. 경기 전 WP와 경기 중 WP를 한 엔진으로 계산한다. 실시간 호출 순서는 [`docs/REALTIME.md`](docs/REALTIME.md)에 있다.

- **B4** — B3 보완(구장 중립 능력치, 다항 로지스틱 매치업), 모델 결합(B3와 B2E의 로짓 평균이 기본 서빙 모델), 구속·구종(Statcast 대기), 당일 컨디션(게이트 미통과), Retrosheet 공식 파싱 파일과 행 단위 교차검증.

결과와 설계 근거는 [`reports/b4_report.md`](reports/b4_report.md), [`reports/b0_b1_report.md`](reports/b0_b1_report.md), [`reports/b2_report.md`](reports/b2_report.md), [`reports/b3_report.md`](reports/b3_report.md)에 있다. B3 확장 지점은 [`docs/B3_INTERFACE.md`](docs/B3_INTERFACE.md).

## 데이터

- **출처**: Retrosheet 이벤트 파일 1908–2025, 게임로그 1900–2025. 사용 가능한 모든 시즌을 썼다.
- **받는 곳**: Chadwick Baseball Bureau의 GitHub 미러 `chadwickbureau/retrosheet`. retrosheet.org는 작업 환경에서 차단되어 미러를 썼다.
- **파싱**: Chadwick 도구는 쓰지 못했다(환경 정책상 외부 C 코드 빌드 불가). 대신 순수 파이썬 파서 `src/wp/retro_parse.py`를 만들었고, 경기마다 최종 점수를 게임로그와 대조해 검증했다.
- **2026 시즌**: 아직 미공개다. 추가 절차는 [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)의 TODO에 있다. 사용자가 retrosheet.org에서 직접 받는 방법도 같은 문서에 있다.

## 실행

모든 단계는 `Makefile`에 의존성과 함께 정의되어 있다. 각 산출물 옆의 `*.meta.json`에는 입력·코드 해시가 기록된다. 다음 단계와 리포트 생성기는 입력이 바뀌었는데 다시 만들지 않은 산출물을 발견하면 실패한다(`src/wp/manifest.py`).

```bash
pip install -r requirements.txt
```

```bash
make all
```

개별 단계로 실행하려면 다음과 같다.

```bash
PYTHONPATH=src python -m wp.download --first 1901 --last 2025
```

```bash
PYTHONPATH=src python -m wp.games
```

```bash
PYTHONPATH=src python -m wp.validate
```

```bash
PYTHONPATH=src python -m wp.run_pipeline
```

```bash
PYTHONPATH=src python -m wp.report
```

```bash
PYTHONPATH=src python -m wp.run_b2
```

```bash
PYTHONPATH=src python -m wp.report_b2
```

```bash
python -m pytest -q tests
```

| 단계 | 내용 | 산출물 | 소요 |
|---|---|---|---|
| 다운로드 | 원본 약 1.1 GB | `data/raw/` | 약 1분 |
| 파싱 | 이벤트 파싱, 제외 판정 | `data/processed/*.parquet` | 약 1분 |
| 검증 (0절) | 라인스코어·3아웃·잔루·타순·투구 시퀀스 검증, 위반 경기 제외, `events.parquet` | `data/processed/validation*.parquet` | 약 2분 |
| 파이프라인 | 워크포워드 평가, 최종 테이블 | `models/`, `reports/results.json`, 그림 | 약 1분 |
| 리포트 | 리포트 생성 | `reports/b0_b1_report.md` | — |
| B2 | 전이 추정, DP, 카운트 체인, 워크포워드 평가 | `models/wp_b2_*.parquet`, `reports/b2_results.json` | 약 2분 |

## 구조

```
src/wp/
  download.py     Retrosheet 미러 다운로드
  retro_parse.py  이벤트 파일 파서 (타석 시작 상태, 최종 점수)
  games.py        게임로그 결합, 제외 규칙, parquet 생성
  validate.py     0절 파서 검증 (게임로그 라인스코어·잔루·박스 기록, 3아웃, 타순, 투구 시퀀스)
  states.py       상태 인코딩/디코딩·진행도·가능성 판정 (재사용 모듈)
  b0_table.py     B0 테이블 적합, 평활, 조회(WPTable, lookup_wp), 단조성 점검
  elo.py          누수 없는 Elo
  b1_model.py     B1 로지스틱 결합
  evaluate.py     로그 손실/Brier/캘리브레이션/경기 블록 부트스트랩/층화 (재사용 모듈)
  run_pipeline.py 워크포워드 실행 (B0·B1)
  transitions.py  B2 타석 결과 q, 진루 A, 타석 간 전이 N, 직접 전이 D
  dp.py           B2 정확 DP 엔진 (연장 고정점, 타순 축)
  count_chain.py  B2 볼카운트 체인, IPF 정합 보정
  wp_b2.py        B2 조회 API (PitchWP, PAWP)
  run_b2.py       B2 워크포워드 실행
  manifest.py     산출물 출처 기록과 오래된 입력 검사
  ratings.py      B3 선수 능력치 (경기 전날까지), 구장·플래툰
  matchup.py      B3 매치업 결합 (다항 Log5, INT 리그 수준)
  pitching_policy.py  B3 선발 강판 해저드, 불펜 등급, 타순 바퀴 효과
  dp_b3.py        B3 DP 엔진 (타순×투수 단계 합성 상태)
  b3_game.py      B3 경기별 입력 조립과 타석별 평가
  run_b3_check.py / run_b3_games.py / run_b3_eval.py  B3 중간 점검, 경기별 실행, 평가
  report.py       리포트 생성
tests/            pytest 정합성 테스트
models/           wp_table_b0.parquet (+단조 버전, 평가 시즌별 테이블), b1_coef.json, elo_params.json
reports/          b0_b1_report.md, results.json, figures/
docs/             DATA_SOURCES.md
data/             원본·가공 데이터 (git 제외)
```

## 조회 예시

```python
from wp.b0_table import WPTable
t = WPTable.load("models/wp_table_b0.parquet")
t.wp(inning=9, half=1, outs=2, bases=0b011, diff=-1)   # 9회말 2사 1·2루, 홈 1점 뒤짐 → 홈 승률
t.wp(inning=10, half=0, outs=0, bases=0b010, diff=0, xrule=1)  # 연장(2루 주자 규칙)
```

```python
from wp.wp_b2 import PitchWP
pw = PitchWP.load("models/wp_b2_pitch.parquet")
pw.split(inning=9, half=1, outs=2, bases=0b111, diff=0, balls=3, strikes=2)  # (원정 %, 홈 %)
```

`bases`는 비트로 쓴다. 1루=1, 2루=2, 3루=4. `diff`는 홈 점수에서 원정 점수를 뺀 값이다.

## 출처 고지

     The information used here was obtained free of charge from and is
     copyrighted by Retrosheet.  Interested parties may contact Retrosheet
     at "www.retrosheet.org".

# 데이터 출처와 직접 받는 방법

## 1. 이번 작업에서 쓴 경로 (2026-11-02 기준)

| 항목 | 위치 | 비고 |
|---|---|---|
| 이벤트 파일 | `https://raw.githubusercontent.com/chadwickbureau/retrosheet/master/seasons/<YYYY>/<YYYY><TEAM>.EVA\|EVN` | 1908–2025 MLB 정규시즌 (1901–1907 없음) |
| 팀 목록 | `.../seasons/<YYYY>/TEAM<YYYY>` | 리그 코드 `A`/`N` 또는 `AL`/`NL` |
| 게임로그 | `.../seasons/<YYYY>/GL<YYYY>.TXT` (2024만 `gl2024.txt`) | 1900–2025 |
| 팀 코드 연표 | `.../reference/teams.csv` | Elo 프랜차이즈 연결 확인용 |

- 이 저장소는 Chadwick Baseball Bureau가 관리하는 Retrosheet 미러다. `master` 브랜치는 공식 데이터에 정오표를 반영한 보강판이고, `official` 브랜치에는 Retrosheet 원본이 그대로 있다.
- 작업 환경에서는 `www.retrosheet.org`로 나가는 연결이 프록시에서 차단되어(HTTP 403) 미러를 썼다.
- Retrosheet가 공식으로 제공하는 "파싱된 CSV"가 있는지는 사이트에 접속하지 못해 **확인하지 못했다.**
- Chadwick(cwevent/cwgame)은 이 환경에서 외부 C 코드 빌드가 정책상 막혀 쓰지 못했다. 그래서 순수 파이썬 파서(`src/wp/retro_parse.py`)를 직접 만들었다. 검증 방법은 파싱한 최종 점수를 게임로그와 경기마다 대조하는 것이고, 99.9%가 일치했다.

## 2. 2026 시즌 데이터 — TODO (나중에 찾으면 추가)

- **현재 상태 (2026-11-02 확인)**: 미러에 `seasons/2026/TEAM2026`이 없다(404). 2026 이벤트 파일과 게임로그는 아직 공개되지 않은 것으로 보인다.
- **확인 방법**:
  ```bash
  curl -sI https://raw.githubusercontent.com/chadwickbureau/retrosheet/master/seasons/2026/TEAM2026 | head -1
  curl -sI https://raw.githubusercontent.com/chadwickbureau/retrosheet/master/seasons/2026/GL2026.TXT | head -1
  ```
  두 요청이 모두 `200`이면 받을 수 있다. 아니면 아래 3절처럼 retrosheet.org에서 직접 받는다.
- **추가 절차**:
  1. `PYTHONPATH=src python -m wp.download --first 2026 --last 2026`
     (직접 받은 파일을 쓸 때는 `data/raw/2026/`에 넣는다)
  2. `PYTHONPATH=src python -m wp.games --last 2026`
  3. `PYTHONPATH=src python -m wp.run_pipeline --eval 2024 2025 2026`
  4. `PYTHONPATH=src python -m wp.report`
- **주의**:
  - 2026 연장 규칙이 2023–2025와 같은지 이벤트 파일의 `info,tiebreaker` 값으로 확인한다. 코드는 이 필드로 `xrule`을 정한다.
  - 연고 이전이나 팀 코드 변경이 있었는지 `reference/teams.csv`로 확인하고, 있으면 `src/wp/elo.py`의 `FRANCHISE`를 갱신한다.

## 3. retrosheet.org에서 직접 받는 방법 (사용자용)

이 환경에서는 사이트에 접속할 수 없어 아래 URL을 **검증하지 못했다.** 사이트 메뉴 이름은 바뀌었을 수 있다. 찾는 대상과 넣을 위치를 기준으로 보면 된다.

1. **플레이 단위 이벤트 파일** (가장 중요)
   - retrosheet.org → "Data downloads" → "Play-by-play files"(또는 "Event files")에서 시즌별 정규시즌 zip을 받는다. 예전 형식은 `https://www.retrosheet.org/events/<YYYY>eve.zip`이다.
   - zip을 풀어 `<YYYY><TEAM>.EVA`, `.EVN`, `TEAM<YYYY>` 파일을 `data/raw/<YYYY>/`에 넣는다.
   - 포스트시즌 파일(`.EVE`, `<YYYY>WS` 등)은 넣지 않는다.
2. **게임로그**
   - "Game logs" 페이지에서 `gl<YYYY>.zip`을 받는다. 예전 형식은 `https://www.retrosheet.org/gamelogs/gl<YYYY>.zip`이다.
   - `gl<YYYY>.txt`를 `data/raw/<YYYY>/`에 넣는다. 대소문자는 상관없다.
3. **파싱된 CSV** (있다면)
   - Retrosheet 다운로드 메뉴에 플레이 단위 CSV나 경기 단위 CSV가 있으면 받아 `data/raw/retrosheet_csv/`에 넣고 알려 달라.
   - 그러면 우리 파서 출력(`states.parquet`)과 경기별로 대조하는 교차검증 스크립트를 추가하겠다. 특히 점수 불일치로 제외된 216경기를 복구할 수 있다.
4. **Chadwick 실행파일** (선택)
   - 로컬 PC에서 `cwevent`와 `cwgame`을 돌릴 수 있다면 다음 결과 CSV를 `data/raw/chadwick/`에 넣으면 같은 방식으로 교차검증할 수 있다:
     - `cwevent -y <YYYY> -f 0-96 <YYYY>*.EV? > events<YYYY>.csv`
     - `cwgame -y <YYYY> <YYYY>*.EV? > games<YYYY>.csv`

받은 파일은 git에 올리지 않는다(`data/`는 `.gitignore`에 있다). 파일을 넣은 뒤에는 2절의 2–4단계를 다시 실행한다.

## 4. 출처 고지 (Retrosheet 요구 문구)

     The information used here was obtained free of charge from and is
     copyrighted by Retrosheet.  Interested parties may contact Retrosheet
     at "www.retrosheet.org".

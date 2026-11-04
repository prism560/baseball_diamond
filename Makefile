# Pipeline stages and their dependencies.  Every stage also writes <file>.meta.json
# (input/code hashes, see src/wp/manifest.py); downstream stages and the report
# generators refuse to run on stale inputs.
PY := PYTHONPATH=src python3
P  := data/processed
W  := src/wp

.PHONY: all b0b1 b2 b3 test clean-processed

all: reports/b0_b1_report.md reports/b2_report.md

data/raw/download_manifest.json: $(W)/download.py
	$(PY) -m wp.download --first 1901 --last 2025

$(P)/games_base.parquet $(P)/states.parquet $(P)/gamelogs.parquet $(P)/build_summary.json &: \
		data/raw/download_manifest.json $(W)/games.py $(W)/retro_parse.py
	$(PY) -m wp.games

$(P)/games.parquet $(P)/events.parquet $(P)/validation.parquet $(P)/validation_summary.json &: \
		$(P)/games_base.parquet $(W)/validate.py $(W)/retro_parse.py
	$(PY) -m wp.validate

$(P)/pa.parquet: $(P)/events.parquet $(P)/games.parquet $(W)/transitions.py
	$(PY) -m wp.transitions

reports/results.json: $(P)/games.parquet $(P)/states.parquet $(P)/gamelogs.parquet \
		$(W)/run_pipeline.py $(W)/b0_table.py $(W)/b1_model.py $(W)/elo.py $(W)/evaluate.py $(W)/states.py
	$(PY) -m wp.run_pipeline

reports/b0_b1_report.md: reports/results.json $(W)/report.py
	$(PY) -m wp.report

reports/b2_results.json: $(P)/pa.parquet $(P)/games.parquet $(P)/states.parquet $(P)/gamelogs.parquet \
		$(W)/run_b2.py $(W)/dp.py $(W)/transitions.py $(W)/count_chain.py
	$(PY) -m wp.run_b2

reports/b2_report.md: reports/b2_results.json $(W)/report_b2.py
	$(PY) -m wp.report_b2

b0b1: reports/b0_b1_report.md
b2: reports/b2_report.md

test:
	python3 -m pytest -q tests

reports/b3_check.json: $(P)/pa.parquet $(P)/games.parquet $(W)/run_b3_check.py $(W)/ratings.py $(W)/matchup.py
	$(PY) -m wp.run_b3_check

# B3: per-game runs are resumable (data/processed/b3_runs/); main comparison on all games,
# ablations on a 1/9 subset (scripts/run_b3_all.sh), then evaluation and report.
b3-runs: reports/b3_check.json
	scripts/run_b3_all.sh

reports/b3_results.json: reports/b3_check.json $(P)/pa.parquet $(W)/run_b3_eval.py $(W)/b3_game.py $(W)/dp_b3.py
	$(PY) -m wp.run_b3_eval

reports/b3_report.md: reports/b3_results.json $(W)/report_b3.py
	$(PY) -m wp.report_b3

b3: reports/b3_report.md

# Cross-check against Retrosheet's official parsed play-by-play (zips at the repo root)
$(P)/crosscheck_retrosheet.json: $(P)/games.parquet 2023plays.zip 2024plays.zip 2025plays.zip $(W)/crosscheck.py $(W)/retro_parse.py
	$(PY) -m wp.crosscheck

# B4 step 3: Statcast (collected on a local PC with scripts/fetch_statcast.py)
b4-statcast:
	$(PY) -m wp.run_b4_statcast

# B4: PA-level gates, combination, report
reports/b4_q.json: reports/b3_check.json $(P)/pa.parquet $(W)/run_b4_q.py $(W)/matchup_v2.py $(W)/ratings.py
	$(PY) -m wp.run_b4_q

reports/b4_condition.json: reports/b3_check.json $(P)/pa.parquet $(W)/run_b4_condition.py $(W)/condition.py
	$(PY) -m wp.run_b4_condition

reports/b4_combine.json: reports/b3_results.json $(W)/run_b4_combine.py
	$(PY) -m wp.run_b4_combine

reports/b4_report.md: reports/b4_q.json reports/b4_condition.json reports/b4_combine.json $(P)/crosscheck_retrosheet.json $(W)/report_b4.py
	$(PY) -m wp.run_b4_statcast
	$(PY) -m wp.report_b4

b4: reports/b4_report.md

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

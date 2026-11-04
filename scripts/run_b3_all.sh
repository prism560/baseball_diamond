#!/usr/bin/env bash
# Full B3 evaluation runs (resumable): main comparison on all games,
# ablations on a deterministic 1/9 subset of games.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
python3 -m wp.run_b3_games --ablation full --fraction 1.0 --workers 4 || exit 1
for ab in no_platoon no_park team_avg_bullpen no_tto; do
  python3 -m wp.run_b3_games --ablation "$ab" --fraction 0.1111 --workers 4 || exit 1
done
echo ALLDONE

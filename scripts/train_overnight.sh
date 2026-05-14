#!/usr/bin/env bash
# Overnight training pipeline: train, then evaluate vs random + vs heuristic.
#
# Usage:
#   ./scripts/train_overnight.sh [TOTAL_STEPS]
#
# Or to keep the laptop awake while it runs:
#   caffeinate -i ./scripts/train_overnight.sh
#
# Defaults are tuned for ~9 hours wall on a 4-core mac at ~158 steps/s with
# the default network (512×3 MLP). Override TOTAL_STEPS for shorter runs.
#
# Outputs land under runs/overnight_<timestamp>/:
#   tb/                  — TensorBoard event files (point `tensorboard --logdir` here)
#   snapshots/           — periodic .pt checkpoints
#   final.pt             — final model
#   train.log            — full stdout/stderr of the training run
#   eval_vs_random.md    — Markdown report of the final-model tournament vs random
#   eval_vs_heuristic.md — same, vs the heuristic baseline
#   replays_vs_*/        — per-game JSON replays loadable in the GUI

set -euo pipefail

# Pick up the project venv so `python` resolves to the catan-engine install.
# Tolerant of being invoked from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

TOTAL_STEPS="${1:-4000000}"
NUM_ENVS="${NUM_ENVS:-4}"
EVAL_EVERY="${EVAL_EVERY:-250000}"
EVAL_GAMES="${EVAL_GAMES:-30}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-250000}"
FINAL_EVAL_GAMES="${FINAL_EVAL_GAMES:-200}"

NAME="overnight_$(date +%Y%m%d_%H%M)"
RUN_DIR="runs/$NAME"
mkdir -p "$RUN_DIR"

echo "[train_overnight] run=$NAME total_steps=$TOTAL_STEPS num_envs=$NUM_ENVS"
echo "[train_overnight] output=$RUN_DIR"

python scripts/train.py \
    --total-steps "$TOTAL_STEPS" \
    --num-envs "$NUM_ENVS" \
    --eval-every "$EVAL_EVERY" --eval-games "$EVAL_GAMES" \
    --snapshot-every "$SNAPSHOT_EVERY" \
    --output-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/train.log"

echo "[train_overnight] training done — running final tournament vs random"
python -m rl.cli evaluate \
    --learner "$RUN_DIR/final.pt" --opponent random --games "$FINAL_EVAL_GAMES" \
    --output-dir "$RUN_DIR/replays_vs_random" \
    2>&1 | tee "$RUN_DIR/eval_vs_random.md"

echo "[train_overnight] running final tournament vs heuristic"
python -m rl.cli evaluate \
    --learner "$RUN_DIR/final.pt" --opponent heuristic --games "$FINAL_EVAL_GAMES" \
    --output-dir "$RUN_DIR/replays_vs_heuristic" \
    2>&1 | tee "$RUN_DIR/eval_vs_heuristic.md"

echo "[train_overnight] all done — see $RUN_DIR/{eval_vs_random.md,eval_vs_heuristic.md}"

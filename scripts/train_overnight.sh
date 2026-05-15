#!/usr/bin/env bash
# Overnight training pipeline: train, then evaluate vs random + vs heuristic.
#
# Usage:
#   ./scripts/train_overnight.sh [TOTAL_STEPS]
#
# Or to keep the laptop awake while it runs:
#   caffeinate -i ./scripts/train_overnight.sh
#
# Defaults: shaped reward (with 0.5 stalemate penalty) + 3 HeuristicAgent
# baselines in the OpponentPool at weight 0.6, single-env rollouts so the
# pool actually applies, stdout heartbeat every 25 iters (~50k env steps),
# watchdog aborts if rollout/wins stays 0 for 200 consecutive iters
# (~90 minutes at 75 steps/s). The stalemate penalty is what makes
# stalling unattractive relative to playing out a loss — without it the
# first attempt at a shaped-reward run found "stall the game" as a local
# minimum and the watchdog correctly aborted it. At ~75 steps/s on a
# 4-core mac the 2M default total takes ~7.5 hours; tune via TOTAL_STEPS
# for shorter runs.
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

TOTAL_STEPS="${1:-2000000}"
NUM_ENVS="${NUM_ENVS:-1}"
EVAL_EVERY="${EVAL_EVERY:-100000}"
EVAL_GAMES="${EVAL_GAMES:-30}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-100000}"
FINAL_EVAL_GAMES="${FINAL_EVAL_GAMES:-200}"
REWARD="${REWARD:-shaped}"
STALEMATE_PENALTY="${STALEMATE_PENALTY:-0.8}"
POOL_BASELINES="${POOL_BASELINES:-3}"
BASELINE_WEIGHT="${BASELINE_WEIGHT:-0.3}"
ENTROPY_COEF="${ENTROPY_COEF:-0.03}"
PRINT_EVERY="${PRINT_EVERY:-25}"
WATCHDOG_ZERO_WINS_ITERS="${WATCHDOG_ZERO_WINS_ITERS:-200}"
# Warm-start curriculum: set INIT_FROM=path/to/prev/final.pt and typically
# bump BASELINE_WEIGHT (0.5–0.7) and drop LR (e.g. 1e-4) for refinement.
INIT_FROM="${INIT_FROM:-}"
LR="${LR:-}"

NAME="overnight_$(date +%Y%m%d_%H%M)"
RUN_DIR="runs/$NAME"
mkdir -p "$RUN_DIR"

echo "[train_overnight] run=$NAME total_steps=$TOTAL_STEPS num_envs=$NUM_ENVS"
echo "[train_overnight] reward=$REWARD stalemate_penalty=$STALEMATE_PENALTY"
echo "[train_overnight] baselines=$POOL_BASELINES weight=$BASELINE_WEIGHT entropy=$ENTROPY_COEF"
echo "[train_overnight] watchdog_zero_wins_iters=$WATCHDOG_ZERO_WINS_ITERS"
if [ -n "$INIT_FROM" ]; then
    echo "[train_overnight] warm-start init_from=$INIT_FROM lr=${LR:-default}"
fi
echo "[train_overnight] output=$RUN_DIR"

# Optional flags only land on the command line when set, so cold-start runs
# stay byte-identical to the prior invocation.
EXTRA_ARGS=()
if [ -n "$INIT_FROM" ]; then
    EXTRA_ARGS+=(--init-from "$INIT_FROM")
fi
if [ -n "$LR" ]; then
    EXTRA_ARGS+=(--lr "$LR")
fi

python scripts/train.py \
    --total-steps "$TOTAL_STEPS" \
    --num-envs "$NUM_ENVS" \
    --reward "$REWARD" \
    --stalemate-penalty "$STALEMATE_PENALTY" \
    --pool-baselines "$POOL_BASELINES" \
    --baseline-weight "$BASELINE_WEIGHT" \
    --entropy-coef "$ENTROPY_COEF" \
    --print-every "$PRINT_EVERY" \
    --watchdog-zero-wins-iters "$WATCHDOG_ZERO_WINS_ITERS" \
    --eval-every "$EVAL_EVERY" --eval-games "$EVAL_GAMES" \
    --snapshot-every "$SNAPSHOT_EVERY" \
    --output-dir "$RUN_DIR" \
    "${EXTRA_ARGS[@]}" \
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

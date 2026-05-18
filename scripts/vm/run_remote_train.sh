#!/usr/bin/env bash
# Start or re-attach a Catan AZ training run on the VM, inside a tmux
# session named ``$CATAN_TMUX_SESSION``.
#
# Behaviour is idempotent:
# - If the tmux session does NOT exist on the VM, we git pull (so the VM
#   picks up local commits), then launch ``scripts/train_alphazero.py``
#   inside a fresh tmux window. Control returns to the laptop immediately;
#   the run continues after we disconnect.
# - If the tmux session ALREADY exists, we re-attach instead of starting
#   a second concurrent run. Use ``--force-restart`` to kill the existing
#   session first.
#
# Usage (from laptop):
#   ./scripts/vm/run_remote_train.sh -- --total-iters 30 [other train flags]
#   ./scripts/vm/run_remote_train.sh attach              # re-attach only
#   ./scripts/vm/run_remote_train.sh status              # is it running?
#   ./scripts/vm/run_remote_train.sh stop                # kill the session
#   ./scripts/vm/run_remote_train.sh logs                # tail progress.md
#
# Anything after a literal ``--`` is passed straight through to
# ``train_alphazero.py``. Defaults below match the VM's 2-core / 4 GB
# profile (subprocess workers each load a full model copy; the buffer
# is bounded so we don't OOM on a long run).

set -euo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/_lib.sh"

# Defaults tuned for the 2-core / 4 GB VM. Buffer is the dominant memory
# cost (~340 MB / 50k transitions); 10k keeps RSS under ~1.5 GB headroom
# with the two-worker subprocess fan-out.
DEFAULT_TRAIN_ARGS=(
    --device cpu
    --self-play-workers 2
    --games-per-iter 8
    --batches-per-iter 100
    --batch-size 64
    --mcts-rollouts 30
    --buffer-capacity 10000
    --eval-every 5
    --eval-games 8
    --snapshot-every 5
)

usage() {
    cat <<EOF
Usage:
  $0 [run] -- [train_alphazero.py args]    start or attach the training session
  $0 attach                                 just re-attach to the existing session
  $0 status                                 print whether a session is alive
  $0 stop                                   kill the existing session
  $0 logs                                   cat the run's progress.md
  $0 --force-restart -- [args]              kill the existing session, then start fresh

Env:
  CATAN_VM_HOST=$CATAN_VM_HOST
  CATAN_VM_REPO=$CATAN_VM_REPO
  CATAN_TMUX_SESSION=$CATAN_TMUX_SESSION

VM-tuned default train flags (override via "-- <flags>"):
  ${DEFAULT_TRAIN_ARGS[*]}
EOF
}

# --------------------------------------------------------------------
# Subcommand dispatch
# --------------------------------------------------------------------

FORCE_RESTART=0
ACTION="run"
TRAIN_ARGS=()
seen_double_dash=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --force-restart) FORCE_RESTART=1; shift ;;
        --) seen_double_dash=1; shift; TRAIN_ARGS=("$@"); break ;;
        run|attach|status|stop|logs) ACTION="$1"; shift ;;
        *)
            if [[ $seen_double_dash -eq 0 ]]; then
                log "error: unknown argument: $1 (did you mean to pass '-- $1' to train_alphazero.py?)"
                usage
                exit 2
            fi
            ;;
    esac
done

# --------------------------------------------------------------------
# Helpers (run on VM via SSH)
# --------------------------------------------------------------------

_remote() {
    # Quote each arg for SSH so spaces / special chars round-trip cleanly.
    local quoted=()
    local a
    for a in "$@"; do
        quoted+=("$(_ssh_quote "$a")")
    done
    # shellcheck disable=SC2029 -- we want client-side expansion of the array
    ssh "$CATAN_VM_HOST" "${quoted[@]}"
}

session_alive() {
    _remote tmux has-session -t "$CATAN_TMUX_SESSION" >/dev/null 2>&1
}

kill_session() {
    if session_alive; then
        log "killing existing tmux session $CATAN_TMUX_SESSION on $CATAN_VM_HOST"
        _remote tmux kill-session -t "$CATAN_TMUX_SESSION"
    fi
}

# --------------------------------------------------------------------
# Action: status
# --------------------------------------------------------------------

if [[ "$ACTION" == "status" ]]; then
    if session_alive; then
        log "session '$CATAN_TMUX_SESSION' is ALIVE on $CATAN_VM_HOST"
        _remote tmux list-windows -t "$CATAN_TMUX_SESSION"
        exit 0
    fi
    log "no session named '$CATAN_TMUX_SESSION' on $CATAN_VM_HOST"
    exit 1
fi

# --------------------------------------------------------------------
# Action: stop
# --------------------------------------------------------------------

if [[ "$ACTION" == "stop" ]]; then
    kill_session
    exit 0
fi

# --------------------------------------------------------------------
# Action: attach
# --------------------------------------------------------------------

if [[ "$ACTION" == "attach" ]]; then
    if ! session_alive; then
        log "no session '$CATAN_TMUX_SESSION' on $CATAN_VM_HOST; nothing to attach to"
        exit 1
    fi
    log "attaching to '$CATAN_TMUX_SESSION' on $CATAN_VM_HOST (Ctrl-b d to detach)"
    exec ssh -t "$CATAN_VM_HOST" tmux attach-session -t "$CATAN_TMUX_SESSION"
fi

# --------------------------------------------------------------------
# Action: logs (cat the most recent progress.md)
# --------------------------------------------------------------------

if [[ "$ACTION" == "logs" ]]; then
    # The remote runs land under $CATAN_VM_REPO/runs/az_<timestamp>/.
    # Show the newest one's progress.md.
    log "tailing latest progress.md on $CATAN_VM_HOST"
    _remote bash -c "
        set -euo pipefail
        cd '$CATAN_VM_REPO' 2>/dev/null || cd \"\${CATAN_VM_REPO/#\\~/\$HOME}\"
        latest=\$(ls -dt runs/az_* 2>/dev/null | head -n1 || true)
        if [[ -z \"\$latest\" ]]; then
            echo 'no runs/az_* directory found on VM'
            exit 1
        fi
        echo \"=== \$latest/progress.md ===\"
        cat \"\$latest/progress.md\" 2>/dev/null || echo '(progress.md not yet written)'
    "
    exit $?
fi

# --------------------------------------------------------------------
# Action: run (the default)
# --------------------------------------------------------------------

if [[ $FORCE_RESTART -eq 1 ]]; then
    kill_session
fi

if session_alive; then
    log "session '$CATAN_TMUX_SESSION' already exists; attaching instead of starting a second run"
    log "(use '$0 --force-restart -- <args>' to discard the existing session)"
    exec ssh -t "$CATAN_VM_HOST" tmux attach-session -t "$CATAN_TMUX_SESSION"
fi

# Build the training command. User-supplied flags REPLACE the VM
# defaults so the user can override anything without having to re-list
# the whole default set. To mix-and-match, edit DEFAULT_TRAIN_ARGS or
# pass the full set explicitly.
if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
    TRAIN_ARGS=("${DEFAULT_TRAIN_ARGS[@]}")
fi

# Render the full train command as a single shell-safe string so tmux
# can be told to run it verbatim.
TRAIN_CMD_PARTS=("./venv/bin/python" "scripts/train_alphazero.py")
TRAIN_CMD_PARTS+=("${TRAIN_ARGS[@]}")

# Build a string with each arg single-quoted for the remote shell.
TRAIN_CMD=""
for a in "${TRAIN_CMD_PARTS[@]}"; do
    TRAIN_CMD+=" $(_ssh_quote "$a")"
done
TRAIN_CMD="${TRAIN_CMD# }"  # strip leading space

log "starting tmux session '$CATAN_TMUX_SESSION' on $CATAN_VM_HOST"
log "train cmd: $TRAIN_CMD"

# Remote bootstrap: cd into the repo, git pull, then launch the training
# inside the tmux session so it survives SSH disconnect. We send a
# heredoc so the remote shell can do its own multi-line work cleanly.
ssh "$CATAN_VM_HOST" bash -s -- \
    "$(_ssh_quote "$CATAN_VM_REPO")" \
    "$(_ssh_quote "$CATAN_TMUX_SESSION")" \
    "$(_ssh_quote "$TRAIN_CMD")" <<'REMOTE'
set -euo pipefail
REPO="$1"
SESSION="$2"
CMD="$3"
REPO="${REPO/#\~/$HOME}"

cd "$REPO"
echo "[run] git pull (fast-forward only)"
git pull --ff-only || {
    echo "[run] WARNING: git pull failed; continuing with existing checkout"
}

# Create a detached tmux session, send the train command + Enter. The
# session persists across SSH disconnects; the laptop can later attach
# via 'run_remote_train.sh attach'.
tmux new-session -d -s "$SESSION" -c "$REPO"
# Send the entire command as a single literal, then a newline to execute.
tmux send-keys -t "$SESSION" "$CMD" Enter
echo "[run] launched in tmux session '$SESSION'"
REMOTE

log "training launched. Useful follow-ups:"
log "  $0 status           # is it still running?"
log "  $0 logs             # latest progress.md"
log "  $0 attach           # interactive shell inside the tmux session"
log "  ./scripts/vm/sync_runs_from_vm.sh --dry-run   # preview rsync"

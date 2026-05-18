#!/usr/bin/env bash
# One-time bootstrap of the VM: clone the repo, create the venv, install
# the [rl] extra. Safe to re-run — every step checks for prior state and
# is a no-op when already done.
#
# Usage (from the laptop):
#   ./scripts/vm/setup_vm.sh [GIT_REMOTE_URL]
#
# GIT_REMOTE_URL defaults to the laptop's current origin. Override if
# the VM should clone from a different remote (eg a fork).

set -euo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/_lib.sh"

GIT_REMOTE_URL="${1:-$(cd "$CATAN_LOCAL_REPO" && git config --get remote.origin.url || true)}"
if [[ -z "$GIT_REMOTE_URL" ]]; then
    log "error: no git remote configured locally and none passed as arg 1"
    log "       usage: $0 GIT_REMOTE_URL"
    exit 2
fi

log "bootstrapping $CATAN_VM_HOST:$CATAN_VM_REPO from $GIT_REMOTE_URL"

# The remote script is single-quoted at the SSH boundary so the caller
# sees the literal command in /var/log/auth without re-expansion. Inside
# we re-expand with the safely-quoted URL we computed locally.
ssh "$CATAN_VM_HOST" bash -s -- "$(_ssh_quote "$CATAN_VM_REPO")" "$(_ssh_quote "$GIT_REMOTE_URL")" <<'REMOTE'
set -euo pipefail
REPO="$1"
URL="$2"
# Expand a leading ~ on the remote side (SSH doesn't do this for us).
REPO="${REPO/#\~/$HOME}"

if [[ ! -d "$REPO/.git" ]]; then
    echo "[setup] cloning $URL -> $REPO"
    git clone "$URL" "$REPO"
else
    echo "[setup] $REPO already a git checkout; skipping clone"
fi

cd "$REPO"

# Detect a partially-created venv: directory exists but pip is missing.
# This happens on Debian/Ubuntu hosts that have ``python3`` but not
# ``python3-venv`` installed — ``python3 -m venv`` creates the layout
# without ensurepip. Wipe and re-create rather than try to patch a
# half-built env.
if [[ -d venv && ! -x venv/bin/pip ]]; then
    echo "[setup] existing venv is broken (no venv/bin/pip); rebuilding"
    rm -rf venv
fi

if [[ ! -d venv ]]; then
    echo "[setup] creating venv at $REPO/venv"
    python3 -m venv venv
else
    echo "[setup] venv already exists; skipping create"
fi

# Fail loudly with a fix-it pointer if ensurepip still didn't run.
# Continuing past this would just blow up on ./venv/bin/pip a line later
# with a less obvious error.
if [[ ! -x venv/bin/pip ]]; then
    cat <<'EOF' >&2
[setup] ERROR: python3 -m venv produced a venv without pip.
       This usually means the python3-venv package isn't installed.
       On Debian / Ubuntu:
           sudo apt update && sudo apt install -y python3-venv python3-pip
       Then re-run scripts/vm/setup_vm.sh.
EOF
    exit 1
fi

# Always sync deps so re-running after pyproject changes keeps the VM
# environment current. Pip is idempotent for already-satisfied requirements.
echo "[setup] installing project + [rl] extra"
./venv/bin/pip install --upgrade pip wheel >/dev/null
./venv/bin/pip install -e ".[rl]"
echo "[setup] done"
REMOTE

log "VM ready. Next: ./scripts/vm/run_remote_train.sh -- --total-iters N [other flags]"

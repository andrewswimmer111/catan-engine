#!/usr/bin/env bash
# Shared config + helpers for the scripts/vm/ family.
#
# Sourced by run_remote_train.sh, sync_runs_from_vm.sh, install_cron.sh.
# All knobs are environment variables with sensible defaults so the
# scripts work out-of-the-box for the default Duke VM but stay
# overridable for other hosts.
#
# Conventions:
# - Strict bash everywhere (set -euo pipefail).
# - macOS bash 3.2 compatible (no associative arrays, no readarray).
# - Every script that sources this can rely on the variables below.

set -euo pipefail

# Default remote host + repo path. Override via env or .env wrapper.
: "${CATAN_VM_HOST:=al576@vcm-53452.vm.duke.edu}"
: "${CATAN_VM_REPO:=~/catan-engine}"

# Tmux session name on the VM. One session per concurrent run; if you
# want to run two trainings at once, override this to a second name.
: "${CATAN_TMUX_SESSION:=catan-train}"

# Where to put the rsync log on the local machine. macOS convention is
# ~/Library/Logs, which Console.app surfaces.
: "${CATAN_SYNC_LOG:=$HOME/Library/Logs/catan-sync.log}"

# Cron-side flock path. Same default for every invocation so two cron
# ticks can't run rsync concurrently.
: "${CATAN_SYNC_LOCK:=/tmp/catan-sync.lock}"

# Local repo root (absolute path). Resolved by the caller's $0; if a
# caller pre-sets it we trust it (useful for cron, which has no $0
# we'd recognise).
: "${CATAN_LOCAL_REPO:=}"

_caller_dir() {
    # Resolve the calling script's directory (works around macOS not
    # shipping GNU readlink -f). Output absolute path.
    local src="${BASH_SOURCE[1]:-$0}"
    local dir
    dir="$(cd "$(dirname "$src")" && pwd)"
    printf '%s\n' "$dir"
}

# When CATAN_LOCAL_REPO isn't pre-set, derive it from the caller's
# script location: scripts/vm/<caller>.sh -> repo root is two levels up.
_resolve_local_repo() {
    if [[ -n "${CATAN_LOCAL_REPO}" ]]; then
        return 0
    fi
    local caller_dir
    caller_dir="$(_caller_dir)"
    CATAN_LOCAL_REPO="$(cd "$caller_dir/../.." && pwd)"
}

_resolve_local_repo

# Quote a string for safe inclusion in an SSH-side command. Uses bash's
# built-in ``printf '%q'`` which produces output that round-trips
# through ``bash -c`` correctly for any input — including strings that
# contain both single quotes and spaces (the home-rolled ``'\''``
# pattern collapses on those because the outer single-quote span ends
# before the substituted segment reopens it).
_ssh_quote() {
    printf '%q' "$1"
}

# Print "[catan-vm] <msg>" to stderr so cron logs are scan-able.
log() {
    printf '[catan-vm] %s\n' "$*" >&2
}

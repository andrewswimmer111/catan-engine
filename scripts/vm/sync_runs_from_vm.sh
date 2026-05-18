#!/usr/bin/env bash
# Pull training artefacts from the VM into the laptop's local runs/.
#
# Designed for unattended / cron operation:
# - rsync flags tolerate flaky links (--partial / --append-verify / -z /
#   --timeout) so a dropped connection mid-transfer doesn't waste the
#   bytes already on disk.
# - Excludes things that don't add value on the laptop (__pycache__,
#   process locks, ephemeral .tmp files). Snapshots and TB events are
#   intentionally INCLUDED — the laptop is where you re-launch eval
#   from, and TB scalars are tiny.
# - Idempotent: rsync only copies what changed since the last sync.
#
# Usage:
#   ./scripts/vm/sync_runs_from_vm.sh             # real pull
#   ./scripts/vm/sync_runs_from_vm.sh --dry-run   # preview, no writes
#   ./scripts/vm/sync_runs_from_vm.sh --once      # alias for the default

set -euo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/_lib.sh"

DRY_RUN=0
case "${1:-}" in
    -h|--help)
        cat <<EOF
Usage: $0 [--dry-run]

Pull \$CATAN_VM_HOST:\$CATAN_VM_REPO/runs/ -> \$CATAN_LOCAL_REPO/runs/.

Env (current values):
  CATAN_VM_HOST=$CATAN_VM_HOST
  CATAN_VM_REPO=$CATAN_VM_REPO
  CATAN_LOCAL_REPO=$CATAN_LOCAL_REPO
  CATAN_SYNC_LOG=$CATAN_SYNC_LOG

Excludes (anything matching is skipped at the rsync level):
  __pycache__/, *.pyc, *.tmp, .DS_Store, *.lock
EOF
        exit 0
        ;;
    --dry-run|-n) DRY_RUN=1 ;;
    --once|"") ;;
    *)
        log "error: unknown argument: $1"
        log "       usage: $0 [--dry-run]"
        exit 2
        ;;
esac

LOCAL_RUNS_DIR="$CATAN_LOCAL_REPO/runs"
mkdir -p "$LOCAL_RUNS_DIR"

# rsync flags rationale:
#   -a                preserve timestamps / perms (so partial-checks work)
#   -z                compress in transit; cheap CPU win on a slow link
#   --partial         keep partial files for resumption
#   --append-verify   resume large files (snapshots / final.pt) with checksum
#                     verification at the boundary
#   --timeout=60      bail on stalled connections rather than hanging
#                     until cron's next tick walks over us
#   --info=...        machine-parseable totals at the end; quiet otherwise
#   --exclude         skip noise that doesn't help local eval
RSYNC_FLAGS=(
    -a -z
    --partial
    --append-verify
    --timeout=60
    --info=stats2,progress0
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='*.tmp'
    --exclude='.DS_Store'
    --exclude='*.lock'
)

if [[ $DRY_RUN -eq 1 ]]; then
    RSYNC_FLAGS+=(--dry-run --itemize-changes)
    log "DRY RUN — no files will be written locally"
fi

# Trailing slash on the source means "copy contents of runs/" rather than
# "copy the runs/ directory itself" — keeps the layout flat on the laptop.
SRC="$CATAN_VM_HOST:$CATAN_VM_REPO/runs/"
DST="$LOCAL_RUNS_DIR/"

log "rsync $SRC -> $DST"
rsync "${RSYNC_FLAGS[@]}" "$SRC" "$DST"
log "sync complete"

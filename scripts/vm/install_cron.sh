#!/usr/bin/env bash
# Install (or remove) the hourly rsync cron entry.
#
# Idempotent: re-running with the same arguments leaves the crontab in
# the same state. The entry is fingerprinted with a tag comment so we
# can find + remove our own line without touching other entries.
#
# Usage:
#   ./scripts/vm/install_cron.sh                # install hourly sync
#   ./scripts/vm/install_cron.sh --remove       # uninstall
#   ./scripts/vm/install_cron.sh --print        # print the line we would add
#
# Cron schedule: minute 10 of every hour. ``flock -n`` ensures a
# long-running sync (large rsync over a slow link) doesn't overlap
# with the next hour's tick.

set -euo pipefail

# shellcheck source=./_lib.sh
source "$(cd "$(dirname "$0")" && pwd)/_lib.sh"

MODE="install"
case "${1:-}" in
    -h|--help)
        cat <<EOF
Usage: $0 [--remove|--print]

Install the hourly rsync cron entry (default), uninstall it, or just
print what we would add. The line is tagged so we can find + remove
our own entry without affecting others.

Schedule:        minute 10 every hour
Lock:            $CATAN_SYNC_LOCK (via flock -n; tick skipped if held)
Log:             $CATAN_SYNC_LOG (appended; no rotation)
Source script:   $CATAN_LOCAL_REPO/scripts/vm/sync_runs_from_vm.sh
EOF
        exit 0
        ;;
    --remove) MODE="remove" ;;
    --print) MODE="print" ;;
    "") ;;
    *)
        log "error: unknown argument: $1 (use --help)"; exit 2 ;;
esac

# A tag comment makes the entry easy to locate without grepping for
# the (long) absolute path. Multiple installs of the same tag are
# de-duped before write.
CRON_TAG="# catan-sync (managed by scripts/vm/install_cron.sh)"

SYNC_SCRIPT="$CATAN_LOCAL_REPO/scripts/vm/sync_runs_from_vm.sh"
if [[ ! -x "$SYNC_SCRIPT" ]]; then
    log "error: $SYNC_SCRIPT is not executable. Run: chmod +x $SYNC_SCRIPT"
    exit 2
fi

# flock blocks concurrent runs; -n = exit immediately if the lock is
# held (so a stuck rsync doesn't pile up hourly invocations). The
# subshell with `( cd && exec )` keeps cron's $HOME-less PATH from
# tripping over relative paths in the sync script.
LOG_DIR="$(dirname "$CATAN_SYNC_LOG")"
mkdir -p "$LOG_DIR"

CRON_LINE="10 * * * * /usr/bin/flock -n $CATAN_SYNC_LOCK $SYNC_SCRIPT >> $CATAN_SYNC_LOG 2>&1"

if [[ "$MODE" == "print" ]]; then
    printf '%s\n%s\n' "$CRON_TAG" "$CRON_LINE"
    exit 0
fi

# Read the current crontab (treat "no crontab" as empty), strip any
# existing catan-sync entry, then either append the new one or stop.
TMP="$(mktemp -t catan-cron.XXXXXX)"
trap 'rm -f "$TMP"' EXIT

# crontab -l fails on hosts that have never had a crontab — that's OK.
crontab -l 2>/dev/null > "$TMP" || true

# Filter out the tag line + the line immediately following it
# (which is OUR entry — we control the format above so this is safe).
FILTERED="$(mktemp -t catan-cron-filtered.XXXXXX)"
awk -v tag="$CRON_TAG" '
    $0 == tag { skip = 2; next }
    skip > 0  { skip--; next }
    { print }
' "$TMP" > "$FILTERED"

case "$MODE" in
    remove)
        if ! crontab "$FILTERED"; then
            log "error: failed to write filtered crontab"
            rm -f "$FILTERED"
            exit 1
        fi
        rm -f "$FILTERED"
        log "removed catan-sync entry from crontab"
        ;;
    install)
        {
            cat "$FILTERED"
            printf '%s\n%s\n' "$CRON_TAG" "$CRON_LINE"
        } | crontab -
        rm -f "$FILTERED"
        log "installed hourly sync (minute 10) -> $CATAN_SYNC_LOG"
        log "verify with: crontab -l | grep -A1 catan-sync"
        ;;
esac

# Remote training on the Duke VM

Workflow for offloading AlphaZero training to
`al576@vcm-53452.vm.duke.edu` (2 cores, 4 GB RAM) and pulling artefacts
back to the laptop hourly. All commands assume you're in the repo root
on the laptop unless stated otherwise.

## What ships where

| Direction       | Mechanism | Carries                                           |
| --------------- | --------- | ------------------------------------------------- |
| laptop → git    | `git push` | code, configs, tests                              |
| git → VM        | `git pull` (auto, in `run_remote_train.sh`) | code |
| VM → laptop     | `rsync` (manual + hourly cron)              | `runs/` (TB events, snapshots, `progress.md`, `elo.json`, `final.pt`) |

`runs/` is gitignored — no run artefacts will ever land in a commit.

## One-time setup

1. **SSH key on the VM** (skip if `ssh al576@vcm-53452.vm.duke.edu`
   already works without a password):

   ```bash
   ssh-copy-id al576@vcm-53452.vm.duke.edu
   ```

2. **Push the latest code to a remote the VM can reach.** The bootstrap
   script defaults to your local `origin` URL. If your `origin` is on
   GitHub and you don't have SSH keys to it on the VM, switch the URL
   to HTTPS or set `GIT_REMOTE_URL` explicitly when calling
   `setup_vm.sh`.

3. **Bootstrap the VM** (clones the repo, makes a venv, installs `.[rl]`):

   ```bash
   ./scripts/vm/setup_vm.sh
   ```

   Idempotent — safe to re-run after `pyproject.toml` changes; pip just
   re-syncs the resolved env.

## Starting / monitoring / stopping a run

The `run_remote_train.sh` script is the single entry point. It runs
inside a `tmux` session named `catan-train` so the run survives SSH
disconnects. Subcommands:

```bash
# Start (or attach, if a session already exists). User flags go after `--`.
./scripts/vm/run_remote_train.sh -- --total-iters 30 --seed 0

# Status: prints whether the tmux session is alive on the VM.
./scripts/vm/run_remote_train.sh status

# Print the latest progress.md from the VM (no full SSH attach needed).
./scripts/vm/run_remote_train.sh logs

# Attach the tmux session interactively (Ctrl-b d to detach).
./scripts/vm/run_remote_train.sh attach

# Kill the current session (the run dies; partial artefacts stay on disk).
./scripts/vm/run_remote_train.sh stop

# Discard the existing session and start a fresh run with new flags.
./scripts/vm/run_remote_train.sh --force-restart -- --total-iters 50
```

### VM-tuned defaults

When you don't pass `-- <flags>`, the script uses a profile tuned for
the 2-core / 4 GB VM:

```
--device cpu \
--self-play-workers 2 \
--games-per-iter 8 \
--batches-per-iter 100 \
--batch-size 64 \
--mcts-rollouts 30 \
--buffer-capacity 10000 \
--eval-every 5 --eval-games 8 \
--snapshot-every 5
```

Override semantics: passing `-- <anything>` **replaces** the defaults
entirely — paste the full flag set you want. The replay buffer is the
dominant memory cost (~34 MB per 1 000 transitions); 10 k caps RSS at
~1.5 GB with two subprocess workers (each carries its own ~6 MB model
copy). Push capacity up only if you've measured headroom.

## Pulling artefacts back

Manual one-shot pull:

```bash
# Preview first — no writes, just lists what rsync would do.
./scripts/vm/sync_runs_from_vm.sh --dry-run

# Real pull.
./scripts/vm/sync_runs_from_vm.sh
```

What syncs:

- `runs/<run-dir>/tb/` — TensorBoard event files.
- `runs/<run-dir>/snapshots/iter_N.pt` — every snapshot the trainer
  wrote.
- `runs/<run-dir>/{progress.md, config.json, elo.json, final.pt}`.

What's excluded (would just bloat the local copy):

- `__pycache__/`, `*.pyc` — Python bytecode generated on the VM.
- `*.tmp`, `*.lock`, `.DS_Store` — process-local junk.

The rsync flags (`-a -z --partial --append-verify --timeout=60`) are
tuned for a flaky link: partial files are resumed on the next tick
rather than re-transferred, and a stalled connection bails after 60 s
instead of hanging.

## Hourly cron sync

Install:

```bash
./scripts/vm/install_cron.sh
```

This adds (or replaces) a single tagged entry in your user crontab:

```
# catan-sync (managed by scripts/vm/install_cron.sh)
10 * * * * /usr/bin/flock -n /tmp/catan-sync.lock <ABSOLUTE_PATH>/sync_runs_from_vm.sh >> ~/Library/Logs/catan-sync.log 2>&1
```

- `flock -n` ensures two ticks can't run rsync at the same time; if the
  previous tick is still going, the new one exits immediately.
- Output is appended to `~/Library/Logs/catan-sync.log` (no rotation —
  rotate manually or via `newsyslog.d` if it grows).

To uninstall:

```bash
./scripts/vm/install_cron.sh --remove
```

To inspect:

```bash
crontab -l | grep -A1 catan-sync
tail -n 40 ~/Library/Logs/catan-sync.log
```

## Verification checklist

After first-time setup, run through this list once:

1. `ssh al576@vcm-53452.vm.duke.edu echo ok` — prints `ok` with no
   password prompt.
2. `./scripts/vm/setup_vm.sh` — finishes with `[catan-vm] VM ready`.
3. `./scripts/vm/run_remote_train.sh -- --total-iters 2 --eval-every 0
   --snapshot-every 1` — short calibration run.
4. `./scripts/vm/run_remote_train.sh status` — reports the session
   alive while the calibration runs (~5–10 min on the VM).
5. `./scripts/vm/run_remote_train.sh logs` — `progress.md` shows
   `Iteration: K / 2`, status `running` or `done`.
6. `./scripts/vm/sync_runs_from_vm.sh --dry-run` — lists files but
   doesn't write.
7. `./scripts/vm/sync_runs_from_vm.sh` — `runs/az_*/` appears under
   the local repo.
8. `cat runs/az_*/progress.md` — locally identical to the VM's copy.
9. `./scripts/vm/install_cron.sh` — confirms install.
10. `crontab -l | grep catan-sync` — shows exactly one entry.
11. Wait until the next minute-10 tick, then `tail
    ~/Library/Logs/catan-sync.log` — shows `[catan-vm] rsync
    ... -> ...` followed by `sync complete`.

## Operational gotchas

- **Don't run `setup_vm.sh` while a training run is active.** Pip will
  walk over packages the running Python is still importing; the run
  may crash mid-iteration. Stop the session first.
- **Force-restart loses the buffer.** The replay buffer is in-memory
  only; `--force-restart` (or any session kill) starts the next run
  with an empty buffer. If you want to continue training the same
  network, use `--init-from runs/az_*/final.pt` so weights warm-start
  even though the buffer reseeds.
- **The VM has no GPU.** `--device mps` will fail there; the default
  profile pins `--device cpu`.
- **rsync direction is VM → laptop only.** If you change code on the
  VM by hand, those edits will not flow back to git. Always edit on
  the laptop, commit + push, and let `run_remote_train.sh` pick the
  changes up on its next launch.

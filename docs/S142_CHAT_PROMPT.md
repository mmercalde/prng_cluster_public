# S142 Session Chat Prompt
**Date:** 2026-03-14
**Previous Session:** S141
**Last Commits:** `fda118d` → `b2e6073` → `8c7a432` → `4bff183`

---

## System State

Distributed PRNG functional mimicry system. Modular, configurable, ML & AI
compatible. Not lottery-specific — designed for any deterministic PRNG output.

**Cluster:** Zeus (2×RTX 3080Ti, `rzeus`) + 2 rigs (8×RX6600 each)
**Project root:** `~/distributed_prng_analysis/`
**Public repo:** `github.com/mmercalde/prng_cluster_public`
**Dashboard:** `http://45.32.131.224:5002`

---

## What Was Completed in S140/S141

### S140 — Seed Coverage Tracker (`cbd95c0`)
- `database_system.py` — `get_next_seed_start()`
- `window_optimizer.py` — `--seed-start` CLI + coverage write-back
- `agents/watcher_agent.py` — Step 1 preflight advances seed range
- `agent_manifests/window_optimizer.json` — `seed_start` in manifest
- Coverage DB manually seeded: `java_lcg` 0→5,000,000 logged

### S140b — Trial History + Warm-Start + Downstream Feedback (`c6fde66`)
- `database_system.py` — `step1_trial_history` table + 3 methods
- `window_optimizer_bayesian.py` — per-trial write in callback + dynamic warm-start
- `window_optimizer.py` — thread `trial_history_context`
- `window_optimizer_integration_final.py` — build context at source
- `agents/watcher_agent.py` — warm-start read at Step 1 relaunch
- `chapter_13_orchestrator.py` — downstream score write-back
- `agent_manifests/window_optimizer.json` — warm_start params
- `agent_grammars/chapter_13.gbnf` — `step1_relaunch` scope
- `strategy_advisor.gbnf` — `steps_0_1`, `step_1_only` scopes

### S140b warm-start manifest fix (`9dc369d`)
- Removed `warm_start_*` from `default_params` — were being passed as CLI args
- Added `_INTERNAL_ONLY_PARAMS` strip set in WATCHER CLI builder

### S140b-NP2 fix2 (`4bff183`) — CURRENT HEAD
- `window_optimizer_integration_final.py` — trial history write + dynamic
  warm-start for `n_parallel > 1` production path
- Reads `_warm_start_params` directly from DB in `n_parallel > 1` block
- Per-trial write in `_worker_obj` with child-local SQLite connection
- TB-approved, TB corrections applied (variable naming + seed_range_end)

---

## Two Known Issues for S142

### Issue 1 — NULL sessions in trial history (P1)

The `save_best_so_far` callback in `window_optimizer_bayesian.py` fires during
`n_parallel=2` runs and writes rows with `session=NULL` (uses `trial.params`
which has `session_idx` integer, not `time_of_day` string). These NULL rows
are written first via `INSERT OR IGNORE`, so NP2 writes with correct sessions
are silently ignored for the same `(run_id, trial_number)`.

**Fix:** One guard in `save_best_so_far` callback — skip trial history write
when `trial_history_context.get('n_parallel_gt1')` is True. Set that flag in
`_trial_history_ctx` when `n_parallel > 1` in `optimize_window`.

**Files:** `window_optimizer_bayesian.py` + `window_optimizer_integration_final.py`

### Issue 2 — `No existing session` tmux error (P2)
`tmux ls` shows no sessions on Zeus. Persistent workers require tmux.
Some sieve jobs fail and fall back — throughput reduced.

**Fix:** Investigate if persistent workers can run without tmux, or add tmux
session creation to the launch sequence.

---

## DB State

```
prng_analysis.db:
  exhaustive_progress: java_lcg covered 0→5,000,000
  step1_trial_history: ~7 rows, mix of correct/NULL sessions
```

Best trial so far: `T8: W4_O23_evening score=2,133,473`

---

## Session Start Checklist

```bash
# 1. Clone public repo
git clone --depth 1 https://github.com/mmercalde/prng_cluster_public.git /home/claude/prng_cluster_public

# 2. Check DB state
ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
  PYTHONPATH=. python3 -c \"
import sqlite3
conn = sqlite3.connect('prng_analysis.db')
rows = conn.execute('SELECT trial_number, window_size, offset, session, trial_score FROM step1_trial_history ORDER BY trial_score DESC LIMIT 5').fetchall()
print(f'Trial history: {len(rows)} rows')
for r in rows: print(f'  T{r[0]}: W{r[1]}_O{r[2]}_{r[3]} score={r[4]:.0f}')
cov = conn.execute('SELECT prng_type, seed_range_end FROM exhaustive_progress ORDER BY seed_range_end DESC LIMIT 3').fetchall()
print('Coverage:', cov)
conn.close()
\""

# 3. Check current files
ssh rzeus "ls -la ~/distributed_prng_analysis/optimal_window_config.json \
  ~/distributed_prng_analysis/bidirectional_survivors.json 2>/dev/null || echo no_output_files"
```

---

## Standard Launch Command (confirmed working)

```bash
ssh rzeus "cd ~/distributed_prng_analysis && \
  source ~/venvs/torch/bin/activate && \
  PYTHONPATH=. python3 -c 'from agents.safety import clear_halt; clear_halt(); print(\"cleared\")' && \
  rm -f optimal_window_config.json bidirectional_survivors.json && \
  nohup bash -c 'source ~/venvs/torch/bin/activate && PYTHONPATH=. python3 \
  agents/watcher_agent.py --run-pipeline --start-step 0 --end-step 1 \
  --params \"{\\\"lottery_file\\\":\\\"daily3.json\\\",\\\"window_trials\\\":100,\
\\\"max_seeds\\\":5000000,\\\"enable_pruning\\\":true,\\\"n_parallel\\\":2,\
\\\"use_persistent_workers\\\":true,\\\"worker_pool_size\\\":4,\
\\\"seed_cap_nvidia\\\":5000000,\\\"seed_cap_amd\\\":2000000}\" \
  > ~/distributed_prng_analysis/logs/step01_s142.log 2>&1' &"
```

**Monitor:**
```bash
ssh rzeus "strings ~/distributed_prng_analysis/logs/step01_s142.log | \
  grep '\[P[0-9]\] Trial\|COVERAGE\|WARM_START\|trial-history' | tail -15"
```

**Verify DB after trials:**
```bash
ssh rzeus "cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \
  PYTHONPATH=. python3 -c \"
import sqlite3
conn = sqlite3.connect('prng_analysis.db')
rows = conn.execute('''SELECT trial_number, window_size, offset, session,
                              trial_score FROM step1_trial_history
                       ORDER BY trial_score DESC LIMIT 10''').fetchall()
print(f'{len(rows)} rows:')
for r in rows: print(f'  T{r[0]}: W{r[1]}_O{r[2]}_{r[3]} score={r[4]:.0f}')
conn.close()
\""
```

---

## P1 TODO for S142

1. Fix NULL-session collision (Issue 1) — one guard in callback
2. Verify fix with 10-trial debug run
3. Confirm all sessions populate correctly
4. Run full 100-trial run after fix confirmed
5. Wire `dispatch_selfplay()` into WATCHER post-Step-6
6. Wire Chapter 13 orchestrator into WATCHER daemon

---

## Architecture Invariants
- Step order static: 0→1→2→3→4→5→6
- Authority: Chapter 13 decides, WATCHER executes
- GPU isolation: parent never initializes CUDA before NN subprocess spawn
- Manifest param governance: every CLI param in `default_params` or dropped
- `bidirectional_survivors_binary.npz` never gitignored
- Different `seed_start` = fresh Optuna study
- Never restore from backup — fix forward
- Always dual-push: `git push origin main && git push public main`
- PYTHONPATH=. required for all WATCHER commands

# SESSION CHANGELOG — S141
**Date:** 2026-03-14
**Session:** S141
**Status:** OPEN — debug run completed, known issues identified
**Commits:** `fda118d` → `b2e6073` → `8c7a432` → `4bff183`

---

## Summary

S141 focused on deploying S140/S140b patches, debugging launch failures, and
implementing + partially validating the S140b-NP2 fix for the n_parallel>1
production path. The system is running correctly at a high level but two
remaining issues were identified for S142.

---

## Patches Deployed This Session

### `fda118d` — S140b checkpoint (pre-NP2 fix)
Files committed to both remotes before NP2 work began.

### `b2e6073` — S140b-NP2 first attempt (broken)
- `window_optimizer_integration_final.py` — referenced `trial_history_context`
  which doesn't exist in `optimize_window` scope → NameError on launch

### `8c7a432` — S140b-NP2 fix: read from DB directly
- Fixed Patch A: `_warm_start_params` now reads from `db.get_best_step1_params()`
  directly inside the `n_parallel > 1` block — no reference to
  `trial_history_context`
- Fixed Patch D: `_warm_start_params` → `warm_start_params` variable name
  inconsistency → NameError on warm-start enqueue (broken again)

### `4bff183` — S140b-NP2 fix2: consistent naming
- Fixed Patch D: warm-start enqueue block uses `_warm_start_params` (main
  process variable) not `warm_start_params` (worker parameter)
- This is the current live version on both remotes ✅

---

## Launch Failures Encountered and Resolved

| Error | Cause | Fix |
|-------|-------|-----|
| `--warm-start-* None` unrecognized args | `warm_start_*` in manifest `default_params` → passed as CLI args | `apply_s140b_warmstart_fix.py` — removed from manifest, added strip set in WATCHER |
| `PIPELINE HALTED` on every launch | Previous run left halt file at `/tmp/agent_halt` | `from agents.safety import clear_halt; clear_halt()` with `PYTHONPATH=.` |
| `NameError: trial_history_context` | `trial_history_context` not in `optimize_window` scope | Read from DB directly in NP2 block |
| `NameError: warm_start_params` | Variable name inconsistency between main/worker | Use `_warm_start_params` in main process |
| Step 0 skipped (freshness gate) | `trse_context.json` from earlier still fresh | Expected behavior — delete `trse_context.json` for true fresh Step 0 |
| Two watcher processes running | Launch command ran twice from combined commit+launch command | `kill` old PIDs before relaunching |

---

## Current State After S141

### What's confirmed working
- Coverage tracker: `[COVERAGE]` fires, seed_start advances to 5,000,000 ✅
- Warm-start: `[WARM_START]` fires, reads from DB ✅
- NP2 trial writes: sessions populating correctly for n_parallel=2 trials ✅
- No fatal write errors ✅
- `optimal_window_config.json` produced ✅

### Known issues for S142

**Issue 1 — n_parallel=1 callback writing NULL-session rows**
The `save_best_so_far` Optuna callback in `window_optimizer_bayesian.py` fires
during n_parallel=2 runs and writes trial history rows with `session=NULL`
(because `trial.params` has `session_idx` integer, not `time_of_day` string).
These NULL rows are written first via `INSERT OR IGNORE`, so subsequent NP2
writes with correct session values are silently ignored for the same trial.

**Fix:** Add one guard in `save_best_so_far` callback:
```python
# Skip trial history write in n_parallel>1 — NP2 path handles it
if trial_history_context and trial_history_context.get('n_parallel_gt1'):
    pass  # skip
else:
    # existing write logic
```
Set `'n_parallel_gt1': True` in `_trial_history_ctx` when `n_parallel > 1`.

**Issue 2 — `No existing session` tmux error on persistent workers**
`tmux ls` shows no sessions on Zeus. Persistent workers require a tmux session
to be running. This causes some sieve jobs to fail with
`Remote job failed: No existing session` and fall back gracefully — but
throughput is reduced on affected partitions.

**Fix:** Ensure tmux session exists before launching persistent workers, or
investigate if persistent workers can be launched without tmux dependency.

### DB state
- `step1_trial_history` has rows from multiple runs
- Some rows have correct sessions (T7, T8 from NP2 path)
- Some rows have NULL sessions (from n_parallel=1 callback)
- `exhaustive_progress` has `java_lcg` covered 0→5,000,000

---

## Deploy Sequence Reference (for S142)

```bash
# Standard launch (confirmed working)
ssh rzeus "cd ~/distributed_prng_analysis && \
  source ~/venvs/torch/bin/activate && \
  PYTHONPATH=. python3 -c 'from agents.safety import clear_halt; clear_halt()' && \
  rm -f optimal_window_config.json bidirectional_survivors.json && \
  nohup bash -c 'source ~/venvs/torch/bin/activate && \
  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 0 --end-step 1 \
  --params \"{\\\"lottery_file\\\":\\\"daily3.json\\\",\\\"window_trials\\\":100,\
\\\"max_seeds\\\":5000000,\\\"enable_pruning\\\":true,\\\"n_parallel\\\":2,\
\\\"use_persistent_workers\\\":true,\\\"worker_pool_size\\\":4,\
\\\"seed_cap_nvidia\\\":5000000,\\\"seed_cap_amd\\\":2000000}\" \
  > ~/distributed_prng_analysis/logs/step01_s142.log 2>&1' &"
```

---

## TODO Updates

**COMPLETED this session:**
- ✅ S140 seed coverage tracker deployed and working
- ✅ S140b trial history + warm-start deployed
- ✅ S140b-NP2 fix deployed (n_parallel>1 path)
- ✅ warm_start_* manifest fix deployed
- ✅ Sessions populating in trial history for NP2 path

**P1 for S142:**
- [ ] Fix n_parallel=1 callback NULL-session collision with NP2 writes
- [ ] Investigate tmux dependency for persistent workers
- [ ] Run full 100-trial clean run after fixes confirmed
- [ ] Verify `bidirectional_survivors.json` produced and NPZ converted
- [ ] Wire `dispatch_selfplay()` into WATCHER post-Step-6
- [ ] Wire Chapter 13 orchestrator into WATCHER daemon

**Files to upload to Claude Project:**
- `SESSION_CHANGELOG_20260314_S141.md`

# SESSION CHANGELOG — S142
**Date:** 2026-03-14
**Session:** S142
**Status:** COMPLETE
**Commits:** `4bff183` → `b5ea9f7` → `a2210ad` → `51aed27`

---

## Summary

S142 resolved the `step1_trial_history` missing rows bug in the NP2
(n_parallel=2) execution path. The root cause was architectural: the NP2
block was not terminal, and `_worker_obj` could never observe all Optuna
trials. Three patch rounds were deployed, escalating to TB twice. Final
resolution: canonical trial history is now written from the shared Optuna
study via backfill after all partition workers complete.

---

## Patches Deployed

### S142 P1 — NULL-session collision guard (`b5ea9f7`)
**Files:** `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py`

`save_best_so_far` callback was writing NULL-session rows via `INSERT OR IGNORE`
during NP2 runs, blocking subsequent correct NP2 writes for the same
`(run_id, trial_number)` PK.

Fix: added `n_parallel_gt1: n_parallel > 1` flag to `_trial_history_ctx` in
`window_optimizer_integration_final.py`. Guard in `save_best_so_far` skips
the history write when `n_parallel_gt1=True`.

**Result:** NULL sessions = 0 ✅

---

### S142 P2 — SQLite retry + WAL mode (deployed, partially effective)
**Files:** `window_optimizer_integration_final.py`, `database_system.py`

Attempted fixes for missing rows:
- 3-attempt retry with 0.1–0.5s jitter in `_worker_obj` write block
- `PRAGMA journal_mode=WAL` in `database_system.py:init_database()`

**Result:** NULL sessions = 0, but still ~4/10 rows. WAL and retry did not
fix the missing rows — root cause was not SQLite contention.

---

### S142-TB1 — Partition-scoped run_id (`b5ea9f7`)
**File:** `window_optimizer_integration_final.py`

TB diagnosis: `INSERT OR IGNORE` on `UNIQUE(run_id, trial_number)` with
identical `run_id` across both partitions caused ~50% silent drops. Both
P0 and P1 used `run_id=f"step1_{prng_base}_{seed_start}"`.

Fix: `run_id=f"step1_{prng_base}_{seed_start}_p{partition_idx}"`

**Result:** P0/P1 rows no longer collide, but still only ~4/8 COMPLETE rows.

---

### S142-TB2 — NP2 architectural diagnosis

TB ruling: The NP2 block is not terminal. After it completes, the
single-process `OptunaBayesianSearch.search()` path runs and completes the
remaining Optuna trials. `_worker_obj` only runs for a subset of trials by
design. Canonical `step1_trial_history` must come from the final shared
study object, not from per-partition execution side effects.

P1 ran 0 trials in some runs because shared Optuna storage does not reserve
per-worker trial quotas — the remaining budget was consumed elsewhere.

---

### S142-B — NP2 terminal + canonical backfill (`a2210ad`)
**File:** `window_optimizer_integration_final.py`
**Patcher:** `apply_s142b_np2_terminal.py`

Three patches:

1. **Backfill** `step1_trial_history` from `_fin_study.trials` after all
   partition workers complete. Iterates all COMPLETE trials, resolves
   session string from `session_idx`, writes with
   `run_id=f"step1_{prng_base}_{seed_start}_backfill"`. Single source of
   truth covering all trials regardless of execution path.

2. **NP2 terminal flag**: `_np2_complete = n_parallel > 1` set after NP2
   block. Guards single-process `survivor_accumulator` reinitialization and
   `optimizer.optimize()` call with `if not _np2_complete`.

3. **Branch logging**: `[NP2] EXIT NP2 PATH`, `[NP2] Single-process search
   path SKIPPED`, `[SINGLE] ENTER SINGLE-PROCESS SEARCH PATH`

**Result:** 7-8 rows written, 0 NULL sessions, NP2 terminal ✅
But: duplicate rows — `_p0`/`_p1` from `_worker_obj` + `_backfill` from
canonical backfill.

---

### S142-C — Remove _worker_obj writes, clean run_id (`51aed27`)
**File:** `window_optimizer_integration_final.py`, `prng_analysis.db`
**Patcher:** `apply_s142c_remove_worker_writes.py`

TB Option A ruling: `_worker_obj` is an incomplete execution path and must
not write to the canonical history table. Backfill is the only writer.

1. Removed entire `_worker_obj` retry+write block (-42 lines)
2. Dropped `_backfill` suffix from backfill `run_id`
   → `run_id=f"step1_{prng_base}_{int(seed_start)}"`
3. DB cleanup: deleted all rows with `_p0`, `_p1`, `_backfill` run_ids
   (14 rows removed)

**Final verified result:**
```
5 rows:
  step1_java_lcg_45000000  T0: sess=midday,evening score=84
  step1_java_lcg_45000000  T3: sess=evening score=2334143
  step1_java_lcg_45000000  T5: sess=midday score=0
  step1_java_lcg_45000000  T7: sess=midday,evening score=1828656
  step1_java_lcg_45000000  T9: sess=midday,evening score=0
NULL sessions: 0 (should be 0)
```
5 COMPLETE rows, 5 PRUNED correctly absent, no duplicates ✅

---

## Architecture Changes

### step1_trial_history — canonical design (post-S142)
- **Single writer:** backfill from `_fin_study.trials` after NP2 completes
- **One row per logical Optuna trial** (COMPLETE only; PRUNED absent)
- **run_id:** `step1_{prng_type}_{seed_start}` — no partition or provenance suffix
- **Unique constraint:** `(run_id, trial_number)` — clean, no collisions possible
- **`_worker_obj`** no longer writes to canonical history table

### NP2 execution flow (post-S142)
```
optimize_window()
  ├─ NP2 block (n_parallel > 1)
  │   ├─ launch P0, P1 partition workers
  │   ├─ join workers, merge survivor_accumulator
  │   ├─ load _fin_study, assemble results
  │   ├─ optimizer.save_results()
  │   ├─ [S142-B] backfill step1_trial_history from _fin_study
  │   └─ set _np2_complete = True
  ├─ shared dedup+save survivor block (both paths)
  │   ├─ deduplicate survivors
  │   ├─ save bidirectional_survivors.json
  │   └─ convert to NPZ
  └─ [S142-B] single-process search SKIPPED when _np2_complete=True
```

### WAL mode (permanent)
`database_system.py:init_database()` now sets `PRAGMA journal_mode=WAL`
on first connection. Sticky — all subsequent connections use WAL. Benefit
for any future concurrent writes.

---

## DB State After S142
```
prng_analysis.db:
  step1_trial_history: 5 rows (seed_start=45M run)
  exhaustive_progress: java_lcg covered 0→45,000,000
  journal_mode: WAL
```

---

## Issues Investigated and Ruled Out
- SQLite lock contention (WAL + retry deployed — did not fix root cause)
- INSERT OR IGNORE collision on same run_id (partition-scoped run_id — did
  not fix root cause, but was a real secondary bug)
- Stdout buffering hiding _worker_obj prints (confirmed — not the root cause)
- PRUNED trials causing missing rows (verified — 5 PRUNED = 5 absent rows,
  correct behavior)

---

## TODO Updates

**COMPLETED this session:**
- ✅ NULL-session collision guard (save_best_so_far callback)
- ✅ WAL mode on prng_analysis.db
- ✅ NP2 terminal — single-process path skipped when NP2 ran
- ✅ Canonical trial history backfill from shared Optuna study
- ✅ _worker_obj writes removed from canonical history table
- ✅ Clean run_id (no partition/provenance suffixes)
- ✅ DB duplicate rows cleaned

**Carry-forward (unchanged from S141):**
- [ ] 200-trial full Step 1 run (best so far: 17,247 bidirectional survivors)
- [ ] Investigate tmux dependency for persistent workers (Issue 2)
- [ ] Wire `dispatch_selfplay()` into WATCHER post-Step-6
- [ ] Wire Chapter 13 orchestrator into WATCHER daemon
- [ ] S110 root cleanup (884 files in project root)
- [ ] sklearn warnings in Step 5
- [ ] Remove dead CSV writer from coordinator.py
- [ ] Regression diagnostics gate = True
- [ ] S103 Part 2
- [ ] Phase 9B.3 (deferred)

---

## Files to Upload to Claude Project
- `SESSION_CHANGELOG_20260314_S142.md`

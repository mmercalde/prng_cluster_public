# Session Changelog — S138 / S139
**Date:** 2026-03-13
**Status:** OPEN — 200-trial fresh study running, TRSE Rule A active for first time
**Commits:** `3624e3c` → `889854f` → `4849bff` → `7d035c6` → `25cc2de`
**Both remotes:** origin + public synced at `25cc2de`

---

## Summary

S138 diagnosed and fixed the NPZ pipe deadlock that had prevented all previous
n_parallel=2 runs from persisting survivors. Root cause was `_partition_worker`
putting ~2.4GB pickled dict onto a Linux 64KB pipe buffer — blocking indefinitely.
Fix: write survivors to temp JSON file, put status-only dict on queue, parent
reads temp files after proc.join().

S138B fixed trial ceiling not enforced on resume — existing complete trials were
not subtracted before computing per-partition trial count, causing overshoot.

S139 lowered window_size max from 500 to 50 based on 167-trial empirical evidence
and TRSE short_persistence confirmation (conf=0.8275).

S139B fixed TRSE Rule A never firing in n_parallel path — the partition worker
called `_pstudy.optimize()` directly, bypassing `OptunaBayesianSearch.search()`
where Rule A lives. Fix: apply Rule A inline after `SearchBounds.from_config()`
inside `_partition_worker`. Confirmed firing in both P0 and P1 on relaunch.

200-trial fresh study launched with 50M seeds, TRSE Rule A active (window cap=32),
n_parallel=2, all 26 GPUs. First properly instrumented run ever.

---

## Changes

### S138 — NPZ Pipe Deadlock Fix (`3624e3c`)
**File:** `window_optimizer_integration_final.py`
**Root cause:** `_partition_worker` built `_local_acc` with millions of survivor
dicts (~2.4GB pickled). `result_queue.put(_local_acc)` blocked on Linux 64KB pipe
buffer. Parent's `_rq.get(timeout=7200)` raised `queue.Empty` (deadlock).
**Fix — 5 changes:**
1. `_partition_worker` signature: added `temp_file` parameter
2. OK path: writes `_local_acc` to JSON temp file, puts status-only dict to queue
3. Error path: removed accumulator from error queue put
4. `mp.Process` args: passes `/tmp/partition_{_pi}_survivors_{study_name}.json`
5. Parent collection loop: reads temp files after `proc.join()`, merges, deletes

**Smoke test:** 50k+75k fake survivors, all 5 checks, 14.8s. ✅

### S138B — Trial Ceiling Fix (`4849bff`)
**File:** `window_optimizer_integration_final.py`
**Root cause:** `_trials_per_worker = max_iterations // n_parallel` ignored existing
completed trials. With 51 existing + max=100, each partition ran 50 more → 151 total.
**Fix:** Query DB for existing complete trials before dividing.
`_remaining_trials = max(0, max_iterations - _existing_complete)`.
If remaining=0, skip workers entirely.

### S139 — Window Size Max 500→50 (`7d035c6`)
**Rationale:** 167-trial run confirmed short-term temporal regime. W2 dominant,
W3 next. TRSE confirms short_persistence (conf=0.8275). 500 was wasteland.
**5 locations updated:**
1. `distributed_config.json` — `search_bounds.window_size.max`: 500→50
2. `window_optimizer.py` line 47 — fallback dict window_size max: 500→50
3. `window_optimizer.py` line 50 — fallback dict skip_max max: 500→250
4. `window_optimizer.py` line 115 — `SearchBounds` dataclass default: 500→50
5. `agent_manifests/window_optimizer.json` — informational max: 4096→50

### S139B — TRSE Rule A in Partition Worker (`25cc2de`)
**File:** `window_optimizer_integration_final.py`
**Root cause:** `n_parallel=2` path called `_pstudy.optimize(_worker_obj)` directly
inside `_partition_worker` — bypassing `BayesianOptimization.search()` and
`OptunaBayesianSearch.search()` entirely. TRSE Rule A bounds narrowing lives inside
`OptunaBayesianSearch.search()`. TRSE Rule A had never fired in any parallel run.
**Fix:** Apply TRSE Rule A inline after `_local_bounds = SearchBounds.from_config()`
inside `_partition_worker`. Same logic as `window_optimizer_bayesian.py` lines 380–406.
Passive: no-op if context absent, version < 1.15, confidence < 0.70, or regime
not short_persistence.
**Confirmed firing on relaunch:**
```
[TRSE][P0] Rule A ACTIVE: short_persistence (conf=0.828) → window_size ceiling 50 → 32
[TRSE][P1] Rule A ACTIVE: short_persistence (conf=0.828) → window_size ceiling 50 → 32
```

---

## 167-Trial Study Results (Reference — pre-S138)
```
Best: trial 74 — W2_O14_evening_S7-63_FT0.201_RT0.151 — 1,384,186 survivors
Top configs: W2_O14, W2_O18, W2_O4, W2_O22 (all evening, session_idx=2)
TRSE: regime=short_persistence conf=0.8275 w3_w8_ratio=2.200 stable=True
```
Note: These results reflect the NPZ pipe deadlock — survivor accumulation from
partition workers was likely incomplete. 200-trial run will be the first clean result.

---

## 200-Trial Study Launch
**Command:**
```bash
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \
  --start-step 0 --end-step 1 \
  --params '{"lottery_file":"daily3.json","window_trials":200,"resume_study":false,
"max_seeds":50000000,"enable_pruning":true,"n_parallel":2,
"use_persistent_workers":true,"worker_pool_size":4,
"seed_cap_nvidia":5000000,"seed_cap_amd":2000000}'
```
**Log:** `~/distributed_prng_analysis/logs/step01_200trial_fresh.log`
**Status:** Running — early trials showing pruning active, TPE cold start exploring

---

## New TODOs Added This Session

- [ ] **WATCHER Step 1 timeout** — 480 min hardcoded will kill long runs. Make
  configurable or remove for autonomous operation.
- [ ] **`--force-step N` flag for WATCHER** — bypass freshness gate without
  manually deleting output files.
- [ ] **Persistent worker session drops on AMD rigs** — keepalive/TTL fix needed
  for 24/7 autonomous operation.
- [ ] **Post-run: update search_bounds** — use 200-trial empirical results to
  tighten offset/skip/threshold bounds in `distributed_config.json`.

---

## Commit Log

| Commit | Description |
|--------|-------------|
| `3624e3c` | S138: NPZ pipe deadlock fix — temp file accumulator |
| `889854f` | S138: SESSION_CHANGELOG |
| `4849bff` | S138B: trial ceiling fix — subtract existing complete |
| `7d035c6` | S139: window_size max 500→50 (5 locations) |
| `25cc2de` | S139B: TRSE Rule A in n_parallel partition worker path |

---

*S138/S139 — 2026-03-13*

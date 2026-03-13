# SESSION CHANGELOG — S138
**Date:** 2026-03-12
**Commit:** `3624e3c` (both remotes)
**Status:** COMPLETE

---

## Summary

Resumed Optuna Step 1 study `window_opt_1772507547` for 167 total trials. Discovered W2/evening regime dominance (1.38M survivors vs prior best 433k). Identified and fixed NPZ accumulator pipe deadlock bug — partition workers were building ~2.4GB accumulators that could not pass through multiprocessing.Queue pipe buffer (64KB), causing silent deadlock and zero survivors merged.

---

## Changes

### `window_optimizer_integration_final.py` — S138 pipe deadlock fix
**Commit:** `3624e3c`

**Root cause:** `_partition_worker` builds `_local_acc` with millions of survivor dicts (~2.4GB pickled). `result_queue.put(_local_acc)` blocks on Linux 64KB pipe buffer. Parent's `_rq.get(timeout=7200)` raises `queue.Empty` immediately (deadlock, not true timeout). Exception message is empty string — log showed `Queue timeout/error:` with nothing after colon. Result: zero survivors merged into `survivor_accumulator` after every parallel run.

**Fix — 5 changes:**
1. `_partition_worker` signature: added `temp_file` parameter
2. OK path: writes `_local_acc` to `temp_file` (JSON), puts status-only dict to queue
3. Error path: removed `accumulator` from error queue put (cleanup)
4. `mp.Process` args: passes `f'/tmp/partition_{_pi}_survivors_{_mp_study_name}.json'` as `temp_file`
5. Parent collection loop: reads temp files after `proc.join()`, merges into `survivor_accumulator`, deletes temp files

**Queue now carries only lightweight status dicts — no accumulator payload.**

---

## Run Results (167-trial study)

| Metric | Value |
|--------|-------|
| Total trials | 167 |
| COMPLETE | 59 |
| PRUNED | 105 |
| FAIL | 1 |
| Best score | 1,384,186 |
| Best config | W2_O14_evening_S7-63_FT0.201_RT0.151 |

**Top 5:**
| Trial | Score | Config |
|-------|-------|--------|
| 74 | 1,384,186 | W2_O14_evening_S7-63 |
| 88 | 1,140,740 | W2_O18_evening_S8-45 |
| 65 | 1,095,398 | W2_O4_evening_S8-35 |
| 84 | 1,001,028 | W2_O22_evening_S6-32 |
| 79 | 433,295 | W3_O18_evening_S8-38 |

**Key findings:**
- Window=2 dominates — 26,000× improvement over S112 W8 config
- Evening session only — all top trials session_idx=2
- Offset clustering at 4, 14, 18, 22 — weekly periodicity
- Variable skip produces ~4× more survivors than constant on same config
- FT: 0.19–0.22, RT: 0.15–0.22 — asymmetric thresholds optimal

---

## Bugs Identified

### NPZ accumulator pipe deadlock (FIXED this session)
- **File:** `window_optimizer_integration_final.py`
- **Symptom:** `bidirectional_survivors_binary.npz` contained only 45,867 seeds from old W4-8 trials instead of 1.38M W2 survivors
- **Root cause:** ~2.4GB pickle payload deadlocked multiprocessing.Queue pipe
- **Fix:** temp file approach — commit `3624e3c`

### Trial count ceiling not enforced when study_name bypasses resume condition (OPEN)
- Run targeted 100 new trials but executed 167 total (16 overrun)
- Each partition runs `_trials_to_run = max_iterations - completed` but with pre-existing completed > max_iterations the math yields free-running partitions
- **TODO:** Add hard ceiling check even when `study_name` is explicitly specified

### Persistent worker session drops on AMD rigs after extended runs (OPEN)
- "No existing session" errors on rrig6600b/c after hours of operation
- Not GPU-specific — any AMD GPU can drop
- 15/16 jobs still succeed; trials complete with partial data
- **TODO:** Worker keepalive ping, session TTL refresh, or auto-respawn after N hours

---

## Diagnostic Evidence

```
# Log grep confirming pipe deadlock:
Queue timeout/error:    ← empty exception message = queue.Empty from deadlock

# Pickle size estimate:
One survivor dict pickled: 515 bytes
1,384,186 survivors: ~0.66 GB per partition
Both partitions combined: ~2.4 GB through 64KB pipe → deadlock
```

## Patch Verification — Smoke Test (Zeus)

`test_s138_partition_accumulator.py` run on Zeus after patch applied.
Spawned 2 mock partition workers (50k + 75k fake survivors) using patched plumbing.

```
✅ Both workers signaled OK via queue
✅ Temp files existed after worker join
✅ survivor_accumulator counts correct (expected 125,000, got bid=125,000 fwd=125,000 rev=125,000)
✅ Temp files deleted after merge
✅ Completed without deadlock (14.8s < 60s)
✅ ALL TESTS PASSED — patch is working correctly
```

Patch confirmed working on Zeus. Safe to launch fresh 200-trial study.


---

## TODO Backlog (updated)

1. `[ ]` S110 root cleanup (884 files)
2. `[ ]` sklearn warnings in Step 5
3. `[ ]` Remove CSV writer from `coordinator.py`
4. `[ ]` Regression diagnostic gate = True
5. `[ ]` S103 Part 2
6. `[ ]` Phase 9B.3 (deferred)
7. `[ ]` Trial count ceiling not enforced when study_name bypasses resume condition
8. `[ ]` Persistent worker session drops on AMD rigs — keepalive/TTL fix
9. `[ ]` Fresh 200-trial study with max_seeds=50M and tightened search bounds

---

## Next Session

Launch fresh 200-trial study with:
- `max_seeds=50000000` (5× current)
- `resume_study=false` (new study)
- Tightened bounds: window 2–8, offset 0–30, session_idx=2 only, FT 0.15–0.30, RT 0.12–0.28
- Warm-start enqueue top 4 configs from this study
- Expected: 5–7M bidirectional survivors per top trial

# SESSION CHANGELOG — S166
**Date:** 2026-04-22
**Session:** S166
**Engineer:** Team Alpha (Michael + Claude)
**HEAD:** 8cb2ada (start) → pending commit
**Status:** COMPLETE — multiple fixes deployed, cwsr_enable=0 validated on rrig6600

---

## Session Summary

Long session with significant progress. Fixed Zeus TCP routing, accumulator NPZ
NameError, warm-start DB override, and discovered/fixed bidirectional list OOM.
Applied cwsr_enable=0 + mcbp=0 to rrig6600 — first 75-minute clean run on that rig.
Multiple OOM and crash events helped identify remaining issues.

---

## Fixes Deployed This Session

### Fix 1 — Zeus TCP Routing (isinstance guard removed)
**File:** `persistent_worker_coordinator.py`
**Commit:** `8ebdf99`

Removed `and not isinstance(wh, WorkerNode)` from `_run_once()` guard.
Zeus WorkerNodes now route via TCP like AMD rigs.

**Impact:** Zeus 3080Ti throughput: 8K s/s → 1,060,595 s/s (+13,200%)
**Validated:** Zeus contributing 395 jobs, 574K s/s avg per GPU.

### Fix 2 — Accumulator Purge (forward/reverse objects removed)
**File:** `window_optimizer_integration_final.py`
**Commit:** `8ebdf99`

Stopped accumulating full forward/reverse objects in bayesian mode.
Replaced with counts only. Eliminated 8-hour post-processing runaway.

**Also fixed:** `_superseded` NameError — `len(_superseded)` → `len(_superseded_prior_orig)`
Added `_superseded_prior_orig = []` to else branch for no-prior-NPZ path.

### Fix 3 — Warm-Start Params Flow End-to-End
**Files:** `window_optimizer_integration_final.py`, `window_optimizer.py`,
          `agents/watcher_agent.py`, `agent_manifests/window_optimizer.json`
**Commit:** `8cb2ada`

Warm-start params were being stripped as INTERNAL_ONLY before reaching
`window_optimizer.py`. Fixed by:
- Adding 6 warm_start params to `optimize_window()` signature
- Populating `_trial_history_ctx` with warm_start values
- Adding `--warm-start-*` CLI args to `window_optimizer.py`
- Removing warm_start_* from `_INTERNAL_ONLY_PARAMS` in `watcher_agent.py`
- Adding warm-start-* to manifest `args_map`

**Verified end-to-end:** W6_O0 enqueues as Optuna trial 0 in simulation.

**Remaining issue:** `watcher_agent.py` on Zeus still has old `_INTERNAL_ONLY_PARAMS`
— file was not updated in commit `8cb2ada` (only 1 file changed vs expected 3).
Fix pending before next run.

### Fix 4 — Bidirectional List Clear After Flush (OOM fix)
**File:** `window_optimizer_integration_final.py`
**Status:** Delivered, not yet committed

`_flush_npz_incremental()` wrote survivors to NPZ but never cleared
`accumulator['bidirectional']`. With W2_O30 producing 21.9M survivors,
the list consumed 60GB of Zeus RAM → OOM kill (SIGKILL code -9).

**Fix:** After successful flush, set `accumulator['bidirectional'] = []`.
Reset `_flush_last_count = 0` (since list is now empty).
Data is safe in NPZ — RAM stays bounded regardless of survivor count.

### Fix 5 — monitor_all.sh v6 (crash monitor removed)
**File:** `monitor_all.sh` on ser8
**Status:** Deployed locally, not committed to repo

Window 7 (crash monitor) removed — false-DOWN bug caused noise.
Netconsole is the reliable GPU fault detector.

### Fix 6 — cwsr_enable=0 + mcbp=0 on rrig6600
**File:** `/etc/modprobe.d/amdgpu.conf` on rrig6600
**Status:** Applied, initramfs updated, rebooted

Added `cwsr_enable=0 mcbp=0` to rrig6600 only. These disable Compute Wave
Save/Restore and Mid-Command Buffer Preemption — known ROCm stability fixes
for queue preemption failures under sustained compute workloads.

**Research finding:** Well-documented ROCm issue on GitHub — cwsr_enable=0
is the community-validated fix for `queue preemption failed` → `SMU 0xFFFFFFFF`
→ ring buffer corruption cascade. Previously tried in S162 with mixed results
on stock driver — now retrying with amdgpu-dkms 6.12.12.

**Result:** rrig6600 ran for 75+ minutes without a single GPU crash. ✅
Previous longest clean run on rrig6600: ~40 minutes.

**Caveat:** Throughput on rrig6600 reduced (~22K s/s vs 95K+ on other rigs).
CWSR handles wave context switching — disabling it serializes some operations.
Performance tradeoff vs stability needs further evaluation.

**Note:** Only applied to rrig6600. S162 showed cwsr_enable=0 broke rrig6600b
(KIQ fence timeouts). rrig6600c had it previously on stock driver. With
amdgpu-dkms 6.12.12 behavior may differ — test rrig6600c next if rrig6600
proves stable across multiple sequential runs (without reboot between runs).

---

## Issues Discovered This Session

### Issue 1 — warm_start_offset=0 filtered by `if value:` check
WATCHER's CLI builder skips values that are falsy. `warm_start_offset=0`
evaluated as False → `--warm-start-offset 0` never added to command.
Root cause: old `_INTERNAL_ONLY_PARAMS` on Zeus stripped params before
they even reached the `if value:` check. Both need fixing.

### Issue 2 — step1_trial_history DB overrides explicit warm-start
WATCHER reads `get_best_step1_params()` from `prng_analysis.db` at line ~1444
and OVERWRITES any warm_start_* params passed via --params.
Workaround: `DELETE FROM step1_trial_history` before each run.
Permanent fix: WATCHER should prefer explicit --params over DB lookup.

### Issue 3 — seed_start past 32-bit space
Coverage tracker advanced to seed_start=5,368,709,120 (past 4,294,967,296
= 2³²). Valid java_lcg seeds are 32-bit only. Fixed by `reset_seed_coverage.py`.
Root cause: coverage tracker advances after every run — after 5 runs covering
1B seeds each, it goes past the valid range. Need wrapping/reset logic.

### Issue 4 — bidirectional list OOM (21.9M survivors, 60GB RAM)
W2_O30 config with loose thresholds produced 21.9M bidirectional survivors.
`_flush_npz_incremental()` wrote to NPZ but never cleared the in-memory list.
Zeus OOM killed the process (code -9) after consuming 60GB RAM.
Fixed by clearing `accumulator['bidirectional'] = []` after each flush.

---

## Run Results This Session

| Run | Config | Survivors | Result |
|-----|--------|-----------|--------|
| s166_run1 | W7_O36 (DB override) | 0 | ✅ Complete |
| s166_run2 | W17_O36 (DB override) | 0 | ✅ Complete, card1 crash |
| s166_run3 | W12_O12 (DB override) | 0 | ✅ Complete |
| s166_run4 | W14_O28 (DB override) | 0 | ✅ Complete |
| s166_run5 | W2_O30 (DB override) | 21.9M | ❌ OOM killed (60GB RAM) |

All runs produced 0 quality survivors because DB warm-start kept picking
bad configs. W6_O0 warm-start never fired due to `_INTERNAL_ONLY_PARAMS` bug.

---

## Cluster State — End of Session

| Component | State |
|-----------|-------|
| Zeus HEAD | 8cb2ada |
| rrig6600 | cwsr_enable=0 mcbp=0, amdgpu-dkms 6.12.12, GDM disabled |
| rrig6600b | stock (no cwsr change), amdgpu-dkms 6.12.12, GDM disabled |
| rrig6600c | stock (no cwsr change), amdgpu-dkms 6.12.12, GDM disabled |
| seed_start | Reset to 0 ✅ |
| step1_trial_history | Cleared ✅ |
| optuna_studies | Cleared ✅ |
| Accumulator | 20,916 seeds (S165 data) |

---

## Pending Before Next Run

1. Deploy `watcher_agent.py` with fixed `_INTERNAL_ONLY_PARAMS`
   (only `warm_start_session` — not the full warm_start_* list)
2. Deploy `window_optimizer_integration_final.py` with bidirectional list clear
3. Commit both to both remotes
4. Clear `step1_trial_history` again (gets repopulated each run)
5. Launch with W6_O0 warm-start — should finally work

---

## Next Session Priorities

1. **Verify warm-start W6_O0 fires** — the real validation
2. **Watch rrig6600 across 2 sequential runs** — true cwsr_enable=0 test
   (fresh reboot doesn't prove anything — must survive run 2 without reboot)
3. **DPM harness** — find stable manual profile for rrig6600
4. **BoTorch dual-GPU** — implement after DPM harness validated
5. **Consider applying cwsr_enable=0 to rrig6600c** — different card variant,
   may behave differently with amdgpu-dkms 6.12.12 vs S162 stock driver
6. **Seed coverage wrapping** — prevent seed_start from exceeding 2³²

---

## Key Learnings

- `cwsr_enable=0` is the community-validated fix for ROCm queue preemption
  failures. Works on rrig6600 with amdgpu-dkms 6.12.12.
- W2_O30 (window=2) produces millions of survivors but low-quality signal.
  W6_O0 with threshold ~0.68 is the proven high-quality config.
- The incremental NPZ flush MUST clear the in-memory list after writing or
  it defeats the purpose entirely — RAM grows unboundedly.
- Two separate DBs control warm-start: optuna_studies/ AND prng_analysis.db.
  Both must be cleared for a true cold start.
- seed_start must be reset after completing the full 32-bit range.

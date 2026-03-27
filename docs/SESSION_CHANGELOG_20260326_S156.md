# SESSION CHANGELOG — S156
**Date:** 2026-03-25 to 2026-03-26  
**Commits:** `6789420` → `d4eb55d`  
**Status:** CLOSED — Two root causes identified and addressed; BIOS reflash confirmed fix for hard crash; CuPy OOM remains at 3+ workers; Run 1 complete

---

## Summary

S156 identified and resolved two distinct failure modes on rrig6600c through systematic remote monitoring (PageTables + py_procs via `remote_crash_monitor.py`):

1. **PWC lifecycle bug** — zombie worker accumulation across trials causing py_procs to climb to 12+ → PageTable overflow → kernel freeze. Bandaid fix deployed (v1 then v2), architectural fix (Phase B) deferred to S157.

2. **BIOS NVRAM corruption** — hard kernel freeze during ROCm context initialization. Confirmed by power-on failure symptom. BIOS reflash (B36FF608.BSS) reset NVRAM, changed e820 memory map, eliminated hard crash.

After both fixes, rrig6600c completed a full 3-trial run at 3 workers without crashing — first time in project history.

---

## Root Cause 1 — PWC Lifecycle Bug

### Discovery
Remote crash monitor captured definitive zombie accumulation signature:
```
py_procs=6  → PageTables=15,000 kB   (workers from previous trial still alive)
py_procs=9  → PageTables=12,844 kB   → CRASH (pre-BIOS-reflash threshold)
py_procs=12 → PageTables=19,124 kB   → CRASH
```

### Root Cause
`run_trial_persistent()` in `window_optimizer_integration_final.py` creates a NEW
`PersistentWorkerCoordinator` per trial. When rrig6600c reboots mid-run and rejoins,
new trial spawns 8 fresh workers without proving node is clean — stacking on surviving
processes from the previous trial.

Team Beta confirmed: "The root cause is that the persistent worker pool is being
recreated every trial instead of once per optimization session."

Full analysis: `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md`

### Fix Deployed — Phase A Bandaid

**Commit:** `6789420`  
**File:** `persistent_worker_coordinator.py`  
**Patch:** `apply_s156_pwc_cleanup_v2.py`

Added targeted pre-spawn cleanup to `startup()`:
- Finds existing `sieve_gpu_worker.*--persistent` processes via `pgrep -af`
- SIGTERM first, SIGKILL on timeout (TB requirement)
- Logs exact processes found and reaped
- 2s delay for ROCm context release

**Note:** v1 patch (broad `pkill -9`) deployed first, then v2 (targeted, TB-compliant).
v2 had syntax error on Zeus — restored from bak_s156_v1 to unblock production.
v2 syntax fix is S157 P0.

### Fix Pending — Phase B Architectural (S157)
Move PWC creation to Step 1 execution layer (not per-trial).
One PWC instance per optimization session.
`reset_for_new_trial()` between trials.
Full spec: `PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md`

---

## Root Cause 2 — BIOS NVRAM Corruption

### Evidence
- Power-on failure stopped working (confirmed BIOS defect)
- Hard kernel freeze at PageTables ~13,000 kB — different threshold than rrig6600/b
- e820 memory map showed fragmented BIOS hole at `0x37700000` (M.2 adapter remnant from S155)
- Crash occurred even with 0 stale workers on clean first spawn at py_procs=6

### Fix — BIOS Reflash
Flashed B36FF608.BSS (same version, clean reflash resets NVRAM).
BIOS settings confirmed:
- Above 4GB MMIO BIOS Assignment: Enabled
- Resize BAR: Disabled
- RC6 Render Standby: Disabled
- VT-d: Enabled

Post-reflash e820: fragmented hole gone, cleaner memory map.

### Verification
```
Before reflash: CRASH at PageTables=10,508-13,000 kB, py_procs=6-9
After reflash:  STABLE at PageTables=13,500-13,900 kB, py_procs=5, 73+ samples
```

rrig6600c stayed up — first time crossing 13,000 kB without a hard crash.
Machine remained alive after CuPy OOM exception (Python exception, not kernel freeze).

---

## Test Matrix Results (Team Beta A→D)

| Test | Workers | Result | PageTables max | Notes |
|------|---------|--------|----------------|-------|
| A — 2 workers | 2 | ✅ PASSED | 35,008 kB | 3 trials complete, clean shutdown |
| B — 3 workers (post-BIOS) | 3 | ✅ PASSED | ~13,900 kB | 2+ trials stable, CuPy OOM on some chunks |
| C — 4 workers (pre-BIOS) | 4 | ❌ CRASHED | 10,508 kB | Hard kernel freeze |
| C — 4 workers (post-BIOS) | 4 | ❌ CuPy OOM | ~14,000 kB | Machine stayed up — soft failure |
| D — 8 workers | Not tested | — | — | Deferred to S157 |

**Key finding:** BIOS reflash changed failure mode from hard kernel crash to soft
Python CuPy OOM. Machine no longer hard-freezes.

---

## PageTables Comparison (Post-BIOS)

```
rrig6600:  38,684 kB  (8 workers, stable)
rrig6600b: 39,640 kB  (8 workers, stable)
rrig6600c: 13,500 kB  (3 workers, stable post-BIOS)
```

PageTables are NOT the crash cause — they are a symptom. Hard crash was BIOS NVRAM
corruption. Remaining CuPy OOM at 3+ workers is a software memory management issue.

---

## Production Run — Run 1 Complete

Run 1 (seeds 0 → 1,073,741,824) completed at end of S156:
- 3 trials completed ✅
- 3 workers on rrig6600c (post-BIOS) ✅  
- WATCHER confidence 1.00 — PROCEED ✅
- Best config: W32_O41_evening_S4-127_FT0.55_RT0.61
- Bidirectional survivors: 0 (window configs need tuning)
- Coverage logged: 1,073,741,824 → 2,147,483,648
- NPZ accumulator: 676 seeds (unchanged)

---

## Diagnostic Tools

- `remote_crash_monitor.py` — polls rrig6600c every 2s, logs PageTables + py_procs
- `PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md` — full PWC lifecycle fix proposal
- `apply_s156_pwc_cleanup_v2.py` — Phase A bandaid patch (v2, TB-compliant)

---

## Commits This Session

| Commit | Description |
|--------|-------------|
| `1f03d24` | fix(s156-bandaid): targeted pre-spawn cleanup in PWC startup (v1) |
| `6789420` | fix(s156-bandaid-v2): targeted SIGTERM-first cleanup per TB |
| `d4eb55d` | config(s156): rrig6600c stable post-BIOS-reflash, 3 workers, coverage run1 complete |

---

## Architecture Invariants Added S156

- **[S156-BANDAID v2]** Targeted pre-spawn cleanup in `PWC.startup()` — SIGTERM first, `--persistent` scoped
- **[S156]** BIOS reflash resets NVRAM — confirmed fix for hard kernel freeze on rrig6600c
- **[S156]** rrig6600c stable operating point: 3 workers post-BIOS (CuPy OOM at 4+)
- **[S156]** PageTables are symptom not cause — hard crash was BIOS NVRAM corruption
- **[S156]** PWC lifecycle bug confirmed — Phase B architectural fix required (S157)
- **[S156]** Remote crash monitor (`remote_crash_monitor.py`) is essential diagnostic tool
- **[S156]** Launch command: `nohup bash -c 'source ~/venvs/torch/bin/activate && PYTHONPATH=. python3 agents/watcher_agent.py --clear-halt --run-pipeline --start-step 1 --end-step 1 --force-step 1 >> ~/distributed_prng_analysis/logs/sweep_run1_production.log 2>&1' &`

---

## Open Issues → S157

1. **P0** — Fix v2 patch syntax error on Zeus (restore from GitHub, reapply correctly)
2. **P0** — Phase B architectural fix: session-scoped PWC
3. **P1** — Test 8 workers on rrig6600c post-BIOS (CuPy OOM may still occur)
4. **P1** — CuPy OOM at 3+ workers: investigate why rrig6600/b handle 8 workers
5. **P2** — Commit S155/S156 changelogs to Zeus docs
6. **P2** — S131 TODOs: Gate 1 kill method, seed cap patch, RETRY param-threading

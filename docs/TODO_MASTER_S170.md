# MASTER TODO LIST — S170
**Compiled:** 2026-03-02 (S114) | **Updated:** 2026-04-24 (S167/S168/S169)
**Branch:** s167-clean | **HEAD:** a6cd55e

---

## ✅ COMPLETED THIS SESSION (S167/S168/S169)

| Item | Session |
|------|---------|
| S167: Removed S140b warm-start DB injection from WATCHER | S167 |
| S167: Restored full _INTERNAL_ONLY_PARAMS (all warm_start_* fields) | S167 |
| S167: Added None/empty string guard in CLI builder | S167 |
| S168A-DIAG: Passive startup hammer telemetry in pwc_transport_tcp.py | S168A |
| S168A-DIAG: Confirmed 129ms synchronized first-wave burst via diagnostic data | S168A |
| S168: CRC32-based deterministic first-assignment jitter | S168 |
| S169: Per-worker minimum gap pacing for steady-state burst smoothing | S169 |
| Verified: 2 consecutive trials completed cleanly with S168+S169 enabled | S169 |
| Session changelog committed to both remotes | S167/S168/S169 |

---

## 🔴 P1 — HIGH PRIORITY (Next Session)

### 1. Stability Curve Test
**Status:** Ready to run
**Purpose:** Determine maximum stable seed_cap_amd under sustained multi-trial load

Test matrix:
- `CAP=100000 TRIALS=5`
- `CAP=150000 TRIALS=5`
- `CAP=200000 TRIALS=5`

Fixed params:
```
min_workers=24
PRNG_PWC_FIRST_ASSIGN_JITTER_SEC=3
PRNG_PWC_PER_WORKER_MIN_GAP_SEC=0.02
PRNG_PWC_STARTUP_DIAG=1
```

Track both:
- GPU faults (netconsole)
- Transport failures (script write failed on rrig6600)

Pass criteria: all trials complete, netconsole clean

### 2. DPM Harness
**Status:** Requires TB proposal
**Priority:** P1 — blocking optimal throughput

Target: 900mV / 2100-2200MHz per rig
- Start conservative at 900mV / 2100MHz
- Step up 50MHz until instability
- Record: crash frequency, temp, power, seeds/sec
- Implement as persistent service (requires TB approval)

Previous profile: Kaspa OC 2250MHz / -150mV (not validated for ROCm compute)

---

## 🟠 P2 — MEDIUM PRIORITY

### 3. BoTorch Dual-GPU
**Status:** Requires TB proposal — implement after DPM harness
**Purpose:** Replace/augment Optuna TPE with GPU-accelerated Gaussian process optimization

Integration point: `window_optimizer_bayesian.py`
- After N warmup trials, switch from TPE to BoTorch
- Use both Zeus RTX 3080Ti GPUs (cuda:0, cuda:1)
- Must fall back to Optuna TPE on failure
- Resume passthrough must still work

### 4. rrig6600 Script Write Failed Investigation
**Status:** Open — non-fatal but indicates transport/I/O contention
**Symptom:** `[PWC-TCP] 192.168.3.120:GPU* script write failed` during Trial 2+ worker restart
**Impact:** Non-fatal (trial completes) but may contribute to instability at higher caps

### 5. rrig6600c Page Fault Root Cause
**Status:** Partially mitigated by S169 pacing
**Signature:** `gfxhub / SQC (inst) / GCVM_L2_PROTECTION_FAULT`
**Note:** Different from rrig6600's `qcm fence timeout` — cwsr_enable=0 does NOT help rrig6600c

---

## 🟡 P3 — LOWER PRIORITY

### 6. Selfplay NN Fix
`inner_episode_trainer.py` still has hardcoded forbidden guard blocking NN in selfplay.
Fix: remove forbidden check + add y-normalization to selfplay path (same as train_single_trial.py S121 fix)

### 7. Post-run JSON cleanup
Remove stale JSON files after each run

### 8. S110 root cleanup
Remove legacy S110 files from Zeus root

### 9. sklearn warnings Step 5
Suppress or fix sklearn deprecation warnings in Step 5

### 10. Remove CSV writer from coordinator.py
Cleanup task — CSV writer is unused

### 11. S103 Part 2
Per-seed match rates downstream scoring integration

---

## 🔵 DEFERRED

| Item | Reason |
|------|--------|
| ZMQ job pre-fetch | TCP-PWC is current transport — not needed |
| free_all_blocks() removal | Staged validation conditions per TB |
| crash_forensic_daemon.py 3 bugs | Low priority — 3 known bugs documented |

---

## Architecture Invariants (Do Not Break)

- `cwsr_enable=0 mcbp=0` on rrig6600 ONLY — causes faster crashes on rrig6600c
- rrig6600b: stock amdgpu settings — cwsr caused KIQ fence timeouts in S162
- No `get_best_step1_params()` in Step 1 execution path
- No `enqueue_trial()` for DB warm-start in fresh studies
- `warm_start_*` are INTERNAL_ONLY — never CLI args
- Resume via explicit `study_name` only (S114-S116 design)
- S168/S169 default OFF — must be explicitly enabled via env vars
- Dual-push always: `git push origin s167-clean && git push public s167-clean`
- WATCHER never commits to git — Team Alpha delivers files only
- SESSION_CHANGELOG every session — committed to docs/ on Zeus

---

## Do NOT Include in TODO Lists
- GPU2 failure logging
- `--save-all-models` flag

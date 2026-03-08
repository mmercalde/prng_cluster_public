# SESSION CHANGELOG — S127
**Date:** 2026-03-07
**Session:** S127
**Engineer:** Team Alpha (Michael + Claude)
**Status:** COMPLETE — manifest bug fixed, Optuna study cleanup done.

---

## Session Objectives
1. Diagnose why `study_name` was not being passed through WATCHER → window_optimizer
2. Clean up stale Optuna studies on Zeus

---

## Completed This Session

### 1. Manifest Bug Fix — `window_optimizer.json` ✅

**Root cause:** Three bugs in `agent_manifests/window_optimizer.json` caused `study_name`, `resume_study`, `enable_pruning`, and `n_parallel` to be silently dropped by WATCHER's step-scoped param filter.

**Bug 1:** `enable_pruning` and `n_parallel` were missing from `default_params` entirely. WATCHER's step-scoped filter (`allowed_params = set(default_params.keys())`) silently discarded any param not declared in `default_params` — so both were dropped before CLI construction.

**Bug 2:** `resume-study` and `study-name` were only in the top-level `args_map`, not in `actions[0].args_map`. Worked by accidental fallback but fragile.

**Bug 3:** `study_name` defaulting to `""` in `default_params`. When `resume_study` didn't fire correctly, `--study-name ""` was passed → window_optimizer created a fresh study instead of resuming.

**Fix applied to `agent_manifests/window_optimizer.json`:**
- Added `"enable_pruning": false` and `"n_parallel": 1` to `default_params`
- Added `"enable-pruning": "enable_pruning"`, `"n-parallel": "n_parallel"`, `"resume-study": "resume_study"`, `"study-name": "study_name"` to `actions[0].args_map`

**Verified:** Simulated WATCHER command build confirms all params now flow correctly:
```
python3 window_optimizer.py --lottery-file daily3.json --strategy bayesian \
  --max-seeds 10000000 --prng-type java_lcg --output optimal_window_config.json \
  --test-both-modes --resume-study --trials 200 \
  --study-name window_opt_1772507547 --trse-context trse_context.json \
  --enable-pruning --n-parallel 2
```

**File delivered:** `agent_manifests/window_optimizer.json`
**Deploy:**
```bash
scp ~/Downloads/window_optimizer.json rzeus:~/distributed_prng_analysis/agent_manifests/window_optimizer.json
```

**⚠️ NOT YET COMMITTED TO EITHER REPO** — carry forward to next session.

---

### 2. Optuna Study Cleanup ✅

**Before:** 14 active studies in `optuna_studies/` (plus 13 in `archive_synthetic/`)

**Deleted (13 studies):**
```
window_opt_1772494935  (23 complete — superseded by primary)
window_opt_1772588654  (7 complete)
window_opt_1772672314  (25 complete — superseded by primary)
window_opt_1772683392  (10 complete)
window_opt_1772683520  (0 complete — ghost)
window_opt_1772684596  (1 complete)
window_opt_1772769305  (0 complete — ghost)
window_opt_1772769307  (0 complete — ghost)
window_opt_1772769646  (6 complete)
window_opt_1772937305  (1 complete)
window_opt_1772938484  (1 complete)
window_opt_1772945042  (2 complete — S126 interrupted)
window_opt_1772946380  (9 complete — S126 wrong study accident)
```

**After:** Exactly 1 active study remains:
```
window_opt_1772507547.db  — 24 COMPLETE, 26 PRUNED, 1 FAIL — PRIMARY resume target
```

`archive_synthetic/` (13 pre-real-data era studies) untouched.

---

## Carry Forward to S128

### 🔴 P1 — Immediate
1. **Deploy manifest fix** — `scp ~/Downloads/window_optimizer.json rzeus:~/distributed_prng_analysis/agent_manifests/window_optimizer.json`
2. **Commit manifest fix to both repos** — `agent_manifests/window_optimizer.json`
3. **GPU Throughput Phase A — RTX isolated** — Step-ladder: 100k→500k→1M→2M→5M seeds, single card
4. **GPU Throughput Phase A — RX 6600 isolated** — Step-ladder: 100k→250k→500k→1M→2M seeds, `ROCR_VISIBLE_DEVICES=0`
5. **GPU Throughput Phase B — Full concurrent rigs** — Both RTX / all 8 AMD simultaneously
6. **GPU Throughput Phase C — Stability test** — 50 consecutive jobs at Phase B × 0.85
7. **Update `coordinator.py` seed caps** — Line 233: `seed_cap_nvidia`, `seed_cap_amd`, `seed_cap_default`
8. **Update `gpu_optimizer.py` profiles** — Lines 17/18/35/36: `seeds_per_second` + `scaling_factor`

### Architecture Invariant Added This Session
- **Manifest param governance:** Every new CLI param MUST be in manifest `default_params` AND `actions[0].args_map` or WATCHER silently drops it (confirmed S127)

---

## Key Numbers (End of S127)
- Real draws: 18,068
- Bidirectional survivors: 85 (W8_O43)
- Best NN R²: +0.020538
- TRSE: regime=short_persistence, conf=0.828, Rule A active (window ceiling=32)
- Primary Optuna study: `window_opt_1772507547.db` (24 COMPLETE, 26 PRUNED, 1 FAIL)
- Active studies on Zeus: **1** (clean)
- Zeus GPU compute mode: DEFAULT ✅
- n_parallel=2: OPERATIONAL ✅

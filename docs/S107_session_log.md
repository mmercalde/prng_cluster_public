# S107 Session Log — Step 2 v4.4
**Date:** 2026-02-22  
**Session Focus:** Step 2 Scorer Meta-Optimizer — diagnosis, patching, and first clean run

---

## Issues Discovered & Resolved

### 1. bc_stat NameError (FIXED ✅)
- **File:** `scorer_trial_worker.py` line 484
- **Cause:** v4.3 removed `bc_stat` from `run_trial` but left an orphaned reference in `_log_trial_metrics` metrics dict
- **Effect:** Every trial computed enrichment objective correctly then crashed before saving cleanly
- **Fix:** Removed orphaned `'bc_stat': round(bc_stat, 2)...` line via `apply_s107_scorer_worker_v4_4.py`
- **Verified:** Smoke test trial 998 completed with `status: success`, no NameError

### 2. sample_size=450 Hardcoded / Override Not Propagating (FIXED ✅)
- **File:** `run_scorer_meta_optimizer.sh` lines 231 & 243
- **Cause:** Shell script hardcoded `--sample-size 450` in both legacy and non-legacy branches; no CLI argument parsing for `--sample-size`; Watcher's `--params '{"sample_size":5000}'` override was silently dropped
- **History:** 450 was benchmarked 2026-01-17 against NN training cost (5000=60-90s/trial). After normalization made NN faster, then v4.3 replaced NN training entirely with numpy enrichment arithmetic, timing became identical (2.8s for both 450 and 5000). The cap became obsolete but the comment was never updated.
- **Effect:** All prior runs used only 450 seeds (~36 minority seeds/trial), giving Optuna insufficient statistical resolution
- **Fix:** Added `SAMPLE_SIZE=5000` default, `--sample-size` CLI arg parser, replaced hardcoded 450 with `$SAMPLE_SIZE` in both branches, removed orphaned timing comments — via `apply_s107_run_scorer_v2_3.sh`
- **Verified:** `scorer_jobs.json` confirmed `Unique sample_size across all 100 jobs: {5000}`

### 3. 192.168.3.162 Missing from scp Push Loop (FIXED ✅)
- **File:** `run_scorer_meta_optimizer.sh` line 256
- **Cause:** Third rig added to cluster but scp loop never updated — only pushed to .120 and .154
- **Effect:** rig-6600c ran stale worker code silently on every Step 2 run
- **Fix:** Added `192.168.3.162` to scp loop in `apply_s107_run_scorer_v2_3.sh`
- **Verified:** MD5 `140e022dcb2fecaae80ff03858dee12f` confirmed across all 4 nodes

### 4. 192.168.3.162 Missing from ml_coordinator_config.json (FIXED ✅)
- **Cause:** Collection config never updated when third rig was added to cluster
- **Effect:** 25 results per run silently lost — never pulled from .162, never reported to Optuna. Caused "Cannot tell a COMPLETE trial" warnings (100 pre-allocated, only 75 ever reported)
- **Fix:** Added .162 node entry to `ml_coordinator_config.json` inline

### 5. gpu_count Discrepancy in ml_coordinator_config.json (FIXED ✅)
- **Cause:** .120 and .154 configured with `gpu_count=12 / max_concurrent_script_jobs=12` despite all remote rigs having 8 GPUs
- **Effect:** Coordinator potentially over-dispatching concurrent jobs
- **Fix:** Corrected all three remote nodes to `gpu_count=8 / max_concurrent_script_jobs=8`
- **Verified:** `localhost=2, .120=8, .154=8, .162=8 → 26 total`

---

## First Clean Step 2 Run — v4.4 Results

| Metric | Previous Best (v4.3) | This Run (v4.4) |
|--------|---------------------|-----------------|
| sample_size | 450 (hardcoded) | 5000 ✅ |
| Trials executed | 100/100 | 100/100 |
| Trials collected | 75/100 | 100/100 ✅ |
| Trials reported to Optuna | 75/100 | 100/100 ✅ |
| NameError crashes | Yes (silent) | None ✅ |
| Best trial # | 1 (early — no learning) | 32 (mid-run — Bayesian learning) ✅ |
| Best accuracy | 0.291 | 0.3644 |
| Runtime | 3:55 | 3:56 |

**Best config (trial 32):** `rm=(16,101,975)  max_offset=5  temporal_window=93  temporal_num_windows=10  hidden_layers=128_64`

**optimal_scorer_config.json** — updated with `_run_context` audit block (sample_size, study_name, run_id, trials_reported, session, version)

---

## Files Modified This Session

| File | Change |
|------|--------|
| `scorer_trial_worker.py` | Removed orphaned `bc_stat` from `_log_trial_metrics` |
| `run_scorer_meta_optimizer.sh` | Added `SAMPLE_SIZE=5000`, `--sample-size` CLI arg, added .162 to scp loop, removed orphaned comments |
| `ml_coordinator_config.json` | Added 192.168.3.162 node, corrected .120/.154 gpu_count 12→8 |
| `optimal_scorer_config.json` | Added `_run_context` audit block |

## Patchers Deployed

| Patcher | Target | Patches |
|---------|--------|---------|
| `apply_s107_scorer_worker_v4_4.py` | `scorer_trial_worker.py` | 1 (bc_stat removal) |
| `apply_s107_run_scorer_v2_3.sh` | `run_scorer_meta_optimizer.sh` | 5 (SAMPLE_SIZE default, CLI arg, 2× branch replacement, orphaned comments, .162 scp) |

---

## Current Status

| Component | Status |
|-----------|--------|
| scorer_trial_worker.py | v4.4 — clean, deployed to all 4 nodes (MD5 verified) |
| run_scorer_meta_optimizer.sh | v2.3 — sample_size wired, all rigs in scp loop |
| ml_coordinator_config.json | 4 nodes, all correct GPU counts |
| optimal_scorer_config.json | Best config from first valid 100-trial run, audit context attached |
| Step 2 pipeline | ✅ Fully operational — first clean run complete |

---

## TODO / Pending TB Decisions

### 🔴 Requires TB Ruling
1. **max_offset architecture flaw** — Single `max_offset` applied uniformly across moduli of different scales. Any modulus smaller than `max_offset` becomes a no-op (e.g. `rm1=11, offset=12` → `seeds % 11` always < 12 → dimension blind). Mask never truly uses all three moduli simultaneously. Proposed fix: replace `max_offset` with scale-invariant `keep_fraction` (Optuna searches 0.1→0.9), each modulus independently keeps same fraction: `(seeds % rm) < rm * keep_fraction`. **Needs TB approval before v4.5.**

### 🟡 Infrastructure / Polish
2. **Watcher `file_exists` evaluation** — Step 2 Watcher evaluates completion via `file_exists` (confidence=1.0) rather than inspecting trial quality metrics. No validation that best trial actually enriches the minority. Low priority but worth flagging to TB.
3. **Previous run artifact cleanup** — Must manually `rm optimal_scorer_config.json scorer_jobs.json optuna_studies/*.db` before each re-run. Consider adding `--clean` flag to Watcher or shell script.

### ✅ Closed This Session
- bc_stat NameError
- sample_size=450 obsolete cap
- scp missing rig
- collection missing rig
- gpu_count config discrepancy
- run context audit block in optimal_scorer_config.json

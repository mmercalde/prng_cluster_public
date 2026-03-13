#!/usr/bin/env python3
"""
S136 Documentation Update Patch Script — CORRECTED
Covers: COMPLETE_OPERATING_GUIDE v2.0→v2.1, CHAPTER_9, CHAPTER_10,
        CHAPTER_12, CHAPTER_14, CHAPTER_1, README

Run on Zeus:
  cd ~/distributed_prng_analysis
  python3 apply_s136_doc_updates.py
"""
import os, shutil, sys

DOCS = os.path.expanduser("~/distributed_prng_analysis/docs")
ROOT = os.path.expanduser("~/distributed_prng_analysis")

_errors = 0

def patch(filepath, old_text, new_text, label):
    global _errors
    path = os.path.expanduser(filepath)
    if not os.path.exists(path):
        print(f"  MISS {label}: file not found: {path}")
        _errors += 1
        return
    with open(path) as f:
        content = f.read()
    if old_text not in content:
        print(f"  SKIP {label}: anchor text not found")
        _errors += 1
        return
    bak = path + ".bak_s136"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
    before = len(content.splitlines())
    content = content.replace(old_text, new_text, 1)
    after = len(content.splitlines())
    with open(path, "w") as f:
        f.write(content)
    print(f"  OK   {label}: {before} -> {after} lines")


def append_section(filepath, marker, new_text, label):
    """Append new_text only if marker string not already present."""
    global _errors
    path = os.path.expanduser(filepath)
    if not os.path.exists(path):
        print(f"  MISS {label}: file not found: {path}")
        _errors += 1
        return
    with open(path) as f:
        content = f.read()
    if marker in content:
        print(f"  SKIP {label}: section already present")
        return
    bak = path + ".bak_s136"
    if not os.path.exists(bak):
        shutil.copy(path, bak)
    before = len(content.splitlines())
    combined = content + new_text
    after = len(combined.splitlines())
    with open(path, "w") as f:
        f.write(combined)
    print(f"  OK   {label}: {before} -> {after} lines")


# =============================================================================
# DOC 1 -- COMPLETE_OPERATING_GUIDE_v2_0.md
# =============================================================================
G = f"{DOCS}/COMPLETE_OPERATING_GUIDE_v2_0.md"

# 1a) Version header
patch(G,
    "**Version 2.0.0**  \n**February 2026**  \n**Updated: Session 83 (Feb 13, 2026)**",
    "**Version 2.1.0**  \n**February 2026**  \n**Updated: Session 135 (Mar 10, 2026)**",
    "GUIDE version header")

# 1b) Step 0 section title
patch(G,
    "## 2.0 Step 0: PRNG Fingerprinting",
    "## 2.0 Step 0: Temporal Regime Segmentation Engine (TRSE) -- NEW S122",
    "GUIDE Step 0 title")

# 1c) Step 0 status line
patch(G,
    "**Status:** ARCHIVED (Session 17)",
    "**Status:** ACTIVE (skip_on_fail -- failures never halt pipeline)",
    "GUIDE Step 0 status")

# 1d) Step 0 body -- replace archived investigation text
patch(G,
    "**Original Purpose:** Classify unknown PRNGs by comparing behavioral fingerprints against a library of known PRNGs.",
    "**Module:** `trse_step0.py` v1.15.1\n\n**Purpose:** Segment draw history into temporal regimes before window optimization. Produces `trse_context.json` consumed by Step 1.\n\n**Key behavior:**\n- `skip_on_fail=True` -- TRSE failure logs a warning and pipeline continues to Step 1\n- Regime detection uses relative normalization (v1.15.1 fix S122 -- eliminates `regime_type=unknown`)\n- Outputs `confirmed_windows` back to `trse_context.json` after Step 1 produces survivors\n\n**Note (archived):** Session 17 mod1000 fingerprinting investigation superseded.\nBidirectional sieve is the correct PRNG discriminator.\n",
    "GUIDE Step 0 body")

# 1e) Step 3 feature count
patch(G,
    "extracting 64 ML features per seed (50 per-seed + 14 global) for downstream model training.",
    "extracting **91 features per survivor** (89 training -- excludes score + confidence) for downstream model training.",
    "GUIDE Step 3 feature count")

# 1f) Feature architecture total
patch(G,
    "Total Features: 64 (62 for training after excluding score, confidence)",
    "Total Features: 91 (89 for training after excluding score, confidence)",
    "GUIDE feature architecture total")

# 1g) Hardware table -- fix broken rig-6600c row (blank line splits the table)
patch(G,
    "| rig-6600b | RX 6600 (8GB) | 8 | ROCm / HIP |\n\n| rig-6600c | RX 6600 (8GB) | 8 | ROCm / HIP |",
    "| rig-6600b | RX 6600 (8GB) | 8 | ROCm / HIP |\n| rig-6600c | RX 6600 (8GB) | 8 | ROCm / HIP |",
    "GUIDE hardware table rig-6600c row fix")

# 1h) Hardware notes -- add rig-6600c CPU note, GPU compute mode, fan service
patch(G,
    "**GPU Stability:** udev rule (perf=high auto-set), GFXOFF disabled via kernel parameter",
    ("**GPU Stability:** udev rule (perf=high auto-set), GFXOFF disabled via kernel parameter  \n"
     "**rig-6600c note:** i5-8400T CPU causes ~50% throughput deficit vs other rigs -- hardware limitation  \n"
     "**Zeus GPU compute mode:** DEFAULT (enforced via /etc/rc.local, S125b -- EXCLUSIVE_PROCESS breaks n_parallel>1)  \n"
     "**Fan service:** rocm-fan-curve.service DISABLED on all rigs"),
    "GUIDE hardware stability notes")

# 1i) New modules in core components table
patch(G,
    "| Strategy Advisor | parameter_advisor.py | LLM-guided parameter recommendations |",
    ("| Strategy Advisor | parameter_advisor.py | LLM-guided parameter recommendations |\n"
     "| Persistent Worker Engine | persistent_worker_coordinator.py | "
     "STANDALONE parallel sieve engine -- activated by --use-persistent-workers flag; "
     "zero changes to coordinator.py (S130/S134/S135) |\n"
     "| TRSE Engine | trse_step0.py | Temporal Regime Segmentation, skip_on_fail (S121/S122) |\n"
     "| Holdout Quality | holdout_quality.py | Composite holdout score -- replaces holdout_hits (S111) |"),
    "GUIDE new modules in core components")

# 1j) Seed caps + GPU sps -- add after SSH Alias line
patch(G,
    "**SSH Alias:** `rzeus` (from ser8 to Zeus)",
    """**SSH Alias:** `rzeus` (from ser8 to Zeus)

### Seed Caps -- Validated Operating Points (S131)

| GPU | CLI Flag | Value |
|-----|----------|-------|
| RTX 3080 Ti (Zeus) | `--seed-cap-nvidia` | 5,000,000 seeds/pass |
| RX 6600 (rigs) | `--seed-cap-amd` | 2,000,000 seeds/pass |

Worker pool size: 4 workers/rig (validated against memory pressure, S134/S135)

### GPU Throughput -- Corrected Values (S129B-A)

| GPU | Measured sps | Old stale value |
|-----|-------------|-----------------|
| RTX 3080 Ti | 2,210,000 | 29,000 |
| RX 6600 | 787,950 | 5,000 |

Aggregate throughput:
- Subprocess mode baseline (S128): 849,469 sps
- Persistent workers active (S130): 2,082,140 sps (+150%)""",
    "GUIDE seed caps and GPU sps")

# 1k) ML results block -- insert after Training Diagnostics status table
patch(G,
    "| Selfplay Diagnostics | Phase 8A: eval_set + trend detection |",
    """| Selfplay Diagnostics | Phase 8A: eval_set + trend detection |""",
    "GUIDE selfplay row anchor check")  # no-op to verify anchor exists

patch(G,
    "| Episode Trend Detection | `_check_episode_training_trend()` | \xe2\x9c\x85 Complete (S83) |",
    """| Episode Trend Detection | `_check_episode_training_trend()` | \xe2\x9c\x85 Complete (S83) |

### ML Results Summary -- Real Data (S111-S121)

```
Dataset:      18,068 CA Daily 3 draws (2000-2026)
ML target:    holdout_quality (composite, S111) -- replaced holdout_hits (R2=0.000155)
Best model:   CatBoost (consistently top on tabular survivor data)
Best R2:      +0.0046 (S111 clean baseline)
NN R2:        +0.020538 (first positive, after y-normalization fix S121)
Best run:     17,247 bidirectional survivors (200-trial, pre-S110)
Optuna study: window_opt_1772507547.db (21 trials, resume target)

Feature signal (CatBoost on real data):
  ~32%  mod-8 residue features  -- LCG low-bit algebraic structure
  ~20%  prediction residuals    -- pred_std, residual_abs_mean
  <10%  mod-125 features
  <10%  mod-1000 features
   24/62 base features: zero importance -- Battery Tier 1A (S113) added to address gap
```""",
    "GUIDE ML results block")

# 1l) WATCHER CLI -- update to show 0->6
patch(G,
    "# Run pipeline manually\nPYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 3 --end-step 6",
    """# Full 0->6 run (standard)
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 0 --end-step 6 \\
  --params '{"lottery_file":"daily3.json","trials":200,"resume_study":true,
             "study_name":"window_opt_1772507547"}'

# With persistent workers (window_optimizer direct)
PYTHONPATH=. python3 window_optimizer.py --lottery-file daily3.json --strategy bayesian \\
  --max-seeds 5000000 --prng-type java_lcg --trials 200 \\
  --use-persistent-workers --worker-pool-size 4 \\
  --resume-study --study-name window_opt_1772507547 \\
  --trse-context trse_context.json --enable-pruning --n-parallel 2

# Run pipeline manually (partial)
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 3 --end-step 6""",
    "GUIDE WATCHER CLI examples")


# =============================================================================
# DOC 2 -- CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md
# =============================================================================
C9 = f"{DOCS}/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md"

append_section(C9, "## Persistent Worker Engine (S130/S134/S135)",
"""
---

## Persistent Worker Engine (S130/S134/S135)

**Module:** `persistent_worker_coordinator.py` -- 855-line STANDALONE module.
**Key invariant:** Zero changes to `coordinator.py`. Drop-in parallel path activated by
`--use-persistent-workers` flag on `window_optimizer.py`.

WATCHER passes flag via manifest -> `window_optimizer_integration_final.py`
-> `run_trial_persistent()` shim -> `PersistentWorkerCoordinator` class.

### Call Chain

```
watcher_agent.py
  -> window_optimizer_integration_final.py  (use_persistent_workers=True)
    -> run_trial_persistent()  (shim at persistent_worker_coordinator.py:669)
      -> PersistentWorkerCoordinator.run_sieve_pass()
            Zeus path:   execute_local_sieve_job()  -> sieve_filter.py subprocess
            Remote path: _dispatch_to_worker()      -> sieve_gpu_worker.py --persistent
```

### IPC Protocol (verified against live sieve_gpu_worker.py)

| Direction | Message |
|-----------|---------|
| Worker -> Coordinator (startup) | {"status":"ready","gpu_id":N,"device":"..."} |
| Coordinator -> Worker (job) | {"command":"sieve","job":{...}} |
| Worker -> Coordinator (success) | {"status":"ok","job_id":"...","elapsed_s":N,"result":{...}} |
| Worker -> Coordinator (error) | {"status":"error","job_id":"...","error":"...","traceback":"..."} |
| Coordinator -> Worker (shutdown) | {"command":"shutdown"} |

### Worker Pool

- 24 AMD workers (8/rig x 3 rigs); Zeus uses local path (no persistent worker needed)
- Spawn stagger: 4.0s per gpu_id (ROCm stability, avoids HIP init storm)
- Pool size cap: worker_pool_size (manifest default 8; stable at 4 vs memory pressure)
- SSH banner drain: -q flag + drain loop reads until line contains "status" and "ready" (Bug 8, S135)
- Dispatch lock: threading.Lock per WorkerHandle -- prevents concurrent chunk collision (Bug 9, S135)

### ROCm Stability Env Vars

HSA_OVERRIDE_GFX_VERSION=10.3.0  (correct GFX version for RX 6600)
HSA_ENABLE_SDMA=0                (reduce HIP init crashes)

### Fault Tolerance (Gate 1, validated S131)

Three fallback paths -> execute_remote_job: spawn failure, broken pipe, worker death mid-job.
Empty-readline dead-pipe: proc.poll() check fires within 20ms.

### Throughput

| Mode | Aggregate sps |
|------|--------------|
| Subprocess baseline (S128) | 849,469 |
| Persistent workers (S130) | 2,082,140 (+150%) |

---

## GPU Throughput -- Corrected Values (S129B-A)

Early-probing values caused chunk size miscalculation (chunks sized at 19k-40k seeds):

| GPU | Before (stale) | After (corrected) |
|-----|----------------|-------------------|
| RTX 3080 Ti sps | 29,000 | 2,210,000 |
| RX 6600 sps | 5,000 | 787,950 |

---

## Seed Caps -- Validated Operating Points (S129B-A / S131)

| File | Parameter | Value |
|------|-----------|-------|
| coordinator.py CLI | --seed-cap-nvidia | 5,000,000 |
| coordinator.py CLI | --seed-cap-amd | 2,000,000 |
| coordinator.py constructor default | seed_cap_nvidia | 5,000,000 |
| coordinator.py constructor default | seed_cap_amd | 2,000,000 |
| window_optimizer_integration_final.py | 2 instantiation sites | explicit (S131) |
| agent_manifests/window_optimizer.json | seed_cap_nvidia/amd | declared v1.5.0 (S131) |

---

## rig-6600c Throughput Note

rig-6600c (192.168.3.162) has an i5-8400T CPU -- ~50% throughput deficit vs other rigs.
Hardware limitation. Consider per-node seed budget or partition exclusion in future.
""", "CH9 persistent worker engine")


# =============================================================================
# DOC 3 -- CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md
# =============================================================================
C10 = f"{DOCS}/CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md"

append_section(C10, "## Step 0 Integration: TRSE (S121/S122)",
"""
---

## Step 0 Integration: TRSE (S121/S122)

**Module:** `trse_step0.py` v1.15.1
**Invariant:** skip_on_fail=True -- TRSE failure NEVER halts pipeline.

### confirmed_windows Feedback Loop

After Step 1 produces survivors, their window configs are written back to
`trse_context.json` as `confirmed_windows`. TRSE uses these on subsequent runs
to bias regime boundary detection.

### regime_type Fix (v1.15.1, S122)

Previous versions used absolute normalization causing `regime_type=unknown` on most
segments. v1.15.1 switched to relative normalization -- regime classification functional.

### Pydantic ge=1 -> ge=0 Fix (S122)

Five agent files had `step: int = Field(ge=1)` which rejected step=0 dispatch.
All five fixed to `ge=0` so TRSE can be dispatched through WATCHER.

### Survivor Threshold (S122)

WATCHER validation threshold lowered from >=100 to >=50 survivors to prevent
false ESCALATE with real data on Steps 1 and 3.

### WATCHER CLI with TRSE

```bash
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \\
  --start-step 0 --end-step 6 \\
  --params '{"lottery_file":"daily3.json","trials":200,
             "resume_study":true,"study_name":"window_opt_1772507547",
             "trse_context":"trse_context.json"}'
```
""", "CH10 TRSE Step 0")


# =============================================================================
# DOC 4 -- CHAPTER_12_WATCHER_AGENT.md
# =============================================================================
C12 = f"{DOCS}/CHAPTER_12_WATCHER_AGENT.md"

append_section(C12, "## S114-S131 CLI Parameter Additions",
"""
---

## S114-S131 CLI Parameter Additions

| Parameter | Added | Notes |
|-----------|-------|-------|
| --node-allowlist | S115 | Restrict trial execution to named nodes |
| --enable-pruning | S116/S118 | Enable Optuna MedianPruner |
| --n-parallel | S116/S118 | Parallel Optuna workers |
| --use-persistent-workers | S130 | Activate persistent_worker_coordinator.py engine |
| --worker-pool-size | S130 | Workers per rig (default: 8; stable at 4) |
| --seed-cap-nvidia | S131 | Seed cap for RTX GPUs (default: 5,000,000) |
| --seed-cap-amd | S131 | Seed cap for AMD GPUs (default: 2,000,000) |
| --trse-context | S122 | Path to trse_context.json |

### Manifest v1.5.0 (S131)

agent_manifests/window_optimizer.json bumped to v1.5.0.
seed_cap_nvidia and seed_cap_amd added to default_params.
Invariant: all new CLI params MUST appear in manifest default_params.

### args_map Fix (S123)

watcher_agent.py rebuilt from commit e184a8c. args_map in manifest was missing
4 keys introduced in S116/S118. All gaps closed; full 0->6 smoke test passed
(commits 7a6a63c, 888cf3e, 478ff74).

### Timeout Overrides (S131)

Per-step timeout configurable via manifest actions[N].timeout_seconds.
Step 1 default: 14400s (4 hours) for 200-trial persistent-worker runs.
""", "CH12 S114-S131 params")


# =============================================================================
# DOC 5 -- CHAPTER_14_TRAINING_DIAGNOSTICS.md
# =============================================================================
C14 = f"{DOCS}/CHAPTER_14_TRAINING_DIAGNOSTICS.md"

append_section(C14, "## ML Target Update: holdout_quality (S111)",
"""
---

## ML Target Update: holdout_quality (S111)

### Problem

holdout_hits as ML target produced R2=0.000155 on real CA Daily 3 data -- zero signal.

### Fix (S111)

holdout_quality (composite score) replaced holdout_hits.
CatBoost R2 immediately rose to +0.0046 on clean real data baseline.

### NN Y-Normalization Fix (S121)

train_single_trial.py line 499: added y_mean/y_std normalization with sidecar .json
for Step 6 inverse-transform. First positive NN R2: +0.020538.

### Feature Signal on Real Data (CatBoost)

| Feature Group | Importance | Notes |
|---------------|-----------|-------|
| mod-8 residue | ~32% | LCG low-bit algebraic structure |
| Prediction residuals | ~20% | pred_std, residual_abs_mean |
| mod-125 | <10% | |
| mod-1000 | <10% | |
| 24/62 base features | 0% | Battery Tier 1A added to address gap |

### Battery Tier 1A (S113) -- 23 columns added

F1 Spectral FFT (5): batt_fft_peak_mag, batt_fft_secondary_peak, batt_fft_spectral_conc,
  batt_fft_diff_peak, batt_fft_diff_conc
F5 Autocorrelation (12): batt_ac_lag_01..10, batt_ac_decay_rate, batt_ac_sig_lag_count
F7 Cumulative Sum (3): batt_cs_max_excursion, batt_cs_mean_excursion, batt_cs_zero_crossings
F6 Bit Frequency (3): batt_bf_hamming_mean, batt_bf_hamming_std, batt_bf_popcount_bias

### Z10xZ10xZ10 Digit Transition Features (S119) -- 4 columns

Full utilization requires Z10xZ10xZ10 kernel in sieve_gpu_worker.py -- TB proposal needed first.
""", "CH14 holdout_quality")


# =============================================================================
# DOC 6 -- CHAPTER_1_WINDOW_OPTIMIZER.md
# =============================================================================
C1 = f"{DOCS}/CHAPTER_1_WINDOW_OPTIMIZER.md"

append_section(C1, "## Persistent Worker Call Chain (S130/S134/S135)",
"""
---

## Persistent Worker Call Chain (S130/S134/S135)

When --use-persistent-workers is set, window_optimizer_integration_final.py routes
through the run_trial_persistent() shim in persistent_worker_coordinator.py:669
instead of the standard coordinator path.

Call chain:
```
watcher_agent.py
  -> window_optimizer_integration_final.py  (use_persistent_workers=True)
    -> run_trial_persistent()  (persistent_worker_coordinator.py:669)
      -> PersistentWorkerCoordinator
            Zeus:    execute_local_sieve_job()  -> sieve_filter.py
            Remote:  _dispatch_to_worker()      -> sieve_gpu_worker.py --persistent
```

Invariant: persistent_worker_coordinator.py is STANDALONE.
Zero changes to coordinator.py, window_optimizer.py, or window_optimizer_integration_final.py.
The default subprocess path is completely untouched -- --use-persistent-workers is additive only.

### Optuna Resume

Active study: window_opt_1772507547.db (21 trials as of S132).
Flag: --resume-study --study-name window_opt_1772507547
Storage: JournalStorage (not SQLite). Trial-unique output paths prevent cross-trial collisions.

### enable_pruning / n_parallel Fix History (S116/S118/S123)

Both flags required fixes through the full call chain:
- CLI -> run_bayesian_optimization() signature (S116)
- -> optimize_window() signature -- enable_pruning was missing (S118)
- -> agent_manifests/window_optimizer.json args_map -- 4 keys missing (S123)
""", "CH1 persistent worker call chain")


# =============================================================================
# DOC 7 -- README.md (root)
# =============================================================================
README = f"{ROOT}/README.md"

with open(README) as f:
    readme_content = f.read()

if "## Pipeline (Steps 0" not in readme_content:
    bak = README + ".bak_s136"
    if not os.path.exists(bak):
        shutil.copy(README, bak)
    before = len(readme_content.splitlines())
    new_readme = """\
# Distributed PRNG Analysis System
**Updated:** S135 (2026-03-10)

A distributed system for reverse-engineering PRNG behavioral patterns via bidirectional
sieve validation, ML ensemble scoring, and autonomous agent orchestration.
Modular and configurable -- not tied to any specific PRNG type or data source.

## Pipeline (Steps 0->6)

| Step | Module | Description |
|------|--------|-------------|
| Step 0 | trse_step0.py | Temporal Regime Segmentation -- skip_on_fail |
| Step 1 | window_optimizer.py | Bayesian TPE window search, 200 trials |
| Step 2 | scorer meta-optimizer | Distributed scoring config optimization |
| Step 3 | survivor_scorer.py | 91-feature extraction per survivor |
| Step 4 | anti-overfit training | PRNG-agnostic feature training |
| Step 5 | meta_prediction_optimizer_anti_overfit.py | 4-model ML comparison |
| Step 6 | prediction_generator.py | Final prediction pool generation |

## Cluster Configuration

| Node | GPUs | Backend | IP |
|------|------|---------|-----|
| Zeus (primary) | 2x RTX 3080 Ti | CUDA | localhost |
| rig-6600 | 8x RX 6600 | ROCm | 192.168.3.120 |
| rig-6600b | 8x RX 6600 | ROCm | 192.168.3.154 |
| rig-6600c | 8x RX 6600 | ROCm | 192.168.3.162 |

**Total: 26 GPUs**
Note: rig-6600c has i5-8400T CPU (~50% throughput deficit vs other rigs)

## Key Metrics

- Aggregate throughput: 2,082,140 sps (persistent workers active, S130)
- Best survivor run: 17,247 bidirectional survivors
- Real data: 18,068 draws (CA Daily 3, 2000-2026)
- Primary Optuna study: window_opt_1772507547.db (resume target)
- Best ML model: CatBoost, R2~+0.005-+0.010 (holdout_quality target)

## WATCHER CLI

```bash
# Full 0->6 pipeline
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \\
  --start-step 0 --end-step 6 \\
  --params '{"lottery_file":"daily3.json","trials":200,
             "resume_study":true,"study_name":"window_opt_1772507547"}'

# With persistent workers (direct window_optimizer)
PYTHONPATH=. python3 window_optimizer.py --lottery-file daily3.json \\
  --strategy bayesian --max-seeds 5000000 --prng-type java_lcg \\
  --trials 200 --use-persistent-workers --worker-pool-size 4 \\
  --resume-study --study-name window_opt_1772507547
```

## Environment Setup

```bash
# Zeus (CUDA)
source ~/venvs/torch/bin/activate
cd ~/distributed_prng_analysis

# AMD rigs (ROCm)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
source ~/rocm_env/bin/activate
```

## Repository

- Private: git@github.com:mmercalde/prng_cluster_project.git  (remote: origin)
- Public:  git@github.com:mmercalde/prng_cluster_public.git   (remote: public)
- Push:    git push origin main && git push public main

## Key Documentation (docs/)

| File | Contents |
|------|----------|
| COMPLETE_OPERATING_GUIDE_v2_0.md | Full system reference (v2.1.0) |
| CHAPTER_1_WINDOW_OPTIMIZER.md | Step 1, persistent worker call chain |
| CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md | GPU specs, seed caps, throughput |
| CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md | WATCHER, TRSE, autonomy |
| CHAPTER_12_WATCHER_AGENT.md | CLI params S114-S131, manifest v1.5.0 |
| CHAPTER_14_TRAINING_DIAGNOSTICS.md | ML targets, feature signal |
"""
    with open(README, "w") as f:
        f.write(new_readme)
    after = len(new_readme.splitlines())
    print(f"  OK   README full rewrite: {before} -> {after} lines")
else:
    print("  SKIP README: already updated")


# =============================================================================
# Summary
# =============================================================================
print()
if _errors == 0:
    print("ALL PATCHES APPLIED CLEANLY -- ready to commit")
else:
    print(f"WARNING: {_errors} patch(es) failed -- review SKIP/MISS lines before committing")
    sys.exit(1)

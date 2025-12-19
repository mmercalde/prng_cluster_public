# Complete Workflow Guide v2.1 - scripts_coordinator.py Update

**Document Version:** 2.1  
**Date:** December 18, 2025  
**Change:** Added scripts_coordinator.py v1.4.0 documentation

---

## NEW: scripts_coordinator.py v1.4.0

### Overview

`scripts_coordinator.py` replaces `coordinator.py` for ML script-based jobs (Steps 3 and 5). It provides:
- **100% job success rate** (vs 72% with coordinator.py)
- **File-based success detection** (no stdout parsing)
- **ML-agnostic design** (works with any script-based worker)

### Usage

```bash
# Step 3 - Full Scoring (default)
python3 scripts_coordinator.py --jobs-file scoring_jobs.json

# Step 5 - Anti-Overfit Trials
python3 scripts_coordinator.py --jobs-file anti_overfit_jobs.json \
    --output-dir anti_overfit_results --preserve-paths

# Dry run - preview job distribution
python3 scripts_coordinator.py --jobs-file scoring_jobs.json --dry-run
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--jobs-file` | Required | Job specifications JSON |
| `--config` | `distributed_config.json` | Cluster config |
| `--output-dir` | `full_scoring_results` | Base output directory |
| `--preserve-paths` | `False` | Don't rewrite job output paths |
| `--dry-run` | `False` | Preview without execution |
| `--max-retries` | `3` | Retry failed jobs on localhost |
| `--verbose` | `False` | Show failure details |

### SCRIPT JOB SPEC v1.0

All jobs must conform to this frozen contract:

```json
{
    "job_id": "full_scoring_0000",
    "script": "full_scoring_worker.py",
    "args": ["--seeds-file", "chunk_0000.json", "--output-file", "results.json"],
    "expected_output": "full_scoring_results/chunk_0000.json",
    "timeout": 7200
}
```

### Output Structure

```
full_scoring_results/
└── full_scoring_results_20251218_191950/   # Run-scoped directory
    ├── scripts_run_manifest.json            # Auditability manifest
    ├── chunk_0000.json
    ├── chunk_0001.json
    └── ...
```

### Manifest Format

```json
{
    "manifest_version": "1.0.0",
    "script_job_spec_version": "1.0",
    "run_id": "full_scoring_results_20251218_191950",
    "jobs_expected": 36,
    "jobs_completed": 36,
    "jobs_failed": 0,
    "runtime_seconds": 261.0,
    "outputs": ["chunk_0000.json", "chunk_0001.json", ...],
    "failures": []
}
```

---

## Updated Step 3 Pipeline

### Phase 3 Change

**Before (coordinator.py):**
```bash
python3 coordinator.py --jobs-file scoring_jobs.json
```

**After (scripts_coordinator.py):**
```bash
python3 scripts_coordinator.py --jobs-file scoring_jobs.json --config distributed_config.json
```

### Complete Pipeline

```
Phase 1: generate_step3_scoring_jobs.py
    → scoring_jobs.json, scoring_chunks/

Phase 2: SCP data to remotes
    → Files distributed to rig-6600, rig-6600b

Phase 3: scripts_coordinator.py  ← UPDATED
    → full_scoring_results/{run_id}/chunk_XXXX.json

Phase 4: SCP results from remotes
    → All chunks pulled to localhost

Phase 5: Aggregate results
    → survivors_with_scores.json

Phase 6: Validate output
    → Confirm 50 features per survivor
```

---

## Updated Step 5 Pipeline

### Using scripts_coordinator.py

```bash
# Generate jobs
python3 generate_anti_overfit_jobs.py \
    --trials 100 \
    --survivors survivors_with_scores.json \
    --lottery-data train_history.json \
    --study-name anti_overfit_study \
    --study-db "sqlite:///optuna_studies/anti_overfit.db" \
    --output anti_overfit_jobs.json

# Distribute data
for node in 192.168.3.120 192.168.3.154; do
    scp survivors_with_scores.json train_history.json $node:~/distributed_prng_analysis/
done

# Execute with scripts_coordinator
python3 scripts_coordinator.py \
    --jobs-file anti_overfit_jobs.json \
    --output-dir anti_overfit_results \
    --preserve-paths

# Pull and aggregate results
# (same as before)
```

---

## Execution Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                   scripts_coordinator.py                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Thread 1 (localhost):  GPU0 → GPU1 → GPU0 → ...  (3s stagger)      │
│  Thread 2 (rig-6600):   GPU0 → GPU1 → ... → GPU11 → ...  (0.5s)     │
│  Thread 3 (rig-6600b):  GPU0 → GPU1 → ... → GPU11 → ...  (0.5s)     │
│                                                                      │
│  Success = output file exists AND size > 0                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Environment Protocols (Unchanged)

| Node | Python Env | GPU Env Var |
|------|------------|-------------|
| localhost | `/home/michael/venvs/torch/bin/python` | `CUDA_VISIBLE_DEVICES` |
| rig-6600 | `/home/michael/rocm_env/bin/python` | `HIP_VISIBLE_DEVICES` |
| rig-6600b | `/home/michael/rocm_env/bin/python` | `HIP_VISIBLE_DEVICES` |

ROCm nodes also require: `HSA_OVERRIDE_GFX_VERSION=10.3.0`

---

## Performance Comparison

| Metric | coordinator.py | scripts_coordinator.py |
|--------|---------------|------------------------|
| Success Rate | 72% | **100%** |
| Runtime (36 jobs) | ~500s | **261s** |
| Code Lines | ~1700 | **~580** |
| Stdout Parsing | Required | **Not used** |

---

## Backward Compatibility

- Step 3 command unchanged (same `--jobs-file` format)
- All existing job generators work
- `coordinator_adapter.py` v2.0.0 handles both formats

---

**End of Workflow Guide Update v2.1**

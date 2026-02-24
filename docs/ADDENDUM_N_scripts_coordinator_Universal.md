# Addendum N: scripts_coordinator.py Universal Orchestrator

**Addendum Version:** 1.0.0  
**Date:** December 20, 2025  
**Author:** Claude (AI Assistant)  
**Status:** ✅ IMPLEMENTED  
**Session:** 14-15  
**Parent Document:** PROPOSAL_Unified_Agent_Context_Framework_v3_2_5.md

---

## 1. Executive Summary

**IMPORTANT UPDATE:** As of Session 14, `scripts_coordinator.py` v1.4.0 replaces `coordinator.py` as the primary job orchestrator for Steps 3, 4, 5, and 6.

This addendum corrects references in Addendum L and documents the architectural shift.

---

## 2. Architectural Change

### 2.1 Before (coordinator.py)

```
coordinator.py (~1700 lines)
├── Complex stdout JSON parsing
├── 72% success rate under concurrency
├── SSH connection overload issues
└── Tightly coupled to job format
```

### 2.2 After (scripts_coordinator.py)

```
scripts_coordinator.py (~580 lines)
├── File-based success detection (.done files)
├── 100% success rate
├── Sequential per-node, parallel across nodes
├── ML-agnostic (Steps 3, 4, 5, 6 compatible)
└── Clean, maintainable code
```

### 2.3 Performance Comparison

| Metric | coordinator.py | scripts_coordinator.py |
|--------|----------------|------------------------|
| Success Rate | 72% (26/36) | **100% (36/36)** |
| Runtime | ~500s | **261s** |
| Code Lines | ~1700 | **~580** |
| Concurrency Model | Parallel all | Sequential/node, parallel/cluster |

---

## 3. Updated Pipeline Flow

### 3.1 Step 3: Full Scoring

```
BEFORE (Addendum L):
run_step3_full_scoring.sh
    └── coordinator.py --jobs-file scoring_jobs.json  ❌ DEPRECATED

AFTER:
run_step3_full_scoring.sh
    └── scripts_coordinator.py --jobs-file scoring_jobs.json  ✅ CURRENT
```

### 3.2 Step 5: Anti-Overfit Training

```bash
# Generate jobs
python3 generate_kfold_jobs.py --config reinforcement_engine_config.json

# Execute via scripts_coordinator
python3 scripts_coordinator.py \
    --jobs-file anti_overfit_jobs.json \
    --output-dir anti_overfit_results \
    --preserve-paths
```

### 3.3 Step 6: Prediction

```bash
# Direct execution (no coordinator needed for single job)
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json \
    --lottery-history synthetic_lottery.json
```

---

## 4. scripts_coordinator.py Usage

### 4.1 Basic Usage

```bash
# Step 3 - Full Scoring (default)
python3 scripts_coordinator.py --jobs-file scoring_jobs.json

# Step 5 - Anti-Overfit Trials
python3 scripts_coordinator.py --jobs-file anti_overfit_jobs.json \
    --output-dir anti_overfit_results --preserve-paths

# Dry run (preview only)
python3 scripts_coordinator.py --jobs-file scoring_jobs.json --dry-run
```

### 4.2 Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--jobs-file` | Required | Path to jobs JSON file |
| `--output-dir` | `full_scoring_results` | Base output directory |
| `--preserve-paths` | `False` | Don't rewrite paths (for Step 5) |
| `--dry-run` | `False` | Preview only, no execution |

### 4.3 Job File Format

```json
[
  {
    "job_id": "chunk_0000",
    "script": "full_scoring_worker.py",
    "args": [
      "--seeds-file", "scoring_chunks/chunk_0000.json",
      "--train-history", "train_history.json",
      "--output-file", "full_scoring_results/chunk_0000.json",
      "--prng-type", "java_lcg",
      "--mod", "1000"
    ],
    "node": "192.168.3.120",
    "gpu_id": 0
  }
]
```

---

## 5. Corrected References

### 5.1 Addendum L Section 2.5 (Corrected)

**BEFORE:**
```
| Script | Location | Role | Called By |
|--------|----------|------|-----------|
| `coordinator.py` | Zeus | Distributed executor | Shell script |
```

**AFTER:**
```
| Script | Location | Role | Called By |
|--------|----------|------|-----------|
| `scripts_coordinator.py` | Zeus | Distributed executor | Shell script |
| `coordinator.py` | Zeus | DEPRECATED (legacy) | - |
```

### 5.2 Addendum L Section 2.6 (Corrected)

**BEFORE:**
```
coordinator.py (Zeus)
    │
    ├── Reads scoring_jobs.json (36 jobs)
    ...
```

**AFTER:**
```
scripts_coordinator.py (Zeus)
    │
    ├── Reads scoring_jobs.json (36 jobs)
    │
    ├── For each node (sequential within node, parallel across nodes):
    │   │
    │   ├── SSH to target node
    │   │   └── Execute job script with args
    │   │
    │   └── Check for .done file (success detection)
    │
    └── Aggregate results from all nodes
```

---

## 6. File-Based Success Detection

The key architectural improvement is **file-based success detection**:

```python
# Worker writes .done file on success
output_path = Path(args.output_file)
output_path.with_suffix('.json.done').touch()

# Coordinator checks for .done file
done_file = output_dir / f"{job_id}.json.done"
if done_file.exists():
    return JobResult.SUCCESS
```

This eliminates stdout JSON parsing failures under concurrency.

---

## 7. coordinator_adapter.py

For backward compatibility, `coordinator_adapter.py` v2.0.0 bridges both formats:

```bash
# Automatically routes to correct coordinator
python3 coordinator_adapter.py --jobs-file scoring_jobs.json

# Detects job format and calls:
# - scripts_coordinator.py for script-based jobs
# - coordinator.py for legacy CuPy jobs (rare)
```

---

## 8. Integration with Multi-Model Architecture

The scripts_coordinator.py works seamlessly with Multi-Model v3.1.2:

```bash
# Step 3: Score survivors (scripts_coordinator)
python3 scripts_coordinator.py --jobs-file scoring_jobs.json

# Step 5: Train model (direct, with model selection)
python3 meta_prediction_optimizer_anti_overfit.py \
    --survivors survivors_with_scores.json \
    --lottery-data synthetic_lottery.json \
    --model-type xgboost

# Step 6: Generate predictions (direct, loads from sidecar)
python3 prediction_generator.py \
    --models-dir models/reinforcement \
    --survivors-forward forward_survivors.json
```

---

## 9. Deprecation Notice

| Component | Status | Notes |
|-----------|--------|-------|
| `coordinator.py` | ⚠️ DEPRECATED | Use scripts_coordinator.py |
| `run_step3_full_scoring.sh` (old) | ⚠️ DEPRECATED | Updated v2.0.0 available |
| Stdout JSON parsing | ❌ REMOVED | File-based detection only |

---

## 10. Approval

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | Claude | 2025-12-20 | ✅ |
| Implementation | Claude | 2025-12-18 | ✅ |
| Documentation | Claude | 2025-12-20 | ✅ |
| Final Approval | Michael | 2025-12-20 | Pending |

---

**End of Addendum N v1.0.0 - scripts_coordinator.py Universal Orchestrator**

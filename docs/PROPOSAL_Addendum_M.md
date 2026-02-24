# Addendum M: scripts_coordinator.py v1.4.0 - Universal ML Script Orchestrator

**Addendum Version:** 1.0.0  
**Date:** December 18, 2025  
**Session:** 14  
**Status:** ✅ IMPLEMENTED & TESTED

---

## Overview

This addendum documents `scripts_coordinator.py` v1.4.0, a focused orchestrator that replaces `coordinator.py` for ML script-based jobs (Steps 3, 5).

---

## Problem Statement

`coordinator.py` (~1700 lines) achieved 72% success rate due to:
1. stdout JSON parsing failures under concurrency
2. Complex polymorphic job routing
3. Semantic branching between job types

**Evidence:** Direct SSH execution achieved 100% success; coordinator added failures.

---

## Solution

`scripts_coordinator.py` v1.4.0 (~580 lines):
- **File-based success**: `output file exists && size > 0`
- **No stdout parsing**
- **ML-agnostic**: Works with Steps 3, 4, 5
- **Team Beta recommendations**: Run ID scoping, manifests, explicit failures

---

## Test Results

| Metric | coordinator.py | scripts_coordinator.py |
|--------|---------------|------------------------|
| Success Rate | 72% (26/36) | **100% (36/36)** |
| Survivors | 285,211 | **395,211** |
| Runtime | ~500s | **261s** |

---

## Usage

```bash
# Step 3 (default)
python3 scripts_coordinator.py --jobs-file scoring_jobs.json

# Step 5
python3 scripts_coordinator.py --jobs-file anti_overfit_jobs.json \
    --output-dir anti_overfit_results --preserve-paths
```

---

## SCRIPT JOB SPEC v1.0 (Frozen Contract)

```json
{
    "job_id": "string",
    "script": "worker_script.py",
    "args": ["--arg1", "value1"],
    "expected_output": "path/to/output.json",
    "timeout": 7200
}
```

---

## Files

| File | Version | Purpose |
|------|---------|---------|
| `scripts_coordinator.py` | v1.4.0 | Orchestrator |
| `coordinator_adapter.py` | v2.0.0 | Format bridge |
| `run_step3_full_scoring.sh` | v2.0.0 | Step 3 runner |

---

## Approval

| Role | Name | Date |
|------|------|------|
| Author | Claude | 2025-12-18 |
| Testing | Michael | 2025-12-18 |
| Approval | Michael | 2025-12-18 |

---

**End of Addendum M v1.0.0**

# Chapter 9: GPU Cluster Infrastructure

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 2.1.0 (Bug Fixes + Memory Config)  
**Files:** `coordinator.py`, `scripts_coordinator.py`, `distributed_worker.py`, `gpu_optimizer.py`  
**Purpose:** 26-GPU distributed job orchestration and execution

---

## Table of Contents

1. [Overview](#1-overview)
2. [Cluster Architecture](#2-cluster-architecture)
3. [Two Coordinators](#3-two-coordinators)
4. [coordinator.py — Seed-Based Jobs](#4-coordinatorpy--seed-based-jobs)
5. [scripts_coordinator.py — ML Script Jobs](#5-scripts_coordinatorpy--ml-script-jobs)
6. [distributed_worker.py — GPU Execution Agent](#6-distributed_workerpy--gpu-execution-agent)
7. [gpu_optimizer.py — Workload Distribution](#7-gpu_optimizerpy--workload-distribution)
8. [ROCm Environment Setup](#8-rocm-environment-setup)
9. [Job Types and Routing](#9-job-types-and-routing)
10. [Fault Tolerance](#10-fault-tolerance)
11. [CLI Reference](#11-cli-reference)

---

## 1. Overview

### 1.1 What the Infrastructure Does

The GPU cluster infrastructure provides:

- **Job Distribution** — Optimal workload allocation across 26 heterogeneous GPUs
- **Two Coordinators** — Seed-based (Steps 1, 2) and script-based (Steps 3-6)
- **Worker Execution** — GPU-bound processing on CUDA (NVIDIA) and ROCm (AMD)
- **Fault Tolerance** — Automatic retry, progress persistence, graceful recovery

### 1.2 Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Seed Coordinator | `coordinator.py` | Distribute seed range jobs |
| Script Coordinator | `scripts_coordinator.py` | Distribute ML script jobs |
| Worker Agent | `distributed_worker.py` | Execute jobs on GPU |
| GPU Optimizer | `gpu_optimizer.py` | Optimal chunk sizing |

---

## 2. Cluster Architecture

### 2.1 Hardware Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    ZEUS (Coordinator)                        │
│           Intel i9-9920X • 64GB RAM • CUDA                  │
│           2× RTX 3080 Ti • 24GB VRAM total                  │
│              Job dispatch, result aggregation                │
└─────────────────────────┬───────────────────────────────────┘
                          │ SSH + SFTP
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   rig-6600    │ │   rig-6600b   │ │  (planned)    │
│ Intel i5-9400 │ │ Intel i5-8400 │ │               │
│   7.7GB RAM   │ │   7.7GB RAM   │ │               │
│ 12× RX 6600   │ │ 12× RX 6600   │ │ 12× RX 6600   │
│   ROCm 6.4    │ │   ROCm 6.4    │ │   ROCm 6.4    │
│   96GB VRAM   │ │   96GB VRAM   │ │   96GB VRAM   │
└───────────────┘ └───────────────┘ └───────────────┘

Total: 26 GPUs • ~285 TFLOPS • 216GB VRAM
```

### 2.2 Performance Characteristics

| Node | CPU | RAM | GPUs | Arch | TFLOPS | Role |
|------|-----|-----|------|------|--------|------|
| Zeus | i9-9920X | 64GB | 2× RTX 3080 Ti | CUDA | ~70 | Coordinator + Worker |
| rig-6600 | i5-9400 | 7.7GB | 12× RX 6600 | ROCm | ~108 | Worker |
| rig-6600b | i5-8400 | 7.7GB | 12× RX 6600 | ROCm | ~108 | Worker |

### 2.3 Memory Constraints

| Node | Total RAM | Available | Constraint |
|------|-----------|-----------|------------|
| Zeus | 64 GB | ~50 GB | None |
| rig-6600 | 7.7 GB | ~4.8 GB | **RAM limits concurrency** |
| rig-6600b | 7.7 GB | ~4.8 GB | **RAM limits concurrency** |

> **Critical:** Mining rigs are RAM-constrained, not CPU or GPU constrained. See Section 5.7 for memory-safe configuration.

---

## 3. Two Coordinators

### 3.1 Why Two Coordinators?

| Coordinator | Job Type | Pipeline Steps | Pattern |
|-------------|----------|----------------|---------|
| `coordinator.py` | Seed ranges | 1, 2 | GPU sieve kernels |
| `scripts_coordinator.py` | Python scripts | 3, 4, 5, 6 | ML/scoring scripts |

### 3.2 When to Use Which

```python
# Step 1 (Window Optimizer) - Uses coordinator.py internally
python3 window_optimizer.py --strategy bayesian --trials 50

# Step 2 (Scorer Meta) - Use scripts_coordinator.py with job file
python3 scripts_coordinator.py --jobs-file scorer_jobs.json --output-dir scorer_trial_results --preserve-paths

# Step 3 (Full Scoring) - Use scripts_coordinator.py
bash run_step3_full_scoring.sh --chunk-size 1000

# Step 5 (Training) - Use scripts_coordinator.py
python3 scripts_coordinator.py --jobs-file anti_overfit_jobs.json
```

---

## 4. coordinator.py — Seed-Based Jobs

### 4.1 Purpose

Orchestrates seed-range jobs across the 26-GPU cluster for sieve operations.

### 4.2 Key Features

| Feature | Description |
|---------|-------------|
| SSH Connection Pool | Persistent connections to all nodes |
| Pull Architecture | Workers write locally, coordinator pulls via SCP |
| Fault Tolerance | Automatic retry with exponential backoff |
| Progress Persistence | Resume from checkpoint after interruption |

### 4.3 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      COORDINATOR.PY                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ Job Queue   │───▶│ Dispatcher  │───▶│ SSH Pool    │         │
│  │ (priority)  │    │ (parallel)  │    │ (persistent)│         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                                │                 │
│                                    ┌───────────┴───────────┐    │
│                                    │     Remote Workers    │    │
│                                    │  ┌─────┐ ┌─────┐     │    │
│                                    │  │GPU 0│ │GPU 1│ ... │    │
│                                    │  └─────┘ └─────┘     │    │
│                                    └───────────────────────┘    │
│                                                │                 │
│  ┌─────────────┐    ┌─────────────┐           │                 │
│  │ Results     │◀───│ Collector   │◀──────────┘                 │
│  │ Aggregator  │    │ (SCP pull)  │                             │
│  └─────────────┘    └─────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Job Distribution

```python
# coordinator.py distributes based on GPU capability
chunk_size = gpu_optimizer.calculate_optimal_chunk_size(
    worker.gpu_type,  # "RTX 3080 Ti" or "RX 6600"
    base_chunk_size   # 10000
)

# RTX 3080 Ti: 60,000 seeds (6x scaling)
# RX 6600: 10,000 seeds (1x baseline)
```

### 4.5 Pull Architecture (Step 2)

```
1. Coordinator dispatches trial via SSH
2. Worker writes result locally:
   ~/distributed_prng_analysis/scorer_trial_results/trial_42.json
3. Coordinator pulls via SCP
4. Coordinator deletes remote file
5. Coordinator aggregates all results
```

---

## 5. scripts_coordinator.py — ML Script Jobs

### 5.1 Purpose

Orchestrates Python script execution for ML/scoring steps (3-6).

### 5.2 Key Differences from coordinator.py

| Feature | coordinator.py | scripts_coordinator.py |
|---------|---------------|------------------------|
| Job type | Seed ranges | Script + args |
| Execution | GPU kernel | Python subprocess |
| Concurrency | Per-GPU | Per-node limited |
| Output | JSON stdout | File-based detection |

### 5.3 Per-Node Concurrency Configuration

**Updated: 2026-01-24** — `max_concurrent_script_jobs` now properly enforced.

Concurrency is configured in `distributed_config.json`:

```json
{
  "nodes": [
    {
      "hostname": "localhost",
      "gpu_count": 2,
      "max_concurrent_script_jobs": 2
    },
    {
      "hostname": "192.168.3.120",
      "gpu_count": 12,
      "max_concurrent_script_jobs": 12
    },
    {
      "hostname": "192.168.3.154",
      "gpu_count": 12,
      "max_concurrent_script_jobs": 12
    }
  ]
}
```

| Node | CPU | GPUs | max_concurrent | Rationale |
|------|-----|------|----------------|-----------|
| Zeus | i9-9920X | 2 | 2 | Match GPU count |
| rig-6600 | i5-9400 | 12 | 12 | Full utilization (validated 2026-01-18) |
| rig-6600b | i5-8400 | 12 | 12 | Full utilization (validated 2026-01-18) |

> **Historical Note:** Earlier versions limited mining rigs to 1 concurrent job due to assumed CPU weakness. Benchmarking (2026-01-18) proved i5-9400/i5-8400 CPUs are sufficient for 12 concurrent workers when `sample_size` or `chunk_size` is properly controlled.

### 5.4 Job Format

```json
{
  "jobs": [
    {
      "job_id": "scoring_chunk_0001",
      "script": "full_scoring_worker.py",
      "args": [
        "--survivors", "chunk_0001.json",
        "--output", "results/chunk_0001.json"
      ],
      "gpu_id": 0
    }
  ]
}
```

### 5.5 Execution Flow

```
1. Load jobs from JSON file
2. For each job:
   a. Select least-loaded node (respecting concurrency limits)
   b. SSH to node with GPU binding
   c. Execute: python3 script.py --args
   d. Detect completion via output file
   e. Collect results
3. Aggregate all results
```

### 5.6 Critical Bug Fixes (January 24, 2026)

Three bugs in `scripts_coordinator.py` were discovered and fixed:

#### Bug #1: Job Type Detection (Line 139)

**Problem:** `AttributeError: 'Job' object has no attribute 'get'`

```python
# BEFORE (broken)
job_type = job.get('job_type', '')

# AFTER (fixed)
job_type = getattr(job, 'job_type', '')
```

**Cause:** Job is a dataclass, not a dict.

#### Bug #2: Jobs Orphaned When GPU Limit < gpu_count (Lines 830-833)

**Problem:** With `max_concurrent_script_jobs: 4` on a 12-GPU node, only 61/99 jobs completed.

**Root Cause:** Jobs were distributed to all 12 GPUs during assignment, but only GPUs 0-3 executed.

```python
# BEFORE (broken) - distributed to ALL GPUs
gpu_jobs = {i: [] for i in range(node.gpu_count)}        # 12 slots
for i, job in enumerate(jobs):
    gpu_jobs[i % node.gpu_count].append(job)             # Jobs go to GPUs 4-11 too

# AFTER (fixed) - distribute only to ACTIVE GPUs
num_active_gpus = min(node.gpu_count, node.max_concurrent)
gpu_jobs = {i: [] for i in range(num_active_gpus)}
for i, job in enumerate(jobs):
    gpu_jobs[i % num_active_gpus].append(job)
```

#### Bug #3: Active GPU Slicing (Line 857)

**Context:** The slicing was already present but only worked correctly after Bug #2 was fixed.

```python
active_gpus = [gid for gid, jlist in gpu_jobs.items() if jlist][:node.max_concurrent]
```

#### Verification

```bash
# Test with explicit 4-GPU limit
sed -i 's/"max_concurrent_script_jobs": 12/"max_concurrent_script_jobs": 4/g' distributed_config.json
bash run_step3_full_scoring.sh --chunk-size 1000
# Result: 99/99 jobs complete (was 61/99 before fix)
```

### 5.7 Step 3 Memory-Safe Configuration

**Added: 2026-01-24** — OOM prevention for mining rigs.

#### The Problem

Mining rigs have limited RAM (7.7 GB) but 12 GPUs with 8 GB VRAM each. Step 3 (Full Scoring) loads survivor data into **RAM**, not VRAM. This is the bottleneck.

```
Total RAM:        7,845 MB
System/ROCm:     ~3,000 MB
Available:       ~4,800 MB
12 workers × ?:   Must fit in 4,800 MB
```

#### The Solution

Use `chunk_size=1000` to limit per-worker memory:

```bash
# Memory-safe Step 3 execution
bash run_step3_full_scoring.sh --chunk-size 1000
```

| Parameter | Value | Effect |
|-----------|-------|--------|
| chunk_size | 1000 | ~500 MB per worker |
| Workers | 12 | ~6,000 MB total |
| Result | ✅ | Fits with margin |

#### Why chunk_size, Not Worker Count?

| Approach | Tradeoff |
|----------|----------|
| Reduce workers | ❌ Wastes GPU capacity |
| Reduce chunk_size | ✅ More jobs, same GPU utilization |

With `chunk_size=1000`:
- 98,172 survivors ÷ 1,000 = 99 jobs
- All 26 GPUs stay busy
- Zero OOM errors
- Runtime: ~313 seconds

#### Recommended Configuration

```bash
# distributed_config.json
"max_concurrent_script_jobs": 12    # Full GPU utilization

# Step 3 execution
bash run_step3_full_scoring.sh --chunk-size 1000
```

#### Verification (January 24, 2026)

```
✅ 99/99 jobs completed
✅ Zero OOM errors
✅ All 26 GPUs utilized
✅ Runtime: 312.9 seconds
✅ WATCHER agent evaluation: Confidence 1.00
```

---

## 6. distributed_worker.py — GPU Execution Agent

### 6.1 Purpose

GPU-bound execution agent running on every cluster node.

### 6.2 Critical: ROCm Environment Setup

**MUST be first — before ANY GPU imports:**

```python
import os
import socket

HOST = socket.gethostname()

# ROCm environment for AMD mining rigs
if HOST in ["rig-6600", "rig-6600b"]:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")

os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")

# NOW safe to import GPU libraries
import cupy as cp
import torch
```

### 6.3 GPU Device Binding

```python
# Coordinator sets environment variable
export CUDA_VISIBLE_DEVICES=3  # (or HIP_VISIBLE_DEVICES for ROCm)

# Worker sees only that GPU as device 0
cp.cuda.Device(0).use()
```

### 6.4 Job Type Routing

```python
def route_job(job_spec):
    job_type = job_spec.get('type', 'analysis')
    
    if job_type == 'script':
        return execute_script_job(job_spec)
    elif job_type == 'sieve':
        return execute_sieve_job(job_spec)
    elif job_type == 'reverse_sieve':
        return execute_reverse_sieve_job(job_spec)
    elif job_type == 'correlation':
        return analyze_correlation_gpu(job_spec)
    else:
        return run_statistical_analysis(job_spec)
```

### 6.5 Script Job Execution (v1.8.0)

**Critical:** Script jobs skip GPU initialization because subprocess handles it:

```python
def execute_script_job(job_spec):
    # DO NOT initialize GPU here - subprocess handles own setup
    script = job_spec['script']
    args = job_spec['args']
    
    result = subprocess.run(
        ['python3', script] + args,
        capture_output=True,
        text=True
    )
    
    return parse_result(result)
```

---

## 7. gpu_optimizer.py — Workload Distribution

### 7.1 Purpose

Optimal job sizing based on GPU capability.

### 7.2 Performance Profiles

```python
gpu_performance_profiles = {
    "RTX 3080 Ti": {
        "seeds_per_second": 29000,
        "scaling_factor": 6.0,      # 6x larger jobs than RX 6600
        "architecture": "CUDA",
    },
    "RX 6600": {
        "seeds_per_second": 5000,
        "scaling_factor": 1.0,      # Baseline
        "architecture": "ROCm",
    },
}
```

### 7.3 Chunk Size Calculation

```python
def calculate_optimal_chunk_size(gpu_type: str, base_size: int) -> int:
    profile = get_gpu_profile(gpu_type)
    optimal = int(base_size * profile["scaling_factor"])
    
    # Constraints
    optimal = max(500, optimal)           # Minimum
    optimal = min(base_size * 50, optimal) # Maximum
    
    return optimal
```

### 7.4 Performance Learning

```python
def update_performance(gpu_type, hostname, seeds_processed, execution_time):
    """Update GPU performance metrics from actual job results."""
    current_sps = seeds_processed / execution_time
    
    # Exponential moving average (α = 0.3)
    profile.seeds_per_second = (
        0.3 * current_sps + 
        0.7 * profile.seeds_per_second
    )
```

---

## 8. ROCm Environment Setup

### 8.1 Required Environment Variables

```bash
# For RX 6600 (gfx1032 architecture)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
```

### 8.2 Why HSA_OVERRIDE_GFX_VERSION?

RX 6600 uses gfx1032 architecture, but ROCm may not have native support. The override tells ROCm to use gfx1030 compatibility mode.

### 8.3 PyTorch Memory Configuration

```python
# For RX 6600 (8GB VRAM)
os.environ["PYTORCH_HIP_ALLOC_CONF"] = (
    "garbage_collection_threshold:0.8,max_split_size_mb:128"
)
torch.cuda.set_per_process_memory_fraction(0.8)  # 6.4GB usable
```

### 8.4 ROCm Stability Envelope (RX 6600) — VALIDATED

> **Updated: 2026-01-18** — Based on systematic benchmark testing

#### Root Cause Analysis

Prior assumptions about ROCm instability (HIP initialization collisions, GPU concurrency limits) have been superseded. Systematic benchmarking revealed the **true constraint is host memory pressure during data loading**, not GPU-side limitations.

| ❌ Previous Assumption | ✅ Validated Reality |
|------------------------|---------------------|
| "ROCm can't handle high concurrency" | Full 12-GPU concurrency is stable |
| "Weak CPUs cause contention" | i5-9400/i5-8400 are sufficient |
| "HIP init collision is dominant failure" | Memory pressure during load is the cause |
| "Reduce GPU count for stability" | Reduce sample_size/chunk_size instead |

#### Validated Configuration

```bash
# Validated ROCm configuration (RX 6600)
# Tested: 2026-01-18, 100 trials, 100% success rate

max_concurrent_script_jobs: 12     # Full GPU utilization
sample_size: 450                   # Optimal for Step 2 (scorer meta)
chunk_size: 1000                   # Optimal for Step 3 (full scoring)
ppfeaturemask: 0xffff7fff          # GFXOFF disabled
cleanup: enabled                   # Best-effort GPU allocator cleanup between jobs
```

#### Performance Envelope (Step 2)

| Sample Size | Throughput | Stability |
|-------------|------------|-----------|
| 350 | 14.98 trials/min | ✅ Stable |
| **450** | **15.41 trials/min** | ✅ **Optimal** |
| 550 | 14.66 trials/min | ✅ Stable |
| 650 | 13.13 trials/min | ✅ Stable |
| 750 | 12.45 trials/min | ✅ Stable |
| 1000 | 10.42 trials/min | ✅ Stable |
| 2000 | — | ❌ Freeze risk |

#### Required Settings

1. **GFXOFF Disabled** — Add to kernel boot params:
   ```bash
   amdgpu.ppfeaturemask=0xffff7fff
   ```

2. **Concurrency Configuration** — `distributed_config.json`:
   ```json
   {
     "hostname": "192.168.3.120",
     "max_concurrent_script_jobs": 12
   }
   ```

3. **Sample Size Cap** — `run_scorer_meta_optimizer.sh`:
   ```bash
   --sample-size 450
   ```

#### Troubleshooting: ROCm Freeze / Monitor Desync

**Symptoms:**
- `rocm-smi` shows GPU with `N/A` in SCLK/MCLK columns
- `Perf` column shows `unknown` instead of `auto`
- Jobs hang without error messages
- Monitor shows "Expected integer value" warnings

**Root Cause:** Memory pressure during concurrent data loading causes allocator thrashing.

**Fix Checklist:**
1. ✅ Reduce sample_size or chunk_size (not GPU count)
2. ✅ Verify GFXOFF disabled: `cat /sys/module/amdgpu/parameters/ppfeaturemask`
3. ✅ Verify cleanup enabled between jobs
4. ✅ Reboot rig if GPU shows persistent N/A state
5. ❌ Do NOT reduce max_concurrent_script_jobs as first response

### 8.5 Ramdisk Deployment for Step 3

**Added: 2026-01-22**

Step 3 (Full Scoring) requires ramdisk files on all nodes before distributed execution.

#### Path Structure

```
/dev/shm/prng/step3/
├── train_history.json      # ~28KB
└── holdout_history.json    # ~7KB
```

#### Deployment Commands

```bash
# From Zeus - deploy to all nodes
for node in localhost 192.168.3.120 192.168.3.154; do
    if [ "$node" = "localhost" ]; then
        mkdir -p /dev/shm/prng/step3
        cp train_history.json holdout_history.json /dev/shm/prng/step3/
    else
        ssh $node "mkdir -p /dev/shm/prng/step3"
        scp train_history.json holdout_history.json $node:/dev/shm/prng/step3/
    fi
done
```

#### Persistence Warning

**Ramdisk (`/dev/shm`) does NOT survive reboot.**

After any node reboot, you must repopulate the ramdisk before running Step 3.

#### Verification Script

```bash
#!/bin/bash
# verify_step3_ramdisk.sh
echo "=== Ramdisk Status ==="
for node in localhost 192.168.3.120 192.168.3.154; do
    echo -n "$node: "
    if [ "$node" = "localhost" ]; then
        ls /dev/shm/prng/step3/*.json 2>/dev/null | wc -l | xargs -I{} echo "{} files"
    else
        ssh $node "ls /dev/shm/prng/step3/*.json 2>/dev/null | wc -l" | xargs -I{} echo "{} files"
    fi
done
```

**Expected output:**
```
localhost: 2 files
192.168.3.120: 2 files
192.168.3.154: 2 files
```


### 8.6 GPU Diagnostic Battery

**File:** `debug_gpu_failures.sh` (v1.0.0, January 25, 2026)

Comprehensive cluster diagnostics for troubleshooting GPU failures.

**Usage:**
```bash
cd ~/distributed_prng_analysis
bash debug_gpu_failures.sh
# Output: debug_gpu_failures_YYYYMMDD_HHMMSS.log
```

**Checks Performed:**

| Check | Command | Looking For |
|-------|---------|-------------|
| Memory status | `free -h` | Available RAM < 4GB = OOM risk |
| GPU health | `rocm-smi` | N/A, "unknown", or missing GPUs |
| OOM killer | `dmesg \| grep oom` | "Killed" or "Out of memory" messages |
| ROCm/HIP errors | `dmesg \| grep amdgpu` | Driver or HIP initialization failures |
| Ramdisk status | `ls /dev/shm/prng/step3/` | Missing files |
| HIP cache size | `du -sh ~/.cache/hip_*` | Large cache = memory pressure |

**Interpreting Results:**

| Pattern in Output | Meaning | Action |
|-------------------|---------|--------|
| `Killed` in dmesg | OOM killer terminated process | Reduce `chunk_size` or concurrent workers |
| GPU shows `N/A` | GPU not responding | Reboot node |
| GPU shows `unknown` | SMU communication failure | Reboot node or reseat GPU |
| Ramdisk missing | Files not preloaded | Run `ramdisk_preload.sh` |
| HIP cache > 1GB | Memory pressure | Clear cache with `rm -rf ~/.cache/hip_*` |

### 8.7 Ramdisk Preload v2.1.0

**File:** `ramdisk_preload.sh` (v2.1.0, January 25, 2026)

**Bug Fixed:** v2.0.0 created `.ready` sentinel marker even when file copy failed, causing Step 3 to believe ramdisk was populated when files were missing.

**Changes from v2.0.0:**

| Issue | v2.0.0 | v2.1.0 |
|-------|--------|--------|
| Incomplete copy | `.ready` created anyway | Only if ALL files copied |
| Missing source files | Silent failure | Fails loudly before copying |
| Stale sentinel | Left `.ready` on failure | Removed on incomplete copy |

**Key Fix:**
```bash
# v2.1.0 FIX: Only create .ready if ALL files copied
if [ $copied -eq $expected_count ]; then
    ssh "$NODE" "touch $RAMDISK_SENTINEL"
    echo "    ✓ Preloaded ($copied files)"
else
    echo "    ❌ INCOMPLETE: Only $copied/$expected_count files copied"
    ssh "$NODE" "rm -f $RAMDISK_SENTINEL"  # Remove stale sentinel
fi
```

**Standalone Preload Script:**

For manual ramdisk population after node reboot:

```bash
# Usage
bash ramdisk_preload_fixed.sh 3  # Preload for Step 3

# Output
[11:16:32] Final verification:
[11:16:32]   ✅ localhost: All files present
[11:16:33]   ✅ 192.168.3.120: All files present
[11:16:33]   ✅ 192.168.3.154: All files present
```


---

## 9. Job Types and Routing

### 9.1 Job Type Summary

| Type | Coordinator | Worker Action |
|------|-------------|---------------|
| `sieve` | coordinator.py | `sieve_filter.py` subprocess |
| `reverse_sieve` | coordinator.py | `reverse_sieve_filter.py` subprocess |
| `script` | scripts_coordinator.py | Direct Python subprocess |
| `correlation` | coordinator.py | `analyze_correlation_gpu()` |
| `analysis` | coordinator.py | `run_statistical_analysis()` |

### 9.2 Pipeline Step to Coordinator Mapping

| Step | Coordinator | Job Type |
|------|-------------|----------|
| Step 1 | coordinator.py | sieve, reverse_sieve (via window_optimizer) |
| Step 2 | scripts_coordinator.py | script (scorer_trial_worker.py) |
| Step 3 | scripts_coordinator.py | script (full_scoring_worker.py) |
| Step 4 | scripts_coordinator.py | script (adaptive_meta_optimizer.py) |
| Step 5 | scripts_coordinator.py | script (anti_overfit_trial_worker.py) |
| Step 6 | Direct execution | No coordinator needed |

---

## 10. Fault Tolerance

### 10.1 Retry Mechanism

```python
MAX_RETRIES = 3

for attempt in range(MAX_RETRIES):
    try:
        result = execute_job(job)
        break
    except Exception as e:
        if attempt < MAX_RETRIES - 1:
            sleep(2 ** attempt)  # Exponential backoff
        else:
            mark_failed(job, str(e))
```

### 10.2 Progress Persistence

```python
# Save progress after each job
def save_progress():
    with open('progress.json', 'w') as f:
        json.dump({
            'completed': completed_jobs,
            'failed': failed_jobs,
            'pending': pending_jobs
        }, f)

# Resume from checkpoint
def load_progress():
    if os.path.exists('progress.json'):
        with open('progress.json') as f:
            return json.load(f)
    return None
```

### 10.3 Node Failure Handling

```python
def handle_node_failure(node):
    # Mark all jobs on that node as pending
    for job in node.active_jobs:
        work_queue.put(job)
    
    # Remove node from pool temporarily
    available_nodes.remove(node)
    
    # Schedule reconnection attempt
    schedule_reconnect(node, delay=60)
```

---

## 11. CLI Reference

### 11.1 coordinator.py

```bash
python3 coordinator.py [options]

Options:
  --seed-start N         Starting seed (default: 0)
  --seed-end N           Ending seed (default: 2^32)
  --jobs-file FILE       Pre-created jobs JSON
  --config FILE          Cluster config (default: distributed_config.json)
  --resume               Resume from checkpoint
  --output-dir DIR       Output directory
```

### 11.2 scripts_coordinator.py

```bash
python3 scripts_coordinator.py [options]

Options:
  --jobs-file FILE       Jobs JSON file (required)
  --output-dir DIR       Output directory
  --preserve-paths       Keep original output paths
  --resume               Resume from checkpoint
```

### 11.3 distributed_worker.py

```bash
python3 distributed_worker.py [options]

Options:
  --job-file FILE        Job specification JSON
  --gpu-id N             GPU to use (default: 0)
  --mining-mode          Enable mining rig optimizations
```

---

## 12. Chapter Summary

**Chapter 9: GPU Cluster Infrastructure** covers:

| Component | Lines | Purpose |
|-----------|-------|---------|
| coordinator.py | ~1200 | Seed-based job orchestration |
| scripts_coordinator.py | ~900 | ML script orchestration |
| distributed_worker.py | ~450 | GPU execution agent |
| gpu_optimizer.py | ~120 | Workload optimization |

**Key Points:**
- Two coordinators for different job types
- 12-GPU concurrency validated on mining rigs (2026-01-18)
- RAM (not CPU/GPU) is the constraint on mining rigs
- Use `chunk_size=1000` for Step 3 to prevent OOM
- ROCm environment MUST be set before GPU imports
- Pull architecture avoids NFS/database contention
- Automatic retry with exponential backoff

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.2.0 | 2026-01-25 | GPU diagnostic battery (8.6), ramdisk v2.1.0 fix (8.7) |
| 2.1.0 | 2026-01-24 | Bug fixes (Section 5.6), memory config (Section 5.7), correct hardware specs |
| 2.0.0 | 2026-01-22 | Consolidated, ramdisk deployment (Section 8.5) |
| 1.x | 2025-12 | Initial versions |

---

*End of Chapter 9: GPU Cluster Infrastructure*

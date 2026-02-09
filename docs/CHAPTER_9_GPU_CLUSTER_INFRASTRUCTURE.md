# Chapter 9: GPU Cluster Infrastructure

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 2.0.0 (Consolidated)  
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
- **Two Coordinators** — Seed-based (Steps 2, 2.5) and script-based (Steps 3-6)
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
│           2× RTX 3080 Ti • CUDA • 24GB VRAM                 │
│              Job dispatch, result aggregation                │
└─────────────────────────┬───────────────────────────────────┘
                          │ SSH + SFTP
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   rig-6600    │ │   rig-6600b   │ │  rig-6600c    │
│ 8× RX 6600   │ │ 8× RX 6600   │ │ 8× RX 6600   │
│   ROCm 6.4.3    │ │   ROCm 6.4.3    │ │   ROCm 6.4.3    │
│   64GB VRAM   │ │   64GB VRAM   │ │   64GB VRAM   │
└───────────────┘ └───────────────┘ └───────────────┘

Total: 26 GPUs • ~285 TFLOPS • 216GB VRAM
```

### 2.2 Performance Characteristics

| Node | GPUs | Arch | TFLOPS | Role |
|------|------|------|--------|------|
| Zeus | 2× RTX 3080 Ti | CUDA | ~70 | Coordinator + Worker |
| rig-6600 | 8× RX 6600 | ROCm | ~71 | Worker |
| rig-6600b | 8× RX 6600 | ROCm | ~71 | Worker |
| rig-6600c | 8× RX 6600 | ROCm | ~71 | Worker |

---

## 3. Two Coordinators

### 3.1 Why Two Coordinators?

| Coordinator | Job Type | Pipeline Steps | Pattern |
|-------------|----------|----------------|---------|
| `coordinator.py` | Seed ranges | 2, 2.5 | GPU sieve kernels |
| `scripts_coordinator.py` | Python scripts | 3, 4, 5, 6 | ML/scoring scripts |

### 3.2 When to Use Which

```python
# Step 2 (Sieve) - Use coordinator.py
python3 coordinator.py --seed-start 0 --seed-end 1000000000

# Step 2.5 (Scorer Meta) - Use scripts_coordinator.py with job file
python3 scripts_coordinator.py --jobs-file scorer_jobs.json --output-dir scorer_trial_results --preserve-paths

# Step 3 (Full Scoring) - Use scripts_coordinator.py
python3 scripts_coordinator.py --jobs-file scoring_jobs.json

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

### 4.5 Pull Architecture (Step 2.5)

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

### 5.3 Per-Node Concurrency Limits

**Critical for mining rigs with weak CPUs:**

```python
# scripts_coordinator.py v1.7.4.1
PER_NODE_CONCURRENCY = {
    'zeus': 2,        # Strong CPU, can handle 2 concurrent
    'rig-6600': 1,    # Weak CPU, limit to 1
    'rig-6600b': 1,   # Weak CPU, limit to 1
    'rig-6600c': 1,   # Weak CPU, limit to 1
}
```

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
if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]:
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
| Step 2 | coordinator.py | sieve, reverse_sieve |
| Step 2.5 | scripts_coordinator.py | script (scorer_trial_worker.py) |
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
| scripts_coordinator.py | ~600 | ML script orchestration |
| distributed_worker.py | ~450 | GPU execution agent |
| gpu_optimizer.py | ~120 | Workload optimization |

**Key Points:**
- Two coordinators for different job types
- Per-node concurrency limits protect weak CPUs
- ROCm environment MUST be set before GPU imports
- Pull architecture avoids NFS/database contention
- Automatic retry with exponential backoff

---

*End of Chapter 9: GPU Cluster Infrastructure*

---

### 8.4 ROCm Stability Envelope (RX 6600) — VALIDATED

> **Updated: 2026-01-18** — Based on systematic benchmark testing

#### Root Cause Analysis

Prior assumptions about ROCm instability (HIP initialization collisions, GPU concurrency limits) have been superseded. Systematic benchmarking revealed the **true constraint is host memory pressure during data loading**, not GPU-side limitations.

| ❌ Previous Assumption | ✅ Validated Reality |
|------------------------|---------------------|
| "ROCm can't handle high concurrency" | Full 12-GPU concurrency is stable |
| "Weak CPUs cause contention" | i5-9400/i5-8400 are sufficient |
| "HIP init collision is dominant failure" | Memory pressure during load is the cause |
| "Reduce GPU count for stability" | Reduce sample_size instead |

#### Validated Configuration

```bash
# Validated ROCm configuration (RX 6600)
# Tested: 2026-01-18, 100 trials, 100% success rate

max_concurrent_script_jobs: 8      # Full GPU utilization
sample_size: 450                   # Optimal operating point
ppfeaturemask: 0xffff7fff          # GFXOFF disabled
cleanup: enabled                   # Best-effort GPU allocator cleanup between jobs
```

#### Performance Envelope

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
1. ✅ Reduce sample_size (not GPU count)
2. ✅ Verify GFXOFF disabled: `cat /sys/module/amdgpu/parameters/ppfeaturemask`
3. ✅ Verify cleanup enabled between jobs
4. ✅ Reboot rig if GPU shows persistent N/A state
5. ❌ Do NOT reduce max_concurrent_script_jobs as first response


---

## 8.5 Ramdisk Deployment for Step 3

**Added: 2026-01-22**

Step 3 (Full Scoring) requires ramdisk files on all nodes before distributed execution.

### Path Structure

```
/dev/shm/prng/step3/
├── train_history.json      # ~28KB
└── holdout_history.json    # ~7KB
```

### Deployment Commands

```bash
# From Zeus - deploy to all nodes
for node in localhost 192.168.3.120 192.168.3.154 192.168.3.162; do
    if [ "$node" = "localhost" ]; then
        mkdir -p /dev/shm/prng/step3
        cp train_history.json holdout_history.json /dev/shm/prng/step3/
    else
        ssh $node "mkdir -p /dev/shm/prng/step3"
        scp train_history.json holdout_history.json $node:/dev/shm/prng/step3/
    fi
done
```

### Persistence Warning

**Ramdisk (`/dev/shm`) does NOT survive reboot.**

After any node reboot, you must repopulate the ramdisk before running Step 3.

### Verification Script

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



---

### 5.8 Configuration Location for chunk_size (CRITICAL)

**Added: 2026-01-25** — Prevents OOM confusion.

The `chunk_size=1000` memory-safe setting must be in **TWO places**:

| File | Setting | Used By |
|------|---------|---------|
| `run_step3_full_scoring.sh` | `CHUNK_SIZE=1000` | Manual runs |
| `agent_manifests/full_scoring.json` | `"chunk_size": 1000` | WATCHER automated runs |

#### Why Two Places?

- **Manual run:** `bash run_step3_full_scoring.sh` uses script's `CHUNK_SIZE=` variable
- **WATCHER run:** Agent reads `default_params` from manifest, passes to script

**Warning:** Manifest overrides script default! If WATCHER uses wrong chunk_size, check `agent_manifests/full_scoring.json` FIRST.

#### Verification Commands

```bash
# Check both locations are in sync
echo "=== Script default ==="
grep "^CHUNK_SIZE=" run_step3_full_scoring.sh

echo "=== Manifest default ==="
grep '"chunk_size"' agent_manifests/full_scoring.json | head -1
```

**Expected output:**
```
=== Script default ===
CHUNK_SIZE=1000
=== Manifest default ===
    "chunk_size": 1000,
```

#### If They Don't Match

```bash
# Fix manifest (authoritative for WATCHER)
sed -i 's/"chunk_size": [0-9]*/"chunk_size": 1000/' agent_manifests/full_scoring.json

# Fix script (authoritative for manual runs)
sed -i 's/^CHUNK_SIZE=.*/CHUNK_SIZE=1000/' run_step3_full_scoring.sh
```


---

## 6. GPU Process Isolation (Multi-Model Training)

**Added: Session 72 (Feb 8, 2026)** — Critical design invariant for Step 5.

### 6.1 The Problem: Mixed GPU Runtimes

When running multiple ML frameworks in a single Python process:

| Framework | GPU Runtime | Issue |
|-----------|-------------|-------|
| LightGBM | OpenCL | Must initialize FIRST |
| CatBoost | CUDA | Retains VRAM context |
| XGBoost | CUDA | Retains VRAM context |
| PyTorch | CUDA | Caching allocator |

**These runtimes do NOT coordinate VRAM ownership:**
- CUDA frameworks retain VRAM via caching allocators
- LightGBM's OpenCL cannot initialize after CUDA touches GPU
- Cleanup APIs (`gc.collect()`, cache clears) are **ineffective**

### 6.2 The Solution: Hard Process Isolation

Each model type runs in its own subprocess:
```
Parent Process (meta_prediction_optimizer_anti_overfit.py)
    ├── NO GPU imports
    ├── NO torch.cuda calls  
    ├── CUDA_INITIALIZED = False
    │
    └── Spawns via subprocess_trial_coordinator.py
        │
        ├── Subprocess 1: neural_net → exits (GPU released)
        ├── Subprocess 2: lightgbm  → exits (GPU released)
        ├── Subprocess 3: xgboost   → exits (GPU released)
        └── Subprocess 4: catboost  → exits (GPU released)
```

### 6.3 Design Invariant

> ⚠️ **MANDATORY: GPU-accelerated code must NEVER run in the coordinating process when using subprocess isolation.**

**Violations cause:**
- LightGBM: "Unknown OpenCL Error (-9999)"
- CatBoost: "CUDA error 46: device busy/unavailable"
- Neural net: Silent crashes
- Non-deterministic failures

### 6.4 Implementation Files

| File | Role | GPU Code? |
|------|------|-----------|
| `meta_prediction_optimizer_anti_overfit.py` | Coordinator | ❌ NO |
| `subprocess_trial_coordinator.py` | Subprocess orchestration | ❌ NO |
| `train_single_trial.py` | Single model worker | ✅ YES |

### 6.5 Verification

When `--compare-models` is active, you MUST see:
```
⚡ Mode: Multi-Model Comparison (Subprocess Isolation)
   GPU initialization DEFERRED to subprocesses
✅ CUDA initialized: False
```

Each subprocess reports its own GPU:
```
Trial 0: NEURAL_NET
  ✅ SUCCESS 🚀
  Device: cuda:0,1 (DataParallel)

Trial 1: LIGHTGBM
  ✅ SUCCESS 🚀
  Device: gpu  ← OpenCL works!
```

See `docs/DESIGN_INVARIANT_GPU_ISOLATION.md` for complete documentation.

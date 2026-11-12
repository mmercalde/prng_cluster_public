# Chapter 9: GPU Cluster Infrastructure

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

## TFM Pipeline — Operating Guide

**Version:** 3.0.0 — corrected against code (2026-11-11)
**Status:** **CORRECTED at base `a15b524`** (`a15b5247336255801de8cf67d52653701b1e8c80`),
2026-11-11. Audited in `docs/CHAPTER_9_AUDIT.md`. Not closed — see **§18** for the closure
statement, **§16** for the open items and **§17** for the completion sentinel.
**Subject:** the cluster that executes TFM jobs — nodes, coordinators, workers, chunk sizing,
ROCm environment, fault tolerance — and **RANGE-MINER (§13)**, the engine that now executes
the Step-1 sieve on that cluster.
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md`
(REV1), owner rulings §0. Team Beta is a reviewer, not an approver (§0.1); a Beta ruling that
calls itself "binding" is recorded here as a review position the owner may keep or lift.
**Numbering scheme used in this chapter:** **EXECUTABLE** step numbers
(`agents/watcher_agent.py:387-395` `STEP_SCRIPTS`): Step 1 = `window_optimizer.py` (the
bidirectional sieve runs **inside** Step 1), Step 2 = Scorer Meta-Optimizer
(`run_scorer_meta_optimizer.sh`), Step 3 = Full Scoring, Step 4 = ML Meta-Optimizer, Step 5 =
Anti-Overfit Training, Step 6 = Prediction. **Chapter numbers are not step numbers**
(`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). The pre-2026-11-11 text below (§1–§12 and the
appended sections) was written in the **conceptual** scheme ("Step 2 = sieve, Step 2.5 = scorer
meta"); where it says "Step 2" for the sieve read executable Step 1, and where it says "Step 2.5"
read executable Step 2. Each such place carries a correction note rather than a silent rewrite.

**Original header (retained, SUPERSEDED 2026-11-11 by the header above):**

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 2.0.0 (Consolidated)  
**Files:** `coordinator.py`, `scripts_coordinator.py`, `distributed_worker.py`, `gpu_optimizer.py`  
**Purpose:** 26-GPU distributed job orchestration and execution
*(Corrected 2026-11-11: the fleet is **25** GPUs — one RTX 3080 Ti on VM 101 plus 24 RX 6600 XT
— see §2 note and §14. The file list omits the Step-1 engines `persistent_worker_coordinator.py`,
`zmq_sqlite_coordinator.py` and the `miner/` package, and the fleet authority
`execution_set.py` + `rig_profiles_config.json`.)*

---

## 0. How this chapter was produced (2026-11-11)

1. Read the binding brief (§0, §1, §3, §5), the accepted model `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`
   (not edited), the audit layout `docs/CHAPTER_13_AUDIT.md`, `docs/TODO_CONSOLIDATED.md` and
   `docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1.
2. Extracted every substantive claim in the 980-line chapter and checked each against the live
   tree at `a15b524` on VM 101 — `coordinator.py`, `recovery_manager.py`, `connection_manager.py`,
   `scripts_coordinator.py`, `distributed_worker.py`, `gpu_optimizer.py`,
   `persistent_worker_coordinator.py`, `sieve_gpu_worker.py`, `window_optimizer_integration_final.py`,
   `miner/*.py`, `utils/canonical_arrays.py`, `utils/canonical_records.py`, `execution_set.py`,
   `distributed_config.json`, `ml_coordinator_config.json`, `rig_profiles_config.json`, the
   `agent_manifests/*.json` (including the gitignored `definitions.json`), and the run scripts.
   Governance was searched first (`docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`),
   then chapters, then code.
3. Read-only host checks on VM 101: `hostname`, `nvidia-smi -L`, `systemctl list-units`,
   `systemctl status`/`cat` of one unit, `crontab -l`. **No rig was contacted** (no SSH); rig facts
   come from committed evidence documents only.
4. Corrected the chapter **in place**: every stale statement keeps its text and gains a
   *Corrected 2026-11-11* note or a **SUPERSEDED (2026-11-11)** marker; nothing was deleted. New
   material is §0 and §13–§18. The audit (`docs/CHAPTER_9_AUDIT.md`) carries the claim-by-claim result,
   the labels and the list of files opened.
5. Nothing was executed except read-only commands and one `python3` import of
   `utils/canonical_arrays.py` to count the contract arrays. No code, config or manifest was changed.

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

**Added 2026-11-11:**

0. [How this chapter was produced](#0-how-this-chapter-was-produced-2026-11-11)
13. [RANGE-MINER — the certifying Step-1 cluster engine](#13-range-miner--the-certifying-step-1-cluster-engine)
14. [Topology, boot-selector and the Resolved Execution Set](#14-topology-boot-selector-and-the-resolved-execution-set)
15. [Known gaps at a15b524](#15-known-gaps-at-a15b524)
16. [Open items](#16-open-items)
17. [Verification declaration (VIR-1…6)](#17-verification-declaration-vir-16)
18. [Closure statement](#18-closure-statement)

The pre-existing appended sections (§8.4, §8.5, §5.8, Persistent Worker Engine, GPU Throughput,
Seed Caps, rig-6600c note, and two copies of the PWC invariants section) sit between §12 and §13
in file order; they are retained, each with a correction note.

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

> **Corrected 2026-11-11.** "Two coordinators — Seed-based (Steps 2, 2.5) and script-based
> (Steps 3-6)" is wrong in two ways at `a15b524`.
> 1. **Step-1 sieve execution has four selectable backends**, chosen by a cascade in
>    `window_optimizer_integration_final.py` (`_use_miner` at `:1402`, `_use_pw` at `:1576`,
>    `_use_zmq` at `:1621`, falling through to the legacy `coordinator.py` path):
>    **RANGE-MINER** (`miner/`, §13 — the only certifying engine), **PWC**
>    (`persistent_worker_coordinator.py`), **ZMQ-SQLite** (`zmq_sqlite_coordinator.py`), and the
>    legacy **`coordinator.py`** chunk dispatch. PWC and ZMQ are non-certifying diagnostic paths
>    (§13.1).
> 2. **Steps** in this table are conceptual numbers. In the executable scheme: `coordinator.py`
>    serves Step 1 (legacy path) and the Step-2 result *pull* (`collect_scorer_results`,
>    `coordinator.py:2564`); `scripts_coordinator.py` serves **Step 2** (`run_scorer_meta_optimizer.sh:266`)
>    and **Step 3** (`run_step3_full_scoring.sh:226`). WATCHER's Step 4 is local
>    (`agent_manifests/ml_meta.json:30` `"distributed": false`); WATCHER's Step 5 runs
>    `meta_prediction_optimizer_anti_overfit.py`, whose trials use the **local**
>    `subprocess_trial_coordinator` (`meta_prediction_optimizer_anti_overfit.py:336`); Step 6 is
>    local. A distributed Step-5 route through `scripts_coordinator.py` exists
>    (`generate_anti_overfit_jobs.py` → `anti_overfit_trial_worker.py`) but is not WATCHER's.
> 3. **"26 heterogeneous GPUs"** — the configured fleet is **25**: `distributed_config.json:6`
>    `localhost.gpu_count: 1` (corrected 2→1 at `f255912`) + 3 × 8 rig GPUs. Live
>    `nvidia-smi -L` on VM 101 this session lists one `RTX 3080 Ti`. See §14.

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

> **Corrected 2026-11-11 — §2.1 and §2.2 are SUPERSEDED as a description of the current fleet;
> the diagram and table are retained as the historical (pre-VM-101) layout.** At `a15b524`:
>
> | Node (logical) | Live hostname | Endpoint (bare-metal / Proxmox) | GPUs | Backend |
> |---|---|---|---|---|
> | VM 101 (`zeus-ubuntu`), coordinator + 1 worker | `zeus-ubuntu-vm` | `localhost` | **1 × RTX 3080 Ti** | CUDA |
> | rrig6600 | `rrig6600` | `192.168.3.120` / `192.168.3.122` | 8 × **RX 6600 XT** (gfx1032) | ROCm |
> | rrig6600b | `rrig6600b` | `192.168.3.154` / `192.168.3.156` | 8 × RX 6600 XT | ROCm |
> | rrig6600c | `rrig6600c` | `192.168.3.162` / `192.168.3.164` | 8 × RX 6600 XT | ROCm |
>
> Anchors: hostnames and both endpoint sets `rig_profiles_config.json` `nodes[*].worker_hostname`
> / `endpoints`; GPU counts `distributed_config.json:6,15,26,37`; local hostname and single GPU
> from `hostname` and `nvidia-smi -L` on VM 101 this session; SKU and arch
> `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:46,119-120,135-136` (device name
> `AMD Radeon RX 6600 XT`, `gfx1032`). The second 3080 Ti belongs to Windows VM 100
> (`CLAUDE.md` §3; `f255912`). **Total: 25 GPUs.** The rig ROCm install resolves to
> `/opt/rocm-6.4.3` (`docs/CLAUDE_CODE_REPORT_PRERUN_PROBE_AND_SAMPLER.md:58`), so "ROCm 6.4.3"
> above is consistent with committed evidence.
>
> - `distributed_config.json` still says `"gpu_type": "RX 6600"`. **Recorded, not changed.** The
>   string is functional for the legacy path only by substring match (`coordinator.py:560-568`
>   `is_mining_node` / `max_seeds_for_worker` test `'RX 6600' in gpu_type`; `gpu_optimizer.py:84-87`
>   matches profile names by substring), so the true SKU name would still match.
> - The **TFLOPS and VRAM totals** are nominal vendor figures not reproduced from any committed
>   measurement: **[UNVERIFIED]**. By nominal spec the current fleet is 1 × 12 GB + 24 × 8 GB =
>   204 GB, not 216 GB.
> - The rigs are **not interchangeable** on throughput; see the rig-6600c note correction below.
> - "SSH + SFTP" is accurate for the legacy coordinator and `scripts_coordinator.py`. On the
>   certifying path, RANGE-MINER workers dial the coordinator over TCP (default port 5700,
>   `miner/range_miner_protocol.py:41`); see §13.3.

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

> **Corrected 2026-11-11.**
> - "Step 2 (Sieve) — `coordinator.py --seed-start 0 --seed-end 1000000000`": `coordinator.py`
>   has **no `--seed-end`** (CLI `coordinator.py:2687-2773`; seed count is `-s/--seeds`, and a
>   positional `target_file` or `--jobs-file` is required, `:2783-2784`). In production the sieve is
>   not launched by calling `coordinator.py` directly: WATCHER runs **Step 1**
>   `window_optimizer.py` (`agents/watcher_agent.py:389`), which selects a backend (§1.2 note).
> - "Step 2.5 (Scorer Meta)" = executable **Step 2**; the command shown matches
>   `run_scorer_meta_optimizer.sh:266-269`.
> - "Step 5 (Training) — `scripts_coordinator.py --jobs-file anti_overfit_jobs.json`" is a manual
>   route; WATCHER's Step 5 is local (§1.2 note). The `--output-dir anti_overfit_results
>   --preserve-paths` form is the one the coordinator's own docstring gives
>   (`scripts_coordinator.py:40-42`).

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

> **Verified 2026-11-11** with precisions. SSH pool: `self.ssh_pool = SSHConnectionPool(...)`
> (`coordinator.py:259`; class `connection_manager.py:12`). Pull: `collect_scorer_results` /
> `_pull_remote_scorer_results` (`coordinator.py:2564`, `:2620`) — SFTP `get` then `remove`
> (`:2660-2664`). Retry: `AutoRecoveryManager` (`recovery_manager.py:19` `max_retries = 3`,
> `:23-26` delay `min(300, 5·2^(n−1))` s), used at `coordinator.py:1724-1756` and `:2148-2197`.
> Progress persistence: `save_progress`/`load_progress` under `.recovery/` keyed by an analysis id
> (`recovery_manager.py:16-18`, `:41`; `coordinator.py:2059-2111`); resume is governed by
> `--resume-policy prompt|resume|restart|auto` (`coordinator.py:2719-2730`, `_decide_resume_or_restart`
> `:1148`). The job queue is not a priority queue: `DynamicJobQueue(self.gpu_optimizer)`
> (`coordinator.py:1208`) in the dynamic path; the static path builds a job list
> (`:575-640`). This whole section describes the **legacy, non-certifying** Step-1 path (§13.1).

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

> **Corrected 2026-11-11.** The scaling factor is **2.80**, not 6.0
> (`gpu_optimizer.py:18`, comment "was 6.0, ungrounded"). There is no fixed `base_chunk_size =
> 10000` in this path: `base_allocation = int(total_seeds * capacity_ratio)` where the ratio is the
> worker's profiled seeds/s over the fleet total (`coordinator.py:586-607`), then
> `calculate_optimal_chunk_size` (`:609`), then capped by `max_seeds_for_worker` — 5,000,000 for
> NVIDIA, 2,000,000 for AMD by default (`:563-568`, `:610-611`).

### 4.5 Pull Architecture (Step 2.5)

> *Corrected 2026-11-11: "Step 2.5" = executable Step 2. The five steps below match
> `coordinator.py:2620-2674` (results path `~/distributed_prng_analysis/scorer_trial_results/trial_*.json`,
> `:2637-2638`). The dispatch half is done by `scripts_coordinator.py`; `run_scorer_meta_optimizer.sh:284`
> then constructs `MultiGPUCoordinator('ml_coordinator_config.json')` to pull. Note that
> `ml_coordinator_config.json` still declares `localhost` with `gpu_count: 2` (read this session);
> that file is a live, tracked fact, recorded in `docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1 / §16 D17,
> and is not changed here.*

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

> *Corrected 2026-11-11: in executable numbering it serves **Step 2 and Step 3** in WATCHER
> runs (`run_scorer_meta_optimizer.sh:266`, `run_step3_full_scoring.sh:226`), plus a manual Step-5
> route. Its own docstring says "Steps 3, 4, and 5" (`scripts_coordinator.py:5`) in the conceptual
> scheme. It reads node definitions straight from `distributed_config.json` (`load_config`,
> `:430-448`) and is **not** a consumer of the Resolved Execution Set (§14) — see §16 O-6.*

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

> **SUPERSEDED (2026-11-11).** No `PER_NODE_CONCURRENCY` dict exists in `scripts_coordinator.py`
> at `a15b524` (searched the file; `git grep PER_NODE_CONCURRENCY` over tracked `*.py` hits only
> the doc-patch script `update_docs_rig6600c.py:222-226`, which edits this chapter). Per-node
> concurrency is **read from config**: `max_concurrent=node_data.get('max_concurrent_script_jobs',
> node_data['gpu_count'])` (`scripts_coordinator.py:444`), and `coordinator.py` v1.7.4.1 reads the
> same key (`coordinator.py:373-376`). Configured values: `localhost` 2, each rig **8**
> (`distributed_config.json:10`, `:19`, `:30`, `:41`). On top of that, batching is step-aware:
> `MAX_JOBS_PER_BATCH = 20` (`:91`), at most 6 jobs per node per batch by default and 8 for Step 3
> (`:105`, `:109`, `get_step_aware_limits` `:165`), with a 5 s inter-batch cooldown. The "weak CPU
> → 1" premise was retired by §8.4 below (2026-01-18) and by
> `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md:108`.

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

> *Corrected 2026-11-11: the jobs file is a **top-level JSON list** (`load_jobs`
> `scripts_coordinator.py:450-457` iterates `json.load(f)` directly), each entry following SCRIPT
> JOB SPEC v1.0 — `job_id`, `script`, `args`, **`expected_output`** (required, `:457`),
> `timeout` (default 7200) (`:27-34`, dataclass `Job` `:230-239`). There is no per-job `gpu_id`;
> the coordinator assigns GPUs round-robin per node (`_node_executor` `:826-834`).*

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

> *Corrected 2026-11-11: node selection is **round-robin, not least-loaded**
> (`assign_jobs_to_nodes` `scripts_coordinator.py:534-558`). Before execution it clears the HIP
> cache and warms ROCm once per node per run (`:640-641`, `clear_hip_cache_on_nodes` `:241`,
> `warmup_rocm_on_nodes` `:296`). Remote jobs run as `ssh user@host '<env …> python -u
> distributed_worker.py job_<id>.json --gpu-id N'` (`_build_command` `:964-999`, `_execute_job`
> `:880-905`); success is decided by the expected output file, with failure modes MISSING / EMPTY /
> INVALID_JSON / TIMEOUT / SSH_ERROR / EXECUTION_ERROR (`_check_output` `:1001`, `class FailureMode` `:191-198`).
> Failed jobs are retried **once, on localhost** (`_retry_on_localhost` `:1068-1110`, gated by
> `max_retries > 0` at `:740`) — `--max-retries` is not a per-job attempt count.*

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

> **Corrected 2026-11-11 — the guard is present but never fires on the live fleet (BROKEN,
> known class).** The block is real (`distributed_worker.py:13-29`; it also hardens
> `LD_LIBRARY_PATH`, and sets `HIP_PATH=/opt/rocm/hip`, not `/opt/rocm`). CuPy is imported lazily
> (`_ensure_cupy` `:44`), not at module top. But the live worker hostnames are `rrig6600`,
> `rrig6600b`, `rrig6600c` (`rig_profiles_config.json` `worker_hostname`), so
> `HOST in ["rig-6600", "rig-6600b", "rig-6600c"]` (`:17`) is false on every rig. The same
> hostname-tuple defect is recorded for `sieve_filter.py` in `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`
> §9.5 (F-5), and a second instance sits in the script path at `distributed_worker.py:354`.
> **Harmless today**: ROCm 6.4 supports gfx1032 natively and no override is required or set
> (`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_0_ROCM_PARITY.md:33`). Chapter 2 §9.5's disposition
> applies: do **not** rename hosts to match the tuple; first decide whether the prelude is still
> supported; if retained, key it from the rig profile, never another hostname tuple. The
> statement "MUST be first" therefore describes a mitigation that is currently inert.

### 6.3 GPU Device Binding

```python
# Coordinator sets environment variable
export CUDA_VISIBLE_DEVICES=3  # (or HIP_VISIBLE_DEVICES for ROCm)

# Worker sees only that GPU as device 0
cp.cuda.Device(0).use()
```

> *Verified 2026-11-11: `_logical_device_id` returns 0 when `CUDA_VISIBLE_DEVICES` or
> `HIP_VISIBLE_DEVICES` is set (`distributed_worker.py:74-83`). `scripts_coordinator.py` sets both
> on ROCm nodes and only `CUDA_VISIBLE_DEVICES` on CUDA nodes (`:967-973`). RANGE-MINER uses
> `ROCR_VISIBLE_DEVICES=N` with `--device-index 0` instead (`scripts/launch_fleet_manual.sh:35-39`;
> worker CLI `miner/range_miner_worker.py:2802-2806`).*

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

> **Corrected 2026-11-11.** There is no `route_job` function and the key is **`analysis_type`**,
> not `type` (`execute_analysis_job` `distributed_worker.py:233`, key read `:249`). Branches:
> `sieve` → `sieve_filter.py` subprocess with a 300 s timeout (`:257-277`); `reverse_sieve` →
> **raises `LEGACY_FUSED_ENGINE_CLOSED`** (`:288-297`) — **SWITCHED-OFF-BY-RULING**, Window-Anchor
> Brief I §2.4 (`docs/S172_WINDOW_ANCHOR_BRIEF_I.md`; "the legacy reverse engine is
> execution-closed", `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:70`); `script` →
> threaded subprocess (`:319-411`); `correlation` → `analyze_correlation_gpu` (`:414-424`); default →
> `run_statistical_analysis` (`:425-430`). `main()` additionally routes `job_type ==
> 'advanced_draw_matching'` (`:566-569`).

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

> *Corrected 2026-11-11: the skip-GPU-init behaviour is accurate (`if analysis_type != 'script':
> enforce_gpu_backend(...)`, `distributed_worker.py:249-251`, `:433-434`; changelog `:7-10`). The
> execution is not `subprocess.run(['python3', …])`: it is `subprocess.Popen([sys.executable, '-u',
> script] + args, …, start_new_session=True)` submitted to a 2-thread pool (`_executor`
> `:42`, `:339-411`), with `--gpu-id` appended when absent (`:344-345`), and the result parsed
> from the last non-empty stdout line (`:378-380`).*

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

> **SUPERSEDED (2026-11-11) — values are stale.** Live profiles (`gpu_optimizer.py:15-58`):
> `RTX 3080 Ti` **2,210,000** seeds/s, scaling **2.80**; `RX 6600` **787,950** seeds/s, scaling
> 1.0 (both "S128 measured × 0.85 safety margin", `:17-18`, `:35`). Further profiles exist for RTX
> 3080, RTX 3090 and RX 7900 XTX, plus CUDA/ROCm fallbacks (`:46-59`). The corrected values are
> also stated in the "GPU Throughput — Corrected Values" section below.

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

> *Verified with one precision (2026-11-11).* §7.3 matches `calculate_optimal_chunk_size`
> (`gpu_optimizer.py:90-97`; `min_job_size = 500`, `max_job_size_multiplier = 50`, `:62-63`).
> §7.4 matches `update_performance` (`:99-136`; α = 0.3 at `:64`). **Precision:** the learned rate
> is stored in per-worker `GPUPerformanceProfile` objects (`self.gpu_profiles`, `:105-111`), while
> `get_gpu_profile` — which chunk sizing reads — returns only the static tables (`:82-89`). No
> tracked `*.py` reads `gpu_profiles` outside `gpu_optimizer.py` (searched). So the learned value
> is recorded, not fed back into chunk size. The chapter never claimed feedback; DESCRIPTION-ONLY.

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

> **SUPERSEDED (2026-11-11) for the current rigs.** The cards are **RX 6600 XT** (gfx1032), and
> on the current ROCm 6.4.3 stack **no override is required or set**: "`HSA_OVERRIDE_GFX_VERSION`
> is unnecessary: ROCm 6.4 supports gfx1032 natively" (`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_6_0_ROCM_PARITY.md:33`;
> SKU `:37-40`). The variables are still *exported* by `scripts_coordinator.py:67-74` (`ROCM_ENV_VARS`)
> and by PWC's launch command (`persistent_worker_coordinator.py:140-141`, `:385-386`), so on those
> paths they are set unconditionally; the hostname-guarded copies never fire (§6.2 note).
> RANGE-MINER sets none of them. Kept as the historical reason (the GCVM_L2 fault chase) —
> Chapter 2 §9.5 asks that the explanation be preserved even if the prelude is retired.

### 8.3 PyTorch Memory Configuration

```python
# For RX 6600 (8GB VRAM)
os.environ["PYTORCH_HIP_ALLOC_CONF"] = (
    "garbage_collection_threshold:0.8,max_split_size_mb:128"
)
torch.cuda.set_per_process_memory_fraction(0.8)  # 6.4GB usable
```

> *Verified 2026-11-11, location added: these two settings live in `survivor_scorer.py:28`
> (`os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", …)`) and `:65`
> (`set_per_process_memory_fraction(0.8)`), i.e. in the Step-2/3 scoring workers, not in the
> cluster layer.*

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

> **Corrected 2026-11-11 (§9.1 and §9.2).**
> - §9.1: `reverse_sieve` is **closed** at both dispatch points — `distributed_worker.py:288-297`
>   and `coordinator.py:832-840` (`LEGACY_FUSED_ENGINE_CLOSED`; also `coordinator_sieve_dynamic.py:563`,
>   `run_complete_pipeline.py:76`, and `reverse_sieve_filter.py:123` itself). SWITCHED-OFF-BY-RULING,
>   Window-Anchor Brief I §2.4 (`docs/S172_WINDOW_ANCHOR_BRIEF_I.md:312`). The legacy forward
>   `sieve` / `residue_sieve` route to `sieve_filter.py` is still live (`distributed_worker.py:257-277`,
>   `coordinator.py:825-831`) and non-certifying.
> - §9.2, executable numbering, at `a15b524`:
>
> | Executable step | Script (`agents/watcher_agent.py:387-395`) | Cluster mechanism |
> |---|---|---|
> | Step 1 | `window_optimizer.py` | one of four backends (§1.2 note); certifying = RANGE-MINER (§13) |
> | Step 2 | `run_scorer_meta_optimizer.sh` | `scripts_coordinator.py` (`scorer_trial_worker.py`), then `coordinator.py` pull |
> | Step 3 | `run_step3_full_scoring.sh` | `scripts_coordinator.py` (`full_scoring_worker.py`) |
> | Step 4 | `adaptive_meta_optimizer.py` | local (`agent_manifests/ml_meta.json:30`) |
> | Step 5 | `meta_prediction_optimizer_anti_overfit.py` | local subprocess trials (`:336`) |
> | Step 6 | `prediction_generator.py` | local (`agent_manifests/prediction.json`, `distributed: false`) |
>
> The table above's "Step 4 → scripts_coordinator (adaptive_meta_optimizer.py)" and "Step 5 →
> scripts_coordinator (anti_overfit_trial_worker.py)" rows do not describe WATCHER's runs.

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

> *Corrected 2026-11-11: illustrative only. The real legacy mechanism is round-based re-queue of
> failed jobs, up to `max_retries = 3` per job, with a delay of `min(300, 5 · 2^(round−1))` s
> between rounds (`recovery_manager.py:19-26`, `should_retry_job` `:86`; `coordinator.py:1724-1756`,
> `:2188-2197`). `scripts_coordinator.py` retries once on localhost (§5.5 note). RANGE-MINER uses a
> different, deliberately narrower matrix — no `MAX_ATTEMPTS` (§13.6).*

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

> *Corrected 2026-11-11: there is no `progress.json`. `AutoRecoveryManager` writes a
> `ProgressState` per analysis id under `.recovery/` (`recovery_manager.py:16-18`, `save_progress`
> `:41`, load `:67`), saved every 30 s and on change (`coordinator.py:279` `progress_save_interval = 30`,
> `:2111`, `:2156`), and removed on completion (`:2245`).*

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

> *Corrected 2026-11-11: no `handle_node_failure` or `schedule_reconnect` exists in any tracked
> `*.py` (searched). The legacy path treats a node's failure as job failures fed to the retry
> rounds above. PWC has per-rig quarantine and respawn-on-dead-worker
> (`persistent_worker_coordinator.py:927-954`, per-node respawn locks `:300-353`). RANGE-MINER has
> leases, heartbeats, session recovery and a frozen cohort (§13.6).*

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

> **Corrected 2026-11-11 (§11.1–§11.3).**
> - `coordinator.py` (`:2687-2773`): positional `target_file`; `-c/--config` (default
>   `distributed_config.json`); `-s/--seeds`; `--seed-start`; `--offset`; `-o/--output`;
>   `--jobs-file`; `--resume-policy prompt|resume|restart|auto`; `--seed-cap-nvidia` (5,000,000),
>   `--seed-cap-amd` (2,000,000), `--seed-cap-default`; `--max-concurrent` (26), `--max-per-node`
>   (12), `--max-local-concurrent` (4), `--job-timeout` (1200); sieve options `--method`,
>   `--prng-type`, `--window-size`, `--skip-min/--skip-max`, `--threshold`, `--hybrid`,
>   `--phase1-threshold`, `--phase2-threshold`, `--session-filter`. **There is no `--seed-end`,
>   `--resume` or `--output-dir`.**
> - `scripts_coordinator.py` (`:1142-1155`): `--jobs-file` (required), `--config`, `--output-dir`
>   (default `full_scoring_results`), `--preserve-paths`, `--max-retries` (3), `--dry-run`,
>   `-v/--verbose`. **There is no `--resume`.**
> - `distributed_worker.py` (`:66-72`): **positional** `job_file` (not `--job-file`), `--gpu-id`,
>   `--mining-mode`.
> - RANGE-MINER worker: `miner/range_miner_worker.py` `--host` (required), `--port` (5700),
>   `--gpu-id` (required), `--device-index`, `--miner-output-dir`, `--heartbeat-interval`, the four
>   `--seed-cap-*` flags, and the launch-barrier flags `--run-nonce`, `--session-release-file`,
>   `--release-deadline` (`:2800-2830`).

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

> *Corrected 2026-11-11: line counts at `a15b524` are `coordinator.py` **2846**,
> `scripts_coordinator.py` **1194**, `distributed_worker.py` **607**, `gpu_optimizer.py` **136**
> (`wc -l`). The summary omits the Step-1 engines that now carry the sieve —
> `persistent_worker_coordinator.py` (1957), `sieve_gpu_worker.py` (510) and the `miner/` package
> (`range_miner_coordinator.py` 12555, `range_miner_worker.py` 2947, `range_miner_npz_writer.py`
> 1302, `range_miner_protocol.py` 271, `assembly_shard_worker.py` 887, `assembly_backends.py` 393,
> `dataset_authority.py` 1350, `step1_ingress.py` 288). "ROCm environment MUST be set before GPU
> imports" — see §6.2 note: no override is needed today and the hostname guard is inert.*

---

*End of Chapter 9: GPU Cluster Infrastructure*

> *Note 2026-11-11: this end marker is historical. Sections appended after it (below) and the
> 2026-11-11 sections §13–§18 are part of the chapter.*

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

   > *Corrected 2026-11-11: the committed value is **8** on each rig
   > (`distributed_config.json:19`, `:30`, `:41`), consistent with the "Validated Configuration"
   > block above (`max_concurrent_script_jobs: 8`). The `12` example and "Full 12-GPU
   > concurrency" do not match a rig of 8 GPUs. The benchmark table and the GFXOFF kernel
   > parameter are host/rig state this pass could not read (no rig access): **[UNVERIFIED]**.
   > The "Root cause = host memory pressure" framing applies to the script-job path
   > (Steps 2/3); the Step-1 launch-storm / `GCVM_L2_PROTECTION_FAULT` history that led to
   > RANGE-MINER is a separate matter (§13.1).*

   > **Boot-selector note:** each rig is dual-boot. The `.120`/`.154`/`.162`
   > addresses in this chapter are the **default** bare-Ubuntu endpoints, which is
   > why `distributed_config.json` holds them. When a rig is booted into Proxmox
   > instead, its worker endpoint is the CT100 address (`.122`/`.156`/`.164`). As of
   > 2026-07-22 all three rigs are running under Proxmox. See `CLAUDE.md` §3.
   >
   > *Added 2026-11-11: that sentence is a point-in-time record. The boot-selector model is
   > now encoded in code: `rig_profiles_config.json` joins both address sets per node and
   > `execution_set.py` resolves one profile per run (§14). Its declared `default_profile` is
   > `proxmox`. `distributed_config.json` holding bare-metal addresses is deliberate, **not a
   > bug**. Current boot state of the rigs was not observed this pass (no rig access);
   > `docs/TODO_CONSOLIDATED.md:7` records the fleet as powered off at the last session.*

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

> *Corrected 2026-11-11: the path is correct — Step-3 jobs reference
> `/dev/shm/prng/step3/<file>` (`generate_step3_scoring_jobs.py:236-237`) and nodes declare
> `use_ramdisk` / `ramdisk_path: /dev/shm/prng` (`distributed_config.json:20-21`). But the manual
> loop below is **no longer the only route**: `run_step3_full_scoring.sh:30-40` sources
> `ramdisk_preload.sh` and calls `preload_ramdisk` with `RAMDISK_STEP_ID=3`, which reads node
> hostnames from `distributed_config.json` (`ramdisk_preload.sh:30-33`). The manual loop and
> verification script are retained as an operator fallback. The verification script omits
> `192.168.3.162`. Both target the bare-metal addresses, which is correct for the default boot
> target; under the Proxmox profile they would need the CT100 addresses (§14).*

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

> *Verified 2026-11-11: `run_step3_full_scoring.sh:70` `CHUNK_SIZE=1000` and
> `agent_manifests/full_scoring.json:122` `"chunk_size": 1000` agree at `a15b524`. The `sed -i`
> repair commands below are recorded as written; this pass ran none of them.*

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

## Persistent Worker Engine (S130/S134/S135)

**Module:** `persistent_worker_coordinator.py` -- 855-line STANDALONE module.
**Key invariant:** Zero changes to `coordinator.py`. Drop-in parallel path activated by
`--use-persistent-workers` flag on `window_optimizer.py`.

WATCHER passes flag via manifest -> `window_optimizer_integration_final.py`
-> `run_trial_persistent()` shim -> `PersistentWorkerCoordinator` class.

> **Corrected 2026-11-11 — status change first.** PWC is **not** the production engine. Team Beta
> retired PWC/ZMQ from certifying authority on 2026-07-31 (recorded in
> `docs/TFM_PROJECT_FACTS_SKILL.md` §0.7 `:186-187`, and in Chapter 2 §10); it remains a
> flag-selectable, **non-certifying diagnostic** path, and its hybrid (variable-skip) passes are
> **quarantined**: `assert_pwc_hybrid_not_quarantined` raises
> `PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED` for any `_hybrid` prng_type
> (`persistent_worker_coordinator.py:188-217`; commit `8a55a68`, "TB Ruling 24 items 1-2,
> comparator ruling 5"). SWITCHED-OFF-BY-RULING — see §15.
>
> Other corrections: the module is **1957** lines, not 855; the shim is at
> `persistent_worker_coordinator.py:1632`, not `:669`; the integration gate is `_use_pw`
> (`window_optimizer_integration_final.py:1576-1582`), placed **after** the RANGE-MINER gate
> (`:1402`). The "zero changes to coordinator.py" claim comes from the module's own docstring
> (`:7-8`) and was not re-proved. **WATCHER's manifest still defaults to this path**:
> `agent_manifests/window_optimizer.json` `default_params.use_persistent_workers: true`,
> `use_range_miner: false` (`:261`, `:269`) — so, on reading, a WATCHER Step-1 run with default
> parameters and `test_both_modes: true` (`:254`) reaches the hybrid quarantine at
> `run_sieve_pass` (`persistent_worker_coordinator.py:1230`) and fails closed. Not run. Recorded,
> not changed (brief §5).

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

> *Verified 2026-11-11 for the stdin/stdout (SSH) mode: `sieve_gpu_worker.py:13-17` (contract),
> `:457` (ready), `:486-487` (ok, with `elapsed_s`), `:489-490` (error with traceback), `:473-475`
> (shutdown; the worker answers `{"status":"shutdown","jobs_processed":N}`). Since S161 PWC also
> has a **TCP transport** (`pwc_transport` `ssh|tcp`, `persistent_worker_coordinator.py:265`,
> `:513-529`; `persistent/pwc_transport_tcp.py`), and the WATCHER manifest selects
> `"pwc_transport": "tcp"` (`agent_manifests/window_optimizer.json`). This is the PWC wire format;
> RANGE-MINER uses a different, length-prefixed protocol (§13.3).*

### Worker Pool

- 24 AMD workers (8/rig x 3 rigs); Zeus uses local path (no persistent worker needed)
- Spawn stagger: 4.0s per gpu_id (ROCm stability, avoids HIP init storm)
- Pool size cap: worker_pool_size (manifest default 8; stable at 4 vs memory pressure)
- SSH banner drain: -q flag + drain loop reads until line contains "status" and "ready" (Bug 8, S135)
- Dispatch lock: threading.Lock per WorkerHandle -- prevents concurrent chunk collision (Bug 9, S135)

> *Corrected 2026-11-11: stagger `ROCM_SPAWN_STAGGER_S = 4.0` (`persistent_worker_coordinator.py:116`;
> TCP mode uses `TCP_SPAWN_STAGGER_S = 1.0`, `:117`); `ssh -q` and the ready-drain loop
> (`:617`, `:635-648`) — verified. "Zeus uses local path (no persistent worker)" is out of date:
> since S165 PWC also launches persistent CUDA workers on the local box in TCP mode (`:797-807`).
> The local slot pool is **hardcoded to two GPUs** (`for _gpu_id in range(2):  # Zeus has 2 GPUs`,
> `:295-297`) while `distributed_config.json:6` now declares `gpu_count: 1` — **BROKEN**, §15 B-2.
> Pool cap: `worker_pool_size` manifest default 8 (`agent_manifests/window_optimizer.json:262`).*

### ROCm Stability Env Vars

HSA_OVERRIDE_GFX_VERSION=10.3.0  (correct GFX version for RX 6600)
HSA_ENABLE_SDMA=0                (reduce HIP init crashes)

### Fault Tolerance (Gate 1, validated S131)

Three fallback paths -> execute_remote_job: spawn failure, broken pipe, worker death mid-job.
Empty-readline dead-pipe: proc.poll() check fires within 20ms.

> *Corrected 2026-11-11: no symbol `execute_remote_job` exists in
> `persistent_worker_coordinator.py` at `a15b524` (searched). Dead-worker detection by
> `proc.poll()` before dispatch and respawn under a per-node lock is present (`:927-954`). The
> S131 "Gate 1" validation and the 20 ms figure: **[UNVERIFIED]** (historical, non-certifying
> path).*

### Throughput

| Mode | Aggregate sps |
|------|--------------|
| Subprocess baseline (S128) | 849,469 |
| Persistent workers (S130) | 2,082,140 (+150%) |

> *Note 2026-11-11: historical operating records of a path that no longer carries certifying
> authority; not re-measured, **[UNVERIFIED]**, and not to be cited as comparators (Chapter 2 §10).*

---

## GPU Throughput -- Corrected Values (S129B-A)

Early-probing values caused chunk size miscalculation (chunks sized at 19k-40k seeds):

| GPU | Before (stale) | After (corrected) |
|-----|----------------|-------------------|
| RTX 3080 Ti sps | 29,000 | 2,210,000 |
| RX 6600 sps | 5,000 | 787,950 |

> *Verified 2026-11-11: `gpu_optimizer.py:17` (2,210,000) and `:35` (787,950). These profile
> values drive only the legacy `coordinator.py` path; RANGE-MINER sizes work by seed caps, not
> throughput profiles (§13.5).*

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

> *Verified 2026-11-11 with updates: CLI `coordinator.py:2703-2706`; constructor `:250`; the
> integration passes caps explicitly to every backend (`window_optimizer_integration_final.py:1456-1462`
> miner, `:1594-1595` PWC, `:1633-1634`/`:1652-1653` ZMQ); manifest is now **v1.8.0**
> (`agent_manifests/window_optimizer.json:5`) and declares the two caps at `:186-203`, `:263-264`.
> **RANGE-MINER adds two tighter hybrid caps** — `amd_hybrid` 1,000,000 and `nvidia_hybrid`
> 2,500,000 (`miner/range_miner_worker.py` `VramCaps`, `miner/range_miner_coordinator.py:599-602`) —
> which are **not** in the manifest (§13.5).*

---

## rig-6600c Throughput Note

rig-6600c (192.168.3.162) has an i5-8400T CPU -- ~50% throughput deficit vs other rigs.
Hardware limitation. Consider per-node seed budget or partition exclusion in future.

> **SUPERSEDED (2026-11-11).** `docs/ROOT_CAUSE_ANALYSIS_RRIG6600C_S151.md` (2026-03-21, status
> RESOLVED) found the rig's crashes were caused by `MaxSessions 5` in its `sshd_config`
> (`:11-15`) and lists "Weak CPU (i5-8400T) — CPU was fine — SSH was the bottleneck" among the
> wrong theories (`:108`). The "~50% throughput deficit" figure is not reproduced by any committed
> measurement read this pass: **[UNVERIFIED]**. On the certifying path the rigs ran as CT100s
> (`rrig6600c` at `.164` under Proxmox); Gate 12 and the Brief-I production-shape run used all 24
> rig GPUs (`docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md` §1, "full 25-GPU fleet").
> Rig-to-rig throughput differences should be re-measured, not assumed.

---

## Persistent Worker Coordinator (PWC) — Architecture Invariants (S146)

> **Corrected 2026-11-11 (applies to this whole section).**
> - "PWC is the production Step 1 sieve execution backend" — **SUPERSEDED**: PWC is a
>   non-certifying diagnostic path and its hybrid passes are quarantined (see the correction under
>   "Persistent Worker Engine" above; §15 S-2, S-3). The certifying engine is RANGE-MINER (§13).
> - `worker_pool_size` **4** "validated" vs the manifest default **8**
>   (`agent_manifests/window_optimizer.json:262`) and the class default 8
>   (`persistent_worker_coordinator.py:261`). The table records an S146 operating point, not the
>   configured value.
> - `JOB_TIMEOUT_S = 600` — verified (`persistent_worker_coordinator.py:154`).
> - `_localhost_semaphore = threading.Semaphore(2)` — **no such attribute in PWC**. PWC uses a
>   two-slot queue `_localhost_gpu_slots` (`:295-297`, hardcoded `range(2)`, §15 B-2) and per-node
>   semaphores of `max_per_node` (`:299`, `:351`). The `coordinator.py` attribute it claims to mirror
>   is `threading.Semaphore(self.max_local_concurrent)` at `coordinator.py:286` (not `:269`), whose
>   CLI default is 4 (`:2714`).
> - Hybrid kernel tails — **verified** in `sieve_gpu_worker.py:231-280` (forward `…threshold, a, c`
>   at `:257-268`; reverse `…threshold, int offset` at `:269-278`). The same ABI split is reproduced
>   by RANGE-MINER's builders (`miner/range_miner_worker.py:19-60` docstring, `_reverse_hybrid_tail`
>   `:307`). Under Window-Anchor Brief I that trailing `int offset` carries **`generator_phase`**,
>   not a window start (§13.7).
> - Threshold routing — **verified**: `hybrid_threshold = coerce_threshold(phase2_raw, threshold)`
>   (`sieve_gpu_worker.py:254-255`) and post-filter `if rate >= hybrid_threshold` (`:285`). On the
>   PWC path this value is the `run_sieve_pass` default of 0.5 rather than the trial's directional
>   threshold, which is why the path is quarantined (`persistent_worker_coordinator.py:205-217`).
> - Strategy dict — **verified**: full `to_dict()` (`sieve_gpu_worker.py:240-245`).
> - ProgressWriter calls — **verified**: `log_gpu_result` (`persistent_worker_coordinator.py:1397`),
>   `update_trial_stats` (`:1697`, `:1860`).
> - Dashboard `45.32.131.224:5002` — that address occurs in no live code, only in the doc-patch
>   script `apply_s146_doc_updates.py:188`, `:380`. The dashboard binds `0.0.0.0:5002`
>   (`web_dashboard.py:2259`) and its banner prints `192.168.3.127:5002` (`:2247`). The public
>   address is **[UNVERIFIED]**.
> - S146 validation result — historical; the two hybrid rows could not run on PWC today (quarantine).

**Files:** `persistent_worker_coordinator.py`, `sieve_gpu_worker.py`

### Overview

The Persistent Worker Coordinator (PWC) is the production Step 1 sieve execution backend
when `--use-persistent-workers` is active. Workers are kept alive between trials, accepting
job payloads over SSH. Zeus dispatches local jobs; remote rigs receive jobs via `sieve_gpu_worker.py --persistent`.

### S146 Validated Architecture

| Parameter | Value | Notes |
|-----------|-------|-------|
| `worker_pool_size` | **4** | Validated stable. Do NOT increase to 8. |
| `JOB_TIMEOUT_S` | **600** | (not 300) — allow time for large seed ranges |
| `_localhost_semaphore` | `threading.Semaphore(2)` | Zeus has 2 GPUs; mirrors coordinator.py line 269 |

### Hybrid Kernel Signature Invariants (CRITICAL)

GPU kernel arg tails differ between forward and reverse hybrid kernels.
These **must not be conflated** — they have different argument tails:

| Kernel direction | Tail args | Notes |
|-----------------|-----------|-------|
| Forward hybrid  | `..., threshold, unsigned long long a, unsigned long long c` | a,c passed explicitly |
| Reverse hybrid  | `..., threshold, int offset` | a,c hardcoded inside kernel |

**Rule:** `sieve_gpu_worker.py` must split the hybrid `elif` into separate forward and
reverse branches and pass the correct tail for each. Sending `a, c` to a reverse hybrid
kernel → crash.

### Threshold Routing (Hybrid)

- Hybrid kernels and hybrid post-filter **must use `phase2_threshold`**, not `min_match_threshold`.
- `hybrid_threshold = coerce_threshold(phase2_raw, threshold) if phase2_raw is not None else threshold`
- Post-filter: `if rate >= hybrid_threshold:` (NOT `if rate >= threshold:`)

### Strategy Dict Requirement

`sieve_filter.py` calls `StrategyConfig(**s)` which requires all 6 fields:

```
name, max_consecutive_misses, skip_tolerance,
enable_reseed_search, skip_learning_rate, breakpoint_threshold
```

Both `persistent_worker_coordinator.py` and `sieve_gpu_worker.py` must send the full
`StrategyConfig.to_dict()` result — never a partial dict.

### Dashboard Integration (ProgressWriter)

PWC must call these on every chunk/trial to drive live dashboard throughput:

```python
# After every successful chunk:
self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds_in_chunk, elapsed)

# After every completed trial:
self._progress_writer.update_trial_stats(trial_id, survivors_found, ...)
```

Dashboard: `45.32.131.224:5002`

### S146 Validation Result

3 trials, 10M seeds, all 4 sieve passes — zero errors:

*(S146 historical record, first copy — see the correction at the top of this section.)*

| Pass | Status |
|------|--------|
| java_lcg forward | ✅ |
| java_lcg_reverse | ✅ |
| java_lcg_hybrid forward | ✅ |
| java_lcg_hybrid_reverse | ✅ |

---

## Persistent Worker Coordinator (PWC) — Architecture Invariants (S146)

> **SUPERSEDED (2026-11-11) — verbatim duplicate.** This section repeats the previous section
> word for word (the same duplication defect Chapter 2 §10 recorded for its §15). It is retained
> unedited so no text is lost; the corrections written under the first copy apply to it.

**Files:** `persistent_worker_coordinator.py`, `sieve_gpu_worker.py`

### Overview

The Persistent Worker Coordinator (PWC) is the production Step 1 sieve execution backend
when `--use-persistent-workers` is active. Workers are kept alive between trials, accepting
job payloads over SSH. Zeus dispatches local jobs; remote rigs receive jobs via `sieve_gpu_worker.py --persistent`.

### S146 Validated Architecture

| Parameter | Value | Notes |
|-----------|-------|-------|
| `worker_pool_size` | **4** | Validated stable. Do NOT increase to 8. |
| `JOB_TIMEOUT_S` | **600** | (not 300) — allow time for large seed ranges |
| `_localhost_semaphore` | `threading.Semaphore(2)` | Zeus has 2 GPUs; mirrors coordinator.py line 269 |

### Hybrid Kernel Signature Invariants (CRITICAL)

GPU kernel arg tails differ between forward and reverse hybrid kernels.
These **must not be conflated** — they have different argument tails:

| Kernel direction | Tail args | Notes |
|-----------------|-----------|-------|
| Forward hybrid  | `..., threshold, unsigned long long a, unsigned long long c` | a,c passed explicitly |
| Reverse hybrid  | `..., threshold, int offset` | a,c hardcoded inside kernel |

**Rule:** `sieve_gpu_worker.py` must split the hybrid `elif` into separate forward and
reverse branches and pass the correct tail for each. Sending `a, c` to a reverse hybrid
kernel → crash.

### Threshold Routing (Hybrid)

- Hybrid kernels and hybrid post-filter **must use `phase2_threshold`**, not `min_match_threshold`.
- `hybrid_threshold = coerce_threshold(phase2_raw, threshold) if phase2_raw is not None else threshold`
- Post-filter: `if rate >= hybrid_threshold:` (NOT `if rate >= threshold:`)

### Strategy Dict Requirement

`sieve_filter.py` calls `StrategyConfig(**s)` which requires all 6 fields:

```
name, max_consecutive_misses, skip_tolerance,
enable_reseed_search, skip_learning_rate, breakpoint_threshold
```

Both `persistent_worker_coordinator.py` and `sieve_gpu_worker.py` must send the full
`StrategyConfig.to_dict()` result — never a partial dict.

### Dashboard Integration (ProgressWriter)

PWC must call these on every chunk/trial to drive live dashboard throughput:

```python
# After every successful chunk:
self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds_in_chunk, elapsed)

# After every completed trial:
self._progress_writer.update_trial_stats(trial_id, survivors_found, ...)
```

Dashboard: `45.32.131.224:5002`

### S146 Validation Result

3 trials, 10M seeds, all 4 sieve passes — zero errors:

| Pass | Status |
|------|--------|
| java_lcg forward | ✅ |
| java_lcg_reverse | ✅ |
| java_lcg_hybrid forward | ✅ |
| java_lcg_hybrid_reverse | ✅ |

---

## 13. RANGE-MINER — the certifying Step-1 cluster engine

*New section, 2026-11-11. Every statement is anchored to code at `a15b524` or to a committed
evidence document. Chapter 2 §8 describes the same engine from the sieve's side; this section
describes it from the cluster's side and does not repeat Chapter 2's threshold and residue
derivations.*

### 13.1 What it is, and why it replaced the cluster paths above

RANGE-MINER is a standalone Step-1 backend: **long-lived per-GPU worker daemons** that dial into
one coordinator, take contiguous seed stripes, run the sieve kernels, and stream results back;
the coordinator records every shard in a durable ledger and hands completed trials to a Phase-5
assembler that produces the frozen 22-array survivor bundle (§13.8). It was built because PWC
suffered silent hard resets and `GCVM_L2_PROTECTION_FAULT` on the RX 6600 XT rigs at full-fleet
saturation, traced to launch-storm behaviour (`docs/TFM_PROJECT_FACTS_SKILL.md` §0.7
`:180-187`; Chapter 2 §8.1). The requirement is an **interface** contract — "the remaining steps
must not be able to tell which engine produced their input" — not "match PWC's values"
(`docs/TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md` §1). Team Beta retired PWC and ZMQ from
certifying authority on 2026-07-31 (fact file `:186-187`).

Selection: `use_range_miner` on the coordinator object, tested first in the backend cascade
(`window_optimizer_integration_final.py:1402-1409`), mutually exclusive with PWC and ZMQ
(enforced at argparse in `window_optimizer.py`, per the comment at `:1394-1395`). The call is
`run_trial_miner(...)` (`:1440`; definition `miner/range_miner_coordinator.py:12336`), which
builds a per-trial coordinator (`build_coordinator` `:12279`) and a Phase-5 sink around the
configured assembly backend (`:1434-1439`).

### 13.2 Package structure

| Module (lines at `a15b524`) | Role (from its own docstring / symbols) |
|---|---|
| `miner/range_miner_protocol.py` (271) | 8 message types, framed JSON (§13.3) |
| `miner/range_miner_worker.py` (2947) | per-GPU daemon: REGISTER handshake, sub-stripe loop, per-family kernel-arg builders, VRAM caps, residue-window resolution, inline/spool transport (docstring `:1-60`) |
| `miner/range_miner_coordinator.py` (12555) | durable SQLite shard ledger (`MinerLedger` `:1197`), stripe state machine, macro partitioning, completion reconciliation, retry matrix, staging pipeline, `RangeMinerCoordinator` `:3519` (docstring `:1-25`) |
| `miner/range_miner_npz_writer.py` (1302) | Phase-5 assembly: spool validation, canonical replay, trial assembly |
| `miner/assembly_backends.py` (393) | frozen two-backend interface `serial_reference` / `process_sharded` (`:200`, `:273`, `get_assembly_backend` `:347`) |
| `miner/assembly_shard_worker.py` (887) | shard worker for the `process_sharded` backend |
| `miner/step1_ingress.py` (288) | Step-1 accumulator ingress (`build_assembling_sink`, `resolve_assembly_backend`, used at `window_optimizer_integration_final.py:1427-1439`) |
| `miner/dataset_authority.py` (1350) | run-start dataset freeze (P0.5); `run_frozen_dataset_sha256` used by `resolve_dataset_sha256` (`range_miner_coordinator.py:87-114`) |

### 13.3 How it changes cluster execution

| Aspect | Legacy `coordinator.py` (§4) | PWC (appended sections) | RANGE-MINER |
|---|---|---|---|
| Worker lifetime | one subprocess per job, over SSH | coordinator-spawned, persistent across the 4 passes | long-lived daemon per GPU, started outside the backend (below) |
| Wire | SSH command + stdout/SCP | JSON lines on stdin/stdout, or TCP since S161 | 4-byte big-endian length prefix + compact JSON, **64 MB frame cap** (`range_miner_protocol.py:44-46`, `encode_frame`/`decode_frame` `:234-261`), TCP default port **5700** (`:41`) |
| Message set | ad hoc | `ready/ok/error/shutdown` | **8 types**: `register`, `stripe_assign`, `sub_stripe_result`, `stripe_complete`, `stripe_error`, `heartbeat`, `shutdown`, `status` (`:80-190`); unknown `message_type` → `ValueError` (`from_dict` `:214-224`) |
| Identity | node hostname from config | `(hostname, gpu_id)` | `worker_id = "{hostname}:gpu{id}"`, reached at the **config's** SSH address, never a parsed hostname (`range_miner_coordinator.py:15-18`) |
| Work unit | chunk sized by throughput profile (§4.4) | chunk per call | coordinator: contiguous macro-stripes with no gap or overlap (`partition_macro_stripes` `:761-783`, default `miner_stripe_size` 2^26 `:603`); worker: GPU-safe sub-stripes (`partition_stripe` `range_miner_worker.py:680-695`) |
| Completion | job returned | pass returned | **proved**: sub-stripes done == expected (`expected_substripes_for` `:786`), ranges tile the stripe exactly (`_coverage_exact` `:1114`, `evaluate_stripe_completion` `:1128`) |
| Admission | whatever is in config | ready gate `min_workers` | frozen **Resolved Execution Set**; a worker not in the set registers **quarantined**, never eligible (`_execution_set_admission` `:552-558`; §14) |
| Durability | `.recovery/` progress file | in-memory | SQLite ledger keyed `(run_id, stripe_id, attempt, sub_index)` (`:19-25`) |
| Output | JSON results | JSON results → accumulator | Phase-5 22-array NPZ (§13.8) |

**Worker startup is not owned by the backend.** "RANGE-MINER has no worker-launch mechanism of
its own" (`scripts/launch_fleet_manual.sh:8`); the fleet is raised by hand with that stopgap
script, which reads endpoints from `rig_profiles_config.json` and GPU counts from
`distributed_config.json` (`:25-33`) and is invoked by nothing in production (`:17-20`). Team
Beta's position is that **final Phase-7 certification requires backend-owned startup**
(`docs/CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_PART_B.md` §7). Tracked as open in
`docs/TODO_CONSOLIDATED.md:92`.

**What did not change.** Steps 2 and 3 still distribute through `scripts_coordinator.py` (§5),
and the Step-2 result pull still uses `coordinator.py` (§4.5). RANGE-MINER is Step 1 only.

### 13.4 Family coverage

`COVERED_FAMILIES` = java_lcg, lcg32, minstd, pcg32, xorshift128, xorshift32
(`miner/range_miner_worker.py:105-107`); `UNCOVERED_FAMILIES` = mt19937, philox4x32, sfc64,
xorshift64, xoshiro256pp (`:109-111`). Each covered base advertises exactly four variants
(base, `_reverse`, `_hybrid`, `_hybrid_reverse`; `SUPPORTED_VARIANTS` `:117-122`), validated
against `KERNEL_REGISTRY` at handshake (`supported_variants` `:630-645`). An uncovered family
raises `NotImplementedError` naming the family **before any GPU work** (`resolve_builder`
`:474-485`).

### 13.5 Per-family VRAM caps (Team Beta Q2)

`VramCaps` (`miner/range_miner_worker.py:648-657`): `amd` 2,000,000; `nvidia` 5,000,000;
`amd_hybrid` **1,000,000**; `nvidia_hybrid` **2,500,000**. `select_seed_cap` (`:659-667`) gives
hybrid variants the tighter cap (extra `skip_sequences_gpu` allocation) and branches on backend
`rocm` / `cuda`, not on the `gpu_type` string. The worker advertises its caps in REGISTER; the
coordinator sizes `expected_substripes` from the **advertised** cap
(`advertised_effective_cap`, `range_miner_coordinator.py:740-758`), so both sides partition
identically. Coordinator defaults mirror the worker (`CoordinatorConfig` `:599-602`); worker CLI
flags `--seed-cap-*` (`range_miner_worker.py:2814-2817`); integration passes all four
(`window_optimizer_integration_final.py:1456-1462`). **The two hybrid caps are not in the WATCHER
manifest** — the worker's own note says Phase 4/7 must wire them (`range_miner_worker.py:650-653`,
`:2812-2813`); `agent_manifests/window_optimizer.json` declares only `seed_cap_nvidia` and
`seed_cap_amd` (`:186-203`). Today they come from defaults.

### 13.6 Retry and failure policy (Team Beta Q3)

`handle_stripe_failure` (`miner/range_miner_coordinator.py:8417-8429`, body
`_handle_stripe_failure_locked` `:8437`), implemented as a matrix with **no `MAX_ATTEMPTS`**:

| Failure | Action |
|---|---|
| explicit non-retryable | fail the trial at once; no retry consumed (`:8463-8474`) |
| retryable, workflow phase 1 or 2 (constant skip) | fail the trial at once — constant phases never retry (`:8476-8492`) |
| retryable or lease expiry, phase 3 or 4 (hybrid), first time | clean up the attempt's shards, fence it, requeue the **whole stripe** as attempt 1 for a **different** eligible worker; fail if no alternate exists (`:8494-8530`) |
| phase 3 or 4, second time | fail the trial |

So "one retry then fail the trial" (protocol `range_miner_protocol.py:25`, `:158`) holds **only for
the hybrid phases**; constant phases fail closed with no retry. Liveness is by compute lease,
renewed by heartbeat or active-stripe progress (commit `c4e0037`, F1/F2), and a worker that loses
its session may re-register into the frozen cohort (commit `acd6f13`, Defect A).

### 13.7 Where the window-anchor change touches the miner (Brief I)

Window-Anchor Brief I split the fused `offset` into two quantities (commits `48a8705`, `69ca910`):

- **`window_anchor`** — host-side: which observed records form the residue window. It lives in
  `load_residue_window` and reaches **no kernel argument** (`range_miner_worker.py:22-29`).
- **`generator_phase`** — device-side generator pre-advance, carried in the kernels' trailing
  `int32` still spelled `offset` because the ABI is frozen (`_generator_phase_tail` `:295-304`,
  `_reverse_hybrid_tail` `:307-310`). Four forward hybrids have no such argument; a nonzero phase
  there is undeliverable and raises (`GeneratorPhaseUnsupportedError` `:529`, matrix comment
  `:126-140`).
- **v1 pin: `generator_phase` must be 0**, enforced twice — at the coordinator's assign-payload
  validation (`range_miner_coordinator.py:9314-9325`) and at the worker seam
  (`GENERATOR_PHASE_V1_PIN = 0` `range_miner_worker.py:605`, `assert_generator_phase_permitted`
  `:608-627`; "TB ruling 2026-08-21", `:699`). Nonzero phase is DEP-ABI-V2, "recorded, NOT built".
- The ledger stores both (`window_anchor_val`, `generator_phase` columns, `:1388-1389`; refusal if
  absent `:1422-1431`).
- NPZ array 4 keeps its wire name `offset` and now carries `window_anchor`
  (`utils/canonical_records.py:225`; commit `69ca910`, "the 22-array contract is untouched").
- The legacy fused reverse engine is closed at every dispatch route (§9.1 note).

**Status: implemented, not accepted.** The production-shape run on 2026-08-22 completed 128/128
stripes and then **failed at Phase-5 commit** on `KeyError: 'offset'`
(`docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md` verdict and §1; commit message of
`69ca910`). The repair is committed (`69ca910`); the Run-4 routing patch is committed (`3e2537b`);
the acceptance run ("Run 4") has not happened (`docs/TODO_CONSOLIDATED.md:7`). Separately, Beta
records that `sieve_gpu_worker.py` (the PWC worker) is still an `offset` consumer left to Brief II
(`docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:70`).

### 13.8 The 22-array NPZ contract

The carrier between Step 1 and Step 2 is `CANONICAL_ARRAY_CONTRACT`
(`utils/canonical_arrays.py:98-122`) — **22 arrays**, counted by import this session: `seeds`
(uint32), `forward_matches`, `reverse_matches`, `window_size`, `offset`, `trial_number`, `skip_min`,
`skip_max`, `skip_range`, `forward_count`, `reverse_count`, `bidirectional_count`,
`intersection_count`, `intersection_ratio`, `intersection_weight`, `bidirectional_selectivity`,
`forward_only_count`, `reverse_only_count`, `survivor_overlap_ratio`, `score`, `skip_mode` (uint8),
`prng_type` (uint8). Records → arrays: `records_to_arrays` (`:480`); the fail-hard wall:
`validate_array_bundle` (`:543`). The record side is `utils/canonical_records.py`.

**Naming trap, re-verified:** `EXPECTED_NPZ_KEYS` (named in `CLAUDE.md:168`, §6 Phase 5) is **not a
symbol** in the tree; it occurs only as a forbidden-token string in
`tests/test_s172_phase4_coordinator.py:1608` (`git grep` at `a15b524`). Phase 4 is forbidden to
assemble arrays (`range_miner_coordinator.py:23-25`); assembly is Phase 5, behind the two-backend
interface, `serial_reference` default and `process_sharded` unpromoted (Chapter 2 §8.6;
`window_optimizer_integration_final.py:1416-1422`).

### 13.9 Phase status, with the evidence

| Phase (spec `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md`, frozen at `1f6c0c5`) | Status at `a15b524` | Evidence opened this session |
|---|---|---|
| 0 PRNG_TYPE_ENCODING | done | commit `2389b61` |
| 1 scaffolding | done | `8d0183f` |
| 2 protocol | done | `e0c9d1c`; `miner/range_miner_protocol.py` |
| 3 worker daemon | done | `dbe3d0e` ("Team Beta approved rev-3") |
| 4 coordinator | done | `f1e0c7c` ("7 review rounds") |
| 5 dedup + NPZ contract wall | done | D0 `4c697a8` … D6 `fc37fb5`, `2be51d5` |
| 6 byte-identity / bounded acceptance | **certified, bounded** | `d98298c` (Wall A, Wall B, Miner Known-Answer Transfer Gate); scope limit in Chapter 2 §11.2 (constant-skip generations); ROCm parity `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md` |
| Gate 12 (production-class run) | **PASSED**, attempt 9, 2026-08-17 | `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md:13-25`; tag `gate12-passed-attempt9` → `e9ca800` (`git rev-parse`); Field-6 repair `d8b21e3` |
| 7 WATCHER soak + pool regression | **not run** | `docs/TODO_CONSOLIDATED.md:7`; backend-owned launcher still required (§13.3) |

`CLAUDE.md` §6 still shows Phase 3 as next; that table is stale (brief §0).

---

## 14. Topology, boot-selector and the Resolved Execution Set

*New section, 2026-11-11. Described from committed sources and read-only local checks; no rig
was contacted.*

**Boot-selector model.** Every rig boots its **default** target, bare Ubuntu on its original
address (`.120` / `.154` / `.162`), **or** Proxmox, under which the worker runs in CT100 at
`.122` / `.156` / `.164` — never both at once (`CLAUDE.md` §3). **Both address sets are valid and
both are retained.** `distributed_config.json` holds the bare-metal addresses because they match
the default boot target; **this is deliberate and not a bug** (`CLAUDE.md` §3;
`rig_profiles_config.json` `_default_profile_note`). CT100s carry the rig's canonical hostname, so
`socket.gethostname()` is `rrig6600` / `rrig6600b` / `rrig6600c` under either target
(`rig_profiles_config.json` `_endpoint_note`). CLAUDE.md §3's "all three rigs booted into Proxmox"
is a 2026-07-22 point-in-time record; current boot state was not observed this pass.

**The selectable rig endpoint now exists.** `rig_profiles_config.json` is the join table naming
both endpoints per logical node, `profiles: ["baremetal", "proxmox"]`, `default_profile:
"proxmox"`. `execution_set.py` resolves **one frozen set per run** (`resolve_execution_set`
`:537`), never by probing (docstring `:39-55`), cross-checking against `distributed_config.json`
and the provisioning manifest (`_cross_check` `:496`). Operators select with `--rig-profile`
(`window_optimizer.py:1530-1538`); WATCHER passes `rig_profile` through
(`agents/watcher_agent.py:1623`). Consumers read the resolved set: legacy `coordinator.py`
(`:119-121`, `:310-313`), PWC (`persistent_worker_coordinator.py:105`, `:323`), RANGE-MINER
admission (`range_miner_coordinator.py:552-558`), `preflight_check.py:179-181`, `:317`, and the
Gate-12 scripts. Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_RESOLVED_EXECUTION_SET.md` §1–§2.
`CLAUDE.md` §3's sentence that a selectable `rig_profile` is "explicitly NOT done" is stale against
this code (listed for update, §16).

**Not consumers of the resolved set** (read `distributed_config.json` or literal addresses
directly): `scripts_coordinator.py` (`load_config` `:430-448`, `ROCM_HOSTNAMES` literal `:78`),
`ramdisk_preload.sh:30-33`, the code-push loop in `run_scorer_meta_optimizer.sh:257-261`, and
`ml_coordinator_config.json`. They therefore address the bare-metal endpoints only. The resolved
set's brief scoped six Step-1 mechanisms and does not name these; see §16 O-6.

**Fleet size.** 25 identities: `zeus-ubuntu-vm:gpu0` plus 8 per rig (`f255912`;
`scripts/launch_fleet_manual.sh:30-31`). `ml_coordinator_config.json` still declares 2 local GPUs
(26 total) — a live fact of that file, recorded in `docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1.

**Host state read this session (VM 101, read-only).** `hostname` → `zeus-ubuntu-vm`;
`nvidia-smi -L` → one `NVIDIA GeForce RTX 3080 Ti`; `crontab -l` → none for `michael`;
`systemctl list-units` → no PRNG / miner / WATCHER / dashboard unit, and one failed unit,
`nvidia-gpu-policy.service` ("Set NVIDIA GPU persistence & scheduling for PRNG jobs"), whose third
`ExecStart` is `nvidia-smi -i 1 -c DEFAULT` and exits 6 because there is no GPU 1 on this VM. The
unit is untracked host state (only mentioned in `docs/SESSION_CHANGELOG_20260419_S164.md`,
`…20260420_S165.md`). Not touched. `agent_manifests/definitions.json` (gitignored, 6,635 bytes,
schema 1.1.0) was read: it defines Step 5/6 sidecar and WATCHER fingerprint contracts and contains
**no cluster or node configuration**.

---

## 15. Known gaps at a15b524

### 15.1 BROKEN

- **B-1 — ROCm prelude hostname guards never fire.** `distributed_worker.py:18` and `:354` test
  *(anchor corrected 2026-11-11 post-commit `741822b`: previously `:17`, which is the comment line
  above the guard; the `if HOST in [...]` guard is `:18`)*
  `rig-6600*`; live hostnames are `rrig6600*` (`rig_profiles_config.json`). Same class as Chapter
  2 F-5 (`sieve_filter.py`), which is governed by Chapter 2 §9.5's disposition. Harmless today (no
  override needed on ROCm 6.4). A repair must first decide whether the prelude is still supported
  and, if retained, key it from the declared rig profile — not rename hosts, not another tuple.
- **B-2 — PWC's local GPU slot pool is hardcoded to two GPUs.**
  `persistent_worker_coordinator.py:295-297` (`range(2)`, "Zeus has 2 GPUs") against
  `distributed_config.json:6` `gpu_count: 1` (`f255912`). Not found in the governance trail (the
  2→1 correction records two other stale sites, commit `4643a11`). Non-certifying path. A repair
  must derive the slot count from the resolved execution set / config.

### 15.2 SWITCHED-OFF-BY-RULING

- **S-1 — Legacy fused reverse sieve dispatch closed.** `distributed_worker.py:288-297`,
  `coordinator.py:832-840` and three other sites. Ruling: Window-Anchor Brief I §2.4
  (`docs/S172_WINDOW_ANCHOR_BRIEF_I.md:312`); "execution-closed",
  `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:70`. *Lifting would re-enable
  `reverse_sieve_filter.py`, whose single `offset` drives both window slice and generator advance
  and is coherent only at skip = 0 (`reverse_sieve_filter.py:123`).*
- **S-2 — PWC hybrid passes quarantined.** `persistent_worker_coordinator.py:188-217`, call
  `:1230`. Ruling: commit `8a55a68` ("TB Ruling 24 items 1-2, comparator ruling 5"); report
  `docs/S172_THRESHOLD_PROPAGATION_REPAIR_REPORT.md`. *Lifting would let PWC run variable-skip
  passes again, filtering at the 0.5 default rather than the trial's directional threshold, with
  kernels that ignore `skip_min/skip_max` — unless Option A propagation is built first.*
- **S-3 — PWC and ZMQ have no certifying authority.** Beta, 2026-07-31 (fact file §0.7
  `:186-187`; Chapter 2 §10; no standalone `TB_RULING_*` file found for this date). *Lifting would
  let PWC/ZMQ results serve as comparators or production Step-1 output.*
- **S-4 — RANGE-MINER is not WATCHER's default.** `agent_manifests/window_optimizer.json:212-216`
  ("Do not enable in production until Phase 7 acceptance completes"), `:269`
  (`use_range_miner: false`), `:261` (`use_persistent_workers: true`). Ruling: the approved phase
  plan, `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:503` (Phase 7 = WATCHER soak + production).
  Tracked at `docs/TODO_CONSOLIDATED.md:93`. *Lifting would make every default WATCHER Step-1 run
  use the certifying engine instead of the quarantined-for-hybrid PWC path.*
- **S-5 — `generator_phase` pinned to 0.** `range_miner_worker.py:605`, `:608-627`;
  `range_miner_coordinator.py:9314-9325`. Ruling: TB 2026-08-21 (cited at
  `range_miner_worker.py:699`, `:1140`). *Lifting (DEP-ABI-V2) would make device-side generator
  phase an explorable dimension, on the variants whose ABI carries it.* (Arises from the new
  §13, not from a pre-existing chapter claim.)

### 15.3 DESCRIPTION-ONLY

Each is corrected in place above with a *Corrected 2026-11-11* note; the full numbered list
(D-1…D-33) is in `docs/CHAPTER_9_AUDIT.md` §6. In short: fleet size and SKU (25 GPUs, RX 6600 XT,
`rrig6600*` hostnames); "two coordinators" and conceptual step numbers; Step 4/5 mapping; stale
throughput profiles and 6× scaling; `PER_NODE_CONCURRENCY`, `route_job`, `progress.json`,
`handle_node_failure`, `execute_remote_job` and `_localhost_semaphore`, none of which exist under
those names; three CLI references; line counts; PWC described as production; the
duplicated PWC section; the rig-6600c note.

---

## 16. Open items

| # | Item | Where tracked / anchor |
|---|---|---|
| O-1 | Brief I acceptance: Run 4 not yet run | `docs/TODO_CONSOLIDATED.md:7`; `3e2537b` |
| O-2 | Phase 7 soak not run | `docs/TODO_CONSOLIDATED.md:7`; proposal `:503` |
| O-3 | Backend-owned RANGE-MINER worker startup (Phase-7 certification precondition) | `docs/CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_PART_B.md` §7; `docs/TODO_CONSOLIDATED.md:92` |
| O-4 | Hybrid seed caps not in the WATCHER manifest | `miner/range_miner_worker.py:650-653` |
| O-5 | WATCHER default backend decision after Phase 7 (S-4) | `docs/TODO_CONSOLIDATED.md:93` |
| O-6 | `scripts_coordinator.py`, `ramdisk_preload.sh`, `run_scorer_meta_optimizer.sh:257-261` and `ml_coordinator_config.json` do not consume the resolved execution set, so Steps 2–3 target bare-metal endpoints only. Not found in the governance trail; recorded, not labelled BROKEN, because no chapter claim or ruling says they should consume it | §14 |
| O-7 | `nvidia-gpu-policy.service` on VM 101 fails on `-i 1` (host state) | §14 |
| O-8 | B-1, B-2 repairs (separate brief) | §15.1 |
| O-9 | `distributed_config.json` `gpu_type: "RX 6600"` (true SKU RX 6600 XT); left as is, substring-functional | §2 note |
| O-10 | `sieve_gpu_worker.py` `offset` consumer left to Brief II | `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:70` |
| O-11 | Items marked **[UNVERIFIED]** above (TFLOPS/VRAM totals, §8.4 benchmark and GFXOFF on rigs, PWC throughput and S131/S146 records, dashboard public address, rig-6600c deficit, manifest-over-script precedence) | audit §9 |

---

## 17. Verification declaration (VIR-1…6)

- **VIR-1 execution proof.** Every file opened is listed in `docs/CHAPTER_9_AUDIT.md` §8. Anchors
  were read this session at `a15b524`.
- **VIR-2 clean control.** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was read, not edited.
- **VIR-3 fault-injection control.** Not applicable to a documentation pass; none was invented.
- **VIR-4 completion sentinel.** Below and in §18.
- **VIR-5 unavailable-observer behaviour.** Rig state (boot target, ROCm settings, GFXOFF,
  throughput) was not observable without rig access and is marked **UNAVAILABLE / [UNVERIFIED]**,
  never clean. The private repo and ser8 archives were not available.
- **VIR-6 audit claim scope.** What this chapter says against what the code at `a15b524` does —
  not whether the design is right. Surfaces searched: tracked tree (`git grep`, `/bin/grep` for
  `.json`), gitignored files (`agent_manifests/definitions.json`), git history (`git log`, tags,
  `git show --stat`), host state (read-only). Governance trail searched: `docs/TB_RULING_*`
  (Gate-12 attempt 9, window-anchor set), `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md`,
  `docs/TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md`, `docs/CLAUDE_CODE_INSTRUCTIONS_RESOLVED_EXECUTION_SET.md`,
  `docs/CLAUDE_CODE_INSTRUCTIONS_STAGING_DIR_PART_B.md`, the fact file, `TODO_CONSOLIDATED.md`,
  `PROJECT_FILE_CATALOG.md`. Chapters searched: 2 (§8–§11), 13 audit (layout).

**Completion sentinel (this chapter): `INCOMPLETE`** — every pre-existing claim was checked and
corrected or labelled, and the RANGE-MINER section was written, but eight claims remain
**[UNVERIFIED]** for want of rig access, and the main session's five-anchor spot-check has not yet
been recorded.

---

## 18. Closure statement

**Verified against:** `a15b5247336255801de8cf67d52653701b1e8c80` on VM 101, 2026-11-11.

**Changed this pass:** added the §1 framing block, status header and §0; added correction notes
to §1.2, §2, §3.2, §4.2, §4.4, §4.5, §5.1, §5.3, §5.4, §5.5, §6.2, §6.3, §6.4, §6.5, §7.2, §7.4,
§8.2, §8.3, §8.4, §8.5, §5.8, §9, §10.1–§10.3, §11, §12, and every appended section; marked
SUPERSEDED: the original header, §2.1/§2.2 as a current description, §5.3, §7.2, §8.2, the
rig-6600c note, PWC-as-production, and the duplicate PWC section; added §13–§18. **No text was
removed**; every pre-existing heading is still present. The chapter grew; no removal list is
needed.

**This chapter is NOT:** a certification of RANGE-MINER, a repair of any code path, or a judgement
on whether the design is right. It does not raise, and is not open to, a seed-recovery
objection.

**Open:** §16. **Sentinel: `INCOMPLETE`** (§17).

---

## Version History

| Version | Date | Change |
|---|---|---|
| 2.0.0 | (pre-2026) | Consolidated chapter, §1–§12 |
| — | 2026-01-18 … S146 | appended §8.4, §8.5, §5.8, PWC sections (one duplicated) |
| 3.0.0 | 2026-11-11 | Corrected against `a15b524`; framing, status header, §0, §13–§18; audit `docs/CHAPTER_9_AUDIT.md` |

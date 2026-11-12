# Chapter 4: Full Scoring & Feature Extraction (Step 3)

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

## PRNG Analysis Pipeline — Complete Operating Guide

**Status (2026-11-11):** **AUDITED AND CORRECTED against code at `a15b524`**
(`a15b5247336255801de8cf67d52653701b1e8c80`). HEAD at the time of the pass was `741822b`, whose diff
from `a15b524` touches only `docs/`, so the code is identical. Audit: `docs/CHAPTER_4_AUDIT.md`
(83 claims: 42 accurate, 7 BROKEN, 2 SWITCHED-OFF-BY-RULING, 15 DESCRIPTION-ONLY, 2 UNVERIFIED,
15 absorbed). See **§16** for the known gaps and **§19** for the closure statement.
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1).
**Numbering scheme:** **executable.** "Step 3" here is WATCHER `STEP_SCRIPTS[3] =
"run_step3_full_scoring.sh"` (`agents/watcher_agent.py:391`), manifest `full_scoring.json` (`:402`).
Where this chapter (or the code) says "Step 2.5" it means executable **Step 2**, the scorer
meta-optimizer (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). "Step 5" is executable Step 5.
Chapter numbers are not step numbers.

**Version:** 2.0.0 (Holdout Integration) — *text below is the 2.0.0 chapter, corrected in place; corrections are marked "Corrected 2026-11-11".*  
**File:** `survivor_scorer.py`, `full_scoring_worker.py` — *Step 3 also runs `run_step3_full_scoring.sh`, `generate_step3_scoring_jobs.py`, `scripts_coordinator.py`, `ramdisk_preload.sh`, `holdout_quality.py`, `models/global_state_tracker.py` (§0.2).*  
**Lines:** ~550 — *Corrected 2026-11-11: at `a15b524` `survivor_scorer.py` is 809 lines and `full_scoring_worker.py` 735.*  
**Purpose:** GPU-accelerated ML feature extraction for survivor seeds

---

## Table of Contents

1. [Overview](#1-overview)
2. [Environment Setup](#2-environment-setup)
3. [Constants and Configuration](#3-constants-and-configuration)
4. [SurvivorScorer Class](#4-survivorscorer-class)
5. [Feature Extraction](#5-feature-extraction)
6. [Dual Sieve Intersection](#6-dual-sieve-intersection)
7. [Batch Scoring](#7-batch-scoring)
8. [GPU-Batched Feature Extraction](#8-gpu-batched-feature-extraction)
9. [Complete Feature Reference](#9-complete-feature-reference)
10. [Integration Points](#10-integration-points)
11. [Method Reference](#11-method-reference)

*Added 2026-11-11:* 0. How this chapter was produced · 15. RANGE-MINER, the 22-array NPZ and the
window anchor · 16. Known gaps at `a15b524` · 17. Open items · 18. Verification declaration
(VIR-1…6) · 19. Closure statement.

---

## 0. How this chapter was produced (added 2026-11-11)

### 0.1 Method

This pass audited every substantive claim in the 2.0.0 chapter against the code at `a15b524`, read
on VM 101 as `michael`, and corrected the chapter **in place**. Nothing was deleted: stale text is
either annotated with a "Corrected 2026-11-11" note giving what the code does and where, or marked
**SUPERSEDED (2026-11-11)** with a pointer to what replaced it. Order of search for every
disagreement: governance trail (`docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`, the
system map, `BACKLOG.md`, `TODO_CONSOLIDATED.md`), then chapters, then code, then host state.

Read only. **No scoring, no pipeline step, no rig access.** "Works" below means the path is complete
on reading. The full claim ledger, the files opened, the surfaces searched and the count
reconciliation are in `docs/CHAPTER_4_AUDIT.md`.

### 0.2 What Step 3 actually is, end to end (verified at `a15b524`)

```
WATCHER run_step(3)                         agents/watcher_agent.py:1705, :1799-1820, :2081-2154
 └─ bash run_step3_full_scoring.sh --survivors … --chunk-size 1000 …   (unknown options ignored, :120-123)
     ├─ ramdisk preload → /dev/shm/prng/step3/ on EVERY distributed_config.json node   (:33-40;
     │     ramdisk_preload.sh:79-150): NPZ, optimal_scorer_config.json, train/holdout history
     ├─ generate_step3_scoring_jobs.py → scoring_chunks/chunk_NNNN.json (all 22 NPZ arrays per row)
     │     + scoring_jobs.json                                                     (:160-165)
     ├─ scp chunks to remotes                                                      (:203-215)
     ├─ scripts_coordinator.py → distributed_worker.py → full_scoring_worker.py, one job per GPU,
     │     run-scoped output dir, localhost retry                                  (:226)
     ├─ pull results, aggregate, merge 14 global_* → survivors_with_scores.json    (:237-408)
     └─ validate: ≥ 46 features                                                    (:475-478)
```

The manifest's own `actions` list (`agent_manifests/full_scoring.json:23-57`) names
`generate_full_scoring_jobs.py`, `full_scoring_worker.py` and `aggregate_scoring_results.py`; WATCHER
does not execute those actions, it runs `STEP_SCRIPTS[3]`. This divergence is already recorded
(`docs/PIPELINE_BEHAVIOUR_MODEL.md` D7; `docs/PROJECT_FILE_CATALOG.md` §7 gap 2).
`aggregate_scoring_results.py` does not exist in the tree.

### 0.3 Host state read for this pass (point-in-time, 2026-11-11)

- `agent_manifests/definitions.json` (gitignored): 220 lines, schema 1.1.0, `updated_at`
  2026-01-02; `pipeline_steps` holds only `step5_anti_overfit` and `step6_prediction`. **No Step-3
  entry; no reader in tracked code.** It does not constrain Step 3.
- The last Step-3 output on VM 101 is `survivors_with_scores.json` of **2026-03-11** (84 records,
  91 features each), i.e. before RANGE-MINER. Step 3 has not run on RANGE-MINER output (§15).
- No systemd unit or crontab entry runs Step 3.

---

## 1. Overview

### 1.1 What the Survivor Scorer Does

The Survivor Scorer extracts **47 ML features** from survivor seeds for quality prediction:

> **Corrected 2026-11-11 (D1):** previously said 47. At `a15b524` Step 3 emits **91 features per
> survivor**: **77** per-seed keys from `SurvivorScorer.extract_ml_features_batch()`
> (`survivor_scorer.py:702-791`) plus **14** run-global `global_*` keys
> (`full_scoring_worker.py:347-349`, `:403-405`). Step 5 trains on **89** (it drops `score` and
> `confidence`, `meta_prediction_optimizer_anti_overfit.py:1479-1482`). Host confirmation: the
> 2026-03-11 `survivors_with_scores.json` carries 91 `features` keys, 14 of them `global_*`. The full
> breakdown is §9.8. The 77 assumes the default three residue mods (see §4.2 and B5).

- **PRNG Sequence Generation:** CPU reference or PyTorch GPU
- **Match Analysis:** Exact matches, residue matching, lane agreement
- **Temporal Stability:** Window-based consistency tracking
- **Statistical Features:** Mean, std, entropy, KL divergence
- **Bidirectional Features:** Forward/reverse intersection analysis

### 1.2 Key Fixes in Final Version

```
BUG FIX 1: Added residue_mod_1/2/3 translation for Optuna
BUG FIX 2: Temporal stability optimization (reuse seq, no duplicate generation)
BUG FIX 3: Team Beta's targeted VRAM limit for RX 6600 rigs only
BUG FIX 4: Consolidated launch contention fix (PYTORCH_HIP_ALLOC_CONF)
BUG FIX 5: Explicit two-step NumPy to GPU tensor transfer (ROCm stability)
```

### 1.3 Key Features

| Feature | Description |
|---------|-------------|
| **47 ML Features** | Comprehensive seed quality metrics + holdout_hits |
| **GPU Acceleration** | PyTorch tensors for parallel computation |
| **ROCm Support** | AMD RX 6600 compatibility |
| **Batch Processing** | Vectorized scoring for entire pools |
| **Memory Management** | VRAM limits for 8GB GPUs |
| **Holdout Integration** | Computes y-label for Step 5 ML training |

> **Corrected 2026-11-11:** "47 ML Features" → 91 extracted / 89 trained (D1, §9.8). "ROCm
> Support — AMD RX 6600": the rigs are RX 6600 XT, gfx1032 (D16; `docs/PIPELINE_BEHAVIOUR_MODEL.md:66`,
> not re-checked on the rigs). "Memory Management — VRAM limits for 8GB GPUs": the limit is keyed to
> hostnames that no live rig has, so it is not applied (B1, §2.2). "Holdout Integration — y-label":
> Step 3 computes both `holdout_hits` and `holdout_quality`; since S111 Step 5 trains on
> `holdout_quality` (D2, §1.4).

### 1.4 HOLDOUT_HITS Integration (v2.0)

The scoring pipeline now computes `holdout_hits` — the ML y-label for Step 5:

```
┌─────────────────────────────────────────────────────────────┐
│  TRAIN HISTORY (used for sieving & feature extraction)      │
│  → Features extracted from THESE draws                      │
│  → Using features as y-label = CIRCULAR / OVERFIT           │
├─────────────────────────────────────────────────────────────┤
│  HOLDOUT HISTORY (never seen during sieving)                │
│  → Measures TRUE predictive power                           │
│  → Using this as y-label = HONEST EVALUATION                │
└─────────────────────────────────────────────────────────────┘
```

**Key Implementation Details:**

| Component | Value | Source |
|-----------|-------|--------|
| `offset` | `len(train_history)` | DERIVED, not configurable |
| `holdout_hits` | 0.0 - 1.0 | Hits / total holdout draws |

**Critical Rule (per Team Beta):**
```python
# OFFSET IS A LAW, NOT A PARAMETER
offset = len(train_history)  # THE ONLY SOURCE
```

This ensures temporal causality: holdout data is the CONTINUATION of training data in the PRNG sequence.

> **SUPERSEDED in part (2026-11-11) — the y-label.** The offset law above is implemented exactly as
> written for `holdout_hits` (`full_scoring_worker.py:299-305`, called with
> `train_history_len=len(train_history)` at `:364-370`). But **`holdout_hits` is no longer the Step-5
> y-label.** S111 (`docs/S111_IMPLEMENTATION_PLAN_FINAL.md`, "v1.1 proposal APPROVED by Team Beta",
> `:6`) added `holdout_quality` and `holdout_features` to every record (`full_scoring_worker.py:408-430`),
> and Step 5 trains on `target_field='holdout_quality'`, excluding `holdout_hits`
> (`meta_prediction_optimizer_anti_overfit.py:611`, `:1479-1482`). `holdout_hits` is a top-level
> record field, not one of the 91 `features` (`full_scoring_worker.py:407`). (D2)
>
> **How `holdout_quality` is computed (verified).** `holdout_features` are the same 77 per-seed
> features, extracted by `SurvivorScorer.extract_ml_features()` against `holdout_history`, with
> `skip = get_survivor_skip(meta)` — `skip_best`, else the NPZ `skip_min`, else 0
> (`holdout_quality.py:26-56`; `full_scoring_worker.py:411-418`). **No `len(train_history)` offset is
> applied on this path**: S111 replaced the uniform offset with a per-survivor skip
> (`docs/S111_IMPLEMENTATION_PLAN_FINAL.md:481`). `compute_holdout_quality` is a fixed 50/30/20
> composite of `residue_1000_match_rate`, the three lanes, the three residue coherences and
> `temporal_stability_mean` (`holdout_quality.py:61-82`).
>
> **"Continuation" holds only at skip = 0 (S1, governed).** Both holdout paths regenerate through the
> registry CPU reference, which applies `skip` **once, as a burn-in** (`prng_registry.py:176-177`),
> whereas the sieve kernel applies it **between every draw**. They agree only at skip = 0. This is
> `docs/BACKLOG.md` §8.1, which names exactly `survivor_scorer.py:124` and `full_scoring_worker.py:305`
> and records that Beta wants a read-only reachability audit and that **no fix is authorised**
> (SWITCHED-OFF-BY-RULING; §16).
>
> **Naming hazard.** This "offset" (holdout start = `len(train_history)`) is **not** the NPZ array 4
> `offset`, which after Window-Anchor Brief I is the window anchor (§15).

---

---

## 1.5 Configuration Sources (CRITICAL)

**Added: 2026-01-25** — Understand where parameters come from.

### Parameter Source by Run Method

| Run Method | Config Source | chunk_size Location |
|------------|---------------|---------------------|
| Manual: `bash run_step3_full_scoring.sh` | Script default | Line 70: `CHUNK_SIZE=1000` |
| Manual with override: `bash run_step3_full_scoring.sh --chunk-size 500` | CLI argument | Passed directly |
| WATCHER: `--start-step 3 --end-step 3` | Manifest | `agent_manifests/full_scoring.json` |
| WATCHER with override: `--params '{"chunk_size": 500}'` | CLI params | Overrides manifest |

### Key Insight

**WATCHER ignores script defaults.** It reads `default_params` from the manifest and passes them explicitly to the script.

> **Verified 2026-11-11.** `run_step` loads `default_params` (`agents/watcher_agent.py:1799-1812`),
> admits only `--params` keys the manifest declares (`:1813-1820`) and emits each as `--key value`
> (`:2135-2154`). `run_step3_full_scoring.sh` consumes `--survivors`, `--train-history`,
> `--holdout-history`, `--config`, `--chunk-size` and ignores the rest with a note (`:86-125`;
> manifest `_note_default_params`, `full_scoring.json:141`). Two caveats, both already recorded
> elsewhere: WATCHER's dataset-authority rewrite covers only `lottery_file`/`lottery_data`/
> `lottery_history` (`agents/watcher_agent.py:474`, `:1860`), so `train_history.json` and
> `holdout_history.json` (`full_scoring.json:120-121`) are passed as bare names — committed
> **Chapter 13 audit B2**; and the fleet Step 3 reaches is `distributed_config.json`'s, not the
> resolved execution set WATCHER freezes at `:1748` — committed **Chapter 9 O-6**.
> `parameter_bounds.chunk_size.default` is 5000 (`full_scoring.json:70`) while `default_params` says
> 1000 (`:122`); WATCHER uses the latter.

### To Change chunk_size Permanently

Update **BOTH** files:

```bash
# 1. Update script default (for manual runs)
sed -i 's/^CHUNK_SIZE=.*/CHUNK_SIZE=1000/' run_step3_full_scoring.sh

# 2. Update manifest default (for WATCHER runs)
sed -i 's/"chunk_size": [0-9]*/"chunk_size": 1000/' agent_manifests/full_scoring.json

# 3. Verify
grep "CHUNK_SIZE=" run_step3_full_scoring.sh
grep '"chunk_size"' agent_manifests/full_scoring.json
```

### OOM Prevention Reminder

Use `chunk_size=1000` (not 5000) to prevent OOM on 7.7GB mining rigs:
- 1000 seeds/chunk × ~500MB = safe for 12 concurrent workers
- 5000 seeds/chunk × ~1.5GB = OOM with 7+ concurrent workers

> **Corrected 2026-11-11.** "12 concurrent workers" (D6): `distributed_config.json` gives each rig
> 8 GPUs with `max_concurrent_script_jobs: 8`, and localhost 1 GPU with 2. The memory figures
> (~500 MB, ~1.5 GB, OOM at 7+) are **[UNVERIFIED]** (U1): no measurement was found or opened.


## 2. Environment Setup

### 2.1 ROCm Configuration (CRITICAL)

**MUST BE BEFORE ANY IMPORTS:**

```python
import sys, os, socket
HOST = socket.gethostname()

# AMD ROCm fixes + CRITICAL VRAM LAUNCH FIX
if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
    # Forces allocator to use small chunks (128MB) and aggressively free memory
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", 
                          "garbage_collection_threshold:0.8,max_split_size_mb:128")

os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")
os.environ.setdefault('CUPY_CUDA_MEMORY_POOL_TYPE', 'none')
```

> **Corrected 2026-11-11 (B1, BROKEN — same class as Chapter 2 F-5 and Chapter 9 B-1).** The block
> above is the code at `survivor_scorer.py:18-31`, and `full_scoring_worker.py:90` and `:99` carry
> the same tuple. **It never fires**: the live worker hostnames are `rrig6600`, `rrig6600b`,
> `rrig6600c` (`rig_profiles_config.json:30`, `:42`, `:53`), none of which is in
> `["rig-6600", "rig-6600b", "rig-6600c"]`. So `PYTORCH_HIP_ALLOC_CONF` is not set on any rig, and
> `setup_gpu_environment` takes the CUDA branch there (`full_scoring_worker.py:103-105`); GPU
> isolation still holds because `scripts_coordinator.py:967-971` exports `HIP_VISIBLE_DEVICES`.
> Any repair follows Chapter 2 §9.5: **do not rename hosts**; first decide whether the prelude is
> still supported; if kept, key it from the declared rig profile, never from another hostname tuple.

### 2.2 VRAM Limiting for RX 6600

```python
# Team Beta's targeted VRAM limit for RX 6600 rigs only
if HOST in ["rig-6600", "rig-6600b", "rig-6600c"] and TORCH_AVAILABLE:
    if torch.cuda.is_available():
        # Limit PyTorch to 80% (6.4GB of 8GB) VRAM
        torch.cuda.set_per_process_memory_fraction(0.8)
        # Disable benchmark mode to reduce memory fragmentation
        torch.backends.cudnn.benchmark = False
```

**Why This Matters:**
- RX 6600 has 8GB VRAM
- 12-worker launch can cause instantaneous VRAM spike
- Linux OOM Killer intervention prevented by limiting to 6.4GB

> **Corrected 2026-11-11.** Code as shown is `survivor_scorer.py:59-68`, behind the same dead hostname
> guard (`:61`), so the 80 % cap and `cudnn.benchmark = False` are **not applied** on the live rigs
> (B1). "12-worker launch" (D6): rigs run 8 GPUs. "RX 6600": the rigs are RX 6600 XT (D16).

### 2.3 Safe Entropy Function

```python
from scipy.stats import entropy as _entropy

# CRITICAL: Safe entropy — fixes CuPy → NumPy implicit conversion
def entropy(p, q=None, *args, **kwargs):
    p = p.get() if hasattr(p, 'get') else p  # CuPy → NumPy
    if q is not None:
        q = q.get() if hasattr(q, 'get') else q
    return _entropy(p, q, *args, **kwargs)
```

---

---

## 2.5 Ramdisk Prerequisites (CRITICAL)

**Added: 2026-01-22**

Step 3 jobs expect training data at `/dev/shm/prng/step3/` on **ALL nodes**.

### Required Files

| File | Purpose |
|------|---------|
| `train_history.json` | Training draws for scoring |
| `holdout_history.json` | Holdout draws for validation |

> **Corrected 2026-11-11 (absorbed into D3).** The path is right (`generate_step3_scoring_jobs.py:236-237`;
> `ramdisk_preload.sh:20-22`), but the script preloads **four** files, not two:
> `bidirectional_survivors_binary.npz`, `optimal_scorer_config.json`, `train_history.json`,
> `holdout_history.json` (`run_step3_full_scoring.sh:36-40`). Only the two histories are read from the
> ramdisk by the worker; chunk files are read from `<repo>/scoring_chunks/` (`generate_step3_scoring_jobs.py:235`).

### Why Ramdisk?

Jobs reference paths like `/dev/shm/prng/step3/train_history.json` to avoid:
- Disk I/O contention with 12 concurrent GPU jobs
- NFS/network latency on distributed nodes
- JSON parsing overhead on repeated loads

### Verification (Before Running Step 3)

```bash
# Check all nodes
ssh 192.168.3.120 "ls -la /dev/shm/prng/step3/"
ssh 192.168.3.154 "ls -la /dev/shm/prng/step3/"
ls -la /dev/shm/prng/step3/  # Zeus
```

**Expected:** Both files present on all three nodes.

> **Corrected 2026-11-11 (D4).** These commands name `.120` and `.154` only, and label `.120` as
> both "rig-6600" and "rig-6600c". `distributed_config.json` lists four nodes — localhost, `.120`,
> `.154`, `.162` (the three rigs' default bare-Ubuntu addresses). When the rigs are booted into
> Proxmox the worker endpoints are the CT100 addresses `.122`/`.156`/`.164` (`CLAUDE.md` §3), which
> no Step-3 surface reads (Chapter 9 O-6). Run nothing here from an agent session; rig access is
> Michael's.

### Manual Population (If Missing)

```bash
# On Zeus
mkdir -p /dev/shm/prng/step3
cp train_history.json holdout_history.json /dev/shm/prng/step3/

# On rig-6600 / rig-6600c
ssh 192.168.3.120 "mkdir -p /dev/shm/prng/step3"
scp train_history.json holdout_history.json 192.168.3.120:/dev/shm/prng/step3/

# On rig-6600b
ssh 192.168.3.154 "mkdir -p /dev/shm/prng/step3"
scp train_history.json holdout_history.json 192.168.3.154:/dev/shm/prng/step3/
```

### Automatic Preload

The WATCHER agent runs ramdisk preload but in **"Standalone mode"** — it only populates the local node. For distributed execution, you must manually populate remote nodes or run the preload script on each node.

> **SUPERSEDED (2026-11-11) by `ramdisk_preload.sh` v2.1.0 (D3).** The preload, which
> `run_step3_full_scoring.sh` sources on every run (`:33-40`), copies to **every** node listed in
> `distributed_config.json`, local and remote (`ramdisk_preload.sh:26-40`, `:104-150`), with a
> headroom check and a `.ready` sentinel written only when all files arrived (`:121-129`,
> `:143-150`). "Standalone mode" means only that `WATCHER_MANAGED_RAMDISK` is unset, which makes the
> preload clear the step directory first (`:86-87`, `:114`, `:136`). A remote copy failure is
> non-fatal and logged `INCOMPLETE` (`:148-150`). Manual population is a fallback, not the norm.

### Common Failure Pattern

If Step 3 jobs fail instantly (~3 seconds) with no output file:
1. Check ramdisk exists on failing node
2. Worker expects `--train-history /dev/shm/prng/step3/train_history.json`
3. Missing file = immediate argument parsing failure

> **Corrected 2026-11-11 (D5).** Argparse accepts any path. A missing history file fails in
> `load_lottery_history`; the worker catches it, prints a JSON `{"status": "error", …}` line and
> exits 1 (`full_scoring_worker.py:579-614`). `scripts_coordinator.py` then retries the job on
> localhost (`:1068-1112`).


## 3. Constants and Configuration

### 3.1 Default Constants

```python
DEFAULT_MOD = 1000              # PRNG output modulo
DEFAULT_RESIDUE_MODS = [8, 125, 1000]  # 3-lane validation
DEFAULT_MAX_OFFSET = 5          # Maximum offset for alignment
DEFAULT_TEMPORAL_WINDOW = 100   # Window size for stability
DEFAULT_TEMPORAL_WINDOWS = 5    # Number of windows
DEFAULT_MIN_CONFIDENCE = 0.1    # Minimum confidence threshold
```

### 3.2 Why [8, 125, 1000]?

| Mod | Purpose | Bit Pattern |
|-----|---------|-------------|
| **8** | Fast power-of-2 check | Low 3 bits |
| **125** | Coprime to 8 | Different bit range |
| **1000** | Full 3-digit match | Complete residue |

**Mathematical Property:** 8 × 125 = 1000, so checking all three provides orthogonal validation.

> **Corrected 2026-11-11 (D7).** The constants are as printed (`survivor_scorer.py:73-78`). But
> because gcd(8, 125) = 1 and 8 × 125 = 1000, the Chinese Remainder Theorem makes a mod-1000 match
> **exactly** a mod-8 match **and** a mod-125 match: the third lane is determined by the first two,
> not orthogonal to them. The code computes all three regardless (`:377-390`, `:625-667`). Note too
> that these are only the *defaults*; Step 2 tunes other mods, which Step 3 does not apply (B5, §4.2).

### 3.3 Java LCG Fallback

```python
def java_lcg_sequence(seed: int, n: int, mod: int) -> np.ndarray:
    """CPU-only fallback for tiny sequences"""
    arr = np.empty(n, dtype=np.int64)
    state = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
    for i in range(n):
        state = (state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        arr[i] = (state >> 16) % mod
    return arr
```

> **Corrected 2026-11-11 (D8).** The function exists exactly as shown (`survivor_scorer.py:81-87`)
> but **nothing in live code calls it** (tracked `*.py` searched; the only other reference is the
> patch file `step6_restoration/survivor_scorer_PATCH.py`). Every Step-3 sequence comes from the
> registry (`get_cpu_reference` / `get_pytorch_gpu_reference`, §10.2). It also differs from the
> registry: it applies Java's `seed ^ 0x5DEECE66D` scramble, which the registry reference
> deliberately does not (`prng_registry.py:175`; GPU twin `:120-122`, "no XOR!").

---

## 4. SurvivorScorer Class

### 4.1 Constructor

```python
class SurvivorScorer:
    def __init__(self, 
                 prng_type: str = 'java_lcg', 
                 mod: int = 1000, 
                 residue_mods: List[int] = None, 
                 config_dict: Optional[Dict] = None):
```

### 4.2 Optuna Parameter Translation (BUG FIX 1)

```python
# Optuna passes: {"residue_mod_1": 14, "residue_mod_2": 137, "residue_mod_3": 1136}
# But self.residue_mods expects a LIST: [14, 137, 1136]
if 'residue_mod_1' in config_dict:
    residue_mods = [
        config_dict.get('residue_mod_1', 8),
        config_dict.get('residue_mod_2', 125),
        config_dict.get('residue_mod_3', 1000)
    ]
```

> **Corrected 2026-11-11 (B5, BROKEN — not found in the governance trail).** The translation exists
> exactly as shown (`survivor_scorer.py:97-111`). **Step 3 never uses it.** The worker builds
> `scorer_class(prng_type=prng_type, mod=mod)` with no `config_dict` (`full_scoring_worker.py:344`)
> and has no `--config` argument (`:522-550`); the job generator reads only `prng_type` and `mod`
> from `optimal_scorer_config.json` (`generate_step3_scoring_jobs.py:201-209`) and passes neither the
> file nor its other keys to the jobs (`:241-248`). No commit ever passed it
> (`git log -S config_dict -- full_scoring_worker.py generate_step3_scoring_jobs.py`: none).
> **Host evidence:** the live `optimal_scorer_config.json` holds `residue_mod_1/2/3 = 17/150/1449`,
> `temporal_num_windows = 9`, `min_confidence_threshold ≈ 0.219`, `max_offset = 13`; the
> 2026-03-11 `survivors_with_scores.json` carries `residue_8_*`, `residue_125_*`, `residue_1000_*`
> only. So Step 3 always scores with the defaults of §3.1, and the manifest's description "Applies
> optimized scoring parameters from Step 2.5 to ALL survivors" (`full_scoring.json:3`) does not hold.
> **A repair is a governed schema decision, not a one-line wiring:** applying the config renames the
> residue features (`residue_{mod}_*`, `survivor_scorer.py:631`, `:666-667`), which changes the
> 91/89 name contract, and `compute_holdout_quality` reads `residue_1000_*`, `residue_8_*`,
> `residue_125_*` by name (`holdout_quality.py:61-82`).

### 4.3 Instance Attributes

| Attribute | Type | Default | Purpose |
|-----------|------|---------|---------|
| `prng_type` | str | 'java_lcg' | PRNG algorithm |
| `mod` | int | 1000 | Output modulo |
| `residue_mods` | List[int] | [8, 125, 1000] | Validation lanes |
| `max_offset` | int | 5 | Alignment offset |
| `temporal_window_size` | int | 100 | Window size |
| `temporal_num_windows` | int | 5 | Number of windows |
| `min_confidence_threshold` | float | 0.1 | Minimum confidence |
| `device` | str | 'cuda'/'cpu' | PyTorch device |

> **Corrected 2026-11-11 (B2, BROKEN — not found in the governance trail).** Attributes and defaults
> are as listed (`survivor_scorer.py:107-117`). But `max_offset` is **stored and never read**: the
> only `.max_offset` in the Step-3 code is its assignment at `:108`. There is no alignment search;
> `best_offset` is hardcoded `0.0` (`:373`, `:707`) and has been since the initial commit `0101306`.
> In Step 3 every attribute here holds its default anyway (B5).

### 4.4 Sequence Generation

```python
def _generate_sequence(self, seed: int, n: int, skip: int = 0) -> np.ndarray:
    """
    Generate PRNG sequence using configured prng_type.
    Uses prng_registry for dynamic PRNG lookup - NO HARDCODING.
    """
    raw = self._cpu_func(seed=int(seed), n=n, skip=skip)
    return np.array([v % self.mod for v in raw], dtype=np.int64)
```

### 4.5 Seed Coercion

```python
def _coerce_seed_list(self, items) -> List[int]:
    """Convert mixed list (int or dict with seed) to list of ints."""
    out = []
    for x in items or []:
        if isinstance(x, dict):
            if "seed" in x:
                out.append(int(x["seed"]))
        else:
            out.append(int(x))
    return out
```

---

## 5. Feature Extraction

### 5.1 extract_ml_features()

```python
def extract_ml_features(self, 
                        seed: int, 
                        lottery_history: List[int], 
                        forward_survivors=None, 
                        reverse_survivors=None, 
                        skip: int = 0) -> Dict[str, float]:
    """
    Extract 46 ML features for a single seed.
    Uses PyTorch tensors — proven stable on ROCm.
    """
```

> **Corrected 2026-11-11.** The signature is exact (`survivor_scorer.py:350`); the docstring shown
> is not in the code, and the count is 77 per-seed keys, not 46 (absorbed into D1). (D9) The
> `forward_survivors` and `reverse_survivors` parameters are **accepted and ignored**: the method
> never reads them and zero-fills the 18 bidirectional keys (`:455-461`). In Step 3 this method is
> used only for `holdout_features` and for the sequential fallback (`full_scoring_worker.py:413`,
> `:445`); the main path is §8.

### 5.2 Feature Extraction Flow

```
1. Generate PRNG sequence
2. Convert to PyTorch tensors (two-step transfer for ROCm)
3. Compute base matching score
4. Compute residue features for each mod
5. Compute temporal stability across windows
6. Compute statistical features
7. Compute lane agreement
8. Fill defaults for bidirectional features
9. Clear VRAM
10. Return feature dict
```

> **Corrected 2026-11-11 (absorbed into D1).** Two steps are missing between 7 and 8 at `a15b524`:
> **digit-wise agreement** (4 features, S119, `survivor_scorer.py:426-436`) plus `pred_min`/`pred_max`
> and the four residual features (`:438-447`), and **Battery Tier 1A** (23 features, S113,
> `compute_battery_features`, `:188-348`, called at `:451-452`). If the history is shorter than half a
> temporal window, the five `temporal_stability_*` keys are absent on this path (`:404`), unlike the
> batch path, which zero-fills them (`:691-696`).

### 5.3 Two-Step Tensor Transfer (BUG FIX 5)

```python
# CRITICAL: Explicit two-step transfer (NumPy -> CPU Tensor -> GPU Tensor)
# This stabilizes data transfer on the crowded, low-bandwidth RX 6600 PCIe bus.
pred_cpu = torch.from_numpy(seq).to(torch.long)
pred = pred_cpu.to(self.device)
act = torch.tensor(hist_np, device=self.device, dtype=torch.long)
```

> **Verified 2026-11-11:** code as shown, `survivor_scorer.py:359-363`. The stability rationale is
> **[UNVERIFIED]** (U2): no A/B record was found.

### 5.4 Residue Features

```python
for mod in self.residue_mods:
    p_res = pred % mod
    a_res = act % mod
    match_rate = (p_res == a_res).float().mean().item()
    
    # KL divergence via histogram
    p_hist = torch.histc(p_res.float(), bins=mod, min=0, max=mod-1)
    a_hist = torch.histc(a_res.float(), bins=mod, min=0, max=mod-1)
    p_dist = (p_hist / p_hist.sum()).clamp(min=1e-10)
    a_dist = (a_hist / a_hist.sum()).clamp(min=1e-10)
    
    kl = entropy(p_dist.cpu().numpy(), a_dist.cpu().numpy())
    
    features[f'residue_{mod}_match_rate'] = match_rate
    features[f'residue_{mod}_coherence'] = 1.0 / (1.0 + kl)
    features[f'residue_{mod}_kl_divergence'] = kl
```

### 5.5 Temporal Stability (BUG FIX 2)

```python
# OPTIMIZED: Reuse seq already generated, no duplicate generation
scores = []
stride = max(1, (n - self.temporal_window_size) // self.temporal_num_windows)

for i in range(self.temporal_num_windows):
    s = i * stride
    e = min(s + self.temporal_window_size, n)
    if e - s < self.temporal_window_size // 2:
        break
    # Use seq[s:e] instead of regenerating
    scores.append(np.mean(seq[s:e] == hist_np[s:e]))

if scores:
    arr = np.array(scores)
    trend = np.polyfit(np.arange(len(arr)), arr, 1)[0] if len(arr) > 1 else 0.0
    features.update({
        'temporal_stability_mean': float(arr.mean()),
        'temporal_stability_std': float(arr.std()),
        'temporal_stability_min': float(arr.min()),
        'temporal_stability_max': float(arr.max()),
        'temporal_stability_trend': float(trend)
    })
```

---

## 6. Dual Sieve Intersection

### 6.1 compute_dual_sieve_intersection()

```python
def compute_dual_sieve_intersection(
    self,
    forward_survivors: List[int],
    reverse_survivors: List[int]
) -> Dict[str, Any]:
    """
    Compute intersection of forward and reverse sieve survivors.
    Per Team Beta: NEVER discard valid intersection, Jaccard is metadata.
    """
```

### 6.2 Return Structure

```python
{
    "intersection": [123, 456, 789],  # Sorted list of common seeds
    "jaccard": 0.15,                   # Jaccard similarity coefficient
    "counts": {
        "forward": 1000,
        "reverse": 800,
        "intersection": 150,
        "union": 1650
    }
}
```

### 6.3 Jaccard Similarity

```python
forward_set = set(forward_survivors)
reverse_set = set(reverse_survivors)
intersection = forward_set & reverse_set
union = forward_set | reverse_set
jaccard = len(intersection) / len(union) if union else 0.0
```

> **Verified 2026-11-11:** `survivor_scorer.py:139-180`, including the empty-input return
> (`:153-163`). Step 3 does not call this method; its live caller is Step 6
> (`prediction_generator.py:610`). The bidirectional *features* Step 3 emits come from the NPZ
> metadata (§8.5), not from this method.

---

## 7. Batch Scoring

### 7.1 batch_score_vectorized()

```python
def batch_score_vectorized(self, 
                           seeds: Union[List[int], torch.Tensor], 
                           lottery_history: Union[List[int], torch.Tensor],
                           device: Optional[str] = None, 
                           return_dict: bool = False):
    """
    Vectorized scoring using GPU-accelerated PRNG generation.
    Returns either tensor of scores or list of dicts.
    """
```

### 7.2 Vectorized Kernel

```python
def _vectorized_scoring_kernel(self, seeds_tensor, lottery_history_tensor, device):
    batch_size = seeds_tensor.shape[0]
    history_len = lottery_history_tensor.shape[0]
    
    if has_pytorch_gpu(self.prng_type):
        prng_func = get_pytorch_gpu_reference(self.prng_type)
        predictions = prng_func(
            seeds=seeds_tensor, 
            n=history_len, 
            mod=self.mod,
            device=device, 
            skip=0
        )
    else:
        # CPU fallback
        predictions = self._cpu_batch_generate(...)
    
    # Vectorized matching
    matches = (predictions == lottery_history_tensor.unsqueeze(0))
    scores = matches.float().sum(dim=1) / history_len
    
    return scores
```

> **Corrected 2026-11-11 (B3, BROKEN — not found in the governance trail).** The real kernel
> (`survivor_scorer.py:492-516`) wraps the GPU call in `try/except` and passes the family's
> `default_params` (`:497-504`), and it has no `else`. `predictions` is assigned only inside
> `if has_pytorch_gpu(...)`, so for a family without a PyTorch implementation `if predictions is None`
> (`:505`) raises `UnboundLocalError`. When reached through a GPU exception, the CPU fallback stores
> the registry's raw 32-bit outputs (`:510-511`; `prng_registry.py:181`) without `% self.mod`, and
> compares them with 3-digit draws. Reach: nothing in the pipeline calls `batch_score_vectorized` at
> `a15b524` (`scorer_trial_worker.py:36` mentions it only in a docstring).

### 7.3 Legacy batch_score()

```python
def batch_score(self, 
                seeds: List[int], 
                lottery_history: List[int], 
                use_dual_gpu: bool = False, 
                window_metadata=None) -> List[Dict]:
    """
    Legacy per-seed scoring with full feature extraction.
    Slower but provides complete feature dicts.
    """
    results = []
    for seed in seeds:
        features = self.extract_ml_features(seed, lottery_history)
        results.append({
            'seed': seed, 
            'features': features, 
            'score': features['score']
        })
    return results
```

---

## 8. GPU-Batched Feature Extraction

### 8.1 extract_ml_features_batch()

```python
def extract_ml_features_batch(self, 
                               seeds: List[int], 
                               lottery_history: List[int], 
                               forward_survivors=None, 
                               reverse_survivors=None, 
                               survivor_metadata=None) -> List[Dict[str, float]]:
    """
    GPU-BATCHED ML feature extraction - CRYPTO MINER STYLE
    Processes ALL seeds in parallel on GPU, single CPU transfer at end.
    """
```

### 8.2 Batch Processing Flow

```
STEP 1: Generate ALL predictions on GPU
    └─→ seeds_t, hist_t tensors
    └─→ PyTorch GPU PRNG or CPU fallback
    └─→ predictions shape: (batch_size, n)

STEP 2: Compute ALL features on GPU
    └─→ Base matching
    └─→ Stats (mean, std, min, max)
    └─→ Residuals
    └─→ Lane agreement
    └─→ Residue features (vectorized histogram)
    └─→ Temporal stability

STEP 3: SINGLE CPU TRANSFER
    └─→ Stack all tensors
    └─→ One .cpu().numpy() call
    └─→ Build result dicts
```

> **Corrected 2026-11-11.**
> - **B4 (BROKEN — not found in the governance trail): the CPU fallback is not mod-reduced.**
>   `_cpu_batch_generate` copies the registry's raw 32-bit outputs (`survivor_scorer.py:802-809`;
>   `prng_registry.py:181`), unlike `_generate_sequence`, which applies `% self.mod` (`:125`). It is
>   used on any GPU exception (`:576-579`) and for every family without a PyTorch implementation
>   (`:580-582`) — **42 of the 44 registry families**; only `java_lcg` and `java_lcg_hybrid` have one
>   (`prng_registry.py:3961`, `:3974`). The resulting features compare raw values with 3-digit draws
>   and the record still carries all 91 keys, so nothing downstream notices. Today's sieve target is
>   `java_lcg`, so the default path does not reach it.
> - **Absorbed into D1:** STEP 2 also computes `pred_min`/`pred_max`, the four residuals
>   (`:599-606`) and the four digit features (`:616-621`); after the transfer, Battery Tier 1A is
>   computed per seed **on the CPU** (`:745-768`).
> - **D11:** there are two device-to-host transfers, not one — the stacked tensor (`:739`) and
>   `predictions.cpu()` for the battery (`:747`).

### 8.3 Vectorized Histogram (No Python Loops)

```python
# VECTORIZED batch histogram using scatter_add - NO PYTHON LOOPS!

# Create batch indices: [0,0,0,...,1,1,1,...,2,2,2,...]
batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, n)

# Flatten for scatter: (batch_size * n,)
p_flat = p_res.reshape(-1)
batch_flat = batch_idx.reshape(-1)

# Compute combined index: batch_idx * mod + residue_value
p_scatter_idx = batch_flat * mod + p_flat

# Scatter to build histograms: (batch_size * mod,)
p_hists = torch.zeros(batch_size * mod, device=device)
ones = torch.ones_like(p_flat, dtype=torch.float32)
p_hists.scatter_add_(0, p_scatter_idx, ones)

# Reshape to (batch_size, mod)
p_hists = p_hists.reshape(batch_size, mod)
```

### 8.4 Single Transfer Optimization

```python
# Collect all tensors and transfer once
results_gpu = {
    'score': base_scores * 100,
    'confidence': torch.clamp(base_scores, min=self.min_confidence_threshold),
    'exact_matches': match_counts,
    # ... all other features
}

# Stack all into one tensor, transfer, unpack
keys = list(results_gpu.keys())
stacked = torch.stack([results_gpu[k] for k in keys], dim=1)
stacked_cpu = stacked.cpu().numpy()  # SINGLE TRANSFER!
```

### 8.5 Metadata Integration

```python
# Add metadata if available
if survivor_metadata and seed in survivor_metadata:
    meta = survivor_metadata[seed]
    for field in ['forward_count', 'reverse_count', 'bidirectional_count',
                  'intersection_count', 'intersection_ratio', 
                  'skip_min', 'skip_max', 'skip_range', 
                  'skip_mean', 'skip_std', 'skip_entropy', ...]:
        if field in meta and meta[field] is not None:
            features[field] = float(meta[field])
```

> **Corrected 2026-11-11.** The batch path merges exactly **18** named fields
> (`survivor_scorer.py:770-781`, list `:774-779`) and zero-fills any that are absent (`:784-791`).
> `survivor_metadata` is the chunk row, i.e. one NPZ row with all 22 arrays
> (`generate_step3_scoring_jobs.py:76-93`; `full_scoring_worker.py:582-584`). Three governed gaps:
> - **B6 (BROKEN, governed — system map §7.3 defect 2; P0-B "Authorized, P0").** If the batch call
>   throws, the sequential fallback (`full_scoring_worker.py:438-443`) merges only six fields —
>   `forward_count, reverse_count, bidirectional_count, skip_min, skip_max, skip_range` (`:451-456`) —
>   so seven NPZ-backed features silently become `0.0` while the record keeps 91 keys. Gate F-PARITY,
>   which Beta requires before fallback output may train a model, exists in no tracked `*.py`.
> - **B7 (BROKEN, governed as a fact — system map §7.1; P0-A).** `skip_mean`, `skip_std`,
>   `skip_entropy`, `survivor_velocity`, `velocity_acceleration` are in the list but in neither the 24
>   canonical record fields (`utils/canonical_records.py:115-123`) nor the 22 NPZ arrays
>   (`convert_survivors_to_binary.py:51-74`), so they are always `0.0`.
> - **S2 (SWITCHED-OFF-BY-RULING).** The per-seed sieve match rates `forward_matches` /
>   `reverse_matches` reach every chunk but are **not** in the list. TB §8 ruling, recorded at
>   `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:165-171` and `:276`: "do NOT silently add
>   them"; a separate feature-schema decision (Option A or B) is required.

---

## 9. Complete Feature Reference

### 9.1 Base Features (5)

| Feature | Description | Range |
|---------|-------------|-------|
| `score` | Base match rate × 100 | 0-100 |
| `confidence` | Clamped match rate | 0.1-1.0 |
| `exact_matches` | Count of exact matches | 0-n |
| `total_predictions` | Sequence length | n |
| `best_offset` | Best alignment offset | 0-5 |

> **Corrected 2026-11-11.** Batch anchors `survivor_scorer.py:703-707`. `best_offset` is always
> `0.0`; there is no alignment search (B2). `total_predictions` is `n = len(train_history)`, the same
> for every seed (`:706`). Both are constant columns that Step 5 still trains on (committed
> Chapter 14 audit Q3).

### 9.2 Residue Features (9 per default config)

For each mod in [8, 125, 1000]:

| Feature | Description |
|---------|-------------|
| `residue_{mod}_match_rate` | Match rate at this modulo |
| `residue_{mod}_coherence` | 1 / (1 + KL divergence) |
| `residue_{mod}_kl_divergence` | KL divergence of distributions |

### 9.3 Temporal Stability Features (5)

| Feature | Description |
|---------|-------------|
| `temporal_stability_mean` | Average window match rate |
| `temporal_stability_std` | Std dev of window rates |
| `temporal_stability_min` | Minimum window rate |
| `temporal_stability_max` | Maximum window rate |
| `temporal_stability_trend` | Linear regression slope |

### 9.4 Statistical Features (10)

| Feature | Description |
|---------|-------------|
| `pred_mean` | Mean of predictions |
| `pred_std` | Std dev of predictions |
| `pred_min` | Minimum prediction |
| `pred_max` | Maximum prediction |
| `actual_mean` | Mean of actual values |
| `actual_std` | Std dev of actual values |
| `residual_mean` | Mean of (pred - actual) |
| `residual_std` | Std dev of residuals |
| `residual_abs_mean` | Mean absolute residual |
| `residual_max_abs` | Maximum absolute residual |

### 9.5 Lane Agreement Features (3)

| Feature | Description |
|---------|-------------|
| `lane_agreement_8` | Match rate mod 8 |
| `lane_agreement_125` | Match rate mod 125 |
| `lane_consistency` | Average of lane agreements |

### 9.6 Bidirectional/Metadata Features (14)

> **Corrected 2026-11-11 (absorbed into D1).** The heading says 14; the table below and the code
> both have **18** (`survivor_scorer.py:774-779`). **13** are live NPZ-backed trial-level values;
> `skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity` and `velocity_acceleration` have no
> producer and are always `0.0` (B7). The per-seed sieve match rates
> are not among them (S2). `actual_mean`/`actual_std` (§9.4) are the history's mean and standard
> deviation, the same for every seed (`:608-609`, `:716-717`).

| Feature | Source | Description |
|---------|--------|-------------|
| `forward_count` | Metadata | Forward sieve survivors |
| `reverse_count` | Metadata | Reverse sieve survivors |
| `intersection_count` | Metadata | Bidirectional intersection |
| `intersection_ratio` | Metadata | intersection / union |
| `survivor_overlap_ratio` | Metadata | Overlap ratio |
| `forward_only_count` | Metadata | Forward-only seeds |
| `reverse_only_count` | Metadata | Reverse-only seeds |
| `bidirectional_count` | Metadata | Bidirectional seeds |
| `bidirectional_selectivity` | Metadata | Selectivity score |
| `skip_min` | Metadata | Minimum skip value |
| `skip_max` | Metadata | Maximum skip value |
| `skip_range` | Metadata | skip_max - skip_min |
| `skip_mean` | Metadata | Average skip |
| `skip_std` | Metadata | Skip std dev |
| `skip_entropy` | Metadata | Skip entropy |
| `survivor_velocity` | Metadata | Survivor count velocity |
| `velocity_acceleration` | Metadata | Velocity change rate |
| `intersection_weight` | Metadata | Weighted intersection |

### 9.7 Holdout Performance Feature (1)

| Feature | Description | Range | Purpose |
|---------|-------------|-------|---------|
| `holdout_hits` | Match rate on holdout data | 0.0 - 1.0 | ML y-label for Step 5 |

**Why This Feature Exists:**

This is the ONLY non-circular training target. All other features measure training performance (what the sieve already validated). `holdout_hits` measures future performance (generalization).

> **SUPERSEDED (2026-11-11) by S111 `holdout_quality` (D2).** `holdout_hits` is still computed
> (`full_scoring_worker.py:364-370`, `:407`) but it is a **top-level record field**, not one of the 91
> `features`, and it is **not** the training target: Step 5 trains on `holdout_quality` and excludes
> `holdout_hits` (`meta_prediction_optimizer_anti_overfit.py:611`, `:1479-1482`). See §1.4 for how
> `holdout_quality` is built and for the skip-semantics caveat (S1).

### 9.8 Total Feature Count: 47

```
Base:                5
Residue (3 mods):    9
Temporal:            5
Statistical:        10
Lane Agreement:      3
Bidirectional:      14
Holdout:             1   ← NEW
─────────────────────────
TOTAL:              47
```

> **SUPERSEDED (2026-11-11) by §9.9.** The 47 was the 2.0.0 contract. Code at `a15b524` emits 91.

### 9.9 The feature contract at `a15b524` — 91 extracted / 89 trained (added 2026-11-11)

Re-derived from the batch producer, not taken from any report. It agrees with committed
`docs/CHAPTER_14_AUDIT.md` Q3 and `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` §7.1–§7.2.

```
Per-seed (SurvivorScorer.extract_ml_features_batch)          anchor (survivor_scorer.py)
  Base (score, confidence, exact_matches, total_predictions,
        best_offset)                                     5   :703-707
  Statistical (pred_mean/std/min/max, actual_mean/std,
        residual_mean/std/abs_mean/max_abs)             10   :708-717
  Lane agreement (8, 125, consistency)                   3   :718-720
  Digit agreement (hundreds, tens, ones, expected count)
        — S119                                           4   :722-725
  Temporal stability                                     5   :726-730
  Residue (3 default mods × match_rate/coherence/KL)     9   :624-667, merged :734
  Battery Tier 1A (FFT 5, autocorr 12, cusum 3, bits 3)
        — S113                                          23   :188-348, :754-768
  Metadata merge list (13 live + 5 dead)                18   :774-779, defaults :784-791
                                                       ───
                                                        77
Run-global (GlobalStateTracker, prefixed global_)       14   full_scoring_worker.py:347-349, :403-405;
                                                             names models/global_state_tracker.py:366-381
                                                       ───
Extracted                                               91
Trained = 91 − score − confidence                       89   meta_prediction_optimizer_anti_overfit.py:1479-1482
```

- **Three namespaces** (system map §8): 72 survivor-local, 14 run-global (identical for every
  survivor in a run), 5 permanently zero.
- **Not features:** `holdout_hits`, `holdout_quality`, `holdout_features` (77 keys, the same per-seed
  set computed on the holdout), `metadata` — all top-level record fields
  (`full_scoring_worker.py:390-430`).
- **The count is conditional on the default residue mods.** With three mods the residue block is 9;
  it is keyed by mod value (`residue_{mod}_*`), so any other mod set changes the names (B5).
- The globals are merged twice — by the worker (`full_scoring_worker.py:403-405`) and again by the
  aggregator (`run_step3_full_scoring.sh:395-401`); the second overwrites the first with values from
  the same history.
- The output validation floor is still `< 46` (`run_step3_full_scoring.sh:475-478`), already
  recorded in `docs/BACKLOG.md` §15.

---

## 10. Integration Points

### 10.1 Required Files

| File | Purpose |
|------|---------|
| `prng_registry.py` | PRNG implementations |

### 10.2 Imports from prng_registry

```python
from prng_registry import (
    get_cpu_reference,        # CPU PRNG function
    get_pytorch_gpu_reference, # PyTorch GPU function
    has_pytorch_gpu,          # Check GPU support
    list_pytorch_gpu_prngs,   # List GPU PRNGs
    get_kernel_info           # Kernel metadata
)
```

### 10.3 Usage by Other Modules

| Module | Usage |
|--------|-------|
| `reinforcement_engine.py` | Feature extraction |
| `meta_prediction_optimizer.py` | Scoring |
| `coordinator.py` | Batch processing |

> **Corrected 2026-11-11 (D12).** `reinforcement_engine.py` (`:100`, `:395`) and
> `meta_prediction_optimizer.py` (`:34`) do import it. `coordinator.py` does **not** (searched).
> Missing from the table: **`full_scoring_worker.py`**, Step 3's only caller, which imports the
> class after GPU isolation is set (`:559-566`), and **`prediction_generator.py`** (Step 6,
> `:58`, `:263`).

### 10.4 Input Data Formats

**Seeds:**
```python
# Format 1: List of ints
seeds = [123, 456, 789]

# Format 2: List of dicts
seeds = [{'seed': 123, 'match_rate': 0.8}, ...]
```

**Lottery History:**
```python
lottery_history = [123, 456, 789, ...]  # List of draw values
```

**Survivor Metadata:**
```python
survivor_metadata = {
    123: {
        'forward_count': 50,
        'reverse_count': 30,
        'intersection_count': 15,
        'skip_min': 1,
        'skip_max': 10,
        ...
    },
    456: {...}
}
```

> **Corrected 2026-11-11.** In Step 3 the seeds and metadata come from the 22-array NPZ, one dict
> per row with every array as a key (`seeds` renamed `seed`), written into the chunk file
> (`generate_step3_scoring_jobs.py:62-102`, `:226-233`) and read back by the worker as
> `{seed: row}` (`full_scoring_worker.py:176-199`, `:582-584`). `prng_type` and `skip_mode` arrive as
> their encoded integers. The training history is the file at
> `/dev/shm/prng/step3/train_history.json` (`generate_step3_scoring_jobs.py:236`). See §15.

### 10.5 Holdout History Integration

**Files Added/Modified:**

| File | Change |
|------|--------|
| `full_scoring_worker.py` | Added `--holdout-history` CLI, `compute_holdout_hits_batch()` |
| `generate_step3_scoring_jobs.py` | Added `--holdout-history` to job args |
| `run_step3_full_scoring.sh` | Added `HOLDOUT_HISTORY` variable, scp to remotes |

> **Corrected 2026-11-11 (D13).** Worker `--holdout-history` and `compute_holdout_hits_batch` are at
> `full_scoring_worker.py:546-547` and `:274`; the generator's flag at
> `generate_step3_scoring_jobs.py:325`, emitted per job at `:244`; `HOLDOUT_HISTORY` at
> `run_step3_full_scoring.sh:68`. The holdout file reaches remotes by the **ramdisk preload**
> (`:36-40`), not by scp: Phase 2 copies only chunk files (`:201`, `:212`).

**compute_holdout_hits_batch() Function:**
```python
def compute_holdout_hits_batch(
    seeds: List[int],
    holdout_history: List[int],
    train_history_len: int,      # Used to derive offset
    prng_type: str = 'java_lcg',
    mod: int = 1000
) -> Dict[int, float]:
    """
    Compute holdout Hit@K for multiple seeds.
    
    CRITICAL: offset is DERIVED from train_history_len, not configurable.
    """
    offset = train_history_len  # THE LAW
    
    for seed in seeds:
        predictions = prng_func(seed, n_holdout, skip=offset)
        hits = sum(1 for pos, actual in enumerate(holdout_history) 
                   if predictions[pos] % mod == actual)
        results[seed] = hits / n_holdout
    
    return results
```

**Output Format (per survivor):**
```json
{
  "seed": 12345,
  "holdout_hits": 1.0,
  "score": 85.5,
  "features": { ... }
}
```

> **Corrected 2026-11-11 (absorbed into D2).** A record at `a15b524` has seven top-level keys:
> `seed`, `score`, `features` (91), `metadata` (`prng_type`, `mod`, `worker_hostname`, `worker_gpu`,
> `timestamp`), `holdout_hits`, `holdout_quality`, `holdout_features` (77)
> (`full_scoring_worker.py:390-430`; host `survivors_with_scores.json` of 2026-03-11 has exactly these).
> `score` is the Step-3 match rate × 100, not the NPZ `score` array.

---

## 11. Method Reference

### 11.1 SurvivorScorer Methods

| Method | Parameters | Returns | Purpose |
|--------|------------|---------|---------|
| `__init__()` | `prng_type, mod, residue_mods, config_dict` | — | Initialize |
| `_generate_sequence()` | `seed, n, skip` | `np.ndarray` | Generate PRNG sequence |
| `_coerce_seed_list()` | `items` | `List[int]` | Convert mixed to ints |
| `compute_dual_sieve_intersection()` | `forward, reverse` | `Dict` | Intersection analysis |
| `extract_ml_features()` | `seed, lottery_history, ...` | `Dict[str, float]` | Single seed features |
| `extract_ml_features_batch()` | `seeds, lottery_history, ...` | `List[Dict]` | Batch GPU features |
| `batch_score_vectorized()` | `seeds, lottery_history, ...` | `Tensor/List` | Fast vectorized scoring |
| `batch_score()` | `seeds, lottery_history, ...` | `List[Dict]` | Legacy full features |
| `_vectorized_scoring_kernel()` | `seeds_t, hist_t, device` | `Tensor` | Core scoring kernel |
| `_cpu_batch_generate()` | `seeds, n` | `np.ndarray` | CPU fallback generation |
| `_empty_ml_features()` | — | `Dict` | Zero-filled features |

> **Corrected 2026-11-11 (absorbed into D1).** Every row exists (`survivor_scorer.py:90-809`).
> Missing: `compute_battery_features(seq)` → `dict` of 23 Battery Tier 1A columns, computed from the
> PRNG sequence only, never from the draw history (`:188-348`). `_empty_ml_features()` returns 53
> keys (`:469-490`): it omits the 18 metadata names and also `pred_min`, `pred_max` and the four
> residual features, so the empty-history early return (`:351-352`, `:556-557`) has fewer keys than a
> normal record.

### 11.2 Module-Level Functions

| Function | Parameters | Returns | Purpose |
|----------|------------|---------|---------|
| `entropy()` | `p, q, ...` | `float` | Safe CuPy → NumPy entropy |
| `java_lcg_sequence()` | `seed, n, mod` | `np.ndarray` | CPU fallback LCG |

---

## 12. Example Usage

### 12.1 Basic Feature Extraction

```python
from survivor_scorer import SurvivorScorer

# Initialize
scorer = SurvivorScorer(prng_type='java_lcg', mod=1000)

# Single seed
seed = 12345
lottery_history = [123, 456, 789, ...]  # 5000 draws
features = scorer.extract_ml_features(seed, lottery_history)

print(f"Score: {features['score']:.2f}")
print(f"Temporal stability: {features['temporal_stability_mean']:.4f}")
```

### 12.2 Batch Scoring

```python
# Vectorized (fast, scores only)
seeds = list(range(1000000, 1001000))
scores = scorer.batch_score_vectorized(seeds, lottery_history)
print(f"Top score: {scores.max():.4f}")

# Full features (slower)
results = scorer.batch_score(seeds[:100], lottery_history)
for r in results[:5]:
    print(f"Seed {r['seed']}: {r['score']:.2f}")
```

### 12.3 GPU-Batched Features

```python
# Crypto miner style - all parallel
features_list = scorer.extract_ml_features_batch(
    seeds=seeds[:1000],
    lottery_history=lottery_history,
    survivor_metadata=metadata_dict
)

for features in features_list[:3]:
    print(f"Score: {features['score']:.2f}, "
          f"Residue 8: {features['residue_8_match_rate']:.4f}")
```

### 12.4 Dual Sieve Intersection

```python
forward = [123, 456, 789, 101, 102]
reverse = [789, 101, 102, 103, 104]

result = scorer.compute_dual_sieve_intersection(forward, reverse)

print(f"Intersection: {result['intersection']}")  # [789, 101, 102]
print(f"Jaccard: {result['jaccard']:.4f}")        # 0.375
```

> **Corrected 2026-11-11 (D14).** The intersection is returned sorted (`survivor_scorer.py:172`), so
> it prints `[101, 102, 789]`; the union has 7 seeds, so Jaccard is 3 / 7 = `0.4286`, not 0.375.

### 12.5 With Optuna Config

```python
# Optuna passes individual params
config_dict = {
    'residue_mod_1': 14,
    'residue_mod_2': 137,
    'residue_mod_3': 1136,
    'max_offset': 3,
    'temporal_window_size': 50
}

scorer = SurvivorScorer(
    prng_type='java_lcg',
    mod=1000,
    config_dict=config_dict
)

# residue_mods is now [14, 137, 1136]
```

---

## 13. Performance Notes

### 13.1 Memory Management

```python
# After batch operations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 13.2 RX 6600 Specific

- VRAM limited to 80% (6.4GB)
- PYTORCH_HIP_ALLOC_CONF for fragmentation
- Two-step tensor transfer for PCIe stability
- cudnn.benchmark disabled

> **Corrected 2026-11-11 (absorbed into B1).** On the live rigs (`rrig6600*`) only the two-step
> transfer is in effect. The 80 % cap, `PYTORCH_HIP_ALLOC_CONF` and `cudnn.benchmark = False` sit
> behind the dead `rig-6600*` guard (`survivor_scorer.py:22`, `:61`). See §2.1.

### 13.3 Batch vs Single

| Method | Speed | Features | Use Case |
|--------|-------|----------|----------|
| `batch_score_vectorized()` | Fastest | Score only | Initial filtering |
| `extract_ml_features_batch()` | Fast | All 46 | GPU training |
| `extract_ml_features()` | Slow | All 46 | Single seed analysis |
| `batch_score()` | Slowest | All 46 | Legacy compatibility |

> **Corrected 2026-11-11 (absorbed into D1).** "All 46" → 77 per-seed keys from each of the three
> feature methods (Step 3 adds the 14 `global_*`). In Step 3, `extract_ml_features_batch()` is the
> main path, `extract_ml_features()` serves the holdout features and the sequential fallback, and
> neither `batch_score_vectorized()` nor `batch_score()` is called.

---

## 14. Chapter Summary

**Chapter 6: Survivor Scorer** covers the feature extraction pipeline:

| Component | Lines | Purpose |
|-----------|-------|---------|
| Environment setup | ~30 | ROCm/VRAM configuration |
| SurvivorScorer class | ~100 | Core scorer |
| `extract_ml_features()` | ~100 | Single seed features |
| `extract_ml_features_batch()` | ~150 | GPU batch features |
| Batch scoring | ~50 | Vectorized scoring |
| Helpers | ~30 | Utilities |

**Key Achievements:**
- 46 ML features per seed
- GPU-accelerated batch processing
- ROCm compatibility for AMD GPUs
- Memory management for 8GB GPUs
- Single transfer optimization

> **Corrected 2026-11-11 (D15; already observed in `docs/PIPELINE_BEHAVIOUR_MODEL.md` §15.1 item 4).**
> This is **Chapter 4**, not Chapter 6. "46 ML features per seed" → 77 per-seed + 14 global = 91
> (§9.9). The 8 GB memory limit is not applied on the live rigs (B1). There are two device-to-host
> transfers, not one (D11).

---

## 15. RANGE-MINER, the 22-array NPZ contract and the window anchor — what changed for Step 3 (added 2026-11-11)

The 2.0.0 chapter predates S172. Step 3's code did not change for RANGE-MINER; what changed is the
artifact it reads.

1. **The input bundle is RANGE-MINER output.** On VM 101 `bidirectional_survivors_binary.npz` is a
   symlink to `.s172_accumulator/current/bidirectional_survivors_binary.npz`, carrying exactly the
   22 frozen arrays in contract order (live `np.load(...).files`; contract
   `convert_survivors_to_binary.py:51-74`). **Point-in-time:** that bundle has 0 rows
   (`provenance.json` `final_row_count: 0`, generation `gen-20260818T013219550933Z-step1_java_lcg_0`),
   and Step 3 has not run since 2026-03-11, so **no Step-3 output from RANGE-MINER survivors exists
   yet.**
2. **What Step 3 takes from the 22 arrays.** Every array becomes a field of each chunk row
   (`generate_step3_scoring_jobs.py:76-93`). Of those, **13** become features through the merge list
   (`survivor_scorer.py:774-779`), `seeds` drives all 59 computed features, and `skip_min` sets the
   burn-in for `holdout_features` (`holdout_quality.py:47`). Carried but not used as features:
   `forward_matches`, `reverse_matches` (S2), `window_size`, `offset`, `trial_number`, `score`,
   `skip_mode`, `prng_type` (Chapter 14 audit Q4). The 5 dead metadata names are not among the 22
   (B7). None of this is caused by, or fixable by, RANGE-MINER (system map §7.3).
3. **The window anchor (Brief I).** NPZ array 4 keeps the wire name `offset` but after Brief I it
   **is** the window anchor and never the generator phase (`utils/canonical_records.py:216-225`,
   `:368-374`). Step 3 neither merges it nor uses it: every sequence is regenerated from PRNG
   position 0 with `skip=0` and compared with the whole training history
   (`survivor_scorer.py:574-575`). So Brief I changes nothing Step 3 computes, and the §1.4 "offset is
   a law" is a **different quantity** (holdout start = `len(train_history)`).
4. **Per-row family is not used.** Each row carries its encoded `prng_type`, but the worker
   regenerates every seed with the single `--prng-type` of the job (`full_scoring_worker.py:530-531`,
   `:344`), which the generator takes from `optimal_scorer_config.json` or its default `java_lcg`
   (`generate_step3_scoring_jobs.py:206`, `:309`). The live config has no `prng_type` key, so every
   row — including any the sieve labelled `java_lcg_hybrid` — is regenerated as `java_lcg`. Recorded
   as an observation, not labelled: this chapter never claimed per-row families, and the hybrid
   contact point is already recorded (`docs/CHAPTER_3_ALIGNMENT_AUDIT.md` F8).
5. **The fleet.** Step 3 dispatches to `distributed_config.json`'s hostnames
   (`run_step3_full_scoring.sh:74-81`; `ramdisk_preload.sh:26-40`; `scripts_coordinator.py:430-448`),
   not to the Resolved Execution Set that governs RANGE-MINER's Step 1 — committed Chapter 9 **O-6**.

---

## 16. Known gaps at `a15b524` (added 2026-11-11)

Full evidence for every item is in `docs/CHAPTER_4_AUDIT.md` §4–§6.

### 16.1 BROKEN

| # | Gap | Stops at | A repair must |
|---|---|---|---|
| B1 | ROCm prelude and 80 % VRAM cap keyed to `rig-6600*`; live rigs are `rrig6600*` — never fires *(class governed: Ch2 F-5, Ch9 B-1; these four sites not named there)* | `survivor_scorer.py:22`, `:61`; `full_scoring_worker.py:90`, `:99` | follow Chapter 2 §9.5: decide if the prelude is still supported; if kept, key it from the rig profile; never rename hosts |
| B2 | `best_offset` always 0; `max_offset` never read; no alignment search *(not in governance)* | `survivor_scorer.py:373`, `:707`, `:108` | implement the search, or classify `best_offset` as a dead column under P0-A |
| B3 | `batch_score_vectorized` CPU fallback: `UnboundLocalError` with no PyTorch family; raw outputs not mod-reduced *(not in governance; no pipeline caller)* | `survivor_scorer.py:505`, `:510-511` | initialise `predictions`; reduce by `self.mod` |
| B4 | Batch-feature CPU fallback not mod-reduced; silent, keeps 91 keys; hit by 42 of 44 families and any GPU exception *(not in governance)* | `survivor_scorer.py:808` | reduce by `self.mod`; cover the fallback in F-PARITY |
| B5 | Step 2's `optimal_scorer_config.json` scorer parameters never reach Step 3 *(not in governance)* | `full_scoring_worker.py:344`; `generate_step3_scoring_jobs.py:201-209` | a governed decision first; if applied, carry the parameters to the worker, version the feature schema, and key `holdout_quality` from the configured mods |
| B6 | Sequential fallback merges 6 of 18 metadata fields *(governed: system map §7.3 defect 2; P0-B authorized; no F-PARITY in code)* | `full_scoring_worker.py:451-456` | what P0-B authorizes: identical merges, exact field validation, F-PARITY, fail closed |
| B7 | Five metadata names have no producer *(governed as fact: system map §7.1; P0-A)* | `utils/canonical_records.py:115-123` | classify them dead in the canonical manifest, or give them a producer under a new schema version |

### 16.2 SWITCHED-OFF-BY-RULING

| # | What is held back | Held by | Lifting it would enable |
|---|---|---|---|
| S1 | Repair of the Step-3 CPU skip semantics (`java_lcg_cpu` applies skip once; the kernel between draws), so "holdout is the continuation" holds only at skip 0 | `docs/BACKLOG.md` §8.1 — Beta's read-only reachability audit, "No fix is authorised"; `docs/TODO_CONSOLIDATED.md:65`, `:164`. No standalone `TB_RULING_*` file found | `holdout_hits` and `holdout_features` computed at the PRNG positions the sieve actually tested, for skip > 0 survivors |
| S2 | `forward_matches` / `reverse_matches` as Step-3 features | TB §8 ruling, `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:165-171`, `:276` | Option A: the two per-seed sieve match rates in the training vector under a new schema version, after leakage and redundancy analysis and full retraining; or Option B: a recorded reason they stay out |

### 16.3 DESCRIPTION-ONLY

D1 feature count 46/47 → 91/89 (§1.1, §9.9) · D2 `holdout_hits` is no longer the y-label (§1.4,
§9.7) · D3 ramdisk preload covers every node (§2.5) · D4 stale node list in §2.5 · D5 missing file
is a load error, not argparse (§2.5) · D6 8 workers per rig, not 12 (§1.5, §2.2) · D7 the mod-1000
lane is not orthogonal (§3.2) · D8 `java_lcg_sequence` is dead code (§3.3) · D9 survivor-list
parameters ignored (§5.1) · D11 two transfers, not one (§8.2) · D12 module-usage table (§10.3) ·
D13 holdout reaches remotes by ramdisk (§10.5) · D14 example output (§12.4) · D15 self-label and
next-chapter pointer (§14) · D16 RX 6600 XT (§1.3).

---

## 17. Open items (added 2026-11-11)

1. **B5 needs a decision before any code.** Whether Step 2's tuned residue mods and temporal
   parameters should drive Step 3 is a design question with schema consequences (91/89 names,
   `holdout_quality`). For Michael, with Beta as reviewer.
2. **U1, U2 stay UNVERIFIED** — chunk memory figures and the two-step-transfer rationale need a
   measured record on a rig.
3. **First Step 3 on RANGE-MINER output** has not happened (§15.1); every count above is from code
   and from the 2026-03-11 artifact.
4. **Inherited, owned elsewhere:** P0-A / P0-B (system map §13); Ch13 B2 (dataset authority for
   Step 3's histories); Ch9 O-6 (fleet authority); BACKLOG §8.1 (S1), §15 (validation floor),
   §16 (26-GPU manifest note); behaviour model D7 (manifest actions).
5. **Not read:** `step3_debug.log`, `step3_diagnostic.log`, `fix_scoring_worker_metadata.py`,
   `results_archive/`, the `apply_s*` patch corpus. A "not in governance" label was not checked
   against the patch corpus.

---

## 18. Verification declaration (VIR-1…6) (added 2026-11-11)

- **Execution proof.** Files opened are listed in `docs/CHAPTER_4_AUDIT.md` §8. Every anchor in this
  chapter was opened in this session; `agents/watcher_agent.py` anchors were taken from
  `git show a15b524:agents/watcher_agent.py`, because another task had uncommitted edits in that
  file's working copy during this pass.
- **Clean control.** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was read, not edited.
- **Fault-injection control.** Not applicable to a documentation pass; none was invented.
- **Completion sentinel.** §19.4.
- **Unavailable-observer behaviour.** Rig state (ramdisk contents, live hostnames, GPU SKU) was not
  observed — no rig access by instruction — and is reported **UNAVAILABLE**, never clean. The rig
  hostnames used for B1 come from the committed `rig_profiles_config.json`.
- **Audit claim scope.** What this chapter says against what the code at `a15b524` does — not whether
  the design is right.
- **Searched / unavailable surfaces.** Tracked repo; gitignored files (`git status --ignored`,
  `git ls-files -o -i --exclude-standard`, `agent_manifests/definitions.json`); git history
  (`git log -S`, `git show 0101306`); host (`systemctl list-units`, `crontab -l`, Step-3 artifacts).
  Unavailable: the rigs.
- **Governance trail searched.** All 30 `docs/TB_RULING_*` (listed), system map §7 and §13–§15,
  `BACKLOG.md`, `TODO_CONSOLIDATED.md`, `S111_IMPLEMENTATION_PLAN_FINAL.md`,
  `KNOWN_ANSWER_VALIDATION_INVENTORY.md`, `PIPELINE_BEHAVIOUR_MODEL.md`; greps across `docs/` listed
  in the audit.
- **Chapters searched.** 2, 3 (alignment audit), 9 (+ audit), 13 audit, 14 audit.

---

## 19. Closure statement (added 2026-11-11)

### 19.1 Verified against

Code at `a15b524` (identical at `741822b`), VM 101, 2026-11-11.

### 19.2 What this pass did

Audited 83 claims (42 accurate, 7 BROKEN, 2 SWITCHED-OFF-BY-RULING, 15 DESCRIPTION-ONLY,
2 UNVERIFIED, 15 absorbed; 42 + 7 + 2 + 15 + 2 + 15 = 83). Added the framing block, status header,
§0, §9.9, §15–§19 and inline correction notes. Marked §1.4 (y-label), §2.5 (Automatic Preload), §9.7
and §9.8 SUPERSEDED, with pointers.

### 19.3 What was removed

Nothing. Every original heading is still present. Three header lines (**Version**, **File**,
**Lines**) were amended in place: their original wording is kept and an annotation appended. No other
line of the 2.0.0 text was changed or deleted. The chapter grew from 1037 to about 1566 lines; it did
not shrink.

### 19.4 Completion sentinel

**PASS** — every substantive claim was checked and placed; the counts reconcile; the two UNVERIFIED
items and the unavailable rig surfaces are stated, not hidden.

---

## Next Chapter

**Chapter 7: GPU Optimizer** will cover:
- `gpu_optimizer.py` — GPU binding and distribution
- Platform detection (CUDA/ROCm)
- Memory management
- Job allocation

> **Corrected 2026-11-11 (absorbed into D15).** The next chapter is
> `docs/CHAPTER_5_ML_ARCHITECTURE_OPTIMIZER_v2.md` (executable Step 4). The live Chapter 7 is the
> prediction generator (`docs/CHAPTER_7_PREDICTION_GENERATOR.md`), not a GPU optimizer.

---

*End of Chapter 6: Survivor Scorer*

*Corrected 2026-11-11: end of **Chapter 4**: Full Scoring & Feature Extraction (Step 3).*

# Chapter 3: Scorer Meta-Optimizer (Step 2.5)

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

## PRNG Analysis Pipeline — Complete Operating Guide

**Chapter version:** 4.3-doc — corrected 2026-11-11 (was 4.2)
**Status:** **CORRECTED at `a15b524`, 2026-11-11** — audited against the code, corrected in place,
history preserved. Not closed: see §20 for the closure statement and sentinel, §18 for open items.
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1).
Audit of record: `docs/CHAPTER_3_ALIGNMENT_AUDIT.md` (original audit at `575378e`, plus the
"Addendum — live re-verification, 2026-11-11").
**Base:** `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`), VM 101, working tree. HEAD at the time
of this pass is `741822b`, which differs from `a15b524` only under `docs/`, so the code is identical.
**Numbering scheme:** **EXECUTABLE** (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). Bare "Step N" in text
added on 2026-11-11 is the step WATCHER runs: `STEP_SCRIPTS[2] = "run_scorer_meta_optimizer.sh"`,
`STEP_MANIFESTS[2] = "scorer_meta.json"`, `STEP_NAMES[2] = "Scorer Meta-Optimizer"`
(`agents/watcher_agent.py:390`, `:401`, `:412`). **This chapter documents executable Step 2.** The
bidirectional sieve (Chapter 2) runs **inside executable Step 1**. "Step 2.5" in the title and the
original text is the older conceptual label for the same stage; it survives in
`agent_manifests/scorer_meta.json:3` (`description`, while `:4` says `"pipeline_step": 2`) and as a
batching label in `scripts_coordinator.py:141-160`. Chapter numbers are not step numbers.

**Original header (retained):**

**Version:** 4.2  
**File:** `scorer_trial_worker.py`  
**Lines:** ~640  
**Purpose:** Execute single scorer meta-optimization trial on remote GPU

> **Corrected 2026-11-11 — header.** The objective that runs is **v4.3** (enrichment over
> `skip_mode`, `scorer_trial_worker.py:413-444`, "TB ruling S107"); `docs/S107_session_log.md` calls
> the deployed worker **v4.4** (an orphaned `bc_stat` reference removed, objective unchanged). The
> file's own docstrings still say v4.2 (`:3`, `:298`, `:311-322`). The file is **641** lines. "On remote
> GPU": the worker **binds** a GPU (`:75-98`) but computes on the CPU with NumPy (`run_trial`,
> `:286-466`); its only torch/CuPy use is best-effort cleanup (`:125-146`). DESCRIPTION-ONLY.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Position](#2-architecture-position)
3. [Pull Architecture](#3-pull-architecture)
4. [Version History](#4-version-history)
5. [Execution Flow](#5-execution-flow)
6. [Trial Parameters](#6-trial-parameters)
7. [Training/Holdout Split](#7-trainingholdout-split)
8. [GPU-Vectorized Scoring](#8-gpu-vectorized-scoring)
9. [Adaptive Memory Batching](#9-adaptive-memory-batching)
10. [Result Serialization](#10-result-serialization)
11. [CLI Interface](#11-cli-interface)
12. [Integration with Coordinator](#12-integration-with-coordinator)
13. [Troubleshooting](#13-troubleshooting)
14. [Complete Method Reference](#14-complete-method-reference)

Added 2026-11-11:

0. [How this chapter was produced](#0-how-this-chapter-was-produced) (below)
16. [What RANGE-MINER, the 22-array NPZ and Brief I changed for Step 2](#16-what-range-miner-the-22-array-npz-and-brief-i-changed-for-step-2)
17. [Known gaps at a15b524](#17-known-gaps-at-a15b524)
18. [Open items](#18-open-items)
19. [Verification declaration (VIR-1…6)](#19-verification-declaration-vir-16)
20. [Closure statement](#20-closure-statement)

> The chapter carries two pre-existing sections numbered **4.2**, a §15 that names itself "Chapter 13",
> and a §9.4 appended after the end marker. Their numbers and positions are kept so earlier references
> still resolve; the new sections start at 16 and follow §9.4.

---

## 0. How this chapter was produced

**Added 2026-11-11.** The 2026-02-23 text (v4.2, 958 lines, last touched `05b0e6b`) was audited claim by
claim in `docs/CHAPTER_3_ALIGNMENT_AUDIT.md` at `575378e`. This pass re-verified that audit against the
code at `a15b524` and the live VM 101 host and appended an addendum to it; it did not redo it. Method:

1. Governance trail first: `docs/TB_RULING_*` (including `TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md` and
   `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`), `docs/PROPOSAL_*` (including
   `PROPOSAL_NPZ_Auto_Conversion_Step2.md`), `docs/TEAM_ALPHA_*` (including
   `TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` and `TEAM_ALPHA_D3_0_B_AND_ITEM1_NOTICE.md`), `docs/S107_session_log.md`,
   then `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/PIPELINE_BEHAVIOUR_MODEL.md`, the fact file, then the
   committed corrected Chapter 12 (§13, §14), then code.
2. Code read this session: `scorer_trial_worker.py`, `generate_scorer_jobs.py` and
   `run_scorer_meta_optimizer.sh` in full; `ramdisk_preload.sh`, `convert_survivors_to_binary.py`,
   `coordinator.py` (`collect_scorer_results` and its two helpers), `scripts_coordinator.py`,
   `agents/watcher_agent.py` (step maps, arg building, run loop, argparse), `agent_manifests/scorer_meta.json`,
   the gitignored `agent_manifests/definitions.json`, `preflight_check.py`, `utils/canonical_arrays.py`,
   `utils/canonical_records.py`, `utils/run_finalizer.py`, `utils/survivor_loader.py`, `utils/prng_encoding.py`,
   `window_optimizer.py`, `window_optimizer_integration_final.py`, `survivor_scorer.py`,
   `generate_anti_overfit_jobs.py`. The full list is in the audit addendum §A.9.
3. Host state, read-only: `ls`/`stat` of the Step-2 inputs and outputs, `test -nt`, JSON parses of
   `scorer_jobs.json`, `optimal_scorer_config.json`, the history files and both node configs, `systemctl`,
   `crontab -l`, `ps`.
4. **Nothing was executed.** The scorer, Optuna, the converter, WATCHER and every pipeline step were not run.
   "Works" below means the code path is complete on reading.

**How to read the corrections.** Original text is kept. Each disagreement is followed by a
`Corrected 2026-11-11` note naming what the code does and its label (BROKEN / SWITCHED-OFF-BY-RULING /
DESCRIPTION-ONLY); superseded sections carry **SUPERSEDED (2026-11-11)** and a pointer. All labels are
collected in §17. The labelling rule (when a stale statement is DESCRIPTION-ONLY rather than BROKEN) is in
the audit addendum §A.2: where the code is the deliberate later state — a ruled objective change, a removed
GPU path, changed ranges — the label is DESCRIPTION-ONLY.

**What changed underneath this chapter since v4.2.**
- **The objective changed twice.** v4.3 replaced the v4.2 `bidirectional_count` percentile objective with
  enrichment over `skip_mode` (`scorer_trial_worker.py:413-444`, "TB ruling S107"); v4.4 removed an
  orphaned reference (`docs/S107_session_log.md`). The change landed in `ca975f8`, a commit whose subject is
  about moving docs, the commit directly after this chapter's own v4.2 update (`ca975f8^` = `05b0e6b`).
- **RANGE-MINER** became the certifying Step-1 engine. Step 2 consumes its 22-array NPZ unchanged — and its
  legacy auto-conversion block is held by an owner/Beta record (§16, §17 S1).
- **The window-anchor change** (Brief I) does not reach Step 2 (§16.4). Brief I is implemented and committed;
  final acceptance is pending a completed production-shape run
  (`docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md`).

---

## 1. Overview

### 1.1 What the Scorer Trial Worker Does

The scorer trial worker executes a **single Optuna trial** for scorer hyperparameter optimization:

- **Receives trial parameters** from coordinator via JSON file or CLI args
- **Scores survivors** using specified residue/temporal parameters
- **Evaluates on holdout set** to measure generalization
- **Writes result JSON** locally for coordinator to pull

> **Corrected 2026-11-11 — §1.1.** At `a15b524` the worker (1) receives its parameters as **inline JSON at
> argv position 5** (`generate_scorer_jobs.py:130` → `scorer_trial_worker.py:560-563`); the `--params-file`
> branch (`:570-589`) works but nothing generates it, and there are no per-parameter CLI flags;
> (2) builds a **k-of-3 residue mask** from `residue_mod_1/2/3` and `max_offset` (`:379-389`) — it does not
> "score" survivors, and the temporal parameters are not used here (they are consumed by Step 3's
> `SurvivorScorer`, `survivor_scorer.py:97-111`); (3) **does not evaluate on a holdout set** — the history
> files are accepted and ignored (`:527-531`, `:600-603`), removed at v4.0 (`:5-10`); (4) writes its result
> locally — accurate (`save_local_result`, `:495-518`). DESCRIPTION-ONLY.
>
> **Corrected (post-review, 2026-11-11) — item (2) above.** "Consumed by Step 3's `SurvivorScorer`" is true of
> the class constructor only (`survivor_scorer.py:97-111`), not of the pipeline. The Step-3 call site builds
> the scorer with no config: `scorer = scorer_class(prng_type=prng_type, mod=mod)` (`full_scoring_worker.py:344`),
> and the job generator reads only `prng_type` and `mod` from `optimal_scorer_config.json`
> (`generate_step3_scoring_jobs.py:201-209`). So the temporal parameters, like the residue ones, reach no step.
> Recorded as **Chapter 4 B5** (`docs/CHAPTER_4_AUDIT.md`); cross-referenced here, not counted as a Chapter 3 break.

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Single Trial Execution** | One GPU, one parameter configuration |
| **Pull Architecture** | Results written locally, coordinator pulls via SCP |
| **GPU-Vectorized Scoring** | PyTorch/CuPy batch processing |
| **Adaptive Memory Batching** | Adjusts batch size for 8GB VRAM |
| **Holdout Evaluation** | v3.4 critical fix for proper validation |
| **NPZ-Based Objective** | v4.2: bidirectional_count from NPZ (no draw history) |
| **Subset Selection** | v4.2: Per-trial survivor subsets with unique seeds |

> **Corrected 2026-11-11 — §1.2, row by row.** *Single Trial Execution* — accurate. *Pull Architecture* —
> accurate, but the remote pull is **SFTP** over a pooled SSH connection (`coordinator.py`
> `_pull_remote_scorer_results`, `:2620`: `ssh.open_sftp()`, `sftp.get`, `sftp.remove`), not `scp`.
> *GPU-Vectorized Scoring* and *Adaptive Memory Batching* — **removed at v4.0**; `run_trial` is pure NumPy
> (§8, §9 SUPERSEDED). *Holdout Evaluation* — removed at v4.0 (`scorer_trial_worker.py:5-10`, `:528`).
> *NPZ-Based Objective* — still NPZ-only, but the signal is **`skip_mode` enrichment (v4.3)**, not
> `bidirectional_count` (§7.2). *Subset Selection* — accurate: per-trial sampling seeded by
> `optuna_trial_number` (`:353-367`), then the residue mask selects the subset. DESCRIPTION-ONLY.

### 1.3 Pipeline Position

```
Step 2.5: Scorer Meta-Optimizer
    │
    ├── run_scorer_meta_optimizer.py (orchestrator)
    │       │
    │       ├── generate_scorer_jobs.py (creates trial jobs)
    │       │
    │       └── coordinator.py (distributes to 26 GPUs)
    │               │
    │               └── scorer_trial_worker.py ◄── THIS FILE
    │                       │
    │                       └── Writes: scorer_trial_results/trial_*.json
    │
    └── coordinator.collect_scorer_results() (pulls results)
```

> **SUPERSEDED (2026-11-11) — the diagram above.** `run_scorer_meta_optimizer.py` is **deprecated** and
> called by nothing (`run_scorer_meta_optimizer.py:1-8`, *"DEPRECATED - January 9, 2026 … Use
> run_scorer_meta_optimizer.sh instead"*), and dispatch is `scripts_coordinator.py`, not `coordinator.py`
> (this chapter's own §12.1). As built at `a15b524` (DESCRIPTION-ONLY):
>
> ```
> Executable Step 2: Scorer Meta-Optimizer   (STEP_SCRIPTS[2], agents/watcher_agent.py:390)
>     │
>     └── run_scorer_meta_optimizer.sh
>             ├── preload_ramdisk  NPZ + train/holdout → /dev/shm/prng/step2 on every node   (:18-21)
>             ├── NPZ auto-conversion block  (:140-150)  ── HELD, see §12.5 and §17 S1
>             ├── validate the three inputs exist  (:213-218)
>             ├── generate_scorer_jobs.py  → scorer_jobs.json (Optuna ask × TRIALS)  (:239-246)
>             ├── scp worker code to three hardcoded rig IPs  (:257-261)
>             ├── scripts_coordinator.py --jobs-file scorer_jobs.json  (:266-269)
>             │       └── per job: distributed_worker.py job_<id>.json --gpu-id N  (scripts_coordinator.py:991-997)
>             │               └── scorer_trial_worker.py  ◄── THIS FILE
>             │                       └── writes ~/distributed_prng_analysis/scorer_trial_results/trial_NNNN.json
>             └── coordinator.MultiGPUCoordinator('ml_coordinator_config.json')
>                     .collect_scorer_results(1)  → study.tell(...) → optimal_scorer_config.json  (:278-310)
> ```

---

## 2. Architecture Position

### 2.1 In the Optimization Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SCORER META-OPTIMIZER (Step 2.5)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Optuna Study (Zeus)                                                     │
│       │                                                                  │
│       ├── Trial 0 ──► GPU 0 ──► scorer_trial_worker.py                  │
│       ├── Trial 1 ──► GPU 1 ──► scorer_trial_worker.py                  │
│       ├── Trial 2 ──► GPU 2 ──► scorer_trial_worker.py                  │
│       │   ...                                                            │
│       └── Trial 99 ──► GPU 25 ──► scorer_trial_worker.py                │
│                                                                          │
│  Each worker:                                                            │
│   1. Receives params (residue_mod_1/2/3, max_offset, temporal_window)   │
│   2. Scores survivors with those params                                  │
│   3. Evaluates on holdout data                                          │
│   4. Writes result JSON locally                                         │
│                                                                          │
│  Coordinator pulls all results → Optuna aggregates → Best params        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

> **Corrected 2026-11-11 — §2.1.** The study is created and **pre-sampled on VM 101** before dispatch:
> `generate_scorer_jobs.py` calls `study.ask()` once per trial and writes the parameters into each job
> (`:116-119`); workers never touch Optuna (`scorer_trial_worker.py:48-51`, `:605`). Trials are batched per
> node, at most **6 jobs per node per batch** for this step (`scripts_coordinator.py:105`, `:177-179`), so
> "Trial N → GPU N" is not the mapping. Step 2 of the worker list ("scores survivors") is the residue mask
> (§1.1 note); step 3 ("evaluates on holdout data") does not happen. The GPU count depends on the file:
> dispatch reads `distributed_config.json` (1 + 8 + 8 + 8 = **25**, `distributed_config.json:6`), collection
> reads `ml_coordinator_config.json` (2 + 8 + 8 + 8 = **26**, `ml_coordinator_config.json:6`). DESCRIPTION-ONLY.

### 2.2 Why Separate Worker?

| Approach | Problem |
|----------|---------|
| Direct Optuna workers | Database contention with 26 concurrent connections |
| Shared storage | NFS/Samba bottleneck, file locking issues |
| **Pull architecture** | ✅ Workers write locally, coordinator pulls via SCP |

---

## 3. Pull Architecture

### 3.1 How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                         PULL ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  STEP 1: Coordinator dispatches trial                            │
│          Zeus ───SSH───► rig-6600 (GPU 5)                        │
│                          scorer_trial_worker.py --trial-id 42    │
│                                                                   │
│  STEP 2: Worker executes and writes locally                      │
│          rig-6600:/home/michael/distributed_prng_analysis/       │
│                   scorer_trial_results/trial_42.json             │
│                                                                   │
│  STEP 3: Coordinator pulls result via SCP                        │
│          Zeus ───SCP───► rig-6600                                │
│          GET scorer_trial_results/trial_42.json                  │
│          DELETE remote file after successful pull                │
│                                                                   │
│  STEP 4: Coordinator aggregates all results                      │
│          Optuna study updated with trial results                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

> **Corrected 2026-11-11 — §3.1.** The architecture is accurate as drawn, with four details. (1) There is
> no `--trial-id` flag; the trial id is positional `argv[4]` (`scorer_trial_worker.py:531`). (2) Each job
> is shipped as a JSON file and run through `distributed_worker.py … --gpu-id N`
> (`scripts_coordinator.py:991-997`). (3) The pull is SFTP (`coordinator.py` `_pull_remote_scorer_results`,
> `:2620`), and the remote file is deleted after a successful pull (`sftp.remove`); local results are read and
> deleted too (`_read_local_scorer_results`, `os.remove` at `:2614`). (4) The file is `trial_0042.json`
> (§3.2 note). DESCRIPTION-ONLY.

### 3.2 Result File Location

```
~/distributed_prng_analysis/
    scorer_trial_results/
        trial_0.json
        trial_1.json
        trial_2.json
        ...
        trial_99.json
```

> **Corrected 2026-11-11 — §3.2.** File names are 4-digit zero-padded: `trial_{trial_id:04d}.json`
> (`scorer_trial_worker.py:501`; `generate_scorer_jobs.py:134`), i.e. `trial_0000.json … trial_0099.json`.
> The directory is accurate (`:498`). DESCRIPTION-ONLY.

### 3.3 Why Pull Instead of Push?

| Push (worker → coordinator) | Pull (coordinator → worker) |
|----------------------------|----------------------------|
| Workers need coordinator address | Workers only need local filesystem |
| Network errors lose results | Results persist until pulled |
| Complex retry logic | Simple: pull, verify, delete |
| Coordinator must be listening | Coordinator polls when ready |

---

## 4. Version History

### 4.1 Version Timeline

```
Version 4.2 - February 2026 (CURRENT — S107/S108)
├── bidirectional_count as primary objective signal
├── bidirectional_selectivity dropped (98.8% at floor — unusable)
├── Secondary bonus: intersection_ratio (weight=0.10)
├── Global percentile rank objective (stable across trials)
├── Median for robustness (TB Q2 ruling)
├── ir_disabled guard added
└── 640 lines total

Version 4.1 - February 2026 (S107 draft, superseded)
├── Weighted Survivor Index (WSI) objective proposed
├── Identified as tautological (batch_score measures same as sieve)
└── Replaced by bidirectional_count in v4.2

Version 4.0 - February 2026 (S105/S106)
├── WSI + IQR composite objective designed
├── IQR component confirmed tautological by Team Beta
└── Blocked pending TB ruling → led to v4.1/v4.2

Version 3.6 - February 2026 (S101)
├── neg-MSE → Spearman correlation (rank-based, more robust)
├── random.seed(42) → per-trial unique seed
└── Fixed all-identical survivor sampling across trials

Version 3.5 - January 2026 (S101)
├── Degenerate guard: std < 1e-12 early exit
└── Draw-history dependency still present (architectural issue)

Version 3.4 - December 2025
├── CRITICAL FIX: Holdout evaluation uses sampled seeds
├── Previous: Holdout evaluated on ALL seeds (wrong!)
└── Now: Holdout evaluated on SAME sampled seeds as training

Version 3.3 - November 2025
├── GPU-vectorized scoring via SurvivorScorer.extract_ml_features_batch()
├── Adaptive memory batching for 8GB VRAM
└── 3.8x performance improvement

Version 3.2 - November 2025
├── --params-file support for shorter SSH commands
└── Params written to JSON file, worker reads file

Version 3.1 - October 2025
├── ROCm environment setup at file top
└── Mining mode support for AMD rigs

Version 3.0 - October 2025
├── Initial distributed implementation
└── Basic trial execution
```

> **Corrected 2026-11-11 — §4.1.** Two entries are missing and two are mis-attributed. **Missing:**
> **v4.3** (S107, 2026-02-22/23) — the objective became enrichment over `skip_mode == 1`, plus coverage over
> `trial_number`, minus a size penalty, because the v4.2 `bc_score` was *"structurally dead: 79.2% of pool at
> bc>=11300"* (`scorer_trial_worker.py:413-444`, "TB ruling S107"); landed in `ca975f8`. **v4.4** (S107) — the
> orphaned `bc_stat` reference removed from `_log_trial_metrics`; first clean 100/100 run
> (`docs/S107_session_log.md`). The v4.2 line "640 lines total" now reads 641. **Mis-attributed:** the
> Spearman and per-trial-seed fixes are **v3.5** (2026-02-20) in the file's own changelog
> (`scorer_trial_worker.py:15-23`); **v3.6** is *"BUG FIX: NPZ branch never read prng_type from config"*
> (`:12-13`). The v3.3 "3.8x" figure is **UNVERIFIED** (no artifact; the code it measured is gone). The ruled
> lineage v4.0 → v4.4 is recorded in `docs/TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md`,
> `docs/TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md` and `docs/S107_session_log.md`. DESCRIPTION-ONLY.

### 4.2 v4.2 Architectural Redesign (S103-S108)

**The Problem (v3.4-v3.6):** Step 2 scored survivors by comparing PRNG outputs
to draw history — duplicating Chapter 13's function. Under mod1000, all scores
were 0.0 (degenerate). Spearman correlation failed on zero-variance input.

**The Fix (v4.2):** Step 2 now uses NPZ intrinsic fields as ground truth:
- `bidirectional_count`: How many seeds survived both sieves (primary signal)
- `intersection_ratio`: Jaccard overlap quality (secondary, weight=0.10)
- NO draw history dependency. The sieve itself is the ground truth.

**Clean Separation Restored:**
```
Step 2: Find optimal SCORING PARAMETERS using SIEVE QUALITY as ground truth
Chapter 13: Compare predictions to real draws (post-prediction validation)
```

> **Corrected 2026-11-11 — §4.2 (v4.2 redesign).** The separation principle stands: Step 2 reads no draw
> history (`scorer_trial_worker.py:151-157`, `:527-531`). The *signal* in "The Fix (v4.2)" was replaced at
> v4.3: neither `bidirectional_count` nor `intersection_ratio` enters the live objective — both are still
> loaded (`:198-222`) and sliced (`:360-361`, `:371-372`) and then unused. The live signal is `skip_mode`
> enrichment (§7.2 note). DESCRIPTION-ONLY.

### 4.2 v3.4 Critical Fix Details

**The Bug (v3.3 and earlier):**
```python
# WRONG: Training used sampled seeds, holdout used ALL seeds
train_score = score_survivors(sampled_seeds, train_history)
holdout_score = score_survivors(ALL_survivors, holdout_history)  # BUG!
```

**The Fix (v3.4):**
```python
# CORRECT: Both use the SAME sampled seeds
sampled_seeds = random.sample(survivors, sample_size)
train_score = score_survivors(sampled_seeds, train_history)
holdout_score = score_survivors(sampled_seeds, holdout_history)  # FIXED!
```

**Why It Matters:**
- Holdout evaluation must measure how well the model generalizes
- Using different seeds biases the holdout metric
- v3.4 ensures apples-to-apples comparison

> **Note 2026-11-11 — second §4.2.** This is history. Its record in the code is accurate
> (`scorer_trial_worker.py:29-33`); v4.0 removed holdout evaluation entirely (`:5-10`), as §7.3 says. The
> duplicate number 4.2 is kept so references resolve.

---

## 5. Execution Flow

> **SUPERSEDED (2026-11-11) — §5.1 and §5.2 describe the v3.x worker** (JSON survivors, `SurvivorScorer`,
> GPU scoring, train/holdout metrics). None of `parse_arguments`, `load_params_from_file`,
> `extract_params_from_args`, `load_json` or `save_result` exists; the file defines exactly
> `_best_effort_gpu_cleanup` (`:125`), `load_data` (`:151`), `run_trial` (`:286`, with nested `_reject` `:400`),
> `_log_trial_metrics` (`:469`), `save_local_result` (`:495`) and `main` (`:522`). The flow at `a15b524`
> (DESCRIPTION-ONLY except where marked):
>
> ```
> main()                                                          scorer_trial_worker.py:522
>   ├─ argv[1] survivors NPZ, argv[2..3] history files (ignored), argv[4] trial_id      :527-531
>   ├─ params: --params-file | --params-json | inline JSON at argv[5]                  :540-596
>   ├─ load_data(): load_survivors(..., return_format='array')  (fallback allowed)     :166
>   │    reads seeds, forward_matches, reverse_matches (hard-required), bidirectional_count,
>   │    intersection_ratio, trial_number, skip_mode; optimal_window_config.json beside
>   │    the NPZ for prng_type/mod (returned, then unused)                             :179-282
>   ├─ run_trial():
>   │    if n_seeds > sample_size: random.seed(optuna_trial_number); sample          :353-367
>   │    else: use all seeds  ── BROKEN: sm_arr is never assigned here → UnboundLocalError at :421  (§17 B3)
>   │    k-of-3 residue mask from residue_mod_1/2/3 + max_offset                     :379-389
>   │    guards: subset_n < 10 / keep < 0.01 / keep > 0.40 → -1.0                     :406-411
>   │    objective = 0.70·tanh(enrich) + 0.20·coverage − 0.10·size_penalty, clipped   :420-444
>   │    returns (objective, full-length 0/1 membership array)                       :459-466
>   └─ save_local_result(trial_id, params, accuracy, status, error, scores)          :495-518
> ```
>
> The sample size that reaches the worker is not `min(10000, len(survivors))` (§5.1 below): it is the job's
> `sample_size`, which the shell sets to **5000** (`run_scorer_meta_optimizer.sh:171`; host `scorer_jobs.json`
> of 2026-03-11: 5000 in all 100 jobs). Worker default if absent: 50000 (`:349`); generator default: 25000
> (`generate_scorer_jobs.py:85`).

### 5.1 Main Flow Diagram

```
scorer_trial_worker.py main()
    │
    ├──► Parse arguments (--params-file or CLI args)
    │
    ├──► Load data files:
    │    ├─ survivors.json (from Step 1)
    │    ├─ train_history.json (80% of draws)
    │    └─ holdout_history.json (20% of draws)
    │
    ├──► Initialize SurvivorScorer with trial params:
    │    ├─ residue_mod_1 (5-20)
    │    ├─ residue_mod_2 (50-150)
    │    ├─ residue_mod_3 (500-1500)
    │    ├─ max_offset (1-15)
    │    └─ temporal_window_size (50-200)
    │
    ├──► Sample survivors (if too many):
    │    └─ sample_size = min(10000, len(survivors))
    │
    ├──► Score on training data:
    │    └─ GPU-vectorized batch scoring
    │
    ├──► Evaluate on holdout data (v3.4 fix):
    │    └─ SAME sampled seeds, different history
    │
    ├──► Compute trial metrics:
    │    ├─ mean_train_score
    │    ├─ mean_holdout_score
    │    ├─ generalization_gap
    │    └─ top_k_holdout_score
    │
    └──► Write result JSON:
         └─ scorer_trial_results/trial_{trial_id}.json
```

### 5.2 Code Structure

```python
def main():
    args = parse_arguments()
    
    # Load params from file or CLI
    if args.params_file:
        params = load_params_from_file(args.params_file, args.trial_id)
    else:
        params = extract_params_from_args(args)
    
    # Load data
    survivors = load_json(args.survivors_file)
    train_history = load_json(args.train_history_file)
    holdout_history = load_json(args.holdout_history_file)
    
    # Run trial
    result = run_trial(
        survivors=survivors,
        train_history=train_history,
        holdout_history=holdout_history,
        params=params,
        trial_id=args.trial_id
    )
    
    # Save result
    save_result(result, args.trial_id)
```

---

## 6. Trial Parameters

### 6.1 Optuna Search Space (v4.2)

| Parameter | Range | Purpose | Consumer |
|-----------|-------|---------|----------|
| `residue_mod_1` | 5-20 | Small residue modulus | Step 3 SurvivorScorer |
| `residue_mod_2` | 50-150 | Medium residue modulus | Step 3 SurvivorScorer |
| `residue_mod_3` | 500-1500 | Large residue modulus | Step 3 SurvivorScorer |
| `max_offset` | 1-15 | Temporal alignment offset | Step 3 SurvivorScorer |
| `temporal_window_size` | 50-200 | Stability analysis window | Step 3 SurvivorScorer |
| `temporal_num_windows` | 1-10 | Number of temporal windows | Step 3 SurvivorScorer |
| `min_confidence_threshold` | 0.05-0.50 | Min confidence for inclusion | Step 3 SurvivorScorer |
| `hidden_layers` | categorical | NN architecture string | Step 5 anti_overfit |
| `dropout` | 0.1-0.5 | NN dropout rate | Step 5 anti_overfit |
| `learning_rate` | 1e-4 to 1e-2 | NN learning rate | Step 5 anti_overfit |
| `batch_size` | [32,64,128,256] | NN batch size | Step 5 anti_overfit |

**Note (S108 audit):** All parameters are downstream consumers. rm1/rm2/rm3 +
max_offset + tw_size + tw_windows + min_conf feed Step 3 SurvivorScorer.
hidden_layers + dropout + lr + batch_size feed Step 5 anti_overfit_trial_worker.
Nothing to remove from the search space.

> **Corrected 2026-11-11 — §6.1.** **Ranges.** Four rows are stale; the live space is
> `generate_scorer_jobs.py:52-70`: `temporal_window_size` **50–100** (`:58`), `temporal_num_windows` **3–10**
> (`:59`), `min_confidence_threshold` **0.05–0.25** (`:60`), `batch_size` **[32, 64, 128]** (`:66`). The other
> seven ranges are accurate. Two keys are added to every job and are not in the table: `sample_size` (`:69`)
> and `optuna_trial_number` (`:119`). DESCRIPTION-ONLY.
>
> **Consumers.** The Step-3 half of the note is accurate: `SurvivorScorer` reads `residue_mod_1/2/3`,
> `max_offset`, `temporal_window_size`, `temporal_num_windows` and `min_confidence_threshold`
> (`survivor_scorer.py:97-111`, used at `:396-400`, `:670-675`, `:704`). **The Step-5 half is not:** the four
> NN parameters have **no consumer**. `optimal_scorer_config` occurs 0 times in
> `meta_prediction_optimizer_anti_overfit.py` (executable Step 5), `generate_anti_overfit_jobs.py`,
> `anti_overfit_trial_worker.py` and `run_anti_overfit_optimizer.sh`; Step 5 samples its own
> `hidden_layers`/`dropout`/`learning_rate`/`batch_size` with incompatible encodings and ranges
> (`generate_anti_overfit_jobs.py:29-39`: `'[128, 64]'` list-strings vs Step 2's `'128_64'`; lr 1e-5–1e-2;
> batch [16, 32, 64, 128]). **BROKEN — §17 B2.**
>
> **Corrected (post-review, 2026-11-11) — the Step-3 half is also not true of the pipeline.** `SurvivorScorer`'s
> constructor accepts these seven parameters, but Step 3 never supplies them: `full_scoring_worker.py:344` builds
> `scorer_class(prng_type=prng_type, mod=mod)` with no config (the worker takes no `--config` argument), and
> `generate_step3_scoring_jobs.py:201-209` reads only `prng_type` and `mod` from `optimal_scorer_config.json`.
> Host evidence (via the Chapter 4 audit): the live config has `residue_mod_1/2/3 = 17/150/1449`, but the
> 2026-03-11 `survivors_with_scores.json` carries the default `residue_8/125/1000` features. This is **Chapter 4
> B5** (`docs/CHAPTER_4_AUDIT.md`). It is absorbed there and not counted again as a Chapter 3 break.
> **Taken with B2:** of the eleven parameters Step 2 tunes, none reaches any downstream step. Only `prng_type`
> and `mod` are read from `optimal_scorer_config.json` downstream, and Step 2 does not tune those.
>
> **Objective signal.** Separately from who consumes them, only `residue_mod_1/2/3` and `max_offset` reach
> Step 2's own objective (`scorer_trial_worker.py:379-444`); `temporal_window_size` is read (`:347`) and
> unused. That is a property of the ruled v4.3 objective, recorded as governed in
> `docs/PIPELINE_BEHAVIOUR_MODEL.md` §4; it is noted here, not labelled. "Nothing to remove" is a judgement,
> not a code path; this pass makes no proposal about the search space.

### 6.2 Parameter File Format (v3.2+)

**scorer_jobs.json:**
```json
{
    "study_name": "scorer_meta_v1",
    "study_db": "sqlite:///optuna_scorer.db",
    "trials": [
        {
            "trial_id": 0,
            "params": {
                "residue_mod_1": 14,
                "residue_mod_2": 137,
                "residue_mod_3": 1136,
                "max_offset": 3,
                "temporal_window_size": 50
            }
        },
        {
            "trial_id": 1,
            "params": {
                "residue_mod_1": 8,
                "residue_mod_2": 92,
                "residue_mod_3": 847,
                "max_offset": 7,
                "temporal_window_size": 100
            }
        }
    ]
}
```

> **SUPERSEDED (2026-11-11) — §6.2.** `scorer_jobs.json` is a flat **list** of jobs, each
> `{"job_id": "scorer_trial_<i>", "analysis_type": "script", "script": "scorer_trial_worker.py", "args": [...],
> "expected_output": "scorer_trial_results/trial_<iiii>.json", "timeout": 3600}`
> (`generate_scorer_jobs.py:121-136`). `args` are the three ramdisk paths
> `/dev/shm/prng/step2/<basename>` (`:33-37`, `:126-128`), the trial index, **the parameters as one inline JSON
> string** (`:130`), then `--optuna-study-name`, `--optuna-study-db` (`:131-132`), plus `--use-legacy-scoring`
> if requested (`:139-140`). There is no `study_name`/`study_db`/`trials` envelope. The host copy (2026-03-11)
> has exactly this shape (audit addendum §A.1). DESCRIPTION-ONLY.

### 6.3 Why --params-file?

```bash
# OLD (v3.1): Very long SSH command
ssh rig-6600 'python scorer_trial_worker.py survivors.json train.json holdout.json \
    --trial-id 42 --residue-mod-1 14 --residue-mod-2 137 --residue-mod-3 1136 \
    --max-offset 3 --temporal-window-size 50 --optuna-study-name scorer_meta_v1 \
    --optuna-study-db sqlite:///optuna.db'

# NEW (v3.2+): Short command, params in file
ssh rig-6600 'python scorer_trial_worker.py survivors.json train.json holdout.json \
    42 --params-file scorer_jobs.json --optuna-study-name scorer_meta_v1 \
    --optuna-study-db sqlite:///optuna.db'
```

> **SUPERSEDED (2026-11-11) — §6.3.** Neither form is generated. The generator emits inline JSON at argv
> position 5 (`generate_scorer_jobs.py:130`), consumed by the backward-compatibility branch
> (`scorer_trial_worker.py:560-563`), and the coordinator ships the whole job as a JSON file over one SSH
> command (`scripts_coordinator.py:991-997`), which is what keeps the command short. The `--params-file`
> branch still works (`:570-589`) but has no producer. DESCRIPTION-ONLY.

---

## 7. Training/Holdout Split

### 7.1 Data Flow from Step 1

```
Step 1 (Window Optimizer) outputs:
    │
    ├── bidirectional_survivors.json (intersection of forward+reverse)
    │
    ├── train_history.json (80% of lottery draws)
    │
    └── holdout_history.json (20% of lottery draws)

Step 2.5 (Scorer Meta-Optimizer) uses:
    │
    ├── survivors from Step 1
    │
    ├── train_history for scoring during optimization
    │
    └── holdout_history for validation (unseen data)
```

> **Corrected 2026-11-11 — §7.1.** **Survivors.** `bidirectional_survivors.json` is no longer survivor data.
> On the certified path Step 1 writes it as a **post-success summary** of the generation — generation id,
> SHA-256s, counts, no seeds (`window_optimizer_integration_final.py:3068-3077`; on VM 101 a 1193-byte, 19-key
> dict). Step 2's survivor input is the **22-array NPZ** `bidirectional_survivors_binary.npz`, at repo root a
> finalizer-owned symlink into `.s172_accumulator/current/` (`utils/run_finalizer.py:129-136`, `:1400-1413`)
> (§16). **History.** The 80/20 split is accurate: `window_optimizer.py:1086-1105` writes
> `train_history.json` / `holdout_history.json` (host: 14454 / 3614 of 18068). Step 2 requires both to exist
> (`run_scorer_meta_optimizer.sh:213-218`) and then **ignores their contents** (`scorer_trial_worker.py:527-531`).
> DESCRIPTION-ONLY.

### 7.2 Objective Function (v4.2)

**v4.2 uses NPZ intrinsic fields — NO draw history dependency.**

```python
def run_trial(params, npz_data, trial_number):
    """
    v4.2: Score using bidirectional_count from NPZ as ground truth.
    
    The sieve's own evidence (how many seeds survived both directions)
    is the objective signal. No draw history needed.
    """
    # Load NPZ fields for this trial's survivor subset
    trial_mask = npz_data['trial_number'] == trial_number
    bid_counts = npz_data['bidirectional_count'][trial_mask]
    ir_values = npz_data['intersection_ratio'][trial_mask]

    if len(bid_counts) == 0:
        return {'accuracy': 0.0, 'params': params}

    # Primary signal: bidirectional_count (survival frequency)
    # Use median for robustness against heavy-tail distributions (TB Q2)
    bid_score = np.median(bid_counts)

    # Normalize to global percentile rank (stable across trials)
    all_bid = npz_data['bidirectional_count']
    percentile = np.searchsorted(np.sort(all_bid), bid_score) / len(all_bid)

    # Secondary bonus: intersection_ratio (quality of overlap)
    ir_bonus = 0.0
    if np.std(ir_values) > 1e-12:  # ir_disabled guard
        ir_bonus = 0.10 * np.median(ir_values)

    accuracy = percentile + ir_bonus

    return {
        'accuracy': accuracy,
        'params': params,
        'bid_median': float(bid_score),
        'ir_median': float(np.median(ir_values)),
        'n_survivors': int(trial_mask.sum()),
    }
```

**Key Design Decisions (S103-S108):**
- `bidirectional_selectivity` dropped (98.8% at floor — unusable signal)
- Median over mean: robust to heavy-tail count distributions
- Global percentile: stable across different trial sizes
- `ir_disabled` guard: prevents NaN when intersection_ratio is constant

> **SUPERSEDED (2026-11-11) — §7.2. The code block above is not the code that runs.** The live objective is
> **v4.3** (`scorer_trial_worker.py:413-444`, "TB ruling S107"):
>
> ```python
> # after the k-of-3 residue mask (:379-389) and the three degenerate guards (:406-411)
> is_rare_global = (sm_arr == 1)            # skip_mode == 1  → 'variable' (utils/prng_encoding.py:37)
> is_rare_subset = (sm_arr[mask] == 1)
> enrich   = log((mean(is_rare_subset) + 1e-6) / (mean(is_rare_global) + 1e-6))
> coverage = |unique(trial_number[mask])| / |unique(trial_number[sample])|
> size_penalty = min(|log((keep + 1e-9) / 0.10)|, 5.0)
> objective = clip(0.70 * tanh(enrich) + 0.20 * coverage - 0.10 * size_penalty, -1, 1)
> ```
>
> Why: the v4.2 median-percentile `bc_score` was *"structurally dead: 79.2% of pool at bc>=11300 => any large
> subset has constant median"* (`:414-416`). `bidirectional_count`, `intersection_ratio`, `forward_matches`
> and `reverse_matches` are still loaded and sliced and do not reach the objective; `LAMBDA_SIZE` and
> `IR_WEIGHT` are defined (`:339`, `:341`) and unused. The `bidirectional_selectivity` note above is accurate
> as v4.2 history (`:303-304`); the array remains position 16 of the 22-array contract. Ruled lineage:
> `docs/TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`, `docs/S107_session_log.md`. DESCRIPTION-ONLY.

### 7.3 Trial Metrics (v4.2)

| Metric | Meaning |
|--------|---------|
| `accuracy` | Primary objective: percentile(bid_count) + 0.10×ir_bonus |
| `bid_median` | Median bidirectional_count for this trial's survivors |
| `ir_median` | Median intersection_ratio for this trial's survivors |
| `n_survivors` | Number of survivors in this trial's subset |

**Note:** v3.4's holdout train/test split is no longer relevant in v4.2
since the objective uses NPZ intrinsic fields, not draw-based scoring.

> **SUPERSEDED (2026-11-11) — §7.3 table.** The per-trial diagnostic metrics logged as `[TRIAL_METRICS]` are
> `trial_num`, `params`, `subset_n`, `keep`, `enrich`, `p_global`, `p_sub`, `coverage`, `size_penalty`,
> `objective`, `reason` (`scorer_trial_worker.py:479-492`). They go to the log only; the result file carries
> the objective as `accuracy` (§10.1 note). The closing note (holdout split no longer relevant) is accurate
> (`:527-531`). DESCRIPTION-ONLY.

---

## 8. GPU-Vectorized Scoring

> **SUPERSEDED (2026-11-11) — all of §8.** This GPU path was removed at **v4.0** with `SurvivorScorer` and
> `ReinforcementEngine` (`scorer_trial_worker.py:5-10`, `:103`, "TB ruling S102/S103"). `score_survivors_gpu`
> does not exist in the file, and `SurvivorScorer` is not imported. `SurvivorScorer.extract_ml_features_batch`
> still exists (`survivor_scorer.py:547`) and belongs to **Step 3**. The 3.8× figure (§8.2) is **UNVERIFIED**:
> no benchmark artifact, and the measured code is gone. Step 2 binds a GPU (`:75-98`) and computes on the CPU.
> Replaced by: §5 SUPERSEDED note (live flow) and §7.2 note (live objective). DESCRIPTION-ONLY.

### 8.1 Batch Processing (v3.3+)

```python
def score_survivors_gpu(scorer, seeds, lottery_history):
    """
    GPU-vectorized scoring using SurvivorScorer.extract_ml_features_batch()
    
    v3.3: Crypto miner style - all parallel, single CPU transfer at end
    """
    features_list = scorer.extract_ml_features_batch(
        seeds=seeds,
        lottery_history=lottery_history
    )
    return features_list
```

### 8.2 Performance Comparison

| Method | 10K Seeds | Notes |
|--------|-----------|-------|
| Single-seed loop | ~300 sec | CPU-bound |
| GPU batch (v3.3) | ~80 sec | 3.8x faster |

### 8.3 Under the Hood

```
SurvivorScorer.extract_ml_features_batch():
    │
    ├── STEP 1: Generate ALL predictions on GPU
    │   └── PyTorch tensors, single kernel launch
    │
    ├── STEP 2: Compute ALL features on GPU
    │   ├── Base matching (vectorized)
    │   ├── Stats (mean, std, etc.)
    │   ├── Residuals (vectorized histogram via scatter_add)
    │   └── Temporal stability
    │
    └── STEP 3: SINGLE CPU transfer
        └── Stack all tensors → one .cpu().numpy() call
```

---

## 9. Adaptive Memory Batching

> **SUPERSEDED (2026-11-11) — §9.1–§9.3.** Removed at v4.0 with §8. `score_with_adaptive_batching` does not
> exist; neither `PYTORCH_HIP_ALLOC_CONF` nor `set_per_process_memory_fraction` occurs in
> `scorer_trial_worker.py` (`/bin/grep -c` → 0). The file's top-of-file code is GPU-id binding only
> (`:72-98`); the only torch/CuPy calls are in `_best_effort_gpu_cleanup` (`:125-146`). Memory pressure on the
> rigs for this step is bounded instead by the per-node batch cap of 6 (`scripts_coordinator.py:105`). §9.4,
> appended after the end of the chapter, is a separate section — see its own note. DESCRIPTION-ONLY.

### 9.1 The Problem

```
8GB VRAM (RX 6600):
    10K seeds × 5000 draws × float32 = 200MB predictions
    + features = ~400MB
    + overhead = ~600MB
    
    Fine for 10K seeds, but what about 100K?
```

### 9.2 Adaptive Batching

```python
def score_with_adaptive_batching(scorer, seeds, lottery_history, target_vram_mb=4000):
    """
    Adaptively batch to fit in VRAM.
    
    v3.3: Starts with estimated batch size, reduces on OOM.
    """
    # Estimate batch size
    bytes_per_seed = len(lottery_history) * 4 * 3  # float32 × 3 tensors
    estimated_batch_size = int(target_vram_mb * 1024 * 1024 / bytes_per_seed)
    batch_size = min(estimated_batch_size, 10000)  # Cap at 10K
    
    all_features = []
    
    for i in range(0, len(seeds), batch_size):
        batch_seeds = seeds[i:i+batch_size]
        
        try:
            features = scorer.extract_ml_features_batch(
                seeds=batch_seeds,
                lottery_history=lottery_history
            )
            all_features.extend(features)
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # Reduce batch size and retry
                batch_size = batch_size // 2
                torch.cuda.empty_cache()
                # Retry with smaller batch
                ...
            else:
                raise
        
        # Clear cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_features
```

### 9.3 Memory Settings for RX 6600

```python
# At top of scorer_trial_worker.py

import os
os.environ["PYTORCH_HIP_ALLOC_CONF"] = \
    "garbage_collection_threshold:0.8,max_split_size_mb:128"

import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of 8GB = 6.4GB
```

---

## 10. Result Serialization

> **SUPERSEDED (2026-11-11) — §10.1–§10.3.** The result file written by `save_local_result`
> (`scorer_trial_worker.py:495-518`) has exactly these keys: `trial_id`, `params`, `accuracy` (the objective),
> `status` (`success` / `pruned` / `error`), `error`, `hostname`, `timestamp` (epoch float), and on success
> `scores` — a full-length **0/1 membership array**, not a score (`:459-466`, `:513-514`). There is no
> `metrics` block, no `runtime_seconds`, no `gpu_id`. The path is
> `~/distributed_prng_analysis/scorer_trial_results/trial_{trial_id:04d}.json` (`:498-501`). There is no
> `save_result` and no `run_trial_safe`: errors are handled inline in `main` (`:620-637`), which writes
> `accuracy: -inf`, `status: "error"` and exits 1. The collection step reads only `trial_id` and `accuracy`
> (`run_scorer_meta_optimizer.sh:291-295`), so the `scores` array is transported and discarded every trial.
> DESCRIPTION-ONLY.

### 10.1 Result JSON Format

```json
{
    "trial_id": 42,
    "status": "success",
    "params": {
        "residue_mod_1": 14,
        "residue_mod_2": 137,
        "residue_mod_3": 1136,
        "max_offset": 3,
        "temporal_window_size": 50
    },
    "metrics": {
        "mean_train_score": 0.0847,
        "mean_holdout_score": 0.0823,
        "generalization_gap": 0.0024,
        "std_train_score": 0.0312,
        "std_holdout_score": 0.0298,
        "top_k_holdout_score": 0.1456,
        "n_survivors_scored": 10000
    },
    "runtime_seconds": 78.4,
    "gpu_id": 5,
    "hostname": "rig-6600",
    "timestamp": "2025-12-15T14:32:45"
}
```

### 10.2 save_result()

```python
def save_result(result: dict, trial_id: int):
    """
    Save trial result to local filesystem.
    
    Path: ~/distributed_prng_analysis/scorer_trial_results/trial_{trial_id}.json
    """
    results_dir = Path.home() / "distributed_prng_analysis" / "scorer_trial_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = results_dir / f"trial_{trial_id}.json"
    
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Result saved: {result_path}")
```

### 10.3 Error Handling

```python
def run_trial_safe(survivors, train_history, holdout_history, params, trial_id):
    """Run trial with error handling."""
    try:
        result = run_trial(survivors, train_history, holdout_history, params, trial_id)
        result['status'] = 'success'
        
    except Exception as e:
        result = {
            'trial_id': trial_id,
            'status': 'failed',
            'error': str(e),
            'params': params
        }
    
    return result
```

---

## 11. CLI Interface

> **SUPERSEDED (2026-11-11) — §11.1–§11.2.** The worker has no argparse. It reads `argv[1]` survivors NPZ,
> `argv[2]`/`argv[3]` history files (accepted, ignored), `argv[4]` trial id (`scorer_trial_worker.py:527-531`),
> then scans for `--params-file FILE`, `--params-json JSON`, `--optuna-study-name`, `--optuna-study-db`,
> `--gpu-id N` (consumed at the top of the file, `:79-82`), `--use-legacy-scoring`, or inline JSON at argv
> position 5 (`:540-566`). **No `--residue-mod-*`, `--max-offset` or `--temporal-window-size` flag exists**, so
> the "With CLI params" example raises *"Must provide either --params-file or --params-json"* (`:596`).
> `--use-legacy-scoring` is accepted and passed to `run_trial`, which never reads it (`:296`). The survivors
> argument is the NPZ, not a JSON file. DESCRIPTION-ONLY.

### 11.1 Arguments

```bash
python3 scorer_trial_worker.py <survivors> <train_history> <holdout_history> <trial_id> [options]

Positional Arguments:
  survivors           Path to survivors JSON from Step 1
  train_history       Path to training lottery history (80%)
  holdout_history     Path to holdout lottery history (20%)
  trial_id            Unique trial identifier

Options:
  --params-file FILE          JSON file with trial parameters
  --residue-mod-1 INT         Small residue modulus (5-20)
  --residue-mod-2 INT         Medium residue modulus (50-150)
  --residue-mod-3 INT         Large residue modulus (500-1500)
  --max-offset INT            Temporal alignment offset (1-15)
  --temporal-window-size INT  Stability window (50-200)
  --optuna-study-name NAME    Optuna study name
  --optuna-study-db PATH      Optuna database path
  --gpu-id INT                GPU device ID (default: 0)
  -h, --help                  Show help
```

### 11.2 Usage Examples

**With params file (recommended):**
```bash
python3 scorer_trial_worker.py \
    bidirectional_survivors.json \
    train_history.json \
    holdout_history.json \
    42 \
    --params-file scorer_jobs.json \
    --optuna-study-name scorer_meta_v1 \
    --optuna-study-db sqlite:///optuna_scorer.db
```

**With CLI params:**
```bash
python3 scorer_trial_worker.py \
    bidirectional_survivors.json \
    train_history.json \
    holdout_history.json \
    42 \
    --residue-mod-1 14 \
    --residue-mod-2 137 \
    --residue-mod-3 1136 \
    --max-offset 3 \
    --temporal-window-size 50
```

---

## 12. Integration with scripts_coordinator.py

### 12.1 Architectural Change (January 2026)

**IMPORTANT:** Step 2.5 now uses `scripts_coordinator.py` instead of `coordinator.py`.

| Aspect | Old (Deprecated) | New (Current) |
|--------|------------------|---------------|
| Coordinator | `coordinator.py` | `scripts_coordinator.py` |
| Job dispatch | Parallel SSH storm | Staggered, controlled |
| Success detection | stdout parsing | File-based |
| Failure modes | Ambiguous | Explicit (MISSING, EMPTY, TIMEOUT) |

**Rationale:** Script-based jobs require autonomy-safe execution semantics.

> **Verified 2026-11-11 — §12.1 is accurate.** `run_scorer_meta_optimizer.sh:264-269` launches
> `scripts_coordinator.py`; its `FailureMode` class defines `FILE_MISSING`, `FILE_EMPTY`, `INVALID_JSON`,
> `TIMEOUT`, `SSH_ERROR`, `EXECUTION_ERROR` (`scripts_coordinator.py:191-198`). It reads nodes from
> `distributed_config.json` (`:431-444`) and classifies these jobs as `'step2.5'` for batching
> (`:141-160`, cap 6 per node per batch, `:105`). "Step 2.5" here is executable Step 2.

### 12.2 Job Creation (generate_scorer_jobs.py)
```python
def generate_scorer_jobs(study_name, n_trials):
    """Generate job specifications for all trials."""
    jobs = []
    
    for trial_id in range(n_trials):
        job = {
            "job_id": f"scorer_trial_{trial_id}",
            "script": "scorer_trial_worker.py",
            "args": [
                "bidirectional_survivors_binary.npz",  # NPZ format (88x faster)
                "train_history.json",
                "holdout_history.json",
                str(trial_id),
                "--params-file", "scorer_jobs.json",
                "--optuna-study-name", study_name,
                "--optuna-study-db", f"sqlite:///optuna_{study_name}.db"
            ],
            "expected_output": f"scorer_trial_results/trial_{trial_id:04d}.json",
            "timeout": 3600
        }
        jobs.append(job)
    
    return jobs
```

> **SUPERSEDED (2026-11-11) — §12.2.** There is no `generate_scorer_jobs()` function; the logic is in `main`
> (`generate_scorer_jobs.py:74-157`). Each job adds `"analysis_type": "script"`, uses ramdisk paths
> `/dev/shm/prng/step2/<basename>` for all three inputs (`:33-37`, `:126-128`), carries the parameters as
> inline JSON (`:130`) rather than `--params-file`, and passes the study DB URL given on the command line
> (`:132`; the shell gives `sqlite:///./optuna_studies/scorer_meta_opt_<epoch>.db`,
> `run_scorer_meta_optimizer.sh:193`, `:223`). `expected_output` is accurate (`:134`). See §6.2 note.
> DESCRIPTION-ONLY.

### 12.3 Coordinator Dispatch
```bash
# Step 2.5 uses scripts_coordinator.py (NOT coordinator.py)
python3 scripts_coordinator.py \
    --jobs-file scorer_jobs.json \
    --output-dir scorer_trial_results \
    --preserve-paths
```

### 12.4 Result Collection

Result collection still uses `coordinator.py` as a transport utility (SCP pull only):
```python
from coordinator import MultiGPUCoordinator

coord = MultiGPUCoordinator('ml_coordinator_config.json')
all_results = coord.collect_scorer_results(total_trials)
```

> **Verified 2026-11-11 — §12.3 is accurate** (matches `run_scorer_meta_optimizer.sh:266-269`).
>
> **Corrected 2026-11-11 — §12.4. The chapter states the right call; the shell does not make it.**
> `run_scorer_meta_optimizer.sh:285` calls `coord.collect_scorer_results(1)` — a literal `1`, whatever
> `$TRIALS` is. `collect_scorer_results` (`coordinator.py:2564`) uses the argument only for the shortfall
> warning (`:2591-2593`), so any run that collects one or more results raises no warning and logs
> *"Found N / 1 results"*. A run that returns 3 of 500 trials reports no problem. **BROKEN — §17 B1.** The
> collection config is `ml_coordinator_config.json` (26 GPUs), a different fleet file from the one dispatch
> uses (`distributed_config.json`, 25); neither is the frozen execution set (`execution_set` occurs 0 times in
> the shell, the generator, the worker or `scripts_coordinator.py`). Recorded, not labelled (no chapter claim).

### 12.5 Binary Survivor Loading (NPZ Format)

| Format | File Size | Load Time |
|--------|-----------|-----------|
| JSON | 258 MB | 4.2s |
| NPZ | 0.6 MB | 0.05s |

**Conversion:**
```bash
python3 convert_survivors_to_binary.py bidirectional_survivors.json
```

> **Corrected 2026-11-11 — §12.5. SWITCHED-OFF-BY-RULING — §17 S1 (= Chapter 12 S2; not a new finding).**
> The size/time table is recorded in the approved `docs/PROPOSAL_NPZ_Auto_Conversion_Step2.md:38-41`
> ("58-258 MB", 4.2 s / 0.6 MB, 0.05 s) and was not re-measured: **UNVERIFIED**.
>
> **The conversion command must not be run.** `convert_survivors_to_binary.py` is the legacy converter. The
> owner/Beta record says it is **not invoked until D3.0-B closes**, and Phase 6 is certified only for the
> miner/finalizer path that **never invoked** it (`docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` §1). On RANGE-MINER
> output the canonical NPZ is produced by the D3.5 finalizer, not by conversion (§16).
>
> **The shell still auto-converts.** Committed Chapter 12 §13/§14 (S2) records that
> `run_scorer_meta_optimizer.sh` calls the converter at `:87` and `mv`s a regular file onto the
> finalizer-owned root symlink at `:97`; the finalizer then fails closed on the next publication
> (`utils/run_finalizer.py:1400-1413`). That is the governed item; this chapter cross-references it. Live-host
> detail added by this pass (audit addendum §A.4 F1), mechanism only: on every certified Step-1 run the summary
> `bidirectional_survivors.json` is written (`window_optimizer_integration_final.py:3073`) **after**
> `finalize_run` (`:3001`), so it is newer than the NPZ and the `-nt` test at `:144` is true (on VM 101 now:
> 26 ms newer, `test -nt` TRUE). The converter is therefore entered on certified output, receives a summary dict
> rather than a survivor list, and fails at `convert_survivors_to_binary.py:112`; the shell exits 1 at `:88-89`
> before `:97`. Production-class launches pin `--start-step 1 --end-step 1` so Step 2 is not reached
> (`gate12_launch.sh:73`, `:378`). **What lifting the hold would enable:** running Steps 2–6 through WATCHER on
> certified miner output — once D3.0-B's bounded scope (`docs/BACKLOG.md` §12) is done and the auto-conversion
> branch is reconciled with the finalizer-owned alias. No recommendation is made here.

---


## 13. Troubleshooting

### 13.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "No result files found" | Worker crashed | Check stderr, look for Python errors |
| "OOM" | Batch too large | Reduce sample size or batch size |
| "Holdout score = 0" | Wrong holdout file | Verify holdout_history.json exists |
| "Trial timeout" | Slow scoring | Increase timeout, reduce sample size |
| "Permission denied" | SCP auth | Check SSH keys on coordinator |

> **Corrected 2026-11-11 — §13.1.** "Holdout score = 0" cannot occur: there is no holdout score (§7.3 note).
> "OOM … reduce batch size" refers to the removed GPU path (§8, §9); the step's NN `batch_size` is never used
> by the worker. Two failure signatures this table lacks: (a) **every trial `status: "error"` with
> `UnboundLocalError: … 'sm_arr'`** — the pool is at or below the job's `sample_size` (§17 B3); (b) the shell
> dying with **`--legacy-scoring: command not found`** (exit 127) — the legacy-scoring branch (§17 B4). And
> **"NPZ Conversion Required" / "JSON newer than NPZ - reconversion required"** at the start of the run means
> the held legacy converter is about to run (§12.5 note, §17 S1). DESCRIPTION-ONLY.

### 13.2 Debug Mode

```bash
# Run locally with verbose output
python3 scorer_trial_worker.py \
    test_survivors.json \
    train_history.json \
    holdout_history.json \
    0 \
    --residue-mod-1 10 \
    --residue-mod-2 100 \
    --residue-mod-3 1000 \
    --max-offset 5 \
    --temporal-window-size 100 \
    2>&1 | tee worker_debug.log
```

> **Corrected 2026-11-11 — §13.2.** The `--residue-mod-*`, `--max-offset` and `--temporal-window-size` flags do
> not exist (§11 note); this command stops at `scorer_trial_worker.py:596`. The worker's input is the NPZ, and
> parameters go in as `--params-json '<json>'` or inline JSON at argv position 5 (`:540-563`). Running the
> worker is a Step-2 execution and is outside a documentation pass; no replacement command is given here.
> DESCRIPTION-ONLY.

### 13.3 Verify Result Files

```bash
# Check local results
ls -la ~/distributed_prng_analysis/scorer_trial_results/

# Check remote results
ssh rig-6600 'ls -la ~/distributed_prng_analysis/scorer_trial_results/'

# View a result
cat ~/distributed_prng_analysis/scorer_trial_results/trial_0.json | jq .
```

> **Corrected 2026-11-11 — §13.3.** Files are `trial_0000.json` etc. (§3.2 note), and the collector deletes
> them after reading (§3.1 note), so an empty directory after a completed run is expected. `rig-6600` is an old
> host name; the rig endpoints depend on the boot target (`CLAUDE.md` §3). DESCRIPTION-ONLY.

---

## 14. Complete Method Reference

### 14.1 Main Functions

| Function | Purpose |
|----------|---------|
| `main()` | Entry point, parse args, run trial |
| `parse_arguments()` | Parse CLI arguments |
| `load_params_from_file(path, trial_id)` | Load params from JSON file |
| `run_trial(...)` | Execute single trial |
| `save_result(result, trial_id)` | Write result JSON |

### 14.2 Scoring Functions

| Function | Purpose |
|----------|---------|
| `score_survivors_gpu(scorer, seeds, history)` | GPU-vectorized batch scoring |
| `score_with_adaptive_batching(...)` | OOM-safe batched scoring |
| `compute_metrics(train_scores, holdout_scores)` | Compute trial metrics |

### 14.3 Utility Functions

| Function | Purpose |
|----------|---------|
| `sample_survivors(survivors, n)` | Random sample for scoring |
| `load_json(path)` | Load JSON file |
| `setup_gpu(gpu_id)` | Initialize GPU device |

> **SUPERSEDED (2026-11-11) — §14.1–§14.3.** Of the eleven functions listed, only `main` and `run_trial` exist.
> The complete function list of `scorer_trial_worker.py` at `a15b524` (DESCRIPTION-ONLY):
>
> | Function | Line | Purpose |
> |---|---|---|
> | `_best_effort_gpu_cleanup()` | `:125` | gc + CuPy pool free + torch cache empty; registered `atexit` (`:149`) |
> | `load_data(survivors_file, train_history_file=None, holdout_history_file=None)` | `:151` | one-time NPZ load into module globals; hard-requires forward/reverse matches and `skip_mode`; reads `optimal_window_config.json` beside the NPZ |
> | `run_trial(seeds_to_score, …, params, prng_type, mod, trial, use_legacy_scoring)` | `:286` | sampling, residue mask, guards, v4.3 objective; returns `(objective, membership_array)` |
> | `_reject(reason)` (nested) | `:400` | degenerate-guard exit with `-1.0` |
> | `_log_trial_metrics(...)` | `:469` | logs `[TRIAL_METRICS]` |
> | `save_local_result(trial_id, params, accuracy, state, error, scores)` | `:495` | writes `trial_NNNN.json` |
> | `main()` | `:522` | argv parsing, params, load, run, save, exit code |

---

## 15. Chapter Summary

**Chapter 13: Scorer Trial Worker** covers single trial execution:

| Component | Lines | Purpose |
|-----------|-------|---------|
| ROCm setup | ~20 | Environment before imports |
| Argument parsing | ~50 | CLI and params-file support |
| Trial execution | ~100 | Score + evaluate + metrics |
| GPU batching | ~80 | Vectorized scoring |
| Result handling | ~30 | JSON serialization |

**Key Points:**
- Part of pull architecture (write locally, coordinator pulls)
- v3.4 critical fix: Holdout uses SAME sampled seeds
- v3.3: GPU-vectorized scoring (3.8x faster)
- v3.2: --params-file for shorter SSH commands
- Adaptive batching for 8GB VRAM

> **Corrected 2026-11-11 — §15.** This file is **Chapter 3**, not "Chapter 13" (Chapter 13 is
> `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md`). Of the key points, the pull architecture is accurate; the v3.4
> holdout fix is history (removed at v4.0); the v3.3 GPU scoring and adaptive batching are removed (§8, §9);
> `--params-file` works but is not generated (§6.3 note). The component table describes the v3.x file. The
> current summary: one trial per job, NPZ-only, CPU NumPy, v4.3 enrichment objective, local result file pulled
> and deleted by the collector. DESCRIPTION-ONLY.

---

## Next Chapter

**Chapter 14: Feature Importance** will cover:
- `feature_importance.py` — Permutation, gradient, SHAP methods
- `feature_visualizer.py` — 13 chart types
- AI-powered interpretation

> **Corrected 2026-11-11 — Next Chapter.** The next chapter is **Chapter 4: Full Scoring**
> (`docs/CHAPTER_4_FULL_SCORING.md`, executable Step 3). Feature importance is
> `docs/CHAPTER_11_FEATURE_IMPORTANCE_VISUALIZATION.md`. DESCRIPTION-ONLY.

---

*End of Chapter 13: Scorer Trial Worker*

> **Corrected 2026-11-11 — end marker.** This is the end of the original Chapter 3 body, not of Chapter 13.
> §9.4 below belongs to §9; §16–§20 were added on 2026-11-11.

---

### 9.4 Resource Scaling and Performance Constraints — VALIDATED

> **Updated: 2026-01-18** — Based on systematic benchmark testing

> **Corrected 2026-11-11 — §9.4 as a whole.** (1) **450 is not the operating point.** S107 found the 450 cap
> obsolete once v4.3 replaced NN training with NumPy arithmetic (*"timing became identical (2.8s for both 450
> and 5000)"*) and set the shell default to **5000** (`docs/S107_session_log.md` item 2;
> `run_scorer_meta_optimizer.sh:171`). The host's last `scorer_jobs.json` (2026-03-11) carries 5000 in all 100
> jobs. DESCRIPTION-ONLY. (2) **The manifest's `sample_size: 450` cannot reach the run.**
> `agent_manifests/scorer_meta.json:126` sets it, but the manifest is `arg_style: positional` with
> `positional_args: ["trials"]` (`:136-139`), and WATCHER emits only declared positional args and flag args
> (`agents/watcher_agent.py:2089-2103`), so `--sample-size` is never passed and the shell default runs. S107
> recorded the intent that WATCHER's value reach the shell. **BROKEN — §17 B5.** (3) **Concurrency.**
> `max_concurrent_script_jobs` is **8** per rig in both node configs (S107 corrected 12 → 8, item 5; localhost 2),
> and `scripts_coordinator` caps this step at **6 jobs per node per batch** (`scripts_coordinator.py:105`).
> DESCRIPTION-ONLY. (4) The throughput table and "TPE ranking is preserved" were measured against the v3.x NN
> scorer, which no longer exists; no benchmark artifact is in the repo. **UNVERIFIED.**

#### Sample Size vs Throughput Trade-off

At 12-way ROCm concurrency, sample sizes above 450 increase memory residency time without improving Optuna convergence. The relationship is inverse: smaller samples = higher throughput.

| Sample Size | Throughput | Signal Quality |
|-------------|------------|----------------|
| 350 | 14.98 trials/min | ✅ Preserved |
| **450** | **15.41 trials/min** | ✅ **Optimal** |
| 550 | 14.66 trials/min | ✅ Preserved |
| 1000 | 10.42 trials/min | ✅ Preserved |
| 2000+ | — | ❌ Freeze risk |

#### Why Signal Quality Is Preserved

Optuna's TPE (Tree-structured Parzen Estimator) sampler is designed for noisy objectives:
- Hyperparameter **ranking** is preserved across sample sizes
- More trials with lower precision beats fewer trials with higher precision
- 500 trials × moderate precision > 100 trials × high precision for global coverage

#### Validated Operating Point

```bash
# Optimal Step 2.5 configuration
sample_size=450              # Maximize throughput
max_concurrent_script_jobs=12  # Full GPU utilization
trials=200-500               # Good Bayesian coverage
```

#### Performance Improvement

| Configuration | Throughput | Factor |
|---------------|------------|--------|
| Old (5000 samples @ 4 concurrent) | ~3.4 trials/min | 1× |
| New (450 samples @ 12 concurrent) | 15.41 trials/min | **4.5×** |

---

## 16. What RANGE-MINER, the 22-array NPZ and Brief I changed for Step 2

**Added 2026-11-11.** Numbering: executable (Step 1 = window optimizer containing the sieve; Step 2 = this stage).

### 16.1 What Step 2 consumes

Step 2's only survivor input is the **22-array NPZ** `bidirectional_survivors_binary.npz`. The worker reads seven
arrays by name and nothing else from it (`scorer_trial_worker.py:179-246`):

| Array | Contract position | Used by the v4.3 objective? |
|---|---|---|
| `seeds` | 1 | yes — residue mask (`:379-389`) |
| `forward_matches` | 2 | no — **hard-required** (`:183-196`), then unused |
| `reverse_matches` | 3 | no — hard-required, then unused |
| `trial_number` | 6 | yes — coverage (`:428-430`) |
| `bidirectional_count` | 12 | no — loaded with a ones fallback (`:201-211`), unused |
| `intersection_ratio` | 14 | no — loaded with a zeros fallback (`:213-222`), unused |
| `skip_mode` | 21 | yes — enrichment (`:420-425`); required after load (`:258-261`) |

Positions are from `utils.canonical_arrays.CANONICAL_ARRAY_CONTRACT` (`utils/canonical_arrays.py:98`), listed by
import this pass: 22 `(name, dtype)` entries, `seeds` `uint32`, `skip_mode` `uint8`, the six `*_count` arrays
`float32`. **All seven are in the contract, so RANGE-MINER's certified bundle satisfies every column Step 2 needs;
the consumer cannot tell which engine produced it.** The worker also reads `optimal_window_config.json` from the
NPZ's directory for `prng_type`/`mod` (`:263-280`), which v4.3 does not use; under the ramdisk path that directory
is `/dev/shm/prng/step2`, where the preload does not place that file (`run_scorer_meta_optimizer.sh:18-21`), so the
defaults apply. Step 2 reads no draw file and no generation sidecar (§7.1 note).

### 16.2 Where the NPZ comes from now

With RANGE-MINER, the certified NPZ is assembled by the miner path and published by the D3.5 finalizer into an
immutable generation directory; the repo-root name is a **finalizer-owned compatibility symlink**
`bidirectional_survivors_binary.npz → .s172_accumulator/current/bidirectional_survivors_binary.npz`
(`utils/run_finalizer.py:129-136`, `:1400-1413`). On VM 101 that symlink exists and resolves to the 2026-08-17
generation `gen-20260818T013219550933Z-step1_java_lcg_0…`. `bidirectional_survivors.json` is now a post-success
summary, not survivor data (`window_optimizer_integration_final.py:3068-3077`). **Step 2 has not run since this
change on VM 101:** its newest outputs (`optimal_scorer_config.json`, `scorer_jobs.json`) are dated 2026-03-11.

### 16.3 The converter, and why Step 2 on miner output is held

`run_scorer_meta_optimizer.sh` predates the finalizer. Its auto-conversion block (`:140-150`) treats the NPZ as
something it may regenerate from `bidirectional_survivors.json` with the legacy converter
`convert_survivors_to_binary.py` (`:87`) and `mv` into place (`:97`). On miner output that is both held by ruling
and destructive to the finalizer's alias; see §12.5 note and §17 S1 (= Chapter 12 S2). The converter itself now
emits the same 22 names and dtypes (`convert_survivors_to_binary.py:46-73`, `:201-225`), but it is the
uncertified legacy writer (D3.0-B open). **Ordering detail:** the ramdisk preload (`:18-21`) runs before the
conversion block, so on a run where conversion succeeded the workers would read the preloaded certified bytes
(`ramdisk_preload.sh:117`, `:139`; `cp`/`scp` follow the symlink) while the root alias had been replaced.

### 16.4 Brief I (window-anchor / generator-phase split) does not reach Step 2

Brief I made the canonical `offset` array (5th of 22; "array 4" in the code's 0-based wording) carry the
**window anchor** only (`utils/canonical_records.py:225`,
`:374`, *"a legacy WIRE NAME whose one post-F-4 meaning IS the window anchor — never the generator phase"*), and
keeps `generator_phase` in miner generation metadata (`miner/range_miner_coordinator.py:1389`, `:2616`), never in
the NPZ. **Step 2 reads neither `offset` nor `window_size`**, so the change alters nothing this stage computes.
Brief I final acceptance is pending a completed production-shape run
(`docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md`).

### 16.5 What Step 2 passes on

Only `optimal_scorer_config.json`: `study.best_params`, eleven scalars (`run_scorer_meta_optimizer.sh:302-307`; host
copy: the 11 sampled keys). *(Post-review, 2026-11-11.)* None of those eleven values is applied downstream. Step 3
reads only `prng_type`/`mod` from the file (`generate_step3_scoring_jobs.py:201-209`) and builds its scorer
without config (`full_scoring_worker.py:344`), which is Chapter 4 B5. Step 5 never opens the file (§17 B2).
Nothing per-seed reaches Step 3 from here; the per-trial membership arrays are
discarded at collection (§10 note). The `forward_matches`/`reverse_matches` gap at the Step-3 merge list is
Step 3's (`docs/BACKLOG.md` §1) and is not touched by this stage.

---

## 17. Known gaps at a15b524

Labels per `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` §3.1. Detail, anchors and search
surfaces: `docs/CHAPTER_3_ALIGNMENT_AUDIT.md` (F-findings and the 2026-11-11 addendum). Nothing here was repaired.

### 17.1 BROKEN

| # | What the chapter says (or describes) | What the code does | Anchor | Audit |
|---|---|---|---|---|
| **B1** | §12.4: results collected with `collect_scorer_results(total_trials)` | shell passes a literal `1`; the shortfall warning cannot fire | `run_scorer_meta_optimizer.sh:285`; `coordinator.py:2591-2593` | F10, #47 |
| **B2** | §6.1: `hidden_layers`, `dropout`, `learning_rate`, `batch_size` feed Step 5 | no Step-4/5 code reads `optimal_scorer_config.json`; Step 5 samples its own, in an incompatible encoding | `generate_anti_overfit_jobs.py:29-39`; `/bin/grep -c optimal_scorer_config` = 0 in the four Step-5 files | F7, #28 |
| **B3** | §5.1: sample survivors "if too many", otherwise score all | the no-sampling branch never assigns `sm_arr`; every trial raises `UnboundLocalError` when the pool ≤ `sample_size` | `scorer_trial_worker.py:365-366` vs `:368-375`, read at `:421` | F5, N1 |
| **B4** | §11 / `--legacy-scoring` path (WATCHER `flag_args`) | the shell's legacy branch lacks a line continuation; `--legacy-scoring` runs as a command, exit 127 under `set -e` | `run_scorer_meta_optimizer.sh:236-237`, `:165`; `agent_manifests/scorer_meta.json:140-142`; `agents/watcher_agent.py:2100-2103` | F11, N2 |
| **B5** | §9.4: the configured `sample_size` is what runs | manifest `sample_size: 450` has no path under `arg_style: positional`; the shell default 5000 runs | `agent_manifests/scorer_meta.json:126`, `:136-139`; `agents/watcher_agent.py:2089-2098`; `run_scorer_meta_optimizer.sh:171` | F13, N3 |

**Absorbed (post-review, 2026-11-11), not counted here:** the §6.1 claim that the seven residue and temporal
parameters feed Step 3's `SurvivorScorer` (audit claim #27). It is false in the pipeline:
`full_scoring_worker.py:344` constructs the scorer with `prng_type`/`mod` only, and
`generate_step3_scoring_jobs.py:201-209` reads nothing else from `optimal_scorer_config.json`. It is **Chapter 4
B5** (`docs/CHAPTER_4_AUDIT.md`) and is counted once, there. With B2, none of the eleven parameters Step 2 tunes
reaches a downstream step. B2 itself stays a single item: the four NN parameters and Step 5.

What a repair must do (no code here): **B1** pass the real trial count to the collector so a shortfall is
reported; **B2** give the four NN parameters a real consumer with a compatible encoding, or have the owner record
that they are intentionally unconsumed; **B3** assign the `skip_mode` slice on both branches so small pools run;
**B4** make the legacy branch's argument list one command; **B5** make the configured sample size and the executed
one the same value, with a path from manifest to shell (reconciling 450 with S107's 5000).

### 17.2 SWITCHED-OFF-BY-RULING

| # | Item | Held by | Lifting it would enable |
|---|---|---|---|
| **S1** | §12.5 conversion and the shell's auto-conversion: `run_scorer_meta_optimizer.sh:87` calls the legacy converter, `:97` `mv`s onto the finalizer-owned symlink. **= Chapter 12 S2; not a new finding** | `docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` §1 (D3.0-B open; legacy converter not invoked until it closes; Phase 6 certified only for the path that never invoked it); scope in `docs/BACKLOG.md` §12; launch rule `--start-step 1 --end-step 1` (`gate12_launch.sh:73`, `:378`); tracked in `docs/TODO_CONSOLIDATED.md:157`. Original approval of the block: `docs/PROPOSAL_NPZ_Auto_Conversion_Step2.md` | Step 2 (and so Steps 3–6) through WATCHER on certified RANGE-MINER output; a full 0→6 pipeline and Chapter-13 retrain requests on miner output |

Per §0 of the authority brief, a Beta ruling described as binding is recorded as a review position the owner may
keep or lift. No recommendation is made here.

### 17.3 DESCRIPTION-ONLY

D1 version 4.2 → objective v4.3 / worker v4.4 (header, §4.1) · D2 "on remote GPU" → binds a GPU, CPU NumPy
(header, §8, §9) · D3 parameter delivery: inline JSON, no per-parameter flags (§1.1, §6.3, §11) · D4 "scores
survivors" → residue mask (§1.1, §2.1) · D5 holdout evaluation removed at v4.0 (§1.1, §2.1, §7.1, §13.1) · D6 §1.3
diagram (deprecated `.py`, `coordinator.py`) · D7 "26 GPUs" depends on the file: 25 dispatch / 26 collection (§2.1)
· D8 file names `trial_NNNN.json` (§3.2, §13.3) · D9 §4.1 v3.5/v3.6 attribution and missing v4.3/v4.4 · D10 §4.2 v4.2
signal superseded · D11 §5 flow and functions · D12 §5.1 sample-size default · D13 four §6.1 ranges · D14 §6.1
"nothing to remove" is a judgement · D15 §6.2 job-file format · D16 §7.1 `bidirectional_survivors.json` is a
summary · D17 §7.2 objective · D18 §7.3 metrics · D19 §8 GPU scoring (removed v4.0) · D20 §9.1–§9.3 memory
batching (removed v4.0) · D21 §10 result format, `save_result`, `run_trial_safe` · D22 §11 CLI · D23 §12.2 job
generator · D24 §13.2 debug flags · D25 §14 method reference · D26 §15 "Chapter 13" and the end marker · D27 "Next
Chapter: 14" · D28 §9.4 450 operating point (S107 retired it) · D29 §9.4 `max_concurrent_script_jobs=12` → 8,
batch cap 6 · D30 "Step 2.5" vs executable Step 2 · D31 `--trial-id` flag (§3.1) · D32 pull transport is SFTP, not
SCP (§1.2, §2.2, §3.1). Each has its note in place. The per-claim count (43 DESCRIPTION-ONLY claims, some grouped
here) is in the audit addendum §A.8.

---

## 18. Open items

| Item | Where tracked |
|---|---|
| S1 — D3.0-B and Step 2 on miner output | `docs/BACKLOG.md` §12; `docs/TODO_CONSOLIDATED.md:157`; Chapter 12 §14.2 S2 |
| B1–B5 — for the repair list | `docs/CHAPTER_3_ALIGNMENT_AUDIT.md` F5, F7, F10, F11, F13 and addendum §A.6; in no other governance document |
| The F1 `-nt` mechanism (certified-run summary JSON is always newer than the NPZ) | audit addendum §A.4 F1; documented nowhere else — belongs with S1/D3.0-B |
| Objective blind to 7 of 11 dimensions; objective steers toward `skip_mode == 1` | audit F6, F8; governed per `docs/PIPELINE_BEHAVIOUR_MODEL.md` §4 (S107 lineage); `max_offset` scale flaw "Needs TB approval before v4.5" (`docs/S107_session_log.md`) |
| Fleet surfaces outside the execution set (`ml_coordinator_config.json`, hardcoded push IPs `:257`) | audit F9; `docs/PIPELINE_BEHAVIOUR_MODEL.md` §16 D17 |
| Loader fallback enabled (`scorer_trial_worker.py:166`) | audit F14 |
| Step-2 module docstrings still say v4.2 | audit F3 (code comment; not changed by a documentation pass) |
| UNVERIFIED: 3.8× (§4.1/§8.2), JSON/NPZ table (§12.5), §9.4 throughput table and TPE-ranking claim | audit addendum §A.5; needs a measurement artifact |
| Deployed rig copies of `scorer_trial_worker.py` | UNAVAILABLE this pass (no rig access) |

---

## 19. Verification declaration (VIR-1…6)

**Execution proof (VIR-1).** Every anchor in the 2026-11-11 notes was opened on VM 101 at `a15b524` in this session
with `Read`, `sed -n`, `cat -n` or `/bin/grep -n`; the full file list is in the audit addendum §A.9. Nothing was
executed except read-only host queries, `test -nt` on two existing files, JSON parses of existing artifacts, and an
import of `utils.canonical_arrays` to list the contract. **Anchor base:** during this pass another session modified
`agents/watcher_agent.py` and `agents/watcher_dispatch.py` in the working tree (not this pass's change; mtime
2026-11-11 17:08). Every `agents/watcher_agent.py` line cited here was re-checked against
`git show a15b524:agents/watcher_agent.py` and is correct **at `a15b524`**; in that modified working tree, lines after
`:308` are shifted.

**Clean control (VIR-2).** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was read, not modified. Claims verified accurate
and left unchanged: 20 (audit addendum §A.5, §A.6, §A.11; was 21 before the post-review correction of #27).

**Fault-injection control.** **Not applicable** to a documentation pass; none was invented.

**Completion sentinel (VIR-3).** §20.

**Unavailable-observer behaviour (VIR-5).** The rigs and their deployed worker copies, a populated
`/dev/shm/prng`, `.127`, root/other-user crontabs, and the S107/S108 benchmark artifacts are **UNAVAILABLE**,
never clean. Rig reachability for WATCHER's preflight (audit F16) was not re-checked.

**Audit claim scope (VIR-6).** What this chapter says against what the code at `a15b524` does — not whether the
Step-2 objective or search space is right. Searched surfaces: the tracked tree; gitignored files (`git check-ignore
-v`, `agent_manifests/definitions.json` — 220 lines, no Step-2 entry, read by no tracked code — and the root `*.json`
artifacts); git history (`git log -S`, `git log -- <path>`, `git show --numstat`, `git rev-parse ca975f8^`); host
(`ls`/`stat` of inputs and outputs, `systemctl list-unit-files --state=enabled`, `systemctl list-timers --all`,
`crontab -l`, `ps`). Governance trail searched first: `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`,
`docs/S107_session_log.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, the fact file. Chapters searched:
`CHAPTER_2`, `CHAPTER_12` (§13, §14), `CHAPTER_13_AUDIT.md`, `PIPELINE_BEHAVIOUR_MODEL.md`.

---

## 20. Closure statement

**Verified against** `a15b524` on VM 101, 2026-11-11. **Corrected:** 74 claims checked (55 from the original audit +
8 it did not number + 11 search-space rows) — **20 accurate, 5 BROKEN, 1 SWITCHED-OFF-BY-RULING, 43
DESCRIPTION-ONLY, 4 UNVERIFIED, 1 absorbed** (#27 → Chapter 4 B5); 20 + 5 + 1 + 43 + 4 + 1 = 74. *(Post-review
correction, 2026-11-11: the first version of this line read 21 accurate / 0 absorbed.)* **Added:** framing block, status header,
§0, an as-built diagram in §1.3, the live objective in §7.2, the live function list in §14, §16–§20. **Superseded,
kept:** §1.3 diagram, §5, §6.2, §6.3, §7.2 code, §7.3 table, §8, §9.1–§9.3, §10, §11, §12.2, §14.
**Removed:** nothing — every pre-existing heading is present; the chapter grew.

**What this chapter is NOT:** not a certification that Step 2 is correct, not a runbook (nothing here is to be
executed; Step 2 through WATCHER on miner output is held, and every run is Michael-initiated), and not a judgement on
the v4.3 objective.

```
CHAPTER 3 CORRECTION:  PASS
```

`PASS` for the declared scope: every claim audited and labelled, the chapter corrected in place with a visible trail,
the RANGE-MINER / 22-array / Brief I effect on Step 2 documented from live code, gaps listed. It is not a claim that
any gap is repaired. Four claims remain UNVERIFIED by name. Files changed: this chapter and
`docs/CHAPTER_3_ALIGNMENT_AUDIT.md` only.


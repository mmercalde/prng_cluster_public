# Chapter 12: WATCHER Agent & Fingerprint Registry

> **What TFM is.** TFM filters candidate generators against observed draws in two directions,
> keeps a population of survivors on purpose, ranks them, and lets the top candidates vote their
> next value into a prediction pool. A survivor is a scored candidate, not a verdict. The
> generator family is a substrate for mimicry, not a claim about the machine. The target is a
> pool hit rate. TFM does not recover a seed or reconstruct state, and an objection that assumes
> it does is out of scope.

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 1.5.0 — corrected 2026-11-11 (was 1.4.0)
**Status:** **CORRECTED at `a15b524`, 2026-11-11** — audited against the code, corrected in place,
history preserved. Not closed: see §17 for the closure statement and sentinel, §15 for open items.
**Authority for this pass:** `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` (REV1).
Audit of record: `docs/CHAPTER_12_AUDIT.md`.
**Base:** `a15b5247336255801de8cf67d52653701b1e8c80` (`a15b524`), VM 101, working tree.
**Numbering scheme:** **EXECUTABLE** (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). Bare "Step N" is the
step WATCHER runs: `STEP_SCRIPTS` / `STEP_MANIFESTS` / `STEP_NAMES` at `agents/watcher_agent.py:387-417`.
Step 0 = TRSE, **Step 1 = Window Optimizer, inside which the bidirectional sieve runs** (backend
RANGE-MINER / PWC / ZMQ / legacy), **Step 2 = Scorer Meta-Optimizer** (the older "Step 2.5" label in
this chapter and in `agent_manifests/scorer_meta.json` `description` is the conceptual name; its
`pipeline_step` is `2`), Steps 3–6 as in §3.4. Chapter numbers are not step numbers.

**Original header (retained):**
**Version:** 1.4.0  
**Date:** February 3, 2026  
**Files:** `agents/watcher_agent.py`, `agents/watcher_registry_hooks.py`, `agents/fingerprint_registry.py`  
**Purpose:** Autonomous pipeline orchestration with PRNG attempt tracking

> **Corrected 2026-11-11 — "PRNG attempt tracking".** The registry and hooks exist as modules but
> **no WATCHER code path calls them** at `a15b524` (§5, Known gaps B2). WATCHER itself is version
> **2.0.0** (`agents/watcher_agent.py:12`, `:3467`); "v1.8.0" in S172 documents is the
> **`window_optimizer.json` manifest version** (`agent_manifests/window_optimizer.json:5`), not WATCHER's.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Watcher Agent](#3-watcher-agent)
4. [Fingerprint Registry](#4-fingerprint-registry)
5. [Registry Hooks](#5-registry-hooks)
6. [Agent Manifests](#6-agent-manifests)
7. [Heuristic Evaluation](#7-heuristic-evaluation)
8. [Safety Controls](#8-safety-controls)
9. [CLI Reference](#9-cli-reference)
10. [Troubleshooting](#10-troubleshooting)

Added 2026-11-11:

0. [How this chapter was produced](#0-how-this-chapter-was-produced) (below)
13. [WATCHER → RANGE-MINER: how Step 1 is routed](#13-watcher--range-miner-how-step-1-is-routed)
14. [Known gaps at a15b524](#14-known-gaps-at-a15b524)
15. [Open items](#15-open-items)
16. [Verification declaration (VIR-1…6)](#16-verification-declaration-vir-16)
17. [Closure statement](#17-closure-statement)

> The chapter carries two pre-existing sections numbered **11** and two numbered **12** (one of the
> §12s was a verbatim duplicate). Their numbers are kept so earlier references still resolve; the
> new sections start at 13.

---

## 0. How this chapter was produced

**Added 2026-11-11.** The 2026-02-03 text (v1.4.0, 880 lines) was audited claim by claim against
the code at `a15b524` on VM 101; the audit is `docs/CHAPTER_12_AUDIT.md`. Method:

1. Governance trail first (`docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*`), then the
   fact file, `docs/TODO_CONSOLIDATED.md`, `docs/PIPELINE_BEHAVIOUR_MODEL.md`, then code.
2. Code read this session: `agents/watcher_agent.py` (in the parts named in the anchors),
   `agents/watcher_dispatch.py`, `agents/watcher_registry_hooks.py`, `agents/fingerprint_registry.py`,
   `agents/safety/kill_switch.py`, `agents/contexts/{__init__,window_optimizer_context,bundle_factory}.py`,
   `agents/full_agent_context.py`, `agents/manifest/agent_manifest.py`, all nine files in
   `agent_manifests/` (including the gitignored `definitions.json`), `window_optimizer.py` argparse,
   `window_optimizer_integration_final.py` miner gate, `miner/range_miner_coordinator.py`
   (`run_trial_miner`), `gate12_launch.sh`, `ramdisk_preload.sh`, `run_scorer_meta_optimizer.{py,sh}`.
3. Host state, read-only: `ls` of WATCHER state files, `systemctl list-units`, `crontab -l`, and a
   parse of the 714-record `watcher_decisions.jsonl`.
4. **Nothing was executed.** WATCHER, the pipeline and `--help` were not run; argparse was read.
   "Works" below means the code path is complete on reading.

**How to read the corrections.** Original text is kept. Each disagreement is followed by a
`Corrected 2026-11-11` note naming what the code does and its label (BROKEN /
SWITCHED-OFF-BY-RULING / DESCRIPTION-ONLY); superseded sections carry **SUPERSEDED (2026-11-11)**
and a pointer. All labels are collected in §14.

**What changed underneath this chapter since v1.4.0.**
- **RANGE-MINER** is now the certifying Step-1 sieve engine; PWC and ZMQ are non-certifying. WATCHER
  routes to it by parameter, not by default (§13).
- **The 22-array NPZ contract** is Step 1's real output; WATCHER does not see it — Step 1's WATCHER
  verdict is still decided by the existence of `optimal_window_config.json` (§7, §14 B1).
- **The window-anchor change** (Brief I: `window_anchor` + `generator_phase` split) does not touch
  WATCHER's dispatch code; the miner call site passes `window_anchor`
  (`miner/range_miner_coordinator.py` `run_trial_miner` signature). WATCHER's only Brief-I-era change is
  the Run-4 Route-A pin routing (§13.4). Brief I is implemented; **final acceptance is pending a
  completed production-shape run** (`docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md`).

---

## 1. Overview

### 1.1 What the WATCHER Does

The WATCHER Agent provides autonomous pipeline orchestration:

- **Monitors** pipeline step outputs
- **Evaluates** results against success criteria
- **Decides** next action (PROCEED / RETRY / ESCALATE)
- **Tracks** PRNG attempt history via Fingerprint Registry
- **Prevents** redundant work on known-failed combinations

> **Corrected 2026-11-11.** Monitor / evaluate / decide / act are real: `WatcherAgent.evaluate_results`
> (`agents/watcher_agent.py:913`), `execute_decision` (`:1317`), `run_pipeline` (`:2717`). The last two
> bullets are **not done by WATCHER**: no WATCHER code imports `watcher_registry_hooks` or
> `fingerprint_registry` (the only importer is `agents/registry_inspector.py:30`), and the registry
> DB `agents/data/fingerprint_registry.db` was last written 2026-01-02. **BROKEN — B2** (§14).

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Heuristic Evaluation** | Rule-based decisions without LLM |
| **LLM Evaluation** | GBNF-constrained LLM decisions (optional) |
| **Fingerprint Registry** | Track dataset + PRNG combinations |
| **PRNG Priority Order** | Systematic PRNG family testing |
| **Safety Controls** | Kill switch, max retries, escalation |
| **Dispatch Module** | Selfplay, learning loop, request processing (v1.4.0) |
| **Bundle Factory** | Unified LLM context assembly — 7 bundle types (v1.4.0) |
| **LLM Lifecycle** | Auto stop/restart around GPU-intensive dispatch phases (v1.4.0) |
| **Chapter 13 Integration** | Process watcher_requests/, dispatch retrain/selfplay (v1.4.0) |

> **Corrected 2026-11-11.**
> - *Heuristic / LLM evaluation* exist (`_evaluate_heuristic` `:1269`, `_evaluate_with_llm` `:1039`)
>   but **are not reached for any of the seven step manifests**: all seven declare
>   `"evaluation_type": "file_exists"`, which short-circuits evaluation at `agents/watcher_agent.py:946`
>   before either evaluator runs. Host evidence: of 714 records in `watcher_decisions.jsonl`, 692 are
>   `parse_method: file_exists`, 17 `skip_on_fail`, 1 `llm_router_grammar`, 0 `heuristic`, 4 unlabelled.
>   The short-circuit was intended (`docs/SESSION_CHANGELOG_20260110_WATCHER_FIXES.md` Issue 3).
>   **DESCRIPTION-ONLY — D3.** Its consequence for failed steps is **B1** (§14).
> - *Fingerprint Registry / PRNG Priority Order*: modules exist, not wired — **B2**.
> - *Safety controls*: real, but the halt file is split across two names — **B3** (§8.1).
> - *Dispatch module*: `agents/watcher_dispatch.py`, bound onto `WatcherAgent` by `bind_to_watcher`
>   (`agents/watcher_dispatch.py:761`; call at `agents/watcher_agent.py:3855`). Accurate.
> - *Bundle Factory — "7 bundle types"*: `agents/contexts/bundle_factory.py:587-596` names **eight**
>   (steps 1–6, 13 = Chapter 13, 99 = selfplay evaluation); there is no Step-0 bundle. **DESCRIPTION-ONLY.**
> - *LLM Lifecycle*: `llm_services/llm_lifecycle.py`, initialised at `agents/watcher_agent.py:879-888` (`get_lifecycle_manager()` at `:883`). Accurate.
> - *Not in this table*: run-scoped fleet authority (`_ensure_execution_set`, `:1578`), the P0.5 dataset
>   authority gate (`run_step`, `:1860` onward), the S145 seed-domain wall (Step 1), and the Run-4
>   operator pin seam (§13.4).

### 1.3 Current Status

> **SUPERSEDED (2026-11-11).** The block below is the output of the mock-results test
> `test_watcher_agent.py` `test_all_steps` (`:156-179`) from before the file-exists short-circuit;
> it is not pipeline status. With the seven manifests at `a15b524` a step evaluated through
> `evaluate_results` gets confidence **1.0 or 0.0** (`agents/watcher_agent.py:990-996`; `confidence=1.0 if success else 0.0` at `:992`), so the
> 0.79 / 0.85 / 0.93 values cannot be produced on the live path. The test was not run in this pass
> (U1). For current status see §13.6.

```
✅ Step 1 (Window Optimizer): proceed (conf=0.79)
✅ Step 2 (Scorer Meta-Optimizer): proceed (conf=0.85)
✅ Step 3 (Full Scoring): proceed (conf=0.93)
✅ Step 4 (ML Meta-Optimizer): proceed (conf=0.93)
✅ Step 5 (Anti-Overfit Training): proceed (conf=0.85)
✅ Step 6 (Prediction Generator): proceed (conf=0.93)
```

**Overall Autonomy: ~85%**

---

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      WATCHER AGENT                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Agent      │    │   Heuristic  │    │    LLM       │       │
│  │  Manifests   │───▶│  Evaluator   │◀──▶│  Evaluator   │       │
│  │  (6 steps)   │    │              │    │  (optional)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Decision Engine                          │       │
│  │  PROCEED (conf > 0.70) │ RETRY │ ESCALATE (conf < 0.50)│     │
│  └──────────────────────────────────────────────────────┘       │
│                            │                                     │
│         ┌──────────────────┼──────────────────┐                 │
│         ▼                  ▼                  ▼                  │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐             │
│  │ Fingerprint│    │  History   │    │   Safety   │             │
│  │  Registry  │    │  Logger    │    │  Controls  │             │
│  └────────────┘    └────────────┘    └────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 File Locations

| File | Purpose |
|------|---------|
| `agents/watcher_agent.py` | Main WATCHER implementation |
| `agents/watcher_dispatch.py` | Dispatch functions (selfplay, learning loop, requests) |
| `agents/contexts/bundle_factory.py` | Step awareness bundle assembly engine |
| `agents/watcher_registry_hooks.py` | Registry integration hooks |
| `agents/fingerprint_registry.py` | Dataset fingerprint tracking |
| `llm_services/llm_lifecycle.py` | LLM lifecycle management (stop/restart) |
| `agent_manifests/*.json` | Step configurations |
| `agent_grammars/*.gbnf` | GBNF grammar constraint files (v1.1) |
| `watcher_history.json` | Decision history log |
| `watcher_decisions.jsonl` | Detailed decision audit |
| `watcher_requests/` | Chapter 13 request queue (JSON files) |

> **Corrected 2026-11-11.** The diagram's "Agent Manifests (6 steps)" is **seven**: `trse.json`
> (Step 0) was added at S121 (`STEP_MANIFESTS[0]`, `agents/watcher_agent.py:399`). The two thresholds are
> the heuristic path's (`WatcherConfig.auto_proceed_threshold = 0.70`, `escalate_threshold = 0.50`,
> `:328-329`) and do not apply to the file-exists path (§1.2 note). The "Fingerprint Registry" box
> is not connected to anything in code (B2). Further files the table omits and WATCHER now depends
> on: `agents/safety/kill_switch.py` (halt file, `:60`), `execution_set.py` (run fleet authority,
> imported at `agents/watcher_agent.py:1586`), `miner/dataset_authority.py` (P0.5),
> `pending_approval.json`, `watcher_daemon.pid`, `daemon_state.json` (`WatcherConfig` `:360-362`).
> `agent_manifests/definitions.json` is present on VM 101 but **gitignored and read by no code**
> (`/bin/grep -rln definitions.json` over `*.py`/`*.sh` returns nothing); see §6.4.

---

## 3. Watcher Agent

### 3.1 Configuration

```python
from agents.watcher_agent import WatcherAgent, WatcherConfig

config = WatcherConfig(
    auto_proceed_threshold=0.70,   # Confidence to auto-proceed
    escalate_threshold=0.50,       # Confidence to escalate to human
    max_retries_per_step=3,        # Max retries before escalate
    max_total_retries=10,          # Max retries across pipeline
    poll_interval_seconds=30,      # Status check interval
    step_timeout_minutes=120,      # Max time per step
    use_llm=False,                 # Use heuristic by default
    use_grammar=True               # GBNF constraints for LLM
)

watcher = WatcherAgent(config)
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY).** The field names and numeric defaults match
> `WatcherConfig` (`agents/watcher_agent.py:324-366`), but **`use_llm` defaults to `True`**
> (`:356`), not to heuristic; the example only works because it passes `use_llm=False`. The CLI sets
> `use_llm = not --no-llm` (`:3571-3576`). `WatcherConfig` also carries `halt_file="/tmp/agent_halt"`
> (`:366`), `force_steps` (S152) and `step_timeout_overrides`; the CLI passes
> `{0: 1, 1: 0, 5: 360}` minutes, where `0` means **no timeout** for Step 1 (`:3575`, `:2636-2637`).

### 3.2 Evaluation Flow

```python
# Evaluate a step's results
decision, context = watcher.evaluate_results(
    step=2,
    results={
        'best_trial': {'accuracy': 0.85, 'params': {...}},
        'total_trials': 100,
        'completed_trials': 98
    }
)

# decision.action: "proceed" | "retry" | "escalate"
# decision.confidence: 0.0 - 1.0
# decision.reasoning: Human-readable explanation
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY).** The return is `(AgentDecision, FullAgentContext)`
> (`agents/watcher_agent.py:913-918`); the action field is **`decision.recommended_action`**, not
> `decision.action` (`:1364-1371`). For every current manifest the call returns from the
> file-exists block (`:946-1000`), so `results` is not inspected at all — only the manifest's
> `success_condition` files on disk are.

### 3.3 Decision Actions

| Action | Confidence | Behavior |
|--------|------------|----------|
| PROCEED | ≥ 0.70 | Trigger next step automatically |
| RETRY | 0.50 - 0.70 | Re-run current step with adjustments |
| ESCALATE | < 0.50 | Alert human for review |

> **Corrected 2026-11-11.** This table is the heuristic rule (`_evaluate_heuristic`,
> `agents/watcher_agent.py:1283-1291`): PROCEED needs `success` **and** confidence ≥ 0.70; RETRY needs
> `not success`, confidence ≥ 0.50 and retries left; otherwise ESCALATE. On the file-exists path that
> every manifest uses, the only outcomes are **PROCEED (1.0)** or **ESCALATE (0.0)**, plus
> `skip_on_fail` PROCEED for Step 0 (`:966-988`); RETRY is not produced.
> "Re-run current step **with adjustments**": `_handle_retry` only **logs** the suggested adjustments
> (`:1452-1454`) and re-runs with the same `params`. Only the Step-5 training-health retry builds new
> parameters (`_build_retry_params`, `:2360`). **BROKEN — B5** (dormant: unreachable under the current
> manifests).

### 3.4 Script Execution Mappings (CRITICAL)

The Watcher Agent uses **two dictionaries** to determine what to run for each step. Understanding both is essential for troubleshooting.

#### 3.4.1 STEP_SCRIPTS (Lines 136-145)

**This dictionary defines what script is ACTUALLY EXECUTED:**

```python
STEP_SCRIPTS = {
    1: "window_optimizer.py",
    2: "run_scorer_meta_optimizer.sh",  # NOTE: .sh not .py (PULL architecture)
    3: "generate_full_scoring_jobs.py",
    4: "adaptive_meta_optimizer.py",
    5: "meta_prediction_optimizer_anti_overfit.py",
    6: "reinforcement_engine.py"
}
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY).** At `a15b524` `STEP_SCRIPTS` is at
> **`agents/watcher_agent.py:387-395`** and reads:
> `0: "trse_step0.py"`, `1: "window_optimizer.py"`, `2: "run_scorer_meta_optimizer.sh"`,
> **`3: "run_step3_full_scoring.sh"`** (not `generate_full_scoring_jobs.py`), `4: "adaptive_meta_optimizer.py"`,
> `5: "meta_prediction_optimizer_anti_overfit.py"`, **`6: "prediction_generator.py"`** (not
> `reinforcement_engine.py`). A stale root-level `watcher_agent.py` (72,042 B, its own `STEP_SCRIPTS` at
> `:314`) still sits beside the live file; Beta's R2 ruling records it as known and unrepaired, and
> deletion is not proposed (`docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md`, dispositions
> table). Always read `agents/watcher_agent.py`.

#### 3.4.2 STEP_MANIFESTS (Lines 147-155)

**This dictionary defines where to load default parameters and evaluation criteria:**

```python
STEP_MANIFESTS = {
    1: "window_optimizer.json",
    2: "scorer_meta.json",
    3: "full_scoring.json",
    4: "ml_meta.json",
    5: "reinforcement.json",
    6: "prediction.json"
}
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY).** Content correct for 1–6; the dictionary is at
> `agents/watcher_agent.py:398-406` and adds `0: "trse.json"`. `STEP_NAMES` follows at `:409-417`.

#### 3.4.3 Why Step 2 Uses `.sh` Instead of `.py`

| Script | Architecture | Status |
|--------|--------------|--------|
| `run_scorer_meta_optimizer.py` | Assumes `/shared/ml/` NFS mount | ❌ **BROKEN** |
| `run_scorer_meta_optimizer.sh` | PULL architecture via `scripts_coordinator.py` | ✅ **CORRECT** |

**The `.py` version has hardcoded paths:**
```python
# run_scorer_meta_optimizer.py lines 35-36 (BROKEN)
JOB_DIR = Path("/shared/ml/scorer_evaluation_jobs")      # Does not exist!
RESULTS_DIR = Path("/shared/ml/scorer_evaluation_results") # Does not exist!
```

**The `.sh` version implements PULL architecture correctly:**
1. Calls `generate_scorer_jobs.py` to create trial jobs
2. SCPs data to remote nodes
3. Launches via `scripts_coordinator.py` (per January 3, 2026 fix)
4. Workers write results locally on each rig
5. Coordinator PULLS results back via SCP
6. Aggregates and reports to Optuna

**See Chapter 3 Section 3 for full PULL architecture details.**

> **Corrected 2026-11-11.** The `/shared/ml` paths are at `run_scorer_meta_optimizer.py:43-44`
> (not 35-36); the `.sh` calls `generate_scorer_jobs.py` (`run_scorer_meta_optimizer.sh:229`, `:239`)
> and `scripts_coordinator.py` (`:264-266`). **What changed:** on RANGE-MINER output the `.sh` is no
> longer "✅ CORRECT" to run through WATCHER. It invokes the legacy converter
> `convert_survivors_to_binary.py` (`:87`) and `mv`s a regular file onto the finalizer-owned
> survivors symlink (`:97`). The owner/Beta record says the legacy converter is not invoked until
> D3.0-B closes (`docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` §1), and every production-class launch
> pins `--start-step 1 --end-step 1` for this reason (`gate12_launch.sh:73-76`, `:377-378`).
> **SWITCHED-OFF-BY-RULING — S2** (§14).

#### 3.4.4 How Execution Works (Line 748)

```python
def run_step(self, step: int):
    # Script comes from STEP_SCRIPTS (hardcoded)
    script = STEP_SCRIPTS.get(step)
    
    # Default params come from manifest (STEP_MANIFESTS)
    manifest_name = STEP_MANIFESTS.get(step)
    manifest = load_manifest(manifest_name)
    default_params = manifest.get('default_params', {})
    
    # Execute the script
    subprocess.run([script] + build_args(default_params))
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY).** `run_step` is at `agents/watcher_agent.py:1705`, and
> the sketch above omits most of what it now does, in this order:
> 1. freeze the run's **execution set** once per process (`_ensure_execution_set`, `:1578`; backend chosen
>    `use_range_miner` → miner, else PWC, else ZMQ, else legacy, `:1604-1611`);
> 2. preflight (HARD failures block, SOFT continue);
> 3. freshness check, bypassed by `--force-step N`; a fresh output **skips** the step;
> 4. load the manifest's `default_params` and merge caller params **step-scoped** — a key not declared in
>    `default_params` is dropped (`:1811-1820`);
> 5. Step 1 only: merge the operator warm-start pin, if any (`:1828`, §13.4);
> 6. P0.5 dataset authority: rewrite `lottery_file` / `lottery_data` / `lottery_history` to the frozen
>    immutable dataset path and verify the fleet (`:1860` onward);
> 7. Step 1 only: S145 certified-coverage cursor and seed-domain wall;
> 8. build argv from `actions[].args_map` (booleans are flag-only, `True` emits the flag, `False`
>    omits it, `None`/`''` are skipped, `:2140-2155`), strip internal-only `warm_start_*` keys;
> 9. run with live streaming via `subprocess.Popen` (`_run_step_streaming`, `:2583`; `:2601`).

#### 3.4.4a Manifest `arg_style`

> **Added 2026-11-11.** A manifest may declare `"arg_style": "positional"` with `positional_args` and
> `flag_args` (`agents/watcher_agent.py:2086-2101`); `scorer_meta.json` does. Others use named
> `--key value` arguments.

#### 3.4.5 Future Improvement

Currently `STEP_SCRIPTS` is hardcoded. A more robust design would read the script from the manifest's `actions[0].script` field:

```python
# Current (hardcoded - error-prone):
script = STEP_SCRIPTS.get(step)

# Future (manifest-driven - single source of truth):
manifest = load_manifest(STEP_MANIFESTS.get(step))
script = manifest['actions'][0]['script']
```

This would eliminate the dual-dictionary pattern and ensure consistency.

> **Verified 2026-11-11 (accurate).** `STEP_SCRIPTS` is still hardcoded; `run_step` reads the script
> from it (`agents/watcher_agent.py:1722`) and uses the manifest's `actions[].args_map` only to name
> CLI flags (`:2104-2113`). Recorded as a design note, not a proposal.

---

## 4. Fingerprint Registry

### 4.1 Purpose

Track dataset + PRNG combinations to prevent redundant work:

- **Fingerprint:** SHA256 hash of dataset characteristics
- **Attempts:** List of PRNG types tried on this fingerprint
- **Outcomes:** Success/failure history per combination

### 4.2 Usage

```python
from agents.fingerprint_registry import FingerprintRegistry

registry = FingerprintRegistry()

# Create fingerprint from dataset
fingerprint = registry.create_fingerprint(
    dataset_path="daily3.json",
    draw_count=1000,
    session_types=["midday", "evening"]
)

# Check if PRNG already tried
status = registry.get_status(fingerprint, "mt19937")
# status: "untried" | "in_progress" | "success" | "failed"

# Record attempt outcome
registry.record_attempt(
    fingerprint=fingerprint,
    prng_type="mt19937",
    outcome="failed",
    details={"best_match_rate": 0.12}
)
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY).** The example calls an API that does not exist.
> `FingerprintRegistry` (`agents/fingerprint_registry.py:76`) is an SQLite store at
> `agents/data/fingerprint_registry.db` (`:99`) with **no `create_fingerprint` and no `get_status`**. Its
> methods are `record_attempt(fingerprint, prng_type, outcome, sidecar=None, run_id=None,
> signal_confidence=0.0, exit_code=0, duration_seconds=0, policy_version, watcher_version)` (`:205-216`;
> no `details` argument; outcomes are upper-case, e.g. `SUCCESS`, `DEGENERATE_SIGNAL`, `:224`),
> `is_combination_tried` (`:321`), `get_entry` (`:341`), `get_untried_prngs` (`:379`),
> `get_attempt_count`, `get_failure_count`, `expire_old_entries`, `clear_fingerprint`, `get_stats`,
> `export_to_json`. The fingerprint is **not computed by the registry**: by design it is the
> Step-5 sidecar's `data_context.fingerprint_hash` — `sha256` of
> `{train_start}:{train_end}|{holdout_start}:{holdout_end}|{survivor_count}|{prng_type}|{mod}`, truncated
> to 8 (`agent_manifests/definitions.json:154-158`, gitignored). "Dataset characteristics" therefore
> means the training/holdout windows and survivor count, not the dataset file. Whether the module is
> used: B2.

---

## 5. Registry Hooks

### 5.1 Purpose

Bridge between WATCHER and Fingerprint Registry:

- **Pre-run checks:** Should we try this PRNG?
- **Post-run recording:** Log the outcome
- **Next PRNG selection:** Get untried PRNG in priority order

### 5.2 PRNG Priority Order

```python
PRNG_PRIORITY_ORDER = [
    # Forward fixed (most common)
    "java_lcg", "mt19937", "xorshift32", "pcg32", "lcg32",
    "minstd", "xorshift64", "xorshift128", "xoshiro256pp", 
    "philox4x32", "sfc64",
    
    # Forward hybrid
    "java_lcg_hybrid", "mt19937_hybrid", "xorshift32_hybrid", ...
    
    # Reverse fixed
    "java_lcg_reverse", "mt19937_reverse", ...
    
    # Reverse hybrid
    "java_lcg_hybrid_reverse", "mt19937_hybrid_reverse", ...
]
```

### 5.3 Hook Methods

```python
from agents.watcher_registry_hooks import WatcherRegistryHooks

hooks = WatcherRegistryHooks()

# Before running pipeline
decision = hooks.pre_run_check(fingerprint, "mt19937")
# decision.action: "PROCEED" | "SKIP_PRNG" | "REJECT"
# decision.suggested_prng: Next untried PRNG if skipping

# After pipeline completes
hooks.post_run_record(
    fingerprint=fingerprint,
    prng_type="mt19937",
    outcome="success",
    sidecar_path="results/run_123.meta.json"
)

# Get next untried PRNG
next_prng = hooks.get_next_prng(fingerprint)
```

> **Corrected 2026-11-11.**
> - **Accurate:** `PRNG_PRIORITY_ORDER` (44 names, forward/hybrid/reverse/hybrid-reverse) at
>   `agents/watcher_registry_hooks.py:50-67`; `pre_run_check` `:151`, `post_run_record` `:252`
>   (signature matches the example), `get_next_prng` `:325`.
> - **DESCRIPTION-ONLY:** `PreRunDecision.action` is one of `PROCEED`, `SKIP_PRNG`,
>   `REJECT_DATA_WINDOW`, `EXPAND_HOLDOUT` (`:91`, `:181-240`), not `REJECT`. Policy thresholds:
>   `MAX_PRNG_ATTEMPTS=5`, `MIN_HOLDOUT_DRAWS=500`, `MAX_TEMPORAL_GAP=1000`, `FINGERPRINT_TTL_DAYS=7`,
>   `MIN_SIGNAL_CONFIDENCE=0.3` (`:75-82`).
> - **BROKEN — B2.** "Bridge between WATCHER and Fingerprint Registry": nothing calls the bridge.
>   Searched: all tracked and untracked `*.py` (`/bin/grep -rn` for `watcher_registry_hooks`,
>   `WatcherRegistryHooks`, `fingerprint_registry`, `FingerprintRegistry`) — the only importer outside
>   the two modules is `agents/registry_inspector.py:30`; `agents/watcher_agent.py` and
>   `agents/watcher_dispatch.py` import neither. The wiring design is in
>   `docs/PROPOSAL_Modular_Step_Automation_Framework_v1_1.md` (`registry_hooks.pre_run_check` in
>   `run_step`), whose status is **"DRAFT - Awaiting Team Beta Approval"** (`:6`) — a draft, not a
>   ruling that holds it back. No `TB_RULING_*` / `TEAM_ALPHA_*` file mentions the hooks. Host: the
>   DB file's mtime is 2026-01-02. The hooks also import `fingerprint_registry` as a bare module after
>   inserting their own directory on `sys.path` (`:38-40`). Note also that the 44-name order is wider
>   than the sieve: TFM sieves `java_lcg`, and RANGE-MINER raises `NotImplementedError` for the five
>   uncovered base families (CLAUDE.md §7), so `get_next_prng` could suggest families that cannot run.

---

## 6. Agent Manifests

### 6.1 Location

`agent_manifests/` directory contains one JSON file per step:

| Manifest | Step | Purpose |
|----------|------|---------|
| `window_optimizer.json` | 1 | Bayesian parameter optimization |
| `scorer_meta.json` | 2.5 | Scorer hyperparameter tuning |
| `full_scoring.json` | 3 | Distributed feature extraction |
| `ml_meta.json` | 4 | Neural network optimization |
| `reinforcement.json` | 5 | Anti-overfit model training |
| `prediction.json` | 6 | Final prediction generation |

> **Corrected 2026-11-11 (DESCRIPTION-ONLY).** Seven step manifests, parsed this session:
>
> | Manifest | `pipeline_step` | `version` | `success_condition` (file-exists) |
> |---|---|---|---|
> | `trse.json` | 0 | 1.15.1 | `trse_context.json` (`skip_on_fail: true`) |
> | `window_optimizer.json` | 1 | **1.8.0** | `optimal_window_config.json` |
> | `scorer_meta.json` | **2** (described as "Step 2.5") | 1.3.0 | `optimal_scorer_config.json` |
> | `full_scoring.json` | 3 | 1.3.0 | `survivors_with_scores.json` |
> | `ml_meta.json` | 4 | 2.0.0 | `reinforcement_engine_config.json` |
> | `reinforcement.json` | 5 | 1.10.0 | `models/reinforcement/best_model.meta.json` |
> | `prediction.json` | 6 | 1.5.0 | `predictions/next_draw_prediction.json` |
>
> All seven declare `evaluation_type: "file_exists"`, `disable_llm_parsing: true`,
> `disable_heuristic_parsing: true`. Only `evaluation_type` (and `skip_on_fail`) is read by code
> (`agents/watcher_agent.py:946`, `:968`); the two `disable_*` keys have no reader (searched all
> `*.py`). The directory also holds `scorer_meta.json.bak` (stale) and the gitignored
> `definitions.json` (§6.4). Step 1's "Bayesian parameter optimization" label is the Optuna layer;
> the **sieve runs inside Step 1** (§13).

### 6.2 Manifest Structure

```json
{
  "agent_name": "window_optimizer_agent",
  "description": "Bayesian window optimization",
  "pipeline_step": 1,
  
  "inputs": ["lottery_file", "seed_count"],
  "outputs": ["bidirectional_survivors.json", "optimal_window_config.json"],
  
  "actions": [{
    "type": "run_script",
    "script": "window_optimizer.py",
    "args_map": {
      "lottery-file": "lottery_file",
      "trials": "window_trials"
    }
  }],
  
  "follow_up_agents": ["scorer_meta_agent"],
  "success_condition": "optimal_window_config.json exists",
  "retry": 2
}
```

### 6.3 Schema Versioning

Manifests support both v1.0 (simple) and v1.1 (extended) formats:

```python
# v1.0: Simple string outputs
"outputs": ["model.pth", "model.meta.json"]

# v1.1: Rich output descriptors (auto-converted to strings)
"outputs": [
  {"name": "model", "file_pattern": "model.pth", "required": true}
]
```

The `AgentManifest` Pydantic model normalizes both formats.

> **Verified 2026-11-11.** §6.2 is illustrative (DESCRIPTION-ONLY: the live `window_optimizer.json`
> has a list `success_condition`, `default_params`, `parameter_bounds`, an `args_map` of 39 entries and
> no `lottery_file`→`bidirectional_survivors.json` contract — see
> `agent_manifests/window_optimizer.json:20-75`). §6.3 is accurate: `AgentManifest`
> (`agents/manifest/agent_manifest.py:119`) normalises `outputs` in a `field_validator`
> (`:187-189`). Note that `run_step` reads the manifest JSON directly, not through this model
> (`agents/watcher_agent.py:1797-1807`).

### 6.4 `agent_manifests/definitions.json` (gitignored)

**Added 2026-11-11.** Present on VM 101 (6,635 B, mtime 2026-01-01), matched by `.gitignore:41`
(`*.json`), never tracked (`git log --all -- agent_manifests/definitions.json` is empty), so a fresh
clone does not have it. It is a **schema document, not a step manifest**: `schema_version` 1.1.0,
`pipeline_steps` (Step 5 `emits` `signal_quality` / `data_context` / `feature_schema`; Step 6 `consumes`
them and lists `predictions/next_draw_prediction.json`, `prediction_pools.json`, `ranked_survivors.json`),
`watcher_protocol` (a degenerate-fingerprint registry at `agent_state/degenerate_fingerprints.json`, a
five-step recovery workflow, the fingerprint formula quoted in §4), and `sidecar_schema` 3.1.0. **No
code reads it** (searched `*.py` and `*.sh`, tracked and untracked, with `/bin/grep`), and
`agent_state/` does not exist on the host. Its recovery workflow is therefore design-only. No ruling
on it was found (`docs/PROJECT_FILE_CATALOG.md` §7 gap 4 records only that it is untracked).

---

## 7. Heuristic Evaluation

### 7.1 When Used

Heuristic evaluation is used when:
- `use_llm=False` in config
- LLM server unavailable
- Fast decisions needed

### 7.2 Step-Specific Heuristics

| Step | Key Metrics | Thresholds |
|------|-------------|------------|
| 1 | survivor_count, bidirectional_count | >1000 survivors |
| 2.5 | best_accuracy, completed_trials | >50 trials, acc >0.7 |
| 3 | scored_count, mean_score | >90% scored |
| 4 | best_r2, validation_r2 | r2 >0.5 |
| 5 | test_r2, overfit_ratio | ratio <1.5 |
| 6 | prediction_confidence | conf >0.6 |

### 7.3 Confidence Calculation

```python
def calculate_confidence(step: int, results: dict) -> float:
    """Calculate confidence score 0.0 - 1.0"""
    
    if step == 1:
        survivors = results.get('survivor_count', 0)
        if survivors > 10000:
            return 0.95
        elif survivors > 1000:
            return 0.80
        elif survivors > 100:
            return 0.60
        else:
            return 0.30
    
    # ... similar logic for other steps
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY — D3).**
> - **§7.1 "When used".** The precondition is missing. Evaluation runs in this order
>   (`agents/watcher_agent.py:913-1036`): (1) if the step's manifest exists and declares
>   `evaluation_type: "file_exists"` — **all seven do** — return PROCEED/1.0 when every
>   `success_condition` file passes `evaluate_file_exists` (size and JSON-structure validation,
>   `FILE_VALIDATION_CONFIG` `:157`), else ESCALATE/0.0 (or PROCEED/0.0 with `skip_on_fail`); (2) safety
>   check; (3) LLM if `use_llm`; (4) heuristic. Steps (2)–(4) are reached only when a manifest is
>   missing (path is CWD-relative, `agents/watcher_agent.py:934-938`) or declares another type.
> - **§7.2 thresholds** and **§7.3 `calculate_confidence`** do not exist as written. The heuristic
>   delegates to the per-step context (`agents/full_agent_context.py:312-325`, factories
>   `agents/contexts/__init__.py:41-49`); Step 1's is rate-based — zero bidirectional survivors → (False,
>   0.1); `bidirectional_rate ≤ 0.02` and overlap ≥ 0.25 → 0.95; ≤ 0.10 → 0.75; ≤ 0.30 → (False, 0.55);
>   else 0.30 (`agents/contexts/window_optimizer_context.py:110-144`) — not survivor-count bands.
> - **Consequence that matters (BROKEN — B1, governed):** the file-exists verdict never reads the
>   step's own result. `run_step` returns `{"success": False, "return_code": …}` on a non-zero exit
>   (`:2177-2185`), and evaluation ignores it. After the Gate-12 attempt-3 Step 1 failed with code 1,
>   WATCHER reported `file_exists`, confidence 1.00, PASSED and "Triggering Step 2", because
>   `optimal_window_config.json` had been written before the finalizer raised. Recorded as open in
>   `docs/TFM_PROJECT_FACTS_SKILL.md` §2.34 ("an explicit subprocess/finalization failure must dominate
>   file-existence heuristics … Required before Phase-7 autonomy"), §2.51 item 9, and
>   `docs/SESSION_CHANGELOG_20260811_CLEANTREE_ADMISSION.md:220-223`. Not repaired at `a15b524`.

---

## 8. Safety Controls

### 8.1 Kill Switch

Create a halt file to stop WATCHER immediately:

```bash
# Create halt
python3 -m agents.watcher_agent --halt "Reason for stopping"

# Clear halt
rm watcher_halt.flag
```

> **Corrected 2026-11-11 — BROKEN — B3.** WATCHER has **two halt files**:
> - `--halt` / `create_halt` write **`/tmp/agent_halt`** (`agents/safety/kill_switch.py:60`, `:367-371`;
>   `WatcherConfig.halt_file`, `agents/watcher_agent.py:366`). This is what `run_pipeline`
>   (`:2777`, `:2804`), the daemon loop (`:3175`) and escalation (`:1473`) check or write. Clear it with
>   **`--clear-halt`** (`:3579-3581`), not `rm watcher_halt.flag`.
> - The dispatch module checks a different file, **`watcher_halt.flag`** in the CWD, before selfplay
>   and learning-loop dispatch and request processing (`agents/watcher_dispatch.py:126`, `:296`, `:386`).
>   Nothing in the tracked code creates it.
>
> So `--halt` does **not** stop `--dispatch-selfplay` / `--dispatch-learning-loop`, and the documented
> `rm watcher_halt.flag` does not clear a `--halt`. (The daemon's own loop does pause on
> `/tmp/agent_halt` before it scans requests, `:3175-3183`.) No governance document records the split.
> Host, read-only: neither file exists as seen from this session (`/tmp` visibility from the agent
> sandbox is not guaranteed — treat the `/tmp` result as UNAVAILABLE).

### 8.2 Retry Limits

| Limit | Default | Purpose |
|-------|---------|---------|
| `max_retries_per_step` | 3 | Prevent infinite loops on single step |
| `max_total_retries` | 10 | Prevent runaway pipeline |

### 8.3 Escalation

When confidence < 0.50 or retries exhausted:

1. WATCHER logs escalation to `watcher_decisions.jsonl`
2. Pipeline halts
3. Human review required
4. Resume with `--resume` flag after fixing issue

> **Corrected 2026-11-11.** §8.2 is accurate: `max_retries_per_step=3`, `max_total_retries=10`
> (`agents/watcher_agent.py:332-333`), enforced in `_handle_retry` (`:1440-1450`). §8.3 steps 1–3 are
> accurate: `execute_decision` appends to `watcher_decisions.jsonl` (`:1340-1350`), and
> `_handle_escalate` writes the halt file, prints the review banner and sends a Telegram CRITICAL
> (`:1459-1489`). **Step 4 is DESCRIPTION-ONLY: there is no `--resume` flag** (argparse
> `:3466-3565`). The printed resume command is
> `python3 watcher_agent.py --clear-halt --run-pipeline --start-step N` (`:1534`). Do not confuse it with
> `window_optimizer.py --resume-study` / `resume_checkpoint`, which are Optuna/miner checkpoint options.

---

## 9. CLI Reference

### 9.1 Commands

```bash
# Check status
python3 -m agents.watcher_agent --status

# Evaluate a result file
python3 -m agents.watcher_agent --evaluate results.json

# Run full pipeline (no LLM)
python3 -m agents.watcher_agent --run-pipeline --no-llm

# Run with LLM
python3 -m agents.watcher_agent --run-pipeline

# Create halt
python3 -m agents.watcher_agent --halt "Reason"

# Resume after halt
python3 -m agents.watcher_agent --run-pipeline --resume
```

> **Corrected 2026-11-11 (DESCRIPTION-ONLY).** `--resume` (above) and `--verbose` (§10.2) are **not
> defined** (argparse `agents/watcher_agent.py:3466-3565`); argparse would reject them. The complete
> flag set at `a15b524` is:
>
> | Flag | Effect | Anchor |
> |---|---|---|
> | `--daemon` | long-running daemon, PID file, signal handlers, request scan | `:3471`, `run_daemon` `:3115` |
> | `--run-pipeline` `--start-step N` `--end-step M` | run steps N..M (defaults **1..6**) | `:3476`, `:3481-3491` |
> | `--params '<json>'` | override manifest defaults (step-scoped, §3.6); the **only** operator pin seam (§13.4) | `:3531`, `:3820` |
> | `--force-step N` (repeatable) | bypass freshness for step N | `:3537-3545` |
> | `--evaluate FILE` / `--status` / `--explain [N]` | read-only reporting | `:3493`, `:3498`, `:3561` |
> | `--halt [REASON]` / `--clear-halt` | write / remove `/tmp/agent_halt` | `:3508`, `:3503` |
> | `--stop` | SIGTERM the daemon via its PID file | `:3559` |
> | `--approve` | run a pending Chapter-13 approval: `run_pipeline(min(steps), max(steps))` | `:3563`, `:3774` |
> | `--no-llm` / `--no-grammar` / `--threshold X` | evaluator settings | `:3515-3529` |
> | `--dispatch-selfplay [--episodes N]` / `--dispatch-learning-loop [SCOPE]` / `--process-requests` | Phase 7 Part B dispatch | `:3547-3555` |
> | `--dry-run` | **only** affects the three dispatch commands (`:3836`, `:3840`, `:3845`) — it does **not** make `--run-pipeline` a dry run | `:3556` |
>
> **Operator warning, from the code:** `--run-pipeline --dry-run` executes the pipeline for real.
> `--run-pipeline` spins the fleet; it is Michael-initiated only (CLAUDE.md rule 3).
> `--approve` runs the contiguous range `min..max`, so `[3, 5, 6]` also runs Step 4
> (`docs/CHAPTER_13_AUDIT.md` B6).

### 9.2 Verification Test

```bash
# Run full verification test
python3 test_watcher_agent.py

# Expected output:
# ✅ Step 1 (Window Optimizer): proceed (conf=0.79)
# ✅ Step 2 (Scorer Meta-Optimizer): proceed (conf=0.85)
# ... all 6 steps pass
# ALL TESTS PASSED ✅
```

> **UNVERIFIED (U1), 2026-11-11.** `test_watcher_agent.py` exists at the repo root (326 lines,
> `test_all_steps` `:156`). It was **not run** in this documentation pass. Its expected confidences
> predate the file-exists short-circuit; with the current manifests its results depend on which
> output files exist in the CWD (§7 note). Would verify: one run, output attached.

---

## 10. Troubleshooting

### 10.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Manifest validation error" | Schema mismatch | Check manifest JSON format |
| "LLM unavailable" | Server not running | Use `--no-llm` or start LLM |
| "Halt flag exists" | Previous halt | Remove `watcher_halt.flag` |
| "Max retries exceeded" | Persistent failure | Review logs, fix root cause |
| "/shared/ml/ not found" | Step 2 using wrong script | Ensure line 138 uses `.sh` not `.py` |

### 10.2 Debug Mode

```bash
# Verbose output
python3 -m agents.watcher_agent --run-pipeline --verbose

# Check decision history
cat watcher_decisions.jsonl | jq .
```

### 10.3 Log Files

| File | Contents |
|------|----------|
| `watcher_history.json` | Run history summary |
| `watcher_decisions.jsonl` | Detailed decision audit (JSONL) |
| `logs/watcher_agent.log` | Application logs |

> **Corrected 2026-11-11 (DESCRIPTION-ONLY).**
> - §10.1 "Halt flag exists → remove `watcher_halt.flag`": the pipeline's halt is `/tmp/agent_halt`;
>   use `--clear-halt` (B3). "/shared/ml/ not found → ensure line 138 uses `.sh`": `STEP_SCRIPTS[2]` is
>   at `agents/watcher_agent.py:390` and is `.sh`.
> - §10.2: `--verbose` does not exist (see §9.1 note). `watcher_decisions.jsonl` is real (714 records on
>   VM 101, last 2026-08-23).
> - §10.3: **there is no `logs/watcher_agent.log`.** Logging is `logging.basicConfig` with no file
>   handler (`agents/watcher_agent.py:311-315`), i.e. stderr. Launches redirect it: `> log 2>&1`
>   (`gate12_launch.sh:377-383`). Additional state files: `watcher_history.json`, `daemon_state.json`,
>   `watcher_daemon.pid`, `pending_approval.json` (host: present, `status: failed`, request
>   `ch13_20260215_084821`), `decisions/*.json`, `watcher_requests/archive/`.

---

---

## 10.5 Ramdisk Preload Limitation

> **SUPERSEDED (2026-11-11) — DESCRIPTION-ONLY.** The limitation below no longer matches the code.
> `run_step3_full_scoring.sh` sources `ramdisk_preload.sh` (`:31-36`), whose `preload_ramdisk` loops over
> **every node in `distributed_config.json`** (`ramdisk_preload.sh:25-40`) and `scp`s the files to each
> remote node (`:130-150`). "Standalone mode" vs "WATCHER-managed mode" (`:86-87`) only decides whether
> an existing ramdisk directory is cleared first (`:114`, `:136`). A remote copy failure is logged and
> non-fatal for that node (`:150`). One live caveat, not a defect: `distributed_config.json` holds the
> **bare-metal** rig addresses (the default boot target, CLAUDE.md §3); when the rigs are booted into
> Proxmox those addresses are down and the remote copies fail. Ramdisk code was not traced further in
> this pass. The original text is kept below.

**Added: 2026-01-22**

### Standalone Mode Behavior

When WATCHER runs Step 3, it performs ramdisk preload:

```
[INFO] Ramdisk preload for Step 3 (4 files)...
[INFO] Target: /dev/shm/prng/step3
[INFO] Standalone mode
[INFO] Ramdisk preload complete
```

**"Standalone mode" means:** Only the local node (Zeus) is populated.

Remote nodes (rig-6600, rig-6600b, rig-6600c) are **NOT** automatically populated.

### Impact on Distributed Execution

| Node | Ramdisk Populated | Jobs Will... |
|------|-------------------|--------------|
| Zeus (local) | ✅ Yes | Work |
| rig-6600 | ❌ No | Fail immediately |
| rig-6600b | ❌ No | Fail immediately |
| rig-6600c | ❌ No | Fail immediately |

### Workaround

Before running Step 3 via WATCHER, manually populate remote ramdisks:

```bash
# Run on Zeus before WATCHER pipeline
for node in 192.168.3.120 192.168.3.154; do
    ssh $node "mkdir -p /dev/shm/prng/step3"
    scp train_history.json holdout_history.json $node:/dev/shm/prng/step3/
done
```

### Future Enhancement

TODO: Modify `run_step3_full_scoring.sh` ramdisk preload to:
1. Detect distributed mode
2. SCP files to all configured nodes in `distributed_config.json`
3. Verify files exist before proceeding


## 11. Chapter Summary

**Chapter 12: WATCHER Agent & Fingerprint Registry** covers:

| Component | Lines | Purpose |
|-----------|-------|---------|
| `watcher_agent.py` | ~500 | Main orchestrator |
| `watcher_registry_hooks.py` | ~350 | Registry integration |
| `fingerprint_registry.py` | ~250 | Dataset tracking |
| Agent manifests | 6 files | Step configurations |

**Key Points:**
- Heuristic evaluation provides fast, reliable decisions
- LLM evaluation available for complex cases
- Fingerprint Registry prevents redundant PRNG attempts
- Safety controls ensure human oversight
- All 6 pipeline steps validated and working
- **STEP_SCRIPTS dict (line 136) determines actual script execution**
- **Step 2 must use `.sh` (PULL architecture) not `.py` (broken NFS paths)**

> **Corrected 2026-11-11.** Sizes at `a15b524`: `agents/watcher_agent.py` **3,859** lines,
> `agents/watcher_registry_hooks.py` **477**, `agents/fingerprint_registry.py` **655**, plus
> `agents/watcher_dispatch.py` 937; manifests: **7** step files (DESCRIPTION-ONLY). "Fingerprint
> Registry prevents redundant PRNG attempts": not wired (B2). "Safety controls ensure human
> oversight": real, with the halt-file split B3. "All 6 pipeline steps validated and working" is not
> a claim this pass can verify, and on RANGE-MINER output Steps 2–6 are **not** run through WATCHER
> today (S2; launches pin `--end-step 1`). "STEP_SCRIPTS dict (line 136)": now `:387-395`.

---

## 12. Changelog

### v1.5.0 (November 11, 2026) — correction pass at `a15b524`
- Added the §1 framing block, status header, §0 (how produced), §6.4 (`definitions.json`), §13
  (WATCHER → RANGE-MINER routing, Route-A pin, pool=8 / soak status), §14 Known gaps, §15 open items,
  §16 VIR declaration, §17 closure.
- Correction notes in place throughout; §1.3, §10.5, the v1.5.0 manifest note and both PWC §12
  sections marked SUPERSEDED. Nothing deleted. Audit: `docs/CHAPTER_12_AUDIT.md`.
- Versions between 1.1.0 and 1.4.0 (1.2.0 Phase 9A, 1.3.0, 1.4.0 dispatch) were never logged here;
  see the un-folded addenda named in §15.

### v1.1.0 (January 9, 2026)
- Added Section 3.4: Script Execution Mappings (CRITICAL)
- Documented STEP_SCRIPTS vs STEP_MANIFESTS dictionaries
- Explained why Step 2 uses `.sh` instead of `.py`
- Added troubleshooting entry for `/shared/ml/` error

### v1.0.0 (January 3, 2026)
- Initial release

---

*End of Chapter 12: WATCHER Agent & Fingerprint Registry*


---

## 3.6 Manifest Parameter Precedence (CRITICAL)

**Added: 2026-01-25** — Lesson learned from OOM debugging.

### The Rule

`agent_manifests/*.json` `default_params` **override** script defaults!

| Priority | Source | Example |
|----------|--------|---------|
| 1 (highest) | CLI `--params '{...}'` | `--params '{"chunk_size": 500}'` |
| 2 | `agent_manifests/*.json` `default_params` | `"chunk_size": 1000` |
| 3 (lowest) | Script hardcoded default | `CHUNK_SIZE=5000` in .sh |

### Why This Matters

When WATCHER runs a step, it:
1. Loads `agent_manifests/{step}.json`
2. Reads `default_params` section
3. Passes ALL params to the script
4. Script's internal defaults are **never used**

> **Corrected 2026-11-11 (DESCRIPTION-ONLY).** The precedence order is right, but the merge is
> **step-scoped**: a `--params` key is merged only if it is declared in that step's `default_params`;
> any other key is dropped with a debug log (`agents/watcher_agent.py:1811-1820`). So (a) a CLI key the
> manifest does not declare never reaches the script — e.g. `seed_count` is dropped and `max_seeds`
> must be used (`gate12_launch.sh:68-70`); (b) script defaults **are** used for every argument the
> manifest does not declare. One deliberate exception: the seven Step-1 warm-start keys, which are not
> in `default_params` and enter only through the operator pin channel (§13.4). `output_file` is always
> removed (`:1838`). The 2026-01-25 fix is still in place: `full_scoring.json` `default_params.chunk_size`
> is `1000` (`agent_manifests/full_scoring.json:122`) and `CHUNK_SIZE=1000` in
> `run_step3_full_scoring.sh:70`.

### Real Example (2026-01-25 OOM Bug)

**Symptom:** Step 3 ran with `chunk_size=5000` despite script having `CHUNK_SIZE=1000`

**Root cause:**
```json
// agent_manifests/full_scoring.json
"default_params": {
    "chunk_size": 5000,  // ← THIS overrode script default
    ...
}
```

**Fix:** Changed manifest to `"chunk_size": 1000`

### Debugging Checklist

When a WATCHER-run step uses unexpected parameters:

1. ✅ Check `agent_manifests/{step}.json` `default_params` FIRST
2. ✅ Check script hardcoded defaults SECOND
3. ✅ Check CLI `--params` if passed

### Best Practice

Keep manifest `default_params` and script defaults **in sync**:

```bash
# Verify consistency
grep "chunk_size" agent_manifests/full_scoring.json
grep "CHUNK_SIZE" run_step3_full_scoring.sh
```


### ⚠️ CRITICAL LESSON (2026-01-25)

**When fixing params via CLI testing, ALWAYS update the manifest too!**

| What Happens | CLI Run | WATCHER Run |
|--------------|---------|-------------|
| Config source | Script defaults / CLI args | Manifest `default_params` |
| Your fix applies? | ✅ Yes | ❌ NO! |

**Pattern that burns you:**
1. Test via CLI: `python3 script.py --param 450` → Works!
2. Update shell script: `--param 450` → Works!
3. Run via WATCHER → Uses manifest's old value (25000) → Fails!

**Fix:** After ANY successful CLI parameter tuning:
```bash
# Immediately update the manifest
sed -i 's/"param": OLD/"param": NEW/' agent_manifests/step.json
```

---

## 11. Phase 9A: Selfplay Request Handling

**Added:** 2026-01-30 | **Version:** 1.2.0

### 11.1 Overview

Phase 9A adds selfplay request handling. WATCHER monitors `watcher_requests/` for retrain requests from Chapter 13.

### 11.2 Request Directory
```
watcher_requests/
├── selfplay_20260130_021543.json
├── selfplay_20260130_031022.json
└── ...
```

**Request Schema:**
```json
{
  "request_id": "selfplay_20260130_021543",
  "request_type": "selfplay_retrain",
  "created_at": "2026-01-30T02:15:43.860849+00:00",
  "status": "pending",
  "reason": "Diagnostics flagged SELFPLAY_RECOMMENDED",
  "source_policy": "policy_selfplay_20260129_234039_ep010",
  "priority": "normal",
  "requested_by": "chapter_13_triggers",
  "requires_watcher_approval": true
}
```

### 11.3 Authorization Flow
```
Chapter 13                    WATCHER                      Selfplay
──────────                    ───────                      ────────
request_selfplay()
        │
        └──► watcher_requests/*.json
                    │
                    └──► validate_request()
                              │
                              ▼
                         APPROVE / REJECT
                              │
                              ▼ (if APPROVE)
                    dispatch_selfplay() ──────────► selfplay_orchestrator.py
```

### 11.4 Validation Rules

| Check | Condition | Action if Failed |
|-------|-----------|------------------|
| Request age | < 24 hours | REJECT (stale) |
| Priority valid | "normal" or "high" | REJECT (invalid) |
| Cooldown elapsed | > 1 hour since last | DEFER |
| Resources available | Cluster not busy | DEFER |

### 11.5 Invariants

- **WATCHER authorizes, does not learn** — No policy decisions
- **Selfplay cannot self-dispatch** — Must go through WATCHER
- **Requests are append-only** — Audit trail preserved
- **Cooldown enforced** — Prevent selfplay storms

> **Corrected 2026-11-11.**
> - **Accurate:** the daemon scans `watcher_requests/*.json` every cycle, excluding `archive/`
>   (`agents/watcher_agent.py:3189`; `_scan_watcher_requests`, `agents/watcher_dispatch.py:555-578`),
>   and routes by `request_type`: `selfplay_retrain` → `dispatch_selfplay`, `learning_loop` →
>   `dispatch_learning_loop(scope)`, `pipeline_rerun` → `dispatch_learning_loop(scope="full")`
>   (`:503-540`). An optional LLM bundle check can reject a request first (`:443-466`). Dispatch is only
>   through WATCHER; selfplay never dispatches itself.
> - **BROKEN — B4.** The §11.4 validation table is not implemented. `_validate_selfplay_request`
>   (`agents/watcher_dispatch.py:598-617`) checks only `1 ≤ episodes ≤ 50` and that the survivors file
>   exists. There is **no request-age check, no priority check, no cooldown, no cluster-busy deferral**;
>   `created_at`, `priority`, `cooldown` and any time arithmetic do not occur in the file (searched).
>   No governance document carries these rules (they occur only in this chapter). The Chapter-13-side
>   cooldown is a separate defect (`docs/CHAPTER_13_AUDIT.md` B7).
> - **DESCRIPTION-ONLY:** "append-only" — processed requests are **moved** to
>   `watcher_requests/archive/<STATUS>_<timestamp>_<name>` (`_archive_request`, `:718-732`); the trail is
>   kept, but the queue is not append-only. The `pending_approval.json` path (Chapter 13 approvals) is
>   separate: the daemon notifies and waits unless test mode auto-approves
>   (`_poll_pending_approval`, `agents/watcher_agent.py:3245`).

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

> **Corrected 2026-11-11.** These are `window_optimizer.py` (Step 1) flags, which WATCHER emits from
> `default_params` via `args_map`. Verified in `window_optimizer.py`: `--enable-pruning` `:1485`,
> `--n-parallel` `:1487`, `--trse-context` `:1489`, `--use-persistent-workers` `:1493`,
> `--worker-pool-size` `:1504` (default **8**), `--seed-cap-nvidia` `:1506`, `--seed-cap-amd` `:1508`.
> **`--node-allowlist` is not a CLI flag** (DESCRIPTION-ONLY): `node_allowlist` is a
> `MultiGPUCoordinator` constructor argument (`coordinator.py:253`); the CLI way to name a partial fleet
> is now `--execution-set-nodes` / `--rig-profile` (`window_optimizer.py:1530`, `:1539`), and neither is in
> the manifest's `args_map` (`docs/TODO_CONSOLIDATED.md` row "--rig-profile / --execution-set-nodes not
> in args_map"). Later additions not listed: the RANGE-MINER and staging flags (§13.2).
> **`--n-parallel`: SWITCHED-OFF-BY-RULING — S3.** On the certified miner path `n_parallel` is held at
> 1 (D6.2 certification, `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md` §0.1); the manifest
> default is `1` (`agent_manifests/window_optimizer.json:260`).

### Manifest v1.5.0 (S131)

> **SUPERSEDED (2026-11-11).** The manifest is now **v1.8.0** (`agent_manifests/window_optimizer.json:5`),
> bumped from 1.7.0 by S172 Phase 1 (`8d0183f`). The invariant below still holds and is enforced by the
> step-scoped filter (§3.6 note).

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

> **Corrected 2026-11-11 (DESCRIPTION-ONLY).** WATCHER does **not** read a manifest timeout:
> `timeout_seconds` / `timeout_minutes` occur nowhere in `agents/watcher_agent.py` (searched), and the
> manifest's `"timeout_minutes": 1440` (`agent_manifests/window_optimizer.json:66`) is inert. The timeout
> comes from `WatcherConfig.get_step_timeout_minutes` (`agents/watcher_agent.py:341-345`) with the CLI's
> overrides `{0: 1, 1: 0, 5: 360}` (`:3575`); **Step 1 = 0 = no timeout** (`:2636-2637`, `[S147 Q1]`),
> others 120 min. The S123 args_map commits cited above (`e184a8c`, `7a6a63c`, `888cf3e`, `478ff74`)
> exist in history (`git cat-file -t`); their content was not re-read.

---

## 12. Step 1 Execution Path — Persistent Worker Mode (S146)

> **SUPERSEDED (2026-11-11) → see §13.** PWC is no longer the certifying Step-1 engine; RANGE-MINER
> is. PWC remains selectable and is **non-certifying** (`docs/TODO_CONSOLIDATED.md`, standing
> paragraph). Corrections to the text below, all DESCRIPTION-ONLY:
> - `use_persistent_workers` is in **`default_params`** (`true`, `agent_manifests/window_optimizer.json:261`);
>   `args_map` only maps the flag name (`:37`). (The same manifest's `parameter_bounds` block still says
>   `default: false`, `:172` — an internal inconsistency in the manifest; `default_params` is what WATCHER
>   merges.)
> - **PROCEED on Step 1 does not test survivor count.** It is the file-exists test on
>   `optimal_window_config.json`; the "confidence 1.00" is that test's constant
>   (`agents/watcher_agent.py:992`). "313 survivors → 1.00" coincided; it was not caused. See B1.
> - `worker_pool_size` default is **8** (`agent_manifests/window_optimizer.json:262`), not 4; on the miner
>   backend it is the **admission count** (`agents/watcher_agent.py:1612`; Gate 12 passed 25,
>   `gate12_launch.sh:380`).
> - `JOB_TIMEOUT_S = 600` is correct (`persistent_worker_coordinator.py:154`). The PWC per-node semaphore
>   is `threading.Semaphore(self.max_per_node)` with default `max_per_node=8` (`:264`, `:351`); the
>   "localhost `Semaphore(2)`" value was not found (U3). Commit `7e4ae02` exists ("send full
>   StrategyConfig dict — sieve_filter.py requires all 6 fields").

### Updated Step 1 Dispatch

When WATCHER executes Step 1 with `--use-persistent-workers`, the sieve backend
is `persistent_worker_coordinator.py` (PWC) rather than the legacy `coordinator.py` path.

**Manifest flag:** `window_optimizer.json` must include `"use_persistent_workers": true` in
the `args_map` section (or pass `--use-persistent-workers` in the launch args).

### WATCHER Step 1 Confidence Threshold

WATCHER issues PROCEED on Step 1 when:
- Bidirectional survivor count > 0
- WATCHER confidence = 1.00 (no anomalies)

S146 preprod run: 313 survivors → confidence 1.00 → PROCEED.

### PWC Architecture Invariants for WATCHER Integration

| Invariant | Value |
|-----------|-------|
| `worker_pool_size` | 4 (not 8) |
| `JOB_TIMEOUT_S` | 600 |
| Localhost semaphore | `threading.Semaphore(2)` |
| Strategy dict fields | All 6 required (`name`, `max_consecutive_misses`, `skip_tolerance`, `enable_reseed_search`, `skip_learning_rate`, `breakpoint_threshold`) |

These invariants are enforced in `persistent_worker_coordinator.py` and
`sieve_gpu_worker.py` as of commit `7e4ae02` (S146).

---

## 12. Step 1 Execution Path — Persistent Worker Mode (S146)

> **SUPERSEDED (2026-11-11) — verbatim duplicate** of the section above (appended twice). Kept so
> nothing is deleted; read the first copy and its correction note, then §13.

### Updated Step 1 Dispatch

When WATCHER executes Step 1 with `--use-persistent-workers`, the sieve backend
is `persistent_worker_coordinator.py` (PWC) rather than the legacy `coordinator.py` path.

**Manifest flag:** `window_optimizer.json` must include `"use_persistent_workers": true` in
the `args_map` section (or pass `--use-persistent-workers` in the launch args).

### WATCHER Step 1 Confidence Threshold

WATCHER issues PROCEED on Step 1 when:
- Bidirectional survivor count > 0
- WATCHER confidence = 1.00 (no anomalies)

S146 preprod run: 313 survivors → confidence 1.00 → PROCEED.

### PWC Architecture Invariants for WATCHER Integration

| Invariant | Value |
|-----------|-------|
| `worker_pool_size` | 4 (not 8) |
| `JOB_TIMEOUT_S` | 600 |
| Localhost semaphore | `threading.Semaphore(2)` |
| Strategy dict fields | All 6 required (`name`, `max_consecutive_misses`, `skip_tolerance`, `enable_reseed_search`, `skip_learning_rate`, `breakpoint_threshold`) |

These invariants are enforced in `persistent_worker_coordinator.py` and
`sieve_gpu_worker.py` as of commit `7e4ae02` (S146).

---

## 13. WATCHER → RANGE-MINER: how Step 1 is routed

**Added 2026-11-11.** Earlier versions of this chapter do not mention RANGE-MINER. Everything below
was read in code at `a15b524`; nothing was run. Step numbers are executable: the sieve runs inside
Step 1 (`window_optimizer.py`), so "routing to RANGE-MINER" means choosing Step 1's sieve backend.

### 13.1 What RANGE-MINER is, from WATCHER's side

RANGE-MINER (S172, spec `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md`, scoped there as a "Step 1
replacement", `:21`) is the **certifying** Step-1 sieve backend: long-lived per-GPU worker daemons
(`miner/range_miner_worker.py`) served by a coordinator (`miner/range_miner_coordinator.py`), which
assigns seed stripes, assembles results in Phase 5 and publishes the **22-array NPZ** survivor bundle.
WATCHER does not talk to the miner. It launches `window_optimizer.py` with the right flags, and
`window_optimizer.py` → `window_optimizer_integration_final.py` selects the miner. PWC and ZMQ remain
selectable and are non-certifying.

### 13.2 The chain of switches

| Hop | What decides | Anchor |
|---|---|---|
| 1. Manifest | `use_range_miner` declared in `default_params` (**`false`**), mapped `"use-range-miner": "use_range_miner"` in `actions[0].args_map`; also `miner_stripe_size` 67108864, `miner_substripes` 8, `miner_output_dir` null, and the staging keys | `agent_manifests/window_optimizer.json:269`, `:52-62`, `:270-279` |
| 2. WATCHER merge | operator `--params` may set `use_range_miner` because it is declared; step-scoped merge | `agents/watcher_agent.py:1811-1820` |
| 3. WATCHER fleet freeze | `_ensure_execution_set` picks backend **`miner`** when `use_range_miner` is truthy (checked first), and uses `worker_pool_size` as the miner's **admission count** | `agents/watcher_agent.py:1604-1613` |
| 4. WATCHER argv | `True` → `--use-range-miner` emitted; `False` → omitted | `agents/watcher_agent.py:2143-2148` |
| 5. Optimizer argparse | `--use-range-miner` (`store_true`), `--miner-stripe-size`, `--miner-substripes`, `--miner-output-dir`, staging flags | `window_optimizer.py:1552-1563` |
| 6. **Argparse gate (mutex)** | at most one of `--use-persistent-workers`, `--use-zmq-sqlite`, `--use-range-miner`; more → `parser.error` | `window_optimizer.py:1686-1697` |
| 7. Optimizer fleet freeze | same resolver, backend `miner` | `window_optimizer.py:1726-1733` |
| 8. Cascade gate | `_use_miner = getattr(coordinator, 'use_range_miner', False)` at the top of the backend cascade → `run_trial_miner(...)` with a Phase-5 assembling sink | `window_optimizer_integration_final.py:1402-1440` |
| 9. Miner | `run_trial_miner` drives the real `serve_trial`; skip bounds, `window_size` and `window_anchor` are mandatory metadata (fail-closed `None` defaults) | `miner/range_miner_coordinator.py:12336` |

**Version that added it.** S172 Phase 1, commit `8d0183f` ("Phase 1 — miner scaffolding, argparse gate,
integration hook, WATCHER v1.8.0"). "WATCHER v1.8.0" there is the **manifest** bump
`window_optimizer.json` 1.7.0 → 1.8.0 (`agent_manifests/window_optimizer.json:5`); the WATCHER program is
v2.0.0 (`agents/watcher_agent.py:12`). Phase 1 shipped the flag, the mutex and the cascade hook with
`run_trial_miner` raising `NotImplementedError`; Phases 2–6 replaced the stub, and at `a15b524`
`run_trial_miner` is real. Phase-1 acceptance: `tests/test_s172_phase1_scaffolding.py` (not run here).

**Stale text that still says otherwise** (DESCRIPTION-ONLY, outside this chapter, not edited): the
manifest's `parameter_bounds.use_range_miner.description` ("Phase 1 is scaffolding-only: setting true will
raise NotImplementedError until Phases 2-5 land", `agent_manifests/window_optimizer.json:215`) and the
comment above the flag (`window_optimizer.py:1548-1551`).

### 13.3 How an operator actually routes Step 1 to RANGE-MINER today

Because the manifest defaults are `use_persistent_workers: true` (`:261`) **and** `use_range_miner:
false` (`:269`), setting only `use_range_miner: true` would make WATCHER emit **both**
`--use-persistent-workers` and `--use-range-miner`, and the optimizer's mutex would abort Step 1
(hop 6). Combined with B1, a stale `optimal_window_config.json` could then let WATCHER report Step 1 as
PASSED. The committed production launch therefore overrides both:

```
python3 agents/watcher_agent.py --clear-halt --run-pipeline --start-step 1 --end-step 1 \
  --params '{"use_persistent_workers": false, "use_range_miner": true, "worker_pool_size": 25, ...,
             "n_parallel": 1}'  > "$LOG" 2>&1
```
(`gate12_launch.sh:377-383`). Its own trap list (`:67-76`) records why: `max_seeds` not `seed_count`
(the undeclared key is dropped), booleans are flag-only, and `--end-step 1` is mandatory (S2). The
worker daemons are started by hand (`scripts/launch_fleet_manual.sh`); WATCHER has no worker-launch
mechanism for the miner.

### 13.4 Run-4 Route-A: explicit operator warm-start pin routing (`3e2537b`)

Approved by Beta at R2 (`docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md`, "APPROVED FOR COMMIT");
routing ruling `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md`. What the code does:

- **The seven keys** `warm_start_window`, `_offset`, `_skip_min`, `_skip_max`, `_fwd_thresh`,
  `_rev_thresh`, `_session_idx` are a literal frozenset, not derived from the manifest
  (`STEP1_EXPLICIT_PIN_KEYS`, `agents/watcher_agent.py:493`). An eighth, `warm_start_session`, is always
  stripped (it has no optimizer flag).
- **Origin, not presence, carries authority.** The CLI `--run-pipeline` branch **moves** the seven out of
  `--params` into a keyword-only `_operator_pin_params` channel (`split_operator_pin_params`, `:630`; call
  `:3820`; `run_pipeline(..., _operator_pin_params=...)` `:3827`). Every other caller — the Chapter-13
  trigger (`chapter_13_triggers.py:616`), `--approve`, the dispatch paths — gets no authority, and if the
  seven appear in its ordinary `params` the pipeline is refused before Step 1
  (`assert_no_unauthorized_pin_keys`, `:668`; `blocked_by: step1_unauthorized_warm_start_pin`).
- **All seven or none, and usable.** `capture_step1_pin_bundle` (`:560`) rejects a partial pin and any
  member that the argv builder would silently skip (`None`, `''`, `True`, `False`); the result is an
  immutable, sorted, invocation-local mapping, captured once before the retry/LLM loop.
- **Two walls.** WALL 1 merges the bundle into Step 1's `final_params` (`_step1_explicit_pin`, `:696`;
  `:1828`); WALL 2 narrows the internal-only strip so the seven reach argv (`:2131-2134`). The manifest's
  `default_params` are untouched — Run-4 values as defaults are **prohibited**
  (`TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:12`).
- **Provenance.** On a pinned run only, `step1_pin_source = explicit_operator_warm_start` and
  `step1_pin_argv` are stamped onto Step 1's result (`_step1_stamp_pin_provenance`, `:708`). Absent on
  every unpinned run. No decision logic reads them.
- **Standing constraint (Beta R2):** `_operator_pin_params` and `_pin_bundle` are privileged internal
  seams; no new production caller may populate them without governance review.
- **Claim boundary (R2):** the patch certifies routing and containment only. Run 4 — the Brief-I
  acceptance run — is Michael-authorized separately and **has not run** (`docs/TODO_CONSOLIDATED.md`
  standing paragraph). Recorded-but-unrepaired at R2: the six-of-seven optimizer check, `_ws_source`
  mislabelling operator pins, the stale root `watcher_agent.py`, and `agents/step_runner/` as a separate
  uncertified dispatch surface. Gates: `tests/test_s172_run4_routing_patch.py`,
  `tests/run4_prelaunch_provenance_check.py` (not run here).

### 13.5 The pool=8 regression and the soak

- **§11.K "Pool=8 cliff regression"** (`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:594-596`): a full 50-trial
  run at default settings, 3 rigs × 8 GPUs, with no `launch_s174` FAULT_KEYWORD. Deferred to Phase 7
  (`docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md:94-107`). **No document records it as run or passed**
  (searched `docs/` for `11.K`, `pool=8`, `launch_s174`; U2). The manifest default `worker_pool_size` is 8
  (`agent_manifests/window_optimizer.json:262`); on the miner path that number is the admission count, and
  production launches pass it explicitly (25 at Gate 12).
- **§11.J WATCHER manifest dry-run** asks for `watcher_agent.py --run-pipeline --start-step 1 --end-step 1
  --dry-run` (`PROPOSAL_S172_RANGE_MINER_v1_4_4.md:445-450`, `:592`). **`--dry-run` has no effect on
  `--run-pipeline`** (§9.1 note), so the command as written would launch Step 1 for real. **BROKEN — B6.**
  Do not run it as a "dry run".
- **S172 Phase 7 soak** (50-trial WATCHER run, `--start-step 1 --end-step 1`, `n_parallel=1`, 25 frozen
  identities): **UNBLOCKED** since Gate-12 attempt 9 was accepted
  (`docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md`, dispositions "Phase 7 UNBLOCKED"), **not yet run**. A
  2026-08-04 soak attempt failed, and the certifying verifier `tests/gate_s172_prod_shape.py`
  (G-PROD-SHAPE) **ran on 2026-08-22 and failed** — the production-shape run failed at Phase-5 commit
  (`docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:4`, `:12`). "Phase 7 COMPLETE" in older
  chapters means WATCHER dispatch integration (Feb 2026), a different milestone.

### 13.6 Status at `a15b524`, in one place

| Item | State |
|---|---|
| Miner reachable from WATCHER | **yes**, by explicit `--params` override (§13.3) |
| Miner the WATCHER default | **no** — `use_range_miner: false`, `use_persistent_workers: true` (S1) |
| Gate 12 (production-class Step 1 via WATCHER) | PASSED, attempt 9, 2026-08-17 |
| Run-4 pin routing | committed `3e2537b`; Run 4 not run |
| Brief I (window-anchor) | implemented; acceptance pending a completed production-shape run |
| Phase-7 soak / §11.K pool=8 regression | not run |
| Steps 2–6 through WATCHER on miner output | held (S2) |
| Step-1 failure authority | open (B1) |

**Hit-rate note (so the two are not conflated).** WATCHER itself computes no hit rate
(`hit_rate`, `exact_hits`, `top_20`, `recall` do not occur in `agents/watcher_agent.py` or
`agents/watcher_dispatch.py`). The two quantities live elsewhere: backtest pool-membership recall
(`hit = actual_draw in numbers`, `evaluate_pools.py:34`; Top-20) and the live Chapter-13 trigger metric
`current_hit_rate = exact_hits / max(pool_size, 1)` (`chapter_13_diagnostics.py:531`). The KPI rulings
bind to the live-trigger metric and keep the two classes separate (`docs/TB_RULING_S176_WATCHER_KPI.md:497`).

---

## 14. Known gaps at a15b524

Labels per `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` §3.1. Detail and search
surfaces: `docs/CHAPTER_12_AUDIT.md`. Nothing here was repaired.

### 14.1 BROKEN

| # | What the chapter says | What the code does | Stops at |
|---|---|---|---|
| **B1** *(governed, open)* | WATCHER evaluates results and escalates failures | file-exists evaluation ignores `run_step`'s `success:False`/`return_code`; a stale or pre-written `optimal_window_config.json` passes a failed Step 1 | `agents/watcher_agent.py:946` vs `:2177` |
| **B2** | registry hooks track PRNG attempts and prevent redundant work | no WATCHER code calls the hooks or the registry | `agents/watcher_agent.py` (no import); `agents/watcher_registry_hooks.py:151` uncalled |
| **B3** | `--halt` stops WATCHER; `rm watcher_halt.flag` clears it | two halt files: `/tmp/agent_halt` (pipeline, daemon) vs `watcher_halt.flag` (dispatch) | `agents/safety/kill_switch.py:60`; `agents/watcher_dispatch.py:126` |
| **B4** | requests validated for age, priority, cooldown, resources | only `episodes` range and survivors-file existence | `agents/watcher_dispatch.py:598-617` |
| **B5** *(dormant)* | RETRY re-runs with adjustments | adjustments logged, not applied (outside Step-5 health retry) | `agents/watcher_agent.py:1452-1454` |
| **B6** | (spec §11.J) `--run-pipeline … --dry-run` is a dry run | `--dry-run` read only by dispatch commands; pipeline would execute | `agents/watcher_agent.py:3836-3845` |

### 14.2 SWITCHED-OFF-BY-RULING

| # | Item | Held by | Lifting it would enable |
|---|---|---|---|
| **S1** | `use_range_miner` defaults to `false` in `agent_manifests/window_optimizer.json:269` (PWC `use_persistent_workers: true`, `:261`) | spec `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:436` (`"use_range_miner": true ── once TB approves`); manifest `:215` "Do not enable in production until Phase 7 acceptance completes"; Phase 7 unblocked but not run (`TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md`); decision recorded as owed in `docs/TODO_CONSOLIDATED.md` (§(a), "Decision owed after Phase-7 acceptance") | WATCHER Step 1 on RANGE-MINER with no per-run override; G-PROD-SHAPE's "manifest defaults" path checking a real default. Flipping only this key without `use_persistent_workers: false` trips the mutex (§13.3) |
| **S2** | Steps 2–6 through WATCHER on RANGE-MINER output; `run_scorer_meta_optimizer.sh` calls the legacy converter | `docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` §1 (legacy converter not invoked until D3.0-B closes; Beta's corrected Phase-6 scope); launch rule `--end-step 1` (`gate12_launch.sh:73-76`) | a full 0→6 WATCHER pipeline, Chapter-13 `learning_loop` / `pipeline_rerun` requests, and `--approve`, on certified miner output |
| **S3** | `--n-parallel` > 1 on the miner path | D6.2 certification, "n_parallel = 1 — BINDING (Beta)" (`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md` §0.1) | concurrent Optuna partitions with certified checkpoint recovery (needs a separate checkpoint transaction design) |

Per §0 of the authority brief, a Beta "binding" ruling is recorded as a review position the owner may
keep or lift. No recommendation is made here.

### 14.3 DESCRIPTION-ONLY

D1 WATCHER version vs manifest v1.8.0 · D2 bundle types (8, not 7) · D3 file-exists short-circuit
instead of heuristic/LLM (§1.2, §7) · D4 §1.3 status block · D5 manifest count 6→7 / Step 0 · D6
`use_llm` default · D7 `decision.action` → `recommended_action` · D8 `STEP_SCRIPTS` content and lines ·
D9 `STEP_MANIFESTS` lines · D10 `/shared/ml` line numbers · D11 `run_step` sketch · D12 registry API ·
D13 `PreRunDecision` action names · D14 "Step 2.5" vs `pipeline_step: 2` · D15 manifest-structure example ·
D16 `--resume` / `--verbose` · D17 no log file · D18 ramdisk limitation superseded · D19 summary sizes ·
D20 "script defaults never used" · D21 request queue "append-only" · D22 `--node-allowlist` · D23
manifest v1.5.0 · D24 timeout source · D25 PWC §12 (flag location, PROCEED criterion, pool size) · D26
duplicate §12 · D27 fingerprint definition. Each has its note in place.

---

## 15. Open items

| Item | Where tracked |
|---|---|
| B1 failure authority — "required before Phase-7 autonomy" | fact file §2.34, §2.51 #9; `docs/TODO_CONSOLIDATED.md` §(a) |
| B2–B6 | `docs/CHAPTER_12_AUDIT.md` §4 — new this pass, for the repair list |
| S1 manifest-default decision | `docs/TODO_CONSOLIDATED.md` §(a); owner |
| Phase-7 soak, §11.K pool=8, Run 4 | `docs/TODO_CONSOLIDATED.md` §(a); Michael-initiated runs only |
| R2 recorded-but-unrepaired list (six-of-seven check, `_ws_source`, stale root `watcher_agent.py`, `step_runner`) | `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md` dispositions |
| `--rig-profile` / `--execution-set-nodes` not in `args_map` | `docs/TODO_CONSOLIDATED.md` |
| Two un-folded addenda `CHAPTER_12_ADDENDUM_v1_3_0.md`, `CHAPTER_12_ADDENDUM_PHASE1_v1_1_2.md` | `docs/PROJECT_FILE_CATALOG.md` chapter table; not read in this pass |
| U1–U5 | `docs/CHAPTER_12_AUDIT.md` §9 |

---

## 16. Verification declaration (VIR-1…6)

**Execution proof (VIR-1).** Every anchor in the 2026-11-11 notes was opened on VM 101 at `a15b524` in
this session with `Read`, `sed -n` or `/bin/grep -n`; the full file list is in `docs/CHAPTER_12_AUDIT.md`
§8. Nothing was executed except read-only host queries and a JSON parse of `watcher_decisions.jsonl`
and the manifests.

**Clean control (VIR-2).** `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` was read, not modified. Claims verified
accurate and left unchanged are listed in the audit §9 (24 items).

**Fault-injection control.** **Not applicable** to a documentation pass; none was invented.

**Completion sentinel (VIR-3).** §17.

**Unavailable-observer behaviour (VIR-5).** `/tmp` halt-file state, rig/CT100 state, the Phase-7 soak and
§11.K results, `test_watcher_agent.py` output and Beta ruling texts held outside the repo are
**UNAVAILABLE**, never clean.

**Audit claim scope (VIR-6).** What this chapter says against what the code at `a15b524` does — not
whether WATCHER's design is right. Searched surfaces: tracked tree; gitignored/untracked files
(`git check-ignore`, `agent_manifests/definitions.json`, `agents/data/`); git history
(`git log -S`, `git log --all -- <path>`, `git cat-file`); host (`ls` of state files, `systemctl
list-units`, `crontab -l`). Governance trail searched: `docs/TB_RULING_*`, `docs/PROPOSAL_*`,
`docs/TEAM_ALPHA_*` (138 files), the fact file, `docs/TODO_CONSOLIDATED.md`. Chapters searched:
`CHAPTER_2`, `CHAPTER_13_AUDIT.md`, `PIPELINE_BEHAVIOUR_MODEL.md`.

---

## 17. Closure statement

**Verified against** `a15b524` on VM 101, 2026-11-11. **Corrected:** 65 claims checked — 24 accurate,
6 BROKEN, 3 SWITCHED-OFF-BY-RULING, 27 DESCRIPTION-ONLY, 5 UNVERIFIED. **Added:** §0, §3.4.4a, §6.4,
§13 (RANGE-MINER routing), §14–§17. **Superseded, kept:** §1.3, §10.5, manifest v1.5.0 note, both §12
PWC sections. **Removed:** nothing — every pre-existing heading is present; the chapter grew.

**What this chapter is NOT:** not a certification that WATCHER is correct, not an operator runbook
(nothing here is to be executed as written; `--run-pipeline` is Michael-initiated only), and not a
statement about the miner beyond how WATCHER reaches it.

```
CHAPTER 12 CORRECTION:  PASS
```

`PASS` for the declared scope: every substantive claim audited and labelled, the chapter corrected in
place with a visible trail, RANGE-MINER routing documented from live code, gaps listed. It is not a
claim that any gap is repaired. Files changed: this chapter and `docs/CHAPTER_12_AUDIT.md` only.

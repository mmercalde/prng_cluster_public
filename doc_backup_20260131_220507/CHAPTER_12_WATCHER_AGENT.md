# Chapter 12: WATCHER Agent & Fingerprint Registry

## PRNG Analysis Pipeline — Complete Operating Guide

**Version:** 1.1.0  
**Date:** January 9, 2026  
**Files:** `agents/watcher_agent.py`, `agents/watcher_registry_hooks.py`, `agents/fingerprint_registry.py`  
**Purpose:** Autonomous pipeline orchestration with PRNG attempt tracking

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

---

## 1. Overview

### 1.1 What the WATCHER Does

The WATCHER Agent provides autonomous pipeline orchestration:

- **Monitors** pipeline step outputs
- **Evaluates** results against success criteria
- **Decides** next action (PROCEED / RETRY / ESCALATE)
- **Tracks** PRNG attempt history via Fingerprint Registry
- **Prevents** redundant work on known-failed combinations

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Heuristic Evaluation** | Rule-based decisions without LLM |
| **LLM Evaluation** | GBNF-constrained LLM decisions (optional) |
| **Fingerprint Registry** | Track dataset + PRNG combinations |
| **PRNG Priority Order** | Systematic PRNG family testing |
| **Safety Controls** | Kill switch, max retries, escalation |

### 1.3 Current Status

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
| `agents/watcher_registry_hooks.py` | Registry integration hooks |
| `agents/fingerprint_registry.py` | Dataset fingerprint tracking |
| `agent_manifests/*.json` | Step configurations |
| `watcher_history.json` | Decision history log |
| `watcher_decisions.jsonl` | Detailed decision audit |

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

### 3.3 Decision Actions

| Action | Confidence | Behavior |
|--------|------------|----------|
| PROCEED | ≥ 0.70 | Trigger next step automatically |
| RETRY | 0.50 - 0.70 | Re-run current step with adjustments |
| ESCALATE | < 0.50 | Alert human for review |

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

---

---

## 10.5 Ramdisk Preload Limitation

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

Remote nodes (rig-6600, rig-6600b) are **NOT** automatically populated.

### Impact on Distributed Execution

| Node | Ramdisk Populated | Jobs Will... |
|------|-------------------|--------------|
| Zeus (local) | ✅ Yes | Work |
| rig-6600 | ❌ No | Fail immediately |
| rig-6600b | ❌ No | Fail immediately |

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

---

## 12. Changelog

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

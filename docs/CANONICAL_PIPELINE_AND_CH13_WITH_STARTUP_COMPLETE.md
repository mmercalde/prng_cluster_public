# CANONICAL PIPELINE + CHAPTER 13 (LIVE FEEDBACK LOOP)
## Complete Operational Guide (No Placeholders)

---

## 1. Purpose of the System

This system is a **PRNG-mimicking and prediction framework** designed to:

- Analyze historical draw data
- Infer latent structure consistent with deterministic PRNG behavior
- Rank candidate generators (seeds + skips) by predictive power
- Generate a **high‑value prediction pool**
- Continuously improve via **closed‑loop learning**

The system is **PRNG‑agnostic** by design.  
No step assumes a specific generator.  
All generator behavior is abstracted via `prng_registry.py`.

---

## 2. High-Level Architecture

```
Historical / Synthetic Draws
        ↓
Steps 1–6 (Discovery + Modeling)
        ↓
Prediction Pool (Step 6)
        ↓
Live Draw Occurs
        ↓
Chapter 13 Diagnostics
        ↓
WATCHER Dispatch (Selfplay / Retrain)
        ↓
Repeat (Autonomous Loop)
```

---

## 3. Core Design Invariants

- **Code is immutable**
- **Only parameters are adjustable**
- **LLMs never execute code**
- **LLMs propose, WATCHER disposes**
- **All learning is evidence‑driven**
- **Every decision is auditable**
- **Selfplay explores. Chapter 13 decides. WATCHER enforces.**

---

## 4. Pipeline Steps (1–6)

### Step 1 — Window Optimizer
**Goal:** Find optimal slicing of draw history.

- Uses Bayesian optimization (Optuna)
- Explores window size, offsets, skip ranges
- Applies forward + reverse sieves
- Outputs:
  - `optimal_window_config.json`
  - `bidirectional_survivors.json`
  - `train_history.json`
  - `holdout_history.json`

**Runtime driver:** number of candidate seeds × trials  
**GPU‑accelerated**

---

### Step 2 — Scorer Meta‑Optimizer
**Goal:** Optimize feature weighting.

- Tests feature hyperparameters
- Distributed across GPUs
- Outputs:
  - `optimal_scorer_config.json`

---

### Step 3 — Full Scoring
**Goal:** Feature extraction + ground‑truth labeling.

For each survivor:
- Extracts ~50 per‑seed features
- Computes `holdout_hits` using unseen data

Outputs:
- `survivors_with_scores.json`

This file is the **single source of truth** for ML.

---

### Step 4 — Adaptive Meta‑Optimizer
**Goal:** ML capacity planning.

- Determines model families + constraints
- Does *not* consume survivor data directly
- Outputs:
  - `reinforcement_engine_config.json`

---

### Step 5 — Anti‑Overfit Training
**Goal:** Learn generalizable patterns.

- Models: XGBoost, LightGBM, CatBoost, NN
- Target: `holdout_hits`
- Uses k‑fold cross‑validation
- Outputs:
  - `models/reinforcement/best_model.*`
  - `models/reinforcement/best_model.meta.json`

The sidecar metadata is a **hard contract**.

---

### Step 6 — Prediction Generator
**Goal:** Produce ranked predictions.

- Loads model **only via sidecar**
- Validates feature schema hash
- Ranks survivors
- Produces:
  - `prediction_pool.json`
  - `confidence_map.json`

No learning happens here.  
This is a **read‑only inference step**.

---

## 5. Chapter 13 — Live Feedback Loop

Chapter 13 governs **post‑prediction learning**.

### Responsibilities
- Detect new draws
- Compare predictions vs reality
- Generate diagnostics
- Decide whether retraining is warranted
- Propose parameter changes (via LLM)
- Dispatch via WATCHER (autonomous LLM evaluation, grammar-constrained)

---

## 6. TEST MODE (Synthetic Convergence Validation)

### 6.1 Enable Test Mode

```bash
cd ~/distributed_prng_analysis

cp watcher_policies.json watcher_policies.json.bak

python3 - << 'EOF'
import json
with open('watcher_policies.json') as f:
    p = json.load(f)

p['test_mode'] = True
p['synthetic_injection']['enabled'] = True

with open('watcher_policies.json','w') as f:
    json.dump(p,f,indent=2)

print("Test mode enabled")
print("True seed:", p['synthetic_injection']['true_seed'])
EOF
```

---

### 6.2 Bootstrap Synthetic History (Required)

Because the injector depends on `optimal_window_config.json`
and Step 1 depends on `lottery_history.json`,
we **bootstrap explicitly**.

```bash
cd ~/distributed_prng_analysis

python3 << 'EOF'
import json
from datetime import datetime, timedelta
from prng_registry import get_cpu_reference

TRUE_SEED = 12345
PRNG_TYPE = "java_lcg"
NUM_DRAWS = 5000
MOD = 1000

prng = get_cpu_reference(PRNG_TYPE)
raw = prng(TRUE_SEED, NUM_DRAWS)
vals = [v % MOD for v in raw]

draws = []
date = datetime(2024,1,1)

for i,v in enumerate(vals):
    digits = [(v//100)%10,(v//10)%10,v%10]
    session = "midday" if i%2==0 else "evening"
    draws.append({
        "date": date.strftime("%Y-%m-%d"),
        "session": session,
        "draw": digits,
        "value": v,
        "draw_source": "synthetic_bootstrap",
        "true_seed": TRUE_SEED
    })
    if session=="evening":
        date += timedelta(days=1)

with open("lottery_history.json","w") as f:
    json.dump({"draws":draws},f,indent=2)

with open("optimal_window_config.json","w") as f:
    json.dump({
        "prng_type": PRNG_TYPE,
        "mod": MOD,
        "bootstrap": True
    },f,indent=2)

print("Bootstrap complete:", len(draws), "draws")
EOF
```

---

## 7. Running the Pipeline

```bash
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 6
```

Expected runtime (test mode):
- 30–90 minutes depending on seed range + trials

---

## 8. Running Chapter 13

### 8.1 Direct (Standalone)

```bash
python3 chapter_13_orchestrator.py
```

### 8.2 Via WATCHER (Autonomous — Phase 7)

```bash
# Process pending Chapter 13 requests
PYTHONPATH=. python3 agents/watcher_agent.py --process-requests

# Dispatch selfplay directly
PYTHONPATH=. python3 agents/watcher_agent.py --dispatch-selfplay

# Dispatch learning loop (Steps 3→5→6)
PYTHONPATH=. python3 agents/watcher_agent.py --dispatch-learning-loop steps_3_5_6

# Daemon mode (monitors watcher_requests/ continuously)
PYTHONPATH=. python3 agents/watcher_agent.py --daemon
```

It will:
- Monitor new draws
- Evaluate prediction quality
- Trigger retraining if thresholds are crossed
- Execute via WATCHER dispatch (autonomous LLM evaluation, grammar-constrained)

---

## 9. Convergence Expectations (Test Mode)

**Functional Mimicry Paradigm:** This system learns output patterns to predict future draws. Seeds generate candidate sequences for heuristic extraction. Success is measured by hit rate and confidence calibration, NOT seed ranking.

| Metric | Target |
|------|-------|
| Hit rate improvement | Increases after N draws |
| Confidence calibration | Correlation > 0.7 |
| Confidence trend | Increasing |
| Diagnostics | Stable or improving |

Failure to converge is **information**, not error.

---

## 10. What Learns vs What Does Not

| Component | Learns? |
|--------|--------|
| Sieves (Steps 1–3) | ❌ |
| ML Model (Step 5) | ✅ |
| Selfplay (tree models + bandit) | ✅ |
| Prediction ranking | ❌ |
| Chapter 13 triggers | ❌ |
| LLM advisor | ❌ (advisory only) |

**Step 5** updates model weights. **Selfplay** learns via tree models and bandit algorithms. Both are statistical, evidence-driven, and auditable.

---

## 11. Autonomy Boundary

Current (v2 — Phase 7 Complete):
- Chapter 13 triggers retrain/selfplay requests
- WATCHER evaluates via grammar-constrained LLM (DeepSeek-R1)
- WATCHER dispatches selfplay or learning loop autonomously
- LLM lifecycle managed (stop before GPU phase, restart after)
- All decisions audited in `watcher_requests/` archive

Authority separation:
- **Chapter 13 decides** (ground truth, promotion/rejection)
- **WATCHER enforces** (dispatch, safety, halt flag)
- **Selfplay explores** (never self-promotes)

Future:
- Phase 9B.3: Automatic policy proposal heuristics
- Parameter advisor: LLM-recommended tuning for Steps 4-6

---

## 12. Final Truth

This system does **not guess numbers**.

It:
- Searches structure
- Learns patterns
- Measures reality
- Adjusts only when evidence demands it

That is why autonomy is possible.

---

**End of Canonical Document**

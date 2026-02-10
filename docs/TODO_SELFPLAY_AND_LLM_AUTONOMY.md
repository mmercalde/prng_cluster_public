# TODO: Selfplay Testing + LLM Autonomy Wiring

**Created:** 2026-01-30  
**Status:** Planning  
**Goal:** Complete autonomous operation pipeline

---

## Overview

| Part | Description | Tasks | Est. Sessions |
|------|-------------|-------|---------------|
| **A** | Selfplay System Testing | 7 | 0.5 |
| **B** | LLM → WATCHER Autonomy Wiring | 20 | 1.5-2 |
| **Total** | | **27** | **2-3 sessions** |

---

## Part A: Selfplay System Testing

**Goal:** Validate Phase 9B.2 works end-to-end before building governance layer

| # | Task | Command / Action | Status |
|---|------|------------------|--------|
| A1 | Run multi-episode selfplay | `python3 selfplay_orchestrator.py --survivors survivors_with_scores.json --episodes 5 --policy-conditioned` | 🔲 |
| A2 | Verify candidate emission | `cat learned_policy_candidate.json` | 🔲 |
| A3 | Verify policy history archive | `ls -la policy_history/` | 🔲 |
| A4 | Test with active policy | Create `learned_policy_active.json`, re-run selfplay | 🔲 |
| A5 | Test filter transform | Add filter to active policy, verify survivor reduction | 🔲 |
| A6 | Verify telemetry health | `cat telemetry/learning_health_latest.json` | 🔲 |
| A7 | Test Chapter 13 candidate validation | `python3 chapter_13_acceptance.py --validate-selfplay learned_policy_candidate.json` | 🔲 |

### Part A Commands Reference

```bash
cd ~/distributed_prng_analysis

# A1: Multi-episode selfplay
python3 selfplay_orchestrator.py \
    --survivors survivors_with_scores.json \
    --episodes 5 \
    --policy-conditioned

# A2: Check candidate
cat learned_policy_candidate.json | jq .

# A3: Check archive
ls -la policy_history/

# A4: Create active policy for testing
echo '{"policy_id": "test_active", "transforms": {}}' > learned_policy_active.json
python3 selfplay_orchestrator.py --survivors survivors_with_scores.json --single-episode --policy-conditioned

# A5: Test filter transform
cat > learned_policy_active.json << 'EOF'
{
  "policy_id": "test_filter",
  "fitness": 0.5,
  "transforms": {
    "filter": {
      "enabled": true,
      "min_score": 0.01
    }
  }
}
EOF
python3 selfplay_orchestrator.py --survivors survivors_with_scores.json --single-episode --policy-conditioned

# A6: Check telemetry
cat telemetry/learning_health_latest.json | jq .

# A7: Validate candidate
python3 chapter_13_acceptance.py --validate-selfplay learned_policy_candidate.json
```

---

## Part B: LLM → WATCHER Autonomy Wiring

**Goal:** Connect LLM recommendations to WATCHER execution

### Phase 10A: Schema & Grammar (Foundation)

| # | Task | File | Lines | Status |
|---|------|------|-------|--------|
| B1 | Create proposal Pydantic models | `llm_proposal_schema.py` | ~80 | 🔲 |
| B2 | Create GBNF grammar | `agent_grammars/chapter_13.gbnf` | ~50 | 🔲 |
| B3 | Extract tunable parameters from manifests | `chapter_13_parameter_vocabulary.py` | ~100 | 🔲 |

**B1 Details — `llm_proposal_schema.py`:**
```python
# Pydantic models for:
# - ParameterProposal (parameter, current, proposed, delta, confidence, rationale)
# - LLMProposal (analysis_summary, failure_mode, confidence, recommended_action, 
#                retrain_scope, parameter_proposals, risk_level, requires_human_review)
# - ValidationResult (accepted, reason, action)
```

**B2 Details — `chapter_13.gbnf`:**
```
# GBNF grammar constraining LLM output to valid JSON structure
# Ensures: recommended_action ∈ {RETRAIN, WAIT, ESCALATE}
# Ensures: risk_level ∈ {low, medium, high}
# Ensures: confidence ∈ [0.0, 1.0]
```

**B3 Details — `chapter_13_parameter_vocabulary.py`:**
```python
# Auto-extract from:
# - agent_manifests/*.json
# - watcher_policies.json
# Output: JSON dict of tunable parameters with bounds, locations, frozen list
```

---

### Phase 10B: Diagnostics Engine (Fact Substrate)

| # | Task | File | Lines | Status |
|---|------|------|-------|--------|
| B4 | Generate post-draw diagnostics | `chapter_13_diagnostics.py` | ~150 | 🔲 |
| B5 | Create diagnostics history archiver | In above | — | 🔲 |
| B6 | Test: generate diagnostics from real run | CLI test | — | 🔲 |

**B4-B5 Details — `chapter_13_diagnostics.py`:**
```python
# Inputs:
# - predictions (from Step 6)
# - actual outcomes (from draw history)
# - telemetry/learning_health_latest.json
# - recent run summaries

# Outputs:
# - post_draw_diagnostics.json (current)
# - diagnostics_history/diagnostics_{timestamp}.json (archive)

# Metrics computed:
# - hit_rate (predictions vs reality)
# - calibration_error
# - survivor_variance
# - fitness_trend (last N runs)
# - train_val_gap_trend
```

---

### Phase 10C: LLM Advisor (The PhD Reviewer)

| # | Task | File | Lines | Status |
|---|------|------|-------|--------|
| B7 | Build prompt with diagnostics + vocabulary | `chapter_13_llm_advisor.py` | ~200 | 🔲 |
| B8 | Integrate with `llm_router.py` | In above | — | 🔲 |
| B9 | Parse LLM response to Pydantic | In above | — | 🔲 |
| B10 | Test: mock diagnostics → LLM → proposal | CLI test | — | 🔲 |

**B7-B9 Details — `chapter_13_llm_advisor.py`:**
```python
# Core function:
def get_llm_recommendation(diagnostics: dict, run_history: list) -> LLMProposal:
    """
    1. Load parameter vocabulary
    2. Build system prompt (constraints, mission)
    3. Build user prompt (diagnostics, history, tasks)
    4. Call llm_router.route(prompt)
    5. Parse response with GBNF validation
    6. Return LLMProposal
    """
```

**System Prompt Template:**
```
You are an analytical advisor for a probabilistic research system.

HARD CONSTRAINTS:
- You do NOT execute actions
- You do NOT modify parameters directly
- You do NOT assume stationarity
- You MUST express uncertainty

TUNABLE PARAMETERS:
{{ parameter_vocabulary }}

FROZEN COMPONENTS (never touch):
{{ frozen_list }}

YOUR TASK:
Interpret diagnostic deltas and propose cautious, reversible adjustments.
If uncertainty is high, recommend WAIT.
```

---

### Phase 10D: WATCHER Execution (Validate + Act)

| # | Task | File | Lines | Status |
|---|------|------|-------|--------|
| B11 | Add `validate_proposal()` | `agents/watcher_agent.py` | ~80 | 🔲 |
| B12 | Add `apply_parameter_changes()` | `agents/watcher_agent.py` | ~50 | 🔲 |
| B13 | Add `dispatch_selfplay()` | `agents/watcher_agent.py` | ~40 | 🔲 |
| B14 | Add `dispatch_learning_loop()` | `agents/watcher_agent.py` | ~30 | 🔲 |
| B15 | Wire LLM advisor into WATCHER daemon | `agents/watcher_agent.py` | ~50 | 🔲 |

**B11 Details — `validate_proposal()`:**
```python
def validate_proposal(proposal: LLMProposal) -> ValidationResult:
    """
    Rejection rules:
    - confidence < 0.60 → REJECT
    - risk_level in [medium, high] → ESCALATE
    - any parameter delta > 30% → REJECT
    - parameter in FROZEN_PARAMETERS → REJECT
    - len(parameter_proposals) > 3 → REJECT
    - cooldown not elapsed → REJECT
    
    Acceptance rules:
    - risk_level == "low"
    - confidence >= 0.75
    - ≤ 2 parameters affected
    - cooldown elapsed (≥ 3 runs since last change)
    """
```

**B13 Details — `dispatch_selfplay()`:**
```python
def dispatch_selfplay(request: dict) -> bool:
    """
    Execute selfplay_orchestrator.py with policy conditioning.
    
    cmd = [
        "python3", "selfplay_orchestrator.py",
        "--survivors", "survivors_with_scores.json",
        "--episodes", str(request.get("episodes", 5)),
        "--policy-conditioned",
        "--project-root", PROJECT_ROOT,
    ]
    """
```

---

### Phase 10E: Integration Testing

| # | Task | Action | Status |
|---|------|--------|--------|
| B16 | End-to-end: diagnostics → LLM → proposal → validate | Full flow test | 🔲 |
| B17 | Test auto-accept (low risk, high conf) | Verify execution | 🔲 |
| B18 | Test auto-reject (delta too large) | Verify rejection logged | 🔲 |
| B19 | Test escalation (medium risk) | Verify human alert | 🔲 |
| B20 | Test selfplay dispatch from WATCHER | `dispatch_selfplay()` works | 🔲 |

**Integration Test Commands:**
```bash
cd ~/distributed_prng_analysis

# B16: Full flow test
python3 chapter_13_diagnostics.py --generate
python3 chapter_13_llm_advisor.py --diagnose post_draw_diagnostics.json
# Review proposal, then:
python3 agents/watcher_agent.py --validate-proposal llm_proposals/latest.json

# B17: Test auto-accept
# Create mock proposal with low risk, high confidence
python3 agents/watcher_agent.py --validate-proposal test_accept_proposal.json

# B18: Test auto-reject
# Create mock proposal with delta > 30%
python3 agents/watcher_agent.py --validate-proposal test_reject_proposal.json

# B19: Test escalation
# Create mock proposal with medium risk
python3 agents/watcher_agent.py --validate-proposal test_escalate_proposal.json

# B20: Test selfplay dispatch
python3 agents/watcher_agent.py --dispatch-selfplay --dry-run
```

---

## Dependency Chain

```
llm_proposal_schema.py (B1)
       ↓
chapter_13.gbnf (B2)
       ↓
chapter_13_parameter_vocabulary.py (B3)
       ↓
chapter_13_diagnostics.py (B4-B6)
       ↓
chapter_13_llm_advisor.py (B7-B10)
       ↓
watcher_agent.py additions (B11-B15)
       ↓
Integration tests (B16-B20)
```

---

## Suggested Schedule

### Day 1
| Time | Tasks | Focus |
|------|-------|-------|
| 30-60 min | A1-A7 | Selfplay testing |
| 60 min | B1-B3 | Schema + Grammar + Vocabulary |
| 60 min | B4-B6 | Diagnostics engine |

### Day 2
| Time | Tasks | Focus |
|------|-------|-------|
| 90 min | B7-B10 | LLM Advisor |
| 60 min | B11-B15 | WATCHER execution additions |

### Day 3
| Time | Tasks | Focus |
|------|-------|-------|
| 60-90 min | B16-B20 | Integration testing + fixes |

---

## Files Created/Modified Summary

| File | Action | Location |
|------|--------|----------|
| `llm_proposal_schema.py` | CREATE | `~/distributed_prng_analysis/` |
| `chapter_13.gbnf` | CREATE | `~/distributed_prng_analysis/agent_grammars/` |
| `chapter_13_parameter_vocabulary.py` | CREATE | `~/distributed_prng_analysis/` |
| `chapter_13_diagnostics.py` | CREATE | `~/distributed_prng_analysis/` |
| `chapter_13_llm_advisor.py` | CREATE | `~/distributed_prng_analysis/` |
| `watcher_agent.py` | MODIFY | `~/distributed_prng_analysis/agents/` |
| `watcher_policies.json` | MODIFY | `~/distributed_prng_analysis/` |

---

## Success Criteria

### Part A Complete When:
- [ ] Selfplay runs 5+ episodes without error
- [ ] Candidates emitted to `learned_policy_candidate.json`
- [ ] Policy history archived
- [ ] Active policy loaded and applied
- [ ] Filter transform reduces survivors as expected

### Part B Complete When:
- [ ] LLM receives diagnostics + vocabulary
- [ ] LLM returns GBNF-valid proposal
- [ ] WATCHER validates proposal against policy
- [ ] Auto-accept triggers execution
- [ ] Auto-reject logs reason
- [ ] Escalation alerts human
- [ ] `dispatch_selfplay()` starts selfplay_orchestrator

### Full Autonomy Achieved When:
```
Diagnostics → LLM → Proposal → WATCHER → Execute
       ↑                                    ↓
       └────────── Next cycle ──────────────┘
```

No human in the loop for routine decisions.

---

## Notes

- **LLM is advisory only** — WATCHER decides
- **GBNF grammar** ensures parseable output
- **Cooldowns** prevent oscillation
- **Frozen components** (Steps 1, 2, 4) never modified
- **Audit trail** for all decisions

---

**END OF TODO**

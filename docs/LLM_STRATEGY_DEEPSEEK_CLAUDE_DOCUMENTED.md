# DeepSeek & Claude Opus Strategy
## Documented LLM Roles and Decision Making

**Version:** 1.0.0 (Verified from Project Documentation)  
**Date:** February 10, 2026  
**Status:** Extracted from Actual Implementation

---

## Overview

The system uses **two LLMs in a Primary + Backup architecture** for strategic decision-making about pipeline parameters. They do NOT execute code or make predictions directly - they **analyze diagnostics and propose parameter adjustments**.

**Source:** CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md, SESSION_CHANGELOG_20260208_S68.md

---

## Architecture: Primary + Backup

```
┌─────────────────────────────────────────────┐
│ Parameter Advisor (parameter_advisor.py)    │
│                                             │
│ Step 1: Compute Metrics                    │
│   - Hit@20, Hit@100, Hit@300               │
│   - Calibration correlation                │
│   - Survivor churn                         │
│   - Model performance trends               │
│                                             │
│ Step 2: Call DeepSeek (Primary)            │
│   - Grammar-constrained JSON output        │
│   - 51 tok/sec throughput                  │
│   - Confidence score returned              │
│                                             │
│ Step 3: Check Confidence                   │
│   If confidence < 0.3 → Escalate to Claude │
│   If confident → Use recommendation        │
│                                             │
│ Step 4: Escalate to Claude (Backup)        │
│   - Called only when DeepSeek uncertain    │
│   - 38 tok/sec throughput                  │
│   - Strategic deep analysis                │
│                                             │
│ Step 5: Heuristic Fallback (Emergency)     │
│   - Only if BOTH LLMs unreachable          │
│   - Logs "DEGRADED_MODE" warning           │
│   - Tags decision with degraded flag       │
└─────────────────────────────────────────────┘
```

**Source:** PROPOSAL_STRATEGY_ADVISOR_LIFECYCLE_INTEGRATION_v1_0.md, Section 3

---

## DeepSeek-R1-14B Strategy (Primary)

### Role: Routine Strategic Analysis

**Model:** DeepSeek-R1-14B (Q4_K_M quantized)  
**Speed:** 51 tok/sec  
**Access:** llama.cpp server on port 8080  
**Constraint:** Grammar-constrained via GBNF

### What DeepSeek Does

**Input:** Diagnostic metrics from last 15-20 draws
**Output:** Structured JSON recommendation (grammar-constrained)
**Frequency:** Every Chapter 13 cycle (after new draw)

**Documented responsibilities:**
1. **Focus Area Classification** (7 categories)
2. **Parameter Adjustment Proposals**
3. **Selfplay Configuration Recommendations**
4. **Confidence Score** (0.0-1.0)

### Focus Area Classification

**Source:** CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md, Section 4.1

DeepSeek classifies the system state into one of **7 focus areas:**

| Focus Area | Trigger | DeepSeek Strategy |
|------------|---------|-------------------|
| **POOL_PRECISION** | Hit@100 > 70% but Hit@20 < 10% | "Pool captures draws but top predictions unfocused. Recommend: Increase weight on survivor consistency, tighten pool concentration." |
| **POOL_COVERAGE** | Hit@300 < 80% | "Pool missing draws. Recommend: Broaden exploration, increase episode count, diversify model types." |
| **CONFIDENCE_CALIBRATION** | Calibration correlation < 0.3 | "Confidence scores not predictive. Recommend: Focus on fold stability, minimize train/val gap." |
| **MODEL_DIVERSITY** | Single model dominates (>80%) | "Over-reliance on one model. Recommend: Force model rotation, increase diversity in episodes." |
| **FEATURE_RELEVANCE** | Feature drift > 0.3 | "Features may be stale. Recommend: Trigger learning loop (Steps 3→5→6)." |
| **REGIME_SHIFT** | Window decay > 0.5 AND survivor churn > 0.4 | "Pattern change detected. Recommend: Full pipeline rerun (Steps 1→6)." |
| **STEADY_STATE** | All metrics within bounds | "System healthy. Recommend: Reduce episode count to maintenance mode." |

**Priority order (documented):**
```
REGIME_SHIFT > POOL_COVERAGE > CONFIDENCE_CALIBRATION > 
POOL_PRECISION > MODEL_DIVERSITY > FEATURE_RELEVANCE > STEADY_STATE
```

### Example DeepSeek Output

**Source:** CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md, Section 5.1

```json
{
  "schema_version": "1.0.0",
  "generated_at": "2026-02-03T12:00:00Z",
  "advisor_model": "deepseek-r1-14b",
  "draws_analyzed": 20,
  
  "focus_area": "POOL_PRECISION",
  "focus_confidence": 0.82,
  "focus_rationale": "Hit@100 at 73% but Hit@20 at 4%. Pool is capturing draws but top predictions are unfocused.",
  
  "recommended_action": "RETRAIN",
  "retrain_scope": "selfplay_only",
  
  "selfplay_overrides": {
    "max_episodes": 15,
    "model_types": ["catboost", "lightgbm"],
    "min_fitness_threshold": 0.55,
    "priority_metrics": ["pool_concentration", "model_agreement"],
    "exploration_ratio": 0.3
  },
  
  "parameter_proposals": [
    {
      "parameter": "n_estimators",
      "current_value": 100,
      "proposed_value": 200,
      "delta": "+100",
      "confidence": 0.75,
      "rationale": "Underfitting signal: fold std decreasing but R² plateau suggests capacity limit"
    }
  ],
  
  "pool_strategy": {
    "tight_pool_guidance": "Increase weight on survivor consistency (current: 0.15, suggest: 0.25)"
  },
  
  "risk_level": "low",
  "requires_human_review": false
}
```

### Grammar Constraint (GBNF)

**Source:** grammars/strategy_advisor.gbnf (3,576 bytes), SESSION_CHANGELOG_20260208_S68.md

DeepSeek's output is **grammar-constrained** - it CANNOT produce invalid JSON. The GBNF grammar forces:
- Specific focus area values (7 options only)
- Structured parameter proposals
- Confidence scores 0.0-1.0
- Valid action types (RETRAIN, WAIT, ESCALATE, REFOCUS, FULL_RESET)

**Example constraint:**
```
focus-area ::= "\"POOL_PRECISION\"" | "\"POOL_COVERAGE\"" |
               "\"CONFIDENCE_CALIBRATION\"" | "\"MODEL_DIVERSITY\"" |
               "\"FEATURE_RELEVANCE\"" | "\"REGIME_SHIFT\"" |
               "\"STEADY_STATE\""
```

### Escalation to Claude

**When DeepSeek escalates:**

```python
# From parameter_advisor.py
if recommendation.get("focus_confidence", 0) < 0.3:
    logger.info("DeepSeek low confidence (%.2f) — escalating to Claude",
                recommendation["focus_confidence"])
    raise EscalationRequired("Low confidence triggers backup LLM")
```

**Source:** PROPOSAL_STRATEGY_ADVISOR_LIFECYCLE_INTEGRATION_v1_0.md, Section 4.1

**Escalation threshold:** confidence < 0.3 (30%)

---

## Claude Opus 4.6 Strategy (Backup)

### Role: Strategic Deep Analysis

**Model:** Claude Opus 4.6  
**Speed:** 38 tok/sec  
**Access:** Claude Code CLI (external API)  
**Constraint:** None (free-form analysis)

### When Claude Is Called

**Source:** CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md, Section 10

**Automatic escalation triggers:**
1. DeepSeek confidence < 0.3
2. DeepSeek returns "ESCALATE" action
3. DeepSeek server unreachable
4. REGIME_SHIFT detected with complex patterns

**Manual consultation triggers:**
- Multi-draw trend analysis (20+ draws)
- Strategy pivot evaluation
- Cross-PRNG algorithm comparison
- Feature engineering suggestions
- Regime shift root cause analysis

### What Claude Analyzes

**Documented inputs provided to Claude:**

**Source:** CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md, Section 10.2

1. **diagnostics_history/** (last 20+ draw diagnostics)
2. **telemetry/** (last 20+ selfplay episode summaries)
3. **policy_history/** (all promoted and rejected policies)
4. **strategy_history/** (last 10 advisor recommendations)
5. **Pool performance CSV** (Hit@K by draw, timestamps)
6. **Active configuration** (current parameter values)

### Claude's Unique Capabilities

**Source:** CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md, Section 10.3

**What Claude can do that DeepSeek cannot:**

1. **Cross-pattern analysis** across PRNG algorithm families
2. **Non-obvious feature interactions** identification
3. **Novel fitness function proposals** (beyond current proxy rewards)
4. **Mathematical proofs** for concentration bound improvements
5. **Strategic reasoning** requiring multi-step inference

### Example Claude Analysis

**Hypothetical scenario (documented style):**

```
User provides:
  - diagnostics_history/ (20 files)
  - Hit@20 declining from 12% → 4% over 20 draws
  - Survivor churn: 0.38 (high)
  - Best model changed: XGBoost → CatBoost → LightGBM → CatBoost

Claude analyzes:
  "Pattern analysis reveals oscillating model selection correlated with 
   survivor churn cycles. When survivor set turns over >30%, the previously
   best model becomes obsolete within 3-5 draws.
   
   Root cause hypothesis: Step 3 feature extraction relies on survivor
   consistency features (weight 0.15), but high churn invalidates these.
   
   Recommendation: 
   1. Reduce survivor_consistency weight from 0.15 → 0.05
   2. Increase temporal_stability weight from 0.12 → 0.22
   3. Add 3-draw rolling average to smooth churn artifacts
   4. Test with selfplay episodes: 25 (increased from 15)
   
   Expected outcome: Model selection stabilizes, Hit@20 recovers to 8-10%
   within 10 draws. Monitor survivor_churn metric."
```

### Claude Output Format

**Unlike DeepSeek:** Claude is NOT grammar-constrained. Output is free-form analysis.

**Documented processing:**

**Source:** SESSION_CHANGELOG_20260208_S68.md

```python
# Claude backup path
response = llm_router.evaluate(
    prompt=bundle,
    grammar_file="strategy_advisor.gbnf",  # Ignored for Claude
    force_backup=True
)

# Response: Markdown-wrapped JSON (needs stripping)
# Processing: Extract JSON, validate against schema
```

**Verification note:** Session 68 verified Claude backup operational:
```
Claude Backup:
  ✅ Routing via force_backup=True
  ✅ Valid JSON (markdown wrapper stripped)
  ✅ Substantive analysis: focus=CONFIDENCE_CALIBRATION
```

---

## Parameter Proposal Validation

### Bounds Clamping (Team Beta Approved)

**Source:** SESSION_CHANGELOG_20260208_S68.md

**Problem:** LLMs sometimes propose out-of-bounds values
- Example: DeepSeek proposed max_episodes=1000
- Bound: max_episodes ≤ 50 (Pydantic validation)

**Solution:** Option D - Clamp + Explicit Tagging

```python
def _clamp_llm_recommendation(recommendation):
    """Clamp LLM proposals to safety bounds."""
    bounds = {
        'max_episodes': (1, 50),
        'min_fitness_threshold': (0.0, 1.0),
        'exploration_ratio': (0.0, 1.0)
    }
    
    adjusted = {}
    for param, (min_val, max_val) in bounds.items():
        if param in recommendation['selfplay_overrides']:
            original = recommendation['selfplay_overrides'][param]
            clamped = max(min_val, min(max_val, original))
            
            if clamped != original:
                adjusted[param] = {
                    'original': original,
                    'applied': clamped
                }
                recommendation['selfplay_overrides'][param] = clamped
    
    if adjusted:
        recommendation['metadata']['bounds_adjusted'] = {
            'fields': list(adjusted.keys()),
            'original_values': {k: v['original'] for k, v in adjusted.items()},
            'applied_limits': {k: v['applied'] for k, v in adjusted.items()}
        }
    
    return recommendation
```

**Result:** LLM analysis preserved, safety enforced, full audit trail

---

## What LLMs Do NOT Do

**Critical limitations (documented):**

**Source:** CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md, Section 2

**LLMs CANNOT:**
- ❌ Execute code directly
- ❌ Modify pipeline structure
- ❌ Change step ordering
- ❌ Invent new features
- ❌ Bypass validation
- ❌ Mutate core algorithms
- ❌ Access raw survivor data
- ❌ Make final decisions (proposals only)

**LLMs CAN:**
- ✅ Interpret diagnostics
- ✅ Detect drift patterns
- ✅ Propose parameter adjustments
- ✅ Recommend retraining
- ✅ Classify focus areas
- ✅ Suggest strategic pivots

**Documented quote:**
> "The LLM cannot: Rewrite mathematical logic, Invent new features,
> Bypass validation, Mutate control flow, Change step ordering.
> The LLM can: Interpret diagnostics, Detect drift patterns,
> Propose parameter adjustments, Recommend retraining."

---

## Execution Flow

### Complete Decision Path

```
New Draw Arrives
    ↓
Chapter 13 Diagnostics
    ↓
WATCHER Policy Check
    ↓
Parameter Advisor Called
    │
    ├─ Compute Metrics (Hit@K, churn, calibration)
    │
    ├─ Start DeepSeek (Primary)
    │   ├─ llm_lifecycle.ensure_running() → 3 seconds if cold
    │   ├─ Build context bundle
    │   ├─ Call with grammar constraint
    │   └─ Return structured JSON
    │
    ├─ Check Confidence
    │   ├─ If confidence ≥ 0.3 → Validate bounds → ACCEPT
    │   └─ If confidence < 0.3 → ESCALATE
    │
    ├─ Escalate to Claude (Backup)
    │   ├─ force_backup=True
    │   ├─ Provide full diagnostic history
    │   ├─ Free-form strategic analysis
    │   └─ Return recommendation
    │
    └─ Heuristic Fallback (Emergency)
        ├─ Log "DEGRADED_MODE"
        ├─ Apply threshold rules
        └─ Tag with degraded flag
    ↓
WATCHER Validation
    ├─ Are parameters within bounds? (clamp if needed)
    ├─ Does diagnosis make sense?
    └─ Is action safe?
    ↓
If valid → Execute (dispatch selfplay with parameters)
If invalid → Reject + log reason
```

**Source:** PROPOSAL_STRATEGY_ADVISOR_LIFECYCLE_INTEGRATION_v1_0.md, Section 3

---

## Documented Performance

### Verified Operation (Session 68)

**Source:** SESSION_CHANGELOG_20260208_S68.md

```
DeepSeek Primary:
  ✅ LLM server auto-started via lifecycle
  ✅ Grammar-constrained JSON response
  ✅ Bounds clamping: 1000 → 50 episodes
  ✅ Audit metadata present
  ✅ Recommendation saved: focus=REGIME_SHIFT, action=REFOCUS

Claude Backup:
  ✅ Routing via force_backup=True
  ✅ Valid JSON (markdown wrapper stripped)
  ✅ Substantive analysis: focus=CONFIDENCE_CALIBRATION
```

### Deployment Status (Session 75)

**Source:** SESSION_CHANGELOG_20260209_S75.md

```
Verified Files (Feb 7, 2026 on Zeus):
  ✅ parameter_advisor.py: 50,258 bytes
  ✅ agents/contexts/advisor_bundle.py: 23,630 bytes
  ✅ grammars/strategy_advisor.gbnf: 3,576 bytes
  ✅ llm_router.py: evaluate_with_grammar() integrated
  ✅ watcher_dispatch.py: Advisor integration present
  ✅ Python import test: PASSED
```

---

## Summary

### DeepSeek Strategy (Primary)
- **Analyzes:** Diagnostic metrics from last 15-20 draws
- **Classifies:** System state into 7 focus areas
- **Proposes:** Parameter adjustments with confidence scores
- **Constraint:** Grammar-constrained JSON (GBNF)
- **Speed:** 51 tok/sec
- **Escalates:** When confidence < 0.3

### Claude Strategy (Backup)
- **Analyzes:** Complex multi-draw patterns
- **Provides:** Deep strategic reasoning
- **Proposes:** Novel approaches beyond threshold logic
- **Constraint:** None (free-form)
- **Speed:** 38 tok/sec
- **Called:** When DeepSeek uncertain or explicitly requested

### Combined Strategy
- **DeepSeek:** 95% of decisions (routine analysis)
- **Claude:** 5% of decisions (complex patterns)
- **Heuristic:** <1% emergency fallback

**Key Innovation:** Neither LLM executes decisions. They propose, WATCHER validates, system executes.

**Documented quote:**
> "The result is a data-driven mimicry engine—an evolving model of the
> true PRNG's behavior that converges through iterative feedback."

---

**Document Status:** Verified from project documentation  
**Sources:** CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md, Session 68/75 changelogs, PROPOSAL docs  
**Version:** 1.0.0


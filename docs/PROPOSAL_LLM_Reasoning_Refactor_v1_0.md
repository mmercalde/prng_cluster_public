# PROPOSAL: LLM Reasoning Refactor v1.0

**Date:** 2026-01-04  
**Status:** PENDING APPROVAL  
**Author:** Claude (AI Assistant)  
**Reviewers:** Michael, Team Beta  

---

## 1. Problem Statement

### Current Architecture (Broken)
```
Raw Metrics → prompt_builder.py interprets → Pre-written phrases → LLM echoes → JSON output
```

The LLM is a **narrator**, not a **decision-maker**. It receives pre-interpreted conclusions and reformats them.

### Evidence
| LLM Output | Actual Source |
|------------|---------------|
| "bidirectional_count is very high, consider narrowing search space" | Hardcoded in `prompt_builder.py:161` |
| "bidirectional_count=353911 is fail" | Hardcoded threshold in `window_optimizer_context.py:52` |

### Consequence
- LLM GPU cycles wasted on text reformatting
- No actual reasoning about data characteristics
- Thresholds don't adapt to seed count, data type, or regime

---

## 2. Proposed Architecture

### New Flow
```
Raw Metrics + Context + Derived Rates → LLM reasons → Decision + Reasoning → Pydantic validation
```

### Separation of Concerns

| Component | Responsibility | Does NOT Do |
|-----------|---------------|-------------|
| `metrics_extractor.py` | Compute raw counts + derived rates | Interpret good/bad |
| `distributed_config.json` | Define threshold priors (rates, not absolutes) | Make decisions |
| `prompt_builder.py` | Format context + metrics for LLM | Pre-interpret outcomes |
| **LLM** | **Reason about rates + context → decision** | Echo templates |
| `watcher_agent.py` | Validate output, execute decision | Override LLM reasoning |
| Heuristic fallback | Handle LLM unavailable | Replace LLM reasoning |

---

## 3. Schema Definitions

### 3.1 Input to LLM (All Steps)

```python
# schemas/watcher_input.py
from pydantic import BaseModel
from typing import Dict, Any, Optional, Literal

class DataSource(BaseModel):
    type: Literal["synthetic", "real", "hybrid"]
    generator: Optional[str] = None  # If synthetic
    notes: Optional[str] = None      # If real, e.g., "Daily 3 CA"

class RunContext(BaseModel):
    pipeline_step: int
    step_id: str
    prng_hypothesis: Dict[str, Any]
    data_source: DataSource
    window: Optional[Dict[str, int]] = None

class WatcherInput(BaseModel):
    """Universal input schema for all steps."""
    context: RunContext
    raw_metrics: Dict[str, Any]
    derived_metrics: Dict[str, float]
    threshold_priors: Optional[Dict[str, Any]] = None  # From config
```

### 3.2 Output from LLM (All Steps)

```python
# schemas/watcher_decision.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Literal

class ReasoningChecks(BaseModel):
    """Self-report: did LLM follow the rules?"""
    used_rates: bool = Field(..., description="Used rates/ratios, not just absolute counts")
    mentioned_data_source: bool = Field(..., description="Referenced data_source.type")
    avoided_absolute_only: bool = Field(..., description="Did not base decision on counts alone")

class WatcherDecision(BaseModel):
    """Universal output schema for all steps."""
    decision: Literal["proceed", "retry", "escalate", "tighten", "widen", "investigate"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=20, max_length=500)
    primary_signal: str = Field(..., description="The metric that drove the decision")
    suggested_params: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    checks: ReasoningChecks
```

### 3.3 GBNF Grammar (Shared)

```gbnf
# grammars/watcher_decision.gbnf
root ::= "{" ws decision-block "}"

decision-block ::= 
  "\"decision\"" ws ":" ws decision-value ws "," ws
  "\"confidence\"" ws ":" ws number ws "," ws
  "\"reasoning\"" ws ":" ws string ws "," ws
  "\"primary_signal\"" ws ":" ws string ws "," ws
  "\"suggested_params\"" ws ":" ws (object | "null") ws "," ws
  "\"warnings\"" ws ":" ws array ws "," ws
  "\"checks\"" ws ":" ws checks-block

decision-value ::= "\"proceed\"" | "\"retry\"" | "\"escalate\"" | "\"tighten\"" | "\"widen\"" | "\"investigate\""

checks-block ::= "{" ws
  "\"used_rates\"" ws ":" ws boolean ws "," ws
  "\"mentioned_data_source\"" ws ":" ws boolean ws "," ws
  "\"avoided_absolute_only\"" ws ":" ws boolean ws
"}"

# ... standard JSON primitives ...
```

---

## 4. Per-Step Configuration

### 4.1 distributed_config.json Addition

```json
{
  "evaluation_thresholds": {
    "step_1_window_optimizer": {
      "bidirectional_rate": {"good_max": 0.05, "warn_max": 0.20, "fail_max": 0.50},
      "overlap_ratio": {"good_min": 0.20, "warn_min": 0.10},
      "min_seeds_tested": 100000
    },
    "step_2_scorer_meta": {
      "score_improvement": {"good_min": 0.05, "warn_min": 0.01},
      "convergence_rate": {"good_max": 50, "warn_max": 100}
    },
    "step_3_full_scoring": {
      "completion_rate": {"good_min": 0.95, "warn_min": 0.80},
      "feature_coverage": {"good_min": 62, "warn_min": 50}
    },
    "step_4_ml_meta": {
      "score_range": {"good_min": 0.01, "warn_min": 0.001},
      "feature_variance": {"good_min": 0.1, "warn_min": 0.01}
    },
    "step_5_anti_overfit": {
      "val_train_gap": {"good_max": 0.05, "warn_max": 0.15},
      "holdout_accuracy": {"good_min": 0.60, "warn_min": 0.50}
    },
    "step_6_prediction": {
      "confidence_mean": {"good_min": 0.70, "warn_min": 0.50},
      "prediction_entropy": {"good_max": 2.0, "warn_max": 3.0}
    }
  }
}
```

### 4.2 Per-Step Derived Metrics

| Step | Raw Metrics | Derived Metrics |
|------|-------------|-----------------|
| 1 | seeds_tested, forward_count, reverse_count, bidirectional_count | forward_rate, reverse_rate, bidirectional_rate, overlap_ratio |
| 2 | trials, best_score, convergence_trial | score_improvement, convergence_rate |
| 3 | survivors_scored, features_extracted, failed_chunks | completion_rate, feature_coverage |
| 4 | score_min, score_max, feature_variances | score_range, avg_feature_variance |
| 5 | train_loss, val_loss, holdout_hits | val_train_gap, holdout_accuracy |
| 6 | predictions, confidence_scores | confidence_mean, prediction_entropy |

---

## 5. File Modifications

### 5.1 Files to Modify

| File | Change |
|------|--------|
| `agents/prompt_builder.py` | Remove all semantic interpretation, emit raw + derived only |
| `agents/contexts/window_optimizer_context.py` | Remove hardcoded thresholds, load from config |
| `agents/contexts/*.py` (all 6) | Same pattern: remove hardcoded, load from config |
| `agents/watcher_agent.py` | Update evaluate_results() to use new schema |
| `distributed_config.json` | Add `evaluation_thresholds` section |

### 5.2 Files to Create

| File | Purpose |
|------|---------|
| `schemas/watcher_input.py` | Pydantic input schema |
| `schemas/watcher_decision.py` | Pydantic output schema |
| `grammars/watcher_decision.gbnf` | GBNF grammar for LLM output |
| `utils/metrics_extractor.py` | Compute derived metrics from raw |
| `tests/test_llm_reasoning_quality.py` | CI tests to catch template echoing |

### 5.3 Files Unchanged

| File | Reason |
|------|--------|
| `llm_services/llm_router.py` | Already supports grammar-constrained decoding |
| `agents/safety.py` | Halt mechanism unchanged |
| `agent_manifests/*.json` | May add `derived_metrics` field later |

---

## 6. Implementation Phases

### Phase 1: Infrastructure (Day 1)
1. Create `schemas/watcher_input.py`
2. Create `schemas/watcher_decision.py`
3. Create `grammars/watcher_decision.gbnf`
4. Create `utils/metrics_extractor.py`
5. Add `evaluation_thresholds` to `distributed_config.json`

### Phase 2: Step 1 Refactor (Day 1-2)
1. Modify `window_optimizer_context.py` - remove hardcoded thresholds
2. Modify `prompt_builder.py` - strip semantic interpretation
3. Update `watcher_agent.py` evaluate_results() for Step 1
4. Test Step 1 with LLM reasoning

### Phase 3: Steps 2-6 Refactor (Day 2-3)
1. Apply same pattern to each context file
2. Add derived metrics for each step
3. Test each step

### Phase 4: CI Tests (Day 3)
1. Create `test_llm_reasoning_quality.py`
2. Add pytest markers for LLM integration tests
3. Document test requirements

---

## 7. Example: Step 1 Before/After

### BEFORE (Current)

```python
# prompt_builder.py line 161
if bi_count > 10000:
    interpretation_parts.append(
        f"High survivor count ({bi_count}) - consider adjusting parameters to narrow search space."
    )
```

**LLM receives:** "High survivor count (353911) - consider adjusting parameters to narrow search space."  
**LLM outputs:** Same thing, reformatted.

### AFTER (Proposed)

```python
# metrics_extractor.py
def extract_step1_metrics(results: dict) -> WatcherInput:
    raw = {
        "seeds_tested": results["seed_count"],
        "forward_count": results["forward_count"],
        "reverse_count": results["reverse_count"],
        "bidirectional_count": results["bidirectional_count"],
    }
    derived = {
        "forward_rate": raw["forward_count"] / raw["seeds_tested"],
        "reverse_rate": raw["reverse_count"] / raw["seeds_tested"],
        "bidirectional_rate": raw["bidirectional_count"] / raw["seeds_tested"],
        "overlap_ratio": raw["bidirectional_count"] / min(raw["forward_count"], raw["reverse_count"]),
    }
    return WatcherInput(
        context=RunContext(
            pipeline_step=1,
            step_id="window_optimizer",
            prng_hypothesis={"prng_type": "java_lcg", "skip_mode": "constant"},
            data_source=DataSource(type="synthetic"),
        ),
        raw_metrics=raw,
        derived_metrics=derived,
    )
```

**LLM receives:**
```json
{
  "context": {"pipeline_step": 1, "data_source": {"type": "synthetic"}},
  "raw_metrics": {"seeds_tested": 1000000, "bidirectional_count": 353911},
  "derived_metrics": {"bidirectional_rate": 0.354, "overlap_ratio": 0.82}
}
```

**LLM reasons:** "bidirectional_rate of 35.4% is very high for synthetic data - this suggests thresholds are too loose. Recommend tightening forward_threshold and reverse_threshold."

---

## 8. Heuristic Fallback

When LLM is unavailable, use config-driven heuristics:

```python
def heuristic_evaluate_step1(derived: dict, thresholds: dict) -> WatcherDecision:
    """Fallback when LLM unavailable."""
    bi_rate = derived["bidirectional_rate"]
    th = thresholds["bidirectional_rate"]
    
    if bi_rate <= th["good_max"]:
        return WatcherDecision(
            decision="proceed",
            confidence=0.80,
            reasoning=f"bidirectional_rate {bi_rate:.4f} within good range",
            primary_signal="bidirectional_rate",
            checks=ReasoningChecks(used_rates=True, mentioned_data_source=False, avoided_absolute_only=True)
        )
    elif bi_rate <= th["warn_max"]:
        return WatcherDecision(
            decision="retry",
            confidence=0.60,
            reasoning=f"bidirectional_rate {bi_rate:.4f} in warning range",
            primary_signal="bidirectional_rate",
            suggested_params={"forward_threshold": 0.005, "reverse_threshold": 0.005},
            checks=ReasoningChecks(used_rates=True, mentioned_data_source=False, avoided_absolute_only=True)
        )
    else:
        return WatcherDecision(
            decision="escalate",
            confidence=0.70,
            reasoning=f"bidirectional_rate {bi_rate:.4f} exceeds fail threshold",
            primary_signal="bidirectional_rate",
            checks=ReasoningChecks(used_rates=True, mentioned_data_source=False, avoided_absolute_only=True)
        )
```

---

## 9. CI Test Summary

| Test | What It Catches |
|------|-----------------|
| `test_no_template_echoing` | LLM reasoning contains banned phrases |
| `test_rate_sensitivity` | Same counts, different seeds_tested → decision unchanged |
| `test_data_source_mentioned` | LLM doesn't reference synthetic vs real |
| `test_schema_compliance` | Output doesn't match Pydantic schema |

---

## 10. Approval Checklist

- [ ] **Michael:** Architecture approach approved
- [ ] **Team Beta:** Schema definitions approved
- [ ] **Michael:** Threshold values in config reasonable
- [ ] **Michael:** Implementation phases acceptable
- [ ] **Team Beta:** CI test strategy approved

---

## 11. Questions for Approval

1. **Threshold values:** Are the proposed rate thresholds (e.g., `bidirectional_rate.good_max: 0.05`) reasonable starting points?

2. **Decision vocabulary:** Is `["proceed", "retry", "escalate", "tighten", "widen", "investigate"]` sufficient or do we need more actions?

3. **Phase priority:** Should we complete Step 1 fully before touching Steps 2-6, or parallelize?

4. **Heuristic parity:** Should heuristic fallback produce identical decisions to LLM for the same inputs (deterministic)?

---

**END OF PROPOSAL**

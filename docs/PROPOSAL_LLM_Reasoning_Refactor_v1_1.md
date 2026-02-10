# PROPOSAL: LLM Reasoning Refactor v1.1

**Date:** 2026-01-04  
**Status:** TEAM BETA APPROVED (with conditions)  
**Author:** Claude (AI Assistant)  
**Reviewers:** Michael, Team Beta  

---

## Team Beta Approval Status

✅ **APPROVED** with 2 conditions (incorporated below):

1. ✅ Threshold priors keyed by `data_source.type` (synthetic/real/hybrid)
2. ✅ Decision vocabulary reduced to `{proceed, retry, escalate}`

---

## 1. Problem Statement

### Current Architecture (Broken)
```
Raw Metrics → prompt_builder.py interprets → Pre-written phrases → LLM echoes → JSON output
```

The LLM is a **narrator**, not a **decision-maker**.

### Evidence
| LLM Output | Actual Source |
|------------|---------------|
| "bidirectional_count is very high, consider narrowing search space" | Hardcoded in `prompt_builder.py:161` |
| "bidirectional_count=353911 is fail" | Hardcoded threshold in `window_optimizer_context.py:52` |

---

## 2. Proposed Architecture

### New Flow
```
Raw Metrics + Context + Derived Rates → LLM reasons → Decision + Reasoning → Pydantic validation
```

---

## 3. Schema Definitions (Updated per Team Beta)

### 3.1 Decision Schema (3 verbs only)

```python
# schemas/watcher_decision.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Literal
from enum import Enum

class RetryReason(str, Enum):
    """Why retry was recommended - expresses tighten/widen/investigate."""
    tighten = "tighten"          # Thresholds too loose, narrow search
    widen = "widen"              # Thresholds too tight, expand search
    rerun = "rerun"              # Transient failure, same params
    investigate = "investigate"  # Needs analysis before proceeding

class ReasoningChecks(BaseModel):
    """Self-report: did LLM follow the rules?"""
    used_rates: bool = Field(..., description="Used rates/ratios, not just absolute counts")
    mentioned_data_source: bool = Field(..., description="Referenced data_source.type")
    avoided_absolute_only: bool = Field(..., description="Did not base decision on counts alone")

class WatcherDecision(BaseModel):
    """Universal output schema for all steps - 3 VERBS ONLY."""
    decision: Literal["proceed", "retry", "escalate"]
    retry_reason: Optional[RetryReason] = None  # Required if decision=retry
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=20, max_length=500)
    primary_signal: str = Field(..., description="The metric that drove the decision")
    suggested_params: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    checks: ReasoningChecks
```

### 3.2 GBNF Grammar (Updated)

```gbnf
# grammars/watcher_decision.gbnf
root ::= "{" ws decision-block "}"

decision-block ::= 
  "\"decision\"" ws ":" ws decision-value ws "," ws
  "\"retry_reason\"" ws ":" ws (retry-reason-value | "null") ws "," ws
  "\"confidence\"" ws ":" ws number ws "," ws
  "\"reasoning\"" ws ":" ws string ws "," ws
  "\"primary_signal\"" ws ":" ws string ws "," ws
  "\"suggested_params\"" ws ":" ws (object | "null") ws "," ws
  "\"warnings\"" ws ":" ws array ws "," ws
  "\"checks\"" ws ":" ws checks-block

decision-value ::= "\"proceed\"" | "\"retry\"" | "\"escalate\""
retry-reason-value ::= "\"tighten\"" | "\"widen\"" | "\"rerun\"" | "\"investigate\""

checks-block ::= "{" ws
  "\"used_rates\"" ws ":" ws boolean ws "," ws
  "\"mentioned_data_source\"" ws ":" ws boolean ws "," ws
  "\"avoided_absolute_only\"" ws ":" ws boolean ws
"}"

ws ::= [ \t\n]*
boolean ::= "true" | "false"
number ::= [0-9]+ ("." [0-9]+)?
string ::= "\"" [^"]* "\""
array ::= "[" ws (string (ws "," ws string)*)? ws "]"
object ::= "{" ws (string ws ":" ws (string | number | boolean | "null") (ws "," ws string ws ":" ws (string | number | boolean | "null"))*)? ws "}"
```

---

## 4. Threshold Priors (Keyed by data_source.type)

### 4.1 distributed_config.json Structure

```json
{
  "evaluation_thresholds": {
    "step_1_window_optimizer": {
      "synthetic": {
        "bidirectional_rate": {"good_max": 0.02, "warn_max": 0.10, "fail_max": 0.30},
        "forward_rate": {"warn_max": 0.25},
        "reverse_rate": {"warn_max": 0.25},
        "overlap_ratio": {"good_min": 0.25, "warn_min": 0.12},
        "min_seeds_tested": 250000
      },
      "real": {
        "bidirectional_rate": {"good_max": 0.005, "warn_max": 0.03, "fail_max": 0.10},
        "forward_rate": {"warn_max": 0.10},
        "reverse_rate": {"warn_max": 0.10},
        "overlap_ratio": {"good_min": 0.20, "warn_min": 0.10},
        "min_seeds_tested": 500000
      },
      "hybrid": {
        "bidirectional_rate": {"good_max": 0.01, "warn_max": 0.06, "fail_max": 0.20},
        "forward_rate": {"warn_max": 0.15},
        "reverse_rate": {"warn_max": 0.15},
        "overlap_ratio": {"good_min": 0.20, "warn_min": 0.10},
        "min_seeds_tested": 350000
      }
    }
  }
}
```

### 4.2 Team Beta Rationale

| Source Type | bidirectional_rate.good_max | Rationale |
|-------------|----------------------------|-----------|
| synthetic | 0.02 (2%) | Clean overlap, avoid proceeding while too wide |
| real | 0.005 (0.5%) | Operational noise should NOT yield high rates |
| hybrid | 0.01 (1%) | Conservative interpolation |

---

## 5. Decision Mapping (Old → New)

| Old Decision | New Representation |
|--------------|-------------------|
| `"proceed"` | `decision="proceed"` |
| `"tighten"` | `decision="retry", retry_reason="tighten", suggested_params={thresholds↑}` |
| `"widen"` | `decision="retry", retry_reason="widen", suggested_params={thresholds↓}` |
| `"investigate"` | `decision="escalate", warnings=["needs_investigation"]` |
| `"escalate"` | `decision="escalate"` |

---

## 6. Implementation Phases

### Phase 1: Infrastructure (Day 1)
1. Create `schemas/watcher_input.py`
2. Create `schemas/watcher_decision.py` (with 3 verbs)
3. Create `grammars/watcher_decision.gbnf`
4. Create `utils/metrics_extractor.py`
5. Add `evaluation_thresholds` (keyed by data_source.type) to `distributed_config.json`

### Phase 2: Step 1 Refactor (Day 1-2)
1. Modify `window_optimizer_context.py` - remove hardcoded thresholds
2. Modify `prompt_builder.py` - strip semantic interpretation
3. Update `watcher_agent.py` evaluate_results() for Step 1
4. Test Step 1 with LLM reasoning

### Phase 3: Steps 2-6 Refactor (Day 2-3)
1. Apply same pattern sequentially (not parallel)
2. Add threshold priors for each step
3. Test each step

### Phase 4: CI Tests (Day 3)
1. Create `test_llm_reasoning_quality.py`
2. Add pytest markers for LLM integration tests

---

## 7. Heuristic Fallback (Deterministic)

```python
def heuristic_evaluate_step1(
    derived: dict, 
    thresholds: dict,  # Keyed by data_source.type
    data_source_type: str
) -> WatcherDecision:
    """
    Deterministic fallback when LLM unavailable.
    - Config-driven, no hardcoded values
    - Conservative: when in doubt, retry or escalate
    """
    th = thresholds.get(data_source_type, thresholds.get("real"))  # Default to strictest
    bi_rate = derived["bidirectional_rate"]
    
    if bi_rate <= th["bidirectional_rate"]["good_max"]:
        return WatcherDecision(
            decision="proceed",
            confidence=0.80,
            reasoning=f"bidirectional_rate {bi_rate:.4f} within good range for {data_source_type} data",
            primary_signal="bidirectional_rate",
            checks=ReasoningChecks(used_rates=True, mentioned_data_source=True, avoided_absolute_only=True)
        )
    elif bi_rate <= th["bidirectional_rate"]["warn_max"]:
        return WatcherDecision(
            decision="retry",
            retry_reason=RetryReason.tighten,
            confidence=0.60,
            reasoning=f"bidirectional_rate {bi_rate:.4f} in warning range for {data_source_type}",
            primary_signal="bidirectional_rate",
            suggested_params={"forward_threshold": 0.005, "reverse_threshold": 0.005},
            checks=ReasoningChecks(used_rates=True, mentioned_data_source=True, avoided_absolute_only=True)
        )
    else:
        return WatcherDecision(
            decision="escalate",
            confidence=0.70,
            reasoning=f"bidirectional_rate {bi_rate:.4f} exceeds fail threshold for {data_source_type}",
            primary_signal="bidirectional_rate",
            warnings=["rate_too_high", "human_review_recommended"],
            checks=ReasoningChecks(used_rates=True, mentioned_data_source=True, avoided_absolute_only=True)
        )
```

---

## 8. Files Summary

### To Modify
| File | Change |
|------|--------|
| `agents/prompt_builder.py` | Remove semantic interpretation |
| `agents/contexts/window_optimizer_context.py` | Remove hardcoded thresholds |
| `agents/watcher_agent.py` | Update evaluate_results() |
| `distributed_config.json` | Add evaluation_thresholds (keyed by type) |

### To Create
| File | Purpose |
|------|---------|
| `schemas/watcher_input.py` | Pydantic input schema |
| `schemas/watcher_decision.py` | Pydantic output schema (3 verbs) |
| `grammars/watcher_decision.gbnf` | GBNF grammar |
| `utils/metrics_extractor.py` | Compute derived metrics |
| `tests/test_llm_reasoning_quality.py` | CI tests |

---

## 9. Approval Checklist

- [x] **Team Beta:** Architecture approach approved
- [x] **Team Beta:** Threshold priors keyed by data_source.type
- [x] **Team Beta:** Decision vocabulary = {proceed, retry, escalate}
- [ ] **Michael:** Final approval to implement

---

## 10. Guiding Principle

> "If a line of code can be rewritten as an English conclusion, it does not belong in prompt_builder.py. It belongs in config (as priors) or in the LLM output (as decisions)."
> — Team Beta

---

**END OF PROPOSAL v1.1**

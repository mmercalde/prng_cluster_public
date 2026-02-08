# PROPOSAL: Search Strategy Visibility Fix

**Version:** 1.0.0  
**Date:** February 7, 2026  
**Author:** Claude (AI Assistant)  
**Status:** PROPOSED  
**Severity:** Integration Gap — Advisory Layer Blindness  
**Affects:** Strategy Advisor, Chapter 13, Chapter 14, WATCHER, GBNF Grammars, Manifests, Parameter Registry

---

## 1. Problem Statement

The `search_strategy` parameter — which controls the Optuna sampler used in Step 1 (Window Optimizer) — is **fully implemented at the execution layer** but **invisible to every advisory and governance layer** in the system.

**What works today:**
- CLI: `--strategy bayesian|random|grid|evolutionary` ✅
- Manifest `window_optimizer.json`: `"search_strategy": {"type": "choice", "choices": ["bayesian", "random", "grid"]}` ✅
- WATCHER dispatch: `--params '{"strategy":"random"}'` ✅
- All 4 Optuna samplers (TPE, Random, Grid, CmaES) implemented ✅

**What is broken:**
- Section 18.1 parameter bounds table: **`search_strategy` NOT LISTED**
- Strategy Advisor contract (Section 5.1): `selfplay_overrides` schema has **no `search_strategy` field**
- `strategy_advisor.gbnf`: grammar cannot produce `search_strategy` recommendations
- Chapter 14 `_is_within_policy_bounds()`: whitelist does **not include `search_strategy`**
- `parameter_adjustment.gbnf`: free-form string allows it, but bounds validation silently rejects it
- `chapter_13.gbnf`: same — LLM could propose it, WATCHER would reject as unknown parameter
- Manifest `window_optimizer.json`: lists only 3 of 4 strategies (missing `evolutionary`)

**Impact:** The Strategy Advisor — the component specifically designed to make intelligent decisions about *how* to optimize — literally cannot recommend changing the optimization strategy. This is the equivalent of a chess engine that can't recommend which opening to play.

---

## 2. Root Cause Analysis

The gap emerged from **bottom-up implementation** without **top-down schema synchronization**:

| When | What Happened | Gap Created |
|------|---------------|-------------|
| Session 8 (Dec 2025) | 4 search strategies implemented in `window_optimizer.py` | Execution layer complete |
| Session 8 (Dec 2025) | CLI expanded to `bayesian\|random\|grid\|evolutionary` | CLI complete |
| Session 8 (Dec 2025) | Manifest updated with `search_strategy` choices | Manifest mostly complete (missing evolutionary) |
| Jan 12, 2026 | `chapter_13.gbnf` v1.0.0 created | No `search_strategy` awareness |
| Jan 30, 2026 | Chapter 13 Section 18 parameter table written | `search_strategy` omitted from Step 1 table |
| Feb 1, 2026 | `parameter_adjustment.gbnf` created | Free-form string, but bounds reject unknowns |
| Feb 3, 2026 | Strategy Advisor contract v1.0 proposed | `selfplay_overrides` schema omits strategy |
| Feb 4, 2026 | Chapter 14 `_is_within_policy_bounds()` implemented | Whitelist does not include `search_strategy` |

Every layer was built correctly in isolation. The parameter simply fell through the cracks between execution and governance.

---

## 3. Affected Components — Complete Inventory

### 3.1 Files Requiring Modification

| # | File | Location | What's Missing | Fix Type |
|---|------|----------|----------------|----------|
| 1 | `CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` | Section 18.1 | `search_strategy` row in parameter bounds table | Documentation |
| 2 | `agent_manifests/window_optimizer.json` | `choices` array | `evolutionary` not in choices list | Config |
| 3 | `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md` | Section 5.1 | `selfplay_overrides` missing `search_strategy` | Documentation |
| 4 | `strategy_advisor.gbnf` | `selfplay-overrides` rule | No `search_strategy` production rule | Grammar |
| 5 | `chapter_13.gbnf` | N/A (free-form string) | Not grammar-blocked, but bounds-blocked | No change needed* |
| 6 | `parameter_adjustment.gbnf` | N/A (free-form string) | Not grammar-blocked, but bounds-blocked | No change needed* |
| 7 | Chapter 14 `_is_within_policy_bounds()` | Whitelist dict | `search_strategy` not in allowed params | Code |
| 8 | `watcher_policies.json` | Parameter bounds section | No `search_strategy` entry | Config |
| 9 | `bundle_factory.py` | STEP_GUARDRAILS[1] | No mention of strategy as adjustable | Code |
| 10 | `CHAPTER_1_WINDOW_OPTIMIZER.md` | Section 10 | CLI documents 4 strategies but no governance link | Documentation |
| 11 | `Cluster_operating_manual.txt` | Strategy section | Documents 4 strategies but no governance link | Documentation |
| 12 | `PROPOSAL_Unified_Agent_Context_Framework_v3_2_0.md` | Example prompt | Example shows trials/max-seeds but not strategy | Documentation |

*`chapter_13.gbnf` and `parameter_adjustment.gbnf` use free-form strings for parameter names, so they don't block `search_strategy` at the grammar level. However, the bounds validation in WATCHER and Chapter 14 silently rejects unknown parameter names — which is the real gate.

### 3.2 Files NOT Requiring Modification (Confirmed Clean)

| File | Why Clean |
|------|-----------|
| `window_optimizer.py` | CLI already supports all 4 strategies |
| `coordinator.py` | Passes strategy through correctly |
| `window_optimizer_bayesian.py` | TPE sampler works; others route correctly |
| `window_optimizer_integration_final.py` | Strategy map has all 4 entries |
| `selfplay_orchestrator.py` | Receives overrides as config overlay — passthrough |
| `agents/watcher_agent.py` | Dispatch logic passes params through — no filtering |

---

## 4. Proposed Fixes

### Fix 1: Section 18.1 Parameter Bounds Table (Chapter 13 v1.1)

**File:** `CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md`  
**Location:** Section 18.1 — Step 1 Window Optimizer

Add row to the parameter table:

```markdown
| Parameter | Location | Type | Bounds | Effect |
|-----------|----------|------|--------|--------|
| `window_size` | `optimal_window_config.json` | int | 100-500 | History window for sieve |
| `offset` | `optimal_window_config.json` | int | 0-50 | Starting offset |
| `skip_min` / `skip_max` | `optimal_window_config.json` | int | 1-500 | Skip range for PRNG |
| `forward_threshold` | `optimal_window_config.json` | float | 0.01-0.15 | Forward sieve tolerance |
| `reverse_threshold` | `optimal_window_config.json` | float | 0.01-0.15 | Reverse sieve tolerance |
| `trials` | `agent_manifests/window_optimizer.json` | int | 10-200 | Optuna trials |
| **`search_strategy`** | **`agent_manifests/window_optimizer.json`** | **choice** | **bayesian/random/grid/evolutionary** | **Optuna sampler selection** |
```

**Rationale:** This is THE parameter table that WATCHER and the Strategy Advisor validate against. Without this row, any proposal touching `search_strategy` is silently rejected by Invariant 3 ("All parameter proposals MUST fall within Section 18 bounds").

---

### Fix 2: Window Optimizer Manifest — Add `evolutionary`

**File:** `agent_manifests/window_optimizer.json` (on Zeus)  
**Current:**
```json
"search_strategy": {"type": "choice", "choices": ["bayesian", "random", "grid"], "default": "bayesian"}
```

**Proposed:**
```json
"search_strategy": {"type": "choice", "choices": ["bayesian", "random", "grid", "evolutionary"], "default": "bayesian"}
```

**Rationale:** The CLI and `window_optimizer_integration_final.py` both support `evolutionary`. The manifest should match.

---

### Fix 3: Strategy Advisor Schema — Add `search_strategy` to `selfplay_overrides`

**File:** `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md`  
**Location:** Section 5.1 JSON Schema

**Current `selfplay_overrides`:**
```json
"selfplay_overrides": {
    "max_episodes": 15,
    "model_types": ["catboost", "lightgbm"],
    "min_fitness_threshold": 0.55,
    "priority_metrics": ["pool_concentration", "model_agreement"],
    "exploration_ratio": 0.3
}
```

**Proposed `selfplay_overrides`:**
```json
"selfplay_overrides": {
    "max_episodes": 15,
    "model_types": ["catboost", "lightgbm"],
    "min_fitness_threshold": 0.55,
    "priority_metrics": ["pool_concentration", "model_agreement"],
    "exploration_ratio": 0.3,
    "search_strategy": "random"
}
```

**Also add to Section 5.1 `parameter_proposals` example:**
```json
{
    "parameter": "search_strategy",
    "current_value": "bayesian",
    "proposed_value": "random",
    "delta": "change",
    "confidence": 0.80,
    "rationale": "First run on new PRNG — random survey of parameter space before Bayesian narrowing"
}
```

---

### Fix 4: Strategy Advisor GBNF — Add `search_strategy` to grammar

**File:** `strategy_advisor.gbnf` (to be created)  
**Location:** `selfplay-overrides` rule

The contract shows a grammar summary but the full `selfplay-overrides` rule is not expanded. When implementing, include:

```gbnf
# Strategy Advisor Grammar — selfplay overrides section
# Must include search_strategy as a producible field

selfplay-overrides ::= "{" ws
    selfplay-override-pair (ws "," ws selfplay-override-pair)*
    ws "}"

selfplay-override-pair ::= 
    "\"max_episodes\"" ws ":" ws number |
    "\"model_types\"" ws ":" ws string-array |
    "\"min_fitness_threshold\"" ws ":" ws number |
    "\"priority_metrics\"" ws ":" ws string-array |
    "\"exploration_ratio\"" ws ":" ws number |
    "\"search_strategy\"" ws ":" ws search-strategy-value

search-strategy-value ::= "\"bayesian\"" | "\"random\"" | "\"grid\"" | "\"evolutionary\""
```

**Rationale:** Without this grammar rule, the LLM physically cannot output `search_strategy` recommendations even if the prompt tells it about the parameter.

---

### Fix 5: Chapter 14 `_is_within_policy_bounds()` — Add `search_strategy`

**File:** Chapter 14 implementation (when built)  
**Location:** `_is_within_policy_bounds()` method

**Current whitelist:**
```python
allowed_params = {
    'normalize_features': {'type': 'bool'},
    'nn_activation': {'type': 'enum', 'values': [0, 1, 2]},
    'learning_rate': {'type': 'float', 'min': 1e-6, 'max': 0.1},
    'dropout': {'type': 'float', 'min': 0.0, 'max': 0.5},
    'n_estimators': {'type': 'int', 'min': 50, 'max': 2000},
    'max_depth': {'type': 'int', 'min': 3, 'max': 15},
}
```

**Proposed addition:**
```python
allowed_params = {
    # ... existing entries ...
    'search_strategy': {'type': 'enum', 'values': ['bayesian', 'random', 'grid', 'evolutionary']},
}
```

**Rationale:** Chapter 14's `_update_strategy_advisor()` writes `strategy_recommendation.json` which WATCHER reads. If the recommendation includes `search_strategy`, `_is_within_policy_bounds()` must recognize it or the proposal is silently dropped.

---

### Fix 6: `watcher_policies.json` — Add bounds entry

**File:** `watcher_policies.json` (on Zeus)  
**Location:** Parameter bounds section (if present; otherwise add)

```json
{
    "parameter_bounds": {
        "search_strategy": {
            "type": "choice",
            "choices": ["bayesian", "random", "grid", "evolutionary"],
            "default": "bayesian"
        }
    }
}
```

---

### Fix 7: `bundle_factory.py` — Add guardrail mention

**File:** `bundle_factory.py`  
**Location:** `STEP_GUARDRAILS[1]` list

**Current:**
```python
1: [
    "Low thresholds (0.01-0.15) maximize seed discovery — do not recommend high thresholds.",
    "Bidirectional intersection handles filtering, not individual thresholds.",
    ...
],
```

**Add:**
```python
1: [
    "Low thresholds (0.01-0.15) maximize seed discovery — do not recommend high thresholds.",
    "Bidirectional intersection handles filtering, not individual thresholds.",
    "search_strategy is adjustable: bayesian (default, best for narrowing), random (exploration/survey), grid (exhaustive near known params), evolutionary (complex spaces, many iterations).",
    ...
],
```

**Rationale:** The guardrails are injected into the LLM prompt. Without this line, the LLM evaluating Step 1 results doesn't know `search_strategy` exists as a knob it can recommend adjusting.

---

### Fix 8: Strategy Advisor Prompt Template — Add context

**File:** `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md`  
**Location:** Section 7 (Advisor Prompt Template)

Add to `CURRENT CONFIGURATION` section:
```
CURRENT SEARCH STRATEGY: {{ current search_strategy from last window_optimizer run }}

STRATEGY OPTIONS:
- bayesian: Optuna TPE sampler — learns from previous trials, best for narrowing (RECOMMENDED for most cases)
- random: Random sampling — good for first run on new PRNG, quick survey of parameter space
- grid: Exhaustive grid search — good for testing specific hypotheses near known good params
- evolutionary: CMA-ES — good for complex spaces with many iterations, avoids local optima
```

---

### Fix 9: Appendix A Quick Reference — Add strategy scenarios

**File:** `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md`  
**Location:** Appendix A

**Add:**
```
Q: "First run on a new PRNG algorithm"
A: Focus = POOL_COVERAGE
   → search_strategy: random, trials: 10 (quick survey)
   → Then switch to bayesian once promising region found

Q: "Bayesian stuck in local optimum, survivors not improving"
A: Focus = FEATURE_RELEVANCE
   → search_strategy: evolutionary (CMA-ES escapes local optima)
   → Increase trials to 100+

Q: "Known good window, testing threshold edge cases"
A: Focus = POOL_PRECISION
   → search_strategy: grid (exhaustive near known params)
   → Reduce trials, increase precision

Q: "Stable regime, maintenance mode"
A: Focus = STEADY_STATE
   → search_strategy: bayesian (default, efficient)
   → Reduce trials to save compute
```

---

## 5. Integration Chain — How the Fix Flows

```
Strategy Advisor prompt (Fix 8)
    │ LLM sees search_strategy as adjustable
    ▼
strategy_advisor.gbnf (Fix 4)
    │ Grammar allows producing search_strategy value
    ▼
strategy_recommendation.json
    │ Contains {"parameter": "search_strategy", "proposed_value": "random"}
    ▼
WATCHER reads recommendation
    │ Validates against Section 18 bounds (Fix 1)
    │ Checks watcher_policies.json bounds (Fix 6)
    ▼
dispatch_selfplay() / dispatch_pipeline()
    │ Passes --params '{"strategy":"random"}' 
    ▼
window_optimizer.py CLI
    │ --strategy random (already works)
    ▼
Optuna RandomSampler (already works)
```

Without the fixes, the chain breaks at step 2 (grammar can't produce it) and step 4 (WATCHER silently rejects it).

---

## 6. Implementation Order

| Priority | Fix | Effort | Dependency |
|----------|-----|--------|------------|
| **P0** | Fix 1: Section 18.1 parameter table | 2 min | None — documentation |
| **P0** | Fix 2: Manifest add evolutionary | 1 min | None — config |
| **P1** | Fix 5: Chapter 14 whitelist | 5 min | Chapter 14 implementation |
| **P1** | Fix 6: watcher_policies.json | 2 min | None — config |
| **P1** | Fix 7: bundle_factory.py guardrail | 2 min | None — code |
| **P2** | Fix 3: Strategy Advisor schema | 5 min | Contract acceptance |
| **P2** | Fix 4: strategy_advisor.gbnf | 10 min | Grammar implementation |
| **P2** | Fix 8: Advisor prompt template | 5 min | Contract acceptance |
| **P2** | Fix 9: Appendix A scenarios | 3 min | Contract acceptance |

**Total estimated effort: ~35 minutes**

P0 fixes can be applied immediately (no code changes, just docs and config).  
P1 fixes apply during Chapter 14 implementation.  
P2 fixes apply during Strategy Advisor implementation.

---

## 7. Verification Checklist

After all fixes applied:

- [ ] `grep -r "search_strategy" agent_manifests/` shows all 4 choices including evolutionary
- [ ] Section 18.1 has `search_strategy` row with `bayesian/random/grid/evolutionary` bounds
- [ ] `strategy_advisor.gbnf` can produce `"search_strategy": "random"` (grammar test)
- [ ] `_is_within_policy_bounds("search_strategy", "random")` returns `True`
- [ ] `_is_within_policy_bounds("search_strategy", "invalid")` returns `False`
- [ ] Bundle factory Step 1 guardrails mention search_strategy
- [ ] Strategy Advisor prompt template includes strategy context
- [ ] Strategy Advisor example JSON includes `search_strategy` in both `selfplay_overrides` and `parameter_proposals`
- [ ] WATCHER can apply `search_strategy` override from `strategy_recommendation.json`
- [ ] End-to-end: Advisor recommends "random" → WATCHER validates → dispatch passes `--strategy random` → window_optimizer uses RandomSampler

---

## 8. Design Note: Why Not a Frozen Parameter?

One might argue `search_strategy` should be frozen (like PRNG algorithms and sieve math). It should NOT be frozen because:

1. **Strategy selection is inherently adaptive** — the best strategy changes based on the optimization phase
2. **It's already configurable** — the CLI, manifest, and WATCHER dispatch all support it
3. **The Strategy Advisor exists precisely for this** — deciding WHEN to use which strategy is a core advisory function
4. **No safety risk** — changing the sampler doesn't affect data integrity, just exploration efficiency

The only parameters that should remain frozen are those where changes would invalidate mathematical invariants (sieve math, feature schema, PRNG algorithm list).

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-07 | Initial proposal — 9 fixes across 12 files |

---

**END OF PROPOSAL**

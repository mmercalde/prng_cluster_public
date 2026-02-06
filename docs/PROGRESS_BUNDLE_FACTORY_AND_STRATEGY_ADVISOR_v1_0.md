# PROGRESS: Bundle Factory v1.1.0 + Strategy Advisor Implementation

**Document:** PROGRESS_BUNDLE_FACTORY_AND_STRATEGY_ADVISOR_v1_0.md  
**Version:** 1.0.0  
**Date:** 2026-02-05  
**Session:** 60  
**Status:** Ready for Execution  

---

## 1. Current Status Summary

| Milestone | Status | Notes |
|-----------|--------|-------|
| Soak Test A (daemon endurance) | ✅ COMPLETE | Verified in chat history |
| Soak Test B (10 back-to-back requests) | ✅ COMPLETE + CERTIFIED | Session 59 |
| Soak Test C (full autonomous loop) | ⬜ NEXT | User executing after this document |
| bundle_factory v1.1.0 | ✅ READY | Files prepared this session |
| Strategy Advisor code | ⬜ IMPLEMENT IMMEDIATELY | After Soak C + v1.1.0 deploy |

---

## 2. Contract Erratum: CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md

### 2.1 The Inconsistency

The Strategy Advisor contract contains an internal contradiction:

| Location | Statement |
|----------|-----------|
| Section 14, Line 671 | `bundle_factory.py` — Extended with `build_advisor_bundle()` |
| Section 8.1 File Structure | `agents/contexts/advisor_bundle.py` — Bundle factory extension (~150 lines) |

These are mutually exclusive approaches:
- Option A: Add function to existing `bundle_factory.py`
- Option B: Create separate `advisor_bundle.py` file

### 2.2 Resolution

**DECISION:** Option A — Add `build_advisor_bundle()` directly to `bundle_factory.py`

**Rationale:**
1. Matches Chapter 14's pattern (`build_diagnostics_bundle()` goes in `bundle_factory.py`)
2. Matches v1.1.0's pattern (selfplay evaluation added to `bundle_factory.py`)
3. Single source of truth for all bundle construction
4. Avoids import complexity and circular dependencies
5. All missions, schemas, guardrails in one file for easy audit

**Contract Amendment:**

Section 8.1 should read:

```
| File | Purpose | Size Est. |
|------|---------|-----------|
| `parameter_advisor.py` | Main advisor module | ~400 lines |
| `strategy_advisor.gbnf` | Grammar constraint | ~80 lines |
| `strategy_recommendation.json` | Output file (overwritten each cycle) | ~2KB |
| `strategy_history/` | Archive of past recommendations | ~2KB each |
| `bundle_factory.py` | Add build_advisor_bundle() function | ~100 lines added |
```

~~`agents/contexts/advisor_bundle.py`~~ — REMOVED (consolidated into bundle_factory.py)

### 2.3 Authority

This erratum is proposed by Session 60 for Team Beta approval. The contract intent is preserved; only the file organization is clarified.

---

## 3. Execution Sequence

### Phase 1: Soak Test C (User executes NOW)
```
Duration: 2-4 hours
Command: PYTHONPATH=. python3 agents/watcher_agent.py --daemon
Test: Full autonomous loop (Ch13 → WATCHER → Selfplay → Ch13)
Pass Criteria: No crashes, proper state transitions, audit logs clean
```

### Phase 2: Deploy bundle_factory v1.1.0 (After Soak C passes)
```bash
# On ser8:
scp ~/Downloads/bundle_factory.py zeus:~/distributed_prng_analysis/agents/contexts/
scp ~/Downloads/SPEC_BUNDLE_FACTORY_v1_1_0.md zeus:~/distributed_prng_analysis/docs/

# On Zeus:
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 agents/contexts/bundle_factory.py  # Self-test

# Git commit:
git add agents/contexts/bundle_factory.py docs/SPEC_BUNDLE_FACTORY_v1_1_0.md
git commit -m "bundle_factory v1.1.0: MAIN_MISSION + selfplay evaluation context"
git push
```

### Phase 3: Implement Strategy Advisor (Immediately after Phase 2)

#### 3.1 Create `agent_grammars/strategy_advisor.gbnf`
```
# Strategy Advisor Grammar
# LLM output constraint for strategy recommendations
#
# VERSION: 1.0.0
# DATE: 2026-02-05

root ::= "{" ws
    "\"focus_area\"" ws ":" ws focus-area ws "," ws
    "\"focus_confidence\"" ws ":" ws confidence-value ws "," ws
    "\"focus_rationale\"" ws ":" ws string ws "," ws
    "\"recommended_action\"" ws ":" ws advisor-action ws "," ws
    "\"selfplay_overrides\"" ws ":" ws selfplay-overrides ws "," ws
    "\"parameter_proposals\"" ws ":" ws parameter-proposals ws "," ws
    "\"risk_level\"" ws ":" ws risk-level ws "," ws
    "\"requires_human_review\"" ws ":" ws boolean
    ws "}"

focus-area ::= "\"POOL_PRECISION\"" | "\"POOL_COVERAGE\"" |
               "\"CONFIDENCE_CALIBRATION\"" | "\"MODEL_DIVERSITY\"" |
               "\"FEATURE_RELEVANCE\"" | "\"REGIME_SHIFT\"" |
               "\"STEADY_STATE\""

advisor-action ::= "\"RETRAIN\"" | "\"WAIT\"" | "\"ESCALATE\"" |
                   "\"REFOCUS\"" | "\"FULL_RESET\""

selfplay-overrides ::= "{" ws
    (selfplay-override (ws "," ws selfplay-override)*)? ws "}"

selfplay-override ::= string ws ":" ws (string | number | boolean)

parameter-proposals ::= "[]" |
    "[" ws parameter-proposal (ws "," ws parameter-proposal)* ws "]"

parameter-proposal ::= "{" ws
    "\"parameter\"" ws ":" ws string ws "," ws
    "\"current_value\"" ws ":" ws (number | "null") ws "," ws
    "\"proposed_value\"" ws ":" ws number ws "," ws
    "\"confidence\"" ws ":" ws confidence-value ws "," ws
    "\"rationale\"" ws ":" ws string
    ws "}"

risk-level ::= "\"low\"" | "\"medium\"" | "\"high\""

confidence-value ::= "0" ("." [0-9]+)? | "1" (".0")? | "0." [0-9]+

boolean ::= "true" | "false"

number ::= "-"? [0-9]+ ("." [0-9]+)?

string ::= "\"" string-content "\""
string-content ::= ([^"\\] | "\\" ["\\/bfnrt])*

ws ::= [ \t\n\r]*
```

#### 3.2 Add to `bundle_factory.py` (after v1.1.0 deployed)

```python
# =============================================================================
# STRATEGY ADVISOR CONTEXT (CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md)
# =============================================================================

STRATEGY_ADVISOR_MISSION = (
    "Strategy Advisor: Analyze Chapter 13 diagnostics and selfplay telemetry "
    "to produce mathematically grounded recommendations for selfplay focus. "
    "You consume: post_draw_diagnostics (last 15-20 draws), telemetry history "
    "(last 20 selfplay episodes), pool performance metrics, and policy history "
    "(promoted + rejected). Classify the system's current state into a focus "
    "area, propose selfplay configuration overrides, and suggest parameter "
    "adjustments with specific rationale citing diagnostic values. "
    "Your recommendations are PROPOSALS — WATCHER validates against policy "
    "bounds before any execution. If uncertainty is high, recommend WAIT."
)

STRATEGY_ADVISOR_SCHEMA_EXCERPT = (
    "StrategyRecommendation: key_fields=[focus_area, focus_confidence, "
    "focus_rationale, recommended_action, selfplay_overrides, parameter_proposals, "
    "risk_level, requires_human_review]. "
    "focus_area enum: POOL_PRECISION, POOL_COVERAGE, CONFIDENCE_CALIBRATION, "
    "MODEL_DIVERSITY, FEATURE_RELEVANCE, REGIME_SHIFT, STEADY_STATE. "
    "recommended_action enum: RETRAIN, WAIT, ESCALATE, REFOCUS, FULL_RESET. "
    "parameter_proposals: array of {parameter, current_value, proposed_value, "
    "confidence, rationale}. Every proposal MUST cite specific diagnostic values."
)

STRATEGY_ADVISOR_GRAMMAR = "strategy_advisor.gbnf"

STRATEGY_ADVISOR_GUARDRAILS = [
    "You are ADVISORY ONLY — you do NOT execute, modify files, or bypass WATCHER.",
    "Every parameter proposal MUST cite specific diagnostic values. 'Increase X' without data is INVALID.",
    "All proposals must fall within Section 18 bounds (watcher_policies.json). Out-of-bounds = rejected.",
    "You MUST NOT propose changes to: step ordering, feature schema, PRNG algorithms, sieve math.",
    "Activation requires ≥15 draws in diagnostics_history/. If insufficient data, recommend WAIT.",
    "Check cooldown_runs before proposing changes. Proposals violating cooldown are deferred.",
    "If uncertainty is high (no clear signal), recommend WAIT — noise is worse than inaction.",
    "You do NOT predict draws or recommend specific numbers. You optimize the learning system.",
]


def build_advisor_bundle(
    diagnostics_history: List[Dict],
    telemetry_history: List[Dict],
    pool_metrics: Dict,
    policy_history: List[Dict],
    current_config: Dict,
    budgets: TokenBudget = None,
) -> StepAwarenessBundle:
    """
    Build an LLM context bundle for Strategy Advisor analysis.
    
    Called by parameter_advisor.py when activation gate is satisfied
    (≥15 draws, ≥10 selfplay episodes, ≥1 promoted policy).
    
    Args:
        diagnostics_history: Last 15-20 post_draw_diagnostics.json entries
        telemetry_history: Last 20 selfplay episode summaries
        pool_metrics: Current pool performance (Hit@K, concentration, stability)
        policy_history: Last 5-10 promoted + rejected policies with reasons
        current_config: Active selfplay config + parameter values
        budgets: Token budget override
        
    Returns:
        StepAwarenessBundle ready for render_prompt_from_bundle()
    """
    if budgets is None:
        budgets = TokenBudget()
    
    run_id = f"advisor_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    # Activation gate check
    if len(diagnostics_history) < 15:
        raise ValueError(
            f"Activation gate not met: {len(diagnostics_history)}/15 draws. "
            "Strategy Advisor requires ≥15 draws in diagnostics_history/."
        )
    
    # Build Tier 1: Inputs summary
    inputs_summary = {
        'draws_analyzed': len(diagnostics_history),
        'episodes_analyzed': len(telemetry_history),
        'policies_reviewed': len(policy_history),
        'current_hit_at_20': pool_metrics.get('hit_at_20', 0),
        'current_hit_at_100': pool_metrics.get('hit_at_100', 0),
        'current_hit_at_300': pool_metrics.get('hit_at_300', 0),
        'confidence_calibration': pool_metrics.get('confidence_calibration', 0),
        'pool_concentration': pool_metrics.get('concentration_ratio', 0),
        'pool_stability': pool_metrics.get('stability_jaccard', 0),
        'recent_diagnostics': diagnostics_history[-5:],  # Last 5 for context
        'recent_telemetry': telemetry_history[-5:],  # Last 5 episodes
        'active_config': current_config,
    }
    
    # Build Tier 1: Evaluation summary (trends)
    hit_20_trend = [d.get('hit_at_20', 0) for d in diagnostics_history[-10:]]
    hit_100_trend = [d.get('hit_at_100', 0) for d in diagnostics_history[-10:]]
    
    evaluation_summary = {
        'hit_20_trend': hit_20_trend,
        'hit_20_improving': hit_20_trend[-1] > hit_20_trend[0] if len(hit_20_trend) > 1 else None,
        'hit_100_trend': hit_100_trend,
        'hit_100_improving': hit_100_trend[-1] > hit_100_trend[0] if len(hit_100_trend) > 1 else None,
        'promoted_policies': [p for p in policy_history if p.get('status') == 'promoted'],
        'rejected_policies': [p for p in policy_history if p.get('status') == 'rejected'],
    }
    
    # Build Tier 2: Recent outcomes
    recent_outcomes = [
        OutcomeRecord(
            step=13,
            run_id=d.get('draw_id', f"draw_{i}"),
            result='hit' if d.get('hit_at_100', 0) > 0 else 'miss',
            metric_delta=d.get('hit_at_100', 0) - diagnostics_history[i-1].get('hit_at_100', 0) if i > 0 else 0,
            key_metric='hit_at_100',
            timestamp=d.get('timestamp', ''),
        )
        for i, d in enumerate(diagnostics_history[-10:])
    ]
    
    # Assemble bundle
    bundle_context = BundleContext(
        mission=STRATEGY_ADVISOR_MISSION,
        schema_excerpt=STRATEGY_ADVISOR_SCHEMA_EXCERPT,
        grammar_name=STRATEGY_ADVISOR_GRAMMAR,
        contracts=['CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md'],
        guardrails=list(STRATEGY_ADVISOR_GUARDRAILS),
        inputs_summary=inputs_summary,
        evaluation_summary=evaluation_summary,
        recent_outcomes=recent_outcomes,
    )
    
    bundle = StepAwarenessBundle(
        step_id=101,  # Strategy Advisor
        step_name="strategy_advisor",
        run_id=run_id,
        is_chapter_13=False,
        context=bundle_context,
        budgets=budgets,
        provenance=[],  # No file provenance — data passed directly
    )
    
    return bundle
```

#### 3.3 Create `parameter_advisor.py` (~400 lines)

Core module that:
- Loads diagnostics_history/, telemetry/, policy_history/
- Computes mathematical metrics (PCS, CC, FPD, MDI, SCS per contract Section 6)
- Calls `build_advisor_bundle()`
- Invokes LLM via router with `strategy_advisor.gbnf`
- Validates proposals against `watcher_policies.json` bounds
- Writes `strategy_recommendation.json`
- Archives to `strategy_history/`

#### 3.4 Wire into WATCHER

Add to `watcher_dispatch.py`:
```python
def dispatch_strategy_advisor(self) -> Optional[Dict]:
    """
    Run Strategy Advisor analysis if activation gate is met.
    Returns strategy_recommendation or None if gate not met.
    """
    # Check activation gate
    diagnostics_count = len(list(Path('diagnostics_history').glob('*.json')))
    if diagnostics_count < 15:
        logger.info("Strategy Advisor gate not met: %d/15 draws", diagnostics_count)
        return None
    
    # Run advisor
    from parameter_advisor import StrategyAdvisor
    advisor = StrategyAdvisor(project_root=self.project_root)
    recommendation = advisor.analyze()
    
    # Validate and apply
    if self._validate_advisor_recommendation(recommendation):
        self._apply_selfplay_overrides(recommendation.get('selfplay_overrides', {}))
    
    return recommendation
```

---

## 4. File Deliverables

### From This Session (Ready Now)

| File | Purpose | Deploy To |
|------|---------|-----------|
| `bundle_factory.py` | v1.1.0 with MAIN_MISSION + selfplay eval | `agents/contexts/` |
| `SPEC_BUNDLE_FACTORY_v1_1_0.md` | Formal specification | `docs/` |
| `bundle_factory_v1.1.0_patch.py` | Reference documentation | `docs/` |
| `PROGRESS_BUNDLE_FACTORY_AND_STRATEGY_ADVISOR_v1_0.md` | This document | `docs/` |

### To Be Created (Phase 3)

| File | Purpose | Deploy To |
|------|---------|-----------|
| `strategy_advisor.gbnf` | Grammar constraint | `agent_grammars/` |
| `parameter_advisor.py` | Main advisor module | Project root |
| Strategy Advisor additions to `bundle_factory.py` | `build_advisor_bundle()` | `agents/contexts/` |
| WATCHER dispatch additions | `dispatch_strategy_advisor()` | `agents/watcher_dispatch.py` |

---

## 5. Verification Checklist

### After Phase 1 (Soak C)
- [ ] Daemon ran for 2+ hours without crash
- [ ] Chapter 13 → WATCHER → Selfplay flow completed at least once
- [ ] Audit logs in `watcher_requests/` show proper state transitions
- [ ] No orphaned processes or GPU memory leaks

### After Phase 2 (bundle_factory v1.1.0)
- [ ] Self-test passes: `PYTHONPATH=. python3 agents/contexts/bundle_factory.py`
- [ ] MAIN_MISSION appears in rendered prompts
- [ ] step_id=99 (selfplay evaluation) works
- [ ] Existing Steps 1-6 and Chapter 13 bundles unaffected
- [ ] Git commit pushed

### After Phase 3 (Strategy Advisor)
- [ ] `strategy_advisor.gbnf` syntax validates
- [ ] `build_advisor_bundle()` self-test passes
- [ ] `parameter_advisor.py` can load sample data and produce recommendation
- [ ] Activation gate correctly blocks with <15 draws
- [ ] WATCHER dispatch integration works
- [ ] Git commit pushed

---

## 6. Timeline

| Phase | Duration | Blocker |
|-------|----------|---------|
| Phase 1: Soak C | 2-4 hours | None — user executing now |
| Phase 2: Deploy v1.1.0 | 15 minutes | Soak C pass |
| Phase 3: Strategy Advisor | 2-3 hours coding | Phase 2 complete |
| Activation Gate | Days/weeks | ≥15 real draws accumulated |

**Note:** Strategy Advisor CODE will be complete after Phase 3. The activation gate (≥15 draws) controls when it RUNS, not when it's implemented.

---

## 7. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-05 | Initial progress document |

---

**END OF PROGRESS DOCUMENT**

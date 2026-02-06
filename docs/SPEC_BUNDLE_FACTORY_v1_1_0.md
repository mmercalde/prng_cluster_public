# SPECIFICATION: Bundle Factory v1.1.0

**Document:** SPEC_BUNDLE_FACTORY_v1_1_0.md  
**Version:** 1.0.0  
**Date:** 2026-02-05  
**Status:** Ready for Deployment  
**Authority:** Team Alpha recommendation  
**Approval Required:** Team Beta  

---

## 1. Purpose

This specification documents the v1.1.0 update to `agents/contexts/bundle_factory.py`, which adds:

1. **MAIN_MISSION** — Global context injected into every LLM prompt
2. **Selfplay Evaluation Context** — Mission, schema, and guardrails for evaluating selfplay outcomes (step_id=99)

These additions close the Chapter 10 §10 Known Gap ("MAIN_MISSION in doctrine.py — No global mission context") and provide LLM context for selfplay outcome evaluation.

---

## 2. Changes Summary

| Change | Type | Lines | Impact |
|--------|------|-------|--------|
| `MAIN_MISSION` constant | Addition | +12 | Global "functional mimicry" context for all LLM calls |
| `SELFPLAY_EVALUATION_MISSION` | Addition | +14 | Selfplay outcome evaluation context |
| `SELFPLAY_SCHEMA_EXCERPT` | Addition | +8 | Describes `agent_decision.gbnf` output for selfplay |
| `SELFPLAY_GRAMMAR` | Addition | +1 | Points to existing `agent_decision.gbnf` |
| `SELFPLAY_GUARDRAILS` | Addition | +7 | Authority boundaries for selfplay evaluation |
| `step_id=99` in `step_names` | Addition | +1 | Route selfplay evaluations |
| `elif step_id == 99` branch | Addition | +5 | Wire selfplay constants into Tier 0 |
| `=== SYSTEM MISSION ===` prepend | Addition | +2 | MAIN_MISSION appears first in all prompts |
| `bundle_version` bump | Modification | 1 | `"1.0.0"` → `"1.1.0"` |
| Selfplay self-test | Addition | +18 | Verification in self-test |
| MAIN_MISSION assertion | Addition | +3 | Verify injection works |

**Total:** ~93 lines added, 1 line modified, 0 lines removed.

---

## 3. MAIN_MISSION Specification

### 3.1 Constant Definition

```python
MAIN_MISSION = (
    "You are an evaluator within a distributed PRNG analysis system that uses "
    "functional mimicry — learning surface-level output patterns and statistical "
    "heuristics from PRNG-generated sequences to predict future draws, rather than "
    "attempting to discover or reconstruct actual seeds. The system operates a "
    "6-step pipeline (window optimization → sieve → scoring → ML architecture → "
    "anti-overfit training → prediction generation) across a 26-GPU cluster. "
    "Success is measured by hit rate and confidence calibration on held-out draws. "
    "All evaluations serve this goal: improving the system's ability to learn "
    "exploitable patterns in PRNG output through iterative ML refinement, "
    "selfplay exploration, and autonomous feedback loops."
)
```

### 3.2 Injection Point

In `render_prompt_from_bundle()`, MAIN_MISSION is prepended as the first Tier 0 section:

```python
tier0_parts.append(f"=== SYSTEM MISSION ===\n{MAIN_MISSION}")
tier0_parts.append(f"=== STEP MISSION ===\n{ctx.mission}")
```

### 3.3 Token Budget Impact

| Metric | Value |
|--------|-------|
| MAIN_MISSION word count | ~90 words |
| Estimated tokens | ~117 tokens |
| Section header overhead | ~5 tokens |
| **Total Tier 0 increase** | **~122 tokens** |
| Current `tier0_reserved_tokens` | 1200 |
| Typical Tier 0 usage before | 400-600 tokens |
| Typical Tier 0 usage after | 520-720 tokens |
| Remaining headroom | 480-680 tokens |

**Verdict:** No budget adjustment required.

---

## 4. Selfplay Evaluation Context Specification

### 4.1 Mission

```python
SELFPLAY_EVALUATION_MISSION = (
    "Selfplay Evaluation: Assess the quality and strategic value of selfplay "
    "episode outcomes. Selfplay generates HYPOTHETICAL policy candidates by "
    "exploring parameter variations (feature weights, pool sizing, model "
    "architecture choices) against historical draw data ONLY — it never sees "
    "live ground truth. Evaluate candidates on: fitness score trajectory "
    "(improving across episodes indicates productive exploration), diversity "
    "of parameter space explored (avoid mode collapse into narrow optima), "
    "alignment with the current pipeline's bottleneck (does the candidate "
    "address the weakest link?), and feasibility of integration (can the "
    "proposed parameters be adopted without destabilizing existing models). "
    "Your evaluation informs Chapter 13's promotion decision — you do NOT "
    "promote candidates yourself."
)
```

### 4.2 Schema Excerpt (Aligned with agent_decision.gbnf)

```python
SELFPLAY_SCHEMA_EXCERPT = (
    "AgentDecision (shared with Steps 1-6): key_fields=[decision, confidence, reasoning]. "
    "decision enum: proceed (candidate worth promoting to Chapter 13), "
    "retry (selfplay should re-explore with adjusted parameters), "
    "escalate (results anomalous or outside expected bounds — flag for human review). "
    "confidence: 0.0-1.0 in your evaluation. "
    "reasoning: Include assessment of candidate quality (strong/moderate/weak/reject), "
    "exploration breadth (mode collapse risk), fitness trajectory (improving/plateau/declining), "
    "bottleneck alignment, and integration risk. These inform Chapter 13's promotion decision."
)
```

### 4.3 Grammar

```python
SELFPLAY_GRAMMAR = "agent_decision.gbnf"  # Reuse existing decision grammar
```

**Rationale:** No new grammar required. Selfplay evaluation uses the same `{decision, confidence, reasoning}` output structure as Steps 1-6. Evaluation criteria (quality, breadth, trajectory, etc.) are instructed to appear in the `reasoning` string field.

### 4.4 Guardrails

```python
SELFPLAY_GUARDRAILS = [
    "Selfplay candidates are HYPOTHESES — you evaluate, Chapter 13 promotes.",
    "Selfplay uses historical data ONLY — never assume it validated against live draws.",
    "Mode collapse (all episodes converging to same parameters) is a critical warning.",
    "A candidate that improves one model type but degrades others needs careful weighting.",
    "Exploration breadth matters: a moderate-fitness diverse search beats a high-fitness narrow one.",
    "Do not penalize candidates for low absolute fitness — trajectory matters more than level.",
    "WATCHER executes; Chapter 13 decides; selfplay explores. You evaluate.",
]
```

### 4.5 Routing

```python
# In step_names dict:
99: "selfplay_evaluation"

# In Tier 0 resolution:
elif step_id == 99:  # Selfplay evaluation
    mission = SELFPLAY_EVALUATION_MISSION
    schema_excerpt = SELFPLAY_SCHEMA_EXCERPT
    grammar_name = SELFPLAY_GRAMMAR
    guardrails = list(SELFPLAY_GUARDRAILS)
```

### 4.6 Dispatch Usage

```python
# In watcher_dispatch.py or chapter_13_*.py:
prompt, grammar, bundle = build_llm_context(
    step_id=99,
    results={
        "episodes_completed": episode_count,
        "candidates_emitted": len(candidates),
        "best_fitness": best_fitness_score,
        "exploration_breadth": diversity_metric,
        "mode_collapse_detected": collapse_flag,
    },
)
llm_output = call_llm(prompt, grammar=grammar)
```

---

## 5. Alignment Verification

### 5.1 Grammar Alignment

| Component | Grammar File | Status |
|-----------|--------------|--------|
| Steps 1, 3, 4, 5, 6 | `agent_decision.gbnf` | ✅ Unchanged |
| Step 2 | `sieve_analysis.gbnf` | ✅ Unchanged |
| Chapter 13 | `chapter_13.gbnf` | ✅ Unchanged |
| Selfplay Evaluation (NEW) | `agent_decision.gbnf` | ✅ Reuses existing |

**No new grammar files required.**

### 5.2 Pydantic Alignment

| Component | Pydantic Model | Status |
|-----------|----------------|--------|
| Steps 1-6 evaluation | WATCHER JSON parser | ✅ Unchanged |
| Chapter 13 proposals | `LLMProposal` | ✅ Unchanged |
| Selfplay Evaluation (NEW) | WATCHER JSON parser | ✅ Same as Steps 1-6 |

**No new Pydantic models required.**

### 5.3 Agent Manifest Alignment

| Component | Manifest Required | Status |
|-----------|-------------------|--------|
| Steps 1-6 | Yes (`agent_manifests/*.json`) | ✅ Unchanged |
| Chapter 13 | No (uses `watcher_policies.json`) | ✅ Unchanged |
| Selfplay Evaluation (NEW) | No (meta-evaluation, not a step) | ✅ N/A |

**No new agent manifests required.**

### 5.4 Authority Contract Alignment

Per `CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md`:

| Invariant | v1.1.0 Compliance |
|-----------|-------------------|
| Selfplay cannot see live outcomes | ✅ Guardrail explicitly states "historical data ONLY" |
| Only Chapter 13 promotes | ✅ Guardrail states "you evaluate, Chapter 13 promotes" |
| WATCHER mediates execution | ✅ Guardrail states "WATCHER executes; Chapter 13 decides; selfplay explores" |
| Selfplay outputs are hypotheses | ✅ Mission and guardrails reinforce this |

### 5.5 Chapter 14 Future Compatibility

| Chapter 14 Element | v1.1.0 Collision | Notes |
|--------------------|------------------|-------|
| `step_id=14` | ✅ None | v1.1.0 uses `step_id=99` |
| `DIAGNOSTICS_MISSION` | ✅ None | Different constant name |
| `DIAGNOSTICS_GUARDRAILS` | ✅ None | Different constant name |
| `build_diagnostics_bundle()` | ✅ None | Separate function, not step routing |
| `diagnostics_analysis.gbnf` | ✅ None | Chapter 14 will add its own |

**Chapter 14 implementation will add alongside v1.1.0 changes without conflict.**

---

## 6. Files Changed

| File | Change Type | Location |
|------|-------------|----------|
| `agents/contexts/bundle_factory.py` | Modified | Zeus: `~/distributed_prng_analysis/` |

**No other files require changes.**

---

## 7. Deployment Instructions

### 7.1 Copy File to Zeus

```bash
# On ser8:
scp ~/Downloads/bundle_factory.py zeus:~/distributed_prng_analysis/agents/contexts/bundle_factory.py
```

### 7.2 Run Self-Test

```bash
# On Zeus:
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 agents/contexts/bundle_factory.py
```

### 7.3 Expected Self-Test Output

```
======================================================================
Bundle Factory v1.1.0 — Self-Test
======================================================================

  Step 1 (window_optimizer):
    Mission: Step 1 — Window Optimizer: Find optimal window parameters...
    Grammar: agent_decision.gbnf
    Guardrails: 2
    Contracts: ['CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md']

  [... Steps 2-6 ...]

  Chapter 13 (chapter_13_feedback):
    Mission: Chapter 13 — Live Feedback Loop: Monitors for new draws...
    Grammar: chapter_13.gbnf
    Guardrails: 4

  Selfplay Evaluation (selfplay_evaluation):
    Mission: Selfplay Evaluation: Assess the quality and strategic value...
    Grammar: agent_decision.gbnf
    Guardrails: 7
    MAIN_MISSION injected: OK

  [... remaining tests ...]

======================================================================
Self-test complete
======================================================================
```

### 7.4 Verify MAIN_MISSION in Rendered Prompt

```bash
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 -c "
from agents.contexts.bundle_factory import build_llm_context

prompt, grammar, bundle = build_llm_context(step_id=3, results={'completion_rate': 0.99})
print('=== FIRST 500 CHARS ===')
print(prompt[:500])
print('...')
print('=== MAIN_MISSION CHECK ===')
assert '=== SYSTEM MISSION ===' in prompt
assert 'functional mimicry' in prompt
print('MAIN_MISSION present: OK')
"
```

### 7.5 Git Commit

```bash
cd ~/distributed_prng_analysis
git add agents/contexts/bundle_factory.py
git commit -m "bundle_factory v1.1.0: Add MAIN_MISSION + selfplay evaluation context

Closes Chapter 10 §10 Known Gap (MAIN_MISSION).
Adds step_id=99 for selfplay outcome evaluation.
No new grammars, Pydantic models, or manifests.
Reuses agent_decision.gbnf for selfplay.
All 8 bundle types pass self-test."
git push
```

---

## 8. Rollback Instructions

If issues are discovered:

```bash
cd ~/distributed_prng_analysis
git checkout HEAD~1 -- agents/contexts/bundle_factory.py
```

Or restore from the v1.0.0 backup in project knowledge.

---

## 9. Known Limitations

1. **Selfplay evaluation not yet wired in dispatch code** — `dispatch_selfplay()` does not currently call `build_llm_context(step_id=99)` after selfplay completes. This is a future integration point, not a v1.1.0 blocker.

2. **MAIN_MISSION is static** — The global mission text is a constant. If the project's goals evolve significantly, this text would need manual update.

3. **No new grammar for rich selfplay evaluation** — By design, selfplay evaluation reuses `agent_decision.gbnf`. If richer structured output is needed later (e.g., separate fields for exploration_breadth, fitness_trajectory), a dedicated `selfplay_evaluation.gbnf` would need to be created.

---

## 10. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-05 | Initial specification |

---

## 11. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | Claude (Session 60) | 2026-02-05 | — |
| Team Alpha | | | |
| Team Beta | | | |

---

**END OF SPECIFICATION**

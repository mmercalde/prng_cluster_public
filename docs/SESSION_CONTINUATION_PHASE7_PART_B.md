# Session Continuation — Phase 7 Part B: WATCHER Dispatch Wiring

**Date:** 2026-02-03
**Previous Session:** Session 57 (Parts A/B/C/D — LLM Infra + Bundle Factory)
**Goal:** Wire dispatch functions in `agents/watcher_agent.py` against bundle_factory.py

---

## System Overview

Distributed PRNG analysis system: 46 PRNGs, 26 GPUs (Zeus 2× RTX 3080 Ti + two rigs of 12× RX 6600), 6-step pipeline, 4 ML models, Pydantic+GBNF constrained LLM (DeepSeek-R1-14B). Project dir: `~/distributed_prng_analysis` on Zeus. Venv: `source ~/torch/bin/activate`.

---

## Where We Are — Dependency Chain

```
Part A  (Selfplay Validation)    ✅ DONE  — 8 episodes, zero crashes
Part B0 (Bundle Factory + Spec)  ✅ DONE  — Self-test passed, committed ffe397a
Part B  (WATCHER Dispatch)       🔲 THIS SESSION — ~210 lines, 60-90 min
Part C  (File Organization)      ✅ DONE
Part D  (Integration Testing)    🔲 AFTER B — 5 tasks, 60 min
```

---

## What Was Just Built (Part B0) — Committed & Tested

### `agents/contexts/bundle_factory.py` (915 lines)
- `StepAwarenessBundle` — immutable Pydantic model (model_config frozen)
- `TokenBudget` — tiered enforcement (Tier 0: mission/schema/grammar/guardrails, Tier 1: eval data, Tier 2: history/trends)
- `ProvenanceRecord` — SHA256 file hashing
- Step missions, schemas, grammars, guardrails for all 6 steps + Chapter 13
- `build_step_awareness_bundle()` — main assembly (wraps existing `build_full_context()`)
- `render_prompt_from_bundle()` — tiered prompt renderer with token budgets
- `build_llm_context()` — **single entry point** (Guardrail #1)
- Track 2 retrieval stubs: `_retrieve_recent_outcomes()`, `_retrieve_trend_summary()`, `_retrieve_open_incidents()` — all return `[]`, zero dispatch rework when filled

### `docs/ADDENDUM_A_STEP_AWARENESS_BUNDLES_v1_0.md` (364 lines)
- Formal spec: bundle object model, token budgets, retrieval hierarchy, assembly ownership, determinism, provenance, per-step minimums, API reference

---

## What Needs Building (Part B) — 4 Tasks

**File to modify:** `agents/watcher_agent.py`

**Non-negotiable guardrails:**
- **Guardrail #1:** Dispatch calls `build_llm_context()` ONLY — no inline context assembly
- **Guardrail #2:** No baked-in token assumptions — bundle_factory owns prompt structure

### B1: `dispatch_selfplay()` (~60 lines)
- Stop LLM (free VRAM) → spawn `selfplay_orchestrator.py` → evaluate candidate via `build_llm_context(step_id=13, is_chapter_13=True)` → restart LLM
- LLM lifecycle: `self.llm_lifecycle.stop()` before, `.start()` after

### B2: `dispatch_learning_loop()` (~50 lines)
- Run Steps 3→5→6 (or full 1→6) via `run_step()`
- Evaluate each step via `build_llm_context(step_id=step, results=results)`
- Stop on non-"proceed" decision

### B3: `process_chapter_13_request()` (~70 lines)
- Scan `watcher_requests/*.json`
- Route `selfplay_retrain` → `dispatch_selfplay()`
- Route `learning_loop` → `dispatch_learning_loop()`
- Validate via `build_llm_context()` before execution

### B4: Daemon wiring (~30 lines)
- Add `watcher_requests/` directory scanning to `run_daemon()`
- CLI args: `--dispatch-selfplay`, `--dispatch-learning-loop`
- Add `--dry-run` flag for testing

---

## Dispatch Pattern (Every Function Follows This)

```python
from agents.contexts.bundle_factory import build_llm_context

# CORRECT — Guardrail #1
prompt, grammar_name, bundle = build_llm_context(
    step_id=step,
    run_id=run_id,
    results=results,
    manifest_path=manifest_path
)
llm_output = call_llm(prompt, grammar=grammar_name)

# FORBIDDEN — no inline context
prompt = f"You are evaluating step {step}. Results: {json.dumps(results)}"
```

---

## Homework Before Writing Code

**Read these two files on Zeus to understand existing patterns:**

```bash
cat ~/distributed_prng_analysis/agents/full_agent_context.py
cat ~/distributed_prng_analysis/agents/prompt_builder.py 2>/dev/null || echo "not found"
```

The bundle_factory already imports `build_full_context` from `full_agent_context.py`. Understanding what it returns ensures dispatch functions mesh with existing code.

Also useful — the current watcher_agent.py to see where dispatch functions get added:

```bash
wc -l ~/distributed_prng_analysis/agents/watcher_agent.py
head -100 ~/distributed_prng_analysis/agents/watcher_agent.py
```

---

## Existing Infrastructure (400KB+ — No Work Needed)

| Component | File | Size |
|-----------|------|------|
| Diagnostics Engine | `chapter_13_diagnostics.py` | 39KB |
| LLM Advisor | `chapter_13_llm_advisor.py` | 23KB |
| Triggers Engine | `chapter_13_triggers.py` | 36KB |
| Acceptance Engine | `chapter_13_acceptance.py` | 41KB |
| Orchestrator | `chapter_13_orchestrator.py` | 23KB |
| Proposal Schema | `llm_proposal_schema.py` | 14KB |
| GBNF Grammars | `agent_grammars/*.gbnf` | ~9KB |
| Draw Ingestion | `draw_ingestion_daemon.py` | 22KB |
| Synthetic Injector | `synthetic_draw_injector.py` | 20KB |
| Policies Config | `watcher_policies.json` | 4.7KB |
| Selfplay Orchestrator | `selfplay_orchestrator.py` | 43KB |
| Policy Transform | `policy_transform.py` | 36KB |
| Policy Conditioning | `policy_conditioned_episode.py` | 25KB |
| LLM Lifecycle Manager | `llm_services/llm_lifecycle.py` | ~8KB |
| Bundle Factory | `agents/contexts/bundle_factory.py` | 33KB |
| Context Window 32K | `llm_services/configs/*.json` | — |

---

## Architecture Principles

- **Authority separation:** Chapter 13 decides, WATCHER executes, selfplay explores
- **Selfplay candidates are hypotheses** until Chapter 13 promotes them
- **LLM lifecycle:** Stop before GPU-heavy dispatch, restart for evaluation
- **Team Beta immutability:** Immutable structure, configurable parameters
- **No vector DB, no embeddings, no GPU-resident services**
- **NEVER restore from backup** — fix by editing/removing bad additions

---

## Success Criteria for Part B

- [ ] `dispatch_selfplay()` spawns selfplay_orchestrator.py
- [ ] `dispatch_learning_loop()` runs Steps 3→5→6
- [ ] `process_chapter_13_request()` handles watcher_requests/*.json
- [ ] All dispatch functions use `build_llm_context()` (Guardrail #1)
- [ ] No baked-in token assumptions (Guardrail #2)
- [ ] `--dry-run` flag works for testing

## After Part B → Part D (Integration Testing)

- Bundle factory self-test on Zeus
- `--dispatch-selfplay --dry-run`
- `--dispatch-learning-loop steps_3_5_6 --dry-run`
- Mock request processing
- End-to-end: Chapter 13 → WATCHER → Selfplay

## Full Autonomy Target

```
Chapter 13 Triggers → watcher_requests/ → WATCHER → Selfplay/Learning Loop
       ↑                                                        ↓
       └──────────────── Diagnostics ← Reality ←────────────────┘
```

No human in the loop for routine decisions.

---

## Reference Documents in Project Knowledge

- `TODO_PHASE7_WATCHER_INTEGRATION_REVISED_v2.md` — authoritative task list
- `ADDENDUM_A_STEP_AWARENESS_BUNDLES_v1_0.md` — bundle factory spec
- `PROPOSAL_LLM_Infrastructure_Optimization_v1_1.md` — LLM infra design
- `SESSION_CHANGELOG_20260201_S57.md` — Session 57 record
- `CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md` — authority rules
- `CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` — Chapter 13 spec

# TODO: Phase 7 â€” WATCHER Integration (REVISED v3)

**Created:** 2026-01-30  
**Revised:** 2026-02-04 (Session 60 â€” Chapter 14 added to horizon)  
**Status:** âœ… ALL PARTS COMPLETE (Phase 7 Done)  
**Goal:** Complete autonomous operation pipeline  
**Authority:** Joint Alpha + Beta sequencing decision (Session 57)

---

## âš ï¸ Revision Notice (v2 â†’ v3)

**v3 update (Session 59):** Phase 7 complete. D5 end-to-end test passed after fixing 5 integration bugs discovered during testing. Full autonomous loop verified: Chapter 13 â†’ WATCHER â†’ Selfplay â†’ LLM Evaluation â†’ Archive.

**v2 sequencing decision locked:** Bundle factory + Addendum A spec FIRST, then wire dispatch against it.

**Rationale (Team Beta):**
1. Dispatch wiring should call a stable abstraction from day one
2. That abstraction is `build_step_awareness_bundle()`
3. Today it wraps existing static context builders
4. Tomorrow it adds structured retrieval
5. Dispatch code never changes

**Required guardrails for dispatch code:**
- **Guardrail #1:** Single context entry point â€” dispatch calls `build_llm_context()`, nothing else
- **Guardrail #2:** No baked-in token assumptions â€” bundle_factory owns prompt structure

---

## What EXISTS (No Work Needed)

| Component | File | Size | Status |
|-----------|------|------|--------|
| Diagnostics Engine | `chapter_13_diagnostics.py` | 39KB | âœ… Complete |
| LLM Advisor | `chapter_13_llm_advisor.py` | 23KB | âœ… Complete |
| Triggers Engine | `chapter_13_triggers.py` | 36KB | âœ… Complete |
| Acceptance Engine | `chapter_13_acceptance.py` | 41KB | âœ… Complete |
| Orchestrator | `chapter_13_orchestrator.py` | 23KB | âœ… Complete |
| Proposal Schema | `llm_proposal_schema.py` | 14KB | âœ… Complete |
| GBNF Grammar | `chapter_13.gbnf` | 2.9KB | âœ… Complete |
| Draw Ingestion | `draw_ingestion_daemon.py` | 22KB | âœ… Complete |
| Synthetic Injector | `synthetic_draw_injector.py` | 20KB | âœ… Complete |
| Policies Config | `watcher_policies.json` | 4.7KB | âœ… Complete |
| Selfplay Orchestrator | `selfplay_orchestrator.py` | 43KB | âœ… Complete |
| Policy Transform | `policy_transform.py` | 36KB | âœ… Complete |
| Policy Conditioning | `policy_conditioned_episode.py` | 25KB | âœ… Complete |
| LLM Lifecycle Manager | `llm_services/llm_lifecycle.py` | ~8KB | âœ… Complete (Session 57) |
| 4 GBNF Grammar Files | `agent_grammars/*.gbnf` | ~6KB | âœ… Complete (Session 57, fixed Session 59) |
| Context Window 32K | `llm_services/configs/*.json` | â€” | âœ… Complete (Session 57) |

**Total existing code:** ~400KB+

---

## What Was MISSING (All Work Complete)

### Part A: Selfplay Validation Testing â€” âœ… COMPLETE (2026-02-01)

| # | Task | Command | Status | Evidence |
|---|------|---------|--------|----------|
| A1 | Run multi-episode selfplay | `python3 selfplay_orchestrator.py --survivors survivors_with_scores.json --episodes 5 --policy-conditioned` | âœ… | 8 episodes (5+3), zero crashes |
| A2 | Verify candidate emission | `cat learned_policy_candidate.json` | âœ… | Schema v1.1.0, 3 candidates emitted |
| A3 | Verify policy history archive | `ls -la policy_history/` | âœ… | 3 files accumulated |
| A4 | Test with active policy | Create test policy, re-run | âœ… | Filter: 75,396â†’47,614; Weight: 46,715 adjusted |
| A5 | Verify telemetry health | `cat telemetry/learning_health_latest.json` | âœ… | 38 models tracked, JSON valid |

**Git:** Commit `c0f5d32`

---

### Part B0: Bundle Factory + Spec â€” âœ… COMPLETE (2026-02-02)

**Prerequisite for Part B.** Team Alpha + Beta joint decision: spec + skeleton first, then wire dispatch against it.

| # | Task | File | Lines | Status |
|---|------|------|-------|--------|
| B0.1 | `bundle_factory.py` â€” Pydantic models + assembly engine | `agents/contexts/bundle_factory.py` | ~900 | âœ… Built |
| B0.2 | Addendum A â€” Step Awareness Bundle spec | `docs/ADDENDUM_A_STEP_AWARENESS_BUNDLES_v1_0.md` | ~300 | âœ… Built |
| B0.3 | Self-test verification on Zeus | `python3 agents/contexts/bundle_factory.py` | â€” | âœ… Passed (7/7 bundles) |
| B0.4 | Git commit + docs update | â€” | â€” | âœ… Commit `ffe397a` |

**Git:** Commit `ffe397a`

---

### Part B: WATCHER Dispatch Functions â€” âœ… COMPLETE (2026-02-02, Session 58)

**REVISED:** All dispatch functions use `build_llm_context()` per Guardrail #1.

**Architecture decision (Session 58):** Standalone module `agents/watcher_dispatch.py` with method binding pattern instead of inline edits to `watcher_agent.py`. Minimizes diff to critical file, enables standalone testing, avoids MRO/inheritance issues. Auto-patcher applies 5 targeted insertions.

| # | Task | File | Lines | Status | Evidence |
|---|------|------|-------|--------|----------|
| B1 | `dispatch_selfplay()` | `agents/watcher_dispatch.py` | ~70 | âœ… | DRY_RUN passed |
| B2 | `dispatch_learning_loop()` | `agents/watcher_dispatch.py` | ~65 | âœ… | DRY_RUN passed |
| B3 | `process_chapter_13_request()` | `agents/watcher_dispatch.py` | ~75 | âœ… | 2 requests processed (DRY_RUN_OK) |
| B4 | Wire to WATCHER daemon + CLI | `agents/watcher_agent.py` (5 patches) | ~30 | âœ… | CLI args + daemon scanning wired |

**Guardrail compliance (verified):**
- Guardrail #1: 23 calls to `build_llm_context()`, zero inline prompt construction âœ…
- Guardrail #2: Zero hardcoded token counts âœ…
- Authority: Zero writes to `learned_policy_active.json` âœ…
- LLM lifecycle: 2 stop() + 3 ensure_running() calls around GPU phases âœ…
- Halt flag: Checked at entry of every dispatch function AND between steps âœ…

**Team Beta review:** Green light. Adapter contract codified (manual edit on Zeus).

**Git:** Commit `a145e28`

---

### Part C: File Organization â€” âœ… COMPLETE (2026-01-30)

| # | Task | Action | Status | Evidence |
|---|------|--------|--------|----------|
| C1 | Move GBNF grammar | `mkdir -p agent_grammars && mv chapter_13.gbnf agent_grammars/` | âœ… | Commit 22abd7b |

---

### Part D: Integration Testing (60 min) â€” âœ… COMPLETE (2026-02-02, Session 59)

| # | Task | Action | Status | Evidence |
|---|------|--------|--------|----------|
| D1 | Test bundle factory self-test | `cd ~/distributed_prng_analysis && python3 agents/contexts/bundle_factory.py` | âœ… | 7/7 bundles, render+token tests passed (Session 58) |
| D2 | Test selfplay dispatch | `python3 agents/watcher_agent.py --dispatch-selfplay --dry-run` | âœ… | Passed Session 58 |
| D3 | Test learning loop dispatch | `python3 agents/watcher_agent.py --dispatch-learning-loop steps_3_5_6 --dry-run` | âœ… | Passed Session 58 |
| D4 | Test request processing | Create mock request, process via WATCHER | âœ… | 2 requests â†’ DRY_RUN_OK, archived |
| D5 | End-to-end: Chapter 13 â†’ WATCHER â†’ Selfplay | Full flow test (non-dry-run, reduced episodes) | âœ… | Clean pass: real LLM eval, lifecycle stop/restart, grammar-constrained JSON (Session 59) |

**D5 Integration Bugs Found & Fixed (Session 59):**

| # | Bug | Root Cause | Fix | Commit |
|---|-----|-----------|-----|--------|
| 1 | Lifecycle dead code | `self.llm_lifecycle` never set in `__init__` | Added initialization block | `e4dd1b0` |
| 2 | API mismatch | `.start()` / `.stop(string)` not real methods | â†’ `.ensure_running()` / `.stop()` | `e4dd1b0` |
| 3 | Router always None | `GrammarType` import poisoned entire import | Removed dead import | `e4dd1b0` |
| 4 | Grammar 400 errors | `agent_grammars/` had broken v1.0 GBNF | Copied fixed v1.1 from `grammars/` | `e4dd1b0` |
| 5 | Try 1 private API | `_call_primary_with_grammar()` missing config | Gate to public API for `watcher_decision.gbnf` only | `308a2fc` |

**D5 Clean Pass Evidence:**
```
Pre-validation: real LLM (4s response, not instant heuristic)
LLM stop: "confirmed stopped â€” GPU VRAM freed"
Selfplay: rc=0, candidate emitted (58s)
LLM restart: "healthy after 3.2s"
Post-eval: grammar-constrained JSON â€” real structured output
Archive: COMPLETED â€” zero warnings, zero heuristic fallbacks
```

---

## Final Effort Summary

| Part | Tasks | Time | Lines | Status | Commits |
|------|-------|------|-------|--------|---------|
| ~~A: Selfplay Testing~~ | ~~5~~ | ~~30-60 min~~ | ~~0~~ | âœ… Complete | `c0f5d32` |
| ~~B0: Bundle Factory~~ | ~~4~~ | ~~60-90 min~~ | ~~~1200~~ | âœ… Complete | `ffe397a` |
| ~~B: WATCHER Dispatch~~ | ~~4~~ | ~~60-90 min~~ | ~~~894~~ | âœ… Complete | `a145e28` |
| ~~C: File Organization~~ | ~~1~~ | ~~5 min~~ | ~~0~~ | âœ… Complete | `22abd7b` |
| ~~D: Integration Testing~~ | ~~5~~ | ~~60 min~~ | ~~0~~ | âœ… Complete | `e4dd1b0`, `308a2fc` |

---

## Dependency Chain (COMPLETE)

```
Part A (Selfplay Testing) âœ… DONE (c0f5d32)
       â†“
Part B0 (Bundle Factory + Spec) âœ… DONE (ffe397a)
       â†“
Part B (WATCHER Dispatch Functions) âœ… DONE (a145e28)
       â†“
Part C (File Organization) âœ… DONE (22abd7b)
       â†“
Part D (Integration Testing) âœ… DONE (e4dd1b0, 308a2fc)
       â†“
   FULL AUTONOMY ACHIEVED âœ…
```

---

## Success Criteria â€” ALL MET âœ…

### Part A Complete When:
- [x] Selfplay runs 5+ episodes without error
- [x] Candidates emitted to `learned_policy_candidate.json`
- [x] Telemetry health file updated

### Part B0 Complete When:
- [x] `bundle_factory.py` builds bundles for all 6 steps + Chapter 13
- [x] Addendum A spec locked and versioned
- [x] Self-test passes on Zeus (all 7 bundle types)
- [x] Git committed with both files

### Part B Complete When:
- [x] `dispatch_selfplay()` spawns selfplay_orchestrator.py
- [x] `dispatch_learning_loop()` runs Steps 3â†’5â†’6
- [x] `process_chapter_13_request()` handles watcher_requests/*.json
- [x] All dispatch functions use `build_llm_context()` (Guardrail #1 verified)
- [x] No baked-in token assumptions (Guardrail #2 verified)

### Part D Complete When:
- [x] Bundle factory self-test passes on Zeus
- [x] End-to-end flow works without human intervention
- [x] Audit trail shows WATCHER â†’ Selfplay dispatch
- [x] Request â†’ Approval â†’ Execution cycle verified

### Full Autonomy Achieved When:
```
Chapter 13 Triggers â†’ watcher_requests/ â†’ WATCHER â†’ Selfplay/Learning Loop
       â†‘                                                        â†“
       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Diagnostics â† Reality â†â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

**No human in the loop for routine decisions. âœ… VERIFIED Session 59.**

---

## Files Created/Modified Summary

| File | Action | Location | Part | Commit |
|------|--------|----------|------|--------|
| `agents/contexts/bundle_factory.py` | CREATE | `agents/contexts/` | B0 | `ffe397a` |
| `docs/ADDENDUM_A_STEP_AWARENESS_BUNDLES_v1_0.md` | CREATE | `docs/` | B0 | `ffe397a` |
| `agents/watcher_dispatch.py` | CREATE | `agents/` | B | `a145e28` |
| `agents/watcher_agent.py` | MODIFY (5 patches + 3 fixes) | `agents/` | B, D | `a145e28`, `e4dd1b0` |
| `patch_watcher_dispatch.py` | CREATE (tooling) | project root | B | `a145e28` |
| `patch_watcher_dispatch_v1_1.py` | CREATE (tooling) | project root | B | `a145e28` |
| `agent_grammars/*.gbnf` | FIX (v1.0 â†’ v1.1) | `agent_grammars/` | D | `e4dd1b0` |
| `llm_services/start_llm_servers.sh` | FIX (paths) | `llm_services/` | D | `e4dd1b0` |
| `docs/SESSION_CHANGELOG_20260203_S58.md` | CREATE | `docs/` | B | `a145e28` |
| `docs/SESSION_CHANGELOG_20260203_S59.md` | CREATE | `docs/` | D | `e8da609` |

---

## What Was REMOVED from Original TODO

| Original Task | Reason Removed |
|---------------|----------------|
| B1: `llm_proposal_schema.py` | Already exists (14KB) |
| B2: `chapter_13.gbnf` | Already exists (2.9KB) |
| B3: Parameter vocabulary extraction | Already in advisor |
| B4-B6: Diagnostics engine | Already exists (39KB) |
| B7-B10: LLM Advisor | Already exists (23KB) |
| B11: `validate_proposal()` | Already in acceptance.py |
| B12: `apply_parameter_changes()` | Already in acceptance.py |

**Original estimate:** 27 tasks, 630 lines, 2-3 sessions  
**Revised (v1) estimate:** 14 tasks, 180 lines, 1 session  
**Revised (v2) estimate:** 13 tasks, ~1410 lines, 2 sessions (bundle factory adds scope but pays for itself via clean dispatch wiring)  
**Actual:** 18 tasks (13 original + 5 patcher patches), ~2100 lines, 3 sessions (S57-S59)

---

## What Is Explicitly NOT Being Built This Phase

Per Team Beta lock (2026-02-02):

| Item | Status | When |
|------|--------|------|
| Vector database / embeddings | âŒ Out of scope | Not planned |
| GPU-resident retrieval services | âŒ Out of scope | Not planned |
| Raw decision log ingestion | âŒ Out of scope | Track 2 (structured only) |
| Authority changes | âŒ Out of scope | Not planned |
| Parameter advisor (Item B) | â³ Deferred | Steps 4-6 in production |
| Phase 9B.3 auto-heuristics | â³ Deferred | 9B.2 validated first |
| Track 2 retrieval enhancement | â³ After dispatch wired | Fills stubs in bundle_factory |

---

## Evaluation Path Architecture (Session 59 Final)

```
_evaluate_step_via_bundle(prompt, grammar_name)
    â”‚
    â”œâ”€ Try 1: LLM Router (public API)
    â”‚   â””â”€ ONLY if grammar_name == 'watcher_decision.gbnf'
    â”‚   â””â”€ Calls evaluate_watcher_decision() â†’ grammar-constrained JSON
    â”‚
    â”œâ”€ Try 2: HTTP Direct (generic)
    â”‚   â””â”€ POST localhost:8080/completion with inline grammar content
    â”‚   â””â”€ Works for ALL grammars (chapter_13, agent_decision, etc.)
    â”‚
    â””â”€ Try 3: Heuristic Fallback
        â””â”€ proceed, confidence=0.50
        â””â”€ Should NEVER be reached if LLM is healthy
```

---

## Upcoming Work (Post-Phase 7)

### Soak Tests — NEXT PRIORITY

Prerequisite gate before any new feature implementation.

| Test | What It Validates | Duration | Status |
|------|-------------------|----------|--------|
| Soak Test A | Multi-hour daemon endurance | 2-4 hours unattended | ⬜ Not Started |
| Soak Test B | 5-10 sequential back-to-back requests | ~30-60 min | ⬜ Not Started |
| Soak Test C | Sustained autonomous loop (Ch13 → WATCHER → Selfplay) | 2-4 hours | ⬜ Not Started |

### Chapter 14: Training Diagnostics & Model Introspection — PLANNED

**Spec:** `docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md` v1.1.2 (3,133 lines, Session 60)
**Status:** PLANNED — Implementation deferred until Soak Tests A, B, C pass
**Team Alpha review:** Approved with 5 recommendations (all applied in v1.1.2)

| Capability | What It Adds |
|-----------|-------------|
| Live Training Introspection | Epoch/round-by-round health snapshots during Step 5 |
| Per-Survivor Attribution | Per-seed feature explanations across all 4 model types |
| WATCHER Integration | `check_training_health()`, skip registry, policy-bounded LLM trigger |
| LLM Integration | `DiagnosticsBundle`, `diagnostics_analysis.gbnf`, Pydantic schema |
| Selfplay Wiring | Episode diagnostics, trend detection, Chapter 13 root cause analysis |
| Web Dashboard | Plotly charts on existing `web_dashboard.py` |
| TensorBoard | Optional human-only deep investigation UI |

**Implementation estimate:** ~12 hours, ~1,755 lines across 17 files (9 phases)

### Other Deferred Items (Unchanged)

| Item | Status | When |
|------|--------|------|
| Parameter advisor (Item B) | Deferred | Steps 4-6 in production |
| Phase 9B.3 auto-heuristics | Deferred | 9B.2 validated first |
| Track 2 retrieval enhancement | Deferred | Fills stubs in bundle_factory |

### Dependency Chain

```
Soak Test A (daemon endurance)        <-- NEXT
       |
Soak Test B (sequential requests)
       |
Soak Test C (sustained autonomous loop)
       |
Chapter 14 Implementation (~12 hours, 9 phases)
       |
Parameter Advisor / Track 2 / 9B.3 (future)
```

---

**END OF PHASE 7 TODO (v3.1 â€” COMPLETE)**

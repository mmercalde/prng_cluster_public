# TODO: Phase 7 — WATCHER Integration (REVISED v3)

**Created:** 2026-01-30  
**Revised:** 2026-02-02 (Session 59 — ALL PARTS COMPLETE)  
**Status:** ✅ ALL PARTS COMPLETE (Phase 7 Done)  
**Goal:** Complete autonomous operation pipeline  
**Authority:** Joint Alpha + Beta sequencing decision (Session 57)

---

## ⚠️ Revision Notice (v2 → v3)

**v3 update (Session 59):** Phase 7 complete. D5 end-to-end test passed after fixing 5 integration bugs discovered during testing. Full autonomous loop verified: Chapter 13 → WATCHER → Selfplay → LLM Evaluation → Archive.

**v2 sequencing decision locked:** Bundle factory + Addendum A spec FIRST, then wire dispatch against it.

**Rationale (Team Beta):**
1. Dispatch wiring should call a stable abstraction from day one
2. That abstraction is `build_step_awareness_bundle()`
3. Today it wraps existing static context builders
4. Tomorrow it adds structured retrieval
5. Dispatch code never changes

**Required guardrails for dispatch code:**
- **Guardrail #1:** Single context entry point — dispatch calls `build_llm_context()`, nothing else
- **Guardrail #2:** No baked-in token assumptions — bundle_factory owns prompt structure

---

## What EXISTS (No Work Needed)

| Component | File | Size | Status |
|-----------|------|------|--------|
| Diagnostics Engine | `chapter_13_diagnostics.py` | 39KB | ✅ Complete |
| LLM Advisor | `chapter_13_llm_advisor.py` | 23KB | ✅ Complete |
| Triggers Engine | `chapter_13_triggers.py` | 36KB | ✅ Complete |
| Acceptance Engine | `chapter_13_acceptance.py` | 41KB | ✅ Complete |
| Orchestrator | `chapter_13_orchestrator.py` | 23KB | ✅ Complete |
| Proposal Schema | `llm_proposal_schema.py` | 14KB | ✅ Complete |
| GBNF Grammar | `chapter_13.gbnf` | 2.9KB | ✅ Complete |
| Draw Ingestion | `draw_ingestion_daemon.py` | 22KB | ✅ Complete |
| Synthetic Injector | `synthetic_draw_injector.py` | 20KB | ✅ Complete |
| Policies Config | `watcher_policies.json` | 4.7KB | ✅ Complete |
| Selfplay Orchestrator | `selfplay_orchestrator.py` | 43KB | ✅ Complete |
| Policy Transform | `policy_transform.py` | 36KB | ✅ Complete |
| Policy Conditioning | `policy_conditioned_episode.py` | 25KB | ✅ Complete |
| LLM Lifecycle Manager | `llm_services/llm_lifecycle.py` | ~8KB | ✅ Complete (Session 57) |
| 4 GBNF Grammar Files | `agent_grammars/*.gbnf` | ~6KB | ✅ Complete (Session 57, fixed Session 59) |
| Context Window 32K | `llm_services/configs/*.json` | — | ✅ Complete (Session 57) |

**Total existing code:** ~400KB+

---

## What Was MISSING (All Work Complete)

### Part A: Selfplay Validation Testing — ✅ COMPLETE (2026-02-01)

| # | Task | Command | Status | Evidence |
|---|------|---------|--------|----------|
| A1 | Run multi-episode selfplay | `python3 selfplay_orchestrator.py --survivors survivors_with_scores.json --episodes 5 --policy-conditioned` | ✅ | 8 episodes (5+3), zero crashes |
| A2 | Verify candidate emission | `cat learned_policy_candidate.json` | ✅ | Schema v1.1.0, 3 candidates emitted |
| A3 | Verify policy history archive | `ls -la policy_history/` | ✅ | 3 files accumulated |
| A4 | Test with active policy | Create test policy, re-run | ✅ | Filter: 75,396→47,614; Weight: 46,715 adjusted |
| A5 | Verify telemetry health | `cat telemetry/learning_health_latest.json` | ✅ | 38 models tracked, JSON valid |

**Git:** Commit `c0f5d32`

---

### Part B0: Bundle Factory + Spec — ✅ COMPLETE (2026-02-02)

**Prerequisite for Part B.** Team Alpha + Beta joint decision: spec + skeleton first, then wire dispatch against it.

| # | Task | File | Lines | Status |
|---|------|------|-------|--------|
| B0.1 | `bundle_factory.py` — Pydantic models + assembly engine | `agents/contexts/bundle_factory.py` | ~900 | ✅ Built |
| B0.2 | Addendum A — Step Awareness Bundle spec | `docs/ADDENDUM_A_STEP_AWARENESS_BUNDLES_v1_0.md` | ~300 | ✅ Built |
| B0.3 | Self-test verification on Zeus | `python3 agents/contexts/bundle_factory.py` | — | ✅ Passed (7/7 bundles) |
| B0.4 | Git commit + docs update | — | — | ✅ Commit `ffe397a` |

**Git:** Commit `ffe397a`

---

### Part B: WATCHER Dispatch Functions — ✅ COMPLETE (2026-02-02, Session 58)

**REVISED:** All dispatch functions use `build_llm_context()` per Guardrail #1.

**Architecture decision (Session 58):** Standalone module `agents/watcher_dispatch.py` with method binding pattern instead of inline edits to `watcher_agent.py`. Minimizes diff to critical file, enables standalone testing, avoids MRO/inheritance issues. Auto-patcher applies 5 targeted insertions.

| # | Task | File | Lines | Status | Evidence |
|---|------|------|-------|--------|----------|
| B1 | `dispatch_selfplay()` | `agents/watcher_dispatch.py` | ~70 | ✅ | DRY_RUN passed |
| B2 | `dispatch_learning_loop()` | `agents/watcher_dispatch.py` | ~65 | ✅ | DRY_RUN passed |
| B3 | `process_chapter_13_request()` | `agents/watcher_dispatch.py` | ~75 | ✅ | 2 requests processed (DRY_RUN_OK) |
| B4 | Wire to WATCHER daemon + CLI | `agents/watcher_agent.py` (5 patches) | ~30 | ✅ | CLI args + daemon scanning wired |

**Guardrail compliance (verified):**
- Guardrail #1: 23 calls to `build_llm_context()`, zero inline prompt construction ✅
- Guardrail #2: Zero hardcoded token counts ✅
- Authority: Zero writes to `learned_policy_active.json` ✅
- LLM lifecycle: 2 stop() + 3 ensure_running() calls around GPU phases ✅
- Halt flag: Checked at entry of every dispatch function AND between steps ✅

**Team Beta review:** Green light. Adapter contract codified (manual edit on Zeus).

**Git:** Commit `a145e28`

---

### Part C: File Organization — ✅ COMPLETE (2026-01-30)

| # | Task | Action | Status | Evidence |
|---|------|--------|--------|----------|
| C1 | Move GBNF grammar | `mkdir -p agent_grammars && mv chapter_13.gbnf agent_grammars/` | ✅ | Commit 22abd7b |

---

### Part D: Integration Testing (60 min) — ✅ COMPLETE (2026-02-02, Session 59)

| # | Task | Action | Status | Evidence |
|---|------|--------|--------|----------|
| D1 | Test bundle factory self-test | `cd ~/distributed_prng_analysis && python3 agents/contexts/bundle_factory.py` | ✅ | 7/7 bundles, render+token tests passed (Session 58) |
| D2 | Test selfplay dispatch | `python3 agents/watcher_agent.py --dispatch-selfplay --dry-run` | ✅ | Passed Session 58 |
| D3 | Test learning loop dispatch | `python3 agents/watcher_agent.py --dispatch-learning-loop steps_3_5_6 --dry-run` | ✅ | Passed Session 58 |
| D4 | Test request processing | Create mock request, process via WATCHER | ✅ | 2 requests → DRY_RUN_OK, archived |
| D5 | End-to-end: Chapter 13 → WATCHER → Selfplay | Full flow test (non-dry-run, reduced episodes) | ✅ | Clean pass: real LLM eval, lifecycle stop/restart, grammar-constrained JSON (Session 59) |

**D5 Integration Bugs Found & Fixed (Session 59):**

| # | Bug | Root Cause | Fix | Commit |
|---|-----|-----------|-----|--------|
| 1 | Lifecycle dead code | `self.llm_lifecycle` never set in `__init__` | Added initialization block | `e4dd1b0` |
| 2 | API mismatch | `.start()` / `.stop(string)` not real methods | → `.ensure_running()` / `.stop()` | `e4dd1b0` |
| 3 | Router always None | `GrammarType` import poisoned entire import | Removed dead import | `e4dd1b0` |
| 4 | Grammar 400 errors | `agent_grammars/` had broken v1.0 GBNF | Copied fixed v1.1 from `grammars/` | `e4dd1b0` |
| 5 | Try 1 private API | `_call_primary_with_grammar()` missing config | Gate to public API for `watcher_decision.gbnf` only | `308a2fc` |

**D5 Clean Pass Evidence:**
```
Pre-validation: real LLM (4s response, not instant heuristic)
LLM stop: "confirmed stopped — GPU VRAM freed"
Selfplay: rc=0, candidate emitted (58s)
LLM restart: "healthy after 3.2s"
Post-eval: grammar-constrained JSON — real structured output
Archive: COMPLETED — zero warnings, zero heuristic fallbacks
```

---

## Final Effort Summary

| Part | Tasks | Time | Lines | Status | Commits |
|------|-------|------|-------|--------|---------|
| ~~A: Selfplay Testing~~ | ~~5~~ | ~~30-60 min~~ | ~~0~~ | ✅ Complete | `c0f5d32` |
| ~~B0: Bundle Factory~~ | ~~4~~ | ~~60-90 min~~ | ~~~1200~~ | ✅ Complete | `ffe397a` |
| ~~B: WATCHER Dispatch~~ | ~~4~~ | ~~60-90 min~~ | ~~~894~~ | ✅ Complete | `a145e28` |
| ~~C: File Organization~~ | ~~1~~ | ~~5 min~~ | ~~0~~ | ✅ Complete | `22abd7b` |
| ~~D: Integration Testing~~ | ~~5~~ | ~~60 min~~ | ~~0~~ | ✅ Complete | `e4dd1b0`, `308a2fc` |

---

## Dependency Chain (COMPLETE)

```
Part A (Selfplay Testing) ✅ DONE (c0f5d32)
       ↓
Part B0 (Bundle Factory + Spec) ✅ DONE (ffe397a)
       ↓
Part B (WATCHER Dispatch Functions) ✅ DONE (a145e28)
       ↓
Part C (File Organization) ✅ DONE (22abd7b)
       ↓
Part D (Integration Testing) ✅ DONE (e4dd1b0, 308a2fc)
       ↓
   FULL AUTONOMY ACHIEVED ✅
```

---

## Success Criteria — ALL MET ✅

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
- [x] `dispatch_learning_loop()` runs Steps 3→5→6
- [x] `process_chapter_13_request()` handles watcher_requests/*.json
- [x] All dispatch functions use `build_llm_context()` (Guardrail #1 verified)
- [x] No baked-in token assumptions (Guardrail #2 verified)

### Part D Complete When:
- [x] Bundle factory self-test passes on Zeus
- [x] End-to-end flow works without human intervention
- [x] Audit trail shows WATCHER → Selfplay dispatch
- [x] Request → Approval → Execution cycle verified

### Full Autonomy Achieved When:
```
Chapter 13 Triggers → watcher_requests/ → WATCHER → Selfplay/Learning Loop
       ↑                                                        ↓
       └──────────────── Diagnostics ← Reality ←────────────────┘
```

**No human in the loop for routine decisions. ✅ VERIFIED Session 59.**

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
| `agent_grammars/*.gbnf` | FIX (v1.0 → v1.1) | `agent_grammars/` | D | `e4dd1b0` |
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
| Vector database / embeddings | ❌ Out of scope | Not planned |
| GPU-resident retrieval services | ❌ Out of scope | Not planned |
| Raw decision log ingestion | ❌ Out of scope | Track 2 (structured only) |
| Authority changes | ❌ Out of scope | Not planned |
| Parameter advisor (Item B) | ⏳ Deferred | Steps 4-6 in production |
| Phase 9B.3 auto-heuristics | ⏳ Deferred | 9B.2 validated first |
| Track 2 retrieval enhancement | ⏳ After dispatch wired | Fills stubs in bundle_factory |

---

## Evaluation Path Architecture (Session 59 Final)

```
_evaluate_step_via_bundle(prompt, grammar_name)
    │
    ├─ Try 1: LLM Router (public API)
    │   └─ ONLY if grammar_name == 'watcher_decision.gbnf'
    │   └─ Calls evaluate_watcher_decision() → grammar-constrained JSON
    │
    ├─ Try 2: HTTP Direct (generic)
    │   └─ POST localhost:8080/completion with inline grammar content
    │   └─ Works for ALL grammars (chapter_13, agent_decision, etc.)
    │
    └─ Try 3: Heuristic Fallback
        └─ proceed, confidence=0.50
        └─ Should NEVER be reached if LLM is healthy
```

---

**END OF PHASE 7 TODO (v3 — COMPLETE)**

# Chapter 13 Implementation Progress

**Last Updated:** 2026-01-30  
**Document Version:** 2.0.0  
**Status:** Phases 1-6 Code Complete → Phase 7 (WATCHER Integration) Required  
**Team Beta Endorsement:** ✅ Approved (Phase 9B.2 complete)

---

## ⚠️ Documentation Sync Notice (2026-01-30)

**Issue Identified:** Section 19 checklist in `CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md` showed unchecked boxes despite code being complete since January 12.

**Resolution:** This document now serves as the authoritative progress tracker. Section 19 has been updated separately.

**Lesson Learned:** When code is completed, update BOTH the progress tracker AND the original chapter checklist.

---

## Overall Progress

| Phase | Status | Owner | Completion | Verified |
|-------|--------|-------|------------|----------|
| 1. Draw Ingestion | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| 2. Diagnostics Engine | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| 3. Retrain Triggers | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| 4. LLM Integration | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| 5. Acceptance Engine | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| 6. Chapter 13 Orchestration | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| **7. WATCHER Integration** | ❌ **NOT STARTED** | TBD | — | — |
| 8. Selfplay Integration | ✅ Complete | Team Beta | 2026-01-30 | 2026-01-30 |
| 9A. Chapter 13 ↔ Selfplay Hooks | ✅ Complete | Team Beta | 2026-01-30 | 2026-01-30 |
| 9B.1 Policy Transform Module | ✅ Complete | Claude | 2026-01-30 | 2026-01-30 |
| 9B.2 Policy-Conditioned Mode | ✅ Complete | Claude | 2026-01-30 | 2026-01-30 |
| 9B.3 Policy Proposal Heuristics | 🔲 Future | TBD | — | — |

**Legend:** 🔲 Not Started | 🟡 In Progress | ✅ Complete | ❌ Blocked/Missing

---

## Files Inventory (Verified 2026-01-30)

### Chapter 13 Core Files

| File | Size | Created | Updated | Purpose |
|------|------|---------|---------|---------|
| `chapter_13_diagnostics.py` | 39KB | Jan 12 | Jan 29 | Diagnostics engine |
| `chapter_13_llm_advisor.py` | 23KB | Jan 12 | Jan 12 | LLM analysis module |
| `chapter_13_triggers.py` | 36KB | Jan 12 | Jan 29 | Retrain trigger logic |
| `chapter_13_acceptance.py` | 41KB | Jan 12 | Jan 29 | Proposal validation |
| `chapter_13_orchestrator.py` | 23KB | Jan 12 | Jan 12 | Main orchestrator |
| `llm_proposal_schema.py` | 14KB | Jan 12 | Jan 12 | Pydantic models |
| `chapter_13.gbnf` | 2.9KB | Jan 12 | Jan 12 | LLM grammar constraint |
| `draw_ingestion_daemon.py` | 22KB | Jan 12 | Jan 12 | Draw monitoring |
| `synthetic_draw_injector.py` | 20KB | Jan 12 | Jan 12 | Test mode injection |
| `watcher_policies.json` | 4.7KB | Jan 12 | Jan 29 | Policy thresholds |

**Total:** ~226KB of Chapter 13 code

### Phase 9B Files (Selfplay)

| File | Size | Created | Purpose |
|------|------|---------|---------|
| `selfplay_orchestrator.py` | 43KB | Jan 29 | Main selfplay loop (v1.1.0) |
| `policy_transform.py` | 36KB | Jan 30 | Transform engine (v1.0.0) |
| `policy_conditioned_episode.py` | 25KB | Jan 30 | Episode conditioning (v1.0.0) |
| `inner_episode_trainer.py` | — | Jan 29 | Tree model trainer |
| `modules/learning_telemetry.py` | — | Jan 29 | Telemetry system |

---

## ❌ Phase 7: WATCHER Integration (THE GAP)

**This is the only remaining work for full autonomy.**

### What's Missing in `agents/watcher_agent.py`

| Function | Purpose | Status |
|----------|---------|--------|
| `dispatch_selfplay()` | Spawn selfplay_orchestrator.py | ❌ Missing |
| `dispatch_learning_loop()` | Run Steps 3→5→6 | ❌ Missing |
| `process_chapter_13_request()` | Handle watcher_requests/*.json | ❌ Missing |
| Chapter 13 daemon integration | Wire orchestrator to WATCHER | ❌ Missing |

### Integration Flow (Designed but not wired)

```
Chapter 13 Triggers                WATCHER                    Execution
──────────────────                 ───────                    ─────────
request_selfplay()
        │
        └──► watcher_requests/*.json
                    │
                    └──► process_chapter_13_request()  ← MISSING
                              │
                              ▼
                         validate_request()
                              │
                              ▼ (if APPROVE)
                    dispatch_selfplay()  ← MISSING
                              │
                              └──────────────────────► selfplay_orchestrator.py
```

### Files to Modify

| File | Changes Needed | Est. Lines |
|------|----------------|------------|
| `agents/watcher_agent.py` | Add dispatch functions | ~150 |
| Move `chapter_13.gbnf` | → `agent_grammars/chapter_13.gbnf` | 0 (file move) |

---

## ✅ What Works Today

### Can Run Directly (CLI)

```bash
# Selfplay (verified 2026-01-30)
python3 selfplay_orchestrator.py --survivors survivors_with_scores.json --policy-conditioned

# Chapter 13 Orchestrator
python3 chapter_13_orchestrator.py

# Diagnostics
python3 chapter_13_diagnostics.py --generate

# LLM Advisor
python3 chapter_13_llm_advisor.py --diagnose post_draw_diagnostics.json

# Acceptance Validation
python3 chapter_13_acceptance.py --validate-proposal proposal.json
```

### Can Run via WATCHER

```bash
# Pipeline Steps 1-6
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 6
```

### CANNOT Run via WATCHER (Gap)

```bash
# These don't work yet:
PYTHONPATH=. python3 agents/watcher_agent.py --dispatch-selfplay  # ❌ Not implemented
PYTHONPATH=. python3 agents/watcher_agent.py --chapter-13-daemon  # ❌ Not implemented
```

---

## Critical Design Invariants

### Chapter 13 Invariant
**Chapter 13 v1 does not alter model weights directly. All learning occurs through controlled re-execution of Step 5 with expanded labels.**

### Selfplay Invariant
**GPU sieving work MUST use coordinator.py / scripts_coordinator.py. Direct SSH to rigs for GPU work is FORBIDDEN.**

### Learning Authority Invariant
**Learning is statistical (tree models + bandit). Verification is deterministic (Chapter 13). LLM is advisory only. Telemetry is observational only.**

### Policy Transform Invariant
**`apply_policy()` is pure functional: stateless, deterministic, never fabricates data. Same inputs always produce same outputs.**

### Documentation Sync Invariant (NEW)
**When code is completed, update BOTH the progress tracker AND the original chapter checklist within the same session.**

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-12 | 1.0.0 | Initial document, Phases 1-6 code complete |
| 2026-01-18 | 1.1.0 | Added Phase 7 testing framework |
| 2026-01-23 | 1.2.0 | NPZ v3.0 integration notes |
| 2026-01-27 | 1.3.0 | GPU stability improvements |
| 2026-01-29 | 1.5.0 | Phase 8 Selfplay architecture approved |
| 2026-01-30 | 1.6.0 | Phase 8 COMPLETE — Zeus integration verified |
| 2026-01-30 | 1.7.0 | Phase 9A COMPLETE — Hooks verified |
| 2026-01-30 | 1.8.0 | Phase 9B.1 COMPLETE — Policy Transform Module |
| 2026-01-30 | 1.9.0 | Phase 9B.2 COMPLETE — Integration verified |
| **2026-01-30** | **2.0.0** | **Documentation audit — Identified Phase 7 as actual gap** |

---

## Next Steps

1. **Complete Phase 7** — WATCHER → Chapter 13 → Selfplay integration
2. **Integration Testing** — End-to-end autonomous operation
3. **Phase 9B.3** (Optional) — Automatic policy proposal heuristics

---

*Update this document as implementation progresses.*

# CONTRACT ADDENDUM: Section 8.5 — LLM Lifecycle Dependency

**Add to:** `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md`, after Section 8.4 (Activation Gate)
**Date:** 2026-02-08
**Authority:** Team Beta (approved in Session 67)
**Ref:** PROPOSAL_STRATEGY_ADVISOR_LIFECYCLE_INTEGRATION_v1_0.md

---

### 8.5 LLM Lifecycle Dependency

The Strategy Advisor MUST use `llm_lifecycle.ensure_running()` before any LLM 
analysis call. The lifecycle manager (`llm_services/llm_lifecycle.py`, Session 57)
guarantees LLM availability in ~3 seconds from cold state.

**Decision hierarchy:**

```
1. DeepSeek (primary) — routine analysis, grammar-constrained
2. Claude (backup)    — escalation on low confidence + risky action
3. Heuristic (emergency) — ONLY when both LLMs unreachable
   MUST log as DEGRADED_MODE warning
   MUST tag recommendation with mode=heuristic_degraded
```

**Escalation trigger (decision-type gated):**

Escalation from DeepSeek to Claude occurs when BOTH conditions are met:
- `focus_confidence < 0.3`
- `recommended_action` is one of: `RETRAIN`, `ESCALATE`, `FULL_RESET`

Low-confidence `WAIT` or `REFOCUS` actions do NOT trigger escalation, as they
are informational and non-destructive.

**VRAM context:**

The advisor runs BEFORE GPU-heavy dispatch phases, so VRAM is available for the
LLM server. There is no legitimate VRAM contention at the advisor's decision point.

Lifecycle `stop()` is NOT called by the advisor. The subsequent `dispatch_selfplay()`
call handles LLM shutdown when GPU VRAM is needed for compute.

**Router dependency:**

The advisor calls `llm_router.evaluate_with_grammar()` (added v2.1.0) for 
grammar-constrained LLM calls. This method supports `force_backup=True` for 
explicit Claude escalation without router ambiguity.

---

### 8.6 Degraded Mode Invariant

```
If the advisor produces a recommendation with advisor_model="heuristic_degraded",
downstream consumers (WATCHER, Chapter 13 logs) MUST treat this as a reduced-quality
decision. The degraded_reason field in metadata provides forensic context.

In normal operation with functional LLM infrastructure, the heuristic path 
MUST NEVER be reached. Any occurrence of heuristic_degraded in production logs
indicates an infrastructure failure requiring investigation.
```
